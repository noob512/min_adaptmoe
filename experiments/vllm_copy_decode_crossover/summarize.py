#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


LABELS = ("baseline", "uva", "copy")
T_CRITICAL_95 = {
    1: 12.706204736,
    2: 4.302652730,
    3: 3.182446305,
    4: 2.776445105,
    5: 2.570581836,
    6: 2.446911851,
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize empty values")
    result: dict[str, Any] = {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "ci95_low": None,
        "ci95_high": None,
    }
    if len(values) > 1:
        critical = T_CRITICAL_95.get(len(values) - 1, 1.96)
        half_width = critical * statistics.stdev(values) / math.sqrt(len(values))
        result["ci95_low"] = result["mean"] - half_width
        result["ci95_high"] = result["mean"] + half_width
    return result


def parameter_signature(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": record.get("full_parameter_name"),
            "bytes": int(record.get("bytes") or 0),
            "shape": record.get("shape"),
            "dtype": record.get("dtype"),
            "order": int(record.get("offload_order") or 0),
        }
        for record in records
        if record.get("event") == "cpu_offload_parameter"
    ]


def signature_hash(signature: list[dict[str, Any]]) -> str:
    return sha256_bytes(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    )


def validate_phase(phase: dict[str, Any], label: str) -> None:
    batch = int(phase["batch_tokens"])
    measured = phase.get("measured_decode_steps") or []
    expected_steps = int(phase["decode_step_count"])
    if len(measured) != expected_steps or expected_steps <= 0:
        raise SystemExit(f"invalid measured steps for {label}/b{batch}")
    for step in measured:
        schedule = step.get("schedule") or {}
        values = list((schedule.get("num_scheduled_tokens") or {}).values())
        if (
            int(schedule.get("total_num_scheduled_tokens", -1)) != batch
            or len(values) != batch
            or any(int(value) != 1 for value in values)
            or schedule.get("new_request_ids")
            or schedule.get("resumed_request_ids")
            or schedule.get("scheduled_spec_decode_request_ids")
            or schedule.get("preempted_request_ids")
        ):
            raise SystemExit(f"non-strict decode step for {label}/b{batch}: {step}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_root / "snapshot.json"

    fixture = read_json(args.run_root / "fixture.json")
    fixture_metadata = read_json(args.run_root / "fixture_metadata.json")
    cases = []
    signatures: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for case_root in sorted(args.run_root.glob("r*_*")):
        repeat_part, label = case_root.name.split("_", 1)
        if not repeat_part.startswith("r") or label not in LABELS:
            continue
        repeat = int(repeat_part[1:])
        returncode = int((case_root / "returncode.txt").read_text().strip())
        result = read_json(case_root / "result.json")
        if returncode != 0 or result.get("status") != "ok":
            raise SystemExit(
                f"invalid case {case_root.name}: rc={returncode} "
                f"status={result.get('status')}"
            )
        manifest = read_jsonl(case_root / "cpu_offload_manifest.jsonl")
        parameter_records = [
            record
            for record in manifest
            if record.get("event") == "cpu_offload_parameter"
        ]
        copy_trace_records = [
            record
            for record in manifest
            if record.get("event") == "cpu_offload_copy_forward"
        ]
        signature = parameter_signature(manifest)
        signatures[(repeat, label)] = signature
        if label == "baseline" and signature:
            raise SystemExit(f"baseline unexpectedly offloaded parameters: {case_root}")
        if label != "baseline" and not signature:
            raise SystemExit(f"offload case has no parameters: {case_root}")
        if copy_trace_records:
            raise SystemExit(f"performance case unexpectedly enabled copy trace: {case_root}")
        if label == "uva" and not all(
            record.get("offload_backend") == "uva"
            and bool(record.get("uva_offloading"))
            and bool(record.get("cpu_data_pinned"))
            for record in parameter_records
        ):
            raise SystemExit(f"invalid UVA backend manifest: {case_root}")
        if label == "copy" and not all(
            record.get("offload_backend") == "copy"
            and bool(record.get("copy_on_forward"))
            and bool(record.get("cpu_data_pinned"))
            and str(record.get("device_after")) == "cpu"
            for record in parameter_records
        ):
            raise SystemExit(f"invalid copy backend manifest: {case_root}")

        phases = []
        for phase in result.get("phases") or []:
            validate_phase(phase, case_root.name)
            batch = int(phase["batch_tokens"])
            measured_ms = [
                float(step["elapsed_ms"])
                for step in phase["measured_decode_steps"]
            ]
            phases.append(
                {
                    "batch_tokens": batch,
                    "actual_prompt_tokens": int(phase["actual_prompt_tokens"]),
                    "prefill_elapsed_ms": float(phase["prefill_elapsed_ms"]),
                    "decode_step_count": len(measured_ms),
                    "decode_step_ms": measured_ms,
                    "decode_step_mean_ms": statistics.mean(measured_ms),
                    "decode_step_median_ms": statistics.median(measured_ms),
                    "decode_step_p90_ms": float(phase["decode_step_p90_ms"]),
                    "steady_decode_tokens_per_second": (
                        batch * 1000.0 / statistics.median(measured_ms)
                    ),
                    "output_token_hash": str(phase["output_token_hash"]),
                }
            )
        phases.sort(key=lambda phase: phase["batch_tokens"])
        cases.append(
            {
                "repeat": repeat,
                "label": label,
                "scan_order": (case_root / "scan_order.txt").read_text().strip(),
                "batch_order": (case_root / "batch_order.txt").read_text().strip(),
                "backend": (case_root / "backend.txt").read_text().strip(),
                "load_seconds": float(result["load_seconds"]),
                "gpu_after_load": result.get("gpu_after_load"),
                "offloaded_parameter_count": len(signature),
                "offloaded_bytes": sum(item["bytes"] for item in signature),
                "offload_signature_sha256": signature_hash(signature),
                "all_parameters_pinned": (
                    bool(signature)
                    and all(
                        bool(record.get("cpu_data_pinned"))
                        for record in parameter_records
                    )
                ),
                "copy_trace_record_count": len(copy_trace_records),
                "phases": phases,
            }
        )

    repeats = sorted({case["repeat"] for case in cases})
    by_case = {(case["repeat"], case["label"]): case for case in cases}
    for repeat in repeats:
        missing = [label for label in LABELS if (repeat, label) not in by_case]
        if missing:
            raise SystemExit(f"repeat {repeat} missing cases: {missing}")
        if signatures[(repeat, "uva")] != signatures[(repeat, "copy")]:
            raise SystemExit(f"repeat {repeat} UVA/copy signatures differ")
    offload_signatures = {
        signature_hash(signatures[(repeat, label)])
        for repeat in repeats
        for label in ("uva", "copy")
    }
    if len(offload_signatures) != 1:
        raise SystemExit(f"offload signatures differ across runs: {offload_signatures}")

    batch_sets = [
        {phase["batch_tokens"] for phase in case["phases"]} for case in cases
    ]
    if not batch_sets or any(batch_set != batch_sets[0] for batch_set in batch_sets):
        raise SystemExit(f"case batch sets differ: {batch_sets}")
    batches = sorted(batch_sets[0])

    comparison_rows = []
    all_outputs_match = True
    for batch in batches:
        paired = []
        hashes = set()
        for repeat in repeats:
            phase_by_label = {
                label: next(
                    phase
                    for phase in by_case[(repeat, label)]["phases"]
                    if phase["batch_tokens"] == batch
                )
                for label in LABELS
            }
            hashes.update(
                phase_by_label[label]["output_token_hash"] for label in LABELS
            )
            baseline_ms = float(phase_by_label["baseline"]["decode_step_median_ms"])
            uva_ms = float(phase_by_label["uva"]["decode_step_median_ms"])
            copy_ms = float(phase_by_label["copy"]["decode_step_median_ms"])
            paired.append(
                {
                    "repeat": repeat,
                    "scan_order": by_case[(repeat, "baseline")]["scan_order"],
                    "baseline_median_step_ms": baseline_ms,
                    "uva_median_step_ms": uva_ms,
                    "copy_median_step_ms": copy_ms,
                    "uva_vs_baseline_slowdown_pct": (
                        uva_ms / baseline_ms - 1.0
                    )
                    * 100.0,
                    "copy_vs_baseline_slowdown_pct": (
                        copy_ms / baseline_ms - 1.0
                    )
                    * 100.0,
                    "copy_vs_uva_step_reduction_pct": (
                        1.0 - copy_ms / uva_ms
                    )
                    * 100.0,
                    "baseline_decode_tokens_per_second": batch
                    * 1000.0
                    / baseline_ms,
                    "uva_decode_tokens_per_second": batch * 1000.0 / uva_ms,
                    "copy_decode_tokens_per_second": batch * 1000.0 / copy_ms,
                    "copy_vs_uva_throughput_change_pct": (
                        uva_ms / copy_ms - 1.0
                    )
                    * 100.0,
                    "copy_faster_than_uva": copy_ms < uva_ms,
                }
            )
        outputs_match = len(hashes) == 1
        all_outputs_match = all_outputs_match and outputs_match
        reduction_stats = stats(
            [pair["copy_vs_uva_step_reduction_pct"] for pair in paired]
        )
        comparison_rows.append(
            {
                "batch_tokens": batch,
                "paired": paired,
                "baseline_median_step_ms": stats(
                    [pair["baseline_median_step_ms"] for pair in paired]
                ),
                "uva_median_step_ms": stats(
                    [pair["uva_median_step_ms"] for pair in paired]
                ),
                "copy_median_step_ms": stats(
                    [pair["copy_median_step_ms"] for pair in paired]
                ),
                "copy_vs_uva_step_reduction_pct": reduction_stats,
                "baseline_decode_tokens_per_second": stats(
                    [pair["baseline_decode_tokens_per_second"] for pair in paired]
                ),
                "uva_decode_tokens_per_second": stats(
                    [pair["uva_decode_tokens_per_second"] for pair in paired]
                ),
                "copy_decode_tokens_per_second": stats(
                    [pair["copy_decode_tokens_per_second"] for pair in paired]
                ),
                "copy_vs_uva_throughput_change_pct": stats(
                    [pair["copy_vs_uva_throughput_change_pct"] for pair in paired]
                ),
                "copy_faster_repeat_count": sum(
                    bool(pair["copy_faster_than_uva"]) for pair in paired
                ),
                "median_copy_faster_than_uva": reduction_stats["median"] > 0,
                "all_outputs_match": outputs_match,
                "output_token_hash": next(iter(hashes)) if outputs_match else None,
            }
        )

    crossovers = [
        row["batch_tokens"]
        for row in comparison_rows
        if row["median_copy_faster_than_uva"]
    ]
    first_crossover = min(crossovers) if crossovers else None
    representative_signature = signatures[(repeats[0], "copy")]
    snapshot = {
        "experiment": "vllm_copy_decode_crossover",
        "run_root": str(args.run_root),
        "fixture": fixture_metadata
        | {
            "fixture_sha256_recomputed": sha256_file(args.run_root / "fixture.json"),
            "request_count_in_file": len(fixture.get("requests") or []),
        },
        "method": {
            "primary_metric": "median synchronized in-process decode engine-step wall time",
            "strict_decode_invariant": (
                "each measured step schedules exactly batch_tokens cached requests, "
                "one token per request, with no new/resumed/spec/preempted requests"
            ),
            "copy_trace_and_per_layer_sync_disabled": True,
            "prefill_excluded_from_measured_steps": True,
            "independent_cold_process_per_case": True,
            "paired_repeats": len(repeats),
        },
        "cases": cases,
        "batches": comparison_rows,
        "offload_signature": representative_signature,
        "integrity": {
            "all_returncodes_zero": True,
            "all_status_ok": True,
            "strict_decode_schedule_validated": True,
            "all_outputs_match": all_outputs_match,
            "uva_copy_parameter_signatures_equal": True,
            "offload_signature_sha256": next(iter(offload_signatures)),
            "offloaded_parameter_count": len(representative_signature),
            "offloaded_bytes": sum(
                item["bytes"] for item in representative_signature
            ),
        },
        "conclusion": {
            "first_sampled_median_copy_faster_batch_tokens": first_crossover,
            "last_sampled_batch_tokens": max(batches),
            "crossover_observed": first_crossover is not None,
            "crossover_statement": (
                f"first sampled crossover at {first_crossover} tokens/forward"
                if first_crossover is not None
                else (
                    "not observed at sampled points up to "
                    f"{max(batches)} tokens/forward"
                )
            ),
        },
        "limitations": {
            "short_sharegpt_prompts_1_to_8_fixture_tokens": True,
            "greedy_fixed_length_decode": True,
            "single_fixed_prompt_set": True,
            "only_two_gib_nominal_offload": True,
            "no_persistent_gpu_weight_cache": True,
        },
    }
    output.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
