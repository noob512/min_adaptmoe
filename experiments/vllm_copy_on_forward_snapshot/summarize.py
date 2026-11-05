#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


CASES = ("baseline", "uva", "copy")
PHASES = ("warm", "eval1", "eval2", "eval3", "eval4", "eval5")
EVAL_PHASES = ("eval1", "eval2", "eval3", "eval4", "eval5")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            records.append(value)
    return records


def output_hash(phase: dict[str, Any]) -> str:
    payload = json.dumps(
        [request["output_token_ids"] for request in phase["requests"]],
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


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
    return hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-requests", type=int, default=4)
    parser.add_argument("--expected-prompt-tokens", type=int, default=1217)
    args = parser.parse_args()
    output = args.output or args.run_root / "snapshot.json"

    results: dict[str, Any] = {}
    signatures: dict[str, list[dict[str, Any]]] = {}
    hashes: set[str] = set()

    def summarize_copy_records(records: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "record_count": len(records),
            "modules": sorted(
                {str(record.get("module_prefix")) for record in records}
            ),
            "bytes": sum(int(record.get("copy_bytes") or 0) for record in records),
            "tensor_count": sum(
                int(record.get("copy_tensor_count") or 0) for record in records
            ),
            "copy_cuda_sum_ms": sum(
                float(record.get("copy_cuda_ms") or 0) for record in records
            ),
            "copy_submit_sum_ms": sum(
                float(record.get("copy_submit_us") or 0) for record in records
            )
            / 1e3,
            "records": records,
        }

    for case in CASES:
        root = args.run_root / case
        returncode = int((root / "returncode.txt").read_text().strip())
        result = read_json(root / "result.json")
        raw_phases = result.get("phases") or []
        if (
            returncode != 0
            or result.get("status") != "ok"
            or len(raw_phases) != len(PHASES)
        ):
            raise SystemExit(
                f"invalid case {case}: rc={returncode} "
                f"status={result.get('status')} phases={len(raw_phases)}"
            )
        phases = {str(phase.get("phase")): phase for phase in raw_phases}
        if tuple(phases) != PHASES:
            raise SystemExit(f"invalid phase order for {case}: {tuple(phases)}")

        expected = {
            "request_count": args.expected_requests,
            "completed_requests": args.expected_requests,
            "prompt_tokens": args.expected_prompt_tokens,
            "output_tokens": args.expected_requests,
            "output_tokens_per_request": 1,
        }
        for phase_name, phase in phases.items():
            mismatch = {
                key: {"actual": phase.get(key), "expected": value}
                for key, value in expected.items()
                if int(phase.get(key, -1)) != value
            }
            if mismatch:
                raise SystemExit(
                    f"invalid phase for {case}/{phase_name}: {mismatch}"
                )

        progress = read_jsonl(root / "progress.jsonl")
        windows: dict[str, tuple[float, float]] = {}
        for phase_name in PHASES:
            begin = next(
                record
                for record in progress
                if record.get("action") == "phase_begin"
                and record.get("phase") == phase_name
            )
            end = next(
                record
                for record in progress
                if record.get("action") == "phase_end"
                and record.get("phase") == phase_name
            )
            windows[phase_name] = (float(begin["timestamp"]), float(end["timestamp"]))

        manifest = read_jsonl(root / "cpu_offload_manifest.jsonl")
        signature = parameter_signature(manifest)
        signatures[case] = signature
        all_copy_records = [
            record
            for record in manifest
            if record.get("event") == "cpu_offload_copy_forward"
        ]
        records_by_phase = {
            phase_name: [
                record
                for record in all_copy_records
                if begin <= float(record.get("timestamp") or 0) <= end
            ]
            for phase_name, (begin, end) in windows.items()
        }
        in_phase_ids = {
            id(record)
            for records in records_by_phase.values()
            for record in records
        }
        pre_phase_records = [
            record for record in all_copy_records if id(record) not in in_phase_ids
        ]
        if case == "baseline" and signature:
            raise SystemExit("baseline unexpectedly offloaded parameters")
        if case == "copy":
            for phase_name, records in records_by_phase.items():
                if not records or not all(
                    bool(record.get("ok"))
                    and bool(record.get("copy_sync"))
                    and record.get("copy_cuda_ms") is not None
                    for record in records
                ):
                    raise SystemExit(
                        f"invalid copy trace for {phase_name}: {records!r}"
                    )
        elif all_copy_records:
            raise SystemExit(f"unexpected copy trace for {case}")

        phase_results = {}
        for phase_name, phase in phases.items():
            token_hash = output_hash(phase)
            hashes.add(token_hash)
            phase_results[phase_name] = {
                "elapsed_seconds": phase["elapsed_seconds"],
                "mean_ttft_ms": phase["mean_ttft_ms"],
                "p99_ttft_ms": phase["p99_ttft_ms"],
                "output_tokens_per_second": phase["output_tokens_per_second"],
                "output_token_hash": token_hash,
                "copy_trace": summarize_copy_records(records_by_phase[phase_name]),
            }
        eval_elapsed = [
            float(phase_results[name]["elapsed_seconds"]) for name in EVAL_PHASES
        ]
        eval_ttft = [
            float(phase_results[name]["mean_ttft_ms"]) for name in EVAL_PHASES
        ]
        results[case] = {
            "load_seconds": result["load_seconds"],
            "gpu_after_load": result.get("gpu_after_load"),
            "eval_elapsed_mean_seconds": sum(eval_elapsed) / len(eval_elapsed),
            "eval_elapsed_median_seconds": statistics.median(eval_elapsed),
            "eval_mean_ttft_ms": sum(eval_ttft) / len(eval_ttft),
            "eval_median_ttft_ms": statistics.median(eval_ttft),
            "phases": phase_results,
            "offloaded_parameter_count": len(signature),
            "offloaded_bytes": sum(item["bytes"] for item in signature),
            "offload_signature_sha256": signature_hash(signature),
            "pre_phase_copy_trace": summarize_copy_records(pre_phase_records),
        }

    if signatures["uva"] != signatures["copy"]:
        raise SystemExit("UVA and copy parameter signatures differ")
    if not signatures["uva"]:
        raise SystemExit("offload cases contain no parameters")
    offloaded_bytes = sum(item["bytes"] for item in signatures["copy"])
    phase_copy_bytes = []
    for phase_name in PHASES:
        copied = int(
            results["copy"]["phases"][phase_name]["copy_trace"]["bytes"]
        )
        phase_copy_bytes.append(copied)
        # Post-load quantization may replace or deduplicate small parameters,
        # so the actual state_dict copy can be slightly below the construction
        # manifest. It must still cover essentially the complete payload.
        if not 0.99 * offloaded_bytes <= copied <= offloaded_bytes:
            raise SystemExit(
                f"copy bytes invalid for {phase_name}: "
                f"copied={copied} offloaded={offloaded_bytes}"
            )
    if len(set(phase_copy_bytes)) != 1:
        raise SystemExit(f"copy bytes differ across phases: {phase_copy_bytes}")
    if len(hashes) != 1:
        raise SystemExit(f"output token hashes differ: {hashes}")

    elapsed_mean = {
        case: float(results[case]["eval_elapsed_mean_seconds"])
        for case in CASES
    }
    elapsed_median = {
        case: float(results[case]["eval_elapsed_median_seconds"])
        for case in CASES
    }
    ttft_median = {
        case: float(results[case]["eval_median_ttft_ms"]) for case in CASES
    }
    snapshot = {
        "experiment": "vllm_copy_on_forward_snapshot",
        "run_root": str(args.run_root),
        "fixture": read_json(args.run_root / "fixtures" / "fixture_metadata.json"),
        "results": results,
        "offload_signature": signatures["copy"],
        "comparisons": {
            "uva_vs_baseline_elapsed_median_overhead_pct": (
                elapsed_median["uva"] / elapsed_median["baseline"] - 1.0
            )
            * 100.0,
            "copy_vs_baseline_elapsed_median_overhead_pct": (
                elapsed_median["copy"] / elapsed_median["baseline"] - 1.0
            )
            * 100.0,
            "copy_vs_uva_elapsed_median_ratio": (
                elapsed_median["copy"] / elapsed_median["uva"]
            ),
            "copy_vs_uva_elapsed_median_reduction_pct": (
                1.0 - elapsed_median["copy"] / elapsed_median["uva"]
            )
            * 100.0,
            "copy_vs_uva_ttft_median_ratio": (
                ttft_median["copy"] / ttft_median["uva"]
            ),
            "copy_vs_uva_elapsed_mean_ratio": (
                elapsed_mean["copy"] / elapsed_mean["uva"]
            ),
        },
        "integrity": {
            "all_returncodes_zero": True,
            "all_status_ok": True,
            "all_output_token_hashes_equal": True,
            "output_token_hash": next(iter(hashes)),
            "uva_copy_parameter_signatures_equal": True,
            "offload_signature_sha256": signature_hash(signatures["copy"]),
        },
        "limitations": {
            "single_cold_process_per_case": True,
            "five_eval_phases_after_one_in_process_warm_phase": True,
            "copy_backend_is_dormant_vllm_branch_exposed_by_experiment": True,
            "copy_is_layer_parameter_granularity_not_expert_granularity": True,
            "copy_sync_before_compute": True,
            "prefill_only_max_tokens": 1,
        },
    }
    output.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
