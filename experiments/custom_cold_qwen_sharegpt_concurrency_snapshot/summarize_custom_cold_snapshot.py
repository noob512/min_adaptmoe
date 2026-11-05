#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CONCURRENCIES = (1, 2, 3, 4, 8, 16, 24, 32)
TARGETS = (2, 6, 10, 14, 18)
CASE_BY_TARGET = {2: "g2", 6: "g6", 10: "g10", 14: "g14", 18: "g18"}
BASELINE_BY_TARGET = {
    2: "baseline_asc",
    6: "baseline_asc",
    10: "baseline_asc",
    14: "baseline_desc",
    18: "baseline_desc",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def phase_map(result: dict[str, Any]) -> dict[int, dict[str, Any]]:
    phases = {}
    for phase in result.get("phases", []):
        name = str(phase.get("phase", ""))
        if name.startswith("sharegpt_c"):
            phases[int(name.removeprefix("sharegpt_c"))] = phase
    return phases


def token_hash(phase: dict[str, Any]) -> str:
    values = [item.get("output_token_ids", []) for item in phase.get("requests", [])]
    payload = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def slowdown_pct(baseline: float | None, value: float | None, *, lower_better: bool) -> float | None:
    if baseline is None or value is None or baseline == 0:
        return None
    if lower_better:
        return (value / baseline - 1.0) * 100.0
    return (1.0 - value / baseline) * 100.0


def first_observed(points: list[dict[str, Any]], key: str) -> int | None:
    for point in points:
        value = point.get(key)
        if value is not None and value >= 10.0:
            return int(point["concurrency"])
    return None


def first_confirmed(points: list[dict[str, Any]], key: str) -> int | None:
    for index, point in enumerate(points[:-1]):
        value = point.get(key)
        next_value = points[index + 1].get(key)
        if (
            value is not None
            and next_value is not None
            and value >= 10.0
            and next_value >= 10.0
        ):
            return int(point["concurrency"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--native-snapshot", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    plan_summary = read_json(args.run_root / "plans" / "plan_summary.json")
    fixture_metadata = read_json(args.run_root / "fixtures" / "fixture_metadata.json")
    driver_metadata = read_json(args.run_root / "driver_metadata.json")
    native = read_json(args.native_snapshot)
    native_by_target = {
        int(item["requested_offload_gib"]): item for item in native["snapshots"]
    }
    plan_by_target = {
        int(item["target_gib"]): item for item in plan_summary["plans"]
    }

    cases: dict[str, dict[str, Any]] = {}
    required_cases = ["pilot", "baseline_asc", "baseline_desc"] + [
        CASE_BY_TARGET[target] for target in TARGETS
    ]
    for name in required_cases:
        case_root = args.run_root / name
        result = read_json(case_root / "result.json")
        cases[name] = {
            "returncode": int((case_root / "returncode.txt").read_text().strip()),
            "status": result.get("status"),
            "load_seconds": result.get("load_seconds"),
            "gpu_after_load": result.get("gpu_after_load"),
            "scan_order": (case_root / "scan_order.txt").read_text().strip(),
            "result": result,
        }

    snapshots = []
    all_outputs_match = True
    all_outputs_match_native = True
    native_output_validation_available = True
    all_phase_sets_match = True
    for target in TARGETS:
        case_name = CASE_BY_TARGET[target]
        baseline_name = BASELINE_BY_TARGET[target]
        custom_case = cases[case_name]
        baseline_case = cases[baseline_name]
        custom_phases = phase_map(custom_case["result"])
        baseline_phases = phase_map(baseline_case["result"])
        native_snapshot = native_by_target[target]
        native_points = {
            int(item["concurrency"]): item for item in native_snapshot["results"]
        }
        native_baseline_path = (
            Path(native["run_root"])
            / f"pair{native_snapshot['pair']}_g{target}_baseline"
            / "native_uva"
            / "result.json"
        )
        native_baseline_phases = (
            phase_map(read_json(native_baseline_path))
            if native_baseline_path.exists()
            else None
        )
        native_output_validation_available = (
            native_output_validation_available and native_baseline_phases is not None
        )
        phase_sets_match = set(custom_phases) == set(baseline_phases) == set(CONCURRENCIES)
        all_phase_sets_match = all_phase_sets_match and phase_sets_match
        points = []
        for concurrency in CONCURRENCIES:
            baseline_phase = baseline_phases[concurrency]
            custom_phase = custom_phases[concurrency]
            native_point = native_points[concurrency]
            outputs_match = token_hash(baseline_phase) == token_hash(custom_phase)
            all_outputs_match = all_outputs_match and outputs_match
            outputs_match_native = (
                token_hash(custom_phase) == token_hash(native_baseline_phases[concurrency])
                if native_baseline_phases is not None
                else None
            )
            if outputs_match_native is False:
                all_outputs_match_native = False
            custom_throughput = float(custom_phase["output_tokens_per_second"])
            native_offload_throughput = float(native_point["offload_output_tok_s"])
            points.append(
                {
                    "concurrency": concurrency,
                    "prompt_tokens": custom_phase["prompt_tokens"],
                    "baseline_output_tok_s": baseline_phase["output_tokens_per_second"],
                    "custom_output_tok_s": custom_throughput,
                    "throughput_slowdown_pct": slowdown_pct(
                        baseline_phase["output_tokens_per_second"],
                        custom_throughput,
                        lower_better=False,
                    ),
                    "baseline_mean_tpot_ms": baseline_phase["mean_tpot_ms"],
                    "custom_mean_tpot_ms": custom_phase["mean_tpot_ms"],
                    "tpot_slowdown_pct": slowdown_pct(
                        baseline_phase["mean_tpot_ms"],
                        custom_phase["mean_tpot_ms"],
                        lower_better=True,
                    ),
                    "baseline_mean_ttft_ms": baseline_phase["mean_ttft_ms"],
                    "custom_mean_ttft_ms": custom_phase["mean_ttft_ms"],
                    "ttft_slowdown_pct": slowdown_pct(
                        baseline_phase["mean_ttft_ms"],
                        custom_phase["mean_ttft_ms"],
                        lower_better=True,
                    ),
                    "native_offload_output_tok_s": native_offload_throughput,
                    "native_throughput_slowdown_pct": native_point[
                        "throughput_slowdown_pct"
                    ],
                    "custom_vs_native_offload_throughput_gain_pct": (
                        custom_throughput / native_offload_throughput - 1.0
                    )
                    * 100.0,
                    "outputs_match_custom_baseline": outputs_match,
                    "outputs_match_native_baseline": outputs_match_native,
                    "baseline_output_token_hash": token_hash(baseline_phase),
                    "custom_output_token_hash": token_hash(custom_phase),
                }
            )

        baseline_used = int(baseline_case["gpu_after_load"]["used_mib"])
        custom_used = int(custom_case["gpu_after_load"]["used_mib"])
        snapshots.append(
            {
                "requested_offload_gib": target,
                "case": case_name,
                "baseline_case": baseline_name,
                "scan_order": custom_case["scan_order"],
                "plan": plan_by_target[target],
                "native_actual_offload_gib": native_snapshot["actual_offload_gib"],
                "native_expert_offload_gib": native_snapshot["expert_offload_gib"],
                "custom_load_seconds": custom_case["load_seconds"],
                "baseline_load_seconds": baseline_case["load_seconds"],
                "gpu_after_load_used_mib": custom_used,
                "baseline_gpu_after_load_used_mib": baseline_used,
                "observed_gpu_used_reduction_mib": baseline_used - custom_used,
                "phase_sets_match": phase_sets_match,
                "all_outputs_match": all(item["outputs_match_custom_baseline"] for item in points),
                "first_observed_throughput_drop_10pct": first_observed(
                    points, "throughput_slowdown_pct"
                ),
                "first_confirmed_throughput_drop_10pct": first_confirmed(
                    points, "throughput_slowdown_pct"
                ),
                "first_observed_tpot_drop_10pct": first_observed(
                    points, "tpot_slowdown_pct"
                ),
                "first_confirmed_tpot_drop_10pct": first_confirmed(
                    points, "tpot_slowdown_pct"
                ),
                "results": points,
            }
        )

    baseline_path_overhead = []
    for target in TARGETS:
        custom_baseline = phase_map(cases[BASELINE_BY_TARGET[target]]["result"])
        native_points = {
            int(item["concurrency"]): item
            for item in native_by_target[target]["results"]
        }
        baseline_path_overhead.append(
            {
                "target_pair": target,
                "scan_order": cases[BASELINE_BY_TARGET[target]]["scan_order"],
                "throughput_delta_pct_by_concurrency": {
                    str(concurrency): (
                        float(custom_baseline[concurrency]["output_tokens_per_second"])
                        / float(native_points[concurrency]["baseline_output_tok_s"])
                        - 1.0
                    )
                    * 100.0
                    for concurrency in CONCURRENCIES
                },
            }
        )

    all_cases_ok = all(
        item["returncode"] == 0 and item["status"] == "ok" for item in cases.values()
    )
    summary = {
        "experiment": "custom_cold_expert_sharegpt_concurrency_snapshot",
        "run_root": str(args.run_root),
        "model": fixture_metadata["model"],
        "configuration": {
            "concurrencies": list(CONCURRENCIES),
            "output_tokens": fixture_metadata["output_tokens"],
            "kv_cache_bytes": 1_288_490_189,
            "max_model_len": 512,
            "max_num_seqs": 32,
            "max_num_batched_tokens": 2048,
            "cpu_offload_gb": 0,
            "expert_access_mode": "custom_managed_uvm_remote_read",
            "baseline": "same custom managed/envelope path with an empty cold plan",
            "calibration": "32 disjoint ShareGPT prompts, one c32 x 64-token pilot",
        },
        "fixtures": fixture_metadata,
        "driver": driver_metadata,
        "plan_summary": plan_summary,
        "snapshots": snapshots,
        "custom_zero_plan_vs_native_zero_offload_baselines": baseline_path_overhead,
        "integrity": {
            "all_cases_ok": all_cases_ok,
            "all_phase_sets_match": all_phase_sets_match,
            "all_outputs_match_custom_baseline": all_outputs_match,
            "native_output_validation_available": native_output_validation_available,
            "all_outputs_match_native_baseline": (
                all_outputs_match_native if native_output_validation_available else None
            ),
            "all_plans_nested": plan_summary["all_plans_nested"],
            "evaluation_matches_native_snapshot": fixture_metadata[
                "evaluation_matches_native_snapshot"
            ],
            "calibration_and_evaluation_disjoint": fixture_metadata[
                "disjoint_source_indices"
            ],
        },
        "limitations": {
            "single_snapshot_no_confidence_intervals": True,
            "concurrency_is_offline_burst": True,
            "native_and_custom_requested_gib_are_not_exactly_equal_bytes": True,
            "expert_migration_tested": False,
        },
    }
    if not all_cases_ok or not all_phase_sets_match or not all_outputs_match:
        raise SystemExit(
            "integrity failure: "
            f"cases={all_cases_ok} phases={all_phase_sets_match} outputs={all_outputs_match}"
        )
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    output = args.run_root / "snapshot_summary.json"
    output.write_text(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
