#!/usr/bin/env python3
"""Verify and aggregate six single-run, independent-process E7 arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ARM_ORDER = [
    "frequency_cold",
    "random",
    "frozen_static",
    "gpu_heat_cpu_ewma",
    "oracle",
    "hot_adversarial",
]
FIXED_ARMS = [
    "frequency_cold",
    "random",
    "frozen_static",
    "oracle",
    "hot_adversarial",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: object required")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct_ratio(value: float, reference: float) -> float:
    return (value / reference - 1.0) * 100.0


def main() -> int:
    args = parse_args()
    root = args.run_root.resolve()
    frozen_path = root / "frozen_targets.json"
    frozen = load_json(frozen_path)
    require(
        frozen.get("schema") == "eurosys_e7_independent_cold_targets_v1",
        "target schema",
    )
    require(frozen.get("frozen_before_run") is True, "targets not frozen before run")
    require(list(frozen.get("targets", {})) == ARM_ORDER, "target arm order")
    source_path = Path(frozen["source_result"])
    require(source_path.is_file(), "target source result absent")
    require(sha256_file(source_path) == frozen["source_result_sha256"], "target source drift")

    before_module = load_json(root / "module.before.json")
    custom_module = load_json(root / "module.custom.json")
    after_module = load_json(root / "module.after.json")
    require(before_module["gpu_clean"] is True, "suite GPU before")
    require(custom_module["gpu_clean"] is True, "custom-module GPU state")
    require(after_module["gpu_clean"] is True, "suite GPU after")
    require(
        before_module["loaded_srcversion"] == after_module["loaded_srcversion"],
        "stock UVM restore",
    )
    require(
        custom_module["loaded_srcversion"] != before_module["loaded_srcversion"],
        "custom UVM was not loaded",
    )

    results: dict[str, dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    pids: list[int] = []
    output_hashes: set[str] = set()
    route_hashes: set[str] = set()
    target_hashes: set[str] = set()
    for name in ARM_ORDER:
        arm_root = root / "arms" / name
        require((arm_root / "returncode.txt").read_text().strip() == "0", f"{name}: rc")
        gpu_before = load_json(arm_root / "gpu.before.json")
        gpu_after = load_json(arm_root / "gpu.after.json")
        require(gpu_before["gpu_clean"] is True, f"{name}: GPU before")
        require(gpu_after["gpu_clean"] is True, f"{name}: GPU after")
        result = load_json(arm_root / "result.json")
        results[name] = result
        require(result.get("schema") == "eurosys_e7_independent_cold_arm_v1", f"{name}: schema")
        require(result.get("status") == "ok", f"{name}: {result.get('status')}")
        require(result.get("arm_name") == name, f"{name}: arm identity")
        require(result.get("qualification_only") is True, f"{name}: qualification")
        require(result.get("statistical_repetitions") == 1, f"{name}: repetitions")
        require(result.get("independent_cold_process") is True, f"{name}: cold process")
        require(result.get("model_load_count") == 1, f"{name}: model load count")
        require(result.get("arm_execution_count") == 1, f"{name}: execution count")
        protocol = result["protocol"]
        require(protocol["calibration_requests"] == 0, f"{name}: calibration rerun")
        require(protocol["shift_requests"] == 32, f"{name}: shift count")
        require(protocol["prompt_tokens"] == 128, f"{name}: prompt length")
        require(protocol["output_tokens"] == 32, f"{name}: output length")
        require(protocol["window_size"] == 4, f"{name}: window size")
        require(protocol["dynamic_slots_per_layer"] == 4, f"{name}: dynamic slots")
        require(
            result["inputs"]["frozen_targets_sha256"] == sha256_file(frozen_path),
            f"{name}: target file drift",
        )
        require(
            result["inputs"]["target_source_result_sha256"]
            == frozen["source_result_sha256"],
            f"{name}: target source differs",
        )
        require(result["dynamic_slot_initialization"]["ok"] is True, f"{name}: init")
        require(len(result["dynamic_slot_initialization"]["rounds"]) == 4, f"{name}: init rounds")
        require(result["initial_placement"]["layer_count"] == 48, f"{name}: layers")
        require(result["initial_placement"]["remote_expert_count"] == 192, f"{name}: slots")
        require(result["graph_count_after_load"] == 2, f"{name}: Graph count after load")
        require(result["graph_count_final"] == 2, f"{name}: Graph count final")
        require(result.get("distributed_cleanup") is True, f"{name}: distributed cleanup")
        require(result.get("cleanup_errors") == [], f"{name}: cleanup errors")
        arm = result["arm"]
        require(arm["name"] == name, f"{name}: result name")
        require(arm["request_count"] == 32, f"{name}: requests")
        require(arm["window_count"] == 8 and len(arm["windows"]) == 8, f"{name}: windows")
        require(arm["start_target"] == frozen["targets"][name], f"{name}: start target")
        require(arm["graph_count_before"] == arm["graph_count_after"] == 2, f"{name}: recapture")
        require(arm["setup_transition"]["ok"] is True, f"{name}: setup")
        require(arm["expert_assignments"] > 0, f"{name}: heat assignments")
        require(arm["request_latency_median_ns"] > 0, f"{name}: latency")
        require(arm["output_tokens_per_second"] > 0, f"{name}: throughput")
        require(arm["end_to_end_output_tokens_per_second"] > 0, f"{name}: cycle throughput")
        require(0 <= arm["dynamic_remote_assignment_rate"] <= 1, f"{name}: remote rate")
        require(arm["final_overlap_with_oracle"]["slot_count"] == 192, f"{name}: oracle slots")
        for index, window in enumerate(arm["windows"]):
            require(window["window_index"] == index, f"{name}: window order")
            require(window["request_count"] == 4, f"{name}: window requests")
            require(len(window["requests"]) == 4, f"{name}: request records")
            require(window["maintenance"]["ok"] is True, f"{name}: maintenance")
            require(len(window["heat_epoch"]) == 48, f"{name}: heat epochs")
            require(window["expert_assignments"] > 0, f"{name}: window heat")
        if name == "gpu_heat_cpu_ewma":
            require(arm["adaptive"] is True, "adaptive flag")
            require(arm["online_swap_count"] > 0, "adaptive swaps")
            require(arm["online_maintenance_planned_h2d_bytes"] > 0, "adaptive H2D")
            require(arm["online_maintenance_planned_d2h_bytes"] > 0, "adaptive D2H")
        else:
            require(arm["adaptive"] is False, f"{name}: adaptive flag")
            require(arm["online_swap_count"] == 0, f"{name}: fixed swaps")
            require(arm["start_target"] == arm["final_target"], f"{name}: placement drift")
        pid = int(result["process"]["pid"])
        pids.append(pid)
        output_hashes.add(arm["output_sha256"])
        route_hashes.add(arm["route_counts_sha256"])
        target_hashes.add(result["inputs"]["frozen_targets_sha256"])
        summaries.append(
            {
                "name": name,
                "pid": pid,
                "model_load_seconds": result["load_ns"] / 1e9,
                "request_latency_median_ms": arm["request_latency_median_ns"] / 1e6,
                "request_latency_mean_ms": arm["request_latency_mean_ns"] / 1e6,
                "request_only_output_tokens_per_second": arm["output_tokens_per_second"],
                "cycle_output_tokens_per_second": arm["end_to_end_output_tokens_per_second"],
                "dynamic_remote_assignments": arm["dynamic_remote_assignments"],
                "dynamic_remote_assignment_rate": arm["dynamic_remote_assignment_rate"],
                "total_remote_assignment_rate": arm["total_remote_assignment_rate"],
                "oracle_plan_hit_start": arm["start_overlap_with_oracle"]["plan_hit_rate"],
                "oracle_plan_hit_final": arm["final_overlap_with_oracle"]["plan_hit_rate"],
                "maintenance_ms": arm["maintenance_ns"] / 1e6,
                "setup_swap_count": arm["setup_transition"]["selected_swaps"],
                "online_swap_count": arm["online_swap_count"],
                "online_h2d_bytes": arm["online_maintenance_planned_h2d_bytes"],
                "online_d2h_bytes": arm["online_maintenance_planned_d2h_bytes"],
                "graph_capture_count": arm["graph_count_after"],
            }
        )

    require(len(set(pids)) == 6, "arm processes did not have six distinct PIDs")
    require(len(output_hashes) == 1, "greedy outputs differ across cold processes")
    require(len(route_hashes) == 1, "route traces differ across cold processes")
    require(len(target_hashes) == 1, "target inputs differ across arms")
    arms = {name: results[name]["arm"] for name in ARM_ORDER}
    oracle = arms["oracle"]
    adversarial = arms["hot_adversarial"]
    adaptive = arms["gpu_heat_cpu_ewma"]
    frequency = arms["frequency_cold"]
    require(oracle["final_overlap_with_oracle"]["plan_hit_rate"] == 1.0, "oracle hit")
    require(
        oracle["dynamic_remote_assignments"]
        == min(arms[name]["dynamic_remote_assignments"] for name in FIXED_ARMS),
        "oracle is not the fixed-plan remote-assignment minimum",
    )
    require(
        adversarial["dynamic_remote_assignments"]
        == max(arms[name]["dynamic_remote_assignments"] for name in FIXED_ARMS),
        "adversarial is not the fixed-plan remote-assignment maximum",
    )
    require(
        adaptive["final_overlap_with_oracle"]["plan_hit_rate"]
        > adaptive["start_overlap_with_oracle"]["plan_hit_rate"],
        "adaptive placement did not improve",
    )
    comparisons = {
        "reference_arm": "frequency_cold",
        "adaptive_dynamic_remote_assignment_reduction_fraction": 1.0
        - adaptive["dynamic_remote_assignments"] / frequency["dynamic_remote_assignments"],
        "adaptive_request_median_latency_change_percent": pct_ratio(
            adaptive["request_latency_median_ns"], frequency["request_latency_median_ns"]
        ),
        "adaptive_cycle_throughput_change_percent": pct_ratio(
            adaptive["end_to_end_output_tokens_per_second"],
            frequency["end_to_end_output_tokens_per_second"],
        ),
        "oracle_request_median_latency_change_percent": pct_ratio(
            oracle["request_latency_median_ns"], frequency["request_latency_median_ns"]
        ),
        "oracle_cycle_throughput_change_percent": pct_ratio(
            oracle["end_to_end_output_tokens_per_second"],
            frequency["end_to_end_output_tokens_per_second"],
        ),
    }
    verification = {
        "schema": "eurosys_e7_independent_cold_verification_v1",
        "status": "ok",
        "qualification_only": True,
        "statistical_repetitions_per_arm": 1,
        "arm_count": 6,
        "measured_request_count_total": 32 * 6,
        "calibration_request_count": 0,
        "independent_process_count": 6,
        "all_gpu_boundaries_clean": True,
        "all_graph_counts_stable": True,
        "output_sha256": next(iter(output_hashes)),
        "route_counts_sha256": next(iter(route_hashes)),
        "frozen_targets_sha256": next(iter(target_hashes)),
        "target_source_result_sha256": frozen["source_result_sha256"],
        "arms": summaries,
        "comparisons": comparisons,
        "adaptive_convergence_window": adaptive.get("ewma_convergence_window"),
        "adaptive_convergence_requests": adaptive.get("ewma_convergence_requests"),
        "interpretation_scope": "one qualification observation per arm; no confidence interval",
    }
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
