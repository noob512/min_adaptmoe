#!/usr/bin/env python3
"""Verify and summarize one E7 workload-shift qualification artifact."""

from __future__ import annotations

import argparse
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
    require(isinstance(value, dict), f"{path}: expected object")
    return value


def main() -> int:
    args = parse_args()
    root = args.run_root.resolve()
    result = load_json(root / "result.json")
    require(result.get("schema") == "eurosys_e7_workload_shift_qualification_v1", "schema")
    require(result.get("status") == "ok", f"runner status: {result.get('status')}")
    require(result.get("qualification_only") is True, "qualification label")
    require(result.get("statistical_repetitions") == 1, "repetition count")
    protocol = result["protocol"]
    require(protocol["pre_shift_requests"] == 32, "pre-shift request count")
    require(protocol["shift_requests_per_arm"] == 32, "shift request count")
    require(protocol["prompt_tokens"] == 128, "prompt length")
    require(protocol["output_tokens"] == 32, "output length")
    require(protocol["window_size"] == 4, "window size")
    require(protocol["dynamic_slots_per_layer"] == 4, "dynamic slots")
    require(protocol["maintenance_mode"].startswith("synchronous"), "maintenance mode")
    require(result["calibration"]["request_count"] == 32, "calibration execution")
    calibration_workers = result["calibration"]["heat"]
    require(len(calibration_workers) == 1, "calibration worker count")
    require(
        all(
            layer["counter_mode"] == "graph_stable_synchronous"
            for layer in calibration_workers[0]["layers"].values()
        ),
        "Graph-stable heat counter mode",
    )
    require(result["initial_placement"]["layer_count"] == 48, "layer count")
    require(result["initial_placement"]["remote_expert_count"] == 192, "initial slots")
    require(result["initial_dynamic_remote_region_bytes"] > 0, "dynamic remote bytes")
    require(result["static_alias_selected_bytes"] > 0, "static alias bytes")
    require(len(result["dynamic_slot_initialization"]["rounds"]) == 4, "slot rounds")
    require(result["dynamic_slot_initialization"]["ok"] is True, "slot initialization")
    require(result["byte_evidence"]["per_arm_remote_demand_bytes"]["available"] is False, "byte limitation")

    arms = result.get("arms")
    require(isinstance(arms, list) and len(arms) == 6, "arm count")
    require([arm["name"] for arm in arms] == ARM_ORDER, "arm order")
    reference_output = result["reference_output_sha256"]
    reference_route = result["reference_route_counts_sha256"]
    summaries: list[dict[str, Any]] = []
    for arm in arms:
        name = arm["name"]
        require(arm["request_count"] == 32, f"{name}: request count")
        require(arm["window_count"] == 8, f"{name}: window count")
        require(len(arm["windows"]) == 8, f"{name}: windows")
        require(arm["output_sha256"] == reference_output, f"{name}: output parity")
        require(arm["route_counts_sha256"] == reference_route, f"{name}: route parity")
        require(arm["graph_count_before"] == arm["graph_count_after"], f"{name}: Graph recapture")
        require(arm["setup_transition"]["ok"] is True, f"{name}: setup transition")
        require(arm["expert_assignments"] > 0, f"{name}: assignments")
        require(0 <= arm["dynamic_remote_assignment_rate"] <= 1, f"{name}: dynamic rate")
        require(0 <= arm["total_remote_assignment_rate"] <= 1, f"{name}: total rate")
        require(arm["request_latency_median_ns"] > 0, f"{name}: latency")
        require(arm["output_tokens_per_second"] > 0, f"{name}: throughput")
        require(arm["maintenance_ns"] > 0, f"{name}: maintenance")
        require(arm["dynamic_remote_plan_bytes"] > 0, f"{name}: plan bytes")
        require(
            arm["final_overlap_with_oracle"]["slot_count"] == 192,
            f"{name}: oracle slot count",
        )
        for window_index, window in enumerate(arm["windows"]):
            require(window["window_index"] == window_index, f"{name}: window ordering")
            require(window["request_count"] == 4, f"{name}: window request count")
            require(len(window["requests"]) == 4, f"{name}: request records")
            require(window["maintenance"]["ok"] is True, f"{name}: maintenance")
            require(len(window["heat_epoch"]) == 48, f"{name}: heat epochs")
            require(window["expert_assignments"] > 0, f"{name}: window heat")
        summaries.append(
            {
                "name": name,
                "request_latency_median_ms": arm["request_latency_median_ns"] / 1e6,
                "output_tokens_per_second": arm["output_tokens_per_second"],
                "dynamic_remote_assignments": arm["dynamic_remote_assignments"],
                "dynamic_remote_assignment_rate": arm["dynamic_remote_assignment_rate"],
                "total_remote_assignment_rate": arm["total_remote_assignment_rate"],
                "plan_hit_rate_final": arm["final_overlap_with_oracle"]["plan_hit_rate"],
                "mean_layer_jaccard_final": arm["final_overlap_with_oracle"]["mean_layer_jaccard"],
                "maintenance_ms": arm["maintenance_ns"] / 1e6,
                "setup_swap_count": arm["setup_transition"]["selected_swaps"],
                "online_swap_count": int(arm.get("online_swap_count", 0)),
                "online_h2d_bytes": int(arm["online_maintenance_planned_h2d_bytes"]),
                "online_d2h_bytes": int(arm["online_maintenance_planned_d2h_bytes"]),
            }
        )

    by_name = {arm["name"]: arm for arm in arms}
    oracle = by_name["oracle"]
    adversarial = by_name["hot_adversarial"]
    adaptive = by_name["gpu_heat_cpu_ewma"]
    require(oracle["final_overlap_with_oracle"]["plan_hit_rate"] == 1.0, "oracle hit rate")
    require(
        oracle["dynamic_remote_assignments"]
        == min(
            arm["dynamic_remote_assignments"]
            for arm in arms
            if arm["name"] != "gpu_heat_cpu_ewma"
        ),
        "oracle assignment lower bound",
    )
    require(
        adversarial["dynamic_remote_assignments"]
        == max(
            arm["dynamic_remote_assignments"]
            for arm in arms
            if arm["name"] != "gpu_heat_cpu_ewma"
        ),
        "adversarial assignment upper bound",
    )
    require(adaptive["online_swap_count"] > 0, "adaptive swaps")
    require(adaptive["online_maintenance_planned_h2d_bytes"] > 0, "adaptive H2D bytes")
    require(adaptive["online_maintenance_planned_d2h_bytes"] > 0, "adaptive D2H bytes")
    require("performance_recovery" in adaptive, "adaptive recovery metric")
    require(result["graph_count_after_load"] == result["graph_count_final"], "global Graph recapture")
    require(result.get("distributed_cleanup") is True, "distributed cleanup")
    require(result.get("cleanup_errors") == [], "cleanup errors")
    require(load_json(root / "gpu.before.json")["gpu_clean"] is True, "GPU before")
    require(load_json(root / "gpu.after.json")["gpu_clean"] is True, "GPU after")
    before_module = load_json(root / "module.before.json")
    custom_module = load_json(root / "module.custom.json")
    after_module = load_json(root / "module.after.json")
    require(before_module["loaded_srcversion"] == after_module["loaded_srcversion"], "stock UVM restore")
    require(custom_module["loaded_srcversion"] != before_module["loaded_srcversion"], "custom UVM load")

    verification = {
        "schema": "eurosys_e7_workload_shift_verification_v1",
        "status": "ok",
        "qualification_only": True,
        "arm_count": len(arms),
        "request_count_total": 32 + 32 * len(arms),
        "statistical_repetitions": 1,
        "maintenance_mode": protocol["maintenance_mode"],
        "actual_remote_demand_bytes_available": False,
        "dynamic_remote_region_bytes": result["initial_dynamic_remote_region_bytes"],
        "static_alias_selected_bytes": result["static_alias_selected_bytes"],
        "graph_capture_count": result["graph_count_final"],
        "ewma_convergence_window": adaptive["ewma_convergence_window"],
        "ewma_convergence_requests": adaptive["ewma_convergence_requests"],
        "adaptive_performance_recovery": adaptive["performance_recovery"],
        "arms": summaries,
    }
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
