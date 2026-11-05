#!/usr/bin/env python3
"""Analyze E13 same-process ABBA Decode diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


TIERS = ("k4", "k6")
CONDITIONS = (
    "a1_direct",
    "b1_switched",
    "b2_switched",
    "a2_direct",
    "s_switch_only",
    "c_target_condition",
)
PHASES = ("medium_b1", "medium_b8")


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("empty median")
    return statistics.median(values)


def load_result(path: Path, *, heavy: bool) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "ok":
        raise RuntimeError(f"incomplete result: {path}")
    if bool(value.get("heavy_diagnostics")) != heavy:
        raise RuntimeError(f"diagnostic mode differs: {path}")
    labels = [item.get("label") for item in value.get("conditions", ())]
    if labels != list(CONDITIONS):
        raise RuntimeError(f"condition order differs: {path}")
    return value


def condition_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["label"]): item for item in result["conditions"]}


def phase_map(condition: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["label"]): item for item in condition["phases"]}


def pcie_window(rows: list[dict[str, str]], phase: dict[str, Any]) -> dict[str, Any]:
    start = float(phase["measurement_unix_started"])
    finish = float(phase["measurement_unix_finished"])
    values = []
    for row in rows:
        try:
            timestamp = float(row["unix_s"])
            rx = float(row["pcie_rx_kib_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if start <= timestamp <= finish:
            values.append(rx)
    duration = max(finish - start, 0.0)
    mean_rx = statistics.fmean(values) if values else None
    return {
        "sample_count": len(values),
        "mean_rx_gib_s": mean_rx / (1024.0**2) if mean_rx is not None else None,
        "peak_rx_gib_s": max(values) / (1024.0**2) if values else None,
        "estimated_rx_bytes": mean_rx * 1024.0 * duration if mean_rx is not None else None,
        "window_seconds": duration,
    }


def diagnostics_for_phase(
    result: dict[str, Any], condition: str, phase: str
) -> list[dict[str, Any]]:
    return [
        item
        for item in result.get("step_diagnostics", ())
        if item.get("condition") == condition
        and item.get("phase") == phase
        and item.get("stage") == "decode_measured"
    ]


def route_signature(steps: list[dict[str, Any]]) -> list[str]:
    return [str(item["route"]["route_sha256"]) for item in steps]


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def plan_by_layer(condition: dict[str, Any]) -> dict[int, set[int]]:
    return {
        int(layer_id): {int(expert_id) for expert_id in expert_ids}
        for layer_id, expert_ids in condition["before"]["plan"]["expert_plan"]
    }


def canonical_plan_signature(condition: dict[str, Any]) -> str:
    canonical = [
        [layer_id, sorted(expert_ids)]
        for layer_id, expert_ids in sorted(plan_by_layer(condition).items())
    ]
    return digest(canonical)


def remote_assignments(
    steps: list[dict[str, Any]], condition: dict[str, Any]
) -> list[dict[str, Any]]:
    remote_plan = plan_by_layer(condition)
    output = []
    for step in steps:
        layers = {}
        total = 0
        for layer_name, layer in step["route"]["layers"].items():
            layer_id = int(layer["layer_id"])
            remote = remote_plan.get(layer_id, set())
            active = [
                pair
                for pair in layer["active_expert_counts"]
                if int(pair[0]) in remote
            ]
            count = sum(int(pair[1]) for pair in active)
            total += count
            layers[layer_name] = active
        output.append(
            {
                "count": total,
                "assignment_sha256": digest(layers),
            }
        )
    return output


def kv_signature(step: dict[str, Any]) -> str:
    tables = step["schedule"]["kv_blocks"]["request_tables"]
    normalized = [tables[key]["groups"] for key in sorted(tables)]
    return digest(normalized)


def kernel_categories(steps: list[dict[str, Any]]) -> dict[str, float] | None:
    profiled = [item.get("kernel_profile") for item in steps if item.get("kernel_profile")]
    if len(profiled) != 1 or not profiled[0].get("ok"):
        return None
    return {
        key: float(value)
        for key, value in profiled[0].get("category_total_us", {}).items()
    }


def ratio(left: float, right: float) -> float:
    return left / right if right else float("inf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    output_path = args.output or root / "analysis.json"
    rows: list[dict[str, Any]] = []
    conclusions: dict[str, Any] = {}

    for tier in TIERS:
        perf_dir = root / f"performance_{tier}"
        heavy_dir = root / f"heavy_{tier}"
        perf = load_result(perf_dir / "result.json", heavy=False)
        heavy = load_result(heavy_dir / "result.json", heavy=True)
        perf_conditions = condition_map(perf)
        heavy_conditions = condition_map(heavy)
        with (perf_dir / "nvml.csv").open(newline="", encoding="utf-8") as stream:
            perf_nvml = list(csv.DictReader(stream))
        tier_flags = {
            "all_routes_equal": True,
            "all_remote_assignment_counts_equal": True,
            "all_kv_block_tables_equal": True,
            "all_plan_signatures_equal": True,
            "steady_decode_pte_counter_changes": False,
        }
        phase_findings: dict[str, Any] = {}
        for phase_name in PHASES:
            direct_values = [
                float(phase_map(perf_conditions[label])[phase_name]["steady_decode_step_ms"]["median"])
                for label in ("a1_direct", "a2_direct")
            ]
            switched_values = [
                float(phase_map(perf_conditions[label])[phase_name]["steady_decode_step_ms"]["median"])
                for label in ("b1_switched", "b2_switched")
            ]
            direct_center = median(direct_values)
            switched_center = median(switched_values)
            baseline = direct_center
            heavy_steps_by_condition = {
                label: diagnostics_for_phase(heavy, label, phase_name)
                for label in CONDITIONS
            }
            if any(len(value) != 16 for value in heavy_steps_by_condition.values()):
                raise RuntimeError(f"{tier}/{phase_name}: incomplete heavy Decode steps")
            route_sigs = {
                label: route_signature(value)
                for label, value in heavy_steps_by_condition.items()
            }
            remote_counts = {
                label: remote_assignments(value, heavy_conditions[label])
                for label, value in heavy_steps_by_condition.items()
            }
            kv_sigs = {
                label: [kv_signature(item) for item in value]
                for label, value in heavy_steps_by_condition.items()
            }
            reference_route = route_sigs["a1_direct"]
            reference_remote = remote_counts["a1_direct"]
            reference_kv = kv_sigs["a1_direct"]
            routes_equal = all(value == reference_route for value in route_sigs.values())
            remote_equal = all(value == reference_remote for value in remote_counts.values())
            kv_equal = all(value == reference_kv for value in kv_sigs.values())
            plan_sigs = {
                label: canonical_plan_signature(heavy_conditions[label])
                for label in CONDITIONS
            }
            plans_equal = len(set(plan_sigs.values())) == 1
            pte_changes = {
                label: [
                    item.get("driver_counter_delta", {})
                    for item in heavy_steps_by_condition[label]
                    if item.get("driver_counter_delta")
                ]
                for label in CONDITIONS
            }
            steady_pte_changed = any(value for value in pte_changes.values())
            kernel = {
                label: kernel_categories(value)
                for label, value in heavy_steps_by_condition.items()
            }
            controls = {}
            for label in CONDITIONS:
                phase_value = phase_map(perf_conditions[label])[phase_name]
                latency = float(phase_value["steady_decode_step_ms"]["median"])
                controls[label] = {
                    "decode_median_ms": latency,
                    "over_abba_direct_center": ratio(latency, baseline),
                    "pcie": pcie_window(perf_nvml, phase_value),
                }
                rows.append({
                    "tier": tier,
                    "phase": phase_name,
                    "condition": label,
                    **controls[label],
                    "route_equal_to_a1": route_sigs[label] == reference_route,
                    "remote_assignments_equal_to_a1": remote_counts[label] == reference_remote,
                    "kv_blocks_equal_to_a1": kv_sigs[label] == reference_kv,
                    "kernel_category_us": kernel[label],
                })
            phase_findings[phase_name] = {
                "direct_condition_medians_ms": direct_values,
                "switched_condition_medians_ms": switched_values,
                "abba_direct_center_ms": direct_center,
                "abba_switched_center_ms": switched_center,
                "switched_over_direct": ratio(switched_center, direct_center),
                "routes_equal_all_conditions": routes_equal,
                "remote_assignment_counts_equal_all_conditions": remote_equal,
                "kv_block_tables_equal_all_conditions": kv_equal,
                "plan_signatures_equal_all_conditions": plans_equal,
                "steady_decode_driver_counter_delta": pte_changes,
                "kernel_category_us": kernel,
                "controls": controls,
            }
            tier_flags["all_routes_equal"] &= routes_equal
            tier_flags["all_remote_assignment_counts_equal"] &= remote_equal
            tier_flags["all_kv_block_tables_equal"] &= kv_equal
            tier_flags["all_plan_signatures_equal"] &= plans_equal
            tier_flags["steady_decode_pte_counter_changes"] |= steady_pte_changed
        conclusions[tier] = {"flags": tier_flags, "phases": phase_findings}

    output = {
        "schema": "eurosys_e13_decode_abba_analysis_v1",
        "status": "ok",
        "run_root": str(root),
        "tiers": conclusions,
        "rows": rows,
        "limitations": [
            "ABBA is one same-process diagnostic sequence, not independent repetition.",
            "NVML PCIe throughput is sampled, so estimated bytes are approximate.",
            "Nsight Compute exposes system-L2 lookup counters on GB202 but no metric named as a hardware TLB counter; driver PTE/TLB transaction counters are reported instead.",
            "The heavy pass enables route counters and profiles one Decode step, so its latency is diagnostic only.",
        ],
    }
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
