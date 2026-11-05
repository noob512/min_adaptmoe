#!/usr/bin/env python3
"""Verify the reduced cold-process KV/expert partition motivation experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


WORKLOADS = ("long_c2", "short_b128")
ARMS = ("fixed_k2", "fixed_k4", "fixed_k6")
TIERS = {
    "fixed_k2": ("k2", 1364, "cold70"),
    "fixed_k4": ("k4", 2730, "cold80"),
    "fixed_k6": ("k6", 4095, "cold89"),
}
EXPECTED = {
    "long_c2": {"batch": 2, "prompt": 26000, "output": 128, "blocks": 3266},
    "short_b128": {"batch": 128, "prompt": 128, "output": 32, "blocks": 1280},
}
BLOCK_BYTES = 1_572_864
BACKUP_BLOCKS = 16


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def performance_row(
    root: Path, workload: str, arm: str, repeat_index: int
) -> dict[str, Any]:
    case_dir = root / "performance" / workload / arm / f"repeat-{repeat_index}"
    result = load_json(case_dir / "result.json")
    measurement = result["segments"][0]["measurement"]
    tier_name, target_blocks, plan_class = TIERS[arm]
    before = result["segments"][0]["audit_before"]
    after = result["segments"][0]["audit_after"]
    returncode = int((case_dir / "returncode.txt").read_text().strip())
    gpu_after = load_json(case_dir / "gpu.after.json")
    kernel_text = (case_dir / "kernel.since.log").read_text(
        encoding="utf-8", errors="replace"
    ).lower()
    expected = EXPECTED[workload]
    checks = {
        "returncode": returncode == 0,
        "result_status": result.get("status") == "ok",
        "scope": result.get("configuration", {}).get("measurement_scope")
        == "partition_motivation",
        "workload": result.get("configuration", {}).get("motivation_workload")
        == workload,
        "not_diagnostic": result.get("configuration", {}).get(
            "motivation_diagnostic"
        )
        is False,
        "eager": result.get("configuration", {}).get("full_separate_graph") is False,
        "shape": (
            int(measurement.get("batch_size", -1)) == expected["batch"]
            and int(measurement.get("prompt_tokens_per_request", -1))
            == expected["prompt"]
            and int(measurement.get("output_tokens_per_request", -1))
            == expected["output"]
            and int(measurement.get("required_kv_blocks", -1)) == expected["blocks"]
        ),
        "tier_fixed": (
            result["segments"][0].get("tier") == tier_name
            and int(before["kv"].get("current_target_blocks", -1)) == target_blocks
            and int(after["kv"].get("current_target_blocks", -1)) == target_blocks
            and before["plan"].get("active_plan_class") == plan_class
            and after["plan"].get("active_plan_class") == plan_class
        ),
        "gpu_clean": gpu_after.get("gpu_clean") is True,
        "no_gpu_fault": not any(
            marker in kernel_text
            for marker in (
                "xid",
                "mmu fault",
                "pte fault",
                "pde fault",
                "illegal address",
            )
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"{case_dir}: failed checks {checks}")
    status = str(measurement.get("status"))
    has_metrics = finite(measurement.get("output_token_throughput_per_second"))
    if status == "ok" and not has_metrics:
        raise ValueError(f"{case_dir}: fully admitted cell lacks performance")
    if has_metrics:
        for field in (
            "output_token_throughput_per_second",
            "steady_decode_output_tokens_per_second",
            "max_observed_active_kv_gib",
        ):
            if not finite(measurement.get(field)):
                raise ValueError(f"{case_dir}: missing {field}")
        steady = measurement.get("steady_decode_after8_ms") or {}
        if int(steady.get("count", 0)) < 16:
            raise ValueError(f"{case_dir}: too few steady Decode samples")
    return {
        "kind": "performance",
        "workload": workload,
        "arm": arm,
        "tier": tier_name,
        "repeat_index": repeat_index,
        "seed": int(result["seed"]),
        "target_blocks": target_blocks,
        "target_gib": target_blocks * BLOCK_BYTES / 2**30,
        "usable_blocks": target_blocks - BACKUP_BLOCKS,
        "required_kv_blocks": int(measurement["required_kv_blocks"]),
        "required_kv_gib": float(measurement["required_kv_gib"]),
        "observed_peak_gpu_kv_gib": (
            float(measurement["max_observed_active_kv_gib"])
            if has_metrics
            else None
        ),
        "batch_size": expected["batch"],
        "prompt_tokens_per_request": expected["prompt"],
        "output_tokens_per_request": expected["output"],
        "status": status,
        "max_running": int(measurement.get("max_scheduler_running_requests", 0)),
        "max_waiting": int(measurement.get("max_scheduler_waiting_requests", 0)),
        "tpot_p50_ms": (
            float(measurement["steady_decode_after8_ms"]["p50"])
            if has_metrics
            else None
        ),
        "tpot_p95_ms": (
            float(measurement["steady_decode_after8_ms"]["p95"])
            if has_metrics
            else None
        ),
        "output_tokens_per_s": (
            float(measurement["output_token_throughput_per_second"])
            if has_metrics
            else None
        ),
        "steady_decode_output_tokens_per_s": (
            float(measurement["steady_decode_output_tokens_per_second"])
            if has_metrics
            else None
        ),
        "output_hash": (
            str(measurement["output_token_ids_sha256"]) if has_metrics else None
        ),
        "source_result": str((case_dir / "result.json").relative_to(root)),
        "validity": (
            "valid_full_admission"
            if status == "ok"
            else "valid_capacity_limited_not_ranked"
            if status == "kv_capacity_limited"
            else "kv_capacity_exceeded_not_numeric"
        ),
    }


def diagnostic_row(root: Path, workload: str, arm: str) -> dict[str, Any]:
    case_dir = root / "diagnostic" / workload / arm
    result = load_json(case_dir / "result.json")
    measurement = result["segments"][0]["measurement"]
    route = measurement.get("route_metrics") or {}
    if (
        int((case_dir / "returncode.txt").read_text().strip()) != 0
        or result.get("status") != "ok"
        or result.get("configuration", {}).get("motivation_diagnostic") is not True
        or measurement.get("status") != "ok"
        or int(route.get("sampled_decode_steps", 0)) != 16
    ):
        raise ValueError(f"invalid diagnostic case: {case_dir}")
    gpu_after = load_json(case_dir / "gpu.after.json")
    if gpu_after.get("gpu_clean") is not True:
        raise ValueError(f"diagnostic GPU not clean: {case_dir}")
    return {
        "kind": "diagnostic",
        "workload": workload,
        "arm": arm,
        "sampled_decode_steps": 16,
        "unique_active_experts_per_layer_step": float(
            route["unique_active_experts_per_layer_step"]["median"]
        ),
        "resident_hit_rate": float(route["resident_hit_rate"]["median"]),
        "remote_bytes_per_step": float(
            route["logical_remote_bytes_per_step"]["median"]
        ),
        "total_assignments_per_step": float(
            route["total_assignments_per_step"]["median"]
        ),
        "route_sha256": str(route["route_sha256"]),
        "source_result": str((case_dir / "result.json").relative_to(root)),
        "validity": "diagnostic_counter_enabled_not_used_for_performance",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    output = args.output or root / "verification.json"

    with (root / "execution_status.tsv").open(newline="") as handle:
        execution = list(csv.DictReader(handle, delimiter="\t"))
    if len(execution) != 14 or any(int(row["returncode"]) != 0 for row in execution):
        raise ValueError("execution matrix is incomplete")

    rows = [
        performance_row(root, workload, arm, repeat_index)
        for workload in WORKLOADS
        for arm in ARMS
        for repeat_index in (0, 1)
    ]
    diagnostics = [
        diagnostic_row(root, "long_c2", "fixed_k6"),
        diagnostic_row(root, "short_b128", "fixed_k2"),
    ]

    def matching(workload: str, arm: str) -> list[dict[str, Any]]:
        return [row for row in rows if row["workload"] == workload and row["arm"] == arm]

    for arm in ARMS:
        values = matching("short_b128", arm)
        if not all(row["status"] == "ok" and row["max_running"] == 128 for row in values):
            raise ValueError(f"Short-B128 not fully feasible for {arm}")
    if not all(
        row["status"] == "kv_capacity_limited" and row["max_running"] == 1
        for row in matching("long_c2", "fixed_k2")
    ):
        raise ValueError("Long-C2 K2 capacity classification differs")
    if not all(
        row["status"] == "kv_capacity_limited" and row["max_running"] == 1
        for row in matching("long_c2", "fixed_k4")
    ):
        raise ValueError("Long-C2 K4 capacity classification differs")
    if not all(
        row["status"] == "ok" and row["max_running"] == 2
        for row in matching("long_c2", "fixed_k6")
    ):
        raise ValueError("Long-C2 K6 feasibility differs")

    for workload in WORKLOADS:
        hashes = {
            row["output_hash"]
            for row in rows
            if row["workload"] == workload and row["output_hash"] is not None
        }
        if len(hashes) != 1:
            raise ValueError(f"{workload}: output hashes differ: {hashes}")

    medians = {
        workload: {
            arm: statistics.median(
                float(row["output_tokens_per_s"])
                for row in matching(workload, arm)
                if row["status"] == "ok"
            )
            for arm in ARMS
            if any(row["status"] == "ok" for row in matching(workload, arm))
        }
        for workload in WORKLOADS
    }
    short = medians["short_b128"]
    if not (short["fixed_k2"] > short["fixed_k4"] > short["fixed_k6"]):
        raise ValueError(f"Short-B128 crossover differs: {short}")
    if not (
        diagnostics[1]["unique_active_experts_per_layer_step"]
        > diagnostics[0]["unique_active_experts_per_layer_step"]
    ):
        raise ValueError("expert working-set direction differs")

    payload = {
        "schema": "eurosys_e13_partition_motivation_verification_v1",
        "status": "ok",
        "protocol": {
            "model": "Qwen3-30B-A3B-FP8",
            "performance_cold_processes_per_cell": 2,
            "performance_processes": 12,
            "diagnostic_processes": 2,
            "total_cold_processes": 14,
            "decode_warmup_steps_excluded": 8,
            "performance_counter_instrumentation": False,
            "diagnostic_counter_instrumentation": True,
            "graph_mode": "NONE/eager",
            "autoscale": False,
        },
        "rows": rows,
        "diagnostics": diagnostics,
        "median_output_tokens_per_s": medians,
        "claims": {
            "long_c2_requires_gib": EXPECTED["long_c2"]["blocks"]
            * BLOCK_BYTES
            / 2**30,
            "short_b128_requires_gib": EXPECTED["short_b128"]["blocks"]
            * BLOCK_BYTES
            / 2**30,
            "long_k2_max_running": 1,
            "long_k4_max_running": 1,
            "long_k6_max_running": 2,
            "short_throughput_order": ["fixed_k2", "fixed_k4", "fixed_k6"],
        },
    }
    atomic_text(output, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    summary = [
        "# E13 KV/expert fixed-partition motivation",
        "",
        "Two independent cold processes per performance cell; diagnostic routing "
        "counters were collected in two separate cold processes and were not used "
        "for performance.",
        "",
        "| Workload | Plan | Status | max running | output tok/s (two runs; median) |",
        "|---|---|---|---:|---:|",
    ]
    for workload in WORKLOADS:
        for arm in ARMS:
            values = matching(workload, arm)
            numeric = [float(row["output_tokens_per_s"]) for row in values if row["status"] == "ok"]
            display = (
                f"{' / '.join(f'{value:.4f}' for value in numeric)}; "
                f"{statistics.median(numeric):.4f}"
                if numeric
                else "N/A"
            )
            summary.append(
                f"| {workload} | {arm} | {values[0]['status']} | "
                f"{values[0]['max_running']}/{values[0]['batch_size']} | {display} |"
            )
    atomic_text(root / "summary.md", "\n".join(summary) + "\n")
    print(f"verification: ok; wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
