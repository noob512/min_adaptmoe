#!/usr/bin/env python3
"""Summarize per-step GPU execution from an E13 Nsight Systems export."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def kernel_category(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("fused_moe", "grouped_gemm", "expert")):
        return "moe"
    if any(token in lowered for token in ("attention", "attn", "paged", "flash")):
        return "attention"
    return "other"


def merge_duration(intervals: list[tuple[int, int]]) -> int:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def main() -> int:
    args = parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    connection = sqlite3.connect(args.sqlite)
    strings = dict(connection.execute("SELECT id, value FROM StringIds"))
    kernels = list(
        connection.execute(
            "SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
        )
    )
    elapsed_by_key = {
        (
            str(item["condition"]),
            str(item["phase"]),
            str(item["stage"]),
            int(item["stage_index"]),
        ): float(item["elapsed_ms"])
        for item in result["step_diagnostics"]
    }

    steps: list[dict[str, Any]] = []
    for start, end, text, text_id in connection.execute(
        "SELECT start, end, text, textId FROM NVTX_EVENTS WHERE end IS NOT NULL"
    ):
        name = text or strings.get(text_id, "")
        parts = str(name).split("/")
        if len(parts) != 6 or parts[0] != "e13":
            continue
        _, tier, condition, phase, stage, index_text = parts
        if stage not in {"decode_warmup", "decode_measured", "drain"}:
            continue
        index = int(index_text)
        selected = [item for item in kernels if item[0] >= start and item[1] <= end]
        intervals = [(item[0], item[1]) for item in selected]
        kernel_total_ns = sum(item[1] - item[0] for item in selected)
        busy_union_ns = merge_duration(intervals)
        kernel_span_ns = (
            max(item[1] for item in selected) - min(item[0] for item in selected)
            if selected
            else 0
        )
        categories = {"moe": 0, "attention": 0, "other": 0}
        for kernel_start, kernel_end, name_id in selected:
            categories[kernel_category(strings[name_id])] += kernel_end - kernel_start
        key = (condition, phase, stage, index)
        steps.append(
            {
                "tier": tier,
                "condition": condition,
                "phase": phase,
                "stage": stage,
                "stage_index": index,
                "application_elapsed_ms": elapsed_by_key.get(key),
                "nvtx_host_wall_ms": (end - start) / 1e6,
                "gpu_kernel_count": len(selected),
                "gpu_kernel_total_ms": kernel_total_ns / 1e6,
                "gpu_busy_union_ms": busy_union_ns / 1e6,
                "gpu_kernel_span_ms": kernel_span_ns / 1e6,
                "gpu_inter_kernel_idle_ms": (kernel_span_ns - busy_union_ns) / 1e6,
                "kernel_category_ms": {
                    category: value / 1e6 for category, value in categories.items()
                },
            }
        )

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for step in steps:
        groups.setdefault((step["phase"], step["stage"]), []).append(step)
    summaries = []
    for (phase, stage), items in sorted(groups.items()):
        summaries.append(
            {
                "phase": phase,
                "stage": stage,
                "step_count": len(items),
                "median_nvtx_host_wall_ms": median(
                    [item["nvtx_host_wall_ms"] for item in items]
                ),
                "median_gpu_kernel_total_ms": median(
                    [item["gpu_kernel_total_ms"] for item in items]
                ),
                "median_gpu_kernel_span_ms": median(
                    [item["gpu_kernel_span_ms"] for item in items]
                ),
                "median_gpu_inter_kernel_idle_ms": median(
                    [item["gpu_inter_kernel_idle_ms"] for item in items]
                ),
                "median_moe_kernel_ms": median(
                    [item["kernel_category_ms"]["moe"] for item in items]
                ),
                "median_attention_kernel_ms": median(
                    [item["kernel_category_ms"]["attention"] for item in items]
                ),
            }
        )

    output = {
        "schema": "eurosys_e13_decode_nsys_analysis_v1",
        "status": "ok" if steps else "error",
        "sqlite": str(args.sqlite.resolve()),
        "result": str(args.result.resolve()),
        "step_count": len(steps),
        "steps": steps,
        "summaries": summaries,
        "interpretation": {
            "profiling_perturbs_wall_time": True,
            "safe_uses": [
                "per-kernel CUDA duration",
                "MoE versus attention category attribution",
                "kernel launch span and traced idle-gap diagnosis",
            ],
            "unsafe_uses": [
                "using profiled wall time as the unprofiled Decode latency",
                "claiming a specific GMMU, TLB, or PCIe submechanism without hardware counters",
            ],
        },
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if steps else 1


if __name__ == "__main__":
    raise SystemExit(main())
