#!/usr/bin/env python3
"""Verify and summarize the matched AdaptMoE/native Prefill rerun."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any


CASE_RE = re.compile(
    r"^b(?P<block>[0-9]+)_(?P<workload>.+)_b(?P<batch>1|8)_"
    r"(?P<mechanism>adaptmoe|native_copy)$"
)
WORKLOAD_LABELS = {
    "short_dialogue": "Short 128",
    "ordinary_chat": "Ordinary 512",
    "medium_context": "Medium 2048",
    "rag_summary": "RAG 8192",
    "long_generation": "Long 512",
}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile input")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values),
        "p10": percentile(values, 0.10),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p90": percentile(values, 0.90),
        "max": max(values),
        "samples": values,
    }


def bootstrap_median_ci(values: list[float], *, samples: int = 20_000) -> list[float]:
    rng = random.Random(0)
    size = len(values)
    medians = [
        statistics.median(values[rng.randrange(size)] for _ in range(size))
        for _ in range(samples)
    ]
    return [percentile(medians, 0.025), percentile(medians, 0.975)]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt_interval(item: dict[str, Any]) -> str:
    return f"{item['median']:.2f} [{item['p10']:.2f}, {item['p90']:.2f}]"


def fmt_ratio(item: dict[str, Any]) -> str:
    interval = item["bootstrap_median_ci95"]
    if interval is None:
        return f"{item['median']:.3f}× (diagnostic)"
    low, high = interval
    return f"{item['median']:.3f}× [{low:.3f}, {high:.3f}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    configuration = json.loads((root / "configuration.json").read_text())
    process_blocks = int(configuration["process_blocks"])
    conditioning_repeats = int(configuration["conditioning_repeats"])
    measurement_repeats = int(configuration["measurement_repeats"])
    workloads = [
        value for value in WORKLOAD_LABELS
        if value in configuration["workload_filter"].split(",")
    ]
    batches = [
        int(value) for value in configuration["batch_filter"].split(",")
    ]
    mechanisms = configuration["arm_filter"].split(",")

    errors: list[str] = []
    records: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for case_dir in sorted((root / "cases").iterdir()):
        match = CASE_RE.match(case_dir.name)
        if match is None:
            errors.append(f"unexpected case directory: {case_dir.name}")
            continue
        key = (
            match.group("workload"),
            int(match.group("batch")),
            match.group("mechanism"),
            int(match.group("block")),
        )
        result_path = case_dir / "result.json"
        if not result_path.is_file():
            errors.append(f"missing result: {case_dir.name}")
            continue
        result = json.loads(result_path.read_text())
        cycles = result.get("cycles", [])
        conditioning = [item for item in cycles if item.get("role") == "conditioning"]
        measurement = [item for item in cycles if item.get("role") == "measurement"]
        if result.get("status") != "ok":
            errors.append(f"non-ok result: {case_dir.name}")
        if len(conditioning) != conditioning_repeats:
            errors.append(f"conditioning count differs: {case_dir.name}")
        if len(measurement) != measurement_repeats:
            errors.append(f"measurement count differs: {case_dir.name}")
        if any(
            item.get("graph_wrapper_delta_prefill", {}).get("full_replay") != 1
            for item in cycles
        ):
            errors.append(f"non-FULL replay cycle: {case_dir.name}")
        hashes = {item.get("output_token_ids_sha256") for item in cycles}
        if len(hashes) != 1 or None in hashes:
            errors.append(f"within-process output hashes differ: {case_dir.name}")
        if not cycles:
            continue
        if key[2] == "adaptmoe":
            actual_bytes = int(
                result["diagnostics"]["after_load"]["model"]
                ["selected_parameter_bytes_per_forward"]
            )
        else:
            actual_bytes = int(result["manifest"]["offloaded_parameter_bytes"])
        records[key] = {
            "case_dir": str(case_dir),
            "result_sha256": sha256_file(result_path),
            "first_ms": float(cycles[0]["prefill_step_ms"]),
            "measurement_samples_ms": [
                float(item["prefill_step_ms"]) for item in measurement
            ],
            "steady_block_median_ms": statistics.median(
                float(item["prefill_step_ms"]) for item in measurement
            ),
            "output_hash": next(iter(hashes)),
            "actual_bytes": actual_bytes,
            "arm": result["arm"],
        }

    expected_keys = {
        (workload, batch, mechanism, block)
        for workload in workloads
        for batch in batches
        for mechanism in mechanisms
        for block in range(process_blocks)
    }
    missing = sorted(expected_keys - set(records))
    extra = sorted(set(records) - expected_keys)
    errors.extend(f"missing case: {item}" for item in missing)
    errors.extend(f"extra case: {item}" for item in extra)

    cells: list[dict[str, Any]] = []
    if set(mechanisms) == {"adaptmoe", "native_copy"}:
        for workload in workloads:
            for batch in batches:
                paired = []
                for block in range(process_blocks):
                    adapt = records.get((workload, batch, "adaptmoe", block))
                    native = records.get((workload, batch, "native_copy", block))
                    if adapt is None or native is None:
                        continue
                    if adapt["output_hash"] != native["output_hash"]:
                        errors.append(
                            f"cross-arm output hash differs: {workload}/B{batch}/b{block}"
                        )
                    paired.append((block, adapt, native))
                if not paired:
                    continue

                adapt_bytes = {item[1]["actual_bytes"] for item in paired}
                native_bytes = {item[2]["actual_bytes"] for item in paired}
                if len(adapt_bytes) != 1 or len(native_bytes) != 1:
                    errors.append(f"bytes vary across blocks: {workload}/B{batch}")
                adapt_byte_value = next(iter(adapt_bytes))
                native_byte_value = next(iter(native_bytes))
                byte_difference = abs(adapt_byte_value - native_byte_value) / native_byte_value
                if byte_difference >= 0.001:
                    errors.append(f"byte mismatch >=0.1%: {workload}/B{batch}")

                cold_adapt = [item[1]["first_ms"] for item in paired]
                cold_native = [item[2]["first_ms"] for item in paired]
                steady_adapt = [item[1]["steady_block_median_ms"] for item in paired]
                steady_native = [item[2]["steady_block_median_ms"] for item in paired]
                cold_ratios = [native / adapt for adapt, native in zip(cold_adapt, cold_native)]
                steady_ratios = [
                    native / adapt for adapt, native in zip(steady_adapt, steady_native)
                ]
                cold_deltas = [native - adapt for adapt, native in zip(cold_adapt, cold_native)]
                steady_deltas = [
                    native - adapt for adapt, native in zip(steady_adapt, steady_native)
                ]
                cold_ratio_summary = summarize(cold_ratios)
                steady_ratio_summary = summarize(steady_ratios)
                cold_ratio_summary["bootstrap_median_ci95"] = (
                    bootstrap_median_ci(cold_ratios)
                    if len(cold_ratios) >= 5 else None
                )
                steady_ratio_summary["bootstrap_median_ci95"] = (
                    bootstrap_median_ci(steady_ratios)
                    if len(steady_ratios) >= 5 else None
                )
                cells.append(
                    {
                        "workload": workload,
                        "label": WORKLOAD_LABELS[workload],
                        "batch": batch,
                        "paired_blocks": len(paired),
                        "adaptmoe_bytes": adapt_byte_value,
                        "native_copy_bytes": native_byte_value,
                        "byte_difference_fraction": byte_difference,
                        "output_hash": paired[0][1]["output_hash"],
                        "cold": {
                            "adaptmoe_ms": summarize(cold_adapt),
                            "native_copy_ms": summarize(cold_native),
                            "native_over_adaptmoe": cold_ratio_summary,
                            "native_minus_adaptmoe_ms": summarize(cold_deltas),
                            "adaptmoe_faster_blocks": sum(value > 0 for value in cold_deltas),
                        },
                        "steady": {
                            "statistical_unit": "per-process median of measurement repeats",
                            "adaptmoe_ms": summarize(steady_adapt),
                            "native_copy_ms": summarize(steady_native),
                            "native_over_adaptmoe": steady_ratio_summary,
                            "native_minus_adaptmoe_ms": summarize(steady_deltas),
                            "adaptmoe_faster_blocks": sum(value > 0 for value in steady_deltas),
                        },
                    }
                )

    kernel_log = root / "kernel.since.log"
    kernel_fault_lines: list[str] = []
    if kernel_log.is_file():
        pattern = re.compile(
            r"CUDA error|illegal memory access|\bXid\b|MMU fault|"
            r"PDE.*fault|PTE.*fault|GPU page fault|GPU has fallen off|"
            r"Segmentation fault|Traceback \(most recent call last\)",
            re.IGNORECASE,
        )
        kernel_fault_lines = [
            line for line in kernel_log.read_text(errors="replace").splitlines()
            if pattern.search(line)
        ]
        if kernel_fault_lines:
            errors.append("kernel fault signatures found")

    gpu_after = root / "gpu.after.csv"
    gpu_used_mib = None
    if gpu_after.is_file():
        fields = [value.strip() for value in gpu_after.read_text().split(",")]
        if len(fields) >= 2:
            gpu_used_mib = int(fields[1])
            if gpu_used_mib != 0:
                errors.append(f"GPU not clean after run: {gpu_used_mib} MiB")

    summary = {
        "schema": "eurosys_e3_prefill_transport_fair_summary_v1",
        "status": "ok" if not errors else "error",
        "configuration": configuration,
        "verification": {
            "expected_case_count": len(expected_keys),
            "observed_case_count": len(records),
            "all_cycles_full_graph": not any("FULL replay" in item for item in errors),
            "all_bytes_within_0_1_percent": not any("byte mismatch" in item for item in errors),
            "all_output_hashes_match": not any("output hash" in item for item in errors),
            "kernel_fault_signature_count": len(kernel_fault_lines),
            "gpu_used_mib_after": gpu_used_mib,
            "errors": errors,
        },
        "cells": cells,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    cold_wins = sum(item["cold"]["adaptmoe_faster_blocks"] for item in cells)
    steady_wins = sum(item["steady"]["adaptmoe_faster_blocks"] for item in cells)
    total_pairs = sum(item["paired_blocks"] for item in cells)
    lines = [
        "# E3 Prefill transport fair rerun",
        "",
        f"Status: `{summary['status']}`.",
        "",
        (
            f"Each cell has {process_blocks} independent process blocks; each block "
            f"runs {conditioning_repeats} complete real conditioning batch and "
            f"{measurement_repeats} measured batches. Both arms use TRITON_ATTN and "
            "FULL_SEPARATE CUDA Graph."
        ),
        "",
        "Cold values are first-real-request medians across process blocks. Brackets are "
        "P10/P90. A block-bootstrap 95% CI is emitted only with at least five "
        "independent blocks; smaller runs are explicitly diagnostic.",
        "",
        "| Request | B | AdaptMoE ms | Native copy ms | Native/AdaptMoE | AdaptMoE wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in cells:
        lines.append(
            f"| {item['label']} | {item['batch']} | "
            f"{fmt_interval(item['cold']['adaptmoe_ms'])} | "
            f"{fmt_interval(item['cold']['native_copy_ms'])} | "
            f"{fmt_ratio(item['cold']['native_over_adaptmoe'])} | "
            f"{item['cold']['adaptmoe_faster_blocks']}/{item['paired_blocks']} |"
        )
    lines.extend(
        [
            "",
            "Steady values first take the median of measured requests inside each process, "
            "then summarize those independent block medians.",
            "",
            "| Request | B | AdaptMoE ms | Native copy ms | Native/AdaptMoE | AdaptMoE wins |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in cells:
        lines.append(
            f"| {item['label']} | {item['batch']} | "
            f"{fmt_interval(item['steady']['adaptmoe_ms'])} | "
            f"{fmt_interval(item['steady']['native_copy_ms'])} | "
            f"{fmt_ratio(item['steady']['native_over_adaptmoe'])} | "
            f"{item['steady']['adaptmoe_faster_blocks']}/{item['paired_blocks']} |"
        )
    lines.extend(
        [
            "",
            f"Cold paired wins: {cold_wins}/{total_pairs}; steady paired wins: "
            f"{steady_wins}/{total_pairs}.",
            "",
            f"Verification errors: {len(errors)}; kernel fault signatures: "
            f"{len(kernel_fault_lines)}; final GPU used: {gpu_used_mib} MiB.",
        ]
    )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "status": summary["status"],
        "cells": len(cells),
        "paired_blocks": total_pairs,
        "cold_adaptmoe_wins": cold_wins,
        "steady_adaptmoe_wins": steady_wins,
        "summary": str(root / "summary.json"),
    }, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
