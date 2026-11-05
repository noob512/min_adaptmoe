#!/usr/bin/env python3
"""Aggregate E13 cold-process Decode backing and PCIe diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


PHASES = ("medium_b1", "medium_b8")
CHUNK_BYTES = {"4k": 4 << 10, "64k": 64 << 10, "2m": 2 << 20}


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def window_rows(rows: list[dict[str, str]], start: float, finish: float) -> list[dict[str, str]]:
    return [row for row in rows if start <= float(row["unix_s"]) <= finish]


def values(rows: list[dict[str, str]], key: str) -> list[float]:
    output = []
    for row in rows:
        try:
            if row.get(key) not in (None, ""):
                output.append(float(row[key]))
        except ValueError:
            pass
    return output


def stats(items: list[float]) -> dict[str, float] | None:
    if not items:
        return None
    return {
        "mean": statistics.fmean(items),
        "median": statistics.median(items),
        "min": min(items),
        "max": max(items),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    output_path = args.output or root / "analysis.json"
    cases: list[dict[str, Any]] = []

    for tier in ("k4", "k6"):
        for trial in (1, 2, 3):
            case_dir = root / f"{tier}_trial{trial}"
            result = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
            if result.get("status") != "ok":
                raise RuntimeError(f"incomplete case: {case_dir}")
            rows = list(csv.DictReader((case_dir / "nvml.csv").open(encoding="utf-8")))
            condition = result["conditions"][0]
            phase_map = {item["label"]: item for item in condition["phases"]}
            counters = result["final_driver_counters"]
            chunks = {
                size: int(counters[f"cpu_chunk_live_{size}"])
                for size in CHUNK_BYTES
            }
            chunk_bytes = {
                size: chunks[size] * CHUNK_BYTES[size]
                for size in CHUNK_BYTES
            }
            total_chunk_bytes = sum(chunk_bytes.values())
            phases: dict[str, Any] = {}
            for phase_name in PHASES:
                phase = phase_map[phase_name]
                measured_steps = [
                    float(item["elapsed_ms"])
                    for item in result["step_diagnostics"]
                    if item.get("condition") == "a1_direct"
                    and item.get("phase") == phase_name
                    and item.get("stage") == "decode_measured"
                ]
                selected = window_rows(
                    rows,
                    float(phase["measurement_unix_started"]),
                    float(phase["measurement_unix_finished"]),
                )
                phases[phase_name] = {
                    "decode_median_ms": float(phase["steady_decode_step_ms"]["median"]),
                    "first_half_median_ms": median(measured_steps[:8]),
                    "second_half_median_ms": median(measured_steps[8:]),
                    "sm_clock_mhz": stats(values(selected, "sm_clock_mhz")),
                    "memory_clock_mhz": stats(values(selected, "memory_clock_mhz")),
                    "pstate": sorted(set(values(selected, "pstate"))),
                    "pcie_generation": sorted(set(values(selected, "pcie_gen"))),
                    "pcie_width": sorted(set(values(selected, "pcie_width"))),
                    "pcie_rx_gib_s": stats(
                        [value / (1024.0**2) for value in values(selected, "pcie_rx_kib_s")]
                    ),
                    "gpu_memory_used_mib": stats(values(selected, "memory_used_mib")),
                }
            cases.append({
                "tier": tier,
                "trial": trial,
                "directory": str(case_dir),
                "cpu_backing_chunks": chunks,
                "cpu_backing_bytes": chunk_bytes,
                "cpu_backing_byte_fraction": {
                    size: chunk_bytes[size] / total_chunk_bytes for size in CHUNK_BYTES
                },
                "phases": phases,
            })

    signatures = {
        tuple(case["cpu_backing_chunks"][size] for size in CHUNK_BYTES)
        for case in cases
    }
    latency_ranges = {}
    for tier in ("k4", "k6"):
        latency_ranges[tier] = {}
        for phase in PHASES:
            samples = [
                case["phases"][phase]["decode_median_ms"]
                for case in cases
                if case["tier"] == tier
            ]
            latency_ranges[tier][phase] = {
                "samples_ms": samples,
                "min_ms": min(samples),
                "max_ms": max(samples),
                "max_over_min": max(samples) / min(samples),
            }

    output = {
        "schema": "eurosys_e13_decode_backing_analysis_v1",
        "status": "ok",
        "run_root": str(root),
        "case_count": len(cases),
        "cases": cases,
        "invariants": {
            "cpu_backing_chunk_signature_count": len(signatures),
            "cpu_backing_chunk_counts_identical": len(signatures) == 1,
            "cpu_backing_chunk_signatures": [list(value) for value in sorted(signatures)],
        },
        "latency_ranges": latency_ranges,
        "interpretation": [
            "Identical CPU backing chunk counts with different Decode latency exclude chunk-size composition as the cause.",
            "A within-process slow-to-fast transition with unchanged backing and GPU residency excludes cold-process physical layout as a sufficient cause.",
            "Stable clocks, P-state, and PCIe link exclude DVFS and link retraining when those fields are available.",
        ],
    }
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
