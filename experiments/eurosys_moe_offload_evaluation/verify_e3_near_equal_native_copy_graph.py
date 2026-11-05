#!/usr/bin/env python3
"""Verify near-equal Native copy FULL_SEPARATE Graph against E3 Ours."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from verify_e3_near_equal_offload_plus_one_expert import (
    EXPERT_MAIN_BYTES,
    FATAL,
    WIDTHS,
    WORKLOADS,
    graph_checks,
    load_json,
    manifest_stats,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    checks: dict[str, bool] = {}
    errors: dict[str, str] = {}
    cells: list[dict[str, Any]] = []

    execution = {
        fields[0]: int(fields[1])
        for line in (root / "execution.tsv").read_text().splitlines()[1:]
        if len(fields := line.split("\t")) == 2 and fields[0] != "verification"
    }
    checks["launchers_succeeded"] = execution == {"native_copy": 0, "ours": 0}

    for workload in WORKLOADS:
        for batch in (1, 16):
            prefix = f"{workload}_b{batch}"
            try:
                width = WIDTHS[(workload, batch)]
                native_case = (
                    root / "native_copy" / "cases" / f"{workload}_b{batch}_copy"
                )
                ours_case = (
                    root / "ours" / "ours" / f"{workload}_b{batch}_D" / "custom"
                    / f"b0_{workload}_b{batch}_D"
                )
                native = load_json(native_case / "result.json")
                ours = load_json(ours_case / "result.json")
                native_log = (native_case / "run.log").read_text(errors="replace")
                ours_log = (ours_case / "run.log").read_text(errors="replace")
                native_manifest_path = native_case / "manifest.jsonl"
                native_events = [
                    json.loads(line)
                    for line in native_manifest_path.read_text().splitlines()
                    if line
                ]
                native_parameters = [
                    event
                    for event in native_events
                    if event.get("event") == "cpu_offload_parameter"
                ]
                native_manifest = manifest_stats(native_manifest_path, False)
                ours_manifest = manifest_stats(ours_case / "manifest.jsonl", True)
                native_summary = native["phase_summary"][str(batch)]
                native_measurement = next(
                    cycle
                    for cycle in native["cycles"]
                    if cycle.get("role") == "measurement"
                )
                expected_ours_bytes = width * 48 * EXPERT_MAIN_BYTES
                symmetric_ratio = max(
                    native_manifest["realized_bytes"], ours_manifest["realized_bytes"]
                ) / min(
                    native_manifest["realized_bytes"], ours_manifest["realized_bytes"]
                )

                checks[f"{prefix}_native_result_ok"] = native.get("status") == "ok"
                checks[f"{prefix}_ours_result_ok"] = ours.get("status") == "ok"
                checks[f"{prefix}_native_copy_backend"] = bool(native_parameters) and all(
                    event.get("offload_backend") == "copy"
                    and event.get("copy_on_forward") is True
                    for event in native_parameters
                )
                checks[f"{prefix}_ours_width"] = (
                    ours_manifest["expert_count"] == width * 48
                )
                checks[f"{prefix}_ours_bytes"] = (
                    ours_manifest["realized_bytes"] == expected_ours_bytes
                )
                checks[f"{prefix}_realized_bytes_within_10_percent"] = (
                    symmetric_ratio <= 1.10
                )
                checks[f"{prefix}_native_minimum_prefix"] = (
                    native_manifest["previous_prefix_bytes"]
                    < native_manifest["requested_bytes"]
                    <= native_manifest["realized_bytes"]
                )
                checks[f"{prefix}_output_hash_match"] = (
                    native_measurement["output_token_ids_sha256"]
                    == ours["output_token_ids_sha256"]
                )
                checks[f"{prefix}_no_fatal_log"] = not (
                    FATAL.search(native_log) or FATAL.search(ours_log)
                )
                for arm, values in (
                    ("native_copy", graph_checks(native_log, False)),
                    ("ours", graph_checks(ours_log, True)),
                ):
                    checks.update(
                        {f"{prefix}_{arm}_{key}": value for key, value in values.items()}
                    )

                native_metrics = {
                    "prefill_ms": float(native_summary["prefill_step_ms"]["median"]),
                    "decode_ms": float(
                        native_summary["decode_repeat_median_ms"]["median"]
                    ),
                    "e2e_ms": float(native_summary["cycle_e2e_ms"]["median"]),
                    "ttft_ms": float(native_summary["request_ttft_p50_ms"]["median"]),
                    "tpot_ms": float(native_summary["request_tpot_p50_ms"]["median"]),
                }
                ours_metrics = {
                    "prefill_ms": float(ours["timing"]["prefill_step_ms"]),
                    "decode_ms": float(
                        ours["timing"]["steady_decode_step_ms"]["median"]
                    ),
                    "e2e_ms": float(ours["timing"]["cycle_e2e_ms"]),
                    "cutover_ms": float(ours["timing"]["cutover_wall_ms"]),
                }
                cells.append(
                    {
                        "workload": workload,
                        "batch": batch,
                        "cold_width": width,
                        "ours": {"manifest": ours_manifest, "metrics": ours_metrics},
                        "native_copy": {
                            "manifest": native_manifest,
                            "metrics": native_metrics,
                        },
                        "offload": {
                            "symmetric_realized_ratio": symmetric_ratio,
                            "ours_gib": ours_manifest["realized_bytes"] / 2**30,
                            "native_copy_gib": native_manifest["realized_bytes"] / 2**30,
                        },
                        "output_hash_match": checks[f"{prefix}_output_hash_match"],
                        "native_copy_over_ours": {
                            key: native_metrics[key] / ours_metrics[key]
                            for key in ("prefill_ms", "decode_ms", "e2e_ms")
                        },
                    }
                )
                checks[f"{prefix}_cell_processed"] = True
            except Exception as exc:  # preserve a formal failure receipt
                checks[f"{prefix}_cell_processed"] = False
                errors[prefix] = f"{type(exc).__name__}: {exc}"

    kernel_log = (root / "kernel.since.log").read_text(errors="replace")
    checks["no_kernel_fault"] = not FATAL.search(kernel_log)
    gpu_rows = [
        line.split(",")
        for line in (root / "gpu.after.csv").read_text().splitlines()
    ]
    checks["gpu_zero_after"] = bool(gpu_rows) and all(
        int(row[1].strip()) == 0 for row in gpu_rows
    )
    status = "ok" if all(checks.values()) and len(cells) == 10 else "error"
    result = {
        "schema": "eurosys_e3_near_equal_native_copy_graph_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "protocol": {
            "repetitions": 1,
            "native_arm": "copy-on-forward",
            "native_graph": "FULL_SEPARATE",
            "ours_graph": "FULL_SEPARATE plus full dual-page-table P-to-D",
            "ours_widths": [11, 39, 90],
            "byte_match": "realized byte counts differ by no more than 10 percent",
        },
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "errors": errors,
        "cells": cells,
    }
    for name in ("verification.json", "summary.json"):
        write_json(root / name, result)
    print(
        json.dumps(
            {"status": status, "failed_checks": result["failed_checks"], "errors": errors},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
