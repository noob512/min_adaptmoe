#!/usr/bin/env python3
"""Verify E3 Ours cold11/39/90 against near-equal Native UVA Graph."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKLOADS = (
    "short_dialogue",
    "ordinary_chat",
    "medium_context",
    "rag_summary",
    "long_generation",
)
WIDTHS = {
    ("short_dialogue", 1): 11,
    ("short_dialogue", 16): 11,
    ("ordinary_chat", 1): 11,
    ("ordinary_chat", 16): 39,
    ("medium_context", 1): 11,
    ("medium_context", 16): 39,
    ("rag_summary", 1): 39,
    ("rag_summary", 16): 90,
    ("long_generation", 1): 11,
    ("long_generation", 16): 39,
}
EXPERT_MAIN_BYTES = 4_718_592
FATAL = re.compile(
    r"Xid|MMU fault|PDE fault|PTE fault|illegal memory access|"
    r"CUDA error|out of memory|kernel BUG|Objects remaining|"
    r"kmem_cache_destroy|Traceback \(most recent call last\)",
    re.IGNORECASE,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_stats(path: Path, ours: bool) -> dict[str, Any]:
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    config = next(event for event in events if event.get("event") == "cpu_offload_config")
    if ours:
        count = int(config["graph_static_alias_selected_expert_count"])
        return {
            "expert_count": count,
            "realized_bytes": count * EXPERT_MAIN_BYTES,
            "plan_file": config["graph_static_alias_expert_plan_file"],
        }
    parameters = [
        event for event in events if event.get("event") == "cpu_offload_parameter"
    ]
    if not parameters:
        raise ValueError(f"no Native offload records in {path}")
    return {
        "requested_bytes": int(config["max_offload_bytes"]),
        "realized_bytes": int(parameters[-1]["cumulative_offload_bytes"]),
        "previous_prefix_bytes": (
            int(parameters[-2]["cumulative_offload_bytes"])
            if len(parameters) > 1 else 0
        ),
        "parameter_count": len(parameters),
    }


def graph_checks(log: str, ours: bool) -> dict[str, bool]:
    audit = [line for line in log.splitlines() if "CUDA_GRAPH_RUNTIME_AUDIT" in line]
    checks = {
        "full_separate_configured": (
            "CUDAGraphMode.FULL_SEPARATE" in log and "enforce_eager=False" in log
        ),
        "prefill_graph_captured": "Capturing CUDA graphs (pure prefill, FULL)" in log,
        "decode_graph_captured": "Capturing CUDA graphs (decode, FULL)" in log,
        "prefill_full_replay": any(
            "phase=enabled:prefill_step" in line and "runtime_mode=FULL" in line
            for line in audit
        ),
        "decode_full_replay": any(
            "phase=enabled:decode_step" in line and "runtime_mode=FULL" in line
            for line in audit
        ),
        "no_eager_dispatch": not any("runtime_mode=NONE" in line for line in audit),
    }
    if ours:
        checks["prebuilt_dual_view_transition"] = (
            "remote_map_operation': 'prebuilt_full_dual_view'" in log
            and "'dual_view_full_coverage': True" in log
        )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--ours-long-b16-root",
        type=Path,
        help=(
            "optional successful Long/B16 cell root used when the main launcher "
            "stopped before that previously unexecuted cell"
        ),
    )
    args = parser.parse_args()
    root = args.run_root.resolve()
    checks: dict[str, bool] = {}
    cells: list[dict[str, Any]] = []

    execution = {
        fields[0]: int(fields[1])
        for line in (root / "execution.tsv").read_text().splitlines()[1:]
        if len(fields := line.split("\t")) == 2 and fields[0] != "verification"
    }
    checks["launchers_succeeded"] = execution == {"native": 0, "ours": 0}

    for workload in WORKLOADS:
        for batch in (1, 16):
            width = WIDTHS[(workload, batch)]
            native_case = root / "native" / "cases" / f"{workload}_b{batch}_uva"
            if workload == "long_generation" and batch == 16 \
                    and args.ours_long_b16_root is not None:
                ours_case = (
                    args.ours_long_b16_root.resolve() / "custom"
                    / "b0_long_generation_b16_D"
                )
            else:
                ours_case = (
                    root / "ours" / "ours" / f"{workload}_b{batch}_D" / "custom"
                    / f"b0_{workload}_b{batch}_D"
                )
            native = load_json(native_case / "result.json")
            ours = load_json(ours_case / "result.json")
            native_log = (native_case / "run.log").read_text(errors="replace")
            ours_log = (ours_case / "run.log").read_text(errors="replace")
            native_manifest = manifest_stats(native_case / "manifest.jsonl", False)
            ours_manifest = manifest_stats(ours_case / "manifest.jsonl", True)
            native_summary = native["phase_summary"][str(batch)]
            native_measurement = next(
                cycle for cycle in native["cycles"]
                if cycle.get("role") == "measurement"
            )
            expected_ours_bytes = width * 48 * EXPERT_MAIN_BYTES
            symmetric_ratio = max(
                native_manifest["realized_bytes"], ours_manifest["realized_bytes"]
            ) / min(native_manifest["realized_bytes"], ours_manifest["realized_bytes"])
            prefix = f"{workload}_b{batch}"
            checks[f"{prefix}_native_result_ok"] = native.get("status") == "ok"
            checks[f"{prefix}_ours_result_ok"] = ours.get("status") == "ok"
            checks[f"{prefix}_ours_width"] = ours_manifest["expert_count"] == width * 48
            checks[f"{prefix}_ours_bytes"] = ours_manifest["realized_bytes"] == expected_ours_bytes
            checks[f"{prefix}_realized_bytes_within_10_percent"] = symmetric_ratio <= 1.10
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
                ("native", graph_checks(native_log, False)),
                ("ours", graph_checks(ours_log, True)),
            ):
                checks.update({f"{prefix}_{arm}_{key}": value for key, value in values.items()})

            native_metrics = {
                "prefill_ms": float(native_summary["prefill_step_ms"]["median"]),
                "decode_ms": float(native_summary["decode_repeat_median_ms"]["median"]),
                "e2e_ms": float(native_summary["cycle_e2e_ms"]["median"]),
                "ttft_ms": float(native_summary["request_ttft_p50_ms"]["median"]),
                "tpot_ms": float(native_summary["request_tpot_p50_ms"]["median"]),
            }
            ours_metrics = {
                "prefill_ms": float(ours["timing"]["prefill_step_ms"]),
                "decode_ms": float(ours["timing"]["steady_decode_step_ms"]["median"]),
                "e2e_ms": float(ours["timing"]["cycle_e2e_ms"]),
                "cutover_ms": float(ours["timing"]["cutover_wall_ms"]),
            }
            cells.append({
                "workload": workload,
                "batch": batch,
                "cold_width": width,
                "ours": {"manifest": ours_manifest, "metrics": ours_metrics},
                "native": {"manifest": native_manifest, "metrics": native_metrics},
                "offload": {
                    "symmetric_realized_ratio": symmetric_ratio,
                    "ours_gib": ours_manifest["realized_bytes"] / 2**30,
                    "native_gib": native_manifest["realized_bytes"] / 2**30,
                },
                "output_hash_match": checks[f"{prefix}_output_hash_match"],
                "native_over_ours": {
                    key: native_metrics[key] / ours_metrics[key]
                    for key in ("prefill_ms", "decode_ms", "e2e_ms")
                },
            })

    kernel_log = (root / "kernel.since.log").read_text(errors="replace")
    checks["no_kernel_fault"] = not FATAL.search(kernel_log)
    gpu_rows = [line.split(",") for line in (root / "gpu.after.csv").read_text().splitlines()]
    checks["gpu_zero_after"] = bool(gpu_rows) and all(int(row[1].strip()) == 0 for row in gpu_rows)
    status = "ok" if all(checks.values()) else "error"
    result = {
        "schema": "eurosys_e3_near_equal_offload_plus_one_expert_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "protocol": {
            "repetitions": 1,
            "ours_widths": [11, 39, 90],
            "native_graph": "FULL_SEPARATE",
            "ours_graph": "FULL_SEPARATE plus full dual-page-table P-to-D",
            "byte_match": "realized byte counts differ by no more than 10 percent",
            "ours_long_b16_continuation": (
                str(args.ours_long_b16_root.resolve())
                if args.ours_long_b16_root is not None else None
            ),
        },
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "cells": cells,
    }
    for name in ("verification.json", "summary.json"):
        (root / name).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({"status": status, "failed_checks": result["failed_checks"]}, indent=2))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
