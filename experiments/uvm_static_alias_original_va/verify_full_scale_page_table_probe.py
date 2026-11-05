#!/usr/bin/env python3
"""Verify the one-shot synthetic full-scale page-table probe."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


KERNEL_ERROR_RE = re.compile(
    r"NVRM: Xid|(?:^|\s)BUG:|Oops:|general protection fault|"
    r"NULL pointer|uvm_gpu_fault_entry",
    re.IGNORECASE,
)


def _ms(section: dict[str, object], field: str) -> float:
    return float(section.get(field, 0)) / 1e6


def verify(result_path: Path, kernel_path: Path) -> dict[str, object]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["single_run"] is True
    assert result["statistical_repetitions"] == 0
    assert result["vllm_model_loaded"] is False
    assert result["dual_parent_switch_enabled"] is False
    assert result["geometry"]["consumer_count"] == 34_560
    assert result["geometry"]["total_directory_records"] == 36_000
    assert result["resume"]["ok"] is True
    assert result["resume"]["rm_status"] == 0
    assert result["resume"]["processed_alias_count"] == 34_560
    assert result["park"]["ok"] is True
    assert result["park"]["processed_alias_count"] == 34_560
    assert all(item["correct"] is True for item in result["correctness_samples"])

    kernel_text = kernel_path.read_text(encoding="utf-8", errors="replace")
    bad = [line for line in kernel_text.splitlines() if KERNEL_ERROR_RE.search(line)]
    assert not bad, bad

    resume = result["resume"]
    park = result["park"]
    return {
        "schema": "synthetic_full_scale_static_alias_page_table_verification_v1",
        "status": "ok",
        "single_run": True,
        "statistical_repetitions": 0,
        "vllm_model_loaded": False,
        "dual_parent_switch_enabled": False,
        "pte_mode": result["pte_mode_requested"],
        "geometry": result["geometry"],
        "resume": {
            "outer_wall_ms": result["resume_outer_wall_ms"],
            "driver_wall_ms": _ms(resume, "wall_ns"),
            "ioctl_wall_ms": _ms(resume, "ioctl_wall_ns"),
            "pte_map_ms": _ms(resume, "pte_map_ns"),
            "build_sort_ms": _ms(resume, "build_sort_ns"),
            "preflight_ms": _ms(resume, "preflight_ns"),
            "metadata_ms": _ms(resume, "metadata_ns"),
            "cpu_unmap_ms": _ms(resume, "cpu_unmap_ns"),
            "wait_ms": _ms(resume, "wait_ns"),
            "bookkeeping_ms": _ms(resume, "bookkeeping_ns"),
            "transition_batches": resume["transition_batch_count"],
            "tlb_invalidates": resume["tlb_invalidate_count"],
        },
        "park": {
            "outer_wall_ms": result["park_outer_wall_ms"],
            "driver_wall_ms": _ms(park, "wall_ns"),
            "ioctl_wall_ms": _ms(park, "ioctl_wall_ns"),
            "pte_map_ms": _ms(park, "pte_map_ns"),
            "transition_batches": park["transition_batch_count"],
            "tlb_invalidates": park["tlb_invalidate_count"],
        },
        "correctness_samples": result["correctness_samples"],
        "kernel_faults": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--kernel-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verification = verify(args.result, args.kernel_log)
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
