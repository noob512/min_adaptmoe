#!/usr/bin/env python3
"""Verify an E12 dual-view PARK/RESUME soak artifact."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-cycles", type=int, default=1000)
    args = parser.parse_args()
    errors: list[str] = []
    result = load(args.run_root / "result.json")
    soak = result.get("soak") if isinstance(result.get("soak"), dict) else {}
    completed = int(soak.get("completed_cycles", 0))
    if result.get("ok") is not True:
        errors.append("probe did not complete successfully")
    if completed < args.minimum_cycles:
        errors.append(
            f"completed {completed} cycles, expected at least {args.minimum_cycles}"
        )
    for operation in ("park", "resume"):
        summary = soak.get(f"{operation}_ms", {})
        if int(summary.get("count", -1)) != completed:
            errors.append(f"{operation} timing count differs from completed cycles")
    for operation in ("park", "resume"):
        record = result.get(operation, {})
        if record.get("ok") is not True:
            errors.append(f"initial {operation} failed")
        if record.get("dual_view_full_coverage") is not True:
            errors.append(f"initial {operation} lacks full dual-view coverage")

    progress_path = args.run_root / "progress.jsonl"
    progress: list[dict[str, Any]] = []
    if not progress_path.is_file():
        errors.append("progress checkpoint file is absent")
    else:
        for line_number, line in enumerate(
            progress_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"progress line {line_number} is invalid JSON")
                continue
            progress.append(item)
        counts = [int(item.get("completed_cycles", -1)) for item in progress]
        if counts != sorted(set(counts)):
            errors.append("progress cycle checkpoints are not strictly increasing")
        if not counts or counts[-1] != completed:
            errors.append("final progress checkpoint differs from final result")
        if any(item.get("verified") is not True for item in progress):
            errors.append("an unverified progress checkpoint was recorded")

    kernel = (args.run_root / "kernel.since.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if re.search(
        r"NVRM:\s+Xid|FAULT_(?:PDE|PTE)|MMU fault|\bBUG:|Oops:|NULL pointer|general protection fault",
        kernel,
        re.I,
    ):
        errors.append("kernel log contains a GPU/kernel fault")
    module_after = load(args.run_root / "module.after.json")
    if module_after.get("gpu_clean") is not True:
        errors.append("final stock module snapshot is not GPU-clean")
    before = load(args.run_root / "module.before.json")
    if module_after.get("loaded_srcversion") != before.get("loaded_srcversion"):
        errors.append("final UVM srcversion differs from initial stock module")

    verification = {
        "schema": "eurosys_e12_dual_view_soak_verification_v1",
        "status": "ok" if not errors else "error",
        "errors": errors,
        "scope": "PARK/RESUME mechanism soak; not the six-hour model-serving soak",
        "completed_cycles": completed,
        "elapsed_seconds": soak.get("elapsed_seconds"),
        "park_ms": soak.get("park_ms"),
        "resume_ms": soak.get("resume_ms"),
        "progress_checkpoints": len(progress),
    }
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
