#!/usr/bin/env python3
"""Fail closed unless an ours case used the full prebuilt dual-page-table path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_root", type=Path)
    args = parser.parse_args()

    phase_path = args.case_root / "phase_transition.json"
    try:
        phase = json.loads(phase_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result = {
            "schema": "ours_full_dual_view_verification_v1",
            "status": "error",
            "reason": f"phase_transition_unavailable: {exc}",
            "checks": {},
        }
        write_json(args.case_root / "dual_view_verification.json", result)
        return 1

    transition = phase.get("transition") or {}
    park = transition.get("park") or {}
    remote_map = transition.get("remote_map") or {}
    operation = transition.get("remote_map_operation")
    if operation is None:
        operation = remote_map.get("operation") or remote_map.get("reason")
    block_count = int(park.get("dual_view_block_count") or 0)
    checks = {
        "phase_transition_ok": phase.get("ok") is True,
        "dual_view_full_coverage": park.get("dual_view_full_coverage") is True,
        "dual_view_block_count_positive": block_count > 0,
        "prebuilt_full_dual_view_remote_map": operation == "prebuilt_full_dual_view",
        "no_cpu_range_map_ioctl": int(remote_map.get("ioctl_calls") or 0) == 0,
    }
    result = {
        "schema": "ours_full_dual_view_verification_v1",
        "status": "ok" if all(checks.values()) else "error",
        "checks": checks,
        "dual_view_block_count": block_count,
        "remote_map_operation": operation,
    }
    write_json(args.case_root / "dual_view_verification.json", result)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
