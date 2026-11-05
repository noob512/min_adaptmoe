#!/usr/bin/env python3
"""Verify one complete E6 serial-wave qualification chain."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.run_root.resolve()
    result = load(root / "block_01" / "result.json")
    manifest = load(root / "inputs" / "serial_plan_classes.json")
    paths = load(root / "inputs" / "serial_transition_paths.json")
    expected_sequence = [
        "base", "d005", "base", "d010", "base", "d025", "base",
        "d050", "base", "d100", "base",
    ]
    require(result.get("status") == "ok", "runner status is not ok")
    require(result.get("transition_mode") == "serial_paths", "transition mode")
    require(result.get("sequence") == expected_sequence, "sequence mismatch")
    require(
        result.get("capture_count_after_load") == result.get("capture_count_final") == 2,
        "Graph capture count changed",
    )
    require(result.get("cleanup_errors") == [], "cleanup errors")
    require(result.get("reference_release", {}).get("ok") is True, "reference release")
    records = result.get("records")
    require(isinstance(records, list) and len(records) == 11, "record count")
    hashes = {str(record.get("output_sha256")) for record in records}
    require(len(hashes) == 1 and "None" not in hashes, "output drift")

    selected_bytes = int(manifest["selected_main_bytes"])
    aggregate: dict[str, dict[str, Any]] = {}
    for previous, record in zip(records, records[1:]):
        old_class = str(previous["class_name"])
        new_class = str(record["class_name"])
        if old_class == "base" and new_class.startswith("d"):
            target, direction = new_class, "outbound"
        elif new_class == "base" and old_class.startswith("d"):
            target, direction = old_class, "return"
        else:
            raise RuntimeError(f"unexpected transition {old_class}->{new_class}")
        key = f"{old_class}->{new_class}"
        expected_path = paths[key]
        require(record.get("transition_path") == expected_path, f"{key}: path")
        activations = record.get("wave_activations")
        require(
            isinstance(activations, list) and len(activations) == len(expected_path),
            f"{key}: wave activation count",
        )
        plan_wall_ns = 0
        for activation in activations:
            require(activation.get("ok") is True, f"{key}: activation")
            switch = activation.get("plan_switch")
            require(isinstance(switch, dict) and switch.get("ok") is True, f"{key}: switch")
            require(switch.get("session_reused") is True, f"{key}: session reuse")
            require(switch.get("graphs_retained") is True, f"{key}: Graph retention")
            require(switch.get("incremental_topology") is True, f"{key}: incremental")
            require(
                int(switch.get("old_selected_bytes", -1))
                == int(switch.get("new_selected_bytes", -2))
                == selected_bytes,
                f"{key}: selected bytes",
            )
            require(
                int(switch.get("old_backing_bytes", -1))
                == int(switch.get("new_backing_bytes", -2)),
                f"{key}: backing bytes",
            )
            require(switch.get("destroy") is None, f"{key}: session destroyed")
            plan_wall_ns += int(switch.get("wall_ns", 0))
        recorded_ns = int(record.get("transition_path_plan_wall_ns", -1))
        require(recorded_ns == plan_wall_ns > 0, f"{key}: timing sum")
        wrapper_ms = float(record.get("switch_ns", 0)) / 1e6
        plan_ms = plan_wall_ns / 1e6
        require(
            math.isfinite(wrapper_ms) and wrapper_ms >= plan_ms > 0,
            f"{key}: invalid timing",
        )
        aggregate.setdefault(target, {})[direction] = {
            "wave_count": len(expected_path),
            "plan_switch_ms": plan_ms,
            "activation_wall_ms": wrapper_ms,
            "request_ms": float(record["request_ns"]) / 1e6,
        }

    require(
        all(set(value) == {"outbound", "return"} for value in aggregate.values()),
        "direction coverage",
    )
    payload = {
        "schema": "eurosys_e6_serial_wave_qualification_verification_v1",
        "status": "ok",
        "run_root": str(root),
        "process_blocks": 1,
        "output_sha256": next(iter(hashes)),
        "capture_count": 2,
        "selected_main_bytes": selected_bytes,
        "backing_bytes": int(records[0]["backing_bytes"]),
        "aggregate": aggregate,
    }
    require(not args.output.exists(), f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
