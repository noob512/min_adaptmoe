#!/usr/bin/env python3
"""Verify one complete E6 full-rebuild qualification chain."""

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
    result = load(root / "result.json")
    expected_sequence = [
        "base", "d005", "base", "d010", "base", "d025", "base",
        "d050", "base", "d100", "base",
    ]
    require(result.get("status") == "ok", "runner status")
    require(result.get("sequence") == expected_sequence, "sequence")
    require(result.get("capture_count_after_load") == 2, "initial capture count")
    require(result.get("capture_count_final") == 22, "final capture count")
    require(result.get("cleanup_errors") == [], "cleanup errors")
    require(result.get("distributed_cleanup") is True, "distributed cleanup")
    require(
        result.get("controlled_hard_exit_after_cleanup") is True,
        "controlled hard-exit contract",
    )
    require(result.get("reference_release", {}).get("ok") is True, "reference release")
    records = result.get("records")
    require(isinstance(records, list) and len(records) == 11, "record count")
    hashes = {str(record.get("output_sha256")) for record in records}
    require(len(hashes) == 1 and "None" not in hashes, "output drift")
    parameter_vas = result.get("parameter_vas")
    aggregate: dict[str, dict[str, Any]] = {}
    for index, (previous, record) in enumerate(zip(records, records[1:]), 1):
        old_class, new_class = str(previous["class_name"]), str(record["class_name"])
        if old_class == "base" and new_class.startswith("d"):
            target, direction = new_class, "outbound"
        elif new_class == "base" and old_class.startswith("d"):
            target, direction = old_class, "return"
        else:
            raise RuntimeError(f"unexpected transition {old_class}->{new_class}")
        rebuild = record.get("rebuild")
        require(isinstance(rebuild, dict) and rebuild.get("ok") is True, "rebuild")
        require(rebuild.get("destroyed_graph_count") == 2, "destroyed Graph count")
        require(rebuild.get("graph_pool_renewed") is True, "Graph pool renewal")
        require(rebuild.get("parameter_va_mode") == "stable_rebuild_recapture", "VA mode")
        inner = rebuild.get("rebuild")
        require(isinstance(inner, dict) and inner.get("ok") is True, "inner rebuild")
        require(inner.get("parameter_vas_before") == inner.get("parameter_vas_after"), "inner VA drift")
        require(record.get("parameter_vas") == parameter_vas, "record VA drift")
        require(record.get("graph_count_before") == 2 * index, "Graph count before")
        require(record.get("graph_count_after_rebuild") == 2 * (index + 1), "Graph count after")
        require(record.get("graph_count_after_request") == 2 * (index + 1), "request recapture")
        metrics = {
            "total_ms": float(rebuild["wall_ns"]) / 1e6,
            "graph_reset_ms": float(rebuild["graph_reset_ns"]) / 1e6,
            "plan_rebuild_ms": float(rebuild["plan_rebuild_ns"]) / 1e6,
            "graph_capture_ms": float(rebuild["graph_capture_ns"]) / 1e6,
            "request_ms": float(record["request_ns"]) / 1e6,
        }
        require(all(math.isfinite(value) and value > 0 for value in metrics.values()), "timing")
        aggregate.setdefault(target, {})[direction] = metrics
    require(all(set(value) == {"outbound", "return"} for value in aggregate.values()), "coverage")
    payload = {
        "schema": "eurosys_e6_full_rebuild_qualification_verification_v1",
        "status": "ok",
        "run_root": str(root),
        "process_blocks": 1,
        "output_sha256": next(iter(hashes)),
        "capture_count_after_load": 2,
        "capture_count_final": 22,
        "selected_main_bytes": int(records[0]["selected_bytes"]),
        "backing_bytes": int(records[0]["backing_bytes"]),
        "aggregate": aggregate,
    }
    require(not args.output.exists(), f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
