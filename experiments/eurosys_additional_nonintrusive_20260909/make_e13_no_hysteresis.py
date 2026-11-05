#!/usr/bin/env python3
"""Derive an E13 Auto-No-Hysteresis config without editing the source config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    source = args.input.read_bytes()
    config = json.loads(source)
    config["description"] += "; externally derived Auto-No-Hysteresis qualification"
    config["runtime"]["low_pressure_streak"] = 1
    config["runtime"]["minimum_residency_steps"] = 0
    config["arms"].append(
        {
            "name": "adaptive_no_hysteresis",
            "strategy": "adaptive",
            "initial_tier": "k4",
        }
    )
    config["external_derivation"] = {
        "base_config": str(args.input.resolve()),
        "base_sha256": hashlib.sha256(source).hexdigest(),
        "changes": {
            "runtime.low_pressure_streak": 1,
            "runtime.minimum_residency_steps": 0,
            "added_arm": "adaptive_no_hysteresis",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
