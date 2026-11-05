#!/usr/bin/env python3
"""Verify and aggregate the E6 controlled plan-difference formal matrix."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: root is not an object")
    return value


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated sample percentile."""
    require(values, "percentile requires at least one value")
    require(0.0 <= quantile <= 1.0, "percentile quantile out of range")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_median_ci(
    values: list[float], *, seed: int, resamples: int = 10_000
) -> list[float]:
    """Return a deterministic percentile-bootstrap 95% CI for the median."""
    require(values, "bootstrap requires at least one value")
    require(resamples > 0, "bootstrap resample count must be positive")
    rng = random.Random(seed)
    count = len(values)
    medians = [
        statistics.median(values[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    ]
    return [percentile(medians, 0.025), percentile(medians, 0.975)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    config = load(root / "run_config.json")
    manifest = load(root / "inputs" / "plan_classes.json")
    blocks = int(config.get("process_blocks", 0))
    mode = str(config.get("mode", "formal"))
    require(mode in {"qualification", "formal"}, "invalid run mode")
    minimum_blocks = 1 if mode == "qualification" else 10
    require(
        blocks >= minimum_blocks,
        f"fewer than {minimum_blocks} independent process blocks for {mode}",
    )
    classes = manifest.get("classes")
    require(isinstance(classes, dict), "plan classes absent")
    expected = {f"d{value:03d}" for value in (5, 10, 25, 50, 100)}
    require(expected <= set(classes), "difference classes incomplete")

    status_lines = (root / "execution_status.tsv").read_text().splitlines()
    rows = [line.split("\t") for line in status_lines[1:] if line]
    require(len(rows) == blocks, "process block count mismatch")
    require(all(row[-1] == "0" for row in rows), "one or more blocks failed")

    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    output_hashes: set[str] = set()
    block_summaries: list[dict[str, Any]] = []
    for row in rows:
        block = int(row[0])
        block_root = root / row[-2]
        result = load(block_root / "result.json")
        require(result.get("status") == "ok", f"block {block}: status")
        require(
            result.get("capture_count_after_load") == result.get("capture_count_final"),
            f"block {block}: Graph recapture",
        )
        records = result.get("records")
        require(isinstance(records, list) and records, f"block {block}: records absent")
        hashes = {str(item.get("output_sha256")) for item in records}
        require(len(hashes) == 1 and "None" not in hashes, f"block {block}: output drift")
        output_hashes |= hashes
        sequence = [str(value) for value in result.get("sequence", ())]
        require(sequence == [str(item.get("class_name")) for item in records], f"block {block}: sequence")
        block_samples = 0
        for previous, item in zip(records, records[1:]):
            old_class = str(previous.get("class_name"))
            new_class = str(item.get("class_name"))
            if old_class == "base" and new_class in expected:
                direction = "outbound"
                difference_class = new_class
            elif new_class == "base" and old_class in expected:
                direction = "return"
                difference_class = old_class
            else:
                continue
            activation = item.get("activation")
            require(isinstance(activation, dict) and activation.get("ok") is True, f"block {block}: activation")
            switch = activation.get("plan_switch")
            require(isinstance(switch, dict) and switch.get("ok") is True, f"block {block}: plan switch")
            require(switch.get("session_reused") is True, f"block {block}: session not reused")
            require(switch.get("graphs_retained") is True, f"block {block}: Graphs not retained")
            require(switch.get("incremental_topology") is True, f"block {block}: not incremental")
            require(
                int(switch.get("old_selected_bytes", -1))
                == int(switch.get("new_selected_bytes", -2))
                == int(manifest["selected_main_bytes"]),
                f"block {block}: selected bytes changed",
            )
            require(
                int(switch.get("old_backing_bytes", -1))
                == int(switch.get("new_backing_bytes", -2)),
                f"block {block}: backing bytes changed",
            )
            require(switch.get("destroy") is None, f"block {block}: session destroyed")
            wall_ms = float(switch.get("wall_ns", 0)) / 1e6
            require(wall_ms > 0 and math.isfinite(wall_ms), f"block {block}: timing")
            samples[(difference_class, direction)].append(wall_ms)
            block_samples += 1
        require(block_samples == 10, f"block {block}: expected ten measured transitions")
        block_summaries.append(
            {
                "block": block,
                "output_sha256": next(iter(hashes)),
                "capture_count": int(result["capture_count_final"]),
                "measured_transition_count": block_samples,
            }
        )
    require(len(output_hashes) == 1, "cross-block output drift")

    aggregate: dict[str, Any] = {}
    for class_name in sorted(expected):
        record = classes[class_name]
        per_direction: dict[str, Any] = {}
        for direction in ("outbound", "return"):
            values = samples[(class_name, direction)]
            require(len(values) == blocks, f"{class_name}/{direction}: sample count")
            bootstrap_seed = 20260828 + int(class_name[1:]) * 2 + (
                0 if direction == "outbound" else 1
            )
            per_direction[direction] = {
                "count": len(values),
                "median_ms": statistics.median(values),
                "p95_ms": percentile(values, 0.95),
                "p99_ms": percentile(values, 0.99),
                "median_bootstrap_95_ci_ms": bootstrap_median_ci(
                    values, seed=bootstrap_seed
                ),
                "bootstrap_resamples": 10_000,
                "bootstrap_seed": bootstrap_seed,
                "mean_ms": statistics.fmean(values),
                "min_ms": min(values),
                "max_ms": max(values),
            }
        aggregate[class_name] = {
            "requested_difference_percent": int(record["difference_percent"]),
            "actual_difference_percent": float(record["actual_difference_percent"]),
            "changed_expert_slots": int(record["changed_expert_slots"]),
            "directions": per_direction,
        }

    verification = {
        "schema": "eurosys_e6_difference_formal_verification_v2",
        "status": "ok",
        "mode": mode,
        "run_root": str(root),
        "process_blocks": blocks,
        "output_sha256": next(iter(output_hashes)),
        "block_summaries": block_summaries,
        "aggregate": aggregate,
    }
    output = args.output or root / "verification.json"
    output.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
