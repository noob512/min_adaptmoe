#!/usr/bin/env python3
"""Compare paired E5 stable-VA and non-stable-VA rebuild/recapture runs."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def timing_summary(values: list[float], *, single_round: bool = False) -> dict[str, Any]:
    if single_round:
        require(len(values) == 1, "single-round summary requires exactly one sample")
        return {"count": 1, "point_ms": values[0], "median_ms": None, "p95_ms": None}
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def paired_ratio(
    numerator: list[float], denominator: list[float], *, seed: int,
    single_round: bool = False,
) -> dict[str, Any]:
    if single_round:
        require(len(numerator) == len(denominator) == 1, "expected one pair")
        require(numerator[0] > 0 and denominator[0] > 0, "non-positive time")
        return {"point_estimate": numerator[0] / denominator[0],
                "paired_bootstrap_95_ci": None, "bootstrap_resamples": 0}
    require(
        len(numerator) == len(denominator) and len(numerator) >= 3,
        "expected at least three pairs",
    )
    require(all(value > 0 for value in numerator + denominator), "non-positive time")
    pair_count = len(numerator)
    point = statistics.median(numerator) / statistics.median(denominator)
    rng = random.Random(seed)
    bootstrapped: list[float] = []
    for _ in range(10_000):
        indices = [rng.randrange(pair_count) for _ in range(pair_count)]
        bootstrapped.append(
            statistics.median(numerator[index] for index in indices)
            / statistics.median(denominator[index] for index in indices)
        )
    return {
        "point_estimate": point,
        "paired_bootstrap_95_ci": [
            percentile(bootstrapped, 0.025),
            percentile(bootstrapped, 0.975),
        ],
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": seed,
    }


def samples_by_label(
    verification: dict[str, Any], *, name: str, single_round: bool = False
) -> dict[str, dict[int, dict[str, Any]]]:
    output: dict[str, dict[int, dict[str, Any]]] = {}
    for sample in verification.get("samples", []):
        label = str(sample.get("label"))
        repeat = int(sample.get("repeat", 0))
        require(repeat > 0, f"{name}: invalid repeat")
        bucket = output.setdefault(label, {})
        require(repeat not in bucket, f"{name}: duplicate {label}/r{repeat}")
        bucket[repeat] = sample
    require(set(output) == {"matched_b1", "matched_b4"}, f"{name}: labels")
    repeat_sets = [set(bucket) for bucket in output.values()]
    require(repeat_sets[0] == repeat_sets[1], f"{name}: repeat sets differ")
    if single_round:
        require(len(repeat_sets[0]) == 1, f"{name}: single round requires one repeat")
    else:
        require(len(repeat_sets[0]) >= 3, f"{name}: requires at least three repeats")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stable_root", type=Path)
    parser.add_argument("nonstable_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-round", action="store_true",
                        help="compare one process pair as point estimates without confidence intervals")
    args = parser.parse_args()

    stable_root = args.stable_root.resolve()
    nonstable_root = args.nonstable_root.resolve()
    stable_verification = load(stable_root / "verification.json")
    nonstable_verification = load(nonstable_root / "verification.json")
    stable_config = load(stable_root / "run_config.json")
    nonstable_config = load(nonstable_root / "run_config.json")

    require(stable_verification.get("status") == "ok", "stable verifier status")
    require(nonstable_verification.get("status") == "ok", "non-stable verifier status")
    require(stable_verification.get("stable_va_preserved") is True, "stable VA gate")
    require(
        nonstable_verification.get("all_parameter_vas_changed") is True,
        "non-stable VA gate",
    )
    require(
        nonstable_verification.get("all_graphs_recaptured") is True,
        "non-stable Graph gate",
    )
    require(
        stable_verification.get("all_fault_scans_clean") is True
        and nonstable_verification.get("all_fault_scans_clean") is True,
        "fault scan gate",
    )
    require(
        stable_verification.get("all_resources_released") is True
        and nonstable_verification.get("all_resources_released") is True,
        "cleanup gate",
    )

    geometry_checks = {
        "model": stable_config.get("model") == nonstable_config.get("model"),
        "layers": stable_config.get("layers") == nonstable_config.get("layers"),
        "parameter_count": stable_config.get("parameter_count")
        == nonstable_config.get("parameter_count")
        == 8,
        "selected_bytes": stable_config.get("logical_alias_bytes")
        == nonstable_config.get("selected_bytes")
        == 2_415_919_104,
        "physical_kv_bytes": stable_config.get("physical_kv_bytes")
        == nonstable_config.get("physical_kv_bytes")
        == 134_217_728,
        "sequence": [
            str(value).rsplit("_", 1)[-1]
            for value in stable_config.get("sequence", [])
        ]
        == [
            str(value).rsplit("_", 1)[-1]
            for value in nonstable_config.get("sequence", [])
        ]
        == ["b1", "b4", "b1"],
        "process_isolation": stable_config.get("process_isolation")
        == nonstable_config.get("process_isolation"),
    }
    require(all(geometry_checks.values()), f"geometry mismatch: {geometry_checks}")

    stable = samples_by_label(stable_verification, name="stable", single_round=args.single_round)
    nonstable = samples_by_label(nonstable_verification, name="non-stable", single_round=args.single_round)
    repeat_ids = sorted(stable["matched_b1"])
    require(
        repeat_ids == sorted(stable["matched_b4"])
        and repeat_ids == sorted(nonstable["matched_b1"])
        and repeat_ids == sorted(nonstable["matched_b4"]),
        "stable and non-stable repeat IDs differ",
    )
    require(
        len(repeat_ids)
        == int(stable_verification.get("repeat_count", 0))
        == int(nonstable_verification.get("repeat_count", 0)),
        "repeat count metadata differs",
    )
    comparison: dict[str, Any] = {}
    for label_index, label in enumerate(("matched_b1", "matched_b4"), 1):
        stable_total: list[float] = []
        nonstable_total: list[float] = []
        nonstable_reallocate: list[float] = []
        for repeat in repeat_ids:
            stable_sample = stable[label][repeat]
            nonstable_sample = nonstable[label][repeat]
            require(
                stable_sample.get("output_sha256")
                == nonstable_sample.get("output_sha256"),
                f"output mismatch: {label}/r{repeat}",
            )
            stable_total.append(float(stable_sample["rebuild_total_ms"]))
            nonstable_total.append(float(nonstable_sample["rebuild_total_ms"]))
            nonstable_reallocate.append(
                float(nonstable_sample["parameter_reallocate_ms"])
            )
        comparison[label] = {
            "output_sha256": stable[label][1]["output_sha256"],
            "stable_rebuild_recapture": timing_summary(stable_total, single_round=args.single_round),
            "nonstable_reallocate_recapture": timing_summary(nonstable_total, single_round=args.single_round),
            "nonstable_parameter_reallocate": timing_summary(nonstable_reallocate, single_round=args.single_round),
            "nonstable_over_stable_ratio": paired_ratio(
                nonstable_total,
                stable_total,
                seed=20260828 + label_index,
                single_round=args.single_round,
            ),
            "median_added_cost_ms": None if args.single_round else (
                statistics.median(nonstable_total) - statistics.median(stable_total)
            ),
            "point_added_cost_ms": nonstable_total[0] - stable_total[0] if args.single_round else None,
        }

    payload = {
        "schema": "e5_stable_vs_nonstable_va_comparison_v2",
        "status": "ok",
        "qualification_scope": "one_process_pair_point_estimate" if args.single_round else "repeated_process_pairs",
        "stable_root": str(stable_root),
        "nonstable_root": str(nonstable_root),
        "process_pairs_per_generation": len(repeat_ids),
        "geometry_checks": geometry_checks,
        "all_outputs_equal": True,
        "all_fault_scans_clean": True,
        "all_resources_released": True,
        "comparison": comparison,
    }
    require(not args.output.exists(), f"refusing to overwrite {args.output}")
    # The output file itself is immutable, but a conventional destination such
    # as /tmp already exists.  Only create missing parent directories here.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
