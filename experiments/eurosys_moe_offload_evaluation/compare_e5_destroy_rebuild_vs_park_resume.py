#!/usr/bin/env python3
"""Compare E5 DESTROY/rebuild with the frozen grouped PARK/RESUME formal."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
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


def summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
    }


def paired_ratio(
    numerator: list[float], denominator: list[float], *, seed: int
) -> dict[str, Any]:
    require(
        len(numerator) == len(denominator) and len(numerator) >= 3,
        "expected at least three process pairs",
    )
    require(all(value > 0 for value in numerator + denominator), "non-positive time")
    pair_count = len(numerator)
    point = statistics.median(numerator) / statistics.median(denominator)
    rng = random.Random(seed)
    ratios = []
    for _ in range(10_000):
        indices = [rng.randrange(pair_count) for _ in range(pair_count)]
        ratios.append(
            statistics.median(numerator[index] for index in indices)
            / statistics.median(denominator[index] for index in indices)
        )
    return {
        "definition": "median(DESTROY/rebuild_ms) / median(PARK/RESUME_ms)",
        "point_estimate": point,
        "paired_bootstrap_95_ci": [
            percentile(ratios, 0.025),
            percentile(ratios, 0.975),
        ],
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": seed,
    }


def park_samples(verification: dict[str, Any]) -> dict[int, dict[int, dict[str, Any]]]:
    output: dict[int, dict[int, dict[str, Any]]] = {1: {}, 4: {}}
    for item in verification.get("samples", []):
        if item.get("audit_trace") is True:
            continue
        match = re.search(r"_s(\d+)_b(\d+)_", str(item.get("case", "")))
        require(match is not None, "cannot parse PARK performance sample")
        sample, batch = int(match.group(1)), int(match.group(2))
        output[batch][sample] = item
    repeat_sets = [set(output[batch]) for batch in output]
    require(repeat_sets[0] == repeat_sets[1], "PARK repeat sets differ by batch")
    require(len(repeat_sets[0]) >= 3, "PARK requires at least three repeats")
    return output


def destroy_samples(verification: dict[str, Any]) -> dict[int, dict[int, dict[str, Any]]]:
    output: dict[int, dict[int, dict[str, Any]]] = {1: {}, 4: {}}
    for item in verification.get("samples", []):
        batch, repeat = int(item["batch_size"]), int(item["repeat"])
        require(repeat not in output[batch], f"duplicate DESTROY B{batch}/r{repeat}")
        output[batch][repeat] = item
    repeat_sets = [set(output[batch]) for batch in output]
    require(repeat_sets[0] == repeat_sets[1], "DESTROY repeat sets differ by batch")
    require(len(repeat_sets[0]) >= 3, "DESTROY requires at least three repeats")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destroy_root", type=Path)
    parser.add_argument("park_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    destroy_root = args.destroy_root.resolve()
    park_root = args.park_root.resolve()
    destroy_verification = load(destroy_root / "verification.json")
    park_verification = load(park_root / "verification.json")
    require(destroy_verification.get("status") == "ok", "DESTROY verifier status")
    destroy_repeat_count = int(destroy_verification.get("repeat_count", 0))
    require(destroy_repeat_count >= 3, "DESTROY repeats")
    require(destroy_verification.get("all_fault_scans_clean") is True, "DESTROY faults")
    require(park_verification.get("status") == "ok", "PARK verifier status")
    park_samples_raw = park_verification.get("samples", [])
    require(
        int(park_verification.get("case_count", -1)) == len(park_samples_raw),
        "PARK case count",
    )

    park_config = park_verification.get("run_config", {})
    destroy_config = load(destroy_root / "run_config.json")
    park_prebuild_enabled = bool(park_config.get("static_alias_dual_view_enable"))
    geometry_checks = {
        "model": str(park_config.get("model", "")).endswith(
            "Qwen3-30B-A3B-FP8/snapshots/d206ba732169f29bb77fbf80fc2c4b81d4d30782"
        ),
        "logical_alias_bytes": park_config.get("logical_bytes_per_forward")
        == destroy_config.get("logical_alias_bytes")
        == 2_415_919_104,
        "retained_backing_bytes": park_config.get("dual_bank_backing_bytes")
        == destroy_config.get("retained_backing_bytes")
        == 1_207_959_552,
        "physical_kv_bytes": park_config.get("kv_cache_memory_bytes")
        == destroy_config.get("physical_kv_bytes")
        == 134_217_728,
        "batch_sizes": park_config.get("batch_sizes") == [1, 4],
        "graph_mode": park_config.get("cudagraph_mode") == "FULL_SEPARATE",
        "custom_module": park_config.get("custom_module_sha256")
        == load(destroy_root / "repeat_01" / "module.custom.json").get(
            "module_sha256"
        ),
    }
    require(all(geometry_checks.values()), f"geometry mismatch: {geometry_checks}")

    destroy = destroy_samples(destroy_verification)
    park = park_samples(park_verification)
    repeat_ids = sorted(destroy[1])
    require(
        repeat_ids == sorted(park[1])
        and repeat_ids == sorted(destroy[4])
        and repeat_ids == sorted(park[4]),
        "PARK and DESTROY repeat IDs differ",
    )
    require(len(repeat_ids) == destroy_repeat_count, "repeat count metadata differs")
    require(
        len(repeat_ids) == int(park_config.get("performance_repeats", 0)),
        "PARK repeat count metadata differs",
    )
    comparison: dict[str, Any] = {}
    for batch in (1, 4):
        destroy_total = [destroy[batch][index]["rebuild_total_ms"] for index in repeat_ids]
        destroy_plan = [destroy[batch][index]["plan_rebuild_ms"] for index in repeat_ids]
        park_resume = [park[batch][index]["resume_wall_ms"] for index in repeat_ids]
        park_roundtrip = [
            park[batch][index]["resume_wall_ms"] + park[batch][index]["repark_wall_ms"]
            for index in repeat_ids
        ]
        comparison[str(batch)] = {
            "destroy_rebuild_total": summary(destroy_total),
            "destroy_plan_rebuild": summary(destroy_plan),
            "park_resume": summary(park_resume),
            "park_resume_plus_repark": summary(park_roundtrip),
            "total_rebuild_over_resume_ratio": paired_ratio(
                destroy_total, park_resume, seed=20260828 + batch
            ),
            "plan_rebuild_over_resume_ratio": paired_ratio(
                destroy_plan, park_resume, seed=20260928 + batch
            ),
            "total_rebuild_over_roundtrip_ratio": paired_ratio(
                destroy_total, park_roundtrip, seed=20261028 + batch
            ),
        }

    payload = {
        "schema": "e5_destroy_rebuild_vs_park_resume_comparison_v2",
        "status": "ok",
        "destroy_root": str(destroy_root),
        "park_root": str(park_root),
        "process_pairs_per_batch": len(repeat_ids),
        "controlled_geometry_equal": True,
        "geometry_checks": geometry_checks,
        "retained_backing_cost_bytes": 1_207_959_552,
        "park_prebuild_enabled": park_prebuild_enabled,
        "strict_execution_input_hashes_equal": False,
        "provenance_note": (
            "Model, request geometry, expert set, KV size, Graph mode and custom "
            "module match. "
            + (
                "The PARK/RESUME arm uses fully prebuilt dual views, which are a "
                "required dependency of direct PDE switching; the incremental "
                "prebuild contribution is reported in a separate on/off ablation. "
                if park_prebuild_enabled
                else "The PARK/RESUME arm uses runtime-installed ranges with "
                "dual-view prebuild disabled. "
            )
            + "Only transition timings are compared."
        ),
        "comparison": comparison,
    }
    require(not args.output.exists(), f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
