#!/usr/bin/env python3
"""Verify native-versus-managed allocation under contiguous/fragmented free space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MIB = 1024 * 1024


def load_result(path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("stage") == "probe_result":
            records.append(value)
    if len(records) != 1:
        raise RuntimeError(f"expected one probe_result in {path}, got {len(records)}")
    return records[0]


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if float(denominator) == 0 else float(numerator) / float(denominator)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "native_contiguous",
        "managed_contiguous",
        "native_fragmented",
        "managed_fragmented",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")

    results = {
        name: load_result(getattr(args, name))
        for name in (
            "native_contiguous",
            "managed_contiguous",
            "native_fragmented",
            "managed_fragmented",
        )
    }
    expected_identity = {
        "native_contiguous": ("native", "contiguous"),
        "managed_contiguous": ("managed", "contiguous"),
        "native_fragmented": ("native", "fragmented"),
        "managed_fragmented": ("managed", "fragmented"),
    }
    identity_ok = all(
        results[name].get("allocation_kind") == allocation
        and results[name].get("layout") == layout
        for name, (allocation, layout) in expected_identity.items()
    )
    geometry_keys = ("arena_bytes", "release_bytes", "native_bytes", "stripe_bytes")
    geometry = {tuple(results[name].get(key) for key in geometry_keys) for name in results}
    starting_free = [int(value.get("pre_release_free_bytes", -1)) for value in results.values()]
    released_free = [int(value.get("after_release_free_bytes", -1)) for value in results.values()]
    native_bytes = int(results["native_contiguous"].get("native_bytes", 0))

    nc = results["native_contiguous"]
    nf = results["native_fragmented"]
    mc = results["managed_contiguous"]
    mf = results["managed_fragmented"]
    checks = {
        "all_measurements_valid": all(
            value.get("measurement_valid") is True for value in results.values()
        ),
        "case_identity": identity_ok,
        "same_geometry": len(geometry) == 1,
        "same_starting_free_within_64mib": max(starting_free) - min(starting_free) <= 64 * MIB,
        "same_released_free_within_64mib": max(released_free) - min(released_free) <= 64 * MIB,
        "automatic_fault_migration_disabled": all(
            value.get("manual_migration_fault_mode") == 2
            for value in results.values()
        ),
        "native_reproduces_fragmentation_eviction": (
            int(nc.get("hot_evicted_bytes", -1)) <= 64 * MIB
            and abs(native_bytes - int(nc.get("free_drop_bytes", -1))) <= 64 * MIB
            and int(nf.get("hot_evicted_bytes", -1)) >= 128 * MIB
        ),
        "managed_target_explicitly_gpu_resident": (
            mc.get("target_gpu_ok") is True and mf.get("target_gpu_ok") is True
        ),
        "managed_uses_fragmented_free_without_hot_eviction": all(
            int(value.get("hot_evicted_bytes", -1)) <= 64 * MIB
            and abs(native_bytes - int(value.get("free_drop_bytes", -1))) <= 64 * MIB
            for value in (mc, mf)
        ),
        "managed_avoids_native_fragmentation_eviction": (
            int(nf.get("hot_evicted_bytes", -1))
            - int(mf.get("hot_evicted_bytes", -1))
            >= 128 * MIB
        ),
        "performance_metrics_present": all(
            int(value.get("time_to_ready_ns", 0)) > 0
            and int(value.get("bandwidth_elapsed_ns", 0)) > 0
            and float(value.get("steady_state_memset_gbps", 0.0)) > 0.0
            for value in results.values()
        ),
    }
    performance: dict[str, Any] = {}
    for layout in ("contiguous", "fragmented"):
        native = results[f"native_{layout}"]
        managed = results[f"managed_{layout}"]
        performance[layout] = {
            "native": {
                key: native[key]
                for key in (
                    "allocation_ns",
                    "placement_ns",
                    "initial_touch_ns",
                    "time_to_ready_ns",
                    "steady_state_memset_gbps",
                )
            },
            "managed": {
                key: managed[key]
                for key in (
                    "allocation_ns",
                    "placement_ns",
                    "initial_touch_ns",
                    "time_to_ready_ns",
                    "steady_state_memset_gbps",
                )
            },
            "managed_to_native_time_to_ready_ratio": ratio(
                managed["time_to_ready_ns"], native["time_to_ready_ns"]
            ),
            "managed_to_native_steady_bandwidth_ratio": ratio(
                managed["steady_state_memset_gbps"],
                native["steady_state_memset_gbps"],
            ),
        }

    passed = all(checks.values())
    output = {
        "schema": "uvm_fragmentation_native_managed_compare_v1",
        "status": "proved" if passed else "inconclusive",
        "managed_avoids_fragmentation_eviction": passed,
        "checks": checks,
        "residency_comparison": {
            name: {
                "free_drop_bytes": value["free_drop_bytes"],
                "hot_evicted_bytes": value["hot_evicted_bytes"],
                "target_gpu_ok": value["target_gpu_ok"],
            }
            for name, value in results.items()
        },
        "performance": performance,
        "results": results,
    }
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
