#!/usr/bin/env python3
"""Verify the controlled contiguous-versus-fragmented cudaMalloc probe."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contiguous", required=True, type=Path)
    parser.add_argument("--fragmented", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")

    contiguous = load_result(args.contiguous)
    fragmented = load_result(args.fragmented)
    same_geometry = all(
        contiguous.get(key) == fragmented.get(key)
        for key in ("arena_bytes", "release_bytes", "native_bytes", "stripe_bytes")
    )
    starting_free_delta = abs(
        int(contiguous.get("pre_release_free_bytes", -1))
        - int(fragmented.get("pre_release_free_bytes", -1))
    )
    released_free_delta = abs(
        int(contiguous.get("after_release_free_bytes", -1))
        - int(fragmented.get("after_release_free_bytes", -1))
    )
    native_bytes = int(contiguous.get("native_bytes", 0))
    contiguous_eviction = int(contiguous.get("hot_evicted_bytes", -1))
    fragmented_eviction = int(fragmented.get("hot_evicted_bytes", -1))
    contiguous_free_drop = int(contiguous.get("free_drop_bytes", -1))
    fragmented_free_drop = int(fragmented.get("free_drop_bytes", -1))
    checks = {
        "both_measurements_valid": (
            contiguous.get("measurement_valid") is True
            and fragmented.get("measurement_valid") is True
        ),
        "same_geometry": same_geometry,
        "same_initial_free_within_64mib": starting_free_delta <= 64 * MIB,
        "same_released_free_within_64mib": released_free_delta <= 64 * MIB,
        "contiguous_uses_free_without_hot_eviction": (
            contiguous_eviction <= 64 * MIB
            and abs(native_bytes - contiguous_free_drop) <= 64 * MIB
        ),
        "fragmented_forces_material_hot_eviction": (
            fragmented_eviction >= 128 * MIB
            and fragmented_free_drop <= native_bytes - 128 * MIB
        ),
        "fragmented_byte_accounting_closes": (
            abs(native_bytes - fragmented_free_drop - fragmented_eviction)
            <= 64 * MIB
        ),
    }
    passed = all(checks.values())
    output = {
        "schema": "uvm_physical_fragmentation_cuda_malloc_ab_v1",
        "status": "proved" if passed else "inconclusive",
        "supports_fragmentation_hypothesis": passed,
        "checks": checks,
        "comparison": {
            "starting_free_delta_bytes": starting_free_delta,
            "released_free_delta_bytes": released_free_delta,
            "contiguous_hot_evicted_bytes": contiguous_eviction,
            "fragmented_hot_evicted_bytes": fragmented_eviction,
            "contiguous_free_drop_bytes": contiguous_free_drop,
            "fragmented_free_drop_bytes": fragmented_free_drop,
            "native_bytes": native_bytes,
        },
        "contiguous": contiguous,
        "fragmented": fragmented,
    }
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
