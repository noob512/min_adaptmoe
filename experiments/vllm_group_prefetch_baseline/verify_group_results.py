#!/usr/bin/env python3
"""Verify saved grouped-offload benchmark evidence without starting vLLM."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def load(path: Path) -> dict[str, object]:
    document: dict[str, object] = {"results": {}}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("benchmark_begin "):
            document["begin"] = parse_fields(line)
        elif line.startswith("SCHEDULE "):
            document["schedule"] = parse_fields(line)
        elif line.startswith("RESULT "):
            result = parse_fields(line)
            results = document["results"]
            assert isinstance(results, dict)
            results[result["backend"]] = result
        elif line.startswith("COMPARISON "):
            document["comparison"] = parse_fields(line)
        elif line.startswith("benchmark_result "):
            document["final"] = parse_fields(line)
    for required in ("begin", "schedule", "comparison", "final"):
        if required not in document:
            raise ValueError(f"missing {required} in {path}")
    return document


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
    print(f"PASS {message}")


def number(record: dict[str, str], key: str) -> float:
    value = float(record[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}={record[key]}")
    return value


def verify_functional(document: dict[str, object], label: str) -> None:
    results = document["results"]
    final = document["final"]
    assert isinstance(results, dict) and isinstance(final, dict)
    require(set(results) == {"pinned", "managed", "uva"}, f"{label} has 3 backends")
    require(
        all(result.get("functional_ok") == "1" for result in results.values()),
        f"{label} checksums and managed placement pass",
    )
    require(
        len({result["checksum"] for result in results.values()}) == 1,
        f"{label} backends produce the same checksum",
    )
    require(
        results["pinned"].get("consumer_va_stable") == "1"
        and results["managed"].get("in_place_va") == "1"
        and results["managed"].get("consumer_va_stable") == "1"
        and results["uva"].get("in_place_va") == "1"
        and results["uva"].get("consumer_va_stable") == "1",
        f"{label} static-slot and fixed-VA consumers stay stable",
    )
    require(
        final.get("functional_success") == "1"
        and final.get("total_cases") == "3"
        and final.get("managed_va_preserved") == "1"
        and final.get("cpu_physical_contiguity_controlled") == "0",
        f"{label} final contract passes without claiming PFN continuity",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", type=Path, default=SCRIPT_DIR / "RESULTS_GROUP_FULL_20260814.txt"
    )
    parser.add_argument(
        "--smoke",
        type=Path,
        default=SCRIPT_DIR / "RESULTS_GROUP_SMOKE_20260814.txt",
    )
    args = parser.parse_args()

    full = load(args.full)
    smoke = load(args.smoke)
    expected_schedule = {
        "selected_layers": "6,7,14,15",
        "slots": "0,1,0,1",
        "initial": "6,7",
        "triggers": "6->14,7->15",
    }
    require(full["schedule"] == expected_schedule, "full group selector/slot/trigger schedule")
    require(smoke["schedule"] == expected_schedule, "smoke group selector/slot/trigger schedule")
    verify_functional(full, "full")
    verify_functional(smoke, "smoke")

    begin = full["begin"]
    results = full["results"]
    comparison = full["comparison"]
    assert isinstance(begin, dict) and isinstance(results, dict)
    assert isinstance(comparison, dict)
    require(
        begin.get("total_layers") == "16"
        and begin.get("group_size") == "8"
        and begin.get("num_in_group") == "2"
        and begin.get("prefetch_step") == "2",
        "full run records upstream-style N=8/M=2/K=2 policy",
    )
    pinned_bw = number(results["pinned"], "transfer_gib_s")
    managed_bw = number(results["managed"], "transfer_gib_s")
    uva_bw = number(results["uva"], "read_gib_s")
    managed_ratio = managed_bw / pinned_bw
    managed_e2e_ratio = number(results["pinned"], "e2e_ms_p50") / number(
        results["managed"], "e2e_ms_p50"
    )
    require(
        math.isclose(
            number(comparison, "managed_transfer_vs_pinned"),
            managed_ratio,
            rel_tol=2e-3,
        ),
        "full managed/pinned transfer ratio matches RESULT records",
    )
    require(
        math.isclose(
            number(comparison, "managed_e2e_speed_vs_pinned"),
            managed_e2e_ratio,
            rel_tol=2e-3,
        ),
        "full managed/pinned e2e ratio matches RESULT records",
    )
    require(
        number(results["pinned"], "logical_bytes")
        == number(results["managed"], "logical_bytes")
        == number(results["uva"], "logical_bytes")
        == 256 * 1024**2,
        "full backends consume the same 256MiB logical weights",
    )
    print(
        "verification_result success=1 "
        f"pinned_gib_s={pinned_bw:.3f} managed_gib_s={managed_bw:.3f} "
        f"uva_gib_s={uva_bw:.3f} managed_vs_pinned={managed_ratio:.3f} "
        f"managed_e2e_vs_pinned={managed_e2e_ratio:.3f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"verification_result success=0 error={error}")
        raise SystemExit(1) from error
