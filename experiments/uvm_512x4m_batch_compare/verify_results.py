#!/usr/bin/env python3
"""Verify and aggregate the three saved 512x4MiB cpp-batch runs."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def fields(line: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def finite(record: dict[str, str], key: str) -> float:
    value = float(record[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}={record[key]}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
    print(f"PASS {message}")


def load(path: Path) -> dict[str, object]:
    document: dict[str, object] = {"samples": [], "results": {}}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("benchmark_begin "):
            document["begin"] = fields(line)
        elif line.startswith("LAYOUT "):
            document["layout"] = fields(line)
        elif line.startswith("SEMANTICS "):
            document["semantics"] = fields(line)
        elif line.startswith("SAMPLE "):
            samples = document["samples"]
            assert isinstance(samples, list)
            samples.append(fields(line))
        elif line.startswith("RESULT "):
            result = fields(line)
            results = document["results"]
            assert isinstance(results, dict)
            results[result["mode"]] = result
        elif line.startswith("COMPARISON "):
            document["comparison"] = fields(line)
        elif line.startswith("benchmark_result "):
            document["final"] = fields(line)
    for key in ("begin", "layout", "semantics", "comparison", "final"):
        if key not in document:
            raise ValueError(f"missing {key} in {path}")
    return document


def verify_run(path: Path, document: dict[str, object]) -> tuple[float, float, float, list[float]]:
    begin = document["begin"]
    layout = document["layout"]
    semantics = document["semantics"]
    samples = document["samples"]
    results = document["results"]
    comparison = document["comparison"]
    final = document["final"]
    assert isinstance(begin, dict) and isinstance(layout, dict)
    assert isinstance(semantics, dict) and isinstance(samples, list)
    assert isinstance(results, dict) and isinstance(comparison, dict)
    assert isinstance(final, dict)

    require(
        begin.get("allocations") == "512"
        and begin.get("allocation_bytes") == "4194304"
        and begin.get("logical_bytes") == "2147483648"
        and begin.get("batch_size") == "128"
        and begin.get("batches") == "4",
        f"{path.name}: exact 512x4MiB and 4x128 configuration",
    )
    require(
        layout.get("adjacent_test_ranges") == "0"
        and int(layout["min_gap_bytes"]) >= 4194304,
        f"{path.name}: all test ranges are VA-separated",
    )
    require(
        semantics.get("same_allocations") == "1"
        and semantics.get("same_stream") == "1"
        and semantics.get("same_order") == "1"
        and semantics.get("runtime_calls_both") == "512",
        f"{path.name}: paths share addresses/order/stream and 512 runtime calls",
    )
    measured = [sample for sample in samples if sample.get("phase") == "measured"]
    direct_samples = [sample for sample in measured if sample.get("mode") == "direct"]
    batch_samples = [sample for sample in measured if sample.get("mode") == "batch"]
    require(
        len(direct_samples) == len(batch_samples) == 8,
        f"{path.name}: 8 measured ABBA samples per mode",
    )
    checksums = {sample.get("checksum") for sample in measured}
    require(
        len(checksums) == 1
        and all(
            sample.get("accepted") == "512"
            and sample.get("runtime_prefetch_calls") == "512"
            and sample.get("cpu_last_prefetch") == "512"
            and sample.get("gpu_last_prefetch") == "512"
            and sample.get("va_stable") == "1"
            and sample.get("checksum_ok") == "1"
            for sample in measured
        ),
        f"{path.name}: all migrations, VAs, locations, and checksums pass",
    )
    require(
        all(sample.get("one_outer_calls") == "512" for sample in direct_samples)
        and all(sample.get("batch_outer_calls") == "4" for sample in batch_samples),
        f"{path.name}: outer call counts are exactly 512 versus 4",
    )
    require(
        set(results) == {"direct", "batch"}
        and final.get("functional_success") == "1"
        and final.get("runtime_calls_both") == "512",
        f"{path.name}: summary contract passes",
    )

    direct = results["direct"]
    batch = results["batch"]
    direct_migration = finite(direct, "migration_ms_p50")
    batch_migration = finite(batch, "migration_ms_p50")
    migration_speedup = direct_migration / batch_migration
    host_speedup = finite(direct, "host_submit_us_p50") / finite(
        batch, "host_submit_us_p50"
    )
    e2e_speedup = finite(direct, "e2e_ms_p50") / finite(batch, "e2e_ms_p50")
    require(
        math.isclose(
            finite(comparison, "batch_migration_speedup"),
            migration_speedup,
            rel_tol=1e-3,
        ),
        f"{path.name}: comparison ratio matches RESULT records",
    )

    ordered = sorted(measured, key=lambda sample: int(sample["sequence_index"]))
    paired_speedups: list[float] = []
    for offset in range(0, len(ordered), 4):
        block = ordered[offset : offset + 4]
        direct_block = [finite(sample, "migration_ms") for sample in block if sample["mode"] == "direct"]
        batch_block = [finite(sample, "migration_ms") for sample in block if sample["mode"] == "batch"]
        if len(direct_block) != 2 or len(batch_block) != 2:
            raise ValueError(f"invalid ABBA block in {path}")
        paired_speedups.append(statistics.mean(direct_block) / statistics.mean(batch_block))
    return host_speedup, migration_speedup, e2e_speedup, paired_speedups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[
            SCRIPT_DIR / f"RESULTS_FULL_RUN{index}_20260815.txt"
            for index in range(1, 4)
        ],
    )
    args = parser.parse_args()
    if len(args.paths) < 3:
        parser.error("at least three independent full runs are required")

    host_speedups: list[float] = []
    migration_speedups: list[float] = []
    e2e_speedups: list[float] = []
    paired_speedups: list[float] = []
    for path in args.paths:
        host, migration, e2e, pairs = verify_run(path, load(path))
        host_speedups.append(host)
        migration_speedups.append(migration)
        e2e_speedups.append(e2e)
        paired_speedups.extend(pairs)
        print(
            f"RUN file={path.name} host_speedup={host:.4f} "
            f"migration_speedup={migration:.4f} e2e_speedup={e2e:.4f}"
        )

    median_host = statistics.median(host_speedups)
    median_migration = statistics.median(migration_speedups)
    median_e2e = statistics.median(e2e_speedups)
    median_pair = statistics.median(paired_speedups)
    require(
        0.95 <= median_migration <= 1.05,
        "4x128 cpp_batch has no material (>5%) migration change",
    )
    print(
        "verification_result success=1 runs="
        f"{len(args.paths)} measured_samples={len(args.paths) * 16} "
        f"host_submit_speedup_median={median_host:.4f} "
        f"migration_speedup_median={median_migration:.4f} "
        f"e2e_speedup_median={median_e2e:.4f} "
        f"abba_pair_speedup_median={median_pair:.4f} "
        f"abba_pair_speedup_min={min(paired_speedups):.4f} "
        f"abba_pair_speedup_max={max(paired_speedups):.4f} "
        "outer_call_reduction=128.0 runtime_call_reduction=1.0"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"verification_result success=0 error={error}")
        raise SystemExit(1) from error
