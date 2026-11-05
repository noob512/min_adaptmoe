#!/usr/bin/env python3
"""Verify saved three-mode synchronous driver-batch benchmark results."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODES = (
    "native_async",
    "native_4x128_sync",
    "driver_4x128_sync",
)
DRIVER_COUNTERS = (
    "prefetch_gpu_batch_fast_ioctl_calls",
    "prefetch_gpu_batch_fast_core_calls",
    "prefetch_gpu_batch_fast_requested_ranges",
    "prefetch_gpu_batch_fast_completed_ranges",
    "prefetch_gpu_batch_fast_requested_bytes",
    "prefetch_gpu_batch_fast_completed_bytes",
    "prefetch_gpu_batch_fast_waits",
    "prefetch_gpu_batch_fast_fallbacks",
    "prefetch_gpu_batch_fast_errors",
)


def fields(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for token in line.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            values[key] = value
    return values


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
    print(f"PASS {message}")


def integer(record: dict[str, str], name: str) -> int:
    return int(record[name], 0)


def finite(record: dict[str, str], name: str) -> float:
    value = float(record[name])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}={record[name]}")
    return value


def load(path: Path) -> dict[str, object]:
    document: dict[str, object] = {
        "samples": [],
        "results": {},
        "orders": [],
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("benchmark_begin "):
            document["begin"] = fields(line)
        elif line.startswith("DRIVER_ABI "):
            document["abi"] = fields(line)
        elif line.startswith("LAYOUT "):
            document["layout"] = fields(line)
        elif line.startswith("SEMANTICS "):
            document["semantics"] = fields(line)
        elif line.startswith("ORDER "):
            orders = document["orders"]
            assert isinstance(orders, list)
            orders.append(fields(line))
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
    for name in (
        "begin",
        "abi",
        "layout",
        "semantics",
        "comparison",
        "final",
    ):
        if name not in document:
            raise ValueError(f"missing {name} record in {path}")
    return document


def verify_sample(
    sample: dict[str, str], allocations: int, logical_bytes: int
) -> None:
    mode = sample["mode"]
    require(mode in MODES, f"sample mode {mode} is known")
    common_ok = (
        integer(sample, "logical_bytes") == logical_bytes
        and integer(sample, "accepted") == allocations
        and integer(sample, "failed") == 0
        and integer(sample, "cpu_last_prefetch") == allocations
        and integer(sample, "resident_gpu_bytes") == logical_bytes
        and integer(sample, "resident_unknown_bytes") == 0
        and sample.get("va_stable") == "1"
        and sample.get("checksum_ok") == "1"
    )
    if not common_ok:
        raise ValueError(
            f"sample {sample.get('sequence_index')} {mode} failed a common invariant"
        )
    if finite(sample, "migration_wall_ms") <= 0:
        raise ValueError(f"sample {sample.get('sequence_index')} has non-positive time")

    if mode == "native_async":
        mode_ok = (
            integer(sample, "outer_calls") == allocations
            and integer(sample, "runtime_prefetch_calls") == allocations
            and integer(sample, "completion_waits") == 1
            and sample.get("submit_measured") == "1"
            and finite(sample, "submit_us") >= 0
            and sample.get("last_prefetch_checked") == "1"
            and integer(sample, "gpu_last_prefetch") == allocations
            and integer(sample, "legacy_api_calls") == allocations
        )
    elif mode == "native_4x128_sync":
        mode_ok = (
            integer(sample, "outer_calls") == 4
            and integer(sample, "runtime_prefetch_calls") == allocations
            and integer(sample, "completion_waits") == 4
            and sample.get("submit_measured") == "0"
            and sample.get("last_prefetch_checked") == "1"
            and integer(sample, "gpu_last_prefetch") == allocations
            and integer(sample, "legacy_api_calls") == allocations
        )
    else:
        mode_ok = (
            integer(sample, "outer_calls") == 4
            and integer(sample, "runtime_prefetch_calls") == 0
            and integer(sample, "completion_waits") == 4
            and sample.get("submit_measured") == "0"
            and sample.get("last_prefetch_checked") == "0"
            and integer(sample, "gpu_last_prefetch") == -1
            and integer(sample, "legacy_api_calls") == 0
        )
    if not mode_ok:
        raise ValueError(
            f"sample {sample.get('sequence_index')} {mode} failed its call contract"
        )

    expected_driver = {
        "prefetch_gpu_batch_fast_ioctl_calls": 4,
        "prefetch_gpu_batch_fast_core_calls": 4,
        "prefetch_gpu_batch_fast_requested_ranges": allocations,
        "prefetch_gpu_batch_fast_completed_ranges": allocations,
        "prefetch_gpu_batch_fast_requested_bytes": logical_bytes,
        "prefetch_gpu_batch_fast_completed_bytes": logical_bytes,
        "prefetch_gpu_batch_fast_waits": 4,
        "prefetch_gpu_batch_fast_fallbacks": 0,
        "prefetch_gpu_batch_fast_errors": 0,
    }
    for name in DRIVER_COUNTERS:
        expected = expected_driver[name] if mode == "driver_4x128_sync" else 0
        if integer(sample, name) != expected:
            raise ValueError(
                f"sample {sample.get('sequence_index')} {name}="
                f"{sample.get(name)}/{expected}"
            )


def verify_run(path: Path, document: dict[str, object], require_full: bool) -> tuple[float, float]:
    begin = document["begin"]
    abi = document["abi"]
    layout = document["layout"]
    semantics = document["semantics"]
    samples = document["samples"]
    results = document["results"]
    comparison = document["comparison"]
    final = document["final"]
    assert isinstance(begin, dict) and isinstance(abi, dict)
    assert isinstance(layout, dict) and isinstance(semantics, dict)
    assert isinstance(samples, list) and isinstance(results, dict)
    assert isinstance(comparison, dict) and isinstance(final, dict)

    allocations = integer(begin, "allocations")
    allocation_bytes = integer(begin, "allocation_bytes")
    logical_bytes = integer(begin, "logical_bytes")
    measured_per_mode = integer(begin, "measured_per_mode")
    warmup_per_mode = integer(begin, "warmup_per_mode")
    require(
        allocations == 4 * integer(begin, "batch_size")
        and integer(begin, "batches") == 4
        and logical_bytes == allocations * allocation_bytes,
        f"{path.name}: four equal batches and byte geometry are consistent",
    )
    if require_full:
        require(
            allocations == 512
            and allocation_bytes == 4 * 1024 * 1024
            and integer(begin, "batch_size") == 128
            and logical_bytes == 2 * 1024 * 1024 * 1024,
            f"{path.name}: exact full 512x4MiB / 4x128 configuration",
        )
    require(
        begin.get("modes") == ",".join(MODES)
        and integer(abi, "ioctl_command") == 90
        and integer(abi, "max_ranges") == 128
        and abi.get("synchronous") == "1"
        and abi.get("completion_on_return") == "1",
        f"{path.name}: versioned synchronous 128-range ABI",
    )
    require(
        layout.get("adjacent_test_ranges") == "0"
        and integer(layout, "min_gap_bytes") >= allocation_bytes,
        f"{path.name}: test ranges are VA-separated",
    )
    require(
        semantics.get("timing") == "host_monotonic_to_explicit_completion"
        and semantics.get("profile_before_after_cpu_reset") == "1"
        and semantics.get("profile_snapshot_before_validation") == "1"
        and semantics.get("driver_last_prefetch_attribute_required") == "0"
        and semantics.get("same_allocations") == "1"
        and semantics.get("same_order") == "1",
        f"{path.name}: timing, profiling, and validation boundaries",
    )

    expected_total = (warmup_per_mode + measured_per_mode) * len(MODES)
    require(
        len(samples) == expected_total,
        f"{path.name}: exact warmup+measured sample count",
    )
    for sample in samples:
        verify_sample(sample, allocations, logical_bytes)
    for phase, expected in (
        ("warmup", warmup_per_mode),
        ("measured", measured_per_mode),
    ):
        for mode in MODES:
            count = sum(
                sample.get("phase") == phase and sample.get("mode") == mode
                for sample in samples
            )
            if count != expected:
                raise ValueError(
                    f"{path.name}: {phase}/{mode} samples={count}/{expected}"
                )
    require(True, f"{path.name}: all sample contracts pass")

    require(
        set(results) == set(MODES)
        and all(
            result.get("profile_contract_ok") == "1"
            and result.get("residency_ok") == "1"
            and result.get("checksum_ok") == "1"
            and integer(result, "samples") == measured_per_mode
            for result in results.values()
        ),
        f"{path.name}: three RESULT records pass",
    )
    native_sync_ms = finite(
        results["native_4x128_sync"], "migration_wall_ms_p50"
    )
    driver_sync_ms = finite(
        results["driver_4x128_sync"], "migration_wall_ms_p50"
    )
    native_async_ms = finite(results["native_async"], "migration_wall_ms_p50")
    driver_vs_sync = native_sync_ms / driver_sync_ms
    driver_vs_async = native_async_ms / driver_sync_ms
    require(
        math.isclose(
            finite(comparison, "driver_vs_native_4x128_sync_speedup"),
            driver_vs_sync,
            rel_tol=1e-3,
        )
        and math.isclose(
            finite(comparison, "driver_vs_native_async_speedup"),
            driver_vs_async,
            rel_tol=1e-3,
        ),
        f"{path.name}: comparison ratios match RESULT medians",
    )
    require(
        final.get("functional_success") == "1"
        and final.get("profile_contract_ok") == "1"
        and final.get("residency_ok") == "1"
        and final.get("checksum_ok") == "1"
        and integer(final, "driver_ioctl_command") == 90
        and integer(final, "driver_max_ranges") == 128,
        f"{path.name}: final functional contract passes",
    )
    return driver_vs_sync, driver_vs_async


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[SCRIPT_DIR / "RESULTS_TRUE_BATCH_FULL_20260815.txt"],
    )
    parser.add_argument("--require-full", action="store_true")
    args = parser.parse_args()

    sync_speedups: list[float] = []
    async_speedups: list[float] = []
    for path in args.paths:
        sync_speedup, async_speedup = verify_run(
            path, load(path), args.require_full
        )
        sync_speedups.append(sync_speedup)
        async_speedups.append(async_speedup)
        print(
            f"RUN file={path.name} "
            f"driver_vs_native_4x128_sync_speedup={sync_speedup:.4f} "
            f"driver_vs_native_async_speedup={async_speedup:.4f}"
        )
    print(
        "verification_result success=1 "
        f"runs={len(args.paths)} "
        "functional_contract=pass "
        f"driver_vs_native_4x128_sync_speedup_median="
        f"{statistics.median(sync_speedups):.4f} "
        f"driver_vs_native_async_speedup_median="
        f"{statistics.median(async_speedups):.4f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"verification_result success=0 error={error}")
        raise SystemExit(1) from error
