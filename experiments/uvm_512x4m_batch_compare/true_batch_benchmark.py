#!/usr/bin/env python3
"""Compare native UVM prefetch with the synchronous 128-range driver batch.

This is intentionally separate from ``benchmark.py``.  The saved two-mode
direct/cpp_batch format remains unchanged while this program exercises the
versioned production ioctl which has synchronous completion semantics.
"""

from __future__ import annotations

import argparse
import ctypes
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import benchmark as legacy


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STATS = Path("/proc/driver/nvidia-uvm/latency_stats")

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
PROFILE_DIAGNOSTICS = (
    "migrate_profile.copy_push_memcopy.calls",
    "migrate_profile.copy_push_memcopy.pages",
    "migrate_profile.copy_push_memcopy.bytes",
    "migrate_profile.populate_gpu.calls",
    "migrate_profile.populate_gpu.pages",
    "migrate_profile.populate_gpu.bytes",
    "migrate_profile.map_total.calls",
    "migrate_profile.map_total.pages",
    "migrate_profile.map_total.bytes",
    "migrate_profile.map_gpu_to.calls",
    "migrate_profile.map_gpu_to.pages",
    "migrate_profile.map_gpu_to.bytes",
    "migrate_profile.tlb_phys_invalidate.calls",
    "migrate_profile.tlb_phys_invalidate.pages",
    "migrate_profile.tlb_phys_invalidate.bytes",
)

_SECTION_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):\s*$")
_PROFILE_RE = re.compile(
    r"^\s+([a-z][a-z0-9_]*)\s+"
    r"calls=(\d+)\s+ns=(\d+)\s+avg_ns=(\d+)\s+"
    r"pages=(\d+)\s+bytes=(\d+)\s*$"
)
_COUNTER_RE = re.compile(
    r"^\s*([a-z][a-z0-9_]*)\s+(0x[0-9a-fA-F]+|\d+)\s*$"
)


@dataclass(frozen=True)
class DriverCall:
    accepted: int
    failed: int
    requested_bytes: int
    completed_bytes: int
    rm_status: int


@dataclass(frozen=True)
class Sample:
    phase: str
    sequence_index: int
    mode_index: int
    mode: str
    accepted: int
    failed: int
    outer_calls: int
    runtime_calls: int
    completion_waits: int
    submit_us: float | None
    migration_wall_ms: float
    checksum_wall_ms: float
    migration_gib_s: float
    checksum: int
    cpu_location_before: int
    gpu_location_after: int
    last_prefetch_checked: bool
    resident_gpu_bytes: int
    resident_unknown_bytes: int
    va_stable: bool
    checksum_ok: bool
    profile: Mapping[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=SCRIPT_DIR / "libuvm_batch_shim.so",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--allocations", type=int, default=512)
    parser.add_argument("--allocation-mib", type=float, default=4.0)
    parser.add_argument("--guard-mib", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-per-mode", type=int, default=2)
    parser.add_argument("--measured-per-mode", type=int, default=6)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    args = parser.parse_args()

    if (
        args.device < 0
        or args.allocations < 4
        or args.batch_size < 1
        or args.batch_size > 128
        or args.allocations != 4 * args.batch_size
        or args.allocation_mib <= 0
        or args.guard_mib < 0
        or args.warmup_per_mode < 0
        or args.measured_per_mode < 2
        or args.warmup_per_mode % 2 != 0
        or args.measured_per_mode % 2 != 0
    ):
        parser.error(
            "require allocations=4*batch_size, batch_size<=128, positive "
            "allocation size, and even warmup/measured counts per mode"
        )
    args.allocation_bytes = int(args.allocation_mib * legacy.MIB)
    args.guard_bytes = int(args.guard_mib * legacy.MIB)
    if (
        args.allocation_bytes <= 0
        or args.allocation_bytes % ctypes.sizeof(ctypes.c_uint32) != 0
        or args.guard_bytes < 0
    ):
        parser.error("allocation/guard byte sizes are invalid")
    return args


def load_library(path: Path) -> ctypes.CDLL:
    library = legacy.load_library(path)
    void_p = ctypes.c_void_p
    size_t = ctypes.c_size_t
    uint32 = ctypes.c_uint32
    uint64 = ctypes.c_uint64

    library.uvm512_driver_batch_context_create.argtypes = [
        ctypes.c_int,
        size_t,
        size_t,
    ]
    library.uvm512_driver_batch_context_create.restype = void_p
    library.uvm512_driver_batch_context_destroy.argtypes = [void_p]
    library.uvm512_driver_batch_context_destroy.restype = None
    library.uvm512_driver_last_error.argtypes = []
    library.uvm512_driver_last_error.restype = ctypes.c_char_p
    library.uvm512_driver_batch_ioctl_command.argtypes = []
    library.uvm512_driver_batch_ioctl_command.restype = uint32
    library.uvm512_driver_batch_max_ranges.argtypes = []
    library.uvm512_driver_batch_max_ranges.restype = size_t
    library.uvm512_driver_batch_params_size.argtypes = []
    library.uvm512_driver_batch_params_size.restype = size_t
    library.uvm512_driver_prefetch_batch_fast.argtypes = [
        void_p,
        ctypes.POINTER(size_t),
        ctypes.POINTER(size_t),
        size_t,
        ctypes.POINTER(uint32),
        ctypes.POINTER(uint32),
        ctypes.POINTER(uint64),
        ctypes.POINTER(uint64),
        ctypes.POINTER(uint32),
    ]
    library.uvm512_driver_prefetch_batch_fast.restype = ctypes.c_int
    library.uvm512_driver_query_residency.argtypes = [
        void_p,
        ctypes.POINTER(size_t),
        ctypes.POINTER(size_t),
        size_t,
        ctypes.POINTER(uint32),
        ctypes.POINTER(uint64),
        ctypes.POINTER(uint64),
    ]
    library.uvm512_driver_query_residency.restype = ctypes.c_int
    return library


def driver_error(library: ctypes.CDLL) -> str:
    message = library.uvm512_driver_last_error()
    return message.decode("utf-8", errors="replace") if message else "unknown"


def parse_latency_stats(text: str) -> dict[str, int]:
    """Flatten the profile while preserving scalar counter names."""
    values: dict[str, int] = {}
    section: str | None = None
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        section_match = _SECTION_RE.fullmatch(line)
        if section_match:
            section = section_match.group(1)
            continue
        profile_match = _PROFILE_RE.fullmatch(line)
        if profile_match:
            if section is None:
                raise ValueError(
                    f"profile row before section at line {line_number}"
                )
            name, calls, nanoseconds, _average, pages, bytes_ = (
                profile_match.groups()
            )
            prefix = f"{section}.{name}"
            values[f"{prefix}.calls"] = int(calls)
            values[f"{prefix}.ns"] = int(nanoseconds)
            values[f"{prefix}.pages"] = int(pages)
            values[f"{prefix}.bytes"] = int(bytes_)
            continue
        counter_match = _COUNTER_RE.fullmatch(line)
        if counter_match:
            name, value = counter_match.groups()
            values[name] = int(value, 0)
            section = None
            continue
        raise ValueError(
            f"unrecognized latency_stats line {line_number}: {line!r}"
        )
    return values


def read_profile(path: Path) -> dict[str, int]:
    try:
        return parse_latency_stats(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RuntimeError(f"cannot read driver profile {path}: {error}") from error


def profile_delta(
    before: Mapping[str, int], after: Mapping[str, int]
) -> dict[str, int]:
    delta: dict[str, int] = {}
    for name in set(before) | set(after):
        value = int(after.get(name, 0)) - int(before.get(name, 0))
        if value < 0:
            raise RuntimeError(
                f"driver profile counter moved backwards: {name}={value}"
            )
        delta[name] = value
    return delta


def balanced_sequence(count_per_mode: int) -> list[str]:
    """Use adjacent forward/reverse permutations to counter order effects."""
    permutations = (
        (MODES[0], MODES[1], MODES[2]),
        (MODES[2], MODES[1], MODES[0]),
        (MODES[1], MODES[2], MODES[0]),
        (MODES[0], MODES[2], MODES[1]),
        (MODES[2], MODES[0], MODES[1]),
        (MODES[1], MODES[0], MODES[2]),
    )
    sequence: list[str] = []
    for index in range(count_per_mode):
        sequence.extend(permutations[index % len(permutations)])
    return sequence


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def profile_value(profile: Mapping[str, int], name: str) -> int:
    if name not in profile:
        raise RuntimeError(f"driver profile is missing required field {name}")
    return int(profile[name])


def validate_profile(
    profile: Mapping[str, int], mode: str, allocations: int, logical_bytes: int
) -> None:
    api_calls = profile_value(profile, "migrate_profile.api_total.calls")
    expected_api_calls = 0 if mode == "driver_4x128_sync" else allocations
    if api_calls != expected_api_calls:
        raise RuntimeError(
            f"{mode} legacy api_total.calls={api_calls}/{expected_api_calls}"
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
        observed = profile_value(profile, name)
        if observed != expected:
            raise RuntimeError(f"{mode} {name}={observed}/{expected}")


def driver_call(
    library: ctypes.CDLL,
    context: ctypes.c_void_p,
    pointer_array: ctypes.Array,
    size_array: ctypes.Array,
    count: int,
) -> DriverCall:
    success = ctypes.c_uint32()
    failure = ctypes.c_uint32()
    requested = ctypes.c_uint64()
    completed = ctypes.c_uint64()
    rm_status = ctypes.c_uint32()
    result = library.uvm512_driver_prefetch_batch_fast(
        context,
        pointer_array,
        size_array,
        count,
        ctypes.byref(success),
        ctypes.byref(failure),
        ctypes.byref(requested),
        ctypes.byref(completed),
        ctypes.byref(rm_status),
    )
    if result != 0:
        raise RuntimeError(
            f"driver batch ioctl failed: wrapper={result} "
            f"rmStatus={rm_status.value} error={driver_error(library)}"
        )
    return DriverCall(
        accepted=int(success.value),
        failed=int(failure.value),
        requested_bytes=int(requested.value),
        completed_bytes=int(completed.value),
        rm_status=int(rm_status.value),
    )


def query_residency(
    library: ctypes.CDLL,
    context: ctypes.c_void_p,
    pointers: ctypes.Array,
    sizes: ctypes.Array,
    count: int,
) -> tuple[int, int, int]:
    success = ctypes.c_uint32()
    gpu_bytes = ctypes.c_uint64()
    unknown_bytes = ctypes.c_uint64()
    result = library.uvm512_driver_query_residency(
        context,
        pointers,
        sizes,
        count,
        ctypes.byref(success),
        ctypes.byref(gpu_bytes),
        ctypes.byref(unknown_bytes),
    )
    if result != 0:
        raise RuntimeError(
            f"residency query failed: wrapper={result} error={driver_error(library)}"
        )
    return int(success.value), int(gpu_bytes.value), int(unknown_bytes.value)


def run_one_sample(
    *,
    args: argparse.Namespace,
    library: ctypes.CDLL,
    dataset: ctypes.c_void_p,
    stream: ctypes.c_void_p,
    driver_context: ctypes.c_void_p,
    pointers: tuple[int, ...],
    all_pointer_array: ctypes.Array,
    all_size_array: ctypes.Array,
    batches: list[tuple[ctypes.Array, ctypes.Array]],
    expected_checksum: int,
    phase: str,
    sequence_index: int,
    mode_index: int,
    mode: str,
) -> Sample:
    reset_accepted = library.uvm512_dataset_prefetch_cpu_and_sync(dataset, stream)
    if reset_accepted != args.allocations:
        raise RuntimeError(
            f"CPU reset accepted {reset_accepted}/{args.allocations} ranges"
        )
    cpu_location = legacy.count_last_location(
        library, dataset, legacy.CUDA_CPU_DEVICE_ID
    )
    if cpu_location != args.allocations:
        raise RuntimeError(
            f"CPU LastPrefetchLocation {cpu_location}/{args.allocations}"
        )
    if legacy.pointer_snapshot(library, dataset, args.allocations) != pointers:
        raise RuntimeError("managed virtual addresses changed before sample")

    # Read after the reset migration so it is excluded from the measured delta.
    profile_before = read_profile(args.stats)
    profile_before_confirm = read_profile(args.stats)
    boundary_names = (
        "migrate_profile.api_total.calls",
        "migrate_profile.map_total.calls",
        "prefetch_gpu_batch_fast_ioctl_calls",
    )
    print(
        "PROFILE_BEFORE "
        f"sequence_index={sequence_index} mode={mode} "
        + " ".join(
            f"{name.replace('.', '_')}={profile_before_confirm[name]} "
            f"prior={profile_before[name]}"
            for name in boundary_names
        )
    )
    profile_before = profile_before_confirm
    library.uvm512_prefetch_counter_reset()
    accepted = 0
    failed = 0
    requested_bytes = 0
    completed_bytes = 0
    submit_us: float | None = None

    wall_begin = time.perf_counter_ns()
    if mode == "native_async":
        for pointer in pointers:
            accepted += library.uvm512_prefetch_one(
                pointer, args.allocation_bytes, args.device, stream
            )
        submit_end = time.perf_counter_ns()
        legacy.check_cuda(
            library,
            library.uvm512_stream_synchronize(stream),
            "native_async completion synchronize",
        )
        submit_us = (submit_end - wall_begin) / 1000.0
        outer_calls = args.allocations
        runtime_calls = args.allocations
        completion_waits = 1
    elif mode == "native_4x128_sync":
        for pointer_array, size_array in batches:
            accepted += library.uvm512_prefetch_batch(
                pointer_array,
                size_array,
                args.batch_size,
                args.device,
                stream,
            )
            legacy.check_cuda(
                library,
                library.uvm512_stream_synchronize(stream),
                "native_4x128_sync batch synchronize",
            )
        outer_calls = 4
        runtime_calls = args.allocations
        completion_waits = 4
    elif mode == "driver_4x128_sync":
        for pointer_array, size_array in batches:
            outcome = driver_call(
                library,
                driver_context,
                pointer_array,
                size_array,
                args.batch_size,
            )
            accepted += outcome.accepted
            failed += outcome.failed
            requested_bytes += outcome.requested_bytes
            completed_bytes += outcome.completed_bytes
            if outcome.rm_status != 0:
                raise RuntimeError(
                    f"driver batch returned rmStatus={outcome.rm_status}"
                )
        outer_calls = 4
        runtime_calls = 0
        completion_waits = 4
    else:
        raise AssertionError(f"unknown mode: {mode}")
    wall_end = time.perf_counter_ns()

    profile_after = read_profile(args.stats)
    print(
        "PROFILE_AFTER "
        f"sequence_index={sequence_index} mode={mode} "
        + " ".join(
            f"{name.replace('.', '_')}={profile_after[name]}"
            for name in boundary_names
        )
    )
    profile = profile_delta(profile_before, profile_after)
    logical_bytes = args.allocations * args.allocation_bytes
    validate_profile(profile, mode, args.allocations, logical_bytes)

    one_calls = int(library.uvm512_prefetch_one_outer_calls())
    batch_calls = int(library.uvm512_prefetch_batch_outer_calls())
    shim_runtime_calls = int(library.uvm512_runtime_prefetch_calls())
    expected_one = args.allocations if mode == "native_async" else 0
    expected_batch = 4 if mode == "native_4x128_sync" else 0
    if (one_calls, batch_calls, shim_runtime_calls) != (
        expected_one,
        expected_batch,
        runtime_calls,
    ):
        raise RuntimeError(
            "shim call-count mismatch: "
            f"one={one_calls}/{expected_one} batch={batch_calls}/{expected_batch} "
            f"runtime={shim_runtime_calls}/{runtime_calls}"
        )
    if accepted != args.allocations or failed != 0:
        raise RuntimeError(
            f"{mode} accepted={accepted}/{args.allocations} failed={failed}"
        )
    if mode == "driver_4x128_sync" and (
        requested_bytes != logical_bytes or completed_bytes != logical_bytes
    ):
        raise RuntimeError(
            f"driver byte result requested={requested_bytes}/{logical_bytes} "
            f"completed={completed_bytes}/{logical_bytes}"
        )

    # This query is deliberately after the profile snapshot so cmd 81 does not
    # contaminate the measured driver-fast counters.
    resident_count, resident_gpu_bytes, resident_unknown_bytes = query_residency(
        library,
        driver_context,
        all_pointer_array,
        all_size_array,
        args.allocations,
    )
    if (
        resident_count != args.allocations
        or resident_gpu_bytes != logical_bytes
        or resident_unknown_bytes != 0
    ):
        raise RuntimeError(
            f"{mode} residency count={resident_count}/{args.allocations} "
            f"gpu_bytes={resident_gpu_bytes}/{logical_bytes} "
            f"unknown_bytes={resident_unknown_bytes}"
        )

    checksum_begin = time.perf_counter_ns()
    legacy.check_cuda(
        library,
        library.uvm512_dataset_launch_checksum(dataset, stream),
        "launch full checksum",
    )
    legacy.check_cuda(
        library,
        library.uvm512_stream_synchronize(stream),
        "synchronize full checksum",
    )
    checksum_end = time.perf_counter_ns()
    checksum_value = ctypes.c_uint64()
    legacy.check_cuda(
        library,
        library.uvm512_dataset_read_checksum(
            dataset, ctypes.byref(checksum_value)
        ),
        "read checksum",
    )

    last_prefetch_checked = mode != "driver_4x128_sync"
    gpu_location = -1
    if last_prefetch_checked:
        gpu_location = legacy.count_last_location(library, dataset, args.device)
        if gpu_location != args.allocations:
            raise RuntimeError(
                f"GPU LastPrefetchLocation {gpu_location}/{args.allocations}"
            )
    va_stable = (
        legacy.pointer_snapshot(library, dataset, args.allocations) == pointers
    )
    checksum_ok = int(checksum_value.value) == expected_checksum
    if not va_stable or not checksum_ok:
        raise RuntimeError(
            f"postcondition failed: va_stable={va_stable} "
            f"checksum={checksum_value.value}/{expected_checksum}"
        )

    migration_wall_ms = (wall_end - wall_begin) / 1_000_000.0
    return Sample(
        phase=phase,
        sequence_index=sequence_index,
        mode_index=mode_index,
        mode=mode,
        accepted=accepted,
        failed=failed,
        outer_calls=outer_calls,
        runtime_calls=runtime_calls,
        completion_waits=completion_waits,
        submit_us=submit_us,
        migration_wall_ms=migration_wall_ms,
        checksum_wall_ms=(checksum_end - checksum_begin) / 1_000_000.0,
        migration_gib_s=(logical_bytes / legacy.GIB)
        / (migration_wall_ms / 1000.0),
        checksum=int(checksum_value.value),
        cpu_location_before=cpu_location,
        gpu_location_after=gpu_location,
        last_prefetch_checked=last_prefetch_checked,
        resident_gpu_bytes=resident_gpu_bytes,
        resident_unknown_bytes=resident_unknown_bytes,
        va_stable=va_stable,
        checksum_ok=checksum_ok,
        profile=profile,
    )


def print_sample(sample: Sample, logical_bytes: int) -> None:
    submit_value = sample.submit_us if sample.submit_us is not None else -1.0
    fields = [
        "SAMPLE",
        f"phase={sample.phase}",
        f"sequence_index={sample.sequence_index}",
        f"mode_index={sample.mode_index}",
        f"mode={sample.mode}",
        f"logical_bytes={logical_bytes}",
        f"accepted={sample.accepted}",
        f"failed={sample.failed}",
        f"outer_calls={sample.outer_calls}",
        f"runtime_prefetch_calls={sample.runtime_calls}",
        f"completion_waits={sample.completion_waits}",
        f"submit_measured={int(sample.submit_us is not None)}",
        f"submit_us={submit_value:.3f}",
        f"migration_wall_ms={sample.migration_wall_ms:.3f}",
        f"checksum_wall_ms={sample.checksum_wall_ms:.3f}",
        f"migration_gib_s={sample.migration_gib_s:.3f}",
        f"cpu_last_prefetch={sample.cpu_location_before}",
        f"gpu_last_prefetch={sample.gpu_location_after}",
        f"last_prefetch_checked={int(sample.last_prefetch_checked)}",
        f"resident_gpu_bytes={sample.resident_gpu_bytes}",
        f"resident_unknown_bytes={sample.resident_unknown_bytes}",
        f"va_stable={int(sample.va_stable)}",
        f"checksum={sample.checksum}",
        f"checksum_ok={int(sample.checksum_ok)}",
        f"legacy_api_calls={sample.profile['migrate_profile.api_total.calls']}",
    ]
    for name in DRIVER_COUNTERS:
        fields.append(f"{name}={sample.profile[name]}")
    for name in PROFILE_DIAGNOSTICS:
        fields.append(f"diag_{name.replace('.', '_')}={sample.profile.get(name, -1)}")
    print(" ".join(fields))


def print_result(mode: str, samples: list[Sample]) -> None:
    migration = [sample.migration_wall_ms for sample in samples]
    bandwidth = [sample.migration_gib_s for sample in samples]
    checksum = [sample.checksum_wall_ms for sample in samples]
    submit = [sample.submit_us for sample in samples if sample.submit_us is not None]
    submit_p50 = statistics.median(submit) if submit else -1.0
    print(
        "RESULT "
        f"mode={mode} samples={len(samples)} "
        f"outer_calls_per_sample={samples[0].outer_calls} "
        f"runtime_prefetch_calls_per_sample={samples[0].runtime_calls} "
        f"completion_waits_per_sample={samples[0].completion_waits} "
        f"submit_us_p50={submit_p50:.3f} "
        f"migration_wall_ms_p50={statistics.median(migration):.3f} "
        f"migration_wall_ms_p95={percentile(migration, 0.95):.3f} "
        f"migration_gib_s_p50={statistics.median(bandwidth):.3f} "
        f"checksum_wall_ms_p50={statistics.median(checksum):.3f} "
        "va_stable=1 checksum_ok=1 residency_ok=1 profile_contract_ok=1"
    )


def run(args: argparse.Namespace, library: ctypes.CDLL) -> int:
    legacy.check_cuda(library, library.uvm512_set_device(args.device), "cudaSetDevice")
    dataset = library.uvm512_dataset_create(
        args.allocations, args.allocation_bytes, args.guard_bytes
    )
    if not dataset:
        raise RuntimeError("failed to create managed dataset")
    stream = None
    driver_context = None
    try:
        pointers = legacy.pointer_snapshot(library, dataset, args.allocations)
        adjacent, min_gap = legacy.layout_metrics(pointers, args.allocation_bytes)
        if args.guard_bytes > 0 and adjacent != 0:
            raise RuntimeError(
                f"guard allocations failed to separate {adjacent} test ranges"
            )
        expected_checksum = int(
            library.uvm512_dataset_expected_checksum(dataset)
        )
        logical_bytes = args.allocations * args.allocation_bytes
        stream = library.uvm512_stream_create()
        if not stream:
            raise RuntimeError("failed to create CUDA stream")

        all_pointer_type = ctypes.c_size_t * args.allocations
        all_size_type = ctypes.c_size_t * args.allocations
        all_pointer_array = all_pointer_type(*pointers)
        all_size_array = all_size_type(
            *([args.allocation_bytes] * args.allocations)
        )
        batch_pointer_type = ctypes.c_size_t * args.batch_size
        batch_size_type = ctypes.c_size_t * args.batch_size
        batches: list[tuple[ctypes.Array, ctypes.Array]] = []
        for begin in range(0, args.allocations, args.batch_size):
            end = begin + args.batch_size
            batches.append(
                (
                    batch_pointer_type(*pointers[begin:end]),
                    batch_size_type(*([args.allocation_bytes] * args.batch_size)),
                )
            )
        if len(batches) != 4:
            raise AssertionError("each path must execute exactly four batches")

        driver_context = library.uvm512_driver_batch_context_create(
            args.device, pointers[0], args.allocation_bytes
        )
        if not driver_context:
            raise RuntimeError(
                "synchronous driver batch ioctl is unavailable: "
                f"{driver_error(library)}"
            )
        ioctl_command = int(library.uvm512_driver_batch_ioctl_command())
        driver_max_ranges = int(library.uvm512_driver_batch_max_ranges())
        params_size = int(library.uvm512_driver_batch_params_size())
        if ioctl_command != 90 or driver_max_ranges != 128:
            raise RuntimeError(
                f"unexpected driver ABI cmd={ioctl_command} "
                f"max_ranges={driver_max_ranges}"
            )

        print(
            "benchmark_begin "
            f"device={args.device} allocations={args.allocations} "
            f"allocation_bytes={args.allocation_bytes} "
            f"logical_bytes={logical_bytes} batch_size={args.batch_size} "
            f"batches=4 warmup_per_mode={args.warmup_per_mode} "
            f"measured_per_mode={args.measured_per_mode} modes={','.join(MODES)}"
        )
        print(
            "DRIVER_ABI "
            f"ioctl_command={ioctl_command} max_ranges={driver_max_ranges} "
            f"params_size={params_size} synchronous=1 completion_on_return=1"
        )
        print(
            "LAYOUT "
            f"test_allocations={args.allocations} "
            f"guard_allocations={args.allocations - 1 if args.guard_bytes else 0} "
            f"guard_bytes={args.guard_bytes} adjacent_test_ranges={adjacent} "
            f"min_gap_bytes={min_gap} guard_initialized=0 guard_prefetched=0"
        )
        print(
            "SEMANTICS "
            f"native_async={args.allocations}_async_then_1_sync "
            f"native_4x128_sync=4_groups_of_{args.batch_size}_then_sync_each "
            f"driver_4x128_sync=4_synchronous_ioctl_{ioctl_command} "
            "timing=host_monotonic_to_explicit_completion "
            "profile_before_after_cpu_reset=1 profile_snapshot_before_validation=1 "
            "residency_query=production_cmd81_action4_after_profile "
            "driver_last_prefetch_attribute_required=0 same_allocations=1 "
            "same_order=1 same_cuda_checksum_stream=1"
        )

        mode_indices = {mode: 0 for mode in MODES}
        all_samples: list[Sample] = []
        phases = (
            ("warmup", balanced_sequence(args.warmup_per_mode)),
            ("measured", balanced_sequence(args.measured_per_mode)),
        )
        sequence_index = 0
        for phase, sequence in phases:
            print(f"ORDER phase={phase} modes={','.join(sequence)}")
            for mode in sequence:
                sample = run_one_sample(
                    args=args,
                    library=library,
                    dataset=dataset,
                    stream=stream,
                    driver_context=driver_context,
                    pointers=pointers,
                    all_pointer_array=all_pointer_array,
                    all_size_array=all_size_array,
                    batches=batches,
                    expected_checksum=expected_checksum,
                    phase=phase,
                    sequence_index=sequence_index,
                    mode_index=mode_indices[mode],
                    mode=mode,
                )
                print_sample(sample, logical_bytes)
                all_samples.append(sample)
                mode_indices[mode] += 1
                sequence_index += 1

        measured_by_mode = {
            mode: [
                sample
                for sample in all_samples
                if sample.phase == "measured" and sample.mode == mode
            ]
            for mode in MODES
        }
        for mode, samples in measured_by_mode.items():
            if len(samples) != args.measured_per_mode:
                raise RuntimeError(f"{mode} measured sample count mismatch")
            print_result(mode, samples)

        native_async = statistics.median(
            sample.migration_wall_ms
            for sample in measured_by_mode["native_async"]
        )
        native_sync = statistics.median(
            sample.migration_wall_ms
            for sample in measured_by_mode["native_4x128_sync"]
        )
        driver_sync = statistics.median(
            sample.migration_wall_ms
            for sample in measured_by_mode["driver_4x128_sync"]
        )
        print(
            "COMPARISON "
            f"driver_vs_native_4x128_sync_speedup={native_sync / driver_sync:.4f} "
            f"driver_vs_native_async_speedup={native_async / driver_sync:.4f} "
            f"native_async_vs_native_4x128_sync_speedup={native_sync / native_async:.4f}"
        )
        print(
            "benchmark_result functional_success=1 "
            f"allocations={args.allocations} allocation_bytes={args.allocation_bytes} "
            "modes=3 batches_per_grouped_mode=4 "
            f"driver_ioctl_command={ioctl_command} driver_max_ranges={driver_max_ranges} "
            "va_stable=1 checksum_ok=1 residency_ok=1 profile_contract_ok=1 "
            "driver_last_prefetch_attribute_required=0"
        )
        return 0
    finally:
        if driver_context:
            library.uvm512_driver_batch_context_destroy(driver_context)
        if stream:
            library.uvm512_stream_destroy(stream)
        library.uvm512_dataset_destroy(dataset)


def main() -> int:
    args = parse_args()
    try:
        library = load_library(args.library)
        return run(args, library)
    except Exception as error:  # noqa: BLE001 - emit a machine-readable failure.
        print(f"benchmark_result functional_success=0 error={error}")
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
