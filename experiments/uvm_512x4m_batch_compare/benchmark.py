#!/usr/bin/env python3
"""Compare Python direct UVM prefetch calls with the current cpp_batch shape."""

from __future__ import annotations

import argparse
import ctypes
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024
CUDA_CPU_DEVICE_ID = -1


@dataclass(frozen=True)
class Sample:
    phase: str
    sequence_index: int
    mode_index: int
    mode: str
    accepted: int
    one_outer_calls: int
    batch_outer_calls: int
    runtime_calls: int
    host_submit_us: float
    migration_ms: float
    checksum_read_ms: float
    e2e_ms: float
    migration_gib_s: float
    checksum: int
    cpu_location_before: int
    gpu_location_after: int
    va_stable: bool
    checksum_ok: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use one managed allocation set to compare 512 Python calls with "
            "four Python-to-C batch calls whose C++ body still issues 512 "
            "cudaMemPrefetchAsync calls."
        )
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=Path(__file__).with_name("libuvm_batch_shim.so"),
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--allocations", type=int, default=512)
    parser.add_argument("--allocation-mib", type=float, default=4.0)
    parser.add_argument("--guard-mib", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-per-mode", type=int, default=2)
    parser.add_argument("--measured-per-mode", type=int, default=8)
    args = parser.parse_args()
    if (
        args.device < 0
        or args.allocations < 2
        or args.batch_size < 1
        or args.allocations != 4 * args.batch_size
        or args.allocation_mib <= 0
        or args.guard_mib < 0
        or args.warmup_per_mode < 0
        or args.measured_per_mode < 2
        or args.warmup_per_mode % 2 != 0
        or args.measured_per_mode % 2 != 0
    ):
        parser.error(
            "require allocations=4*batch_size, allocations>=2, positive "
            "allocation size, and even warmup/measured counts per mode"
        )
    args.allocation_bytes = int(args.allocation_mib * MIB)
    args.guard_bytes = int(args.guard_mib * MIB)
    if (
        args.allocation_bytes <= 0
        or args.allocation_bytes % ctypes.sizeof(ctypes.c_uint32) != 0
        or args.guard_bytes < 0
    ):
        parser.error("allocation/guard byte sizes are invalid")
    return args


def load_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path.resolve()))
    void_p = ctypes.c_void_p
    size_t = ctypes.c_size_t
    uint64 = ctypes.c_uint64

    library.uvm512_set_device.argtypes = [ctypes.c_int]
    library.uvm512_set_device.restype = ctypes.c_int
    library.uvm512_cuda_error_string.argtypes = [ctypes.c_int]
    library.uvm512_cuda_error_string.restype = ctypes.c_char_p
    library.uvm512_dataset_create.argtypes = [size_t, size_t, size_t]
    library.uvm512_dataset_create.restype = void_p
    library.uvm512_dataset_create_view.argtypes = [
        size_t,
        size_t,
        size_t,
        size_t,
    ]
    library.uvm512_dataset_create_view.restype = void_p
    # Additive ABI used by the fragmented-plan probe. Keep loading compatible
    # with an older already-built shim when that probe arm is not requested.
    try:
        create_offset_cycle = library.uvm512_dataset_create_offset_cycle
    except AttributeError:
        create_offset_cycle = None
    if create_offset_cycle is not None:
        create_offset_cycle.argtypes = [size_t] * 6
        create_offset_cycle.restype = void_p
    library.uvm512_dataset_destroy.argtypes = [void_p]
    library.uvm512_dataset_destroy.restype = None
    library.uvm512_dataset_count.argtypes = [void_p]
    library.uvm512_dataset_count.restype = size_t
    library.uvm512_dataset_allocation_bytes.argtypes = [void_p]
    library.uvm512_dataset_allocation_bytes.restype = size_t
    library.uvm512_dataset_pointer.argtypes = [void_p, size_t]
    library.uvm512_dataset_pointer.restype = size_t
    library.uvm512_dataset_expected_checksum.argtypes = [void_p]
    library.uvm512_dataset_expected_checksum.restype = uint64
    library.uvm512_dataset_reseed.argtypes = [void_p, ctypes.c_uint32]
    library.uvm512_dataset_reseed.restype = ctypes.c_int

    library.uvm512_stream_create.argtypes = []
    library.uvm512_stream_create.restype = void_p
    library.uvm512_stream_synchronize.argtypes = [void_p]
    library.uvm512_stream_synchronize.restype = ctypes.c_int
    library.uvm512_stream_destroy.argtypes = [void_p]
    library.uvm512_stream_destroy.restype = None
    library.uvm512_event_create.argtypes = []
    library.uvm512_event_create.restype = void_p
    library.uvm512_event_record.argtypes = [void_p, void_p]
    library.uvm512_event_record.restype = ctypes.c_int
    library.uvm512_event_synchronize.argtypes = [void_p]
    library.uvm512_event_synchronize.restype = ctypes.c_int
    library.uvm512_event_elapsed_ms.argtypes = [
        void_p,
        void_p,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.uvm512_event_elapsed_ms.restype = ctypes.c_int
    library.uvm512_event_destroy.argtypes = [void_p]
    library.uvm512_event_destroy.restype = None

    library.uvm512_dataset_prefetch_cpu_and_sync.argtypes = [void_p, void_p]
    library.uvm512_dataset_prefetch_cpu_and_sync.restype = ctypes.c_int
    library.uvm512_prefetch_counter_reset.argtypes = []
    library.uvm512_prefetch_counter_reset.restype = None
    library.uvm512_prefetch_one.argtypes = [
        size_t,
        size_t,
        ctypes.c_int,
        void_p,
    ]
    library.uvm512_prefetch_one.restype = ctypes.c_int
    library.uvm512_prefetch_batch.argtypes = [
        ctypes.POINTER(size_t),
        ctypes.POINTER(size_t),
        size_t,
        ctypes.c_int,
        void_p,
    ]
    library.uvm512_prefetch_batch.restype = ctypes.c_int
    library.uvm512_prefetch_one_outer_calls.argtypes = []
    library.uvm512_prefetch_one_outer_calls.restype = uint64
    library.uvm512_prefetch_batch_outer_calls.argtypes = []
    library.uvm512_prefetch_batch_outer_calls.restype = uint64
    library.uvm512_runtime_prefetch_calls.argtypes = []
    library.uvm512_runtime_prefetch_calls.restype = uint64

    library.uvm512_dataset_launch_checksum.argtypes = [void_p, void_p]
    library.uvm512_dataset_launch_checksum.restype = ctypes.c_int
    library.uvm512_dataset_read_checksum.argtypes = [
        void_p,
        ctypes.POINTER(uint64),
    ]
    library.uvm512_dataset_read_checksum.restype = ctypes.c_int
    library.uvm512_dataset_count_last_prefetch_location.argtypes = [
        void_p,
        ctypes.c_int,
        ctypes.POINTER(size_t),
    ]
    library.uvm512_dataset_count_last_prefetch_location.restype = ctypes.c_int
    return library


def check_cuda(library: ctypes.CDLL, status: int, operation: str) -> None:
    if status == 0:
        return
    message = library.uvm512_cuda_error_string(status)
    decoded = message.decode("utf-8", errors="replace") if message else "unknown"
    raise RuntimeError(f"{operation} failed: {decoded} ({status})")


def elapsed_ms(
    library: ctypes.CDLL, begin: ctypes.c_void_p, end: ctypes.c_void_p
) -> float:
    value = ctypes.c_float()
    check_cuda(
        library,
        library.uvm512_event_elapsed_ms(begin, end, ctypes.byref(value)),
        "cudaEventElapsedTime",
    )
    return float(value.value)


def abba_sequence(count_per_mode: int) -> list[str]:
    sequence: list[str] = []
    for _ in range(count_per_mode // 2):
        sequence.extend(("direct", "batch", "batch", "direct"))
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


def pointer_snapshot(
    library: ctypes.CDLL, dataset: ctypes.c_void_p, count: int
) -> tuple[int, ...]:
    result = tuple(
        int(library.uvm512_dataset_pointer(dataset, index))
        for index in range(count)
    )
    if any(pointer == 0 for pointer in result):
        raise RuntimeError("dataset returned a null test-allocation pointer")
    return result


def layout_metrics(pointers: tuple[int, ...], allocation_bytes: int) -> tuple[int, int]:
    ordered = sorted(pointers)
    gaps: list[int] = []
    for first, second in zip(ordered, ordered[1:]):
        gap = second - (first + allocation_bytes)
        if gap < 0:
            raise RuntimeError("managed test allocations overlap")
        gaps.append(gap)
    adjacent = sum(gap == 0 for gap in gaps)
    return adjacent, min(gaps) if gaps else 0


def count_last_location(
    library: ctypes.CDLL,
    dataset: ctypes.c_void_p,
    location: int,
) -> int:
    matched = ctypes.c_size_t()
    check_cuda(
        library,
        library.uvm512_dataset_count_last_prefetch_location(
            dataset, location, ctypes.byref(matched)
        ),
        "cudaMemRangeGetAttribute(LastPrefetchLocation)",
    )
    return int(matched.value)


def run_one_sample(
    *,
    args: argparse.Namespace,
    library: ctypes.CDLL,
    dataset: ctypes.c_void_p,
    stream: ctypes.c_void_p,
    events: dict[str, ctypes.c_void_p],
    pointers: tuple[int, ...],
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
    cpu_location = count_last_location(library, dataset, CUDA_CPU_DEVICE_ID)
    if cpu_location != args.allocations:
        raise RuntimeError(
            f"CPU LastPrefetchLocation {cpu_location}/{args.allocations}"
        )
    if pointer_snapshot(library, dataset, args.allocations) != pointers:
        raise RuntimeError("managed virtual addresses changed before sample")

    library.uvm512_prefetch_counter_reset()
    check_cuda(
        library,
        library.uvm512_event_record(events["begin"], stream),
        "record begin event",
    )
    host_begin = time.perf_counter_ns()
    accepted = 0
    if mode == "direct":
        for pointer in pointers:
            accepted += library.uvm512_prefetch_one(
                pointer, args.allocation_bytes, args.device, stream
            )
    elif mode == "batch":
        for pointer_array, size_array in batches:
            accepted += library.uvm512_prefetch_batch(
                pointer_array,
                size_array,
                args.batch_size,
                args.device,
                stream,
            )
    else:
        raise AssertionError(f"unknown mode: {mode}")
    host_end = time.perf_counter_ns()

    check_cuda(
        library,
        library.uvm512_event_record(events["migration_end"], stream),
        "record migration-end event",
    )
    check_cuda(
        library,
        library.uvm512_event_record(events["read_begin"], stream),
        "record checksum-read begin event",
    )
    check_cuda(
        library,
        library.uvm512_dataset_launch_checksum(dataset, stream),
        "launch full checksum",
    )
    check_cuda(
        library,
        library.uvm512_event_record(events["read_end"], stream),
        "record checksum-read end event",
    )
    check_cuda(
        library,
        library.uvm512_event_synchronize(events["read_end"]),
        "synchronize checksum-read end event",
    )

    checksum_value = ctypes.c_uint64()
    check_cuda(
        library,
        library.uvm512_dataset_read_checksum(
            dataset, ctypes.byref(checksum_value)
        ),
        "read checksum",
    )
    one_calls = int(library.uvm512_prefetch_one_outer_calls())
    batch_calls = int(library.uvm512_prefetch_batch_outer_calls())
    runtime_calls = int(library.uvm512_runtime_prefetch_calls())
    expected_one_calls = args.allocations if mode == "direct" else 0
    expected_batch_calls = 4 if mode == "batch" else 0
    if accepted != args.allocations:
        raise RuntimeError(f"{mode} accepted {accepted}/{args.allocations}")
    if (
        one_calls != expected_one_calls
        or batch_calls != expected_batch_calls
        or runtime_calls != args.allocations
    ):
        raise RuntimeError(
            "call-count mismatch: "
            f"one={one_calls}/{expected_one_calls} "
            f"batch={batch_calls}/{expected_batch_calls} "
            f"runtime={runtime_calls}/{args.allocations}"
        )
    gpu_location = count_last_location(library, dataset, args.device)
    if gpu_location != args.allocations:
        raise RuntimeError(
            f"GPU LastPrefetchLocation {gpu_location}/{args.allocations}"
        )
    va_stable = pointer_snapshot(library, dataset, args.allocations) == pointers
    checksum_ok = int(checksum_value.value) == expected_checksum
    if not va_stable or not checksum_ok:
        raise RuntimeError(
            f"postcondition failed: va_stable={va_stable} "
            f"checksum={checksum_value.value}/{expected_checksum}"
        )

    migration_ms = elapsed_ms(library, events["begin"], events["migration_end"])
    checksum_read_ms = elapsed_ms(
        library, events["read_begin"], events["read_end"]
    )
    e2e_ms = elapsed_ms(library, events["begin"], events["read_end"])
    logical_gib = (args.allocations * args.allocation_bytes) / GIB
    return Sample(
        phase=phase,
        sequence_index=sequence_index,
        mode_index=mode_index,
        mode=mode,
        accepted=accepted,
        one_outer_calls=one_calls,
        batch_outer_calls=batch_calls,
        runtime_calls=runtime_calls,
        host_submit_us=(host_end - host_begin) / 1000.0,
        migration_ms=migration_ms,
        checksum_read_ms=checksum_read_ms,
        e2e_ms=e2e_ms,
        migration_gib_s=logical_gib / (migration_ms / 1000.0),
        checksum=int(checksum_value.value),
        cpu_location_before=cpu_location,
        gpu_location_after=gpu_location,
        va_stable=va_stable,
        checksum_ok=checksum_ok,
    )


def print_sample(sample: Sample, logical_bytes: int) -> None:
    print(
        "SAMPLE "
        f"phase={sample.phase} sequence_index={sample.sequence_index} "
        f"mode_index={sample.mode_index} mode={sample.mode} "
        f"logical_bytes={logical_bytes} accepted={sample.accepted} "
        f"one_outer_calls={sample.one_outer_calls} "
        f"batch_outer_calls={sample.batch_outer_calls} "
        f"runtime_prefetch_calls={sample.runtime_calls} "
        f"host_submit_us={sample.host_submit_us:.3f} "
        f"migration_ms={sample.migration_ms:.3f} "
        f"checksum_read_ms={sample.checksum_read_ms:.3f} "
        f"e2e_ms={sample.e2e_ms:.3f} "
        f"migration_gib_s={sample.migration_gib_s:.3f} "
        f"cpu_last_prefetch={sample.cpu_location_before} "
        f"gpu_last_prefetch={sample.gpu_location_after} "
        f"va_stable={int(sample.va_stable)} checksum={sample.checksum} "
        f"checksum_ok={int(sample.checksum_ok)}"
    )


def print_result(mode: str, samples: list[Sample], allocations: int) -> None:
    host_submit = [sample.host_submit_us for sample in samples]
    migration = [sample.migration_ms for sample in samples]
    checksum_read = [sample.checksum_read_ms for sample in samples]
    e2e = [sample.e2e_ms for sample in samples]
    bandwidth = [sample.migration_gib_s for sample in samples]
    outer_calls = allocations if mode == "direct" else 4
    print(
        "RESULT "
        f"mode={mode} samples={len(samples)} accepted_per_sample={allocations} "
        f"outer_calls_per_sample={outer_calls} "
        f"runtime_prefetch_calls_per_sample={allocations} "
        f"host_submit_us_p50={statistics.median(host_submit):.3f} "
        f"host_submit_us_p95={percentile(host_submit, 0.95):.3f} "
        f"migration_ms_p50={statistics.median(migration):.3f} "
        f"migration_ms_p95={percentile(migration, 0.95):.3f} "
        f"checksum_read_ms_p50={statistics.median(checksum_read):.3f} "
        f"e2e_ms_p50={statistics.median(e2e):.3f} "
        f"migration_gib_s_p50={statistics.median(bandwidth):.3f} "
        "va_stable=1 checksum_ok=1 last_prefetch_location_ok=1"
    )


def run(args: argparse.Namespace, library: ctypes.CDLL) -> int:
    check_cuda(library, library.uvm512_set_device(args.device), "cudaSetDevice")
    dataset = library.uvm512_dataset_create(
        args.allocations, args.allocation_bytes, args.guard_bytes
    )
    if not dataset:
        raise RuntimeError("failed to create managed dataset")
    stream = None
    events: dict[str, ctypes.c_void_p] = {}
    try:
        if int(library.uvm512_dataset_count(dataset)) != args.allocations:
            raise RuntimeError("dataset allocation count mismatch")
        if (
            int(library.uvm512_dataset_allocation_bytes(dataset))
            != args.allocation_bytes
        ):
            raise RuntimeError("dataset allocation size mismatch")
        pointers = pointer_snapshot(library, dataset, args.allocations)
        adjacent, min_gap = layout_metrics(pointers, args.allocation_bytes)
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
        for name in ("begin", "migration_end", "read_begin", "read_end"):
            event = library.uvm512_event_create()
            if not event:
                raise RuntimeError(f"failed to create {name} CUDA event")
            events[name] = event

        pointer_array_type = ctypes.c_size_t * args.batch_size
        size_array_type = ctypes.c_size_t * args.batch_size
        batches: list[tuple[ctypes.Array, ctypes.Array]] = []
        for begin in range(0, args.allocations, args.batch_size):
            end = begin + args.batch_size
            batches.append(
                (
                    pointer_array_type(*pointers[begin:end]),
                    size_array_type(
                        *([args.allocation_bytes] * args.batch_size)
                    ),
                )
            )
        if len(batches) != 4:
            raise AssertionError("batch path must make exactly four outer calls")

        print(
            "benchmark_begin "
            f"device={args.device} allocations={args.allocations} "
            f"allocation_bytes={args.allocation_bytes} "
            f"logical_bytes={logical_bytes} batch_size={args.batch_size} "
            f"batches=4 warmup_per_mode={args.warmup_per_mode} "
            f"measured_per_mode={args.measured_per_mode}"
        )
        print(
            "LAYOUT "
            f"test_allocations={args.allocations} "
            f"guard_allocations={args.allocations - 1 if args.guard_bytes else 0} "
            f"guard_bytes={args.guard_bytes} "
            f"adjacent_test_ranges={adjacent} min_gap_bytes={min_gap} "
            "guard_initialized=0 guard_prefetched=0"
        )
        print(
            "SEMANTICS "
            f"direct=Python_{args.allocations}_calls "
            f"batch=Python_4_calls_of_{args.batch_size} "
            "batch_internal=strict_cudaMemPrefetchAsync_loop "
            f"direct_outer_calls={args.allocations} batch_outer_calls=4 "
            f"ranges_per_batch={args.batch_size} "
            f"runtime_calls_both={args.allocations} same_allocations=1 "
            "same_stream=1 same_order=1 reset_cpu_outside_timing=1"
        )

        mode_indices = {"direct": 0, "batch": 0}
        all_samples: list[Sample] = []
        phases = (
            ("warmup", abba_sequence(args.warmup_per_mode)),
            ("measured", abba_sequence(args.measured_per_mode)),
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
                    events=events,
                    pointers=pointers,
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

        measured = [sample for sample in all_samples if sample.phase == "measured"]
        direct = [sample for sample in measured if sample.mode == "direct"]
        batch = [sample for sample in measured if sample.mode == "batch"]
        if len(direct) != args.measured_per_mode or len(batch) != args.measured_per_mode:
            raise RuntimeError("measured ABBA sample count mismatch")
        print_result("direct", direct, args.allocations)
        print_result("batch", batch, args.allocations)

        direct_host = statistics.median(
            sample.host_submit_us for sample in direct
        )
        batch_host = statistics.median(sample.host_submit_us for sample in batch)
        direct_migration = statistics.median(
            sample.migration_ms for sample in direct
        )
        batch_migration = statistics.median(
            sample.migration_ms for sample in batch
        )
        print(
            "COMPARISON "
            f"batch_host_submit_speedup={direct_host / batch_host:.3f} "
            f"batch_migration_speedup={direct_migration / batch_migration:.3f} "
            "runtime_call_reduction=1.000 outer_call_reduction="
            f"{args.allocations / 4:.3f}"
        )
        print(
            "benchmark_result functional_success=1 "
            f"allocations={args.allocations} allocation_bytes={args.allocation_bytes} "
            f"direct_outer_calls={args.allocations} batch_outer_calls=4 "
            f"runtime_calls_both={args.allocations} accepted_both={args.allocations} "
            f"adjacent_test_ranges={adjacent} min_gap_bytes={min_gap} "
            "va_stable=1 checksum_ok=1 last_prefetch_location_ok=1"
        )
        return 0
    finally:
        for event in events.values():
            library.uvm512_event_destroy(event)
        if stream:
            library.uvm512_stream_destroy(stream)
        library.uvm512_dataset_destroy(dataset)


def main() -> int:
    args = parse_args()
    try:
        library = load_library(args.library)
        return run(args, library)
    except Exception as error:  # noqa: BLE001 - CLI must emit one failure record.
        print(f"benchmark_result functional_success=0 error={error}")
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
