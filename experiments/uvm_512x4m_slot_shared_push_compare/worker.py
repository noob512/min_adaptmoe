#!/usr/bin/env python3
"""Run one isolated A/B/C slot-copy sample in a fresh process."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
LEGACY_DIR = SCRIPT_DIR.parent / "uvm_512x4m_batch_compare"
sys.path.insert(0, str(LEGACY_DIR))
import benchmark as legacy  # noqa: E402


DEFAULT_STATS = Path("/proc/driver/nvidia-uvm/latency_stats")
MODES = (
    "driver_4x128_sync",
    "slot_per_block_push",
    "slot_shared_push",
)
GENERIC_COUNTERS = (
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

# Filled by the paired driver implementation. These names are intentionally a
# strict ABI contract: an older module must fail instead of producing a timing
# that merely looks like the requested slot fast path.
SLOT_COUNTERS = (
    "slot_direct_ioctl_calls",
    "slot_direct_per_block_calls",
    "slot_direct_shared_calls",
    "slot_direct_successes",
    "slot_direct_requested_ranges",
    "slot_direct_completed_ranges",
    "slot_direct_requested_bytes",
    "slot_direct_completed_bytes",
    "slot_direct_blocks",
    "slot_direct_copy_pushes",
    "slot_direct_memcopy_methods",
    "slot_direct_tracker_waits",
    "slot_direct_vidmem_allocations",
    "slot_direct_vidmem_evictions",
    "slot_direct_pte_map_ranges",
    "slot_direct_pte_map_bytes",
    "slot_direct_fallbacks",
    "slot_direct_errors",
)
SLOT_PROFILE_STAGES = ("total", "prepare", "copy", "map", "wait")

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
class Residency:
    ranges: int
    gpu_bytes: int
    cpu_bytes: int
    any_bytes: int
    unknown_bytes: int


@dataclass(frozen=True)
class BatchOutcome:
    calls: int = 0
    success: int = 0
    failure: int = 0
    conflict: int = 0
    remap: int = 0
    tlb_invalidate: int = 0
    requested_bytes: int = 0
    accepted_bytes: int = 0
    bad_entries: int = 0
    copy_pushes: int = 0
    completion_waits: int = 0

    def plus(self, other: "BatchOutcome") -> "BatchOutcome":
        return BatchOutcome(
            **{
                name: int(getattr(self, name)) + int(getattr(other, name))
                for name in self.__dataclass_fields__
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--allocations", type=int, required=True)
    parser.add_argument("--allocation-bytes", type=int, required=True)
    parser.add_argument("--guard-bytes", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--slot-layer-id", type=lambda value: int(value, 0), default=0x5124)
    parser.add_argument("--slot-expert-id", type=lambda value: int(value, 0), default=0xC01D)
    parser.add_argument("--sample-id", required=True)
    args = parser.parse_args()
    if (
        args.device < 0
        or args.allocations <= 0
        or args.allocation_bytes <= 0
        or args.guard_bytes < 0
        or args.batch_size <= 0
        or args.batch_size > 128
        or args.allocations % args.batch_size != 0
        or args.allocations // args.batch_size != 4
        or args.warmups < 0
        or args.allocation_bytes % ctypes.sizeof(ctypes.c_uint32) != 0
    ):
        parser.error("require four batches, batch_size<=128, and valid byte sizes")
    return args


def load_library(path: Path) -> ctypes.CDLL:
    library = legacy.load_library(path)
    void_p = ctypes.c_void_p
    size_t = ctypes.c_size_t
    uint32 = ctypes.c_uint32
    uint64 = ctypes.c_uint64
    size_ptr = ctypes.POINTER(size_t)
    u32_ptr = ctypes.POINTER(uint32)

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
        size_ptr,
        size_ptr,
        size_t,
        u32_ptr,
        u32_ptr,
        ctypes.POINTER(uint64),
        ctypes.POINTER(uint64),
        u32_ptr,
    ]
    library.uvm512_driver_prefetch_batch_fast.restype = ctypes.c_int

    library.uvm512_slot_batch_ioctl_command.argtypes = []
    library.uvm512_slot_batch_ioctl_command.restype = uint32
    library.uvm512_slot_batch_max_entries.argtypes = []
    library.uvm512_slot_batch_max_entries.restype = size_t
    library.uvm512_slot_batch_params_size.argtypes = []
    library.uvm512_slot_batch_params_size.restype = size_t
    for name in (
        "uvm512_slot_mode_vidmem_prealloc",
        "uvm512_slot_mode_direct_per_block_push",
        "uvm512_slot_mode_direct_shared_push",
    ):
        function = getattr(library, name)
        function.argtypes = []
        function.restype = uint32
    library.uvm512_slot_remap_batch.argtypes = [
        void_p,
        uint32,
        size_ptr,
        size_ptr,
        u32_ptr,
        size_t,
        uint32,
        uint32,
        u32_ptr,
        u32_ptr,
        u32_ptr,
        u32_ptr,
        u32_ptr,
        ctypes.POINTER(uint64),
        ctypes.POINTER(uint64),
        u32_ptr,
        u32_ptr,
        u32_ptr,
    ]
    library.uvm512_slot_remap_batch.restype = ctypes.c_int
    library.uvm512_driver_query_residency_full.argtypes = [
        void_p,
        size_ptr,
        size_ptr,
        size_t,
        u32_ptr,
        ctypes.POINTER(uint64),
        ctypes.POINTER(uint64),
        ctypes.POINTER(uint64),
        ctypes.POINTER(uint64),
    ]
    library.uvm512_driver_query_residency_full.restype = ctypes.c_int
    return library


def driver_error(library: ctypes.CDLL) -> str:
    value = library.uvm512_driver_last_error()
    return value.decode("utf-8", errors="replace") if value else "unknown"


def parse_latency_stats(text: str) -> dict[str, int]:
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
                raise ValueError(f"profile row before section at line {line_number}")
            name, calls, nanoseconds, _average, pages, bytes_ = profile_match.groups()
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
        raise ValueError(f"unrecognized latency_stats line {line_number}: {line!r}")
    return values


def read_profile(path: Path) -> dict[str, int]:
    try:
        return parse_latency_stats(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RuntimeError(f"cannot read driver profile {path}: {error}") from error


def profile_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    delta: dict[str, int] = {}
    for name in set(before) | set(after):
        value = int(after.get(name, 0)) - int(before.get(name, 0))
        if value < 0:
            raise RuntimeError(f"driver profile moved backwards: {name}={value}")
        delta[name] = value
    return delta


def require_fields(profile: Mapping[str, int], expected: Mapping[str, int]) -> None:
    for name, value in expected.items():
        if name not in profile:
            raise RuntimeError(f"driver profile is missing required field {name}")
        if int(profile[name]) != int(value):
            raise RuntimeError(f"driver profile {name}={profile[name]}/{value}")


def query_residency(
    library: ctypes.CDLL,
    context: ctypes.c_void_p,
    pointers: ctypes.Array,
    sizes: ctypes.Array,
    count: int,
) -> Residency:
    success = ctypes.c_uint32()
    gpu = ctypes.c_uint64()
    cpu = ctypes.c_uint64()
    any_ = ctypes.c_uint64()
    unknown = ctypes.c_uint64()
    result = library.uvm512_driver_query_residency_full(
        context,
        pointers,
        sizes,
        count,
        ctypes.byref(success),
        ctypes.byref(gpu),
        ctypes.byref(cpu),
        ctypes.byref(any_),
        ctypes.byref(unknown),
    )
    if result != 0:
        raise RuntimeError(f"residency query failed: {driver_error(library)}")
    return Residency(
        ranges=int(success.value),
        gpu_bytes=int(gpu.value),
        cpu_bytes=int(cpu.value),
        any_bytes=int(any_.value),
        unknown_bytes=int(unknown.value),
    )


def generic_call(
    library: ctypes.CDLL,
    context: ctypes.c_void_p,
    pointers: ctypes.Array,
    sizes: ctypes.Array,
    count: int,
) -> BatchOutcome:
    success = ctypes.c_uint32()
    failure = ctypes.c_uint32()
    requested = ctypes.c_uint64()
    completed = ctypes.c_uint64()
    rm_status = ctypes.c_uint32()
    result = library.uvm512_driver_prefetch_batch_fast(
        context,
        pointers,
        sizes,
        count,
        ctypes.byref(success),
        ctypes.byref(failure),
        ctypes.byref(requested),
        ctypes.byref(completed),
        ctypes.byref(rm_status),
    )
    if result != 0 or rm_status.value != 0:
        raise RuntimeError(
            f"ioctl90 failed wrapper={result} rmStatus={rm_status.value}: "
            f"{driver_error(library)}"
        )
    return BatchOutcome(
        calls=1,
        success=int(success.value),
        failure=int(failure.value),
        requested_bytes=int(requested.value),
        accepted_bytes=int(completed.value),
        completion_waits=1,
    )


def slot_call(
    library: ctypes.CDLL,
    context: ctypes.c_void_p,
    mode: int,
    pointers: ctypes.Array,
    sizes: ctypes.Array,
    slot_ids: ctypes.Array,
    count: int,
    layer_id: int,
    expert_id: int,
) -> BatchOutcome:
    outputs32 = [ctypes.c_uint32() for _ in range(8)]
    requested = ctypes.c_uint64()
    accepted = ctypes.c_uint64()
    (
        success,
        failure,
        conflict,
        remap,
        tlb,
        bad_entries,
        copy_pushes,
        rm_status,
    ) = outputs32
    result = library.uvm512_slot_remap_batch(
        context,
        mode,
        pointers,
        sizes,
        slot_ids,
        count,
        layer_id,
        expert_id,
        ctypes.byref(success),
        ctypes.byref(failure),
        ctypes.byref(conflict),
        ctypes.byref(remap),
        ctypes.byref(tlb),
        ctypes.byref(requested),
        ctypes.byref(accepted),
        ctypes.byref(bad_entries),
        ctypes.byref(copy_pushes),
        ctypes.byref(rm_status),
    )
    if result != 0 or rm_status.value != 0:
        raise RuntimeError(
            f"ioctl91 mode={mode} failed wrapper={result} "
            f"rmStatus={rm_status.value}: {driver_error(library)}"
        )
    return BatchOutcome(
        calls=1,
        success=int(success.value),
        failure=int(failure.value),
        conflict=int(conflict.value),
        remap=int(remap.value),
        tlb_invalidate=int(tlb.value),
        requested_bytes=int(requested.value),
        accepted_bytes=int(accepted.value),
        bad_entries=int(bad_entries.value),
        copy_pushes=int(copy_pushes.value),
        completion_waits=1,
    )


def validate_outcome(outcome: BatchOutcome, allocations: int, logical_bytes: int) -> None:
    if (
        outcome.calls != 4
        or outcome.success != allocations
        or outcome.failure != 0
        or outcome.conflict != 0
        or outcome.requested_bytes != logical_bytes
        or outcome.accepted_bytes != logical_bytes
        or outcome.bad_entries != 0
        or outcome.completion_waits != 4
    ):
        raise RuntimeError(f"four-batch result contract failed: {outcome}")


def execute_transfer(
    *,
    args: argparse.Namespace,
    library: ctypes.CDLL,
    context: ctypes.c_void_p,
    batches: list[tuple[ctypes.Array, ctypes.Array, ctypes.Array]],
    slot_modes: Mapping[str, int],
) -> BatchOutcome:
    aggregate = BatchOutcome()
    for pointers, sizes, slots in batches:
        if args.mode == "driver_4x128_sync":
            outcome = generic_call(library, context, pointers, sizes, args.batch_size)
        else:
            outcome = slot_call(
                library,
                context,
                slot_modes[args.mode],
                pointers,
                sizes,
                slots,
                args.batch_size,
                args.slot_layer_id,
                args.slot_expert_id,
            )
        aggregate = aggregate.plus(outcome)
    validate_outcome(
        aggregate,
        args.allocations,
        args.allocations * args.allocation_bytes,
    )
    return aggregate


def reset_to_cpu(
    *,
    args: argparse.Namespace,
    library: ctypes.CDLL,
    dataset: ctypes.c_void_p,
    stream: ctypes.c_void_p,
    context: ctypes.c_void_p,
    pointers_tuple: tuple[int, ...],
    pointer_array: ctypes.Array,
    size_array: ctypes.Array,
) -> dict[str, object]:
    accepted = int(library.uvm512_dataset_prefetch_cpu_and_sync(dataset, stream))
    if accepted != args.allocations:
        raise RuntimeError(f"CPU reset accepted {accepted}/{args.allocations}")
    last_location = legacy.count_last_location(
        library, dataset, legacy.CUDA_CPU_DEVICE_ID
    )
    if last_location != args.allocations:
        raise RuntimeError(f"CPU LastPrefetchLocation {last_location}/{args.allocations}")
    if legacy.pointer_snapshot(library, dataset, args.allocations) != pointers_tuple:
        raise RuntimeError("managed VA changed during CPU reset")
    residency = query_residency(
        library, context, pointer_array, size_array, args.allocations
    )
    logical_bytes = args.allocations * args.allocation_bytes
    expected = Residency(args.allocations, 0, logical_bytes, logical_bytes, 0)
    if residency != expected:
        raise RuntimeError(f"CPU source residency={residency}, expected={expected}")
    return {
        "accepted": accepted,
        "last_prefetch_cpu_ranges": last_location,
        "source_residency": asdict(residency),
        "va_stable": True,
    }


def checksum(
    library: ctypes.CDLL,
    dataset: ctypes.c_void_p,
    stream: ctypes.c_void_p,
    expected: int,
) -> tuple[int, float]:
    begin = time.perf_counter_ns()
    legacy.check_cuda(
        library,
        library.uvm512_dataset_launch_checksum(dataset, stream),
        "launch full GPU checksum",
    )
    legacy.check_cuda(
        library,
        library.uvm512_stream_synchronize(stream),
        "synchronize full GPU checksum",
    )
    end = time.perf_counter_ns()
    value = ctypes.c_uint64()
    legacy.check_cuda(
        library,
        library.uvm512_dataset_read_checksum(dataset, ctypes.byref(value)),
        "read full GPU checksum",
    )
    if int(value.value) != expected:
        raise RuntimeError(f"checksum={value.value}/{expected}")
    return int(value.value), (end - begin) / 1_000_000.0


def expected_profile(
    mode: str,
    allocations: int,
    logical_bytes: int,
    observed: Mapping[str, int],
    outcome: BatchOutcome,
) -> dict[str, int]:
    generic_values = {
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
    expected: dict[str, int] = {}
    for name in GENERIC_COUNTERS:
        expected[name] = generic_values[name] if mode == "driver_4x128_sync" else 0

    if mode == "driver_4x128_sync":
        for name in SLOT_COUNTERS:
            expected[name] = 0
        for stage in SLOT_PROFILE_STAGES:
            for field in ("calls", "ns", "pages", "bytes"):
                expected[f"slot_direct_profile.{stage}.{field}"] = 0
        if outcome.remap != 0 or outcome.tlb_invalidate != 0 or outcome.copy_pushes != 0:
            raise RuntimeError(f"generic ioctl returned slot fields: {outcome}")
        return expected

    shared = mode == "slot_shared_push"
    blocks = int(observed.get("slot_direct_blocks", -1))
    if blocks <= 0:
        raise RuntimeError(f"invalid slot_direct_blocks={blocks}")
    methods = int(observed.get("slot_direct_memcopy_methods", -1))
    copy_pushes = int(observed.get("slot_direct_copy_pushes", -1))
    if methods < blocks:
        raise RuntimeError(f"invalid slot methods={methods}, blocks={blocks}")
    if shared:
        if copy_pushes < 4 or copy_pushes > blocks:
            raise RuntimeError(
                "shared-push capacity contract failed: "
                f"pushes={copy_pushes} blocks={blocks}"
            )
    elif copy_pushes != blocks:
        raise RuntimeError(
            f"per-block push contract failed: pushes={copy_pushes} blocks={blocks}"
        )
    if (
        outcome.remap != allocations
        or outcome.tlb_invalidate != blocks
        or outcome.copy_pushes != copy_pushes
    ):
        raise RuntimeError(
            "slot ioctl output/profile mismatch: "
            f"outcome={outcome} blocks={blocks} pushes={copy_pushes}"
        )
    slot_values = {
        "slot_direct_ioctl_calls": 4,
        "slot_direct_per_block_calls": 0 if shared else 4,
        "slot_direct_shared_calls": 4 if shared else 0,
        "slot_direct_successes": 4,
        "slot_direct_requested_ranges": allocations,
        "slot_direct_completed_ranges": allocations,
        "slot_direct_requested_bytes": logical_bytes,
        "slot_direct_completed_bytes": logical_bytes,
        "slot_direct_blocks": blocks,
        "slot_direct_memcopy_methods": methods,
        # Shared mode returns its actual capacity-driven flush count in
        # params.reserved; never assume one push per ioctl.
        "slot_direct_copy_pushes": copy_pushes,
        "slot_direct_tracker_waits": 4,
        "slot_direct_vidmem_allocations": 0,
        "slot_direct_vidmem_evictions": 0,
        "slot_direct_pte_map_ranges": allocations,
        "slot_direct_pte_map_bytes": logical_bytes,
        "slot_direct_fallbacks": 0,
        "slot_direct_errors": 0,
    }
    expected.update(slot_values)
    for stage in SLOT_PROFILE_STAGES:
        expected[f"slot_direct_profile.{stage}.calls"] = 4
        expected[f"slot_direct_profile.{stage}.pages"] = logical_bytes // 4096
        expected[f"slot_direct_profile.{stage}.bytes"] = logical_bytes
    return expected


def run(args: argparse.Namespace, library: ctypes.CDLL) -> dict[str, object]:
    logical_bytes = args.allocations * args.allocation_bytes
    legacy.check_cuda(library, library.uvm512_set_device(args.device), "cudaSetDevice")
    dataset = library.uvm512_dataset_create(
        args.allocations, args.allocation_bytes, args.guard_bytes
    )
    if not dataset:
        raise RuntimeError("failed to create managed dataset")
    stream = None
    context = None
    try:
        pointers = legacy.pointer_snapshot(library, dataset, args.allocations)
        adjacent, min_gap = legacy.layout_metrics(pointers, args.allocation_bytes)
        if args.guard_bytes > 0 and adjacent != 0:
            raise RuntimeError(f"test allocations have {adjacent} adjacent ranges")
        expected_checksum = int(library.uvm512_dataset_expected_checksum(dataset))
        stream = library.uvm512_stream_create()
        if not stream:
            raise RuntimeError("failed to create CUDA stream")

        pointer_array = (ctypes.c_size_t * args.allocations)(*pointers)
        size_array = (ctypes.c_size_t * args.allocations)(
            *([args.allocation_bytes] * args.allocations)
        )
        slot_ids_tuple = tuple(range(args.allocations))
        slot_array = (ctypes.c_uint32 * args.allocations)(*slot_ids_tuple)
        batches: list[tuple[ctypes.Array, ctypes.Array, ctypes.Array]] = []
        pointer_type = ctypes.c_size_t * args.batch_size
        size_type = ctypes.c_size_t * args.batch_size
        slot_type = ctypes.c_uint32 * args.batch_size
        for begin in range(0, args.allocations, args.batch_size):
            end = begin + args.batch_size
            batches.append(
                (
                    pointer_type(*pointers[begin:end]),
                    size_type(*([args.allocation_bytes] * args.batch_size)),
                    slot_type(*slot_ids_tuple[begin:end]),
                )
            )
        if len(batches) != 4:
            raise AssertionError("all modes require exactly four batches")

        context = library.uvm512_driver_batch_context_create(
            args.device, pointers[0], args.allocation_bytes
        )
        if not context:
            raise RuntimeError(f"driver context unavailable: {driver_error(library)}")
        abi = {
            "generic_ioctl": int(library.uvm512_driver_batch_ioctl_command()),
            "generic_max_ranges": int(library.uvm512_driver_batch_max_ranges()),
            "generic_params_size": int(library.uvm512_driver_batch_params_size()),
            "slot_ioctl": int(library.uvm512_slot_batch_ioctl_command()),
            "slot_max_entries": int(library.uvm512_slot_batch_max_entries()),
            "slot_params_size": int(library.uvm512_slot_batch_params_size()),
            "slot_mode_prealloc": int(library.uvm512_slot_mode_vidmem_prealloc()),
            "slot_mode_per_block": int(
                library.uvm512_slot_mode_direct_per_block_push()
            ),
            "slot_mode_shared": int(library.uvm512_slot_mode_direct_shared_push()),
        }
        expected_abi = {
            "generic_ioctl": 90,
            "generic_max_ranges": 128,
            "slot_ioctl": 91,
            "slot_max_entries": 128,
            "slot_params_size": 6224,
            "slot_mode_prealloc": 8,
            "slot_mode_per_block": 10,
            "slot_mode_shared": 11,
        }
        for name, value in expected_abi.items():
            if abi[name] != value:
                raise RuntimeError(f"ABI {name}={abi[name]}/{value}")

        slot_modes = {
            "slot_per_block_push": abi["slot_mode_per_block"],
            "slot_shared_push": abi["slot_mode_shared"],
        }

        # The ioctl-90 context probe migrates the first allocation. Restore the
        # complete source set before any slot setup or warmup.
        reset_to_cpu(
            args=args,
            library=library,
            dataset=dataset,
            stream=stream,
            context=context,
            pointers_tuple=pointers,
            pointer_array=pointer_array,
            size_array=size_array,
        )

        prealloc = BatchOutcome()
        if args.mode != "driver_4x128_sync":
            for batch_pointers, batch_sizes, batch_slots in batches:
                prealloc = prealloc.plus(
                    slot_call(
                        library,
                        context,
                        abi["slot_mode_prealloc"],
                        batch_pointers,
                        batch_sizes,
                        batch_slots,
                        args.batch_size,
                        args.slot_layer_id,
                        args.slot_expert_id,
                    )
                )
            validate_outcome(prealloc, args.allocations, logical_bytes)
            if (
                prealloc.remap != 0
                or prealloc.tlb_invalidate != 0
                or prealloc.copy_pushes != 0
            ):
                raise RuntimeError(f"preallocation unexpectedly remapped: {prealloc}")

        for _ in range(args.warmups):
            reset_to_cpu(
                args=args,
                library=library,
                dataset=dataset,
                stream=stream,
                context=context,
                pointers_tuple=pointers,
                pointer_array=pointer_array,
                size_array=size_array,
            )
            execute_transfer(
                args=args,
                library=library,
                context=context,
                batches=batches,
                slot_modes=slot_modes,
            )
            checksum(library, dataset, stream, expected_checksum)

        reset = reset_to_cpu(
            args=args,
            library=library,
            dataset=dataset,
            stream=stream,
            context=context,
            pointers_tuple=pointers,
            pointer_array=pointer_array,
            size_array=size_array,
        )
        profile_before_first = read_profile(args.stats)
        profile_before = read_profile(args.stats)
        boundary_fields = (
            "migrate_profile.api_total.calls",
            "prefetch_gpu_batch_fast_ioctl_calls",
            "slot_direct_ioctl_calls",
        )
        for name in boundary_fields:
            if name not in profile_before:
                raise RuntimeError(f"profile boundary field missing: {name}")

        wall_begin = time.perf_counter_ns()
        outcome = execute_transfer(
            args=args,
            library=library,
            context=context,
            batches=batches,
            slot_modes=slot_modes,
        )
        wall_end = time.perf_counter_ns()
        profile_after = read_profile(args.stats)
        profile = profile_delta(profile_before, profile_after)
        required_profile = expected_profile(
            args.mode, args.allocations, logical_bytes, profile, outcome
        )
        require_fields(profile, required_profile)

        if args.mode == "driver_4x128_sync":
            copy_ms: float | None = None
            slot_map_ms: float | None = None
            driver_profile_total_ms: float | None = None
            component_source: str | None = None
            noncomparable_generic_diagnostics = {
                "migrate_copy_between_total_cpu_ms": profile[
                    "migrate_profile.copy_between_total.ns"
                ]
                / 1_000_000.0,
                "migrate_block_map_cpu_ms": profile[
                    "migrate_profile.block_map.ns"
                ]
                / 1_000_000.0,
                "migrate_api_tracker_wait_cpu_ms": profile[
                    "migrate_profile.api_tracker_wait.ns"
                ]
                / 1_000_000.0,
                "note": "nested CPU stages; not the slot copy/map timing contract",
            }
        else:
            copy_ms = profile["slot_direct_profile.copy.ns"] / 1_000_000.0
            slot_map_ms = profile["slot_direct_profile.map.ns"] / 1_000_000.0
            driver_profile_total_ms = (
                profile["slot_direct_profile.total.ns"] / 1_000_000.0
            )
            component_source = "slot_direct_profile_cumulative_nested_stages"
            noncomparable_generic_diagnostics = None
            if copy_ms <= 0 or slot_map_ms <= 0 or driver_profile_total_ms <= 0:
                raise RuntimeError(
                    "slot profile returned a non-positive timing component: "
                    f"copy={copy_ms} map={slot_map_ms} "
                    f"total={driver_profile_total_ms}"
                )

        post_residency = query_residency(
            library, context, pointer_array, size_array, args.allocations
        )
        if args.mode == "driver_4x128_sync":
            expected_post = Residency(
                args.allocations, logical_bytes, 0, logical_bytes, 0
            )
        else:
            # Slot PTEs are deliberately not normal UVM GPU residency. The
            # backing managed pages remain on CPU while GPU reads the slot.
            expected_post = Residency(
                args.allocations, 0, logical_bytes, logical_bytes, 0
            )
        if post_residency != expected_post:
            raise RuntimeError(
                f"post residency={post_residency}, expected={expected_post}"
            )

        checksum_value, checksum_ms = checksum(
            library, dataset, stream, expected_checksum
        )
        post_checksum_residency = query_residency(
            library, context, pointer_array, size_array, args.allocations
        )
        if post_checksum_residency != expected_post:
            raise RuntimeError(
                "GPU checksum changed UVM source residency: "
                f"{post_checksum_residency}/{expected_post}"
            )
        cpu_last_after = legacy.count_last_location(
            library, dataset, legacy.CUDA_CPU_DEVICE_ID
        )
        if args.mode != "driver_4x128_sync" and cpu_last_after != args.allocations:
            raise RuntimeError(
                f"slot mode changed CPU LastPrefetchLocation "
                f"{cpu_last_after}/{args.allocations}"
            )
        va_stable = (
            legacy.pointer_snapshot(library, dataset, args.allocations) == pointers
        )
        if not va_stable:
            raise RuntimeError("managed VA changed after transfer")

        wall_ns = wall_end - wall_begin
        if wall_ns <= 0:
            raise RuntimeError("non-positive measurement")
        gib_s = (logical_bytes / legacy.GIB) / (wall_ns / 1_000_000_000.0)
        if not math.isfinite(gib_s) or gib_s <= 0:
            raise RuntimeError(f"invalid bandwidth {gib_s}")
        return {
            "schema_version": 1,
            "status": "ok",
            "sample_id": args.sample_id,
            "pid": os.getpid(),
            "mode": args.mode,
            "config": {
                "device": args.device,
                "allocations": args.allocations,
                "allocation_bytes": args.allocation_bytes,
                "guard_bytes": args.guard_bytes,
                "batch_size": args.batch_size,
                "batches": len(batches),
                "logical_bytes": logical_bytes,
                "warmups": args.warmups,
                "slot_layer_id": args.slot_layer_id,
                "slot_expert_id": args.slot_expert_id,
            },
            "abi": abi,
            "layout": {
                "adjacent_test_ranges": adjacent,
                "min_gap_bytes": min_gap,
                "va_snapshot": [f"0x{pointer:x}" for pointer in pointers],
            },
            "setup": {
                "slot_vidmem_preallocated_before_timing": args.mode
                != "driver_4x128_sync",
                "prealloc": asdict(prealloc),
            },
            "reset": reset,
            "measurement": {
                "cpu_reset_immediately_before_profile_and_timing": True,
                "host_monotonic_to_synchronous_ioctl_completion": True,
                "wall_ns": wall_ns,
                "wall_ms": wall_ns / 1_000_000.0,
                "copy_ms": copy_ms,
                "slot_map_ms": slot_map_ms,
                "driver_profile_total_ms": driver_profile_total_ms,
                "component_source": component_source,
                "components_are_nested_not_additive": args.mode
                != "driver_4x128_sync",
                "noncomparable_generic_diagnostics": noncomparable_generic_diagnostics,
                "gib_s": gib_s,
                "outcome": asdict(outcome),
            },
            "validation": {
                "post_residency": asdict(post_residency),
                "post_checksum_residency": asdict(post_checksum_residency),
                "slot_mapping_is_not_uvm_gpu_residency": args.mode
                != "driver_4x128_sync",
                "cpu_last_prefetch_after": cpu_last_after,
                "checksum": checksum_value,
                "expected_checksum": expected_checksum,
                "full_gpu_checksum": True,
                "checksum_ms": checksum_ms,
                "va_stable": va_stable,
            },
            "profile": {
                "snapshot_after_cpu_reset": True,
                "first_boundary": {
                    name: profile_before_first[name] for name in boundary_fields
                },
                "timing_boundary": {
                    name: profile_before[name] for name in boundary_fields
                },
                "required": required_profile,
                "delta": profile,
            },
        }
    finally:
        if context:
            library.uvm512_driver_batch_context_destroy(context)
        if stream:
            library.uvm512_stream_destroy(stream)
        library.uvm512_dataset_destroy(dataset)


def main() -> int:
    args = parse_args()
    try:
        result = run(args, load_library(args.library))
        print("SAMPLE_JSON " + json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except Exception as error:  # noqa: BLE001 - preserve a machine-readable failure.
        failure = {
            "schema_version": 1,
            "status": "error",
            "sample_id": args.sample_id,
            "pid": os.getpid(),
            "mode": args.mode,
            "error": str(error),
        }
        print("SAMPLE_JSON " + json.dumps(failure, sort_keys=True, separators=(",", ":")))
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
