#!/usr/bin/env python3
"""Run one isolated normal-64K/strict-overwrite-64K sample."""

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
SCHEMA_VERSION = 2
MIB = 1024 * 1024
VA_BLOCK_BYTES = 2 * MIB
REGION_BYTES = 4 * MIB
MODES = (
    "normal_64k_e2e",
    "overwrite_64k_e2e",
    "normal_64k_stage_sync",
    "overwrite_64k_stage_sync",
)
MODE_EXPORTS = {
    "normal_64k_e2e": "uvm512_slot_mode_direct_64k",
    "overwrite_64k_e2e": "uvm512_slot_mode_direct_overwrite_64k",
    "normal_64k_stage_sync": "uvm512_slot_mode_direct_64k_stage_sync",
    "overwrite_64k_stage_sync": (
        "uvm512_slot_mode_direct_overwrite_64k_stage_sync"
    ),
}
OVERWRITE_MODES = frozenset(("overwrite_64k_e2e", "overwrite_64k_stage_sync"))
STAGE_SYNC_MODES = frozenset(("normal_64k_stage_sync", "overwrite_64k_stage_sync"))
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
    "slot_direct_pte_4k_blocks",
    "slot_direct_pte_64k_blocks",
    "slot_direct_pte_2m_blocks",
    "slot_direct_pte_write_entries",
    "slot_direct_skipped_unmap_blocks",
    "slot_direct_stage_sync_calls",
    "slot_direct_fallbacks",
    "slot_direct_errors",
)
LEGACY_SLOT_PROFILE_STAGES = (
    "total",
    "prepare",
    "unmap",
    "copy",
    "map",
    "wait",
    "prepare_complete",
    "copy_complete",
    "map_complete",
)
COMPLETE_PROFILE_STAGES = (
    "prepare_complete",
    "copy_complete",
    "map_complete",
)
PARENT_PROFILE_STAGES = (
    "pre_helper_total",
    "helper_total",
    "post_helper_total",
)
PRE_HELPER_PROFILE_STAGES = (
    "gate",
    "state_alloc",
    "gpu_retain",
    "outer_lock_acquire",
    "batch_bookkeeping",
    "slot_validate",
    "flatten_sort",
    "slot_tracker_init",
)
HELPER_RESIDUAL_PROFILE_STAGES = (
    "helper_context",
    "block_lock_acquire",
    "locked_revalidate",
    "copy_to_map",
    "block_unlock",
    "helper_context_free",
)
POST_HELPER_PROFILE_STAGES = (
    "result_check",
    "pte_metadata",
    "status_resolve",
    "slot_state_commit",
    "telemetry_cleanup",
    "outer_lock_release",
    "resource_teardown",
)
RESIDUAL_PROFILE_STAGES = (
    PRE_HELPER_PROFILE_STAGES
    + HELPER_RESIDUAL_PROFILE_STAGES
    + POST_HELPER_PROFILE_STAGES
)
SLOT_PROFILE_STAGES = (
    LEGACY_SLOT_PROFILE_STAGES + PARENT_PROFILE_STAGES + RESIDUAL_PROFILE_STAGES
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
    parser.add_argument(
        "--slot-layer-id", type=lambda value: int(value, 0), default=0x5124
    )
    parser.add_argument(
        "--slot-expert-id", type=lambda value: int(value, 0), default=0xC01D
    )
    parser.add_argument("--sample-id", required=True)
    args = parser.parse_args()
    geometry = (args.allocations, args.allocation_bytes, args.guard_bytes, args.batch_size)
    if geometry not in (
        (4, REGION_BYTES, REGION_BYTES, 1),
        (512, REGION_BYTES, REGION_BYTES, 128),
    ):
        parser.error(
            "only smoke=4x4MiB/4x1 or full=512x4MiB/4x128 with 4MiB guards "
            "is supported"
        )
    if args.device < 0 or args.warmups < 0:
        parser.error("device and warmup count are invalid")
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

    for name in (
        "uvm512_phase_f_batch_executor_ioctl_command",
        "uvm512_phase_f_action_prepare_remote_read",
        "uvm512_phase_f_action_force_remote_read_64k",
    ):
        function = getattr(library, name)
        function.argtypes = []
        function.restype = uint32
    library.uvm512_phase_f_batch_executor_max_ranges.argtypes = []
    library.uvm512_phase_f_batch_executor_max_ranges.restype = size_t
    library.uvm512_phase_f_batch_executor_params_size.argtypes = []
    library.uvm512_phase_f_batch_executor_params_size.restype = size_t

    library.uvm512_slot_batch_ioctl_command.argtypes = []
    library.uvm512_slot_batch_ioctl_command.restype = uint32
    library.uvm512_slot_batch_max_entries.argtypes = []
    library.uvm512_slot_batch_max_entries.restype = size_t
    library.uvm512_slot_batch_params_size.argtypes = []
    library.uvm512_slot_batch_params_size.restype = size_t
    for name in ("uvm512_slot_mode_vidmem_prealloc", *MODE_EXPORTS.values()):
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
    library.uvm512_driver_prepare_remote_read_full.argtypes = [
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
    library.uvm512_driver_prepare_remote_read_full.restype = ctypes.c_int
    library.uvm512_driver_force_remote_read_64k_full.argtypes = [
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
    library.uvm512_driver_force_remote_read_64k_full.restype = ctypes.c_int
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
    ):
        raise RuntimeError(f"four-batch result contract failed: {outcome}")


def execute_transfer(
    *,
    args: argparse.Namespace,
    library: ctypes.CDLL,
    context: ctypes.c_void_p,
    batches: list[tuple[ctypes.Array, ctypes.Array, ctypes.Array]],
    mode: int,
) -> BatchOutcome:
    aggregate = BatchOutcome()
    for pointers, sizes, slots in batches:
        aggregate = aggregate.plus(
            slot_call(
                library,
                context,
                mode,
                pointers,
                sizes,
                slots,
                args.batch_size,
                args.slot_layer_id,
                args.slot_expert_id,
            )
        )
    logical_bytes = args.allocations * args.allocation_bytes
    validate_outcome(aggregate, args.allocations, logical_bytes)
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


def prepare_remote_read(
    *,
    args: argparse.Namespace,
    library: ctypes.CDLL,
    context: ctypes.c_void_p,
    pointer_array: ctypes.Array,
    size_array: ctypes.Array,
) -> dict[str, int]:
    success = ctypes.c_uint32()
    newly_mapped = ctypes.c_uint64()
    cpu_candidate = ctypes.c_uint64()
    covered = ctypes.c_uint64()
    unknown = ctypes.c_uint64()
    result = library.uvm512_driver_prepare_remote_read_full(
        context,
        pointer_array,
        size_array,
        args.allocations,
        ctypes.byref(success),
        ctypes.byref(newly_mapped),
        ctypes.byref(cpu_candidate),
        ctypes.byref(covered),
        ctypes.byref(unknown),
    )
    logical_bytes = args.allocations * args.allocation_bytes
    if result != 0:
        raise RuntimeError(f"remote-read preparation failed: {driver_error(library)}")
    if (
        int(success.value) != args.allocations
        or int(cpu_candidate.value) != logical_bytes
        or int(covered.value) != logical_bytes
        or int(unknown.value) != 0
    ):
        raise RuntimeError(
            "remote-read preparation contract failed: "
            f"success={success.value}/{args.allocations} "
            f"newly_mapped={newly_mapped.value} "
            f"cpu_candidate={cpu_candidate.value}/{logical_bytes} "
            f"covered={covered.value}/{logical_bytes} unknown={unknown.value}"
        )
    return {
        "action": 5,
        "ioctl_calls": math.ceil(args.allocations / 64),
        "success_ranges": int(success.value),
        "newly_mapped_bytes": int(newly_mapped.value),
        "cpu_candidate_bytes": int(cpu_candidate.value),
        "covered_bytes": int(covered.value),
        "unknown_bytes": int(unknown.value),
    }


def force_remote_read_64k(
    *,
    args: argparse.Namespace,
    library: ctypes.CDLL,
    context: ctypes.c_void_p,
    pointer_array: ctypes.Array,
    size_array: ctypes.Array,
) -> dict[str, int]:
    success = ctypes.c_uint32()
    split = ctypes.c_uint64()
    already = ctypes.c_uint64()
    validated = ctypes.c_uint64()
    unknown = ctypes.c_uint64()
    result = library.uvm512_driver_force_remote_read_64k_full(
        context,
        pointer_array,
        size_array,
        args.allocations,
        ctypes.byref(success),
        ctypes.byref(split),
        ctypes.byref(already),
        ctypes.byref(validated),
        ctypes.byref(unknown),
    )
    logical_bytes = args.allocations * args.allocation_bytes
    if result != 0:
        raise RuntimeError(f"force-remote-64K failed: {driver_error(library)}")
    if (
        int(success.value) != args.allocations
        or int(split.value) + int(already.value) != logical_bytes
        or int(validated.value) != logical_bytes
        or int(unknown.value) != 0
    ):
        raise RuntimeError(
            "force-remote-64K contract failed: "
            f"success={success.value}/{args.allocations} "
            f"split={split.value} already={already.value} "
            f"validated={validated.value}/{logical_bytes} unknown={unknown.value}"
        )
    return {
        "action": 20,
        "ioctl_calls": math.ceil(args.allocations / 64),
        "success_ranges": int(success.value),
        "split_2m_to_64k_bytes": int(split.value),
        "already_64k_bytes": int(already.value),
        "validated_64k_bytes": int(validated.value),
        "unknown_bytes": int(unknown.value),
    }


def establish_cpu_remote_64k(
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
    reset = reset_to_cpu(
        args=args,
        library=library,
        dataset=dataset,
        stream=stream,
        context=context,
        pointers_tuple=pointers_tuple,
        pointer_array=pointer_array,
        size_array=size_array,
    )
    remote_read = prepare_remote_read(
        args=args,
        library=library,
        context=context,
        pointer_array=pointer_array,
        size_array=size_array,
    )
    force_64k = force_remote_read_64k(
        args=args,
        library=library,
        context=context,
        pointer_array=pointer_array,
        size_array=size_array,
    )
    residency = query_residency(
        library, context, pointer_array, size_array, args.allocations
    )
    logical_bytes = args.allocations * args.allocation_bytes
    expected = Residency(args.allocations, 0, logical_bytes, logical_bytes, 0)
    if residency != expected:
        raise RuntimeError(
            f"force-remote-64K changed ordinary residency: {residency}/{expected}"
        )
    if legacy.pointer_snapshot(library, dataset, args.allocations) != pointers_tuple:
        raise RuntimeError("managed VA changed while forcing remote 64K mappings")
    return {
        "order": ["cpu_reset", "prepare_remote_read_action_5", "force_64k_action_20"],
        "cpu_reset": reset,
        "prepare_remote_read": remote_read,
        "force_remote_read_64k": force_64k,
        "post_action_residency": asdict(residency),
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
    blocks = logical_bytes // VA_BLOCK_BYTES
    pages = logical_bytes // 4096
    if logical_bytes % VA_BLOCK_BYTES != 0:
        raise RuntimeError("logical bytes are not an integral number of VA blocks")

    expected: dict[str, int] = {name: 0 for name in GENERIC_COUNTERS}
    expected["migrate_profile.api_total.calls"] = 0

    observed_grain = tuple(
        int(observed.get(name, -1))
        for name in (
            "slot_direct_pte_4k_blocks",
            "slot_direct_pte_64k_blocks",
            "slot_direct_pte_2m_blocks",
        )
    )
    if observed_grain != (0, blocks, 0):
        raise RuntimeError(
            f"timed path did not use strict 64K PTEs: {observed_grain}/(0,{blocks},0)"
        )
    pte_entries = blocks * (VA_BLOCK_BYTES // (64 * 1024))
    stage_sync = mode in STAGE_SYNC_MODES
    overwrite = mode in OVERWRITE_MODES

    slot_values = {
        "slot_direct_ioctl_calls": 4,
        "slot_direct_per_block_calls": 4,
        "slot_direct_shared_calls": 0,
        "slot_direct_successes": 4,
        "slot_direct_requested_ranges": allocations,
        "slot_direct_completed_ranges": allocations,
        "slot_direct_requested_bytes": logical_bytes,
        "slot_direct_completed_bytes": logical_bytes,
        "slot_direct_blocks": blocks,
        "slot_direct_copy_pushes": blocks,
        "slot_direct_memcopy_methods": blocks,
        "slot_direct_tracker_waits": 16 if stage_sync else 4,
        "slot_direct_vidmem_allocations": 0,
        "slot_direct_vidmem_evictions": 0,
        "slot_direct_pte_map_ranges": allocations,
        "slot_direct_pte_map_bytes": logical_bytes,
        "slot_direct_pte_4k_blocks": 0,
        "slot_direct_pte_64k_blocks": blocks,
        "slot_direct_pte_2m_blocks": 0,
        "slot_direct_pte_write_entries": pte_entries,
        "slot_direct_skipped_unmap_blocks": (
            blocks if overwrite else 0
        ),
        "slot_direct_stage_sync_calls": 4 if stage_sync else 0,
        "slot_direct_fallbacks": 0,
        "slot_direct_errors": 0,
    }
    expected.update(slot_values)

    if (
        outcome.remap != allocations
        or outcome.tlb_invalidate != blocks
        or outcome.copy_pushes != blocks
    ):
        raise RuntimeError(
            "slot ioctl output/profile contract failed: "
            f"outcome={outcome} blocks={blocks}"
        )

    for stage in SLOT_PROFILE_STAGES:
        completed_stage = stage in COMPLETE_PROFILE_STAGES
        enabled = not completed_stage or stage_sync
        expected[f"slot_direct_profile.{stage}.calls"] = 4 if enabled else 0
        expected[f"slot_direct_profile.{stage}.pages"] = pages if enabled else 0
        expected[f"slot_direct_profile.{stage}.bytes"] = logical_bytes if enabled else 0
        if not enabled:
            expected[f"slot_direct_profile.{stage}.ns"] = 0
        elif stage == "unmap" and overwrite:
            expected[f"slot_direct_profile.{stage}.ns"] = 0
    return expected


def run(args: argparse.Namespace, library: ctypes.CDLL) -> dict[str, object]:
    logical_bytes = args.allocations * args.allocation_bytes
    expected_blocks = logical_bytes // VA_BLOCK_BYTES
    legacy.check_cuda(library, library.uvm512_set_device(args.device), "cudaSetDevice")
    dataset = library.uvm512_dataset_create(
        args.allocations, args.allocation_bytes, args.guard_bytes
    )
    if not dataset:
        raise RuntimeError("failed to create managed dataset")
    stream = None
    context = None
    try:
        if int(library.uvm512_dataset_count(dataset)) != args.allocations:
            raise RuntimeError("dataset allocation count mismatch")
        if int(library.uvm512_dataset_allocation_bytes(dataset)) != args.allocation_bytes:
            raise RuntimeError("dataset allocation size mismatch")
        pointers = legacy.pointer_snapshot(library, dataset, args.allocations)
        if len(set(pointers)) != args.allocations:
            raise RuntimeError("dataset returned duplicate managed VAs")
        if any(pointer % VA_BLOCK_BYTES != 0 for pointer in pointers):
            raise RuntimeError("every 4MiB region must begin on a 2MiB VA-block boundary")
        adjacent, min_gap = legacy.layout_metrics(pointers, args.allocation_bytes)
        if adjacent != 0 or min_gap < args.guard_bytes:
            raise RuntimeError(
                f"test layout is not guard-separated: adjacent={adjacent} gap={min_gap}"
            )
        expected_checksum = int(library.uvm512_dataset_expected_checksum(dataset))
        stream = library.uvm512_stream_create()
        if not stream:
            raise RuntimeError("failed to create CUDA stream")

        pointer_array = (ctypes.c_size_t * args.allocations)(*pointers)
        size_array = (ctypes.c_size_t * args.allocations)(
            *([args.allocation_bytes] * args.allocations)
        )
        slot_ids = tuple(range(args.allocations))
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
                    slot_type(*slot_ids[begin:end]),
                )
            )
        if len(batches) != 4:
            raise AssertionError("all modes require exactly four ioctl batches")

        context = library.uvm512_driver_batch_context_create(
            args.device, pointers[0], args.allocation_bytes
        )
        if not context:
            raise RuntimeError(f"driver context unavailable: {driver_error(library)}")

        abi = {
            "generic_ioctl": int(library.uvm512_driver_batch_ioctl_command()),
            "generic_max_ranges": int(library.uvm512_driver_batch_max_ranges()),
            "generic_params_size": int(library.uvm512_driver_batch_params_size()),
            "phase_f_ioctl": int(
                library.uvm512_phase_f_batch_executor_ioctl_command()
            ),
            "phase_f_max_ranges": int(
                library.uvm512_phase_f_batch_executor_max_ranges()
            ),
            "phase_f_params_size": int(
                library.uvm512_phase_f_batch_executor_params_size()
            ),
            "phase_f_action_prepare_remote_read": int(
                library.uvm512_phase_f_action_prepare_remote_read()
            ),
            "phase_f_action_force_remote_read_64k": int(
                library.uvm512_phase_f_action_force_remote_read_64k()
            ),
            "slot_ioctl": int(library.uvm512_slot_batch_ioctl_command()),
            "slot_max_entries": int(library.uvm512_slot_batch_max_entries()),
            "slot_params_size": int(library.uvm512_slot_batch_params_size()),
            "slot_mode_prealloc": int(library.uvm512_slot_mode_vidmem_prealloc()),
        }
        for mode_name, export in MODE_EXPORTS.items():
            abi[f"slot_mode_{mode_name}"] = int(getattr(library, export)())
        expected_abi = {
            "generic_ioctl": 90,
            "generic_max_ranges": 128,
            "generic_params_size": 9280,
            "phase_f_ioctl": 81,
            "phase_f_max_ranges": 64,
            "phase_f_params_size": 4656,
            "phase_f_action_prepare_remote_read": 5,
            "phase_f_action_force_remote_read_64k": 20,
            "slot_ioctl": 91,
            "slot_max_entries": 128,
            "slot_params_size": 6224,
            "slot_mode_prealloc": 8,
            "slot_mode_normal_64k_e2e": 12,
            "slot_mode_overwrite_64k_e2e": 17,
            "slot_mode_normal_64k_stage_sync": 16,
            "slot_mode_overwrite_64k_stage_sync": 18,
        }
        for name, value in expected_abi.items():
            if abi.get(name) != value:
                raise RuntimeError(f"ABI {name}={abi.get(name)}/{value}")
        selected_mode = abi[f"slot_mode_{args.mode}"]

        # Context creation probes ioctl 90 and migrates the first range. That
        # setup is outside the sample; restore the complete source set first.
        initial_reset = reset_to_cpu(
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
        if prealloc.remap or prealloc.tlb_invalidate or prealloc.copy_pushes:
            raise RuntimeError(f"preallocation unexpectedly remapped: {prealloc}")

        for _ in range(args.warmups):
            establish_cpu_remote_64k(
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
                mode=selected_mode,
            )
            checksum(library, dataset, stream, expected_checksum)

        measured_precondition = establish_cpu_remote_64k(
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
            if profile_before_first.get(name) != profile_before.get(name):
                raise RuntimeError(f"profile changed across baseline reads: {name}")

        wall_begin = time.perf_counter_ns()
        outcome = execute_transfer(
            args=args,
            library=library,
            context=context,
            batches=batches,
            mode=selected_mode,
        )
        wall_end = time.perf_counter_ns()
        profile_after = read_profile(args.stats)
        profile = profile_delta(profile_before, profile_after)
        required_profile = expected_profile(
            args.mode, args.allocations, logical_bytes, profile, outcome
        )
        require_fields(profile, required_profile)

        overwrite = args.mode in OVERWRITE_MODES
        stage_sync = args.mode in STAGE_SYNC_MODES
        positive_stages = [
            "total",
            "prepare",
            "copy",
            "map",
            "wait",
            *PARENT_PROFILE_STAGES,
            *RESIDUAL_PROFILE_STAGES,
        ]
        if not overwrite:
            positive_stages.append("unmap")
        for stage in positive_stages:
            if profile[f"slot_direct_profile.{stage}.ns"] <= 0:
                raise RuntimeError(f"slot profile stage {stage} has non-positive time")
        if (
            overwrite
            and profile["slot_direct_profile.unmap.ns"] != 0
        ):
            raise RuntimeError("direct-overwrite unexpectedly ran a separate unmap")
        if stage_sync:
            if any(
                profile[f"slot_direct_profile.{stage}.ns"] <= 0
                for stage in COMPLETE_PROFILE_STAGES
            ):
                raise RuntimeError("stage-sync completion timing is non-positive")

        wall_ns = wall_end - wall_begin
        if wall_ns <= 0:
            raise RuntimeError("non-positive measurement")
        gib_s = (logical_bytes / legacy.GIB) / (wall_ns / 1_000_000_000.0)
        if not math.isfinite(gib_s) or gib_s <= 0:
            raise RuntimeError(f"invalid bandwidth {gib_s}")

        def profile_ns(stage: str) -> int:
            return int(profile[f"slot_direct_profile.{stage}.ns"])

        def profile_ms(stage: str) -> float:
            return profile_ns(stage) / 1_000_000.0

        transfer_stages = (
            COMPLETE_PROFILE_STAGES
            if stage_sync
            else ("prepare", "copy", "map")
        )
        core_stages = transfer_stages + ("wait",)
        parent_components = {
            "pre_helper_total": PRE_HELPER_PROFILE_STAGES,
            "helper_total": transfer_stages + HELPER_RESIDUAL_PROFILE_STAGES,
            "post_helper_total": ("wait",) + POST_HELPER_PROFILE_STAGES,
        }

        parent_residual_ns = {
            parent: profile_ns(parent)
            - sum(profile_ns(component) for component in components)
            for parent, components in parent_components.items()
        }
        parent_partition_residual_ns = profile_ns("total") - sum(
            profile_ns(parent) for parent in PARENT_PROFILE_STAGES
        )
        driver_accounted_ns = sum(profile_ns(stage) for stage in core_stages) + sum(
            profile_ns(stage) for stage in RESIDUAL_PROFILE_STAGES
        )
        driver_residual_ns = profile_ns("total") - driver_accounted_ns
        host_outer_ns = wall_ns - profile_ns("total")
        reconstructed_driver_residual_ns = (
            parent_partition_residual_ns + sum(parent_residual_ns.values())
        )

        if parent_partition_residual_ns != 0:
            raise RuntimeError(
                "slot profile parent partition does not close: "
                f"residual_ns={parent_partition_residual_ns}"
            )
        if any(value < 0 for value in parent_residual_ns.values()):
            raise RuntimeError(
                f"slot profile parent has negative residual: {parent_residual_ns}"
            )
        if driver_residual_ns < 0 or driver_residual_ns != reconstructed_driver_residual_ns:
            raise RuntimeError(
                "slot profile leaf closure is inconsistent: "
                f"driver_residual_ns={driver_residual_ns} "
                f"reconstructed_ns={reconstructed_driver_residual_ns}"
            )
        if host_outer_ns < 0:
            raise RuntimeError(
                "host wall time is below driver profile total: "
                f"wall_ns={wall_ns} total_ns={profile_ns('total')}"
            )

        driver_closure_fraction = driver_accounted_ns / profile_ns("total")
        if (
            not math.isfinite(driver_closure_fraction)
            or driver_closure_fraction <= 0
            or driver_closure_fraction > 1
        ):
            raise RuntimeError(
                f"invalid driver closure fraction {driver_closure_fraction}"
            )

        detailed_profile_ms = {
            f"{stage}_ms": profile_ms(stage)
            for stage in PARENT_PROFILE_STAGES + RESIDUAL_PROFILE_STAGES
        }

        copy_complete_ms = profile_ms("copy_complete") if stage_sync else None
        copy_tail_ms = (
            copy_complete_ms - profile_ms("copy") if stage_sync else None
        )
        copy_complete_gib_s = (
            (logical_bytes / legacy.GIB) / (copy_complete_ms / 1000.0)
            if stage_sync
            else None
        )
        if stage_sync and (
            copy_tail_ms is None
            or copy_tail_ms < 0
            or copy_complete_gib_s is None
            or not math.isfinite(copy_complete_gib_s)
            or copy_complete_gib_s <= 0
        ):
            raise RuntimeError(
                f"invalid copy completion diagnostic: tail={copy_tail_ms} "
                f"gib_s={copy_complete_gib_s}"
            )
        post_residency = query_residency(
            library, context, pointer_array, size_array, args.allocations
        )
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
        if cpu_last_after != args.allocations:
            raise RuntimeError(
                f"slot mapping changed CPU LastPrefetchLocation: "
                f"{cpu_last_after}/{args.allocations}"
            )
        va_stable = (
            legacy.pointer_snapshot(library, dataset, args.allocations) == pointers
        )
        if not va_stable:
            raise RuntimeError("managed VA changed after transfer")

        return {
            "schema_version": SCHEMA_VERSION,
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
                "expected_va_blocks": expected_blocks,
                "warmups": args.warmups,
                "slot_layer_id": args.slot_layer_id,
                "slot_expert_id": args.slot_expert_id,
            },
            "abi": abi,
            "layout": {
                "adjacent_test_ranges": adjacent,
                "min_gap_bytes": min_gap,
                "all_regions_2m_aligned": True,
                "va_snapshot": [f"0x{pointer:x}" for pointer in pointers],
            },
            "setup": {
                "slot_vidmem_preallocated_before_timing": True,
                "initial_cpu_reset_after_context_probe": initial_reset,
                "prealloc": asdict(prealloc),
            },
            "precondition": measured_precondition,
            "measurement": {
                "cpu_reset_action5_action20_immediately_before_profile_and_timing": True,
                "host_monotonic_to_synchronous_ioctl_completion": True,
                "wall_ns": wall_ns,
                "wall_ms": wall_ns / 1_000_000.0,
                "driver_profile_total_ms": profile_ms("total"),
                "prepare_ms": profile_ms("prepare"),
                "unmap_ms": profile_ms("unmap"),
                "copy_ms": profile_ms("copy"),
                "map_ms": profile_ms("map"),
                "wait_ms": profile_ms("wait"),
                "prepare_complete_ms": (
                    profile_ms("prepare_complete") if stage_sync else None
                ),
                "copy_complete_ms": (
                    copy_complete_ms
                ),
                "map_complete_ms": (
                    profile_ms("map_complete") if stage_sync else None
                ),
                **detailed_profile_ms,
                "driver_accounted_ms": driver_accounted_ns / 1_000_000.0,
                "driver_residual_ms": driver_residual_ns / 1_000_000.0,
                "driver_closure_fraction": driver_closure_fraction,
                "host_outer_ms": host_outer_ns / 1_000_000.0,
                "parent_partition_residual_ms": (
                    parent_partition_residual_ns / 1_000_000.0
                ),
                "pre_helper_residual_ms": (
                    parent_residual_ns["pre_helper_total"] / 1_000_000.0
                ),
                "helper_residual_ms": (
                    parent_residual_ns["helper_total"] / 1_000_000.0
                ),
                "post_helper_residual_ms": (
                    parent_residual_ns["post_helper_total"] / 1_000_000.0
                ),
                "component_source": "slot_direct_profile",
                "submit_stages_are_not_ce_completion": True,
                "stage_sync_completion_boundaries": stage_sync,
                "copy_complete_gib_s": copy_complete_gib_s,
                "copy_complete_minus_submit_ms": copy_tail_ms,
                "gib_s": gib_s,
                "outcome": asdict(outcome),
            },
            "validation": {
                "post_residency": asdict(post_residency),
                "post_checksum_residency": asdict(post_checksum_residency),
                "slot_mapping_is_not_uvm_gpu_residency": True,
                "cpu_last_prefetch_after": cpu_last_after,
                "checksum": checksum_value,
                "expected_checksum": expected_checksum,
                "full_gpu_checksum": True,
                "checksum_ms": checksum_ms,
                "va_stable": va_stable,
            },
            "profile": {
                "snapshot_after_cpu_reset_action5_action20": True,
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
            "schema_version": SCHEMA_VERSION,
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
