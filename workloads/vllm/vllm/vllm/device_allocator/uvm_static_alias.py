# SPDX-License-Identifier: Apache-2.0
"""Userspace wrapper for the experimental UVM static-alias UAPI.

The static-alias commands are control-plane operations and are never issued
during CUDA Graph capture or replay. A caller maps one or more original managed
VAs, seals the immutable session, and captures Graphs which use those VAs. A
scheduler barrier may PARK read-only consumers for remote decode and RESUME
them for alias prefill while retaining upload VAs and backing. Final DESTROY
still runs only after every Graph and CUDA operation has completed.

This module deliberately owns state independently from ``uvm.py``'s dynamic
cmd96 slot-transition session. It reuses only that module's low-level ioctl,
GPU-UUID, and initialized-UVM-fd discovery helpers so MAP targets the UVM VA
space which owns the submitted managed addresses.
"""

from __future__ import annotations

import atexit
import ctypes
import dataclasses
import os
import threading
import time
from contextlib import suppress
from typing import Any, Sequence

import torch

from vllm.device_allocator import uvm as _uvm

UVM_MOE_STATIC_ALIAS_ABI_VERSION = 1
UVM_MOE_STATIC_ALIAS_MAX_ENTRIES = 4608

_UVM_MOE_STATIC_ALIAS_MAP_IOCTL = 97
_UVM_MOE_STATIC_ALIAS_SEAL_IOCTL = 98
_UVM_MOE_STATIC_ALIAS_DESTROY_IOCTL = 99
_UVM_MOE_STATIC_ALIAS_PARK_IOCTL = 101
_UVM_MOE_STATIC_ALIAS_RESUME_IOCTL = 102
_UVM_MOE_STATIC_ALIAS_REBIND_IOCTL = 103
_UVM_MOE_STATIC_ALIAS_QUIESCED_REMOVE_IOCTL = 104

UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_FORCE_SEQUENTIAL = 1 << 0
UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_DEFER_CPU_RESTORE = 1 << 1
UVM_MOE_STATIC_ALIAS_MAP_FLAG_QUIESCED_APPEND = 1 << 0
UVM_MOE_STATIC_ALIAS_MAP_FLAG_DEFER_DUAL_REFRESH = 1 << 1
UVM_MOE_STATIC_ALIAS_SEAL_FLAG_REFRESH_DUAL = 1 << 0
UVM_MOE_STATIC_ALIAS_SEAL_FLAG_REFRESH_DUAL_NESTED = 1 << 1
UVM_MOE_STATIC_ALIAS_SEAL_FLAG_DISCARD_DUAL = 1 << 2
_UVM_MOE_STATIC_ALIAS_MAP_FLAGS_ALL = (
    UVM_MOE_STATIC_ALIAS_MAP_FLAG_QUIESCED_APPEND
    | UVM_MOE_STATIC_ALIAS_MAP_FLAG_DEFER_DUAL_REFRESH
)
UVM_MOE_STATIC_ALIAS_QUIESCED_REMOVE_FLAG_RELEASE_BACKING = 1 << 0

UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW = 1 << 0
UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K = 1 << 1
_UVM_MOE_STATIC_ALIAS_ENTRY_FLAGS_ALL = (
    UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW
    | UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K
)
UVM_MOE_STATIC_ALIAS_DESTROY_FLAG_RELEASE_BACKING = 1 << 0
UVM_MOE_STATIC_ALIAS_DESTROY_FLAG_DEFER_CPU_RESTORE = 1 << 1
UVM_MOE_STATIC_ALIAS_REBIND_FLAG_BEGIN = 1 << 0
UVM_MOE_STATIC_ALIAS_REBIND_FLAG_COMMIT = 1 << 1
UVM_MOE_STATIC_ALIAS_REBIND_FLAG_ABORT = 1 << 2
UVM_MOE_STATIC_ALIAS_REBIND_FLAG_RESIZE = 1 << 3
_UVM_MOE_STATIC_ALIAS_REBIND_FLAGS_ALL = (
    UVM_MOE_STATIC_ALIAS_REBIND_FLAG_BEGIN
    | UVM_MOE_STATIC_ALIAS_REBIND_FLAG_COMMIT
    | UVM_MOE_STATIC_ALIAS_REBIND_FLAG_ABORT
    | UVM_MOE_STATIC_ALIAS_REBIND_FLAG_RESIZE
)
_UVM_MOE_STATIC_ALIAS_MAX_REBIND_CONSUMERS = 65536

_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_PAGE_BYTES = 4096
_PTE_64K_BYTES = 64 * 1024

# Keep these aliases patchable by CPU-only tests. They are the only coupling to
# the older wrapper and do not share any cmd96 owner, plan, or generation state.
_libc_ioctl = _uvm._libc_ioctl
_phase_f_gpu_uuid = _uvm._phase_f_gpu_uuid
_iter_existing_uvm_fd_sources = _uvm._iter_existing_uvm_fd_sources
_initialize_duplicated_uvm_fd = _uvm._initialize_duplicated_uvm_fd


class _StaticAliasUuid(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)]


class _StaticAliasEntry(ctypes.Structure):
    _fields_ = [
        ("backing_layer_id", ctypes.c_uint32),
        ("backing_expert_id", ctypes.c_uint32),
        ("backing_slot_id", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("va_base", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("slot_offset", ctypes.c_uint64),
        ("status", ctypes.c_uint32),
    ]


class _StaticAliasMapParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("session_id", ctypes.c_uint64),
        ("gpu_uuid", _StaticAliasUuid),
        ("requested_bytes", ctypes.c_uint64),
        ("mapped_bytes", ctypes.c_uint64),
        ("total_alias_bytes", ctypes.c_uint64),
        ("backing_bytes", ctypes.c_uint64),
        ("count", ctypes.c_uint32),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("alias_count", ctypes.c_uint32),
        ("tlb_invalidate_count", ctypes.c_uint32),
        ("poisoned", ctypes.c_uint32),
        (
            "entries",
            _StaticAliasEntry * UVM_MOE_STATIC_ALIAS_MAX_ENTRIES,
        ),
        ("rmStatus", ctypes.c_uint32),
    ]


class _StaticAliasSealParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("session_id", ctypes.c_uint64),
        ("expected_alias_bytes", ctypes.c_uint64),
        ("alias_bytes", ctypes.c_uint64),
        ("backing_bytes", ctypes.c_uint64),
        ("expected_alias_count", ctypes.c_uint32),
        ("sealed", ctypes.c_uint32),
        ("alias_count", ctypes.c_uint32),
        ("poisoned", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
    ]


class _StaticAliasDestroyParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("session_id", ctypes.c_uint64),
        ("released_alias_bytes", ctypes.c_uint64),
        ("remaining_alias_bytes", ctypes.c_uint64),
        ("released_backing_bytes", ctypes.c_uint64),
        ("remaining_backing_bytes", ctypes.c_uint64),
        ("released_alias_count", ctypes.c_uint32),
        ("remaining_alias_count", ctypes.c_uint32),
        ("session_cleared", ctypes.c_uint32),
        ("poisoned", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
    ]


class _StaticAliasTransitionParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("session_id", ctypes.c_uint64),
        ("processed_alias_bytes", ctypes.c_uint64),
        ("consumer_alias_bytes", ctypes.c_uint64),
        ("backing_bytes", ctypes.c_uint64),
        ("processed_alias_count", ctypes.c_uint32),
        ("consumer_alias_count", ctypes.c_uint32),
        ("upload_alias_count", ctypes.c_uint32),
        ("tlb_invalidate_count", ctypes.c_uint32),
        ("parked", ctypes.c_uint32),
        ("sealed", ctypes.c_uint32),
        ("poisoned", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
        ("validation_ns", ctypes.c_uint64),
        ("build_sort_ns", ctypes.c_uint64),
        ("preflight_ns", ctypes.c_uint64),
        ("metadata_ns", ctypes.c_uint64),
        ("cpu_unmap_ns", ctypes.c_uint64),
        ("pte_map_ns", ctypes.c_uint64),
        ("wait_ns", ctypes.c_uint64),
        ("bookkeeping_ns", ctypes.c_uint64),
        ("dual_view_full_coverage", ctypes.c_uint32),
        ("dual_view_block_count", ctypes.c_uint32),
    ]


class _StaticAliasRebindParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("session_id", ctypes.c_uint64),
        ("rebind_id", ctypes.c_uint64),
        ("expected_consumer_alias_bytes", ctypes.c_uint64),
        ("staged_consumer_alias_bytes", ctypes.c_uint64),
        ("backing_bytes", ctypes.c_uint64),
        ("total_alias_bytes", ctypes.c_uint64),
        ("count", ctypes.c_uint32),
        ("accepted_count", ctypes.c_uint32),
        ("expected_consumer_alias_count", ctypes.c_uint32),
        ("staged_consumer_alias_count", ctypes.c_uint32),
        ("upload_alias_count", ctypes.c_uint32),
        ("committed", ctypes.c_uint32),
        ("poisoned", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        (
            "entries",
            _StaticAliasEntry * UVM_MOE_STATIC_ALIAS_MAX_ENTRIES,
        ),
        ("rmStatus", ctypes.c_uint32),
    ]


class _StaticAliasQuiescedRemoveParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("session_id", ctypes.c_uint64),
        ("gpu_uuid", _StaticAliasUuid),
        ("requested_bytes", ctypes.c_uint64),
        ("released_alias_bytes", ctypes.c_uint64),
        ("total_alias_bytes", ctypes.c_uint64),
        ("backing_bytes", ctypes.c_uint64),
        ("count", ctypes.c_uint32),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("remaining_alias_count", ctypes.c_uint32),
        ("tlb_invalidate_count", ctypes.c_uint32),
        ("poisoned", ctypes.c_uint32),
        (
            "entries",
            _StaticAliasEntry * UVM_MOE_STATIC_ALIAS_MAX_ENTRIES,
        ),
        ("rmStatus", ctypes.c_uint32),
    ]


def _assert_static_alias_abi() -> None:
    """Fail before an ioctl if the hand-written ctypes layout drifts."""
    expected = {
        "entry_size": (ctypes.sizeof(_StaticAliasEntry), 48),
        "entry_va_base": (_StaticAliasEntry.va_base.offset, 16),
        "entry_status": (_StaticAliasEntry.status.offset, 40),
        "map_size": (ctypes.sizeof(_StaticAliasMapParams), 221288),
        "map_session_id": (_StaticAliasMapParams.session_id.offset, 16),
        "map_requested_bytes": (
            _StaticAliasMapParams.requested_bytes.offset,
            40,
        ),
        "map_count": (_StaticAliasMapParams.count.offset, 72),
        "map_entries": (_StaticAliasMapParams.entries.offset, 96),
        "map_rm_status": (_StaticAliasMapParams.rmStatus.offset, 221280),
        "seal_size": (ctypes.sizeof(_StaticAliasSealParams), 72),
        "seal_session_id": (_StaticAliasSealParams.session_id.offset, 16),
        "seal_expected_count": (
            _StaticAliasSealParams.expected_alias_count.offset,
            48,
        ),
        "seal_rm_status": (_StaticAliasSealParams.rmStatus.offset, 64),
        "destroy_size": (ctypes.sizeof(_StaticAliasDestroyParams), 80),
        "destroy_session_id": (
            _StaticAliasDestroyParams.session_id.offset,
            16,
        ),
        "destroy_released_aliases": (
            _StaticAliasDestroyParams.released_alias_count.offset,
            56,
        ),
        "destroy_rm_status": (
            _StaticAliasDestroyParams.rmStatus.offset,
            72,
        ),
        "transition_size": (ctypes.sizeof(_StaticAliasTransitionParams), 152),
        "transition_batch_count": (
            _StaticAliasTransitionParams.reserved.offset,
            12,
        ),
        "transition_session_id": (
            _StaticAliasTransitionParams.session_id.offset,
            16,
        ),
        "transition_processed_count": (
            _StaticAliasTransitionParams.processed_alias_count.offset,
            48,
        ),
        "transition_rm_status": (
            _StaticAliasTransitionParams.rmStatus.offset,
            76,
        ),
        "rebind_size": (ctypes.sizeof(_StaticAliasRebindParams), 221288),
        "rebind_session_id": (_StaticAliasRebindParams.session_id.offset, 16),
        "rebind_id": (_StaticAliasRebindParams.rebind_id.offset, 24),
        "rebind_count": (_StaticAliasRebindParams.count.offset, 64),
        "rebind_entries": (_StaticAliasRebindParams.entries.offset, 96),
        "rebind_rm_status": (_StaticAliasRebindParams.rmStatus.offset, 221280),
        "quiesced_remove_size": (
            ctypes.sizeof(_StaticAliasQuiescedRemoveParams),
            221288,
        ),
        "quiesced_remove_session_id": (
            _StaticAliasQuiescedRemoveParams.session_id.offset,
            16,
        ),
        "quiesced_remove_count": (
            _StaticAliasQuiescedRemoveParams.count.offset,
            72,
        ),
        "quiesced_remove_entries": (
            _StaticAliasQuiescedRemoveParams.entries.offset,
            96,
        ),
        "quiesced_remove_rm_status": (
            _StaticAliasQuiescedRemoveParams.rmStatus.offset,
            221280,
        ),
    }
    mismatches = [
        f"{name}={actual} (expected {wanted})"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise RuntimeError(
            "UVM MoE static-alias ioctl ABI mismatch: " + ", ".join(mismatches)
        )


_assert_static_alias_abi()


@dataclasses.dataclass
class _StaticAliasSession:
    fd: int
    uuid_bytes: bytes
    session_id: int
    owner_pid: int
    sealed: bool = False
    parked: bool = False
    poisoned: bool = False
    alias_count: int = 0
    alias_bytes: int = 0
    backing_bytes: int = 0
    target_ranges: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    backing_identities: set[tuple[int, int, int]] = dataclasses.field(
        default_factory=set
    )
    alias_entries: list[tuple[int, int, int, int, int, int, int]] = (
        dataclasses.field(default_factory=list)
    )


_static_alias_sessions: dict[int, _StaticAliasSession] = {}
_static_alias_lock = threading.RLock()
_static_alias_close_registered = False


def _capture_rejection() -> tuple[str, str] | None:
    try:
        capturing = bool(torch.cuda.is_current_stream_capturing())
    except Exception as exc:
        return ("cuda_graph_capture_query_failed", repr(exc))
    if capturing:
        return (
            "cuda_graph_capture_unsupported",
            "static-alias ioctls are initialization/teardown operations",
        )
    return None


def _normalize_device(device: int) -> tuple[int, str]:
    if not isinstance(device, int) or isinstance(device, bool) or device < 0:
        return 0, "device must be a non-negative integer"
    return int(device), ""


def _normalize_entries(
    entries: Sequence[Sequence[int]],
    *,
    maximum_entries: int | None = UVM_MOE_STATIC_ALIAS_MAX_ENTRIES,
) -> tuple[list[tuple[int, int, int, int, int, int, int]], str]:
    if isinstance(entries, (str, bytes)):
        return [], "entries must be a sequence of seven-integer records"
    try:
        records = list(entries)
    except TypeError:
        return [], "entries must be a sequence"
    if not records:
        return [], "at least one alias entry is required"
    if maximum_entries is not None and len(records) > maximum_entries:
        return [], f"entry count exceeds the {maximum_entries}-entry limit"

    normalized: list[tuple[int, int, int, int, int, int, int]] = []
    target_ranges: list[tuple[int, int]] = []
    for index, record in enumerate(records):
        if isinstance(record, (str, bytes)):
            return [], f"entry {index} is not a seven-integer record"
        try:
            values = tuple(record)
        except TypeError:
            return [], f"entry {index} is not a sequence"
        if len(values) != 7:
            return [], f"entry {index} must contain seven integers"
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            return [], f"entry {index} contains a non-integer field"
        (
            backing_layer_id,
            backing_expert_id,
            backing_slot_id,
            flags,
            va_base,
            length,
            slot_offset,
        ) = (int(value) for value in values)
        if any(
            value < 0 or value > _UINT32_MAX
            for value in (
                backing_layer_id,
                backing_expert_id,
                backing_slot_id,
                flags,
            )
        ):
            return [], f"entry {index} contains a uint32 overflow"
        if flags & ~_UVM_MOE_STATIC_ALIAS_ENTRY_FLAGS_ALL:
            return [], f"entry {index} contains unknown flags"
        if va_base <= 0 or length <= 0 or slot_offset < 0:
            return [], f"entry {index} has an invalid VA, length, or offset"
        if any(value > _UINT64_MAX for value in (va_base, length, slot_offset)):
            return [], f"entry {index} contains a uint64 overflow"
        if va_base > _UINT64_MAX - length or slot_offset > _UINT64_MAX - length:
            return [], f"entry {index} range overflows uint64"
        alignment = (
            _PTE_64K_BYTES
            if flags & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K
            else _PAGE_BYTES
        )
        if any(value % alignment for value in (va_base, length, slot_offset)):
            return [], f"entry {index} is not {alignment}-byte aligned"
        end = va_base + length
        target_ranges.append((va_base, end))
        normalized.append(
            (
                backing_layer_id,
                backing_expert_id,
                backing_slot_id,
                flags,
                va_base,
                length,
                slot_offset,
            )
        )

    ordered = sorted(target_ranges)
    if any(right[0] < left[1] for left, right in zip(ordered, ordered[1:])):
        return [], "target alias VA ranges overlap"
    pte_modes = {
        bool(record[3] & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K)
        for record in normalized
    }
    if len(pte_modes) != 1:
        return [], "one MAP cannot mix 4K and 64K PTE entries"
    return normalized, ""


def _make_uuid(device: int) -> tuple[_StaticAliasUuid | None, bytes, str]:
    try:
        source = _phase_f_gpu_uuid(int(device))
        uuid_bytes = bytes(source.uuid)
    except Exception as exc:
        return None, b"", repr(exc)
    if len(uuid_bytes) != 16:
        return None, b"", f"unexpected GPU UUID size {len(uuid_bytes)}"
    value = _StaticAliasUuid()
    value.uuid[:] = uuid_bytes
    return value, uuid_bytes, ""


def _base_result(operation: str, **values: Any) -> dict[str, object]:
    return {
        "ok": False,
        "operation": operation,
        "abi_version": UVM_MOE_STATIC_ALIAS_ABI_VERSION,
        **values,
    }


def _session_block_reason(
    session: _StaticAliasSession,
    *,
    allow_poisoned: bool = False,
) -> tuple[str, str] | None:
    if session.owner_pid != os.getpid():
        return (
            "fork_inherited_static_alias_session",
            f"session belongs to pid {session.owner_pid}, current pid is {os.getpid()}",
        )
    if session.poisoned and not allow_poisoned:
        return ("static_alias_session_poisoned", "session is destroy-only")
    return None


def _fill_map_params(
    normalized: Sequence[tuple[int, int, int, int, int, int, int]],
    *,
    session_id: int,
    gpu_uuid: _StaticAliasUuid,
    flags: int,
) -> _StaticAliasMapParams:
    params = _StaticAliasMapParams()
    params.abi_version = UVM_MOE_STATIC_ALIAS_ABI_VERSION
    params.struct_size = ctypes.sizeof(_StaticAliasMapParams)
    params.flags = int(flags)
    params.session_id = int(session_id)
    params.gpu_uuid = gpu_uuid
    params.count = len(normalized)
    for index, record in enumerate(normalized):
        (
            backing_layer_id,
            backing_expert_id,
            backing_slot_id,
            entry_flags,
            va_base,
            length,
            slot_offset,
        ) = record
        entry = params.entries[index]
        entry.backing_layer_id = backing_layer_id
        entry.backing_expert_id = backing_expert_id
        entry.backing_slot_id = backing_slot_id
        entry.flags = entry_flags
        entry.va_base = va_base
        entry.bytes = length
        entry.slot_offset = slot_offset
    return params


def _submit_map(
    fd: int,
    normalized: Sequence[tuple[int, int, int, int, int, int, int]],
    *,
    session_id: int,
    gpu_uuid: _StaticAliasUuid,
    flags: int,
    prior_alias_count: int,
    prior_alias_bytes: int,
) -> dict[str, object]:
    params = _fill_map_params(
        normalized,
        session_id=session_id,
        gpu_uuid=gpu_uuid,
        flags=flags,
    )
    expected_bytes = sum(record[5] for record in normalized)
    started_ns = time.perf_counter_ns()
    ctypes.set_errno(0)
    try:
        ret = int(_libc_ioctl(fd, _UVM_MOE_STATIC_ALIAS_MAP_IOCTL, params))
    except Exception as exc:
        return _base_result(
            "map",
            reason="map_ioctl_exception",
            error=repr(exc),
            requested_session_id=int(session_id),
            requested_count=len(normalized),
            expected_requested_bytes=expected_bytes,
            ioctl_wall_ns=time.perf_counter_ns() - started_ns,
            side_effect_indeterminate=True,
            made_progress=True,
            poisoned=True,
        )
    wall_ns = time.perf_counter_ns() - started_ns
    errno_value = ctypes.get_errno() if ret != 0 else 0
    returned_session_id = int(params.session_id)
    statuses = [int(params.entries[index].status) for index in range(len(normalized))]
    poisoned_raw = int(params.poisoned)
    boolean_fields_ok = poisoned_raw in (0, 1)
    expected_total_alias_bytes = prior_alias_bytes + expected_bytes
    expected_total_alias_count = prior_alias_count + len(normalized)
    returned_alias_count = int(params.alias_count)
    returned_total_alias_count = prior_alias_count + returned_alias_count
    success_contract_ok = (
        ret == 0
        and int(params.rmStatus) == 0
        and boolean_fields_ok
        and poisoned_raw == 0
        and returned_session_id > 0
        and (session_id == 0 or returned_session_id == session_id)
        and int(params.requested_bytes) == expected_bytes
        and int(params.mapped_bytes) == expected_bytes
        and int(params.total_alias_bytes) == expected_total_alias_bytes
        and int(params.backing_bytes) > 0
        and int(params.success_count) == len(normalized)
        and int(params.failure_count) == 0
        and int(params.alias_count) == len(normalized)
        and statuses == [0] * len(normalized)
    )
    # A failed probe may come from an initialized UVM file description which
    # does not own the CUDA VA space. A bare returned session id has no PTE or
    # backing side effect and must not stop candidate probing. Any mutation is
    # represented by one of the counters below or by the poisoned bit.
    made_progress = (
        int(params.mapped_bytes) != 0
        or int(params.total_alias_bytes) != 0
        or int(params.backing_bytes) != 0
        or int(params.success_count) != 0
        or int(params.alias_count) != 0
        or int(params.tlb_invalidate_count) != 0
        or poisoned_raw != 0
    )
    ioctl_completed = ret == 0 and int(params.rmStatus) == 0
    contract_violation = ioctl_completed and not success_contract_ok
    return _base_result(
        "map",
        ok=success_contract_ok,
        reason=(
            "ok"
            if success_contract_ok
            else "map_contract_mismatch"
            if contract_violation
            else "map_ioctl_failed"
        ),
        requested_session_id=int(session_id),
        session_id=returned_session_id,
        requested_count=len(normalized),
        requested_bytes=int(params.requested_bytes),
        expected_requested_bytes=expected_bytes,
        mapped_bytes=int(params.mapped_bytes),
        total_alias_bytes=int(params.total_alias_bytes),
        backing_bytes=int(params.backing_bytes),
        success_count=int(params.success_count),
        failure_count=int(params.failure_count),
        alias_count=returned_alias_count,
        # The v1 UAPI reports alias_count for this MAP. Combining it with the
        # cached pre-call count also describes a failed MAP which committed
        # its complete topology before a later PTE operation failed.
        total_alias_count=returned_total_alias_count,
        tlb_invalidate_count=int(params.tlb_invalidate_count),
        entry_statuses=statuses,
        poisoned=(poisoned_raw == 1 or contract_violation),
        kernel_poisoned=poisoned_raw == 1,
        boolean_fields_ok=boolean_fields_ok,
        success_contract_ok=success_contract_ok,
        contract_violation=contract_violation,
        made_progress=made_progress,
        side_effect_indeterminate=False,
        ioctl_ret=ret,
        errno=errno_value,
        rm_status=int(params.rmStatus),
        ioctl_calls=1,
        ioctl_wall_ns=wall_ns,
    )


def _sync_cached_map_result(
    cached: _StaticAliasSession,
    normalized: Sequence[tuple[int, int, int, int, int, int, int]],
    result: dict[str, object],
) -> None:
    """Mirror kernel-visible MAP progress into the destroy contract cache."""
    alias_count = int(result.get("alias_count") or 0)
    total_alias_count = result.get("total_alias_count")
    if total_alias_count is None:
        total_alias_count = cached.alias_count + alias_count

    cached.alias_count = int(total_alias_count)
    if "total_alias_bytes" in result:
        cached.alias_bytes = int(result["total_alias_bytes"])
    if "backing_bytes" in result:
        cached.backing_bytes = int(result["backing_bytes"])

    # MAP v1 commits a whole request topology at once. Only claim its ranges
    # and backing identities when the kernel confirms the full batch count.
    # An indeterminate/partial result remains destroy-only without inventing
    # which subset became active.
    if alias_count == len(normalized):
        cached.alias_entries.extend(normalized)
        cached.target_ranges.extend(
            (record[4], record[4] + record[5]) for record in normalized
        )
        cached.backing_identities.update(
            (record[0], record[1], record[2]) for record in normalized
        )


def _register_close_once() -> None:
    global _static_alias_close_registered
    if not _static_alias_close_registered:
        atexit.register(_close_static_alias_sessions)
        _static_alias_close_registered = True


def driver_static_alias_map(
    entries: Sequence[Sequence[int]],
    *,
    session_id: int = 0,
    device: int = 0,
    flags: int = 0,
) -> dict[str, object]:
    """Create/append aliases, including a scheduler-quiesced sealed append."""
    started_ns = time.perf_counter_ns()
    normalized_device, device_error = _normalize_device(device)
    normalized, validation_error = _normalize_entries(entries)
    if (
        not isinstance(session_id, int)
        or isinstance(session_id, bool)
        or session_id < 0
        or session_id > _UINT64_MAX
    ):
        validation_error = "session_id is outside uint64"
    if (
        not isinstance(flags, int)
        or isinstance(flags, bool)
        or flags < 0
        or flags & ~_UVM_MOE_STATIC_ALIAS_MAP_FLAGS_ALL
    ):
        validation_error = "MAP flags contain an unsupported bit"
    quiesced_append = bool(
        type(flags) is int
        and flags & UVM_MOE_STATIC_ALIAS_MAP_FLAG_QUIESCED_APPEND
    )
    defer_dual_refresh = bool(
        type(flags) is int
        and flags & UVM_MOE_STATIC_ALIAS_MAP_FLAG_DEFER_DUAL_REFRESH
    )
    if quiesced_append and session_id == 0:
        validation_error = "a quiesced append requires an existing session"
    if defer_dual_refresh and not quiesced_append:
        validation_error = "deferred dual refresh requires a quiesced append"
    if device_error:
        validation_error = device_error
    if validation_error:
        return _base_result(
            "map",
            reason="validation_failed",
            error=validation_error,
            session_id=int(session_id) if type(session_id) is int else session_id,
            wall_ns=time.perf_counter_ns() - started_ns,
        )
    capture = _capture_rejection()
    if capture is not None:
        reason, error = capture
        return _base_result(
            "map",
            reason=reason,
            error=error,
            session_id=int(session_id),
            wall_ns=time.perf_counter_ns() - started_ns,
        )
    gpu_uuid, uuid_bytes, uuid_error = _make_uuid(normalized_device)
    if gpu_uuid is None:
        return _base_result(
            "map",
            reason="gpu_uuid_failed",
            error=uuid_error,
            session_id=int(session_id),
            wall_ns=time.perf_counter_ns() - started_ns,
        )

    with _static_alias_lock:
        cached = _static_alias_sessions.get(normalized_device)
        if session_id != 0:
            if cached is None or cached.session_id != int(session_id):
                return _base_result(
                    "map",
                    reason="static_alias_session_not_found",
                    session_id=int(session_id),
                    wall_ns=time.perf_counter_ns() - started_ns,
                )
            blocked = _session_block_reason(cached)
            if blocked is not None:
                reason, error = blocked
                return _base_result(
                    "map",
                    reason=reason,
                    error=error,
                    session_id=int(session_id),
                    wall_ns=time.perf_counter_ns() - started_ns,
                )
            if cached.sealed and not quiesced_append:
                return _base_result(
                    "map",
                    reason="static_alias_session_sealed",
                    session_id=int(session_id),
                    wall_ns=time.perf_counter_ns() - started_ns,
                )
            if quiesced_append and (not cached.sealed or cached.parked):
                return _base_result(
                    "map",
                    reason="invalid_static_alias_transition_state",
                    error=(
                        "a quiesced append requires sealed=True and "
                        f"parked=False, got sealed={cached.sealed}, "
                        f"parked={cached.parked}"
                    ),
                    session_id=int(session_id),
                    wall_ns=time.perf_counter_ns() - started_ns,
                )
            if cached.uuid_bytes != uuid_bytes:
                return _base_result(
                    "map",
                    reason="cached_gpu_uuid_mismatch",
                    session_id=int(session_id),
                    wall_ns=time.perf_counter_ns() - started_ns,
                )
            # Do not rescan every cached target range here. MAP performs the
            # authoritative overlap and managed-VA ownership checks while it
            # holds the driver session lock. Repeating its O(new * existing)
            # check in Python made scheduler-quiesced appends needlessly slow.
            result = _submit_map(
                cached.fd,
                normalized,
                session_id=int(session_id),
                gpu_uuid=gpu_uuid,
                flags=int(flags),
                prior_alias_count=cached.alias_count,
                prior_alias_bytes=cached.alias_bytes,
            )
            if bool(result["ok"]) or bool(result.get("made_progress")):
                _sync_cached_map_result(cached, normalized, result)
            if not bool(result["ok"]) and (
                bool(result.get("made_progress")) or bool(result.get("poisoned"))
            ):
                cached.poisoned = True
            result["quiesced_append"] = quiesced_append
            result["dual_view_refresh_deferred"] = defer_dual_refresh
            result["wall_ns"] = time.perf_counter_ns() - started_ns
            return result

        if cached is not None:
            return _base_result(
                "map",
                reason="static_alias_session_already_active",
                session_id=cached.session_id,
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        sources = _iter_existing_uvm_fd_sources()
        # Slot backing is currently created through uvm.py's mode-8 command.
        # Prefer that command's retained file description so MAP targets the
        # exact same UVM VA space. Keep generic discovery as a fallback for
        # callers which arrange backing without the Python slot wrapper.
        legacy_cached = _uvm._uvm_slot_direct_sessions.get(normalized_device)
        if legacy_cached is not None:
            preferred_fd = int(legacy_cached[0])
            sources = [preferred_fd] + [fd for fd in sources if fd != preferred_fd]
        if not sources:
            return _base_result(
                "map",
                reason="no_existing_uvm_fd",
                session_id=0,
                wall_ns=time.perf_counter_ns() - started_ns,
            )

        last_result: dict[str, object] | None = None
        candidate_attempts = 0
        candidate_ioctl_calls = 0
        for source_fd in sources:
            fd = _initialize_duplicated_uvm_fd(source_fd)
            if fd is None:
                continue
            candidate_attempts += 1
            result = _submit_map(
                fd,
                normalized,
                session_id=0,
                gpu_uuid=gpu_uuid,
                flags=int(flags),
                prior_alias_count=0,
                prior_alias_bytes=0,
            )
            candidate_ioctl_calls += int(result.get("ioctl_calls", 0))
            if bool(result["ok"]) or bool(result.get("made_progress")):
                returned_session_id = int(result.get("session_id") or 0)
                cached = _StaticAliasSession(
                    fd=fd,
                    uuid_bytes=uuid_bytes,
                    session_id=returned_session_id,
                    owner_pid=os.getpid(),
                    poisoned=not bool(result["ok"]),
                )
                _sync_cached_map_result(cached, normalized, result)
                _static_alias_sessions[normalized_device] = cached
                _register_close_once()
                result["fd_candidate_attempts"] = candidate_attempts
                result["fd_candidate_ioctl_calls"] = candidate_ioctl_calls
                result["wall_ns"] = time.perf_counter_ns() - started_ns
                return result
            last_result = result
            with suppress(OSError):
                os.close(fd)

        if last_result is None:
            last_result = _base_result(
                "map",
                reason="no_initialized_uvm_fd",
                session_id=0,
            )
        else:
            last_result["reason"] = "no_matching_existing_uvm_fd"
        last_result["fd_candidate_attempts"] = candidate_attempts
        last_result["fd_candidate_ioctl_calls"] = candidate_ioctl_calls
        last_result["wall_ns"] = time.perf_counter_ns() - started_ns
        return last_result


def driver_static_alias_seal(
    session_id: int,
    *,
    expected_alias_count: int | None = None,
    expected_alias_bytes: int | None = None,
    device: int = 0,
    refresh_dual: bool = False,
    nested_refresh: bool = False,
    discard_dual: bool = False,
) -> dict[str, object]:
    """Seal a session, or refresh its frozen dual children at a safe point."""
    started_ns = time.perf_counter_ns()
    normalized_device, device_error = _normalize_device(device)
    if (
        device_error
        or not isinstance(session_id, int)
        or isinstance(session_id, bool)
        or session_id <= 0
        or session_id > _UINT64_MAX
        or not isinstance(refresh_dual, bool)
        or not isinstance(nested_refresh, bool)
        or not isinstance(discard_dual, bool)
        or (nested_refresh and not refresh_dual)
        or (discard_dual and (refresh_dual or nested_refresh))
    ):
        return _base_result(
            "seal",
            reason="validation_failed",
            error=device_error or "session_id must be a non-zero uint64",
            session_id=session_id,
            wall_ns=time.perf_counter_ns() - started_ns,
        )
    capture = _capture_rejection()
    if capture is not None:
        reason, error = capture
        return _base_result(
            "seal",
            reason=reason,
            error=error,
            session_id=int(session_id),
            wall_ns=time.perf_counter_ns() - started_ns,
        )

    with _static_alias_lock:
        cached = _static_alias_sessions.get(normalized_device)
        if cached is None or cached.session_id != int(session_id):
            return _base_result(
                "seal",
                reason="static_alias_session_not_found",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        blocked = _session_block_reason(cached)
        if blocked is not None:
            reason, error = blocked
            return _base_result(
                "seal",
                reason=reason,
                error=error,
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        if cached.sealed and not refresh_dual and not discard_dual:
            return _base_result(
                "seal",
                reason="static_alias_session_already_sealed",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        if (refresh_dual or discard_dual) and not cached.sealed:
            return _base_result(
                "seal",
                reason="static_alias_session_not_sealed",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        expected_count = (
            cached.alias_count
            if expected_alias_count is None
            else int(expected_alias_count)
        )
        expected_bytes = (
            cached.alias_bytes
            if expected_alias_bytes is None
            else int(expected_alias_bytes)
        )
        if expected_count != cached.alias_count or expected_bytes != cached.alias_bytes:
            return _base_result(
                "seal",
                reason="validation_failed",
                error="seal expectations disagree with the userspace session",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )

        params = _StaticAliasSealParams()
        params.abi_version = UVM_MOE_STATIC_ALIAS_ABI_VERSION
        params.struct_size = ctypes.sizeof(_StaticAliasSealParams)
        params.flags = 0
        if refresh_dual:
            params.flags |= UVM_MOE_STATIC_ALIAS_SEAL_FLAG_REFRESH_DUAL
        if nested_refresh:
            params.flags |= UVM_MOE_STATIC_ALIAS_SEAL_FLAG_REFRESH_DUAL_NESTED
        if discard_dual:
            params.flags |= UVM_MOE_STATIC_ALIAS_SEAL_FLAG_DISCARD_DUAL
        params.session_id = int(session_id)
        params.expected_alias_count = expected_count
        params.expected_alias_bytes = expected_bytes
        ctypes.set_errno(0)
        ioctl_started_ns = time.perf_counter_ns()
        try:
            ret = int(_libc_ioctl(cached.fd, _UVM_MOE_STATIC_ALIAS_SEAL_IOCTL, params))
        except Exception as exc:
            cached.poisoned = True
            return _base_result(
                "seal",
                reason="seal_ioctl_exception",
                error=repr(exc),
                session_id=int(session_id),
                poisoned=True,
                side_effect_indeterminate=True,
                ioctl_wall_ns=time.perf_counter_ns() - ioctl_started_ns,
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        ioctl_wall_ns = time.perf_counter_ns() - ioctl_started_ns
        errno_value = ctypes.get_errno() if ret != 0 else 0
        sealed_raw = int(params.sealed)
        poisoned_raw = int(params.poisoned)
        boolean_fields_ok = sealed_raw in (0, 1) and poisoned_raw in (0, 1)
        ok = (
            ret == 0
            and int(params.rmStatus) == 0
            and boolean_fields_ok
            and sealed_raw == 1
            and poisoned_raw == 0
            and int(params.alias_count) == expected_count
            and int(params.alias_bytes) == expected_bytes
            and int(params.backing_bytes) == cached.backing_bytes
        )
        contract_violation = ret == 0 and int(params.rmStatus) == 0 and not ok
        if ok:
            cached.sealed = True
        elif poisoned_raw == 1 or contract_violation:
            cached.poisoned = True
        return _base_result(
            "seal",
            ok=ok,
            reason=(
                "ok"
                if ok
                else "seal_contract_mismatch"
                if contract_violation
                else "seal_ioctl_failed"
            ),
            session_id=int(session_id),
            sealed=sealed_raw == 1,
            alias_count=int(params.alias_count),
            alias_bytes=int(params.alias_bytes),
            backing_bytes=int(params.backing_bytes),
            poisoned=cached.poisoned,
            kernel_poisoned=poisoned_raw == 1,
            boolean_fields_ok=boolean_fields_ok,
            contract_violation=contract_violation,
            ioctl_ret=ret,
            errno=errno_value,
            rm_status=int(params.rmStatus),
            ioctl_calls=1,
            ioctl_wall_ns=ioctl_wall_ns,
            wall_ns=time.perf_counter_ns() - started_ns,
            refresh_dual=bool(refresh_dual),
            nested_refresh=bool(nested_refresh),
            discard_dual=bool(discard_dual),
        )


def _driver_static_alias_transition(
    operation: str,
    session_id: int,
    *,
    device: int,
    synchronize: bool,
    sequential: bool | None,
    defer_cpu_restore: bool,
) -> dict[str, object]:
    """Park or resume the consumer PTEs of one sealed alias session."""
    started_ns = time.perf_counter_ns()
    normalized_device, device_error = _normalize_device(device)
    if (
        operation not in ("park", "resume")
        or device_error
        or not isinstance(session_id, int)
        or isinstance(session_id, bool)
        or session_id <= 0
        or session_id > _UINT64_MAX
        or not isinstance(synchronize, bool)
        or (sequential is not None and not isinstance(sequential, bool))
        or not isinstance(defer_cpu_restore, bool)
        or (defer_cpu_restore and operation != "park")
    ):
        return _base_result(
            operation,
            reason="validation_failed",
            error=device_error or "invalid static-alias transition argument",
            session_id=session_id,
            wall_ns=time.perf_counter_ns() - started_ns,
        )
    capture = _capture_rejection()
    if capture is not None:
        reason, error = capture
        return _base_result(
            operation,
            reason=reason,
            error=error,
            session_id=int(session_id),
            wall_ns=time.perf_counter_ns() - started_ns,
        )
    if synchronize:
        try:
            torch.cuda.synchronize(normalized_device)
        except Exception as exc:
            return _base_result(
                operation,
                reason="cuda_synchronize_failed",
                error=repr(exc),
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )

    with _static_alias_lock:
        cached = _static_alias_sessions.get(normalized_device)
        if cached is None or cached.session_id != int(session_id):
            return _base_result(
                operation,
                reason="static_alias_session_not_found",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        blocked = _session_block_reason(cached)
        if blocked is not None:
            reason, error = blocked
            return _base_result(
                operation,
                reason=reason,
                error=error,
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        expected_pre_parked = operation == "resume"
        expected_post_parked = operation == "park"
        if not cached.sealed or cached.parked != expected_pre_parked:
            return _base_result(
                operation,
                reason="invalid_static_alias_transition_state",
                error=(
                    f"sealed={cached.sealed}, parked={cached.parked}, "
                    f"operation={operation}"
                ),
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )

        if sequential is None:
            transition_mode = os.environ.get(
                "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_TRANSITION_MODE",
                "grouped",
            ).strip().lower()
            if transition_mode not in ("grouped", "sequential"):
                return _base_result(
                    operation,
                    reason="validation_failed",
                    error=(
                        "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_TRANSITION_MODE "
                        f"must be grouped or sequential, got {transition_mode!r}"
                    ),
                    session_id=int(session_id),
                    wall_ns=time.perf_counter_ns() - started_ns,
                )
            sequential = transition_mode == "sequential"

        params = _StaticAliasTransitionParams()
        params.abi_version = UVM_MOE_STATIC_ALIAS_ABI_VERSION
        params.struct_size = ctypes.sizeof(_StaticAliasTransitionParams)
        params.flags = (
            UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_FORCE_SEQUENTIAL
            if sequential
            else 0
        )
        if defer_cpu_restore:
            params.flags |= (
                UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_DEFER_CPU_RESTORE
            )
        params.session_id = int(session_id)
        ioctl_command = (
            _UVM_MOE_STATIC_ALIAS_PARK_IOCTL
            if operation == "park"
            else _UVM_MOE_STATIC_ALIAS_RESUME_IOCTL
        )
        ctypes.set_errno(0)
        ioctl_started_ns = time.perf_counter_ns()
        try:
            ret = int(_libc_ioctl(cached.fd, ioctl_command, params))
        except Exception as exc:
            cached.poisoned = True
            return _base_result(
                operation,
                reason=f"{operation}_ioctl_exception",
                error=repr(exc),
                session_id=int(session_id),
                poisoned=True,
                side_effect_indeterminate=True,
                ioctl_wall_ns=time.perf_counter_ns() - ioctl_started_ns,
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        ioctl_wall_ns = time.perf_counter_ns() - ioctl_started_ns
        errno_value = ctypes.get_errno() if ret != 0 else 0
        parked_raw = int(params.parked)
        sealed_raw = int(params.sealed)
        poisoned_raw = int(params.poisoned)
        boolean_fields_ok = all(
            value in (0, 1)
            for value in (
                parked_raw,
                sealed_raw,
                poisoned_raw,
                int(params.dual_view_full_coverage),
            )
        )
        processed_count = int(params.processed_alias_count)
        consumer_count = int(params.consumer_alias_count)
        upload_count = int(params.upload_alias_count)
        processed_bytes = int(params.processed_alias_bytes)
        consumer_bytes = int(params.consumer_alias_bytes)
        transition_batch_count = int(params.reserved)
        batch_count_valid = (
            transition_batch_count == 0
            or 0 < transition_batch_count <= processed_count
        )
        ok = (
            ret == 0
            and int(params.rmStatus) == 0
            and boolean_fields_ok
            and sealed_raw == 1
            and poisoned_raw == 0
            and bool(parked_raw) is expected_post_parked
            and consumer_count > 0
            and consumer_bytes > 0
            and upload_count > 0
            and consumer_count + upload_count == cached.alias_count
            and processed_count == consumer_count
            and processed_bytes == consumer_bytes
            and int(params.backing_bytes) == cached.backing_bytes
            and batch_count_valid
        )
        contract_violation = ret == 0 and int(params.rmStatus) == 0 and not ok
        if ok:
            cached.parked = expected_post_parked
        elif poisoned_raw == 1 or contract_violation or processed_count != 0:
            cached.poisoned = True
        return _base_result(
            operation,
            ok=ok,
            reason=(
                "ok"
                if ok
                else f"{operation}_contract_mismatch"
                if contract_violation
                else f"{operation}_ioctl_failed"
            ),
            session_id=int(session_id),
            processed_alias_count=processed_count,
            consumer_alias_count=consumer_count,
            upload_alias_count=upload_count,
            processed_alias_bytes=processed_bytes,
            consumer_alias_bytes=consumer_bytes,
            backing_bytes=int(params.backing_bytes),
            tlb_invalidate_count=int(params.tlb_invalidate_count),
            transition_batch_count=transition_batch_count,
            transition_mode="sequential" if sequential else "grouped",
            sequential_requested=bool(sequential),
            batch_count_reported=transition_batch_count > 0,
            batch_count_valid=batch_count_valid,
            parked=parked_raw == 1,
            sealed=sealed_raw == 1,
            poisoned=cached.poisoned,
            kernel_poisoned=poisoned_raw == 1,
            boolean_fields_ok=boolean_fields_ok,
            contract_violation=contract_violation,
            ioctl_ret=ret,
            errno=errno_value,
            rm_status=int(params.rmStatus),
            validation_ns=int(params.validation_ns),
            build_sort_ns=int(params.build_sort_ns),
            preflight_ns=int(params.preflight_ns),
            metadata_ns=int(params.metadata_ns),
            cpu_unmap_ns=int(params.cpu_unmap_ns),
            pte_map_ns=int(params.pte_map_ns),
            wait_ns=int(params.wait_ns),
            bookkeeping_ns=int(params.bookkeeping_ns),
            dual_view_full_coverage=(
                int(params.dual_view_full_coverage) == 1
            ),
            dual_view_block_count=int(params.dual_view_block_count),
            ioctl_calls=1,
            ioctl_wall_ns=ioctl_wall_ns,
            wall_ns=time.perf_counter_ns() - started_ns,
        )


def driver_static_alias_park(
    session_id: int,
    *,
    device: int = 0,
    synchronize: bool = False,
    sequential: bool | None = None,
    defer_cpu_restore: bool = False,
) -> dict[str, object]:
    """Detach read-only consumers while retaining upload aliases/backing."""
    return _driver_static_alias_transition(
        "park",
        session_id,
        device=device,
        synchronize=synchronize,
        sequential=sequential,
        defer_cpu_restore=defer_cpu_restore,
    )


def driver_static_alias_resume(
    session_id: int,
    *,
    device: int = 0,
    synchronize: bool = False,
    sequential: bool | None = None,
) -> dict[str, object]:
    """Restore parked consumers to their retained GPU slot backing."""
    return _driver_static_alias_transition(
        "resume",
        session_id,
        device=device,
        synchronize=synchronize,
        sequential=sequential,
        defer_cpu_restore=False,
    )


def _fill_rebind_params(
    normalized: Sequence[tuple[int, int, int, int, int, int, int]],
    *,
    session_id: int,
    rebind_id: int,
    flags: int,
    expected_count: int,
    expected_bytes: int,
) -> _StaticAliasRebindParams:
    params = _StaticAliasRebindParams()
    params.abi_version = UVM_MOE_STATIC_ALIAS_ABI_VERSION
    params.struct_size = ctypes.sizeof(_StaticAliasRebindParams)
    params.flags = int(flags)
    params.session_id = int(session_id)
    params.rebind_id = int(rebind_id)
    params.expected_consumer_alias_count = int(expected_count)
    params.expected_consumer_alias_bytes = int(expected_bytes)
    params.count = len(normalized)
    for index, record in enumerate(normalized):
        entry = params.entries[index]
        (
            entry.backing_layer_id,
            entry.backing_expert_id,
            entry.backing_slot_id,
            entry.flags,
            entry.va_base,
            entry.bytes,
            entry.slot_offset,
        ) = record
    return params


def _abort_static_alias_rebind(
    cached: _StaticAliasSession,
    *,
    rebind_id: int,
) -> dict[str, object]:
    params = _fill_rebind_params(
        (),
        session_id=cached.session_id,
        rebind_id=rebind_id,
        flags=UVM_MOE_STATIC_ALIAS_REBIND_FLAG_ABORT,
        expected_count=0,
        expected_bytes=0,
    )
    ctypes.set_errno(0)
    started_ns = time.perf_counter_ns()
    try:
        ret = int(_libc_ioctl(cached.fd, _UVM_MOE_STATIC_ALIAS_REBIND_IOCTL, params))
    except Exception as exc:
        return _base_result(
            "rebind_abort",
            reason="rebind_abort_ioctl_exception",
            error=repr(exc),
            session_id=cached.session_id,
            rebind_id=rebind_id,
            side_effect_indeterminate=True,
            ioctl_wall_ns=time.perf_counter_ns() - started_ns,
        )
    return _base_result(
        "rebind_abort",
        ok=ret == 0 and int(params.rmStatus) == 0 and int(params.rebind_id) == 0,
        reason=(
            "ok"
            if ret == 0 and int(params.rmStatus) == 0 and int(params.rebind_id) == 0
            else "rebind_abort_ioctl_failed"
        ),
        session_id=cached.session_id,
        rebind_id=int(params.rebind_id),
        ioctl_ret=ret,
        errno=ctypes.get_errno() if ret != 0 else 0,
        rm_status=int(params.rmStatus),
        ioctl_wall_ns=time.perf_counter_ns() - started_ns,
    )


def driver_static_alias_rebind(
    session_id: int,
    consumer_entries: Sequence[Sequence[int]],
    *,
    device: int = 0,
    synchronize: bool = False,
    resize: bool = False,
    expected_backing_bytes: int | None = None,
) -> dict[str, object]:
    """Atomically replace aliases in one parked session.

    The wrapper streams large full-model topologies through the transaction
    UAPI.  The default form replaces only read-only consumers and retains all
    backing.  ``resize=True`` submits the complete upload + consumer topology;
    the driver keeps common backing identities, adopts preallocated additions,
    and releases removed identities without changing the session id.

    A failed transaction is aborted and never changes the committed alias set
    unless the kernel explicitly reports a poisoned, partially mutated resize.
    Ambiguous ioctl failures poison the userspace handle so a later RESUME
    cannot accidentally publish an unknown candidate.
    """
    started_ns = time.perf_counter_ns()
    normalized_device, device_error = _normalize_device(device)
    normalized, validation_error = _normalize_entries(
        consumer_entries,
        maximum_entries=_UVM_MOE_STATIC_ALIAS_MAX_REBIND_CONSUMERS,
    )
    if (
        device_error
        or not isinstance(session_id, int)
        or isinstance(session_id, bool)
        or session_id <= 0
        or session_id > _UINT64_MAX
        or not isinstance(synchronize, bool)
        or not isinstance(resize, bool)
        or (
            expected_backing_bytes is not None
            and (
                not isinstance(expected_backing_bytes, int)
                or isinstance(expected_backing_bytes, bool)
                or expected_backing_bytes <= 0
                or expected_backing_bytes > _UINT64_MAX
            )
        )
    ):
        validation_error = device_error or "invalid REBIND argument"
    if not resize and any(
        record[3] & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW for record in normalized
    ):
        validation_error = "REBIND accepts read-only consumer entries only"
    if resize:
        target_uploads = [
            record
            for record in normalized
            if record[3] & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW
        ]
        target_consumers = [
            record
            for record in normalized
            if not (record[3] & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW)
        ]
        if not target_uploads or not target_consumers:
            validation_error = (
                "A resizing REBIND requires complete upload and consumer entries"
            )
        upload_identities = [record[:3] for record in target_uploads]
        if len(upload_identities) != len(set(upload_identities)):
            validation_error = (
                "A resizing REBIND requires exactly one upload per backing identity"
            )
    if validation_error:
        return _base_result(
            "rebind",
            reason="validation_failed",
            error=validation_error,
            session_id=session_id,
            wall_ns=time.perf_counter_ns() - started_ns,
        )
    capture = _capture_rejection()
    if capture is not None:
        reason, error = capture
        return _base_result(
            "rebind",
            reason=reason,
            error=error,
            session_id=int(session_id),
            wall_ns=time.perf_counter_ns() - started_ns,
        )
    if synchronize:
        try:
            torch.cuda.synchronize(normalized_device)
        except Exception as exc:
            return _base_result(
                "rebind",
                reason="cuda_synchronize_failed",
                error=repr(exc),
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )

    with _static_alias_lock:
        cached = _static_alias_sessions.get(normalized_device)
        if cached is None or cached.session_id != int(session_id):
            return _base_result(
                "rebind",
                reason="static_alias_session_not_found",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        blocked = _session_block_reason(cached)
        if blocked is not None:
            reason, error = blocked
            return _base_result(
                "rebind",
                reason=reason,
                error=error,
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        if not cached.sealed or not cached.parked:
            return _base_result(
                "rebind",
                reason="invalid_static_alias_transition_state",
                error=f"sealed={cached.sealed}, parked={cached.parked}",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )

        retained_upload_entries = [
            record
            for record in cached.alias_entries
            if record[3] & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW
        ]
        if not retained_upload_entries:
            return _base_result(
                "rebind",
                reason="cached_upload_aliases_missing",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        upload_entries = (
            [
                record
                for record in normalized
                if record[3] & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW
            ]
            if resize
            else retained_upload_entries
        )
        replacement_consumers = (
            [
                record
                for record in normalized
                if not (record[3] & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW)
            ]
            if resize
            else normalized
        )
        upload_keys = {
            (
                record[0],
                record[1],
                record[2],
                record[5],
                record[6],
                bool(record[3] & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K),
            )
            for record in upload_entries
        }
        unmatched = [
            index
            for index, record in enumerate(replacement_consumers)
            if (
                record[0],
                record[1],
                record[2],
                record[5],
                record[6],
                bool(record[3] & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K),
            )
            not in upload_keys
        ]
        if unmatched:
            return _base_result(
                "rebind",
                reason="consumer_backing_geometry_mismatch",
                error=f"entries without retained upload backing: {unmatched[:8]!r}",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )

        transaction_entries = normalized
        expected_count = len(transaction_entries)
        expected_bytes = sum(record[5] for record in transaction_entries)
        upload_bytes = sum(record[5] for record in upload_entries)
        expected_total_alias_bytes = (
            expected_bytes if resize else upload_bytes + expected_bytes
        )
        target_backing_bytes = sum(record[5] for record in upload_entries)
        if (
            expected_backing_bytes is not None
            and target_backing_bytes != int(expected_backing_bytes)
        ):
            return _base_result(
                "rebind",
                reason="validation_failed",
                error=(
                    "resizing REBIND backing expectation disagrees with the "
                    f"target uploads: expected={expected_backing_bytes}, "
                    f"target={target_backing_bytes}"
                ),
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        old_upload_count = len(retained_upload_entries)
        old_backing_bytes = int(cached.backing_bytes)
        rebind_id = 0
        accepted_total = 0
        accepted_bytes = 0
        batch_results: list[dict[str, object]] = []
        ioctl_wall_ns = 0
        failure: dict[str, object] | None = None
        for offset in range(0, expected_count, UVM_MOE_STATIC_ALIAS_MAX_ENTRIES):
            chunk = transaction_entries[
                offset : offset + UVM_MOE_STATIC_ALIAS_MAX_ENTRIES
            ]
            final = offset + len(chunk) == expected_count
            flags = 0
            if offset == 0:
                flags |= UVM_MOE_STATIC_ALIAS_REBIND_FLAG_BEGIN
            if final:
                flags |= UVM_MOE_STATIC_ALIAS_REBIND_FLAG_COMMIT
            if resize:
                flags |= UVM_MOE_STATIC_ALIAS_REBIND_FLAG_RESIZE
            params = _fill_rebind_params(
                chunk,
                session_id=int(session_id),
                rebind_id=rebind_id,
                flags=flags,
                expected_count=expected_count,
                expected_bytes=expected_bytes,
            )
            ctypes.set_errno(0)
            ioctl_started_ns = time.perf_counter_ns()
            try:
                ret = int(
                    _libc_ioctl(
                        cached.fd,
                        _UVM_MOE_STATIC_ALIAS_REBIND_IOCTL,
                        params,
                    )
                )
            except Exception as exc:
                ioctl_wall_ns += time.perf_counter_ns() - ioctl_started_ns
                failure = {
                    "reason": "rebind_ioctl_exception",
                    "error": repr(exc),
                    "side_effect_indeterminate": True,
                }
                break
            one_wall_ns = time.perf_counter_ns() - ioctl_started_ns
            ioctl_wall_ns += one_wall_ns
            statuses = [int(params.entries[index].status) for index in range(len(chunk))]
            committed_raw = int(params.committed)
            poisoned_raw = int(params.poisoned)
            returned_rebind_id = int(params.rebind_id)
            expected_staged_count = 0 if final else accepted_total + len(chunk)
            expected_staged_bytes = 0 if final else accepted_bytes + sum(
                record[5] for record in chunk
            )
            expected_upload_count = (
                len(upload_entries) if resize and final else old_upload_count
            )
            expected_kernel_backing_bytes = (
                target_backing_bytes if resize and final else old_backing_bytes
            )
            contract_ok = (
                ret == 0
                and int(params.rmStatus) == 0
                and committed_raw in (0, 1)
                and poisoned_raw in (0, 1)
                and poisoned_raw == 0
                and int(params.accepted_count) == len(chunk)
                and statuses == [0] * len(chunk)
                and int(params.staged_consumer_alias_count)
                == expected_staged_count
                and int(params.staged_consumer_alias_bytes)
                == expected_staged_bytes
                and int(params.upload_alias_count) == expected_upload_count
                and int(params.backing_bytes) == expected_kernel_backing_bytes
                and committed_raw == int(final)
                and (returned_rebind_id == 0 if final else returned_rebind_id > 0)
                and (
                    not final
                    or int(params.total_alias_bytes) == expected_total_alias_bytes
                )
            )
            batch_result = {
                "batch_index": len(batch_results),
                "offset": offset,
                "count": len(chunk),
                "flags": flags,
                "rebind_id": returned_rebind_id,
                "accepted_count": int(params.accepted_count),
                "staged_consumer_alias_count": int(
                    params.staged_consumer_alias_count
                ),
                "staged_consumer_alias_bytes": int(
                    params.staged_consumer_alias_bytes
                ),
                "committed": committed_raw == 1,
                "poisoned": poisoned_raw == 1,
                "entry_statuses": statuses,
                "ioctl_ret": ret,
                "errno": ctypes.get_errno() if ret != 0 else 0,
                "rm_status": int(params.rmStatus),
                "contract_ok": contract_ok,
                "ioctl_wall_ns": one_wall_ns,
            }
            batch_results.append(batch_result)
            if not contract_ok:
                failure = {
                    "reason": (
                        "rebind_contract_mismatch"
                        if ret == 0 and int(params.rmStatus) == 0
                        else "rebind_ioctl_failed"
                    ),
                    "kernel_poisoned": poisoned_raw == 1,
                    "contract_violation": ret == 0 and int(params.rmStatus) == 0,
                }
                rebind_id = returned_rebind_id
                break
            accepted_total += len(chunk)
            accepted_bytes += sum(record[5] for record in chunk)
            rebind_id = returned_rebind_id

        if failure is not None:
            abort_result = None
            if rebind_id > 0:
                abort_result = _abort_static_alias_rebind(
                    cached, rebind_id=rebind_id
                )
            ambiguous = bool(failure.get("side_effect_indeterminate")) or bool(
                failure.get("contract_violation")
            )
            if ambiguous or bool(failure.get("kernel_poisoned")) or (
                abort_result is not None and not bool(abort_result.get("ok"))
            ):
                cached.poisoned = True
            return _base_result(
                "rebind",
                reason=str(failure["reason"]),
                error=failure.get("error"),
                session_id=int(session_id),
                expected_consumer_alias_count=expected_count,
                expected_consumer_alias_bytes=expected_bytes,
                accepted_consumer_alias_count=accepted_total,
                accepted_consumer_alias_bytes=accepted_bytes,
                batch_results=batch_results,
                abort_result=abort_result,
                poisoned=cached.poisoned,
                side_effect_indeterminate=bool(
                    failure.get("side_effect_indeterminate")
                ),
                ioctl_calls=len(batch_results),
                ioctl_wall_ns=ioctl_wall_ns,
                wall_ns=time.perf_counter_ns() - started_ns,
            )

        old_backing_identities = set(cached.backing_identities)
        cached.alias_entries = (
            list(transaction_entries)
            if resize
            else [*upload_entries, *replacement_consumers]
        )
        cached.alias_count = len(cached.alias_entries)
        cached.alias_bytes = expected_total_alias_bytes
        cached.backing_bytes = target_backing_bytes
        cached.target_ranges = [
            (record[4], record[4] + record[5]) for record in cached.alias_entries
        ]
        cached.backing_identities = {
            (record[0], record[1], record[2]) for record in cached.alias_entries
        }
        if resize:
            legacy_owners = _uvm._uvm_slot_direct_owners.get(normalized_device)
            if legacy_owners is not None:
                for layer_id, expert_id, slot_id in (
                    old_backing_identities - cached.backing_identities
                ):
                    key = (layer_id, slot_id)
                    if legacy_owners.get(key) == expert_id:
                        legacy_owners.pop(key, None)
        return _base_result(
            "rebind",
            ok=True,
            reason="ok",
            session_id=int(session_id),
            consumer_alias_count=expected_count,
            consumer_alias_bytes=expected_bytes,
            upload_alias_count=len(upload_entries),
            upload_alias_bytes=upload_bytes,
            alias_count=cached.alias_count,
            alias_bytes=cached.alias_bytes,
            backing_bytes=cached.backing_bytes,
            resized=resize,
            retained_backing_count=len(
                old_backing_identities & cached.backing_identities
            ),
            added_backing_count=len(
                cached.backing_identities - old_backing_identities
            ),
            removed_backing_count=len(
                old_backing_identities - cached.backing_identities
            ),
            batch_count=len(batch_results),
            batch_results=batch_results,
            committed=True,
            poisoned=False,
            ioctl_calls=len(batch_results),
            ioctl_wall_ns=ioctl_wall_ns,
            wall_ns=time.perf_counter_ns() - started_ns,
        )


def _fill_quiesced_remove_params(
    normalized: Sequence[tuple[int, int, int, int, int, int, int]],
    *,
    session_id: int,
    gpu_uuid: _StaticAliasUuid,
) -> _StaticAliasQuiescedRemoveParams:
    params = _StaticAliasQuiescedRemoveParams()
    params.abi_version = UVM_MOE_STATIC_ALIAS_ABI_VERSION
    params.struct_size = ctypes.sizeof(_StaticAliasQuiescedRemoveParams)
    params.flags = UVM_MOE_STATIC_ALIAS_QUIESCED_REMOVE_FLAG_RELEASE_BACKING
    params.session_id = int(session_id)
    params.gpu_uuid = gpu_uuid
    params.count = len(normalized)
    for index, record in enumerate(normalized):
        (
            backing_layer_id,
            backing_expert_id,
            backing_slot_id,
            entry_flags,
            va_base,
            length,
            slot_offset,
        ) = record
        entry = params.entries[index]
        entry.backing_layer_id = backing_layer_id
        entry.backing_expert_id = backing_expert_id
        entry.backing_slot_id = backing_slot_id
        entry.flags = entry_flags
        entry.va_base = va_base
        entry.bytes = length
        entry.slot_offset = slot_offset
    return params


def driver_static_alias_quiesced_remove(
    session_id: int,
    entries: Sequence[Sequence[int]],
    *,
    device: int = 0,
    synchronize: bool = False,
) -> dict[str, object]:
    """Remove complete backing identities at a scheduler quiescence point."""
    started_ns = time.perf_counter_ns()
    normalized_device, device_error = _normalize_device(device)
    normalized, validation_error = _normalize_entries(entries)
    if (
        device_error
        or not isinstance(session_id, int)
        or isinstance(session_id, bool)
        or session_id <= 0
        or session_id > _UINT64_MAX
        or not isinstance(synchronize, bool)
    ):
        validation_error = device_error or "invalid quiesced-remove argument"
    if validation_error:
        return _base_result(
            "quiesced_remove",
            reason="validation_failed",
            error=validation_error,
            session_id=session_id,
            wall_ns=time.perf_counter_ns() - started_ns,
        )
    capture = _capture_rejection()
    if capture is not None:
        reason, error = capture
        return _base_result(
            "quiesced_remove",
            reason=reason,
            error=error,
            session_id=int(session_id),
            wall_ns=time.perf_counter_ns() - started_ns,
        )
    if synchronize:
        try:
            torch.cuda.synchronize(normalized_device)
        except Exception as exc:
            return _base_result(
                "quiesced_remove",
                reason="cuda_synchronize_failed",
                error=repr(exc),
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
    gpu_uuid, uuid_bytes, uuid_error = _make_uuid(normalized_device)
    if gpu_uuid is None:
        return _base_result(
            "quiesced_remove",
            reason="gpu_uuid_failed",
            error=uuid_error,
            session_id=int(session_id),
            wall_ns=time.perf_counter_ns() - started_ns,
        )

    with _static_alias_lock:
        cached = _static_alias_sessions.get(normalized_device)
        if cached is None or cached.session_id != int(session_id):
            return _base_result(
                "quiesced_remove",
                reason="static_alias_session_not_found",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        blocked = _session_block_reason(cached)
        if blocked is not None:
            reason, error = blocked
            return _base_result(
                "quiesced_remove",
                reason=reason,
                error=error,
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        if not cached.sealed:
            return _base_result(
                "quiesced_remove",
                reason="invalid_static_alias_transition_state",
                error=(
                    "quiesced remove requires sealed=True, "
                    f"got sealed={cached.sealed}, parked={cached.parked}"
                ),
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        if cached.uuid_bytes != uuid_bytes:
            return _base_result(
                "quiesced_remove",
                reason="cached_gpu_uuid_mismatch",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )

        requested_set = set(normalized)
        cached_set = set(cached.alias_entries)
        if len(requested_set) != len(normalized) or not requested_set <= cached_set:
            return _base_result(
                "quiesced_remove",
                reason="validation_failed",
                error="remove entries must be unique exact records in the session",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        identities = {(record[0], record[1], record[2]) for record in normalized}
        complete_records = {
            record
            for record in cached.alias_entries
            if (record[0], record[1], record[2]) in identities
        }
        if requested_set != complete_records:
            return _base_result(
                "quiesced_remove",
                reason="validation_failed",
                error="every alias of each removed backing identity is required",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        removed_uploads = [
            record
            for record in normalized
            if record[3] & UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW
        ]
        if len(removed_uploads) != len(identities):
            return _base_result(
                "quiesced_remove",
                reason="validation_failed",
                error="each removed backing identity requires exactly one upload alias",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )

        requested_bytes = sum(record[5] for record in normalized)
        removed_backing_bytes = sum(record[5] for record in removed_uploads)
        expected_alias_count = cached.alias_count - len(normalized)
        expected_alias_bytes = cached.alias_bytes - requested_bytes
        expected_backing_bytes = cached.backing_bytes - removed_backing_bytes
        if min(expected_alias_count, expected_alias_bytes, expected_backing_bytes) < 0:
            return _base_result(
                "quiesced_remove",
                reason="validation_failed",
                error="remove delta exceeds the cached session totals",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )

        params = _fill_quiesced_remove_params(
            normalized,
            session_id=int(session_id),
            gpu_uuid=gpu_uuid,
        )
        ctypes.set_errno(0)
        ioctl_started_ns = time.perf_counter_ns()
        try:
            ret = int(
                _libc_ioctl(
                    cached.fd,
                    _UVM_MOE_STATIC_ALIAS_QUIESCED_REMOVE_IOCTL,
                    params,
                )
            )
        except Exception as exc:
            cached.poisoned = True
            return _base_result(
                "quiesced_remove",
                reason="quiesced_remove_ioctl_exception",
                error=repr(exc),
                session_id=int(session_id),
                poisoned=True,
                side_effect_indeterminate=True,
                ioctl_wall_ns=time.perf_counter_ns() - ioctl_started_ns,
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        ioctl_wall_ns = time.perf_counter_ns() - ioctl_started_ns
        statuses = [int(params.entries[index].status) for index in range(len(normalized))]
        poisoned_raw = int(params.poisoned)
        boolean_fields_ok = poisoned_raw in (0, 1)
        ok = (
            ret == 0
            and int(params.rmStatus) == 0
            and boolean_fields_ok
            and poisoned_raw == 0
            and int(params.requested_bytes) == requested_bytes
            and int(params.released_alias_bytes) == requested_bytes
            and int(params.total_alias_bytes) == expected_alias_bytes
            and int(params.backing_bytes) == expected_backing_bytes
            and int(params.success_count) == len(normalized)
            and int(params.failure_count) == 0
            and int(params.remaining_alias_count) == expected_alias_count
            and statuses == [0] * len(normalized)
        )
        contract_violation = ret == 0 and int(params.rmStatus) == 0 and not ok
        if ok:
            cached.alias_entries = [
                record for record in cached.alias_entries if record not in requested_set
            ]
            cached.alias_count = expected_alias_count
            cached.alias_bytes = expected_alias_bytes
            cached.backing_bytes = expected_backing_bytes
            cached.target_ranges = [
                (record[4], record[4] + record[5])
                for record in cached.alias_entries
            ]
            cached.backing_identities = {
                (record[0], record[1], record[2])
                for record in cached.alias_entries
            }
            legacy_owners = _uvm._uvm_slot_direct_owners.get(normalized_device)
            if legacy_owners is not None:
                for layer_id, expert_id, slot_id in identities:
                    key = (layer_id, slot_id)
                    if legacy_owners.get(key) == expert_id:
                        legacy_owners.pop(key, None)
        elif poisoned_raw == 1 or contract_violation or int(params.success_count) != 0:
            cached.poisoned = True
        return _base_result(
            "quiesced_remove",
            ok=ok,
            reason=(
                "ok"
                if ok
                else "quiesced_remove_contract_mismatch"
                if contract_violation
                else "quiesced_remove_ioctl_failed"
            ),
            session_id=int(session_id),
            requested_count=len(normalized),
            requested_bytes=int(params.requested_bytes),
            released_alias_bytes=int(params.released_alias_bytes),
            alias_count=int(params.remaining_alias_count),
            alias_bytes=int(params.total_alias_bytes),
            backing_bytes=int(params.backing_bytes),
            released_backing_bytes=max(
                0, cached.backing_bytes - int(params.backing_bytes)
            )
            if not ok
            else removed_backing_bytes,
            success_count=int(params.success_count),
            failure_count=int(params.failure_count),
            tlb_invalidate_count=int(params.tlb_invalidate_count),
            entry_statuses=statuses,
            poisoned=cached.poisoned,
            kernel_poisoned=poisoned_raw == 1,
            boolean_fields_ok=boolean_fields_ok,
            contract_violation=contract_violation,
            side_effect_indeterminate=False,
            ioctl_ret=ret,
            errno=ctypes.get_errno() if ret != 0 else 0,
            rm_status=int(params.rmStatus),
            ioctl_calls=1,
            ioctl_wall_ns=ioctl_wall_ns,
            wall_ns=time.perf_counter_ns() - started_ns,
        )


def driver_static_alias_destroy(
    session_id: int,
    *,
    device: int = 0,
    release_backing: bool = True,
    defer_cpu_restore: bool = False,
    synchronize: bool = False,
) -> dict[str, object]:
    """Destroy aliases after Graph teardown and an optional device barrier."""
    started_ns = time.perf_counter_ns()
    normalized_device, device_error = _normalize_device(device)
    if (
        device_error
        or not isinstance(session_id, int)
        or isinstance(session_id, bool)
        or session_id <= 0
        or session_id > _UINT64_MAX
        or not isinstance(release_backing, bool)
        or not isinstance(defer_cpu_restore, bool)
        or (defer_cpu_restore and not release_backing)
        or not isinstance(synchronize, bool)
    ):
        return _base_result(
            "destroy",
            reason="validation_failed",
            error=device_error or "invalid destroy argument",
            session_id=session_id,
            wall_ns=time.perf_counter_ns() - started_ns,
        )
    capture = _capture_rejection()
    if capture is not None:
        reason, error = capture
        return _base_result(
            "destroy",
            reason=reason,
            error=error,
            session_id=int(session_id),
            wall_ns=time.perf_counter_ns() - started_ns,
        )
    if synchronize:
        try:
            torch.cuda.synchronize(normalized_device)
        except Exception as exc:
            return _base_result(
                "destroy",
                reason="cuda_synchronize_failed",
                error=repr(exc),
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )

    with _static_alias_lock:
        cached = _static_alias_sessions.get(normalized_device)
        if cached is None or cached.session_id != int(session_id):
            return _base_result(
                "destroy",
                reason="static_alias_session_not_found",
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        blocked = _session_block_reason(cached, allow_poisoned=True)
        if blocked is not None:
            reason, error = blocked
            return _base_result(
                "destroy",
                reason=reason,
                error=error,
                session_id=int(session_id),
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        params = _StaticAliasDestroyParams()
        params.abi_version = UVM_MOE_STATIC_ALIAS_ABI_VERSION
        params.struct_size = ctypes.sizeof(_StaticAliasDestroyParams)
        params.flags = 0
        if release_backing:
            params.flags |= UVM_MOE_STATIC_ALIAS_DESTROY_FLAG_RELEASE_BACKING
        if defer_cpu_restore:
            params.flags |= UVM_MOE_STATIC_ALIAS_DESTROY_FLAG_DEFER_CPU_RESTORE
        params.session_id = int(session_id)
        ctypes.set_errno(0)
        ioctl_started_ns = time.perf_counter_ns()
        try:
            ret = int(
                _libc_ioctl(cached.fd, _UVM_MOE_STATIC_ALIAS_DESTROY_IOCTL, params)
            )
        except Exception as exc:
            cached.poisoned = True
            return _base_result(
                "destroy",
                reason="destroy_ioctl_exception",
                error=repr(exc),
                session_id=int(session_id),
                poisoned=True,
                side_effect_indeterminate=True,
                ioctl_wall_ns=time.perf_counter_ns() - ioctl_started_ns,
                wall_ns=time.perf_counter_ns() - started_ns,
            )
        ioctl_wall_ns = time.perf_counter_ns() - ioctl_started_ns
        errno_value = ctypes.get_errno() if ret != 0 else 0
        cleared_raw = int(params.session_cleared)
        poisoned_raw = int(params.poisoned)
        boolean_fields_ok = cleared_raw in (0, 1) and poisoned_raw in (0, 1)
        aliases_removed = (
            int(params.released_alias_count) == cached.alias_count
            and int(params.released_alias_bytes) == cached.alias_bytes
            and int(params.remaining_alias_count) == 0
            and int(params.remaining_alias_bytes) == 0
        )
        if release_backing:
            backing_contract_ok = (
                int(params.released_backing_bytes) == cached.backing_bytes
                and int(params.remaining_backing_bytes) == 0
                and cleared_raw == 1
            )
        else:
            backing_contract_ok = (
                int(params.released_backing_bytes) == 0
                and int(params.remaining_backing_bytes) == cached.backing_bytes
                and cleared_raw == 0
            )
        kernel_session_cleared = (
            ret == 0
            and int(params.rmStatus) == 0
            and cleared_raw == 1
            and int(params.remaining_alias_count) == 0
            and int(params.remaining_alias_bytes) == 0
            and int(params.remaining_backing_bytes) == 0
        )
        ok = (
            ret == 0
            and int(params.rmStatus) == 0
            and boolean_fields_ok
            and poisoned_raw == 0
            and aliases_removed
            and backing_contract_ok
        )
        contract_violation = ret == 0 and int(params.rmStatus) == 0 and not ok
        result = _base_result(
            "destroy",
            ok=ok,
            reason=(
                "ok"
                if ok
                else "destroy_contract_mismatch"
                if contract_violation
                else "destroy_ioctl_failed"
            ),
            session_id=int(session_id),
            release_backing=release_backing,
            defer_cpu_restore=defer_cpu_restore,
            released_alias_count=int(params.released_alias_count),
            remaining_alias_count=int(params.remaining_alias_count),
            released_alias_bytes=int(params.released_alias_bytes),
            remaining_alias_bytes=int(params.remaining_alias_bytes),
            released_backing_bytes=int(params.released_backing_bytes),
            remaining_backing_bytes=int(params.remaining_backing_bytes),
            session_cleared=cleared_raw == 1,
            cache_cleared=kernel_session_cleared,
            poisoned=(poisoned_raw == 1 or contract_violation),
            kernel_poisoned=poisoned_raw == 1,
            boolean_fields_ok=boolean_fields_ok,
            contract_violation=contract_violation,
            ioctl_ret=ret,
            errno=errno_value,
            rm_status=int(params.rmStatus),
            ioctl_calls=1,
            ioctl_wall_ns=ioctl_wall_ns,
            wall_ns=time.perf_counter_ns() - started_ns,
        )
        if kernel_session_cleared:
            # released_* is a per-call value and can legitimately disagree
            # with stale userspace totals after a prior partially successful
            # append. The authoritative cleared+remaining tuple means there
            # is no kernel session left to retry, so always retire our fd and
            # cache even while returning the contract mismatch to the caller.
            legacy_owners = _uvm._uvm_slot_direct_owners.get(normalized_device)
            if legacy_owners is not None:
                for layer_id, expert_id, slot_id in cached.backing_identities:
                    key = (layer_id, slot_id)
                    if legacy_owners.get(key) == expert_id:
                        legacy_owners.pop(key, None)
            _static_alias_sessions.pop(normalized_device, None)
            with suppress(OSError):
                os.close(cached.fd)
        elif ok:
            cached.alias_count = 0
            cached.alias_bytes = 0
            cached.backing_bytes = int(params.remaining_backing_bytes)
            cached.target_ranges.clear()
            cached.parked = False
        else:
            cached.alias_count = int(params.remaining_alias_count)
            cached.alias_bytes = int(params.remaining_alias_bytes)
            cached.backing_bytes = int(params.remaining_backing_bytes)
            cached.poisoned = poisoned_raw == 1 or contract_violation
        return result


def get_static_alias_diagnostics() -> dict[str, object]:
    """Return process-local lifecycle state without issuing an ioctl."""
    with _static_alias_lock:
        sessions = [
            {
                "device": device,
                "session_id": session.session_id,
                "owner_pid": session.owner_pid,
                "sealed": session.sealed,
                "parked": session.parked,
                "poisoned": session.poisoned,
                "alias_count": session.alias_count,
                "alias_bytes": session.alias_bytes,
                "backing_bytes": session.backing_bytes,
            }
            for device, session in sorted(_static_alias_sessions.items())
        ]
    return {"session_count": len(sessions), "sessions": sessions}


def _close_static_alias_sessions() -> None:
    """Best-effort process-exit cleanup; never destroy a fork-inherited session."""
    with _static_alias_lock:
        snapshot = [
            (device, session.session_id, session.owner_pid, session.fd)
            for device, session in _static_alias_sessions.items()
        ]
    for device, session_id, owner_pid, fd in snapshot:
        if owner_pid != os.getpid() or session_id == 0:
            with suppress(OSError):
                os.close(fd)
            continue
        try:
            result = driver_static_alias_destroy(
                session_id,
                device=device,
                release_backing=True,
                synchronize=True,
            )
            if not bool(result.get("ok")):
                with suppress(OSError):
                    os.close(fd)
        except Exception:
            with suppress(OSError):
                os.close(fd)
