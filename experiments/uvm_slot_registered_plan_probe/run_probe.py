#!/usr/bin/env python3
"""Probe immutable registered slot plans without launching vLLM.

The controller starts isolated short-lived GPU workers and verifies that the
GPU is idle before and after each one. The happy worker registers one initial L0 plan plus
four transition targets, then executes L0/L1/L2/L3/L0 while checking the
generation, managed VAs, checksums, stale-generation rejection, and cleanup.

The controller also runs isolated one-block negative workers for registration
geometry, legacy-ioctl exclusion, cross-session generation ABA, and managed
allocation identity invalidation. This script never loads a kernel module or
writes a module parameter.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
VLLM_ROOT = REPO_ROOT / "workloads" / "vllm" / "vllm"
LEGACY_DIR = REPO_ROOT / "experiments" / "uvm_512x4m_batch_compare"
DEFAULT_LIBRARY = LEGACY_DIR / "libuvm_batch_shim.so"
DEFAULT_STATS = Path("/proc/driver/nvidia-uvm/latency_stats")
DEFAULT_FAST_INDEX = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_transition_fast_index"
)
DEFAULT_DIRECT_CUTOVER = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_transition_direct_cutover"
)
DEFAULT_FAST_VALIDATION = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_fast_validation"
)
DEFAULT_BUDGET = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes"
)
DEFAULT_SRCVERSION = Path("/sys/module/nvidia_uvm/srcversion")

MODE_PREALLOC = 8
MODE_TRANSITION = 20
MODE_INITIAL = 21
NO_EVICT = 1
REUSE_BACKING_REQUIRED = 1 << 1
BLOCK_BYTES = 2 * 1024 * 1024
SLICE_BYTES = 512 * 1024
SLICES_PER_BLOCK = BLOCK_BYTES // SLICE_BYTES
PTE_BYTES = 64 * 1024
PTE_PER_BLOCK = BLOCK_BYTES // PTE_BYTES
MAX_BLOCKS = 288
LAYER_IDS = (0, 1, 2, 3)
LAYER_SEEDS = (0x10203041, 0x20304052, 0x30405063, 0x40506074)
DRIVER_BANK_LAYER = 0x51A6
INITIAL_PLAN_ID = 1
TRANSITION_PLAN_IDS = {layer: (layer << 2) | 2 for layer in LAYER_IDS}
NEGATIVE_CASES = (
    "duplicate_bootstrap",
    "geometry_count",
    "geometry_slot",
    "geometry_bytes",
    "geometry_offset",
    "geometry_uuid",
    "geometry_logical_layer",
    "legacy_mode20",
    "generation_aba",
    "backing_invalidation",
    "allocation_invalidation",
)

# These counters are updated only after the registered-plan execute path has
# entered the slot-direct core. A stale generation must leave all of them at
# zero delta.
CORE_ACTIVITY_COUNTERS = (
    "slot_direct_ioctl_calls",
    "slot_direct_successes",
    "slot_direct_fallbacks",
    "slot_direct_errors",
    "slot_direct_requested_ranges",
    "slot_direct_completed_ranges",
    "slot_direct_requested_bytes",
    "slot_direct_completed_bytes",
    "slot_direct_blocks",
    "slot_direct_memcopy_methods",
    "slot_direct_copy_pushes",
    "slot_direct_pte_map_ranges",
    "slot_direct_pte_map_bytes",
    "slot_direct_pte_write_entries",
    "slot_direct_transition_fast_index_calls",
    "slot_direct_transition_group_clear_calls",
    "slot_direct_transition_direct_cutover_calls",
    "slot_direct_transition_direct_cutover_pages",
)

CONTROL_PLANE_COUNTERS = (
    "slot_plan_register_calls",
    "slot_plan_execute_calls",
    "slot_plan_destroy_calls",
    "slot_legacy_ioctl_calls",
    "slot_legacy_transition_calls",
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


class ProbeError(RuntimeError):
    """A probe invariant was not satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def parse_stats(text: str) -> dict[str, int]:
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
            require(section is not None, f"profile row before section at {line_number}")
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
        raise ProbeError(f"unrecognized latency_stats line {line_number}: {line!r}")
    return values


def stats_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    delta = {
        name: int(after.get(name, 0)) - int(before.get(name, 0))
        for name in set(before) | set(after)
    }
    require(all(value >= 0 for value in delta.values()), "driver counters moved backwards")
    return delta


def require_no_core_activity(delta: Mapping[str, int], stage: str) -> None:
    for name in CORE_ACTIVITY_COUNTERS:
        require(name in delta, f"{stage}: missing counter {name}")
        require(int(delta[name]) == 0, f"{stage}: {name} changed by {delta[name]}")


def require_control_activity(
    delta: Mapping[str, int],
    stage: str,
    *,
    register: int = 0,
    execute: int = 0,
    destroy: int = 0,
    legacy: int = 0,
    legacy_transition: int = 0,
) -> None:
    expected = {
        "slot_plan_register_calls": register,
        "slot_plan_execute_calls": execute,
        "slot_plan_destroy_calls": destroy,
        "slot_legacy_ioctl_calls": legacy,
        "slot_legacy_transition_calls": legacy_transition,
    }
    for name, value in expected.items():
        require(name in delta, f"{stage}: missing control counter {name}")
        require(
            int(delta[name]) == value,
            f"{stage}: {name}={delta[name]}, expected {value}",
        )


def transition_plan_id(layer_id: int) -> int:
    return TRANSITION_PLAN_IDS[int(layer_id)]


def build_entries(
    pointers: Sequence[int],
    *,
    layer_id: int,
    reuse_required: bool,
) -> tuple[tuple[int, ...], ...]:
    flags = NO_EVICT | (REUSE_BACKING_REQUIRED if reuse_required else 0)
    entries: list[tuple[int, ...]] = []
    owner_base = 0x10000 * (int(layer_id) + 1)
    for block_index, pointer in enumerate(pointers):
        for slice_index in range(SLICES_PER_BLOCK):
            entry_index = block_index * SLICES_PER_BLOCK + slice_index
            entries.append(
                (
                    int(pointer) + slice_index * SLICE_BYTES,
                    SLICE_BYTES,
                    DRIVER_BANK_LAYER,
                    owner_base + entry_index,
                    entry_index + 1,
                    0,
                    flags,
                )
            )
    return tuple(entries)


class ProbeBackend(Protocol):
    def create_dataset(self, layer_id: int, seed: int, blocks: int) -> object: ...

    def dataset_pointers(self, dataset: object, blocks: int) -> tuple[int, ...]: ...

    def checksum(self, dataset: object) -> tuple[int, int]: ...

    def read_stats(self) -> dict[str, int]: ...

    def preallocate(self, entries: Sequence[Sequence[int]]) -> dict[str, object]: ...

    def register_plan(
        self,
        plan_id: int,
        layer_id: int,
        entries: Sequence[Sequence[int]],
        *,
        transition: bool,
    ) -> dict[str, object]: ...

    def register_plan_raw(
        self,
        plan_id: int,
        layer_id: int,
        entries: Sequence[Sequence[int]],
        *,
        mode: int,
        count_override: int | None = None,
        uuid_mismatch: bool = False,
    ) -> dict[str, object]: ...

    def execute_plan(
        self, plan_id: int, layer_id: int, generation: int
    ) -> dict[str, object]: ...

    def execute_stale_raw(
        self, plan_id: int, layer_id: int, generation: int
    ) -> dict[str, object]: ...

    def execute_plan_raw(
        self, plan_id: int, layer_id: int, generation: int
    ) -> dict[str, object]: ...

    def legacy_transition_raw(
        self, entries: Sequence[Sequence[int]]
    ) -> dict[str, object]: ...

    def destroy_plan(self, plan_id: int, generation: int) -> dict[str, object]: ...

    def close(self) -> dict[str, object]: ...

    def release_ranges(
        self, entries: Sequence[Sequence[int]]
    ) -> dict[str, object]: ...

    def destroy_dataset(self, dataset: object) -> None: ...

    def state_snapshot(self) -> dict[str, object]: ...


def require_preallocate(
    result: Mapping[str, object], *, entries: int, logical_bytes: int
) -> None:
    expected = {
        "mode": MODE_PREALLOC,
        "requested_count": entries,
        "success_count": entries,
        "failure_count": 0,
        "requested_bytes": logical_bytes,
        "accepted_bytes": logical_bytes,
        "remap_count": 0,
        "ioctl_calls": 1,
    }
    require(bool(result.get("ok")), f"mode8 preallocation failed: {result}")
    for name, value in expected.items():
        require(int(result.get(name, -1)) == value, f"mode8 {name} mismatch")
    require(result.get("entry_statuses") == [0] * entries, "mode8 entry status mismatch")


def require_register(
    result: Mapping[str, object],
    *,
    plan_id: int,
    layer_id: int,
    mode: int,
    entries: int,
    logical_bytes: int,
    generation: int | None = None,
) -> None:
    require(bool(result.get("ok")), f"plan {plan_id} registration failed: {result}")
    expected = {
        "plan_id": plan_id,
        "layer_id": layer_id,
        "mode": mode,
        "requested_count": entries,
        "requested_bytes": logical_bytes,
        "ioctl_ret": 0,
        "rm_status": 0,
    }
    for name, value in expected.items():
        require(int(result.get(name, -1)) == value, f"register {plan_id}: {name} mismatch")
    actual_generation = int(result.get("generation", 0))
    require(actual_generation > 0, f"register {plan_id}: generation is not positive")
    if generation is not None:
        require(
            actual_generation == generation,
            f"register {plan_id}: generation={actual_generation}, expected {generation}",
        )


def require_execute(
    result: Mapping[str, object],
    *,
    plan_id: int,
    layer_id: int,
    generation: int,
    mode: int,
    entries: int,
    blocks: int,
    logical_bytes: int,
) -> None:
    transition = mode == MODE_TRANSITION
    require(bool(result.get("ok")), f"plan {plan_id} execute failed: {result}")
    expected = {
        "plan_id": plan_id,
        "layer_id": layer_id,
        "generation": generation,
        "new_generation": generation + 1,
        "mode": mode,
        "requested_count": entries,
        "success_count": entries,
        "failure_count": 0,
        "conflict_count": 0,
        "requested_bytes": logical_bytes,
        "accepted_bytes": logical_bytes,
        "remap_count": entries,
        "tlb_invalidate_count": blocks,
        "ioctl_calls": 1,
        "rm_status": 0,
    }
    for name, value in expected.items():
        require(int(result.get(name, -1)) == value, f"execute {plan_id}: {name} mismatch")
    require(result.get("plan_execute") is True, "compact plan execute was not reported")
    require(bool(result.get("owner_cutover")) is transition, "owner cutover mismatch")
    require(result.get("owner_cutover_trusted") is True, "owner cutover is untrusted")
    require(result.get("owner_state_indeterminate") is False, "owner state is indeterminate")
    require(result.get("poisoned") is False, "registered plan session was poisoned")
    pushes = int(result.get("copy_push_count", 0))
    require(pushes > 0, "execute submitted no copy push")


def require_execute_profile(
    delta: Mapping[str, int],
    result: Mapping[str, object],
    *,
    transition: bool,
    blocks: int,
    entries: int,
    logical_bytes: int,
) -> None:
    expected = {
        "slot_direct_ioctl_calls": 1,
        "slot_direct_successes": 1,
        "slot_direct_fallbacks": 0,
        "slot_direct_errors": 0,
        "slot_direct_requested_ranges": entries,
        "slot_direct_completed_ranges": entries,
        "slot_direct_requested_bytes": logical_bytes,
        "slot_direct_completed_bytes": logical_bytes,
        "slot_direct_blocks": blocks,
        "slot_direct_pte_4k_blocks": 0,
        "slot_direct_pte_64k_blocks": blocks,
        "slot_direct_pte_2m_blocks": 0,
        "slot_direct_pte_write_entries": blocks * PTE_PER_BLOCK,
        "slot_direct_pte_map_ranges": entries,
        "slot_direct_pte_map_bytes": logical_bytes,
        "slot_direct_transition_fast_index_calls": 1 if transition else 0,
        "slot_direct_transition_group_clear_calls": 1 if transition else 0,
        "slot_direct_transition_group_clear_fallbacks": 0,
        "slot_direct_transition_direct_cutover_calls": 1 if transition else 0,
        "slot_direct_transition_direct_cutover_pages": (
            logical_bytes // 4096 if transition else 0
        ),
    }
    for name, value in expected.items():
        require(name in delta, f"execute profile missing {name}")
        require(int(delta[name]) == value, f"execute profile {name} mismatch")
    require(
        int(delta.get("slot_direct_copy_pushes", -1))
        == int(result.get("copy_push_count", -2)),
        "wrapper/profile copy-push count mismatch",
    )
    require(
        int(delta.get("slot_direct_memcopy_methods", -1)) >= entries,
        "too few CE copy methods",
    )
    require_control_activity(delta, "registered plan execute", execute=1)


def require_stale_rejection(
    result: Mapping[str, object], *, generation: int, current_generation: int
) -> None:
    require_execute_rejection(
        result,
        generation=generation,
        current_generation=current_generation,
        poisoned=False,
        stage="stale execute",
    )


def require_execute_rejection(
    result: Mapping[str, object],
    *,
    generation: int,
    current_generation: int,
    poisoned: bool,
    stage: str,
) -> None:
    require(result.get("rejected") is True, f"{stage}: execute was accepted: {result}")
    require(
        int(result.get("generation", -1)) == generation,
        f"{stage}: input generation changed",
    )
    require(
        int(result.get("new_generation", -1)) == current_generation,
        f"{stage}: rejection changed or hid the current generation",
    )
    require(
        bool(result.get("poisoned")) is poisoned,
        f"{stage}: poisoned={result.get('poisoned')!r}, expected {poisoned}",
    )
    for name in (
        "requested_bytes",
        "accepted_bytes",
        "mode",
        "requested_count",
        "success_count",
        "failure_count",
        "conflict_count",
        "remap_count",
        "tlb_invalidate_count",
        "copy_push_count",
        "owner_cutover",
    ):
        require(
            int(result.get(name, -1)) == 0,
            f"{stage}: rejected execute populated {name}",
        )


def require_allocation_identity_rejection(
    result: Mapping[str, object],
    delta: Mapping[str, int],
    *,
    generation: int,
    entries: int,
    logical_bytes: int,
) -> None:
    stage = "allocation_invalidation"
    require(result.get("rejected") is True, f"{stage}: execute was accepted")
    expected_result = {
        "generation": generation,
        "new_generation": generation,
        "mode": MODE_TRANSITION,
        "requested_count": entries,
        "success_count": 0,
        "failure_count": entries,
        "conflict_count": 0,
        "requested_bytes": logical_bytes,
        "accepted_bytes": 0,
        "remap_count": 0,
        "tlb_invalidate_count": 0,
        "copy_push_count": 0,
        "owner_cutover": 0,
    }
    for name, value in expected_result.items():
        require(
            int(result.get(name, -1)) == value,
            f"{stage}: {name}={result.get(name)!r}, expected {value}",
        )
    require(
        result.get("poisoned") is False,
        f"{stage}: preflight rejection poisoned the kernel registry",
    )

    expected_core = {
        "slot_direct_ioctl_calls": 1,
        "slot_direct_per_block_calls": 0,
        "slot_direct_shared_calls": 1,
        "slot_direct_successes": 0,
        "slot_direct_fallbacks": 0,
        "slot_direct_errors": 1,
        "slot_direct_requested_ranges": entries,
        "slot_direct_completed_ranges": 0,
        "slot_direct_requested_bytes": logical_bytes,
        "slot_direct_completed_bytes": 0,
        "slot_direct_blocks": 0,
        "slot_direct_memcopy_methods": 0,
        "slot_direct_copy_pushes": 0,
        "slot_direct_pte_map_ranges": 0,
        "slot_direct_pte_map_bytes": 0,
        "slot_direct_pte_write_entries": 0,
        "slot_direct_skipped_unmap_blocks": 0,
        "slot_direct_stage_sync_calls": 0,
        "slot_direct_tracker_waits": 0,
        "slot_direct_vidmem_allocations": 0,
        "slot_direct_vidmem_evictions": 0,
        "slot_direct_transition_fast_index_calls": 0,
        "slot_direct_transition_group_clear_calls": 0,
        "slot_direct_transition_clear_ranges": 0,
        "slot_direct_transition_clear_unique_blocks": 0,
        "slot_direct_transition_direct_cutover_calls": 0,
        "slot_direct_transition_direct_cutover_pages": 0,
    }
    for name, value in expected_core.items():
        require(name in delta, f"{stage}: missing counter {name}")
        require(
            int(delta[name]) == value,
            f"{stage}: {name}={delta[name]}, expected {value}",
        )
    for name in (
        "slot_direct_pte_4k_blocks",
        "slot_direct_pte_64k_blocks",
        "slot_direct_pte_2m_blocks",
        "slot_direct_transition_group_clear_fallbacks",
    ):
        require(name in delta, f"{stage}: missing counter {name}")
        require(int(delta[name]) == 0, f"{stage}: {name} changed")
    require_control_activity(delta, stage, execute=1)


def require_backing_identity_rejection(
    result: Mapping[str, object],
    delta: Mapping[str, int],
    *,
    generation: int,
    entries: int,
    logical_bytes: int,
) -> None:
    """Prove cached slot backing rejection happens before transaction start."""
    stage = "backing_invalidation"
    require(result.get("rejected") is True, f"{stage}: execute was accepted")
    expected_result = {
        "generation": generation,
        "new_generation": generation,
        "mode": MODE_TRANSITION,
        "requested_count": entries,
        "success_count": 0,
        "failure_count": entries,
        "conflict_count": 0,
        "requested_bytes": logical_bytes,
        "accepted_bytes": 0,
        "remap_count": 0,
        "tlb_invalidate_count": 0,
        "copy_push_count": 0,
        "owner_cutover": 0,
    }
    for name, value in expected_result.items():
        require(
            int(result.get(name, -1)) == value,
            f"{stage}: {name}={result.get(name)!r}, expected {value}",
        )
    require(
        result.get("poisoned") is False,
        f"{stage}: preflight rejection poisoned the kernel registry",
    )

    expected_core = {
        "slot_direct_ioctl_calls": 1,
        "slot_direct_per_block_calls": 0,
        "slot_direct_shared_calls": 1,
        "slot_direct_successes": 0,
        "slot_direct_fallbacks": 0,
        "slot_direct_errors": 1,
        "slot_direct_requested_ranges": entries,
        "slot_direct_completed_ranges": 0,
        "slot_direct_requested_bytes": logical_bytes,
        "slot_direct_completed_bytes": 0,
        "slot_direct_blocks": 0,
        "slot_direct_memcopy_methods": 0,
        "slot_direct_copy_pushes": 0,
        "slot_direct_pte_map_ranges": 0,
        "slot_direct_pte_map_bytes": 0,
        "slot_direct_pte_write_entries": 0,
        "slot_direct_skipped_unmap_blocks": 0,
        "slot_direct_stage_sync_calls": 0,
        "slot_direct_tracker_waits": 0,
        "slot_direct_vidmem_allocations": 0,
        "slot_direct_vidmem_evictions": 0,
        # The cached validator is reached after the immutable mode strategy
        # has selected the fast-index path, but before clear/copy/map work.
        "slot_direct_transition_fast_index_calls": 1,
        "slot_direct_transition_group_clear_calls": 0,
        "slot_direct_transition_clear_ranges": 0,
        "slot_direct_transition_clear_unique_blocks": 0,
        "slot_direct_transition_direct_cutover_calls": 0,
        "slot_direct_transition_direct_cutover_pages": 0,
    }
    for name, value in expected_core.items():
        require(name in delta, f"{stage}: missing counter {name}")
        require(
            int(delta[name]) == value,
            f"{stage}: {name}={delta[name]}, expected {value}",
        )
    for name in (
        "slot_direct_pte_4k_blocks",
        "slot_direct_pte_64k_blocks",
        "slot_direct_pte_2m_blocks",
        "slot_direct_transition_group_clear_fallbacks",
    ):
        require(name in delta, f"{stage}: missing counter {name}")
        require(int(delta[name]) == 0, f"{stage}: {name} changed")
    require_control_activity(delta, stage, execute=1)


def summarize(result: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "ok",
        "reason",
        "plan_id",
        "layer_id",
        "generation",
        "new_generation",
        "current_generation",
        "mode",
        "requested_count",
        "success_count",
        "failure_count",
        "requested_bytes",
        "accepted_bytes",
        "remap_count",
        "tlb_invalidate_count",
        "copy_push_count",
        "owner_cutover",
        "owner_cutover_count",
        "owner_cutover_trusted",
        "owner_state_indeterminate",
        "poisoned",
        "plan_execute",
        "destroyed",
        "ioctl_ret",
        "errno",
        "rm_status",
        "ioctl_calls",
        "ioctl_wall_ns",
        "wall_ns",
        "rejected",
    )
    return {key: result.get(key) for key in keys if key in result}


def exercise_registered_plans(backend: ProbeBackend, *, blocks: int) -> dict[str, Any]:
    require(1 <= blocks <= MAX_BLOCKS, f"blocks must be in [1, {MAX_BLOCKS}]")
    entries_count = blocks * SLICES_PER_BLOCK
    logical_bytes = blocks * BLOCK_BYTES
    datasets: dict[int, object] = {}
    pointers: dict[int, tuple[int, ...]] = {}
    for layer_id, seed in zip(LAYER_IDS, LAYER_SEEDS):
        dataset = backend.create_dataset(layer_id, seed, blocks)
        datasets[layer_id] = dataset
        layer_pointers = backend.dataset_pointers(dataset, blocks)
        require(len(layer_pointers) == blocks, f"L{layer_id} pointer count mismatch")
        require(all(value and value % BLOCK_BYTES == 0 for value in layer_pointers),
                f"L{layer_id} does not have 2MiB-aligned VAs")
        require(len(set(layer_pointers)) == blocks, f"L{layer_id} has duplicate VAs")
        pointers[layer_id] = layer_pointers

    all_vas = tuple(value for layer in LAYER_IDS for value in pointers[layer])
    require(len(set(all_vas)) == len(all_vas), "managed VAs overlap across layers")

    def require_va_stable(stage: str) -> None:
        current = tuple(
            value
            for layer in LAYER_IDS
            for value in backend.dataset_pointers(datasets[layer], blocks)
        )
        require(current == all_vas, f"managed VA changed after {stage}")

    initial_entries = build_entries(pointers[0], layer_id=0, reuse_required=False)
    transition_entries = {
        layer: build_entries(pointers[layer], layer_id=layer, reuse_required=True)
        for layer in LAYER_IDS
    }

    lifecycle_before = backend.read_stats()
    before_preallocate = lifecycle_before
    preallocate = backend.preallocate(initial_entries)
    after_preallocate = backend.read_stats()
    preallocate_delta = stats_delta(before_preallocate, after_preallocate)
    require_preallocate(preallocate, entries=entries_count, logical_bytes=logical_bytes)
    preallocate_legacy_calls = int(preallocate.get("fd_candidate_ioctl_calls", 1))
    require(preallocate_legacy_calls >= 1, "mode8 submitted no candidate ioctl")
    require_control_activity(
        preallocate_delta,
        "mode8 preallocate",
        legacy=preallocate_legacy_calls,
    )
    require_va_stable("preallocate")

    before_register = backend.read_stats()
    registrations: list[dict[str, object]] = []
    initial_register = backend.register_plan(
        INITIAL_PLAN_ID, 0, initial_entries, transition=False
    )
    require_register(
        initial_register,
        plan_id=INITIAL_PLAN_ID,
        layer_id=0,
        mode=MODE_INITIAL,
        entries=entries_count,
        logical_bytes=logical_bytes,
    )
    generation_start = int(initial_register["generation"])
    registrations.append(summarize(initial_register))
    for layer_id in LAYER_IDS:
        plan_id = transition_plan_id(layer_id)
        result = backend.register_plan(
            plan_id,
            layer_id,
            transition_entries[layer_id],
            transition=True,
        )
        require_register(
            result,
            plan_id=plan_id,
            layer_id=layer_id,
            mode=MODE_TRANSITION,
            entries=entries_count,
            logical_bytes=logical_bytes,
            generation=generation_start,
        )
        registrations.append(summarize(result))
    after_register = backend.read_stats()
    register_delta = stats_delta(before_register, after_register)
    require_no_core_activity(register_delta, "plan registration")
    require_control_activity(register_delta, "plan registration", register=5)
    require_va_stable("plan registration")

    execution_order = (0, 1, 2, 3, 0)
    generation = generation_start
    executions: list[dict[str, object]] = []
    profile_deltas: list[dict[str, int]] = []
    checksums: list[dict[str, int]] = []
    first_l0_checksum: int | None = None

    for sequence_index, layer_id in enumerate(execution_order):
        initial = sequence_index == 0
        plan_id = INITIAL_PLAN_ID if initial else transition_plan_id(layer_id)
        mode = MODE_INITIAL if initial else MODE_TRANSITION
        before_execute = backend.read_stats()
        execute_result = backend.execute_plan(plan_id, layer_id, generation)
        after_execute = backend.read_stats()
        execute_delta = stats_delta(before_execute, after_execute)
        require_execute(
            execute_result,
            plan_id=plan_id,
            layer_id=layer_id,
            generation=generation,
            mode=mode,
            entries=entries_count,
            blocks=blocks,
            logical_bytes=logical_bytes,
        )
        require_execute_profile(
            execute_delta,
            execute_result,
            transition=not initial,
            blocks=blocks,
            entries=entries_count,
            logical_bytes=logical_bytes,
        )
        generation += 1
        require_va_stable(f"L{layer_id} execute")
        checksum, expected = backend.checksum(datasets[layer_id])
        require(checksum == expected, f"L{layer_id} checksum mismatch")
        require_va_stable(f"L{layer_id} checksum")
        checksums.append(
            {"sequence_index": sequence_index, "layer_id": layer_id,
             "checksum": checksum, "expected": expected}
        )
        if initial:
            first_l0_checksum = checksum
        elif sequence_index == len(execution_order) - 1:
            require(checksum == first_l0_checksum, "return-to-L0 checksum changed")

        executions.append(summarize(execute_result))
        profile_deltas.append(execute_delta)

        # Test the kernel generation gate, rather than the userspace fast
        # rejection, after the first valid execute. L0 remains the active
        # mapping and must be byte-identical before and after the stale ioctl.
        if sequence_index == 0:
            stale_before = backend.read_stats()
            stale = backend.execute_stale_raw(
                transition_plan_id(1), 1, generation - 1
            )
            stale_after = backend.read_stats()
            stale_delta = stats_delta(stale_before, stale_after)
            require_stale_rejection(
                stale, generation=generation - 1, current_generation=generation
            )
            require_no_core_activity(stale_delta, "stale generation")
            require_control_activity(stale_delta, "stale generation", execute=1)
            require_va_stable("stale generation")
            stale_checksum, stale_expected = backend.checksum(datasets[0])
            require(
                stale_checksum == stale_expected == first_l0_checksum,
                "stale generation changed the active L0 bytes",
            )
            require_va_stable("stale generation checksum")
            stale_record = {
                "result": summarize(stale),
                "profile_delta": stale_delta,
                "active_checksum": stale_checksum,
                "active_checksum_expected": stale_expected,
            }

    require(
        generation == generation_start + len(execution_order),
        f"final generation is {generation}, expected "
        f"{generation_start + len(execution_order)}",
    )
    require(len({item["expected"] for item in checksums[:4]}) == 4,
            "layer checksum signatures are not distinct")

    before_destroy = backend.read_stats()
    destroy_results: list[dict[str, object]] = []
    plan_ids = (INITIAL_PLAN_ID,) + tuple(
        transition_plan_id(layer) for layer in LAYER_IDS
    )
    for index, plan_id in enumerate(plan_ids):
        result = backend.destroy_plan(plan_id, generation)
        require(bool(result.get("ok")), f"destroy plan {plan_id} failed: {result}")
        require(result.get("destroyed") is True, f"plan {plan_id} was not destroyed")
        expected_current = 0 if index == len(plan_ids) - 1 else generation
        require(
            int(result.get("current_generation", -1)) == expected_current,
            f"destroy plan {plan_id} current generation mismatch",
        )
        destroy_results.append(summarize(result))
    after_destroy = backend.read_stats()
    destroy_delta = stats_delta(before_destroy, after_destroy)
    require_no_core_activity(destroy_delta, "plan destroy")
    require_control_activity(destroy_delta, "plan destroy", destroy=5)
    require_va_stable("plan destroy")
    destroy_checksum, destroy_expected = backend.checksum(datasets[0])
    require(
        destroy_checksum == destroy_expected == first_l0_checksum,
        "destroying descriptors changed the live L0 mapping",
    )

    pre_close_state = backend.state_snapshot()
    require(int(pre_close_state.get("plan_count", -1)) == 0,
            "userspace still tracks a destroyed plan")
    before_close = backend.read_stats()
    close_result = backend.close()
    after_close = backend.read_stats()
    close_delta = stats_delta(before_close, after_close)
    require_control_activity(close_delta, "mode19 close", legacy=1)
    require(bool(close_result.get("ok")), f"slot session close failed: {close_result}")
    require(close_result.get("closed") is True, "slot session was not closed")
    require(not close_result.get("plan_destroy"), "close unexpectedly destroyed a plan")
    release = close_result.get("release")
    require(isinstance(release, Mapping), "close did not return a mode19 release result")
    require(bool(release.get("ok")), f"mode19 close release failed: {release}")
    require(int(release.get("requested_count", -1)) == entries_count,
            "close released the wrong slot count")
    require(int(release.get("success_count", -1)) == entries_count,
            "close did not release every slot")
    require(int(release.get("failure_count", -1)) == 0,
            "close reported a failed slot release")
    require(int(release.get("accepted_bytes", -1)) == logical_bytes,
            "close released the wrong number of bytes")
    require_va_stable("slot close")

    post_close_state = backend.state_snapshot()
    expected_zero = {
        "allocated_slots": 0,
        "plan_count": 0,
        "tracked_generation": None,
        "session_open": False,
    }
    for name, value in expected_zero.items():
        require(post_close_state.get(name) == value,
                f"post-close {name}={post_close_state.get(name)!r}, expected {value!r}")

    lifecycle_delta = stats_delta(lifecycle_before, after_close)
    require_control_activity(
        lifecycle_delta,
        "complete registered-plan lifecycle",
        register=5,
        execute=6,
        destroy=5,
        legacy=preallocate_legacy_calls + 1,
        legacy_transition=0,
    )

    return {
        "status": "pass",
        "blocks": blocks,
        "entries_per_layer": entries_count,
        "logical_bytes_per_layer": logical_bytes,
        "managed_vas": {str(layer): list(pointers[layer]) for layer in LAYER_IDS},
        "va_stable": True,
        "execution_order": list(execution_order),
        "generation_start": generation_start,
        "generation_end": generation,
        "preallocate": summarize(preallocate),
        "preallocate_profile_delta": preallocate_delta,
        "preallocate_legacy_ioctl_calls": preallocate_legacy_calls,
        "registrations": registrations,
        "register_profile_delta": register_delta,
        "executions": executions,
        "execute_profile_deltas": profile_deltas,
        "checksums": checksums,
        "stale_generation": stale_record,
        "destroy_results": destroy_results,
        "destroy_profile_delta": destroy_delta,
        "destroy_checksum": destroy_checksum,
        "destroy_checksum_expected": destroy_expected,
        "pre_close_state": pre_close_state,
        "close": close_result,
        "close_profile_delta": close_delta,
        "post_close_state": post_close_state,
        "lifecycle_control_delta": {
            name: lifecycle_delta[name] for name in CONTROL_PLANE_COUNTERS
        },
        "zero_slots_after_close": True,
    }


def _mutate_entry(
    entries: Sequence[Sequence[int]], entry_index: int, field_index: int, value: int
) -> tuple[tuple[int, ...], ...]:
    copied = [list(map(int, entry)) for entry in entries]
    copied[entry_index][field_index] = int(value)
    return tuple(tuple(entry) for entry in copied)


def _require_raw_rejection(
    result: Mapping[str, object],
    delta: Mapping[str, int],
    *,
    stage: str,
    register: int = 0,
    execute: int = 0,
    legacy: int = 0,
    legacy_transition: int = 0,
) -> None:
    require(result.get("rejected") is True, f"{stage}: request was accepted: {result}")
    require_no_core_activity(delta, stage)
    require_control_activity(
        delta,
        stage,
        register=register,
        execute=execute,
        legacy=legacy,
        legacy_transition=legacy_transition,
    )


def _require_closed_state(backend: ProbeBackend, stage: str) -> dict[str, object]:
    state = backend.state_snapshot()
    expected = {
        "allocated_slots": 0,
        "plan_count": 0,
        "tracked_generation": None,
        "session_open": False,
    }
    for name, value in expected.items():
        require(
            state.get(name) == value,
            f"{stage}: {name}={state.get(name)!r}, expected {value!r}",
        )
    return state


def _close_one_block_session(
    backend: ProbeBackend, *, stage: str
) -> tuple[dict[str, object], dict[str, object]]:
    result = backend.close()
    require(bool(result.get("ok")), f"{stage}: close failed: {result}")
    require(result.get("closed") is True, f"{stage}: session was not closed")
    release = result.get("release")
    require(isinstance(release, Mapping), f"{stage}: close has no release result")
    require(bool(release.get("ok")), f"{stage}: release failed: {release}")
    require(
        int(release.get("requested_count", -1)) == SLICES_PER_BLOCK,
        f"{stage}: released slot count mismatch",
    )
    require(
        int(release.get("accepted_bytes", -1)) == BLOCK_BYTES,
        f"{stage}: released byte count mismatch",
    )
    return dict(result), _require_closed_state(backend, stage)


def _destroy_plans(
    backend: ProbeBackend,
    plan_ids: Sequence[int],
    generation: int,
    *,
    stage: str,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for plan_id in plan_ids:
        result = backend.destroy_plan(int(plan_id), int(generation))
        require(bool(result.get("ok")), f"{stage}: destroy {plan_id} failed: {result}")
        require(result.get("destroyed") is True, f"{stage}: plan {plan_id} survived")
        results.append(summarize(result))
    return results


def _create_one_block_layer(
    backend: ProbeBackend, layer_id: int, seed: int
) -> tuple[object, tuple[int, ...], tuple[tuple[int, ...], ...]]:
    dataset = backend.create_dataset(layer_id, seed, 1)
    pointers = backend.dataset_pointers(dataset, 1)
    require(
        len(pointers) == 1 and pointers[0] > 0 and pointers[0] % BLOCK_BYTES == 0,
        f"negative L{layer_id}: invalid managed VA",
    )
    return (
        dataset,
        pointers,
        build_entries(
            pointers,
            layer_id=layer_id,
            reuse_required=layer_id != 0,
        ),
    )


def _prepare_negative_bank(
    backend: ProbeBackend, *, with_target: bool
) -> dict[str, object]:
    dataset0, pointers0, initial_entries = _create_one_block_layer(
        backend, 0, LAYER_SEEDS[0]
    )
    target: tuple[object, tuple[int, ...], tuple[tuple[int, ...], ...]] | None = None
    if with_target:
        target = _create_one_block_layer(backend, 1, LAYER_SEEDS[1])

    preallocate = backend.preallocate(initial_entries)
    require_preallocate(
        preallocate,
        entries=SLICES_PER_BLOCK,
        logical_bytes=BLOCK_BYTES,
    )
    bootstrap = backend.register_plan(
        INITIAL_PLAN_ID,
        0,
        initial_entries,
        transition=False,
    )
    require_register(
        bootstrap,
        plan_id=INITIAL_PLAN_ID,
        layer_id=0,
        mode=MODE_INITIAL,
        entries=SLICES_PER_BLOCK,
        logical_bytes=BLOCK_BYTES,
    )
    return {
        "dataset0": dataset0,
        "pointers0": pointers0,
        "initial_entries": initial_entries,
        "target": target,
        "generation": int(bootstrap["generation"]),
        "bootstrap": bootstrap,
    }


def exercise_negative_case(
    backend: ProbeBackend, *, case: str
) -> dict[str, Any]:
    """Run one isolated one-block fail-closed contract check."""
    require(case in NEGATIVE_CASES, f"unknown negative case {case!r}")
    setup = _prepare_negative_bank(
        backend,
        with_target=case
        not in ("duplicate_bootstrap", "generation_aba"),
    )
    generation = int(setup["generation"])
    initial_entries = setup["initial_entries"]
    require(isinstance(initial_entries, Sequence), "missing initial entries")
    plan_ids = [INITIAL_PLAN_ID]
    evidence: dict[str, object] = {}

    if case == "duplicate_bootstrap":
        before = backend.read_stats()
        rejected = backend.register_plan_raw(
            0x101,
            9,
            initial_entries,
            mode=MODE_INITIAL,
        )
        after = backend.read_stats()
        delta = stats_delta(before, after)
        _require_raw_rejection(
            rejected, delta, stage=case, register=1
        )

    elif case.startswith("geometry_"):
        target = setup["target"]
        require(isinstance(target, tuple), f"{case}: missing target dataset")
        _target_dataset, _target_pointers, target_entries = target
        raw_entries = target_entries
        count_override: int | None = None
        uuid_mismatch = False
        logical_layer = 1

        if case == "geometry_count":
            count_override = len(target_entries) - 1
        elif case == "geometry_slot":
            raw_entries = _mutate_entry(
                target_entries, 0, 4, len(target_entries) + 1
            )
        elif case == "geometry_bytes":
            raw_entries = _mutate_entry(
                target_entries, 0, 1, int(target_entries[0][1]) - PTE_BYTES
            )
        elif case == "geometry_offset":
            raw_entries = _mutate_entry(
                target_entries, 0, 5, int(target_entries[0][5]) + PTE_BYTES
            )
        elif case == "geometry_uuid":
            uuid_mismatch = True
        elif case == "geometry_logical_layer":
            valid = backend.register_plan(
                transition_plan_id(1),
                1,
                target_entries,
                transition=True,
            )
            require_register(
                valid,
                plan_id=transition_plan_id(1),
                layer_id=1,
                mode=MODE_TRANSITION,
                entries=SLICES_PER_BLOCK,
                logical_bytes=BLOCK_BYTES,
                generation=generation,
            )
            plan_ids.append(transition_plan_id(1))
            logical_layer = 1
        else:  # pragma: no cover - guarded by NEGATIVE_CASES
            raise ProbeError(f"unhandled geometry case {case}")

        before = backend.read_stats()
        rejected = backend.register_plan_raw(
            0x200 + NEGATIVE_CASES.index(case),
            logical_layer,
            raw_entries,
            mode=MODE_TRANSITION,
            count_override=count_override,
            uuid_mismatch=uuid_mismatch,
        )
        after = backend.read_stats()
        delta = stats_delta(before, after)
        _require_raw_rejection(rejected, delta, stage=case, register=1)

    elif case == "legacy_mode20":
        target = setup["target"]
        require(isinstance(target, tuple), "legacy case has no target")
        _target_dataset, _target_pointers, target_entries = target
        before = backend.read_stats()
        rejected = backend.legacy_transition_raw(target_entries)
        after = backend.read_stats()
        delta = stats_delta(before, after)
        _require_raw_rejection(
            rejected,
            delta,
            stage=case,
            legacy=1,
            legacy_transition=1,
        )

    elif case == "generation_aba":
        old_generation = generation
        first_destroy = _destroy_plans(
            backend,
            plan_ids,
            generation,
            stage="generation_aba old session",
        )
        first_close, first_closed_state = _close_one_block_session(
            backend, stage="generation_aba old session"
        )

        second_preallocate = backend.preallocate(initial_entries)
        require_preallocate(
            second_preallocate,
            entries=SLICES_PER_BLOCK,
            logical_bytes=BLOCK_BYTES,
        )
        second_register = backend.register_plan(
            INITIAL_PLAN_ID, 0, initial_entries, transition=False
        )
        require_register(
            second_register,
            plan_id=INITIAL_PLAN_ID,
            layer_id=0,
            mode=MODE_INITIAL,
            entries=SLICES_PER_BLOCK,
            logical_bytes=BLOCK_BYTES,
        )
        generation = int(second_register["generation"])
        require(
            generation > old_generation,
            "generation_aba: new session generation did not advance past the "
            "old session",
        )
        plan_ids = [INITIAL_PLAN_ID]
        before = backend.read_stats()
        rejected = backend.execute_plan_raw(INITIAL_PLAN_ID, 0, old_generation)
        after = backend.read_stats()
        delta = stats_delta(before, after)
        _require_raw_rejection(rejected, delta, stage=case, execute=1)
        require_execute_rejection(
            rejected,
            generation=old_generation,
            current_generation=generation,
            poisoned=False,
            stage=case,
        )
        evidence.update(
            {
                "old_generation": old_generation,
                "old_session_final_generation": old_generation,
                "new_generation": generation,
                "first_destroy": first_destroy,
                "first_close": summarize(first_close),
                "first_closed_state": first_closed_state,
                "second_preallocate": summarize(second_preallocate),
                "second_register": summarize(second_register),
            }
        )

    elif case == "backing_invalidation":
        target = setup["target"]
        require(isinstance(target, tuple), "backing case has no target")
        _target_dataset, _target_pointers, target_entries = target
        target_plan_id = transition_plan_id(1)
        registered = backend.register_plan(
            target_plan_id, 1, target_entries, transition=True
        )
        require_register(
            registered,
            plan_id=target_plan_id,
            layer_id=1,
            mode=MODE_TRANSITION,
            entries=SLICES_PER_BLOCK,
            logical_bytes=BLOCK_BYTES,
            generation=generation,
        )
        plan_ids.append(target_plan_id)

        release_before = backend.read_stats()
        released = backend.release_ranges(initial_entries)
        release_after = backend.read_stats()
        release_delta = stats_delta(release_before, release_after)
        require(bool(released.get("ok")), f"{case}: backing release failed: {released}")
        require(
            int(released.get("requested_count", -1)) == SLICES_PER_BLOCK,
            f"{case}: backing release count mismatch",
        )
        require(
            int(released.get("success_count", -1)) == SLICES_PER_BLOCK,
            f"{case}: not every cached lane was released",
        )
        require(
            int(released.get("failure_count", -1)) == 0,
            f"{case}: backing release reported failures",
        )
        require(
            int(released.get("accepted_bytes", -1)) == BLOCK_BYTES,
            f"{case}: backing release byte mismatch",
        )
        require_no_core_activity(release_delta, f"{case} release")
        require_control_activity(
            release_delta,
            f"{case} release",
            legacy=int(released.get("fd_candidate_ioctl_calls", 1)),
        )
        post_release_state = backend.state_snapshot()
        require(
            int(post_release_state.get("allocated_slots", -1)) == 0,
            f"{case}: released backing remains tracked",
        )
        require(
            int(post_release_state.get("plan_count", -1)) == len(plan_ids),
            f"{case}: backing release changed plan registry",
        )
        require(
            int(post_release_state.get("tracked_generation", -1)) == generation,
            f"{case}: backing release changed generation",
        )
        require(
            post_release_state.get("session_open") is True,
            f"{case}: backing release closed the session",
        )

        reallocate_before = backend.read_stats()
        reallocated = backend.preallocate(initial_entries)
        reallocate_after = backend.read_stats()
        reallocate_delta = stats_delta(reallocate_before, reallocate_after)
        require_preallocate(
            reallocated,
            entries=SLICES_PER_BLOCK,
            logical_bytes=BLOCK_BYTES,
        )
        require_no_core_activity(reallocate_delta, f"{case} reallocate")
        require_control_activity(
            reallocate_delta,
            f"{case} reallocate",
            legacy=int(reallocated.get("fd_candidate_ioctl_calls", 1)),
        )
        post_reallocate_state = backend.state_snapshot()
        require(
            int(post_reallocate_state.get("allocated_slots", -1))
            == SLICES_PER_BLOCK,
            f"{case}: replacement backing is not fully tracked",
        )
        require(
            int(post_reallocate_state.get("plan_count", -1)) == len(plan_ids),
            f"{case}: backing replacement changed plan registry",
        )
        require(
            int(post_reallocate_state.get("tracked_generation", -1)) == generation,
            f"{case}: backing replacement changed plan generation",
        )
        require(
            post_reallocate_state.get("session_open") is True,
            f"{case}: backing replacement closed the session",
        )

        before = backend.read_stats()
        rejected = backend.execute_plan_raw(target_plan_id, 1, generation)
        after = backend.read_stats()
        delta = stats_delta(before, after)
        require_backing_identity_rejection(
            rejected,
            delta,
            generation=generation,
            entries=SLICES_PER_BLOCK,
            logical_bytes=BLOCK_BYTES,
        )
        post_rejection_state = backend.state_snapshot()
        require(
            post_rejection_state == post_reallocate_state,
            f"{case}: rejected execute changed session state",
        )
        evidence.update(
            {
                "registered": summarize(registered),
                "backing_release": summarize(released),
                "backing_release_profile_delta": release_delta,
                "backing_reallocate": summarize(reallocated),
                "backing_reallocate_profile_delta": reallocate_delta,
                "post_release_state": post_release_state,
                "post_reallocate_state": post_reallocate_state,
                "post_rejection_state": post_rejection_state,
            }
        )

    elif case == "allocation_invalidation":
        target = setup["target"]
        require(isinstance(target, tuple), "allocation case has no target")
        target_dataset, old_pointers, target_entries = target
        target_plan_id = transition_plan_id(1)
        registered = backend.register_plan(
            target_plan_id, 1, target_entries, transition=True
        )
        require_register(
            registered,
            plan_id=target_plan_id,
            layer_id=1,
            mode=MODE_TRANSITION,
            entries=SLICES_PER_BLOCK,
            logical_bytes=BLOCK_BYTES,
            generation=generation,
        )
        plan_ids.append(target_plan_id)
        backend.destroy_dataset(target_dataset)
        recreated, new_pointers, _new_entries = _create_one_block_layer(
            backend, 1, LAYER_SEEDS[1] ^ 0x55AA55AA
        )
        require(recreated is not None, "allocation_invalidation: recreate failed")

        before = backend.read_stats()
        rejected = backend.execute_plan_raw(target_plan_id, 1, generation)
        after = backend.read_stats()
        delta = stats_delta(before, after)
        require_allocation_identity_rejection(
            rejected,
            delta,
            generation=generation,
            entries=SLICES_PER_BLOCK,
            logical_bytes=BLOCK_BYTES,
        )
        post_rejection_state = backend.state_snapshot()
        require(
            int(post_rejection_state.get("tracked_generation", -1)) == generation,
            "allocation_invalidation: preflight rejection changed generation",
        )
        require(
            int(post_rejection_state.get("plan_count", -1)) == len(plan_ids),
            "allocation_invalidation: preflight rejection changed plan registry",
        )
        require(
            post_rejection_state.get("session_open") is True,
            "allocation_invalidation: preflight rejection closed active session",
        )
        evidence.update(
            {
                "registered": summarize(registered),
                "old_target_vas": list(old_pointers),
                "recreated_target_vas": list(new_pointers),
                "same_va_reused": tuple(old_pointers) == tuple(new_pointers),
                "allocation_recreated": True,
                "post_rejection_state": post_rejection_state,
            }
        )

    else:  # pragma: no cover - guarded above
        raise ProbeError(f"unhandled negative case {case}")

    destroy_results = _destroy_plans(
        backend, plan_ids, generation, stage=f"{case} cleanup"
    )
    close_result, post_close_state = _close_one_block_session(
        backend, stage=f"{case} cleanup"
    )
    evidence.update(
        {
            "rejection": summarize(rejected),
            "rejection_profile_delta": delta,
            "destroy_results": destroy_results,
            "close": summarize(close_result),
            "post_close_state": post_close_state,
        }
    )
    return {
        "status": "pass",
        "negative_case": case,
        "blocks": 1,
        "generation": generation,
        "zero_slots_after_close": True,
        **evidence,
    }


class GpuBackend:
    """Thin adapter around the existing CUDA shim and low-level UVM wrappers."""

    def __init__(self, *, library: Path, stats: Path, device: int) -> None:
        os.environ["VLLM_UVM_SLOT_DIRECT_SHARED_PUSH"] = "1"
        os.environ["VLLM_UVM_SLOT_DIRECT_64K_PTE"] = "1"
        os.environ["VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION"] = "1"
        sys.path.insert(0, str(VLLM_ROOT))
        sys.path.insert(0, str(LEGACY_DIR))
        import benchmark as legacy  # type: ignore[import-not-found]
        import vllm.device_allocator.uvm as uvm

        self.legacy = legacy
        self.uvm = uvm
        self.library = legacy.load_library(library)
        self.stats_path = stats
        self.device = int(device)
        self.datasets: list[int] = []
        self.closed = False
        self.stream = 0
        self._check_cuda(self.library.uvm512_set_device(self.device), "set device")
        self.stream = int(self.library.uvm512_stream_create())
        require(self.stream != 0, "stream_create failed")

    def _check_cuda(self, status: int, operation: str) -> None:
        self.legacy.check_cuda(self.library, int(status), operation)

    def create_dataset(self, layer_id: int, seed: int, blocks: int) -> object:
        dataset = int(self.library.uvm512_dataset_create(blocks, BLOCK_BYTES, 0))
        require(dataset != 0, f"L{layer_id} dataset_create failed")
        self.datasets.append(dataset)
        self._check_cuda(
            self.library.uvm512_dataset_reseed(dataset, seed),
            f"L{layer_id} reseed",
        )
        return dataset

    def dataset_pointers(self, dataset: object, blocks: int) -> tuple[int, ...]:
        return tuple(
            int(self.library.uvm512_dataset_pointer(int(dataset), index))
            for index in range(blocks)
        )

    def checksum(self, dataset: object) -> tuple[int, int]:
        handle = int(dataset)
        self._check_cuda(
            self.library.uvm512_dataset_launch_checksum(handle, self.stream),
            "launch checksum",
        )
        self._check_cuda(
            self.library.uvm512_stream_synchronize(self.stream),
            "synchronize checksum",
        )
        actual = ctypes.c_uint64()
        self._check_cuda(
            self.library.uvm512_dataset_read_checksum(handle, ctypes.byref(actual)),
            "read checksum",
        )
        expected = int(self.library.uvm512_dataset_expected_checksum(handle))
        return int(actual.value), expected

    def read_stats(self) -> dict[str, int]:
        return parse_stats(self.stats_path.read_text(encoding="utf-8"))

    def preallocate(self, entries: Sequence[Sequence[int]]) -> dict[str, object]:
        result = self.uvm.driver_slot_preallocate_ranges(
            entries, device=self.device
        )
        if bool(result.get("ok")):
            self.closed = False
        return result

    def register_plan(
        self,
        plan_id: int,
        layer_id: int,
        entries: Sequence[Sequence[int]],
        *,
        transition: bool,
    ) -> dict[str, object]:
        return self.uvm.driver_slot_register_plan(
            plan_id,
            layer_id,
            entries,
            transition=transition,
            device=self.device,
        )

    def _raw_entries(
        self, entries: Sequence[Sequence[int]], *, mode: int
    ) -> list[tuple[int, ...]]:
        normalized, validation_error = self.uvm._normalize_moe_slot_ranges(entries)
        require(not validation_error, f"raw entry validation failed: {validation_error}")
        require(
            len(normalized) <= self.uvm._UVM_MOE_SLOT_REMAP_MAX_ENTRIES,
            "raw entry count exceeds the ABI",
        )
        if mode == MODE_TRANSITION:
            policy_flags, policy_error = self.uvm._slot_direct_policy_flags()
            require(not policy_error, f"raw transition policy failed: {policy_error}")
            normalized = [
                (*record[:6], record[6] | policy_flags)
                for record in normalized
            ]
        return [tuple(map(int, record)) for record in normalized]

    def _raw_gpu_uuid(self, *, mismatch: bool = False):
        gpu_uuid = self.uvm._phase_f_gpu_uuid(self.device)
        if not mismatch:
            return gpu_uuid
        mutated = bytearray(bytes(gpu_uuid.uuid))
        require(len(mutated) == 16, "unexpected raw GPU UUID length")
        mutated[0] ^= 0x80
        result = self.uvm._PhaseFUuid()
        for index, value in enumerate(mutated):
            result.uuid[index] = value
        return result

    def register_plan_raw(
        self,
        plan_id: int,
        layer_id: int,
        entries: Sequence[Sequence[int]],
        *,
        mode: int,
        count_override: int | None = None,
        uuid_mismatch: bool = False,
    ) -> dict[str, object]:
        normalized = self._raw_entries(entries, mode=mode)
        count = len(normalized) if count_override is None else int(count_override)
        require(0 < count <= len(normalized), "invalid raw register count override")
        gpu_uuid = self._raw_gpu_uuid(mismatch=uuid_mismatch)
        with self.uvm._uvm_slot_direct_lock:
            cached = self.uvm._uvm_slot_direct_sessions.get(self.device)
            require(cached is not None, "slot session disappeared before raw register")
            fd, _cached_uuid = cached
            params = self.uvm._MoeSlotPlanRegisterParams()
            params.plan_id = int(plan_id)
            params.layer_id = int(layer_id)
            params.mode = int(mode)
            params.count = count
            params.gpu_uuid = gpu_uuid
            for index, record in enumerate(normalized):
                base, length, driver_layer, expert, slot, offset, flags = record
                entry = params.entries[index]
                entry.layer_id = driver_layer
                entry.expert_id = expert
                entry.slot_id = slot
                entry.va_base = base
                entry.bytes = length
                entry.slot_offset = offset
                entry.flags = flags
            ctypes.set_errno(0)
            started_ns = time.perf_counter_ns()
            ret = self.uvm._libc_ioctl(
                fd, self.uvm._UVM_MOE_SLOT_PLAN_REGISTER_IOCTL, params
            )
            wall_ns = time.perf_counter_ns() - started_ns
            errno_value = ctypes.get_errno() if ret != 0 else 0
        rm_status = int(params.rmStatus)
        return {
            "ok": ret == 0 and rm_status == 0,
            "reason": "unexpected_accept" if ret == 0 and rm_status == 0 else "rejected",
            "rejected": not (ret == 0 and rm_status == 0),
            "plan_id": int(plan_id),
            "layer_id": int(layer_id),
            "mode": int(mode),
            "generation": int(params.generation),
            "requested_count": count,
            "requested_bytes": int(params.requested_bytes),
            "ioctl_ret": int(ret),
            "errno": errno_value,
            "rm_status": rm_status,
            "ioctl_calls": 1,
            "ioctl_wall_ns": wall_ns,
        }

    def execute_plan(
        self, plan_id: int, layer_id: int, generation: int
    ) -> dict[str, object]:
        return self.uvm.driver_slot_execute_plan(
            plan_id, layer_id, generation, device=self.device
        )

    def execute_stale_raw(
        self, plan_id: int, layer_id: int, generation: int
    ) -> dict[str, object]:
        # The public wrapper intentionally rejects stale userspace generations
        # before ioctl. This raw call is confined to the probe so the kernel's
        # own generation gate is exercised without changing the production API.
        return self._execute_plan_raw(
            plan_id,
            layer_id,
            generation,
            reason="expected_stale_generation_rejection",
        )

    def execute_plan_raw(
        self, plan_id: int, layer_id: int, generation: int
    ) -> dict[str, object]:
        return self._execute_plan_raw(
            plan_id, layer_id, generation, reason="expected_negative_rejection"
        )

    def _execute_plan_raw(
        self,
        plan_id: int,
        layer_id: int,
        generation: int,
        *,
        reason: str,
    ) -> dict[str, object]:
        with self.uvm._uvm_slot_direct_lock:
            cached = self.uvm._uvm_slot_direct_sessions.get(self.device)
            require(cached is not None, "slot session disappeared before raw execute")
            fd, _uuid = cached
            params = self.uvm._MoeSlotPlanExecuteParams()
            params.plan_id = int(plan_id)
            params.layer_id = int(layer_id)
            params.generation = int(generation)
            ctypes.set_errno(0)
            started_ns = time.perf_counter_ns()
            ret = self.uvm._libc_ioctl(
                fd, self.uvm._UVM_MOE_SLOT_PLAN_EXECUTE_IOCTL, params
            )
            wall_ns = time.perf_counter_ns() - started_ns
            errno_value = ctypes.get_errno() if ret != 0 else 0
        rm_status = int(params.rmStatus)
        return {
            "ok": False,
            "reason": reason,
            "rejected": not (ret == 0 and rm_status == 0),
            "plan_id": int(plan_id),
            "layer_id": int(layer_id),
            "generation": int(generation),
            "new_generation": int(params.new_generation),
            "requested_bytes": int(params.requested_bytes),
            "accepted_bytes": int(params.accepted_bytes),
            "mode": int(params.mode),
            "requested_count": int(params.count),
            "success_count": int(params.success_count),
            "failure_count": int(params.failure_count),
            "conflict_count": int(params.conflict_count),
            "remap_count": int(params.remap_count),
            "tlb_invalidate_count": int(params.tlb_invalidate_count),
            "copy_push_count": int(params.copy_push_count),
            "owner_cutover": int(params.owner_cutover),
            "poisoned": bool(params.poisoned),
            "ioctl_ret": int(ret),
            "errno": errno_value,
            "rm_status": rm_status,
            "ioctl_calls": 1,
            "ioctl_wall_ns": wall_ns,
        }

    def legacy_transition_raw(
        self, entries: Sequence[Sequence[int]]
    ) -> dict[str, object]:
        normalized = self._raw_entries(entries, mode=MODE_TRANSITION)
        gpu_uuid = self._raw_gpu_uuid()
        with self.uvm._uvm_slot_direct_lock:
            cached = self.uvm._uvm_slot_direct_sessions.get(self.device)
            require(cached is not None, "slot session disappeared before legacy ioctl")
            fd, _cached_uuid = cached
            params = self.uvm._MoeSlotRemapBatchParams()
            params.mode = MODE_TRANSITION
            params.count = len(normalized)
            params.gpu_uuid = gpu_uuid
            params.validate_managed = 1
            for index, record in enumerate(normalized):
                base, length, driver_layer, expert, slot, offset, flags = record
                entry = params.entries[index]
                entry.layer_id = driver_layer
                entry.expert_id = expert
                entry.slot_id = slot
                entry.va_base = base
                entry.bytes = length
                entry.slot_offset = offset
                entry.flags = flags
            ctypes.set_errno(0)
            started_ns = time.perf_counter_ns()
            ret = self.uvm._libc_ioctl(
                fd, self.uvm._UVM_MOE_SLOT_REMAP_BATCH_IOCTL, params
            )
            wall_ns = time.perf_counter_ns() - started_ns
            errno_value = ctypes.get_errno() if ret != 0 else 0
        rm_status = int(params.rmStatus)
        return {
            "ok": ret == 0 and rm_status == 0,
            "reason": "unexpected_accept" if ret == 0 and rm_status == 0 else "rejected",
            "rejected": not (ret == 0 and rm_status == 0),
            "mode": MODE_TRANSITION,
            "requested_count": int(params.count),
            "success_count": int(params.success_count),
            "failure_count": int(params.failure_count),
            "conflict_count": int(params.conflict_count),
            "requested_bytes": int(params.requested_bytes),
            "accepted_bytes": int(params.accepted_bytes),
            "remap_count": int(params.remap_count),
            "tlb_invalidate_count": int(params.tlb_invalidate_count),
            "copy_push_count": int(params.reserved),
            "owner_cutover": int(params.remap_count) == len(normalized),
            "ioctl_ret": int(ret),
            "errno": errno_value,
            "rm_status": rm_status,
            "ioctl_calls": 1,
            "ioctl_wall_ns": wall_ns,
        }

    def destroy_plan(self, plan_id: int, generation: int) -> dict[str, object]:
        return self.uvm.driver_slot_destroy_plan(
            plan_id, generation, device=self.device
        )

    def close(self) -> dict[str, object]:
        result = self.uvm.close_uvm_slot_direct_session(device=self.device)
        if bool(result.get("ok")):
            self.closed = True
        return result

    def release_ranges(
        self, entries: Sequence[Sequence[int]]
    ) -> dict[str, object]:
        return self.uvm.driver_slot_release_ranges(entries, device=self.device)

    def destroy_dataset(self, dataset: object) -> None:
        handle = int(dataset)
        require(handle in self.datasets, "attempted to destroy an unknown dataset")
        self.library.uvm512_dataset_destroy(handle)
        self.datasets.remove(handle)

    def state_snapshot(self) -> dict[str, object]:
        with self.uvm._uvm_slot_direct_lock:
            return {
                "allocated_slots": len(
                    self.uvm._uvm_slot_direct_owners.get(self.device, {})
                ),
                "plan_count": len(
                    self.uvm._uvm_slot_direct_plans.get(self.device, {})
                ),
                "tracked_generation": self.uvm._uvm_slot_direct_plan_generations.get(
                    self.device
                ),
                "session_open": self.device in self.uvm._uvm_slot_direct_sessions,
            }

    def finish(self) -> dict[str, object]:
        errors: list[str] = []
        emergency_close: dict[str, object] | None = None
        if not self.closed:
            try:
                emergency_close = self.close()
                if not bool(emergency_close.get("ok")):
                    errors.append(f"emergency close failed: {emergency_close}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"emergency close raised: {exc!r}")
        if self.stream:
            try:
                self.library.uvm512_stream_destroy(self.stream)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"stream destroy raised: {exc!r}")
            self.stream = 0
        for dataset in reversed(self.datasets):
            try:
                self.library.uvm512_dataset_destroy(dataset)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"dataset destroy raised: {exc!r}")
        self.datasets.clear()
        return {
            "ok": not errors,
            "errors": errors,
            "emergency_close": emergency_close,
        }


def read_integer(path: Path, name: str) -> int:
    require(path.is_file(), f"missing {name}: {path}")
    try:
        return int(path.read_text(encoding="utf-8").strip(), 0)
    except ValueError as exc:
        raise ProbeError(f"invalid {name}: {path}") from exc


def run_gpu_worker(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "status": "failed",
        "worker": True,
        "negative_case": args.negative_case,
    }
    backend: GpuBackend | None = None
    try:
        require(args.library.is_file(), f"missing CUDA shim: {args.library}")
        require(args.stats.is_file(), f"missing latency stats: {args.stats}")
        fast_index = read_integer(args.fast_index_parameter, "fast-index parameter")
        direct_cutover = read_integer(
            args.direct_cutover_parameter, "direct-cutover parameter"
        )
        fast_validation = read_integer(
            args.fast_validation_parameter, "fast-validation parameter"
        )
        budget = read_integer(args.budget_parameter, "slot VIDMEM budget")
        worker_blocks = 1 if args.negative_case is not None else int(args.blocks)
        logical_bytes = worker_blocks * BLOCK_BYTES
        require(fast_index == 1, "uvm_moe_slot_transition_fast_index must already be 1")
        require(direct_cutover == 1,
                "uvm_moe_slot_transition_direct_cutover must already be 1")
        require(fast_validation == 1,
                "uvm_moe_slot_plan_fast_validation must already be 1")
        require(budget >= logical_bytes,
                f"slot VIDMEM budget {budget} is below required {logical_bytes}")
        require(args.srcversion.is_file(), f"missing module srcversion: {args.srcversion}")
        payload.update(
            {
                "module_srcversion": args.srcversion.read_text(
                    encoding="utf-8"
                ).strip(),
                "module_parameters": {
                    "fast_index": fast_index,
                    "direct_cutover": direct_cutover,
                    "fast_validation": fast_validation,
                    "vidmem_budget_bytes": budget,
                },
            }
        )
        backend = GpuBackend(
            library=args.library, stats=args.stats, device=args.device
        )
        if args.negative_case is None:
            payload.update(exercise_registered_plans(backend, blocks=args.blocks))
        else:
            require(args.blocks == 1, "negative workers must use exactly one block")
            payload.update(exercise_negative_case(backend, case=args.negative_case))
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if backend is not None:
            payload["worker_cleanup"] = backend.finish()
            if not bool(payload["worker_cleanup"].get("ok")):
                payload["status"] = "failed"
                payload.setdefault("error", "worker cleanup failed")

    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if payload.get("status") == "pass" else 1


def gpu_used_mib(device: int) -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(device),
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
    )
    return int(output.strip().splitlines()[0])


def wait_for_zero_mib(device: int, timeout_seconds: float) -> tuple[int, list[int]]:
    deadline = time.monotonic() + timeout_seconds
    observations: list[int] = []
    while True:
        current = gpu_used_mib(device)
        observations.append(current)
        if current == 0 or time.monotonic() >= deadline:
            return current, observations
        time.sleep(0.25)


def run_controller(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "status": "failed",
        "device": args.device,
        "blocks": args.blocks,
        "require_zero_gpu_memory": True,
    }
    try:
        require(args.output is not None, "--output is required")
        require(1 <= args.blocks <= MAX_BLOCKS,
                f"blocks must be in [1, {MAX_BLOCKS}]")
        require(subprocess.call(
            ["sh", "-c", "command -v nvidia-smi >/dev/null 2>&1"]
        ) == 0, "nvidia-smi is unavailable")
        initial_mib = gpu_used_mib(args.device)
        payload["gpu_memory_initial_mib"] = initial_mib
        require(initial_mib == 0,
                f"GPU {args.device} is not idle before the probe: {initial_mib} MiB")

        with tempfile.TemporaryDirectory(prefix="uvm-registered-plan-probe-") as temp:
            temp_path = Path(temp)

            def launch_worker(
                label: str, *, blocks: int, negative_case: str | None
            ) -> dict[str, object]:
                worker_output = temp_path / f"{label}.json"
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-output",
                    str(worker_output),
                    "--library",
                    str(args.library),
                    "--stats",
                    str(args.stats),
                    "--fast-index-parameter",
                    str(args.fast_index_parameter),
                    "--direct-cutover-parameter",
                    str(args.direct_cutover_parameter),
                    "--fast-validation-parameter",
                    str(args.fast_validation_parameter),
                    "--budget-parameter",
                    str(args.budget_parameter),
                    "--srcversion",
                    str(args.srcversion),
                    "--device",
                    str(args.device),
                    "--blocks",
                    str(blocks),
                ]
                if negative_case is not None:
                    command.extend(["--negative-case", negative_case])
                completed = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    timeout=args.worker_timeout_seconds,
                    check=False,
                )
                if worker_output.is_file():
                    result = json.loads(worker_output.read_text(encoding="utf-8"))
                else:
                    result = {
                        "status": "failed",
                        "error": "worker produced no JSON artifact",
                    }
                final_mib, observations = wait_for_zero_mib(
                    args.device, args.zero_memory_timeout_seconds
                )
                record: dict[str, object] = {
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "result": result,
                    "gpu_memory_final_mib": final_mib,
                    "gpu_memory_observations_mib": observations,
                    "zero_mib_after_worker_exit": final_mib == 0,
                }
                require(final_mib == 0, f"{label} left {final_mib} MiB allocated")
                return record

            happy_record = launch_worker(
                "happy", blocks=args.blocks, negative_case=None
            )
            payload["worker_command"] = happy_record["command"]
            payload["worker_returncode"] = happy_record["returncode"]
            payload["worker_stdout"] = happy_record["stdout"]
            payload["worker_stderr"] = happy_record["stderr"]
            worker_payload = happy_record["result"]
            require(isinstance(worker_payload, Mapping), "invalid happy worker result")
            payload["worker_result"] = worker_payload
            require(happy_record["returncode"] == 0, "happy GPU worker failed")
            require(worker_payload.get("status") == "pass", "happy worker artifact failed")

            negative_records: dict[str, dict[str, object]] = {}
            payload["negative_worker_results"] = negative_records
            for negative_case in NEGATIVE_CASES:
                record = launch_worker(
                    f"negative_{negative_case}",
                    blocks=1,
                    negative_case=negative_case,
                )
                negative_records[negative_case] = record
                result = record["result"]
                require(isinstance(result, Mapping),
                        f"{negative_case}: invalid worker result")
                require(record["returncode"] == 0,
                        f"negative worker {negative_case} failed")
                require(result.get("status") == "pass",
                        f"negative artifact {negative_case} failed")
                require(result.get("negative_case") == negative_case,
                        f"negative artifact {negative_case} mislabeled itself")
                require(result.get("zero_slots_after_close") is True,
                        f"negative worker {negative_case} leaked slots")

        last_record = negative_records[NEGATIVE_CASES[-1]]
        final_mib = int(last_record["gpu_memory_final_mib"])
        observations = last_record["gpu_memory_observations_mib"]
        payload["gpu_memory_final_mib"] = final_mib
        payload["gpu_memory_final_observations_mib"] = observations
        payload["zero_mib_after_worker_exit"] = final_mib == 0
        post_close = worker_payload.get("post_close_state", {})
        payload["zero_slots_after_close"] = (
            isinstance(post_close, Mapping)
            and int(post_close.get("allocated_slots", -1)) == 0
        )
        payload["lifecycle_control_delta"] = worker_payload.get(
            "lifecycle_control_delta"
        )
        expected_lifecycle = {
                "slot_plan_register_calls": 5,
                "slot_plan_execute_calls": 6,
                "slot_plan_destroy_calls": 5,
                "slot_legacy_ioctl_calls": int(
                    worker_payload.get("preallocate_legacy_ioctl_calls", 1)
                )
                + 1,
                "slot_legacy_transition_calls": 0,
            }
        require(
            payload["lifecycle_control_delta"] == expected_lifecycle,
            "worker control-plane summary is not exact",
        )
        require(payload["zero_slots_after_close"] is True,
                "worker did not prove zero slots after close")
        payload["negative_cases_passed"] = list(NEGATIVE_CASES)
        require(final_mib == 0,
                f"GPU {args.device} still uses {final_mib} MiB after worker exit")
        payload["status"] = "pass"
    except Exception as exc:  # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()

    assert args.output is not None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "pass" else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--negative-case", choices=NEGATIVE_CASES, help=argparse.SUPPRESS
    )
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument(
        "--fast-index-parameter", type=Path, default=DEFAULT_FAST_INDEX
    )
    parser.add_argument(
        "--direct-cutover-parameter", type=Path, default=DEFAULT_DIRECT_CUTOVER
    )
    parser.add_argument(
        "--fast-validation-parameter", type=Path, default=DEFAULT_FAST_VALIDATION
    )
    parser.add_argument("--budget-parameter", type=Path, default=DEFAULT_BUDGET)
    parser.add_argument("--srcversion", type=Path, default=DEFAULT_SRCVERSION)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--worker-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--zero-memory-timeout-seconds", type=float, default=15.0)
    args = parser.parse_args(argv)
    if args.device < 0 or not 1 <= args.blocks <= MAX_BLOCKS:
        parser.error(f"require device>=0 and blocks in [1, {MAX_BLOCKS}]")
    if args.worker_timeout_seconds <= 0 or args.zero_memory_timeout_seconds < 0:
        parser.error("timeouts must be non-negative and worker timeout must be positive")
    if args.worker_output is None and args.output is None:
        parser.error("--output is required in controller mode")
    if args.worker_output is None and args.negative_case is not None:
        parser.error("--negative-case is internal to worker mode")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_output is not None:
        return run_gpu_worker(args)
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
