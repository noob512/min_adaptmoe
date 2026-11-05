#!/usr/bin/env python3
"""Compare packed and fully-fragmented registered-plan VA geometries.

The probe does not load a module or change module parameters.  It runs one
happy dual-bank lifecycle in an isolated worker, records every execute wall
time and latency_stats delta, validates every generation/checksum, closes the
session, and requires the worker process to return the selected GPU to 0 MiB.

Five source layouts exercise complete and partial VA-block geometry:

* packed: 288 independent 2 MiB managed allocations, split into four entries;
* gapped: 288 independent 4 MiB allocations with only the first 2 MiB active,
  proving that global VA continuity is not required for complete blocks;
* fragmented: 1,152 independent 2 MiB managed allocations with only the first
  512 KiB active, so every envelope occupies a distinct partial VA block;
* sparse-experts: a selectable non-contiguous subset of 128 experts, with the
  Qwen shape of nine independent 512 KiB envelopes per selected expert.
* qwen-packed-sparse: the same selectable subset cut from Qwen's real two
  contiguous FP8 tensors: w13=[128,1536,2048] and w2=[128,2048,768].

``64k`` names the complete-block control. ``partial64k`` uses the same existing
mode-21 bootstrap and mode-20 + 64K transition policy; it is only a stricter
probe label which requires partial-block 64 KiB success and forbids a silent
registered 4 KiB fallback. No new UAPI mode or flag is introduced.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
BASE_PROBE_PATH = (
    REPO_ROOT / "experiments" / "uvm_slot_registered_plan_probe" / "run_probe.py"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "artifacts" / "fragmented_plan.json"
DEFAULT_GUARD_LIBRARY = SCRIPT_DIR / "libuvm_fragment_guard.so"


def load_base_probe():
    spec = importlib.util.spec_from_file_location(
        "fragmented_registered_plan_base", BASE_PROBE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_PROBE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_probe()

ENTRY_COUNT = 1152
LOGICAL_BYTES = ENTRY_COUNT * base.SLICE_BYTES
PACKED_ALLOCATION_COUNT = ENTRY_COUNT // base.SLICES_PER_BLOCK
TOTAL_EXPERTS = 128
W13_ENVELOPES_PER_EXPERT = 6
W2_ENVELOPES_PER_EXPERT = 3
ENVELOPES_PER_EXPERT = W13_ENVELOPES_PER_EXPERT + W2_ENVELOPES_PER_EXPERT
SPARSE_LAYOUTS = ("sparse-experts", "qwen-packed-sparse")
PTE_64K_BYTES = 64 * 1024
PTE_64K_PER_ENVELOPE = base.SLICE_BYTES // PTE_64K_BYTES
PTE_64K_PER_BLOCK = base.BLOCK_BYTES // PTE_64K_BYTES
GAPPED_ALLOCATION_BYTES = 2 * base.BLOCK_BYTES
FRAGMENT_ALLOCATION_BYTES = base.BLOCK_BYTES
FRAGMENT_GUARD_BYTES = 0
ACTIVE_OFFSET_STEP_BYTES = base.SLICE_BYTES
ACTIVE_OFFSET_PERIOD = FRAGMENT_ALLOCATION_BYTES // ACTIVE_OFFSET_STEP_BYTES
QWEN_W13_SHAPE = (TOTAL_EXPERTS, 1536, 2048)
QWEN_W2_SHAPE = (TOTAL_EXPERTS, 2048, 768)
QWEN_W13_EXPERT_BYTES = W13_ENVELOPES_PER_EXPERT * base.SLICE_BYTES
QWEN_W2_EXPERT_BYTES = W2_ENVELOPES_PER_EXPERT * base.SLICE_BYTES
QWEN_W13_TENSOR_BYTES = TOTAL_EXPERTS * QWEN_W13_EXPERT_BYTES
QWEN_W2_TENSOR_BYTES = TOTAL_EXPERTS * QWEN_W2_EXPERT_BYTES
QWEN_MAIN_TENSOR_BYTES = QWEN_W13_TENSOR_BYTES + QWEN_W2_TENSOR_BYTES
CUDA_CPU_DEVICE_ID = -1
QWEN_TENSOR_OFFSET_CYCLE = tuple(
    index * base.SLICE_BYTES for index in range(base.SLICES_PER_BLOCK)
)
MODE_INITIAL_4K = 11
MODE_INITIAL_64K = base.MODE_INITIAL
DEFAULT_PARTIAL_64K_INITIAL_MODE = base.MODE_INITIAL
EXECUTION_ORDER = (0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0)
# Index 0 is bootstrap. Indices 1-3 install L1-L3 hierarchy for the first
# time. From index 4 onward every layer has already executed once, so the 4K
# arm no longer charges lazy PTE hierarchy allocation to steady state.
STEADY_INDICES = tuple(range(4, len(EXECUTION_ORDER)))

DEFAULT_DUAL_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_dual_bank_event"
)
DEFAULT_CACHE_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_cache_copy_runs"
)
DEFAULT_PREWARM_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_register_pte_prewarm"
)
DEFAULT_OVERLAP_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_overlap_copy_prepare"
)
DEFAULT_RECIPE_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_cache_metadata_recipe"
)
DEFAULT_FAST_VALIDATION_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_fast_validation"
)
DEFAULT_EPOCH_VALIDATION_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_epoch_validation"
)
DEFAULT_SHADOW_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_shadow_pte_prefill"
)
DEFAULT_CUDA_FENCE_PARAM = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_plan_cuda_fence"
)


class ProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkloadShape:
    layout: str
    expert_ids: tuple[int, ...]
    entry_count: int
    logical_bytes: int

    @property
    def sparse(self) -> bool:
        return self.layout in SPARSE_LAYOUTS


class ComplementGuard:
    """GPU checksum context for the unowned 1.5 MiB of partial VA blocks."""

    def __init__(
        self,
        *,
        library: Path,
        backend: Any,
        pointers: Sequence[int],
        offsets: Sequence[int] = (),
        seed: int,
    ) -> None:
        require(library.is_file(), f"missing partial complement shim: {library}")
        self.backend = backend
        self.library = base.ctypes.CDLL(str(library))
        void_p = base.ctypes.c_void_p
        size_t = base.ctypes.c_size_t
        uint64_p = base.ctypes.POINTER(base.ctypes.c_uint64)
        self.library.uvm_fragment_guard_create.argtypes = [
            uint64_p,
            size_t,
            size_t,
            size_t,
            base.ctypes.c_uint32,
            void_p,
        ]
        self.library.uvm_fragment_guard_create.restype = void_p
        self.library.uvm_fragment_guard_create_offset.argtypes = [
            uint64_p,
            base.ctypes.POINTER(size_t),
            size_t,
            size_t,
            size_t,
            base.ctypes.c_uint32,
            void_p,
        ]
        self.library.uvm_fragment_guard_create_offset.restype = void_p
        self.library.uvm_fragment_guard_restore_active_cpu.argtypes = [
            void_p,
            void_p,
        ]
        self.library.uvm_fragment_guard_restore_active_cpu.restype = (
            base.ctypes.c_int
        )
        self.library.uvm_fragment_guard_destroy.argtypes = [void_p]
        self.library.uvm_fragment_guard_destroy.restype = None
        self.library.uvm_fragment_guard_launch_checksum.argtypes = [void_p, void_p]
        self.library.uvm_fragment_guard_launch_checksum.restype = base.ctypes.c_int
        self.library.uvm_fragment_guard_read_checksum.argtypes = [void_p, uint64_p]
        self.library.uvm_fragment_guard_read_checksum.restype = base.ctypes.c_int
        self.library.uvm_fragment_guard_expected_checksum.argtypes = [void_p]
        self.library.uvm_fragment_guard_expected_checksum.restype = base.ctypes.c_uint64

        pointer_array_type = base.ctypes.c_uint64 * len(pointers)
        pointer_array = pointer_array_type(*(int(pointer) for pointer in pointers))
        selected_offsets = tuple(int(offset) for offset in offsets)
        require(
            not selected_offsets or len(selected_offsets) == len(pointers),
            "partial complement offset count mismatch",
        )
        require(
            all(
                offset >= 0
                and offset % PTE_64K_BYTES == 0
                and offset + base.SLICE_BYTES <= FRAGMENT_ALLOCATION_BYTES
                for offset in selected_offsets
            ),
            "partial complement contains an invalid active offset",
        )
        if selected_offsets:
            offset_array_type = size_t * len(selected_offsets)
            offset_array = offset_array_type(*selected_offsets)
            handle = self.library.uvm_fragment_guard_create_offset(
                pointer_array,
                offset_array,
                len(pointers),
                FRAGMENT_ALLOCATION_BYTES,
                base.SLICE_BYTES,
                int(seed) & 0xFFFFFFFF,
                backend.stream,
            )
        else:
            selected_offsets = (0,) * len(pointers)
            handle = self.library.uvm_fragment_guard_create(
                pointer_array,
                len(pointers),
                FRAGMENT_ALLOCATION_BYTES,
                base.SLICE_BYTES,
                int(seed) & 0xFFFFFFFF,
                backend.stream,
            )
        self.handle = int(handle or 0)
        require(self.handle != 0, "partial complement context creation failed")
        self.expected = int(
            self.library.uvm_fragment_guard_expected_checksum(self.handle)
        )
        self.bytes = len(pointers) * (
            FRAGMENT_ALLOCATION_BYTES - base.SLICE_BYTES
        )
        self.active_range_count = len(pointers)
        self.active_bytes = len(pointers) * base.SLICE_BYTES
        self.active_offsets = selected_offsets
        self.complement_prefix_bytes = sum(selected_offsets)
        self.complement_suffix_bytes = sum(
            FRAGMENT_ALLOCATION_BYTES - offset - base.SLICE_BYTES
            for offset in selected_offsets
        )

    def restore_active_cpu(self) -> None:
        self.backend._check_cuda(  # noqa: SLF001 - probe CUDA shim
            self.library.uvm_fragment_guard_restore_active_cpu(
                self.handle, self.backend.stream
            ),
            "restore active fragmented sources to CPU",
        )

    def launch(self) -> None:
        self.backend._check_cuda(  # noqa: SLF001 - probe CUDA shim
            self.library.uvm_fragment_guard_launch_checksum(
                self.handle, self.backend.stream
            ),
            "launch partial complement checksum",
        )

    def read(self) -> int:
        actual = base.ctypes.c_uint64()
        self.backend._check_cuda(  # noqa: SLF001 - probe CUDA shim
            self.library.uvm_fragment_guard_read_checksum(
                self.handle, base.ctypes.byref(actual)
            ),
            "read partial complement checksum",
        )
        return int(actual.value)

    def close(self) -> None:
        if self.handle:
            self.library.uvm_fragment_guard_destroy(self.handle)
            self.handle = 0


def restore_guard_active_sources_cpu(
    guards: Mapping[int, ComplementGuard],
) -> list[dict[str, int]]:
    """Undo guard-induced GPU residency on only the active source ranges."""
    records: list[dict[str, int]] = []
    for layer_id in sorted(guards):
        guard = guards[layer_id]
        guard.restore_active_cpu()
        records.append(
            {
                "layer_id": layer_id,
                "range_count": guard.active_range_count,
                "bytes": guard.active_bytes,
                "complement_prefix_bytes": guard.complement_prefix_bytes,
                "complement_suffix_bytes": guard.complement_suffix_bytes,
            }
        )
    return records


def active_offsets(entry_count: int, *, cycle: bool) -> tuple[int, ...]:
    require(entry_count > 0, "active offset entry count must be positive")
    if not cycle:
        return (0,) * entry_count
    return tuple(
        (index % ACTIVE_OFFSET_PERIOD) * ACTIVE_OFFSET_STEP_BYTES
        for index in range(entry_count)
    )


def active_offset_summary(offsets: Sequence[int]) -> dict[str, Any]:
    require(offsets, "active offset summary requires at least one range")
    histogram = {
        str(offset): sum(int(value) == offset for value in offsets)
        for offset in sorted({int(value) for value in offsets})
    }
    return {
        "cycle_bytes": [
            index * ACTIVE_OFFSET_STEP_BYTES
            for index in range(ACTIVE_OFFSET_PERIOD)
        ],
        "histogram": histogram,
        "nonzero_offset_ranges": sum(int(offset) != 0 for offset in offsets),
        "two_mib_aligned_ranges": sum(int(offset) == 0 for offset in offsets),
    }


def qwen_tensor_offsets(layer_index: int, *, cycle: bool) -> tuple[int, int]:
    """Return requested 2 MiB-relative w13/w2 tensor-base offsets.

    The real tensors are separate allocations.  CUDA commonly happens to make
    their bases 2 MiB aligned, but the registered-plan contract only requires
    512 KiB alignment.  The optional cycle therefore covers every legal base
    residue without changing the packed tensor strides.
    """
    if not cycle:
        return 0, 0
    index = int(layer_index) % len(QWEN_TENSOR_OFFSET_CYCLE)
    return (
        QWEN_TENSOR_OFFSET_CYCLE[index],
        QWEN_TENSOR_OFFSET_CYCLE[(index + 1) % len(QWEN_TENSOR_OFFSET_CYCLE)],
    )


def qwen_packed_region_pointers(
    w13_base: int,
    w2_base: int,
    expert_ids: Sequence[int],
) -> tuple[int, ...]:
    """Materialize expert-major 6+3 region VAs from two packed tensors."""
    require_sparse_expert_ids(
        expert_ids, label="Qwen packed", require_gap=False
    )
    w13_base = int(w13_base)
    w2_base = int(w2_base)
    require(
        w13_base > 0
        and w2_base > 0
        and w13_base % base.SLICE_BYTES == 0
        and w2_base % base.SLICE_BYTES == 0,
        "Qwen tensor bases must be positive and 512 KiB aligned",
    )
    w13_end = w13_base + QWEN_W13_TENSOR_BYTES
    w2_end = w2_base + QWEN_W2_TENSOR_BYTES
    require(
        w13_end <= w2_base or w2_end <= w13_base,
        "Qwen w13 and w2 tensor ranges overlap",
    )

    pointers: list[int] = []
    for raw_expert_id in expert_ids:
        expert_id = int(raw_expert_id)
        w13_expert = w13_base + expert_id * QWEN_W13_EXPERT_BYTES
        pointers.extend(
            w13_expert + envelope * base.SLICE_BYTES
            for envelope in range(W13_ENVELOPES_PER_EXPERT)
        )
        w2_expert = w2_base + expert_id * QWEN_W2_EXPERT_BYTES
        pointers.extend(
            w2_expert + envelope * base.SLICE_BYTES
            for envelope in range(W2_ENVELOPES_PER_EXPERT)
        )
    return tuple(pointers)


def qwen_tensor_bases_from_regions(
    pointers: Sequence[int], expert_ids: Sequence[int]
) -> tuple[int, int]:
    """Recover and validate tensor bases returned by the CUDA dataset ABI."""
    require_sparse_expert_ids(
        expert_ids, label="Qwen packed", require_gap=False
    )
    require(
        len(pointers) == len(expert_ids) * ENVELOPES_PER_EXPERT,
        "Qwen packed region pointer count mismatch",
    )
    first_expert = int(expert_ids[0])
    w13_base = int(pointers[0]) - first_expert * QWEN_W13_EXPERT_BYTES
    w2_base = (
        int(pointers[W13_ENVELOPES_PER_EXPERT])
        - first_expert * QWEN_W2_EXPERT_BYTES
    )
    require(
        tuple(int(pointer) for pointer in pointers)
        == qwen_packed_region_pointers(w13_base, w2_base, expert_ids),
        "Qwen packed dataset does not match contiguous tensor strides",
    )
    return w13_base, w2_base


def qwen_tensor_layout_summary(
    pointers: Sequence[int], expert_ids: Sequence[int]
) -> dict[str, Any]:
    w13_base, w2_base = qwen_tensor_bases_from_regions(pointers, expert_ids)
    return {
        "dtype": "torch.float8_e4m3fn",
        "envelope_bytes": base.SLICE_BYTES,
        "full_tensor_checksum_bytes": QWEN_MAIN_TENSOR_BYTES,
        "w13": {
            "shape": list(QWEN_W13_SHAPE),
            "stride": [QWEN_W13_EXPERT_BYTES, QWEN_W13_SHAPE[2], 1],
            "bytes": QWEN_W13_TENSOR_BYTES,
            "expert_stride_bytes": QWEN_W13_EXPERT_BYTES,
            "envelopes_per_expert": W13_ENVELOPES_PER_EXPERT,
            "base": w13_base,
            "base_offset_in_2m": w13_base % base.BLOCK_BYTES,
        },
        "w2": {
            "shape": list(QWEN_W2_SHAPE),
            "stride": [QWEN_W2_EXPERT_BYTES, QWEN_W2_SHAPE[2], 1],
            "bytes": QWEN_W2_TENSOR_BYTES,
            "expert_stride_bytes": QWEN_W2_EXPERT_BYTES,
            "envelopes_per_expert": W2_ENVELOPES_PER_EXPERT,
            "base": w2_base,
            "base_offset_in_2m": w2_base % base.BLOCK_BYTES,
        },
    }


def qwen_registration_residency_partition(
    pointers: Sequence[int], expert_ids: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return full, cold, and hot 512 KiB regions for one packed layer."""
    cold = tuple(int(pointer) for pointer in pointers)
    w13_base, w2_base = qwen_tensor_bases_from_regions(cold, expert_ids)
    full = qwen_packed_region_pointers(
        w13_base, w2_base, tuple(range(TOTAL_EXPERTS))
    )
    cold_set = set(cold)
    require(
        len(cold_set) == len(cold) and cold_set.issubset(set(full)),
        "Qwen cold regions are not a unique subset of the packed tensors",
    )
    hot = tuple(pointer for pointer in full if pointer not in cold_set)
    require(
        len(full) == TOTAL_EXPERTS * ENVELOPES_PER_EXPERT
        and len(cold) == len(expert_ids) * ENVELOPES_PER_EXPERT
        and len(full) == len(cold) + len(hot),
        "Qwen full/cold/hot residency partition mismatch",
    )
    return full, cold, hot


def query_qwen_region_residency(
    backend: Any, pointers: Sequence[int], *, label: str
) -> dict[str, int | bool]:
    """Query exact page residency for a set of 512 KiB regions."""
    ranges = tuple((int(pointer), base.SLICE_BYTES) for pointer in pointers)
    result = backend.uvm.driver_batch_query_ranges_residency_bytes(
        ranges, device=backend.device, wait_for_trackers=True
    )
    expected_bytes = len(ranges) * base.SLICE_BYTES
    require(result.get("ok") is True, f"{label} residency query failed: {result}")
    # The production wrapper returns the number of successfully queried
    # ranges, not a zero errno. ok=True alone only means no negative wrapper
    # failure and can otherwise hide a partial result.
    require(
        int(result.get("status", -1)) == len(ranges),
        f"{label} residency range-count status failed",
    )
    require(
        int(result.get("queried_bytes", -1)) == expected_bytes,
        f"{label} residency queried-byte mismatch",
    )
    return {
        "ok": True,
        "range_count": len(ranges),
        "queried_bytes": int(result.get("queried_bytes", 0)),
        "gpu_bytes": int(result.get("gpu_bytes", 0)),
        "cpu_bytes": int(result.get("cpu_bytes", 0)),
        "any_bytes": int(result.get("any_bytes", 0)),
        "unknown_bytes": int(result.get("unknown_bytes", 0)),
    }


def prepare_qwen_registration_residency(
    backend: Any,
    datasets: Mapping[int, int],
    pointers: Mapping[int, Sequence[int]],
    shapes: Mapping[int, WorkloadShape],
) -> dict[str, Any]:
    """Model GPU-resident hot experts and CPU-resident cold regions.

    The checksum kernel covers all 1,152 envelopes in w13+w2, not merely the
    selected public cold pointers. Only after that full-tensor GPU touch is
    synchronized do we prefetch the selected 512 KiB regions back to CPU.
    No GPU work is submitted after the final residency queries, so this is the
    exact state consumed by bank preallocation and registered-plan creation.
    """
    layer_ids = tuple(sorted(int(layer_id) for layer_id in datasets))
    require(layer_ids, "Qwen residency setup requires at least one layer")
    require(
        set(layer_ids) == {int(layer_id) for layer_id in pointers}
        == {int(layer_id) for layer_id in shapes},
        "Qwen residency setup layer sets differ",
    )

    partitions: dict[int, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = {}
    records: dict[str, dict[str, Any]] = {}
    touch_started_ns = time.perf_counter_ns()
    for layer_id in layer_ids:
        partitions[layer_id] = qwen_registration_residency_partition(
            pointers[layer_id], shapes[layer_id].expert_ids
        )
        actual, expected = backend.checksum(datasets[layer_id])
        require(actual == expected, f"L{layer_id} full Qwen GPU-touch checksum mismatch")
        full, cold, hot = partitions[layer_id]
        records[str(layer_id)] = {
            "layer_id": layer_id,
            "full_region_count": len(full),
            "full_bytes": len(full) * base.SLICE_BYTES,
            "cold_region_count": len(cold),
            "cold_bytes": len(cold) * base.SLICE_BYTES,
            "hot_region_count": len(hot),
            "hot_bytes": len(hot) * base.SLICE_BYTES,
            "gpu_touch_checksum_actual": actual,
            "gpu_touch_checksum_expected": expected,
            "gpu_touch_checksum_ok": True,
        }

    for layer_id in layer_ids:
        full, _cold, _hot = partitions[layer_id]
        observed = query_qwen_region_residency(
            backend, full, label=f"L{layer_id} post-touch full tensor"
        )
        require(
            observed["gpu_bytes"] == QWEN_MAIN_TENSOR_BYTES
            and observed["any_bytes"] == QWEN_MAIN_TENSOR_BYTES
            and observed["unknown_bytes"] == 0,
            f"L{layer_id} full tensor did not become GPU-resident: {observed}",
        )
        records[str(layer_id)]["post_gpu_touch_residency"] = observed
    touch_finished_ns = time.perf_counter_ns()

    backend.library.uvm512_prefetch_counter_reset()
    prefetch_started_ns = time.perf_counter_ns()
    for layer_id in layer_ids:
        _full, cold, _hot = partitions[layer_id]
        pointer_array_type = base.ctypes.c_size_t * len(cold)
        size_array_type = base.ctypes.c_size_t * len(cold)
        pointer_array = pointer_array_type(*cold)
        size_array = size_array_type(*(base.SLICE_BYTES for _ in cold))
        accepted = int(
            backend.library.uvm512_prefetch_batch(
                pointer_array,
                size_array,
                len(cold),
                CUDA_CPU_DEVICE_ID,
                backend.stream,
            )
        )
        require(
            accepted == len(cold),
            f"L{layer_id} cold CPU prefetch accepted {accepted}/{len(cold)}",
        )
        records[str(layer_id)]["cold_prefetch_accepted"] = accepted

    backend._check_cuda(  # noqa: SLF001 - probe CUDA shim
        backend.library.uvm512_stream_synchronize(backend.stream),
        "synchronize exact Qwen cold-region CPU prefetch",
    )
    prefetch_finished_ns = time.perf_counter_ns()
    prefetch_counters = {
        "one_outer_calls": int(
            backend.library.uvm512_prefetch_one_outer_calls()
        ),
        "batch_outer_calls": int(
            backend.library.uvm512_prefetch_batch_outer_calls()
        ),
        "runtime_prefetch_calls": int(
            backend.library.uvm512_runtime_prefetch_calls()
        ),
    }
    expected_cold_regions = sum(len(partitions[layer_id][1]) for layer_id in layer_ids)
    require(
        prefetch_counters
        == {
            "one_outer_calls": 0,
            "batch_outer_calls": len(layer_ids),
            "runtime_prefetch_calls": expected_cold_regions,
        },
        f"Qwen cold prefetch call topology mismatch: {prefetch_counters}",
    )

    for layer_id in layer_ids:
        _full, cold, hot = partitions[layer_id]
        cold_observed = query_qwen_region_residency(
            backend, cold, label=f"L{layer_id} final cold regions"
        )
        hot_observed = query_qwen_region_residency(
            backend, hot, label=f"L{layer_id} final hot regions"
        )
        expected_cold_bytes = len(cold) * base.SLICE_BYTES
        expected_hot_bytes = len(hot) * base.SLICE_BYTES
        require(
            cold_observed["cpu_bytes"] == expected_cold_bytes
            and cold_observed["gpu_bytes"] == 0
            and cold_observed["any_bytes"] == expected_cold_bytes
            and cold_observed["unknown_bytes"] == 0,
            f"L{layer_id} cold regions are not exclusively CPU-resident: {cold_observed}",
        )
        require(
            hot_observed["gpu_bytes"] == expected_hot_bytes
            and hot_observed["any_bytes"] == expected_hot_bytes
            and hot_observed["unknown_bytes"] == 0,
            f"L{layer_id} hot regions did not remain GPU-resident: {hot_observed}",
        )
        records[str(layer_id)]["final_cold_residency"] = cold_observed
        records[str(layer_id)]["final_hot_residency"] = hot_observed

    return {
        "enabled": True,
        "operation_order": [
            "gpu_checksum_touch_full_w13_w2",
            "synchronize_and_validate_full_gpu_residency",
            "prefetch_selected_512k_cold_regions_to_cpu",
            "synchronize_cold_prefetch",
            "validate_cold_cpu_and_hot_gpu_residency",
        ],
        "region_bytes": base.SLICE_BYTES,
        "full_tensor_bytes_per_layer": QWEN_MAIN_TENSOR_BYTES,
        "layer_count": len(layer_ids),
        "touch_and_validate_wall_ns": touch_finished_ns - touch_started_ns,
        "cold_prefetch_and_sync_wall_ns": prefetch_finished_ns - prefetch_started_ns,
        "prefetch_counters": prefetch_counters,
        "layers": records,
        "contract": {
            "full_tensor_gpu_touch_before_cold_prefetch": True,
            "only_selected_cold_regions_prefetched_to_cpu": True,
            "cold_regions_exclusively_cpu_resident": True,
            "hot_regions_gpu_resident": True,
            "final_operation_before_preallocate_register": True,
        },
    }


def bind_offset_dataset_abi(library: Any) -> Any:
    """Bind the additive offset-view ABI without relying on ctypes defaults."""
    try:
        operation = library.uvm512_dataset_create_offset_cycle
    except AttributeError as exc:
        raise ProbeError(
            "dataset shim lacks uvm512_dataset_create_offset_cycle; rebuild it"
        ) from exc
    operation.argtypes = [base.ctypes.c_size_t] * 6
    operation.restype = base.ctypes.c_void_p
    return operation


def bind_qwen_packed_dataset_abi(library: Any) -> Any:
    """Bind the additive two-tensor packed-sparse dataset constructor."""
    try:
        operation = library.uvm512_dataset_create_qwen_packed_sparse
    except AttributeError as exc:
        raise ProbeError(
            "dataset shim lacks uvm512_dataset_create_qwen_packed_sparse; "
            "rebuild experiments/uvm_512x4m_batch_compare"
        ) from exc
    operation.argtypes = [
        base.ctypes.POINTER(base.ctypes.c_uint32),
        base.ctypes.c_size_t,
        base.ctypes.c_size_t,
        base.ctypes.c_size_t,
        base.ctypes.c_size_t,
        base.ctypes.c_size_t,
        base.ctypes.c_size_t,
        base.ctypes.c_size_t,
    ]
    operation.restype = base.ctypes.c_void_p
    return operation


def parse_expert_ids(value: str) -> tuple[int, ...]:
    """Parse comma-separated IDs and inclusive ``begin-end`` ranges."""
    selected: set[int] = set()
    for token in (item.strip() for item in value.split(",")):
        if not token:
            continue
        if "-" in token:
            begin_text, end_text = token.split("-", 1)
            begin = int(begin_text, 0)
            end = int(end_text, 0)
            require(begin <= end, f"invalid expert range {token!r}")
            selected.update(range(begin, end + 1))
        else:
            selected.add(int(token, 0))
    require(selected, "expert ID selection is empty")
    require(
        min(selected) >= 0 and max(selected) < TOTAL_EXPERTS,
        "expert IDs must be in [0, 127]",
    )
    return tuple(sorted(selected))


def interleaved_expert_ids(count: int) -> tuple[int, ...]:
    """Choose a deterministic, non-contiguous prefix of the expert space."""
    require(1 <= count <= TOTAL_EXPERTS, "cold expert count must be in [1, 128]")
    # Reverse the odd half so count=127 still omits an interior ID instead of
    # accidentally producing the contiguous prefix 0..126.
    interleaved = tuple(range(0, TOTAL_EXPERTS, 2)) + tuple(
        range(TOTAL_EXPERTS - 1, 0, -2)
    )
    return tuple(sorted(interleaved[:count]))


def require_sparse_expert_ids(
    expert_ids: Sequence[int], *, label: str, require_gap: bool = True
) -> None:
    require(expert_ids, f"{label} expert ID selection is empty")
    require(
        len(set(int(value) for value in expert_ids)) == len(expert_ids),
        f"{label} expert IDs contain duplicates",
    )
    require(
        min(expert_ids) >= 0 and max(expert_ids) < TOTAL_EXPERTS,
        f"{label} expert IDs must be in [0, 127]",
    )
    if require_gap and 1 < len(expert_ids) < TOTAL_EXPERTS:
        require(
            any(
                expert_ids[index] - expert_ids[index - 1] > 1
                for index in range(1, len(expert_ids))
            ),
            f"{label} sparse expert IDs must contain at least one gap",
        )


def deterministic_layer_expert_ids(
    count: int,
    *,
    layer_ids: Sequence[int] = base.LAYER_IDS,
) -> dict[int, tuple[int, ...]]:
    """Generate equal-size, deterministic sparse selections for each layer."""
    require(1 <= count <= TOTAL_EXPERTS, "cold expert count must be in [1, 128]")
    # Odd strides are permutations modulo 128. The first N values therefore
    # stay unique within a layer, while the four stride/offset pairs produce
    # distinct selections for every N < 128, including N=64 and N=127.
    permutations = ((31, 0), (47, 1), (53, 2), (61, 3))
    require(
        len(layer_ids) <= len(permutations),
        "deterministic sparse selection supports at most four layers",
    )
    selections: dict[int, tuple[int, ...]] = {}
    for index, layer_id in enumerate(layer_ids):
        stride, offset = permutations[index]
        expert_ids = tuple(
            sorted((offset + position * stride) % TOTAL_EXPERTS for position in range(count))
        )
        require_sparse_expert_ids(expert_ids, label=f"layer {layer_id}")
        selections[int(layer_id)] = expert_ids
    if count < TOTAL_EXPERTS:
        require(
            len(set(selections.values())) == len(selections),
            "deterministic sparse selections must differ across layers",
        )
    return selections


def parse_layer_expert_ids(
    value: str,
    *,
    layer_ids: Sequence[int] = base.LAYER_IDS,
    require_gap: bool = True,
) -> dict[int, tuple[int, ...]]:
    """Parse ``L0=0,7;L1=1,8;...`` selections with equal layer counts."""
    expected_layers = tuple(int(layer_id) for layer_id in layer_ids)
    selections: dict[int, tuple[int, ...]] = {}
    for raw_group in value.split(";"):
        group = raw_group.strip()
        require(group, "per-layer expert ID selection contains an empty group")
        layer_text, separator, expert_text = group.partition("=")
        require(separator == "=", f"invalid per-layer expert group {group!r}")
        layer_text = layer_text.strip()
        if layer_text.lower().startswith("l"):
            layer_text = layer_text[1:]
        layer_id = int(layer_text, 0)
        require(layer_id in expected_layers, f"unexpected layer ID {layer_id}")
        require(layer_id not in selections, f"layer {layer_id} is repeated")
        expert_ids = parse_expert_ids(expert_text)
        require_sparse_expert_ids(
            expert_ids, label=f"layer {layer_id}", require_gap=require_gap
        )
        selections[layer_id] = expert_ids

    require(
        set(selections) == set(expected_layers),
        "per-layer expert IDs must specify every probe layer exactly once",
    )
    require(
        len({len(expert_ids) for expert_ids in selections.values()}) == 1,
        "per-layer expert ID selections must have identical counts",
    )
    return {layer_id: selections[layer_id] for layer_id in expected_layers}


def workload_shape(
    layout: str,
    *,
    cold_expert_count: int,
    cold_expert_ids: str,
) -> WorkloadShape:
    if layout in SPARSE_LAYOUTS:
        expert_ids = (
            parse_expert_ids(cold_expert_ids)
            if cold_expert_ids.strip()
            else interleaved_expert_ids(cold_expert_count)
        )
        require_sparse_expert_ids(
            expert_ids,
            label="sparse",
            require_gap=layout != "qwen-packed-sparse",
        )
        entry_count = len(expert_ids) * ENVELOPES_PER_EXPERT
    else:
        require(
            not cold_expert_ids.strip(),
            "--cold-expert-ids is valid only for sparse expert layouts",
        )
        expert_ids = tuple(range(TOTAL_EXPERTS))
        entry_count = ENTRY_COUNT

    return WorkloadShape(
        layout=layout,
        expert_ids=expert_ids,
        entry_count=entry_count,
        logical_bytes=entry_count * base.SLICE_BYTES,
    )


def workload_shapes_by_layer(
    layout: str,
    *,
    cold_expert_count: int,
    cold_expert_ids: str,
    cold_expert_ids_by_layer: str,
) -> dict[int, WorkloadShape]:
    """Resolve one equal-geometry workload shape for each probe layer."""
    layer_ids = tuple(int(layer_id) for layer_id in base.LAYER_IDS)
    if layout not in SPARSE_LAYOUTS:
        require(
            not cold_expert_ids_by_layer.strip(),
            "--cold-expert-ids-by-layer is valid only for sparse expert layouts",
        )
        shape = workload_shape(
            layout,
            cold_expert_count=cold_expert_count,
            cold_expert_ids=cold_expert_ids,
        )
        return {layer_id: shape for layer_id in layer_ids}

    require(
        not (cold_expert_ids.strip() and cold_expert_ids_by_layer.strip()),
        "--cold-expert-ids and --cold-expert-ids-by-layer are mutually exclusive",
    )
    if cold_expert_ids_by_layer.strip():
        selections = parse_layer_expert_ids(
            cold_expert_ids_by_layer,
            layer_ids=layer_ids,
            require_gap=layout != "qwen-packed-sparse",
        )
    elif cold_expert_ids.strip():
        expert_ids = parse_expert_ids(cold_expert_ids)
        require_sparse_expert_ids(
            expert_ids,
            label="sparse",
            require_gap=layout != "qwen-packed-sparse",
        )
        selections = {layer_id: expert_ids for layer_id in layer_ids}
    else:
        selections = deterministic_layer_expert_ids(
            cold_expert_count,
            layer_ids=layer_ids,
        )

    shapes = {
        layer_id: WorkloadShape(
            layout=layout,
            expert_ids=selections[layer_id],
            entry_count=len(selections[layer_id]) * ENVELOPES_PER_EXPERT,
            logical_bytes=(
                len(selections[layer_id]) * ENVELOPES_PER_EXPERT * base.SLICE_BYTES
            ),
        )
        for layer_id in layer_ids
    }
    require(
        len({shape.entry_count for shape in shapes.values()}) == 1,
        "per-layer sparse plans must have identical lane counts",
    )
    require(
        len({shape.logical_bytes for shape in shapes.values()}) == 1,
        "per-layer sparse plans must have identical lane-size geometry",
    )
    return shapes


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeError(message)


def read_int(path: Path, name: str) -> int:
    require(path.is_file(), f"missing {name}: {path}")
    try:
        return int(path.read_text(encoding="utf-8").strip(), 0)
    except ValueError as exc:
        raise ProbeError(f"invalid {name}: {path}") from exc


def profile_stage_ns(delta: Mapping[str, int]) -> dict[str, int]:
    prefix = "slot_direct_profile."
    suffix = ".ns"
    return {
        name[len(prefix) : -len(suffix)]: int(value)
        for name, value in sorted(delta.items())
        if name.startswith(prefix) and name.endswith(suffix)
    }


def build_entries(
    pointers: Sequence[int],
    *,
    layout: str,
    logical_layer: int,
    bank_id: int,
    reuse_required: bool,
    expert_ids: Sequence[int] = (),
) -> tuple[tuple[int, ...], ...]:
    require(bank_id in (0, 1), "bank_id must be 0 or 1")
    flags = base.NO_EVICT | (
        base.REUSE_BACKING_REQUIRED if reuse_required else 0
    )
    driver_layer = base.DRIVER_BANK_LAYER + bank_id
    owner_base = 0x10000 * (logical_layer + 1)
    entries: list[tuple[int, ...]] = []

    if layout in ("packed", "gapped"):
        require(
            len(pointers) == PACKED_ALLOCATION_COUNT,
            "packed pointer count mismatch",
        )
        for allocation_index, pointer in enumerate(pointers):
            for slice_index in range(base.SLICES_PER_BLOCK):
                entry_index = (
                    allocation_index * base.SLICES_PER_BLOCK + slice_index
                )
                entries.append(
                    (
                        int(pointer) + slice_index * base.SLICE_BYTES,
                        base.SLICE_BYTES,
                        driver_layer,
                        owner_base + entry_index,
                        entry_index + 1,
                        0,
                        flags,
                    )
                )
    elif layout == "fragmented":
        require(len(pointers) == ENTRY_COUNT, "fragmented pointer count mismatch")
        for entry_index, pointer in enumerate(pointers):
            entries.append(
                (
                    int(pointer),
                    base.SLICE_BYTES,
                    driver_layer,
                    owner_base + entry_index,
                    entry_index + 1,
                    0,
                    flags,
                )
            )
    elif layout in SPARSE_LAYOUTS:
        require(expert_ids, f"{layout} requires selected expert IDs")
        require(
            len(pointers) == len(expert_ids) * ENVELOPES_PER_EXPERT,
            "sparse expert pointer count mismatch",
        )
        for entry_index, pointer in enumerate(pointers):
            expert_index = entry_index // ENVELOPES_PER_EXPERT
            expert_id = int(expert_ids[expert_index])
            entries.append(
                (
                    int(pointer),
                    base.SLICE_BYTES,
                    driver_layer,
                    owner_base + expert_id,
                    entry_index + 1,
                    0,
                    flags,
                )
            )
    else:
        raise ProbeError(f"unsupported layout {layout!r}")

    expected_entry_count = (
        len(expert_ids) * ENVELOPES_PER_EXPERT
        if layout in SPARSE_LAYOUTS
        else ENTRY_COUNT
    )
    require(len(entries) == expected_entry_count, "entry count mismatch")
    require(
        sum(int(entry[1]) for entry in entries)
        == expected_entry_count * base.SLICE_BYTES,
        "logical byte count mismatch",
    )
    return tuple(entries)


def layout_metrics(entries: Sequence[Sequence[int]]) -> dict[str, Any]:
    ranges = sorted((int(entry[0]), int(entry[0]) + int(entry[1])) for entry in entries)
    require(all(start > 0 and end > start for start, end in ranges), "invalid VA range")
    require(
        all(ranges[index - 1][1] <= ranges[index][0] for index in range(1, len(ranges))),
        "VA ranges overlap",
    )

    block_intervals: dict[int, list[tuple[int, int]]] = {}
    for start, end in ranges:
        cursor = start
        while cursor < end:
            block_start = cursor - cursor % base.BLOCK_BYTES
            slice_end = min(end, block_start + base.BLOCK_BYTES)
            block_intervals.setdefault(block_start, []).append((cursor, slice_end))
            cursor = slice_end

    complete_blocks = 0
    partial_blocks = 0
    maximum_slices = 0
    multi_entry_blocks = 0
    entry_slices_histogram: dict[str, int] = {}
    active_bytes_histogram: dict[str, int] = {}
    cold_leaf_mask_histogram: dict[str, int] = {}
    partial_preserve_leaf_entries = 0
    for block_start, intervals in block_intervals.items():
        intervals.sort()
        maximum_slices = max(maximum_slices, len(intervals))
        if len(intervals) > 1:
            multi_entry_blocks += 1
        slice_key = str(len(intervals))
        entry_slices_histogram[slice_key] = entry_slices_histogram.get(slice_key, 0) + 1
        active_bytes = sum(end - start for start, end in intervals)
        byte_key = str(active_bytes)
        active_bytes_histogram[byte_key] = active_bytes_histogram.get(byte_key, 0) + 1
        cold_leaf_mask = 0
        for start, end in intervals:
            require(
                start % PTE_64K_BYTES == 0 and end % PTE_64K_BYTES == 0,
                "layout range is not representable by 64 KiB leaves",
            )
            first_leaf = (start - block_start) // PTE_64K_BYTES
            leaf_count = (end - start) // PTE_64K_BYTES
            cold_leaf_mask |= ((1 << leaf_count) - 1) << first_leaf
        mask_key = f"0x{cold_leaf_mask:08x}"
        cold_leaf_mask_histogram[mask_key] = (
            cold_leaf_mask_histogram.get(mask_key, 0) + 1
        )
        cursor = block_start
        complete = True
        for start, end in intervals:
            if start != cursor:
                complete = False
                break
            cursor = end
        if complete and cursor == block_start + base.BLOCK_BYTES:
            complete_blocks += 1
        else:
            partial_blocks += 1
            partial_preserve_leaf_entries += (
                PTE_64K_PER_BLOCK - cold_leaf_mask.bit_count()
            )

    gaps = [
        ranges[index][0] - ranges[index - 1][1]
        for index in range(1, len(ranges))
    ]
    positive_gaps = [gap for gap in gaps if gap > 0]
    return {
        "entry_count": len(entries),
        "logical_bytes": sum(end - start for start, end in ranges),
        "minimum_va": ranges[0][0],
        "maximum_va_end": ranges[-1][1],
        "adjacent_range_pairs": sum(
            ranges[index - 1][1] == ranges[index][0]
            for index in range(1, len(ranges))
        ),
        "minimum_gap_bytes": min(gaps, default=0),
        "nonzero_gap_pairs": len(positive_gaps),
        "minimum_positive_gap_bytes": min(positive_gaps, default=0),
        "va_block_count": len(block_intervals),
        "complete_2m_blocks": complete_blocks,
        "partial_2m_blocks": partial_blocks,
        "max_entry_slices_per_2m_block": maximum_slices,
        "multi_entry_2m_blocks": multi_entry_blocks,
        "entry_slices_per_2m_block_histogram": entry_slices_histogram,
        "active_bytes_per_2m_block_histogram": active_bytes_histogram,
        "cold_64k_leaf_mask_histogram": cold_leaf_mask_histogram,
        "partial_preserve_64k_leaf_entries": partial_preserve_leaf_entries,
        "mixed_full_partial_2m_layout": complete_blocks > 0 and partial_blocks > 0,
        "complete_gap_free_2m_layout": partial_blocks == 0,
    }


def pte_contract(pte_mode: str, metrics: Mapping[str, Any]) -> dict[str, int]:
    entry_count = int(metrics["entry_count"])
    logical_bytes = int(metrics["logical_bytes"])
    touched_blocks = int(metrics["va_block_count"])
    uses_64k = pte_mode in ("64k", "partial64k")
    return {
        "touched_2m_blocks": touched_blocks,
        "full_2m_blocks": int(metrics.get("complete_2m_blocks", 0)),
        "partial_2m_blocks": int(metrics.get("partial_2m_blocks", 0)),
        "active_64k_leaf_entries": (
            logical_bytes // PTE_64K_BYTES if uses_64k else 0
        ),
        "active_4k_leaf_entries": logical_bytes // 4096 if not uses_64k else 0,
        # A partial-64K implementation must initialize or otherwise prove the
        # state of every leaf in each activated child table. This capacity is
        # deliberately reported separately from active mapped leaves.
        "activated_child_table_leaf_capacity": (
            touched_blocks * PTE_64K_PER_BLOCK if uses_64k else 0
        ),
        "active_envelopes": entry_count,
    }


def require_layout_contract(
    layout: str,
    metrics: Mapping[str, Any],
    *,
    expected_entry_count: int = ENTRY_COUNT,
    expected_logical_bytes: int = LOGICAL_BYTES,
    allow_adjacent_partial: bool = False,
) -> None:
    require(
        metrics["entry_count"] == expected_entry_count,
        "layout entry count mismatch",
    )
    require(
        metrics["logical_bytes"] == expected_logical_bytes,
        "layout byte count mismatch",
    )
    if layout in ("packed", "gapped"):
        require(
            metrics["complete_2m_blocks"] == PACKED_ALLOCATION_COUNT,
            "packed layout does not contain 288 complete blocks",
        )
        require(metrics["partial_2m_blocks"] == 0, "packed layout has partial blocks")
        require(
            metrics["max_entry_slices_per_2m_block"] == base.SLICES_PER_BLOCK,
            "packed block does not contain four envelopes",
        )
        if layout == "gapped":
            require(
                metrics["adjacent_range_pairs"]
                == PACKED_ALLOCATION_COUNT * (base.SLICES_PER_BLOCK - 1),
                "gapped layout has unexpected within-block adjacency",
            )
            require(
                metrics["nonzero_gap_pairs"] == PACKED_ALLOCATION_COUNT - 1,
                "gapped layout does not separate every payload block",
            )
            require(
                metrics["minimum_positive_gap_bytes"] >= base.BLOCK_BYTES,
                "gapped layout has less than one VA block between payload blocks",
            )
    elif layout in ("fragmented", "sparse-experts"):
        require(
            layout in ("fragmented", "sparse-experts"),
            f"unsupported independent partial layout {layout!r}",
        )
        if not allow_adjacent_partial:
            require(metrics["adjacent_range_pairs"] == 0, "partial ranges are adjacent")
        require(
            metrics["va_block_count"] == expected_entry_count,
            "partial envelopes do not occupy distinct 2 MiB VA blocks",
        )
        require(metrics["complete_2m_blocks"] == 0, "partial layout has a full block")
        require(
            metrics["partial_2m_blocks"] == expected_entry_count,
            "partial layout partial-block count mismatch",
        )
        require(
            metrics["max_entry_slices_per_2m_block"] == 1,
            "partial VA block contains more than one envelope",
        )
    elif layout == "qwen-packed-sparse":
        require(
            metrics["complete_2m_blocks"] + metrics["partial_2m_blocks"]
            == metrics["va_block_count"],
            "Qwen packed block accounting mismatch",
        )
        require(
            1 < metrics["max_entry_slices_per_2m_block"]
            <= base.SLICES_PER_BLOCK,
            "Qwen packed layout did not place multiple envelopes in a block",
        )
        require(
            metrics["multi_entry_2m_blocks"] > 0,
            "Qwen packed layout contains no multi-entry VA block",
        )
        if expected_entry_count < ENTRY_COUNT:
            require(
                metrics["partial_2m_blocks"] > 0,
                "sparse Qwen selection contains no partial block",
            )
            require(
                metrics["partial_preserve_64k_leaf_entries"] > 0,
                "sparse Qwen selection has no hot leaves to preserve",
            )
    else:
        raise ProbeError(f"unsupported layout {layout!r}")


def create_dataset(
    backend: Any,
    *,
    layout: str,
    layer_id: int,
    seed: int,
    guard_bytes: int,
    entry_count: int = ENTRY_COUNT,
    cycle_active_offsets: bool = False,
    expert_ids: Sequence[int] = (),
    qwen_base_offsets: tuple[int, int] = (0, 0),
) -> tuple[int, tuple[int, ...]]:
    if layout == "packed":
        handle = int(backend.create_dataset(layer_id, seed, PACKED_ALLOCATION_COUNT))
        pointers = backend.dataset_pointers(handle, PACKED_ALLOCATION_COUNT)
    elif layout == "gapped":
        handle = int(
            backend.library.uvm512_dataset_create_view(
                PACKED_ALLOCATION_COUNT,
                GAPPED_ALLOCATION_BYTES,
                base.BLOCK_BYTES,
                0,
            )
        )
        require(handle != 0, f"L{layer_id} gapped dataset_create failed")
        backend.datasets.append(handle)
        backend._check_cuda(  # noqa: SLF001 - isolated probe backend
            backend.library.uvm512_dataset_reseed(handle, seed),
            f"L{layer_id} gapped reseed",
        )
        pointers = backend.dataset_pointers(handle, PACKED_ALLOCATION_COUNT)
    elif layout in ("fragmented", "sparse-experts"):
        require(
            layout in ("fragmented", "sparse-experts"),
            f"unsupported dataset layout {layout!r}",
        )
        create_operation = (
            backend.library.uvm512_dataset_create_offset_cycle
            if cycle_active_offsets
            else backend.library.uvm512_dataset_create_view
        )
        create_arguments = (
            (
                entry_count,
                FRAGMENT_ALLOCATION_BYTES,
                base.SLICE_BYTES,
                ACTIVE_OFFSET_STEP_BYTES,
                ACTIVE_OFFSET_PERIOD,
                guard_bytes,
            )
            if cycle_active_offsets
            else (
                entry_count,
                FRAGMENT_ALLOCATION_BYTES,
                base.SLICE_BYTES,
                guard_bytes,
            )
        )
        handle = int(create_operation(*create_arguments))
        require(handle != 0, f"L{layer_id} fragmented dataset_create failed")
        backend.datasets.append(handle)
        backend._check_cuda(  # noqa: SLF001 - isolated probe backend
            backend.library.uvm512_dataset_reseed(handle, seed),
            f"L{layer_id} fragmented reseed",
        )
        pointers = backend.dataset_pointers(handle, entry_count)
    elif layout == "qwen-packed-sparse":
        require_sparse_expert_ids(
            expert_ids, label=f"L{layer_id} Qwen packed", require_gap=False
        )
        require(
            entry_count == len(expert_ids) * ENVELOPES_PER_EXPERT,
            f"L{layer_id} Qwen packed entry count mismatch",
        )
        require(not cycle_active_offsets, "Qwen packed layout has fixed tensor strides")
        require(
            len(qwen_base_offsets) == 2
            and all(
                int(offset) in QWEN_TENSOR_OFFSET_CYCLE
                for offset in qwen_base_offsets
            ),
            "Qwen tensor offsets must be 0/512KiB/1MiB/1.5MiB",
        )
        create_operation = bind_qwen_packed_dataset_abi(backend.library)
        expert_array_type = base.ctypes.c_uint32 * len(expert_ids)
        expert_array = expert_array_type(*(int(value) for value in expert_ids))
        handle = int(
            create_operation(
                expert_array,
                len(expert_ids),
                TOTAL_EXPERTS,
                base.SLICE_BYTES,
                W13_ENVELOPES_PER_EXPERT,
                W2_ENVELOPES_PER_EXPERT,
                int(qwen_base_offsets[0]),
                int(qwen_base_offsets[1]),
            )
            or 0
        )
        require(handle != 0, f"L{layer_id} Qwen packed dataset_create failed")
        backend.datasets.append(handle)
        backend._check_cuda(  # noqa: SLF001 - isolated probe backend
            backend.library.uvm512_dataset_reseed(handle, seed),
            f"L{layer_id} Qwen packed reseed",
        )
        pointers = backend.dataset_pointers(handle, entry_count)
        w13_base, w2_base = qwen_tensor_bases_from_regions(pointers, expert_ids)
        require(
            w13_base % base.BLOCK_BYTES == int(qwen_base_offsets[0])
            and w2_base % base.BLOCK_BYTES == int(qwen_base_offsets[1]),
            f"L{layer_id} Qwen tensor base offset mismatch",
        )
    else:
        raise ProbeError(f"unsupported dataset layout {layout!r}")

    require(all(pointer > 0 for pointer in pointers), f"L{layer_id} has a null VA")
    require(
        all(pointer % 4096 == 0 for pointer in pointers),
        f"L{layer_id} contains a non-4K-aligned VA",
    )
    require(len(set(pointers)) == len(pointers), f"L{layer_id} has duplicate VAs")
    return handle, pointers


def summarize_result(result: Mapping[str, object]) -> dict[str, object]:
    summary = base.summarize(result)
    for key in ("generation", "requested_bytes", "accepted_bytes"):
        if key in result:
            summary[key] = result[key]
    return summary


def timed_call(backend: Any, operation: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    before = backend.read_stats()
    started_ns = time.perf_counter_ns()
    result = operation(*args, **kwargs)
    ended_ns = time.perf_counter_ns()
    after = backend.read_stats()
    delta = base.stats_delta(before, after)
    return {
        "wall_ns": ended_ns - started_ns,
        "result": summarize_result(result),
        "latency_stats_delta": delta,
        "stage_ns": profile_stage_ns(delta),
        "_raw_result": result,
    }


def validate_registration_prewarm(
    record: dict[str, Any],
    *,
    pte_mode: str,
    entry_count: int,
    logical_bytes: int,
    expected_pte: Mapping[str, int],
) -> None:
    """Require registration-time hierarchy preparation for partial 64K plans."""
    if pte_mode != "partial64k":
        return

    touched_blocks = int(expected_pte["touched_2m_blocks"])
    full_blocks = int(expected_pte["full_2m_blocks"])
    partial_blocks = int(expected_pte["partial_2m_blocks"])
    delta = record["latency_stats_delta"]
    expected = {
        "slot_plan_pte_prewarm_calls": 1,
        "slot_plan_pte_prewarm_successes": 1,
        "slot_plan_pte_prewarm_failures": 0,
        "slot_plan_pte_prewarm_ranges": entry_count,
        "slot_plan_pte_prewarm_blocks": touched_blocks,
        "slot_plan_pte_prewarm_full_blocks": full_blocks,
        "slot_plan_pte_prewarm_partial_blocks": partial_blocks,
        "slot_plan_pte_prewarm_bytes": logical_bytes,
    }
    for key, value in expected.items():
        require(int(delta.get(key, -1)) == value, f"registration counter {key} mismatch")

    allocated_blocks = int(
        delta.get("slot_plan_pte_prewarm_allocated_blocks", -1)
    )
    reused_blocks = int(delta.get("slot_plan_pte_prewarm_reused_blocks", -1))
    require(
        allocated_blocks >= 0
        and reused_blocks >= 0
        and allocated_blocks + reused_blocks == touched_blocks,
        "registration prewarm allocated/reused block accounting mismatch",
    )
    record["pte_prewarm_contract"] = dict(expected_pte)
    record["pte_prewarm_observed"] = {
        "calls": 1,
        "ranges": entry_count,
        "blocks": touched_blocks,
        "full_blocks": full_blocks,
        "partial_blocks": partial_blocks,
        "allocated_blocks": allocated_blocks,
        "reused_blocks": reused_blocks,
        "bytes": logical_bytes,
    }


def validate_execute(
    record: Mapping[str, Any],
    *,
    plan_id: int,
    layer_id: int,
    generation: int,
    initial_mode: int,
    initial: bool,
    dual_bank_event: bool,
    entry_count: int,
    logical_bytes: int,
    pte_mode: str,
    expected_pte: Mapping[str, int],
    shadow_requested: bool,
) -> None:
    result = record["_raw_result"]
    require(isinstance(result, Mapping), "execute result is not an object")
    expected_mode = initial_mode if initial else base.MODE_TRANSITION
    expected = {
        "plan_id": plan_id,
        "layer_id": layer_id,
        "generation": generation,
        "new_generation": generation + 1,
        "mode": expected_mode,
        "requested_count": entry_count,
        "success_count": entry_count,
        "failure_count": 0,
        "requested_bytes": logical_bytes,
        "accepted_bytes": logical_bytes,
        "remap_count": entry_count,
        "ioctl_ret": 0,
        "rm_status": 0,
    }
    require(result.get("ok") is True, f"execute failed: {result}")
    for key, value in expected.items():
        require(int(result.get(key, -1)) == value, f"execute {key} mismatch")
    require(result.get("poisoned") is False, "execute poisoned the session")
    require(result.get("plan_execute") is True, "execute did not use compact plan ioctl")
    require(int(result.get("copy_push_count", 0)) > 0, "execute emitted no copy push")

    delta = record["latency_stats_delta"]
    expected_counters = {
        "slot_plan_execute_calls": 1,
        "slot_direct_ioctl_calls": 1,
        "slot_direct_successes": 1,
        "slot_direct_errors": 0,
        "slot_direct_fallbacks": 0,
        "slot_direct_requested_ranges": entry_count,
        "slot_direct_completed_ranges": entry_count,
        "slot_direct_requested_bytes": logical_bytes,
        "slot_direct_completed_bytes": logical_bytes,
    }
    for key, value in expected_counters.items():
        require(int(delta.get(key, -1)) == value, f"execute counter {key} mismatch")
    require(
        int(delta.get("slot_direct_dual_bank_copy_first_calls", -1))
        == (1 if dual_bank_event and not initial else 0),
        "dual-bank copy-first counter does not match the selected A/B arm",
    )

    touched_blocks = int(expected_pte["touched_2m_blocks"])
    full_blocks = int(expected_pte["full_2m_blocks"])
    partial_blocks = int(expected_pte["partial_2m_blocks"])
    active_64k = int(expected_pte["active_64k_leaf_entries"])
    active_4k = int(expected_pte["active_4k_leaf_entries"])
    pte_write_entries = int(delta.get("slot_direct_pte_write_entries", -1))
    observed = {
        "pte_4k_blocks": int(delta.get("slot_direct_pte_4k_blocks", -1)),
        "pte_64k_blocks": int(delta.get("slot_direct_pte_64k_blocks", -1)),
        "partial_64k_blocks": int(
            delta.get("slot_direct_partial_64k_blocks", -1)
        ),
        "pte_write_entries": pte_write_entries,
        # For a 64K operation the kernel's PTE-write count is the number of
        # active leaves selected by the plan. Never synthesize a successful
        # value when the telemetry counter is missing.
        "active_64k_leaf_entries": pte_write_entries if pte_mode != "4k" else 0,
        "shadow_prefill_blocks": int(
            delta.get("slot_direct_shadow_pte_prefill_blocks", 0)
        ),
    }
    if pte_mode == "4k":
        require(
            observed["partial_64k_blocks"] == 0,
            "4K arm emitted partial-64K blocks",
        )
        require(observed["pte_64k_blocks"] == 0, "4K arm emitted 64K blocks")
        require(
            observed["pte_4k_blocks"] == entry_count,
            "4K arm mapped the wrong entry count",
        )
        require(
            observed["pte_write_entries"] == active_4k,
            "4K arm wrote the wrong leaf count",
        )
    else:
        require(observed["pte_4k_blocks"] == 0, "64K arm silently used 4K PTEs")
        require(
            observed["pte_64k_blocks"] == touched_blocks,
            "64K arm touched the wrong number of VA blocks",
        )
        require(
            observed["active_64k_leaf_entries"] == active_64k,
            "64K arm active leaf telemetry mismatch",
        )
        require(
            observed["pte_write_entries"] >= active_64k,
            "64K arm wrote fewer leaves than the active mapping requires",
        )

        if pte_mode == "partial64k":
            require(
                observed["partial_64k_blocks"] == partial_blocks,
                "partial-64K arm reported the wrong partial-block count",
            )
        else:
            require(
                observed["partial_64k_blocks"] == 0,
                "complete 64K arm unexpectedly reported partial blocks",
            )

        # Shadow eligibility is per block: complete groups may prefill an
        # unreachable leaf table while partial groups preserve hot leaves via
        # the live-leaf path. A mixed plan must therefore report exactly its
        # complete-block count, never zero-for-the-whole-plan or every block.
        if not initial:
            expected_shadow_blocks = full_blocks if shadow_requested else 0
            require(
                observed["shadow_prefill_blocks"] == expected_shadow_blocks,
                "shadow prefill did not match complete-block eligibility",
            )

    if isinstance(record, dict):
        record["pte_contract"] = dict(expected_pte)
        record["pte_observed"] = observed


def run_worker_body(args: argparse.Namespace) -> dict[str, Any]:
    shapes = workload_shapes_by_layer(
        args.layout,
        cold_expert_count=args.cold_expert_count,
        cold_expert_ids=args.cold_expert_ids,
        cold_expert_ids_by_layer=args.cold_expert_ids_by_layer,
    )
    shape = shapes[int(base.LAYER_IDS[0])]
    entry_count = shape.entry_count
    logical_bytes = shape.logical_bytes
    selected_active_offsets = active_offsets(
        entry_count, cycle=args.cycle_active_offsets
    )
    expert_ids_by_layer = {
        str(layer_id): list(shapes[int(layer_id)].expert_ids)
        for layer_id in base.LAYER_IDS
    }
    expert_ids_vary_by_layer = len(
        {shapes[int(layer_id)].expert_ids for layer_id in base.LAYER_IDS}
    ) > 1
    parameters = {
        "dual_bank_event": read_int(args.dual_parameter, "dual-bank parameter"),
        "copy_run_cache": read_int(args.cache_parameter, "copy-cache parameter"),
        "pte_prewarm": read_int(args.prewarm_parameter, "PTE-prewarm parameter"),
        "legacy_overlap": read_int(args.overlap_parameter, "overlap parameter"),
        "metadata_recipe": read_int(args.recipe_parameter, "recipe parameter"),
        "fast_validation": read_int(
            args.fast_validation_parameter, "fast-validation parameter"
        ),
        "epoch_validation": read_int(
            args.epoch_validation_parameter, "epoch-validation parameter"
        ),
        "shadow_pte_prefill": read_int(args.shadow_parameter, "shadow parameter"),
        "cuda_fence": read_int(args.cuda_fence_parameter, "CUDA-fence parameter"),
        "fast_index": read_int(args.fast_index_parameter, "fast-index parameter"),
        "direct_cutover": read_int(
            args.direct_cutover_parameter, "direct-cutover parameter"
        ),
        "vidmem_budget_bytes": read_int(args.budget_parameter, "VIDMEM budget"),
    }
    require(
        parameters["dual_bank_event"] in (0, 1),
        "dual-bank event parameter must be 0 or 1",
    )
    require(
        parameters["copy_run_cache"] in (0, 1),
        "copy-run cache parameter must be 0 or 1",
    )
    require(
        parameters["epoch_validation"] in (0, 1),
        "epoch-validation parameter must be 0 or 1",
    )
    if parameters["epoch_validation"] == 1:
        require(
            parameters["copy_run_cache"] == 1
            and parameters["metadata_recipe"] == 1
            and parameters["fast_validation"] == 1,
            "epoch validation requires copy, metadata, and fast-validation caches",
        )
    if parameters["dual_bank_event"] == 1:
        require(
            parameters["copy_run_cache"] == 1,
            "dual-bank copy-first requires the registered copy-run cache",
        )
    require(parameters["legacy_overlap"] == 0, "legacy overlap must be disabled")
    require(parameters["fast_index"] == 1, "fast index must be enabled")
    require(parameters["direct_cutover"] == 1, "direct cutover must be enabled")
    require(parameters["cuda_fence"] == 0, "host-event probe requires CUDA fence=0")
    require(
        parameters["vidmem_budget_bytes"]
        >= (2 if parameters["dual_bank_event"] else 1) * logical_bytes,
        "VIDMEM budget is below the configured bank payload",
    )
    bank_count = 2 if parameters["dual_bank_event"] else 1

    def plan_bank(layer_id: int) -> int:
        return (layer_id & 1) if bank_count == 2 else 0
    if args.pte_mode in ("64k", "partial64k"):
        require(parameters["pte_prewarm"] == 1, "64K arm requires PTE prewarm")
    elif args.pte_mode == "4k":
        require(
            parameters["shadow_pte_prefill"] == 0,
            "4K fallback cannot use 64K shadow leaf prefill",
        )

    backend = base.GpuBackend(
        library=args.library, stats=args.stats, device=args.device
    )
    if args.cycle_active_offsets:
        bind_offset_dataset_abi(backend.library)
    if args.layout == "qwen-packed-sparse":
        bind_qwen_packed_dataset_abi(backend.library)
    # GpuBackend defaults to the historical 64K test policy.  Policy flags are
    # consumed at registration time, so override only the PTE grain after the
    # backend has loaded the production wrapper.
    os.environ["VLLM_UVM_SLOT_DIRECT_SHARED_PUSH"] = "1"
    os.environ["VLLM_UVM_SLOT_DIRECT_64K_PTE"] = (
        "0" if args.pte_mode == "4k" else "1"
    )

    event = 0
    datasets: dict[int, int] = {}
    pointers: dict[int, tuple[int, ...]] = {}
    complement_guards: dict[int, ComplementGuard] = {}
    active_cpu_restore_records: list[dict[str, int]] = []
    qwen_registration_residency: dict[str, Any] = {"enabled": False}
    initial_entries: tuple[tuple[int, ...], ...] = ()
    bank1_prealloc: tuple[tuple[int, ...], ...] = ()
    try:
        event = int(backend.library.uvm512_event_create())
        require(event != 0, "cudaEventCreate failed")

        all_layout_metrics: dict[str, dict[str, Any]] = {}
        for layer_index, (layer_id, seed) in enumerate(
            zip(base.LAYER_IDS, base.LAYER_SEEDS)
        ):
            dataset, layer_pointers = create_dataset(
                backend,
                layout=args.layout,
                layer_id=layer_id,
                seed=seed,
                guard_bytes=args.fragment_guard_bytes,
                entry_count=entry_count,
                cycle_active_offsets=args.cycle_active_offsets,
                expert_ids=shapes[layer_id].expert_ids,
                qwen_base_offsets=qwen_tensor_offsets(
                    layer_index, cycle=args.qwen_base_offset_cycle
                ),
            )
            datasets[layer_id] = dataset
            pointers[layer_id] = layer_pointers
            if args.pte_mode in ("64k", "partial64k"):
                require(
                    all(pointer % PTE_64K_BYTES == 0 for pointer in layer_pointers),
                    f"L{layer_id} contains a non-64K-aligned VA",
                )
            observed_active_offsets = tuple(
                int(pointer) % FRAGMENT_ALLOCATION_BYTES
                for pointer in layer_pointers
            )
            if args.cycle_active_offsets:
                require(
                    observed_active_offsets == selected_active_offsets,
                    f"L{layer_id} active VA offset cycle mismatch",
                )
            probe_entries = build_entries(
                layer_pointers,
                layout=args.layout,
                logical_layer=layer_id,
                bank_id=plan_bank(layer_id),
                reuse_required=True,
                expert_ids=shapes[layer_id].expert_ids if shape.sparse else (),
            )
            metrics = layout_metrics(probe_entries)
            require_layout_contract(
                args.layout,
                metrics,
                expected_entry_count=entry_count,
                expected_logical_bytes=logical_bytes,
                allow_adjacent_partial=args.cycle_active_offsets,
            )
            metrics["pte_contract"] = pte_contract(args.pte_mode, metrics)
            metrics["active_offsets"] = active_offset_summary(
                observed_active_offsets
            )
            if args.layout == "qwen-packed-sparse":
                metrics["qwen_tensor_layout"] = qwen_tensor_layout_summary(
                    layer_pointers, shapes[layer_id].expert_ids
                )
            all_layout_metrics[str(layer_id)] = metrics

        guard_required = (
            args.layout != "qwen-packed-sparse"
            and (args.pte_mode == "partial64k" or args.cycle_active_offsets)
        )
        if guard_required:
            for layer_id, seed in zip(base.LAYER_IDS, base.LAYER_SEEDS):
                complement_guards[layer_id] = ComplementGuard(
                    library=args.guard_library,
                    backend=backend,
                    pointers=pointers[layer_id],
                    offsets=(
                        selected_active_offsets
                        if args.cycle_active_offsets
                        else ()
                    ),
                    seed=seed ^ 0xA5A55A5A,
                )
            # This must be the final managed-residency operation before slot
            # preallocation/registration. Complement GPU faults can pull in
            # neighbouring active pages at 64K/2M granularity, while the copy
            # cache requires every active source page to be CPU-resident.
            active_cpu_restore_records = restore_guard_active_sources_cpu(
                complement_guards
            )

        all_test_vas = [pointer for values in pointers.values() for pointer in values]
        require(len(set(all_test_vas)) == len(all_test_vas), "layer VAs overlap")

        def require_va_stable(stage: str) -> None:
            for layer_id, dataset in datasets.items():
                current = backend.dataset_pointers(dataset, len(pointers[layer_id]))
                require(current == pointers[layer_id], f"L{layer_id} VA changed after {stage}")

        if args.layout == "qwen-packed-sparse":
            # Reproduce the serving state before any slot or plan exists:
            # full w13+w2 tensors were used on GPU, then only the selected cold
            # 512 KiB regions were offloaded to CPU. This must be the final
            # residency-changing phase before preallocate/register so cached
            # physical runs describe the exact cold source backing.
            qwen_registration_residency = prepare_qwen_registration_residency(
                backend, datasets, pointers, shapes
            )
            require_va_stable("Qwen registration residency setup")

        initial_entries = build_entries(
            pointers[0],
            layout=args.layout,
            logical_layer=0,
            bank_id=0,
            reuse_required=False,
            expert_ids=shapes[0].expert_ids if shape.sparse else (),
        )
        if bank_count == 2:
            bank1_prealloc = build_entries(
                pointers[1],
                layout=args.layout,
                logical_layer=1,
                bank_id=1,
                reuse_required=False,
                expert_ids=shapes[1].expert_ids if shape.sparse else (),
            )
        transition_entries = {
            layer_id: build_entries(
                pointers[layer_id],
                layout=args.layout,
                logical_layer=layer_id,
                bank_id=plan_bank(layer_id),
                reuse_required=True,
                expert_ids=shapes[layer_id].expert_ids if shape.sparse else (),
            )
            for layer_id in base.LAYER_IDS
        }

        preallocations = [timed_call(backend, backend.preallocate, initial_entries)]
        if bank_count == 2:
            preallocations.append(
                timed_call(backend, backend.preallocate, bank1_prealloc)
            )
        for record in preallocations:
            result = record["_raw_result"]
            base.require_preallocate(
                result, entries=entry_count, logical_bytes=logical_bytes
            )
        require_va_stable("preallocation")

        if args.pte_mode == "4k":
            initial_mode = MODE_INITIAL_4K
        elif args.pte_mode == "partial64k":
            initial_mode = args.partial_64k_initial_mode
        else:
            initial_mode = MODE_INITIAL_64K
        registration_records: list[dict[str, Any]] = []
        initial_registration = timed_call(
            backend,
            backend.register_plan,
            base.INITIAL_PLAN_ID,
            0,
            initial_entries,
            transition=False,
        )
        registration_records.append(initial_registration)
        initial_result = initial_registration["_raw_result"]

        if args.expect_registration_rejection:
            require(
                args.layout in ("fragmented", "sparse-experts", "qwen-packed-sparse"),
                "rejection arm requires a partial layout",
            )
            require(args.pte_mode == "64k", "rejection arm requires 64K mode")
            require(initial_result.get("ok") is False, "fragmented 64K plan was accepted")
            require(
                initial_result.get("reason") == "validation_failed",
                f"unexpected rejection reason: {initial_result}",
            )
            error = str(initial_result.get("error", ""))
            require(
                "complete gap-free 2 MiB blocks" in error,
                f"rejection did not identify partial 2MiB layout: {error}",
            )
            require(
                int(initial_registration["latency_stats_delta"].get(
                    "slot_plan_register_calls", 0
                ))
                == 0,
                "userspace layout rejection unexpectedly reached register ioctl",
            )
            checksum, expected = backend.checksum(datasets[0])
            require(checksum == expected, "rejected plan changed source data")
            close = timed_call(backend, backend.close)
            close_result = close["_raw_result"]
            require(close_result.get("ok") is True, f"close failed: {close_result}")
            require_va_stable("expected registration rejection")
            post_close = backend.state_snapshot()
            require(post_close.get("allocated_slots") == 0, "rejection leaked slots")
            return {
                "status": "pass",
                "outcome": "expected_fragmented_64k_registration_rejection",
                "layout": args.layout,
                "pte_mode": args.pte_mode,
                "entry_count": entry_count,
                "expert_ids": list(shape.expert_ids),
                "expert_ids_by_layer": expert_ids_by_layer,
                "expert_ids_vary_by_layer": expert_ids_vary_by_layer,
                "expert_count": len(shape.expert_ids),
                "bank_count": bank_count,
                "bytes_per_bank": logical_bytes,
                "total_bank_bytes": bank_count * logical_bytes,
                "layout_metrics": all_layout_metrics,
                "registration_residency_setup": qwen_registration_residency,
                "module_parameters": parameters,
                "module_srcversion": args.srcversion.read_text(encoding="utf-8").strip(),
                "preallocations": [strip_private(record) for record in preallocations],
                "registration": strip_private(initial_registration),
                "checksum": {"actual": checksum, "expected": expected},
                "close": strip_private(close),
                "post_close_state": post_close,
            }

        base.require_register(
            initial_result,
            plan_id=base.INITIAL_PLAN_ID,
            layer_id=0,
            mode=initial_mode,
            entries=entry_count,
            logical_bytes=logical_bytes,
        )
        validate_registration_prewarm(
            initial_registration,
            pte_mode=args.pte_mode,
            entry_count=entry_count,
            logical_bytes=logical_bytes,
            expected_pte=all_layout_metrics["0"]["pte_contract"],
        )
        generation = int(initial_result["generation"])

        for layer_id in base.LAYER_IDS:
            record = timed_call(
                backend,
                backend.register_plan,
                base.transition_plan_id(layer_id),
                layer_id,
                transition_entries[layer_id],
                transition=True,
            )
            registration_records.append(record)
            base.require_register(
                record["_raw_result"],
                plan_id=base.transition_plan_id(layer_id),
                layer_id=layer_id,
                mode=base.MODE_TRANSITION,
                entries=entry_count,
                logical_bytes=logical_bytes,
                generation=generation,
            )
            validate_registration_prewarm(
                record,
                pte_mode=args.pte_mode,
                entry_count=entry_count,
                logical_bytes=logical_bytes,
                expected_pte=all_layout_metrics[str(layer_id)]["pte_contract"],
            )
        require_va_stable("registration")

        execution_records: list[dict[str, Any]] = []
        checksums: list[dict[str, int]] = []
        preserve_checksums: list[dict[str, int]] = []
        event_completed_ns: int | None = None
        for sequence_index, layer_id in enumerate(EXECUTION_ORDER):
            initial = sequence_index == 0
            plan_id = (
                base.INITIAL_PLAN_ID if initial else base.transition_plan_id(layer_id)
            )
            execute_started_ns = time.perf_counter_ns()
            if event_completed_ns is not None:
                require(
                    execute_started_ns >= event_completed_ns,
                    "execute started before compute event completed",
                )
            record = timed_call(
                backend,
                backend.execute_plan,
                plan_id,
                layer_id,
                generation,
            )
            validate_execute(
                record,
                plan_id=plan_id,
                layer_id=layer_id,
                generation=generation,
                initial_mode=initial_mode,
                initial=initial,
                dual_bank_event=bool(parameters["dual_bank_event"]),
                entry_count=entry_count,
                logical_bytes=logical_bytes,
                pte_mode=args.pte_mode,
                expected_pte=all_layout_metrics[str(layer_id)]["pte_contract"],
                shadow_requested=bool(parameters["shadow_pte_prefill"]),
            )
            generation += 1
            require_va_stable(f"execute {sequence_index}")

            barrier_started_ns = time.perf_counter_ns()
            backend._check_cuda(  # noqa: SLF001
                backend.library.uvm512_dataset_launch_checksum(
                    datasets[layer_id], backend.stream
                ),
                "launch checksum",
            )
            complement_guard = complement_guards.get(layer_id)
            if complement_guard is not None:
                complement_guard.launch()
            backend._check_cuda(  # noqa: SLF001
                backend.library.uvm512_event_record(event, backend.stream),
                "record compute-done event",
            )
            backend._check_cuda(  # noqa: SLF001
                backend.library.uvm512_event_synchronize(event),
                "wait compute-done event",
            )
            event_completed_ns = time.perf_counter_ns()
            actual = base.ctypes.c_uint64()
            backend._check_cuda(  # noqa: SLF001
                backend.library.uvm512_dataset_read_checksum(
                    datasets[layer_id], base.ctypes.byref(actual)
                ),
                "read checksum",
            )
            expected = int(
                backend.library.uvm512_dataset_expected_checksum(datasets[layer_id])
            )
            require(int(actual.value) == expected, f"L{layer_id} checksum mismatch")
            if complement_guard is not None:
                preserve_actual = complement_guard.read()
                require(
                    preserve_actual == complement_guard.expected,
                    f"L{layer_id} unowned 1.5MiB complement checksum mismatch",
                )
                preserve_checksums.append(
                    {
                        "sequence_index": sequence_index,
                        "layer_id": layer_id,
                        "actual": preserve_actual,
                        "expected": complement_guard.expected,
                        "preserved_bytes": complement_guard.bytes,
                        "preserved_prefix_bytes": (
                            complement_guard.complement_prefix_bytes
                        ),
                        "preserved_suffix_bytes": (
                            complement_guard.complement_suffix_bytes
                        ),
                    }
                )
            checksums.append(
                {
                    "sequence_index": sequence_index,
                    "layer_id": layer_id,
                    "actual": int(actual.value),
                    "expected": expected,
                    "event_barrier_ns": event_completed_ns - barrier_started_ns,
                }
            )
            record["sequence_index"] = sequence_index
            record["layer_id"] = layer_id
            record["bank_id"] = plan_bank(layer_id)
            record["generation_after"] = generation
            record["compute_event_completed_ns"] = event_completed_ns
            execution_records.append(record)

        require(
            generation
            == int(initial_result["generation"]) + len(EXECUTION_ORDER),
            "generation gap",
        )
        require(
            len({item["expected"] for item in checksums[:4]}) == 4,
            "layer checksum signatures are not distinct",
        )
        require(checksums[0]["actual"] == checksums[-1]["actual"], "L0 checksum changed")
        complement_bytes_per_layer = entry_count * (
            FRAGMENT_ALLOCATION_BYTES - base.SLICE_BYTES
        )
        if guard_required:
            require(
                len(preserve_checksums) == len(EXECUTION_ORDER),
                "partial-64K complement was not checked after every execute",
            )
            require(
                all(
                    item["preserved_bytes"] == complement_bytes_per_layer
                    and item["preserved_prefix_bytes"]
                    + item["preserved_suffix_bytes"]
                    == complement_bytes_per_layer
                    for item in preserve_checksums
                ),
                "partial-64K complement byte coverage mismatch",
            )

        destroy_records = []
        for plan_id in (base.INITIAL_PLAN_ID,) + tuple(
            base.transition_plan_id(layer_id) for layer_id in base.LAYER_IDS
        ):
            record = timed_call(
                backend, backend.destroy_plan, plan_id, generation
            )
            require(record["_raw_result"].get("ok") is True, "plan destroy failed")
            destroy_records.append(record)

        pre_close = backend.state_snapshot()
        require(
            pre_close.get("allocated_slots") == bank_count * entry_count,
            "bank slot count mismatch",
        )
        require(pre_close.get("plan_count") == 0, "destroy left a registered plan")
        close = timed_call(backend, backend.close)
        close_result = close["_raw_result"]
        require(close_result.get("ok") is True, f"session close failed: {close_result}")
        post_close = backend.state_snapshot()
        expected_zero = {
            "allocated_slots": 0,
            "plan_count": 0,
            "tracked_generation": None,
            "session_open": False,
        }
        for key, value in expected_zero.items():
            require(post_close.get(key) == value, f"post-close {key} mismatch")

        steady_wall = [
            int(execution_records[index]["wall_ns"]) for index in STEADY_INDICES
        ]
        return {
            "status": "pass",
            "outcome": "successful_registered_plan_lifecycle",
            "layout": args.layout,
            "pte_mode": args.pte_mode,
            "entry_count": entry_count,
            "expert_ids": list(shape.expert_ids),
            "expert_ids_by_layer": expert_ids_by_layer,
            "expert_ids_vary_by_layer": expert_ids_vary_by_layer,
            "expert_count": len(shape.expert_ids),
            "envelopes_per_expert": ENVELOPES_PER_EXPERT,
            "w13_envelopes_per_expert": W13_ENVELOPES_PER_EXPERT,
            "w2_envelopes_per_expert": W2_ENVELOPES_PER_EXPERT,
            "bank_count": bank_count,
            "bytes_per_bank": logical_bytes,
            "total_bank_bytes": bank_count * logical_bytes,
            "layout_metrics": all_layout_metrics,
            "module_parameters": parameters,
            "module_srcversion": args.srcversion.read_text(encoding="utf-8").strip(),
            "execution_order": list(EXECUTION_ORDER),
            "bank_order": [plan_bank(layer_id) for layer_id in EXECUTION_ORDER],
            "compute_copy_overlap_allowed": False,
            "compute_event_waited_before_each_transition": True,
            "preallocations": [strip_private(record) for record in preallocations],
            "registrations": [strip_private(record) for record in registration_records],
            "executions": [strip_private(record) for record in execution_records],
            "checksums": checksums,
            "cycle_active_offsets": args.cycle_active_offsets,
            "qwen_base_offset_cycle": args.qwen_base_offset_cycle,
            "active_offset_contract": (
                all_layout_metrics["0"]["active_offsets"]
                if args.layout == "qwen-packed-sparse"
                else active_offset_summary(selected_active_offsets)
            ),
            "active_source_cpu_restore": active_cpu_restore_records,
            "registration_residency_setup": qwen_registration_residency,
            "complement_preserve_required": guard_required,
            "full_qwen_tensor_checksum_required": (
                args.layout == "qwen-packed-sparse"
            ),
            "full_qwen_tensor_checksum_bytes_per_layer": (
                QWEN_MAIN_TENSOR_BYTES
                if args.layout == "qwen-packed-sparse"
                else 0
            ),
            "complement_preserved_bytes_per_layer": (
                complement_bytes_per_layer
                if guard_required
                else 0
            ),
            "complement_preserve_checksums": preserve_checksums,
            "generation_start": int(initial_result["generation"]),
            "generation_end": generation,
            "steady_indices": list(STEADY_INDICES),
            "steady_ioctl_wall_ns": steady_wall,
            "steady_ioctl_wall_ns_median": int(statistics.median(steady_wall)),
            "destroy_records": [strip_private(record) for record in destroy_records],
            "pre_close_state": pre_close,
            "close": strip_private(close),
            "post_close_state": post_close,
        }
    finally:
        for complement_guard in complement_guards.values():
            complement_guard.close()
        if event:
            backend.library.uvm512_event_destroy(event)
        cleanup = backend.finish()
        require(bool(cleanup.get("ok")), f"backend cleanup failed: {cleanup}")


def strip_private(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


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


def wait_zero_mib(device: int, timeout_seconds: float) -> tuple[int, list[int]]:
    deadline = time.monotonic() + timeout_seconds
    observations: list[int] = []
    while True:
        current = gpu_used_mib(device)
        observations.append(current)
        if current == 0 or time.monotonic() >= deadline:
            return current, observations
        time.sleep(0.25)


def write_payload(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_worker(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "status": "failed",
        "layout": args.layout,
        "pte_mode": args.pte_mode,
        "cold_expert_count": args.cold_expert_count,
        "cold_expert_ids": args.cold_expert_ids,
        "cold_expert_ids_by_layer": args.cold_expert_ids_by_layer,
        "cycle_active_offsets": args.cycle_active_offsets,
        "qwen_base_offset_cycle": args.qwen_base_offset_cycle,
    }
    try:
        require(gpu_used_mib(args.device) == 0, "GPU is not idle before worker")
        payload = run_worker_body(args)
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    require(args.worker_output is not None, "worker output is missing")
    write_payload(args.worker_output, payload)
    return 0 if payload.get("status") == "pass" else 1


def worker_command(args: argparse.Namespace, worker_output: Path) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-output",
        str(worker_output),
        "--layout",
        args.layout,
        "--pte-mode",
        args.pte_mode,
        "--cold-expert-count",
        str(args.cold_expert_count),
        "--cold-expert-ids",
        args.cold_expert_ids,
        "--cold-expert-ids-by-layer",
        args.cold_expert_ids_by_layer,
        "--partial-64k-initial-mode",
        str(args.partial_64k_initial_mode),
        "--fragment-guard-bytes",
        str(args.fragment_guard_bytes),
        "--guard-library",
        str(args.guard_library),
        "--library",
        str(args.library),
        "--stats",
        str(args.stats),
        "--srcversion",
        str(args.srcversion),
        "--fast-index-parameter",
        str(args.fast_index_parameter),
        "--direct-cutover-parameter",
        str(args.direct_cutover_parameter),
        "--budget-parameter",
        str(args.budget_parameter),
        "--dual-parameter",
        str(args.dual_parameter),
        "--cache-parameter",
        str(args.cache_parameter),
        "--prewarm-parameter",
        str(args.prewarm_parameter),
        "--overlap-parameter",
        str(args.overlap_parameter),
        "--recipe-parameter",
        str(args.recipe_parameter),
        "--fast-validation-parameter",
        str(args.fast_validation_parameter),
        "--epoch-validation-parameter",
        str(args.epoch_validation_parameter),
        "--shadow-parameter",
        str(args.shadow_parameter),
        "--cuda-fence-parameter",
        str(args.cuda_fence_parameter),
        "--device",
        str(args.device),
    ]
    if args.expect_registration_rejection:
        command.append("--expect-registration-rejection")
    if args.cycle_active_offsets:
        command.append("--cycle-active-offsets")
    if args.qwen_base_offset_cycle:
        command.append("--qwen-base-offset-cycle")
    return command


def run_controller(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "status": "failed",
        "layout": args.layout,
        "pte_mode": args.pte_mode,
        "cold_expert_count": args.cold_expert_count,
        "cold_expert_ids": args.cold_expert_ids,
        "cold_expert_ids_by_layer": args.cold_expert_ids_by_layer,
        "cycle_active_offsets": args.cycle_active_offsets,
        "qwen_base_offset_cycle": args.qwen_base_offset_cycle,
    }
    try:
        require(gpu_used_mib(args.device) == 0, "GPU is not idle before probe")
        with tempfile.TemporaryDirectory(prefix="uvm-fragmented-plan-") as temp:
            worker_output = Path(temp) / "worker.json"
            command = worker_command(args, worker_output)
            completed = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=args.worker_timeout_seconds,
                check=False,
            )
            require(worker_output.is_file(), "worker produced no JSON artifact")
            payload = json.loads(worker_output.read_text(encoding="utf-8"))

        final_mib, observations = wait_zero_mib(
            args.device, args.zero_memory_timeout_seconds
        )
        payload["worker_command"] = command
        payload["worker_returncode"] = completed.returncode
        payload["worker_stdout"] = completed.stdout
        payload["worker_stderr"] = completed.stderr
        payload["gpu_memory_final_mib"] = final_mib
        payload["gpu_memory_observations_mib"] = observations
        require(completed.returncode == 0, "GPU worker failed")
        require(payload.get("status") == "pass", "worker artifact failed")
        require(final_mib == 0, "GPU memory did not return to zero")
    except Exception as exc:  # noqa: BLE001
        controller_failure = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        # Once the child artifact has been read, its exception is the primary
        # diagnostic. Do not replace it with the controller's generic
        # "GPU worker failed" wrapper; preserve both failure layers instead.
        if payload.get("status") == "failed" and "error" in payload:
            payload["controller_error"] = controller_failure["error"]
            payload["controller_traceback"] = controller_failure["traceback"]
        else:
            payload.update(controller_failure)
        payload["status"] = "failed"
        try:
            final_mib, observations = wait_zero_mib(
                args.device, args.zero_memory_timeout_seconds
            )
            payload["gpu_memory_final_mib"] = final_mib
            payload["gpu_memory_observations_mib"] = observations
        except Exception as gpu_exc:  # noqa: BLE001
            payload["gpu_memory_final_error"] = repr(gpu_exc)

    write_payload(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "pass" else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--layout",
        choices=(
            "packed",
            "gapped",
            "fragmented",
            "sparse-experts",
            "qwen-packed-sparse",
        ),
        required=True,
    )
    parser.add_argument(
        "--pte-mode", choices=("4k", "64k", "partial64k"), required=True
    )
    parser.add_argument("--expect-registration-rejection", action="store_true")
    parser.add_argument(
        "--cycle-active-offsets",
        action="store_true",
        help=(
            "cycle each partial allocation's active 512KiB range through "
            "0/512KiB/1MiB/1.5MiB offsets"
        ),
    )
    parser.add_argument(
        "--qwen-base-offset-cycle",
        action="store_true",
        help=(
            "cycle packed w13/w2 tensor bases through all legal 512KiB "
            "offsets relative to a 2MiB VA block"
        ),
    )
    parser.add_argument("--cold-expert-count", type=int, default=16)
    parser.add_argument("--cold-expert-ids", default="")
    parser.add_argument(
        "--cold-expert-ids-by-layer",
        default="",
        help="L0=0,7;L1=1,8;L2=2,9;L3=3,10 for equal-size sparse sets",
    )
    parser.add_argument(
        "--partial-64k-initial-mode",
        type=int,
        default=DEFAULT_PARTIAL_64K_INITIAL_MODE,
    )
    parser.add_argument(
        "--fragment-guard-bytes", type=int, default=FRAGMENT_GUARD_BYTES
    )
    parser.add_argument("--library", type=Path, default=base.DEFAULT_LIBRARY)
    parser.add_argument("--guard-library", type=Path, default=DEFAULT_GUARD_LIBRARY)
    parser.add_argument("--stats", type=Path, default=base.DEFAULT_STATS)
    parser.add_argument("--srcversion", type=Path, default=base.DEFAULT_SRCVERSION)
    parser.add_argument("--fast-index-parameter", type=Path, default=base.DEFAULT_FAST_INDEX)
    parser.add_argument(
        "--direct-cutover-parameter", type=Path, default=base.DEFAULT_DIRECT_CUTOVER
    )
    parser.add_argument("--budget-parameter", type=Path, default=base.DEFAULT_BUDGET)
    parser.add_argument("--dual-parameter", type=Path, default=DEFAULT_DUAL_PARAM)
    parser.add_argument("--cache-parameter", type=Path, default=DEFAULT_CACHE_PARAM)
    parser.add_argument("--prewarm-parameter", type=Path, default=DEFAULT_PREWARM_PARAM)
    parser.add_argument("--overlap-parameter", type=Path, default=DEFAULT_OVERLAP_PARAM)
    parser.add_argument("--recipe-parameter", type=Path, default=DEFAULT_RECIPE_PARAM)
    parser.add_argument(
        "--fast-validation-parameter",
        type=Path,
        default=DEFAULT_FAST_VALIDATION_PARAM,
    )
    parser.add_argument(
        "--epoch-validation-parameter",
        type=Path,
        default=DEFAULT_EPOCH_VALIDATION_PARAM,
    )
    parser.add_argument("--shadow-parameter", type=Path, default=DEFAULT_SHADOW_PARAM)
    parser.add_argument(
        "--cuda-fence-parameter", type=Path, default=DEFAULT_CUDA_FENCE_PARAM
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--worker-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--zero-memory-timeout-seconds", type=float, default=20.0)
    args = parser.parse_args(argv)
    if args.device < 0 or args.fragment_guard_bytes < 0:
        parser.error("device and fragment guard bytes must be non-negative")
    if not 1 <= args.cold_expert_count <= TOTAL_EXPERTS:
        parser.error("cold expert count must be in [1, 128]")
    if args.partial_64k_initial_mode <= 0:
        parser.error("partial-64K initial mode must be positive")
    if args.worker_timeout_seconds <= 0 or args.zero_memory_timeout_seconds < 0:
        parser.error("invalid timeout")
    if args.expect_registration_rejection and not (
        args.layout in ("fragmented", "sparse-experts", "qwen-packed-sparse")
        and args.pte_mode == "64k"
    ):
        parser.error("expected rejection requires a partial layout + legacy 64k")
    if args.pte_mode == "partial64k" and args.layout not in (
        "fragmented",
        "sparse-experts",
        "qwen-packed-sparse",
    ):
        parser.error("partial64k requires fragmented or sparse-experts layout")
    if args.cycle_active_offsets and args.layout not in (
        "fragmented",
        "sparse-experts",
    ):
        parser.error("--cycle-active-offsets requires a partial layout")
    if args.qwen_base_offset_cycle and args.layout != "qwen-packed-sparse":
        parser.error("--qwen-base-offset-cycle requires qwen-packed-sparse")
    try:
        workload_shapes_by_layer(
            args.layout,
            cold_expert_count=args.cold_expert_count,
            cold_expert_ids=args.cold_expert_ids,
            cold_expert_ids_by_layer=args.cold_expert_ids_by_layer,
        )
    except (ProbeError, ValueError) as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.worker_output is not None:
        return run_worker(args)
    return run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
