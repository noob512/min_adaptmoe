# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CUDA UVM (Unified Virtual Memory) allocator for vLLM.

This module provides a custom CUDA memory allocator that uses cudaMallocManaged
to enable memory oversubscription - allocating more GPU memory than physically
available by using CPU memory as backing store.

Usage:
    # Enable UVM before any CUDA operations
    from vllm.device_allocator.uvm import enable_uvm_allocator

    # Option 1: Call explicitly
    enable_uvm_allocator()

    # Option 2: Set environment variable VLLM_USE_UVM=1

Example:
    VLLM_USE_UVM=1 python -m vllm.entrypoints.openai.api_server \
        --model meta-llama/Llama-2-7b-hf

Note:
    UVM allows running models larger than GPU memory, but with significant
    performance overhead due to page migration between CPU and GPU.
    This is useful for:
    - Testing large models on smaller GPUs
    - Development and debugging
    - Scenarios where memory > throughput

Warning:
    UVM mode is NOT recommended for production use due to:
    - Unpredictable latency from page faults
    - Lower throughput compared to native GPU memory
    - Potential compatibility issues with CUDA graphs
"""

import atexit
import ctypes
import gc
import mmap
import os
import threading
import time
import uuid as uuidlib
from collections.abc import Callable, Sequence
from contextlib import contextmanager, suppress

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Global state
_uvm_enabled = False
_uvm_lib: ctypes.CDLL | None = None
_uvm_log_close_registered = False
_uvm_local_allocator = None
_uvm_local_mem_pool = None
_uvm_local_prefetch_stack: list[bool] = []
_uvm_phase_stack: list[str] = []
_UVM_DRIVER_BATCH_MAX_RANGES = 64
_UVM_INITIALIZE_IOCTL = 0x30000001
_UVM_PHASE_F_BATCH_EXECUTOR_IOCTL = 81
_UVM_PHASE_F_BATCH_REGIONS_IOCTL = 84
_UVM_PHASE_F_MAX_RANGES = 64
_UVM_EXPERT_OFFLOAD_RELEASE_IOCTL = 100
_UVM_PAGE_SIZE = 0
_UVM_MOE_SLOT_REMAP_BATCH_IOCTL = 91
_UVM_MOE_SLOT_PLAN_REGISTER_IOCTL = 92
_UVM_MOE_SLOT_PLAN_EXECUTE_IOCTL = 93
_UVM_MOE_SLOT_PLAN_DESTROY_IOCTL = 94
_UVM_MOE_SLOT_PLAN_ABORT_SESSION_IOCTL = 95
_UVM_MOE_SLOT_PLAN_EXECUTE_FENCED_IOCTL = 96
_UVM_ALLOC_SEMAPHORE_POOL_IOCTL = 68
_UVM_FREE_IOCTL = 34
_UVM_MOE_SLOT_REMAP_MAX_ENTRIES = 4608
_UVM_MOE_SLOT_REMAP_MAX_ENTRY_BYTES = 4 * 1024 * 1024
_UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC = 8
_UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_PER_BLOCK_PUSH = 10
_UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH = 11
_UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_PTE = 12
_UVM_MOE_SLOT_REMAP_MODE_RELEASE = 19
_UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION = 20
_UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE = 21
UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT = 1
UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED = 1 << 1
UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_SHARED_PUSH = 1 << 2
UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_64K_PTE = 1 << 3
_UVM_MOE_SLOT_REMAP_POLICY_FLAGS = (
    UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_SHARED_PUSH
    | UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_64K_PTE
)
_UVM_MOE_SLOT_PAGE_BYTES = 4096
_UVM_MOE_SLOT_VA_BLOCK_BYTES = 2 * 1024 * 1024
_UVM_MOE_SLOT_UINT32_MAX = (1 << 32) - 1
_UVM_MOE_SLOT_UINT64_MAX = (1 << 64) - 1
_uvm_slot_direct_sessions: dict[int, tuple[int, bytes]] = {}
_uvm_slot_direct_owners: dict[int, dict[tuple[int, int], int]] = {}
_uvm_slot_direct_plans: dict[int, dict[int, dict[str, object]]] = {}
_uvm_slot_direct_plan_generations: dict[int, int] = {}
# Registered plans already own immutable per-lane descriptors.  When the
# bank-level commit experiment is enabled, this is the only dynamic owner
# selector needed on the execute fast path; the legacy per-slot owner table is
# retained for non-registered ioctls and can stay untouched until teardown.
_uvm_slot_direct_active_plan_ids: dict[int, int] = {}
_uvm_slot_direct_session_pids: dict[int, int] = {}
_uvm_slot_direct_poisoned: dict[int, str] = {}
_uvm_slot_direct_cuda_fences: dict[int, "_UvmSlotCudaFence"] = {}
_uvm_slot_direct_cuda_fence_cleanup_deferred: set[int] = set()
_uvm_slot_cuda_fence_kernel_lib: ctypes.CDLL | None = None
_uvm_slot_cuda_fence_kernel_path_loaded = ""
_uvm_slot_cuda_fence_kernel_signal: object | None = None
_uvm_slot_cuda_fence_kernel_wait: object | None = None
_uvm_slot_direct_lock = threading.RLock()
_uvm_slot_direct_close_registered = False
_uvm_slot_direct_policy_cleanup: Callable[[int], None] | None = None
_UVM_SLOT_DIRECT_SHARED_PUSH_ENV = "VLLM_UVM_SLOT_DIRECT_SHARED_PUSH"
_UVM_SLOT_DIRECT_64K_PTE_ENV = "VLLM_UVM_SLOT_DIRECT_64K_PTE"
_UVM_SLOT_DIRECT_LAYER_TRANSITION_ENV = (
    "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION"
)
_UVM_SLOT_DIRECT_CUDA_FENCE_ENV = "VLLM_UVM_SLOT_DIRECT_CUDA_FENCE"
_UVM_SLOT_DIRECT_BANK_LEVEL_COMMIT_ENV = (
    "VLLM_UVM_SLOT_DIRECT_BANK_LEVEL_COMMIT"
)
_UVM_SLOT_DIRECT_CUDA_FENCE_KERNEL_LIB_ENV = (
    "VLLM_UVM_SLOT_DIRECT_CUDA_FENCE_KERNEL_LIB"
)
_UVM_MOE_SLOT_PLAN_FENCE_FLAGS_ALL = 0x3
_UVM_MAX_GPUS = 32 * 8
_UVM_SLOT_FENCE_POOL_BYTES = 4096
_UVM_SLOT_FENCE_COMPUTE_OFFSET = 0
_UVM_SLOT_FENCE_COMPLETION_OFFSET = 64
_UVM_SLOT_FENCE_COMPUTE_TIMEOUT_SEC = 30.0

UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ = 5
UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG = 6
UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED = 7
UVM_PHASE_F_ACTION_START_BG_PROMOTION = 8
UVM_PHASE_F_ACTION_PAUSE_BG_PROMOTION = 16
UVM_PHASE_F_ACTION_RESUME_BG_PROMOTION = 17
UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST = 18
UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_AUDITED = 21
UVM_PHASE_F_ACTION_IMMUTABLE_CPU_SHADOW_CUTOVER = 22
UVM_PHASE_F_ACTION_REBUILD_GPU_READ_MAPPINGS = 23
UVM_PHASE_F_ACTION_SYNC_CPU_REMOTE_CUTOVER = 24
_NV_ERR_INVALID_DEVICE = 0x00000026

UVM_REGION_STATE_UNKNOWN = 0
UVM_REGION_STATE_COLD = 1
UVM_REGION_STATE_WARM = 2
UVM_REGION_STATE_HOT = 3
UVM_REGION_STATE_ACTIVE = 4
UVM_REGION_STATE_INACTIVE = 5

UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K = 1 << 2
UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ = 1 << 3
UVM_EXPERT_OFFLOAD_RELEASE_FLAG_IMMUTABLE_CPU_SHADOW = 1 << 4


class _UvmInitializeParams(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint64),
        ("rmStatus", ctypes.c_uint32),
    ]


class _PhaseFUuid(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)]


class _UvmGpuMappingAttributes(ctypes.Structure):
    _fields_ = [
        ("gpu_uuid", _PhaseFUuid),
        ("gpu_mapping_type", ctypes.c_uint32),
        ("gpu_caching_type", ctypes.c_uint32),
        ("gpu_format_type", ctypes.c_uint32),
        ("gpu_element_bits", ctypes.c_uint32),
        ("gpu_compression_type", ctypes.c_uint32),
    ]


class _UvmAllocSemaphorePoolParams(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
        (
            "per_gpu_attributes",
            _UvmGpuMappingAttributes * _UVM_MAX_GPUS,
        ),
        ("gpu_attributes_count", ctypes.c_uint64),
        ("rmStatus", ctypes.c_uint32),
    ]


class _UvmFreeParams(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_uint64),
        ("rmStatus", ctypes.c_uint32),
    ]


class _PhaseFRange(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
    ]


class _PhaseFRangeResult(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("resident_on_count", ctypes.c_uint32),
        ("resident_on_cpu", ctypes.c_uint8),
        ("resident_on_gpu", ctypes.c_uint8),
        ("queried_bytes", ctypes.c_uint64),
        ("resident_gpu_bytes", ctypes.c_uint64),
        ("resident_cpu_bytes", ctypes.c_uint64),
        ("resident_any_bytes", ctypes.c_uint64),
        ("resident_unknown_bytes", ctypes.c_uint64),
    ]


class _PhaseFBatchExecutorParams(ctypes.Structure):
    _fields_ = [
        ("action", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("gpu_uuid", _PhaseFUuid),
        ("async_", ctypes.c_uint8),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("ranges", _PhaseFRange * _UVM_PHASE_F_MAX_RANGES),
        ("results", _PhaseFRangeResult * _UVM_PHASE_F_MAX_RANGES),
        ("rmStatus", ctypes.c_uint32),
    ]


class _PhaseFBatchRegionsParams(ctypes.Structure):
    _fields_ = [
        ("action", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("gpu_uuid", _PhaseFUuid),
        ("async_", ctypes.c_uint8),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("region_ids", ctypes.c_uint64 * _UVM_PHASE_F_MAX_RANGES),
        ("results", _PhaseFRangeResult * _UVM_PHASE_F_MAX_RANGES),
        ("rmStatus", ctypes.c_uint32),
    ]


def _phase_f_batch_abi_matches_kernel() -> bool:
    """Guard the hand-written ctypes ABI before issuing mutable ioctls."""
    return (
        ctypes.sizeof(_PhaseFRangeResult) == 56
        and ctypes.sizeof(_PhaseFBatchExecutorParams) == 4656
        and _PhaseFBatchExecutorParams.ranges.offset == 40
        and _PhaseFBatchExecutorParams.results.offset == 1064
        and _PhaseFBatchExecutorParams.rmStatus.offset == 4648
        and ctypes.sizeof(_PhaseFBatchRegionsParams) == 4144
        and _PhaseFBatchRegionsParams.region_ids.offset == 40
        and _PhaseFBatchRegionsParams.results.offset == 552
        and _PhaseFBatchRegionsParams.rmStatus.offset == 4136
    )


class _ExpertOffloadReleaseParams(ctypes.Structure):
    _fields_ = [
        ("region_id", ctypes.c_uint64),
        ("gpu_uuid", _PhaseFUuid),
        ("flags", ctypes.c_uint32),
        ("base", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
        ("block_count", ctypes.c_uint64),
        ("attempted_blocks", ctypes.c_uint32),
        ("success_blocks", ctypes.c_uint32),
        ("failure_blocks", ctypes.c_uint32),
        ("skipped_blocks", ctypes.c_uint32),
        ("attempted_bytes", ctypes.c_uint64),
        ("released_bytes", ctypes.c_uint64),
        ("failed_bytes", ctypes.c_uint64),
        ("remote_map_blocks", ctypes.c_uint32),
        ("remote_map_failure_blocks", ctypes.c_uint32),
        ("cpu_resident_bytes", ctypes.c_uint64),
        ("remote_mapped_bytes", ctypes.c_uint64),
        ("newly_remote_mapped_bytes", ctypes.c_uint64),
        ("first_failed_block", ctypes.c_uint64),
        ("first_error", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
    ]


class _MoeSlotRemapEntry(ctypes.Structure):
    _fields_ = [
        ("layer_id", ctypes.c_uint32),
        ("expert_id", ctypes.c_uint32),
        ("slot_id", ctypes.c_uint32),
        ("role", ctypes.c_uint32),
        ("va_base", ctypes.c_uint64),
        ("bytes", ctypes.c_uint64),
        ("slot_offset", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
        ("status", ctypes.c_uint32),
    ]


class _MoeSlotRemapBatchParams(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("gpu_uuid", _PhaseFUuid),
        ("validate_managed", ctypes.c_uint32),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("conflict_count", ctypes.c_uint32),
        ("remap_count", ctypes.c_uint32),
        ("tlb_invalidate_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("requested_bytes", ctypes.c_uint64),
        ("accepted_bytes", ctypes.c_uint64),
        ("entries", _MoeSlotRemapEntry * _UVM_MOE_SLOT_REMAP_MAX_ENTRIES),
        ("rmStatus", ctypes.c_uint32),
    ]


class _MoeSlotPlanRegisterParams(ctypes.Structure):
    _fields_ = [
        ("plan_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("layer_id", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("gpu_uuid", _PhaseFUuid),
        ("requested_bytes", ctypes.c_uint64),
        ("entries", _MoeSlotRemapEntry * _UVM_MOE_SLOT_REMAP_MAX_ENTRIES),
        ("rmStatus", ctypes.c_uint32),
    ]


class _MoeSlotPlanExecuteParams(ctypes.Structure):
    _fields_ = [
        ("plan_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("new_generation", ctypes.c_uint64),
        ("requested_bytes", ctypes.c_uint64),
        ("accepted_bytes", ctypes.c_uint64),
        ("layer_id", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("conflict_count", ctypes.c_uint32),
        ("remap_count", ctypes.c_uint32),
        ("tlb_invalidate_count", ctypes.c_uint32),
        ("copy_push_count", ctypes.c_uint32),
        ("owner_cutover", ctypes.c_uint32),
        ("poisoned", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
    ]


class _MoeSlotPlanExecuteFencedParams(ctypes.Structure):
    _fields_ = [
        ("plan_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("new_generation", ctypes.c_uint64),
        ("requested_bytes", ctypes.c_uint64),
        ("accepted_bytes", ctypes.c_uint64),
        ("layer_id", ctypes.c_uint32),
        ("mode", ctypes.c_uint32),
        ("count", ctypes.c_uint32),
        ("success_count", ctypes.c_uint32),
        ("failure_count", ctypes.c_uint32),
        ("conflict_count", ctypes.c_uint32),
        ("remap_count", ctypes.c_uint32),
        ("tlb_invalidate_count", ctypes.c_uint32),
        ("copy_push_count", ctypes.c_uint32),
        ("owner_cutover", ctypes.c_uint32),
        ("poisoned", ctypes.c_uint32),
        ("fence_flags", ctypes.c_uint32),
        ("compute_semaphore_address", ctypes.c_uint64),
        ("completion_semaphore_address", ctypes.c_uint64),
        ("compute_payload", ctypes.c_uint32),
        ("completion_payload", ctypes.c_uint32),
        ("fence_accepted", ctypes.c_uint32),
        ("completion_signaled", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
    ]


class _UvmSlotCudaFence:
    __slots__ = (
        "base",
        "cuda_host_registered",
        "device_base",
        "fd",
        "length",
        "kernel_library_path",
        "next_ticket",
        "transport",
        "warmed",
    )

    def __init__(
        self,
        *,
        fd: int,
        base: int,
        device_base: int,
        length: int,
        transport: str = "stream_memop",
        cuda_host_registered: bool = True,
        kernel_library_path: str = "",
    ) -> None:
        self.fd = int(fd)
        self.base = int(base)
        self.device_base = int(device_base)
        self.length = int(length)
        self.next_ticket = 1
        self.transport = str(transport)
        self.cuda_host_registered = bool(cuda_host_registered)
        self.kernel_library_path = str(kernel_library_path)
        # A raw-UVA helper launch is intentionally deferred until persistent
        # slot backing has been allocated. Launching any CUDA work while mode
        # 8 is still selecting its owning UVM file description changes the
        # initialization state seen by that selection and can make every bank
        # candidate fail.
        self.warmed = self.transport != "raw_kernel"

    @property
    def compute_address(self) -> int:
        return self.base + _UVM_SLOT_FENCE_COMPUTE_OFFSET

    @property
    def completion_address(self) -> int:
        return self.base + _UVM_SLOT_FENCE_COMPLETION_OFFSET

    @property
    def compute_device_address(self) -> int:
        return self.device_base + _UVM_SLOT_FENCE_COMPUTE_OFFSET

    @property
    def completion_device_address(self) -> int:
        return self.device_base + _UVM_SLOT_FENCE_COMPLETION_OFFSET

    def reserve_ticket(self) -> int:
        ticket = int(self.next_ticket)
        # Both CUDA and UVM wait with unsigned GEQ semantics. Reusing 1 after
        # 0xffffffff would therefore let the stale 0xffffffff value satisfy a
        # new wait before the corresponding stream write executes. A session
        # this long-lived must be torn down and recreated with a zeroed pool.
        if ticket <= 0 or ticket > _UVM_MOE_SLOT_UINT32_MAX:
            raise RuntimeError(
                "CUDA fence ticket space exhausted; recreate the slot session"
            )
        self.next_ticket = ticket + 1
        return ticket


class _UvmSlotCudaFenceSignalTimeout(RuntimeError):
    pass


class _MoeSlotPlanDestroyParams(ctypes.Structure):
    _fields_ = [
        ("plan_id", ctypes.c_uint64),
        ("generation", ctypes.c_uint64),
        ("current_generation", ctypes.c_uint64),
        ("destroyed", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
    ]


class _MoeSlotPlanAbortSessionParams(ctypes.Structure):
    _fields_ = [
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("generation_before", ctypes.c_uint64),
        ("released_bytes", ctypes.c_uint64),
        ("remaining_bytes", ctypes.c_uint64),
        ("destroyed_plan_count", ctypes.c_uint32),
        ("released_slot_count", ctypes.c_uint32),
        ("remaining_slot_count", ctypes.c_uint32),
        ("had_registered_session", ctypes.c_uint32),
        ("session_cleared", ctypes.c_uint32),
        ("poisoned", ctypes.c_uint32),
        ("rmStatus", ctypes.c_uint32),
    ]


def _assert_moe_slot_remap_abi() -> None:
    """Reject a userspace/kernel ABI mismatch before an ioctl can be issued."""
    expected = {
        "entry_size": (ctypes.sizeof(_MoeSlotRemapEntry), 48),
        "entry_va_base": (_MoeSlotRemapEntry.va_base.offset, 16),
        "entry_status": (_MoeSlotRemapEntry.status.offset, 44),
        "params_size": (ctypes.sizeof(_MoeSlotRemapBatchParams), 221264),
        "params_requested_bytes": (
            _MoeSlotRemapBatchParams.requested_bytes.offset,
            56,
        ),
        "params_entries": (_MoeSlotRemapBatchParams.entries.offset, 72),
        "params_rm_status": (_MoeSlotRemapBatchParams.rmStatus.offset, 221256),
        "plan_register_size": (ctypes.sizeof(_MoeSlotPlanRegisterParams), 221248),
        "plan_register_entries": (_MoeSlotPlanRegisterParams.entries.offset, 56),
        "plan_execute_size": (ctypes.sizeof(_MoeSlotPlanExecuteParams), 88),
        "plan_execute_layer": (_MoeSlotPlanExecuteParams.layer_id.offset, 40),
        "plan_execute_fenced_size": (
            ctypes.sizeof(_MoeSlotPlanExecuteFencedParams),
            128,
        ),
        "plan_execute_fenced_flags": (
            _MoeSlotPlanExecuteFencedParams.fence_flags.offset,
            84,
        ),
        "plan_execute_fenced_compute_address": (
            _MoeSlotPlanExecuteFencedParams.compute_semaphore_address.offset,
            88,
        ),
        "plan_execute_fenced_rm_status": (
            _MoeSlotPlanExecuteFencedParams.rmStatus.offset,
            124,
        ),
        "plan_destroy_size": (ctypes.sizeof(_MoeSlotPlanDestroyParams), 40),
        "plan_abort_size": (
            ctypes.sizeof(_MoeSlotPlanAbortSessionParams),
            64,
        ),
        "plan_abort_generation": (
            _MoeSlotPlanAbortSessionParams.generation_before.offset,
            8,
        ),
        "plan_abort_released_bytes": (
            _MoeSlotPlanAbortSessionParams.released_bytes.offset,
            16,
        ),
        "plan_abort_destroyed_plans": (
            _MoeSlotPlanAbortSessionParams.destroyed_plan_count.offset,
            32,
        ),
        "plan_abort_had_session": (
            _MoeSlotPlanAbortSessionParams.had_registered_session.offset,
            44,
        ),
        "plan_abort_rm_status": (
            _MoeSlotPlanAbortSessionParams.rmStatus.offset,
            56,
        ),
    }
    mismatches = [
        f"{name}={actual} (expected {wanted})"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise RuntimeError("UVM MoE slot ioctl ABI mismatch: " + ", ".join(mismatches))


_assert_moe_slot_remap_abi()


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.ioctl.restype = ctypes.c_int
_LIBC.mmap.restype = ctypes.c_void_p
_LIBC.munmap.restype = ctypes.c_int
_CUDA_DRIVER: ctypes.CDLL | None = None
_CUDA_STREAM_WRITE_VALUE32 = None
_CUDA_STREAM_WAIT_VALUE32 = None
_CUDA_MEM_HOST_REGISTER = None
_CUDA_MEM_HOST_GET_DEVICE_POINTER = None
_CUDA_MEM_HOST_UNREGISTER = None


def _libc_ioctl(fd: int, request: int, params: ctypes.Structure) -> int:
    return int(
        _LIBC.ioctl(
            ctypes.c_int(fd),
            ctypes.c_ulong(request),
            ctypes.byref(params),
        )
    )


def uvm_slot_cuda_fence_enabled() -> bool:
    raw = os.environ.get(_UVM_SLOT_DIRECT_CUDA_FENCE_ENV, "0")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _uvm_slot_bank_level_commit_enabled() -> bool:
    raw = os.environ.get(_UVM_SLOT_DIRECT_BANK_LEVEL_COMMIT_ENV, "0").strip()
    if raw not in ("0", "1"):
        raise RuntimeError(
            f"{_UVM_SLOT_DIRECT_BANK_LEVEL_COMMIT_ENV} must be 0 or 1"
        )
    return raw == "1"


def _uvm_slot_cuda_fence_kernel_path() -> str:
    return os.environ.get(
        _UVM_SLOT_DIRECT_CUDA_FENCE_KERNEL_LIB_ENV, ""
    ).strip()


def _uvm_slot_cuda_fence_uses_raw_kernel() -> bool:
    return bool(_uvm_slot_cuda_fence_kernel_path())


def _load_cuda_fence_kernel_ops(
    path: str | None = None,
) -> tuple[object, object]:
    global _uvm_slot_cuda_fence_kernel_lib
    global _uvm_slot_cuda_fence_kernel_path_loaded
    global _uvm_slot_cuda_fence_kernel_signal
    global _uvm_slot_cuda_fence_kernel_wait

    requested_path = str(path or "").strip()
    if requested_path:
        requested_path = os.path.realpath(requested_path)
    if (
        _uvm_slot_cuda_fence_kernel_signal is not None
        and _uvm_slot_cuda_fence_kernel_wait is not None
    ):
        if (
            requested_path
            and requested_path != _uvm_slot_cuda_fence_kernel_path_loaded
        ):
            raise RuntimeError(
                "CUDA fence kernel library changed during this process"
            )
        return (
            _uvm_slot_cuda_fence_kernel_signal,
            _uvm_slot_cuda_fence_kernel_wait,
        )

    if not requested_path:
        configured_path = _uvm_slot_cuda_fence_kernel_path()
        if configured_path:
            requested_path = os.path.realpath(configured_path)
    if not requested_path:
        raise RuntimeError(
            f"{_UVM_SLOT_DIRECT_CUDA_FENCE_KERNEL_LIB_ENV} is not configured"
        )
    library = ctypes.CDLL(requested_path)
    signal = getattr(library, "vllm_uvm_slot_fence_signal")
    wait = getattr(library, "vllm_uvm_slot_fence_wait")
    argument_types = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32]
    signal.argtypes = argument_types
    signal.restype = ctypes.c_int
    wait.argtypes = argument_types
    wait.restype = ctypes.c_int
    _uvm_slot_cuda_fence_kernel_lib = library
    _uvm_slot_cuda_fence_kernel_path_loaded = requested_path
    _uvm_slot_cuda_fence_kernel_signal = signal
    _uvm_slot_cuda_fence_kernel_wait = wait
    return signal, wait


def _uvm_slot_cuda_fence_transport_for_address(address: int) -> str:
    address = int(address)
    for fence in _uvm_slot_direct_cuda_fences.values():
        start = int(fence.device_base)
        if start <= address < start + int(fence.length):
            return fence.transport
    return (
        "raw_kernel"
        if _uvm_slot_cuda_fence_uses_raw_kernel()
        else "stream_memop"
    )


def _load_cuda_stream_memops() -> tuple[object, object]:
    global _CUDA_DRIVER, _CUDA_STREAM_WRITE_VALUE32
    global _CUDA_STREAM_WAIT_VALUE32

    if (
        _CUDA_STREAM_WRITE_VALUE32 is not None
        and _CUDA_STREAM_WAIT_VALUE32 is not None
    ):
        return _CUDA_STREAM_WRITE_VALUE32, _CUDA_STREAM_WAIT_VALUE32

    driver = ctypes.CDLL("libcuda.so.1")
    write = getattr(driver, "cuStreamWriteValue32_v2")
    wait = getattr(driver, "cuStreamWaitValue32_v2")
    argument_types = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    write.argtypes = argument_types
    write.restype = ctypes.c_int
    wait.argtypes = argument_types
    wait.restype = ctypes.c_int
    _CUDA_DRIVER = driver
    _CUDA_STREAM_WRITE_VALUE32 = write
    _CUDA_STREAM_WAIT_VALUE32 = wait
    return write, wait


def _load_cuda_host_mapping_ops() -> tuple[object, object, object]:
    global _CUDA_DRIVER, _CUDA_MEM_HOST_REGISTER
    global _CUDA_MEM_HOST_GET_DEVICE_POINTER, _CUDA_MEM_HOST_UNREGISTER

    if (
        _CUDA_MEM_HOST_REGISTER is not None
        and _CUDA_MEM_HOST_GET_DEVICE_POINTER is not None
        and _CUDA_MEM_HOST_UNREGISTER is not None
    ):
        return (
            _CUDA_MEM_HOST_REGISTER,
            _CUDA_MEM_HOST_GET_DEVICE_POINTER,
            _CUDA_MEM_HOST_UNREGISTER,
        )

    driver = _CUDA_DRIVER or ctypes.CDLL("libcuda.so.1")
    register = getattr(driver, "cuMemHostRegister_v2")
    get_device_pointer = getattr(driver, "cuMemHostGetDevicePointer_v2")
    unregister = getattr(driver, "cuMemHostUnregister")
    register.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    register.restype = ctypes.c_int
    get_device_pointer.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    get_device_pointer.restype = ctypes.c_int
    unregister.argtypes = [ctypes.c_void_p]
    unregister.restype = ctypes.c_int
    _CUDA_DRIVER = driver
    _CUDA_MEM_HOST_REGISTER = register
    _CUDA_MEM_HOST_GET_DEVICE_POINTER = get_device_pointer
    _CUDA_MEM_HOST_UNREGISTER = unregister
    return register, get_device_pointer, unregister


def _uvm_slot_cuda_fence_create_locked(
    *, device: int, fd: int
) -> _UvmSlotCudaFence:
    if not _uvm_slot_cuda_fence_uses_raw_kernel():
        raise RuntimeError(
            "registered P3 requires the bounded raw-UVA CUDA fence transport; "
            f"set {_UVM_SLOT_DIRECT_CUDA_FENCE_KERNEL_LIB_ENV}"
        )

    existing = _uvm_slot_direct_cuda_fences.get(int(device))
    if existing is not None:
        if existing.fd != int(fd):
            raise RuntimeError("CUDA fence pool belongs to another UVM session")
        requested_transport = (
            "raw_kernel"
            if _uvm_slot_cuda_fence_uses_raw_kernel()
            else "stream_memop"
        )
        if existing.transport != requested_transport:
            raise RuntimeError("CUDA fence transport changed during the session")
        if requested_transport == "raw_kernel":
            requested_path = os.path.realpath(
                _uvm_slot_cuda_fence_kernel_path()
            )
            if requested_path != existing.kernel_library_path:
                raise RuntimeError(
                    "CUDA fence kernel library changed during the session"
                )
        return existing

    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    length = max(_UVM_SLOT_FENCE_POOL_BYTES, page_size)
    map_failed = ctypes.c_void_p(-1).value
    reserved_raw = _LIBC.mmap(
        None,
        ctypes.c_size_t(length),
        ctypes.c_int(getattr(mmap, "PROT_NONE", 0)),
        ctypes.c_int(mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS),
        ctypes.c_int(-1),
        ctypes.c_longlong(0),
    )
    reserved = int(reserved_raw or 0)
    if reserved == 0 or reserved == map_failed:
        saved_errno = ctypes.get_errno()
        raise OSError(saved_errno, "failed to reserve CUDA fence VA")

    allocated = False
    cuda_registered = False
    try:
        params = _UvmAllocSemaphorePoolParams()
        params.base = reserved
        params.length = length
        # A zero attribute count selects UVM's default uncached, atomic
        # read/write mapping for every GPU registered in this VA space.
        params.gpu_attributes_count = 0
        ctypes.set_errno(0)
        ret = _libc_ioctl(fd, _UVM_ALLOC_SEMAPHORE_POOL_IOCTL, params)
        if ret != 0 or int(params.rmStatus) != 0:
            saved_errno = ctypes.get_errno()
            raise RuntimeError(
                "UVM_ALLOC_SEMAPHORE_POOL failed: "
                f"ret={ret} errno={saved_errno} rmStatus={int(params.rmStatus)}"
            )
        allocated = True

        fixed_flags = mmap.MAP_SHARED | getattr(mmap, "MAP_FIXED", 0x10)
        mapped_raw = _LIBC.mmap(
            ctypes.c_void_p(reserved),
            ctypes.c_size_t(length),
            ctypes.c_int(mmap.PROT_READ | mmap.PROT_WRITE),
            ctypes.c_int(fixed_flags),
            ctypes.c_int(fd),
            ctypes.c_longlong(reserved),
        )
        mapped = int(mapped_raw or 0)
        if mapped != reserved:
            saved_errno = ctypes.get_errno()
            raise OSError(saved_errno, "failed to map UVM CUDA fence pool")

        ctypes.memset(ctypes.c_void_p(reserved), 0, length)
        raw_kernel = _uvm_slot_cuda_fence_uses_raw_kernel()
        if raw_kernel:
            # UVM already maps this semaphore-pool VA into the session GPU
            # page table. A tiny CUDA kernel can consume that raw UVA without
            # libcuda's allocation-metadata check, unlike StreamMemOp. Load
            # the helper now so a missing/incompatible binary fails before
            # the plan or persistent slot backing is registered.
            kernel_library_path = os.path.realpath(
                _uvm_slot_cuda_fence_kernel_path()
            )
            _load_cuda_fence_kernel_ops(kernel_library_path)
            device_pointer_value = reserved
        else:
            kernel_library_path = ""
            register, get_device_pointer, _unregister = (
                _load_cuda_host_mapping_ops()
            )
            # StreamMemOp requires CUDA allocation metadata for the address.
            cuda_result = int(
                register(
                    ctypes.c_void_p(reserved),
                    ctypes.c_size_t(length),
                    ctypes.c_uint32(0x2),  # CU_MEMHOSTREGISTER_DEVICEMAP
                )
            )
            if cuda_result != 0:
                raise RuntimeError(
                    "cuMemHostRegister_v2 fence pool failed: "
                    f"CUresult={cuda_result}"
                )
            cuda_registered = True
            device_pointer = ctypes.c_uint64()
            cuda_result = int(
                get_device_pointer(
                    ctypes.byref(device_pointer),
                    ctypes.c_void_p(reserved),
                    ctypes.c_uint32(0),
                )
            )
            if cuda_result != 0 or int(device_pointer.value) == 0:
                raise RuntimeError(
                    "cuMemHostGetDevicePointer_v2 fence pool failed: "
                    f"CUresult={cuda_result}"
                )
            device_pointer_value = int(device_pointer.value)

        fence = _UvmSlotCudaFence(
            fd=fd,
            base=reserved,
            device_base=device_pointer_value,
            length=length,
            transport="raw_kernel" if raw_kernel else "stream_memop",
            cuda_host_registered=cuda_registered,
            kernel_library_path=kernel_library_path,
        )
        _uvm_slot_direct_cuda_fences[int(device)] = fence
        return fence
    except Exception:
        if cuda_registered:
            with suppress(Exception):
                _register, _get_device_pointer, unregister = (
                    _load_cuda_host_mapping_ops()
                )
                unregister(ctypes.c_void_p(reserved))
        _LIBC.munmap(ctypes.c_void_p(reserved), ctypes.c_size_t(length))
        if allocated:
            free_params = _UvmFreeParams()
            free_params.base = reserved
            with suppress(Exception):
                _libc_ioctl(fd, _UVM_FREE_IOCTL, free_params)
        raise


def _uvm_slot_cuda_fence_warm_locked(*, device: int, fd: int) -> None:
    """Warm raw CUDA fence helpers after slot-bank preallocation.

    The fence pool itself must exist while selecting the UVM fd, but launching
    a CUDA kernel at that point perturbs mode-8 bank ownership. Registered-plan
    callers reach this function only after their persistent bank(s) exist and
    before any plan is registered, which is the safe one-time boundary.
    """
    fence = _uvm_slot_direct_cuda_fences.get(int(device))
    if fence is None:
        raise RuntimeError("CUDA fence pool is missing during transport warmup")
    if fence.fd != int(fd):
        raise RuntimeError("CUDA fence pool/session fd mismatch during warmup")
    if fence.warmed:
        return
    if fence.transport != "raw_kernel":
        fence.warmed = True
        return

    signal, wait = _load_cuda_fence_kernel_ops(fence.kernel_library_path)
    with torch.cuda.device(int(device)):
        warm_stream = torch.cuda.Stream(device=int(device))
        signal_status = int(
            signal(
                int(warm_stream.cuda_stream),
                int(fence.compute_device_address),
                0,
            )
        )
        wait_status = int(
            wait(
                int(warm_stream.cuda_stream),
                int(fence.completion_device_address),
                0,
            )
        )
        if signal_status != 0 or wait_status != 0:
            raise RuntimeError(
                "CUDA fence raw transport warmup failed: "
                f"signal={signal_status} wait={wait_status}"
            )
        warm_stream.synchronize()

    ctypes.c_uint32.from_address(fence.compute_address).value = 0
    ctypes.c_uint32.from_address(fence.completion_address).value = 0
    fence.warmed = True


def _uvm_slot_cuda_fence_destroy_locked(*, device: int, fd: int) -> None:
    fence = _uvm_slot_direct_cuda_fences.get(int(device))
    if fence is None:
        return
    if fence.fd != int(fd):
        raise RuntimeError("CUDA fence pool/session fd mismatch during cleanup")

    if fence.device_base != 0:
        if fence.cuda_host_registered:
            _register, _get_device_pointer, unregister = (
                _load_cuda_host_mapping_ops()
            )
            cuda_result = int(unregister(ctypes.c_void_p(fence.base)))
            if cuda_result != 0:
                raise RuntimeError(
                    "cuMemHostUnregister fence pool failed: "
                    f"CUresult={cuda_result}"
                )
            fence.cuda_host_registered = False
        # UVM_FREE intentionally rejects a semaphore pool while its CPU VMA is
        # present. Drop any CUDA host registration first, then the VMA, then
        # the UVM range itself.
        if _LIBC.munmap(
            ctypes.c_void_p(fence.base), ctypes.c_size_t(fence.length)
        ) != 0:
            saved_errno = ctypes.get_errno()
            raise OSError(saved_errno, "munmap CUDA fence pool failed")
        fence.device_base = 0

    free_params = _UvmFreeParams()
    free_params.base = int(fence.base)
    ctypes.set_errno(0)
    ret = _libc_ioctl(fd, _UVM_FREE_IOCTL, free_params)
    if ret != 0 or int(free_params.rmStatus) != 0:
        saved_errno = ctypes.get_errno()
        raise RuntimeError(
            "UVM_FREE CUDA fence pool failed: "
            f"ret={ret} errno={saved_errno} rmStatus={int(free_params.rmStatus)}"
        )
    _uvm_slot_direct_cuda_fences.pop(int(device), None)


def _uvm_slot_cuda_fence_current_stream(device: int) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA stream fence requested without CUDA")
    return int(torch.cuda.current_stream(int(device)).cuda_stream)


def _uvm_slot_cuda_fence_enqueue_compute(
    *, stream: int, address: int, ticket: int
) -> None:
    if _uvm_slot_cuda_fence_transport_for_address(address) == "raw_kernel":
        signal, _wait = _load_cuda_fence_kernel_ops()
        result = int(signal(int(stream), int(address), int(ticket)))
        if result != 0:
            raise RuntimeError(
                f"CUDA fence signal kernel failed: cudaError={result}"
            )
        return
    write, _wait = _load_cuda_stream_memops()
    result = int(
        write(
            ctypes.c_void_p(int(stream)),
            ctypes.c_uint64(int(address)),
            ctypes.c_uint32(int(ticket)),
            ctypes.c_uint32(0),
        )
    )
    if result != 0:
        raise RuntimeError(f"cuStreamWriteValue32_v2 failed: CUresult={result}")


def _uvm_slot_cuda_fence_wait_compute_visible(
    *, address: int, ticket: int
) -> None:
    if _uvm_slot_cuda_fence_transport_for_address(address) != "raw_kernel":
        return
    # A UVM push waiting on a GPU semaphore has no finite cancellation path.
    # Observe the same system-memory ticket with a bounded CPU poll before
    # entering ioctl 96.
    cell = ctypes.c_uint32.from_address(int(address))
    deadline = time.monotonic() + _UVM_SLOT_FENCE_COMPUTE_TIMEOUT_SEC
    while int(cell.value) < int(ticket):
        if time.monotonic() >= deadline:
            raise _UvmSlotCudaFenceSignalTimeout(
                "timed out waiting for raw CUDA compute fence ticket"
            )
        time.sleep(0)


def _uvm_slot_cuda_fence_write_compute(
    *, stream: int, address: int, ticket: int
) -> None:
    _uvm_slot_cuda_fence_enqueue_compute(
        stream=int(stream), address=int(address), ticket=int(ticket)
    )
    _uvm_slot_cuda_fence_wait_compute_visible(
        address=int(address), ticket=int(ticket)
    )


def _uvm_slot_cuda_fence_wait_completion(
    *, stream: int, address: int, ticket: int
) -> None:
    if _uvm_slot_cuda_fence_transport_for_address(address) == "raw_kernel":
        _signal, wait = _load_cuda_fence_kernel_ops()
        result = int(wait(int(stream), int(address), int(ticket)))
        if result != 0:
            raise RuntimeError(
                f"CUDA fence wait kernel failed: cudaError={result}"
            )
        return
    _write, wait = _load_cuda_stream_memops()
    result = int(
        wait(
            ctypes.c_void_p(int(stream)),
            ctypes.c_uint64(int(address)),
            ctypes.c_uint32(int(ticket)),
            ctypes.c_uint32(0),
        )
    )
    if result != 0:
        raise RuntimeError(f"cuStreamWaitValue32_v2 failed: CUresult={result}")


def _uvm_slot_abort_after_fence_failure_locked(
    *, device: int, fence_enabled: bool
) -> tuple[dict[str, object] | None, str | None]:
    """Establish the old-compute boundary before destructive recovery.

    A fenced execute can fail before the kernel accepts and waits on the
    compute ticket. In that case, the stream write and the work preceding it
    may still be outstanding, so aborting the slot session could clear PTEs or
    backing still in use. The synchronization is failure-only; if it cannot be
    proven, leave all driver resources alive for process teardown.
    """
    if fence_enabled:
        try:
            torch.cuda.synchronize(int(device))
        except Exception as exc:
            return None, repr(exc)

    return _driver_slot_abort_session_impl(device=int(device)), None


def _phase_f_gpu_uuid(device: int) -> _PhaseFUuid:
    if device < 0:
        device = torch.cuda.current_device()
    raw_uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    if raw_uuid is None:
        raise RuntimeError("torch.cuda device properties do not expose UUID")
    if isinstance(raw_uuid, bytes):
        uuid_bytes = raw_uuid
    else:
        uuid_text = str(raw_uuid)
        if uuid_text.startswith("GPU-"):
            uuid_text = uuid_text[4:]
        uuid_bytes = uuidlib.UUID(uuid_text).bytes
    if len(uuid_bytes) != 16:
        raise RuntimeError(f"Unexpected CUDA UUID size: {len(uuid_bytes)}")
    return _PhaseFUuid((ctypes.c_uint8 * 16).from_buffer_copy(uuid_bytes))


def _open_initialized_uvm_fd() -> int:
    fd = os.open("/dev/nvidia-uvm", os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
    params = _UvmInitializeParams()
    ret = _libc_ioctl(fd, _UVM_INITIALIZE_IOCTL, params)
    if ret != 0 or int(params.rmStatus) != 0:
        saved_errno = ctypes.get_errno()
        os.close(fd)
        raise RuntimeError(
            "UVM_INITIALIZE failed: "
            f"ret={ret} errno={saved_errno} rmStatus={int(params.rmStatus)}"
        )
    return fd


def _fd_link_is_uvm(link_target: str) -> bool:
    base = os.path.basename(link_target)
    return base == "nvidia-uvm" or base.startswith("nvidia-uvm ")


def _iter_existing_uvm_fd_sources() -> list[int]:
    try:
        fd_names = sorted(
            os.listdir("/proc/self/fd"),
            key=lambda value: int(value) if value.isdigit() else -1,
        )
    except OSError:
        return []

    sources: list[int] = []
    for name in fd_names:
        if not name.isdigit():
            continue
        source_fd = int(name)
        if source_fd < 3:
            continue
        try:
            link_target = os.readlink(f"/proc/self/fd/{source_fd}")
        except OSError:
            continue
        if not _fd_link_is_uvm(link_target):
            continue
        sources.append(source_fd)
    return sources


def _initialize_duplicated_uvm_fd(source_fd: int) -> int | None:
    try:
        fd = os.dup(source_fd)
    except OSError:
        return None

    params = _UvmInitializeParams()
    ret = _libc_ioctl(fd, _UVM_INITIALIZE_IOCTL, params)
    if ret == 0 and int(params.rmStatus) == 0:
        return fd

    with suppress(OSError):
        os.close(fd)
    return None


def _adopt_existing_uvm_fd() -> int | None:
    for source_fd in _iter_existing_uvm_fd_sources():
        fd = _initialize_duplicated_uvm_fd(source_fd)
        if fd is not None:
            return fd
    return None


def _iter_initialized_existing_uvm_fds():
    for source_fd in _iter_existing_uvm_fd_sources():
        fd = _initialize_duplicated_uvm_fd(source_fd)
        if fd is not None:
            yield fd


def _open_or_adopt_initialized_uvm_fd(*, prefer_existing: bool = False) -> int:
    if prefer_existing:
        fd = _adopt_existing_uvm_fd()
        if fd is not None:
            return fd

    return _open_initialized_uvm_fd()


def _uvm_slot_direct_session_owner_pid_locked(device: int) -> int | None:
    """Return the creator PID, backfilling only legacy in-process state."""
    device = int(device)
    if device not in _uvm_slot_direct_sessions:
        return None
    owner_pid = _uvm_slot_direct_session_pids.get(device)
    if owner_pid is None:
        # This path keeps old serialized/tests-created state usable. Every real
        # session created below records its PID before it becomes observable,
        # so a forked production session can never be adopted here.
        owner_pid = os.getpid()
        _uvm_slot_direct_session_pids[device] = owner_pid
    return owner_pid


def _uvm_slot_direct_inherited_pid_locked(device: int) -> int | None:
    owner_pid = _uvm_slot_direct_session_owner_pid_locked(int(device))
    if owner_pid is None or owner_pid == os.getpid():
        return None
    return owner_pid


def _uvm_slot_direct_clear_resource_cache_locked(device: int) -> None:
    """Forget only driver resources proven gone; retain the fd until close."""
    device = int(device)
    _uvm_slot_direct_owners.pop(device, None)
    _uvm_slot_direct_plans.pop(device, None)
    _uvm_slot_direct_plan_generations.pop(device, None)
    _uvm_slot_direct_active_plan_ids.pop(device, None)
    _uvm_slot_direct_poisoned.pop(device, None)


def _uvm_slot_direct_clear_session_cache_locked(device: int) -> None:
    device = int(device)
    _uvm_slot_direct_clear_resource_cache_locked(device)
    _uvm_slot_direct_sessions.pop(device, None)
    _uvm_slot_direct_session_pids.pop(device, None)
    _uvm_slot_direct_cuda_fences.pop(device, None)
    _uvm_slot_direct_cuda_fence_cleanup_deferred.discard(device)


def _uvm_slot_direct_mark_poisoned_locked(device: int, reason: str) -> None:
    _uvm_slot_direct_poisoned[int(device)] = str(reason)


def _uvm_slot_direct_block_reason_locked(
    device: int,
    *,
    allow_poisoned: bool = False,
) -> tuple[str, str] | None:
    device = int(device)
    inherited_pid = _uvm_slot_direct_inherited_pid_locked(device)
    if inherited_pid is not None:
        return (
            "fork_inherited_slot_session",
            f"session belongs to pid {inherited_pid}, current pid is {os.getpid()}",
        )
    poison_reason = _uvm_slot_direct_poisoned.get(device)
    if poison_reason is not None and not allow_poisoned:
        return ("slot_session_poisoned", poison_reason)
    return None


def _close_uvm_slot_direct_sessions() -> None:
    """Best-effort exit cleanup with an explicit CUDA completion boundary."""
    with _uvm_slot_direct_lock:
        devices = [
            (
                device,
                _uvm_slot_direct_inherited_pid_locked(device) is not None,
            )
            for device in _uvm_slot_direct_sessions
        ]
    for device, inherited in devices:
        with _uvm_slot_direct_lock:
            cleanup_deferred = (
                device in _uvm_slot_direct_cuda_fence_cleanup_deferred
            )
        # A bounded synchronous fence failure can leave the old-compute
        # boundary unproven. Never call cudaDeviceSynchronize in that state:
        # the process/driver teardown boundary is the only safe cleanup.
        if not inherited and cleanup_deferred:
            logger.error(
                "UVM slot session on device %s requires process-teardown-only "
                "cleanup; skipping CUDA synchronization",
                device,
            )
            continue
        if not inherited and not cleanup_deferred:
            try:
                torch.cuda.synchronize(device)
            except Exception:
                # Releasing slot PTE/backing without proving all user kernels
                # complete can create a use-after-free. Leave the descriptor
                # open and let final process/driver teardown reclaim it after
                # CUDA channels are stopped.
                logger.exception(
                    "UVM slot session exit synchronization failed on device %s; "
                    "deferring cleanup to process teardown",
                    device,
                )
                continue
        result = close_uvm_slot_direct_session(device=device)
        if not bool(result.get("ok")):
            logger.error(
                "UVM slot session cleanup failed at process exit: %s",
                result,
            )


def _register_uvm_slot_direct_close() -> None:
    global _uvm_slot_direct_close_registered
    if not _uvm_slot_direct_close_registered:
        atexit.register(_close_uvm_slot_direct_sessions)
        _uvm_slot_direct_close_registered = True


def register_uvm_slot_direct_policy_cleanup(
    callback: Callable[[int], None],
) -> None:
    """Register the policy-state cleanup paired with low-level session close."""
    global _uvm_slot_direct_policy_cleanup

    with _uvm_slot_direct_lock:
        _uvm_slot_direct_policy_cleanup = callback


def _run_uvm_slot_direct_policy_cleanup(device: int) -> None:
    with _uvm_slot_direct_lock:
        callback = _uvm_slot_direct_policy_cleanup
    if callback is not None:
        callback(int(device))


def _moe_slot_result(
    *,
    ok: bool,
    reason: str,
    mode: int,
    requested_count: int,
    requested_bytes: int,
    error: str = "",
) -> dict[str, object]:
    return {
        "ok": bool(ok),
        "reason": str(reason),
        "error": str(error),
        "mode": int(mode),
        "requested_count": int(requested_count),
        "success_count": 0,
        "failure_count": 0,
        "conflict_count": 0,
        "requested_bytes": int(requested_bytes),
        "accepted_bytes": 0,
        "remap_count": 0,
        "tlb_invalidate_count": 0,
        "copy_push_count": 0,
        "backing_reuse_count": 0,
        "owner_cutover": False,
        "owner_cutover_count": 0,
        "owner_cutover_trusted": True,
        "owner_state_indeterminate": False,
        "ioctl_calls": 0,
        "ioctl_wall_ns": 0,
        "wall_ns": 0,
        "errno": 0,
        "rm_status": 0,
        "entry_statuses": [],
        "batches": [],
    }


def _normalize_moe_slot_ranges(
    ranges: Sequence[Sequence[int]],
) -> tuple[list[tuple[int, int, int, int, int, int, int]], str]:
    """Validate ioctl-91 entries without allowing lossy integer coercion."""
    try:
        records = list(ranges)
    except TypeError:
        return [], "ranges must be a finite sequence"
    if not records:
        return [], "at least one slot range is required"

    normalized: list[tuple[int, int, int, int, int, int, int]] = []
    slot_keys: set[tuple[int, int]] = set()
    for index, record in enumerate(records):
        try:
            values = tuple(record)
        except TypeError:
            return [], f"range {index} is not a six-field sequence"
        if len(values) not in (6, 7):
            return [], f"range {index} must contain six or seven fields"
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in values
        ):
            return [], f"range {index} fields must be plain integers"

        base, length, layer_id, expert_id, slot_id, slot_offset = values[:6]
        flags = values[6] if len(values) == 7 else UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT
        if base <= 0 or length <= 0:
            return [], f"range {index} has a non-positive base or length"
        if length > _UVM_MOE_SLOT_REMAP_MAX_ENTRY_BYTES:
            return [], f"range {index} exceeds the 4 MiB slot-direct limit"
        if (
            base % _UVM_MOE_SLOT_PAGE_BYTES
            or length % _UVM_MOE_SLOT_PAGE_BYTES
            or slot_offset % _UVM_MOE_SLOT_PAGE_BYTES
        ):
            return [], f"range {index} is not 4 KiB aligned"
        if base > _UVM_MOE_SLOT_UINT64_MAX - length:
            return [], f"range {index} virtual-address end overflows uint64"
        if slot_offset < 0 or slot_offset > _UVM_MOE_SLOT_UINT64_MAX - length:
            return [], f"range {index} slot end overflows uint64"
        for name, value in (
            ("layer_id", layer_id),
            ("expert_id", expert_id),
            ("slot_id", slot_id),
            ("flags", flags),
        ):
            if value < 0 or value > _UVM_MOE_SLOT_UINT32_MAX:
                return [], f"range {index} {name} is outside uint32"
        if flags & ~(
            UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT
            | UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
        ):
            return [], f"range {index} contains unsupported slot flags"

        slot_key = (layer_id, slot_id)
        if slot_key in slot_keys:
            return [], f"range {index} repeats logical slot {slot_key}"
        slot_keys.add(slot_key)
        normalized.append(
            (base, length, layer_id, expert_id, slot_id, slot_offset, flags)
        )

    by_address = sorted(normalized, key=lambda item: item[0])
    for previous, current in zip(by_address, by_address[1:]):
        if current[0] < previous[0] + previous[1]:
            return [], "source managed ranges overlap"
    return normalized, ""


def _moe_slot_batch_result(
    *,
    params: _MoeSlotRemapBatchParams,
    ret: int,
    errno_value: int,
    mode: int,
    expected_bytes: int,
    input_indices: list[int],
) -> dict[str, object]:
    count = int(params.count)
    raw_entry_statuses = [
        int(params.entries[index].status) for index in range(count)
    ]
    reported_successes = int(params.success_count)
    reported_failures = int(params.failure_count)
    zero_statuses = sum(status == 0 for status in raw_entry_statuses)
    nonzero_statuses = count - zero_statuses
    successful_bytes = sum(
        int(params.entries[index].bytes)
        for index, status in enumerate(raw_entry_statuses)
        if status == 0
    )
    status_summary_consistent = (
        ret == 0
        and reported_successes == zero_statuses
        and reported_failures == nonzero_statuses
        and reported_successes + reported_failures == count
        and int(params.requested_bytes) == expected_bytes
        and (
            (reported_failures == 0 and int(params.rmStatus) == 0)
            or (reported_failures > 0 and int(params.rmStatus) != 0)
        )
        and (
            mode == _UVM_MOE_SLOT_REMAP_MODE_RELEASE
            or int(params.accepted_bytes) == successful_bytes
        )
    )
    # ctypes zero-initializes entry.status. If ioctl fails before the UVM
    # handler writes its output, a raw zero is not proof that the entry was
    # committed. Expose only successes corroborated by the aggregate counters;
    # callers use None to keep owner/rebind state fail-closed.
    entry_statuses: list[int | None] = [
        status if status != 0 or status_summary_consistent else None
        for status in raw_entry_statuses
    ]
    owner_cutover_trusted = (
        mode == _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION
        and status_summary_consistent
        and (
            (reported_failures == 0 and int(params.remap_count) == count)
            or (reported_failures > 0 and int(params.remap_count) == 0)
        )
    )
    owner_cutover: bool | None = False
    if mode == _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION:
        owner_cutover = (
            int(params.remap_count) == count
            if owner_cutover_trusted
            else None
        )
    problems: list[str] = []
    if ret != 0:
        problems.append("ioctl_ret")
    if int(params.rmStatus) != 0:
        problems.append("rm_status")
    if int(params.success_count) != count:
        problems.append("short_success")
    if int(params.failure_count) != 0:
        problems.append("failure_count")
    if int(params.conflict_count) != 0:
        problems.append("conflict_count")
    if int(params.requested_bytes) != expected_bytes:
        problems.append("requested_bytes")
    if mode == _UVM_MOE_SLOT_REMAP_MODE_RELEASE:
        if count and int(params.accepted_bytes) <= 0:
            problems.append("missing_reclaimed_bytes")
    elif int(params.accepted_bytes) != expected_bytes:
        problems.append("accepted_bytes")
    if any(status is None for status in entry_statuses):
        problems.append("untrusted_entry_status")
    elif any(status != 0 for status in entry_statuses):
        problems.append("entry_status")

    if mode == _UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC:
        required_reuses = sum(
            bool(
                int(params.entries[index].flags)
                & UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
            )
            for index in range(count)
        )
        if (
            int(params.remap_count) != 0
            or int(params.tlb_invalidate_count) != 0
        ):
            problems.append("unexpected_prealloc_side_effect")
        if int(params.reserved) < required_reuses or int(params.reserved) > count:
            problems.append("backing_reuse_count")
    elif mode in (
        _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_PER_BLOCK_PUSH,
        _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH,
        _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_PTE,
        _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION,
        _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE,
    ):
        if int(params.remap_count) != count:
            problems.append("short_remap")
        if count and int(params.tlb_invalidate_count) == 0:
            problems.append("missing_tlb_invalidate")
        if count and int(params.reserved) == 0:
            problems.append("missing_copy_push")
    elif mode == _UVM_MOE_SLOT_REMAP_MODE_RELEASE:
        if int(params.remap_count) != count:
            problems.append("short_release")
        if int(params.tlb_invalidate_count) > count:
            problems.append("excess_tlb_invalidate")
        if int(params.reserved) != 0:
            problems.append("unexpected_release_copy_push")
    else:
        problems.append("unsupported_mode")

    return {
        "ok": not problems,
        "reason": "ok" if not problems else problems[0],
        "problems": problems,
        "input_indices": list(input_indices),
        "count": count,
        "success_count": int(params.success_count),
        "failure_count": int(params.failure_count),
        "conflict_count": int(params.conflict_count),
        "requested_bytes": int(params.requested_bytes),
        "accepted_bytes": int(params.accepted_bytes),
        "remap_count": int(params.remap_count),
        "tlb_invalidate_count": int(params.tlb_invalidate_count),
        "copy_push_count": (
            int(params.reserved)
            if mode
            in (
                _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_PER_BLOCK_PUSH,
                _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH,
                _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_PTE,
                _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION,
                _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE,
            )
            else 0
        ),
        "backing_reuse_count": (
            int(params.reserved)
            if mode == _UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC
            else 0
        ),
        "ioctl_ret": int(ret),
        "errno": int(errno_value),
        "rm_status": int(params.rmStatus),
        "entry_statuses": entry_statuses,
        "owner_cutover": owner_cutover,
        "owner_cutover_count": (
            int(params.remap_count)
            if mode == _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION
            else 0
        ),
        "owner_cutover_trusted": owner_cutover_trusted,
    }


def _partition_moe_slot_batches(
    normalized: list[tuple[int, int, int, int, int, int, int]],
    *,
    mode: int,
) -> list[list[tuple[int, tuple[int, int, int, int, int, int, int]]]]:
    indexed = list(enumerate(normalized))
    # Preserve complete Qwen layers in one transaction.  The expanded ABI also
    # supplies scratch space for a 4,500-entry static-alias tier delta; normal
    # per-layer callers still submit only their own 1,152 envelopes.
    return [
        indexed[start : start + _UVM_MOE_SLOT_REMAP_MAX_ENTRIES]
        for start in range(0, len(indexed), _UVM_MOE_SLOT_REMAP_MAX_ENTRIES)
    ]


def _run_moe_slot_batches(
    fd: int,
    normalized: list[tuple[int, int, int, int, int, int, int]],
    *,
    gpu_uuid: _PhaseFUuid,
    mode: int,
) -> dict[str, object]:
    total_bytes = sum(record[1] for record in normalized)
    result = _moe_slot_result(
        ok=False,
        reason="not_submitted",
        mode=mode,
        requested_count=len(normalized),
        requested_bytes=total_bytes,
    )
    if mode == _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION:
        result["owner_cutover"] = None
        result["owner_cutover_trusted"] = False
    statuses: list[int | None] = [None] * len(normalized)
    batches: list[dict[str, object]] = []
    partitioned = _partition_moe_slot_batches(normalized, mode=mode)

    for indexed_chunk in partitioned:
        input_indices = [index for index, _record in indexed_chunk]
        chunk = [record for _index, record in indexed_chunk]
        params = _MoeSlotRemapBatchParams()
        params.mode = int(mode)
        params.count = len(chunk)
        params.gpu_uuid = gpu_uuid
        params.validate_managed = (
            0 if mode == _UVM_MOE_SLOT_REMAP_MODE_RELEASE else 1
        )
        expected_bytes = 0
        for entry_index, record in enumerate(chunk):
            base, length, layer_id, expert_id, slot_id, slot_offset, flags = record
            entry = params.entries[entry_index]
            entry.layer_id = layer_id
            entry.expert_id = expert_id
            entry.slot_id = slot_id
            entry.role = 0
            entry.va_base = base
            entry.bytes = length
            entry.slot_offset = slot_offset
            entry.flags = flags
            expected_bytes += length

        ctypes.set_errno(0)
        ioctl_start_ns = time.perf_counter_ns()
        try:
            ret = _libc_ioctl(fd, _UVM_MOE_SLOT_REMAP_BATCH_IOCTL, params)
            errno_value = ctypes.get_errno() if int(ret) != 0 else 0
        except Exception as exc:
            result["ioctl_wall_ns"] = int(result["ioctl_wall_ns"]) + (
                time.perf_counter_ns() - ioctl_start_ns
            )
            result["reason"] = "ioctl_exception"
            result["error"] = repr(exc)
            break
        ioctl_wall_ns = time.perf_counter_ns() - ioctl_start_ns

        batch = _moe_slot_batch_result(
            params=params,
            ret=int(ret),
            errno_value=errno_value,
            mode=mode,
            expected_bytes=expected_bytes,
            input_indices=input_indices,
        )
        batch["wall_ns"] = ioctl_wall_ns
        batches.append(batch)
        for entry_index, status in enumerate(batch["entry_statuses"]):
            statuses[input_indices[entry_index]] = (
                status if type(status) is int else None
            )

        for field in (
            "success_count",
            "failure_count",
            "conflict_count",
            "accepted_bytes",
            "remap_count",
            "tlb_invalidate_count",
            "copy_push_count",
            "backing_reuse_count",
        ):
            result[field] = int(result[field]) + int(batch[field])
        result["ioctl_calls"] = int(result["ioctl_calls"]) + 1
        result["ioctl_wall_ns"] = int(result["ioctl_wall_ns"]) + ioctl_wall_ns
        result["errno"] = int(batch["errno"])
        result["rm_status"] = int(batch["rm_status"])
        if mode == _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION:
            result["owner_cutover"] = batch["owner_cutover"]
            result["owner_cutover_count"] = int(batch["owner_cutover_count"])
            result["owner_cutover_trusted"] = bool(
                batch["owner_cutover_trusted"]
            )
        if not bool(batch["ok"]):
            result["reason"] = str(batch["reason"])
            break

    result["entry_statuses"] = statuses
    result["batches"] = batches
    result["ok"] = (
        len(batches) == len(partitioned)
        and all(bool(batch["ok"]) for batch in batches)
        and int(result["success_count"]) == len(normalized)
        and (
            mode == _UVM_MOE_SLOT_REMAP_MODE_RELEASE
            or int(result["accepted_bytes"]) == total_bytes
        )
    )
    if bool(result["ok"]):
        result["reason"] = "ok"
    return result


def _record_preallocated_slot_owners(
    device: int,
    normalized: list[tuple[int, int, int, int, int, int, int]],
    result: dict[str, object],
) -> None:
    owners = _uvm_slot_direct_owners.setdefault(int(device), {})
    batches = result.get("batches")
    if not isinstance(batches, list):
        return
    for batch in batches:
        if not isinstance(batch, dict):
            continue
        indices = batch.get("input_indices")
        statuses = batch.get("entry_statuses")
        if not isinstance(indices, list) or not isinstance(statuses, list):
            continue
        for input_index, status in zip(indices, statuses):
            if type(status) is not int or status != 0:
                continue
            _base, _length, layer_id, expert_id, slot_id, _offset, _flags = (
                normalized[int(input_index)]
            )
            owners[(layer_id, slot_id)] = expert_id


def _driver_slot_preallocate_ranges_impl(
    ranges: Sequence[Sequence[int]],
    *,
    device: int = 0,
) -> dict[str, object]:
    """Preallocate/rebind driver VIDMEM slots on the current CUDA VA-space.

    Each entry is ``(base, length, layer_id, expert_id, slot_id, slot_offset)``
    with an optional seventh ``flags`` field (default ``NO_EVICT``).
    Only already-open UVM file descriptions are duplicated. The selected fd is
    retained for the matching mode-10 restore, so restore cannot accidentally
    target another CUDA VA-space.
    """
    normalized, validation_error = _normalize_moe_slot_ranges(ranges)
    total_bytes = sum(record[1] for record in normalized)
    with _uvm_slot_direct_lock:
        blocked = _uvm_slot_direct_block_reason_locked(int(device))
    if blocked is not None:
        reason, error = blocked
        return _moe_slot_result(
            ok=False,
            reason=reason,
            mode=_UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC,
            requested_count=len(normalized),
            requested_bytes=total_bytes,
            error=error,
        )
    if validation_error:
        return _moe_slot_result(
            ok=False,
            reason="validation_failed",
            mode=_UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC,
            requested_count=len(normalized),
            requested_bytes=total_bytes,
            error=validation_error,
        )
    try:
        gpu_uuid = _phase_f_gpu_uuid(device)
    except Exception as exc:
        return _moe_slot_result(
            ok=False,
            reason="gpu_uuid_failed",
            mode=_UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC,
            requested_count=len(normalized),
            requested_bytes=total_bytes,
            error=repr(exc),
        )
    uuid_bytes = bytes(gpu_uuid.uuid)

    with _uvm_slot_direct_lock:
        blocked = _uvm_slot_direct_block_reason_locked(int(device))
        if blocked is not None:
            reason, error = blocked
            return _moe_slot_result(
                ok=False,
                reason=reason,
                mode=_UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC,
                requested_count=len(normalized),
                requested_bytes=total_bytes,
                error=error,
            )
        cached = _uvm_slot_direct_sessions.get(int(device))
        if cached is not None:
            fd, cached_uuid = cached
            if cached_uuid != uuid_bytes:
                return _moe_slot_result(
                    ok=False,
                    reason="cached_gpu_uuid_mismatch",
                    mode=_UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC,
                    requested_count=len(normalized),
                    requested_bytes=total_bytes,
                )
            result = _run_moe_slot_batches(
                fd,
                normalized,
                gpu_uuid=gpu_uuid,
                mode=_UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC,
            )
            result["fd_candidate_attempts"] = 1
            result["fd_candidate_ioctl_calls"] = int(result["ioctl_calls"])
            _record_preallocated_slot_owners(device, normalized, result)
            return result

        sources = _iter_existing_uvm_fd_sources()
        if not sources:
            return _moe_slot_result(
                ok=False,
                reason="no_existing_uvm_fd",
                mode=_UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC,
                requested_count=len(normalized),
                requested_bytes=total_bytes,
            )

        last_result: dict[str, object] | None = None
        candidate_attempts = 0
        candidate_ioctl_calls = 0
        fence_enabled = uvm_slot_cuda_fence_enabled()
        fence_candidate_attempts = 0
        fence_candidate_rejections = 0
        last_fence_error = ""
        for source_fd in sources:
            fd = _initialize_duplicated_uvm_fd(source_fd)
            if fd is None:
                continue
            candidate_attempts += 1

            # A process can have several initialized UVM file descriptions.
            # The first one which owns a managed allocation is not necessarily
            # usable as backing for cuMemHostRegister in the active PyTorch CUDA
            # context. Select a fenced session only from candidates satisfying
            # both contracts, before mode 8 can allocate any persistent slot
            # backing on the wrong file description.
            if fence_enabled:
                fence_candidate_attempts += 1
                try:
                    _uvm_slot_cuda_fence_create_locked(device=int(device), fd=fd)
                except Exception as exc:
                    fence_candidate_rejections += 1
                    last_fence_error = repr(exc)
                    with suppress(OSError):
                        os.close(fd)
                    continue

            candidate = _run_moe_slot_batches(
                fd,
                normalized,
                gpu_uuid=gpu_uuid,
                mode=_UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC,
            )
            candidate_ioctl_calls += int(candidate["ioctl_calls"])
            made_progress = (
                int(candidate["success_count"]) > 0
                or int(candidate["accepted_bytes"]) > 0
            )
            if bool(candidate["ok"]) or made_progress:
                candidate["fd_candidate_attempts"] = candidate_attempts
                candidate["fd_candidate_ioctl_calls"] = candidate_ioctl_calls
                candidate["cuda_fence_fd_candidate_attempts"] = (
                    fence_candidate_attempts
                )
                candidate["cuda_fence_fd_candidate_rejections"] = (
                    fence_candidate_rejections
                )
                _uvm_slot_direct_sessions[int(device)] = (fd, uuid_bytes)
                _uvm_slot_direct_session_pids[int(device)] = os.getpid()
                _uvm_slot_direct_poisoned.pop(int(device), None)
                _register_uvm_slot_direct_close()
                _record_preallocated_slot_owners(device, normalized, candidate)
                return candidate
            last_result = candidate
            if fence_enabled:
                try:
                    _uvm_slot_cuda_fence_destroy_locked(
                        device=int(device), fd=fd
                    )
                except Exception as exc:
                    # The candidate cannot be discarded while a registered
                    # host mapping or semaphore-pool range may remain. Retain
                    # the fd as an abort-only session so process cleanup can
                    # retry the ordered unregister/munmap/UVM_FREE sequence.
                    _uvm_slot_direct_sessions[int(device)] = (fd, uuid_bytes)
                    _uvm_slot_direct_session_pids[int(device)] = os.getpid()
                    _uvm_slot_direct_mark_poisoned_locked(
                        int(device),
                        f"CUDA fence candidate cleanup failed: {exc!r}",
                    )
                    _register_uvm_slot_direct_close()
                    candidate["ok"] = False
                    candidate["reason"] = "cuda_fence_candidate_cleanup_failed"
                    candidate["error"] = repr(exc)
                    candidate["cuda_fence_fd_candidate_attempts"] = (
                        fence_candidate_attempts
                    )
                    candidate["cuda_fence_fd_candidate_rejections"] = (
                        fence_candidate_rejections
                    )
                    return candidate
            with suppress(OSError):
                os.close(fd)

        if last_result is not None:
            last_result["reason"] = "no_matching_existing_uvm_fd"
            last_result["fd_candidate_attempts"] = candidate_attempts
            last_result["fd_candidate_ioctl_calls"] = candidate_ioctl_calls
            last_result["cuda_fence_fd_candidate_attempts"] = (
                fence_candidate_attempts
            )
            last_result["cuda_fence_fd_candidate_rejections"] = (
                fence_candidate_rejections
            )
            if fence_enabled and last_fence_error:
                last_result["cuda_fence_last_error"] = last_fence_error
            return last_result
        if fence_enabled and fence_candidate_attempts:
            result = _moe_slot_result(
                ok=False,
                reason="no_cuda_fence_compatible_uvm_fd",
                mode=_UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC,
                requested_count=len(normalized),
                requested_bytes=total_bytes,
                error=last_fence_error,
            )
            result["fd_candidate_attempts"] = candidate_attempts
            result["fd_candidate_ioctl_calls"] = candidate_ioctl_calls
            result["cuda_fence_fd_candidate_attempts"] = (
                fence_candidate_attempts
            )
            result["cuda_fence_fd_candidate_rejections"] = (
                fence_candidate_rejections
            )
            return result
        return _moe_slot_result(
            ok=False,
            reason="no_initializable_existing_uvm_fd",
            mode=_UVM_MOE_SLOT_REMAP_MODE_VIDMEM_PREALLOC,
            requested_count=len(normalized),
            requested_bytes=total_bytes,
        )


def _slot_direct_policy_flags() -> tuple[int, str]:
    """Translate the two orthogonal restore gates into ioctl entry flags."""
    flags = 0
    for name, flag in (
        (
            _UVM_SLOT_DIRECT_SHARED_PUSH_ENV,
            UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_SHARED_PUSH,
        ),
        (
            _UVM_SLOT_DIRECT_64K_PTE_ENV,
            UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_64K_PTE,
        ),
    ):
        value = os.environ.get(name, "0").strip()
        if value not in ("0", "1"):
            return 0, f"{name} must be 0 or 1"
        if value == "1":
            flags |= flag
    return flags, ""


def _slot_direct_restore_mode() -> tuple[int, str]:
    """Select the restore ABI mode from two independently gated policies."""
    flags, error = _slot_direct_policy_flags()
    if error:
        return _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_PER_BLOCK_PUSH, error
    if flags == _UVM_MOE_SLOT_REMAP_POLICY_FLAGS:
        return _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE, ""
    if flags == UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_SHARED_PUSH:
        return _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH, ""
    if flags == UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_64K_PTE:
        return _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_PTE, ""
    if flags:
        return (
            _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_PER_BLOCK_PUSH,
            f"unsupported slot-direct policy flags 0x{flags:x}",
        )
    return _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_PER_BLOCK_PUSH, ""


def _validate_moe_slot_64k_layout(
    normalized: list[tuple[int, int, int, int, int, int, int]],
) -> str:
    """Validate sparse 64 KiB leaves without selecting a 4 KiB fallback.

    A submitted range may cover only part of a 2 MiB VA block. Gaps are
    intentional: they represent leaves which remain owned by the ordinary UVM
    mapping (for example, a hot expert next to a cold 512 KiB envelope). The
    kernel must preserve those leaves while replacing only the submitted
    ranges. Userspace still rejects overlap and any geometry which cannot be
    represented exactly by 64 KiB PTEs.
    """
    block_slices: dict[int, list[tuple[int, int]]] = {}
    page_bytes = 64 * 1024
    for index, record in enumerate(normalized):
        base, length, _layer, _expert, _slot, slot_offset, _flags = record
        if base % page_bytes or length % page_bytes or slot_offset % page_bytes:
            return f"range {index} is not 64 KiB aligned"

        cursor = base
        end = base + length
        while cursor < end:
            block_base = cursor - cursor % _UVM_MOE_SLOT_VA_BLOCK_BYTES
            slice_end = min(end, block_base + _UVM_MOE_SLOT_VA_BLOCK_BYTES)
            block_slices.setdefault(block_base, []).append((cursor, slice_end))
            cursor = slice_end

    for block_base, slices in block_slices.items():
        previous_end: int | None = None
        for start, end in sorted(slices):
            if previous_end is not None and start < previous_end:
                return (
                    "64 KiB mode ranges overlap; "
                    f"block 0x{block_base:x} contains overlapping slices"
                )
            previous_end = end

    return ""


def _driver_slot_restore_ranges_impl(
    ranges: Sequence[Sequence[int]],
    *,
    device: int = 0,
) -> dict[str, object]:
    """Copy CPU-resident managed ranges into slots and remap their original VA.

    Mode 10 remains the default. Explicit environment gates select mode 11
    (shared CE pushes), mode 12 (sparse-capable 64 KiB PTEs), or mode 21 when
    both policies are enabled. Geometry never changes the selected mode: an
    invalid 64 KiB request fails validation instead of falling back to 4 KiB,
    cudaMemPrefetchAsync, or a newly opened UVM fd.
    """
    restore_mode, mode_error = _slot_direct_restore_mode()
    normalized, validation_error = _normalize_moe_slot_ranges(ranges)
    total_bytes = sum(record[1] for record in normalized)
    with _uvm_slot_direct_lock:
        blocked = _uvm_slot_direct_block_reason_locked(int(device))
    if blocked is not None:
        reason, error = blocked
        return _moe_slot_result(
            ok=False,
            reason=reason,
            mode=restore_mode,
            requested_count=len(normalized),
            requested_bytes=total_bytes,
            error=error,
        )
    if (
        not validation_error
        and restore_mode
        in (
            _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_64K_PTE,
            _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE,
        )
    ):
        validation_error = _validate_moe_slot_64k_layout(normalized)
    if not validation_error:
        validation_error = mode_error
    if validation_error:
        return _moe_slot_result(
            ok=False,
            reason="validation_failed",
            mode=restore_mode,
            requested_count=len(normalized),
            requested_bytes=total_bytes,
            error=validation_error,
        )
    try:
        gpu_uuid = _phase_f_gpu_uuid(device)
    except Exception as exc:
        return _moe_slot_result(
            ok=False,
            reason="gpu_uuid_failed",
            mode=restore_mode,
            requested_count=len(normalized),
            requested_bytes=total_bytes,
            error=repr(exc),
        )
    uuid_bytes = bytes(gpu_uuid.uuid)

    with _uvm_slot_direct_lock:
        blocked = _uvm_slot_direct_block_reason_locked(int(device))
        if blocked is not None:
            reason, error = blocked
            return _moe_slot_result(
                ok=False,
                reason=reason,
                mode=restore_mode,
                requested_count=len(normalized),
                requested_bytes=total_bytes,
                error=error,
            )
        cached = _uvm_slot_direct_sessions.get(int(device))
        if cached is None:
            return _moe_slot_result(
                ok=False,
                reason="slot_preallocation_required",
                mode=restore_mode,
                requested_count=len(normalized),
                requested_bytes=total_bytes,
            )
        fd, cached_uuid = cached
        if cached_uuid != uuid_bytes:
            return _moe_slot_result(
                ok=False,
                reason="cached_gpu_uuid_mismatch",
                mode=restore_mode,
                requested_count=len(normalized),
                requested_bytes=total_bytes,
            )
        return _run_moe_slot_batches(
            fd,
            normalized,
            gpu_uuid=gpu_uuid,
            mode=restore_mode,
        )


def _driver_slot_transition_ranges_impl(
    ranges: Sequence[Sequence[int]],
    *,
    device: int = 0,
) -> dict[str, object]:
    """Rebind one complete fixed bank and copy/map it in one mode-20 ioctl.

    Every input entry must explicitly carry
    ``NO_EVICT|REUSE_BACKING_REQUIRED``. This wrapper adds the same shared-push
    and/or 64 KiB-PTE policy flags to every submitted entry from the restore
    environment gates. The kernel keeps the old logical owners while it tears
    down old PTEs and runs copy/map work. ``remap_count == count`` is accepted
    only on complete success; every failed transition must report zero so the
    local owner table remains unchanged while the caller poisons and aborts the
    now GPU-unmapped session.
    """
    transition_mode = _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION
    gate = os.environ.get(_UVM_SLOT_DIRECT_LAYER_TRANSITION_ENV, "0").strip()
    normalized, validation_error = _normalize_moe_slot_ranges(ranges)
    total_bytes = sum(record[1] for record in normalized)
    with _uvm_slot_direct_lock:
        blocked = _uvm_slot_direct_block_reason_locked(int(device))
    if blocked is not None:
        reason, error = blocked
        return _moe_slot_result(
            ok=False,
            reason=reason,
            mode=transition_mode,
            requested_count=len(normalized),
            requested_bytes=total_bytes,
            error=error,
        )
    required_flags = (
        UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT
        | UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
    )
    policy_flags, policy_error = _slot_direct_policy_flags()

    if gate not in ("0", "1"):
        validation_error = (
            f"{_UVM_SLOT_DIRECT_LAYER_TRANSITION_ENV} must be 0 or 1"
        )
    elif gate != "1":
        return _moe_slot_result(
            ok=False,
            reason="layer_transition_disabled",
            mode=transition_mode,
            requested_count=len(normalized),
            requested_bytes=total_bytes,
        )
    elif not validation_error and policy_error:
        validation_error = policy_error
    elif not validation_error and len(normalized) > _UVM_MOE_SLOT_REMAP_MAX_ENTRIES:
        validation_error = (
            "one layer transition exceeds the "
            f"{_UVM_MOE_SLOT_REMAP_MAX_ENTRIES}-entry ABI"
        )
    elif not validation_error and any(
        record[6] != required_flags for record in normalized
    ):
        validation_error = (
            "layer transition requires NO_EVICT|REUSE_BACKING_REQUIRED "
            "on every entry"
        )
    elif (
        not validation_error
        and policy_flags & UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_64K_PTE
    ):
        validation_error = _validate_moe_slot_64k_layout(normalized)

    if validation_error:
        return _moe_slot_result(
            ok=False,
            reason="validation_failed",
            mode=transition_mode,
            requested_count=len(normalized),
            requested_bytes=total_bytes,
            error=validation_error,
        )

    normalized = [
        (*record[:6], record[6] | policy_flags) for record in normalized
    ]

    try:
        gpu_uuid = _phase_f_gpu_uuid(device)
    except Exception as exc:
        return _moe_slot_result(
            ok=False,
            reason="gpu_uuid_failed",
            mode=transition_mode,
            requested_count=len(normalized),
            requested_bytes=total_bytes,
            error=repr(exc),
        )
    uuid_bytes = bytes(gpu_uuid.uuid)

    with _uvm_slot_direct_lock:
        blocked = _uvm_slot_direct_block_reason_locked(int(device))
        if blocked is not None:
            reason, error = blocked
            return _moe_slot_result(
                ok=False,
                reason=reason,
                mode=transition_mode,
                requested_count=len(normalized),
                requested_bytes=total_bytes,
                error=error,
            )
        cached = _uvm_slot_direct_sessions.get(int(device))
        if cached is None:
            return _moe_slot_result(
                ok=False,
                reason="slot_preallocation_required",
                mode=transition_mode,
                requested_count=len(normalized),
                requested_bytes=total_bytes,
            )
        fd, cached_uuid = cached
        if cached_uuid != uuid_bytes:
            return _moe_slot_result(
                ok=False,
                reason="cached_gpu_uuid_mismatch",
                mode=transition_mode,
                requested_count=len(normalized),
                requested_bytes=total_bytes,
            )

        owners = _uvm_slot_direct_owners.get(int(device), {})
        missing = [
            (record[2], record[4])
            for record in normalized
            if (record[2], record[4]) not in owners
        ]
        if missing:
            return _moe_slot_result(
                ok=False,
                reason="slot_owner_untracked",
                mode=transition_mode,
                requested_count=len(normalized),
                requested_bytes=total_bytes,
                error=f"missing={missing!r}",
            )

        result = _run_moe_slot_batches(
            fd,
            normalized,
            gpu_uuid=gpu_uuid,
            mode=transition_mode,
        )
        cutover = result.get("owner_cutover")
        if cutover is True:
            for record in normalized:
                _base, _length, layer_id, expert_id, slot_id, _offset, _flags = (
                    record
                )
                owners[(layer_id, slot_id)] = expert_id
        elif cutover is None:
            result["owner_state_indeterminate"] = True
        return result


def _build_registered_plan_execute_contract(
    entries: Sequence[tuple[int, int, int, int, int, int, int]],
    mode: int,
) -> dict[str, object]:
    """Materialize immutable execute checks once, at plan registration.

    The registered-plan ioctl only accepts ``plan_id/layer/generation`` at
    execute time. Rebuilding byte totals, touched 2 MiB blocks, and the lane
    owner view on every invocation is therefore redundant. Keep the complete
    contract next to the immutable descriptor and compare only the returned
    scalar counters on the fast path.
    """
    count = len(entries)
    requested_bytes = sum(int(record[1]) for record in entries)
    uses_64k_grouping = (
        int(mode) == _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE
        or (
            int(mode) == _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION
            and any(
                int(record[6])
                & UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_64K_PTE
                for record in entries
            )
        )
    )
    if uses_64k_grouping:
        touched_blocks = {
            block
            for record in entries
            for block in range(
                int(record[0]) // _UVM_MOE_SLOT_VA_BLOCK_BYTES,
                (int(record[0]) + int(record[1]) - 1)
                // _UVM_MOE_SLOT_VA_BLOCK_BYTES
                + 1,
            )
        }
        tlb_invalidate_count = len(touched_blocks)
    else:
        tlb_invalidate_count = count

    return {
        "count": count,
        "requested_bytes": requested_bytes,
        "tlb_invalidate_count": tlb_invalidate_count,
        "owner_cutover": (
            int(mode) == _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION
        ),
        "copy_push_count_min": 1 if count else 0,
        "copy_push_count_max": requested_bytes // _UVM_MOE_SLOT_PAGE_BYTES,
        "owners": {
            (int(record[2]), int(record[4])): int(record[3])
            for record in entries
        },
    }


def _driver_slot_register_plan_impl(
    plan_id: int,
    layer_id: int,
    ranges: Sequence[Sequence[int]],
    *,
    transition: bool,
    device: int = 0,
) -> dict[str, object]:
    """Register one immutable target-layer descriptor array in the driver."""
    if (
        not isinstance(plan_id, int)
        or isinstance(plan_id, bool)
        or plan_id <= 0
        or plan_id > _UVM_MOE_SLOT_UINT64_MAX
        or not isinstance(layer_id, int)
        or isinstance(layer_id, bool)
        or layer_id < 0
        or layer_id > _UVM_MOE_SLOT_UINT32_MAX
    ):
        return {
            "ok": False,
            "reason": "validation_failed",
            "error": "plan_id/layer_id is outside the ABI",
            "plan_id": plan_id,
            "layer_id": layer_id,
        }

    normalized, validation_error = _normalize_moe_slot_ranges(ranges)
    total_bytes = sum(record[1] for record in normalized)
    if transition:
        mode = _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION
        initial_mode_error = ""
    else:
        mode, initial_mode_error = _slot_direct_restore_mode()
    with _uvm_slot_direct_lock:
        blocked = _uvm_slot_direct_block_reason_locked(int(device))
    if blocked is not None:
        reason, error = blocked
        return {
            "ok": False,
            "reason": reason,
            "error": error,
            "plan_id": int(plan_id),
            "layer_id": int(layer_id),
            "mode": mode,
            "requested_count": len(normalized),
            "requested_bytes": total_bytes,
        }
    if len(normalized) > _UVM_MOE_SLOT_REMAP_MAX_ENTRIES:
        validation_error = (
            "one registered layer exceeds the "
            f"{_UVM_MOE_SLOT_REMAP_MAX_ENTRIES}-entry ABI"
        )

    if transition:
        gate = os.environ.get(
            _UVM_SLOT_DIRECT_LAYER_TRANSITION_ENV, "0"
        ).strip()
        required_flags = (
            UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT
            | UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
        )
        policy_flags, policy_error = _slot_direct_policy_flags()
        if gate != "1":
            validation_error = "layer transition plan is disabled"
        elif not validation_error and policy_error:
            validation_error = policy_error
        elif not validation_error and any(
            record[6] != required_flags for record in normalized
        ):
            validation_error = (
                "transition plan requires NO_EVICT|REUSE_BACKING_REQUIRED"
            )
        elif (
            not validation_error
            and policy_flags
            & UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_64K_PTE
        ):
            validation_error = _validate_moe_slot_64k_layout(normalized)
        if not validation_error:
            normalized = [
                (*record[:6], record[6] | policy_flags)
                for record in normalized
            ]
    else:
        if mode not in (
            _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH,
            _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE,
        ):
            validation_error = (
                initial_mode_error
                or "initial plan requires shared-push mode 11 or "
                "shared-push plus 64 KiB PTE mode 21"
            )
        elif (
            not validation_error
            and mode
            == _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE
        ):
            validation_error = _validate_moe_slot_64k_layout(normalized)

    if validation_error:
        return {
            "ok": False,
            "reason": "validation_failed",
            "error": validation_error,
            "plan_id": int(plan_id),
            "layer_id": int(layer_id),
            "mode": mode,
            "requested_count": len(normalized),
            "requested_bytes": total_bytes,
        }

    try:
        gpu_uuid = _phase_f_gpu_uuid(device)
    except Exception as exc:
        return {
            "ok": False,
            "reason": "gpu_uuid_failed",
            "error": repr(exc),
            "plan_id": int(plan_id),
            "layer_id": int(layer_id),
            "mode": mode,
            "requested_count": len(normalized),
            "requested_bytes": total_bytes,
        }

    uuid_bytes = bytes(gpu_uuid.uuid)
    with _uvm_slot_direct_lock:
        blocked = _uvm_slot_direct_block_reason_locked(int(device))
        if blocked is not None:
            reason, error = blocked
            return {
                "ok": False,
                "reason": reason,
                "error": error,
                "plan_id": int(plan_id),
                "layer_id": int(layer_id),
                "mode": mode,
                "requested_count": len(normalized),
                "requested_bytes": total_bytes,
            }
        cached = _uvm_slot_direct_sessions.get(int(device))
        if cached is None:
            return {
                "ok": False,
                "reason": "slot_preallocation_required",
                "plan_id": int(plan_id),
                "layer_id": int(layer_id),
                "mode": mode,
            }
        fd, cached_uuid = cached
        if cached_uuid != uuid_bytes:
            return {
                "ok": False,
                "reason": "cached_gpu_uuid_mismatch",
                "plan_id": int(plan_id),
                "layer_id": int(layer_id),
                "mode": mode,
            }

        if uvm_slot_cuda_fence_enabled():
            try:
                _uvm_slot_cuda_fence_create_locked(device=int(device), fd=fd)
                _uvm_slot_cuda_fence_warm_locked(device=int(device), fd=fd)
            except Exception as exc:
                return {
                    "ok": False,
                    "reason": "cuda_fence_pool_create_failed",
                    "error": repr(exc),
                    "plan_id": int(plan_id),
                    "layer_id": int(layer_id),
                    "mode": mode,
                }

        params = _MoeSlotPlanRegisterParams()
        params.plan_id = int(plan_id)
        params.layer_id = int(layer_id)
        params.mode = mode
        params.count = len(normalized)
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
        ioctl_started_ns = time.perf_counter_ns()
        try:
            ret = _libc_ioctl(fd, _UVM_MOE_SLOT_PLAN_REGISTER_IOCTL, params)
        except Exception as exc:
            ioctl_wall_ns = time.perf_counter_ns() - ioctl_started_ns
            _uvm_slot_direct_mark_poisoned_locked(
                int(device), f"plan register ioctl raised: {exc!r}"
            )
            abort_result = _driver_slot_abort_session_impl(device=int(device))
            return {
                "ok": False,
                "reason": "plan_register_ioctl_exception",
                "error": repr(exc),
                "plan_id": int(plan_id),
                "layer_id": int(layer_id),
                "mode": mode,
                "ioctl_wall_ns": ioctl_wall_ns,
                "abort": abort_result,
                "session_aborted": bool(abort_result.get("ok")),
            }
        ioctl_wall_ns = time.perf_counter_ns() - ioctl_started_ns
        errno_value = ctypes.get_errno() if ret != 0 else 0
        generation = int(params.generation)
        ok = (
            ret == 0
            and int(params.rmStatus) == 0
            and generation > 0
            and int(params.requested_bytes) == total_bytes
        )
        result: dict[str, object] = {
            "ok": ok,
            "reason": "ok" if ok else "plan_register_ioctl_failed",
            "plan_id": int(plan_id),
            "layer_id": int(layer_id),
            "mode": mode,
            "generation": generation,
            "requested_count": len(normalized),
            "requested_bytes": int(params.requested_bytes),
            "ioctl_ret": int(ret),
            "errno": errno_value,
            "rm_status": int(params.rmStatus),
            "ioctl_wall_ns": ioctl_wall_ns,
        }
        registration_may_have_side_effect = (
            (ret == 0 and int(params.rmStatus) == 0)
            or generation > 0
            or int(params.requested_bytes) > 0
        )
        if not ok and registration_may_have_side_effect:
            result["reason"] = "plan_register_contract_mismatch"
            _uvm_slot_direct_mark_poisoned_locked(
                int(device), "plan register returned an untrusted success contract"
            )
            abort_result = _driver_slot_abort_session_impl(device=int(device))
            result["abort"] = abort_result
            result["session_aborted"] = bool(abort_result.get("ok"))
            return result
        if ok:
            generations = _uvm_slot_direct_plan_generations
            current_generation = generations.get(int(device))
            if current_generation is not None and current_generation != generation:
                result["ok"] = False
                result["reason"] = "plan_generation_mismatch"
                _uvm_slot_direct_mark_poisoned_locked(
                    int(device), "kernel plan generation disagrees with userspace"
                )
                abort_result = _driver_slot_abort_session_impl(device=int(device))
                result["abort"] = abort_result
                result["session_aborted"] = bool(abort_result.get("ok"))
                return result
            try:
                execute_contract = _build_registered_plan_execute_contract(
                    normalized, mode
                )
                generations[int(device)] = generation
                _uvm_slot_direct_plans.setdefault(int(device), {})[
                    int(plan_id)
                ] = {
                    "layer_id": int(layer_id),
                    "mode": mode,
                    "entries": tuple(normalized),
                    "execute_contract": execute_contract,
                }
            except Exception as exc:
                result["ok"] = False
                result["reason"] = "plan_register_postcondition_failed"
                result["error"] = repr(exc)
                _uvm_slot_direct_mark_poisoned_locked(
                    int(device), f"plan register cache update failed: {exc!r}"
                )
                abort_result = _driver_slot_abort_session_impl(device=int(device))
                result["abort"] = abort_result
                result["session_aborted"] = bool(abort_result.get("ok"))
        return result


def _driver_slot_execute_plan_impl(
    plan_id: int,
    layer_id: int,
    generation: int,
    *,
    device: int = 0,
) -> dict[str, object]:
    """Execute one compact plan-id/layer/generation transition ioctl.

    The caller must first stop ordinary CUDA work which can read the old slot
    VAs. Kernel trackers order the transition's UVM pushes, not user kernels.
    """
    with _uvm_slot_direct_lock:
        blocked = _uvm_slot_direct_block_reason_locked(int(device))
        if blocked is not None:
            reason, error = blocked
            return {
                "ok": False,
                "reason": reason,
                "error": error,
                "plan_id": int(plan_id),
                "layer_id": int(layer_id),
                "generation": int(generation),
            }
        cached = _uvm_slot_direct_sessions.get(int(device))
        plan = _uvm_slot_direct_plans.get(int(device), {}).get(int(plan_id))
        tracked_generation = _uvm_slot_direct_plan_generations.get(int(device))
        if cached is None or plan is None:
            return {
                "ok": False,
                "reason": "slot_plan_not_registered",
                "plan_id": int(plan_id),
                "layer_id": int(layer_id),
                "generation": int(generation),
            }
        if tracked_generation != int(generation):
            return {
                "ok": False,
                "reason": "stale_userspace_generation",
                "plan_id": int(plan_id),
                "layer_id": int(layer_id),
                "generation": int(generation),
                "tracked_generation": tracked_generation,
            }

        expected_layer = int(plan["layer_id"])
        expected_mode = int(plan["mode"])
        expected_entries = plan["entries"]
        try:
            bank_level_commit = _uvm_slot_bank_level_commit_enabled()
        except RuntimeError as exc:
            return {
                "ok": False,
                "reason": "bank_level_commit_config_invalid",
                "error": str(exc),
                "plan_id": int(plan_id),
                "layer_id": int(layer_id),
                "generation": int(generation),
            }

        if bank_level_commit:
            contract = plan.get("execute_contract")
            if not isinstance(contract, dict):
                return {
                    "ok": False,
                    "reason": "slot_plan_execute_contract_missing",
                    "plan_id": int(plan_id),
                    "layer_id": int(layer_id),
                    "generation": int(generation),
                }
            expected_count = int(contract["count"])
            expected_bytes = int(contract["requested_bytes"])
            expected_tlb_invalidate_count = int(
                contract["tlb_invalidate_count"]
            )
            expected_owner_cutover = bool(contract["owner_cutover"])
            expected_copy_push_count_min = int(contract["copy_push_count_min"])
            expected_copy_push_count_max = int(contract["copy_push_count_max"])
        else:
            expected_count = len(expected_entries)
            expected_bytes = sum(int(record[1]) for record in expected_entries)
            expected_tlb_blocks = {
                block
                for record in expected_entries
                for block in range(
                    int(record[0]) // _UVM_MOE_SLOT_VA_BLOCK_BYTES,
                    (int(record[0]) + int(record[1]) - 1)
                    // _UVM_MOE_SLOT_VA_BLOCK_BYTES
                    + 1,
                )
            }
            expected_64k_grouping = (
                expected_mode
                == _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE
                or (
                    expected_mode
                    == _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION
                    and any(
                        int(record[6])
                        & UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_64K_PTE
                        for record in expected_entries
                    )
                )
            )
            # Sliced-64K batches coalesce four envelopes in one VA block and
            # publish/invalidate that block once. The registered 4K fallback
            # maps each independently-backed envelope separately.
            expected_tlb_invalidate_count = (
                len(expected_tlb_blocks)
                if expected_64k_grouping
                else expected_count
            )
            expected_owner_cutover = (
                expected_mode
                == _UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION
            )
            expected_copy_push_count_min = 1 if expected_count else 0
            expected_copy_push_count_max = (
                expected_bytes // _UVM_MOE_SLOT_PAGE_BYTES
            )
        if int(layer_id) != expected_layer:
            return {
                "ok": False,
                "reason": "slot_plan_layer_mismatch",
                "plan_id": int(plan_id),
                "layer_id": int(layer_id),
                "expected_layer_id": expected_layer,
                "generation": int(generation),
            }

        fd, _uuid_bytes = cached
        fence_enabled = uvm_slot_cuda_fence_enabled()
        fence: _UvmSlotCudaFence | None = None
        fence_stream = 0
        fence_ticket = 0
        compute_fence_wall_ns = 0
        if fence_enabled:
            fence = _uvm_slot_direct_cuda_fences.get(int(device))
            if fence is None or fence.fd != int(fd):
                return {
                    "ok": False,
                    "reason": "cuda_fence_pool_missing",
                    "plan_id": int(plan_id),
                    "layer_id": int(layer_id),
                    "generation": int(generation),
                }
            if fence.transport != "raw_kernel":
                return {
                    "ok": False,
                    "reason": "cuda_fence_raw_transport_required",
                    "plan_id": int(plan_id),
                    "layer_id": int(layer_id),
                    "generation": int(generation),
                    "poisoned": False,
                }
            try:
                capturing = bool(torch.cuda.is_current_stream_capturing())
            except Exception as exc:
                return {
                    "ok": False,
                    "reason": "cuda_graph_capture_query_failed",
                    "error": repr(exc),
                    "plan_id": int(plan_id),
                    "layer_id": int(layer_id),
                    "generation": int(generation),
                    "poisoned": False,
                }
            if capturing:
                return {
                    "ok": False,
                    "reason": "cuda_graph_capture_unsupported",
                    "plan_id": int(plan_id),
                    "layer_id": int(layer_id),
                    "generation": int(generation),
                    "poisoned": False,
                }
            try:
                compute_fence_started_ns = time.perf_counter_ns()
                fence_stream = _uvm_slot_cuda_fence_current_stream(int(device))
                fence_ticket = fence.reserve_ticket()
                _uvm_slot_cuda_fence_write_compute(
                    stream=fence_stream,
                    address=fence.compute_device_address,
                    ticket=fence_ticket,
                )
                compute_fence_wall_ns = (
                    time.perf_counter_ns() - compute_fence_started_ns
                )
            except Exception as exc:
                if "compute_fence_started_ns" in locals():
                    compute_fence_wall_ns = (
                        time.perf_counter_ns() - compute_fence_started_ns
                    )
                _uvm_slot_direct_mark_poisoned_locked(
                    int(device), f"compute fence enqueue failed: {exc!r}"
                )
                if isinstance(exc, _UvmSlotCudaFenceSignalTimeout):
                    # The old-compute boundary is unproven and CUDA has no
                    # bounded stream-synchronize API. Preserve all mappings
                    # and defer reclamation to process teardown.
                    abort_result = None
                    boundary_error = repr(exc)
                    _uvm_slot_direct_cuda_fence_cleanup_deferred.add(
                        int(device)
                    )
                else:
                    abort_result, boundary_error = (
                        _uvm_slot_abort_after_fence_failure_locked(
                            device=int(device), fence_enabled=True
                        )
                    )
                return {
                    "ok": False,
                    "reason": "cuda_compute_fence_enqueue_failed",
                    "error": repr(exc),
                    "plan_id": int(plan_id),
                    "layer_id": int(layer_id),
                    "generation": int(generation),
                    "cuda_compute_fence_wall_ns": compute_fence_wall_ns,
                    "poisoned": True,
                    "abort": abort_result,
                    "session_aborted": bool(
                        abort_result is not None and abort_result.get("ok")
                    ),
                    "failure_boundary_error": boundary_error,
                    "deferred_to_process_teardown": boundary_error is not None,
                }

        params: _MoeSlotPlanExecuteParams | _MoeSlotPlanExecuteFencedParams
        if fence_enabled:
            assert fence is not None
            fenced_params = _MoeSlotPlanExecuteFencedParams()
            fenced_params.fence_flags = _UVM_MOE_SLOT_PLAN_FENCE_FLAGS_ALL
            fenced_params.compute_semaphore_address = fence.compute_address
            fenced_params.completion_semaphore_address = fence.completion_address
            fenced_params.compute_payload = fence_ticket
            fenced_params.completion_payload = fence_ticket
            params = fenced_params
            execute_ioctl = _UVM_MOE_SLOT_PLAN_EXECUTE_FENCED_IOCTL
        else:
            params = _MoeSlotPlanExecuteParams()
            execute_ioctl = _UVM_MOE_SLOT_PLAN_EXECUTE_IOCTL
        params.plan_id = int(plan_id)
        params.layer_id = int(layer_id)
        params.generation = int(generation)
        ctypes.set_errno(0)
        ioctl_started_ns = time.perf_counter_ns()
        try:
            ret = _libc_ioctl(fd, execute_ioctl, params)
        except Exception as exc:
            ioctl_wall_ns = time.perf_counter_ns() - ioctl_started_ns
            _uvm_slot_direct_mark_poisoned_locked(
                int(device), f"plan execute ioctl raised: {exc!r}"
            )
            abort_result, boundary_error = (
                _uvm_slot_abort_after_fence_failure_locked(
                    device=int(device), fence_enabled=fence_enabled
                )
            )
            return {
                "ok": False,
                "reason": "plan_execute_ioctl_exception",
                "error": repr(exc),
                "plan_id": int(plan_id),
                "layer_id": int(layer_id),
                "generation": int(generation),
                "poisoned": True,
                "ioctl_wall_ns": ioctl_wall_ns,
                "abort": abort_result,
                "session_aborted": bool(
                    abort_result is not None and abort_result.get("ok")
                ),
                "failure_boundary_error": boundary_error,
                "deferred_to_process_teardown": boundary_error is not None,
            }
        ioctl_wall_ns = time.perf_counter_ns() - ioctl_started_ns
        errno_value = ctypes.get_errno() if ret != 0 else 0
        count = int(params.count)
        mode = int(params.mode)
        new_generation = int(params.new_generation)
        owner_cutover_raw = int(params.owner_cutover)
        kernel_poisoned_raw = int(params.poisoned)
        fence_accepted_raw = (
            int(params.fence_accepted) if fence_enabled else 0
        )
        completion_signaled_raw = (
            int(params.completion_signaled) if fence_enabled else 0
        )
        boolean_fields_ok = (
            owner_cutover_raw in (0, 1)
            and kernel_poisoned_raw in (0, 1)
            and fence_accepted_raw in (0, 1)
            and completion_signaled_raw in (0, 1)
        )
        owner_cutover = owner_cutover_raw == 1
        kernel_poisoned = kernel_poisoned_raw == 1
        ioctl_completed = ret == 0 and int(params.rmStatus) == 0
        success_contract_ok = (
            ioctl_completed
            and boolean_fields_ok
            and not kernel_poisoned
            and mode == expected_mode
            and count == expected_count
            and int(params.requested_bytes) == expected_bytes
            and int(params.success_count) == expected_count
            and int(params.failure_count) == 0
            and int(params.conflict_count) == 0
            and int(params.accepted_bytes) == expected_bytes
            and int(params.remap_count) == expected_count
            and int(params.tlb_invalidate_count)
            == expected_tlb_invalidate_count
            and expected_copy_push_count_min
            <= int(params.copy_push_count)
            <= expected_copy_push_count_max
            and owner_cutover == expected_owner_cutover
            and new_generation == int(generation) + 1
            and (not fence_enabled or fence_accepted_raw == 1)
            and (not fence_enabled or completion_signaled_raw == 1)
        )
        # Copy submission and old-PTE teardown may already have happened on a
        # failed transaction. They are deliberately not owner publication. A
        # trustworthy failure keeps both the old owner and active generation,
        # leaves the target VAs GPU-unmapped, and makes the session abort-only.
        failure_contract_ok = (
            not ioctl_completed
            and boolean_fields_ok
            and kernel_poisoned
            and not owner_cutover
            and int(params.remap_count) == 0
            and new_generation == int(generation)
            and (not fence_enabled or completion_signaled_raw == 0)
        )
        publication_observed = (
            owner_cutover
            or int(params.accepted_bytes) > 0
            or int(params.success_count) > 0
            or int(params.remap_count) > 0
            or new_generation not in (0, int(generation))
        )
        precommit_work_observed = (
            int(params.tlb_invalidate_count) > 0
            or int(params.copy_push_count) > 0
        )
        side_effect_observed = (
            publication_observed or precommit_work_observed
        )
        # The kernel enters TRANSITIONING only after immutable identity/device,
        # slot, target, large-leaf, and cleanup-resource preflight. A rejection
        # before that point is trustworthy only when it reports no publication
        # or precommit work and keeps the active generation/session unpoisoned.
        preflight_rejection_ok = (
            not ioctl_completed
            and boolean_fields_ok
            and not kernel_poisoned
            and not owner_cutover
            and mode == expected_mode
            and count == expected_count
            and int(params.requested_bytes) == expected_bytes
            and int(params.success_count) == 0
            and int(params.accepted_bytes) == 0
            and int(params.remap_count) == 0
            and not side_effect_observed
            and new_generation == int(generation)
            and (not fence_enabled or completion_signaled_raw == 0)
        )
        contract_violation = (
            (ioctl_completed and not success_contract_ok)
            or (
                not ioctl_completed
                and not failure_contract_ok
                and not preflight_rejection_ok
            )
        )
        ok = success_contract_ok
        # Serving stays conservative: even a trustworthy preflight rejection is
        # locally poisoned and enters the generation-free abort path, although
        # the kernel has not changed a PTE or poisoned its registered session.
        poisoned = not ok
        new_generation_trusted = (
            ok or failure_contract_ok or preflight_rejection_ok
        )
        result: dict[str, object] = {
            "ok": ok,
            "reason": (
                "ok"
                if ok
                else "plan_execute_preflight_rejected"
                if preflight_rejection_ok
                else "plan_execute_contract_mismatch"
                if contract_violation
                else "plan_execute_ioctl_failed"
            ),
            "plan_id": int(plan_id),
            "layer_id": int(layer_id),
            "generation": int(generation),
            "new_generation": new_generation,
            "mode": mode,
            "requested_count": count,
            "success_count": int(params.success_count),
            "failure_count": int(params.failure_count),
            "conflict_count": int(params.conflict_count),
            "requested_bytes": int(params.requested_bytes),
            "accepted_bytes": int(params.accepted_bytes),
            "remap_count": int(params.remap_count),
            "tlb_invalidate_count": int(params.tlb_invalidate_count),
            "copy_push_count": int(params.copy_push_count),
            "owner_cutover": owner_cutover,
            "owner_cutover_count": int(params.remap_count) if owner_cutover else 0,
            "owner_cutover_trusted": (
                ok or failure_contract_ok or preflight_rejection_ok
            ),
            "owner_state_indeterminate": poisoned and not (
                failure_contract_ok or preflight_rejection_ok
            ),
            "poisoned": poisoned,
            "kernel_poisoned": kernel_poisoned,
            "cuda_fence_used": fence_enabled,
            "cuda_fence_ticket": fence_ticket,
            "cuda_compute_fence_wall_ns": compute_fence_wall_ns,
            "cuda_fence_accepted": fence_accepted_raw == 1,
            "cuda_completion_signaled": completion_signaled_raw == 1,
            "contract_violation": contract_violation,
            "success_contract_ok": success_contract_ok,
            "failure_contract_ok": failure_contract_ok,
            "preflight_rejection_ok": preflight_rejection_ok,
            "publication_observed": publication_observed,
            "precommit_work_observed": precommit_work_observed,
            "side_effect_observed": side_effect_observed,
            "expected_mode": expected_mode,
            "expected_count": expected_count,
            "expected_requested_bytes": expected_bytes,
            "expected_tlb_invalidate_count": expected_tlb_invalidate_count,
            "expected_copy_push_count_min": expected_copy_push_count_min,
            "expected_copy_push_count_max": expected_copy_push_count_max,
            "new_generation_trusted": new_generation_trusted,
            "generation_cleanup_requires_fd_close": not new_generation_trusted,
            "ioctl_ret": int(ret),
            "errno": errno_value,
            "rm_status": int(params.rmStatus),
            "ioctl_calls": 1,
            "ioctl_wall_ns": ioctl_wall_ns,
            "entry_statuses": (
                []
                if bank_level_commit
                else ([0] * count if ok else [None] * count)
            ),
            "entry_statuses_elided": bank_level_commit,
            "batches": [],
            "plan_execute": True,
        }
        if not new_generation_trusted:
            # Never leave a stale generation available to identity-based
            # cleanup after an execute response whose generation contract is
            # malformed. The recovery abort is generation-free and may itself
            # fail, so invalidate this cache before entering that path.
            _uvm_slot_direct_plan_generations.pop(int(device), None)
        if poisoned:
            _uvm_slot_direct_mark_poisoned_locked(
                int(device), str(result["reason"])
            )
            abort_result, boundary_error = (
                _uvm_slot_abort_after_fence_failure_locked(
                    device=int(device), fence_enabled=fence_enabled
                )
            )
            result["abort"] = abort_result
            result["session_aborted"] = bool(
                abort_result is not None and abort_result.get("ok")
            )
            result["failure_boundary_error"] = boundary_error
            result["deferred_to_process_teardown"] = boundary_error is not None
            return result
        if ok and fence_enabled:
            assert fence is not None
            try:
                _uvm_slot_cuda_fence_wait_completion(
                    stream=fence_stream,
                    address=fence.completion_device_address,
                    ticket=fence_ticket,
                )
            except Exception as exc:
                result["ok"] = False
                result["reason"] = "cuda_completion_wait_enqueue_failed"
                result["error"] = repr(exc)
                result["poisoned"] = True
                _uvm_slot_direct_mark_poisoned_locked(
                    int(device), f"completion wait enqueue failed: {exc!r}"
                )
                abort_result, boundary_error = (
                    _uvm_slot_abort_after_fence_failure_locked(
                        device=int(device), fence_enabled=True
                    )
                )
                result["abort"] = abort_result
                result["session_aborted"] = bool(
                    abort_result is not None and abort_result.get("ok")
                )
                result["failure_boundary_error"] = boundary_error
                result["deferred_to_process_teardown"] = (
                    boundary_error is not None
                )
                return result
            result["cuda_stream_wait_enqueued"] = True
        else:
            result["cuda_stream_wait_enqueued"] = False

        if ok and new_generation > 0:
            _uvm_slot_direct_plan_generations[int(device)] = new_generation
        if ok:
            if bank_level_commit:
                # The kernel has atomically committed this descriptor and its
                # generation. Selecting the immutable plan is sufficient; no
                # per-lane owner dictionary needs to be rewritten here.
                _uvm_slot_direct_active_plan_ids[int(device)] = int(plan_id)
            else:
                owners = _uvm_slot_direct_owners.setdefault(int(device), {})
                for record in plan["entries"]:  # type: ignore[union-attr]
                    (
                        _base,
                        _length,
                        driver_layer,
                        expert,
                        slot,
                        _offset,
                        _flags,
                    ) = record
                    owners[(driver_layer, slot)] = expert
        return result


def _driver_slot_destroy_plan_impl(
    plan_id: int,
    generation: int,
    *,
    device: int = 0,
) -> dict[str, object]:
    with _uvm_slot_direct_lock:
        blocked = _uvm_slot_direct_block_reason_locked(int(device))
        if blocked is not None:
            reason, error = blocked
            return {
                "ok": False,
                "reason": reason,
                "error": error,
                "plan_id": int(plan_id),
                "generation": int(generation),
                "destroyed": False,
            }
        cached = _uvm_slot_direct_sessions.get(int(device))
        if cached is None:
            return {"ok": True, "reason": "session_closed", "destroyed": False}
        fd, _uuid_bytes = cached
        params = _MoeSlotPlanDestroyParams()
        params.plan_id = int(plan_id)
        params.generation = int(generation)
        ctypes.set_errno(0)
        ret = _libc_ioctl(fd, _UVM_MOE_SLOT_PLAN_DESTROY_IOCTL, params)
        errno_value = ctypes.get_errno() if ret != 0 else 0
        destroyed = int(params.destroyed) == 1
        ok = ret == 0 and int(params.rmStatus) == 0 and destroyed
        if ok:
            plans = _uvm_slot_direct_plans.get(int(device), {})
            plans.pop(int(plan_id), None)
            if _uvm_slot_direct_active_plan_ids.get(int(device)) == int(plan_id):
                _uvm_slot_direct_active_plan_ids.pop(int(device), None)
            current_generation = int(params.current_generation)
            if plans:
                _uvm_slot_direct_plan_generations[int(device)] = current_generation
            else:
                _uvm_slot_direct_plans.pop(int(device), None)
                _uvm_slot_direct_plan_generations.pop(int(device), None)
        return {
            "ok": ok,
            "reason": "ok" if ok else "plan_destroy_ioctl_failed",
            "plan_id": int(plan_id),
            "generation": int(generation),
            "current_generation": int(params.current_generation),
            "destroyed": destroyed,
            "ioctl_ret": int(ret),
            "errno": errno_value,
            "rm_status": int(params.rmStatus),
        }


def _driver_slot_abort_session_impl(
    *,
    device: int = 0,
) -> dict[str, object]:
    """Abort all slot resources scoped to the current UVM VA-space.

    Unlike plan destroy, this recovery ioctl intentionally takes no userspace
    generation. It is the only safe cleanup when userspace cannot prove which
    generation the kernel accepted. The duplicated fd is retained so callers
    can close it only after the driver reports that no scoped resources remain.
    """
    device = int(device)
    with _uvm_slot_direct_lock:
        cached = _uvm_slot_direct_sessions.get(device)
        if cached is None:
            inconsistent = bool(
                _uvm_slot_direct_owners.get(device)
                or _uvm_slot_direct_plans.get(device)
                or device in _uvm_slot_direct_plan_generations
                or device in _uvm_slot_direct_active_plan_ids
            )
            if inconsistent:
                _uvm_slot_direct_mark_poisoned_locked(
                    device, "slot resource cache exists without a session fd"
                )
                return {
                    "ok": False,
                    "reason": "slot_session_fd_missing",
                    "session_cleared": False,
                    "session_poisoned": True,
                    "ioctl_calls": 0,
                }
            _uvm_slot_direct_session_pids.pop(device, None)
            _uvm_slot_direct_poisoned.pop(device, None)
            return {
                "ok": True,
                "reason": "already_aborted",
                "generation_before": 0,
                "released_bytes": 0,
                "remaining_bytes": 0,
                "destroyed_plan_count": 0,
                "released_slot_count": 0,
                "remaining_slot_count": 0,
                "had_registered_session": False,
                "session_cleared": True,
                "poisoned": False,
                "kernel_poisoned": False,
                "session_poisoned": False,
                "contract_ok": True,
                "ioctl_ret": 0,
                "errno": 0,
                "rm_status": 0,
                "ioctl_calls": 0,
                "ioctl_wall_ns": 0,
            }

        inherited_pid = _uvm_slot_direct_inherited_pid_locked(device)
        if inherited_pid is not None:
            return {
                "ok": False,
                "reason": "fork_inherited_slot_session",
                "error": (
                    f"session belongs to pid {inherited_pid}, "
                    f"current pid is {os.getpid()}"
                ),
                "session_cleared": False,
                "session_poisoned": True,
                "ioctl_calls": 0,
            }

        fd, _uuid_bytes = cached
        params = _MoeSlotPlanAbortSessionParams()
        # flags and reserved are ABI inputs and must remain zero.
        params.flags = 0
        params.reserved = 0
        ctypes.set_errno(0)
        ioctl_started_ns = time.perf_counter_ns()
        try:
            ret = _libc_ioctl(
                fd,
                _UVM_MOE_SLOT_PLAN_ABORT_SESSION_IOCTL,
                params,
            )
        except Exception as exc:
            ioctl_wall_ns = time.perf_counter_ns() - ioctl_started_ns
            _uvm_slot_direct_mark_poisoned_locked(
                device, f"abort ioctl raised: {exc!r}"
            )
            return {
                "ok": False,
                "reason": "slot_session_abort_ioctl_exception",
                "error": repr(exc),
                "session_cleared": False,
                "session_poisoned": True,
                "ioctl_calls": 1,
                "ioctl_wall_ns": ioctl_wall_ns,
            }
        ioctl_wall_ns = time.perf_counter_ns() - ioctl_started_ns
        errno_value = ctypes.get_errno() if ret != 0 else 0

        generation_before = int(params.generation_before)
        released_bytes = int(params.released_bytes)
        remaining_bytes = int(params.remaining_bytes)
        destroyed_plan_count = int(params.destroyed_plan_count)
        released_slot_count = int(params.released_slot_count)
        remaining_slot_count = int(params.remaining_slot_count)
        had_registered_session_raw = int(params.had_registered_session)
        session_cleared_raw = int(params.session_cleared)
        kernel_poisoned_raw = int(params.poisoned)
        rm_status = int(params.rmStatus)

        boolean_fields_ok = all(
            value in (0, 1)
            for value in (
                had_registered_session_raw,
                session_cleared_raw,
                kernel_poisoned_raw,
            )
        )
        had_registered_session = had_registered_session_raw == 1
        session_cleared = session_cleared_raw == 1
        kernel_poisoned = kernel_poisoned_raw == 1
        released_pair_ok = (released_slot_count == 0) == (released_bytes == 0)
        # An invariant-only residue has no trustworthy byte total, and the
        # kernel deliberately reports count=1/bytes=0 to keep teardown
        # fail-closed. Positive bytes, however, must name at least one record.
        remaining_pair_ok = remaining_bytes == 0 or remaining_slot_count > 0
        bytes_aligned = (
            released_bytes % _UVM_MOE_SLOT_PAGE_BYTES == 0
            and remaining_bytes % _UVM_MOE_SLOT_PAGE_BYTES == 0
        )
        # A stale per-plan owner can set had_registered_session while the
        # global session owner (and therefore generation_before) is absent.
        # The converse is impossible.
        generation_contract_ok = generation_before == 0 or had_registered_session
        clear_contract_ok = session_cleared == (
            remaining_slot_count == 0 and remaining_bytes == 0
        )
        poison_contract_ok = (
            (session_cleared and not kernel_poisoned)
            or (
                not session_cleared
                and remaining_slot_count > 0
                and destroyed_plan_count == 0
                and kernel_poisoned
            )
        )
        contract_ok = (
            int(params.flags) == 0
            and int(params.reserved) == 0
            and boolean_fields_ok
            and released_pair_ok
            and remaining_pair_ok
            and bytes_aligned
            and generation_contract_ok
            and clear_contract_ok
            and poison_contract_ok
        )
        ioctl_completed = ret == 0 and rm_status == 0
        ok = (
            ioctl_completed
            and contract_ok
            and session_cleared
            and remaining_slot_count == 0
            and remaining_bytes == 0
        )
        reason = (
            "ok"
            if ok
            else "slot_session_abort_contract_mismatch"
            if ioctl_completed and not contract_ok
            else "slot_session_abort_incomplete"
            if ioctl_completed
            else "slot_session_abort_ioctl_failed"
        )
        if ok:
            _uvm_slot_direct_clear_resource_cache_locked(device)
        else:
            _uvm_slot_direct_mark_poisoned_locked(
                device,
                (
                    f"{reason}: ret={ret} rmStatus={rm_status} "
                    f"remaining_slots={remaining_slot_count} "
                    f"remaining_bytes={remaining_bytes}"
                ),
            )

        return {
            "ok": ok,
            "reason": reason,
            "generation_before": generation_before,
            "released_bytes": released_bytes,
            "remaining_bytes": remaining_bytes,
            "destroyed_plan_count": destroyed_plan_count,
            "released_slot_count": released_slot_count,
            "remaining_slot_count": remaining_slot_count,
            "had_registered_session": had_registered_session,
            "session_cleared": session_cleared,
            "poisoned": kernel_poisoned,
            "kernel_poisoned": kernel_poisoned,
            "session_poisoned": not ok,
            "contract_ok": contract_ok,
            "ioctl_ret": int(ret),
            "errno": errno_value,
            "rm_status": rm_status,
            "ioctl_calls": 1,
            "ioctl_wall_ns": ioctl_wall_ns,
        }


def _normalize_moe_slot_release_entries(
    entries: Sequence[Sequence[int]],
) -> tuple[list[tuple[int, int, int]], str]:
    try:
        records = list(entries)
    except TypeError:
        return [], "release entries must be a finite sequence"
    if not records:
        return [], "at least one release entry is required"

    identities: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for index, record in enumerate(records):
        try:
            values = tuple(record)
        except TypeError:
            return [], f"release entry {index} is not a sequence"
        if len(values) == 3:
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in values
            ):
                return [], f"release entry {index} fields must be plain integers"
            layer_id, expert_id, slot_id = values
        elif len(values) in (6, 7):
            normalized, error = _normalize_moe_slot_ranges([values])
            if error:
                return [], f"release entry {index}: {error}"
            (
                _base,
                _length,
                layer_id,
                expert_id,
                slot_id,
                _offset,
                _flags,
            ) = normalized[0]
        else:
            return [], f"release entry {index} must contain three, six, or seven fields"

        if any(
            value < 0 or value > _UVM_MOE_SLOT_UINT32_MAX
            for value in (layer_id, expert_id, slot_id)
        ):
            return [], f"release entry {index} identity is outside uint32"
        key = (layer_id, slot_id)
        if key in seen:
            return [], f"release entry {index} repeats logical slot {key}"
        seen.add(key)
        identities.append((layer_id, expert_id, slot_id))
    return identities, ""


def _driver_slot_release_ranges_impl(
    entries: Sequence[Sequence[int]],
    *,
    device: int = 0,
) -> dict[str, object]:
    identities, validation_error = _normalize_moe_slot_release_entries(entries)
    if validation_error:
        return _moe_slot_result(
            ok=False,
            reason="validation_failed",
            mode=_UVM_MOE_SLOT_REMAP_MODE_RELEASE,
            requested_count=len(identities),
            requested_bytes=0,
            error=validation_error,
        )

    with _uvm_slot_direct_lock:
        blocked = _uvm_slot_direct_block_reason_locked(int(device))
        if blocked is not None:
            reason, error = blocked
            return _moe_slot_result(
                ok=False,
                reason=reason,
                mode=_UVM_MOE_SLOT_REMAP_MODE_RELEASE,
                requested_count=len(identities),
                requested_bytes=0,
                error=error,
            )
        cached = _uvm_slot_direct_sessions.get(int(device))
        if cached is None:
            return _moe_slot_result(
                ok=False,
                reason="slot_session_required",
                mode=_UVM_MOE_SLOT_REMAP_MODE_RELEASE,
                requested_count=len(identities),
                requested_bytes=0,
            )
        fd, uuid_bytes = cached
        owners = _uvm_slot_direct_owners.get(int(device), {})
        for layer_id, expert_id, slot_id in identities:
            tracked_expert = owners.get((layer_id, slot_id))
            if tracked_expert != expert_id:
                return _moe_slot_result(
                    ok=False,
                    reason="slot_owner_mismatch",
                    mode=_UVM_MOE_SLOT_REMAP_MODE_RELEASE,
                    requested_count=len(identities),
                    requested_bytes=0,
                    error=(
                        f"layer={layer_id} slot={slot_id} "
                        f"requested_expert={expert_id} tracked_expert={tracked_expert}"
                    ),
                )

        gpu_uuid = _PhaseFUuid(
            (ctypes.c_uint8 * len(uuid_bytes)).from_buffer_copy(uuid_bytes)
        )
        release_records = [
            (0, 0, layer_id, expert_id, slot_id, 0, 0)
            for layer_id, expert_id, slot_id in identities
        ]
        result = _run_moe_slot_batches(
            fd,
            release_records,
            gpu_uuid=gpu_uuid,
            mode=_UVM_MOE_SLOT_REMAP_MODE_RELEASE,
        )
        batches = result.get("batches")
        if isinstance(batches, list):
            for batch in batches:
                if not isinstance(batch, dict):
                    continue
                indices = batch.get("input_indices")
                statuses = batch.get("entry_statuses")
                if not isinstance(indices, list) or not isinstance(statuses, list):
                    continue
                for input_index, status in zip(indices, statuses):
                    if type(status) is int and status == 0:
                        layer_id, _expert_id, slot_id = identities[int(input_index)]
                        owners.pop((layer_id, slot_id), None)
        return result


def driver_slot_preallocate_ranges(
    ranges: Sequence[Sequence[int]],
    *,
    device: int = 0,
) -> dict[str, object]:
    """Preallocate/rebind slots and report complete userspace wall time."""
    start_ns = time.perf_counter_ns()
    result = _driver_slot_preallocate_ranges_impl(ranges, device=device)
    result["wall_ns"] = time.perf_counter_ns() - start_ns
    return result


def driver_slot_restore_ranges(
    ranges: Sequence[Sequence[int]],
    *,
    device: int = 0,
) -> dict[str, object]:
    """Run the gated slot-direct restore mode and report complete wall time."""
    start_ns = time.perf_counter_ns()
    result = _driver_slot_restore_ranges_impl(ranges, device=device)
    result["wall_ns"] = time.perf_counter_ns() - start_ns
    return result


def driver_slot_transition_ranges(
    ranges: Sequence[Sequence[int]],
    *,
    device: int = 0,
) -> dict[str, object]:
    """Run the gated fixed-bank owner-rebind plus policy-selected copy/map."""
    start_ns = time.perf_counter_ns()
    result = _driver_slot_transition_ranges_impl(ranges, device=device)
    result["wall_ns"] = time.perf_counter_ns() - start_ns
    return result


def driver_slot_register_plan(
    plan_id: int,
    layer_id: int,
    ranges: Sequence[Sequence[int]],
    *,
    transition: bool,
    device: int = 0,
) -> dict[str, object]:
    start_ns = time.perf_counter_ns()
    result = _driver_slot_register_plan_impl(
        plan_id,
        layer_id,
        ranges,
        transition=transition,
        device=device,
    )
    result["wall_ns"] = time.perf_counter_ns() - start_ns
    return result


def driver_slot_execute_plan(
    plan_id: int,
    layer_id: int,
    generation: int,
    *,
    device: int = 0,
) -> dict[str, object]:
    start_ns = time.perf_counter_ns()
    result = _driver_slot_execute_plan_impl(
        plan_id,
        layer_id,
        generation,
        device=device,
    )
    result["wall_ns"] = time.perf_counter_ns() - start_ns
    return result


def driver_slot_destroy_plan(
    plan_id: int,
    generation: int,
    *,
    device: int = 0,
) -> dict[str, object]:
    start_ns = time.perf_counter_ns()
    result = _driver_slot_destroy_plan_impl(
        plan_id,
        generation,
        device=device,
    )
    result["wall_ns"] = time.perf_counter_ns() - start_ns
    return result


def driver_slot_abort_session(*, device: int = 0) -> dict[str, object]:
    """Recover a VA-space slot session without requiring its generation."""
    start_ns = time.perf_counter_ns()
    result = _driver_slot_abort_session_impl(device=device)
    result["wall_ns"] = time.perf_counter_ns() - start_ns
    return result


def driver_slot_release_ranges(
    entries: Sequence[Sequence[int]],
    *,
    device: int = 0,
) -> dict[str, object]:
    """Release exactly tracked slot owners with the mode-19 audit contract."""
    start_ns = time.perf_counter_ns()
    result = _driver_slot_release_ranges_impl(entries, device=device)
    result["wall_ns"] = time.perf_counter_ns() - start_ns
    return result


def close_uvm_slot_direct_session(*, device: int = 0) -> dict[str, object]:
    """Prove scoped resources gone, then close the duplicated UVM fd."""
    start_ns = time.perf_counter_ns()
    device = int(device)
    run_policy_cleanup = False
    with _uvm_slot_direct_lock:
        cached = _uvm_slot_direct_sessions.get(device)
        if cached is None:
            _uvm_slot_direct_session_pids.pop(device, None)
            _uvm_slot_direct_poisoned.pop(device, None)
            _uvm_slot_direct_cuda_fence_cleanup_deferred.discard(device)
            result: dict[str, object] = {
                "ok": True,
                "reason": "already_closed",
                "closed": False,
                "release": None,
                "abort": None,
                "plan_destroy": [],
            }
            run_policy_cleanup = True
        else:
            fd, _uuid_bytes = cached
            inherited_pid = _uvm_slot_direct_inherited_pid_locked(device)
            if inherited_pid is not None:
                # A fork child shares the parent's UVM file description. It
                # must never mutate that VA-space; dropping its own descriptor
                # reference is the only permitted cleanup operation.
                try:
                    os.close(fd)
                except OSError as exc:
                    return {
                        "ok": False,
                        "reason": "fork_child_fd_close_failed",
                        "closed": False,
                        "fork_inherited": True,
                        "owner_pid": inherited_pid,
                        "current_pid": os.getpid(),
                        "error": repr(exc),
                        "wall_ns": time.perf_counter_ns() - start_ns,
                    }
                _uvm_slot_direct_clear_session_cache_locked(device)
                return {
                    "ok": True,
                    "reason": "fork_child_fd_closed",
                    "closed": True,
                    "fork_inherited": True,
                    "owner_pid": inherited_pid,
                    "current_pid": os.getpid(),
                    "release": None,
                    "abort": None,
                    "plan_destroy": [],
                    "wall_ns": time.perf_counter_ns() - start_ns,
                }

            if device in _uvm_slot_direct_cuda_fence_cleanup_deferred:
                # A bounded compute-ticket poll already failed. Calling an
                # unbounded CUDA synchronize here would undo that guarantee;
                # keep every mapping, plan, pool, and fd alive until process
                # teardown resets the CUDA/UVM address space.
                return {
                    "ok": False,
                    "reason": "cuda_fence_cleanup_process_teardown_only",
                    "closed": False,
                    "poisoned": True,
                    "deferred_to_process_teardown": True,
                    "release": None,
                    "abort": None,
                    "plan_destroy": [],
                    "wall_ns": time.perf_counter_ns() - start_ns,
                }

            if device in _uvm_slot_direct_cuda_fences:
                try:
                    torch.cuda.synchronize(device)
                except Exception as exc:
                    # Do not destroy plans, clear PTEs, unregister the fence
                    # pool, or close its fd unless all stream work is known to
                    # be stopped. Process/driver teardown remains recoverable.
                    _uvm_slot_direct_mark_poisoned_locked(
                        device, f"CUDA fence cleanup boundary failed: {exc!r}"
                    )
                    return {
                        "ok": False,
                        "reason": "cuda_fence_cleanup_boundary_failed",
                        "closed": False,
                        "poisoned": True,
                        "error": repr(exc),
                        "deferred_to_process_teardown": True,
                        "release": None,
                        "abort": None,
                        "plan_destroy": [],
                        "wall_ns": time.perf_counter_ns() - start_ns,
                    }

            plan_destroy_results: list[dict[str, object]] = []
            release_result: dict[str, object] | None = None
            abort_result: dict[str, object] | None = None
            plan_ids = sorted(_uvm_slot_direct_plans.get(device, {}))
            had_registered_plans = bool(plan_ids)
            cleanup_requires_abort = bool(_uvm_slot_direct_poisoned.get(device))
            cleanup_trigger = (
                "session_poisoned" if cleanup_requires_abort else ""
            )

            if plan_ids and not cleanup_requires_abort:
                if device not in _uvm_slot_direct_plan_generations:
                    cleanup_requires_abort = True
                    cleanup_trigger = "untrusted_plan_generation"
                else:
                    for plan_id in plan_ids:
                        generation = _uvm_slot_direct_plan_generations.get(device)
                        if generation is None:
                            cleanup_requires_abort = True
                            cleanup_trigger = "untrusted_plan_generation"
                            break
                        try:
                            destroy_result = _driver_slot_destroy_plan_impl(
                                plan_id,
                                generation,
                                device=device,
                            )
                        except Exception as exc:
                            destroy_result = {
                                "ok": False,
                                "reason": "plan_destroy_exception",
                                "plan_id": plan_id,
                                "generation": generation,
                                "error": repr(exc),
                            }
                        plan_destroy_results.append(destroy_result)
                        if not bool(destroy_result.get("ok")):
                            cleanup_requires_abort = True
                            cleanup_trigger = "plan_destroy_failed"
                            break

            # A registered-plan session always receives the scoped abort audit
            # after explicit destroys. This catches any kernel plan/slot that
            # userspace failed to cache. Poisoned, partially destroyed, and
            # unknown-generation sessions skip identity-based release and go
            # straight to the same generation-free recovery path.
            if had_registered_plans or cleanup_requires_abort:
                abort_result = _driver_slot_abort_session_impl(device=device)
                if not bool(abort_result.get("ok")):
                    return {
                        "ok": False,
                        "reason": "slot_session_abort_failed",
                        "cleanup_trigger": cleanup_trigger,
                        "closed": False,
                        "release": None,
                        "abort": abort_result,
                        "plan_destroy": plan_destroy_results,
                        "poisoned": True,
                        "wall_ns": time.perf_counter_ns() - start_ns,
                    }
            else:
                owners = dict(_uvm_slot_direct_owners.get(device, {}))
                if owners:
                    release_entries = [
                        (layer_id, expert_id, slot_id)
                        for (layer_id, slot_id), expert_id in sorted(owners.items())
                    ]
                    try:
                        release_result = _driver_slot_release_ranges_impl(
                            release_entries,
                            device=device,
                        )
                    except Exception as exc:
                        release_result = {
                            "ok": False,
                            "reason": "slot_release_exception",
                            "error": repr(exc),
                        }
                    if not bool(release_result.get("ok")):
                        _uvm_slot_direct_mark_poisoned_locked(
                            device, "identity-based slot release failed"
                        )
                        abort_result = _driver_slot_abort_session_impl(device=device)
                        if not bool(abort_result.get("ok")):
                            return {
                                "ok": False,
                                "reason": "slot_session_abort_failed",
                                "cleanup_trigger": "slot_release_failed",
                                "closed": False,
                                "release": release_result,
                                "abort": abort_result,
                                "plan_destroy": plan_destroy_results,
                                "poisoned": True,
                                "wall_ns": time.perf_counter_ns() - start_ns,
                            }

            if _uvm_slot_direct_sessions.get(device) != cached:
                _uvm_slot_direct_mark_poisoned_locked(
                    device, "slot session identity changed during close"
                )
                return {
                    "ok": False,
                    "reason": "slot_session_changed",
                    "closed": False,
                    "release": release_result,
                    "abort": abort_result,
                    "plan_destroy": plan_destroy_results,
                    "poisoned": True,
                    "wall_ns": time.perf_counter_ns() - start_ns,
                }
            try:
                _uvm_slot_cuda_fence_destroy_locked(device=device, fd=fd)
            except Exception as exc:
                _uvm_slot_direct_mark_poisoned_locked(
                    device, f"CUDA fence pool cleanup failed: {exc!r}"
                )
                return {
                    "ok": False,
                    "reason": "cuda_fence_pool_cleanup_failed",
                    "closed": False,
                    "release": release_result,
                    "abort": abort_result,
                    "plan_destroy": plan_destroy_results,
                    "error": repr(exc),
                    "poisoned": True,
                    "wall_ns": time.perf_counter_ns() - start_ns,
                }
            try:
                os.close(fd)
            except OSError as exc:
                # Driver resources are already gone, but retain the fd and PID
                # identity in local state so an explicit retry can close it.
                return {
                    "ok": False,
                    "reason": "fd_close_failed",
                    "closed": False,
                    "release": release_result,
                    "abort": abort_result,
                    "plan_destroy": plan_destroy_results,
                    "error": repr(exc),
                    "wall_ns": time.perf_counter_ns() - start_ns,
                }
            _uvm_slot_direct_clear_session_cache_locked(device)
            result = {
                "ok": True,
                "reason": "closed",
                "closed": True,
                "cleanup_trigger": (
                    cleanup_trigger
                    or (
                        "registered_plan_abort_audit"
                        if had_registered_plans
                        else "identity_release"
                    )
                ),
                "release": release_result,
                "abort": abort_result,
                "plan_destroy": plan_destroy_results,
                "forced_fd_cleanup": False,
            }
            run_policy_cleanup = True

    if run_policy_cleanup:
        _run_uvm_slot_direct_policy_cleanup(device)
    result["wall_ns"] = time.perf_counter_ns() - start_ns
    return result


def find_uvm_allocator_library() -> str | None:
    """
    Find the UVM allocator shared library.

    Returns:
        Path to the library if found, None otherwise.
    """
    # Try to find the library in common locations. In local editable layouts,
    # ``import vllm`` may resolve as a namespace package with ``__file__`` set
    # to None, so derive the package path from this module first.
    vllm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    # Possible library names and locations
    lib_names = [
        "uvm_allocator.abi3.so",  # ABI3 build (most common)
        "uvm_allocator.so",
        "libuvm_allocator.so",
    ]
    search_paths = [
        os.path.join(vllm_path, "..", "uvm_test"),  # Local development build
        vllm_path,  # Installed location
        os.path.join(vllm_path, ".."),  # Development location
        os.path.dirname(os.path.abspath(__file__)),  # Same directory
    ]

    for path in search_paths:
        for lib_name in lib_names:
            lib_path = os.path.join(path, lib_name)
            if os.path.exists(lib_path):
                return lib_path

    # Try to find using /proc/self/maps (similar to cumem allocator)
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "uvm_allocator" in line:
                    start = line.index("/")
                    return line[start:].strip()
    except (FileNotFoundError, ValueError):
        pass

    return None


def load_uvm_library() -> ctypes.CDLL:
    """
    Load the UVM allocator library and set up function signatures.

    Returns:
        CDLL handle to the library.

    Raises:
        RuntimeError: If library cannot be found or loaded.
    """
    lib_path = find_uvm_allocator_library()

    if lib_path is None:
        raise RuntimeError(
            "UVM allocator library not found. "
            "Please ensure vLLM is built with UVM support enabled. "
            "The library should be at: <vllm_path>/uvm_allocator.so"
        )

    logger.info("Loading UVM allocator library from: %s", lib_path)

    try:
        lib = ctypes.CDLL(lib_path)
    except OSError as e:
        raise RuntimeError(f"Failed to load UVM allocator library: {e}") from e

    # Define function signatures for statistics API
    lib.uvm_get_allocated_bytes.restype = ctypes.c_size_t
    lib.uvm_get_allocated_bytes.argtypes = []

    lib.uvm_get_peak_allocated_bytes.restype = ctypes.c_size_t
    lib.uvm_get_peak_allocated_bytes.argtypes = []

    lib.uvm_get_num_allocs.restype = ctypes.c_size_t
    lib.uvm_get_num_allocs.argtypes = []

    lib.uvm_get_num_frees.restype = ctypes.c_size_t
    lib.uvm_get_num_frees.argtypes = []

    lib.uvm_reset_peak_stats.restype = None
    lib.uvm_reset_peak_stats.argtypes = []

    lib.uvm_reset_all_stats.restype = None
    lib.uvm_reset_all_stats.argtypes = []

    lib.uvm_set_prefetch.restype = None
    lib.uvm_set_prefetch.argtypes = [ctypes.c_int]

    lib.uvm_set_verbose.restype = None
    lib.uvm_set_verbose.argtypes = [ctypes.c_int]

    lib.uvm_set_phase.restype = None
    lib.uvm_set_phase.argtypes = [ctypes.c_char_p]

    lib.uvm_mark_phase_event.restype = None
    lib.uvm_mark_phase_event.argtypes = [ctypes.c_char_p, ctypes.c_char_p]

    lib.uvm_prefetch.restype = None
    lib.uvm_prefetch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]

    lib.uvm_batch_prefetch.restype = ctypes.c_int
    lib.uvm_batch_prefetch.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]

    lib.uvm_advise.restype = None
    lib.uvm_advise.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
    ]

    lib.uvm_batch_advise.restype = ctypes.c_int
    lib.uvm_batch_advise.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
    ]

    lib.uvm_driver_batch_prefetch_cpu.restype = ctypes.c_int
    lib.uvm_driver_batch_prefetch_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
    ]

    lib.uvm_driver_batch_prefetch_gpu.restype = ctypes.c_int
    lib.uvm_driver_batch_prefetch_gpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
    ]

    lib.uvm_driver_batch_advise_cpu.restype = ctypes.c_int
    lib.uvm_driver_batch_advise_cpu.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    ]

    lib.uvm_driver_batch_query_residency.restype = ctypes.c_int
    lib.uvm_driver_batch_query_residency.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
    ]

    for helper_name in (
        "uvm_driver_batch_prepare_remote_read",
        "uvm_driver_batch_prepare_remote_read_bg_deferred",
        "uvm_driver_batch_start_bg_promotion",
        "uvm_driver_batch_pause_bg_promotion",
        "uvm_driver_batch_resume_bg_promotion",
    ):
        if hasattr(lib, helper_name):
            helper = getattr(lib, helper_name)
            helper.restype = ctypes.c_int
            helper.argtypes = [
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
            ]

    lib.uvm_driver_batch_query_residency_bytes.restype = ctypes.c_int
    lib.uvm_driver_batch_query_residency_bytes.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    if hasattr(lib, "uvm_driver_batch_query_residency_bytes_nowait"):
        lib.uvm_driver_batch_query_residency_bytes_nowait.restype = ctypes.c_int
        lib.uvm_driver_batch_query_residency_bytes_nowait.argtypes = (
            lib.uvm_driver_batch_query_residency_bytes.argtypes
        )

    lib.uvm_driver_phase_f_register_region.restype = ctypes.c_int
    lib.uvm_driver_phase_f_register_region.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]

    lib.uvm_driver_phase_f_unregister_region.restype = ctypes.c_int
    lib.uvm_driver_phase_f_unregister_region.argtypes = [ctypes.c_uint64]

    if hasattr(lib, "uvm_driver_update_region_state"):
        lib.uvm_driver_update_region_state.restype = ctypes.c_int
        lib.uvm_driver_update_region_state.argtypes = [
            ctypes.c_uint64,
            ctypes.c_uint32,
        ]

    if hasattr(lib, "uvm_driver_query_region"):
        lib.uvm_driver_query_region.restype = ctypes.c_int
        lib.uvm_driver_query_region.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]

    if hasattr(lib, "uvm_driver_query_region_by_va"):
        lib.uvm_driver_query_region_by_va.restype = ctypes.c_int
        lib.uvm_driver_query_region_by_va.argtypes = [
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]

    lib.uvm_is_managed_pointer.restype = ctypes.c_int
    lib.uvm_is_managed_pointer.argtypes = [
        ctypes.c_uint64,
        ctypes.c_size_t,
    ]

    lib.uvm_close_log.restype = None
    lib.uvm_close_log.argtypes = []

    return lib


def _close_uvm_log_at_exit() -> None:
    if _uvm_lib is None:
        return
    try:
        _uvm_lib.uvm_close_log()
    except Exception:
        logger.debug("Failed to close UVM allocator log", exc_info=True)


def ensure_uvm_runtime_library() -> ctypes.CDLL:
    """Load the UVM helper library without enabling the global CUDA allocator.

    This is used by isolated experiments that allocate only selected tensors
    through a local PyTorch MemPool while the rest of vLLM keeps the ordinary
    CUDA allocator.
    """
    global _uvm_lib
    if _uvm_lib is None:
        _uvm_lib = load_uvm_library()
    _uvm_lib.uvm_set_prefetch(1 if _uvm_local_prefetch_enabled() else 0)
    _uvm_lib.uvm_set_verbose(0)
    return _uvm_lib


def _uvm_local_prefetch_enabled() -> bool:
    return _uvm_local_prefetch_stack[-1] if _uvm_local_prefetch_stack else False


@contextmanager
def uvm_local_memory_pool(enable_prefetch: bool | None = None):
    """Temporarily allocate CUDA tensors from cudaMallocManaged.

    Unlike ``enable_uvm_allocator()``, this does not call
    ``change_current_allocator``. Only allocations inside this context use the
    UVM pluggable allocator, which lets CPU-offload experiments replace just
    the offloaded weight tensors.

    If ``enable_prefetch`` is omitted, the current nested prefetch state is
    inherited. This keeps outer pre-service UVM scopes intact even when helper
    code opens its own local pool.
    """
    global _uvm_local_allocator, _uvm_local_mem_pool
    if enable_prefetch is None:
        enable_prefetch = _uvm_local_prefetch_enabled()
    nested_pool = bool(_uvm_local_prefetch_stack)
    _uvm_local_prefetch_stack.append(bool(enable_prefetch))
    try:
        ensure_uvm_runtime_library()
        lib_path = find_uvm_allocator_library()
        if lib_path is None:
            raise RuntimeError("UVM allocator library not found")

        if _uvm_local_allocator is None or _uvm_local_mem_pool is None:
            allocator = torch.cuda.memory.CUDAPluggableAllocator(
                lib_path, "uvm_malloc", "uvm_free"
            )
            _uvm_local_allocator = allocator
            _uvm_local_mem_pool = torch.cuda.memory.MemPool(allocator._allocator)

        if _uvm_lib is not None:
            _uvm_lib.uvm_set_prefetch(1 if enable_prefetch else 0)

        if nested_pool:
            # PyTorch cannot begin recording the same MemPool twice. The
            # outer scope already selects it; nested scopes only change the
            # allocator's prefetch setting and restore it on exit.
            yield
        else:
            with torch.cuda.memory.use_mem_pool(_uvm_local_mem_pool):
                yield
    finally:
        if _uvm_local_prefetch_stack:
            _uvm_local_prefetch_stack.pop()
        if _uvm_lib is not None:
            _uvm_lib.uvm_set_prefetch(1 if _uvm_local_prefetch_enabled() else 0)


def release_uvm_local_memory_pool() -> bool:
    """Destroy the process-local managed MemPool before interpreter teardown.

    Standalone tools must first release every tensor allocated from the pool.
    Explicit destruction keeps the pluggable allocator callback library alive
    while PyTorch releases the pool, avoiding undefined module-finalization
    order at Python shutdown. Long-lived vLLM workers normally retain the pool
    for their full lifetime and do not need this helper.
    """
    global _uvm_local_allocator, _uvm_local_mem_pool
    if _uvm_local_prefetch_stack:
        raise RuntimeError("cannot release an active UVM local memory-pool scope")
    pool = _uvm_local_mem_pool
    allocator = _uvm_local_allocator
    if pool is None and allocator is None:
        return False
    if pool is not None and int(pool.use_count()) > 1:
        raise RuntimeError(
            "cannot release UVM local memory pool while it is still in use: "
            f"use_count={pool.use_count()}"
        )
    _uvm_local_mem_pool = None
    _uvm_local_allocator = None
    del pool
    gc.collect()
    del allocator
    gc.collect()
    return True


def enable_uvm_allocator(
    enable_prefetch: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Enable the UVM allocator for PyTorch CUDA operations.

    This must be called before any CUDA memory allocations occur.

    Args:
        enable_prefetch: If True, prefetch allocations to GPU after allocation.
            This can improve performance but uses more GPU memory.
        verbose: If True, log large allocations to stderr.

    Returns:
        True if UVM was enabled successfully, False otherwise.

    Raises:
        RuntimeError: If UVM allocator library is not available.
    """
    global _uvm_enabled, _uvm_lib, _uvm_log_close_registered

    if _uvm_enabled:
        logger.warning("UVM allocator is already enabled")
        return True

    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available, UVM allocator cannot be enabled")
        return False

    # Load the library
    _uvm_lib = load_uvm_library()

    # Configure the allocator
    _uvm_lib.uvm_set_prefetch(1 if enable_prefetch else 0)
    _uvm_lib.uvm_set_verbose(1 if verbose else 0)

    # Find library path for PyTorch pluggable allocator
    lib_path = find_uvm_allocator_library()
    if lib_path is None:
        raise RuntimeError("UVM allocator library not found")

    try:
        # Create and enable the pluggable allocator
        allocator = torch.cuda.memory.CUDAPluggableAllocator(
            lib_path, "uvm_malloc", "uvm_free"
        )
        torch.cuda.memory.change_current_allocator(allocator)
        _uvm_enabled = True
        if not _uvm_log_close_registered:
            atexit.register(_close_uvm_log_at_exit)
            _uvm_log_close_registered = True
        logger.info("UVM allocator enabled successfully")
        _set_uvm_phase("enabled")
        mark_uvm_phase_event("allocator_enabled", "enabled")

        # Pre-initialize cuBLAS after UVM is enabled
        # This ensures cuBLAS workspace is allocated via UVM
        # and won't fail later due to native memory exhaustion
        try:
            logger.info("Pre-initializing cuBLAS with UVM...")
            with uvm_allocation_phase("uvm_enable:cublas_preinit"):
                a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
                b = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
                _ = torch.matmul(a, b)
                torch.cuda.synchronize()
                del a, b
                torch.cuda.empty_cache()
            logger.info("cuBLAS pre-initialized successfully with UVM")
        except Exception as e:
            logger.warning("Failed to pre-initialize cuBLAS: %s", e)

        return True

    except Exception as e:
        logger.error("Failed to enable UVM allocator: %s", e)
        raise RuntimeError(f"Failed to enable UVM allocator: {e}") from e


def is_uvm_enabled() -> bool:
    """Check if UVM allocator is currently enabled."""
    return _uvm_enabled


def _set_uvm_phase(phase: str) -> None:
    if _uvm_lib is None:
        return
    _uvm_lib.uvm_set_phase(phase.encode("utf-8"))


def mark_uvm_phase_event(event: str, phase: str | None = None) -> None:
    if _uvm_lib is None:
        return
    event_bytes = event.encode("utf-8")
    phase_bytes = phase.encode("utf-8") if phase is not None else None
    _uvm_lib.uvm_mark_phase_event(event_bytes, phase_bytes)


@contextmanager
def uvm_allocation_phase(phase: str):
    """Attach allocator traces to a high-level execution phase."""
    if not _uvm_enabled or _uvm_lib is None:
        yield
        return

    _uvm_phase_stack.append(phase)
    mark_uvm_phase_event("enter", phase)
    _set_uvm_phase(phase)
    try:
        yield
    finally:
        mark_uvm_phase_event("exit", phase)
        _uvm_phase_stack.pop()
        restored = _uvm_phase_stack[-1] if _uvm_phase_stack else "enabled"
        _set_uvm_phase(restored)


@contextmanager
def uvm_enabled_allocation_phase(phase: str):
    """Attach a finer phase only while serving is in enabled mode."""
    if not _uvm_enabled or _uvm_lib is None:
        yield
        return

    current_phase = _uvm_phase_stack[-1] if _uvm_phase_stack else "enabled"
    if current_phase == "enabled" or current_phase.startswith("enabled:"):
        with uvm_allocation_phase(phase):
            yield
    else:
        yield


def prefetch_range_to_device(ptr: int, size: int, device: int = 0) -> bool:
    """
    Prefetch a raw UVM address range to the specified GPU device.

    Returns True when a prefetch call was issued. This is intentionally a thin
    wrapper around the allocator .so so higher-level policies can prefetch
    logical tensor slices without constructing tensor views.
    """
    if uvm_residency_tag_only_enabled():
        return False
    if _uvm_lib is None:
        return False
    if ptr <= 0 or size <= 0:
        return False
    return batch_prefetch_ranges_to_device(((ptr, size),), device=device) == 1


def probe_driver_capable_range(ptr: int, size: int) -> tuple[bool, str, int]:
    """Return whether a raw range is CUDA managed and suitable for driver ioctls.

    The Phase F driver executor only works on ranges visible to UVM as managed
    VA blocks. This probe records the CUDA runtime's pointer classification so
    weight traces can distinguish non-managed tensors from driver plumbing bugs.
    """
    try:
        lib = ensure_uvm_runtime_library()
    except Exception:
        logger.debug("Failed to load UVM runtime library", exc_info=True)
        return False, "uvm_library_unavailable", -2
    if ptr <= 0 or size <= 0:
        return False, "invalid_range", -3
    try:
        result = int(
            lib.uvm_is_managed_pointer(
                ctypes.c_uint64(ptr),
                ctypes.c_size_t(size),
            )
        )
    except Exception:
        logger.debug("Failed to probe UVM pointer attributes", exc_info=True)
        return False, "probe_exception", -4
    if result == 1:
        return True, "cuda_managed", result
    if result == 0:
        return False, "not_cuda_managed", result
    return False, f"cuda_pointer_probe_failed:{result}", result


def _prepare_uvm_range_arrays(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> tuple[ctypes.Array[ctypes.c_uint64], ctypes.Array[ctypes.c_size_t], int]:
    valid_ranges = [
        (int(ptr), int(size)) for ptr, size in ranges if int(ptr) > 0 and int(size) > 0
    ]
    count = len(valid_ranges)
    ptr_array = (ctypes.c_uint64 * count)(*(ptr for ptr, _ in valid_ranges))
    size_array = (ctypes.c_size_t * count)(*(size for _, size in valid_ranges))
    return ptr_array, size_array, count


def get_uvm_page_size() -> int:
    global _UVM_PAGE_SIZE
    if _UVM_PAGE_SIZE > 0:
        return _UVM_PAGE_SIZE

    raw = os.environ.get("VLLM_UVM_HYBRID_PAGE_SIZE", "").strip()
    if raw:
        try:
            value = int(raw, 0)
            if value > 0:
                _UVM_PAGE_SIZE = value
                return _UVM_PAGE_SIZE
        except ValueError:
            logger.warning("Invalid VLLM_UVM_HYBRID_PAGE_SIZE=%r", raw)

    try:
        _UVM_PAGE_SIZE = int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        _UVM_PAGE_SIZE = 4096
    if _UVM_PAGE_SIZE <= 0:
        _UVM_PAGE_SIZE = 4096
    return _UVM_PAGE_SIZE


def round_ranges_to_pages(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    page_size: int | None = None,
) -> list[tuple[int, int]]:
    """Round raw byte ranges to page ranges and merge overlaps.

    This is the intentionally aggressive experimental normalization used by
    the hybrid Phase-F ioctl path. It assumes the rounded pages are safe to
    treat as readonly managed weight pages.
    """
    page = int(page_size or get_uvm_page_size())
    if page <= 0:
        page = 4096

    rounded: list[tuple[int, int]] = []
    for ptr, size in ranges:
        start = int(ptr)
        length = int(size)
        if start <= 0 or length <= 0:
            continue
        end = start + length
        rounded_start = start - (start % page)
        rounded_end = ((end + page - 1) // page) * page
        if rounded_end <= rounded_start:
            continue
        rounded.append((rounded_start, rounded_end - rounded_start))

    if not rounded:
        return []

    rounded.sort()
    merged: list[tuple[int, int]] = []
    cur_start, cur_size = rounded[0]
    cur_end = cur_start + cur_size
    for start, size in rounded[1:]:
        end = start + size
        if start <= cur_end:
            cur_end = max(cur_end, end)
            continue
        merged.append((cur_start, cur_end - cur_start))
        cur_start, cur_end = start, end
    merged.append((cur_start, cur_end - cur_start))
    return merged


def _phase_f_cpp_helper_name(action: int) -> str | None:
    if action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ:
        return "uvm_driver_batch_prepare_remote_read"
    if action == UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED:
        return "uvm_driver_batch_prepare_remote_read_bg_deferred"
    if action == UVM_PHASE_F_ACTION_START_BG_PROMOTION:
        return "uvm_driver_batch_start_bg_promotion"
    if action == UVM_PHASE_F_ACTION_PAUSE_BG_PROMOTION:
        return "uvm_driver_batch_pause_bg_promotion"
    if action == UVM_PHASE_F_ACTION_RESUME_BG_PROMOTION:
        return "uvm_driver_batch_resume_bg_promotion"
    return None


def _phase_f_cpp_helper_enabled() -> bool:
    raw = os.environ.get("VLLM_UVM_PHASE_F_CPP_HELPER")
    if raw is None:
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _driver_phase_f_batch_execute_ranges_cpp_helper(
    ranges: list[tuple[int, int]],
    action: int,
    *,
    device: int = 0,
    async_: bool = False,
) -> tuple[int, int, int, int] | None:
    if not _phase_f_cpp_helper_enabled() or _uvm_lib is None:
        return None

    helper_name = _phase_f_cpp_helper_name(action)
    if helper_name is None or not hasattr(_uvm_lib, helper_name):
        return None

    helper = getattr(_uvm_lib, helper_name)
    total_success = 0
    total_failure = 0
    last_rm_status = 0
    for start in range(0, len(ranges), _UVM_DRIVER_BATCH_MAX_RANGES):
        chunk = ranges[start : start + _UVM_DRIVER_BATCH_MAX_RANGES]
        ptr_array, size_array, count = _prepare_uvm_range_arrays(chunk)
        if count <= 0:
            continue
        result = int(
            helper(
                ptr_array,
                size_array,
                ctypes.c_size_t(count),
                ctypes.c_int(device),
                ctypes.c_int(1 if async_ else 0),
            )
        )
        if result < 0:
            if result < -6:
                last_rm_status = -result
            total_failure += count
            logger.warning(
                "Phase-F C++ helper %s failed: result=%s rmStatus=%s "
                "success=%s failure=%s",
                helper_name,
                result,
                last_rm_status,
                total_success,
                total_failure,
            )
            return total_success, total_failure, last_rm_status, result
        total_success += result
        if result < count:
            total_failure += count - result

    return total_success, total_failure, last_rm_status, 0


def driver_phase_f_batch_execute_ranges(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    action: int,
    *,
    device: int = 0,
    async_: bool = False,
) -> tuple[int, int, int, int]:
    """Submit raw VA ranges to the modified UVM Phase-F executor.

    This pure-Python ioctl path is intentionally independent from the
    pluggable allocator .so exports, so isolated vLLM experiments can exercise
    newly added kernel actions without rebuilding vLLM or the allocator helper.

    Returns (success_count, failure_count, rm_status, ioctl_ret). Negative
    ``ioctl_ret`` values are local wrapper failures.
    """
    if uvm_residency_tag_only_enabled():
        return 0, 0, 0, 0

    range_list = [
        (int(ptr), int(size)) for ptr, size in ranges if int(ptr) > 0 and int(size) > 0
    ]
    if not range_list:
        return 0, 0, 0, 0

    helper_result = _driver_phase_f_batch_execute_ranges_cpp_helper(
        range_list,
        action,
        device=device,
        async_=async_,
    )
    if helper_result is not None:
        return helper_result

    try:
        gpu_uuid = _phase_f_gpu_uuid(device)
    except Exception:
        logger.debug("Failed to resolve CUDA UUID for Phase-F ioctl", exc_info=True)
        return 0, len(range_list), 0, -3

    fd = -1
    try:
        fd = _open_or_adopt_initialized_uvm_fd(prefer_existing=True)
        total_success = 0
        total_failure = 0
        last_rm_status = 0

        for start in range(0, len(range_list), _UVM_PHASE_F_MAX_RANGES):
            chunk = range_list[start : start + _UVM_PHASE_F_MAX_RANGES]
            params = _PhaseFBatchExecutorParams()
            params.action = int(action)
            params.count = len(chunk)
            params.gpu_uuid = gpu_uuid
            params.async_ = 1 if async_ else 0
            for index, (ptr, size) in enumerate(chunk):
                params.ranges[index].base = int(ptr)
                params.ranges[index].length = int(size)

            ret = _libc_ioctl(fd, _UVM_PHASE_F_BATCH_EXECUTOR_IOCTL, params)
            if ret != 0:
                saved_errno = ctypes.get_errno()
                logger.warning(
                    "Phase-F ioctl action=%s failed: ret=%s errno=%s "
                    "rmStatus=%s success=%s failure=%s",
                    action,
                    ret,
                    saved_errno,
                    int(params.rmStatus),
                    int(params.success_count),
                    int(params.failure_count),
                )
                total_success += int(params.success_count)
                total_failure += int(params.failure_count)
                return total_success, total_failure, int(params.rmStatus), -6

            total_success += int(params.success_count)
            total_failure += int(params.failure_count)
            last_rm_status = int(params.rmStatus)
            if int(params.rmStatus) != 0 and int(params.success_count) == 0:
                return total_success, total_failure, last_rm_status, -last_rm_status

        return total_success, total_failure, last_rm_status, 0
    except Exception:
        logger.debug("Phase-F direct ioctl failed", exc_info=True)
        return 0, len(range_list), 0, -4
    finally:
        if fd >= 0:
            with suppress(OSError):
                os.close(fd)


def driver_promote_ranges_to_read_duplicate_fast(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    device: int = 0,
) -> dict[str, object]:
    """Re-establish immutable CPU/GPU read-duplicate state for weight slices.

    CUDA Graph capture can touch only a routed subset of logically-hot expert
    pages.  A later Plan Class cutover requires every added-cold page to carry
    UVM's authoritative read-duplicate and GPU-READ-PTE certificate.  Action
    18 promotes exactly those differential ranges without changing their VAs.
    """
    normalized = [
        (int(ptr), int(size))
        for ptr, size in ranges
        if int(ptr) > 0 and int(size) > 0
    ]
    requested_bytes = sum(size for _, size in normalized)
    empty_result: dict[str, object] = {
        "ok": True,
        "action": "expert_read_duplicate_fast",
        "range_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "requested_bytes": 0,
        "gpu_read_duplicate_bytes": 0,
        "cpu_shadow_bytes": 0,
        "rm_status": 0,
        "ioctl_ret": 0,
        "results": (),
    }
    if not normalized:
        return empty_result
    if not _phase_f_batch_abi_matches_kernel():
        return {
            **empty_result,
            "ok": False,
            "range_count": len(normalized),
            "failure_count": len(normalized),
            "requested_bytes": int(requested_bytes),
            "ioctl_ret": -2,
            "reason": "phase_f_abi_mismatch",
        }

    try:
        gpu_uuid = _phase_f_gpu_uuid(int(device))
    except Exception:
        logger.debug("Failed to resolve CUDA UUID for expert promotion", exc_info=True)
        return {
            **empty_result,
            "ok": False,
            "range_count": len(normalized),
            "failure_count": len(normalized),
            "requested_bytes": int(requested_bytes),
            "ioctl_ret": -3,
            "reason": "gpu_uuid_failed",
        }

    def run_candidate(
        fd: int, *, source_fd: int, candidate_index: int
    ) -> dict[str, object]:
        success_count = 0
        failure_count = 0
        gpu_read_duplicate_bytes = 0
        cpu_shadow_bytes = 0
        last_rm_status = 0
        last_ioctl_ret = 0
        records: list[dict[str, int]] = []

        for start in range(0, len(normalized), _UVM_PHASE_F_MAX_RANGES):
            chunk = normalized[start : start + _UVM_PHASE_F_MAX_RANGES]
            params = _PhaseFBatchExecutorParams()
            params.action = UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST
            params.count = len(chunk)
            params.gpu_uuid = gpu_uuid
            params.async_ = 0
            for index, (ptr, size) in enumerate(chunk):
                params.ranges[index].base = int(ptr)
                params.ranges[index].length = int(size)

            ctypes.set_errno(0)
            last_ioctl_ret = int(
                _libc_ioctl(fd, _UVM_PHASE_F_BATCH_EXECUTOR_IOCTL, params)
            )
            last_rm_status = int(params.rmStatus)
            success_count += int(params.success_count)
            failure_count += int(params.failure_count)
            for index, (ptr, size) in enumerate(chunk):
                raw = params.results[index]
                record = {
                    "base": int(ptr),
                    "length": int(size),
                    "status": int(raw.status),
                    "gpu_read_duplicate_bytes": int(raw.resident_gpu_bytes),
                    "cpu_shadow_bytes": int(raw.resident_cpu_bytes),
                    "covered_bytes": int(raw.resident_any_bytes),
                    "unknown_bytes": int(raw.resident_unknown_bytes),
                }
                records.append(record)
                gpu_read_duplicate_bytes += int(raw.resident_gpu_bytes)
                cpu_shadow_bytes += int(raw.resident_cpu_bytes)

            if last_ioctl_ret != 0 or last_rm_status != 0:
                break

        complete = (
            last_ioctl_ret == 0
            and last_rm_status == 0
            and success_count == len(normalized)
            and failure_count == 0
            and len(records) == len(normalized)
            and all(
                int(record["status"]) == 0
                and int(record["covered_bytes"]) == int(record["length"])
                and int(record["unknown_bytes"]) == 0
                for record in records
            )
        )
        made_progress = any(
            (
                success_count > 0,
                gpu_read_duplicate_bytes > 0,
                cpu_shadow_bytes > 0,
            )
        )
        fd_mismatch_proven = (
            not made_progress
            and last_ioctl_ret == 0
            and last_rm_status == _NV_ERR_INVALID_DEVICE
            and success_count == 0
            and failure_count > 0
            and bool(records)
            and all(
                int(record["status"]) == _NV_ERR_INVALID_DEVICE
                and int(record["gpu_read_duplicate_bytes"]) == 0
                and int(record["cpu_shadow_bytes"]) == 0
                for record in records
            )
        )
        return {
            "ok": bool(complete),
            "action": "expert_read_duplicate_fast",
            "range_count": len(normalized),
            "success_count": int(success_count),
            "failure_count": int(failure_count),
            "requested_bytes": int(requested_bytes),
            "gpu_read_duplicate_bytes": int(gpu_read_duplicate_bytes),
            "cpu_shadow_bytes": int(cpu_shadow_bytes),
            "rm_status": int(last_rm_status),
            "ioctl_ret": int(last_ioctl_ret),
            "results": tuple(records),
            "made_progress": bool(made_progress),
            "fd_mismatch_proven": bool(fd_mismatch_proven),
            "fd_candidate_index": int(candidate_index),
            "selected_fd_source": int(source_fd),
        }

    last_candidate: dict[str, object] | None = None
    try:
        for candidate_index, source_fd in enumerate(
            _iter_existing_uvm_fd_sources(), start=1
        ):
            fd = _initialize_duplicated_uvm_fd(source_fd)
            if fd is None:
                continue
            try:
                candidate = run_candidate(
                    fd,
                    source_fd=int(source_fd),
                    candidate_index=candidate_index,
                )
            finally:
                with suppress(OSError):
                    os.close(fd)
            if (
                bool(candidate["ok"])
                or bool(candidate["made_progress"])
                or not bool(candidate["fd_mismatch_proven"])
            ):
                return candidate
            last_candidate = candidate

        fd = _open_initialized_uvm_fd()
        try:
            fallback = run_candidate(
                fd,
                source_fd=-1,
                candidate_index=(
                    int((last_candidate or {}).get("fd_candidate_index", 0)) + 1
                ),
            )
        finally:
            with suppress(OSError):
                os.close(fd)
        if (
            not bool(fallback["ok"])
            and not bool(fallback["made_progress"])
            and bool(fallback["fd_mismatch_proven"])
        ):
            fallback["reason"] = "no_matching_uvm_fd"
        return fallback
    except Exception as exc:
        logger.debug("Expert read-duplicate promotion failed", exc_info=True)
        return {
            **empty_result,
            "ok": False,
            "range_count": len(normalized),
            "failure_count": len(normalized),
            "requested_bytes": int(requested_bytes),
            "ioctl_ret": -4,
            "reason": repr(exc),
        }


def driver_rebuild_ranges_gpu_read_mappings(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    device: int = 0,
) -> dict[str, object]:
    """Rebuild complete Parameter GPU READ mappings without moving data.

    The kernel action is synchronous and audits every page after rebuilding.
    Its authoritative residency masks choose the mapping target: a page with a
    target-GPU copy maps locally, otherwise an existing CPU copy maps remotely.
    The public result intentionally reports operation counts; byte-level local
    versus remote evidence is emitted by the kernel route log and physical
    residency remains available through the normal residency query API.
    """
    normalized = [
        (int(base), int(length))
        for base, length in ranges
        if int(base) > 0 and int(length) > 0
    ]
    requested_count = len(normalized)
    requested_bytes = sum(length for _base, length in normalized)
    if not normalized:
        return {
            "ok": False,
            "reason": "no_ranges",
            "requested_count": 0,
            "requested_bytes": 0,
            "success_count": 0,
            "failure_count": 0,
            "rm_status": 0,
            "ioctl_ret": 0,
        }

    page_size = int(get_uvm_page_size())
    if page_size <= 0:
        page_size = 4096
    if any(
        base % page_size != 0 or length % page_size != 0
        for base, length in normalized
    ):
        return {
            "ok": False,
            "reason": "unaligned_parameter_range",
            "requested_count": requested_count,
            "requested_bytes": requested_bytes,
            "success_count": 0,
            "failure_count": requested_count,
            "rm_status": 0,
            "ioctl_ret": -1,
        }

    if not _phase_f_batch_abi_matches_kernel():
        return {
            "ok": False,
            "reason": "phase_f_abi_mismatch",
            "requested_count": requested_count,
            "requested_bytes": requested_bytes,
            "success_count": 0,
            "failure_count": requested_count,
            "rm_status": 0,
            "ioctl_ret": -2,
        }

    try:
        gpu_uuid = _phase_f_gpu_uuid(int(device))
    except Exception:
        logger.debug("Failed to resolve CUDA UUID for mapping rebuild", exc_info=True)
        return {
            "ok": False,
            "reason": "gpu_uuid_failed",
            "requested_count": requested_count,
            "requested_bytes": requested_bytes,
            "success_count": 0,
            "failure_count": requested_count,
            "rm_status": 0,
            "ioctl_ret": -3,
        }

    def run_candidate(
        fd: int, *, source_fd: int, candidate_index: int
    ) -> dict[str, object]:
        success_count = 0
        failure_count = 0
        local_gpu_bytes = 0
        remote_cpu_bytes = 0
        read_mapped_bytes = 0
        unknown_bytes = 0
        last_rm_status = 0
        last_ioctl_ret = 0
        records: list[dict[str, int]] = []

        for start in range(0, requested_count, _UVM_PHASE_F_MAX_RANGES):
            chunk = normalized[start : start + _UVM_PHASE_F_MAX_RANGES]
            params = _PhaseFBatchExecutorParams()
            params.action = UVM_PHASE_F_ACTION_REBUILD_GPU_READ_MAPPINGS
            params.count = len(chunk)
            params.gpu_uuid = gpu_uuid
            params.async_ = 0
            for index, (base, length) in enumerate(chunk):
                params.ranges[index].base = base
                params.ranges[index].length = length

            ctypes.set_errno(0)
            last_ioctl_ret = int(
                _libc_ioctl(fd, _UVM_PHASE_F_BATCH_EXECUTOR_IOCTL, params)
            )
            last_rm_status = int(params.rmStatus)
            success_count += int(params.success_count)
            failure_count += int(params.failure_count)

            for index, (base, length) in enumerate(chunk):
                raw = params.results[index]
                record = {
                    "base": int(base),
                    "length": int(length),
                    "status": int(raw.status),
                    # Action 23 result semantics are documented in uvm.c.
                    "local_gpu_bytes": int(raw.resident_gpu_bytes),
                    "remote_cpu_bytes": int(raw.resident_cpu_bytes),
                    "read_mapped_bytes": int(raw.resident_any_bytes),
                    "unknown_bytes": int(raw.resident_unknown_bytes),
                }
                records.append(record)
                local_gpu_bytes += int(raw.resident_gpu_bytes)
                remote_cpu_bytes += int(raw.resident_cpu_bytes)
                read_mapped_bytes += int(raw.resident_any_bytes)
                unknown_bytes += int(raw.resident_unknown_bytes)

            if last_ioctl_ret != 0 or last_rm_status != 0:
                break

        complete = (
            last_ioctl_ret == 0
            and last_rm_status == 0
            and success_count == requested_count
            and failure_count == 0
            and local_gpu_bytes + remote_cpu_bytes == requested_bytes
            and read_mapped_bytes == requested_bytes
            and unknown_bytes == 0
            and len(records) == requested_count
            and all(int(record["status"]) == 0 for record in records)
        )
        made_progress = any(
            (
                success_count > 0,
                local_gpu_bytes > 0,
                remote_cpu_bytes > 0,
                read_mapped_bytes > 0,
            )
        )
        fd_mismatch_proven = (
            not made_progress
            and last_ioctl_ret == 0
            and last_rm_status == _NV_ERR_INVALID_DEVICE
            and success_count == 0
            and failure_count > 0
            and bool(records)
            and all(
                int(record["status"]) == _NV_ERR_INVALID_DEVICE
                and int(record["local_gpu_bytes"]) == 0
                and int(record["remote_cpu_bytes"]) == 0
                and int(record["read_mapped_bytes"]) == 0
                and int(record["unknown_bytes"]) == int(record["length"])
                for record in records
            )
        )
        return {
            "ok": bool(complete),
            "reason": "ok" if complete else "driver_rebuild_failed",
            "requested_count": requested_count,
            "requested_bytes": requested_bytes,
            "success_count": int(success_count),
            "failure_count": int(failure_count),
            "local_gpu_bytes": int(local_gpu_bytes),
            "remote_cpu_bytes": int(remote_cpu_bytes),
            "read_mapped_bytes": int(read_mapped_bytes),
            "unknown_bytes": int(unknown_bytes),
            "rm_status": int(last_rm_status),
            "ioctl_ret": int(last_ioctl_ret),
            "errno": int(ctypes.get_errno()) if last_ioctl_ret != 0 else 0,
            "results": tuple(records),
            "made_progress": bool(made_progress),
            "fd_mismatch_proven": bool(fd_mismatch_proven),
            "fd_candidate_index": int(candidate_index),
            "selected_fd_source": int(source_fd),
        }

    last_candidate: dict[str, object] | None = None
    try:
        for candidate_index, source_fd in enumerate(
            _iter_existing_uvm_fd_sources(), start=1
        ):
            fd = _initialize_duplicated_uvm_fd(source_fd)
            if fd is None:
                continue
            try:
                candidate = run_candidate(
                    fd,
                    source_fd=int(source_fd),
                    candidate_index=candidate_index,
                )
            finally:
                with suppress(OSError):
                    os.close(fd)
            if (
                bool(candidate["ok"])
                or bool(candidate["made_progress"])
                or not bool(candidate["fd_mismatch_proven"])
            ):
                return candidate
            last_candidate = candidate

        fd = _open_initialized_uvm_fd()
        try:
            fallback = run_candidate(
                fd,
                source_fd=-1,
                candidate_index=(
                    int((last_candidate or {}).get("fd_candidate_index", 0))
                    + 1
                ),
            )
        finally:
            with suppress(OSError):
                os.close(fd)
        if (
            not bool(fallback["ok"])
            and not bool(fallback["made_progress"])
            and bool(fallback["fd_mismatch_proven"])
        ):
            fallback["reason"] = "no_matching_uvm_fd"
        return fallback
    except Exception as exc:
        logger.debug("GPU READ mapping rebuild failed", exc_info=True)
        return {
            "ok": False,
            "reason": repr(exc),
            "requested_count": requested_count,
            "requested_bytes": requested_bytes,
            "success_count": 0,
            "failure_count": requested_count,
            "rm_status": 0,
            "ioctl_ret": -4,
        }


def driver_cutover_ranges_to_immutable_cpu_shadow(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    device: int = 0,
    force_synchronous_migration: bool = False,
) -> dict[str, object]:
    """Drop GPU read-duplicates without copying them back to CPU.

    Every submitted managed range must already have matching read-only CPU and
    GPU residency. The modified driver rejects the whole range unless it can
    retain the CPU copy, free all corresponding GPU chunks, and install
    read-only GPU-to-CPU mappings. This action is always synchronous.
    """
    range_list = [
        (int(ptr), int(size))
        for ptr, size in ranges
        if int(ptr) > 0 and int(size) > 0
    ]
    requested_bytes = sum(size for _, size in range_list)
    action_name = (
        "sync_cpu_remote_cutover"
        if force_synchronous_migration
        else "immutable_cpu_shadow_cutover"
    )
    action_code = (
        UVM_PHASE_F_ACTION_SYNC_CPU_REMOTE_CUTOVER
        if force_synchronous_migration
        else UVM_PHASE_F_ACTION_IMMUTABLE_CPU_SHADOW_CUTOVER
    )
    empty_result: dict[str, object] = {
        "ok": True,
        "action": action_name,
        "range_count": 0,
        "success_count": 0,
        "failure_count": 0,
        "requested_bytes": 0,
        "released_gpu_bytes": 0,
        "retained_cpu_shadow_bytes": 0,
        "remote_mapped_bytes": 0,
        "rm_status": 0,
        "ioctl_ret": 0,
        "results": (),
    }
    if not range_list:
        return empty_result
    if not _phase_f_batch_abi_matches_kernel():
        return {
            **empty_result,
            "ok": False,
            "range_count": len(range_list),
            "failure_count": len(range_list),
            "requested_bytes": int(requested_bytes),
            "ioctl_ret": -2,
            "reason": "phase_f_abi_mismatch",
        }

    try:
        gpu_uuid = _phase_f_gpu_uuid(device)
    except Exception:
        logger.debug("Failed to resolve CUDA UUID for shadow cutover", exc_info=True)
        return {
            **empty_result,
            "ok": False,
            "range_count": len(range_list),
            "failure_count": len(range_list),
            "requested_bytes": int(requested_bytes),
            "ioctl_ret": -3,
            "reason": "gpu_uuid_failed",
        }

    def run_candidate(
        fd: int, *, source_fd: int, candidate_index: int
    ) -> dict[str, object]:
        success_count = 0
        failure_count = 0
        released_gpu_bytes = 0
        retained_cpu_shadow_bytes = 0
        remote_mapped_bytes = 0
        last_rm_status = 0
        last_ioctl_ret = 0
        records: list[dict[str, object]] = []
        for start in range(0, len(range_list), _UVM_PHASE_F_MAX_RANGES):
            chunk = range_list[start : start + _UVM_PHASE_F_MAX_RANGES]
            params = _PhaseFBatchExecutorParams()
            params.action = action_code
            params.count = len(chunk)
            params.gpu_uuid = gpu_uuid
            params.async_ = 0
            for index, (ptr, size) in enumerate(chunk):
                params.ranges[index].base = int(ptr)
                params.ranges[index].length = int(size)

            ret = int(_libc_ioctl(fd, _UVM_PHASE_F_BATCH_EXECUTOR_IOCTL, params))
            last_ioctl_ret = ret
            last_rm_status = int(params.rmStatus)
            success_count += int(params.success_count)
            failure_count += int(params.failure_count)
            for index, (ptr, size) in enumerate(chunk):
                raw = params.results[index]
                record = {
                    "base": int(ptr),
                    "length": int(size),
                    "status": int(raw.status),
                    # Action 22 deliberately assigns action-specific meanings
                    # to the three byte counters; see uvm.c.
                    "released_gpu_bytes": int(raw.resident_gpu_bytes),
                    "retained_cpu_shadow_bytes": int(raw.resident_cpu_bytes),
                    "remote_mapped_bytes": int(raw.resident_any_bytes),
                    "unknown_bytes": int(raw.resident_unknown_bytes),
                }
                records.append(record)
                released_gpu_bytes += int(raw.resident_gpu_bytes)
                retained_cpu_shadow_bytes += int(raw.resident_cpu_bytes)
                remote_mapped_bytes += int(raw.resident_any_bytes)

            if ret != 0 or last_rm_status != 0:
                break

        complete = (
            last_ioctl_ret == 0
            and last_rm_status == 0
            and success_count == len(range_list)
            and failure_count == 0
            and released_gpu_bytes == requested_bytes
            and retained_cpu_shadow_bytes == requested_bytes
            and remote_mapped_bytes == requested_bytes
            and len(records) == len(range_list)
            and all(int(record["status"]) == 0 for record in records)
        )
        made_progress = any(
            (
                success_count > 0,
                released_gpu_bytes > 0,
                retained_cpu_shadow_bytes > 0,
                remote_mapped_bytes > 0,
            )
        )
        fd_mismatch_proven = (
            not made_progress
            and last_ioctl_ret == 0
            and last_rm_status == _NV_ERR_INVALID_DEVICE
            and success_count == 0
            and failure_count > 0
            and records
            and all(
                int(record["status"]) == _NV_ERR_INVALID_DEVICE
                and int(record["released_gpu_bytes"]) == 0
                and int(record["retained_cpu_shadow_bytes"]) == 0
                and int(record["remote_mapped_bytes"]) == 0
                and int(record["unknown_bytes"]) == int(record["length"])
                for record in records
            )
        )
        return {
            "ok": bool(complete),
            "action": action_name,
            "range_count": len(range_list),
            "success_count": int(success_count),
            "failure_count": int(failure_count),
            "requested_bytes": int(requested_bytes),
            "released_gpu_bytes": int(released_gpu_bytes),
            "retained_cpu_shadow_bytes": int(retained_cpu_shadow_bytes),
            "remote_mapped_bytes": int(remote_mapped_bytes),
            "rm_status": int(last_rm_status),
            "ioctl_ret": int(last_ioctl_ret),
            "results": tuple(records),
            "made_progress": bool(made_progress),
            "fd_mismatch_proven": bool(fd_mismatch_proven),
            "fd_candidate_index": int(candidate_index),
            "selected_fd_source": int(source_fd),
        }

    last_candidate: dict[str, object] | None = None
    try:
        for candidate_index, source_fd in enumerate(
            _iter_existing_uvm_fd_sources(), start=1
        ):
            fd = _initialize_duplicated_uvm_fd(source_fd)
            if fd is None:
                continue
            try:
                candidate = run_candidate(
                    fd,
                    source_fd=int(source_fd),
                    candidate_index=candidate_index,
                )
            finally:
                with suppress(OSError):
                    os.close(fd)
            if (
                bool(candidate["ok"])
                or bool(candidate["made_progress"])
                or not bool(candidate["fd_mismatch_proven"])
            ):
                return candidate
            last_candidate = candidate

        fd = _open_initialized_uvm_fd()
        try:
            fallback = run_candidate(
                fd,
                source_fd=-1,
                candidate_index=(
                    int((last_candidate or {}).get("fd_candidate_index", 0)) + 1
                ),
            )
        finally:
            with suppress(OSError):
                os.close(fd)
        if (
            not bool(fallback["ok"])
            and not bool(fallback["made_progress"])
            and bool(fallback["fd_mismatch_proven"])
        ):
            fallback["reason"] = "no_matching_uvm_fd"
        return fallback
    except Exception as exc:
        logger.debug("Immutable CPU shadow cutover failed", exc_info=True)
        return {
            **empty_result,
            "ok": False,
            "range_count": len(range_list),
            "failure_count": len(range_list),
            "requested_bytes": int(requested_bytes),
            "ioctl_ret": -4,
            "reason": repr(exc),
            "results": (),
        }


def _region_release_outcome(
    params: _ExpertOffloadReleaseParams,
    *,
    ret: int,
) -> tuple[bool, bool]:
    raw_ok = int(ret) == 0 and int(params.rmStatus) == 0
    remote_read = bool(int(params.flags) & UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ)
    activity = (
        int(params.block_count),
        int(params.attempted_blocks),
        int(params.success_blocks),
        int(params.failure_blocks),
        int(params.skipped_blocks),
        int(params.attempted_bytes),
        int(params.released_bytes),
        int(params.failed_bytes),
        int(params.remote_map_blocks),
        int(params.remote_map_failure_blocks),
        int(params.cpu_resident_bytes),
        int(params.remote_mapped_bytes),
        int(params.newly_remote_mapped_bytes),
    )
    idempotent = (
        raw_ok
        and not remote_read
        and int(params.length) > 0
        and int(params.first_error) == 0
        and not any(activity)
    )
    ok = raw_ok and (
        idempotent
        or int(params.block_count) > 0
        or int(params.attempted_blocks) > 0
        or int(params.skipped_blocks) > 0
        or int(params.released_bytes) > 0
    )
    if remote_read:
        ok = (
            ok
            and int(params.remote_map_failure_blocks) == 0
            and int(params.cpu_resident_bytes) >= int(params.length)
            and int(params.remote_mapped_bytes) >= int(params.length)
        )

    made_progress = (
        int(params.attempted_blocks) > 0
        or int(params.released_bytes) > 0
        or int(params.newly_remote_mapped_bytes) > 0
    )
    return ok, made_progress


def _region_release_audit(
    params: _ExpertOffloadReleaseParams,
    *,
    ret: int,
    ok: bool,
    made_progress: bool,
    errno_value: int = 0,
    reason: str = "ioctl_complete",
) -> dict[str, object]:
    """Materialize one release ioctl result without changing the default API."""
    return {
        "region_id": int(params.region_id),
        "ok": bool(ok),
        "made_progress": bool(made_progress),
        "reason": str(reason),
        "ioctl_ret": int(ret),
        "errno": int(errno_value),
        "rm_status": int(params.rmStatus),
        "flags": int(params.flags),
        "remote_read": bool(
            int(params.flags) & UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ
        ),
        "base": int(params.base),
        "length": int(params.length),
        "block_count": int(params.block_count),
        "attempted_blocks": int(params.attempted_blocks),
        "success_blocks": int(params.success_blocks),
        "failure_blocks": int(params.failure_blocks),
        "skipped_blocks": int(params.skipped_blocks),
        "attempted_bytes": int(params.attempted_bytes),
        "released_bytes": int(params.released_bytes),
        "failed_bytes": int(params.failed_bytes),
        "remote_map_blocks": int(params.remote_map_blocks),
        "remote_map_failure_blocks": int(params.remote_map_failure_blocks),
        "cpu_resident_bytes": int(params.cpu_resident_bytes),
        "remote_mapped_bytes": int(params.remote_mapped_bytes),
        "newly_remote_mapped_bytes": int(params.newly_remote_mapped_bytes),
        "first_failed_block": int(params.first_failed_block),
        "first_error": int(params.first_error),
    }


def driver_release_regions(
    region_ids: Sequence[int],
    *,
    device: int = 0,
    remote_read: bool = False,
    immutable_cpu_shadow: bool = False,
    prefer_existing: bool = True,
    include_audit: bool = False,
) -> dict[str, object]:
    """Release explicitly named regions at 64 KiB subchunk granularity.

    The managed VA ranges remain allocated. With ``remote_read=True``, the
    released ranges are also synchronously mapped read-only from the GPU to
    their CPU backing before the ioctl commits success.
    """
    normalized: list[int] = []
    for region_id in region_ids:
        try:
            value = int(region_id)
        except (TypeError, ValueError):
            continue
        if value > 0:
            normalized.append(value)

    def _failure_audits(reason: str) -> dict[int, dict[str, object]]:
        audits: dict[int, dict[str, object]] = {}
        for region_id in normalized:
            params = _ExpertOffloadReleaseParams()
            params.region_id = int(region_id)
            params.flags = int(flags)
            audits[region_id] = _region_release_audit(
                params,
                ret=-1,
                ok=False,
                made_progress=False,
                reason=reason,
            )
        return audits

    def _result(
        statuses: dict[int, bool],
        audits: dict[int, dict[str, object]] | None = None,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "ok": all(statuses.get(region_id, False) for region_id in normalized),
            "statuses": statuses,
        }
        if include_audit:
            result["audits"] = audits or {}
        return result

    if not normalized:
        return _result({})

    flags = UVM_EXPERT_OFFLOAD_RELEASE_FLAG_SUBCHUNK_64K
    if remote_read:
        flags |= UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ
    if immutable_cpu_shadow:
        flags |= UVM_EXPERT_OFFLOAD_RELEASE_FLAG_IMMUTABLE_CPU_SHADOW

    failure_statuses = {region_id: False for region_id in normalized}

    try:
        gpu_uuid = _phase_f_gpu_uuid(device)
    except Exception:
        logger.debug("Failed to resolve CUDA UUID for region release", exc_info=True)
        return _result(failure_statuses, _failure_audits("gpu_uuid_failed"))

    def _run_release_on_fd(
        fd: int,
    ) -> tuple[dict[int, bool], bool, dict[int, dict[str, object]]]:
        statuses: dict[int, bool] = {}
        audits: dict[int, dict[str, object]] = {}
        made_progress = False
        for region_id in normalized:
            params = _ExpertOffloadReleaseParams()
            params.region_id = int(region_id)
            params.gpu_uuid = gpu_uuid
            params.flags = int(flags)
            ret = _libc_ioctl(fd, _UVM_EXPERT_OFFLOAD_RELEASE_IOCTL, params)
            ok, progressed = _region_release_outcome(params, ret=int(ret))
            statuses[region_id] = ok
            made_progress |= progressed
            if include_audit:
                audits[region_id] = _region_release_audit(
                    params,
                    ret=int(ret),
                    ok=ok,
                    made_progress=progressed,
                    errno_value=(ctypes.get_errno() if int(ret) != 0 else 0),
                )
        return statuses, made_progress, audits

    try:
        if prefer_existing:
            for source_fd in _iter_existing_uvm_fd_sources():
                fd = _initialize_duplicated_uvm_fd(source_fd)
                if fd is None:
                    continue
                try:
                    statuses, made_progress, audits = _run_release_on_fd(fd)
                finally:
                    with suppress(OSError):
                        os.close(fd)
                result = _result(statuses, audits)
                if bool(result["ok"]) or made_progress:
                    return result

        fd = _open_initialized_uvm_fd()
        try:
            statuses, _, audits = _run_release_on_fd(fd)
        finally:
            with suppress(OSError):
                os.close(fd)
        return _result(statuses, audits)
    except Exception:
        logger.debug("Region release ioctl failed", exc_info=True)
        return _result(failure_statuses, _failure_audits("release_exception"))


def driver_prepare_ranges_remote_read_bg_deferred(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    device: int = 0,
) -> tuple[int, int, int, int]:
    """Prepare remote READ PTEs and mark background promotion pending."""
    return driver_phase_f_batch_execute_ranges(
        ranges,
        UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_BG_DEFERRED,
        device=device,
    )


def driver_prepare_ranges_remote_read(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    device: int = 0,
) -> tuple[int, int, int, int]:
    """Prepare remote READ PTEs without marking background promotion pending."""
    return driver_phase_f_batch_execute_ranges(
        ranges,
        UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ,
        device=device,
    )


def driver_prepare_ranges_remote_read_audited(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    device: int = 0,
) -> dict[str, object]:
    """Batch-install remote READ PTEs and verify every returned byte count.

    Unlike :func:`driver_prepare_ranges_remote_read`, this wrapper deliberately
    bypasses the C++ count-only helper so the caller can prove that every input
    page was CPU-resident, covered, and actually mapped.  Up to 64 ranges share
    one ioctl.  Existing initialized UVM file descriptions are probed until
    the VA-space which owns the ranges is found.  Only an
    ``NV_ERR_INVALID_DEVICE`` result with no observable progress may advance
    to another candidate.  Partial progress is reported as failure and stops
    candidate probing; callers must fail closed.
    """
    normalized = [
        (int(base), int(length))
        for base, length in ranges
        if int(base) > 0 and int(length) > 0
    ]
    requested_count = len(normalized)
    requested_bytes = sum(length for _base, length in normalized)

    def result(
        *,
        ok: bool,
        reason: str,
        success_count: int = 0,
        failure_count: int = 0,
        ioctl_calls: int = 0,
        cpu_resident_bytes: int = 0,
        remote_mapped_bytes: int = 0,
        covered_bytes: int = 0,
        unknown_bytes: int = 0,
        first_failed_index: int = -1,
        ioctl_ret: int = 0,
        rm_status: int = 0,
        errno_value: int = 0,
        fd_candidate_attempts: int = 0,
        fd_candidate_ioctl_calls: int = 0,
        selected_fd_source: int = -1,
        selected_fd_kind: str = "",
        made_progress: bool = False,
        fd_mismatch_proven: bool = False,
    ) -> dict[str, object]:
        return {
            "ok": bool(ok),
            "reason": reason,
            "requested_count": requested_count,
            "requested_bytes": requested_bytes,
            "success_count": int(success_count),
            "failure_count": int(failure_count),
            "ioctl_calls": int(ioctl_calls),
            "cpu_resident_bytes": int(cpu_resident_bytes),
            "remote_mapped_bytes": int(remote_mapped_bytes),
            "covered_bytes": int(covered_bytes),
            "unknown_bytes": int(unknown_bytes),
            "first_failed_index": int(first_failed_index),
            "ioctl_ret": int(ioctl_ret),
            "rm_status": int(rm_status),
            "errno": int(errno_value),
            "fd_candidate_attempts": int(fd_candidate_attempts),
            "fd_candidate_ioctl_calls": int(fd_candidate_ioctl_calls),
            "selected_fd_source": int(selected_fd_source),
            "selected_fd_kind": str(selected_fd_kind),
            "made_progress": bool(made_progress),
            "fd_mismatch_proven": bool(fd_mismatch_proven),
        }

    if not normalized:
        return result(ok=False, reason="no_ranges")
    try:
        gpu_uuid = _phase_f_gpu_uuid(int(device))
    except Exception:
        logger.debug("Failed to resolve CUDA UUID for audited remote map", exc_info=True)
        return result(
            ok=False,
            reason="gpu_uuid_failed",
            failure_count=requested_count,
        )

    total_ioctl_calls = 0
    candidate_attempts = 0

    def run_candidate(
        fd: int,
        *,
        source_fd: int,
        source_kind: str,
    ) -> dict[str, object]:
        nonlocal total_ioctl_calls
        success_count = 0
        failure_count = 0
        cpu_resident_bytes = 0
        remote_mapped_bytes = 0
        covered_bytes = 0
        unknown_bytes = 0
        made_progress = False
        fd_mismatch_proven = False

        def candidate_result(**kwargs: object) -> dict[str, object]:
            return result(
                fd_candidate_attempts=candidate_attempts,
                fd_candidate_ioctl_calls=total_ioctl_calls,
                selected_fd_source=source_fd,
                selected_fd_kind=source_kind,
                made_progress=made_progress,
                fd_mismatch_proven=fd_mismatch_proven,
                **kwargs,
            )

        try:
            for start in range(0, requested_count, _UVM_PHASE_F_MAX_RANGES):
                chunk = normalized[start : start + _UVM_PHASE_F_MAX_RANGES]
                params = _PhaseFBatchExecutorParams()
                # Action 21 audits the final remote mapping state and is
                # intentionally idempotent.  This matters after cmd99 restores an
                # already-present CPU-backed READ PTE: action 5 reports only newly
                # installed pages and would incorrectly return zero mapped bytes.
                params.action = UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_AUDITED
                params.count = len(chunk)
                params.gpu_uuid = gpu_uuid
                params.async_ = 0
                for offset, (base, length) in enumerate(chunk):
                    params.ranges[offset].base = base
                    params.ranges[offset].length = length

                ctypes.set_errno(0)
                ret = int(_libc_ioctl(fd, _UVM_PHASE_F_BATCH_EXECUTOR_IOCTL, params))
                total_ioctl_calls += 1
                rm_status = int(params.rmStatus)
                chunk_success = int(params.success_count)
                chunk_failure = int(params.failure_count)
                chunk_cpu = sum(
                    int(params.results[offset].resident_cpu_bytes)
                    for offset in range(len(chunk))
                )
                chunk_mapped = sum(
                    int(params.results[offset].resident_gpu_bytes)
                    for offset in range(len(chunk))
                )
                chunk_covered = sum(
                    int(params.results[offset].resident_any_bytes)
                    for offset in range(len(chunk))
                )
                chunk_unknown = sum(
                    int(params.results[offset].resident_unknown_bytes)
                    for offset in range(len(chunk))
                )
                cpu_resident_bytes += chunk_cpu
                remote_mapped_bytes += chunk_mapped
                covered_bytes += chunk_covered
                unknown_bytes += chunk_unknown
                made_progress = made_progress or any(
                    (
                        chunk_success > 0,
                        chunk_cpu > 0,
                        chunk_mapped > 0,
                        chunk_covered > 0,
                    )
                )

                if ret != 0 or rm_status != 0:
                    fd_mismatch_proven = (
                        ret == 0
                        and rm_status == _NV_ERR_INVALID_DEVICE
                        and chunk_success == 0
                        and chunk_failure == len(chunk)
                        and all(
                            int(params.results[offset].status)
                            == _NV_ERR_INVALID_DEVICE
                            and int(params.results[offset].queried_bytes)
                            == int(chunk[offset][1])
                            and int(params.results[offset].resident_cpu_bytes) == 0
                            and int(params.results[offset].resident_gpu_bytes) == 0
                            and int(params.results[offset].resident_any_bytes) == 0
                            and int(params.results[offset].resident_unknown_bytes)
                            == int(chunk[offset][1])
                            for offset in range(len(chunk))
                        )
                    )
                    success_count += chunk_success
                    failure_count += max(
                        chunk_failure,
                        len(chunk) - chunk_success,
                        1,
                    )
                    return candidate_result(
                        ok=False,
                        reason="batch_ioctl_failed",
                        success_count=success_count,
                        failure_count=failure_count,
                        ioctl_calls=total_ioctl_calls,
                        cpu_resident_bytes=cpu_resident_bytes,
                        remote_mapped_bytes=remote_mapped_bytes,
                        covered_bytes=covered_bytes,
                        unknown_bytes=unknown_bytes,
                        first_failed_index=start + min(chunk_success, len(chunk) - 1),
                        ioctl_ret=ret,
                        rm_status=rm_status,
                        errno_value=ctypes.get_errno() if ret != 0 else 0,
                    )

                if chunk_success != len(chunk) or chunk_failure != 0:
                    success_count += chunk_success
                    failure_count += max(
                        chunk_failure,
                        len(chunk) - chunk_success,
                        1,
                    )
                    return candidate_result(
                        ok=False,
                        reason="batch_count_mismatch",
                        success_count=success_count,
                        failure_count=failure_count,
                        ioctl_calls=total_ioctl_calls,
                        cpu_resident_bytes=cpu_resident_bytes,
                        remote_mapped_bytes=remote_mapped_bytes,
                        covered_bytes=covered_bytes,
                        unknown_bytes=unknown_bytes,
                        first_failed_index=start + min(chunk_success, len(chunk) - 1),
                        rm_status=rm_status,
                    )

                for offset, (_base, length) in enumerate(chunk):
                    item = params.results[offset]
                    item_cpu = int(item.resident_cpu_bytes)
                    item_mapped = int(item.resident_gpu_bytes)
                    item_covered = int(item.resident_any_bytes)
                    item_unknown = int(item.resident_unknown_bytes)
                    valid = (
                        int(item.status) == 0
                        and int(item.queried_bytes) == length
                        and item_cpu >= length
                        and item_mapped >= length
                        and item_covered >= length
                        and item_unknown == 0
                    )
                    if not valid:
                        success_count += offset
                        failure_count += 1
                        return candidate_result(
                            ok=False,
                            reason="range_audit_failed",
                            success_count=success_count,
                            failure_count=failure_count,
                            ioctl_calls=total_ioctl_calls,
                            cpu_resident_bytes=cpu_resident_bytes,
                            remote_mapped_bytes=remote_mapped_bytes,
                            covered_bytes=covered_bytes,
                            unknown_bytes=unknown_bytes,
                            first_failed_index=start + offset,
                            rm_status=int(item.status) or rm_status,
                        )
                success_count += len(chunk)

            complete = (
                success_count == requested_count
                and failure_count == 0
                and cpu_resident_bytes >= requested_bytes
                and remote_mapped_bytes >= requested_bytes
                and covered_bytes >= requested_bytes
                and unknown_bytes == 0
            )
            return candidate_result(
                ok=complete,
                reason="ok" if complete else "aggregate_audit_failed",
                success_count=success_count,
                failure_count=failure_count,
                ioctl_calls=total_ioctl_calls,
                cpu_resident_bytes=cpu_resident_bytes,
                remote_mapped_bytes=remote_mapped_bytes,
                covered_bytes=covered_bytes,
                unknown_bytes=unknown_bytes,
            )
        except Exception as exc:
            logger.debug("Audited remote-read batch failed", exc_info=True)
            return candidate_result(
                ok=False,
                reason="batch_exception",
                success_count=success_count,
                failure_count=max(failure_count, requested_count - success_count),
                ioctl_calls=total_ioctl_calls,
                cpu_resident_bytes=cpu_resident_bytes,
                remote_mapped_bytes=remote_mapped_bytes,
                covered_bytes=covered_bytes,
                unknown_bytes=unknown_bytes,
                first_failed_index=success_count,
                errno_value=getattr(exc, "errno", 0) or 0,
            )

    last_candidate: dict[str, object] | None = None
    for source_fd in _iter_existing_uvm_fd_sources():
        fd = _initialize_duplicated_uvm_fd(source_fd)
        if fd is None:
            continue
        candidate_attempts += 1
        try:
            candidate = run_candidate(
                fd,
                source_fd=int(source_fd),
                source_kind="existing",
            )
        finally:
            with suppress(OSError):
                os.close(fd)

        if bool(candidate["ok"]) or bool(candidate["made_progress"]):
            return candidate
        if not bool(candidate["fd_mismatch_proven"]):
            return candidate
        last_candidate = candidate

    # Preserve the legacy fallback only after every existing file description
    # was proven unusable without modifying or auditing any target range.
    try:
        fd = _open_initialized_uvm_fd()
    except Exception as exc:
        logger.debug("Failed to open fallback UVM fd for remote audit", exc_info=True)
        if last_candidate is not None:
            return {
                **last_candidate,
                "reason": "no_matching_existing_uvm_fd",
                "fd_candidate_attempts": candidate_attempts,
                "fd_candidate_ioctl_calls": total_ioctl_calls,
            }
        return result(
            ok=False,
            reason="no_initializable_uvm_fd",
            failure_count=requested_count,
            ioctl_calls=total_ioctl_calls,
            errno_value=getattr(exc, "errno", 0) or 0,
            fd_candidate_attempts=candidate_attempts,
            fd_candidate_ioctl_calls=total_ioctl_calls,
        )

    candidate_attempts += 1
    try:
        fallback = run_candidate(fd, source_fd=-1, source_kind="new")
    finally:
        with suppress(OSError):
            os.close(fd)
    if (
        not bool(fallback["ok"])
        and not bool(fallback["made_progress"])
        and bool(fallback["fd_mismatch_proven"])
    ):
        return {**fallback, "reason": "no_matching_uvm_fd"}
    return fallback


def driver_prepare_regions_remote_read_audited(
    regions: Sequence[Sequence[int]],
    *,
    device: int = 0,
) -> dict[str, object]:
    """One-shot remote-map registered, freshly-unmapped CPU regions.

    Each input is ``(numeric_region_id, base, length)``.  The geometry is
    validated in userspace, while ioctl 84 resolves the registered id on the
    owning UVM file description.  This avoids applying a raw VA to an
    unrelated CUDA VA-space when a process owns more than one UVM fd.

    Action 21 reports final CPU residency and final CPU-backed remote READ PTE
    coverage after installing any missing PTEs.  It is used only after P3
    close has proved that no slot mapping or slot backing remains reachable;
    that close proof makes an already-present READ PTE safe to accept.
    """
    raw_regions = list(regions)
    requested_count = len(raw_regions)
    requested_bytes = 0
    alignment = 64 * 1024

    def result(
        *,
        ok: bool,
        reason: str,
        success_count: int = 0,
        failure_count: int = 0,
        ioctl_calls: int = 0,
        mapping_ioctl_calls: int = 0,
        fd_candidate_attempts: int = 0,
        cpu_resident_bytes: int = 0,
        remote_mapped_bytes: int = 0,
        covered_bytes: int = 0,
        unknown_bytes: int = 0,
        first_failed_index: int = -1,
        ioctl_ret: int = 0,
        rm_status: int = 0,
        errno_value: int = 0,
        made_progress: bool = False,
    ) -> dict[str, object]:
        return {
            "ok": bool(ok),
            "reason": str(reason),
            "requested_count": requested_count,
            "requested_bytes": requested_bytes,
            "success_count": int(success_count),
            "failure_count": int(failure_count),
            "ioctl_calls": int(ioctl_calls),
            "mapping_ioctl_calls": int(mapping_ioctl_calls),
            "fd_candidate_attempts": int(fd_candidate_attempts),
            "cpu_resident_bytes": int(cpu_resident_bytes),
            "remote_mapped_bytes": int(remote_mapped_bytes),
            "covered_bytes": int(covered_bytes),
            "unknown_bytes": int(unknown_bytes),
            "first_failed_index": int(first_failed_index),
            "ioctl_ret": int(ioctl_ret),
            "rm_status": int(rm_status),
            "errno": int(errno_value),
            "made_progress": bool(made_progress),
        }

    if not raw_regions:
        return result(ok=False, reason="no_regions")
    if not _phase_f_batch_abi_matches_kernel():
        return result(
            ok=False,
            reason="phase_f_ctypes_abi_mismatch",
            failure_count=requested_count,
        )

    normalized: list[tuple[int, int, int]] = []
    seen_ids: set[int] = set()
    for index, raw in enumerate(raw_regions):
        if not isinstance(raw, (tuple, list)) or len(raw) != 3:
            return result(
                ok=False,
                reason="invalid_region_tuple",
                failure_count=requested_count,
                first_failed_index=index,
            )
        if any(type(value) is not int for value in raw):
            return result(
                ok=False,
                reason="invalid_region_scalar",
                failure_count=requested_count,
                first_failed_index=index,
            )
        region_id, base, length = raw
        if (
            region_id <= 0
            or region_id > _UVM_MOE_SLOT_UINT64_MAX
            or base <= 0
            or base > _UVM_MOE_SLOT_UINT64_MAX
            or length <= 0
            or length > _UVM_MOE_SLOT_UINT64_MAX
            or base % alignment != 0
            or length % alignment != 0
            or base + length > _UVM_MOE_SLOT_UINT64_MAX + 1
        ):
            return result(
                ok=False,
                reason="invalid_region_geometry",
                failure_count=requested_count,
                first_failed_index=index,
            )
        if region_id in seen_ids:
            return result(
                ok=False,
                reason="duplicate_region_id",
                failure_count=requested_count,
                first_failed_index=index,
            )
        seen_ids.add(region_id)
        normalized.append((region_id, base, length))
        requested_bytes += length

    ordered = sorted(
        ((base, base + length, index) for index, (_id, base, length) in enumerate(normalized))
    )
    previous_end = 0
    for base, end, index in ordered:
        if base < previous_end:
            return result(
                ok=False,
                reason="overlapping_regions",
                failure_count=requested_count,
                first_failed_index=index,
            )
        previous_end = end

    try:
        gpu_uuid = _phase_f_gpu_uuid(int(device))
    except Exception:
        logger.debug("Failed to resolve CUDA UUID for registered remote map", exc_info=True)
        return result(
            ok=False,
            reason="gpu_uuid_failed",
            failure_count=requested_count,
        )

    sources = _iter_existing_uvm_fd_sources()
    if not sources:
        return result(
            ok=False,
            reason="no_existing_uvm_fd",
            failure_count=requested_count,
        )

    total_ioctl_calls = 0
    candidate_attempts = 0
    last_candidate: dict[str, object] | None = None
    for source_fd in sources:
        fd = _initialize_duplicated_uvm_fd(source_fd)
        if fd is None:
            continue
        candidate_attempts += 1
        success_count = 0
        failure_count = 0
        mapping_ioctl_calls = 0
        cpu_resident_bytes = 0
        remote_mapped_bytes = 0
        covered_bytes = 0
        unknown_bytes = 0
        made_progress = False
        candidate: dict[str, object] | None = None
        try:
            for start in range(0, requested_count, _UVM_PHASE_F_MAX_RANGES):
                chunk = normalized[start : start + _UVM_PHASE_F_MAX_RANGES]
                params = _PhaseFBatchRegionsParams()
                params.action = UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_AUDITED
                params.count = len(chunk)
                params.gpu_uuid = gpu_uuid
                params.async_ = 0
                for offset, (region_id, _base, _length) in enumerate(chunk):
                    params.region_ids[offset] = region_id

                ctypes.set_errno(0)
                ret = int(
                    _libc_ioctl(fd, _UVM_PHASE_F_BATCH_REGIONS_IOCTL, params)
                )
                mapping_ioctl_calls += 1
                total_ioctl_calls += 1
                rm_status = int(params.rmStatus)
                chunk_success = int(params.success_count)
                chunk_failure = int(params.failure_count)
                chunk_mapped = sum(
                    int(params.results[offset].resident_gpu_bytes)
                    for offset in range(len(chunk))
                )
                made_progress = made_progress or chunk_success > 0 or chunk_mapped > 0

                if ret != 0 or rm_status != 0:
                    success_count += chunk_success
                    failure_count += max(chunk_failure, len(chunk) - chunk_success, 1)
                    candidate = result(
                        ok=False,
                        reason="batch_ioctl_failed",
                        success_count=success_count,
                        failure_count=failure_count,
                        ioctl_calls=total_ioctl_calls,
                        mapping_ioctl_calls=mapping_ioctl_calls,
                        fd_candidate_attempts=candidate_attempts,
                        cpu_resident_bytes=cpu_resident_bytes,
                        remote_mapped_bytes=remote_mapped_bytes + chunk_mapped,
                        covered_bytes=covered_bytes,
                        unknown_bytes=unknown_bytes,
                        first_failed_index=start + chunk_success,
                        ioctl_ret=ret,
                        rm_status=rm_status,
                        errno_value=ctypes.get_errno() if ret != 0 else 0,
                        made_progress=made_progress,
                    )
                    break

                if chunk_success != len(chunk) or chunk_failure != 0:
                    success_count += chunk_success
                    failure_count += max(chunk_failure, len(chunk) - chunk_success, 1)
                    candidate = result(
                        ok=False,
                        reason="batch_count_mismatch",
                        success_count=success_count,
                        failure_count=failure_count,
                        ioctl_calls=total_ioctl_calls,
                        mapping_ioctl_calls=mapping_ioctl_calls,
                        fd_candidate_attempts=candidate_attempts,
                        cpu_resident_bytes=cpu_resident_bytes,
                        remote_mapped_bytes=remote_mapped_bytes + chunk_mapped,
                        covered_bytes=covered_bytes,
                        unknown_bytes=unknown_bytes,
                        first_failed_index=start + chunk_success,
                        rm_status=rm_status,
                        made_progress=made_progress,
                    )
                    break

                failed_offset = -1
                for offset, (_region_id, _base, length) in enumerate(chunk):
                    item = params.results[offset]
                    item_cpu = int(item.resident_cpu_bytes)
                    item_mapped = int(item.resident_gpu_bytes)
                    item_covered = int(item.resident_any_bytes)
                    item_unknown = int(item.resident_unknown_bytes)
                    cpu_resident_bytes += item_cpu
                    remote_mapped_bytes += item_mapped
                    covered_bytes += item_covered
                    unknown_bytes += item_unknown
                    if not (
                        int(item.status) == 0
                        and int(item.resident_on_count) == 0
                        and int(item.resident_on_cpu) == 0
                        and int(item.resident_on_gpu) == 0
                        and int(item.queried_bytes) == length
                        and item_cpu == length
                        and item_mapped == length
                        and item_covered == length
                        and item_unknown == 0
                    ):
                        failed_offset = offset
                        break
                if failed_offset >= 0:
                    success_count += failed_offset
                    failure_count += 1
                    made_progress = made_progress or remote_mapped_bytes > 0
                    candidate = result(
                        ok=False,
                        reason="range_audit_failed",
                        success_count=success_count,
                        failure_count=failure_count,
                        ioctl_calls=total_ioctl_calls,
                        mapping_ioctl_calls=mapping_ioctl_calls,
                        fd_candidate_attempts=candidate_attempts,
                        cpu_resident_bytes=cpu_resident_bytes,
                        remote_mapped_bytes=remote_mapped_bytes,
                        covered_bytes=covered_bytes,
                        unknown_bytes=unknown_bytes,
                        first_failed_index=start + failed_offset,
                        rm_status=int(params.results[failed_offset].status),
                        made_progress=made_progress,
                    )
                    break
                success_count += len(chunk)

            if candidate is None:
                complete = (
                    success_count == requested_count
                    and failure_count == 0
                    and cpu_resident_bytes == requested_bytes
                    and remote_mapped_bytes == requested_bytes
                    and covered_bytes == requested_bytes
                    and unknown_bytes == 0
                )
                candidate = result(
                    ok=complete,
                    reason="ok" if complete else "aggregate_audit_failed",
                    success_count=success_count,
                    failure_count=failure_count,
                    ioctl_calls=total_ioctl_calls,
                    mapping_ioctl_calls=mapping_ioctl_calls,
                    fd_candidate_attempts=candidate_attempts,
                    cpu_resident_bytes=cpu_resident_bytes,
                    remote_mapped_bytes=remote_mapped_bytes,
                    covered_bytes=covered_bytes,
                    unknown_bytes=unknown_bytes,
                    made_progress=made_progress,
                )
        except Exception as exc:
            logger.debug("Registered remote-read batch failed", exc_info=True)
            candidate = result(
                ok=False,
                reason="batch_exception",
                success_count=success_count,
                failure_count=max(failure_count, requested_count - success_count),
                ioctl_calls=total_ioctl_calls,
                mapping_ioctl_calls=mapping_ioctl_calls,
                fd_candidate_attempts=candidate_attempts,
                cpu_resident_bytes=cpu_resident_bytes,
                remote_mapped_bytes=remote_mapped_bytes,
                covered_bytes=covered_bytes,
                unknown_bytes=unknown_bytes,
                first_failed_index=success_count,
                errno_value=getattr(exc, "errno", 0) or 0,
                made_progress=made_progress,
            )
        finally:
            with suppress(OSError):
                os.close(fd)

        if bool(candidate.get("ok")) or bool(candidate.get("made_progress")):
            return candidate
        last_candidate = candidate

    if last_candidate is not None:
        return {
            **last_candidate,
            "reason": "no_matching_existing_uvm_fd",
            "ioctl_calls": total_ioctl_calls,
            "fd_candidate_attempts": candidate_attempts,
            "failure_count": requested_count,
        }
    return result(
        ok=False,
        reason="no_initializable_existing_uvm_fd",
        failure_count=requested_count,
        ioctl_calls=total_ioctl_calls,
        fd_candidate_attempts=candidate_attempts,
    )


def driver_start_ranges_bg_promotion(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    device: int = 0,
) -> tuple[int, int, int, int]:
    return driver_phase_f_batch_execute_ranges(
        ranges,
        UVM_PHASE_F_ACTION_START_BG_PROMOTION,
        device=device,
    )


def driver_pause_ranges_bg_promotion(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    device: int = 0,
) -> tuple[int, int, int, int]:
    return driver_phase_f_batch_execute_ranges(
        ranges,
        UVM_PHASE_F_ACTION_PAUSE_BG_PROMOTION,
        device=device,
    )


def driver_resume_ranges_bg_promotion(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    device: int = 0,
) -> tuple[int, int, int, int]:
    return driver_phase_f_batch_execute_ranges(
        ranges,
        UVM_PHASE_F_ACTION_RESUME_BG_PROMOTION,
        device=device,
    )


def uvm_residency_tag_only_enabled() -> bool:
    raw = os.environ.get("VLLM_UVM_RESIDENCY_TAG_ONLY")
    return raw is not None and raw.strip().lower() in ("1", "true", "yes", "on")


def batch_prefetch_ranges_to_device(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    device: int = 0,
    *,
    allow_tag_only: bool = False,
) -> int:
    """Batch prefetch raw UVM ranges to a GPU or CPU device.

    Returns the number of CUDA prefetch calls accepted by the allocator shim.
    """
    if uvm_residency_tag_only_enabled() and not allow_tag_only:
        return 0
    if _uvm_lib is None:
        return 0
    ptr_array, size_array, count = _prepare_uvm_range_arrays(ranges)
    if count <= 0:
        return 0
    stream = torch.cuda.current_stream().cuda_stream
    return int(
        _uvm_lib.uvm_batch_prefetch(
            ptr_array,
            size_array,
            ctypes.c_size_t(count),
            ctypes.c_int(device),
            ctypes.c_void_p(stream),
        )
    )


def batch_prefetch_ranges_to_cpu(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    allow_tag_only: bool = False,
) -> int:
    """Batch prefetch raw UVM ranges to CPU memory."""
    return batch_prefetch_ranges_to_device(
        ranges,
        device=-1,
        allow_tag_only=allow_tag_only,
    )


def advise_range_preferred_location(ptr: int, size: int, device: int) -> bool:
    """
    Set cudaMemAdviseSetPreferredLocation for a raw UVM address range.

    Returns True when the advise call was issued. The allocator shim currently
    logs CUDA errors internally instead of returning them, so benchmark health
    and Stage I trace records are used for end-to-end validation.
    """
    if uvm_residency_tag_only_enabled():
        return False
    if _uvm_lib is None:
        return False
    if ptr <= 0 or size <= 0:
        return False
    _uvm_lib.uvm_advise(
        ctypes.c_void_p(ptr),
        ctypes.c_size_t(size),
        ctypes.c_int(2),  # cudaMemAdviseSetPreferredLocation
        ctypes.c_int(device),
    )
    return True


def batch_advise_ranges_preferred_location(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    device: int,
) -> int:
    """Batch set cudaMemAdviseSetPreferredLocation for raw UVM ranges."""
    if uvm_residency_tag_only_enabled():
        return 0
    if _uvm_lib is None:
        return 0
    ptr_array, size_array, count = _prepare_uvm_range_arrays(ranges)
    if count <= 0:
        return 0
    return int(
        _uvm_lib.uvm_batch_advise(
            ptr_array,
            size_array,
            ctypes.c_size_t(count),
            ctypes.c_int(2),  # cudaMemAdviseSetPreferredLocation
            ctypes.c_int(device),
        )
    )


def batch_advise_ranges_read_mostly(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    device: int = 0,
) -> int:
    """Batch set cudaMemAdviseSetReadMostly for raw UVM ranges."""
    if uvm_residency_tag_only_enabled():
        return 0
    if _uvm_lib is None:
        return 0
    ptr_array, size_array, count = _prepare_uvm_range_arrays(ranges)
    if count <= 0:
        return 0
    return int(
        _uvm_lib.uvm_batch_advise(
            ptr_array,
            size_array,
            ctypes.c_size_t(count),
            ctypes.c_int(1),  # uvm_allocator shim: ReadMostly
            ctypes.c_int(device),
        )
    )


def batch_advise_ranges_accessed_by(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    device: int = 0,
) -> int:
    """Batch set cudaMemAdviseSetAccessedBy for raw UVM ranges."""
    if uvm_residency_tag_only_enabled():
        return 0
    if _uvm_lib is None:
        return 0
    ptr_array, size_array, count = _prepare_uvm_range_arrays(ranges)
    if count <= 0:
        return 0
    return int(
        _uvm_lib.uvm_batch_advise(
            ptr_array,
            size_array,
            ctypes.c_size_t(count),
            ctypes.c_int(3),  # uvm_allocator shim: AccessedBy
            ctypes.c_int(device),
        )
    )


def driver_batch_prefetch_ranges_to_device(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    device: int = 0,
    *,
    async_: bool = False,
) -> int:
    """Try Phase F driver batch prefetch. Negative return means unavailable."""
    if uvm_residency_tag_only_enabled():
        return 0
    if not _uvm_enabled or _uvm_lib is None:
        return -1
    range_list = list(ranges)
    if not range_list:
        return 0
    total = 0
    for start in range(0, len(range_list), _UVM_DRIVER_BATCH_MAX_RANGES):
        chunk = range_list[start : start + _UVM_DRIVER_BATCH_MAX_RANGES]
        ptr_array, size_array, count = _prepare_uvm_range_arrays(chunk)
        if count <= 0:
            continue
        if device < 0:
            result = int(
                _uvm_lib.uvm_driver_batch_prefetch_cpu(
                    ptr_array,
                    size_array,
                    ctypes.c_size_t(count),
                    ctypes.c_int(1 if async_ else 0),
                )
            )
        else:
            result = int(
                _uvm_lib.uvm_driver_batch_prefetch_gpu(
                    ptr_array,
                    size_array,
                    ctypes.c_size_t(count),
                    ctypes.c_int(device),
                    ctypes.c_int(1 if async_ else 0),
                )
            )
        if result < 0:
            return result
        total += result
    return total


def driver_batch_advise_ranges_cpu(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> int:
    """Try Phase F driver batch CPU preferred-location advise."""
    if uvm_residency_tag_only_enabled():
        return 0
    if not _uvm_enabled or _uvm_lib is None:
        return -1
    range_list = list(ranges)
    if not range_list:
        return 0
    total = 0
    for start in range(0, len(range_list), _UVM_DRIVER_BATCH_MAX_RANGES):
        chunk = range_list[start : start + _UVM_DRIVER_BATCH_MAX_RANGES]
        ptr_array, size_array, count = _prepare_uvm_range_arrays(chunk)
        if count <= 0:
            continue
        result = int(
            _uvm_lib.uvm_driver_batch_advise_cpu(
                ptr_array,
                size_array,
                ctypes.c_size_t(count),
            )
        )
        if result < 0:
            return result
        total += result
    return total


def driver_batch_query_ranges_residency_bytes(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    device: int = 0,
    *,
    wait_for_trackers: bool = True,
) -> dict[str, int | bool]:
    """Query page-granular real UVM residency bytes for managed ranges.

    A no-wait query is intended for callers which have already synchronized
    CUDA work. It reads the protected resident masks without waiting on each
    VA block tracker, avoiding an unbounded wait on stale experimental tracker
    entries.
    """
    result: dict[str, int | bool] = {
        "ok": False,
        "status": -1,
        "queried_bytes": 0,
        "gpu_bytes": 0,
        "cpu_bytes": 0,
        "any_bytes": 0,
        "unknown_bytes": 0,
    }
    try:
        ensure_uvm_runtime_library()
    except Exception:
        logger.debug("Failed to load UVM runtime library", exc_info=True)
        return result

    range_list = list(ranges)
    if not range_list:
        result["ok"] = True
        result["status"] = 0
        return result

    query_fn = _uvm_lib.uvm_driver_batch_query_residency_bytes
    if not wait_for_trackers:
        query_fn = getattr(
            _uvm_lib,
            "uvm_driver_batch_query_residency_bytes_nowait",
            None,
        )
        if query_fn is None:
            result["status"] = -2
            return result

    final_status = 0
    total_gpu_bytes = 0
    total_cpu_bytes = 0
    total_any_bytes = 0
    total_unknown_bytes = 0
    total_queried_bytes = 0

    for start in range(0, len(range_list), _UVM_DRIVER_BATCH_MAX_RANGES):
        chunk = range_list[start : start + _UVM_DRIVER_BATCH_MAX_RANGES]
        ptr_array, size_array, count = _prepare_uvm_range_arrays(chunk)
        if count <= 0:
            continue
        gpu_bytes = ctypes.c_uint64(0)
        cpu_bytes = ctypes.c_uint64(0)
        any_bytes = ctypes.c_uint64(0)
        unknown_bytes = ctypes.c_uint64(0)
        queried_bytes = ctypes.c_uint64(0)
        status = int(
            query_fn(
                ptr_array,
                size_array,
                ctypes.c_size_t(count),
                ctypes.c_int(device),
                ctypes.byref(gpu_bytes),
                ctypes.byref(cpu_bytes),
                ctypes.byref(any_bytes),
                ctypes.byref(unknown_bytes),
                ctypes.byref(queried_bytes),
            )
        )
        if status < 0:
            final_status = status
            break
        final_status += status
        total_gpu_bytes += int(gpu_bytes.value)
        total_cpu_bytes += int(cpu_bytes.value)
        total_any_bytes += int(any_bytes.value)
        total_unknown_bytes += int(unknown_bytes.value)
        total_queried_bytes += int(queried_bytes.value)

    result.update(
        {
            "ok": final_status >= 0,
            "status": final_status,
            "queried_bytes": total_queried_bytes,
            "gpu_bytes": total_gpu_bytes,
            "cpu_bytes": total_cpu_bytes,
            "any_bytes": total_any_bytes,
            "unknown_bytes": total_unknown_bytes,
        }
    )
    return result


def driver_register_region(
    *,
    region_id: int,
    ptr: int,
    nbytes: int,
) -> int:
    """Register an opaque explicit region by id and VA geometry."""
    try:
        lib = ensure_uvm_runtime_library()
    except Exception:
        logger.debug("Failed to load UVM runtime library", exc_info=True)
        return -1
    if region_id <= 0 or ptr <= 0 or nbytes <= 0:
        return -2
    try:
        return int(
            lib.uvm_driver_phase_f_register_region(
                ctypes.c_uint64(region_id),
                ctypes.c_uint64(ptr),
                ctypes.c_size_t(nbytes),
            )
        )
    except Exception:
        logger.debug("Failed to register Phase F driver region", exc_info=True)
        return -3


def driver_unregister_region(region_id: int) -> int:
    """Unregister a Phase F driver region by numeric region id."""
    try:
        lib = ensure_uvm_runtime_library()
    except Exception:
        logger.debug("Failed to load UVM runtime library", exc_info=True)
        return -1
    if region_id <= 0:
        return -2
    try:
        return int(lib.uvm_driver_phase_f_unregister_region(ctypes.c_uint64(region_id)))
    except Exception:
        logger.debug("Failed to unregister Phase F driver region", exc_info=True)
        return -3


def driver_update_region_state(
    *,
    region_id: int,
    state: int = UVM_REGION_STATE_UNKNOWN,
) -> int:
    """Update the minimal explicit-region state in the driver."""
    if not has_region_state_ioctl():
        return -1
    if region_id <= 0:
        return -2
    try:
        return int(
            _uvm_lib.uvm_driver_update_region_state(
                ctypes.c_uint64(int(region_id)),
                ctypes.c_uint32(max(int(state), 0)),
            )
        )
    except Exception:
        logger.debug("Failed to update driver region state", exc_info=True)
        return -3


def _driver_region_values_to_dict(values: ctypes.Array) -> dict[str, int]:
    return {
        "region_id": int(values[0]),
        "base": int(values[1]),
        "length": int(values[2]),
        "state": int(values[3]),
    }


def driver_query_region(region_id: int) -> dict[str, int] | None:
    """Query explicit region metadata by region id."""
    if not has_region_state_ioctl() or region_id <= 0:
        return None
    values = (ctypes.c_uint64 * 4)()
    try:
        status = int(
            _uvm_lib.uvm_driver_query_region(
                ctypes.c_uint64(int(region_id)),
                values,
                ctypes.c_size_t(len(values)),
            )
        )
    except Exception:
        logger.debug("Failed to query driver region", exc_info=True)
        return None
    if status < 0:
        return None
    return _driver_region_values_to_dict(values)


def driver_query_region_states(region_ids: Sequence[int]) -> dict[int, int]:
    """Return only the state of explicitly queried numeric region IDs."""
    result: dict[int, int] = {}
    for value in region_ids:
        try:
            region_id = int(value)
        except (TypeError, ValueError):
            continue
        snapshot = driver_query_region(region_id)
        if snapshot is not None:
            result[region_id] = int(snapshot.get("state", UVM_REGION_STATE_UNKNOWN))
    return result


def has_region_state_ioctl() -> bool:
    return (
        _uvm_lib is not None
        and all(
            hasattr(_uvm_lib, name)
            for name in (
                "uvm_driver_update_region_state",
                "uvm_driver_query_region",
                "uvm_driver_query_region_by_va",
            )
        )
    )


def batch_prefetch_ranges_to_device_with_executor(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    device: int = 0,
    *,
    executor: str = "cpp_batch",
) -> tuple[int, str, int]:
    """Batch prefetch using cpp_batch, driver_batch, or auto fallback.

    Returns (successes, executor_used, driver_result). driver_result is negative
    when the Phase F driver path was unavailable or failed.
    """
    if executor == "driver_batch":
        driver_result = driver_batch_prefetch_ranges_to_device(ranges, device)
        return max(driver_result, 0), "driver_batch", driver_result
    if executor == "auto":
        driver_result = driver_batch_prefetch_ranges_to_device(ranges, device)
        if driver_result >= 0:
            return driver_result, "driver_batch", driver_result
        return (
            batch_prefetch_ranges_to_device(ranges, device),
            "cpp_batch_fallback",
            driver_result,
        )
    return batch_prefetch_ranges_to_device(ranges, device), "cpp_batch", 0


def batch_offload_ranges_to_cpu_with_executor(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    sync: bool = False,
    executor: str = "cpp_batch",
) -> tuple[int, str, int, int]:
    """Batch CPU offload using cpp_batch, driver_batch, or auto fallback.

    Returns (successes, executor_used, driver_advise_result,
    driver_prefetch_result).
    """
    if executor == "driver_batch":
        advise_result = driver_batch_advise_ranges_cpu(ranges)
        prefetch_result = driver_batch_prefetch_ranges_to_device(ranges, -1)
        successes = min(max(advise_result, 0), max(prefetch_result, 0))
        if successes > 0 and sync:
            try:
                torch.cuda.synchronize()
            except Exception:
                logger.debug(
                    "Failed to synchronize after driver CPU offload", exc_info=True
                )
        return successes, "driver_batch", advise_result, prefetch_result
    if executor == "auto":
        advise_result = driver_batch_advise_ranges_cpu(ranges)
        prefetch_result = driver_batch_prefetch_ranges_to_device(ranges, -1)
        if advise_result >= 0 and prefetch_result >= 0:
            successes = min(advise_result, prefetch_result)
            if successes > 0 and sync:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    logger.debug(
                        "Failed to synchronize after driver CPU offload", exc_info=True
                    )
            return successes, "driver_batch", advise_result, prefetch_result
        return (
            batch_offload_ranges_to_cpu(ranges, sync=sync),
            "cpp_batch_fallback",
            advise_result,
            prefetch_result,
        )
    return batch_offload_ranges_to_cpu(ranges, sync=sync), "cpp_batch", 0, 0


def batch_advise_ranges_cpu_with_executor(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    executor: str = "cpp_batch",
) -> tuple[int, str, int]:
    """Batch CPU preferred-location advise with optional Phase F driver path."""
    if executor == "driver_batch":
        driver_result = driver_batch_advise_ranges_cpu(ranges)
        return max(driver_result, 0), "driver_batch", driver_result
    if executor == "auto":
        driver_result = driver_batch_advise_ranges_cpu(ranges)
        if driver_result >= 0:
            return driver_result, "driver_batch", driver_result
        return (
            batch_advise_ranges_preferred_location(ranges, -1),
            "cpp_batch_fallback",
            driver_result,
        )
    return batch_advise_ranges_preferred_location(ranges, -1), "cpp_batch", 0


def batch_prepare_ranges_for_remote_read_with_executor(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    sync: bool = False,
    executor: str = "cpp_batch",
    device: int = 0,
    accessed_by: bool = False,
) -> tuple[int, str, int, int, int, int]:
    """Prepare UVM ranges for CPU-resident GPU remote reads.

    The range is marked ReadMostly, moved to CPU/preferred CPU, and optionally
    tagged AccessedBy for the GPU. With the modified driver remote-first path
    enabled, the ReadMostly + CPU-resident state lets first GPU reads install
    remote CPU mappings instead of migrating the page to GPU.

    Returns (successes, executor_used, readmostly_result, accessed_by_result,
    driver_advise_result, driver_prefetch_result).
    """
    if not ranges:
        return 0, executor, 0, 0, 0, 0

    readmostly_result = batch_advise_ranges_read_mostly(ranges, device=device)
    (
        offload_successes,
        executor_used,
        driver_advise_result,
        driver_prefetch_result,
    ) = batch_offload_ranges_to_cpu_with_executor(
        ranges,
        sync=sync,
        executor=executor,
    )
    accessed_by_result = 0
    if accessed_by:
        accessed_by_result = batch_advise_ranges_accessed_by(
            ranges,
            device=device,
        )

    successes = min(readmostly_result, offload_successes)
    if accessed_by:
        successes = min(successes, accessed_by_result)
    return (
        successes,
        executor_used,
        readmostly_result,
        accessed_by_result,
        driver_advise_result,
        driver_prefetch_result,
    )


def batch_prepare_ranges_for_native_remote_access_with_executor(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    sync: bool = False,
    executor: str = "cpp_batch",
    device: int = 0,
) -> tuple[int, str, int, int, int]:
    """Prepare UVM ranges for native pure GPU remote access.

    This path matches the microbenchmark's ``uvm_pure_remote`` mechanism:
    place managed ranges on CPU, set PreferredLocation(CPU), then set
    AccessedBy(GPU) so native UVM can pre-map GPU->CPU remote READ mappings.
    It deliberately does not use ReadMostly, modified-driver remote-first, or
    repository-specific driver-batch ioctls.

    Returns (successes, executor_used, accessed_by_result,
    prefetch_cpu_successes, preferred_cpu_result).
    """
    if not ranges:
        return 0, executor, 0, 0, 0

    # Keep this experiment on native CUDA UVM APIs. The driver-batch executor
    # can use repository-specific Phase F ioctls, which would change the test.
    prefetch_cpu_successes = batch_prefetch_ranges_to_device(ranges, -1)
    executor_used = "cpp_batch_native_cuda"
    if prefetch_cpu_successes > 0 and sync:
        try:
            torch.cuda.synchronize()
        except Exception:
            logger.debug(
                "Failed to synchronize after native remote CPU prefetch", exc_info=True
            )

    preferred_cpu_result = batch_advise_ranges_preferred_location(
        ranges,
        device=-1,
    )
    accessed_by_result = batch_advise_ranges_accessed_by(
        ranges,
        device=device,
    )
    successes = min(
        prefetch_cpu_successes,
        preferred_cpu_result,
        accessed_by_result,
    )
    return (
        successes,
        executor_used,
        accessed_by_result,
        prefetch_cpu_successes,
        preferred_cpu_result,
    )


def batch_offload_ranges_to_cpu(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    *,
    sync: bool = False,
) -> int:
    """Batch set CPU preferred location and prefetch UVM ranges to CPU."""
    if not ranges:
        return 0
    advise_successes = batch_advise_ranges_preferred_location(ranges, -1)
    prefetch_successes = batch_prefetch_ranges_to_cpu(ranges)
    successes = min(advise_successes, prefetch_successes)
    if successes > 0 and sync:
        try:
            torch.cuda.synchronize()
        except Exception:
            logger.debug(
                "Failed to synchronize after batch UVM CPU offload", exc_info=True
            )
    return successes


def driver_restore_regions(
    region_ids: Sequence[int],
    *,
    device: int = 0,
    executor: str = "auto",
) -> dict[str, object]:
    """Restore explicitly registered regions by their recorded VA ranges."""
    ranges: list[tuple[int, int]] = []
    valid_ids: list[int] = []
    for value in region_ids:
        try:
            region_id = int(value)
        except (TypeError, ValueError):
            continue
        snapshot = driver_query_region(region_id)
        if snapshot is None:
            continue
        base = int(snapshot.get("base") or 0)
        length = int(snapshot.get("length") or 0)
        if base > 0 and length > 0:
            valid_ids.append(region_id)
            ranges.append((base, length))
    successes, executor_used, driver_status = (
        batch_prefetch_ranges_to_device_with_executor(
            ranges,
            device,
            executor=executor,
        )
        if ranges
        else (0, executor, 0)
    )
    # The range executor does not know region identity. Reflect completed
    # restores back into the explicit-region state machine in submission order.
    for region_id in valid_ids[: max(int(successes), 0)]:
        driver_update_region_state(
            region_id=region_id,
            state=UVM_REGION_STATE_HOT,
        )
    complete = int(successes) == len(ranges)
    return {
        "ok": complete,
        "region_ids": valid_ids,
        "success_count": int(successes),
        "failure_count": len(ranges) - int(successes),
        "executor": executor_used,
        "driver_status": driver_status,
    }
