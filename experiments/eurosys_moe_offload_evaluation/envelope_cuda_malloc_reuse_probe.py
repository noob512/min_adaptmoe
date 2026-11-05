#!/usr/bin/env python3
"""Prove whether original-VA envelope release feeds native cudaMalloc.

This intentionally keeps automatic managed-memory fault migration disabled.
The cold allocation is released with the same explicit-region API used by the
original-VA static-alias path, then a native allocation larger than the
pre-release free-memory balance is allocated and touched.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
from typing import Any

from vllm.device_allocator import uvm
from vllm.device_allocator.uvm_pool_residency import (
    register_uvm_region,
    release_uvm_regions,
    unregister_uvm_regions,
)


MIB = 1024 * 1024
GIB = 1024 * MIB
CPU_DEVICE_ID = -1
MEM_ATTACH_GLOBAL = 1
REGION_ID = "cuda_malloc_reuse_probe:original_va_cold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--hot-gib", type=float, default=20.0)
    parser.add_argument("--expert-gib", type=float, default=8.0)
    parser.add_argument("--native-gib", type=float, default=6.0)
    return parser.parse_args()


def load_cudart() -> ctypes.CDLL:
    # The host ldconfig also exposes CUDA 13, while this vLLM/PyTorch build and
    # uvm_allocator.so use CUDA 12.8.  Loading the generic lookup result would
    # create a second runtime and make a CUDA-12 managed pointer invalid to the
    # CUDA-13 prefetch call.
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    versioned = os.path.join(cuda_home, "lib64", "libcudart.so.12")
    path = (
        versioned
        if os.path.exists(versioned)
        else (ctypes.util.find_library("cudart") or "libcudart.so")
    )
    lib = ctypes.CDLL(path)
    lib.cudaGetErrorString.restype = ctypes.c_char_p
    lib.cudaGetErrorString.argtypes = [ctypes.c_int]
    lib.cudaSetDevice.restype = ctypes.c_int
    lib.cudaSetDevice.argtypes = [ctypes.c_int]
    lib.cudaMallocManaged.restype = ctypes.c_int
    lib.cudaMallocManaged.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaFree.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaMemPrefetchAsync.restype = ctypes.c_int
    lib.cudaMemPrefetchAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.cudaMemset.restype = ctypes.c_int
    lib.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    lib.cudaDeviceSynchronize.restype = ctypes.c_int
    lib.cudaDeviceSynchronize.argtypes = []
    lib.cudaMemGetInfo.restype = ctypes.c_int
    lib.cudaMemGetInfo.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    return lib


def check(lib: ctypes.CDLL, status: int, operation: str) -> None:
    if status == 0:
        return
    raw = lib.cudaGetErrorString(int(status))
    error = raw.decode("utf-8", "replace") if raw else "unknown"
    raise RuntimeError(f"{operation} failed: status={status}, error={error}")


def memory_info(lib: ctypes.CDLL) -> dict[str, int]:
    free_bytes = ctypes.c_size_t()
    total_bytes = ctypes.c_size_t()
    check(
        lib,
        lib.cudaMemGetInfo(ctypes.byref(free_bytes), ctypes.byref(total_bytes)),
        "cudaMemGetInfo",
    )
    return {
        "free_bytes": int(free_bytes.value),
        "used_bytes": int(total_bytes.value - free_bytes.value),
        "total_bytes": int(total_bytes.value),
    }


def residency(ptr: ctypes.c_void_p, nbytes: int, device: int) -> dict[str, Any]:
    return dict(
        uvm.driver_batch_query_ranges_residency_bytes(
            [(int(ptr.value or 0), int(nbytes))], device=int(device)
        )
    )


def emit(stage: str, **payload: Any) -> None:
    print(json.dumps({"stage": stage, **payload}, sort_keys=True), flush=True)


def mostly(value: int, expected: int) -> bool:
    return int(value) + 2 * MIB >= int(expected)


def main() -> int:
    args = parse_args()
    hot_bytes = int(args.hot_gib * GIB)
    expert_bytes = int(args.expert_gib * GIB)
    native_bytes = int(args.native_gib * GIB)
    if min(hot_bytes, expert_bytes, native_bytes) <= 0:
        raise ValueError("all allocation sizes must be positive")
    if expert_bytes % (64 * 1024):
        raise ValueError("expert allocation must be 64 KiB aligned")

    mode_path = "/sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode"
    with open(mode_path, encoding="ascii") as handle:
        fault_mode = int(handle.read().strip())
    if fault_mode != 2:
        raise RuntimeError(f"manual migration fault mode must be 2, got {fault_mode}")

    cudart = load_cudart()
    check(cudart, cudart.cudaSetDevice(args.device), "cudaSetDevice")
    hot = ctypes.c_void_p()
    expert = ctypes.c_void_p()
    native = ctypes.c_void_p()
    registered = False
    summary: dict[str, Any] = {}
    try:
        context_ready = memory_info(cudart)
        emit(
            "context_ready",
            memory=context_ready,
            fault_mode=fault_mode,
            hot_bytes=hot_bytes,
            expert_bytes=expert_bytes,
            native_bytes=native_bytes,
        )

        check(
            cudart,
            cudart.cudaMallocManaged(
                ctypes.byref(hot), hot_bytes, MEM_ATTACH_GLOBAL
            ),
            "cudaMallocManaged hot",
        )
        check(
            cudart,
            cudart.cudaMallocManaged(
                ctypes.byref(expert), expert_bytes, MEM_ATTACH_GLOBAL
            ),
            "cudaMallocManaged expert",
        )
        check(
            cudart,
            cudart.cudaMemPrefetchAsync(
                hot, hot_bytes, args.device, ctypes.c_void_p()
            ),
            "prefetch hot to GPU",
        )
        check(
            cudart,
            cudart.cudaMemPrefetchAsync(
                expert, expert_bytes, args.device, ctypes.c_void_p()
            ),
            "prefetch expert to GPU",
        )
        check(cudart, cudart.cudaDeviceSynchronize(), "initial GPU placement")

        before_release = memory_info(cudart)
        hot_before = residency(hot, hot_bytes, args.device)
        expert_before = residency(expert, expert_bytes, args.device)
        emit(
            "before_envelope_release",
            memory=before_release,
            hot=hot_before,
            expert=expert_before,
        )

        register_uvm_region(
            REGION_ID, int(expert.value or 0), expert_bytes, state="unknown"
        )
        registered = True
        release = release_uvm_regions(
            [REGION_ID],
            device=args.device,
            remote_read=False,
            include_audit=True,
        )
        check(cudart, cudart.cudaDeviceSynchronize(), "envelope release fence")

        after_release = memory_info(cudart)
        hot_after_release = residency(hot, hot_bytes, args.device)
        expert_after_release = residency(expert, expert_bytes, args.device)
        emit(
            "after_envelope_release",
            memory=after_release,
            hot=hot_after_release,
            expert=expert_after_release,
            release=release,
        )

        native_status = int(cudart.cudaMalloc(ctypes.byref(native), native_bytes))
        if native_status != 0:
            raw = cudart.cudaGetErrorString(native_status)
            error = raw.decode("utf-8", "replace") if raw else "unknown"
            emit("native_allocation_failed", status=native_status, error=error)
            return 1
        check(
            cudart,
            cudart.cudaMemset(native, 0x5A, native_bytes),
            "touch native allocation",
        )
        check(cudart, cudart.cudaDeviceSynchronize(), "native allocation fence")

        during_native = memory_info(cudart)
        hot_during = residency(hot, hot_bytes, args.device)
        expert_during = residency(expert, expert_bytes, args.device)
        emit(
            "during_native_allocation",
            memory=during_native,
            hot=hot_during,
            expert=expert_during,
            native_ptr=int(native.value or 0),
        )

        audit = release.get("audits", {}).get(REGION_ID, {})
        free_gain = max(
            0, int(after_release["free_bytes"]) - int(before_release["free_bytes"])
        )
        hot_evicted = max(
            0,
            int(hot_after_release.get("gpu_bytes", 0))
            - int(hot_during.get("gpu_bytes", 0)),
        )
        native_exceeds_initial_free = (
            native_bytes > int(before_release["free_bytes"])
        )
        release_ok = (
            release.get("ok") is True
            and int(audit.get("released_bytes", 0)) + 2 * MIB >= expert_bytes
            and mostly(int(expert_after_release.get("cpu_bytes", 0)), expert_bytes)
            and int(expert_after_release.get("gpu_bytes", 0)) <= 2 * MIB
        )
        hot_preserved = (
            mostly(int(hot_after_release.get("gpu_bytes", 0)), hot_bytes)
            and mostly(int(hot_during.get("gpu_bytes", 0)), hot_bytes)
            and hot_evicted <= 2 * MIB
        )
        cold_preserved = (
            mostly(int(expert_during.get("cpu_bytes", 0)), expert_bytes)
            and int(expert_during.get("gpu_bytes", 0)) <= 2 * MIB
        )
        success = bool(
            release_ok
            and native_exceeds_initial_free
            and hot_preserved
            and cold_preserved
        )
        summary = {
            "success": success,
            "release_ok": release_ok,
            "native_allocation_ok": True,
            "native_exceeds_initial_free": native_exceeds_initial_free,
            "hot_preserved": hot_preserved,
            "cold_preserved": cold_preserved,
            "initial_free_bytes": int(before_release["free_bytes"]),
            "after_release_free_bytes": int(after_release["free_bytes"]),
            "free_gain_bytes": free_gain,
            "native_bytes": native_bytes,
            "released_bytes": int(audit.get("released_bytes", 0)),
            "hot_evicted_bytes": hot_evicted,
        }
        emit("probe_result", **summary)
        return 0 if success else 1
    finally:
        if native.value:
            cudart.cudaFree(native)
        if registered:
            try:
                unregister_uvm_regions([REGION_ID])
            except Exception as exc:  # cleanup evidence, not result mutation
                emit("unregister_failed", error=repr(exc))
        if expert.value:
            cudart.cudaFree(expert)
        if hot.value:
            cudart.cudaFree(hot)
        cudart.cudaDeviceSynchronize()
        emit("cleanup", memory=memory_info(cudart), prior_result=summary or None)


if __name__ == "__main__":
    raise SystemExit(main())
