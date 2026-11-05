#!/usr/bin/env python3
"""Measure fragmentation effects for native or explicitly placed managed memory.

The two invocations of this probe use identical allocation sizes.  The
``contiguous`` layout releases one continuous range, while ``fragmented``
releases alternating fixed-size stripes with the same total byte count.
Automatic managed-memory fault migration must remain disabled so residency
changes can only come from explicit release or PMA eviction.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import time
from typing import Any

from vllm.device_allocator import uvm
from vllm.device_allocator.uvm_pool_residency import (
    register_uvm_region,
    release_uvm_regions,
    unregister_uvm_regions,
)


MIB = 1024 * 1024
GIB = 1024 * MIB
MEM_ATTACH_GLOBAL = 1
ALIGNMENT = 64 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("contiguous", "fragmented"), required=True)
    parser.add_argument(
        "--allocation-kind", choices=("native", "managed"), default="native"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--arena-gib", type=float, default=28.0)
    parser.add_argument("--release-gib", type=float, default=4.0)
    parser.add_argument("--native-gib", type=float, default=4.0)
    parser.add_argument("--stripe-mib", type=float, default=16.0)
    parser.add_argument("--bandwidth-repeats", type=int, default=8)
    return parser.parse_args()


def load_cudart() -> ctypes.CDLL:
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
    if int(status) == 0:
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


def aligned_bytes(value: float, unit: int, name: str) -> int:
    result = int(value * unit)
    if result <= 0 or result % ALIGNMENT:
        raise ValueError(f"{name} must be positive and 64-KiB aligned")
    return result


def release_ranges(
    layout: str, arena_bytes: int, release_bytes: int, stripe_bytes: int
) -> list[tuple[int, int]]:
    if layout == "contiguous":
        if release_bytes > arena_bytes:
            raise ValueError("release exceeds arena")
        return [(0, release_bytes)]
    if release_bytes * 2 > arena_bytes:
        raise ValueError("fragmented layout needs a 2x release-sized arena span")
    if release_bytes % stripe_bytes:
        raise ValueError("release bytes must be divisible by stripe bytes")
    return [
        (index * 2 * stripe_bytes, stripe_bytes)
        for index in range(release_bytes // stripe_bytes)
    ]


def main() -> int:
    args = parse_args()
    arena_bytes = aligned_bytes(args.arena_gib, GIB, "arena")
    release_bytes = aligned_bytes(args.release_gib, GIB, "release")
    native_bytes = aligned_bytes(args.native_gib, GIB, "native")
    stripe_bytes = aligned_bytes(args.stripe_mib, MIB, "stripe")
    ranges = release_ranges(
        args.layout, arena_bytes, release_bytes, stripe_bytes
    )
    if args.bandwidth_repeats <= 0:
        raise ValueError("bandwidth repeats must be positive")

    mode_path = "/sys/module/nvidia_uvm/parameters/uvm_manual_migration_fault_mode"
    with open(mode_path, encoding="ascii") as handle:
        fault_mode = int(handle.read().strip())
    if fault_mode != 2:
        raise RuntimeError(f"manual migration fault mode must be 2, got {fault_mode}")

    cudart = load_cudart()
    check(cudart, cudart.cudaSetDevice(args.device), "cudaSetDevice")
    arena = ctypes.c_void_p()
    native = ctypes.c_void_p()
    registered: list[str] = []
    result: dict[str, Any] | None = None
    try:
        context_memory = memory_info(cudart)
        emit(
            "context_ready",
            layout=args.layout,
            allocation_kind=args.allocation_kind,
            fault_mode=fault_mode,
            arena_bytes=arena_bytes,
            release_bytes=release_bytes,
            native_bytes=native_bytes,
            stripe_bytes=stripe_bytes,
            release_range_count=len(ranges),
            memory=context_memory,
        )
        check(
            cudart,
            cudart.cudaMallocManaged(
                ctypes.byref(arena), arena_bytes, MEM_ATTACH_GLOBAL
            ),
            "cudaMallocManaged arena",
        )
        check(
            cudart,
            cudart.cudaMemPrefetchAsync(
                arena, arena_bytes, args.device, ctypes.c_void_p()
            ),
            "prefetch arena to GPU",
        )
        check(cudart, cudart.cudaDeviceSynchronize(), "initial placement")

        before_release_memory = memory_info(cudart)
        before_release_residency = residency(arena, arena_bytes, args.device)
        emit(
            "before_release",
            memory=before_release_memory,
            arena=before_release_residency,
        )

        base = int(arena.value or 0)
        for index, (offset, length) in enumerate(ranges):
            region_id = (
                f"fragmentation_probe:{os.getpid()}:{args.layout}:{index}"
            )
            register_uvm_region(
                region_id, base + int(offset), int(length), state="unknown"
            )
            registered.append(region_id)
        release = release_uvm_regions(
            registered,
            device=args.device,
            remote_read=False,
            include_audit=True,
        )
        check(cudart, cudart.cudaDeviceSynchronize(), "release fence")

        after_release_memory = memory_info(cudart)
        after_release_residency = residency(arena, arena_bytes, args.device)
        audits = release.get("audits", {})
        audited_released_bytes = sum(
            int(value.get("released_bytes", 0))
            for value in audits.values()
            if isinstance(value, dict)
        )
        emit(
            "after_release",
            memory=after_release_memory,
            arena=after_release_residency,
            release_ok=release.get("ok") is True,
            audited_released_bytes=audited_released_bytes,
            release_range_count=len(registered),
        )

        allocation_started_ns = time.perf_counter_ns()
        if args.allocation_kind == "native":
            allocation_status = int(
                cudart.cudaMalloc(ctypes.byref(native), native_bytes)
            )
        else:
            allocation_status = int(
                cudart.cudaMallocManaged(
                    ctypes.byref(native), native_bytes, MEM_ATTACH_GLOBAL
                )
            )
        allocation_ns = time.perf_counter_ns() - allocation_started_ns
        if allocation_status != 0:
            raw = cudart.cudaGetErrorString(allocation_status)
            error = raw.decode("utf-8", "replace") if raw else "unknown"
            emit(
                "target_allocation_failed",
                layout=args.layout,
                allocation_kind=args.allocation_kind,
                status=allocation_status,
                error=error,
            )
            return 1

        placement_ns = 0
        target_residency_after_placement: dict[str, Any] | None = None
        if args.allocation_kind == "managed":
            placement_started_ns = time.perf_counter_ns()
            check(
                cudart,
                cudart.cudaMemPrefetchAsync(
                    native, native_bytes, args.device, ctypes.c_void_p()
                ),
                "explicitly prefetch managed target to GPU",
            )
            check(
                cudart,
                cudart.cudaDeviceSynchronize(),
                "managed target placement fence",
            )
            placement_ns = time.perf_counter_ns() - placement_started_ns
            target_residency_after_placement = residency(
                native, native_bytes, args.device
            )

        initial_touch_started_ns = time.perf_counter_ns()
        check(
            cudart,
            cudart.cudaMemset(native, 0x5A, native_bytes),
            "touch target allocation",
        )
        check(cudart, cudart.cudaDeviceSynchronize(), "target allocation fence")
        initial_touch_ns = time.perf_counter_ns() - initial_touch_started_ns

        bandwidth_started_ns = time.perf_counter_ns()
        for iteration in range(args.bandwidth_repeats):
            check(
                cudart,
                cudart.cudaMemset(native, (iteration + 1) & 0xFF, native_bytes),
                "steady-state target memset",
            )
        check(cudart, cudart.cudaDeviceSynchronize(), "bandwidth measurement fence")
        bandwidth_elapsed_ns = time.perf_counter_ns() - bandwidth_started_ns
        bandwidth_gbps = (
            native_bytes
            * args.bandwidth_repeats
            / (bandwidth_elapsed_ns / 1_000_000_000)
            / 1_000_000_000
        )

        during_native_memory = memory_info(cudart)
        during_native_residency = residency(arena, arena_bytes, args.device)
        emit(
            "during_target",
            memory=during_native_memory,
            arena=during_native_residency,
            target_ptr=int(native.value or 0),
            allocation_kind=args.allocation_kind,
            target_residency=target_residency_after_placement,
            performance={
                "allocation_ns": allocation_ns,
                "placement_ns": placement_ns,
                "initial_touch_ns": initial_touch_ns,
                "time_to_ready_ns": (
                    allocation_ns + placement_ns + initial_touch_ns
                ),
                "bandwidth_repeats": args.bandwidth_repeats,
                "bandwidth_elapsed_ns": bandwidth_elapsed_ns,
                "steady_state_memset_gbps": bandwidth_gbps,
            },
        )

        initial_gpu_ok = (
            int(before_release_residency.get("cpu_bytes", -1)) == 0
            and int(before_release_residency.get("gpu_bytes", 0)) == arena_bytes
            and int(before_release_residency.get("unknown_bytes", -1)) == 0
        )
        release_ok = (
            release.get("ok") is True
            and audited_released_bytes == release_bytes
            and int(after_release_residency.get("cpu_bytes", -1))
            == release_bytes
            and int(after_release_residency.get("gpu_bytes", -1))
            == arena_bytes - release_bytes
            and int(after_release_residency.get("unknown_bytes", -1)) == 0
        )
        hot_evicted_bytes = max(
            0,
            int(during_native_residency.get("cpu_bytes", 0))
            - int(after_release_residency.get("cpu_bytes", 0)),
        )
        free_drop_bytes = max(
            0,
            int(after_release_memory["free_bytes"])
            - int(during_native_memory["free_bytes"]),
        )
        accounting_error_bytes = abs(
            native_bytes - free_drop_bytes - hot_evicted_bytes
        )
        target_gpu_ok = (
            True
            if args.allocation_kind == "native"
            else (
                target_residency_after_placement is not None
                and int(target_residency_after_placement.get("gpu_bytes", -1))
                == native_bytes
                and int(target_residency_after_placement.get("cpu_bytes", -1)) == 0
                and int(target_residency_after_placement.get("unknown_bytes", -1))
                == 0
            )
        )
        result = {
            "measurement_valid": bool(
                initial_gpu_ok
                and release_ok
                and target_gpu_ok
                and native_bytes > int(before_release_memory["free_bytes"])
                and int(during_native_residency.get("unknown_bytes", -1)) == 0
                and accounting_error_bytes <= 64 * MIB
            ),
            "layout": args.layout,
            "allocation_kind": args.allocation_kind,
            "manual_migration_fault_mode": fault_mode,
            "arena_bytes": arena_bytes,
            "release_bytes": release_bytes,
            "native_bytes": native_bytes,
            "stripe_bytes": stripe_bytes,
            "release_range_count": len(ranges),
            "initial_gpu_ok": initial_gpu_ok,
            "release_ok": release_ok,
            "native_allocation_ok": True,
            "target_gpu_ok": target_gpu_ok,
            "native_exceeds_pre_release_free": (
                native_bytes > int(before_release_memory["free_bytes"])
            ),
            "pre_release_free_bytes": int(before_release_memory["free_bytes"]),
            "after_release_free_bytes": int(after_release_memory["free_bytes"]),
            "during_native_free_bytes": int(during_native_memory["free_bytes"]),
            "free_drop_bytes": free_drop_bytes,
            "cold_cpu_bytes": int(after_release_residency.get("cpu_bytes", 0)),
            "hot_evicted_bytes": hot_evicted_bytes,
            "arena_cpu_bytes_during_native": int(
                during_native_residency.get("cpu_bytes", 0)
            ),
            "arena_gpu_bytes_during_native": int(
                during_native_residency.get("gpu_bytes", 0)
            ),
            "accounting_error_bytes": accounting_error_bytes,
            "allocation_ns": allocation_ns,
            "placement_ns": placement_ns,
            "initial_touch_ns": initial_touch_ns,
            "time_to_ready_ns": allocation_ns + placement_ns + initial_touch_ns,
            "bandwidth_repeats": args.bandwidth_repeats,
            "bandwidth_elapsed_ns": bandwidth_elapsed_ns,
            "steady_state_memset_gbps": bandwidth_gbps,
            "target_residency_after_placement": target_residency_after_placement,
        }
        emit("probe_result", **result)
        return 0 if result["measurement_valid"] else 1
    finally:
        if native.value:
            cudart.cudaFree(native)
        if registered:
            try:
                unregister_uvm_regions(registered)
            except Exception as exc:
                emit("unregister_failed", error=repr(exc))
        if arena.value:
            cudart.cudaFree(arena)
        cudart.cudaDeviceSynchronize()
        emit("cleanup", memory=memory_info(cudart), prior_result=result)


if __name__ == "__main__":
    raise SystemExit(main())
