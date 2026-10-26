#!/usr/bin/env python3
"""Directed original-managed-VA static-alias/CUDA-Graph probe."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import statistics
import time
from pathlib import Path

import torch

from vllm.device_allocator import uvm
from vllm.device_allocator import uvm_static_alias as static_alias


REGION_BYTES = 512 * 1024
ARENA_BYTES = 2 * REGION_BYTES
VA_BLOCK_BYTES = 2 * 1024 * 1024
# Leave enough headroom to select one complete 2 MiB-aligned UVM VA block
# even when the managed allocation itself begins just past such a boundary.
ALLOCATION_BYTES = 2 * VA_BLOCK_BYTES
LAYER_ID = 7001
EXPERT_ID = 17
SLOT_ID = 23

UVM_SET_PREFERRED_LOCATION = 42
UVM_UNSET_PREFERRED_LOCATION = 43
UVM_ENABLE_READ_DUPLICATION = 44
UVM_DISABLE_READ_DUPLICATION = 45
UVM_SET_ACCESSED_BY = 46
UVM_UNSET_ACCESSED_BY = 47
UVM_MIGRATE = 51
NV_ERR_IN_USE = 0x17

# NV_PROCESSOR_UUID_CPU_DEFAULT from nvidia-uvm/nvCpuUuid.c. UUID byte order is
# the in-memory NvUuid order used by the UVM ioctl ABI, not uuid.UUID.bytes.
CPU_PROCESSOR_UUID = bytes(
    (
        0xA6,
        0x5E,
        0x0F,
        0x4E,
        0xD7,
        0xD4,
        0x7B,
        0xA2,
        0x50,
        0x47,
        0x41,
        0x2C,
        0x14,
        0x2A,
        0x77,
        0x73,
    )
)


class _Uuid(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)]


class _PreferredLocationParams(ctypes.Structure):
    _fields_ = [
        ("requestedBase", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
        ("preferredLocation", _Uuid),
        ("preferredCpuNumaNode", ctypes.c_int32),
        ("rmStatus", ctypes.c_uint32),
    ]


class _RangePolicyParams(ctypes.Structure):
    _fields_ = [
        ("requestedBase", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
        ("rmStatus", ctypes.c_uint32),
    ]


class _AccessedByParams(ctypes.Structure):
    _fields_ = [
        ("requestedBase", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
        ("accessedByUuid", _Uuid),
        ("rmStatus", ctypes.c_uint32),
    ]


class _MigrateParams(ctypes.Structure):
    _fields_ = [
        ("base", ctypes.c_uint64),
        ("length", ctypes.c_uint64),
        ("destinationUuid", _Uuid),
        ("flags", ctypes.c_uint32),
        ("semaphoreAddress", ctypes.c_uint64),
        ("semaphorePayload", ctypes.c_uint32),
        ("cpuNumaNode", ctypes.c_int32),
        ("userSpaceStart", ctypes.c_uint64),
        ("userSpaceLength", ctypes.c_uint64),
        ("rmStatus", ctypes.c_uint32),
    ]


def _assert_policy_ioctl_abi() -> None:
    expected = {
        "uuid_size": (ctypes.sizeof(_Uuid), 16),
        "preferred_size": (ctypes.sizeof(_PreferredLocationParams), 40),
        "preferred_uuid": (_PreferredLocationParams.preferredLocation.offset, 16),
        "preferred_status": (_PreferredLocationParams.rmStatus.offset, 36),
        "range_size": (ctypes.sizeof(_RangePolicyParams), 24),
        "range_status": (_RangePolicyParams.rmStatus.offset, 16),
        "accessed_by_size": (ctypes.sizeof(_AccessedByParams), 40),
        "accessed_by_uuid": (_AccessedByParams.accessedByUuid.offset, 16),
        "accessed_by_status": (_AccessedByParams.rmStatus.offset, 32),
        "migrate_size": (ctypes.sizeof(_MigrateParams), 80),
        "migrate_status": (_MigrateParams.rmStatus.offset, 72),
    }
    mismatches = [
        f"{name}: got {actual}, expected {wanted}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise RuntimeError("UVM policy ioctl ABI mismatch: " + ", ".join(mismatches))


_assert_policy_ioctl_abi()


def _require(result: dict[str, object], label: str) -> None:
    if not bool(result.get("ok")):
        raise RuntimeError(f"{label} failed: {result}")


def _all_equal(tensor: torch.Tensor, value: int) -> bool:
    return bool(torch.all(tensor == int(value)).item())


def _submit_uvm_ioctl(
    fd: int, request: int, params: ctypes.Structure
) -> dict[str, int | bool]:
    ctypes.set_errno(0)
    ret = int(uvm._libc_ioctl(fd, request, params))
    errno_value = int(ctypes.get_errno()) if ret != 0 else 0
    rm_status = int(getattr(params, "rmStatus"))
    accepted = ret == 0 and rm_status == 0
    return {
        "ioctl": int(request),
        "ioctl_ret": ret,
        "errno": errno_value,
        "rm_status": rm_status,
        "accepted": accepted,
        "rejected": not accepted,
    }


def _range_policy_params(base: int, length: int) -> _RangePolicyParams:
    params = _RangePolicyParams()
    params.requestedBase = int(base)
    params.length = int(length)
    return params


def _migrate_params(
    base: int, length: int, destination_uuid: bytes
) -> _MigrateParams:
    params = _MigrateParams()
    params.base = int(base)
    params.length = int(length)
    params.destinationUuid.uuid[:] = destination_uuid
    params.cpuNumaNode = -1
    return params


def _policy_mutation_probe(fd: int, base: int, length: int) -> dict[str, object]:
    gpu_uuid = bytes(uvm._phase_f_gpu_uuid(0).uuid)

    preferred = _PreferredLocationParams()
    preferred.requestedBase = int(base)
    preferred.length = int(length)
    preferred.preferredLocation.uuid[:] = gpu_uuid
    preferred.preferredCpuNumaNode = -1

    accessed_by = _AccessedByParams()
    accessed_by.requestedBase = int(base)
    accessed_by.length = int(length)
    accessed_by.accessedByUuid.uuid[:] = gpu_uuid

    operations = {
        "set_preferred_location": _submit_uvm_ioctl(
            fd, UVM_SET_PREFERRED_LOCATION, preferred
        ),
        "unset_preferred_location": _submit_uvm_ioctl(
            fd,
            UVM_UNSET_PREFERRED_LOCATION,
            _range_policy_params(base, length),
        ),
        "set_accessed_by": _submit_uvm_ioctl(
            fd, UVM_SET_ACCESSED_BY, accessed_by
        ),
    }

    unset_accessed_by = _AccessedByParams()
    unset_accessed_by.requestedBase = int(base)
    unset_accessed_by.length = int(length)
    unset_accessed_by.accessedByUuid.uuid[:] = gpu_uuid
    operations.update(
        {
            "unset_accessed_by": _submit_uvm_ioctl(
                fd, UVM_UNSET_ACCESSED_BY, unset_accessed_by
            ),
            "enable_read_duplication": _submit_uvm_ioctl(
                fd,
                UVM_ENABLE_READ_DUPLICATION,
                _range_policy_params(base, length),
            ),
            "disable_read_duplication": _submit_uvm_ioctl(
                fd,
                UVM_DISABLE_READ_DUPLICATION,
                _range_policy_params(base, length),
            ),
            "migrate_to_gpu": _submit_uvm_ioctl(
                fd,
                UVM_MIGRATE,
                _migrate_params(base, length, gpu_uuid),
            ),
            "migrate_to_cpu": _submit_uvm_ioctl(
                fd,
                UVM_MIGRATE,
                _migrate_params(base, length, CPU_PROCESSOR_UUID),
            ),
        }
    )

    all_rejected_in_use = all(
        bool(result["rejected"]) and int(result["rm_status"]) == NV_ERR_IN_USE
        for result in operations.values()
    )
    all_accepted = all(bool(result["accepted"]) for result in operations.values())
    return {
        "expected_sealed_rejection_status": NV_ERR_IN_USE,
        "all_rejected_in_use": all_rejected_in_use,
        "all_accepted": all_accepted,
        "operations": operations,
    }


def _cpu_fault_guard_probe(address: int) -> dict[str, object]:
    """Attempt one direct CPU load in a child after MAP revoked CPU PTEs."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        signal.signal(signal.SIGALRM, lambda _signum, _frame: os._exit(124))
        signal.alarm(3)
        try:
            value = int(ctypes.c_uint8.from_address(int(address)).value)
            os.write(write_fd, f"read_succeeded:{value}".encode())
            os._exit(0)
        except BaseException as exc:
            os.write(write_fd, f"exception:{exc!r}".encode())
            os._exit(125)

    os.close(write_fd)
    _, wait_status = os.waitpid(pid, 0)
    payload = os.read(read_fd, 4096).decode(errors="replace")
    os.close(read_fd)
    signaled = os.WIFSIGNALED(wait_status)
    signal_number = os.WTERMSIG(wait_status) if signaled else 0
    exit_code = os.WEXITSTATUS(wait_status) if os.WIFEXITED(wait_status) else None
    return {
        "rejected": signaled
        and signal_number in (int(signal.SIGBUS), int(signal.SIGSEGV)),
        "signal": signal_number,
        "signal_name": signal.Signals(signal_number).name if signaled else "",
        "exit_code": exit_code,
        "child_payload": payload,
    }


def run(replays: int) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.cuda.set_device(0)
    uvm.ensure_uvm_runtime_library()
    session_id = 0
    graph: torch.cuda.CUDAGraph | None = None
    allocation: torch.Tensor | None = None
    arena: torch.Tensor | None = None
    upload: torch.Tensor | None = None
    consumer: torch.Tensor | None = None
    graph_out: torch.Tensor | None = None
    source: torch.Tensor | None = None
    cleanup: dict[str, object] = {}

    try:
        with uvm.uvm_local_memory_pool(enable_prefetch=False):
            allocation = torch.empty(
                ALLOCATION_BYTES, dtype=torch.uint8, device="cuda"
            )
        allocation_va = int(allocation.data_ptr())
        arena_va = (allocation_va + VA_BLOCK_BYTES - 1) & ~(VA_BLOCK_BYTES - 1)
        arena_offset = arena_va - allocation_va
        if arena_offset + VA_BLOCK_BYTES > ALLOCATION_BYTES:
            raise RuntimeError("managed allocation has no complete aligned VA block")
        arena = allocation[arena_offset : arena_offset + ARENA_BYTES]
        upload = arena[:REGION_BYTES]
        consumer = arena[REGION_BYTES:]
        upload.fill_(17)
        consumer.fill_(29)
        torch.cuda.synchronize()

        upload_va = int(upload.data_ptr())
        consumer_va = int(consumer.data_ptr())
        stable_vas = (upload_va, consumer_va, arena_va, allocation_va)

        remote_prepare = uvm.batch_prepare_ranges_for_native_remote_access_with_executor(
            [(allocation_va, ALLOCATION_BYTES)], sync=True, device=0
        )
        if int(remote_prepare[0]) != 1:
            raise RuntimeError(f"CPU residency preparation failed: {remote_prepare}")

        preallocate = uvm.driver_slot_preallocate_ranges(
            [
                (
                    upload_va,
                    REGION_BYTES,
                    LAYER_ID,
                    EXPERT_ID,
                    SLOT_ID,
                    0,
                    uvm.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT,
                )
            ],
            device=0,
        )
        _require(preallocate, "mode-8 backing preallocation")

        requested_pte_mode = os.environ.get(
            "UVM_STATIC_ALIAS_PTE_MODE", "auto"
        ).strip().lower()
        if requested_pte_mode not in {"auto", "4k", "64k"}:
            raise ValueError(
                "UVM_STATIC_ALIAS_PTE_MODE must be auto, 4k, or 64k"
            )
        layout_supports_64k = all(
            value % (64 * 1024) == 0
            for value in (upload_va, consumer_va, REGION_BYTES)
        )
        if requested_pte_mode == "64k" and not layout_supports_64k:
            raise RuntimeError("requested 64K PTEs but alias layout is unaligned")
        use_64k = layout_supports_64k and requested_pte_mode != "4k"
        pte_flag = (
            static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K
            if use_64k
            else 0
        )
        map_started = time.perf_counter_ns()
        mapped = static_alias.driver_static_alias_map(
            [
                (
                    LAYER_ID,
                    EXPERT_ID,
                    SLOT_ID,
                    static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW | pte_flag,
                    upload_va,
                    REGION_BYTES,
                    0,
                ),
                (
                    LAYER_ID,
                    EXPERT_ID,
                    SLOT_ID,
                    pte_flag,
                    consumer_va,
                    REGION_BYTES,
                    0,
                ),
            ],
            device=0,
        )
        map_wall_ms = (time.perf_counter_ns() - map_started) / 1e6
        _require(mapped, "static alias MAP")
        session_id = int(mapped["session_id"])

        seal_started = time.perf_counter_ns()
        sealed = static_alias.driver_static_alias_seal(session_id, device=0)
        seal_wall_ms = (time.perf_counter_ns() - seal_started) / 1e6
        _require(sealed, "static alias SEAL")

        cpu_fault_guard = _cpu_fault_guard_probe(consumer_va)
        if not bool(cpu_fault_guard["rejected"]):
            raise RuntimeError(
                "sealed alias unexpectedly retained or rebuilt a CPU PTE: "
                f"{cpu_fault_guard}"
            )

        if stable_vas != (
            int(upload.data_ptr()),
            int(consumer.data_ptr()),
            int(arena.data_ptr()),
            int(allocation.data_ptr()),
        ):
            raise RuntimeError("managed parameter VA changed after MAP/SEAL")

        source = torch.empty(REGION_BYTES, dtype=torch.uint8, pin_memory=True)
        graph_out = torch.empty(REGION_BYTES, dtype=torch.uint8, device="cuda")
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            source.fill_(41)
            upload.copy_(source, non_blocking=True)
            graph_out.copy_(consumer)
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        capture_started = time.perf_counter_ns()
        with torch.cuda.graph(graph, stream=capture_stream):
            upload.copy_(source, non_blocking=True)
            graph_out.copy_(consumer)
        capture_wall_ms = (time.perf_counter_ns() - capture_started) / 1e6

        replay_ioctl_calls = 0
        original_static_ioctl = static_alias._libc_ioctl

        def _counting_ioctl(fd: int, request: int, params: ctypes.Structure) -> int:
            nonlocal replay_ioctl_calls
            replay_ioctl_calls += 1
            return int(original_static_ioctl(fd, request, params))

        static_alias._libc_ioctl = _counting_ioctl
        correctness_patterns = [3, 19, 67, 131, 251]
        correctness_ms: list[float] = []
        try:
            for pattern in correctness_patterns:
                source.fill_(pattern)
                started = time.perf_counter_ns()
                graph.replay()
                torch.cuda.synchronize()
                correctness_ms.append((time.perf_counter_ns() - started) / 1e6)
                if not _all_equal(graph_out, pattern):
                    raise RuntimeError(f"alias replay mismatch for pattern {pattern}")

            source.fill_(173)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            submit_started = time.perf_counter_ns()
            for _ in range(replays):
                graph.replay()
            submit_wall_ms = (time.perf_counter_ns() - submit_started) / 1e6
            end_event.record()
            end_event.synchronize()
            gpu_total_ms = float(start_event.elapsed_time(end_event))
        finally:
            static_alias._libc_ioctl = original_static_ioctl

        if not _all_equal(graph_out, 173):
            raise RuntimeError("final replay output mismatch")

        cached = static_alias._static_alias_sessions[0]
        sealed_policy_guards = _policy_mutation_probe(
            cached.fd, consumer_va, REGION_BYTES
        )
        if not bool(sealed_policy_guards["all_rejected_in_use"]):
            raise RuntimeError(
                "sealed alias unexpectedly allowed a policy/residency mutation: "
                f"{sealed_policy_guards}"
            )
        # Preserve the original result key for consumers of older artifacts.
        migrate_guard = sealed_policy_guards["operations"]["migrate_to_gpu"]

        torch.cuda.synchronize()
        del graph
        graph = None
        capture_stream.synchronize()

        destroy_started = time.perf_counter_ns()
        destroyed = static_alias.driver_static_alias_destroy(
            session_id,
            device=0,
            release_backing=True,
            synchronize=True,
        )
        destroy_wall_ms = (time.perf_counter_ns() - destroy_started) / 1e6
        _require(destroyed, "static alias DESTROY")
        session_id = 0

        # cmd99 closes the duplicated static-alias fd. The retained mode-8 fd
        # owns the same VA space and remains live until the legacy slot-session
        # cleanup below, so use it to prove the guards are removed. Every set
        # operation is paired with its inverse, and migration ends on CPU.
        policy_fd = int(uvm._uvm_slot_direct_sessions[0][0])
        post_destroy_policy_recovery = _policy_mutation_probe(
            policy_fd, consumer_va, REGION_BYTES
        )
        if not bool(post_destroy_policy_recovery["all_accepted"]):
            raise RuntimeError(
                "policy/residency mutation remained blocked after DESTROY: "
                f"{post_destroy_policy_recovery}"
            )

        post_destroy_cpu_value = int(
            ctypes.c_uint8.from_address(consumer_va).value
        )
        if post_destroy_cpu_value != 29:
            raise RuntimeError(
                "CPU access after DESTROY did not observe original backing: "
                f"{post_destroy_cpu_value}"
            )

        # Graph H2D updated only VIDMEM. Explicit teardown must restore the
        # original CPU-backed managed allocation at both original VAs. The
        # recovery probe's final migrate-to-CPU also verifies the restored
        # range remains a normal UVM allocation after policy round-trips.
        upload_after = upload.clone()
        consumer_after = consumer.clone()
        torch.cuda.synchronize()
        remote_restored = _all_equal(upload_after, 17) and _all_equal(consumer_after, 29)
        if not remote_restored:
            raise RuntimeError("DESTROY did not restore original CPU backing")

        cleanup = uvm.close_uvm_slot_direct_session(device=0)
        return {
            "ok": True,
            "mechanism": "uvm_original_va_static_alias_graph",
            "bytes_per_alias": REGION_BYTES,
            "alias_count": 2,
            "pte_mode": "64K" if use_64k else "4K",
            "original_vas": {
                "allocation": allocation_va,
                "arena": arena_va,
                "upload_rw": upload_va,
                "consumer_ro": consumer_va,
                "stable": stable_vas
                == (
                    int(upload.data_ptr()),
                    int(consumer.data_ptr()),
                    int(arena.data_ptr()),
                    int(allocation.data_ptr()),
                ),
            },
            "preallocate": preallocate,
            "map": mapped,
            "seal": sealed,
            "migrate_guard": migrate_guard,
            "cpu_fault_guard": cpu_fault_guard,
            "sealed_policy_guards": sealed_policy_guards,
            "graph": {
                "capture_ms": capture_wall_ms,
                "correctness_patterns": correctness_patterns,
                "correctness_replay_ms": correctness_ms,
                "correctness_replay_median_ms": statistics.median(correctness_ms),
                "measured_replays": replays,
                "gpu_total_ms": gpu_total_ms,
                "gpu_ms_per_replay": gpu_total_ms / replays,
                "host_submit_total_ms": submit_wall_ms,
                "host_submit_us_per_replay": submit_wall_ms * 1000 / replays,
                "custom_static_alias_ioctls_during_replay": replay_ioctl_calls,
            },
            "lifecycle_ms": {
                "map": map_wall_ms,
                "seal": seal_wall_ms,
                "destroy": destroy_wall_ms,
            },
            "destroy": destroyed,
            "post_destroy_policy_recovery": post_destroy_policy_recovery,
            "post_destroy_cpu_value": post_destroy_cpu_value,
            "remote_cpu_backing_restored": remote_restored,
            "legacy_slot_session_cleanup": cleanup,
        }
    finally:
        if graph is not None:
            torch.cuda.synchronize()
            del graph
        if session_id:
            cleanup["static_alias_fallback_destroy"] = (
                static_alias.driver_static_alias_destroy(
                    session_id,
                    device=0,
                    release_backing=True,
                    synchronize=True,
                )
            )
        if 0 in uvm._uvm_slot_direct_sessions:
            cleanup["legacy_slot_fallback_close"] = uvm.close_uvm_slot_direct_session(
                device=0
            )
        del source, graph_out, consumer, upload, arena, allocation
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        uvm.release_uvm_local_memory_pool()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replays <= 0:
        raise ValueError("--replays must be positive")

    started = time.time()
    try:
        result = run(args.replays)
    except Exception as exc:
        result = {"ok": False, "error": repr(exc)}
        raise
    finally:
        result["started_unix"] = started
        result["completed_unix"] = time.time()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
