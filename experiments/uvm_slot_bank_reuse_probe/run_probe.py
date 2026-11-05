#!/usr/bin/env python3
"""Validate fixed physical-slot reuse before running the vLLM integration."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
VLLM_ROOT = REPO_ROOT / "workloads" / "vllm" / "vllm"
LEGACY_DIR = REPO_ROOT / "experiments" / "uvm_512x4m_batch_compare"
DEFAULT_LIBRARY = LEGACY_DIR / "libuvm_batch_shim.so"
sys.path.insert(0, str(VLLM_ROOT))
sys.path.insert(0, str(LEGACY_DIR))

import benchmark as legacy  # noqa: E402
from vllm.device_allocator.uvm import (  # noqa: E402
    UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT,
    UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED,
    close_uvm_slot_direct_session,
    driver_slot_preallocate_ranges,
    driver_slot_restore_ranges,
    driver_slot_transition_ranges,
)


DRIVER_BANK_LAYER = (1 << 32) - 1
DATASET_A_SEED = 0x13579BDF
DATASET_B_SEED = 0x2468ACE0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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
    )
    return int(output.strip().splitlines()[0])


def check_cuda(library: ctypes.CDLL, status: int, operation: str) -> None:
    legacy.check_cuda(library, int(status), operation)


def checksum_dataset(
    library: ctypes.CDLL,
    dataset: int,
    stream: int,
) -> tuple[int, int, float]:
    started_ns = time.perf_counter_ns()
    check_cuda(
        library,
        library.uvm512_dataset_launch_checksum(dataset, stream),
        "launch checksum",
    )
    check_cuda(
        library,
        library.uvm512_stream_synchronize(stream),
        "synchronize checksum",
    )
    value = ctypes.c_uint64()
    check_cuda(
        library,
        library.uvm512_dataset_read_checksum(dataset, ctypes.byref(value)),
        "read checksum",
    )
    expected = int(library.uvm512_dataset_expected_checksum(dataset))
    return int(value.value), expected, (time.perf_counter_ns() - started_ns) / 1e6


def dataset_pointers(
    library: ctypes.CDLL,
    dataset: int,
    count: int,
) -> tuple[int, ...]:
    """Re-read every exported VA instead of trusting a cached Python list."""
    return tuple(
        int(library.uvm512_dataset_pointer(dataset, index))
        for index in range(count)
    )


def require_single_layer_ioctl(
    result: dict[str, object],
    *,
    operation: str,
    count: int,
    logical_bytes: int,
    expect_remap: bool,
) -> None:
    require(bool(result.get("ok")), f"{operation} failed")
    require(int(result.get("ioctl_calls", -1)) == 1, f"{operation} was not one ioctl")
    require(int(result.get("requested_count", -1)) == count, f"{operation} count mismatch")
    require(int(result.get("success_count", -1)) == count, f"{operation} short success")
    require(int(result.get("failure_count", -1)) == 0, f"{operation} reported failure")
    require(
        int(result.get("requested_bytes", -1)) == logical_bytes,
        f"{operation} requested-byte mismatch",
    )
    require(
        int(result.get("accepted_bytes", -1)) == logical_bytes,
        f"{operation} accepted-byte mismatch",
    )
    statuses = result.get("entry_statuses")
    require(
        isinstance(statuses, list)
        and len(statuses) == count
        and all(status == 0 for status in statuses),
        f"{operation} entry status mismatch",
    )
    require(
        int(result.get("remap_count", -1)) == (count if expect_remap else 0),
        f"{operation} remap count mismatch",
    )


def make_entries(
    pointers: list[int],
    allocation_bytes: int,
    *,
    owner_token: int,
    reuse_required: bool,
) -> list[tuple[int, ...]]:
    flags = UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT
    if reuse_required:
        flags |= UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
    return [
        (
            pointer,
            allocation_bytes,
            DRIVER_BANK_LAYER,
            owner_token,
            index + 1,
            0,
            flags,
        )
        for index, pointer in enumerate(pointers)
    ]


def summarize_result(result: dict[str, object]) -> dict[str, object]:
    summary = {
        key: result.get(key)
        for key in (
            "ok",
            "reason",
            "requested_count",
            "success_count",
            "failure_count",
            "requested_bytes",
            "accepted_bytes",
            "remap_count",
            "tlb_invalidate_count",
            "copy_push_count",
            "backing_reuse_count",
            "owner_cutover",
            "owner_cutover_count",
            "owner_cutover_trusted",
            "owner_state_indeterminate",
            "ioctl_calls",
            "ioctl_wall_ns",
            "wall_ns",
        )
    }
    statuses = result.get("entry_statuses")
    if isinstance(statuses, list):
        summary["entry_status_count"] = len(statuses)
        summary["bad_entry_status_count"] = sum(status != 0 for status in statuses)
    return summary


def summarize_close(result: dict[str, object]) -> dict[str, object]:
    release = result.get("release")
    return {
        "ok": result.get("ok"),
        "reason": result.get("reason"),
        "closed": result.get("closed"),
        "wall_ns": result.get("wall_ns"),
        "release": summarize_result(release) if isinstance(release, dict) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--slots", type=int, default=1152)
    parser.add_argument("--slot-bytes", type=int, default=512 * 1024)
    parser.add_argument("--guard-bytes", type=int, default=512 * 1024)
    args = parser.parse_args()
    require(args.slots > 0, "slots must be positive")
    require(args.slot_bytes > 0 and args.slot_bytes % 4096 == 0, "invalid slot size")
    require(args.guard_bytes >= 0, "guard size must be non-negative")
    require(args.library.is_file(), f"missing library: {args.library}")
    transition_value = os.environ.get(
        "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION", "0"
    ).strip()
    require(
        transition_value in ("0", "1"),
        "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION must be 0 or 1",
    )
    transition_enabled = transition_value == "1"

    library = legacy.load_library(args.library)
    check_cuda(library, library.uvm512_set_device(args.device), "set device")
    dataset_a = 0
    dataset_b = 0
    stream = 0
    close_result: dict[str, object] | None = None
    payload: dict[str, Any] = {
        "status": "failed",
        "slots": args.slots,
        "slot_bytes": args.slot_bytes,
        "logical_bytes": args.slots * args.slot_bytes,
        "driver_bank_layer": DRIVER_BANK_LAYER,
        "layer_transition_enabled": transition_enabled,
        "module_srcversion": Path("/sys/module/nvidia_uvm/srcversion")
        .read_text(encoding="utf-8")
        .strip(),
    }
    try:
        dataset_a = int(
            library.uvm512_dataset_create(
                args.slots, args.slot_bytes, args.guard_bytes
            )
        )
        require(dataset_a != 0, "failed to allocate dataset A")
        dataset_b = int(
            library.uvm512_dataset_create(
                args.slots, args.slot_bytes, args.guard_bytes
            )
        )
        require(dataset_b != 0, "failed to allocate dataset B")
        check_cuda(
            library,
            library.uvm512_dataset_reseed(dataset_a, DATASET_A_SEED),
            "reseed dataset A",
        )
        check_cuda(
            library,
            library.uvm512_dataset_reseed(dataset_b, DATASET_B_SEED),
            "reseed dataset B",
        )
        expected_seeded_a = int(
            library.uvm512_dataset_expected_checksum(dataset_a)
        )
        expected_seeded_b = int(
            library.uvm512_dataset_expected_checksum(dataset_b)
        )
        require(
            expected_seeded_a != expected_seeded_b,
            "dataset A/B signatures must differ",
        )
        stream = int(library.uvm512_stream_create())
        require(stream != 0, "failed to create CUDA stream")

        pointers_a = list(dataset_pointers(library, dataset_a, args.slots))
        pointers_b = list(dataset_pointers(library, dataset_b, args.slots))
        require(all(pointers_a) and all(pointers_b), "dataset returned a null VA")
        require(len(set(pointers_a + pointers_b)) == 2 * args.slots, "VAs overlap")
        original_pointers = tuple(pointers_a + pointers_b)
        va_requeries: dict[str, bool] = {}

        def verify_va(stage: str) -> None:
            current = dataset_pointers(library, dataset_a, args.slots) + dataset_pointers(
                library, dataset_b, args.slots
            )
            stable = current == original_pointers
            va_requeries[stage] = stable
            require(stable, f"managed virtual addresses changed at {stage}")

        initial_entries = make_entries(
            pointers_a,
            args.slot_bytes,
            owner_token=0xA001,
            reuse_required=False,
        )
        rebound_entries = make_entries(
            pointers_b,
            args.slot_bytes,
            owner_token=0xB002,
            reuse_required=True,
        )
        restore_b_entries = [
            (*entry[:-1], UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT)
            for entry in rebound_entries
        ]

        memory: dict[str, int] = {"before_preallocate": gpu_used_mib(args.device)}
        initial_preallocate = driver_slot_preallocate_ranges(
            initial_entries, device=args.device
        )
        require_single_layer_ioctl(
            initial_preallocate,
            operation="initial mode8",
            count=args.slots,
            logical_bytes=args.slots * args.slot_bytes,
            expect_remap=False,
        )
        require(
            int(initial_preallocate.get("backing_reuse_count", -1)) == 0,
            "initial bank unexpectedly reported reuse",
        )
        verify_va("after_initial_mode8")
        memory["after_initial_preallocate"] = gpu_used_mib(args.device)

        restore_a = driver_slot_restore_ranges(initial_entries, device=args.device)
        require_single_layer_ioctl(
            restore_a,
            operation="first mode10",
            count=args.slots,
            logical_bytes=args.slots * args.slot_bytes,
            expect_remap=True,
        )
        verify_va("after_first_mode10")
        checksum_a, expected_a, checksum_a_ms = checksum_dataset(
            library, dataset_a, stream
        )
        require(checksum_a == expected_a, "dataset A checksum mismatch")
        memory["after_first_restore"] = gpu_used_mib(args.device)

        rebound_preallocate: dict[str, object] | None = None
        if transition_enabled:
            restore_b = driver_slot_transition_ranges(
                rebound_entries, device=args.device
            )
            require_single_layer_ioctl(
                restore_b,
                operation="mode20 layer transition",
                count=args.slots,
                logical_bytes=args.slots * args.slot_bytes,
                expect_remap=True,
            )
            require(
                restore_b.get("owner_cutover") is True
                and restore_b.get("owner_cutover_trusted") is True
                and int(restore_b.get("owner_cutover_count", -1)) == args.slots,
                "mode20 did not prove a complete owner cutover",
            )
            verify_va("after_mode20_transition")
            memory["after_layer_transition"] = gpu_used_mib(args.device)
        else:
            rebound_preallocate = driver_slot_preallocate_ranges(
                rebound_entries, device=args.device
            )
            require_single_layer_ioctl(
                rebound_preallocate,
                operation="rebind mode8",
                count=args.slots,
                logical_bytes=args.slots * args.slot_bytes,
                expect_remap=False,
            )
            require(
                int(rebound_preallocate.get("backing_reuse_count", -1))
                == args.slots,
                "kernel did not acknowledge every retained backing",
            )
            verify_va("after_rebind_mode8")
            memory["after_rebind_preallocate"] = gpu_used_mib(args.device)

            restore_b = driver_slot_restore_ranges(
                restore_b_entries, device=args.device
            )
            require_single_layer_ioctl(
                restore_b,
                operation="second mode10",
                count=args.slots,
                logical_bytes=args.slots * args.slot_bytes,
                expect_remap=True,
            )
            verify_va("after_second_mode10")
        checksum_b, expected_b, checksum_b_ms = checksum_dataset(
            library, dataset_b, stream
        )
        require(checksum_b == expected_b, "dataset B checksum mismatch")
        require(checksum_a != checksum_b, "dataset A/B GPU checksums are not distinct")
        memory["after_second_restore"] = gpu_used_mib(args.device)

        close_result = close_uvm_slot_direct_session(device=args.device)
        require(bool(close_result.get("ok")), "slot session close failed")
        verify_va("after_slot_close")
        memory["after_slot_close"] = gpu_used_mib(args.device)

        initial_ms = int(initial_preallocate["wall_ns"]) / 1e6
        rebind_ms = (
            int(rebound_preallocate["wall_ns"]) / 1e6
            if rebound_preallocate is not None
            else None
        )
        transition_ms = (
            int(restore_b["wall_ns"]) / 1e6 if transition_enabled else None
        )
        payload.update(
            {
                "status": "pass",
                "va_stable": True,
                "va_requeries": va_requeries,
                "dataset_a_seed": DATASET_A_SEED,
                "dataset_b_seed": DATASET_B_SEED,
                "checksum_a": checksum_a,
                "checksum_a_expected": expected_a,
                "checksum_a_ms": checksum_a_ms,
                "checksum_b": checksum_b,
                "checksum_b_expected": expected_b,
                "checksum_b_ms": checksum_b_ms,
                "initial_preallocate": summarize_result(initial_preallocate),
                "first_restore": summarize_result(restore_a),
                "rebind_preallocate": (
                    summarize_result(rebound_preallocate)
                    if rebound_preallocate is not None
                    else None
                ),
                "layer_transition": (
                    summarize_result(restore_b) if transition_enabled else None
                ),
                "second_restore": summarize_result(restore_b),
                "close": summarize_close(close_result),
                "memory_used_mib": memory,
                "mode8_initial_ms": initial_ms,
                "mode8_rebind_ms": rebind_ms,
                "mode8_rebind_speedup": (
                    initial_ms / rebind_ms if rebind_ms is not None else None
                ),
                "mode20_transition_ms": transition_ms,
                "rebind_memory_growth_mib": (
                    memory[
                        "after_layer_transition"
                        if transition_enabled
                        else "after_rebind_preallocate"
                    ]
                    - memory["after_first_restore"]
                ),
            }
        )
    finally:
        if close_result is None:
            try:
                close_uvm_slot_direct_session(device=args.device)
            except Exception:
                pass
        if stream:
            library.uvm512_stream_destroy(stream)
        if dataset_b:
            library.uvm512_dataset_destroy(dataset_b)
        if dataset_a:
            library.uvm512_dataset_destroy(dataset_a)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
