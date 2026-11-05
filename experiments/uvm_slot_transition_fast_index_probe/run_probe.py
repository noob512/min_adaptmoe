#!/usr/bin/env python3
"""Exercise mode-20 validation atomically with fast-index disabled or enabled."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
VLLM_ROOT = REPO_ROOT / "workloads" / "vllm" / "vllm"
LEGACY_DIR = REPO_ROOT / "experiments" / "uvm_512x4m_batch_compare"
DEFAULT_LIBRARY = LEGACY_DIR / "libuvm_batch_shim.so"
DEFAULT_FAST_INDEX = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_transition_fast_index"
)

# The public transition wrapper reads this gate for every call. Invalid cases
# deliberately bypass its duplicate/overlap validation and issue ioctl 91
# directly so the kernel's mode-20 validator is what the probe exercises.
os.environ["VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION"] = "1"
sys.path.insert(0, str(VLLM_ROOT))
sys.path.insert(0, str(LEGACY_DIR))

import benchmark as legacy
import vllm.device_allocator.uvm as uvm_module
from vllm.device_allocator.uvm import (
    UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT,
    UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED,
    close_uvm_slot_direct_session,
    driver_slot_preallocate_ranges,
    driver_slot_transition_ranges,
)

PAGE_BYTES = 4096
DEFAULT_SLOTS = 2
MAX_SLOTS = 1152
DEFAULT_SLOT_BYTES = 512 * 1024
DRIVER_BANK_LAYER = (1 << 32) - 1
MODE_LAYER_TRANSITION = 20
MODE_IOCTL = 91
REQUIRED_TRANSITION_FLAGS = (
    UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT
    | UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
)

ENTRY_OK = 0
ENTRY_INVALID_ARGUMENT = 1
ENTRY_SLOT_CONFLICT = 3
ENTRY_REMAP_FAILED = 7

SEED_INITIAL = 0x1701
SEED_FIRST_VALID = 0x2702
SEED_INVALID = 0x3703
SEED_FINAL_VALID = 0x4704
OWNER_INITIAL = 0x10000
OWNER_FIRST_VALID = 0x20000
OWNER_FINAL_VALID = 0x40000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_cuda(library: ctypes.CDLL, status: int, operation: str) -> None:
    legacy.check_cuda(library, int(status), operation)


def dataset_pointers(
    library: ctypes.CDLL,
    dataset: int,
    count: int,
) -> tuple[int, ...]:
    pointers = tuple(
        int(library.uvm512_dataset_pointer(dataset, index))
        for index in range(count)
    )
    require(all(pointers), "dataset returned a null managed VA")
    return pointers


def dataset_checksum(
    library: ctypes.CDLL,
    dataset: int,
    stream: int,
) -> tuple[int, int]:
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
    actual = ctypes.c_uint64()
    check_cuda(
        library,
        library.uvm512_dataset_read_checksum(dataset, ctypes.byref(actual)),
        "read checksum",
    )
    expected = int(library.uvm512_dataset_expected_checksum(dataset))
    return int(actual.value), expected


def create_dataset(
    library: ctypes.CDLL,
    *,
    count: int,
    allocation_bytes: int,
    guard_bytes: int,
    seed: int,
) -> int:
    dataset = int(
        library.uvm512_dataset_create(count, allocation_bytes, guard_bytes)
    )
    require(dataset != 0, f"dataset_create failed for seed 0x{seed:x}")
    check_cuda(
        library,
        library.uvm512_dataset_reseed(dataset, seed),
        f"reseed dataset 0x{seed:x}",
    )
    return dataset


def make_layer_entries(
    pointers: tuple[int, ...],
    slot_bytes: int,
    *,
    owner_base: int,
    reuse_required: bool,
) -> list[tuple[int, ...]]:
    flags = UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT
    if reuse_required:
        flags |= UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
    return [
        (
            pointer,
            slot_bytes,
            DRIVER_BANK_LAYER,
            owner_base + index,
            index + 1,
            0,
            flags,
        )
        for index, pointer in enumerate(pointers)
    ]


def summarize(result: dict[str, object]) -> dict[str, object]:
    keys = (
        "ok",
        "reason",
        "mode",
        "requested_count",
        "success_count",
        "failure_count",
        "conflict_count",
        "requested_bytes",
        "accepted_bytes",
        "remap_count",
        "tlb_invalidate_count",
        "copy_push_count",
        "owner_cutover",
        "owner_cutover_count",
        "owner_cutover_trusted",
        "owner_state_indeterminate",
        "ioctl_calls",
        "ioctl_wall_ns",
        "wall_ns",
        "errno",
        "rm_status",
        "entry_statuses",
    )
    return {key: result.get(key) for key in keys}


def require_preallocate(
    result: dict[str, object],
    *,
    slots: int,
    logical_bytes: int,
) -> None:
    require(bool(result.get("ok")), f"mode 8 preallocate failed: {result}")
    require(int(result.get("mode", -1)) == 8, "preallocate mode mismatch")
    require(int(result.get("ioctl_calls", -1)) == 1, "preallocate was not one ioctl")
    require(int(result.get("success_count", -1)) == slots, "preallocate short success")
    require(int(result.get("failure_count", -1)) == 0, "preallocate failure")
    require(int(result.get("requested_bytes", -1)) == logical_bytes, "preallocate byte mismatch")
    require(int(result.get("accepted_bytes", -1)) == logical_bytes, "preallocate accepted mismatch")
    require(int(result.get("remap_count", -1)) == 0, "preallocate unexpectedly remapped")
    require(result.get("entry_statuses") == [ENTRY_OK] * slots, "preallocate status mismatch")


def require_valid_transition(
    result: dict[str, object],
    *,
    slots: int,
    logical_bytes: int,
    operation: str,
) -> None:
    require(bool(result.get("ok")), f"{operation} failed: {result}")
    require(int(result.get("mode", -1)) == MODE_LAYER_TRANSITION, f"{operation} mode mismatch")
    require(int(result.get("ioctl_calls", -1)) == 1, f"{operation} was not one ioctl")
    require(int(result.get("success_count", -1)) == slots, f"{operation} short success")
    require(int(result.get("failure_count", -1)) == 0, f"{operation} failure")
    require(int(result.get("requested_bytes", -1)) == logical_bytes, f"{operation} byte mismatch")
    require(int(result.get("accepted_bytes", -1)) == logical_bytes, f"{operation} accepted mismatch")
    require(int(result.get("remap_count", -1)) == slots, f"{operation} cutover mismatch")
    require(int(result.get("tlb_invalidate_count", 0)) > 0, f"{operation} missing TLB invalidate")
    require(int(result.get("copy_push_count", 0)) > 0, f"{operation} missing copy push")
    require(result.get("entry_statuses") == [ENTRY_OK] * slots, f"{operation} status mismatch")
    require(result.get("owner_cutover") is True, f"{operation} cutover not reported")
    require(result.get("owner_cutover_trusted") is True, f"{operation} cutover untrusted")
    require(result.get("owner_state_indeterminate") is False, f"{operation} owner indeterminate")


def raw_mode20_ioctl(
    records: list[tuple[int, int, int, int, int, int, int]],
    *,
    device: int,
) -> dict[str, object]:
    require(0 < len(records) <= MAX_SLOTS, "raw mode-20 record count is invalid")
    with uvm_module._uvm_slot_direct_lock:
        cached = uvm_module._uvm_slot_direct_sessions.get(int(device))
        require(cached is not None, "raw mode-20 ioctl requires an active slot session")
        fd, uuid_bytes = cached
        params = uvm_module._MoeSlotRemapBatchParams()
        params.mode = MODE_LAYER_TRANSITION
        params.count = len(records)
        params.gpu_uuid = uvm_module._PhaseFUuid(
            (ctypes.c_uint8 * len(uuid_bytes)).from_buffer_copy(uuid_bytes)
        )
        params.validate_managed = 1
        for index, record in enumerate(records):
            base, length, layer, expert, slot, slot_offset, flags = record
            entry = params.entries[index]
            entry.layer_id = layer
            entry.expert_id = expert
            entry.slot_id = slot
            entry.role = 0
            entry.va_base = base
            entry.bytes = length
            entry.slot_offset = slot_offset
            entry.flags = flags

        ctypes.set_errno(0)
        start_ns = time.perf_counter_ns()
        ret = uvm_module._libc_ioctl(fd, MODE_IOCTL, params)
        wall_ns = time.perf_counter_ns() - start_ns
        errno_value = ctypes.get_errno() if int(ret) != 0 else 0

    return {
        "ioctl_ret": int(ret),
        "errno": int(errno_value),
        "rm_status": int(params.rmStatus),
        "count": int(params.count),
        "success_count": int(params.success_count),
        "failure_count": int(params.failure_count),
        "conflict_count": int(params.conflict_count),
        "requested_bytes": int(params.requested_bytes),
        "accepted_bytes": int(params.accepted_bytes),
        "remap_count": int(params.remap_count),
        "tlb_invalidate_count": int(params.tlb_invalidate_count),
        "copy_push_count": int(params.reserved),
        "entry_statuses": [
            int(params.entries[index].status) for index in range(int(params.count))
        ],
        "wall_ns": wall_ns,
    }


def require_atomic_rejection(
    result: dict[str, object],
    *,
    operation: str,
    expected_statuses: list[int],
    expected_bytes: int,
    expected_conflicts: int,
) -> None:
    require(int(result.get("count", -1)) == len(expected_statuses), f"{operation} count mismatch")
    require(int(result.get("success_count", -1)) == 0, f"{operation} reported success")
    require(
        int(result.get("failure_count", -1)) == len(expected_statuses),
        f"{operation} failure-count mismatch: {result}",
    )
    require(
        int(result.get("conflict_count", -1)) == expected_conflicts,
        f"{operation} conflict-count mismatch: {result}",
    )
    require(
        int(result.get("requested_bytes", -1)) == expected_bytes,
        f"{operation} requested-byte mismatch",
    )
    require(int(result.get("accepted_bytes", -1)) == 0, f"{operation} accepted bytes")
    require(int(result.get("remap_count", -1)) == 0, f"{operation} changed owner")
    require(int(result.get("tlb_invalidate_count", -1)) == 0, f"{operation} changed PTEs")
    require(int(result.get("copy_push_count", -1)) == 0, f"{operation} submitted a copy")
    require(int(result.get("rm_status", 0)) != 0, f"{operation} returned NV_OK")
    require(
        result.get("entry_statuses") == expected_statuses,
        f"{operation} exact statuses mismatch: {result}",
    )
    require(
        all(status != ENTRY_OK for status in expected_statuses),
        f"{operation} expected status vector contains success",
    )


def read_fast_index(path: Path, expected: int | None) -> int:
    require(path.is_file(), f"missing fast-index parameter: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    require(raw in ("0", "1"), f"invalid fast-index parameter value: {raw!r}")
    actual = int(raw)
    if expected is not None:
        require(actual == expected, f"fast-index is {actual}, expected {expected}")
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--slots", type=int, default=DEFAULT_SLOTS)
    parser.add_argument("--slot-bytes", type=int, default=DEFAULT_SLOT_BYTES)
    parser.add_argument("--guard-bytes", type=int, default=DEFAULT_SLOT_BYTES)
    parser.add_argument("--fast-index-path", type=Path, default=DEFAULT_FAST_INDEX)
    parser.add_argument("--expected-fast-index", type=int, choices=(0, 1))
    args = parser.parse_args()

    require(2 <= args.slots <= MAX_SLOTS, f"slots must be in [2, {MAX_SLOTS}]")
    require(
        0 < args.slot_bytes <= 4 * 1024 * 1024
        and args.slot_bytes % (4 * PAGE_BYTES) == 0,
        "slot-bytes must be <=4 MiB and divisible by 16 KiB",
    )
    require(args.guard_bytes >= 0, "guard-bytes must be non-negative")
    require(args.library.is_file(), f"missing library: {args.library}")

    fast_index = read_fast_index(args.fast_index_path, args.expected_fast_index)
    logical_bytes = args.slots * args.slot_bytes
    half_slot = args.slot_bytes // 2
    quarter_slot = args.slot_bytes // 4
    library = legacy.load_library(args.library)
    check_cuda(library, library.uvm512_set_device(args.device), "set device")

    datasets: list[int] = []
    stream = 0
    closed = False
    failure: str | None = None
    payload: dict[str, Any] = {
        "status": "failed",
        "fast_index": fast_index,
        "slots": args.slots,
        "slot_bytes": args.slot_bytes,
        "logical_bytes": logical_bytes,
        "module_srcversion": Path("/sys/module/nvidia_uvm/srcversion")
        .read_text(encoding="utf-8")
        .strip(),
    }

    try:
        dataset_initial = create_dataset(
            library,
            count=args.slots,
            allocation_bytes=args.slot_bytes,
            guard_bytes=args.guard_bytes,
            seed=SEED_INITIAL,
        )
        datasets.append(dataset_initial)
        dataset_first = create_dataset(
            library,
            count=args.slots,
            allocation_bytes=args.slot_bytes,
            guard_bytes=args.guard_bytes,
            seed=SEED_FIRST_VALID,
        )
        datasets.append(dataset_first)
        dataset_invalid = create_dataset(
            library,
            count=2,
            allocation_bytes=args.slot_bytes,
            guard_bytes=args.guard_bytes,
            seed=SEED_INVALID,
        )
        datasets.append(dataset_invalid)
        dataset_final = create_dataset(
            library,
            count=args.slots,
            allocation_bytes=args.slot_bytes,
            guard_bytes=args.guard_bytes,
            seed=SEED_FINAL_VALID,
        )
        datasets.append(dataset_final)

        pointers_initial = dataset_pointers(library, dataset_initial, args.slots)
        pointers_first = dataset_pointers(library, dataset_first, args.slots)
        pointers_invalid = dataset_pointers(library, dataset_invalid, 2)
        pointers_final = dataset_pointers(library, dataset_final, args.slots)
        original_pointers = (
            pointers_initial + pointers_first + pointers_invalid + pointers_final
        )
        require(
            len(set(original_pointers)) == len(original_pointers),
            "managed allocation bases are not unique",
        )

        def require_va_stable(stage: str) -> None:
            current = (
                dataset_pointers(library, dataset_initial, args.slots)
                + dataset_pointers(library, dataset_first, args.slots)
                + dataset_pointers(library, dataset_invalid, 2)
                + dataset_pointers(library, dataset_final, args.slots)
            )
            require(current == original_pointers, f"managed VA changed after {stage}")

        initial_entries = make_layer_entries(
            pointers_initial,
            args.slot_bytes,
            owner_base=OWNER_INITIAL,
            reuse_required=False,
        )
        first_entries = make_layer_entries(
            pointers_first,
            args.slot_bytes,
            owner_base=OWNER_FIRST_VALID,
            reuse_required=True,
        )
        final_entries = make_layer_entries(
            pointers_final,
            args.slot_bytes,
            owner_base=OWNER_FINAL_VALID,
            reuse_required=True,
        )

        preallocate = driver_slot_preallocate_ranges(
            initial_entries,
            device=args.device,
        )
        require_preallocate(
            preallocate,
            slots=args.slots,
            logical_bytes=logical_bytes,
        )
        first_valid = driver_slot_transition_ranges(
            first_entries,
            device=args.device,
        )
        require_valid_transition(
            first_valid,
            slots=args.slots,
            logical_bytes=logical_bytes,
            operation="first legal mode-20 transition",
        )
        require_va_stable("first legal transition")

        stream = int(library.uvm512_stream_create())
        require(stream != 0, "stream_create failed")
        checksum_first, checksum_first_expected = dataset_checksum(
            library,
            dataset_first,
            stream,
        )
        require(checksum_first == checksum_first_expected, "first legal checksum mismatch")

        duplicate_same = [
            (
                pointers_invalid[0],
                half_slot,
                DRIVER_BANK_LAYER,
                0x31000,
                1,
                0,
                REQUIRED_TRANSITION_FLAGS,
            ),
            (
                pointers_invalid[1],
                half_slot,
                DRIVER_BANK_LAYER,
                0x31000,
                1,
                half_slot,
                REQUIRED_TRANSITION_FLAGS,
            ),
        ]
        duplicate_same_result = raw_mode20_ioctl(
            duplicate_same,
            device=args.device,
        )
        require_atomic_rejection(
            duplicate_same_result,
            operation="duplicate slot with same expert",
            expected_statuses=[ENTRY_REMAP_FAILED, ENTRY_SLOT_CONFLICT],
            expected_bytes=args.slot_bytes,
            expected_conflicts=1,
        )
        require_va_stable("same-expert duplicate rejection")
        checksum_after_same, _ = dataset_checksum(library, dataset_first, stream)
        require(checksum_after_same == checksum_first_expected, "same-expert rejection changed data")

        duplicate_different = [
            (
                pointers_invalid[0],
                half_slot,
                DRIVER_BANK_LAYER,
                0x32000,
                1,
                0,
                REQUIRED_TRANSITION_FLAGS,
            ),
            (
                pointers_invalid[1],
                half_slot,
                DRIVER_BANK_LAYER,
                0x32001,
                1,
                half_slot,
                REQUIRED_TRANSITION_FLAGS,
            ),
        ]
        duplicate_different_result = raw_mode20_ioctl(
            duplicate_different,
            device=args.device,
        )
        require_atomic_rejection(
            duplicate_different_result,
            operation="duplicate slot with different expert",
            expected_statuses=[ENTRY_REMAP_FAILED, ENTRY_SLOT_CONFLICT],
            expected_bytes=args.slot_bytes,
            expected_conflicts=1,
        )
        require_va_stable("different-expert duplicate rejection")
        checksum_after_different, _ = dataset_checksum(library, dataset_first, stream)
        require(
            checksum_after_different == checksum_first_expected,
            "different-expert rejection changed data",
        )

        source_overlap = [
            (
                pointers_invalid[0],
                half_slot,
                DRIVER_BANK_LAYER,
                0x33000,
                1,
                0,
                REQUIRED_TRANSITION_FLAGS,
            ),
            (
                pointers_invalid[0] + quarter_slot,
                half_slot,
                DRIVER_BANK_LAYER,
                0x33001,
                2,
                0,
                REQUIRED_TRANSITION_FLAGS,
            ),
        ]
        source_overlap_result = raw_mode20_ioctl(
            source_overlap,
            device=args.device,
        )
        require_atomic_rejection(
            source_overlap_result,
            operation="overlapping source VA",
            expected_statuses=[ENTRY_REMAP_FAILED, ENTRY_INVALID_ARGUMENT],
            expected_bytes=args.slot_bytes,
            expected_conflicts=0,
        )
        require_va_stable("source-overlap rejection")
        checksum_after_overlap, _ = dataset_checksum(library, dataset_first, stream)
        require(checksum_after_overlap == checksum_first_expected, "overlap rejection changed data")

        expected_first_owners = {
            (DRIVER_BANK_LAYER, index + 1): OWNER_FIRST_VALID + index
            for index in range(args.slots)
        }
        require(
            uvm_module._uvm_slot_direct_owners.get(args.device)
            == expected_first_owners,
            "raw rejected ioctls changed userspace owner state",
        )

        final_valid = driver_slot_transition_ranges(
            final_entries,
            device=args.device,
        )
        require_valid_transition(
            final_valid,
            slots=args.slots,
            logical_bytes=logical_bytes,
            operation="final legal mode-20 transition",
        )
        require_va_stable("final legal transition")
        checksum_final, checksum_final_expected = dataset_checksum(
            library,
            dataset_final,
            stream,
        )
        require(checksum_final == checksum_final_expected, "final legal checksum mismatch")
        require(
            checksum_final != checksum_first,
            "first and final datasets do not have distinct signatures",
        )

        close_result = close_uvm_slot_direct_session(device=args.device)
        require(bool(close_result.get("ok")), f"slot close failed: {close_result}")
        closed = True
        require_va_stable("slot close")
        payload.update(
            {
                "status": "pass",
                "va_stable": True,
                "checksum_first": checksum_first,
                "checksum_first_expected": checksum_first_expected,
                "checksum_final": checksum_final,
                "checksum_final_expected": checksum_final_expected,
                "preallocate": summarize(preallocate),
                "first_valid_transition": summarize(first_valid),
                "duplicate_same_expert": duplicate_same_result,
                "duplicate_different_expert": duplicate_different_result,
                "source_va_overlap": source_overlap_result,
                "final_valid_transition": summarize(final_valid),
                "close": summarize(close_result),
            }
        )
    except Exception as error:  # noqa: BLE001
        failure = f"{type(error).__name__}: {error}"
        payload["error"] = failure
    finally:
        if not closed:
            try:
                cleanup = close_uvm_slot_direct_session(device=args.device)
                payload["cleanup_close"] = summarize(cleanup)
            except Exception as cleanup_error:  # noqa: BLE001
                payload["cleanup_close_error"] = (
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
        if stream:
            library.uvm512_stream_destroy(stream)
        for dataset in reversed(datasets):
            library.uvm512_dataset_destroy(dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failure is not None:
        print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
