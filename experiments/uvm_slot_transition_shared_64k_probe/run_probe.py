#!/usr/bin/env python3
"""Verify a shared-push + 64K fixed-bank A-to-B layer transition.

Use ``--blocks 1`` for the smoke case and ``--blocks 288`` for the full
576 MiB/1,152-slice layer shape. The initial A placement uses mode 21 and the
owner cutover to B uses mode 20 with the shared-push and 64K strategy flags.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
VLLM_ROOT = REPO_ROOT / "workloads" / "vllm" / "vllm"
LEGACY_DIR = REPO_ROOT / "experiments" / "uvm_512x4m_batch_compare"
DEFAULT_LIBRARY = LEGACY_DIR / "libuvm_batch_shim.so"
DEFAULT_STATS = Path("/proc/driver/nvidia-uvm/latency_stats")
DEFAULT_DIRECT_CUTOVER = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_transition_direct_cutover"
)

# The low-level wrappers derive mode 21 and the mode-20 strategy flags from
# these gates for every call. Set them before importing the wrapper as an
# additional guard against future import-time configuration caching.
os.environ["VLLM_UVM_SLOT_DIRECT_SHARED_PUSH"] = "1"
os.environ["VLLM_UVM_SLOT_DIRECT_64K_PTE"] = "1"
os.environ["VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION"] = "1"
sys.path.insert(0, str(VLLM_ROOT))
sys.path.insert(0, str(LEGACY_DIR))

import benchmark as legacy
from vllm.device_allocator.uvm import (
    UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT,
    UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED,
    close_uvm_slot_direct_session,
    driver_slot_preallocate_ranges,
    driver_slot_restore_ranges,
    driver_slot_transition_ranges,
)


MODE_VIDMEM_PREALLOC = 8
MODE_LAYER_TRANSITION = 20
MODE_SHARED_PUSH_64K = 21
BLOCK_BYTES = 2 * 1024 * 1024
SLICE_BYTES = 512 * 1024
SLICES_PER_BLOCK = BLOCK_BYTES // SLICE_BYTES
PTE_BYTES = 64 * 1024
PTE_PER_BLOCK = BLOCK_BYTES // PTE_BYTES
DEFAULT_BLOCKS = 288
MAX_BLOCKS = 288
DRIVER_BANK_LAYER = 0x64C1
OWNER_A = 0x10000
OWNER_B = 0x20000
SEED_A = 0x51A1
SEED_B = 0x62B2
MIN_PUSH_REDUCTION = 4

_SECTION_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):\s*$")
_PROFILE_RE = re.compile(
    r"^\s+([a-z][a-z0-9_]*)\s+"
    r"calls=(\d+)\s+ns=(\d+)\s+avg_ns=(\d+)\s+"
    r"pages=(\d+)\s+bytes=(\d+)\s*$"
)
_COUNTER_RE = re.compile(
    r"^\s*([a-z][a-z0-9_]*)\s+(0x[0-9a-fA-F]+|\d+)\s*$"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_cuda(library: ctypes.CDLL, status: int, operation: str) -> None:
    legacy.check_cuda(library, int(status), operation)


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
            require(section is not None, f"profile row before section: {line_number}")
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
        raise RuntimeError(
            f"unrecognized latency_stats line {line_number}: {line!r}"
        )
    return values


def read_stats(path: Path) -> dict[str, int]:
    return parse_stats(path.read_text(encoding="utf-8"))


def stats_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    delta = {
        name: int(after.get(name, 0)) - int(before.get(name, 0))
        for name in set(before) | set(after)
    }
    require(all(value >= 0 for value in delta.values()), "driver counters moved backwards")
    return delta


def create_dataset(
    library: ctypes.CDLL,
    blocks: int,
    seed: int,
) -> int:
    dataset = int(library.uvm512_dataset_create(blocks, BLOCK_BYTES, 0))
    require(dataset != 0, f"dataset_create failed for seed 0x{seed:x}")
    check_cuda(
        library,
        library.uvm512_dataset_reseed(dataset, seed),
        f"reseed 0x{seed:x}",
    )
    return dataset


def dataset_pointers(
    library: ctypes.CDLL,
    dataset: int,
    blocks: int,
) -> tuple[int, ...]:
    pointers = tuple(
        int(library.uvm512_dataset_pointer(dataset, index))
        for index in range(blocks)
    )
    require(all(pointers), "dataset returned a null managed VA")
    require(len(set(pointers)) == blocks, "dataset VAs overlap")
    require(all(pointer % BLOCK_BYTES == 0 for pointer in pointers), "2 MiB VA alignment failed")
    return pointers


def sliced_entries(
    pointers: tuple[int, ...],
    *,
    owner_base: int,
    reuse_required: bool,
) -> list[tuple[int, ...]]:
    flags = UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT
    if reuse_required:
        flags |= UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
    entries: list[tuple[int, ...]] = []
    for block_index, pointer in enumerate(pointers):
        for slice_index in range(SLICES_PER_BLOCK):
            entry_index = block_index * SLICES_PER_BLOCK + slice_index
            entries.append(
                (
                    pointer + slice_index * SLICE_BYTES,
                    SLICE_BYTES,
                    DRIVER_BANK_LAYER,
                    owner_base + entry_index,
                    entry_index + 1,
                    0,
                    flags,
                )
            )
    return entries


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
    return int(actual.value), int(library.uvm512_dataset_expected_checksum(dataset))


def require_preallocate(
    result: Mapping[str, object],
    *,
    entries: int,
    logical_bytes: int,
) -> None:
    expected = {
        "mode": MODE_VIDMEM_PREALLOC,
        "requested_count": entries,
        "success_count": entries,
        "failure_count": 0,
        "requested_bytes": logical_bytes,
        "accepted_bytes": logical_bytes,
        "remap_count": 0,
        "ioctl_calls": 1,
    }
    require(bool(result.get("ok")), f"mode 8 preallocate failed: {result}")
    for name, value in expected.items():
        require(int(result.get(name, -1)) == value, f"mode 8 {name} mismatch")
    require(result.get("entry_statuses") == [0] * entries, "mode 8 status mismatch")


def require_copy_map_result(
    result: Mapping[str, object],
    *,
    mode: int,
    blocks: int,
    entries: int,
    logical_bytes: int,
    transition: bool,
) -> None:
    label = "mode 20 transition" if transition else "mode 21 restore"
    expected = {
        "mode": mode,
        "requested_count": entries,
        "success_count": entries,
        "failure_count": 0,
        "requested_bytes": logical_bytes,
        "accepted_bytes": logical_bytes,
        "remap_count": entries,
        "tlb_invalidate_count": blocks,
        "ioctl_calls": 1,
    }
    require(bool(result.get("ok")), f"{label} failed: {result}")
    for name, value in expected.items():
        require(int(result.get(name, -1)) == value, f"{label} {name} mismatch")
    pushes = int(result.get("copy_push_count", -1))
    require(
        0 < pushes <= max(1, entries // MIN_PUSH_REDUCTION),
        f"{label} shared-push reduction is below {MIN_PUSH_REDUCTION}x",
    )
    require(result.get("entry_statuses") == [0] * entries, f"{label} status mismatch")
    require(
        result.get("owner_cutover") is transition
        and int(result.get("owner_cutover_count", -1))
        == (entries if transition else 0)
        and result.get("owner_cutover_trusted") is True
        and result.get("owner_state_indeterminate") is False,
        f"{label} owner-cutover report mismatch",
    )


def require_profile(
    delta: Mapping[str, int],
    result: Mapping[str, object],
    *,
    blocks: int,
    entries: int,
    logical_bytes: int,
    operation: str,
    direct_cutover: bool = False,
) -> None:
    pte_entries = blocks * PTE_PER_BLOCK
    exact = {
        "slot_direct_ioctl_calls": 1,
        "slot_direct_per_block_calls": 0,
        "slot_direct_shared_calls": 1,
        "slot_direct_successes": 1,
        "slot_direct_fallbacks": 0,
        "slot_direct_errors": 0,
        "slot_direct_requested_ranges": entries,
        "slot_direct_completed_ranges": entries,
        "slot_direct_blocks": blocks,
        "slot_direct_pte_4k_blocks": 0,
        "slot_direct_pte_64k_blocks": blocks,
        "slot_direct_pte_2m_blocks": 0,
        "slot_direct_pte_write_entries": pte_entries,
        "slot_direct_requested_bytes": logical_bytes,
        "slot_direct_completed_bytes": logical_bytes,
        "slot_direct_pte_map_ranges": entries,
        "slot_direct_pte_map_bytes": logical_bytes,
        "slot_direct_vidmem_allocations": 0,
        "slot_direct_vidmem_evictions": 0,
        "slot_direct_transition_fast_index_calls": 1 if operation.startswith("mode 20") else 0,
        "slot_direct_transition_group_clear_calls": 1 if operation.startswith("mode 20") else 0,
        "slot_direct_transition_group_clear_fallbacks": 0,
        "slot_direct_transition_clear_ranges": entries if operation.startswith("mode 20") else 0,
        "slot_direct_transition_clear_unique_blocks": blocks if operation.startswith("mode 20") else 0,
        "slot_direct_transition_direct_cutover_calls": (
            1 if operation.startswith("mode 20") and direct_cutover else 0
        ),
        "slot_direct_transition_direct_cutover_pages": (
            logical_bytes // 4096
            if operation.startswith("mode 20") and direct_cutover
            else 0
        ),
        "slot_direct_profile.total.calls": 1,
    }
    for name, value in exact.items():
        require(name in delta, f"{operation}: missing counter {name}")
        require(
            int(delta[name]) == value,
            f"{operation}: {name}={delta[name]} expected {value}",
        )
    pushes = int(delta.get("slot_direct_copy_pushes", -1))
    require(
        pushes == int(result.get("copy_push_count", -2)),
        f"{operation}: wrapper/profile push count mismatch",
    )
    require(
        0 < pushes <= max(1, entries // MIN_PUSH_REDUCTION),
        f"{operation}: profile does not prove {MIN_PUSH_REDUCTION}x push reduction",
    )
    require(
        int(delta.get("slot_direct_memcopy_methods", -1)) >= entries,
        f"{operation}: too few CE copy methods",
    )
    if operation.startswith("mode 20"):
        expected_restore_calls = 0 if direct_cutover else blocks
        expected_restore_bytes = 0 if direct_cutover else logical_bytes
        for stage in ("migrate_profile.map_gpu_to", "migrate_profile.map_total"):
            require(
                int(delta.get(f"{stage}.calls", -1)) == expected_restore_calls,
                f"{operation}: {stage}.calls does not prove CPU-map policy",
            )
            require(
                int(delta.get(f"{stage}.bytes", -1)) == expected_restore_bytes,
                f"{operation}: {stage}.bytes does not prove CPU-map policy",
            )
        require(
            int(delta.get("migrate_profile.unmap_gpu.calls", -1)) == blocks,
            f"{operation}: old slot unmap count mismatch",
        )
        require(
            int(delta.get("migrate_profile.unmap_gpu.bytes", -1)) == logical_bytes,
            f"{operation}: old slot unmap bytes mismatch",
        )


def summarize(result: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "ok",
        "reason",
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
        "ioctl_calls",
        "ioctl_wall_ns",
        "wall_ns",
    )
    return {key: result.get(key) for key in keys}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument(
        "--direct-cutover-parameter",
        type=Path,
        default=DEFAULT_DIRECT_CUTOVER,
    )
    parser.add_argument("--expect-direct-cutover", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    args = parser.parse_args()
    require(1 <= args.blocks <= MAX_BLOCKS, f"blocks must be in [1, {MAX_BLOCKS}]")
    require(args.library.is_file(), f"missing library: {args.library}")
    require(args.stats.is_file(), f"missing stats: {args.stats}")
    require(
        args.direct_cutover_parameter.is_file(),
        f"missing direct-cutover parameter: {args.direct_cutover_parameter}",
    )
    direct_cutover_value = int(
        args.direct_cutover_parameter.read_text(encoding="utf-8").strip()
    )
    require(
        direct_cutover_value in (0, 1),
        f"invalid direct-cutover parameter: {direct_cutover_value}",
    )
    require(
        not args.expect_direct_cutover or direct_cutover_value == 1,
        "direct-cutover was required but is disabled",
    )

    entries_count = args.blocks * SLICES_PER_BLOCK
    logical_bytes = args.blocks * BLOCK_BYTES
    pte_entries = args.blocks * PTE_PER_BLOCK
    library = legacy.load_library(args.library)
    check_cuda(library, library.uvm512_set_device(args.device), "set device")

    datasets: list[int] = []
    stream = 0
    closed = False
    failure: str | None = None
    payload: dict[str, Any] = {
        "status": "failed",
        "blocks": args.blocks,
        "slices_per_block": SLICES_PER_BLOCK,
        "entries": entries_count,
        "logical_bytes": logical_bytes,
        "expected_64k_ptes_per_operation": pte_entries,
        "expected_modes": {"initial_restore": 21, "layer_transition": 20},
        "module_srcversion": Path("/sys/module/nvidia_uvm/srcversion")
        .read_text(encoding="utf-8")
        .strip(),
        "direct_cutover_parameter": direct_cutover_value,
    }

    try:
        dataset_a = create_dataset(library, args.blocks, SEED_A)
        datasets.append(dataset_a)
        dataset_b = create_dataset(library, args.blocks, SEED_B)
        datasets.append(dataset_b)
        pointers_a = dataset_pointers(library, dataset_a, args.blocks)
        pointers_b = dataset_pointers(library, dataset_b, args.blocks)
        require(set(pointers_a).isdisjoint(pointers_b), "A/B managed VAs overlap")
        original_pointers = pointers_a + pointers_b

        def require_va_stable(stage: str) -> None:
            current = dataset_pointers(library, dataset_a, args.blocks) + dataset_pointers(
                library, dataset_b, args.blocks
            )
            require(current == original_pointers, f"managed VA changed after {stage}")

        entries_a = sliced_entries(
            pointers_a,
            owner_base=OWNER_A,
            reuse_required=False,
        )
        entries_b = sliced_entries(
            pointers_b,
            owner_base=OWNER_B,
            reuse_required=True,
        )

        preallocate = driver_slot_preallocate_ranges(entries_a, device=args.device)
        require_preallocate(
            preallocate,
            entries=entries_count,
            logical_bytes=logical_bytes,
        )

        before_restore = read_stats(args.stats)
        restore_a = driver_slot_restore_ranges(entries_a, device=args.device)
        after_restore = read_stats(args.stats)
        restore_delta = stats_delta(before_restore, after_restore)
        require_copy_map_result(
            restore_a,
            mode=MODE_SHARED_PUSH_64K,
            blocks=args.blocks,
            entries=entries_count,
            logical_bytes=logical_bytes,
            transition=False,
        )
        require_profile(
            restore_delta,
            restore_a,
            blocks=args.blocks,
            entries=entries_count,
            logical_bytes=logical_bytes,
            operation="mode 21 restore A",
            direct_cutover=False,
        )
        require_va_stable("mode 21 restore A")

        stream = int(library.uvm512_stream_create())
        require(stream != 0, "stream_create failed")
        checksum_a, expected_a = dataset_checksum(library, dataset_a, stream)
        require(checksum_a == expected_a, "A checksum mismatch")

        before_transition = read_stats(args.stats)
        transition_b = driver_slot_transition_ranges(entries_b, device=args.device)
        after_transition = read_stats(args.stats)
        transition_delta = stats_delta(before_transition, after_transition)
        require_copy_map_result(
            transition_b,
            mode=MODE_LAYER_TRANSITION,
            blocks=args.blocks,
            entries=entries_count,
            logical_bytes=logical_bytes,
            transition=True,
        )
        require_profile(
            transition_delta,
            transition_b,
            blocks=args.blocks,
            entries=entries_count,
            logical_bytes=logical_bytes,
            operation="mode 20 transition A-to-B",
            direct_cutover=direct_cutover_value == 1,
        )
        require_va_stable("mode 20 transition A-to-B")
        checksum_b, expected_b = dataset_checksum(library, dataset_b, stream)
        require(checksum_b == expected_b, "B checksum mismatch")
        require(checksum_b != checksum_a, "A/B checksum signatures are not distinct")

        # In direct-cutover mode A is CPU-resident but deliberately GPU-unmapped
        # after ownership moves to B. Reading A from the GPU must fault through
        # ordinary UVM and recover the original CPU data, not observe B's bytes
        # through a stale slot translation. Repeat the read to verify the
        # recovered mapping is stable, then re-read B to prove the live slot
        # mapping was not disturbed by A's fault recovery.
        before_old_va_refault = read_stats(args.stats)
        old_va_refault_started_ns = time.perf_counter_ns()
        checksum_a_after, expected_a_after = dataset_checksum(library, dataset_a, stream)
        old_va_refault_first_ns = time.perf_counter_ns() - old_va_refault_started_ns
        after_old_va_refault_first = read_stats(args.stats)
        old_va_second_started_ns = time.perf_counter_ns()
        checksum_a_second, expected_a_second = dataset_checksum(library, dataset_a, stream)
        old_va_refault_second_ns = time.perf_counter_ns() - old_va_second_started_ns
        after_old_va_refault_second = read_stats(args.stats)
        checksum_b_after_refault, expected_b_after_refault = dataset_checksum(
            library, dataset_b, stream
        )
        require(
            checksum_a_after == expected_a_after == expected_a,
            "old A checksum did not recover from the GPU-unmapped state",
        )
        require(
            checksum_a_second == expected_a_second == expected_a,
            "old A second checksum is not stable after fault recovery",
        )
        require(
            checksum_b_after_refault == expected_b_after_refault == expected_b,
            "B slot checksum changed after old-A fault recovery",
        )
        require(checksum_a_after != checksum_b_after_refault, "A/B refault signatures alias")

        close_result = close_uvm_slot_direct_session(device=args.device)
        require(bool(close_result.get("ok")), f"slot close failed: {close_result}")
        closed = True
        require_va_stable("slot close")
        payload.update(
            {
                "status": "pass",
                "va_stable": True,
                "checksum_a": checksum_a,
                "checksum_a_expected": expected_a,
                "checksum_b": checksum_b,
                "checksum_b_expected": expected_b,
                "checksum_a_after_transition": checksum_a_after,
                "checksum_a_after_transition_expected": expected_a_after,
                "checksum_a_second_after_transition": checksum_a_second,
                "checksum_a_second_after_transition_expected": expected_a_second,
                "checksum_b_after_old_va_refault": checksum_b_after_refault,
                "checksum_b_after_old_va_refault_expected": expected_b_after_refault,
                "old_va_refault_first_wall_ns": old_va_refault_first_ns,
                "old_va_refault_second_wall_ns": old_va_refault_second_ns,
                "old_va_refault_first_profile_delta": stats_delta(
                    before_old_va_refault, after_old_va_refault_first
                ),
                "old_va_refault_second_profile_delta": stats_delta(
                    after_old_va_refault_first, after_old_va_refault_second
                ),
                "preallocate": summarize(preallocate),
                "restore_a": summarize(restore_a),
                "transition_b": summarize(transition_b),
                "restore_profile_delta": restore_delta,
                "transition_profile_delta": transition_delta,
                "close": summarize(close_result),
            }
        )
    except Exception as error:  # noqa: BLE001
        failure = f"{type(error).__name__}: {error}"
        payload["error"] = failure
    finally:
        if not closed:
            try:
                payload["cleanup_close"] = summarize(
                    close_uvm_slot_direct_session(device=args.device)
                )
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
