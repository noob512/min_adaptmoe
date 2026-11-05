#!/usr/bin/env python3
"""Validate mode 12 with configurable 2 MiB blocks split into four slices."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
VLLM_ROOT = REPO_ROOT / "workloads" / "vllm" / "vllm"
LEGACY_DIR = REPO_ROOT / "experiments" / "uvm_512x4m_batch_compare"
DEFAULT_LIBRARY = LEGACY_DIR / "libuvm_batch_shim.so"
DEFAULT_STATS = Path("/proc/driver/nvidia-uvm/latency_stats")

# Select mode 12 before importing the low-level wrapper. The wrapper still
# reads this gate per restore, but setting it here makes the probe standalone.
os.environ["VLLM_UVM_SLOT_DIRECT_SHARED_PUSH"] = "0"
os.environ["VLLM_UVM_SLOT_DIRECT_64K_PTE"] = "1"
sys.path.insert(0, str(VLLM_ROOT))
sys.path.insert(0, str(LEGACY_DIR))

import benchmark as legacy
from vllm.device_allocator.uvm import (
    UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT,
    close_uvm_slot_direct_session,
    driver_slot_preallocate_ranges,
    driver_slot_restore_ranges,
)

DEFAULT_BLOCKS = 288
MAX_BLOCKS = 288
BLOCK_BYTES = 2 * 1024 * 1024
SLICES_PER_BLOCK = 4
SLICE_BYTES = 512 * 1024
DRIVER_LAYER = 0x64C0

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
        raise RuntimeError(f"unrecognized latency_stats line {line_number}: {line!r}")
    return values


def read_stats(path: Path) -> dict[str, int]:
    return parse_stats(path.read_text(encoding="utf-8"))


def stats_delta(before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, int]:
    delta = {
        name: int(after.get(name, 0)) - int(before.get(name, 0))
        for name in set(before) | set(after)
    }
    require(all(value >= 0 for value in delta.values()), "driver counters moved backwards")
    return delta


def dataset_pointers(
    library: ctypes.CDLL,
    dataset: int,
    blocks: int,
) -> tuple[int, ...]:
    return tuple(
        int(library.uvm512_dataset_pointer(dataset, index))
        for index in range(blocks)
    )


def sliced_entries(pointers: tuple[int, ...]) -> list[tuple[int, ...]]:
    entries: list[tuple[int, ...]] = []
    for block_index, pointer in enumerate(pointers):
        for slice_index in range(SLICES_PER_BLOCK):
            entry_index = block_index * SLICES_PER_BLOCK + slice_index
            entries.append(
                (
                    pointer + slice_index * SLICE_BYTES,
                    SLICE_BYTES,
                    DRIVER_LAYER,
                    entry_index // 9,
                    entry_index + 1,
                    0,
                    UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT,
                )
            )
    return entries


def require_restore(
    result: Mapping[str, object],
    *,
    blocks: int,
    entries: int,
    logical_bytes: int,
) -> None:
    require(bool(result.get("ok")), f"mode12 restore failed: {result}")
    expected = {
        "mode": 12,
        "requested_count": entries,
        "success_count": entries,
        "failure_count": 0,
        "requested_bytes": logical_bytes,
        "accepted_bytes": logical_bytes,
        "remap_count": entries,
        "tlb_invalidate_count": blocks,
        "copy_push_count": entries,
        "ioctl_calls": 1,
    }
    for name, value in expected.items():
        require(int(result.get(name, -1)) == value, f"{name} mismatch")
    statuses = result.get("entry_statuses")
    require(
        isinstance(statuses, list)
        and len(statuses) == entries
        and all(status == 0 for status in statuses),
        "entry status mismatch",
    )


def require_profile(
    delta: Mapping[str, int],
    *,
    blocks: int,
    entries: int,
    logical_bytes: int,
    pte_64k_entries: int,
) -> None:
    expected = {
        "slot_direct_ioctl_calls": 1,
        "slot_direct_per_block_calls": 1,
        "slot_direct_shared_calls": 0,
        "slot_direct_successes": 1,
        "slot_direct_fallbacks": 0,
        "slot_direct_errors": 0,
        "slot_direct_requested_ranges": entries,
        "slot_direct_completed_ranges": entries,
        "slot_direct_blocks": blocks,
        "slot_direct_pte_4k_blocks": 0,
        "slot_direct_pte_64k_blocks": blocks,
        "slot_direct_pte_2m_blocks": 0,
        "slot_direct_pte_write_entries": pte_64k_entries,
        "slot_direct_copy_pushes": entries,
        "slot_direct_memcopy_methods": entries,
        "slot_direct_requested_bytes": logical_bytes,
        "slot_direct_completed_bytes": logical_bytes,
        "slot_direct_pte_map_ranges": entries,
        "slot_direct_pte_map_bytes": logical_bytes,
        "slot_direct_vidmem_allocations": 0,
        "slot_direct_vidmem_evictions": 0,
    }
    for name, value in expected.items():
        require(name in delta, f"missing counter {name}")
        require(int(delta[name]) == value, f"{name}={delta[name]} expected {value}")


def checksum(library: ctypes.CDLL, dataset: int, stream: int) -> tuple[int, int]:
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
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    args = parser.parse_args()
    require(1 <= args.blocks <= MAX_BLOCKS, f"blocks must be in [1, {MAX_BLOCKS}]")
    require(args.library.is_file(), f"missing library: {args.library}")
    require(args.stats.is_file(), f"missing stats: {args.stats}")

    entries_count = args.blocks * SLICES_PER_BLOCK
    logical_bytes = args.blocks * BLOCK_BYTES
    pte_64k_entries = logical_bytes // (64 * 1024)

    library = legacy.load_library(args.library)
    check_cuda(library, library.uvm512_set_device(args.device), "set device")
    dataset = 0
    stream = 0
    closed = False
    payload: dict[str, Any] = {
        "status": "failed",
        "blocks": args.blocks,
        "slices_per_block": SLICES_PER_BLOCK,
        "entries": entries_count,
        "block_bytes": BLOCK_BYTES,
        "slice_bytes": SLICE_BYTES,
        "logical_bytes": logical_bytes,
        "expected_64k_ptes": pte_64k_entries,
        "module_srcversion": Path("/sys/module/nvidia_uvm/srcversion")
        .read_text(encoding="utf-8")
        .strip(),
    }

    try:
        dataset = int(library.uvm512_dataset_create(args.blocks, BLOCK_BYTES, 0))
        require(dataset != 0, "dataset_create failed")
        pointers = dataset_pointers(library, dataset, args.blocks)
        require(len(set(pointers)) == args.blocks, "dataset VAs overlap")
        require(all(pointer % BLOCK_BYTES == 0 for pointer in pointers), "2 MiB VA alignment failed")
        entries = sliced_entries(pointers)

        preallocate = driver_slot_preallocate_ranges(entries, device=args.device)
        require(bool(preallocate.get("ok")), f"mode8 failed: {preallocate}")
        require(int(preallocate.get("ioctl_calls", -1)) == 1, "mode8 was not one ioctl")

        before = read_stats(args.stats)
        restore = driver_slot_restore_ranges(entries, device=args.device)
        after = read_stats(args.stats)
        delta = stats_delta(before, after)
        require_restore(
            restore,
            blocks=args.blocks,
            entries=entries_count,
            logical_bytes=logical_bytes,
        )
        require_profile(
            delta,
            blocks=args.blocks,
            entries=entries_count,
            logical_bytes=logical_bytes,
            pte_64k_entries=pte_64k_entries,
        )
        require(
            dataset_pointers(library, dataset, args.blocks) == pointers,
            "managed VA changed",
        )

        stream = int(library.uvm512_stream_create())
        require(stream != 0, "stream_create failed")
        actual, expected = checksum(library, dataset, stream)
        require(actual == expected, "GPU checksum mismatch")

        close_result = close_uvm_slot_direct_session(device=args.device)
        require(bool(close_result.get("ok")), f"session close failed: {close_result}")
        closed = True
        payload.update(
            {
                "status": "pass",
                "va_stable": True,
                "checksum": actual,
                "checksum_expected": expected,
                "preallocate": summarize(preallocate),
                "restore": summarize(restore),
                "profile_delta": delta,
                "close_ok": True,
            }
        )
    finally:
        if not closed:
            try:
                close_uvm_slot_direct_session(device=args.device)
            except (OSError, RuntimeError) as error:
                print(f"warning: slot session cleanup failed: {error}", file=sys.stderr)
        if stream:
            library.uvm512_stream_destroy(stream)
        if dataset:
            library.uvm512_dataset_destroy(dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
