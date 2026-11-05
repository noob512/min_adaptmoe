#!/usr/bin/env python3
"""Exercise complete Prefill/Decode children with model-like mixed 2 MiB blocks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from vllm.device_allocator import uvm
from vllm.device_allocator import uvm_static_alias as static_alias


MIB = 1024 * 1024
BLOCK_BYTES = 2 * MIB
TARGET_BLOCKS = 4
ALLOCATION_BYTES = 7 * BLOCK_BYTES
LAYER_BASE = 9300


def require_ok(result: dict[str, object], label: str) -> None:
    if result.get("ok") is not True:
        raise RuntimeError(f"{label} failed: {result!r}")


def all_equal(tensor: torch.Tensor, value: int) -> bool:
    return bool(torch.all(tensor == value).item())


def check_view(
    target: torch.Tensor,
    cold_regions: list[tuple[int, int]],
    cold_values: list[int],
    block_values: list[int],
) -> None:
    boundaries = {0, TARGET_BLOCKS * BLOCK_BYTES}
    for offset, length in cold_regions:
        boundaries.add(offset)
        boundaries.add(offset + length)
    for block in range(TARGET_BLOCKS):
        boundaries.add(block * BLOCK_BYTES)
        boundaries.add((block + 1) * BLOCK_BYTES)

    points = sorted(boundaries)
    for first, outer in zip(points, points[1:]):
        if first == outer:
            continue
        expected = block_values[first // BLOCK_BYTES]
        for index, (cold_first, cold_length) in enumerate(cold_regions):
            if cold_first <= first and outer <= cold_first + cold_length:
                expected = cold_values[index]
                break
        if not all_equal(target[first:outer], expected):
            raise RuntimeError(
                f"view mismatch in [{first:#x}, {outer:#x}), expected {expected}"
            )


def timing_summary(values: list[float], count: int, total: float,
                   minimum: float, maximum: float) -> dict[str, object]:
    return {
        "count": count,
        "mean": total / count if count else None,
        "sample_count": len(values),
        "sample_p50": statistics.median(values) if values else None,
        "sample_p95": (
            sorted(values)[min(len(values) - 1, int(0.95 * len(values)))]
            if values else None
        ),
        "min": minimum if count else None,
        "max": maximum if count else None,
    }


def run(
    *,
    cycles: int = 1,
    minimum_seconds: float = 0.0,
    verify_every: int = 1,
    checkpoint_every: int = 100,
    progress_path: Path | None = None,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(0)
    uvm.ensure_uvm_runtime_library()

    allocation: torch.Tensor | None = None
    upload_arena: torch.Tensor | None = None
    session_id = 0
    cleanup: dict[str, object] = {}
    try:
        with uvm.uvm_local_memory_pool(enable_prefetch=False):
            allocation = torch.empty(
                ALLOCATION_BYTES, dtype=torch.uint8, device="cuda"
            )
            upload_arena = torch.empty(
                5 * MIB, dtype=torch.uint8, device="cuda"
            )

        allocation_va = int(allocation.data_ptr())
        aligned_va = (allocation_va + BLOCK_BYTES - 1) & ~(BLOCK_BYTES - 1)
        aligned_offset = aligned_va - allocation_va
        target_bytes = TARGET_BLOCKS * BLOCK_BYTES
        if aligned_offset + target_bytes > ALLOCATION_BYTES:
            raise RuntimeError("managed allocation has insufficient aligned blocks")
        target = allocation[aligned_offset : aligned_offset + target_bytes]

        initial_count, initial_executor, initial_status = (
            uvm.batch_prefetch_ranges_to_device_with_executor(
                [
                    (int(allocation.data_ptr()), ALLOCATION_BYTES),
                    (int(upload_arena.data_ptr()), int(upload_arena.numel())),
                ],
                0,
                executor="cpp_batch",
            )
        )
        if int(initial_count) != 2 or int(initial_status) != 0:
            raise RuntimeError(
                "initial GPU residency preparation failed: "
                f"{initial_count}, {initial_executor}, {initial_status}"
            )
        torch.cuda.synchronize()

        # These shapes cover both directions of the Qwen 1.5 MiB expert
        # boundary and multiple disjoint cold runs in one 2 MiB parent.
        cold_regions = [
            (0, 1536 * 1024),
            (BLOCK_BYTES + 512 * 1024, 1536 * 1024),
            (2 * BLOCK_BYTES + 768 * 1024, 512 * 1024),
            (3 * BLOCK_BYTES, 512 * 1024),
            (3 * BLOCK_BYTES + MIB, 512 * 1024),
        ]
        block_values = [31, 37, 43, 49]
        first_slot_values = [71, 73, 75, 77, 79]
        second_slot_values = [81, 83, 85, 87, 89]

        for block, value in enumerate(block_values):
            target[
                block * BLOCK_BYTES : (block + 1) * BLOCK_BYTES
            ].fill_(value)
        upload_arena.fill_(17)
        torch.cuda.synchronize()

        target_ranges = [(int(target.data_ptr()), target_bytes)]
        if int(uvm.batch_advise_ranges_read_mostly(target_ranges, device=0)) != 1:
            raise RuntimeError("ReadMostly setup failed")
        if int(uvm.batch_prefetch_ranges_to_cpu(target_ranges)) != 1:
            raise RuntimeError("full target CPU shadow setup failed")
        torch.cuda.synchronize()

        consumer_ranges = [
            (int(target.data_ptr()) + offset, length)
            for offset, length in cold_regions
        ]
        cutover = dict(
            uvm.driver_cutover_ranges_to_immutable_cpu_shadow(
                consumer_ranges, device=0
            )
        )
        require_ok(cutover, "immutable-shadow cutover")

        hot_ranges: list[tuple[int, int]] = []
        cursor = 0
        for offset, length in cold_regions:
            if cursor < offset:
                hot_ranges.append((int(target.data_ptr()) + cursor, offset - cursor))
            cursor = offset + length
        if cursor < target_bytes:
            hot_ranges.append((int(target.data_ptr()) + cursor, target_bytes - cursor))
        hot_count, hot_executor, hot_status = (
            uvm.batch_prefetch_ranges_to_device_with_executor(
                hot_ranges, 0, executor="cpp_batch"
            )
        )
        if int(hot_count) != len(hot_ranges) or int(hot_status) != 0:
            raise RuntimeError(
                f"hot prefetch failed: {hot_count}, {hot_executor}, {hot_status}"
            )
        torch.cuda.synchronize()

        upload_ranges: list[tuple[int, int]] = []
        upload_offset = 0
        for _, length in cold_regions:
            upload_ranges.append((int(upload_arena.data_ptr()) + upload_offset, length))
            upload_offset += length
        prepared = uvm.batch_prepare_ranges_for_native_remote_access_with_executor(
            upload_ranges, sync=True, device=0
        )
        if int(prepared[0]) != len(upload_ranges):
            raise RuntimeError(f"upload preparation failed: {prepared!r}")

        preallocate_entries = []
        aliases = []
        for index, ((upload_va, length), (consumer_va, _)) in enumerate(
            zip(upload_ranges, consumer_ranges)
        ):
            preallocate_entries.append(
                (
                    upload_va,
                    length,
                    LAYER_BASE + index,
                    0,
                    0,
                    0,
                    uvm.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT,
                )
            )
            aliases.extend(
                (
                    (
                        LAYER_BASE + index,
                        0,
                        0,
                        static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW,
                        upload_va,
                        length,
                        0,
                    ),
                    (LAYER_BASE + index, 0, 0, 0, consumer_va, length, 0),
                )
            )

        preallocated = dict(
            uvm.driver_slot_preallocate_ranges(preallocate_entries, device=0)
        )
        require_ok(preallocated, "slot preallocate")
        mapped = dict(static_alias.driver_static_alias_map(aliases, device=0))
        require_ok(mapped, "static alias MAP")
        session_id = int(mapped["session_id"])

        seal_start = time.perf_counter_ns()
        sealed = dict(
            static_alias.driver_static_alias_seal(
                session_id,
                expected_alias_count=len(aliases),
                expected_alias_bytes=2 * sum(length for _, length in cold_regions),
                device=0,
            )
        )
        seal_ms = (time.perf_counter_ns() - seal_start) / 1e6
        require_ok(sealed, "SEAL")

        refresh_prefill_start = time.perf_counter_ns()
        refreshed_prefill = dict(
            static_alias.driver_static_alias_seal(
                session_id,
                device=0,
                refresh_dual=True,
            )
        )
        refresh_prefill_ms = (
            time.perf_counter_ns() - refresh_prefill_start
        ) / 1e6
        require_ok(refreshed_prefill, "Prefill dual-view refresh")

        first_sources: list[torch.Tensor] = []
        upload_offset = 0
        for (_, length), value in zip(cold_regions, first_slot_values):
            source = torch.full((length,), value, dtype=torch.uint8, pin_memory=True)
            first_sources.append(source)
            upload_arena[upload_offset : upload_offset + length].copy_(
                source, non_blocking=True
            )
            upload_offset += length
        torch.cuda.synchronize()
        check_view(target, cold_regions, first_slot_values, block_values)

        soak_started = time.perf_counter()
        park_start = time.perf_counter_ns()
        parked = dict(
            static_alias.driver_static_alias_park(
                session_id, device=0, synchronize=True
            )
        )
        park_ms = (time.perf_counter_ns() - park_start) / 1e6
        require_ok(parked, "Prefill-to-Decode switch")
        if parked.get("dual_view_full_coverage") is not True:
            raise RuntimeError(f"PARK did not certify complete coverage: {parked!r}")
        if int(parked.get("dual_view_block_count", -1)) != TARGET_BLOCKS:
            raise RuntimeError(f"unexpected dual block count: {parked!r}")

        # Relocate ordinary pages while the Decode child is active, then
        # refresh both children before allowing a GPU read. This models KV or
        # Graph allocation changing a hot page's physical address.
        relocated_hot = hot_ranges[:1]
        if int(uvm.batch_prefetch_ranges_to_cpu(relocated_hot)) != 1:
            raise RuntimeError("ordinary hot-page CPU relocation failed")
        hot_count, hot_executor, hot_status = (
            uvm.batch_prefetch_ranges_to_device_with_executor(
                relocated_hot, 0, executor="cpp_batch"
            )
        )
        if int(hot_count) != 1 or int(hot_status) != 0:
            raise RuntimeError(
                "ordinary hot-page GPU restoration failed: "
                f"{hot_count}, {hot_executor}, {hot_status}"
            )
        torch.cuda.synchronize()
        refresh_decode_start = time.perf_counter_ns()
        refreshed_decode = dict(
            static_alias.driver_static_alias_seal(
                session_id,
                device=0,
                refresh_dual=True,
            )
        )
        refresh_decode_ms = (
            time.perf_counter_ns() - refresh_decode_start
        ) / 1e6
        require_ok(refreshed_decode, "Decode dual-view refresh")
        check_view(target, cold_regions, block_values_for_cold(cold_regions, block_values), block_values)

        second_sources: list[torch.Tensor] = []
        upload_offset = 0
        for (_, length), value in zip(cold_regions, second_slot_values):
            source = torch.full((length,), value, dtype=torch.uint8, pin_memory=True)
            second_sources.append(source)
            upload_arena[upload_offset : upload_offset + length].copy_(
                source, non_blocking=True
            )
            upload_offset += length
        torch.cuda.synchronize()

        resume_start = time.perf_counter_ns()
        resumed = dict(
            static_alias.driver_static_alias_resume(
                session_id, device=0, synchronize=True
            )
        )
        resume_ms = (time.perf_counter_ns() - resume_start) / 1e6
        require_ok(resumed, "Decode-to-Prefill switch")
        check_view(target, cold_regions, second_slot_values, block_values)

        # The first cycle above also validates a hot-page relocation and a
        # dual-view refresh.  Further cycles keep the same session alive and
        # exercise only the production parent-entry PARK/RESUME operations.
        park_samples = [park_ms]
        resume_samples = [resume_ms]
        park_total = park_ms
        resume_total = resume_ms
        park_min = park_max = park_ms
        resume_min = resume_max = resume_ms
        completed_cycles = 1
        checkpoints = 0
        latest_slot_values = list(second_slot_values)

        def checkpoint() -> None:
            nonlocal checkpoints
            if progress_path is None:
                return
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "schema": "uvm_full_dual_view_soak_progress_v1",
                    "completed_cycles": completed_cycles,
                    "elapsed_seconds": time.perf_counter() - soak_started,
                    "park_last_ms": park_samples[-1],
                    "resume_last_ms": resume_samples[-1],
                    "session_id": session_id,
                    "verified": True,
                    "timestamp": time.time(),
                }, sort_keys=True) + "\n")
            checkpoints += 1

        if checkpoint_every > 0 and completed_cycles % checkpoint_every == 0:
            checkpoint()

        while (
            completed_cycles < cycles
            or time.perf_counter() - soak_started < minimum_seconds
        ):
            cycle_number = completed_cycles + 1
            park_started = time.perf_counter_ns()
            cycle_park = dict(
                static_alias.driver_static_alias_park(
                    session_id, device=0, synchronize=True
                )
            )
            cycle_park_ms = (time.perf_counter_ns() - park_started) / 1e6
            require_ok(cycle_park, f"cycle {cycle_number} Prefill-to-Decode")
            if cycle_park.get("dual_view_full_coverage") is not True:
                raise RuntimeError(
                    f"cycle {cycle_number} PARK lost full coverage: {cycle_park!r}"
                )
            if cycle_number % verify_every == 0:
                check_view(
                    target,
                    cold_regions,
                    block_values_for_cold(cold_regions, block_values),
                    block_values,
                )

            latest_slot_values = [
                (101 + cycle_number * 17 + index * 7) & 0xFF
                for index in range(len(cold_regions))
            ]
            upload_offset = 0
            cycle_sources: list[torch.Tensor] = []
            for (_, length), value in zip(cold_regions, latest_slot_values):
                source = torch.full(
                    (length,), value, dtype=torch.uint8, pin_memory=True
                )
                cycle_sources.append(source)
                upload_arena[upload_offset : upload_offset + length].copy_(
                    source, non_blocking=True
                )
                upload_offset += length
            torch.cuda.synchronize()

            resume_started = time.perf_counter_ns()
            cycle_resume = dict(
                static_alias.driver_static_alias_resume(
                    session_id, device=0, synchronize=True
                )
            )
            cycle_resume_ms = (time.perf_counter_ns() - resume_started) / 1e6
            require_ok(cycle_resume, f"cycle {cycle_number} Decode-to-Prefill")
            if cycle_resume.get("dual_view_full_coverage") is not True:
                raise RuntimeError(
                    f"cycle {cycle_number} RESUME lost full coverage: {cycle_resume!r}"
                )
            if cycle_number % verify_every == 0:
                check_view(target, cold_regions, latest_slot_values, block_values)

            completed_cycles = cycle_number
            park_total += cycle_park_ms
            resume_total += cycle_resume_ms
            park_min = min(park_min, cycle_park_ms)
            park_max = max(park_max, cycle_park_ms)
            resume_min = min(resume_min, cycle_resume_ms)
            resume_max = max(resume_max, cycle_resume_ms)
            # Bound formal 6/24-hour artifacts while retaining every sample
            # in the 1,000-cycle qualification.
            if len(park_samples) < 100_000:
                park_samples.append(cycle_park_ms)
                resume_samples.append(cycle_resume_ms)
            if checkpoint_every > 0 and completed_cycles % checkpoint_every == 0:
                checkpoint()

        if checkpoint_every <= 0 or completed_cycles % checkpoint_every != 0:
            checkpoint()
        soak_elapsed = time.perf_counter() - soak_started

        destroyed = dict(
            static_alias.driver_static_alias_destroy(
                session_id, device=0, release_backing=True, synchronize=True
            )
        )
        require_ok(destroyed, "DESTROY")
        session_id = 0
        cleanup["slot_session"] = uvm.close_uvm_slot_direct_session(device=0)
        return {
            "ok": True,
            "schema": "uvm_full_dual_view_geometry_probe_v2",
            "target_blocks": TARGET_BLOCKS,
            "target_bytes": target_bytes,
            "cold_regions": cold_regions,
            "seal": sealed,
            "refresh_prefill": refreshed_prefill,
            "refresh_decode": refreshed_decode,
            "park": parked,
            "resume": resumed,
            "timing_ms": {
                "seal": seal_ms,
                "refresh_prefill": refresh_prefill_ms,
                "refresh_decode": refresh_decode_ms,
                "prefill_to_decode": park_ms,
                "decode_to_prefill": resume_ms,
            },
            "soak": {
                "requested_cycles": cycles,
                "requested_minimum_seconds": minimum_seconds,
                "completed_cycles": completed_cycles,
                "elapsed_seconds": soak_elapsed,
                "verify_every": verify_every,
                "checkpoint_every": checkpoint_every,
                "checkpoint_count": checkpoints,
                "latest_slot_values": latest_slot_values,
                "park_ms": timing_summary(
                    park_samples, completed_cycles, park_total, park_min, park_max
                ),
                "resume_ms": timing_summary(
                    resume_samples,
                    completed_cycles,
                    resume_total,
                    resume_min,
                    resume_max,
                ),
            },
            "cleanup": cleanup,
        }
    finally:
        if session_id:
            cleanup["fallback_destroy"] = static_alias.driver_static_alias_destroy(
                session_id, device=0, release_backing=True, synchronize=True
            )
        if 0 in uvm._uvm_slot_direct_sessions:
            cleanup["fallback_slot_close"] = uvm.close_uvm_slot_direct_session(
                device=0
            )
        allocation = None
        upload_arena = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        uvm.release_uvm_local_memory_pool()


def block_values_for_cold(
    cold_regions: list[tuple[int, int]], block_values: list[int]
) -> list[int]:
    return [block_values[offset // BLOCK_BYTES] for offset, _ in cold_regions]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--minimum-seconds", type=float, default=0.0)
    parser.add_argument("--verify-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--progress", type=Path)
    args = parser.parse_args()
    if args.cycles <= 0:
        raise SystemExit("cycles must be positive")
    if args.minimum_seconds < 0:
        raise SystemExit("minimum-seconds must be non-negative")
    if args.verify_every <= 0:
        raise SystemExit("verify-every must be positive")
    if args.checkpoint_every < 0:
        raise SystemExit("checkpoint-every must be non-negative")
    started = time.time()
    result: dict[str, object] = {}
    try:
        result = run(
            cycles=args.cycles,
            minimum_seconds=args.minimum_seconds,
            verify_every=args.verify_every,
            checkpoint_every=args.checkpoint_every,
            progress_path=args.progress,
        )
    except Exception as exc:
        result = {"ok": False, "error": repr(exc)}
        raise
    finally:
        result["started_unix"] = started
        result["completed_unix"] = time.time()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
