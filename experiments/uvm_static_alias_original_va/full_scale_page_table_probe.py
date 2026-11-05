#!/usr/bin/env python3
"""One-shot synthetic 34,560-range static-alias page-table probe.

This deliberately does not load vLLM or a model. It creates the cold80-sized
managed-memory geometry, stages a parked static-alias directory, then measures
exactly one full RESUME followed by exactly one full PARK.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from vllm.device_allocator import uvm
from vllm.device_allocator import uvm_static_alias as static_alias


REGION_BYTES = 512 * 1024
VA_BLOCK_BYTES = 2 * 1024 * 1024
SLOT_COUNT = 1_440
CONSUMER_COUNT = 34_560
CONSUMERS_PER_SLOT = CONSUMER_COUNT // SLOT_COUNT
BASE_LAYER_ID = 9_600


def _require(result: dict[str, object], label: str) -> None:
    if result.get("ok") is not True:
        raise RuntimeError(f"{label} failed: {result!r}")


def _aligned_arena(
    allocation: torch.Tensor, arena_bytes: int
) -> tuple[torch.Tensor, int]:
    allocation_va = int(allocation.data_ptr())
    arena_va = (allocation_va + VA_BLOCK_BYTES - 1) & ~(VA_BLOCK_BYTES - 1)
    arena_offset = arena_va - allocation_va
    if arena_offset + arena_bytes > int(allocation.numel()):
        raise RuntimeError("managed allocation does not contain the aligned arena")
    return allocation[arena_offset : arena_offset + arena_bytes], arena_va


def _slim(result: dict[str, object], fields: tuple[str, ...]) -> dict[str, object]:
    return {name: result.get(name) for name in fields if name in result}


def _all_equal(tensor: torch.Tensor, value: int) -> bool:
    return bool(torch.all(tensor == int(value)).item())


def run() -> dict[str, object]:
    if CONSUMER_COUNT % SLOT_COUNT != 0:
        raise RuntimeError("consumer geometry must evenly share backing slots")
    if CONSUMERS_PER_SLOT + 1 > 64:
        raise RuntimeError("synthetic geometry exceeds the per-slot alias limit")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.cuda.set_device(0)
    uvm.ensure_uvm_runtime_library()

    upload_bytes = SLOT_COUNT * REGION_BYTES
    consumer_bytes = CONSUMER_COUNT * REGION_BYTES
    upload_allocation: torch.Tensor | None = None
    consumer_allocation: torch.Tensor | None = None
    upload_arena: torch.Tensor | None = None
    consumer_arena: torch.Tensor | None = None
    session_id = 0
    cleanup: dict[str, object] = {}

    setup_started_ns = time.perf_counter_ns()
    try:
        allocation_started_ns = time.perf_counter_ns()
        with uvm.uvm_local_memory_pool(enable_prefetch=False):
            upload_allocation = torch.empty(
                upload_bytes + VA_BLOCK_BYTES,
                dtype=torch.uint8,
                device="cuda",
            )
            consumer_allocation = torch.empty(
                consumer_bytes + VA_BLOCK_BYTES,
                dtype=torch.uint8,
                device="cuda",
            )
        upload_arena, upload_va = _aligned_arena(upload_allocation, upload_bytes)
        consumer_arena, consumer_va = _aligned_arena(
            consumer_allocation, consumer_bytes
        )
        allocation_wall_ns = time.perf_counter_ns() - allocation_started_ns

        residency_started_ns = time.perf_counter_ns()
        prepared = uvm.batch_prepare_ranges_for_native_remote_access_with_executor(
            [
                (upload_va, upload_bytes),
                (consumer_va, consumer_bytes),
            ],
            sync=True,
            device=0,
        )
        residency_wall_ns = time.perf_counter_ns() - residency_started_ns
        if int(prepared[0]) != 2:
            raise RuntimeError(f"CPU residency preparation failed: {prepared!r}")

        preallocate_entries = [
            (
                upload_va + slot * REGION_BYTES,
                REGION_BYTES,
                BASE_LAYER_ID,
                slot,
                slot,
                0,
                uvm.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT,
            )
            for slot in range(SLOT_COUNT)
        ]
        preallocate_started_ns = time.perf_counter_ns()
        preallocate = dict(
            uvm.driver_slot_preallocate_ranges(preallocate_entries, device=0)
        )
        preallocate_wall_ns = time.perf_counter_ns() - preallocate_started_ns
        _require(preallocate, "1,440-slot GPU backing preallocation")

        # Seed a small, valid session containing all upload/backing identities.
        # The measured full topology is installed later through one parked
        # transaction, so setup time is excluded from RESUME timing.
        initial_entries: list[tuple[int, int, int, int, int, int, int]] = []
        for slot in range(SLOT_COUNT):
            identity = (BASE_LAYER_ID, slot, slot)
            initial_entries.append(
                (
                    *identity,
                    static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW,
                    upload_va + slot * REGION_BYTES,
                    REGION_BYTES,
                    0,
                )
            )
            initial_entries.append(
                (
                    *identity,
                    0,
                    consumer_va + slot * REGION_BYTES,
                    REGION_BYTES,
                    0,
                )
            )

        initial_map_started_ns = time.perf_counter_ns()
        initial_map = dict(
            static_alias.driver_static_alias_map(initial_entries, device=0)
        )
        initial_map_wall_ns = time.perf_counter_ns() - initial_map_started_ns
        _require(initial_map, "initial static-alias MAP")
        session_id = int(initial_map["session_id"])

        seal = dict(
            static_alias.driver_static_alias_seal(
                session_id,
                expected_alias_count=len(initial_entries),
                expected_alias_bytes=len(initial_entries) * REGION_BYTES,
                device=0,
            )
        )
        _require(seal, "initial static-alias SEAL")
        initial_park = dict(
            static_alias.driver_static_alias_park(
                session_id, device=0, synchronize=True
            )
        )
        _require(initial_park, "initial static-alias PARK")

        consumer_entries = [
            (
                BASE_LAYER_ID,
                consumer_index % SLOT_COUNT,
                consumer_index % SLOT_COUNT,
                0,
                consumer_va + consumer_index * REGION_BYTES,
                REGION_BYTES,
                0,
            )
            for consumer_index in range(CONSUMER_COUNT)
        ]
        rebind_started_ns = time.perf_counter_ns()
        rebind = dict(
            static_alias.driver_static_alias_rebind(
                session_id,
                consumer_entries,
                device=0,
                synchronize=True,
                resize=False,
            )
        )
        rebind_wall_ns = time.perf_counter_ns() - rebind_started_ns
        _require(rebind, "parked full-directory REBIND")
        setup_before_resume_ns = time.perf_counter_ns() - setup_started_ns

        torch.cuda.synchronize()
        resume_started_ns = time.perf_counter_ns()
        resume = dict(
            static_alias.driver_static_alias_resume(
                session_id, device=0, synchronize=True
            )
        )
        resume_outer_wall_ns = time.perf_counter_ns() - resume_started_ns
        _require(resume, "measured 34,560-range RESUME")

        if int(resume.get("processed_alias_count", -1)) != CONSUMER_COUNT:
            raise RuntimeError(f"RESUME processed the wrong topology: {resume!r}")

        # Verify one low, middle, and high backing identity through a consumer
        # which is not part of the initial seed directory.
        samples: list[dict[str, int | bool]] = []
        for sample_index, (slot, value) in enumerate(
            ((0, 37), (SLOT_COUNT // 2, 83), (SLOT_COUNT - 1, 149))
        ):
            consumer_index = slot + (CONSUMERS_PER_SLOT - 1) * SLOT_COUNT
            upload_view = upload_arena[
                slot * REGION_BYTES : (slot + 1) * REGION_BYTES
            ]
            consumer_view = consumer_arena[
                consumer_index * REGION_BYTES : (consumer_index + 1) * REGION_BYTES
            ]
            source = torch.full(
                (REGION_BYTES,),
                value,
                dtype=torch.uint8,
                pin_memory=True,
            )
            upload_view.copy_(source, non_blocking=True)
            observed = consumer_view.clone()
            torch.cuda.synchronize()
            correct = _all_equal(observed, value)
            if not correct:
                raise RuntimeError(
                    f"sample {sample_index} did not observe slot backing"
                )
            samples.append(
                {
                    "slot": slot,
                    "consumer_index": consumer_index,
                    "value": value,
                    "correct": correct,
                }
            )

        torch.cuda.synchronize()
        park_started_ns = time.perf_counter_ns()
        park = dict(
            static_alias.driver_static_alias_park(
                session_id, device=0, synchronize=True
            )
        )
        park_outer_wall_ns = time.perf_counter_ns() - park_started_ns
        _require(park, "measured 34,560-range PARK")

        destroy = dict(
            static_alias.driver_static_alias_destroy(
                session_id,
                device=0,
                release_backing=True,
                synchronize=True,
            )
        )
        _require(destroy, "static-alias DESTROY")
        session_id = 0
        cleanup["slot_session"] = uvm.close_uvm_slot_direct_session(device=0)

        timing_fields = (
            "wall_ns",
            "ioctl_wall_ns",
            "validation_ns",
            "build_sort_ns",
            "preflight_ns",
            "metadata_ns",
            "cpu_unmap_ns",
            "pte_map_ns",
            "wait_ns",
            "bookkeeping_ns",
        )
        result: dict[str, Any] = {
            "ok": True,
            "schema": "synthetic_full_scale_static_alias_page_table_probe_v1",
            "single_run": True,
            "statistical_repetitions": 0,
            "vllm_model_loaded": False,
            "dual_parent_switch_enabled": False,
            "pte_mode_requested": "4K",
            "geometry": {
                "region_bytes": REGION_BYTES,
                "slot_count": SLOT_COUNT,
                "consumer_count": CONSUMER_COUNT,
                "consumers_per_slot": CONSUMERS_PER_SLOT,
                "upload_bytes": upload_bytes,
                "consumer_bytes": consumer_bytes,
                "total_directory_records": SLOT_COUNT + CONSUMER_COUNT,
            },
            "setup_timing_ms": {
                "managed_allocation": allocation_wall_ns / 1e6,
                "cpu_residency": residency_wall_ns / 1e6,
                "slot_preallocate": preallocate_wall_ns / 1e6,
                "initial_map": initial_map_wall_ns / 1e6,
                "full_directory_rebind": rebind_wall_ns / 1e6,
                "setup_total_before_measured_resume": setup_before_resume_ns / 1e6,
            },
            "resume": _slim(
                resume,
                timing_fields
                + (
                    "ok",
                    "rm_status",
                    "processed_alias_count",
                    "processed_alias_bytes",
                    "consumer_alias_count",
                    "consumer_alias_bytes",
                    "transition_batch_count",
                    "tlb_invalidate_count",
                ),
            ),
            "resume_outer_wall_ms": resume_outer_wall_ns / 1e6,
            "park": _slim(
                park,
                timing_fields
                + (
                    "ok",
                    "rm_status",
                    "processed_alias_count",
                    "processed_alias_bytes",
                    "consumer_alias_count",
                    "consumer_alias_bytes",
                    "transition_batch_count",
                    "tlb_invalidate_count",
                ),
            ),
            "park_outer_wall_ms": park_outer_wall_ns / 1e6,
            "correctness_samples": samples,
            "setup": {
                "cpu_prepare_count": int(prepared[0]),
                "cpu_prepare_executor": str(prepared[1]),
                "cpu_prepare_driver_result": int(prepared[2]),
                "preallocate": _slim(
                    preallocate,
                    (
                        "ok",
                        "rm_status",
                        "requested_count",
                        "success_count",
                        "requested_bytes",
                        "accepted_bytes",
                        "ioctl_calls",
                        "ioctl_wall_ns",
                    ),
                ),
                "initial_map": _slim(
                    initial_map,
                    (
                        "ok",
                        "rm_status",
                        "session_id",
                        "success_count",
                        "alias_count",
                        "mapped_bytes",
                        "wall_ns",
                    ),
                ),
                "seal": _slim(
                    seal,
                    ("ok", "rm_status", "session_id", "wall_ns"),
                ),
                "initial_park": _slim(
                    initial_park,
                    (
                        "ok",
                        "rm_status",
                        "processed_alias_count",
                        "wall_ns",
                    ),
                ),
                "rebind": _slim(
                    rebind,
                    (
                        "ok",
                        "rm_status",
                        "session_id",
                        "expected_consumer_alias_count",
                        "committed_consumer_alias_count",
                        "total_alias_count",
                        "total_alias_bytes",
                        "batch_count",
                        "ioctl_calls",
                        "ioctl_wall_ns",
                        "wall_ns",
                    ),
                ),
                "destroy": _slim(
                    destroy,
                    ("ok", "rm_status", "session_id", "wall_ns"),
                ),
            },
            "cleanup": cleanup,
        }
        return result
    finally:
        if session_id:
            cleanup["fallback_destroy"] = (
                static_alias.driver_static_alias_destroy(
                    session_id,
                    device=0,
                    release_backing=True,
                    synchronize=True,
                )
            )
        if 0 in uvm._uvm_slot_direct_sessions:
            cleanup["fallback_slot_close"] = uvm.close_uvm_slot_direct_session(
                device=0
            )
        upload_arena = None
        consumer_arena = None
        upload_allocation = None
        consumer_allocation = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        uvm.release_uvm_local_memory_pool()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    result: dict[str, object] = {}
    try:
        result = run()
    except Exception as exc:
        result = {
            "ok": False,
            "error": repr(exc),
            "single_run": True,
            "statistical_repetitions": 0,
            "vllm_model_loaded": False,
        }
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
