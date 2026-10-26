#!/usr/bin/env python3
"""One-shot real-GPU probe for partial-block dual parent PTE views."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vllm.device_allocator import uvm
from vllm.device_allocator import uvm_static_alias as static_alias


BLOCK_BYTES = 2 * 1024 * 1024
ALLOCATION_BYTES = 6 * BLOCK_BYTES
COLD_BYTES = 512 * 1024
HOT_PREFIX_BYTES = 768 * 1024
HOT_SUFFIX_BYTES = BLOCK_BYTES - HOT_PREFIX_BYTES - COLD_BYTES
LAYER_ID = 9200
EXPERT_ID = 0
SLOT_ID = 0


def require_ok(result: dict[str, object], label: str) -> None:
    if result.get("ok") is not True:
        raise RuntimeError(f"{label} failed: {result!r}")


def all_equal(tensor: torch.Tensor, value: int) -> bool:
    return bool(torch.all(tensor == value).item())


def run() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(0)
    uvm.ensure_uvm_runtime_library()

    allocation: torch.Tensor | None = None
    session_id = 0
    cleanup: dict[str, object] = {}
    try:
        with uvm.uvm_local_memory_pool(enable_prefetch=False):
            allocation = torch.empty(
                ALLOCATION_BYTES, dtype=torch.uint8, device="cuda"
            )
        allocation_va = int(allocation.data_ptr())
        prefetched, prefetch_executor, prefetch_driver_result = (
            uvm.batch_prefetch_ranges_to_device_with_executor(
                [(allocation_va, ALLOCATION_BYTES)],
                0,
                executor="cpp_batch",
            )
        )
        if int(prefetched) != 1 or int(prefetch_driver_result) != 0:
            raise RuntimeError(
                "initial GPU residency preparation failed: "
                f"count={prefetched}, executor={prefetch_executor!r}, "
                f"driver_result={prefetch_driver_result}"
            )
        torch.cuda.synchronize()
        aligned_va = (allocation_va + BLOCK_BYTES - 1) & ~(BLOCK_BYTES - 1)
        aligned_offset = aligned_va - allocation_va
        if aligned_offset + 2 * BLOCK_BYTES > ALLOCATION_BYTES:
            raise RuntimeError("managed allocation has fewer than two aligned blocks")

        upload = allocation[aligned_offset : aligned_offset + COLD_BYTES]
        target_offset = aligned_offset + BLOCK_BYTES
        target = allocation[target_offset : target_offset + BLOCK_BYTES]
        hot_prefix = target[:HOT_PREFIX_BYTES]
        consumer = target[
            HOT_PREFIX_BYTES : HOT_PREFIX_BYTES + COLD_BYTES
        ]
        hot_suffix = target[HOT_PREFIX_BYTES + COLD_BYTES :]
        upload.fill_(17)
        hot_prefix.fill_(41)
        consumer.fill_(29)
        hot_suffix.fill_(53)
        torch.cuda.synchronize()

        prepared = uvm.batch_prepare_ranges_for_native_remote_access_with_executor(
            [(int(upload.data_ptr()), COLD_BYTES)], sync=True, device=0
        )
        if int(prepared[0]) != 1:
            raise RuntimeError(
                f"upload CPU residency preparation failed: {prepared!r}"
            )

        # Mirror the full E10 initialization contract: apply ReadMostly once
        # to the complete Parameter-sized range so UVM keeps one unsplit 2 MiB
        # policy block, then cut only the selected cold pages to CPU-only.
        target_ranges = [(int(target.data_ptr()), BLOCK_BYTES)]
        full_shadow_advised = int(
            uvm.batch_advise_ranges_read_mostly(target_ranges, device=0)
        )
        full_shadow_prefetched = int(
            uvm.batch_prefetch_ranges_to_cpu(target_ranges)
        )
        if full_shadow_advised != 1 or full_shadow_prefetched != 1:
            raise RuntimeError(
                "full-Parameter immutable shadow preparation failed: "
                f"advise={full_shadow_advised}, "
                f"prefetch={full_shadow_prefetched}"
            )
        torch.cuda.synchronize()
        shadow_cutover = dict(
            uvm.driver_cutover_ranges_to_immutable_cpu_shadow(
                [(int(consumer.data_ptr()), COLD_BYTES)], device=0
            )
        )
        require_ok(shadow_cutover, "cold immutable-shadow cutover")

        hot_ranges = [
            (int(hot_prefix.data_ptr()), HOT_PREFIX_BYTES),
            (int(hot_suffix.data_ptr()), HOT_SUFFIX_BYTES),
        ]
        hot_prefetched, hot_executor, hot_driver_result = (
            uvm.batch_prefetch_ranges_to_device_with_executor(
                hot_ranges, 0, executor="cpp_batch"
            )
        )
        if int(hot_prefetched) != len(hot_ranges) or int(hot_driver_result) != 0:
            raise RuntimeError(
                "hot-page GPU residency preparation failed: "
                f"count={hot_prefetched}, executor={hot_executor!r}, "
                f"driver_result={hot_driver_result}"
            )
        torch.cuda.synchronize()
        cold_residency = dict(
            uvm.driver_batch_query_ranges_residency_bytes(
                [(int(consumer.data_ptr()), COLD_BYTES)],
                device=0,
                wait_for_trackers=False,
            )
        )
        hot_residency = dict(
            uvm.driver_batch_query_ranges_residency_bytes(
                hot_ranges, device=0, wait_for_trackers=False
            )
        )
        if (
            cold_residency.get("ok") is not True
            or int(cold_residency.get("cpu_bytes", -1)) != COLD_BYTES
            or int(cold_residency.get("gpu_bytes", -1)) != 0
        ):
            raise RuntimeError(
                f"cold pages are not CPU-only before MAP: {cold_residency!r}"
            )
        if (
            hot_residency.get("ok") is not True
            or int(hot_residency.get("gpu_bytes", -1))
            != HOT_PREFIX_BYTES + HOT_SUFFIX_BYTES
            or int(hot_residency.get("cpu_bytes", -1))
            != HOT_PREFIX_BYTES + HOT_SUFFIX_BYTES
        ):
            raise RuntimeError(
                "ordinary pages did not retain their GPU copy and full-"
                f"Parameter CPU shadow before MAP: {hot_residency!r}"
            )

        preallocate = dict(
            uvm.driver_slot_preallocate_ranges(
                [
                    (
                        int(upload.data_ptr()),
                        COLD_BYTES,
                        LAYER_ID,
                        EXPERT_ID,
                        SLOT_ID,
                        0,
                        uvm.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT,
                    )
                ],
                device=0,
            )
        )
        require_ok(preallocate, "slot preallocate")

        entries = (
            (
                LAYER_ID,
                EXPERT_ID,
                SLOT_ID,
                static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW,
                int(upload.data_ptr()),
                COLD_BYTES,
                0,
            ),
            (
                LAYER_ID,
                EXPERT_ID,
                SLOT_ID,
                0,
                int(consumer.data_ptr()),
                COLD_BYTES,
                0,
            ),
        )
        mapped = dict(static_alias.driver_static_alias_map(entries, device=0))
        require_ok(mapped, "MAP")
        session_id = int(mapped["session_id"])

        seal_started = time.perf_counter_ns()
        sealed = dict(
            static_alias.driver_static_alias_seal(
                session_id,
                expected_alias_count=2,
                expected_alias_bytes=2 * COLD_BYTES,
                device=0,
            )
        )
        seal_wall_ns = time.perf_counter_ns() - seal_started
        require_ok(sealed, "SEAL and dual-view prepare")

        source = torch.full(
            (COLD_BYTES,), 73, dtype=torch.uint8, pin_memory=True
        )
        upload.copy_(source, non_blocking=True)
        gpu_view = consumer.clone()
        hot_prefix_gpu_view = hot_prefix.clone()
        hot_suffix_gpu_view = hot_suffix.clone()
        torch.cuda.synchronize()
        if not all_equal(gpu_view, 73):
            raise RuntimeError("GPU slot view did not observe uploaded value")
        if not all_equal(hot_prefix_gpu_view, 41) or not all_equal(
            hot_suffix_gpu_view, 53
        ):
            raise RuntimeError("ordinary GPU pages changed in Prefill view")

        park_started = time.perf_counter_ns()
        parked = dict(
            static_alias.driver_static_alias_park(
                session_id, device=0, synchronize=True
            )
        )
        park_wall_ns = time.perf_counter_ns() - park_started
        require_ok(parked, "GPU-to-CPU parent switch")
        cpu_view = consumer.clone()
        hot_prefix_cpu_view = hot_prefix.clone()
        hot_suffix_cpu_view = hot_suffix.clone()
        torch.cuda.synchronize()
        if not all_equal(cpu_view, 29):
            raise RuntimeError("CPU shadow view did not preserve original value")
        if not all_equal(hot_prefix_cpu_view, 41) or not all_equal(
            hot_suffix_cpu_view, 53
        ):
            raise RuntimeError("ordinary GPU pages changed in Decode view")

        source.fill_(91)
        upload.copy_(source, non_blocking=True)
        torch.cuda.synchronize()
        resume_started = time.perf_counter_ns()
        resumed = dict(
            static_alias.driver_static_alias_resume(
                session_id, device=0, synchronize=True
            )
        )
        resume_wall_ns = time.perf_counter_ns() - resume_started
        require_ok(resumed, "CPU-to-GPU parent switch")
        gpu_view_after = consumer.clone()
        hot_prefix_gpu_view_after = hot_prefix.clone()
        hot_suffix_gpu_view_after = hot_suffix.clone()
        torch.cuda.synchronize()
        if not all_equal(gpu_view_after, 91):
            raise RuntimeError("resumed GPU slot view did not observe new value")
        if not all_equal(hot_prefix_gpu_view_after, 41) or not all_equal(
            hot_suffix_gpu_view_after, 53
        ):
            raise RuntimeError("ordinary GPU pages changed after RESUME")

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
            "schema": "uvm_static_alias_dual_parent_partial_block_probe_v4",
            "block_bytes": BLOCK_BYTES,
            "cold_bytes": COLD_BYTES,
            "ordinary_gpu_bytes": HOT_PREFIX_BYTES + HOT_SUFFIX_BYTES,
            "pte_mode": "4K",
            "expected_dual_blocks": 1,
            "values": {
                "original_cpu": 29,
                "first_gpu_slot": 73,
                "second_gpu_slot": 91,
                "hot_prefix": 41,
                "hot_suffix": 53,
            },
            "initial_prefetch": {
                "count": int(prefetched),
                "executor": str(prefetch_executor),
                "driver_result": int(prefetch_driver_result),
            },
            "mixed_block_setup": {
                "full_parameter_shadow_advised": full_shadow_advised,
                "full_parameter_shadow_prefetched": full_shadow_prefetched,
                "cold_shadow_cutover": shadow_cutover,
                "hot_prefetch_count": int(hot_prefetched),
                "hot_prefetch_executor": str(hot_executor),
                "hot_prefetch_driver_result": int(hot_driver_result),
                "cold_residency": cold_residency,
                "hot_residency": hot_residency,
            },
            "preallocate": preallocate,
            "map": mapped,
            "seal": sealed,
            "park": parked,
            "resume": resumed,
            "destroy": destroyed,
            "timing_ms": {
                "seal_and_prepare": seal_wall_ns / 1e6,
                "park_parent_switch": park_wall_ns / 1e6,
                "resume_parent_switch": resume_wall_ns / 1e6,
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
