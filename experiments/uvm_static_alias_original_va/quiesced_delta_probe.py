#!/usr/bin/env python3
"""Small real-GPU lifecycle probe for sealed/parked alias delta updates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from vllm.device_allocator import uvm
from vllm.device_allocator import uvm_static_alias as static_alias


REGION_BYTES = 512 * 1024
VA_BLOCK_BYTES = 2 * 1024 * 1024
LAYER_ID = 9100


def _require(result: dict[str, object], label: str) -> None:
    if result.get("ok") is not True:
        raise RuntimeError(f"{label} failed: {result!r}")


def _all_equal(tensor: torch.Tensor, value: int) -> bool:
    return bool(torch.all(tensor == int(value)).item())


def run() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(0)
    uvm.ensure_uvm_runtime_library()
    allocation: torch.Tensor | None = None
    session_id = 0
    cleanup: dict[str, object] = {}

    try:
        allocation_bytes = 4 * REGION_BYTES + VA_BLOCK_BYTES
        with uvm.uvm_local_memory_pool(enable_prefetch=False):
            allocation = torch.empty(
                allocation_bytes, dtype=torch.uint8, device="cuda"
            )
        allocation_va = int(allocation.data_ptr())
        arena_va = (allocation_va + VA_BLOCK_BYTES - 1) & ~(VA_BLOCK_BYTES - 1)
        arena_offset = arena_va - allocation_va
        arena = allocation[arena_offset : arena_offset + 4 * REGION_BYTES]
        upload0, upload1, consumer0, consumer1 = (
            arena[index * REGION_BYTES : (index + 1) * REGION_BYTES]
            for index in range(4)
        )
        for tensor, value in zip(
            (upload0, upload1, consumer0, consumer1),
            (11, 22, 33, 44),
            strict=True,
        ):
            tensor.fill_(value)
        torch.cuda.synchronize()

        prepared = uvm.batch_prepare_ranges_for_native_remote_access_with_executor(
            [(allocation_va, allocation_bytes)], sync=True, device=0
        )
        if int(prepared[0]) != 1:
            raise RuntimeError(f"CPU residency preparation failed: {prepared!r}")

        def preallocate(expert: int, upload: torch.Tensor) -> dict[str, object]:
            return dict(
                uvm.driver_slot_preallocate_ranges(
                    [
                        (
                            int(upload.data_ptr()),
                            REGION_BYTES,
                            LAYER_ID,
                            expert,
                            expert,
                            0,
                            uvm.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT,
                        )
                    ],
                    device=0,
                )
            )

        def alias_entries(
            expert: int, upload: torch.Tensor, consumer: torch.Tensor
        ) -> tuple[tuple[int, int, int, int, int, int, int], ...]:
            return (
                (
                    LAYER_ID,
                    expert,
                    expert,
                    static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW,
                    int(upload.data_ptr()),
                    REGION_BYTES,
                    0,
                ),
                (
                    LAYER_ID,
                    expert,
                    expert,
                    0,
                    int(consumer.data_ptr()),
                    REGION_BYTES,
                    0,
                ),
            )

        entries0 = alias_entries(0, upload0, consumer0)
        entries1 = alias_entries(1, upload1, consumer1)
        preallocate0 = preallocate(0, upload0)
        _require(preallocate0, "initial backing preallocation")
        mapped0 = dict(static_alias.driver_static_alias_map(entries0, device=0))
        _require(mapped0, "initial MAP")
        session_id = int(mapped0["session_id"])
        sealed = dict(static_alias.driver_static_alias_seal(session_id, device=0))
        _require(sealed, "initial SEAL")

        preallocate1 = preallocate(1, upload1)
        _require(preallocate1, "sealed-session delta preallocation")
        append_started = time.perf_counter_ns()
        appended = dict(
            static_alias.driver_static_alias_map(
                entries1,
                session_id=session_id,
                device=0,
                flags=static_alias.UVM_MOE_STATIC_ALIAS_MAP_FLAG_QUIESCED_APPEND,
            )
        )
        append_ms = (time.perf_counter_ns() - append_started) / 1e6
        _require(appended, "quiesced append")

        source0 = torch.full(
            (REGION_BYTES,), 71, dtype=torch.uint8, pin_memory=True
        )
        source1 = torch.full(
            (REGION_BYTES,), 83, dtype=torch.uint8, pin_memory=True
        )
        out0 = torch.empty(REGION_BYTES, dtype=torch.uint8, device="cuda")
        out1 = torch.empty(REGION_BYTES, dtype=torch.uint8, device="cuda")
        upload0.copy_(source0, non_blocking=True)
        upload1.copy_(source1, non_blocking=True)
        out0.copy_(consumer0)
        out1.copy_(consumer1)
        torch.cuda.synchronize()
        if not (_all_equal(out0, 71) and _all_equal(out1, 83)):
            raise RuntimeError("appended alias did not share its slot backing")

        parked = dict(
            static_alias.driver_static_alias_park(
                session_id, device=0, synchronize=True
            )
        )
        _require(parked, "PARK before parked-session delta")
        remote_map = dict(
            uvm.driver_prepare_ranges_remote_read_audited(
                [
                    (int(consumer0.data_ptr()), REGION_BYTES),
                    (int(consumer1.data_ptr()), REGION_BYTES),
                ],
                device=0,
            )
        )
        _require(remote_map, "parked consumer remote mapping")

        remove_started = time.perf_counter_ns()
        removed = dict(
            static_alias.driver_static_alias_quiesced_remove(
                session_id, entries1, device=0, synchronize=True
            )
        )
        remove_ms = (time.perf_counter_ns() - remove_started) / 1e6
        _require(removed, "quiesced remove")

        # The live session still owns aliases elsewhere in the same managed
        # allocation. Splitting its preferred-location policy can therefore
        # touch retained aliases and is intentionally skipped; prefetch is the
        # actual residency operation required by the plan-class restore.
        advised = 0
        prefetched, executor, driver_result = (
            uvm.batch_prefetch_ranges_to_device_with_executor(
                [(int(consumer1.data_ptr()), REGION_BYTES)],
                0,
                executor="cpp_batch",
            )
        )
        torch.cuda.synchronize()
        if int(prefetched) != 1:
            raise RuntimeError(
                "removed alias did not restore a prefetchable managed range: "
                f"advised={advised}, prefetched={prefetched}, "
                f"executor={executor!r}, driver_result={driver_result}"
            )
        restored1 = consumer1.clone()
        if not _all_equal(restored1, 44):
            raise RuntimeError("removed alias did not restore original backing")

        resumed = dict(
            static_alias.driver_static_alias_resume(
                session_id, device=0, synchronize=True
            )
        )
        _require(resumed, "RESUME after parked-session delta")
        source0.fill_(97)
        upload0.copy_(source0, non_blocking=True)
        out0.copy_(consumer0)
        torch.cuda.synchronize()
        if not _all_equal(out0, 97):
            raise RuntimeError("retained alias stopped working after delta removal")

        destroyed = dict(
            static_alias.driver_static_alias_destroy(
                session_id, device=0, release_backing=True, synchronize=True
            )
        )
        _require(destroyed, "final DESTROY")
        session_id = 0
        cleanup["slot_session"] = uvm.close_uvm_slot_direct_session(device=0)
        return {
            "ok": True,
            "mechanism": "parked_static_alias_quiesced_remove",
            "region_bytes": REGION_BYTES,
            "append_ms": append_ms,
            "remove_ms": remove_ms,
            "initial_preallocate": preallocate0,
            "delta_preallocate": preallocate1,
            "initial_map": mapped0,
            "seal": sealed,
            "append": appended,
            "park": parked,
            "remote_map": remote_map,
            "remove": removed,
            "removed_range_advised_count": advised,
            "removed_range_advice_requested": False,
            "removed_range_prefetched_count": int(prefetched),
            "removed_range_prefetch_executor": str(executor),
            "removed_range_prefetch_driver_result": int(driver_result),
            "removed_original_backing_restored": True,
            "resume": resumed,
            "retained_alias_still_correct": True,
            "destroy": destroyed,
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
        del allocation
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        uvm.release_uvm_local_memory_pool()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=int, default=1, help=argparse.SUPPRESS)
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
