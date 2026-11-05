#!/usr/bin/env python3
"""Force a mid-prepare cmd97 failure and verify convergent cmd99 cleanup."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from probe import (
    CPU_PROCESSOR_UUID,
    UVM_MIGRATE,
    _all_equal,
    _migrate_params,
    _require,
    _submit_uvm_ioctl,
)
from vllm.device_allocator import uvm
from vllm.device_allocator import uvm_static_alias as static_alias


REGION_BYTES = 512 * 1024
VA_BLOCK_BYTES = 2 * 1024 * 1024
ALLOCATION_BYTES = 3 * VA_BLOCK_BYTES
LAYER_ID = 8200


def run() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    torch.cuda.set_device(0)
    uvm.ensure_uvm_runtime_library()
    allocation: torch.Tensor | None = None
    gpu_resident_copy: torch.Tensor | None = None
    session_id = 0
    cleanup: dict[str, object] = {}

    try:
        with uvm.uvm_local_memory_pool(enable_prefetch=False):
            allocation = torch.empty(
                ALLOCATION_BYTES, dtype=torch.uint8, device="cuda"
            )
        allocation_va = int(allocation.data_ptr())
        arena_va = (allocation_va + VA_BLOCK_BYTES - 1) & ~(VA_BLOCK_BYTES - 1)
        arena_offset = arena_va - allocation_va
        if arena_offset + 2 * VA_BLOCK_BYTES > ALLOCATION_BYTES:
            raise RuntimeError("managed allocation lacks two aligned VA blocks")

        first = allocation[arena_offset : arena_offset + REGION_BYTES]
        second_offset = arena_offset + VA_BLOCK_BYTES
        second = allocation[second_offset : second_offset + REGION_BYTES]
        first.fill_(17)
        second.fill_(29)
        torch.cuda.synchronize()
        stable_vas = (int(first.data_ptr()), int(second.data_ptr()))

        remote_prepare = uvm.batch_prepare_ranges_for_native_remote_access_with_executor(
            [(allocation_va, ALLOCATION_BYTES)], sync=True, device=0
        )
        if int(remote_prepare[0]) != 1:
            raise RuntimeError(f"CPU residency preparation failed: {remote_prepare}")

        preallocate = uvm.driver_slot_preallocate_ranges(
            [
                (
                    int(first.data_ptr()),
                    REGION_BYTES,
                    LAYER_ID,
                    0,
                    0,
                    0,
                    uvm.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT,
                ),
                (
                    int(second.data_ptr()),
                    REGION_BYTES,
                    LAYER_ID,
                    1,
                    1,
                    0,
                    uvm.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT,
                ),
            ],
            device=0,
        )
        _require(preallocate, "backing preallocation")
        legacy_fd = int(uvm._uvm_slot_direct_sessions[0][0])

        # Make only the second, higher-VA target GPU-resident. cmd97 sorts by
        # VA: it therefore invalidates/prepares the first CPU target before
        # revalidation rejects this second target. No slot PTE is published,
        # which exercises teardown metadata for a prepare-stage partial change.
        migrate_second_gpu = _submit_uvm_ioctl(
            legacy_fd,
            UVM_MIGRATE,
            _migrate_params(
                int(second.data_ptr()),
                REGION_BYTES,
                bytes(uvm._phase_f_gpu_uuid(0).uuid),
            ),
        )
        if not bool(migrate_second_gpu["accepted"]):
            raise RuntimeError(f"GPU migration setup failed: {migrate_second_gpu}")
        gpu_resident_copy = second.clone()
        torch.cuda.synchronize()
        if not _all_equal(gpu_resident_copy, 29):
            raise RuntimeError("GPU-resident failure target was corrupted")

        mapped = static_alias.driver_static_alias_map(
            [
                (LAYER_ID, 0, 0, 0, int(first.data_ptr()), REGION_BYTES, 0),
                (LAYER_ID, 1, 1, 0, int(second.data_ptr()), REGION_BYTES, 0),
            ],
            device=0,
        )
        if bool(mapped.get("ok")):
            raise RuntimeError("directed MAP unexpectedly succeeded")
        if not bool(mapped.get("made_progress")) or not bool(mapped.get("poisoned")):
            raise RuntimeError(f"failed MAP was not retained as destroy-only: {mapped}")
        session_id = int(mapped.get("session_id") or 0)
        if session_id == 0 or int(mapped.get("alias_count") or 0) != 2:
            raise RuntimeError(f"failed MAP lost its teardown topology: {mapped}")

        destroyed = static_alias.driver_static_alias_destroy(
            session_id,
            device=0,
            release_backing=True,
            synchronize=True,
        )
        _require(destroyed, "poisoned-session DESTROY")
        session_id = 0

        # End in the same CPU-resident/remote-readable state expected by a
        # decode phase with automatic migration disabled.
        migrate_second_cpu = _submit_uvm_ioctl(
            legacy_fd,
            UVM_MIGRATE,
            _migrate_params(
                int(second.data_ptr()), REGION_BYTES, CPU_PROCESSOR_UUID
            ),
        )
        if not bool(migrate_second_cpu["accepted"]):
            raise RuntimeError(f"CPU migration recovery failed: {migrate_second_cpu}")

        first_after = first.clone()
        second_after = second.clone()
        torch.cuda.synchronize()
        restored = _all_equal(first_after, 17) and _all_equal(second_after, 29)
        if not restored:
            raise RuntimeError("cmd99 failed to restore original managed contents")
        if stable_vas != (int(first.data_ptr()), int(second.data_ptr())):
            raise RuntimeError("original managed VAs changed during failure recovery")

        cleanup = uvm.close_uvm_slot_direct_session(device=0)
        return {
            "ok": True,
            "mechanism": "uvm_static_alias_mid_prepare_failure_recovery",
            "directed_failure": "second_target_gpu_resident",
            "original_vas_stable": True,
            "preallocate": preallocate,
            "migrate_second_gpu": migrate_second_gpu,
            "map_failure": mapped,
            "destroy": destroyed,
            "migrate_second_cpu": migrate_second_cpu,
            "original_cpu_contents_restored": restored,
            "legacy_owner_cache_empty": not bool(
                uvm._uvm_slot_direct_owners.get(0)
            ),
            "legacy_slot_session_cleanup": cleanup,
        }
    finally:
        if session_id:
            cleanup["fallback_destroy"] = static_alias.driver_static_alias_destroy(
                session_id,
                device=0,
                release_backing=True,
                synchronize=True,
            )
        if 0 in uvm._uvm_slot_direct_sessions:
            cleanup["fallback_legacy_close"] = uvm.close_uvm_slot_direct_session(
                device=0
            )
        del gpu_resident_copy, allocation
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        uvm.release_uvm_local_memory_pool()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    try:
        result = run()
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
