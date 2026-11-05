#!/usr/bin/env python3
"""Real-GPU probe for the immutable CPU shadow cutover action."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch

from vllm.device_allocator import uvm


VA_BLOCK_BYTES = 2 * 1024 * 1024
ALLOCATION_BYTES = 3 * VA_BLOCK_BYTES
VALUE = 73


def _trace(stage: str, **values: object) -> None:
    print(json.dumps({"trace_stage": stage, **values}, sort_keys=True), flush=True)


def _residency(ranges: list[tuple[int, int]]) -> dict[str, object]:
    return dict(
        uvm.driver_batch_query_ranges_residency_bytes(
            ranges, device=0, wait_for_trackers=False
        )
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(cycles: int) -> dict[str, object]:
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    torch.cuda.set_device(0)
    uvm.ensure_uvm_runtime_library()
    allocation: torch.Tensor | None = None
    region: torch.Tensor | None = None
    remote_copy: torch.Tensor | None = None
    result: dict[str, object] = {}
    try:
        with uvm.uvm_local_memory_pool(enable_prefetch=False):
            allocation = torch.empty(
                ALLOCATION_BYTES, dtype=torch.uint8, device="cuda"
            )
        allocation_va = int(allocation.data_ptr())
        region_va = (allocation_va + VA_BLOCK_BYTES - 1) & ~(
            VA_BLOCK_BYTES - 1
        )
        region_offset = region_va - allocation_va
        region = allocation[region_offset : region_offset + VA_BLOCK_BYTES]
        ranges = [(region_va, VA_BLOCK_BYTES)]
        _trace(
            "allocated",
            allocation_va=allocation_va,
            allocation_bytes=ALLOCATION_BYTES,
            region_va=region_va,
            region_bytes=VA_BLOCK_BYTES,
        )
        initial_prefetch = int(
            uvm.batch_prefetch_ranges_to_device(
                [(allocation_va, ALLOCATION_BYTES)], device=0
            )
        )
        torch.cuda.synchronize()
        _require(initial_prefetch == 1, "initial GPU residency failed")
        _trace("initial_gpu_residency", success_count=initial_prefetch)
        region.fill_(VALUE)
        torch.cuda.synchronize()
        _trace("filled")

        advised = int(uvm.batch_advise_ranges_read_mostly(ranges, device=0))
        prefetched_cpu = int(uvm.batch_prefetch_ranges_to_cpu(ranges))
        torch.cuda.synchronize()
        before = _residency(ranges)
        _trace("shadow_ready", residency=before)
        _require(advised == 1 and prefetched_cpu == 1, "shadow setup failed")
        _require(
            int(before.get("cpu_bytes", -1)) == VA_BLOCK_BYTES
            and int(before.get("gpu_bytes", -1)) == VA_BLOCK_BYTES,
            f"shadow is not CPU/GPU resident: {before!r}",
        )

        cycle_results: list[dict[str, object]] = []
        for cycle_index in range(cycles):
            cycle_number = cycle_index + 1
            cutover_start = time.perf_counter_ns()
            cutover = dict(
                uvm.driver_cutover_ranges_to_immutable_cpu_shadow(
                    ranges, device=0
                )
            )
            cutover_ms = (time.perf_counter_ns() - cutover_start) / 1e6
            _trace(
                "cutover_complete",
                cycle=cycle_number,
                cutover=cutover,
                wall_ms=cutover_ms,
            )
            _require(
                cutover.get("ok") is True,
                f"cycle {cycle_number} cutover failed: {cutover!r}",
            )
            after = _residency(ranges)
            _trace(
                "cutover_residency", cycle=cycle_number, residency=after
            )
            _require(
                int(after.get("cpu_bytes", -1)) == VA_BLOCK_BYTES
                and int(after.get("gpu_bytes", -1)) == 0,
                f"cycle {cycle_number} cutover residency is not CPU-only: "
                f"{after!r}",
            )

            remote_read_start = time.perf_counter_ns()
            if remote_copy is None:
                remote_copy = torch.empty_like(region, device="cuda")
                _trace(
                    "remote_copy_allocated",
                    output_va=int(remote_copy.data_ptr()),
                )
            remote_copy.copy_(region)
            torch.cuda.synchronize()
            _trace("remote_copy_complete", cycle=cycle_number)
            remote_ok = bool(torch.all(remote_copy == VALUE).item())
            torch.cuda.synchronize()
            remote_read_ms = (
                time.perf_counter_ns() - remote_read_start
            ) / 1e6
            _require(
                remote_ok,
                f"cycle {cycle_number} remote read returned incorrect bytes",
            )

            restore_start = time.perf_counter_ns()
            restored, executor, driver_status = (
                uvm.batch_prefetch_ranges_to_device_with_executor(
                    ranges, 0, executor="cpp_batch"
                )
            )
            torch.cuda.synchronize()
            restore_ms = (time.perf_counter_ns() - restore_start) / 1e6
            restored_residency = _residency(ranges)
            _trace(
                "restore_complete",
                cycle=cycle_number,
                restored=int(restored),
                residency=restored_residency,
                wall_ms=restore_ms,
            )
            _require(
                int(restored) == 1,
                f"cycle {cycle_number} GPU restore failed",
            )
            _require(
                int(restored_residency.get("cpu_bytes", -1))
                == VA_BLOCK_BYTES
                and int(restored_residency.get("gpu_bytes", -1))
                == VA_BLOCK_BYTES,
                f"cycle {cycle_number} restore did not retain the reusable "
                f"CPU shadow: {restored_residency!r}",
            )
            restored_ok = bool(torch.all(region == VALUE).item())
            _require(
                restored_ok,
                f"cycle {cycle_number} restored GPU copy returned incorrect "
                "bytes",
            )
            if (
                cycle_number == 1
                and cycles > 1
                and os.environ.get("IMMUTABLE_SHADOW_MUTATE_AFTER_RESTORE")
                == "1"
            ):
                region.fill_(VALUE + 1)
                torch.cuda.synchronize()
                _trace("gpu_mutation_complete", cycle=cycle_number)
            cycle_results.append(
                {
                    "cycle": cycle_number,
                    "cutover": cutover,
                    "cutover_ms": cutover_ms,
                    "after_cutover_residency": after,
                    "remote_read_ok": remote_ok,
                    "remote_read_ms": remote_read_ms,
                    "restore": {
                        "success_count": int(restored),
                        "executor": str(executor),
                        "driver_status": int(driver_status),
                        "wall_ms": restore_ms,
                        "residency": restored_residency,
                        "content_ok": restored_ok,
                    },
                }
            )

        result = {
            "schema": "uvm_immutable_cpu_shadow_probe_v1",
            "status": "ok",
            "range": {"base": region_va, "length": VA_BLOCK_BYTES},
            "setup": {
                "readmostly_count": advised,
                "prefetched_cpu_count": prefetched_cpu,
                "residency": before,
            },
            "cycles": cycle_results,
        }
        return result
    finally:
        # Views keep the MemPool allocation alive even after ``allocation`` is
        # cleared.  Drop every tensor reference while the UVM helper library is
        # still loaded, then explicitly destroy the process-local pool.  This
        # avoids leaving managed allocations to Python's module-finalization
        # order, which can otherwise call the pluggable allocator after its
        # shared library has begun teardown.
        remote_copy = None
        region = None
        allocation = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        uvm.release_uvm_local_memory_pool()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=2)
    # The generic module-cycle wrapper always supplies this argument. It is
    # intentionally ignored because this probe has explicit cutover cycles.
    parser.add_argument("--replays", type=int, default=0)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("--cycles must be positive")
    try:
        payload = run(args.cycles)
    except Exception as exc:
        payload = {
            "schema": "uvm_immutable_cpu_shadow_probe_v1",
            "status": "error",
            "error": repr(exc),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
