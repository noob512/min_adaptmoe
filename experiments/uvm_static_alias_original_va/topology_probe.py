#!/usr/bin/env python3
"""Multi-region, dual-bank original-VA static-alias Graph probe."""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import time
from pathlib import Path

import torch

from vllm.device_allocator import uvm
from vllm.device_allocator import uvm_static_alias as static_alias


REGION_BYTES = 512 * 1024
VA_BLOCK_BYTES = 2 * 1024 * 1024
BANKS = 2
BASE_LAYER_ID = 8100


def _require(result: dict[str, object], label: str) -> None:
    if not bool(result.get("ok")):
        raise RuntimeError(f"{label} failed: {result}")


def _all_equal(tensor: torch.Tensor, value: int) -> bool:
    return bool(torch.all(tensor == int(value)).item())


def _map_in_batches(
    entries: list[tuple[int, int, int, int, int, int, int]],
) -> tuple[int, list[dict[str, object]]]:
    session_id = 0
    results: list[dict[str, object]] = []
    maximum = static_alias.UVM_MOE_STATIC_ALIAS_MAX_ENTRIES
    for offset in range(0, len(entries), maximum):
        result = static_alias.driver_static_alias_map(
            entries[offset : offset + maximum],
            session_id=session_id,
            device=0,
        )
        _require(result, f"static alias MAP batch {len(results)}")
        returned = int(result["session_id"])
        if session_id not in (0, returned):
            raise RuntimeError("MAP append changed the static-alias session id")
        session_id = returned
        results.append(result)
    return session_id, results


def run(regions: int, layers: int, replays: int) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if regions <= 0 or regions > 1152:
        raise ValueError("regions must be in [1, 1152]")
    if layers <= 0 or layers % BANKS != 0:
        raise ValueError("layers must be a positive multiple of two")
    aliases_per_slot = 1 + layers // BANKS
    if aliases_per_slot > 64:
        raise ValueError("topology exceeds the driver's 64 aliases per slot")

    torch.cuda.set_device(0)
    uvm.ensure_uvm_runtime_library()
    bank_bytes = regions * REGION_BYTES
    segment_count = BANKS + layers
    arena_bytes = segment_count * bank_bytes
    allocation_bytes = arena_bytes + VA_BLOCK_BYTES
    session_id = 0
    graph: torch.cuda.CUDAGraph | None = None
    allocation: torch.Tensor | None = None
    arena: torch.Tensor | None = None
    source: torch.Tensor | None = None
    graph_out: torch.Tensor | None = None
    cleanup: dict[str, object] = {}

    try:
        with uvm.uvm_local_memory_pool(enable_prefetch=False):
            allocation = torch.empty(
                allocation_bytes, dtype=torch.uint8, device="cuda"
            )
        allocation_va = int(allocation.data_ptr())
        arena_va = (allocation_va + VA_BLOCK_BYTES - 1) & ~(VA_BLOCK_BYTES - 1)
        arena_offset = arena_va - allocation_va
        if arena_offset + arena_bytes > allocation_bytes:
            raise RuntimeError("managed allocation has no aligned topology arena")
        arena = allocation[arena_offset : arena_offset + arena_bytes]
        segments = [
            arena[index * bank_bytes : (index + 1) * bank_bytes]
            for index in range(segment_count)
        ]
        uploads = segments[:BANKS]
        consumers = segments[BANKS:]
        original_patterns = [
            (11 + index * 7) % 251 for index in range(segment_count)
        ]
        for tensor, pattern in zip(segments, original_patterns, strict=True):
            tensor.fill_(pattern)
        torch.cuda.synchronize()

        stable_vas = tuple(int(tensor.data_ptr()) for tensor in segments)
        remote_prepare = uvm.batch_prepare_ranges_for_native_remote_access_with_executor(
            [(allocation_va, allocation_bytes)], sync=True, device=0
        )
        if int(remote_prepare[0]) != 1:
            raise RuntimeError(f"CPU residency preparation failed: {remote_prepare}")

        preallocate_entries: list[tuple[int, int, int, int, int, int, int]] = []
        for bank in range(BANKS):
            for region in range(regions):
                preallocate_entries.append(
                    (
                        int(uploads[bank].data_ptr()) + region * REGION_BYTES,
                        REGION_BYTES,
                        BASE_LAYER_ID + bank,
                        region,
                        region,
                        0,
                        uvm.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT,
                    )
                )
        preallocate_started = time.perf_counter_ns()
        preallocate = uvm.driver_slot_preallocate_ranges(
            preallocate_entries, device=0
        )
        preallocate_wall_ms = (time.perf_counter_ns() - preallocate_started) / 1e6
        _require(preallocate, "dual-bank backing preallocation")

        pte_flag = static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K
        alias_entries: list[tuple[int, int, int, int, int, int, int]] = []
        for bank in range(BANKS):
            layer_indices = range(bank, layers, BANKS)
            for region in range(regions):
                identity = (BASE_LAYER_ID + bank, region, region)
                byte_offset = region * REGION_BYTES
                alias_entries.append(
                    (
                        *identity,
                        static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW | pte_flag,
                        int(uploads[bank].data_ptr()) + byte_offset,
                        REGION_BYTES,
                        0,
                    )
                )
                for layer_index in layer_indices:
                    alias_entries.append(
                        (
                            *identity,
                            pte_flag,
                            int(consumers[layer_index].data_ptr()) + byte_offset,
                            REGION_BYTES,
                            0,
                        )
                    )

        map_started = time.perf_counter_ns()
        session_id, map_results = _map_in_batches(alias_entries)
        map_wall_ms = (time.perf_counter_ns() - map_started) / 1e6
        expected_alias_count = BANKS * regions * aliases_per_slot
        expected_alias_bytes = expected_alias_count * REGION_BYTES
        if int(map_results[-1]["total_alias_bytes"]) != expected_alias_bytes:
            raise RuntimeError("kernel MAP alias-byte total disagrees with topology")

        seal_started = time.perf_counter_ns()
        seal = static_alias.driver_static_alias_seal(
            session_id,
            expected_alias_count=expected_alias_count,
            expected_alias_bytes=expected_alias_bytes,
            device=0,
        )
        seal_wall_ms = (time.perf_counter_ns() - seal_started) / 1e6
        _require(seal, "static alias SEAL")
        if stable_vas != tuple(int(tensor.data_ptr()) for tensor in segments):
            raise RuntimeError("one or more managed consumer VAs changed")

        source = torch.empty(
            layers * bank_bytes, dtype=torch.uint8, pin_memory=True
        )
        source_layers = [
            source[index * bank_bytes : (index + 1) * bank_bytes]
            for index in range(layers)
        ]
        graph_out = torch.empty(
            layers * bank_bytes, dtype=torch.uint8, device="cuda"
        )
        output_layers = [
            graph_out[index * bank_bytes : (index + 1) * bank_bytes]
            for index in range(layers)
        ]
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())

        def enqueue_pipeline() -> None:
            for layer in range(layers):
                uploads[layer % BANKS].copy_(source_layers[layer], non_blocking=True)
                output_layers[layer].copy_(consumers[layer])

        initial_patterns = [(31 + layer * 36) % 251 for layer in range(layers)]
        for tensor, pattern in zip(source_layers, initial_patterns, strict=True):
            tensor.fill_(pattern)
        with torch.cuda.stream(capture_stream):
            enqueue_pipeline()
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        capture_started = time.perf_counter_ns()
        with torch.cuda.graph(graph, stream=capture_stream):
            enqueue_pipeline()
        capture_wall_ms = (time.perf_counter_ns() - capture_started) / 1e6

        replay_ioctl_calls = 0
        original_ioctl = static_alias._libc_ioctl

        def _counting_ioctl(fd: int, request: int, params: ctypes.Structure) -> int:
            nonlocal replay_ioctl_calls
            replay_ioctl_calls += 1
            return int(original_ioctl(fd, request, params))

        correctness_wall_ms: list[float] = []
        static_alias._libc_ioctl = _counting_ioctl
        try:
            for replay_index in range(3):
                patterns = [
                    (replay_index * 53 + layer * 37 + 5) % 251
                    for layer in range(layers)
                ]
                for tensor, pattern in zip(source_layers, patterns, strict=True):
                    tensor.fill_(pattern)
                started = time.perf_counter_ns()
                graph.replay()
                torch.cuda.synchronize()
                correctness_wall_ms.append(
                    (time.perf_counter_ns() - started) / 1e6
                )
                for layer, pattern in enumerate(patterns):
                    if not _all_equal(output_layers[layer], pattern):
                        raise RuntimeError(
                            f"layer {layer} mismatch after replay {replay_index}"
                        )

            measured_patterns = [
                (23 + layer * 48) % 251 for layer in range(layers)
            ]
            for tensor, pattern in zip(
                source_layers, measured_patterns, strict=True
            ):
                tensor.fill_(pattern)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            submit_started = time.perf_counter_ns()
            for _ in range(replays):
                graph.replay()
            submit_wall_ms = (time.perf_counter_ns() - submit_started) / 1e6
            end_event.record()
            end_event.synchronize()
            gpu_total_ms = float(start_event.elapsed_time(end_event))
        finally:
            static_alias._libc_ioctl = original_ioctl

        for layer, pattern in enumerate(measured_patterns):
            if not _all_equal(output_layers[layer], pattern):
                raise RuntimeError(f"layer {layer} final output mismatch")

        torch.cuda.synchronize()
        del graph
        graph = None
        capture_stream.synchronize()
        destroy_started = time.perf_counter_ns()
        destroy = static_alias.driver_static_alias_destroy(
            session_id,
            device=0,
            release_backing=True,
            synchronize=True,
        )
        destroy_wall_ms = (time.perf_counter_ns() - destroy_started) / 1e6
        _require(destroy, "static alias DESTROY")
        session_id = 0

        restored = True
        for tensor, pattern in zip(segments, original_patterns, strict=True):
            restored = restored and _all_equal(tensor.clone(), pattern)
        torch.cuda.synchronize()
        if not restored:
            raise RuntimeError("DESTROY did not restore every CPU backing segment")

        cleanup = uvm.close_uvm_slot_direct_session(device=0)
        return {
            "ok": True,
            "mechanism": "uvm_original_va_static_alias_dual_bank_topology",
            "regions_per_bank": regions,
            "region_bytes": REGION_BYTES,
            "bank_bytes": bank_bytes,
            "bank_count": BANKS,
            "layer_count": layers,
            "unique_backing_bytes": BANKS * bank_bytes,
            "alias_count": expected_alias_count,
            "alias_bytes": expected_alias_bytes,
            "original_vas_stable": stable_vas
            == tuple(int(tensor.data_ptr()) for tensor in segments),
            "preallocate": preallocate,
            "map_batches": map_results,
            "seal": seal,
            "graph": {
                "capture_ms": capture_wall_ms,
                "correctness_replay_ms": correctness_wall_ms,
                "correctness_replay_median_ms": statistics.median(
                    correctness_wall_ms
                ),
                "measured_replays": replays,
                "gpu_total_ms": gpu_total_ms,
                "gpu_ms_per_replay": gpu_total_ms / replays,
                "host_submit_total_ms": submit_wall_ms,
                "host_submit_us_per_replay": submit_wall_ms * 1000 / replays,
                "custom_static_alias_ioctls_during_replay": replay_ioctl_calls,
                "h2d_bytes_per_replay": layers * bank_bytes,
                "consumer_d2d_bytes_per_replay": layers * bank_bytes,
            },
            "lifecycle_ms": {
                "preallocate": preallocate_wall_ms,
                "map_total": map_wall_ms,
                "map_batches": [
                    float(result["wall_ns"]) / 1e6 for result in map_results
                ],
                "seal": seal_wall_ms,
                "destroy": destroy_wall_ms,
            },
            "destroy": destroy,
            "remote_cpu_backing_restored": restored,
            "legacy_slot_session_cleanup": cleanup,
        }
    finally:
        if graph is not None:
            torch.cuda.synchronize()
            del graph
        if session_id:
            cleanup["static_alias_fallback_destroy"] = (
                static_alias.driver_static_alias_destroy(
                    session_id,
                    device=0,
                    release_backing=True,
                    synchronize=True,
                )
            )
        if 0 in uvm._uvm_slot_direct_sessions:
            cleanup["legacy_slot_fallback_close"] = uvm.close_uvm_slot_direct_session(
                device=0
            )
        del graph_out, source, arena, allocation
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        uvm.release_uvm_local_memory_pool()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replays <= 0:
        raise ValueError("replays must be positive")

    started = time.time()
    try:
        result = run(args.regions, args.layers, args.replays)
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
