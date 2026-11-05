#!/usr/bin/env python3
"""Exercise two VMM static-alias banks inside one torch CUDA Graph.

The default is a small functional smoke.  Passing ``--layer-mib 576`` uses the
same logical bytes and 1.125-GiB two-bank backing as the Qwen L0-L3 fixture.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any


MIB = 1 << 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layer-mib", type=int, default=8)
    parser.add_argument("--warmup-replays", type=int, default=3)
    parser.add_argument("--measured-replays", type=int, default=20)
    return parser.parse_args()


def tensor_sha256(tensor: Any) -> str:
    contiguous = tensor.contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def set_pattern(tensor: Any, layer: int, generation: int) -> None:
    value = (17 + layer * 53 + generation * 91) & 0xFF
    tensor.fill_(value)
    # Make ordering mistakes visible even if a future implementation happens
    # to compare only a few cache lines.
    markers = (
        (0, (value + 1) & 0xFF),
        (min(63, tensor.numel() - 1), (value + 2) & 0xFF),
        (tensor.numel() // 2, (value + 3) & 0xFF),
        (tensor.numel() - 1, (value + 4) & 0xFF),
    )
    for index, marker in markers:
        tensor[index] = marker


def allocation_info(allocation: Any) -> dict[str, Any]:
    info = allocation.info()
    if not isinstance(info, dict):
        raise RuntimeError(f"StaticAliasAllocation.info() returned {type(info)!r}")
    return dict(info)


def run() -> int:
    args = parse_args()
    if args.layer_mib <= 0:
        raise ValueError("layer-mib must be positive")
    if args.warmup_replays < 0 or args.measured_replays <= 0:
        raise ValueError("invalid replay counts")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or Path(str(args.output) + ".manifest.jsonl")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("", encoding="utf-8")

    def manifest(record: dict[str, Any]) -> None:
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    result: dict[str, Any] = {
        "schema": "vllm_static_alias_graph_smoke_v1",
        "status": "starting",
        "config": {
            "device": args.device,
            "layers": 4,
            "banks": 2,
            "consumers_per_bank": 2,
            "layer_bytes": args.layer_mib * MIB,
            "logical_h2d_bytes_per_replay": 4 * args.layer_mib * MIB,
            "warmup_replays": args.warmup_replays,
            "measured_replays": args.measured_replays,
            "manifest": str(manifest_path.resolve()),
        },
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    allocations: list[Any] = []
    try:
        import torch
        from vllm.device_allocator.cumem import StaticAliasAllocation

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if torch.cuda.device_count() != 1 or args.device != 0:
            raise RuntimeError(
                "qualification smoke is fail-closed for one visible GPU at device 0"
            )
        torch.cuda.set_device(args.device)
        layer_bytes = args.layer_mib * MIB

        allocations = [
            StaticAliasAllocation.create(args.device, layer_bytes, 2)
            for _ in range(2)
        ]
        initial_info = [allocation_info(item) for item in allocations]
        for bank, info in enumerate(initial_info):
            manifest({"event": "static_alias_create", "bank": bank, **info})
            if int(info["requested_size"]) != layer_bytes:
                raise RuntimeError(f"bank {bank} requested-size mismatch: {info}")
            if int(info["backing_bytes"]) != int(info["aligned_size"]):
                raise RuntimeError(f"bank {bank} backing-size mismatch: {info}")
            if int(info["aligned_size"]) % int(info["granularity"]):
                raise RuntimeError(f"bank {bank} is not granularity aligned: {info}")
            access_pair = (info.get("upload_access"), info.get("consumer_access"))
            if access_pair != ("rw", "ro"):
                raise RuntimeError(f"bank {bank} access contract differs: {info}")
            if len(info["consumer_vas"]) != 2:
                raise RuntimeError(f"bank {bank} consumer-count mismatch: {info}")

        upload_tensors = [
            allocation.upload_tensor((layer_bytes,), torch.uint8)
            for allocation in allocations
        ]
        consumer_tensors = [
            allocations[layer % 2].consumer_tensor(
                layer // 2, (layer_bytes,), torch.uint8
            )
            for layer in range(4)
        ]
        pinned_sources = [
            torch.empty(layer_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
            for _ in range(4)
        ]
        snapshots = [
            torch.empty(layer_bytes, dtype=torch.uint8, device="cuda")
            for _ in range(4)
        ]
        for layer, source in enumerate(pinned_sources):
            set_pattern(source, layer, generation=0)

        upload_ptrs = [int(item.data_ptr()) for item in upload_tensors]
        consumer_ptrs = [int(item.data_ptr()) for item in consumer_tensors]
        source_ptrs = [int(item.data_ptr()) for item in pinned_sources]
        if len(set(upload_ptrs + consumer_ptrs)) != 6:
            raise RuntimeError("upload and consumer virtual addresses are not unique")
        if upload_ptrs != [int(item["upload_va"]) for item in initial_info]:
            raise RuntimeError("upload tensor pointers differ from allocator metadata")
        expected_consumer_ptrs = [
            int(initial_info[layer % 2]["consumer_vas"][layer // 2])
            for layer in range(4)
        ]
        if consumer_ptrs != expected_consumer_ptrs:
            raise RuntimeError("consumer tensor pointers differ from allocator metadata")
        if not all(item.is_pinned() for item in pinned_sources):
            raise RuntimeError("one or more host sources are not pinned")
        for bank, allocation in enumerate(allocations):
            manifest(
                {
                    "event": "static_alias_tensor_views_created",
                    "bank": bank,
                    **allocation_info(allocation),
                }
            )

        capture_stream = torch.cuda.Stream(device=args.device)

        def enqueue_pipeline() -> None:
            for layer in range(4):
                bank = layer % 2
                upload_tensors[bank].copy_(pinned_sources[layer], non_blocking=True)
                # A D2D copy is a simple consumer of the read-only alias.  It
                # snapshots data before the same bank is overwritten by L+2.
                snapshots[layer].copy_(consumer_tensors[layer])

        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            enqueue_pipeline()
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            enqueue_pipeline()
        graph.replay()
        torch.cuda.synchronize()

        def verify_generation(generation: int) -> tuple[list[str], list[str]]:
            source_hashes: list[str] = []
            output_hashes: list[str] = []
            for layer in range(4):
                output_cpu = snapshots[layer].cpu()
                if not torch.equal(output_cpu, pinned_sources[layer]):
                    mismatch = int(
                        torch.count_nonzero(output_cpu != pinned_sources[layer]).item()
                    )
                    raise RuntimeError(
                        f"generation {generation} layer {layer} mismatch bytes={mismatch}"
                    )
                source_hashes.append(tensor_sha256(pinned_sources[layer]))
                output_hashes.append(tensor_sha256(output_cpu))
            return source_hashes, output_hashes

        generation0_sources, generation0_outputs = verify_generation(0)

        # Mutating the pinned buffers without recapturing must change replay
        # results.  This proves that replay executes H2D nodes, rather than
        # merely reusing data left in a bank at capture time.
        for layer, source in enumerate(pinned_sources):
            set_pattern(source, layer, generation=1)
        graph.replay()
        torch.cuda.synchronize()
        generation1_sources, generation1_outputs = verify_generation(1)
        if generation0_outputs == generation1_outputs:
            raise RuntimeError("source mutation did not change Graph replay outputs")

        for _ in range(args.warmup_replays):
            graph.replay()
        torch.cuda.synchronize()

        gpu_samples: list[float] = []
        host_submit_samples: list[float] = []
        for _ in range(args.measured_replays):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            host_started = time.perf_counter_ns()
            start_event.record()
            graph.replay()
            end_event.record()
            host_submit_samples.append((time.perf_counter_ns() - host_started) / 1e3)
            end_event.synchronize()
            gpu_samples.append(float(start_event.elapsed_time(end_event)))

        final_sources, final_outputs = verify_generation(1)
        if final_outputs != generation1_outputs or final_sources != generation1_sources:
            raise RuntimeError("soak replay changed output hashes")
        if upload_ptrs != [int(item.data_ptr()) for item in upload_tensors]:
            raise RuntimeError("upload virtual addresses changed after replay")
        if consumer_ptrs != [int(item.data_ptr()) for item in consumer_tensors]:
            raise RuntimeError("consumer virtual addresses changed after replay")
        if source_ptrs != [int(item.data_ptr()) for item in pinned_sources]:
            raise RuntimeError("pinned source addresses changed after replay")

        logical_bytes = 4 * layer_bytes
        gpu_median_ms = statistics.median(gpu_samples)
        result.update(
            {
                "status": "ok",
                "allocator_initial": initial_info,
                "addresses": {
                    "upload_vas": upload_ptrs,
                    "consumer_vas": consumer_ptrs,
                    "pinned_source_vas": source_ptrs,
                    "stable_after_replay": True,
                },
                "correctness": {
                    "generation0_source_sha256": generation0_sources,
                    "generation0_output_sha256": generation0_outputs,
                    "generation1_source_sha256": generation1_sources,
                    "generation1_output_sha256": generation1_outputs,
                    "source_mutation_observed": True,
                    "full_byte_compare": True,
                    "errors": 0,
                },
                "performance": {
                    "timing_scope": "4 pinned H2D copies + 4 alias-to-snapshot D2D copies",
                    "logical_h2d_bytes_per_replay": logical_bytes,
                    "gpu_ms": {
                        "samples": gpu_samples,
                        "median": gpu_median_ms,
                        "mean": statistics.mean(gpu_samples),
                        "min": min(gpu_samples),
                        "max": max(gpu_samples),
                    },
                    "host_submit_us": {
                        "samples": host_submit_samples,
                        "median": statistics.median(host_submit_samples),
                        "mean": statistics.mean(host_submit_samples),
                    },
                    "logical_h2d_gib_per_second": (
                        logical_bytes / (1 << 30) / (gpu_median_ms / 1000.0)
                    ),
                },
            }
        )

        # Teardown order is part of the allocator contract: Graph first, then
        # all tensor wrappers, then the VMM reservations/handles.
        del graph
        del capture_stream
        del snapshots
        del consumer_tensors
        del upload_tensors
        gc.collect()
        torch.cuda.synchronize()
        preclose_info = [allocation_info(item) for item in allocations]
        for bank, (allocation, info) in enumerate(zip(allocations, preclose_info)):
            manifest({"event": "static_alias_before_close", "bank": bank, **info})
            token = int(info["token"])
            allocation.close(synchronize=True)
            manifest({"event": "static_alias_closed", "bank": bank, "token": token})
        result["allocator_before_close"] = preclose_info
        result["allocator_closed"] = True
        allocations.clear()
    except Exception as exc:
        result["status"] = "exception"
        result["error"] = repr(exc)
        manifest({"event": "smoke_exception", "error": repr(exc)})
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        raise

    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
