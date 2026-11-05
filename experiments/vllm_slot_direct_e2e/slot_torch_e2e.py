#!/usr/bin/env python3
"""Exercise mode-8 + mode-10 on real PyTorch managed weight views.

The primary geometry mirrors vLLM expert envelopes: adjacent 512 KiB ranges
inside one managed fused allocation. Each range is a partial 2 MiB VA block.
The script keeps every torch Tensor data_ptr unchanged, moves ordinary UVM
residency to CPU, copies/maps the ranges into preallocated driver slots, and
then consumes the original Tensor objects with CUDA GEMM and a full checksum.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
VLLM_ROOT = REPO_ROOT / "workloads" / "vllm" / "vllm"
SHARED_WORKER = (
    REPO_ROOT
    / "experiments"
    / "uvm_512x4m_slot_shared_push_compare"
    / "worker.py"
)
DEFAULT_LIBRARY = SCRIPT_DIR / "libvllm_slot_direct_e2e.so"
DEFAULT_STATS = Path("/proc/driver/nvidia-uvm/latency_stats")

KIB = 1024
MIB = 1024 * KIB
VA_BLOCK_BYTES = 2 * MIB
ENVELOPE_BYTES = 512 * KIB
ELEMENT_BYTES = 2
MATRIX_DIM = 512
CUDA_CPU_DEVICE_ID = -1


def load_shared_worker():
    spec = importlib.util.spec_from_file_location(
        "vllm_slot_e2e_shared_worker", SHARED_WORKER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import shared worker: {SHARED_WORKER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--regions", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--warmup-forwards", type=int, default=2)
    parser.add_argument("--measured-forwards", type=int, default=8)
    parser.add_argument("--layer-id", type=lambda value: int(value, 0), default=0xE2E0)
    parser.add_argument("--expert-id", type=lambda value: int(value, 0), default=0x51A7)
    parser.add_argument("--slot-base", type=lambda value: int(value, 0), default=0)
    args = parser.parse_args()
    if args.device < 0 or not 1 <= args.regions <= 128:
        parser.error("require device>=0 and 1<=regions<=128")
    if not 1 <= args.batch_size <= 128 or args.tokens <= 0:
        parser.error("require 1<=batch-size<=128 and tokens>0")
    if args.warmup_forwards < 0 or args.measured_forwards <= 0:
        parser.error("invalid forward sample counts")
    if args.slot_base < 0 or args.slot_base + args.regions > 2**32:
        parser.error("slot range does not fit uint32")
    return args


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def gpu_snapshot() -> dict[str, Any] | None:
    try:
        import subprocess

        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).splitlines()[0]
        name, used, free, total = [part.strip() for part in output.split(",")]
        return {
            "name": name,
            "used_mib": int(used),
            "free_mib": int(free),
            "total_mib": int(total),
        }
    except Exception:
        return None


def build_unique_block_batches(
    pointers: list[int], batch_size: int
) -> list[list[int]]:
    """Color ranges so one ioctl never receives the same VA block twice."""
    by_block: dict[int, list[int]] = {}
    for index, pointer in enumerate(pointers):
        block = pointer // VA_BLOCK_BYTES
        by_block.setdefault(block, []).append(index)
    batches: list[list[int]] = []
    while any(by_block.values()):
        batch: list[int] = []
        for block in sorted(by_block):
            pending = by_block[block]
            if pending and len(batch) < batch_size:
                batch.append(pending.pop(0))
        if not batch:
            raise RuntimeError("failed to schedule unique-VA-block batch")
        batches.append(batch)
    flattened = [index for batch in batches for index in batch]
    if sorted(flattened) != list(range(len(pointers))):
        raise RuntimeError("batch scheduler lost or duplicated a region")
    for batch in batches:
        blocks = [pointers[index] // VA_BLOCK_BYTES for index in batch]
        if len(blocks) != len(set(blocks)):
            raise RuntimeError("batch contains duplicate VA blocks")
    return batches


def arrays_for_indices(
    pointers: list[int], indices: list[int], slot_base: int
) -> tuple[ctypes.Array, ctypes.Array, ctypes.Array]:
    count = len(indices)
    return (
        (ctypes.c_size_t * count)(*(pointers[index] for index in indices)),
        (ctypes.c_size_t * count)(*([ENVELOPE_BYTES] * count)),
        (ctypes.c_uint32 * count)(*(slot_base + index for index in indices)),
    )


def run_slot_batches(
    shared: Any,
    library: ctypes.CDLL,
    context: ctypes.c_void_p,
    mode: int,
    pointers: list[int],
    batches: list[list[int]],
    args: argparse.Namespace,
) -> Any:
    aggregate = shared.BatchOutcome()
    for indices in batches:
        pointer_array, size_array, slot_array = arrays_for_indices(
            pointers, indices, args.slot_base
        )
        outcome = shared.slot_call(
            library,
            context,
            mode,
            pointer_array,
            size_array,
            slot_array,
            len(indices),
            args.layer_id,
            args.expert_id,
        )
        aggregate = aggregate.plus(outcome)
    expected_bytes = len(pointers) * ENVELOPE_BYTES
    if (
        aggregate.calls != len(batches)
        or aggregate.success != len(pointers)
        or aggregate.failure != 0
        or aggregate.conflict != 0
        or aggregate.remap != (0 if mode == 8 else len(pointers))
        or aggregate.requested_bytes != expected_bytes
        or aggregate.accepted_bytes != expected_bytes
        or aggregate.bad_entries != 0
    ):
        raise RuntimeError(f"slot batch contract failed: {aggregate}")
    return aggregate


def assert_cpu_residency(residency: Any, regions: int) -> None:
    expected_bytes = regions * ENVELOPE_BYTES
    expected = shared_worker.Residency(
        ranges=regions,
        gpu_bytes=0,
        cpu_bytes=expected_bytes,
        any_bytes=expected_bytes,
        unknown_bytes=0,
    )
    if residency != expected:
        raise RuntimeError(f"ordinary source residency={residency}, expected={expected}")


def output_hash(tensor: Any) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def benchmark_forward(
    torch: Any,
    weights: list[Any],
    inputs: Any,
    warmups: int,
    measured: int,
) -> tuple[Any, dict[str, Any]]:
    temporary = torch.empty(
        (inputs.shape[0], MATRIX_DIM),
        device=inputs.device,
        dtype=inputs.dtype,
    )
    accumulator = torch.empty_like(temporary)

    def one_forward() -> Any:
        accumulator.zero_()
        for weight in weights:
            torch.mm(inputs, weight, out=temporary)
            accumulator.add_(temporary)
        return accumulator

    for _ in range(warmups):
        one_forward()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(measured):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        one_forward()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    host_output = one_forward().float().cpu()
    torch.cuda.synchronize()
    return host_output, {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p95_ms": percentile(samples, 95.0),
        "tokens_per_second": inputs.shape[0] * 1000.0 / statistics.median(samples),
    }


def weight_checksums(torch: Any, weights: list[Any]) -> list[float]:
    values = [float(weight.float().sum().item()) for weight in weights]
    torch.cuda.synchronize()
    return values


def require_profile_contract(
    delta: dict[str, int],
    *,
    calls: int,
    regions: int,
    logical_bytes: int,
) -> dict[str, int]:
    expected = {
        "slot_direct_ioctl_calls": calls,
        "slot_direct_per_block_calls": calls,
        "slot_direct_shared_calls": 0,
        "slot_direct_successes": calls,
        "slot_direct_requested_ranges": regions,
        "slot_direct_completed_ranges": regions,
        "slot_direct_requested_bytes": logical_bytes,
        "slot_direct_completed_bytes": logical_bytes,
        "slot_direct_blocks": regions,
        "slot_direct_copy_pushes": regions,
        "slot_direct_tracker_waits": calls,
        "slot_direct_vidmem_allocations": 0,
        "slot_direct_vidmem_evictions": 0,
        "slot_direct_pte_map_ranges": regions,
        "slot_direct_pte_map_bytes": logical_bytes,
        "slot_direct_fallbacks": 0,
        "slot_direct_errors": 0,
    }
    for name, value in expected.items():
        if name not in delta or int(delta[name]) != value:
            raise RuntimeError(f"profile {name}={delta.get(name)!r}/{value}")
    selected = {
        name: int(value)
        for name, value in sorted(delta.items())
        if name.startswith("slot_direct")
    }
    return selected


def main() -> int:
    global shared_worker
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.library.is_file():
        raise FileNotFoundError(args.library)
    if not args.stats.is_file():
        raise FileNotFoundError(args.stats)

    sys.path.insert(0, str(VLLM_ROOT))
    import torch
    from vllm.device_allocator.uvm import (
        release_uvm_local_memory_pool,
        uvm_local_memory_pool,
    )

    shared_worker = load_shared_worker()
    library = shared_worker.load_library(args.library)
    if int(library.uvm512_set_device(args.device)) != 0:
        raise RuntimeError("cudaSetDevice failed")
    torch.cuda.set_device(args.device)

    context = None
    stream = None
    raw = None
    weight = None
    weights: list[Any] = []
    inputs = None
    result: dict[str, Any] = {
        "status": "running",
        "schema": "vllm_slot_direct_torch_e2e_v1",
        "config": {
            **vars(args),
            "library": str(args.library.resolve()),
            "stats": str(args.stats),
            "output": str(args.output),
            "envelope_bytes": ENVELOPE_BYTES,
            "matrix_dim": MATRIX_DIM,
            "dtype": "torch.float16",
            "mode": "mode8_prealloc_then_mode10_per_block_4k",
        },
        "gpu_before": gpu_snapshot(),
    }
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")

    try:
        # One fused managed allocation with adjacent 512 KiB expert envelopes.
        # Start half a MiB into a VA block so the primary case cannot
        # accidentally become the old full-2MiB overwrite geometry.
        alignment_slack = VA_BLOCK_BYTES + ENVELOPE_BYTES
        raw_bytes = alignment_slack + args.regions * ENVELOPE_BYTES
        raw_elements = (raw_bytes + ELEMENT_BYTES - 1) // ELEMENT_BYTES
        with uvm_local_memory_pool(enable_prefetch=False):
            raw = torch.empty(raw_elements, device="cuda", dtype=torch.float16)
        raw_ptr = int(raw.data_ptr())
        first_block = (raw_ptr + VA_BLOCK_BYTES - 1) // VA_BLOCK_BYTES * VA_BLOCK_BYTES
        first_ptr = first_block + ENVELOPE_BYTES
        byte_offset = first_ptr - raw_ptr
        if byte_offset < 0 or byte_offset + args.regions * ENVELOPE_BYTES > raw.numel() * ELEMENT_BYTES:
            raise RuntimeError("managed fused allocation cannot hold aligned envelopes")
        if byte_offset % ELEMENT_BYTES:
            raise RuntimeError("aligned envelope offset is not element aligned")
        offset_elements = byte_offset // ELEMENT_BYTES
        envelope_elements = ENVELOPE_BYTES // ELEMENT_BYTES
        for index in range(args.regions):
            begin = offset_elements + index * envelope_elements
            weight = raw.narrow(0, begin, envelope_elements).view(MATRIX_DIM, MATRIX_DIM)
            weights.append(weight)

        pointers = [int(weight.data_ptr()) for weight in weights]
        original_pointers = tuple(pointers)
        if any(pointer % ENVELOPE_BYTES for pointer in pointers):
            raise RuntimeError("expert envelope is not 512KiB aligned")
        if not ENVELOPE_BYTES < VA_BLOCK_BYTES:
            raise RuntimeError("primary geometry is not a partial VA block")
        if any(
            pointer // VA_BLOCK_BYTES
            != (pointer + ENVELOPE_BYTES - 1) // VA_BLOCK_BYTES
            for pointer in pointers
        ):
            raise RuntimeError("an expert envelope crosses a VA-block boundary")
        ranges = sorted((pointer, pointer + ENVELOPE_BYTES) for pointer in pointers)
        if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
            raise RuntimeError("expert envelopes overlap")

        # Deterministic full-page contents. GEMM consumes every byte; the
        # independent sum catches corruption without relying only on logits.
        for index, weight in enumerate(weights):
            weight.fill_((index + 1) / 4096.0)
            weight.diagonal().add_(0.25)
        inputs = torch.linspace(
            -1.0,
            1.0,
            args.tokens * MATRIX_DIM,
            device="cuda",
            dtype=torch.float32,
        ).to(torch.float16).view(args.tokens, MATRIX_DIM)
        torch.cuda.synchronize()

        baseline_checksums = weight_checksums(torch, weights)
        baseline_output, baseline_forward = benchmark_forward(
            torch,
            weights,
            inputs,
            args.warmup_forwards,
            args.measured_forwards,
        )

        # This constructor only accepts an initialized CUDA-owned UVM fd whose
        # VA space can execute the ioctl-90 probe on the managed pointer. It
        # duplicates that fd; it does not create an unrelated VA space.
        context = library.uvm512_driver_batch_context_create(
            args.device, pointers[0], ENVELOPE_BYTES
        )
        if not context:
            raise RuntimeError(
                "failed to adopt existing CUDA UVM fd: "
                f"{shared_worker.driver_error(library)}"
            )

        all_indices = list(range(args.regions))
        prealloc_batches = [
            all_indices[begin : begin + args.batch_size]
            for begin in range(0, args.regions, args.batch_size)
        ]
        prealloc_started = time.perf_counter_ns()
        prealloc = run_slot_batches(
            shared_worker,
            library,
            context,
            int(library.uvm512_slot_mode_vidmem_prealloc()),
            pointers,
            prealloc_batches,
            args,
        )
        prealloc_ms = (time.perf_counter_ns() - prealloc_started) / 1e6

        # Move only the expert-envelope ranges, not the whole fused allocation.
        pointer_array = (ctypes.c_size_t * args.regions)(*pointers)
        size_array = (ctypes.c_size_t * args.regions)(
            *([ENVELOPE_BYTES] * args.regions)
        )
        stream = library.uvm512_stream_create()
        if not stream:
            raise RuntimeError("failed to create CUDA stream")
        cpu_started = time.perf_counter_ns()
        accepted = int(
            library.uvm512_prefetch_batch(
                pointer_array,
                size_array,
                args.regions,
                CUDA_CPU_DEVICE_ID,
                stream,
            )
        )
        if accepted != args.regions:
            raise RuntimeError(f"CPU prefetch accepted {accepted}/{args.regions}")
        if int(library.uvm512_stream_synchronize(stream)) != 0:
            raise RuntimeError("CPU prefetch synchronize failed")
        cpu_prefetch_ms = (time.perf_counter_ns() - cpu_started) / 1e6
        cpu_residency = shared_worker.query_residency(
            library, context, pointer_array, size_array, args.regions
        )
        assert_cpu_residency(cpu_residency, args.regions)

        direct_batches = build_unique_block_batches(pointers, args.batch_size)
        profile_before = shared_worker.read_profile(args.stats)
        direct_started = time.perf_counter_ns()
        direct = run_slot_batches(
            shared_worker,
            library,
            context,
            int(library.uvm512_slot_mode_direct_per_block_push()),
            pointers,
            direct_batches,
            args,
        )
        direct_ms = (time.perf_counter_ns() - direct_started) / 1e6
        profile_after = shared_worker.read_profile(args.stats)
        profile_delta = shared_worker.profile_delta(profile_before, profile_after)
        logical_bytes = args.regions * ENVELOPE_BYTES
        selected_profile = require_profile_contract(
            profile_delta,
            calls=len(direct_batches),
            regions=args.regions,
            logical_bytes=logical_bytes,
        )

        if tuple(int(weight.data_ptr()) for weight in weights) != original_pointers:
            raise RuntimeError("torch Tensor VA changed after slot-direct remap")
        post_map_residency = shared_worker.query_residency(
            library, context, pointer_array, size_array, args.regions
        )
        assert_cpu_residency(post_map_residency, args.regions)

        slot_checksums = weight_checksums(torch, weights)
        slot_output, slot_forward = benchmark_forward(
            torch,
            weights,
            inputs,
            args.warmup_forwards,
            args.measured_forwards,
        )
        if baseline_checksums != slot_checksums:
            raise RuntimeError(
                f"weight checksum mismatch: {baseline_checksums}/{slot_checksums}"
            )
        torch.testing.assert_close(
            slot_output,
            baseline_output,
            rtol=0.0,
            atol=0.0,
            msg="slot-backed original-VA GEMM differs from GPU-resident baseline",
        )
        post_gemm_residency = shared_worker.query_residency(
            library, context, pointer_array, size_array, args.regions
        )
        assert_cpu_residency(post_gemm_residency, args.regions)

        result.update(
            {
                "status": "ok",
                "geometry": {
                    "raw_managed_base": raw_ptr,
                    "raw_managed_bytes": raw.numel() * ELEMENT_BYTES,
                    "pointers": pointers,
                    "original_pointers_after": [
                        int(weight.data_ptr()) for weight in weights
                    ],
                    "unique_va_blocks": len(
                        {pointer // VA_BLOCK_BYTES for pointer in pointers}
                    ),
                    "regions_per_va_block": {
                        str(block): sum(
                            pointer // VA_BLOCK_BYTES == block for pointer in pointers
                        )
                        for block in sorted(
                            {pointer // VA_BLOCK_BYTES for pointer in pointers}
                        )
                    },
                    "all_ranges_partial_va_blocks": True,
                    "all_ranges_512k_aligned": True,
                    "va_stable": True,
                },
                "fd_contract": {
                    "adopted_existing_cuda_uvm_fd": True,
                    "validated_by_ioctl90_probe": True,
                    "new_unrelated_uvm_va_space_fallback": False,
                },
                "prealloc": {
                    **asdict(prealloc),
                    "wall_ms": prealloc_ms,
                    "outside_mode10_timed_path": True,
                },
                "cpu_prefetch": {
                    "accepted_ranges": accepted,
                    "wall_ms": cpu_prefetch_ms,
                    "ordinary_residency": asdict(cpu_residency),
                },
                "mode10": {
                    **asdict(direct),
                    "wall_ms": direct_ms,
                    "effective_gib_s": logical_bytes / (1024**3) / (direct_ms / 1000.0),
                    "batch_indices": direct_batches,
                    "batch_count": len(direct_batches),
                    "unique_va_block_per_batch": True,
                    "profile_delta": selected_profile,
                    "zero_timed_backing_allocations": (
                        selected_profile["slot_direct_vidmem_allocations"] == 0
                    ),
                    "zero_timed_backing_evictions": (
                        selected_profile["slot_direct_vidmem_evictions"] == 0
                    ),
                },
                "correctness": {
                    "baseline_weight_checksums": baseline_checksums,
                    "slot_weight_checksums": slot_checksums,
                    "checksum_equal": True,
                    "baseline_output_sha256": output_hash(baseline_output),
                    "slot_output_sha256": output_hash(slot_output),
                    "gemm_exact_equal": True,
                    "post_map_ordinary_residency": asdict(post_map_residency),
                    "post_gemm_ordinary_residency": asdict(post_gemm_residency),
                },
                "performance": {
                    "baseline_gpu_resident_forward": baseline_forward,
                    "slot_original_va_forward": slot_forward,
                    "slot_vs_baseline_speed": (
                        baseline_forward["median_ms"] / slot_forward["median_ms"]
                    ),
                },
                "gpu_after": gpu_snapshot(),
            }
        )
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except Exception as exc:
        result["status"] = "exception"
        result["error"] = repr(exc)
        result["gpu_after_error"] = gpu_snapshot()
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        if stream:
            library.uvm512_stream_destroy(stream)
        if context:
            library.uvm512_driver_batch_context_destroy(context)
        weights.clear()
        weight = None
        raw = None
        inputs = None
        gc.collect()
        try:
            torch.cuda.synchronize(args.device)
            torch.cuda.empty_cache()
            release_uvm_local_memory_pool()
        except Exception:
            # Preserve the primary exception/result. Fresh-process execution
            # remains the required harness contract.
            if result.get("status") == "ok":
                raise


if __name__ == "__main__":
    raise SystemExit(main())
