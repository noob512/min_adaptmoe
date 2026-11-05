#!/usr/bin/env python3
"""Measure four 576-MiB pinned H2D layers after controlled idle gaps."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path
from typing import Any


MIB = 1024 * 1024
LAYER_BYTES = 576 * MIB
TOTAL_BYTES = 4 * LAYER_BYTES
LAYER_SEGMENTS = (384 * MIB, 192 * MIB)
DEFAULT_GAPS = "0,1,5,10,20,30,50,80,110,250,500,500,250,110,80,50,30,20,10,5,1,0"


def load_clock_module() -> Any:
    path = Path(__file__).resolve().with_name("benchmark_clock_phases.py")
    spec = importlib.util.spec_from_file_location("pinned_idle_clock_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_gaps(value: str) -> list[int]:
    try:
        gaps = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid gap list") from exc
    if not gaps or any(gap < 0 or gap > 10_000 for gap in gaps):
        raise argparse.ArgumentTypeError("gaps must be in [0, 10000] ms")
    return gaps


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "samples": values,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gaps-ms", type=parse_gaps, default=parse_gaps(DEFAULT_GAPS))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.warmups < 1 or args.device < 0:
        parser.error("warmups must be positive and device non-negative")

    import torch

    helpers = load_clock_module()
    sampler = helpers.NvmlSampler(args.device)
    torch.cuda.set_device(args.device)
    source = torch.empty(TOTAL_BYTES, dtype=torch.uint8, pin_memory=True)
    destination = torch.empty(TOTAL_BYTES, dtype=torch.uint8, device="cuda")
    source.fill_(0x5A)
    stream = torch.cuda.current_stream(args.device)
    ranges: list[list[tuple[int, int]]] = []
    offset = 0
    for _ in range(4):
        layer: list[tuple[int, int]] = []
        for size in LAYER_SEGMENTS:
            layer.append((offset, size))
            offset += size
        ranges.append(layer)
    if offset != TOTAL_BYTES or not source.is_pinned():
        raise RuntimeError("probe geometry or pinned allocation is invalid")

    def copy_batch(measure: bool) -> dict[str, Any] | None:
        starts = [torch.cuda.Event(enable_timing=True) for _ in ranges]
        ends = [torch.cuda.Event(enable_timing=True) for _ in ranges]
        wall_started = time.perf_counter()
        host_started = time.perf_counter()
        for layer_index, layer in enumerate(ranges):
            starts[layer_index].record(stream)
            for begin, size in layer:
                destination.narrow(0, begin, size).copy_(
                    source.narrow(0, begin, size), non_blocking=True
                )
            ends[layer_index].record(stream)
        host_submit_ms = (time.perf_counter() - host_started) * 1000.0
        ends[-1].synchronize()
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        if not measure:
            return None
        layer_ms = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
        return {
            "wall_ms": wall_ms,
            "host_submit_ms": host_submit_ms,
            "cuda_ms": sum(layer_ms),
            "layer_cuda_ms": layer_ms,
        }

    for _ in range(args.warmups):
        copy_batch(False)

    observations: list[dict[str, Any]] = []
    for trial, gap_ms in enumerate(args.gaps_ms):
        copy_batch(False)
        condition_end = time.perf_counter()
        if gap_ms:
            time.sleep(gap_ms / 1000.0)
        actual_gap_ms = (time.perf_counter() - condition_end) * 1000.0
        before = sampler.sample()
        measured = copy_batch(True)
        assert measured is not None
        after = sampler.sample()
        observations.append(
            {
                "trial": trial,
                "requested_gap_ms": gap_ms,
                "actual_gap_ms": actual_gap_ms,
                "before": before,
                "after": after,
                **measured,
            }
        )

    sampler.close()
    torch.cuda.synchronize()
    sample_indices = [0, LAYER_BYTES - 1, LAYER_BYTES, TOTAL_BYTES - 1]
    values = [int(destination[index].item()) for index in sample_indices]
    if values != [0x5A] * len(values):
        raise RuntimeError(f"copy verification failed: {values}")

    by_gap: dict[str, Any] = {}
    for gap in sorted(set(args.gaps_ms)):
        population = [item for item in observations if item["requested_gap_ms"] == gap]
        by_gap[str(gap)] = {
            "wall_ms": summary([float(item["wall_ms"]) for item in population]),
            "cuda_ms": summary([float(item["cuda_ms"]) for item in population]),
            "layer_cuda_ms": [item["layer_cuda_ms"] for item in population],
            "pcie_generation_before": [item["before"]["pcie_generation"] for item in population],
            "pcie_generation_after": [item["after"]["pcie_generation"] for item in population],
            "sm_mhz_before": [item["before"]["sm_mhz"] for item in population],
            "memory_mhz_before": [item["before"]["memory_mhz"] for item in population],
        }
    result = {
        "schema": "pinned_h2d_idle_gap_probe_v1",
        "status": "ok",
        "config": {
            "total_bytes": TOTAL_BYTES,
            "layer_bytes": LAYER_BYTES,
            "layers": 4,
            "segments_per_layer": len(LAYER_SEGMENTS),
            "layer_segments": list(LAYER_SEGMENTS),
            "gaps_ms": args.gaps_ms,
            "warmups": args.warmups,
            "device": args.device,
        },
        "observations": observations,
        "by_gap": by_gap,
        "correctness": {"sample_indices": sample_indices, "values": values},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
