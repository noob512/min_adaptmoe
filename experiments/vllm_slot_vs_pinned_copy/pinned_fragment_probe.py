#!/usr/bin/env python3
"""Compare 8 large pinned H2D copies with 4,608 512-KiB copies."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any


MIB = 1024 * 1024
TOTAL_BYTES = 2_415_919_104
FRAGMENT_BYTES = 512 * 1024
FRAGMENTS = TOTAL_BYTES // FRAGMENT_BYTES
LARGE_SEGMENTS = [384 * MIB, 192 * MIB] * 4


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, Any]:
    return {
        "samples": values,
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.warmups < 1 or args.samples < 3 or args.device < 0:
        parser.error("require warmups>=1, samples>=3, device>=0")

    import torch

    torch.cuda.set_device(args.device)
    source = torch.empty(TOTAL_BYTES, dtype=torch.uint8, pin_memory=True)
    source.fill_(0x5A)
    destination = torch.empty(TOTAL_BYTES, dtype=torch.uint8, device="cuda")
    if not source.is_pinned():
        raise RuntimeError("source allocation is not pinned")
    stream = torch.cuda.current_stream(args.device)

    large_ranges: list[tuple[int, int]] = []
    offset = 0
    for size in LARGE_SEGMENTS:
        large_ranges.append((offset, size))
        offset += size
    if offset != TOTAL_BYTES or FRAGMENTS != 4_608:
        raise RuntimeError("probe geometry is inconsistent")
    fragment_ranges = [
        (index * FRAGMENT_BYTES, FRAGMENT_BYTES) for index in range(FRAGMENTS)
    ]

    def run_once(ranges: list[tuple[int, int]]) -> tuple[float, float]:
        started_event = torch.cuda.Event(enable_timing=True)
        ended_event = torch.cuda.Event(enable_timing=True)
        started_event.record(stream)
        host_started = time.perf_counter_ns()
        for begin, size in ranges:
            destination.narrow(0, begin, size).copy_(
                source.narrow(0, begin, size), non_blocking=True
            )
        host_submit_ms = (time.perf_counter_ns() - host_started) / 1e6
        ended_event.record(stream)
        ended_event.synchronize()
        return host_submit_ms, float(started_event.elapsed_time(ended_event))

    # ABBA within one process controls allocator, temperature, and order drift.
    modes = {
        "large_8": large_ranges,
        "fragmented_4608": fragment_ranges,
    }
    for mode in ("large_8", "fragmented_4608"):
        for _ in range(args.warmups):
            run_once(modes[mode])
    schedule = ["large_8", "fragmented_4608", "fragmented_4608", "large_8"]
    observations: dict[str, dict[str, list[float]]] = {
        mode: {"host_submit_ms": [], "cuda_ms": []} for mode in modes
    }
    while any(len(value["cuda_ms"]) < args.samples for value in observations.values()):
        for mode in schedule:
            if len(observations[mode]["cuda_ms"]) >= args.samples:
                continue
            host_ms, cuda_ms = run_once(modes[mode])
            observations[mode]["host_submit_ms"].append(host_ms)
            observations[mode]["cuda_ms"].append(cuda_ms)

    torch.cuda.synchronize()
    checksum = int(destination.sum(dtype=torch.int64).item())
    expected_checksum = TOTAL_BYTES * 0x5A
    if checksum != expected_checksum:
        raise RuntimeError(f"checksum mismatch: {checksum}/{expected_checksum}")

    result: dict[str, Any] = {
        "status": "ok",
        "schema": "pinned_fragment_probe_v1",
        "config": {
            "total_bytes": TOTAL_BYTES,
            "total_gib": TOTAL_BYTES / 1024**3,
            "fragment_bytes": FRAGMENT_BYTES,
            "fragments": FRAGMENTS,
            "large_segments": LARGE_SEGMENTS,
            "large_calls": len(LARGE_SEGMENTS),
            "warmups": args.warmups,
            "samples": args.samples,
            "device": args.device,
        },
        "correctness": {
            "source_pinned": True,
            "checksum": checksum,
            "expected_checksum": expected_checksum,
            "equal": True,
        },
        "modes": {},
    }
    for mode, values in observations.items():
        cuda_summary = summarize(values["cuda_ms"])
        host_summary = summarize(values["host_submit_ms"])
        result["modes"][mode] = {
            "copy_calls": len(modes[mode]),
            "host_submit_ms": host_summary,
            "cuda_ms": cuda_summary,
            "effective_gib_s": (TOTAL_BYTES / 1024**3)
            / (float(cuda_summary["median"]) / 1000.0),
        }
    large_ms = float(result["modes"]["large_8"]["cuda_ms"]["median"])
    fragment_ms = float(
        result["modes"]["fragmented_4608"]["cuda_ms"]["median"]
    )
    result["comparison"] = {
        "fragmented_over_large_cuda_ratio": fragment_ms / large_ms,
        "fragmented_slowdown_percent": (fragment_ms / large_ms - 1.0) * 100.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
