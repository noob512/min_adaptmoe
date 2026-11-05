#!/usr/bin/env python3
"""Repeat frozen request contents under a low/high/low arrival envelope."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def build(source: Path, output: Path) -> dict:
    rows = [json.loads(line) for line in source.read_text().split("\n") if line]
    arrivals = [float(row["arrival_ms"]) for row in rows]
    intervals = [b - a for a, b in zip(arrivals, arrivals[1:]) if b > a]
    span = arrivals[-1] - arrivals[0] + statistics.median(intervals)
    phases, result = [], []
    start = 0.0
    for name, rate in (("low_initial", 0.25), ("high", 2.0), ("low_recovery", 0.25)):
        duration = len(rows) / rate
        phases.append(dict(name=name, offered_qps=rate, start_s=start,
                           end_s=start + duration, request_count=len(rows)))
        for row, arrival in zip(rows, arrivals):
            clone = dict(row)
            clone.update(request_id=f"{name}/{row['request_id']}",
                         source_request_id=row["request_id"], pressure_phase=name,
                         arrival_ms=(start + (arrival - arrivals[0]) / span * duration) * 1000)
            result.append(clone)
        start += duration
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for row in result:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = dict(source=str(source.resolve()), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    trace=str(output.resolve()), trace_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    request_count=len(result), unique_source_requests=len(rows), phases=phases,
                    arrival_horizon_s=start, repeats_per_source_request=3,
                    description="Same full request sequence per phase; affine rescaling of the frozen burst offsets; no gap or drain between phases.")
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))
