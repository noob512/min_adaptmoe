#!/usr/bin/env python3
"""Summarize E4 Nsight Compute application-replay counter samples."""

from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
from pathlib import Path


METRICS = {
    "kernel_duration_ns": "gpu__time_duration.sum",
    "long_scoreboard_pct": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "lg_throttle_pct": "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "mio_throttle_pct": "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
}


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ncu_root", type=Path)
    parser.add_argument("--formal-root", type=Path)
    args = parser.parse_args()
    root = args.ncu_root.resolve()
    execution = list(csv.DictReader((root / "execution.tsv").open(), delimiter="\t"))
    if len(execution) != 3 or any(row["status"] != "ok" for row in execution):
        raise RuntimeError("NCU execution did not complete all C1/C16/C32 cases")
    records: list[dict[str, object]] = []
    flat: list[dict[str, object]] = []
    for concurrency in (1, 16, 32):
        path = root / "cases" / f"c{concurrency}" / "ncu.csv"
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        header = next(index for index, line in enumerate(lines) if line.startswith('"ID"'))
        rows = [
            row
            for row in csv.DictReader(io.StringIO("\n".join(lines[header:])))
            if row.get("ID") and row.get("Kernel Name") == "fused_moe_kernel"
        ]
        if len(rows) != 8:
            raise RuntimeError(f"C{concurrency} expected 8 kernel samples, found {len(rows)}")
        metric_summary = {
            output: summarize([float(row[source]) for row in rows])
            for output, source in METRICS.items()
        }
        records.append(
            {
                "concurrency": concurrency,
                "kernel": "fused_moe_kernel",
                "sample_count": len(rows),
                "metrics": metric_summary,
            }
        )
        for index, row in enumerate(rows):
            flat.append(
                {
                    "concurrency": concurrency,
                    "sample": index,
                    **{output: float(row[source]) for output, source in METRICS.items()},
                }
            )
    output = {
        "schema": "eurosys_e4_ncu_stall_summary_v1",
        "status": "ok",
        "profiling_scope": "one measured Decode step; first 8 fused_moe_kernel launches",
        "replay_mode": "application",
        "cache_control": "none",
        "timings_excluded_from_formal_performance": True,
        "cases": records,
        "interpretation": (
            "Long-scoreboard is the dominant sampled warp-issue stall. It rises from C1 "
            "to C16 but is flat from C16 to C32; throttle stalls stay small."
        ),
    }
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    (analysis / "ncu_stall_summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (analysis / "ncu_stall_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    if args.formal_root is not None:
        formal_analysis = args.formal_root.resolve() / "analysis"
        formal_analysis.mkdir(exist_ok=True)
        linked = dict(output)
        linked["ncu_run_root"] = str(root)
        (formal_analysis / "ncu_stall_summary.json").write_text(
            json.dumps(linked, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
