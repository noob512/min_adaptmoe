#!/usr/bin/env python3
"""Read-only compact progress snapshot for a pressure-wave run."""
import collections
import json
import re
import statistics
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
for arm in ("adaptmoe", "native_copy", "native_uva"):
    directory = root / arm
    if not directory.exists():
        continue
    path = directory / "replay.progress.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text().split("\n"):
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # Last append may still be in progress.
    log = (directory / "server.log").read_text() if (directory / "server.log").exists() else ""
    lines = log.split("\n")
    stats = [line for line in lines if "Running:" in line]
    commits = [line for line in lines if "autoscale committed" in line]
    errors = [line for line in lines if re.search(r"\bERROR\b|Traceback \(most recent|illegal memory access|CUDA error|Xid |MMU fault|OutOfMemoryError", line)]
    result = dict(arm=arm, complete=(directory / "replay.json").exists(), completed=len(rows),
                  failed=sum(not row["success"] for row in rows),
                  phases=dict(collections.Counter(row["pressure_phase"] for row in rows)),
                  latest_stats=stats[-1:] if stats else lines[-2:],
                  commits=commits, errors=errors[-5:])
    if rows:
        origin = rows[0]["scheduled_at_perf"] - rows[0]["trace_arrival_ms"] / 1000
        result["trace_elapsed_s"] = round(time.perf_counter() - origin, 1)
        recent = [row for row in rows[-16:] if row.get("success")]
        result["last16_ttft_mean_ms"] = round(statistics.fmean(row["ttft_ms"] for row in recent), 2) if recent else None
    print(json.dumps(result))
