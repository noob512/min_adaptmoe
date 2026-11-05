#!/usr/bin/env python3
"""Run a Prefill benchmark under a cudaProfilerApi-bounded Nsight capture."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    args = sys.argv[1:]
    try:
        output = Path(args[args.index("--output") + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise SystemExit("missing benchmark --output") from exc
    real_runner_text = os.environ.get("E3_NSYS_REAL_RUNNER")
    if not real_runner_text:
        raise SystemExit("E3_NSYS_REAL_RUNNER is required")
    real_runner = Path(real_runner_text).resolve()
    nsys = Path(os.environ.get("E3_NSYS", "/usr/local/cuda-12.8/bin/nsys"))
    timeline = output.parent / "prefill-timeline"
    command = [
        str(nsys),
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--cuda-graph-trace="
        + os.environ.get("E3_NSYS_CUDA_GRAPH_TRACE", "node"),
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        f"--output={timeline}",
        sys.executable,
        str(real_runner),
        *args,
    ]
    os.execv(str(nsys), command)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
