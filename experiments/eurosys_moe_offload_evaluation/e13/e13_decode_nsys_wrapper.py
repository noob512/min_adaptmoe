#!/usr/bin/env python3
"""Launch the E13 Decode diagnostic under a bounded Nsight Systems capture."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    args = sys.argv[1:]
    try:
        output = Path(args[args.index("--output") + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise SystemExit("missing application --output") from exc
    real_runner = Path(
        os.environ.get(
            "E13_DECODE_REAL_RUNNER",
            str(Path(__file__).with_name("e13_decode_abba_diagnostic.py")),
        )
    ).resolve()
    nsys = Path(os.environ.get("E13_NSYS", "/usr/local/cuda-12.8/bin/nsys"))
    timeline = output.parent / "timeline"
    command = [
        str(nsys),
        "profile",
        "--trace=cuda,nvtx,osrt",
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
