#!/usr/bin/env python3
"""Run a black-box command while sampling PCIe and GPU state with NVML."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_MONITOR = (
    REPO_ROOT
    / "experiments"
    / "eurosys_moe_offload_evaluation"
    / "monitor_e4_nvml.py"
)
DEFAULT_PYTHON = REPO_ROOT / "workloads" / "vllm" / "vllm" / ".venv" / "bin" / "python"


def write_new(path: Path, value: object) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--interval-ms", type=int, default=100)
    parser.add_argument("--monitor", type=Path, default=DEFAULT_MONITOR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise SystemExit("missing black-box command")
    if args.output_dir.exists():
        raise SystemExit(f"output directory exists: {args.output_dir}")
    if not args.monitor.is_file() or not args.python.is_file():
        raise SystemExit("monitor or Python interpreter is missing")
    args.output_dir.mkdir(parents=True)
    write_new(
        args.output_dir / "configuration.json",
        {
            "command": command,
            "interval_ms": args.interval_ms,
            "monitor": str(args.monitor.resolve()),
            "python": str(args.python.resolve()),
        },
    )
    monitor_stdout = (args.output_dir / "monitor.stdout").open("wb")
    monitor_stderr = (args.output_dir / "monitor.stderr").open("wb")
    monitor = subprocess.Popen(
        [
            str(args.python),
            str(args.monitor),
            "--output",
            str(args.output_dir / "nvml.csv"),
            "--interval-ms",
            str(args.interval_ms),
        ],
        stdout=monitor_stdout,
        stderr=monitor_stderr,
    )
    started = time.monotonic()
    try:
        completed = subprocess.run(command, env=os.environ.copy(), check=False)
        returncode = completed.returncode
    finally:
        monitor.send_signal(signal.SIGTERM)
        try:
            monitor.wait(timeout=10)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait(timeout=10)
        monitor_stdout.close()
        monitor_stderr.close()
    write_new(
        args.output_dir / "status.json",
        {
            "command_returncode": returncode,
            "duration_seconds": time.monotonic() - started,
            "monitor_returncode": monitor.returncode,
        },
    )
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
