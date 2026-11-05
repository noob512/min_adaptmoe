#!/usr/bin/env python3
"""Run one black-box case with immutable-worktree checks and captured output."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
GUARD = HERE / "guard.py"
BASELINE = HERE / "baseline.json"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_new(path: Path, value: object) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gpu_snapshot() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,driver_version,pstate,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    clients = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "query_returncode": result.returncode,
        "query": result.stdout.strip(),
        "query_stderr": result.stderr.strip(),
        "clients_returncode": clients.returncode,
        "clients": clients.stdout.strip(),
    }


def guard(label: str, output: Path) -> bool:
    result = subprocess.run(
        [sys.executable, str(GUARD), "check", "--baseline", str(BASELINE), "--label", label, "--output", str(output)],
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise SystemExit("missing command")
    if args.output_dir.exists():
        raise SystemExit(f"output directory exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    before_ok = guard(f"{args.case}:before", args.output_dir / "git_guard.before.json")
    if not before_ok:
        write_new(args.output_dir / "status.json", {"case": args.case, "status": "blocked", "reason": "pre-run git guard failed"})
        return 2

    environment = os.environ.copy()
    overrides: dict[str, str] = {}
    for assignment in args.env:
        if "=" not in assignment:
            raise SystemExit(f"invalid environment assignment: {assignment}")
        key, value = assignment.split("=", 1)
        overrides[key] = value
        environment[key] = value
    write_new(args.output_dir / "command.json", {"argv": command, "environment_overrides": overrides, "timeout_seconds": args.timeout})
    write_new(args.output_dir / "gpu.before.json", gpu_snapshot())
    started = utc()
    start_perf = time.monotonic()
    timed_out = False
    with (args.output_dir / "stdout.log").open("wb") as stdout, (args.output_dir / "stderr.log").open("wb") as stderr:
        try:
            completed = subprocess.run(command, env=environment, stdout=stdout, stderr=stderr, timeout=args.timeout, check=False)
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124
    duration = time.monotonic() - start_perf
    write_new(args.output_dir / "gpu.after.json", gpu_snapshot())
    after_ok = guard(f"{args.case}:after", args.output_dir / "git_guard.after.json")
    status = "completed" if returncode == 0 and after_ok else "failed"
    write_new(
        args.output_dir / "status.json",
        {
            "case": args.case,
            "status": status,
            "started_at": started,
            "finished_at": utc(),
            "duration_seconds": duration,
            "returncode": returncode,
            "timed_out": timed_out,
            "git_guard_before": before_ok,
            "git_guard_after": after_ok,
        },
    )
    return 0 if status == "completed" else returncode or 2


if __name__ == "__main__":
    raise SystemExit(main())
