#!/usr/bin/env python3
"""Orchestrate pairwise ABBA samples, one fresh worker process per sample."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
MODES = (
    "driver_4x128_sync",
    "slot_per_block_push",
    "slot_shared_push",
)
MODE_LETTERS = dict(zip("ABC", MODES, strict=True))
DEFAULT_STATS = Path("/proc/driver/nvidia-uvm/latency_stats")
DEFAULT_BUDGET = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes"
)
DEFAULT_SRCVERSION = Path("/sys/module/nvidia_uvm/srcversion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--library", type=Path, default=SCRIPT_DIR / "libuvm_slot_batch_shim.so")
    parser.add_argument("--worker", type=Path, default=SCRIPT_DIR / "worker.py")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--budget-path", type=Path, default=DEFAULT_BUDGET)
    parser.add_argument("--srcversion-path", type=Path, default=DEFAULT_SRCVERSION)
    parser.add_argument("--artifact-root", type=Path, default=SCRIPT_DIR / "artifacts")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--allocations", type=int)
    parser.add_argument("--allocation-mib", type=float)
    parser.add_argument("--guard-mib", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--slot-layer-id", type=lambda value: int(value, 0), default=0x5124)
    parser.add_argument("--slot-expert-id", type=lambda value: int(value, 0), default=0xC01D)
    args = parser.parse_args()

    defaults = {
        "smoke": {
            "allocations": 16,
            "allocation_mib": 1.0,
            "guard_mib": 1.0,
            "batch_size": 4,
            "blocks": 1,
            "warmups": 0,
        },
        "full": {
            "allocations": 512,
            "allocation_mib": 4.0,
            "guard_mib": 4.0,
            "batch_size": 128,
            "blocks": 2,
            "warmups": 1,
        },
    }[args.profile]
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    args.allocation_bytes = int(args.allocation_mib * 1024 * 1024)
    args.guard_bytes = int(args.guard_mib * 1024 * 1024)
    if (
        args.device < 0
        or args.allocations <= 0
        or args.batch_size <= 0
        or args.batch_size > 128
        or args.allocations != 4 * args.batch_size
        or args.allocation_bytes <= 0
        or args.guard_bytes < 0
        or args.blocks <= 0
        or args.warmups < 0
        or args.timeout_seconds <= 0
    ):
        parser.error("require allocations=4*batch_size and positive geometry/counts")
    return args


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def percentile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def abba_schedule(block_index: int) -> list[tuple[str, int, str]]:
    # Every pair is a literal ABBA (or its next-block BAAB reversal). This
    # gives each mode four samples per block while avoiding a privileged mode.
    if block_index % 2 == 0:
        pair_orders = (("AB", "ABBA"), ("BC", "BCCB"), ("AC", "ACCA"))
    else:
        pair_orders = (("AC", "CAAC"), ("BC", "CBBC"), ("AB", "BAAB"))
    schedule: list[tuple[str, int, str]] = []
    for pair_name, order in pair_orders:
        for pair_slot, letter in enumerate(order):
            schedule.append((pair_name, pair_slot, MODE_LETTERS[letter]))
    return schedule


def parse_sample(stdout: str) -> dict[str, Any]:
    records = [
        line.removeprefix("SAMPLE_JSON ")
        for line in stdout.splitlines()
        if line.startswith("SAMPLE_JSON ")
    ]
    if len(records) != 1:
        raise RuntimeError(f"worker emitted {len(records)} SAMPLE_JSON records")
    result = json.loads(records[0])
    if result.get("status") != "ok":
        raise RuntimeError(f"worker failed: {result.get('error', 'unknown error')}")
    return result


def collect_gpu_metadata(device: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={device}",
        "--query-gpu=name,uuid,driver_version,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=False, text=True, capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for mode in MODES:
        selected = [sample for sample in samples if sample["mode"] == mode]
        wall = [float(sample["measurement"]["wall_ms"]) for sample in selected]
        copy_ms = [
            float(sample["measurement"]["copy_ms"])
            for sample in selected
            if sample["measurement"]["copy_ms"] is not None
        ]
        map_ms = [
            float(sample["measurement"]["slot_map_ms"])
            for sample in selected
            if sample["measurement"]["slot_map_ms"] is not None
        ]
        bandwidth = [float(sample["measurement"]["gib_s"]) for sample in selected]
        pushes = [
            int(sample["profile"]["delta"]["slot_direct_copy_pushes"])
            for sample in selected
            if mode != "driver_4x128_sync"
        ]
        methods = [
            int(sample["profile"]["delta"]["slot_direct_memcopy_methods"])
            for sample in selected
            if mode != "driver_4x128_sync"
        ]
        results[mode] = {
            "samples": len(selected),
            "wall_ms_p50": statistics.median(wall),
            "wall_ms_p95": percentile(wall, 0.95),
            "copy_ms_p50": statistics.median(copy_ms) if copy_ms else None,
            "slot_map_ms_p50": statistics.median(map_ms) if map_ms else None,
            "gib_s_p50": statistics.median(bandwidth),
            "copy_pushes_p50": statistics.median(pushes) if pushes else None,
            "memcopy_methods_p50": statistics.median(methods) if methods else None,
        }
    a = results[MODES[0]]["wall_ms_p50"]
    b = results[MODES[1]]["wall_ms_p50"]
    c = results[MODES[2]]["wall_ms_p50"]
    return {
        "modes": results,
        "comparisons": {
            "slot_shared_vs_slot_per_block_speedup": b / c,
            "slot_shared_vs_driver_batch_speedup": a / c,
            "slot_per_block_vs_driver_batch_speedup": a / b,
            "slot_shared_push_count_reduction": (
                results[MODES[1]]["copy_pushes_p50"]
                / results[MODES[2]]["copy_pushes_p50"]
            ),
        },
    }


def run(args: argparse.Namespace) -> Path:
    if not args.library.is_file():
        raise RuntimeError(f"benchmark library does not exist: {args.library}")
    if not args.worker.is_file():
        raise RuntimeError(f"worker does not exist: {args.worker}")
    if not os.access(args.stats, os.R_OK):
        raise RuntimeError(f"driver profile is not readable: {args.stats}")

    logical_bytes = args.allocations * args.allocation_bytes
    budget_text = read_text(args.budget_path)
    if budget_text is None:
        raise RuntimeError(f"slot VIDMEM budget is unavailable: {args.budget_path}")
    budget_bytes = int(budget_text, 0)
    if budget_bytes != 0 and budget_bytes < logical_bytes:
        raise RuntimeError(
            f"slot VIDMEM budget {budget_bytes} is below logical bytes {logical_bytes}; "
            "load the test module with a sufficient budget (or zero/unlimited)"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{args.profile}_{uuid.uuid4().hex[:8]}"
    artifact_dir = args.artifact_root.resolve() / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "schema_version": 1,
        "run_id": run_id,
        "profile": args.profile,
        "device": args.device,
        "allocations": args.allocations,
        "allocation_bytes": args.allocation_bytes,
        "guard_bytes": args.guard_bytes,
        "batch_size": args.batch_size,
        "batches": 4,
        "logical_bytes": logical_bytes,
        "blocks": args.blocks,
        "samples_per_mode": args.blocks * 4,
        "worker_warmups": args.warmups,
        "schedule": "pairwise_abba_independent_processes",
        "slot_layer_id": args.slot_layer_id,
        "slot_expert_id": args.slot_expert_id,
        "slot_vidmem_budget_bytes": budget_bytes,
        "stats_path": str(args.stats),
        "library": str(args.library.resolve()),
        "worker": str(args.worker.resolve()),
    }
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "uvm_srcversion": read_text(args.srcversion_path),
        "gpu": collect_gpu_metadata(args.device),
        "python": sys.version,
    }
    write_json(artifact_dir / "config.json", config)
    write_json(artifact_dir / "metadata_start.json", metadata)

    samples: list[dict[str, Any]] = []
    sample_index = 0
    for block_index in range(args.blocks):
        for pair_name, pair_slot, mode in abba_schedule(block_index):
            sample_id = (
                f"b{block_index:02d}_{pair_name}_s{pair_slot + 1}_{mode}"
            )
            sample_dir = artifact_dir / "samples" / f"{sample_index:03d}_{sample_id}"
            sample_dir.mkdir(parents=True, exist_ok=False)
            command = [
                sys.executable,
                str(args.worker.resolve()),
                "--mode",
                mode,
                "--library",
                str(args.library.resolve()),
                "--stats",
                str(args.stats),
                "--device",
                str(args.device),
                "--allocations",
                str(args.allocations),
                "--allocation-bytes",
                str(args.allocation_bytes),
                "--guard-bytes",
                str(args.guard_bytes),
                "--batch-size",
                str(args.batch_size),
                "--warmups",
                str(args.warmups),
                "--slot-layer-id",
                str(args.slot_layer_id),
                "--slot-expert-id",
                str(args.slot_expert_id),
                "--sample-id",
                sample_id,
            ]
            write_json(sample_dir / "command.json", command)
            started_ns = time.time_ns()
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=args.timeout_seconds,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
            except subprocess.TimeoutExpired as error:
                (sample_dir / "stdout.log").write_text(error.stdout or "", encoding="utf-8")
                (sample_dir / "stderr.log").write_text(error.stderr or "", encoding="utf-8")
                raise RuntimeError(f"sample {sample_id} timed out") from error
            (sample_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
            (sample_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")
            write_json(
                sample_dir / "process.json",
                {
                    "returncode": completed.returncode,
                    "started_ns": started_ns,
                    "finished_ns": time.time_ns(),
                },
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"sample {sample_id} failed rc={completed.returncode}; "
                    f"see {sample_dir}"
                )
            sample = parse_sample(completed.stdout)
            if sample["sample_id"] != sample_id or sample["mode"] != mode:
                raise RuntimeError(f"sample identity mismatch in {sample_id}")
            sample["schedule"] = {
                "block_index": block_index,
                "pair": pair_name,
                "pair_slot": pair_slot,
                "global_index": sample_index,
            }
            write_json(sample_dir / "sample.json", sample)
            samples.append(sample)
            copy_value = sample["measurement"]["copy_ms"]
            map_value = sample["measurement"]["slot_map_ms"]
            copy_text = "na" if copy_value is None else f"{float(copy_value):.3f}"
            map_text = "na" if map_value is None else f"{float(map_value):.3f}"
            print(
                f"SAMPLE index={sample_index} block={block_index} pair={pair_name} "
                f"pair_slot={pair_slot} mode={mode} pid={sample['pid']} "
                f"copy_ms={copy_text} "
                f"slot_map_ms={map_text} "
                f"e2e_ms={sample['measurement']['wall_ms']:.3f} "
                f"gib_s={sample['measurement']['gib_s']:.3f}",
                flush=True,
            )
            sample_index += 1

    summary = {
        "schema_version": 1,
        "status": "ok",
        "config": config,
        "metadata": {
            **metadata,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        },
        "sample_count": len(samples),
        "unique_process_count": len({int(sample["pid"]) for sample in samples}),
        "aggregate": aggregate(samples),
        "samples": samples,
    }
    write_json(artifact_dir / "summary.json", summary)
    with (artifact_dir / "samples.jsonl").open("w", encoding="utf-8") as output:
        for sample in samples:
            output.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")
    print("SUMMARY " + json.dumps(summary["aggregate"], sort_keys=True))
    print(f"ARTIFACT_DIR={artifact_dir}")
    return artifact_dir


def main() -> int:
    args = parse_args()
    lock_path = Path("/tmp/uvm_512x4m_slot_shared_push_compare.lock")
    try:
        with lock_path.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("another slot benchmark is already running") from error
            run(args)
        return 0
    except Exception as error:  # noqa: BLE001 - command-line boundary.
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
