#!/usr/bin/env python3
"""Run all-pairwise ABBA slot-PTE/CE samples in fresh processes."""

from __future__ import annotations

import argparse
import fcntl
import itertools
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
MIB = 1024 * 1024
MODES = (
    "slot_4k_baseline",
    "slot_64k",
    "slot_2m",
    "direct_overwrite_existing_grain",
    "stage_sync_4k_ce",
)
MODE_LETTERS = dict(zip("ABCDE", MODES, strict=True))
PAIR_LETTERS = tuple(itertools.combinations(MODE_LETTERS, 2))
DEFAULT_STATS = Path("/proc/driver/nvidia-uvm/latency_stats")
DEFAULT_BUDGET = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes"
)
DEFAULT_SRCVERSION = Path("/sys/module/nvidia_uvm/srcversion")
TIMING_FIELDS = (
    "wall_ms",
    "driver_profile_total_ms",
    "prepare_ms",
    "unmap_ms",
    "copy_ms",
    "map_ms",
    "wait_ms",
    "prepare_complete_ms",
    "copy_complete_ms",
    "map_complete_ms",
    "gib_s",
)
COUNTER_FIELDS = (
    "slot_direct_blocks",
    "slot_direct_copy_pushes",
    "slot_direct_memcopy_methods",
    "slot_direct_tracker_waits",
    "slot_direct_pte_4k_blocks",
    "slot_direct_pte_64k_blocks",
    "slot_direct_pte_2m_blocks",
    "slot_direct_pte_write_entries",
    "slot_direct_skipped_unmap_blocks",
    "slot_direct_stage_sync_calls",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--library",
        type=Path,
        default=SCRIPT_DIR / "libuvm_slot_pte_overwrite_ce_shim.so",
    )
    parser.add_argument("--worker", type=Path, default=SCRIPT_DIR / "worker.py")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--budget-path", type=Path, default=DEFAULT_BUDGET)
    parser.add_argument("--srcversion-path", type=Path, default=DEFAULT_SRCVERSION)
    parser.add_argument("--artifact-root", type=Path, default=SCRIPT_DIR / "artifacts")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--slot-layer-id", type=lambda value: int(value, 0), default=0x5124
    )
    parser.add_argument(
        "--slot-expert-id", type=lambda value: int(value, 0), default=0xC01D
    )
    args = parser.parse_args()

    if args.profile == "smoke":
        args.allocations = 4
        args.batch_size = 1
        default_warmups = 0
    else:
        args.allocations = 512
        args.batch_size = 128
        default_warmups = 1
    args.allocation_bytes = 4 * MIB
    args.guard_bytes = 4 * MIB
    if args.warmups is None:
        args.warmups = default_warmups
    if (
        args.device < 0
        or args.blocks <= 0
        or args.warmups < 0
        or args.timeout_seconds <= 0
    ):
        parser.error("device/count/timeout arguments are invalid")
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


def pairwise_abba_schedule(block_index: int) -> list[tuple[str, int, str]]:
    """Return all ten mode pairs, each as literal ABBA or BAAB."""
    pairs = list(PAIR_LETTERS)
    if block_index % 2:
        pairs.reverse()
    schedule: list[tuple[str, int, str]] = []
    for pair_index, (left, right) in enumerate(pairs):
        if (block_index + pair_index) % 2:
            order = (right, left, left, right)
        else:
            order = (left, right, right, left)
        pair_name = left + right
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
        result = subprocess.run(
            command, check=False, text=True, capture_output=True, timeout=20
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for mode in MODES:
        selected = [sample for sample in samples if sample["mode"] == mode]
        row: dict[str, Any] = {"samples": len(selected)}
        for field in TIMING_FIELDS:
            values = [
                float(sample["measurement"][field])
                for sample in selected
                if sample["measurement"][field] is not None
            ]
            row[f"{field}_p50"] = median_or_none(values)
            if field == "wall_ms":
                row["wall_ms_p95"] = percentile(values, 0.95)
        for field in COUNTER_FIELDS:
            values = [
                float(sample["profile"]["delta"][field]) for sample in selected
            ]
            row[f"{field}_p50"] = median_or_none(values)
        results[mode] = row

    baseline_wall = float(results[MODES[0]]["wall_ms_p50"])
    baseline_entries = float(
        results[MODES[0]]["slot_direct_pte_write_entries_p50"]
    )
    comparisons: dict[str, float] = {}
    for mode in MODES[1:]:
        comparisons[f"{mode}_vs_slot_4k_baseline_speedup"] = (
            baseline_wall / float(results[mode]["wall_ms_p50"])
        )
        comparisons[f"{mode}_pte_entry_reduction_vs_slot_4k"] = (
            baseline_entries
            / float(results[mode]["slot_direct_pte_write_entries_p50"])
        )
    return {"modes": results, "comparisons": comparisons}


def run(args: argparse.Namespace) -> Path:
    if not args.library.is_file():
        raise RuntimeError(f"benchmark library does not exist: {args.library}")
    if not args.worker.is_file():
        raise RuntimeError(f"worker does not exist: {args.worker}")
    if not os.access(args.stats, os.R_OK):
        raise RuntimeError(f"driver profile is not readable: {args.stats}")

    logical_bytes = args.allocations * args.allocation_bytes
    expected_va_blocks = logical_bytes // (2 * MIB)
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
        "expected_va_blocks": expected_va_blocks,
        "blocks": args.blocks,
        "pairs_per_block": len(PAIR_LETTERS),
        "samples_per_mode": args.blocks * 8,
        "samples_per_block": len(PAIR_LETTERS) * 4,
        "worker_warmups": args.warmups,
        "schedule": "all_pairwise_abba_independent_processes",
        "modes": list(MODES),
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
        for pair_name, pair_slot, mode in pairwise_abba_schedule(block_index):
            sample_id = f"b{block_index:02d}_{pair_name}_s{pair_slot + 1}_{mode}"
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
                (sample_dir / "stdout.log").write_text(
                    error.stdout or "", encoding="utf-8"
                )
                (sample_dir / "stderr.log").write_text(
                    error.stderr or "", encoding="utf-8"
                )
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
                    f"sample {sample_id} failed rc={completed.returncode}; see {sample_dir}"
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
            complete_value = sample["measurement"]["copy_complete_ms"]
            complete_text = (
                "na" if complete_value is None else f"{float(complete_value):.3f}"
            )
            print(
                f"SAMPLE index={sample_index} block={block_index} pair={pair_name} "
                f"pair_slot={pair_slot} mode={mode} pid={sample['pid']} "
                f"unmap_ms={sample['measurement']['unmap_ms']:.3f} "
                f"copy_submit_ms={sample['measurement']['copy_ms']:.3f} "
                f"copy_complete_ms={complete_text} "
                f"map_ms={sample['measurement']['map_ms']:.3f} "
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
    lock_path = Path("/tmp/uvm_slot_pte_overwrite_ce_compare.lock")
    try:
        with lock_path.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("another slot-PTE benchmark is already running") from error
            run(args)
        return 0
    except Exception as error:  # noqa: BLE001 - command-line boundary.
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
