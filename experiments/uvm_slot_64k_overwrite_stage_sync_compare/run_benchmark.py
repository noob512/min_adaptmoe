#!/usr/bin/env python3
"""Run matched normal/overwrite existing-64K samples in fresh processes."""

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
from typing import Any, Iterable, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
MIB = 1024 * 1024
SCHEMA_VERSION = 2
PROFILE_CONTRACT = "slot_direct_detailed_closure_v2"
MODES = (
    "normal_64k_e2e",
    "overwrite_64k_e2e",
    "normal_64k_stage_sync",
    "overwrite_64k_stage_sync",
)
PAIRS = (
    ("normal", MODES[0], MODES[1]),
    ("stage_sync", MODES[2], MODES[3]),
)
REQUIRED_SCHEDULE_BLOCKS = 4
DEFAULT_STATS = Path("/proc/driver/nvidia-uvm/latency_stats")
DEFAULT_BUDGET = Path(
    "/sys/module/nvidia_uvm/parameters/uvm_moe_slot_pool_vidmem_budget_bytes"
)
DEFAULT_SRCVERSION = Path("/sys/module/nvidia_uvm/srcversion")
LEGACY_TIMING_FIELDS = (
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
    "copy_complete_gib_s",
    "copy_complete_minus_submit_ms",
    "gib_s",
)
PARENT_PROFILE_STAGES = (
    "pre_helper_total",
    "helper_total",
    "post_helper_total",
)
PRE_HELPER_PROFILE_STAGES = (
    "gate",
    "state_alloc",
    "gpu_retain",
    "outer_lock_acquire",
    "batch_bookkeeping",
    "slot_validate",
    "flatten_sort",
    "slot_tracker_init",
)
HELPER_RESIDUAL_PROFILE_STAGES = (
    "helper_context",
    "block_lock_acquire",
    "locked_revalidate",
    "copy_to_map",
    "block_unlock",
    "helper_context_free",
)
POST_HELPER_PROFILE_STAGES = (
    "result_check",
    "pte_metadata",
    "status_resolve",
    "slot_state_commit",
    "telemetry_cleanup",
    "outer_lock_release",
    "resource_teardown",
)
DETAILED_PROFILE_STAGES = (
    PARENT_PROFILE_STAGES
    + PRE_HELPER_PROFILE_STAGES
    + HELPER_RESIDUAL_PROFILE_STAGES
    + POST_HELPER_PROFILE_STAGES
)
CLOSURE_TIMING_FIELDS = (
    "driver_accounted_ms",
    "driver_residual_ms",
    "driver_closure_fraction",
    "host_outer_ms",
    "parent_partition_residual_ms",
    "pre_helper_residual_ms",
    "helper_residual_ms",
    "post_helper_residual_ms",
)
SCHEMA_V2_TIMING_FIELDS = tuple(
    f"{stage}_ms" for stage in DETAILED_PROFILE_STAGES
) + CLOSURE_TIMING_FIELDS
TIMING_FIELDS = LEGACY_TIMING_FIELDS + SCHEMA_V2_TIMING_FIELDS
COUNTER_FIELDS = (
    "slot_direct_ioctl_calls",
    "slot_direct_per_block_calls",
    "slot_direct_shared_calls",
    "slot_direct_successes",
    "slot_direct_requested_ranges",
    "slot_direct_completed_ranges",
    "slot_direct_requested_bytes",
    "slot_direct_completed_bytes",
    "slot_direct_blocks",
    "slot_direct_copy_pushes",
    "slot_direct_memcopy_methods",
    "slot_direct_tracker_waits",
    "slot_direct_vidmem_allocations",
    "slot_direct_vidmem_evictions",
    "slot_direct_pte_map_ranges",
    "slot_direct_pte_map_bytes",
    "slot_direct_pte_4k_blocks",
    "slot_direct_pte_64k_blocks",
    "slot_direct_pte_2m_blocks",
    "slot_direct_pte_write_entries",
    "slot_direct_skipped_unmap_blocks",
    "slot_direct_stage_sync_calls",
    "slot_direct_fallbacks",
    "slot_direct_errors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--library",
        type=Path,
        default=SCRIPT_DIR / "libuvm_slot_64k_overwrite_stage_sync_shim.so",
    )
    parser.add_argument("--worker", type=Path, default=SCRIPT_DIR / "worker.py")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--budget-path", type=Path, default=DEFAULT_BUDGET)
    parser.add_argument("--srcversion-path", type=Path, default=DEFAULT_SRCVERSION)
    parser.add_argument("--artifact-root", type=Path, default=SCRIPT_DIR / "artifacts")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--schedule-blocks",
        "--blocks",
        dest="schedule_blocks",
        type=int,
        default=REQUIRED_SCHEDULE_BLOCKS,
        help="matched ABBA schedule blocks (fixed at 4)",
    )
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
        or args.schedule_blocks != REQUIRED_SCHEDULE_BLOCKS
        or args.warmups < 0
        or args.timeout_seconds <= 0
    ):
        parser.error("device/count/timeout arguments are invalid; schedule blocks must be 4")
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


def matched_abba_schedule(block_index: int) -> list[tuple[str, int, str]]:
    """Return the two matched pairs, each as literal ABBA or BAAB."""
    pairs = list(PAIRS)
    if block_index % 2:
        pairs.reverse()
    schedule: list[tuple[str, int, str]] = []
    for pair_name, left, right in pairs:
        if block_index % 2:
            order = (right, left, left, right)
        else:
            order = (left, right, right, left)
        for pair_slot, mode in enumerate(order):
            schedule.append((pair_name, pair_slot, mode))
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


def sample_schema_version(sample: Mapping[str, Any]) -> int:
    try:
        version = int(sample.get("schema_version", 1))
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid sample schema_version: {sample.get('schema_version')!r}") from error
    if version not in (1, SCHEMA_VERSION):
        raise ValueError(f"unsupported sample schema_version: {version}")
    return version


def sample_timing_value(sample: Mapping[str, Any], field: str) -> float | None:
    """Read one timing while allowing schema-1 samples to lack v2 fields."""
    version = sample_schema_version(sample)
    measurement = sample.get("measurement")
    if not isinstance(measurement, Mapping):
        raise ValueError("sample measurement is not an object")

    if field not in measurement:
        if version == 1 and field in SCHEMA_V2_TIMING_FIELDS:
            return None
        raise ValueError(f"schema-{version} sample is missing measurement.{field}")

    value = measurement[field]
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"measurement.{field} is not finite: {value!r}")
    return numeric


def format_optional(value: Any, digits: int = 3) -> str:
    if value is None:
        return "na"
    return f"{float(value):.{digits}f}"


def aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    versions = [sample_schema_version(sample) for sample in samples]
    legacy_output = bool(versions) and all(version == 1 for version in versions)
    timing_fields = LEGACY_TIMING_FIELDS if legacy_output else TIMING_FIELDS
    results: dict[str, Any] = {}
    for mode in MODES:
        selected = [sample for sample in samples if sample["mode"] == mode]
        row: dict[str, Any] = {"samples": len(selected)}
        if not legacy_output:
            schema_counts: dict[str, int] = {}
            for sample in selected:
                key = str(sample_schema_version(sample))
                schema_counts[key] = schema_counts.get(key, 0) + 1
            row["sample_schema_versions"] = schema_counts
        for field in timing_fields:
            values = [
                value
                for sample in selected
                if (value := sample_timing_value(sample, field)) is not None
            ]
            if not legacy_output:
                row[f"{field}_samples"] = len(values)
            row[f"{field}_p50"] = median_or_none(values)
            row[f"{field}_p95"] = percentile(values, 0.95) if values else None
        for field in COUNTER_FIELDS:
            values = [
                float(sample["profile"]["delta"][field]) for sample in selected
            ]
            row[f"{field}_p50"] = median_or_none(values)
        results[mode] = row

    def ratio(numerator_mode: str, denominator_mode: str, field: str) -> float:
        numerator = results[numerator_mode][f"{field}_p50"]
        denominator = results[denominator_mode][f"{field}_p50"]
        if numerator is None or denominator is None or float(denominator) <= 0:
            raise ValueError(
                f"cannot compare {field}: {numerator_mode}={numerator}, "
                f"{denominator_mode}={denominator}"
            )
        return float(numerator) / float(denominator)

    comparisons = {
        "normal_overwrite_wall_speedup": ratio(MODES[0], MODES[1], "wall_ms"),
        "stage_overwrite_wall_speedup": ratio(MODES[2], MODES[3], "wall_ms"),
        "stage_overwrite_copy_complete_speedup": ratio(
            MODES[2], MODES[3], "copy_complete_ms"
        ),
        "normal_stage_sync_wall_penalty": ratio(MODES[2], MODES[0], "wall_ms"),
        "overwrite_stage_sync_wall_penalty": ratio(
            MODES[3], MODES[1], "wall_ms"
        ),
    }
    aggregate_result = {"modes": results, "comparisons": comparisons}
    if not legacy_output:
        schema_counts: dict[str, int] = {}
        for version in versions:
            key = str(version)
            schema_counts[key] = schema_counts.get(key, 0) + 1
        aggregate_result["sample_schema_versions"] = schema_counts
    return aggregate_result


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
        "schema_version": SCHEMA_VERSION,
        "sample_schema_version": SCHEMA_VERSION,
        "profile_contract": PROFILE_CONTRACT,
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
        "schedule_blocks": args.schedule_blocks,
        "pairs_per_block": len(PAIRS),
        "samples_per_mode": args.schedule_blocks * 2,
        "samples_per_block": len(PAIRS) * 4,
        "worker_warmups": args.warmups,
        "schedule": "matched_pair_abba_independent_processes",
        "modes": list(MODES),
        "timing_fields": list(TIMING_FIELDS),
        "closure_timing_fields": list(CLOSURE_TIMING_FIELDS),
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
    for block_index in range(args.schedule_blocks):
        for pair_name, pair_slot, mode in matched_abba_schedule(block_index):
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
            if sample_schema_version(sample) != SCHEMA_VERSION:
                raise RuntimeError(
                    f"sample {sample_id} used schema {sample.get('schema_version')}; "
                    f"expected schema {SCHEMA_VERSION}"
                )
            sample["schedule"] = {
                "schedule_block_index": block_index,
                "pair": pair_name,
                "pair_slot": pair_slot,
                "global_index": sample_index,
            }
            write_json(sample_dir / "sample.json", sample)
            samples.append(sample)
            measurement = sample["measurement"]
            closure_text = " ".join(
                f"{field}={format_optional(measurement[field], 6 if field == 'driver_closure_fraction' else 3)}"
                for field in CLOSURE_TIMING_FIELDS
            )
            print(
                f"SAMPLE index={sample_index} block={block_index} pair={pair_name} "
                f"pair_slot={pair_slot} mode={mode} pid={sample['pid']} "
                f"unmap_ms={measurement['unmap_ms']:.3f} "
                f"copy_submit_ms={measurement['copy_ms']:.3f} "
                f"copy_complete_ms={format_optional(measurement['copy_complete_ms'])} "
                f"map_ms={measurement['map_ms']:.3f} "
                f"e2e_ms={measurement['wall_ms']:.3f} "
                f"gib_s={measurement['gib_s']:.3f} "
                f"copy_complete_gib_s="
                f"{format_optional(measurement['copy_complete_gib_s'])} "
                f"{closure_text}",
                flush=True,
            )
            sample_index += 1

    aggregate_result = aggregate(samples)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "config": config,
        "metadata": {
            **metadata,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        },
        "sample_count": len(samples),
        "unique_process_count": len({int(sample["pid"]) for sample in samples}),
        "aggregate": aggregate_result,
        "samples": samples,
    }
    write_json(artifact_dir / "summary.json", summary)
    with (artifact_dir / "samples.jsonl").open("w", encoding="utf-8") as output:
        for sample in samples:
            output.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")
    for mode in MODES:
        row = aggregate_result["modes"][mode]
        closure_text = " ".join(
            f"{field}_p50={format_optional(row[f'{field}_p50'], 6 if field == 'driver_closure_fraction' else 3)}"
            for field in CLOSURE_TIMING_FIELDS
        )
        print(f"CLOSURE mode={mode} {closure_text}")
    print("SUMMARY " + json.dumps(summary["aggregate"], sort_keys=True))
    print(f"ARTIFACT_DIR={artifact_dir}")
    return artifact_dir


def main() -> int:
    args = parse_args()
    lock_path = Path("/tmp/uvm_slot_64k_overwrite_stage_sync_compare.lock")
    try:
        with lock_path.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "another slot-64K overwrite benchmark is already running"
                ) from error
            run(args)
        return 0
    except Exception as error:  # noqa: BLE001 - command-line boundary.
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
