#!/usr/bin/env python3
"""Validate the preregistered MoE-Infinity demand-only thread calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_order(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("order must contain integers") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("order must contain positive integers")
    return result


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration_root", type=Path)
    parser.add_argument(
        "--order", type=parse_order, default=(1, 4, 8, 14, 14, 8, 4, 1)
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    by_thread: dict[int, dict[str, Any]] = {}
    all_output_hashes: set[str] = set()
    all_prompt_hashes: set[str] = set()
    for index, thread_count in enumerate(args.order, start=1):
        label = f"scan_{index:02d}_t{thread_count}"
        run_root = args.calibration_root / "runs" / label
        case_root = run_root / "cases" / "demand-only"
        result_path = case_root / "result.json"
        returncode_path = case_root / "returncode.txt"
        if not result_path.is_file() or not returncode_path.is_file():
            failures.append(f"{label}: missing result or returncode")
            continue
        result = load_json(result_path)
        if returncode_path.read_text(encoding="utf-8").strip() != "0":
            failures.append(f"{label}: nonzero returncode")
        if result.get("status") != "ok":
            failures.append(f"{label}: status={result.get('status')!r}")
        qualification = result.get("qualification", {})
        if qualification.get("status") != "ok":
            failures.append(f"{label}: qualification failed")
        final_counters = qualification.get("instrumentation_final", {})
        for field in (
            "prediction_calls",
            "correction_calls",
            "enqueue_attempts",
            "enqueue_errors",
        ):
            if final_counters.get(field) != 0:
                failures.append(f"{label}: demand counter {field} is not zero")

        protocol = result.get("protocol", {})
        expected_protocol = {
            "batch_size": 1,
            "prompt_tokens": 512,
            "output_tokens": 32,
            "warmup_requests": args.warmups,
            "measured_repetitions": args.repetitions,
            "expert_compute_threads": thread_count,
            "greedy": True,
            "use_native_engine": False,
        }
        for key, expected in expected_protocol.items():
            if protocol.get(key) != expected:
                failures.append(
                    f"{label}: protocol {key}={protocol.get(key)!r}, "
                    f"expected {expected!r}"
                )
        resolved = result.get("resolved_archer_config", {})
        if resolved.get("num_threads") != thread_count:
            failures.append(f"{label}: resolved thread count mismatch")
        if resolved.get("speculative_prefetch") is not False:
            failures.append(f"{label}: speculative prefetch is enabled")

        prompt = result.get("prompt", {})
        if prompt.get("source_file_sha256") != args.fixture_sha256:
            failures.append(f"{label}: calibration fixture SHA-256 mismatch")
        prompt_hash = prompt.get("prompt_token_ids_sha256")
        if isinstance(prompt_hash, str):
            all_prompt_hashes.add(prompt_hash)
        else:
            failures.append(f"{label}: prompt token hash missing")
        output_hashes = result.get("summary", {}).get("output_hashes", [])
        if len(output_hashes) != 1:
            failures.append(f"{label}: measured output hash is unstable")
        else:
            all_output_hashes.add(output_hashes[0])

        trials = result.get("trials", [])
        if not isinstance(trials, list) or len(trials) != args.repetitions:
            failures.append(f"{label}: wrong measured trial count")
            continue
        tpot_values = [float(trial["decode_tpot_seconds"]) for trial in trials]
        candidate = by_thread.setdefault(
            thread_count,
            {"thread_count": thread_count, "processes": [], "tpot_seconds": []},
        )
        candidate["processes"].append(
            {
                "label": label,
                "result_path": str(result_path.resolve()),
                "median_tpot_seconds": statistics.median(tpot_values),
                "output_hash": output_hashes[0]
                if len(output_hashes) == 1
                else None,
            }
        )
        candidate["tpot_seconds"].extend(tpot_values)

    if len(all_prompt_hashes) != 1:
        failures.append(
            f"calibration runs used {len(all_prompt_hashes)} prompt token hashes"
        )
    if len(all_output_hashes) != 1:
        failures.append(
            f"calibration runs produced {len(all_output_hashes)} output hashes"
        )
    expected_processes_per_thread = {
        thread_count: args.order.count(thread_count)
        for thread_count in set(args.order)
    }
    candidates: list[dict[str, Any]] = []
    for thread_count in sorted(expected_processes_per_thread):
        candidate = by_thread.get(thread_count)
        if candidate is None:
            failures.append(f"thread {thread_count}: no completed process")
            continue
        if len(candidate["processes"]) != expected_processes_per_thread[thread_count]:
            failures.append(
                f"thread {thread_count}: completed {len(candidate['processes'])} "
                f"processes, expected {expected_processes_per_thread[thread_count]}"
            )
        candidate["tpot_summary_seconds"] = summary(candidate.pop("tpot_seconds"))
        candidates.append(candidate)

    ranking = sorted(
        candidates,
        key=lambda candidate: candidate["tpot_summary_seconds"]["median"],
    )
    provisional_winner = ranking[0]["thread_count"] if ranking else None
    runner_up = ranking[1]["thread_count"] if len(ranking) > 1 else None
    snapshot = {
        "schema_version": 1,
        "experiment": "moe_infinity_demand_thread_calibration",
        "status": "ok" if not failures else "error",
        "calibration_root": str(args.calibration_root.resolve()),
        "preregistered_order": list(args.order),
        "protocol": {
            "prompt_tokens": 512,
            "output_tokens": 32,
            "warmups_per_process": args.warmups,
            "repetitions_per_process": args.repetitions,
            "fixture_sha256": args.fixture_sha256,
            "selection_metric": "median decode TPOT across all measured trials",
        },
        "candidates": candidates,
        "ranking_threads": [candidate["thread_count"] for candidate in ranking],
        "provisional_winner_threads": provisional_winner,
        "runner_up_threads": runner_up,
        "requires_confirmation": (
            "Run an independent A/B/B/A confirmation of the provisional winner "
            "and runner-up before freezing a formal performance setting."
        ),
        "prompt_token_ids_sha256": next(iter(all_prompt_hashes), None),
        "cross_thread_output_hash_equal": len(all_output_hashes) == 1,
        "failures": failures,
    }
    atomic_text(
        args.output_json,
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    rows = []
    for candidate in ranking:
        stats = candidate["tpot_summary_seconds"]
        rows.append(
            f"| {candidate['thread_count']} | {len(candidate['processes'])} | "
            f"{stats['count']} | {stats['median'] * 1000:.3f} | "
            f"{stats['mean'] * 1000:.3f} | {stats['stdev'] * 1000:.3f} |"
        )
    markdown = "\n".join(
        [
            "# MoE-Infinity demand-only thread calibration",
            "",
            f"Status: `{snapshot['status']}`",
            "",
            "This calibration uses a prompt fixture distinct from formal "
            "evaluation and a preregistered forward/reverse process order.",
            "",
            "| Threads | Cold processes | Trials | Median TPOT ms | Mean TPOT ms | Stdev ms |",
            "|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"Provisional winner: `{provisional_winner}`; runner-up: `{runner_up}`.",
            "An independent A/B/B/A confirmation is required before formal use.",
            "",
            f"Failures: `{failures}`",
            "",
        ]
    )
    atomic_text(args.output_md, markdown)
    snapshot_sha256 = hashlib.sha256(
        args.output_json.read_bytes()
    ).hexdigest()
    terminal = {
        "schema_version": 1,
        "experiment": "moe_infinity_demand_thread_calibration",
        "status": "complete" if not failures else "failed",
        "snapshot_status": snapshot["status"],
        "snapshot_sha256": snapshot_sha256,
        "provisional_winner_threads": provisional_winner,
        "requires_independent_abba_confirmation": True,
    }
    terminal_name = (
        "THREAD_CALIBRATION_COMPLETED.json"
        if not failures
        else "THREAD_CALIBRATION_FAILED.json"
    )
    atomic_text(
        args.calibration_root / terminal_name,
        json.dumps(terminal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
