#!/usr/bin/env python3
"""Validate and summarize custom-empty/custom-remote cold-process ABBA runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ORDER_FIELDS = ["concurrency", "block", "slot", "arm", "profile", "case_root"]
STATUS_FIELDS = [
    *ORDER_FIELDS,
    "status",
    "returncode",
    "started_at",
    "ended_at",
]
MIN_ABBA_BLOCKS = 3


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summary(values: list[float]) -> dict[str, float | int | None]:
    value_mean = mean(values)
    sample_stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": value_mean,
        "median": statistics.median(values) if values else None,
        "sample_stddev": sample_stddev if values else None,
        "cv": (
            sample_stddev / value_mean
            if value_mean is not None and value_mean != 0.0
            else None
        ),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def bootstrap_ratio(
    numerator: list[float], denominator: list[float], *, samples: int = 10_000
) -> dict[str, float]:
    if not numerator or not denominator:
        raise ValueError("bootstrap inputs are empty")
    rng = random.Random(0)
    ratios = []
    for _ in range(samples):
        top = statistics.fmean(rng.choice(numerator) for _ in numerator)
        bottom = statistics.fmean(rng.choice(denominator) for _ in denominator)
        ratios.append(top / bottom)
    ratios.sort()

    def quantile(fraction: float) -> float:
        position = (len(ratios) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ratios[lower]
        weight = position - lower
        return ratios[lower] * (1 - weight) + ratios[upper] * weight

    return {"lower_95": quantile(0.025), "upper_95": quantile(0.975)}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def read_tsv(path: Path, expected_fields: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != expected_fields:
            raise SystemExit(f"unexpected columns in {path}: {reader.fieldnames}")
        return list(reader)


def schedule_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in ORDER_FIELDS)


def metric_mean(measurement: dict[str, Any], name: str) -> float | None:
    value = measurement.get(name)
    if not isinstance(value, dict):
        return None
    result = value.get("mean")
    return float(result) if isinstance(result, (int, float)) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--order", type=Path)
    parser.add_argument("--execution-status", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    order_path = args.order or args.run_root / "abba_order.tsv"
    status_path = args.execution_status or args.run_root / "execution_status.tsv"
    output_path = args.output or args.run_root / "custom_abba_summary.json"
    markdown_path = args.markdown or args.run_root / "CUSTOM_ABBA_RESULTS.md"
    if not order_path.is_file():
        raise SystemExit(f"missing ABBA order: {order_path}")

    rows = read_tsv(order_path, ORDER_FIELDS)
    if not rows:
        raise SystemExit("ABBA order is empty")
    status_rows = read_tsv(status_path, STATUS_FIELDS) if status_path.is_file() else []
    status_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    for status_row in status_rows:
        key = schedule_key(status_row)
        if key in status_by_key:
            raise SystemExit(f"duplicate execution-status row: {key}")
        status_by_key[key] = status_row
    scheduled_keys = {schedule_key(row) for row in rows}
    if len(scheduled_keys) != len(rows):
        raise SystemExit("ABBA order contains duplicate schedule rows")
    case_roots = [row["case_root"] for row in rows]
    if len(set(case_roots)) != len(case_roots):
        raise SystemExit("ABBA order contains duplicate case_root values")
    extra_status = sorted(set(status_by_key) - scheduled_keys)
    if extra_status:
        raise SystemExit(f"execution status contains unscheduled cases: {extra_status[:4]}")

    per_block: dict[tuple[int, int], list[str]] = defaultdict(list)
    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            concurrency = int(row["concurrency"])
            block = int(row["block"])
            slot = int(row["slot"])
        except ValueError as exc:
            raise SystemExit(f"invalid ABBA integer field: {row}") from exc
        arm = row["arm"]
        profile = row["profile"]
        if concurrency <= 0 or block <= 0 or slot not in {1, 2, 3, 4}:
            raise SystemExit(f"invalid ABBA coordinate: {row}")
        if arm not in {"A", "B"}:
            raise SystemExit(f"invalid arm {arm!r}")
        expected_profile = "custom_empty" if arm == "A" else "custom_remote"
        if profile != expected_profile:
            raise SystemExit(f"arm/profile mismatch: {arm}/{profile}")
        per_block[(concurrency, block)].append(arm)
        parsed_rows.append(
            {
                **row,
                "concurrency": concurrency,
                "block": block,
                "slot": slot,
            }
        )

    for key, sequence in sorted(per_block.items()):
        expected = ["A", "B", "B", "A"] if key[1] % 2 == 1 else ["B", "A", "A", "B"]
        if sequence != expected:
            raise SystemExit(f"ABBA sequence mismatch at {key}: {sequence} != {expected}")

    cases: list[dict[str, Any]] = []
    for row in parsed_rows:
        raw_row = {field: str(row[field]) for field in ORDER_FIELDS}
        execution = status_by_key.get(schedule_key(raw_row))
        reasons: list[str] = []
        returncode: int | None = None
        execution_state = "not_recorded"
        if execution is None:
            reasons.append("execution_status_missing")
        else:
            execution_state = execution["status"]
            try:
                returncode = int(execution["returncode"])
            except ValueError:
                reasons.append("execution_returncode_invalid")
            if execution_state != "success":
                reasons.append(f"execution_status_{execution_state or 'empty'}")
            if returncode != 0:
                reasons.append(f"execution_returncode_{returncode}")

        case_root = Path(str(row["case_root"]))
        result_path = case_root / "result.json"
        gate_path = case_root / "integrity_gate.json"
        result: dict[str, Any] | None = None
        gate: dict[str, Any] | None = None
        if not reasons:
            for path, label in ((result_path, "result"), (gate_path, "integrity_gate")):
                if not path.is_file():
                    reasons.append(f"{label}_missing")
            if not reasons:
                try:
                    result = load_json(result_path)
                    gate = load_json(gate_path)
                except (OSError, ValueError, UnicodeError) as exc:
                    reasons.append(f"artifact_parse_error:{exc}")
        measurement: dict[str, Any] | None = None
        if not reasons and result is not None and gate is not None:
            if result.get("status") != "ok":
                reasons.append("benchmark_status_not_ok")
            if gate.get("status") != "pass":
                reasons.append("integrity_gate_not_pass")
            raw_measurement = result.get("measurement")
            if not isinstance(raw_measurement, dict):
                reasons.append("measurement_missing")
            else:
                measurement = raw_measurement
                throughput = measurement.get("output_tokens_per_second")
                if (
                    not isinstance(throughput, (int, float))
                    or not math.isfinite(float(throughput))
                    or throughput <= 0
                ):
                    reasons.append("throughput_invalid")
                elapsed = measurement.get("elapsed_seconds")
                if (
                    not isinstance(elapsed, (int, float))
                    or not math.isfinite(float(elapsed))
                    or elapsed <= 0
                ):
                    reasons.append("elapsed_seconds_invalid")

        case: dict[str, Any] = {
            "concurrency": row["concurrency"],
            "block": row["block"],
            "slot": row["slot"],
            "arm": row["arm"],
            "profile": row["profile"],
            "case_root": str(case_root),
            "scheduled": True,
            "success": not reasons,
            "failure_reasons": reasons,
            "execution_status": execution_state,
            "returncode": returncode,
            "started_at": execution.get("started_at") if execution else None,
            "ended_at": execution.get("ended_at") if execution else None,
        }
        if not reasons and measurement is not None and result is not None and gate is not None:
            ready = result.get("post_warmup_ready_nvml")
            ready_mib = (
                float(ready["used_mib"])
                if isinstance(ready, dict)
                and isinstance(ready.get("used_mib"), (int, float))
                else None
            )
            case.update(
                {
                    "throughput": float(measurement["output_tokens_per_second"]),
                    "elapsed_seconds": float(measurement["elapsed_seconds"]),
                    "ttft_ms": metric_mean(measurement, "ttft_ms"),
                    "tpot_ms": metric_mean(measurement, "tpot_ms"),
                    "ready_used_mib": ready_mib,
                    "output_hash": measurement.get(
                        "aggregate_output_token_ids_sha256"
                    ),
                    "prompt_token_ids_frozen": bool(
                        gate.get("prompt_token_ids_frozen")
                    ),
                    "formal_run": gate.get("formal_run") is True,
                    "metric_counts_complete": gate.get("metric_counts_complete")
                    is True,
                    "metric_values_finite_nonnegative": gate.get(
                        "metric_values_finite_nonnegative"
                    )
                    is True,
                    "aggregate_metrics_finite_positive": gate.get(
                        "aggregate_metrics_finite_positive"
                    )
                    is True,
                    "publication_eligible": bool(gate.get("publication_eligible")),
                    "placement_qualified": gate.get("placement_qualified") is True,
                    "runtime_remote_placement_manifest_available": gate.get(
                        "runtime_remote_placement_manifest_available"
                    )
                    is True,
                    "release_time_remote_mapping_verified": gate.get(
                        "release_time_remote_mapping_verified"
                    )
                    is True,
                    "snapshot_time_remote_cpu_residency_verified": gate.get(
                        "snapshot_time_remote_cpu_residency_verified"
                    )
                    is True,
                }
            )
        cases.append(case)

    successful_cases = [case for case in cases if case["success"]]
    failed_cases = [case for case in cases if not case["success"]]
    any_failure = bool(failed_cases)
    grouped_scheduled: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_success: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        key = (int(case["concurrency"]), str(case["arm"]))
        grouped_scheduled[key].append(case)
        if case["success"]:
            grouped_success[key].append(case)

    concurrencies = sorted({int(case["concurrency"]) for case in cases})
    points: list[dict[str, Any]] = []
    for concurrency in concurrencies:
        scheduled_a = grouped_scheduled[(concurrency, "A")]
        scheduled_b = grouped_scheduled[(concurrency, "B")]
        arm_a = grouped_success[(concurrency, "A")]
        arm_b = grouped_success[(concurrency, "B")]
        throughput_a = [float(item["throughput"]) for item in arm_a]
        throughput_b = [float(item["throughput"]) for item in arm_b]
        point_complete = (
            len(scheduled_a) == len(scheduled_b)
            and bool(scheduled_a)
            and len(arm_a) == len(scheduled_a)
            and len(arm_b) == len(scheduled_b)
        )
        available_ratio = (
            statistics.fmean(throughput_b) / statistics.fmean(throughput_a)
            if throughput_a and throughput_b
            else None
        )
        point_cases = [*arm_a, *arm_b]
        blocks_scheduled = len(
            {int(item["block"]) for item in [*scheduled_a, *scheduled_b]}
        )
        hash_values = [item.get("output_hash") for item in point_cases]
        hash_gate = (
            point_complete
            and len(hash_values) == len(scheduled_a) + len(scheduled_b)
            and all(isinstance(value, str) and value for value in hash_values)
            and len(set(hash_values)) == 1
        )
        prompt_ids_frozen = point_complete and all(
            item.get("prompt_token_ids_frozen") is True for item in point_cases
        )
        formal_case_gates = point_complete and all(
            item.get("formal_run") is True
            and item.get("metric_counts_complete") is True
            and item.get("metric_values_finite_nonnegative") is True
            and item.get("aggregate_metrics_finite_positive") is True
            for item in point_cases
        )
        placement_gates = point_complete and all(
            item.get("placement_qualified") is True
            and item.get("runtime_remote_placement_manifest_available") is True
            and item.get("release_time_remote_mapping_verified") is True
            and item.get("snapshot_time_remote_cpu_residency_verified") is True
            for item in point_cases
        )
        formal_ratio_ready = (
            not any_failure
            and point_complete
            and prompt_ids_frozen
            and hash_gate
            and formal_case_gates
            and placement_gates
        )
        formal_ratio = available_ratio if formal_ratio_ready else None
        ratio_status = (
            "valid_formal_gates"
            if formal_ratio is not None
            else (
                (
                    "screening_only_experiment_has_failed_cases"
                    if any_failure
                    else "screening_only_formal_gate_failed"
                )
                if available_ratio is not None
                else "invalid_insufficient_successful_arms"
            )
        )
        ready_a = [
            float(item["ready_used_mib"])
            for item in arm_a
            if item.get("ready_used_mib") is not None
        ]
        ready_b = [
            float(item["ready_used_mib"])
            for item in arm_b
            if item.get("ready_used_mib") is not None
        ]
        throughput_summary_a = summary(throughput_a)
        throughput_summary_b = summary(throughput_b)
        points.append(
            {
                "concurrency": concurrency,
                "scheduled_per_arm": {
                    "custom_empty": len(scheduled_a),
                    "custom_remote": len(scheduled_b),
                },
                "successful_per_arm": {
                    "custom_empty": len(arm_a),
                    "custom_remote": len(arm_b),
                },
                "schedule_complete": point_complete,
                "abba_blocks_scheduled": blocks_scheduled,
                "minimum_abba_blocks_required": MIN_ABBA_BLOCKS,
                "minimum_abba_blocks_complete": (
                    point_complete and blocks_scheduled >= MIN_ABBA_BLOCKS
                ),
                "formal_case_gates_complete": formal_case_gates,
                "runtime_placement_gates_complete": placement_gates,
                "custom_empty": {
                    "throughput": throughput_summary_a,
                    "ttft_ms": summary(
                        [float(item["ttft_ms"]) for item in arm_a if item.get("ttft_ms") is not None]
                    ),
                    "tpot_ms": summary(
                        [float(item["tpot_ms"]) for item in arm_a if item.get("tpot_ms") is not None]
                    ),
                    "ready_used_mib": summary(ready_a),
                },
                "custom_remote": {
                    "throughput": throughput_summary_b,
                    "ttft_ms": summary(
                        [float(item["ttft_ms"]) for item in arm_b if item.get("ttft_ms") is not None]
                    ),
                    "tpot_ms": summary(
                        [float(item["tpot_ms"]) for item in arm_b if item.get("tpot_ms") is not None]
                    ),
                    "ready_used_mib": summary(ready_b),
                },
                "remote_over_empty_throughput_ratio": formal_ratio,
                "screening_remote_over_empty_throughput_ratio": (
                    available_ratio if formal_ratio is None else None
                ),
                "ratio_status": ratio_status,
                "remote_over_empty_throughput_ratio_bootstrap_95": (
                    bootstrap_ratio(throughput_b, throughput_a)
                    if formal_ratio is not None
                    else None
                ),
                "output_hash_exact_across_arms_and_repeats": hash_gate,
                "output_hash": hash_values[0] if hash_gate else None,
                "prompt_token_ids_frozen": prompt_ids_frozen,
                "needs_more_repeats_cv_gt_5pct": any(
                    (value.get("cv") or 0.0) > 0.05
                    for value in (throughput_summary_a, throughput_summary_b)
                ),
                "ready_vram_mean_delta_mib_remote_minus_empty": (
                    statistics.fmean(ready_b) - statistics.fmean(ready_a)
                    if ready_a and ready_b
                    else None
                ),
            }
        )

    scheduled_count = len(cases)
    success_count = len(successful_cases)
    formal_ratios_valid = bool(points) and all(
        point["remote_over_empty_throughput_ratio"] is not None for point in points
    )
    placement_evidence_complete = bool(successful_cases) and all(
        case.get("placement_qualified") is True
        and case.get("runtime_remote_placement_manifest_available") is True
        and case.get("release_time_remote_mapping_verified") is True
        and case.get("snapshot_time_remote_cpu_residency_verified") is True
        for case in successful_cases
    )
    minimum_blocks_complete = bool(points) and all(
        point["minimum_abba_blocks_complete"] is True for point in points
    )
    blockers: list[str] = []
    if any_failure:
        blockers.append("one or more pre-registered ABBA cases failed or were not run")
    if not formal_ratios_valid:
        blockers.append(
            "formal-run, finite-metric, frozen-prompt-ID, or output-hash gate failed"
        )
    if not placement_evidence_complete:
        blockers.append("runtime remote-placement evidence is incomplete")
    if not minimum_blocks_complete:
        blockers.append(
            f"fewer than {MIN_ABBA_BLOCKS} complete ABBA blocks at one or more points"
        )
    snapshot = {
        "schema_version": 2,
        "experiment": "custom_empty_vs_custom_remote_abba",
        "run_root": str(args.run_root.resolve()),
        "order": str(order_path.resolve()),
        "execution_status": str(status_path.resolve()),
        "cases": cases,
        "points": points,
        "integrity": {
            "balanced_abba_schedule": True,
            "scheduled_cases": scheduled_count,
            "successful_cases": success_count,
            "failed_cases": scheduled_count - success_count,
            "failure_rate": (scheduled_count - success_count) / scheduled_count,
            "all_cases_passed": not any_failure,
            "failed_case_roots": [case["case_root"] for case in failed_cases],
            "ratios_valid_for_formal_use": formal_ratios_valid,
            "output_hash_exact_all_points": bool(points)
            and all(point["output_hash_exact_across_arms_and_repeats"] for point in points),
            "prompt_token_ids_frozen_all_points": bool(points)
            and all(point["prompt_token_ids_frozen"] for point in points),
            "runtime_remote_placement_manifest_available": placement_evidence_complete,
            "minimum_abba_blocks_complete": minimum_blocks_complete,
            "publication_eligible": not blockers,
            "publication_blockers": blockers,
        },
    }
    output_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Custom empty vs remote ABBA",
        "",
        f"Scheduled: {scheduled_count}; successful: {success_count}; "
        f"failed/not-run: {scheduled_count - success_count}; "
        f"failure rate: {(scheduled_count - success_count) / scheduled_count:.2%}.",
        "",
        "| concurrency | success/scheduled | empty tok/s | remote tok/s | ratio | status | hash | more repeats |",
        "|---:|---:|---:|---:|---:|:---|:---:|:---:|",
    ]
    for point in points:
        empty_mean = point["custom_empty"]["throughput"]["mean"]
        remote_mean = point["custom_remote"]["throughput"]["mean"]
        ratio = point["remote_over_empty_throughput_ratio"]
        screening_ratio = point["screening_remote_over_empty_throughput_ratio"]
        ratio_text = (
            f"{ratio:.4f}"
            if ratio is not None
            else (f"screening {screening_ratio:.4f}" if screening_ratio is not None else "invalid")
        )
        successes = sum(point["successful_per_arm"].values())
        scheduled = sum(point["scheduled_per_arm"].values())
        lines.append(
            "| {concurrency} | {successes}/{scheduled} | {a} | {b} | {ratio} | {status} | {hash_gate} | {repeat} |".format(
                concurrency=point["concurrency"],
                successes=successes,
                scheduled=scheduled,
                a=f"{empty_mean:.3f}" if empty_mean is not None else "N/A",
                b=f"{remote_mean:.3f}" if remote_mean is not None else "N/A",
                ratio=ratio_text,
                status=point["ratio_status"],
                hash_gate="pass" if point["output_hash_exact_across_arms_and_repeats"] else "FAIL",
                repeat="yes" if point["needs_more_repeats_cv_gt_5pct"] else "no",
            )
        )
    if failed_cases:
        lines.extend(
            [
                "",
                "## Failed or missing cases",
                "",
                "| case | execution status | return code | reason |",
                "|:---|:---|---:|:---|",
            ]
        )
        for case in failed_cases:
            reason = "; ".join(case["failure_reasons"]).replace("|", "\\|")
            lines.append(
                f"| {case['case_root']} | {case['execution_status']} | "
                f"{case['returncode']} | {reason} |"
            )
    lines.extend(
        [
            "",
            "This is an internal custom-path ABBA, not the cross-framework result.",
            "Any ratio based on an incomplete pre-registered schedule is screening-only.",
            f"Publication eligibility requires at least {MIN_ABBA_BLOCKS} complete ABBA blocks per point and all runtime placement gates.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(snapshot["integrity"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
