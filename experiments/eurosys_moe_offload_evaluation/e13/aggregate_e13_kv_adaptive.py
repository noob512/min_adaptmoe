#!/usr/bin/env python3
"""Aggregate independent cold-process E13 repetitions."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def variation(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("qualification", "formal", "capacity"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def read_nvml(path: Path, start: float, finish: float) -> dict[str, Any]:
    samples: list[dict[str, float]] = []
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    timestamp = float(row["unix_s"])
                    if row.get("sample_ok") != "1" or not start <= timestamp <= finish:
                        continue
                    samples.append(
                        {
                            "gpu_util_percent": float(row["gpu_util_percent"]),
                            "memory_used_mib": float(row["memory_used_mib"]),
                            "power_w": float(row["power_w"]),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    return {
        "sample_count": len(samples),
        "gpu_util_percent_mean": (
            statistics.fmean(item["gpu_util_percent"] for item in samples)
            if samples
            else None
        ),
        "gpu_util_percent_min": min(
            (item["gpu_util_percent"] for item in samples), default=None
        ),
        "gpu_util_percent_max": max(
            (item["gpu_util_percent"] for item in samples), default=None
        ),
        "memory_used_mib_mean": (
            statistics.fmean(item["memory_used_mib"] for item in samples)
            if samples
            else None
        ),
        "memory_used_mib_max": max(
            (item["memory_used_mib"] for item in samples), default=None
        ),
        "power_w_mean": (
            statistics.fmean(item["power_w"] for item in samples)
            if samples
            else None
        ),
    }


def metric(summary: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = summary
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def select_best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return min(
        rows,
        key=lambda row: (
            int(row["failed_requests_mean"] + row["rejected_requests_mean"]),
            -float(row["successful_requests_mean"]),
            -float(row["request_throughput_mean"]),
            float(row["e2e_p99_pooled_ms"] or math.inf),
            row["arm"],
        ),
    )


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.run_root.resolve()
    protocol = load_json(args.config)
    arms = [item["name"] for item in protocol["arms"]]
    strategies = {item["name"]: item["strategy"] for item in protocol["arms"]}
    seeds = [int(value) for value in protocol["profiles"][args.profile]["seeds"]]
    scenarios = list(protocol["profiles"][args.profile]["scenario_names"])
    case_rows: list[dict[str, Any]] = []
    capacity_case_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    request_pool: dict[tuple[str, str], list[dict[str, Any]]] = {}
    result_map: dict[tuple[int, str], dict[str, Any]] = {}

    for repeat_index, seed in enumerate(seeds):
        for arm in arms:
            case_dir = root / f"repeat-{repeat_index:02d}-seed-{seed}" / arm
            result_path = case_dir / "result.json"
            if not result_path.is_file():
                continue
            result = load_json(result_path)
            result_map[(repeat_index, arm)] = result
            if result.get("status") != "ok":
                continue
            trace_start = float(result["trace_started_unix_s"])
            for scenario in scenarios:
                summary = result["scenario_summaries"][scenario]
                requests = [
                    item for item in result["requests"] if item["scenario"] == scenario
                ]
                successful = [item for item in requests if item.get("success")]
                scenario_start = trace_start + min(
                    (float(item["arrival_elapsed_s"]) for item in requests), default=0.0
                )
                scenario_finish = trace_start + max(
                    (
                        float(item["finished_elapsed_s"])
                        for item in successful
                        if item.get("finished_elapsed_s") is not None
                    ),
                    default=scenario_start - trace_start,
                )
                nvml = read_nvml(case_dir / "nvml.csv", scenario_start, scenario_finish)
                row = {
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "arm": arm,
                    "strategy": strategies[arm],
                    "scenario": scenario,
                    "offered_requests": int(summary["offered_requests"]),
                    "successful_requests": int(summary["successful_requests"]),
                    "rejected_requests": int(summary["rejected_requests"]),
                    "failed_requests": int(summary["failed_requests"]),
                    "request_throughput": float(
                        summary["successful_request_throughput_per_second"]
                    ),
                    "output_token_throughput": float(
                        summary["output_token_throughput_per_second"]
                    ),
                    "ttft_p50_ms": metric(summary, "ttft_ms.p50"),
                    "ttft_p95_ms": metric(summary, "ttft_ms.p95"),
                    "ttft_p99_ms": metric(summary, "ttft_ms.p99"),
                    "tpot_p50_ms": metric(summary, "tpot_ms.p50"),
                    "tpot_p95_ms": metric(summary, "tpot_ms.p95"),
                    "tpot_p99_ms": metric(summary, "tpot_ms.p99"),
                    "e2e_p50_ms": metric(summary, "e2e_ms.p50"),
                    "e2e_p95_ms": metric(summary, "e2e_ms.p95"),
                    "e2e_p99_ms": metric(summary, "e2e_ms.p99"),
                    "maximum_scheduler_running_requests": int(
                        summary["maximum_scheduler_running_requests"]
                    ),
                    "maximum_scheduled_requests_per_step": int(
                        summary["maximum_scheduled_requests_per_step"]
                    ),
                    "maximum_prompt_tokens_completed": int(
                        summary["maximum_prompt_tokens_completed"]
                    ),
                    "maximum_context_tokens_completed": int(
                        summary["maximum_context_tokens_completed"]
                    ),
                    "maximum_active_kv_blocks": int(summary["maximum_active_kv_blocks"]),
                    "minimum_target_blocks": summary["minimum_target_blocks"],
                    "maximum_target_blocks": summary["maximum_target_blocks"],
                    "mean_target_blocks": summary["mean_target_blocks"],
                    "resize_count": int(summary["resize_count"]),
                    "resize_attempt_count": int(
                        summary.get("resize_attempt_count", summary["resize_count"])
                    ),
                    "failed_resize_attempt_count": int(
                        summary.get("failed_resize_attempt_count", 0)
                    ),
                    "grow_count": int(summary["grow_count"]),
                    "shrink_count": int(summary["shrink_count"]),
                    "resize_wall_ms": float(summary["resize_wall_ms"]),
                    "resize_response_p50_ms": metric(
                        summary, "resize_response_ms.p50"
                    ),
                    "resize_response_p95_ms": metric(
                        summary, "resize_response_ms.p95"
                    ),
                    "resize_response_max_ms": metric(
                        summary, "resize_response_ms.max"
                    ),
                    "resize_attempt_wall_ms": float(
                        summary.get(
                            "resize_attempt_wall_ms", summary["resize_wall_ms"]
                        )
                    ),
                    "requested_kv_h2d_bytes": int(summary["requested_kv_h2d_bytes"]),
                    "requested_kv_d2h_bytes": int(summary["requested_kv_d2h_bytes"]),
                    **nvml,
                }
                case_rows.append(row)
                request_pool.setdefault((arm, scenario), []).extend(successful)

            for epoch in result.get("epochs") or []:
                if epoch.get("scenario") != "capacity_envelope":
                    continue
                summary = epoch["summary"]
                offered = int(summary["offered_requests"])
                maximum_running = int(
                    summary["maximum_scheduler_running_requests"]
                )
                maximum_context = int(
                    summary["maximum_context_tokens_completed"]
                )
                capacity_case_rows.append(
                    {
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "arm": arm,
                        "strategy": strategies[arm],
                        "epoch": str(epoch["label"]),
                        "offered_concurrency": offered,
                        "successful_requests": int(
                            summary["successful_requests"]
                        ),
                        "rejected_requests": int(summary["rejected_requests"]),
                        "failed_requests": int(summary["failed_requests"]),
                        "maximum_running_requests": maximum_running,
                        "requests_not_simultaneously_resident": max(
                            offered - maximum_running,
                            0,
                        ),
                        "all_offered_immediately_resident": (
                            maximum_running >= offered
                        ),
                        "maximum_context_tokens_per_request": maximum_context,
                        "maximum_live_context_tokens": (
                            maximum_running * maximum_context
                        ),
                        "minimum_target_blocks": summary[
                            "minimum_target_blocks"
                        ],
                        "maximum_target_blocks": summary[
                            "maximum_target_blocks"
                        ],
                    }
                )

            for step in result["time_series"]:
                time_rows.append(
                    {
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "arm": arm,
                        "strategy": strategies[arm],
                        "elapsed_s": step["elapsed_s"],
                        "unix_s": step["unix_s"],
                        "scenario": step["scenario"],
                        "epoch": step["epoch"],
                        "engine_step": step["engine_step"],
                        "step_wall_ms": step["step_wall_ms"],
                        "instant_output_tokens_per_second": step[
                            "instant_output_tokens_per_second"
                        ],
                        "emitted_tokens": step["emitted_tokens"],
                        "running_request_count": step["running_request_count"],
                        "waiting_request_count": step["waiting_request_count"],
                        "preempted_request_count": step["preempted_request_count"],
                        "target_blocks": step["kv"]["current_target_blocks"],
                        "active_blocks": step["kv"]["active_blocks"],
                        "gpu_free_blocks": step["kv"]["gpu_free_blocks"],
                        "gpu_backup_blocks": step["kv"]["gpu_backup_blocks"],
                        "cpu_offline_blocks": step["kv"]["cpu_offline_blocks"],
                        "predicted_blocks": step["kv"]["predicted_blocks"],
                        "low_streak": step["kv"]["low_streak"],
                        "controller_action": step["kv"].get("last_decision", {}).get(
                            "action"
                        ),
                        "active_plan_class": step["active_plan_class"],
                    }
                )

    aggregate_rows: list[dict[str, Any]] = []
    numeric_map = {
        "request_throughput": "request_throughput",
        "output_token_throughput": "output_token_throughput",
        "ttft_p50_ms": "ttft_p50_ms",
        "ttft_p95_ms": "ttft_p95_ms",
        "ttft_p99_ms": "ttft_p99_ms",
        "tpot_p50_ms": "tpot_p50_ms",
        "tpot_p95_ms": "tpot_p95_ms",
        "tpot_p99_ms": "tpot_p99_ms",
        "e2e_p50_ms": "e2e_p50_ms",
        "e2e_p95_ms": "e2e_p95_ms",
        "e2e_p99_ms": "e2e_p99_ms",
        "gpu_util_percent": "gpu_util_percent_mean",
        "memory_used_mib": "memory_used_mib_mean",
        "resize_wall_ms": "resize_wall_ms",
        "resize_attempt_wall_ms": "resize_attempt_wall_ms",
        "resize_response_p50_ms": "resize_response_p50_ms",
        "resize_response_p95_ms": "resize_response_p95_ms",
        "resize_response_max_ms": "resize_response_max_ms",
        "maximum_running_requests": "maximum_scheduler_running_requests",
        "maximum_context_tokens": "maximum_context_tokens_completed",
    }
    for arm in arms:
        for scenario in scenarios:
            rows = [
                row
                for row in case_rows
                if row["arm"] == arm and row["scenario"] == scenario
            ]
            pooled = request_pool.get((arm, scenario), [])
            aggregate_row: dict[str, Any] = {
                "arm": arm,
                "strategy": strategies[arm],
                "scenario": scenario,
                "repeat_count": len(rows),
                "offered_requests_mean": statistics.fmean(
                    row["offered_requests"] for row in rows
                ) if rows else 0.0,
                "successful_requests_mean": statistics.fmean(
                    row["successful_requests"] for row in rows
                ) if rows else 0.0,
                "rejected_requests_mean": statistics.fmean(
                    row["rejected_requests"] for row in rows
                ) if rows else 0.0,
                "failed_requests_mean": statistics.fmean(
                    row["failed_requests"] for row in rows
                ) if rows else 0.0,
                "ttft_p95_pooled_ms": percentile(
                    [float(item["ttft_ms"]) for item in pooled], 95.0
                ),
                "ttft_p99_pooled_ms": percentile(
                    [float(item["ttft_ms"]) for item in pooled], 99.0
                ),
                "tpot_p95_pooled_ms": percentile(
                    [
                        float(item["tpot_ms"])
                        for item in pooled
                        if item.get("tpot_ms") is not None
                    ],
                    95.0,
                ),
                "tpot_p99_pooled_ms": percentile(
                    [
                        float(item["tpot_ms"])
                        for item in pooled
                        if item.get("tpot_ms") is not None
                    ],
                    99.0,
                ),
                "e2e_p95_pooled_ms": percentile(
                    [float(item["e2e_ms"]) for item in pooled], 95.0
                ),
                "e2e_p99_pooled_ms": percentile(
                    [float(item["e2e_ms"]) for item in pooled], 99.0
                ),
                "resize_count_mean": statistics.fmean(
                    row["resize_count"] for row in rows
                ) if rows else 0.0,
                "resize_attempt_count_mean": statistics.fmean(
                    row["resize_attempt_count"] for row in rows
                ) if rows else 0.0,
                "failed_resize_attempt_count_mean": statistics.fmean(
                    row["failed_resize_attempt_count"] for row in rows
                ) if rows else 0.0,
                "grow_count_mean": statistics.fmean(
                    row["grow_count"] for row in rows
                ) if rows else 0.0,
                "shrink_count_mean": statistics.fmean(
                    row["shrink_count"] for row in rows
                ) if rows else 0.0,
            }
            for output_name, source_name in numeric_map.items():
                values = [
                    float(row[source_name])
                    for row in rows
                    if row.get(source_name) is not None
                ]
                stats = variation(values)
                for stat_name, value in stats.items():
                    aggregate_row[f"{output_name}_{stat_name}"] = value
            aggregate_rows.append(aggregate_row)

    best_fixed_by_scenario: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    for scenario in scenarios:
        fixed = [
            row
            for row in aggregate_rows
            if row["scenario"] == scenario and row["strategy"] == "fixed"
        ]
        best = select_best(fixed)
        adaptive = next(
            (
                row
                for row in aggregate_rows
                if row["scenario"] == scenario and row["arm"] == "adaptive"
            ),
            {},
        )
        best_fixed_by_scenario[scenario] = best.get("arm")
        if best and adaptive:
            comparisons[scenario] = {
                "best_fixed": best["arm"],
                "adaptive_request_throughput_over_best_fixed": (
                    adaptive["request_throughput_mean"]
                    / best["request_throughput_mean"]
                    if best["request_throughput_mean"]
                    else None
                ),
                "adaptive_output_token_throughput_over_best_fixed": (
                    adaptive["output_token_throughput_mean"]
                    / best["output_token_throughput_mean"]
                    if best["output_token_throughput_mean"]
                    else None
                ),
                "best_fixed_ttft_p99_over_adaptive": (
                    best["ttft_p99_pooled_ms"] / adaptive["ttft_p99_pooled_ms"]
                    if adaptive["ttft_p99_pooled_ms"]
                    else None
                ),
                "best_fixed_e2e_p99_over_adaptive": (
                    best["e2e_p99_pooled_ms"] / adaptive["e2e_p99_pooled_ms"]
                    if adaptive["e2e_p99_pooled_ms"]
                    else None
                ),
            }

    global_rows: list[dict[str, Any]] = []
    global_pool: dict[str, list[dict[str, Any]]] = {}
    for arm in arms:
        arm_cases: list[dict[str, Any]] = []
        for repeat_index, _seed in enumerate(seeds):
            result = result_map.get((repeat_index, arm))
            if result and result.get("status") == "ok":
                summary = result["summary"]
                arm_cases.append(
                    {
                        "offered": summary["offered_requests"],
                        "success": summary["successful_requests"],
                        "rejected": summary["rejected_requests"],
                        "failed": summary["failed_requests"],
                        "request_throughput": summary[
                            "successful_request_throughput_per_second"
                        ],
                        "output_throughput": summary[
                            "output_token_throughput_per_second"
                        ],
                    }
                )
                global_pool.setdefault(arm, []).extend(
                    item for item in result["requests"] if item.get("success")
                )
        pooled = global_pool.get(arm, [])
        row = {
            "arm": arm,
            "strategy": strategies[arm],
            "repeat_count": len(arm_cases),
            "offered_requests_mean": statistics.fmean(
                item["offered"] for item in arm_cases
            ) if arm_cases else 0.0,
            "successful_requests_mean": statistics.fmean(
                item["success"] for item in arm_cases
            ) if arm_cases else 0.0,
            "rejected_requests_mean": statistics.fmean(
                item["rejected"] for item in arm_cases
            ) if arm_cases else 0.0,
            "failed_requests_mean": statistics.fmean(
                item["failed"] for item in arm_cases
            ) if arm_cases else 0.0,
            "request_throughput_mean": statistics.fmean(
                item["request_throughput"] for item in arm_cases
            ) if arm_cases else 0.0,
            "output_token_throughput_mean": statistics.fmean(
                item["output_throughput"] for item in arm_cases
            ) if arm_cases else 0.0,
            "e2e_p99_pooled_ms": percentile(
                [float(item["e2e_ms"]) for item in pooled], 99.0
            ),
            "ttft_p99_pooled_ms": percentile(
                [float(item["ttft_ms"]) for item in pooled], 99.0
            ),
        }
        global_rows.append(row)
    global_best_fixed = select_best(
        [row for row in global_rows if row["strategy"] == "fixed"]
    )
    adaptive_global = next(
        (row for row in global_rows if row["arm"] == "adaptive"), {}
    )
    global_comparison = {
        "best_fixed": global_best_fixed.get("arm"),
        "adaptive_request_throughput_over_best_fixed": (
            adaptive_global.get("request_throughput_mean", 0)
            / global_best_fixed.get("request_throughput_mean", 1)
            if global_best_fixed.get("request_throughput_mean")
            else None
        ),
        "best_fixed_e2e_p99_over_adaptive": (
            global_best_fixed.get("e2e_p99_pooled_ms")
            / adaptive_global.get("e2e_p99_pooled_ms")
            if adaptive_global.get("e2e_p99_pooled_ms")
            else None
        ),
    }

    capacity_aggregate_rows: list[dict[str, Any]] = []
    capacity_limits: dict[str, Any] = {}
    capacity_epochs = sorted(
        {row["epoch"] for row in capacity_case_rows},
        key=lambda label: next(
            int(row["offered_concurrency"])
            for row in capacity_case_rows
            if row["epoch"] == label
        ),
    )
    for arm in arms:
        arm_aggregate: list[dict[str, Any]] = []
        for epoch in capacity_epochs:
            rows = [
                row
                for row in capacity_case_rows
                if row["arm"] == arm and row["epoch"] == epoch
            ]
            if not rows:
                continue
            running = variation(
                [float(row["maximum_running_requests"]) for row in rows]
            )
            queued = variation(
                [
                    float(row["requests_not_simultaneously_resident"])
                    for row in rows
                ]
            )
            live_context = variation(
                [float(row["maximum_live_context_tokens"]) for row in rows]
            )
            item = {
                "arm": arm,
                "strategy": strategies[arm],
                "epoch": epoch,
                "repeat_count": len(rows),
                "offered_concurrency": int(rows[0]["offered_concurrency"]),
                "successful_requests_mean": statistics.fmean(
                    row["successful_requests"] for row in rows
                ),
                "rejected_requests_mean": statistics.fmean(
                    row["rejected_requests"] for row in rows
                ),
                "failed_requests_mean": statistics.fmean(
                    row["failed_requests"] for row in rows
                ),
                "fully_resident_in_all_repeats": all(
                    bool(row["all_offered_immediately_resident"])
                    for row in rows
                ),
                "maximum_context_tokens_per_request": max(
                    row["maximum_context_tokens_per_request"] for row in rows
                ),
            }
            for prefix, stats in (
                ("maximum_running_requests", running),
                ("requests_not_simultaneously_resident", queued),
                ("maximum_live_context_tokens", live_context),
            ):
                for name, value in stats.items():
                    item[f"{prefix}_{name}"] = value
            capacity_aggregate_rows.append(item)
            arm_aggregate.append(item)
        fully_resident = [
            row
            for row in arm_aggregate
            if row["fully_resident_in_all_repeats"]
        ]
        capacity_limits[arm] = {
            "maximum_fully_resident_offered_concurrency": max(
                (row["offered_concurrency"] for row in fully_resident),
                default=0,
            ),
            "maximum_observed_running_requests": max(
                (
                    int(row["maximum_running_requests_max"])
                    for row in arm_aggregate
                ),
                default=0,
            ),
            "maximum_observed_live_context_tokens": max(
                (
                    int(row["maximum_live_context_tokens_max"])
                    for row in arm_aggregate
                ),
                default=0,
            ),
        }

    return {
        "schema": "eurosys_e13_kv_adaptive_aggregate_v1",
        "generated_at": utc_now(),
        "run_root": str(root),
        "profile": args.profile,
        "expected_repeats": len(seeds),
        "case_rows": case_rows,
        "capacity_case_rows": capacity_case_rows,
        "capacity_aggregate_rows": capacity_aggregate_rows,
        "capacity_limits": capacity_limits,
        "aggregate_rows": aggregate_rows,
        "global_rows": global_rows,
        "best_fixed_by_scenario": best_fixed_by_scenario,
        "global_best_fixed": global_best_fixed.get("arm"),
        "scenario_comparisons": comparisons,
        "global_comparison": global_comparison,
        "time_series": time_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any, digits: int = 2) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# E13 Adaptive 与 Fixed KV 档位汇总",
        "",
        f"- profile：`{payload['profile']}`",
        f"- cold-process repeats：{payload['expected_repeats']}",
        f"- global best-fixed：`{payload['global_best_fixed']}`",
        "",
        "| 场景 | Arm | 成功请求均值 | req/s 均值 [min,max] | TTFT P99 pooled | E2E P99 pooled | GPU util均值 | 切换次数均值 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["aggregate_rows"]:
        lines.append(
            "| {scenario} | {arm} | {success:.1f} | {throughput} [{low},{high}] | {ttft} ms | {e2e} ms | {gpu}% | {switches:.1f} |".format(
                scenario=row["scenario"],
                arm=row["arm"],
                success=float(row["successful_requests_mean"]),
                throughput=format_value(row["request_throughput_mean"], 3),
                low=format_value(row["request_throughput_min"], 3),
                high=format_value(row["request_throughput_max"], 3),
                ttft=format_value(row["ttft_p99_pooled_ms"]),
                e2e=format_value(row["e2e_p99_pooled_ms"]),
                gpu=format_value(row["gpu_util_percent_mean"]),
                switches=float(row["resize_count_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "## 公平比较",
            "",
            "per-scenario best-fixed 是静态场景的 oracle 上界；跨整条 trace 的 global best-fixed 才是长期固定策略的主要对照。",
            "",
            "| 场景 | Best fixed | Adaptive/best-fixed req/s | Adaptive/best-fixed output tok/s | Best-fixed/Adaptive E2E P99 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for scenario, comparison in payload["scenario_comparisons"].items():
        lines.append(
            f"| {scenario} | {comparison['best_fixed']} | "
            f"{format_value(comparison['adaptive_request_throughput_over_best_fixed'], 3)}× | "
            f"{format_value(comparison['adaptive_output_token_throughput_over_best_fixed'], 3)}× | "
            f"{format_value(comparison['best_fixed_e2e_p99_over_adaptive'], 3)}× |"
        )
    if payload.get("capacity_aggregate_rows"):
        lines.extend(
            [
                "",
                "## Capacity envelope",
                "",
                "full-resident 表示所有 offered 请求在每个重复中都曾同时处于 scheduler running 集合；其余请求仍可排队完成，不计作 OOM。",
                "",
                "| Arm | Offered | Max running mean [min,max] | Not resident mean | Full-resident all repeats | Max live context tokens |",
                "| --- | ---: | ---: | ---: | --- | ---: |",
            ]
        )
        for row in payload["capacity_aggregate_rows"]:
            lines.append(
                f"| {row['arm']} | {row['offered_concurrency']} | "
                f"{format_value(row['maximum_running_requests_mean'], 1)} "
                f"[{format_value(row['maximum_running_requests_min'], 0)},"
                f"{format_value(row['maximum_running_requests_max'], 0)}] | "
                f"{format_value(row['requests_not_simultaneously_resident_mean'], 1)} | "
                f"{'yes' if row['fully_resident_in_all_repeats'] else 'no'} | "
                f"{format_value(row['maximum_live_context_tokens_max'], 0)} |"
            )
        lines.extend(["", "容量摘要：", ""])
        for arm, limit in payload["capacity_limits"].items():
            lines.append(
                f"- {arm}: fully-resident offered B"
                f"{limit['maximum_fully_resident_offered_concurrency']}，"
                f"最大 observed running="
                f"{limit['maximum_observed_running_requests']}，"
                f"最大 live context="
                f"{limit['maximum_observed_live_context_tokens']} tokens。"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    payload = aggregate(args)
    output_dir = (args.output_dir or args.run_root / "analysis").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = dict(payload)
    time_rows = json_payload.pop("time_series")
    (output_dir / "e13_summary.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "e13_cases.csv", payload["case_rows"])
    write_csv(output_dir / "e13_summary.csv", payload["aggregate_rows"])
    write_csv(output_dir / "e13_global.csv", payload["global_rows"])
    write_csv(
        output_dir / "e13_capacity_cases.csv",
        payload["capacity_case_rows"],
    )
    write_csv(
        output_dir / "e13_capacity_envelope.csv",
        payload["capacity_aggregate_rows"],
    )
    write_csv(output_dir / "e13_timeseries.csv", time_rows)
    write_markdown(output_dir / "e13_summary.md", payload)
    print(json.dumps({"status": "ok", "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
