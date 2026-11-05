#!/usr/bin/env python3
"""Compare a QPS=8 rerun and audit TTFT against scheduler timestamps."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

LABELS = {"adaptive": "AdaptMoE", "native_uva": "Native UVA", "native_copy": "Native Copy"}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root: Path, label: str) -> tuple[dict, list[dict], list[dict], list[dict]]:
    base = root / "qps8"
    verification = json.loads((base / "verification.json").read_text())
    assert not verification["errors"] and len(verification["cases"]) == 6
    assert verification["protocol"]["arrival_qps"] == 8
    raw, processes, quarters = [], [], []
    for case in verification["cases"]:
        result = json.loads((base / "arms" / case["directory"] / "result.json").read_text())
        assert result["status"] == "ok" and len(result["requests"]) == 128
        steps = {step["scheduler_step"]: step for step in result["scheduler_steps"]}
        first = result["requests"][0]
        origin = (steps[first["first_scheduled_step"]]["finished_perf_counter"]
                  - first["queueing_ms"] / 1000 - first["planned_arrival_offset_seconds"])
        values = []
        for index, request in enumerate(result["requests"]):
            assert request["fixture_index"] == index
            assert abs(request["planned_arrival_offset_seconds"] - index / 8) < 1e-6
            step = steps[request["first_scheduled_step"]]
            arrival = origin + index / 8
            admitted = origin + request["admitted_offset_seconds"]
            scheduled = step["finished_perf_counter"]
            first_token = step["engine_step_finished_perf_counter"]
            # Every 512-token prompt fits in its first scheduled step in this
            # configuration. Cross-check that the first token was returned in it.
            reconstructed = (first_token - arrival) * 1000
            error = abs(reconstructed - request["ttft_ms"])
            assert error < 1e-3, (case["directory"], index, error)
            admission_ms = (admitted - arrival) * 1000
            engine_queue_ms = (scheduled - admitted) * 1000
            forward_ms = (first_token - scheduled) * 1000
            assert abs(admission_ms - request["admission_delay_ms"]) < 1e-3
            assert abs(engine_queue_ms - request["engine_queueing_ms"]) < 1e-3
            assert abs(admission_ms + engine_queue_ms + forward_ms - request["ttft_ms"]) < 1e-3
            row = {
                "run": label, "arm": case["arm"], "sample": case["sample"],
                "request_index": index, "planned_arrival_seconds": index / 8,
                "ttft_ms": request["ttft_ms"], "admission_delay_ms": admission_ms,
                "engine_queueing_ms": engine_queue_ms,
                "scheduled_to_first_token_ms": forward_ms,
                "scheduler_reconstructed_ttft_ms": reconstructed,
                "reconstruction_error_ms": error,
            }
            raw.append(row)
            values.append(row)
        processes.append({
            "run": label, "arm": case["arm"], "sample": case["sample"],
            "actual_qps": case["request_throughput_per_second"],
            **{key: statistics.fmean(row[key] for row in values) for key in (
                "ttft_ms", "admission_delay_ms", "engine_queueing_ms", "scheduled_to_first_token_ms")},
            "max_reconstruction_error_ms": max(row["reconstruction_error_ms"] for row in values),
            "max_waiting_requests": max(len(step["waiting_before"]) for step in steps.values()),
            "max_engine_step_ms": max(step["engine_step_duration_ms"] for step in steps.values()),
        })
        for start in (0, 32, 64, 96):
            quarters.append({
                "run": label, "arm": case["arm"], "sample": case["sample"],
                "first_request_index": start, "last_request_index": start + 31,
                "mean_ttft_ms": statistics.fmean(row["ttft_ms"] for row in values[start:start + 32]),
            })
    return verification, raw, processes, quarters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    all_raw, all_processes, all_quarters, runs = [], [], [], {}
    for label, root in (("previous", args.baseline), ("rerun", args.run_root)):
        verification, raw, processes, quarters = analyze(root, label)
        runs[label] = verification
        all_raw.extend(raw)
        all_processes.extend(processes)
        all_quarters.extend(quarters)
    assert runs["previous"]["protocol"] == runs["rerun"]["protocol"], "rerun configuration differs"
    assert (args.baseline / "qps8/input_sha256.txt").read_text() == (args.run_root / "qps8/input_sha256.txt").read_text(), "input/code/driver hashes differ"
    lines = ["# QPS=8 原配置复跑及 TTFT 核对", "",
             "同一份 128 条 ShareGPT trace，512 输入/16 输出，C16、KV 512 MiB、eager、无 Graph。每机制两个独立进程，ABC/CBA。与上轮 run_config 及 input_sha256 完全一致；没有额外 conditioning，没有剔除样本。", "",
             "| 机制 | 上轮实际 QPS | 本轮实际 QPS | 上轮 TTFT ms | 本轮 TTFT ms | 上轮 TPOT ms | 本轮 TPOT ms |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm, name in LABELS.items():
        a, b = runs["previous"]["aggregate"][arm], runs["rerun"]["aggregate"][arm]
        lines.append(f"| {name} | {a['request_throughput_per_second']:.4f} | {b['request_throughput_per_second']:.4f} | {a['mean_ttft_ms']:.2f} | {b['mean_ttft_ms']:.2f} | {a['mean_tpot_ms']:.2f} | {b['mean_tpot_ms']:.2f} |")
    lines += ["", "TTFT 按每个请求的首个调度 step 时间戳重新计算，并核对下式：", "",
              "TTFT = 计划到达到引擎接收的延迟 + 引擎接收到首次调度的排队 + 首次调度结束到首 token 的时间。", "",
              "第三项包含模型执行、搬运和主机工作，不能等同于纯 Prefill GPU kernel 时间。", "",
              "| 运行 | 机制 | 接收前延迟 ms | 引擎内排队 ms | 首次调度后到首 token ms | TTFT mean ms |",
              "|---|---|---:|---:|---:|---:|"]
    for label in ("previous", "rerun"):
        for arm, name in LABELS.items():
            selected = [row for row in all_raw if row["run"] == label and row["arm"] == arm]
            means = {key: statistics.fmean(row[key] for row in selected) for key in (
                "admission_delay_ms", "engine_queueing_ms", "scheduled_to_first_token_ms", "ttft_ms")}
            lines.append(f"| {label} | {name} | {means['admission_delay_ms']:.2f} | {means['engine_queueing_ms']:.2f} | {means['scheduled_to_first_token_ms']:.2f} | {means['ttft_ms']:.2f} |")
    lines += ["", "将 UVA 的请求按到达顺序分为四组，检查 TTFT 是否随队列累积，而非由单次长停顿拉高：", "",
              "| 运行 | Sample | 请求 0–31 ms | 请求 32–63 ms | 请求 64–95 ms | 请求 96–127 ms |",
              "|---|---:|---:|---:|---:|---:|"]
    for label in ("previous", "rerun"):
        for sample in (1, 2):
            selected = [row for row in all_quarters if row["run"] == label and row["arm"] == "native_uva" and row["sample"] == sample]
            lines.append(f"| {label} | {sample} | " + " | ".join(f"{row['mean_ttft_ms']:.2f}" for row in selected) + " |")
    max_error = max(row["reconstruction_error_ms"] for row in all_raw)
    lines += ["", f"新旧两轮全部 {len(all_raw)} 条请求的 TTFT 都能由 scheduler/engine step 时间戳重建，最大偏差 {max_error:.9f} ms。逐进程最长 step 与队列深度见 latency_processes.csv；完整逐请求拆分见 request_latency.csv。", "",
              "每个机制每轮只有两个独立进程，仍是有限 trace 的观测。三机制输出 hash 有漂移时，以本轮 output_drift.csv 为准，不宣称严格输出等价。", ""]
    (args.run_root / "RERUN_COMPARISON.md").write_text("\n".join(lines))
    write_csv(args.run_root / "request_latency.csv", all_raw)
    write_csv(args.run_root / "latency_processes.csv", all_processes)
    write_csv(args.run_root / "latency_quarters.csv", all_quarters)
    print("\n".join(lines[:13]))


if __name__ == "__main__":
    main()
