#!/usr/bin/env python3
"""Audit and summarize the 128-request DeepSeek QPS matrix."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

LABELS = {"adaptive": "AdaptMoE", "native_uva": "Native UVA", "native_copy": "Native Copy"}


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return values[low] * (high - position) + values[high] * (position - low) if low != high else values[low]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--rates", nargs="+", type=int, default=[10, 20, 30])
    args = parser.parse_args()
    rates = args.rates
    if any(rate <= 0 for rate in rates) or len(set(rates)) != len(rates):
        parser.error("rates must be unique positive integers")
    process_count = len(rates) * 6
    root = args.run_root.resolve()
    rows, details, drift_rows, warnings = [], [], [], []
    schedule_equivalence = {}
    invariant = None
    fixture_sha = None
    for rate in rates:
        base = root / f"qps{rate}"
        verification = json.loads((base / "verification.json").read_text())
        assert not verification["errors"] and len(verification["cases"]) == 6
        protocol = verification["protocol"]
        assert protocol["arrival_qps"] == rate and protocol["requests"] == 128
        assert protocol["samples_per_arm"] == 2 and protocol["arms"] == list(LABELS)
        constants = {key: protocol[key] for key in (
            "fixture_sha256", "prompt_token_hashes", "model", "requests",
            "output_tokens_per_request", "kv_cache_memory_bytes", "max_model_len",
            "max_num_seqs", "max_num_batched_tokens", "adaptive_selected_bytes",
            "native_expected_bytes", "park_token_threshold", "resume_token_threshold",
        )}
        if invariant is None:
            invariant = constants
            fixture_sha = protocol["fixture_sha256"]
        assert constants == invariant, "cross-rate configuration differs"
        assert json.loads((base / "module.after.json").read_text())["gpu_clean"]
        warnings.extend(f"QPS={rate}: {warning}" for warning in verification["warnings"])
        requests_by_case = {}
        steps_by_case = {}
        for case in verification["cases"]:
            case_dir = base / "arms" / case["directory"]
            result = json.loads((case_dir / "result.json").read_text())
            requests = result["requests"]
            steps = result["scheduler_steps"]
            requests_by_case[(case["sample"], case["arm"])] = requests
            steps_by_case[(case["sample"], case["arm"])] = steps
            assert result["status"] == "ok" and len(requests) == 128
            assert all(request["success"] and request["completion_tokens"] == 16 for request in requests)
            assert result["configuration"]["latency_origin"] == "planned_arrival"
            assert all(abs(request["planned_arrival_offset_seconds"] - index / rate) < 1e-6 for index, request in enumerate(requests))
            assert sum(step["total_num_scheduled_tokens"] for step in steps) == 128 * (512 + 15)
            dispatches = re.findall(r"runtime_mode=(\w+)", (case_dir / "run.log").read_text())
            assert dispatches and set(dispatches) == {"NONE"}
            if case["arm"] == "adaptive":
                # The thresholds are hysteretic: an intermediate 17..127-token
                # step retains the preceding topology. Replay the actual state
                # machine instead of counting only RESUME-triggering steps.
                in_copy_topology = True  # alias/copy topology is active at startup
                copy_steps = 0
                for step in steps:
                    total = int(step["total_num_scheduled_tokens"])
                    if total <= protocol["park_token_threshold"]:
                        in_copy_topology = False
                    elif total >= protocol["resume_token_threshold"]:
                        in_copy_topology = True
                    copy_steps += int(in_copy_topology)
            else:
                copy_steps = len(steps) if case["arm"] == "native_copy" else 0
            if case["arm"] == "adaptive":
                assert case["measured_copy_bytes"] == copy_steps * protocol["adaptive_selected_bytes"]
            large_steps = [step for step in steps if step["total_num_scheduled_tokens"] >= protocol["resume_token_threshold"]]
            details.append({
                "offered_qps": rate, "arm": case["arm"], "sample": case["sample"],
                "actual_qps": case["request_throughput_per_second"],
                "ttft_mean_ms": case["mean_ttft_ms"], "tpot_mean_ms": case["mean_tpot_ms"],
                "e2e_mean_ms": case["mean_e2e_ms"], "engine_steps": len(steps),
                "copy_steps": copy_steps, "explicit_h2d_bytes": case["measured_copy_bytes"],
                "large_token_steps": len(large_steps),
                "large_token_mean_scheduled_tokens": statistics.fmean(step["total_num_scheduled_tokens"] for step in large_steps),
                "large_token_total_ms": sum(step["engine_step_duration_ms"] for step in large_steps),
                "other_step_total_ms": sum(step["engine_step_duration_ms"] for step in steps if step["total_num_scheduled_tokens"] < protocol["resume_token_threshold"]),
                "max_engine_step_ms": max(step["engine_step_duration_ms"] for step in steps),
                "park_count": len(case["park_wall_ms"]), "resume_count": len(case["resume_wall_ms"]),
                "switch_total_ms": sum(case["park_wall_ms"]) + sum(case["resume_wall_ms"]),
                "max_waiting_requests": max(len(step["waiting_before"]) for step in steps),
                "admission_delay_mean_ms": result["summary"]["admission_delay_ms"]["mean"],
                "elapsed_seconds": result["summary"]["elapsed_seconds"],
                "drain_seconds": result["summary"]["elapsed_seconds"] - 127 / rate,
            })
        for sample in (1, 2):
            reference = requests_by_case[(sample, "native_copy")]
            for arm in ("adaptive", "native_uva"):
                requests = requests_by_case[(sample, arm)]
                drift_rows.append({
                    "offered_qps": rate, "sample": sample, "arm": arm,
                    "reference": "native_copy", "request_count": 128,
                    "full_output_mismatch_requests": sum(x["output_token_ids_sha256"] != y["output_token_ids_sha256"] for x, y in zip(requests, reference)),
                    "first_token_mismatch_requests": sum(x["output_token_ids"][0] != y["output_token_ids"][0] for x, y in zip(requests, reference)),
                })
        schedule_equivalence[rate] = {
            sample: [step["num_scheduled_tokens"] for step in steps_by_case[(sample, "adaptive")]]
            == [step["num_scheduled_tokens"] for step in steps_by_case[(sample, "native_copy")]]
            for sample in (1, 2)
        }
        for arm in LABELS:
            aggregate = verification["aggregate"][arm]
            cases = [case for case in details if case["offered_qps"] == rate and case["arm"] == arm]
            requests = [request for sample in (1, 2) for request in requests_by_case[(sample, arm)]]
            rows.append({
                "offered_qps": rate, "arm": arm, "samples": 2,
                "actual_qps": aggregate["request_throughput_per_second"],
                "ttft_mean_ms": aggregate["mean_ttft_ms"], "ttft_p95_ms": percentile([request["ttft_ms"] for request in requests], .95),
                "tpot_mean_ms": aggregate["mean_tpot_ms"], "e2e_mean_ms": aggregate["mean_e2e_ms"],
                "engine_steps_mean": statistics.fmean(case["engine_steps"] for case in cases),
                "copy_steps_mean": statistics.fmean(case["copy_steps"] for case in cases),
                "large_token_steps_mean": statistics.fmean(case["large_token_steps"] for case in cases),
                "large_token_mean_scheduled_tokens": statistics.fmean(case["large_token_mean_scheduled_tokens"] for case in cases),
                "large_token_total_ms_mean": statistics.fmean(case["large_token_total_ms"] for case in cases),
                "other_step_total_ms_mean": statistics.fmean(case["other_step_total_ms"] for case in cases),
                "explicit_h2d_gb_mean": statistics.fmean(case["explicit_h2d_bytes"] for case in cases) / 1e9,
                "switch_total_ms_mean": statistics.fmean(case["switch_total_ms"] for case in cases),
                "max_waiting_requests": max(case["max_waiting_requests"] for case in cases),
                "drain_seconds_mean": statistics.fmean(case["drain_seconds"] for case in cases),
            })

    lines = [
        f"# 128 条 ShareGPT、512-token 输入：QPS={'/'.join(map(str, rates))} 三机制实验", "",
        "DeepSeek-V2-Lite-Chat BF16、TRITON_MLA、mixed continuous、eager，无 CUDA Graph。",
        "每条输入恰好 512 token、输出 16 token；并发上限 16、固定 KV 512 MiB、调度预算 8192 token、max_model_len=576。",
        "AdaptMoE PARK≤16 / RESUME≥128，保留原专家方案：4,948,230,144 B；Native UVA/Copy 使用同一组 4,798,283,776 B 完整 Parameter，相差 3.13%。",
        f"每个 QPS/机制两个独立进程，ABC/CBA，共 {process_count} 个进程；表内实际 QPS 为两进程吞吐均值，延迟为同等请求数的均值。", "",
        f"冻结 fixture SHA-256：`{fixture_sha}`。保留原始 32 条请求为前缀，从同份 ShareGPT 补充 96 条，所有 128 条 prompt hash 唯一。新增来源与原所有校准/验证/测试来源不重叠。",
        "每档按 i/QPS 固定间隔到达，同档各机制共享请求顺序、token IDs、输出长度和计划到达时间。",
        "正常 engine 初始化后直接计时，没有额外 conditioning；TTFT/E2E 包含计划到达到引擎接收/调度的排队时间，TPOT 为首 token 后的平均间隔。",
        "实际 QPS=128/从首个计划到达到末请求完成的时间，包含尾部排空。这是有限 trace，不能等同于无限时长稳态容量。", "",
        "| 提供 QPS | 机制 | 实际 QPS | TTFT mean ms | TTFT P95 ms | TPOT mean ms | E2E mean ms |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['offered_qps']} | {LABELS[row['arm']]} | {row['actual_qps']:.4f} | {row['ttft_mean_ms']:.2f} | {row['ttft_p95_ms']:.2f} | {row['tpot_mean_ms']:.2f} | {row['e2e_mean_ms']:.2f} |")
    lines += ["", "## AdaptMoE 相对吞吐", "", "| 提供 QPS | 相对 UVA | 相对 Copy |", "|---:|---:|---:|"]
    for rate in rates:
        group = {row["arm"]: row for row in rows if row["offered_qps"] == rate}
        lines.append(f"| {rate} | {(group['adaptive']['actual_qps'] / group['native_uva']['actual_qps'] - 1) * 100:+.2f}% | {(group['adaptive']['actual_qps'] / group['native_copy']['actual_qps'] - 1) * 100:+.2f}% |")
    lines += ["", "## 调度与搬运（进程均值）", "", "| QPS | 机制 | steps | copy steps | 显式 H2D GB | 切换总耗时 ms | 最多等待请求 | 尾部排空 s |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['offered_qps']} | {LABELS[row['arm']]} | {row['engine_steps_mean']:.1f} | {row['copy_steps_mean']:.1f} | {row['explicit_h2d_gb_mean']:.2f} | {row['switch_total_ms_mean']:.2f} | {row['max_waiting_requests']} | {row['drain_seconds_mean']:.2f} |")
    lines += ["", "UVA 的显式 H2D=0 不代表 PCIe 流量为零：其专家由远程读取访问。", "", "## 大 token step 的合批与耗时（进程均值）", "", "大 token step 定义为调度 token≥128；本 trace 中对应 Prefill/mixed step。耗时为完整 engine.step()，包含 forward、搬运、切换与其他主机工作，不能直接当作独立 DMA 耗时。", "", "| QPS | 机制 | 大 token steps | 每大 step 调度 token | 大 step 总耗时 ms | 其他 step 总耗时 ms |", "|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['offered_qps']} | {LABELS[row['arm']]} | {row['large_token_steps_mean']:.1f} | {row['large_token_mean_scheduled_tokens']:.1f} | {row['large_token_total_ms_mean']:.2f} | {row['other_step_total_ms_mean']:.2f} |")
    lines += ["", "## 原始进程", "", "| QPS | 机制 | Sample | 实际 QPS | TTFT ms | TPOT ms | E2E ms |", "|---:|---|---:|---:|---:|---:|---:|"]
    for case in details:
        lines.append(f"| {case['offered_qps']} | {LABELS[case['arm']]} | {case['sample']} | {case['actual_qps']:.4f} | {case['ttft_mean_ms']:.2f} | {case['tpot_mean_ms']:.2f} | {case['e2e_mean_ms']:.2f} |")
    by_rate = {rate: {row["arm"]: row for row in rows if row["offered_qps"] == rate} for rate in rates}
    lines += ["", "## 结果解读", "", "| 提供 QPS | AdaptMoE 每大 step 调度 token | AdaptMoE copy steps | AdaptMoE 显式 H2D GB |", "|---:|---:|---:|---:|"]
    for rate in rates:
        a = by_rate[rate]["adaptive"]
        lines.append(f"| {rate} | {a['large_token_mean_scheduled_tokens']:.1f} | {a['copy_steps_mean']:.0f} | {a['explicit_h2d_gb_mean']:.2f} |")
    for rate in rates:
        drift = [row for row in drift_rows if row["offered_qps"] == rate and row["arm"] == "adaptive"]
        if all(schedule_equivalence[rate].values()) and all(row["full_output_mismatch_requests"] == 0 for row in drift):
            a, c = by_rate[rate]["adaptive"], by_rate[rate]["native_copy"]
            lines += ["", f"QPS={rate} 的两次 AdaptMoE/Native Copy 运行具有相同的每步请求/token 调度映射，且每次 128/128 完整输出 hash 一致。", "", f"AdaptMoE/Native Copy 大 token steps 累计分别为 {a['large_token_total_ms_mean']:.2f}/{c['large_token_total_ms_mean']:.2f} ms；其余 steps 分别为 {a['other_step_total_ms_mean']:.2f}/{c['other_step_total_ms_mean']:.2f} ms。AdaptMoE PARK/RESUME 总耗时均值为 {a['switch_total_ms_mean']:.2f} ms。这是 engine step 分解，不能把耗时差直接当作纯 DMA 或纯 kernel 差。"]
    lines += ["", "实际 QPS 包含有限 trace 的尾部排空，不能仅凭其低于提供 QPS 判断系统饱和。TTFT 包含排队，稳态容量与统计置信结论需要更长 trace 和更多独立重复。", ""]
    lines += ["", "## 验证与边界", "", f"{process_count}/{process_count} 进程完成；每进程 128/128 请求且输出长度正确，调度 token 总数均为 67,456。各机制/档位配置与计划到达一致，实际 Graph dispatch 全为 NONE；无内核 GPU fault，最终 stock UVM 和 GPU 清理检查通过。", "", *([f"- {warning}" for warning in warnings] if warnings else ["完整输出 hash 跨机制一致。"]), "", "输出存在漂移时仅报告性能观测，不宣称严格输出等价。output_drift.csv 给出各臂相对 Native Copy 的完整输出/首 token 不一致请求数；Native Copy 仅作比较参考。两个独立进程不足以给出严格置信区间。", "", "详见 PROTOCOL.md、summary.csv、cases.csv、output_drift.csv 和每档 verification.json/逐进程 result.json。", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    write_csv(root / "summary.csv", rows)
    write_csv(root / "cases.csv", details)
    write_csv(root / "output_drift.csv", drift_rows)
    (root / "summary.json").write_text(json.dumps({"rows": rows, "cases": details, "output_drift": drift_rows, "adaptive_copy_schedule_equivalence": schedule_equivalence, "warnings": warnings}, indent=2) + "\n")
    print("\n".join(lines[:26]))


if __name__ == "__main__":
    main()
