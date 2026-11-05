#!/usr/bin/env python3
"""Summarize the fixed historical DeepSeek configuration at timed arrival rates."""
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
labels = {"adaptive": "AdaptMoE", "native_uva": "Native UVA", "native_copy": "Native Copy"}
rows = []
warnings = []
lines = [
    "# DeepSeek 512×16：固定到达 QPS 扫描", "",
    "DeepSeek-V2-Lite-Chat BF16，TRITON_MLA，混合连续调度，eager，无 CUDA Graph。",
    "同一组 16 个冻结 ShareGPT prompt，每个输入 512 token、输出 16 token。并发上限 16，固定 KV 512 MiB，调度 token budget 8192。",
    "PARK≤16、RESUME≥128，关闭全量驻留诊断和 runtime read gate，固定原专家卸载方案。",
    "AdaptMoE 4,948,230,144 B (4.6084 GiB)，Native 两臂同一组 4,798,283,776 B (4.46875 GiB)，相差 3.13%。",
    "每个 QPS / 机制两个独立进程，ABC/CBA 顺序；下表为进程均值。", "",
    "正常 engine 初始化后直接计时真实请求，没有额外 conditioning；保留首轮真实请求开销，不能称为充分预热后的稳态数据。", "",
    "原来的 8+8 按 step 到达改为相同请求按 offset=i/QPS 到达。TTFT/E2E 包含计划到达到引擎实际接收的延迟。",
    "实际 QPS = 16 / 从首个计划到达到末请求完成的秒数，包含尾部排空；只有 16 条请求，不是长时稳态容量测试。", "",
    "| 提供 QPS | 机制 | 实际 QPS | TTFT mean ms | TPOT mean ms | E2E mean ms |",
    "|---:|---|---:|---:|---:|---:|",
]
reference_protocol = None
for rate in (1, 4, 8):
    verification = json.loads((root / f"qps{rate}" / "verification.json").read_text())
    if verification["errors"]:
        raise SystemExit(f"qps{rate}: {verification['errors']}")
    warnings.extend(f"QPS={rate}: {warning}" for warning in verification["warnings"])
    protocol = verification["protocol"]
    invariant = {key: protocol[key] for key in ("fixture_sha256", "prompt_token_hashes", "requests", "kv_cache_memory_bytes", "max_num_seqs", "max_num_batched_tokens", "adaptive_selected_bytes", "native_expected_bytes")}
    if reference_protocol is not None and invariant != reference_protocol:
        raise SystemExit("cross-rate protocol mismatch")
    reference_protocol = invariant
    for arm, label in labels.items():
        item = verification["aggregate"][arm]
        cases = [case for case in verification["cases"] if case["arm"] == arm]
        row = {"offered_qps": rate, "arm": arm, **item}
        rows.append(row)
        lines.append(f"| {rate} | {label} | {item['request_throughput_per_second']:.4f} | {item['mean_ttft_ms']:.2f} | {item['mean_tpot_ms']:.2f} | {item['mean_e2e_ms']:.2f} |")

lines += ["", "## 相对吞吐", "", "| 提供 QPS | AdaptMoE 相对 UVA | AdaptMoE 相对 Copy |", "|---:|---:|---:|"]
for rate in (1, 4, 8):
    group = {row['arm']: row for row in rows if row['offered_qps'] == rate}
    adaptive = group['adaptive']['request_throughput_per_second']
    lines.append(f"| {rate} | {(adaptive / group['native_uva']['request_throughput_per_second'] - 1) * 100:+.2f}% | {(adaptive / group['native_copy']['request_throughput_per_second'] - 1) * 100:+.2f}% |")
lines += ["", "## 每个进程与实际调度", "", "| QPS | 机制 | Sample | 实际 QPS | TTFT ms | TPOT ms | mixed steps | PARK/RESUME | H2D GiB | 接收延迟均值 ms |", "|---:|---|---:|---:|---:|---:|---:|---|---:|---:|"]
for rate in (1, 4, 8):
    verification = json.loads((root / f"qps{rate}" / "verification.json").read_text())
    for case in verification["cases"]:
        lines.append(f"| {rate} | {labels[case['arm']]} | {case['sample']} | {case['request_throughput_per_second']:.4f} | {case['mean_ttft_ms']:.2f} | {case['mean_tpot_ms']:.2f} | {case['mixed_steps']} | {len(case['park_wall_ms'])}/{len(case['resume_wall_ms'])} | {case['measured_copy_bytes'] / 2**30:.2f} | {case['mean_admission_delay_ms']:.2f} |")
lines += [
    "", "## 首轮长 step 与协议限制", "",
    "QPS=1 的 AdaptMoE sample 1，第一个 Decode step 为 5956.79 ms，对应 PARK 仅 11.08 ms；sample 2 没有复现。两次 TTFT mean 为 1640.74 / 177.73 ms，汇总均值保留这次长停顿。具体根因未最终定位，不能归因于约 6 秒的页表切换，也不能将汇总均值解释为稳态延迟。",
    "原历史实验先同时加入 8 条、在 engine step 4 再加入 8 条；本次逐条定时到达，Prefill 合批及切换频率自然会变化。即使 prompt/KV/专家方案相同，也不应直接拿实际 QPS 与历史 8+8 批量注入结果比较。",
    "", "## 输出与运行校验", "",
    "18/18 进程成功，CUDA Graph 关闭、卸载量及输入一致性检查通过，无内核 GPU fault；最终已恢复 stock UVM。",
    *([f"- {warning}" for warning in warnings] if warnings else ["所有 rate 的逐请求跨机制输出 hash 一致。"]), "",
    "存在输出 hash 告警时，仅报告性能观测，不宣称严格输出等价；不同机制也会形成不同调度合批。",
    "执行中的监控句柄及清理复查记录见 [EXECUTION_NOTES.md](EXECUTION_NOTES.md)。",
    "两个独立进程不足以给出严格统计置信结论。各 rate 的逐进程配置、时间戳、调度记录、kernel 日志和验证结果保存在 qps1/qps4/qps8 目录。", "",
]
(root / "REPORT.md").write_text("\n".join(lines))
with (root / "summary.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(root / "summary.json").write_text(json.dumps({"rows": rows, "warnings": warnings}, indent=2) + "\n")
print("\n".join(lines[:26]))
