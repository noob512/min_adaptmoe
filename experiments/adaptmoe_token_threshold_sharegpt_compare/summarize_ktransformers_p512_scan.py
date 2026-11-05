#!/usr/bin/env python3
"""Verify KTransformers replay and join immutable historical baselines."""
import argparse
import csv
import hashlib
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ktransformers_p512_runner import distribution, prompt_hash, token_hash, write_json

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "inputs/sharegpt_deepseek_p512_n128_20261023.json"
BASELINE = HERE / "P512_N128_QPS4_8_10_20_30_SUMMARY_20261023.json"


def verify_case(directory: Path) -> dict:
    result = json.loads((directory / "result.json").read_text())
    fixture = json.loads(FIXTURE.read_text())
    assert result["status"] == "ok", result.get("error")
    assert (directory / "returncode.txt").read_text().strip() == "0"
    config = result["configuration"]
    assert config["fixture_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    measurement = result["measurement"]
    assert measurement["successful_requests"] == 128 and measurement["failed_requests"] == 0
    assert result["server_info_before"]["max_total_num_tokens"] == 17248
    rate = config["arrival_qps"]
    requests = measurement["requests"]
    assert len(requests) == 128
    for i, (record, source) in enumerate(zip(requests, fixture["requests"])):
        assert record["fixture_index"] == i and record["success"]
        assert record["prompt_token_ids_sha256"] == prompt_hash(source["prompt_token_ids"])
        assert len(record["output_token_ids"]) == record["completion_tokens"] == 16
        assert record["output_token_ids_sha256"] == token_hash(record["output_token_ids"])
        assert record["planned_arrival_offset_seconds"] == i / rate
        times = record["token_observed_offsets_seconds"]
        assert len(times) == 16 and times == sorted(times)
        assert abs(record["ttft_ms"] - (times[0] - i / rate) * 1000) < 1e-6
        assert abs(record["tpot_ms"] - (times[-1] - times[0]) * 1000 / 15) < 1e-6
    elapsed = max(r["stream_finished_offset_seconds"] for r in requests)
    assert abs(measurement["actual_qps"] - 128 / elapsed) < 1e-9
    for metric in ("ttft", "tpot", "e2e"):
        assert abs(measurement[f"{metric}_ms"]["mean"] - statistics.fmean(r[f"{metric}_ms"] for r in requests)) < 1e-6
    log = (directory / "run.log").read_text()
    placement = re.findall(r"KT GPU experts: layer (\d+) \(MoE\) has (\d+) GPU experts", log)
    assert sorted((int(a), int(b)) for a, b in placement) == [(i, 53) for i in range(1, 27)], placement
    assert "TRACE_END" in log
    kernel = (directory / "kernel.log").read_text()
    # Journal --since accepts local time unless a timezone is explicit. Bound
    # every line by the actual run's UTC window, including older driver snapshots.
    lower = datetime.fromisoformat(result["started_at"]).timestamp() - 1
    upper = datetime.fromisoformat(result["finished_at"]).timestamp() + 60
    year = datetime.fromisoformat(result["started_at"]).year
    run_lines = []
    for line in kernel.splitlines():
        try:
            stamp = datetime.strptime(f"{year} {line[:15]}", "%Y %b %d %H:%M:%S").replace(tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
        except ValueError:
            continue
        if lower <= stamp <= upper:
            run_lines.append(line)
    fatal = [line for line in run_lines if re.search(
        r"NVRM: Xid|\bBUG:|\bOops:|GPU has fallen off|out of memory: Killed process", line, re.I)]
    assert not fatal, fatal[:10]
    (directory / "kernel.run_window.log").write_text("\n".join(run_lines) + "\n")
    return {"status": "ok", "directory": str(directory), "offered_qps": rate,
            "actual_qps": measurement["actual_qps"],
            "ttft_mean_ms": measurement["ttft_ms"]["mean"],
            "tpot_mean_ms": measurement["tpot_ms"]["mean"],
            "e2e_mean_ms": measurement["e2e_ms"]["mean"],
            "elapsed_seconds": elapsed, "successful_requests": 128,
            "grouped_stream_events": measurement["grouped_stream_events"],
            "max_logged_running_requests": max(map(int, re.findall(r"#running-req: (\d+)", log)), default=0),
            "max_logged_queued_requests": max(map(int, re.findall(r"#queue-req: (\d+)", log)), default=0),
            "prefill_batch_log_count": len(re.findall(r"Prefill batch,", log)),
            "decode_batch_log_count": len(re.findall(r"Decode batch,", log)),
            "cpu_backend_avx512_bf16": "backend=AVX512-BF16" in log,
            "missing_tuned_gpu_moe_config": "Using default MoE kernel config" in log}


def baseline_dirs(row: dict) -> list[Path]:
    root = Path(row["source_report"]).parent / f"qps{row['offered_qps']}" / "arms"
    return sorted(p.parent for p in root.glob(f"*_{row['arm']}/result.json"))


def summarize(root: Path) -> None:
    baseline = json.loads(BASELINE.read_text())["rows"]
    cases = []
    rows = []
    drift = []
    names = {"adaptive": "AdaptMoE", "native_uva": "Native UVA", "native_copy": "Native Copy",
             "ktransformers": "KTransformers"}
    for rate in (4, 8, 10, 20, 30):
        dirs = [root / f"qps{rate}/sample{sample}" for sample in (1, 2)]
        verified = [verify_case(p) for p in dirs]
        cases.extend(verified)
        results = [json.loads((p / "result.json").read_text()) for p in dirs]
        measurements = [r["measurement"] for r in results]
        row = {"offered_qps": rate, "arm": "ktransformers", "samples": 2,
               "actual_qps": statistics.fmean(m["actual_qps"] for m in measurements),
               **{f"{metric}_mean_ms": statistics.fmean(m[f"{metric}_ms"]["mean"] for m in measurements)
                  for metric in ("ttft", "tpot", "e2e")},
               "ttft_p95_ms": distribution([r["ttft_ms"] for m in measurements for r in m["requests"]])["p95"],
               "source_results": [str(p / "result.json") for p in dirs]}
        rows.extend([r for r in baseline if r["offered_qps"] == rate] + [row])
        kt_hashes = [[r["output_token_ids_sha256"] for r in m["requests"]] for m in measurements]
        drift.append({"qps": rate, "comparison": "KT sample1 vs sample2",
                      "different_full_outputs": sum(a != b for a, b in zip(*kt_hashes)), "requests": 128})
        for old in (r for r in baseline if r["offered_qps"] == rate):
            refs = baseline_dirs(old)
            assert len(refs) == 2, refs
            for i, ref in enumerate(refs):
                native = json.loads((ref / "result.json").read_text())
                reference = sorted(native["requests"], key=lambda r: r["fixture_index"])
                assert [r["prompt_token_ids_sha256"] for r in reference] == [r["prompt_token_ids_sha256"] for r in measurements[i]["requests"]]
                drift.append({"qps": rate, "comparison": f"KT vs {old['arm']} sample{i + 1}",
                              "reference": str(ref / "result.json"), "requests": 128,
                              "different_full_outputs": sum(a != b["output_token_ids_sha256"] for a, b in zip(kt_hashes[i], reference)),
                              "different_first_tokens": sum(a["output_token_ids"][0] != b["output_token_ids"][0]
                                                            for a, b in zip(measurements[i]["requests"], reference))})
    verification = {"status": "ok_with_output_drift" if any(d["different_full_outputs"] for d in drift) else "ok",
                    "errors": [], "cases": cases, "output_comparisons": drift}
    resources = {}
    cpu_sample = root / "qps10/sample1/cpu.sample_1hz.json"
    gpu_sample = root / "qps10/sample1/gpu.sample_1hz.csv"
    if cpu_sample.exists() and gpu_sample.exists():
        cpu_values = json.loads(cpu_sample.read_text())
        with gpu_sample.open() as handle:
            gpu_values = list(csv.DictReader(handle))
        utilization = [float(next(v for k, v in r.items() if "utilization.gpu" in k).strip().split()[0]) for r in gpu_values]
        resources = {"source_case": "qps10/sample1", "duration_seconds": len(cpu_values),
                     "scheduler_process_cpu_mean_percent": statistics.fmean(r["process_cpu_percent"] for r in cpu_values),
                     "gpu_utilization_mean_percent": statistics.fmean(utilization),
                     "gpu_utilization_min_percent": min(utilization), "gpu_utilization_max_percent": max(utilization)}
    write_json(root / "verification.json", verification)
    write_json(root / "summary.json", {"rows": rows, "raw_ktransformers_cases": cases,
                                       "historical_baseline": str(BASELINE), "output_comparisons": drift,
                                       "resource_sample": resources})
    fields = ["offered_qps", "arm", "samples", "actual_qps", "ttft_mean_ms", "tpot_mean_ms", "e2e_mean_ms"]
    with (root / "summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    text = ["# 相同 P512/N128 trace：补测 KTransformers", "",
            "五档 QPS 各两个独立进程，本次共10个进程；历史三臂使用冻结的最新三机制汇总，其中QPS=8使用最近复跑结果。每次128/128请求完成、每条16输出 token，输入 hash 与原三臂一致。", "",
            "DeepSeek-V2-Lite-Chat BF16、512输入/16输出、C16、8192-token混合调度预算、上下文576、无CUDA Graph、无额外conditioning。KV对齐17,248-token有效容量。", "",
            f"KTransformers KV tensor = {17249 * 31104:,} B；原三臂设置512 MiB预算。每层53个GPU专家/11个CPU专家，26个MoE层，共286个cold专家，4,948,230,144 B，坐标与AdaptMoE完全一致；Native UVA/Copy原样使用4,798,283,776 B。", "",
            "KTransformers：14 CPU线程、1线程池、NUMA0、BF16、gpu_prefill_token_threshold=0、deferred experts=0、动态专家更新关闭。SGLang triton attention、mixed chunk开启、prefix cache关闭。", "",
            "| 提供QPS | 机制 | 实际QPS | TTFT mean s | TPOT mean ms | E2E mean s |",
            "|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        text.append(f"| {r['offered_qps']} | {names[r['arm']]} | {r['actual_qps']:.3f} | {r['ttft_mean_ms']/1000:.3f} | {r['tpot_mean_ms']:.2f} | {r['e2e_mean_ms']/1000:.3f} |")
    text += ["", "## KTransformers 原始样本", "", "| QPS | 进程 | 实际QPS | TTFT s | TPOT ms | grouped events |", "|---:|---|---:|---:|---:|---:|"]
    for c in cases:
        text.append(f"| {int(c['offered_qps'])} | {Path(c['directory']).name} | {c['actual_qps']:.6f} | {c['ttft_mean_ms']/1000:.6f} | {c['tpot_mean_ms']:.6f} | {c['grouped_stream_events']} |")
    if resources:
        text += ["", f"QPS=10/sample1 运行中另有{resources['duration_seconds']}秒、1 Hz资源采样：scheduler进程CPU平均{resources['scheduler_process_cpu_mean_percent']:.1f}%（约{resources['scheduler_process_cpu_mean_percent']/100:.2f}核），GPU利用率平均{resources['gpu_utilization_mean_percent']:.2f}%、范围{resources['gpu_utilization_min_percent']:.0f}%–{resources['gpu_utilization_max_percent']:.0f}%。与饱和后吞吐不再随提供QPS上升的现象一致，支持CPU侧受限；这不是逐kernel剖析，未进一步分离CPU专家计算、内存带宽、线程调度和忙等成本。"]
    text += ["", "## 输出检查与口径限制", "",
             "实际QPS=128/首个计划到达到最后完成时间，含尾部排空；TTFT从计划到达开始含排队，TPOT为首末token间隔/15。客户端没有并发限流，由后端C16限制运行请求。正常初始化后直接计时，与原三臂一样没有额外conditioning。", "",
             "KTransformers使用SGLang的异步直接Engine及多进程IPC/overlap scheduler，原三臂使用vLLM嵌入式Engine。两个框架的attention/MoE算子和调度实现不同，本表是系统配置对比，不能将差异单独归因为专家访问方式。", "",
             "本机14个可用物理CPU核、无AMX标志，实际CPU专家后端为AVX512-BF16；日志提示53 GPU专家的MoE使用默认kernel配置，未进行额外调优。因此这些数值仅代表当前机器与当前KTransformers配置。首次缺失CUDA扩展的构建检查保存在attempt-03/QUALIFICATION.md；正式矩阵在软件构建就绪后重新启动全部五档，未删除正式样本。", "",
             "完整输出hash存在跨机制或跨进程漂移时，不宣称严格数值等价；具体数量如下，原始tokens保存在result.json。两进程均值不构成严格统计置信或无限时长容量结论。", "",
             "| QPS | 比较 | 完整输出不同/128 |", "|---:|---|---:|"]
    text += [f"| {d['qps']} | {d['comparison']} | {d['different_full_outputs']} |" for d in drift]
    text += ["", f"原始artifact：[本轮目录]({root})；[历史三机制汇总]({BASELINE.with_suffix('.md')})。协议、实际配置、源码快照、placement、日志、kernel记录、逐请求时刻和输出均保存在该artifact目录。", ""]
    (root / "REPORT.md").write_text("\n".join(text))
    print(json.dumps({"status": verification["status"], "cases": len(cases),
                      "report": str(root / "REPORT.md"), "ktransformers": [r for r in rows if r["arm"] == "ktransformers"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--verify-case", type=Path)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.verify_case:
        case = verify_case(args.verify_case)
        write_json(args.verify_case / "verification.json", case)
        print(json.dumps(case))
    else:
        summarize(args.root)
