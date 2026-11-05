#!/usr/bin/env python3
"""Summarize the frozen Qwen3 six-mechanism Prefill experiment."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ARTIFACTS = HERE / "artifacts"
CUSTOM_ROOT = ARTIFACTS / "custom_formal_20260821_v1"
VLLM_ROOT = ARTIFACTS / "vllm_formal_20260821_v1"
KT_ROOT = ARTIFACTS / "kt_eager_formal_20260821_v1"
INPUT_MANIFEST = (
    REPO / "experiments/qwen_decode_six_mechanism_compare/inputs/manifest.json"
)
GRAPH_QUALIFICATIONS = {
    "2g": [
        ARTIFACTS / "kt_graph_2g_qual_v1/2g_graph/server.log",
        ARTIFACTS / "kt_graph_2g_qual_no_overlap_v2/2g_graph/server.log",
    ],
    "4g": [ARTIFACTS / "kt_graph_4g_qual_v1/4g_graph/server.log"],
}
OUTPUT_JSON = HERE / "RESULTS_20260821.json"
OUTPUT_MD = HERE / "RESULTS_20260821.md"


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO.resolve()))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measured_cycles(result: dict[str, Any], batch: int) -> list[dict[str, Any]]:
    cycles = [
        item
        for item in result["cycles"]
        if item["role"] == "measurement" and item["batch_size"] == batch
    ]
    if len(cycles) != 3:
        raise AssertionError(f"expected three samples for batch {batch}")
    return cycles


def batch_ttft_median(result: dict[str, Any], batch: int) -> float:
    samples = [float(item["request_ttft_ms"]["max"]) for item in measured_cycles(result, batch)]
    return statistics.median(samples)


def vllm_record(
    tier: str,
    mechanism: str,
    path: Path,
    *,
    graph: bool,
    placement: dict[str, Any],
) -> dict[str, Any]:
    result = load(path)
    if result["status"] != "ok":
        raise AssertionError(f"failed result: {path}")
    batches: dict[str, Any] = {}
    for batch in (1, 4):
        phase = result["phase_summary"][str(batch)]
        batches[str(batch)] = {
            "prompt_tokens": int(phase["prompt_tokens"]),
            "prefill_step_ms": float(phase["prefill_step_ms"]["median"]),
            "prefill_step_samples_ms": [
                float(value) for value in phase["prefill_step_ms"]["samples"]
            ],
            "prefill_tokens_per_second": float(phase["prefill_tokens_per_second"]),
            "batch_ttft_ms": batch_ttft_median(result, batch),
        }
    record = {
        "tier": tier,
        "mechanism": mechanism,
        "status": "ok",
        "timing_boundary": "synchronized_vllm_engine_core_step",
        "cuda_graph": "verified_full_prefill_replay" if graph else "disabled_eager",
        "batches": batches,
        "placement": placement,
        "result_path": rel(path),
        "result_sha256": sha256(path),
    }
    if graph:
        diagnostics = result["static_alias_diagnostics_after_load"]
        prefill_replays = [
            int(item["prefill_graph_wrapper_delta"].get("full_replay", 0))
            for item in result["cycles"]
        ]
        certificate = diagnostics["remote_graph_certificate"]
        if not prefill_replays or set(prefill_replays) != {1}:
            raise AssertionError(f"not every Prefill was one FULL replay: {path}")
        if int(certificate["actual_h2d_bytes"]) != 0:
            raise AssertionError(f"remote Graph captured H2D bytes: {path}")
        record["graph_evidence"] = {
            "realized_cudagraph_mode": result["config"]["realized_cudagraph_mode"],
            "prefill_cycle_count": len(prefill_replays),
            "full_replays_per_prefill": sorted(set(prefill_replays)),
            "capture_actual_h2d_bytes": int(certificate["actual_h2d_bytes"]),
            "remote_map_requested_bytes": int(certificate["remote_map_requested_bytes"]),
            "remote_map_active": bool(diagnostics["remote_map_active"]),
        }
    return record


def kt_eager_record(tier: str, path: Path, summary_path: Path) -> dict[str, Any]:
    result = load(path)
    run_summary = load(summary_path)
    if result["status"] != "ok":
        raise AssertionError(f"failed result: {path}")
    batches: dict[str, Any] = {}
    for batch in (1, 4):
        phase = result["phase_summary"][str(batch)]
        tokens = int(phase["prompt_tokens"])
        ttft = float(phase["client_ttft_batch_ms"]["median"])
        batches[str(batch)] = {
            "prompt_tokens": tokens,
            "client_ttft_ms": ttft,
            "client_ttft_samples_ms": [
                float(value) for value in phase["client_ttft_batch_ms"]["samples"]
            ],
            "client_prompt_tokens_per_second": tokens * 1000.0 / ttft,
        }
    return {
        "tier": tier,
        "mechanism": "ktransformers_eager",
        "status": "ok",
        "timing_boundary": "localhost_http_sse_client_ttft",
        "cuda_graph": "disabled_eager",
        "batches": batches,
        "placement": {
            "policy": "same_calibration_cold_experts",
            "cold_experts": int(run_summary["cold_experts"]),
            "main_weight_bytes": int(run_summary["cold_main_weight_bytes"]),
            "scale_bytes": int(run_summary["cold_scale_bytes"]),
        },
        "result_path": rel(path),
        "result_sha256": sha256(path),
    }


def kt_graph_failure(tier: str, paths: list[Path]) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        observations: dict[str, list[bool]] = {"314": [], "1217": []}
        for tokens, graph in re.findall(
            r"Prefill batch[^\n]*#new-token: (314|1217),[^\n]*cuda graph: (True|False)",
            text,
        ):
            observations[tokens].append(graph == "True")
        attempt = {
            "server_log": rel(path),
            "server_log_sha256": sha256(path),
            "piecewise_capture_started": "Capture piecewise CUDA graph begin" in text,
            "piecewise_capture_completed": "Capture piecewise CUDA graph end" in text,
            "capture_sizes_314_1217": bool(
                re.search(r"Capture cuda graph num tokens \[314, 1217\]", text)
            ),
            "prefill_graph_observations": observations,
            "fatal_segfault": "Fatal Python error: Segmentation fault" in text,
            "failure_inside_cuda_graph_replay": (
                "torch/cuda/graphs.py\", line 141 in replay" in text
            ),
            "overlap_disabled": "disable_overlap_schedule=True" in text,
        }
        if not all(
            (
                attempt["piecewise_capture_started"],
                attempt["piecewise_capture_completed"],
                attempt["capture_sizes_314_1217"],
                attempt["fatal_segfault"],
            )
        ):
            raise AssertionError(f"qualification evidence is incomplete: {path}")
        if len(observations["314"]) < 4 or not all(observations["314"][-4:]):
            raise AssertionError(f"314-token Graph qualification differs: {path}")
        attempts.append(attempt)
    return {
        "tier": tier,
        "mechanism": "ktransformers_piecewise_prefill_graph",
        "status": "unsupported_runtime_crash",
        "timing_boundary": "no_publishable_timing",
        "cuda_graph": "capture_succeeded_but_b4_replay_segfaulted",
        "batches": {
            "1": {"prompt_tokens": 314, "result": "qualification_only_not_reported"},
            "4": {"prompt_tokens": 1217, "result": "segmentation_fault"},
        },
        "attempts": attempts,
    }


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator


def pct_reduction(old: float, new: float) -> float:
    return (old - new) / old * 100.0


def record_by(results: list[dict[str, Any]], tier: str, mechanism: str) -> dict[str, Any]:
    matches = [
        item
        for item in results
        if item["tier"] == tier and item["mechanism"] == mechanism
    ]
    if len(matches) != 1:
        raise AssertionError((tier, mechanism, len(matches)))
    return matches[0]


def build() -> dict[str, Any]:
    inputs = load(INPUT_MANIFEST)
    results: list[dict[str, Any]] = []
    for tier in ("2g", "4g"):
        cold = inputs["tiers"][tier]
        cold_placement = {
            "policy": "calibration_cold_experts_48_layers",
            "cold_experts": int(cold["offload_plan_records"]),
            "experts_per_layer": int(cold["experts_per_layer"]),
            "main_weight_bytes": int(cold["selected_main_bytes"]),
            "scale_bytes": int(cold["selected_scale_bytes"]),
            "plan_sha256": cold["plan_sha256"],
        }
        for mode in ("eager", "graph"):
            results.append(
                vllm_record(
                    tier,
                    f"custom_remote_{mode}",
                    CUSTOM_ROOT / f"{tier}_{mode}/result.json",
                    graph=mode == "graph",
                    placement=cold_placement,
                )
            )
        for backend in ("uva", "copy"):
            placement_path = VLLM_ROOT / f"{tier}_{backend}/placement.json"
            placement_raw = load(placement_path)
            placement = {
                "policy": "vllm_native_parameter_order_layer_prefix",
                **placement_raw,
                "placement_path": rel(placement_path),
            }
            results.append(
                vllm_record(
                    tier,
                    f"vllm_native_{backend}",
                    VLLM_ROOT / f"{tier}_{backend}/result.json",
                    graph=False,
                    placement=placement,
                )
            )
        results.append(
            kt_eager_record(
                tier,
                KT_ROOT / f"{tier}_eager/result.json",
                KT_ROOT / f"{tier}_eager/run_summary.json",
            )
        )
        results.append(kt_graph_failure(tier, GRAPH_QUALIFICATIONS[tier]))

    comparisons: dict[str, Any] = {}
    for tier in ("2g", "4g"):
        custom_eager = record_by(results, tier, "custom_remote_eager")
        custom_graph = record_by(results, tier, "custom_remote_graph")
        native_uva = record_by(results, tier, "vllm_native_uva")
        native_copy = record_by(results, tier, "vllm_native_copy")
        comparisons[tier] = {}
        for batch in ("1", "4"):
            eager_ms = custom_eager["batches"][batch]["prefill_step_ms"]
            graph_ms = custom_graph["batches"][batch]["prefill_step_ms"]
            uva_ms = native_uva["batches"][batch]["prefill_step_ms"]
            copy_ms = native_copy["batches"][batch]["prefill_step_ms"]
            comparisons[tier][batch] = {
                "custom_graph_vs_custom_eager_speedup": ratio(eager_ms, graph_ms),
                "custom_graph_vs_custom_eager_latency_reduction_pct": pct_reduction(
                    eager_ms, graph_ms
                ),
                "custom_graph_vs_vllm_uva_speedup": ratio(uva_ms, graph_ms),
                "custom_graph_vs_vllm_copy_speedup": ratio(copy_ms, graph_ms),
            }

    vllm_hashes = set()
    for item in results:
        if item["status"] != "ok" or item["mechanism"] == "ktransformers_eager":
            continue
        result = load(REPO / item["result_path"])
        for cycle in result["cycles"]:
            if cycle["role"] == "measurement":
                vllm_hashes.add((cycle["batch_size"], cycle["output_token_ids_sha256"]))
    if len({value for batch, value in vllm_hashes if batch == 1}) != 1:
        raise AssertionError("vLLM/custom batch-1 outputs differ")
    if len({value for batch, value in vllm_hashes if batch == 4}) != 1:
        raise AssertionError("vLLM/custom batch-4 outputs differ")

    return {
        "schema": "qwen3_prefill_six_mechanism_results_v1",
        "status": "ok_with_unsupported_ktransformers_graph",
        "date": "2026-08-21",
        "protocol": {
            "model": "Qwen/Qwen3-30B-A3B-FP8",
            "gpu": "NVIDIA GeForce RTX 5090 32 GiB",
            "prompt_cases": {"1": 314, "4": 1217},
            "conditioning_repeats": 1,
            "measurement_repeats": 3,
            "vllm_attention_backend": "TRITON_ATTN",
            "ktransformers_attention_backend": "flashinfer",
            "vllm_timing_boundary": "CUDA-synchronized EngineCore.step",
            "ktransformers_timing_boundary": "localhost batched HTTP/SSE TTFT",
            "fixture_sha256": sha256(
                REPO
                / "experiments/eurosys_moe_offload_evaluation/inputs/legacy_pilot/qwen3_30b_a3b_fp8/fixtures/evaluation_fixture_token_ids.json"
            ),
            "cold_plan_manifest": rel(INPUT_MANIFEST),
            "cold_plan_manifest_sha256": sha256(INPUT_MANIFEST),
        },
        "results": results,
        "comparisons": comparisons,
        "correctness": {
            "custom_and_vllm_measurement_outputs_bit_exact": True,
            "ktransformers_repeats_deterministic_within_each_arm": True,
            "ktransformers_outputs_not_required_bit_exact_to_vllm": True,
        },
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def markdown(payload: dict[str, Any]) -> str:
    results = payload["results"]
    rows: list[str] = []
    labels = {
        "custom_remote_eager": "自定义远程访问 eager",
        "custom_remote_graph": "自定义远程访问 FULL Graph",
        "vllm_native_uva": "vLLM pinned UVA（按层）",
        "vllm_native_copy": "vLLM copy-on-forward（按层搬运）",
        "ktransformers_eager": "KTransformers CPU 专家 eager",
        "ktransformers_piecewise_prefill_graph": "KTransformers piecewise Graph",
    }
    for tier in ("2g", "4g"):
        for mechanism in labels:
            item = record_by(results, tier, mechanism)
            if item["status"] == "ok":
                if mechanism == "ktransformers_eager":
                    b1 = item["batches"]["1"]
                    b4 = item["batches"]["4"]
                    first = f"{fmt(b1['client_ttft_ms'])} ms / {b1['client_prompt_tokens_per_second']:.0f} tok/s"
                    fourth = f"{fmt(b4['client_ttft_ms'])} ms / {b4['client_prompt_tokens_per_second']:.0f} tok/s"
                    boundary = "HTTP/SSE TTFT"
                else:
                    b1 = item["batches"]["1"]
                    b4 = item["batches"]["4"]
                    first = f"{fmt(b1['prefill_step_ms'])} ms / {b1['prefill_tokens_per_second']:.0f} tok/s"
                    fourth = f"{fmt(b4['prefill_step_ms'])} ms / {b4['prefill_tokens_per_second']:.0f} tok/s"
                    boundary = "同步 Engine step"
                graph = item["cuda_graph"]
            else:
                first = "未发布（仅资格运行）"
                fourth = "N/A（回放段错误）"
                boundary = "无有效计时"
                graph = "捕获成功，B4 回放失败"
            rows.append(
                f"| {tier} | {labels[mechanism]} | {first} | {fourth} | {boundary} | {graph} |"
            )

    comparison_lines: list[str] = []
    for tier in ("2g", "4g"):
        for batch, prompt in (("1", 314), ("4", 1217)):
            value = payload["comparisons"][tier][batch]
            comparison_lines.append(
                f"- {tier}、{prompt} token：自定义 Graph 相对自定义 eager 为 "
                f"{value['custom_graph_vs_custom_eager_speedup']:.2f}×（延迟降低 "
                f"{value['custom_graph_vs_custom_eager_latency_reduction_pct']:.1f}%）；"
                f"相对 vLLM UVA 为 {value['custom_graph_vs_vllm_uva_speedup']:.2f}×，"
                f"相对 vLLM 搬运为 {value['custom_graph_vs_vllm_copy_speedup']:.2f}×。"
            )

    return f"""# Qwen3 Prefill 六机制对比（2026-08-21）

## 结论

- 自定义远程访问已经能在 Prefill 使用真正的 FULL CUDA Graph：2/4 GiB 两档的每个 Prefill 都恰好一次 FULL replay，捕获证书 H2D=0，远程只读 PTE 在回放前重新校验。
- 在与 vLLM 直接同边界的四项中，自定义 Graph 在 314-token 小批次最快；1,217-token 时 Graph 启动开销已不是主瓶颈，收益缩小。4 GiB、1,217-token 下 vLLM 搬运为 112.451 ms，略快于自定义远程 Graph 的 118.105 ms。
- vLLM 按层 UVA 随 token 数和卸载量增长明显恶化，因为它把前缀层中的高频 attention、整层专家等一起变成远程访问；冷专家策略只让低频专家远程化。
- KTransformers eager 可运行，但 CPU 专家命中使 TTFT 很高，尤其 4 GiB/B4 达 6.837 s。其 Prefill piecewise Graph 在两档均成功捕获并连续完成 4 次 314-token 回放，随后进入第一次 1,217-token 请求时都段错误；2 GiB 完整栈定位在 `torch.cuda.graph.replay()`，关闭 overlap 后仍复现。因此记为不支持，不发布性能数。

## 正式结果

| 档位 | 机制 | 314 token（B1） | 1,217 token（B4） | 计时边界 | CUDA Graph |
|---|---|---:|---:|---|---|
{chr(10).join(rows)}

数值均为 1 次 conditioning 后 3 次正式样本的中位数。`tok/s` 是该批输入 token 数除以对应延迟。

## 同边界比较

{chr(10).join(comparison_lines)}

这里的倍率只比较自定义与 vLLM 四项，因为它们都使用 CUDA 同步的 `EngineCore.step`。KTransformers 当前只能从本机 HTTP/SSE 记录批次 TTFT，包含调度、采样、输出与传输边界，绝对值不能与 Engine step 当作严格微基准直接相除。

## 放置范围

- 冷专家 2 GiB：48 层 × 10 = 480 个专家，主权重 2.109375 GiB，另有 540 KiB scale；校准路由命中 0/86,016。
- 冷专家 4 GiB：48 层 × 19 = 912 个专家，主权重 4.0078125 GiB，另有 1,026 KiB scale；校准路由命中 5/86,016。
- vLLM 原生 2 GiB：实际 2.134731 GiB、62 个 Parameter，按装载顺序覆盖 L0–L3。
- vLLM 原生 4 GiB：实际 4.064380 GiB、114 个 Parameter，按装载顺序覆盖 L0–L6。

因此这是“各系统原生策略”的系统对比，不是相同 Parameter 集合的纯传输机制对照。自定义/KTransformers 使用相同冷专家计划；vLLM 按用户要求使用原生按层/参数顺序。

## 协议与校验

- 模型：Qwen3-30B-A3B-FP8；GPU：RTX 5090 32 GiB。
- 输入：冻结 token IDs；B1=314 token，严格 B4=1,217 aggregate token；校准与评测请求源索引隔离。
- 自定义/vLLM 固定 `TRITON_ATTN`、关闭 prefix cache/chunked prefill；KTransformers 使用其 FlashInfer 服务栈。
- 自定义与 vLLM 八个正式 case 的生成 token hash 按 batch 完全一致；KTransformers 各 case 内 3 次重复确定，但不要求与 vLLM 位级一致。
- 原始 JSON、日志、失败栈及显存快照均保存在 `experiments/qwen_prefill_six_mechanism_compare/artifacts/`。
"""


def main() -> int:
    payload = build()
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    OUTPUT_MD.write_text(markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "json": str(OUTPUT_JSON),
                "markdown": str(OUTPUT_MD),
                "records": len(payload["results"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
