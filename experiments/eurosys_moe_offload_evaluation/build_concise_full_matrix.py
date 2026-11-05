#!/usr/bin/env python3
"""Build the two-model C1/C16 six-mechanism result document."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
GIB = 1024**3
WORKLOADS_DEEPSEEK = (
    "short_dialogue",
    "ordinary_chat",
    "rag_summary",
    "long_generation",
)
WORKLOADS_QWEN = (
    "short_dialogue",
    "ordinary_chat",
    "medium_context",
    "rag_summary",
    "long_generation",
)
LABELS = {
    "short_dialogue": "Short",
    "ordinary_chat": "Ordinary",
    "medium_context": "Medium",
    "rag_summary": "RAG",
    "long_generation": "Long",
}


@dataclass
class Cell:
    model: str
    workload: str
    concurrency: int
    mechanism: str
    offload_gib: float | None
    ttft_ms: float | None
    tpot_ms: float | None
    prefill_e2e_ms: float | None
    decode_tps: float | None
    e2e_ms: float | None
    status: str
    source: str
    note: str = ""


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def one(paths: Iterable[Path], description: str) -> Path:
    values = sorted(paths)
    if len(values) != 1:
        raise RuntimeError(f"expected one {description}, found {len(values)}: {values}")
    return values[0]


def p50(distribution: Any) -> float | None:
    if not isinstance(distribution, dict):
        return None
    value = distribution.get("p50", distribution.get("median"))
    return float(value) if value is not None else None


def request_median(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in records if item.get(key) is not None]
    return statistics.median(values) if values else None


def actual_manifest_gib(path: Path) -> float | None:
    if not path.is_file():
        return None
    maximum = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        maximum = max(maximum, int(item.get("cumulative_offload_bytes") or 0))
    return maximum / GIB if maximum else None


def e2_client_metrics(path: Path) -> tuple[float, float, float]:
    data = load(path)
    if data.get("status") != "ok":
        raise RuntimeError(f"source result is not ok: {path}")
    phase = data["phases"][0]
    return (
        float(phase["ttft_ms"]["p50"]),
        float(phase["tpot_ms"]["p50"]),
        float(phase["e2e_ms"]["p50"]),
    )


def e2_direct_metrics(path: Path) -> tuple[float, float, float, float | None, float | None]:
    data = load(path)
    if data.get("status") != "ok":
        raise RuntimeError(f"source result is not ok: {path}")
    measurement = data["measurement"]
    phase = measurement.get("phase_metrics") or {}
    return (
        float(measurement["ttft_ms"]["p50"]),
        float(measurement["tpot_ms"]["p50"]),
        float(measurement["e2e_ms"]["max"]),
        (
            float(phase["prefill_e2e_ms"])
            if phase.get("prefill_e2e_ms") is not None
            else None
        ),
        (
            float(phase["steady_decode_tokens_per_second"])
            if phase.get("steady_decode_tokens_per_second") is not None
            else None
        ),
    )


def e2_direct_prefill(path: Path) -> float | None:
    data = load(path)
    measurement = data.get("measurement") or {}
    phase = measurement.get("phase_metrics") or {}
    value = phase.get("prefill_e2e_ms")
    return float(value) if value is not None else None


def compact_error(value: Any) -> str:
    text = str(value or "unknown error")
    lowered = text.lower()
    if "outofmemory" in lowered or "out of memory" in lowered:
        return "CUDA OOM（Graph pool/临时 workspace 容量不足）"
    if "illegal memory access" in lowered:
        return "首次真实 Graph replay 触发 CUDA illegal memory access（Xid 31/MMU fault）"
    if "wrong device" in lowered:
        return "exact-name 参数在权重加载后回到 GPU，Copy wrapper 设备检查失败"
    if "exceeds cpu_offload_gb" in lowered:
        return "exact-name 参数总字节超过 cpu_offload_gb 预算"
    return " ".join(text.split())[:240]


def failure_status(value: Any) -> str:
    text = str(value or "").lower()
    if "outofmemory" in text or "out of memory" in text:
        return "OOM（组件容量限制）"
    if "illegal memory access" in text:
        return "N/A（组件非法访存）"
    return "N/A（组件错误）"


def qwen_phase_metrics(path: Path, concurrency: int) -> tuple[float, float, float, float, float]:
    data = load(path)
    if data.get("status") != "ok":
        raise RuntimeError(f"source result is not ok: {path}")
    summary = data["phase_summary"][str(concurrency)]
    return (
        float(summary["request_ttft_p50_ms"]["median"]),
        float(summary["request_tpot_p50_ms"]["median"]),
        float(summary["prefill_step_ms"]["median"]),
        float(summary["decode_tokens_per_second"]),
        float(summary["cycle_e2e_ms"]["median"]),
    )


def qwen_adapt_metrics(path: Path) -> tuple[float, float, float, float, float, float]:
    data = load(path)
    if data.get("status") != "ok":
        raise RuntimeError(f"source result is not ok: {path}")
    timing = data["timing"]
    outputs = data["outputs"]
    return (
        request_median(outputs, "ttft_ms") or 0.0,
        request_median(outputs, "tpot_ms") or 0.0,
        float(timing["prefill_step_ms"]),
        float(timing["steady_decode_tokens_per_second"]),
        float(timing["cycle_e2e_ms"]),
        float(data["config"]["expected_selected_bytes"]) / GIB,
    )


def qwen_kt_candidates(workload: str, concurrency: int, cold: int) -> list[Path]:
    candidates: list[tuple[float, Path]] = []
    for path in ARTIFACTS.glob("e3-ktransformers-tun*/**/result.json"):
        try:
            data = load(path)
            config = data.get("configuration") or {}
            cells = data.get("cells") or []
            if (
                data.get("status") == "ok"
                and config.get("workload") == workload
                and int(config.get("cpu_experts_per_layer", -1)) == cold
                and int(config.get("cpu_threads", -1)) == 7
                and config.get("cuda_graph_enabled") is True
                and len(cells) == 1
                and int(cells[0].get("batch_size", -1)) == concurrency
            ):
                candidates.append((float(cells[0]["cycle_e2e_ms"]), path))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return [path for _, path in sorted(candidates)]


def qwen_kt_metrics(path: Path) -> tuple[float, float, float, float, float]:
    data = load(path)
    if data.get("status") != "ok":
        raise RuntimeError(f"source result is not ok: {path}")
    cell = data["cells"][0]
    requests = cell["requests"]
    return (
        request_median(requests, "ttft_ms") or 0.0,
        request_median(requests, "tpot_ms") or 0.0,
        float(cell["prefill_step_ms"]),
        float(cell["decode_tokens_per_second"]),
        float(cell["cycle_e2e_ms"]),
    )


def deepseek_cells(gaps_root: Path) -> list[Cell]:
    cells: list[Cell] = []
    reused = ARTIFACTS / "e2-near-equal-offload-plus-one-expert-20260903-attempt-01"
    kt_c1 = ARTIFACTS / "e2-ktransformers-equal-offload-c1-20260911-attempt-02"
    kt_c16 = ARTIFACTS / "e2-ktransformers-equal-offload-c16-20260911-attempt-01"
    kt_rag = ARTIFACTS / "e2-ktransformers-equal-offload-rag-c16-20260911-attempt-02"
    direct_uva_decode_c1 = (
        ARTIFACTS
        / "e2-native-uva-direct-phase-metrics-20260911-attempt-01-full-decode-only-c1"
    )
    direct_uva_decode_c16 = (
        ARTIFACTS
        / "e2-native-uva-direct-phase-metrics-20260911-attempt-01-full-decode-only-c16"
    )
    fixed_uva_graph_c1 = (
        ARTIFACTS
        / "e2-native-uva-direct-phase-metrics-20260911-attempt-01-full-separate-c1"
    )
    fixed_uva_graph_c16_short = (
        ARTIFACTS
        / "e2-native-uva-full-separate-fixed-probe-20260911-short-c16"
    )
    fixed_uva_graph_c16_rest = (
        ARTIFACTS / "e2-native-uva-full-separate-fixed-20260911-c16-rest"
    )
    retry_root = ARTIFACTS / "full-matrix-deepseek-retries-20260911-attempt-02"
    adapt_prefill_ordinary_c16 = (
        ARTIFACTS / "e2-adaptmoe-prefill-ordinary-c16-fixed-20260911-attempt-01"
    )
    copy_eager_c1 = ARTIFACTS / "full-matrix-deepseek-native-copy-eager-c1-20260911-attempt-04"
    copy_eager_rag_c1 = ARTIFACTS / "full-matrix-deepseek-native-copy-eager-rag-c1-20260911-attempt-05"
    copy_eager_short_c16 = ARTIFACTS / "full-matrix-deepseek-native-copy-eager-short-c16-20260911-attempt-04"
    copy_graph_c1 = ARTIFACTS / "e2-native-copy-full-separate-direct-c1-20260911-attempt-02"
    copy_graph_c16 = ARTIFACTS / "e2-native-copy-full-separate-direct-c16-20260911-attempt-01"
    adapt_offload = {
        ("short_dialogue", 1): 0.838,
        ("ordinary_chat", 1): 0.838,
        ("rag_summary", 1): 0.838,
        ("long_generation", 1): 0.838,
        ("short_dialogue", 16): 0.838,
        ("ordinary_chat", 16): 2.514,
        ("rag_summary", 16): 8.379,
        ("long_generation", 16): 2.514,
    }
    native_offload = {
        ("short_dialogue", 1): 0.896,
        ("ordinary_chat", 1): 0.896,
        ("rag_summary", 1): 0.896,
        ("long_generation", 1): 0.896,
        ("short_dialogue", 16): 0.896,
        ("ordinary_chat", 16): 2.330,
        ("rag_summary", 16): 8.522,
        ("long_generation", 16): 2.330,
    }
    kt_cold = {
        (workload, 1): 2 for workload in WORKLOADS_DEEPSEEK
    } | {
        ("short_dialogue", 16): 2,
        ("ordinary_chat", 16): 6,
        ("rag_summary", 16): 20,
        ("long_generation", 16): 6,
    }

    for workload in WORKLOADS_DEEPSEEK:
        for concurrency in (1, 16):
            if concurrency == 1:
                adapt_path = one(
                    (reused / "c1/cases/ours" / workload).glob("**/client.json"),
                    f"DeepSeek AdaptMoE {workload}/C1 client",
                )
                adapt_values = e2_client_metrics(adapt_path)
            else:
                adapt_path = one(
                    (reused / "c16/cases/ours" / workload).glob("**/result.json"),
                    f"DeepSeek AdaptMoE {workload}/C16 result",
                )
                a = e2_direct_metrics(adapt_path)
                adapt_values = a[:3]

            native_decode_root = (
                direct_uva_decode_c1
                if concurrency == 1
                else direct_uva_decode_c16
            )
            native_path = one(
                (native_decode_root / "cases/native_pinned_uva" / workload).glob(
                    "**/result.json"
                ),
                f"direct DeepSeek Native UVA Decode Graph {workload}/C{concurrency}",
            )
            native_values = e2_direct_metrics(native_path)

            if workload == "rag_summary" and concurrency == 1:
                prefill_base = retry_root / "adaptmoe_prefill_rag_c1"
            elif workload == "ordinary_chat" and concurrency == 16:
                prefill_base = adapt_prefill_ordinary_c16
            else:
                prefill_base = gaps_root / f"adaptmoe_prefill_c{concurrency}"
            prefill_path = one(
                (prefill_base / "cases/ours" / workload).glob("**/result.json"),
                f"DeepSeek AdaptMoE Prefill {workload}/C{concurrency}",
            )
            prefill_data = load(prefill_path)
            prefill = (
                e2_direct_prefill(prefill_path)
                if prefill_data.get("status") == "ok"
                and prefill_data.get("measurement") is not None
                else None
            )
            prefill_note = (
                "Decode tok/s 由同步 phase-batched TPOT 计算"
                if prefill is not None
                else "独立 Prefill 重测在真实 Graph replay 触发组件非法访存；Decode tok/s 由同步 phase-batched TPOT 计算"
            )
            cells.append(
                Cell(
                    "DeepSeek-V2-Lite BF16",
                    workload,
                    concurrency,
                    "AdaptMoE 双阶段 Graph",
                    adapt_offload[(workload, concurrency)],
                    adapt_values[0],
                    adapt_values[1],
                    prefill,
                    concurrency * 1000.0 / adapt_values[1],
                    adapt_values[2],
                    (
                        "ok（复用完整运行 + 新测 Prefill）"
                        if prefill is not None
                        else "ok（复用完整运行；Prefill N/A）"
                    ),
                    f"{adapt_path.relative_to(HERE)}; {prefill_path.relative_to(HERE)}",
                    prefill_note,
                )
            )
            cells.append(
                Cell(
                    "DeepSeek-V2-Lite BF16",
                    workload,
                    concurrency,
                    "Native UVA（eager Prefill + Decode Graph）",
                    native_offload[(workload, concurrency)],
                    native_values[0],
                    native_values[1],
                    native_values[3],
                    native_values[4],
                    native_values[2],
                    "ok（direct runner 重测）",
                    str(native_path.relative_to(HERE)),
                    "FULL_DECODE_ONLY：Prefill eager、Decode FULL Graph replay",
                )
            )

            if concurrency == 1:
                copy_base = (
                    copy_eager_rag_c1
                    if workload == "rag_summary"
                    else copy_eager_c1
                )
            elif workload == "short_dialogue":
                copy_base = copy_eager_short_c16
            else:
                copy_base = gaps_root / "native_copy_eager_c16"
            copy_path = one(
                (copy_base / "cases/native_copy_on_forward" / workload).glob("**/result.json"),
                f"DeepSeek Native Copy eager {workload}/C{concurrency}",
            )
            copy_data = load(copy_path)
            if copy_data.get("status") == "ok":
                ttft, tpot, e2e, copy_prefill, copy_decode = e2_direct_metrics(copy_path)
                cells.append(
                    Cell(
                        "DeepSeek-V2-Lite BF16",
                        workload,
                        concurrency,
                        "Native Copy eager",
                        actual_manifest_gib(copy_path.parent / "offload_manifest.jsonl"),
                        ttft,
                        tpot,
                        copy_prefill,
                        copy_decode,
                        e2e,
                        "ok（新测）",
                        str(copy_path.relative_to(HERE)),
                        (
                            "低档为最近可执行 exact-name 组合，0.812744 GiB；相对 0.896 GiB 差 10.24%"
                            if actual_manifest_gib(copy_path.parent / "offload_manifest.jsonl")
                            and actual_manifest_gib(copy_path.parent / "offload_manifest.jsonl") < 1.0
                            else ""
                        ),
                    )
                )
            else:
                copy_error = copy_data.get("error", "unknown error")
                cells.append(
                    Cell(
                        "DeepSeek-V2-Lite BF16", workload, concurrency,
                        "Native Copy eager", None, None, None, None, None, None,
                        failure_status(copy_error), str(copy_path.relative_to(HERE)),
                        compact_error(copy_error),
                    )
                )

            cold = kt_cold[(workload, concurrency)]
            if concurrency == 1:
                kt_path = one(
                    (kt_c1 / "cases/ktransformers" / workload / f"cold_{cold:02d}").glob(
                        "result.json"
                    ),
                    f"DeepSeek KT {workload}/C1",
                )
            else:
                base = kt_rag if workload == "rag_summary" else kt_c16
                kt_path = one(
                    (base / "cases/ktransformers" / workload / f"cold_{cold:02d}").glob("result.json"),
                    f"DeepSeek KT {workload}/C16",
                )
            kt_values = e2_direct_metrics(kt_path)[:3]
            kt_prefill_data = load(kt_path)
            kt_prefill = float(kt_prefill_data["measurement"]["ttft_ms"]["max"])
            cells.append(
                Cell(
                    "DeepSeek-V2-Lite BF16",
                    workload,
                    concurrency,
                    "KTransformers 7T + Decode Graph",
                    cold * 0.4189453125,
                    kt_values[0],
                    kt_values[1],
                    kt_prefill,
                    concurrency * 1000.0 / kt_values[1],
                    kt_values[2],
                    f"ok（等量卸载新测 cold{cold}）",
                    str(kt_path.relative_to(HERE)),
                    "与 AdaptMoE 使用相同 cold 档位；Prefill E2E 为全批请求产生首 token 的时刻；Decode tok/s 由同步 TPOT 计算",
                )
            )

            baseline_decode_tps = native_values[4]
            if concurrency == 1:
                graph_root = fixed_uva_graph_c1
                capture_layout = "与正式请求一致的 1-request 布局"
            elif workload == "short_dialogue":
                graph_root = fixed_uva_graph_c16_short
                capture_layout = "与正式请求一致的 16-request 布局"
            else:
                graph_root = fixed_uva_graph_c16_rest
                capture_layout = (
                    "2×8192 请求布局"
                    if workload == "rag_summary"
                    else "与正式请求一致的 16-request 布局"
                )
            graph_path = one(
                (graph_root / "cases/native_pinned_uva" / workload).glob(
                    "**/result.json"
                ),
                f"fixed DeepSeek UVA dual Graph {workload}/C{concurrency}",
            )
            graph_data = load(graph_path)
            verification = load(graph_path.parent / "verification.json")
            required_graph_checks = (
                "native_uva_prefill_graph_replayed",
                "native_uva_decode_graph_replayed",
                "native_uva_no_prefill_eager_runtime_dispatch",
                "native_uva_no_decode_eager_runtime_dispatch",
                "scheduler_prefilled_all_before_decode",
            )
            checks = verification.get("checks") or {}
            if (
                graph_data.get("status") != "ok"
                or verification.get("status") != "ok"
                or any(not checks.get(key) for key in required_graph_checks)
            ):
                raise RuntimeError(
                    f"fixed dual-Graph result lacks valid replay proof: {graph_path}"
                )
            ttft, tpot, e2e, prefill, graph_decode_tps = e2_direct_metrics(
                graph_path
            )
            if graph_decode_tps is None:
                raise RuntimeError(
                    f"fixed dual-Graph result lacks phase Decode throughput: {graph_path}"
                )
            decode_delta_pct = (
                graph_decode_tps / baseline_decode_tps - 1.0
            ) * 100.0
            if abs(decode_delta_pct) > 2.0:
                raise RuntimeError(
                    f"fixed dual-Graph Decode differs from baseline by "
                    f"{decode_delta_pct:+.2f}%: {graph_path}"
                )
            baseline_tokens = [
                request.get("output_token_ids")
                for request in load(native_path)["measurement"]["requests"]
            ]
            graph_tokens = [
                request.get("output_token_ids")
                for request in graph_data["measurement"]["requests"]
            ]
            if graph_tokens != baseline_tokens:
                raise RuntimeError(
                    f"dual-Graph output differs from Decode-only baseline: {graph_path}"
                )
            cells.append(
                Cell(
                    "DeepSeek-V2-Lite BF16", workload, concurrency,
                    "Native UVA 双阶段 Graph",
                    native_offload[(workload, concurrency)],
                    ttft, tpot, prefill, graph_decode_tps, e2e,
                    "ok（direct runner 重测）", str(graph_path.relative_to(HERE)),
                    "Prefill/Decode 均为 FULL Graph replay；capture 使用"
                    f"{capture_layout}；Decode 相对 Decode-only 基线 "
                    f"{decode_delta_pct:+.2f}%；完整输出一致",
                )
            )

            graph_copy_base = copy_graph_c1 if concurrency == 1 else copy_graph_c16
            graph_copy_path = one(
                (graph_copy_base / "cases/native_copy_on_forward" / workload).glob("**/result.json"),
                f"DeepSeek Native Copy dual Graph {workload}/C{concurrency}",
            )
            graph_copy_data = load(graph_copy_path)
            if graph_copy_data.get("status") == "ok":
                ttft, tpot, e2e, prefill, decode = e2_direct_metrics(graph_copy_path)
                cells.append(
                    Cell(
                        "DeepSeek-V2-Lite BF16", workload, concurrency,
                        "Native Copy 双阶段 Graph",
                        actual_manifest_gib(graph_copy_path.parent / "offload_manifest.jsonl"),
                        ttft, tpot, prefill, decode, e2e, "ok（修正参数后重测）",
                        str(graph_copy_path.relative_to(HERE)),
                    )
                )
            else:
                graph_copy_error = graph_copy_data.get("error", "unknown error")
                cells.append(
                    Cell(
                        "DeepSeek-V2-Lite BF16", workload, concurrency,
                        "Native Copy 双阶段 Graph",
                        actual_manifest_gib(graph_copy_path.parent / "offload_manifest.jsonl"),
                        None, None, None, None, None,
                        failure_status(graph_copy_error), str(graph_copy_path.relative_to(HERE)),
                        compact_error(graph_copy_error),
                    )
                )
    return cells


def qwen_cells(eager_root: Path) -> list[Cell]:
    cells: list[Cell] = []
    paired = ARTIFACTS / "e3-near-equal-offload-plus-one-expert-20260903-attempt-02"
    copy_graph = ARTIFACTS / "e3-near-equal-native-copy-graph-20260903-attempt-01"
    copy_graph_rag_c16_retry = (
        ARTIFACTS
        / "e3-native-copy-full-separate-rag-c16-retry-20260911-attempt-01"
        / "cases/rag_summary_b16_copy"
    )
    cold_map = {
        ("short_dialogue", 1): (11, 10),
        ("short_dialogue", 16): (11, 10),
        ("ordinary_chat", 1): (11, 10),
        ("ordinary_chat", 16): (39, 38),
        ("medium_context", 1): (11, 10),
        ("medium_context", 16): (39, 38),
        ("rag_summary", 1): (39, 38),
        ("rag_summary", 16): (90, 89),
        ("long_generation", 1): (11, 10),
        ("long_generation", 16): (39, 38),
    }
    for workload in WORKLOADS_QWEN:
        for concurrency in (1, 16):
            adapt_cold, kt_cold = cold_map[(workload, concurrency)]
            adapt_path = paired / (
                f"ours/ours/{workload}_b{concurrency}_D/custom/"
                f"b0_{workload}_b{concurrency}_D/result.json"
            )
            ttft, tpot, prefill, decode, e2e, offload = qwen_adapt_metrics(adapt_path)
            cells.append(
                Cell(
                    "Qwen3-30B-A3B FP8", workload, concurrency,
                    "AdaptMoE 双阶段 Graph", offload, ttft, tpot, prefill, decode,
                    e2e, f"ok（已有 cold{adapt_cold}）", str(adapt_path.relative_to(HERE)),
                )
            )
            for arm, mechanism in (("uva", "Native UVA eager"), ("copy", "Native Copy eager")):
                path = eager_root / f"cases/{workload}_b{concurrency}_{arm}/result.json"
                ttft, tpot, prefill, decode, e2e = qwen_phase_metrics(path, concurrency)
                cells.append(
                    Cell(
                        "Qwen3-30B-A3B FP8", workload, concurrency, mechanism,
                        actual_manifest_gib(path.parent / "manifest.jsonl"),
                        ttft, tpot, prefill, decode, e2e, "ok（新测 NONE/NONE）",
                        str(path.relative_to(HERE)),
                    )
                )
            candidates = qwen_kt_candidates(workload, concurrency, kt_cold)
            if not candidates:
                raise RuntimeError(
                    f"no tuned Qwen KT result for {workload}/C{concurrency}/cold{kt_cold}"
                )
            kt_path = candidates[0]
            ttft, tpot, prefill, decode, e2e = qwen_kt_metrics(kt_path)
            cells.append(
                Cell(
                    "Qwen3-30B-A3B FP8", workload, concurrency,
                    "KTransformers 7T + Decode Graph",
                    kt_cold * 48 * 4718592 / GIB, ttft, tpot, prefill, decode, e2e,
                    f"ok（已有最佳 cold{kt_cold}）", str(kt_path.relative_to(HERE)),
                )
            )
            uva_path = paired / f"native/cases/{workload}_b{concurrency}_uva/result.json"
            ttft, tpot, prefill, decode, e2e = qwen_phase_metrics(uva_path, concurrency)
            cells.append(
                Cell(
                    "Qwen3-30B-A3B FP8", workload, concurrency,
                    "Native UVA 双阶段 Graph",
                    actual_manifest_gib(uva_path.parent / "manifest.jsonl"),
                    ttft, tpot, prefill, decode, e2e, "ok（已有）",
                    str(uva_path.relative_to(HERE)),
                )
            )
            copy_path = copy_graph / f"native_copy/cases/{workload}_b{concurrency}_copy/result.json"
            if copy_path.is_file() and load(copy_path).get("status") == "ok":
                ttft, tpot, prefill, decode, e2e = qwen_phase_metrics(copy_path, concurrency)
                cells.append(
                    Cell(
                        "Qwen3-30B-A3B FP8", workload, concurrency,
                        "Native Copy 双阶段 Graph",
                        actual_manifest_gib(copy_path.parent / "manifest.jsonl"),
                        ttft, tpot, prefill, decode, e2e, "ok（已有）",
                        str(copy_path.relative_to(HERE)),
                    )
                )
            else:
                retry_manifest = copy_graph_rag_c16_retry / "manifest.jsonl"
                cells.append(
                    Cell(
                        "Qwen3-30B-A3B FP8", workload, concurrency,
                        "Native Copy 双阶段 Graph",
                        actual_manifest_gib(retry_manifest),
                        None, None, None, None, None, "OOM（重试确认，保留 N/A）",
                        str(copy_graph_rag_c16_retry.relative_to(HERE)),
                        "131,072-token Prefill Graph capture 时 attention 输出申请 1,024 MiB 失败；峰值显存 31,979 MiB",
                    )
                )
    return cells


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"


def render_table(cells: list[Cell]) -> list[str]:
    lines = [
        "| 场景 | C | 机制 | 实际卸载 GiB | TTFT (ms) | TPOT (ms) | Prefill E2E (ms) | Decode tok/s | E2E (ms) | 状态 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cell in cells:
        lines.append(
            "| " + " | ".join(
                (
                    LABELS[cell.workload],
                    str(cell.concurrency),
                    cell.mechanism,
                    fmt(cell.offload_gib, 3),
                    fmt(cell.ttft_ms),
                    fmt(cell.tpot_ms),
                    fmt(cell.prefill_e2e_ms),
                    fmt(cell.decode_tps),
                    fmt(cell.e2e_ms),
                    cell.status,
                )
            ) + " |"
        )
    return lines


def render_sources(cells: list[Cell]) -> list[str]:
    lines = [
        "| 模型 | 场景 | C | 机制 | 原始结果 | 说明 |",
        "|---|---|---:|---|---|---|",
    ]
    for cell in cells:
        lines.append(
            f"| {cell.model} | {LABELS[cell.workload]} | {cell.concurrency} | "
            f"{cell.mechanism} | `{cell.source}` | {cell.note or '—'} |"
        )
    return lines


def validate_cells(cells: list[Cell]) -> None:
    if len(cells) != 108:
        raise RuntimeError(f"expected 108 cells, found {len(cells)}")
    keys = [
        (cell.model, cell.workload, cell.concurrency, cell.mechanism)
        for cell in cells
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("matrix contains duplicate cells")
    coordinate_counts: dict[tuple[str, str, int], int] = {}
    for cell in cells:
        coordinate = (cell.model, cell.workload, cell.concurrency)
        coordinate_counts[coordinate] = coordinate_counts.get(coordinate, 0) + 1
    if any(count != 6 for count in coordinate_counts.values()):
        raise RuntimeError(f"each model/workload/C coordinate must have six mechanisms: {coordinate_counts}")
    for cell in cells:
        values = (
            cell.offload_gib,
            cell.ttft_ms,
            cell.tpot_ms,
            cell.prefill_e2e_ms,
            cell.decode_tps,
            cell.e2e_ms,
        )
        if any(value is not None and (not math.isfinite(value) or value <= 0) for value in values):
            raise RuntimeError(f"invalid numeric value in {cell}")
        if cell.status.startswith("ok") and any(
            value is None
            for value in (cell.ttft_ms, cell.tpot_ms, cell.decode_tps, cell.e2e_ms)
        ):
            raise RuntimeError(f"successful cell lacks a core metric: {cell}")
        if cell.offload_gib is None:
            raise RuntimeError(f"cell lacks actual/static offload quantity: {cell}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deepseek-gaps-root",
        type=Path,
        default=ARTIFACTS / "full-matrix-deepseek-gaps-20260911-attempt-01",
    )
    parser.add_argument(
        "--qwen-eager-root",
        type=Path,
        default=ARTIFACTS / "full-matrix-qwen-native-eager-20260911-attempt-01",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "RESULTS_E0_E14_CONCISE_Full.md",
    )
    args = parser.parse_args()

    deepseek = deepseek_cells(args.deepseek_gaps_root.resolve())
    qwen = qwen_cells(args.qwen_eager_root.resolve())
    all_cells = deepseek + qwen
    validate_cells(all_cells)
    failures = [cell for cell in all_cells if not cell.status.startswith("ok")]
    lines = [
        "# 两模型 C1/C16 六机制完整结果",
        "",
        "本文档扩展 `RESULTS_E0_E14_CONCISE.md` 的 E2/E3 数据，复用已有合法结果并仅补测原矩阵缺口。所有延迟单位均为 ms；每个新增 cell 为一个冷进程样本。",
        "",
        "指标口径：TTFT/TPOT 为请求级 P50；Prefill E2E 为完整 batch 从 EngineCore step 开始到全批 Prefill 完成并产生首 token 的墙钟时间；Decode tok/s 优先使用全批 Prefill 完成后直接累计的稳态 step。新测 Native UVA 两种 Graph 协议均使用该 phase counter；仅没有保存该 counter 的既有 DeepSeek AdaptMoE/KTransformers phase-batched 结果按 `C × 1000 / TPOT` 换算。E2E 为 batch 中最后一个请求完成的时间（Qwen 使用 cycle E2E）。`—` 表示源运行没有独立记录该指标，N/A/OOM 不填性能值。",
        "",
        "## DeepSeek-V2-Lite BF16",
        "",
        "请求为 Short、Ordinary、RAG、Long，共 4 × C1/C16 × 6 = 48 cells。按用户指定，Native UVA 不改为真正的 NONE/NONE；这里用新版 direct runner 重测 eager Prefill + Decode Graph 和双阶段 Graph，以补齐独立 Prefill E2E。KTransformers 已按与 AdaptMoE 相同的 cold2/cold6/cold20 档位重测，使用 7 个 CPU 专家线程和 Decode CUDA Graph；RAG/C16 继续使用 `kt_gpu_prefill_token_threshold=4096` 的 full-GPU Prefill fallback。",
        "",
        *render_table(deepseek),
        "",
        "DeepSeek Native Copy 的安全 exact-name 三档实际卸载量为 0.812744/2.406/8.250 GiB。low 档是当前 Copy wrapper 可执行的最近组合，相对 Native UVA 0.896 GiB 差 10.24%；尝试补入 shared expert、attention 或 router 小张量均因 DeepSeek 自定义权重加载后参数回到 GPU 而失败。mid/high 档相对 2.330/8.522 GiB 分别差 3.3%/3.3%。",
        "",
        "## Qwen3-30B-A3B FP8",
        "",
        "请求为 Short、Ordinary、Medium、RAG、Long，共 5 × C1/C16 × 6 = 60 cells。AdaptMoE、KTransformers 与两种双 Graph 路径复用已有合法结果；Native UVA/Copy eager 为本轮 TRITON_ATTN、C1/C16、Graph NONE/NONE 新测。按当前要求，本轮未重测 Qwen KTransformers，其 cold10/cold38/cold89 行保持不变，尚不是与 AdaptMoE 完全相同的 cold 档位。",
        "",
        *render_table(qwen),
        "",
        "## 覆盖与失败",
        "",
        "本轮 Qwen Native UVA/Copy eager 20/20 点完成。DeepSeek 的 AdaptMoE 独立 Prefill以及等量卸载 KTransformers cold2/cold6/cold20 均完成 8/8；Native Copy eager、两种 Native UVA Graph 协议也均为 8/8。修正 `STATIC_BANK` 参数并采用当前 MLA Graph 修复后，DeepSeek Native Copy 双阶段 Graph 完成 7/8，仅 RAG/C1 因 Graph pool/临时 workspace 容量不足保留 OOM。Qwen Native Copy 双阶段 Graph 的 RAG/C16 已按原配置复核一次，仍在 131,072-token Prefill Graph capture 时 OOM。全部冷进程退出后 GPU 显存恢复为 0 MiB。",
        "",
        "此前 Native UVA 双阶段 Graph 的 N/A 与 128-token 分块产生的全零输出均已废弃。根因有两处：MLA 纯 Prefill metadata 在 capture 时通过减法生成临时 GPU tensor，capture 返回后其 storage 被释放，但 GraphExec 仍保留旧地址；同时 Prefill Graph key 只区分总 token 数，未区分请求布局，使 16×8 capture 可能 replay 到 1×128。修复后纯 Prefill 使用 runner 持久输入 buffer 的 view，并按正式 step 的请求数 capture（RAG/C16 为 2×8192，其余 C16 为 16-request 布局）。8 对同行 direct-runner 结果均确认相应 Graph replay且完整输出一致；按纯 Decode phase counter，双阶段 Graph 相对 eager Prefill + Decode Graph 的吞吐差异绝对值不超过 0.61%。",
        "",
        f"总矩阵为 108 cells；有效完成 {len(all_cells) - len(failures)}，N/A/OOM {len(failures)}。失败点保留原始错误或 OOM 记录，不用其他协议数据填补。",
        "",
    ]
    if failures:
        lines.extend(
            [
                "| 模型 | 场景 | C | 机制 | 状态 |",
                "|---|---|---:|---|---|",
                *[
                    f"| {cell.model} | {LABELS[cell.workload]} | {cell.concurrency} | {cell.mechanism} | {cell.status} |"
                    for cell in failures
                ],
                "",
            ]
        )
    lines.extend(
        [
            "## 原始结果索引",
            "",
            *render_sources(all_cells),
            "",
        ]
    )
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "cells": len(all_cells),
        "ok": len(all_cells) - len(failures),
        "na_or_oom": len(failures),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
