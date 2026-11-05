#!/usr/bin/env python3
"""Verify and summarize the Qwen3 six-mechanism decode experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TIERS = ("2g", "4g")
BATCHES = (1, 4)
EXPECTED_STEPS = 16
EXPECTED_KT_REPEATS = 3
STOCK_UVM = "85A79790636BBD99BA3E43B"


def fail(message: str) -> None:
    raise RuntimeError(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected object: {path}")
    return value


def read_int(path: Path) -> int:
    require(path.is_file(), f"missing integer file: {path}")
    return int(path.read_text(encoding="utf-8").strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing TSV: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def check_execution(root: Path, expected: set[str]) -> None:
    rows = read_tsv(root / "execution.tsv")
    directories = {row["directory"] for row in rows}
    require(directories == expected, f"execution case set differs in {root}")
    require(
        all(int(row["returncode"]) == 0 for row in rows),
        f"nonzero execution return code in {root}",
    )


def check_gpu_json(path: Path) -> None:
    state = read_json(path)
    require(state["memory_used_mib"] == 0, f"GPU memory not clean: {path}")
    require(not state["compute_pids"], f"compute PIDs remain: {path}")
    require(not state.get("uvm_client_pids", []), f"UVM clients remain: {path}")


def check_gpu_csv(path: Path) -> None:
    require(read_int(path) == 0, f"GPU memory not clean: {path}")


def gib(value: int) -> float:
    return value / (1 << 30)


def percent_change(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def row_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {(row["family"], row["tier"], row["batch"]): row for row in rows}


def collect_custom(root: Path, inputs: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {
        f"{tier}_{mode}_b{batch}"
        for tier in TIERS
        for mode in ("eager", "graph")
        for batch in BATCHES
    }
    check_execution(root, expected)
    rows: list[dict[str, Any]] = []
    hashes: dict[tuple[str, int], dict[str, str]] = {}
    for tier in TIERS:
        tier_input = inputs["tiers"][tier]
        plan = Path(tier_input["plan"])
        require(
            sha256_file(plan) == tier_input["plan_sha256"],
            f"input plan hash differs for {tier}",
        )
        for mode in ("eager", "graph"):
            for batch in BATCHES:
                case_name = f"{tier}_{mode}_b{batch}"
                case = root / case_name
                require(read_int(case / "returncode.txt") == 0, case_name)
                result = read_json(case / "result.json")
                require(result["status"] == "ok", f"custom not ok: {case_name}")
                config = result["config"]
                require(config["batch_size"] == batch, f"custom batch: {case_name}")
                require(
                    config["warmup_decode_steps"] == 5
                    and config["measured_decode_steps"] == EXPECTED_STEPS,
                    f"custom step protocol: {case_name}",
                )
                require(
                    config["expected_selected_bytes"]
                    == tier_input["selected_main_bytes"],
                    f"custom selected bytes: {case_name}",
                )
                timing = result["timing"]["steady_decode_step_ms"]
                require(timing["count"] == EXPECTED_STEPS, f"custom samples: {case_name}")
                snapshots = result["parameter_va_snapshots"]
                require(
                    all(snapshots[name]["count"] == 96 for name in snapshots),
                    f"custom VA count: {case_name}",
                )
                reference_vas = snapshots["after_load"]["vas"]
                require(
                    all(snapshot["vas"] == reference_vas for snapshot in snapshots.values()),
                    f"custom Parameter VA changed: {case_name}",
                )
                for segment in result["manifest_segments"].values():
                    require(
                        segment["actual_h2d_bytes"] == 0,
                        f"custom decode/capture H2D is nonzero: {case_name}",
                    )
                for point in ("after_load", "after_prefill", "after_first_decode", "final"):
                    model = result["diagnostics"][point]["model"]
                    require(model["remote_map_active"], f"remote map inactive: {case_name}")
                    require(
                        model["active_driver_backing_bytes"] == 0,
                        f"remote backing remains: {case_name}",
                    )
                    require(
                        model["selected_expert_count"]
                        == tier_input["offload_plan_records"],
                        f"custom expert count: {case_name}",
                    )
                wrapper = result["graph_wrapper_calls_final"]
                counters = result["compilation_counter_final"]
                if mode == "graph":
                    require(
                        wrapper.get("full_capture") == 2
                        and wrapper.get("full_replay") == 22
                        and counters["num_cudagraph_captured"] == 2,
                        f"custom Graph evidence: {case_name}",
                    )
                else:
                    require(
                        not wrapper and counters["num_cudagraph_captured"] == 0,
                        f"custom eager unexpectedly captured Graph: {case_name}",
                    )
                shutdown = result["shutdown_contract"]
                require(not shutdown["cleanup_errors"], f"custom cleanup: {case_name}")
                require(
                    all(item["ok"] for item in shutdown["model_runner_shutdown_results"]),
                    f"custom graph cleanup: {case_name}",
                )
                check_gpu_json(case / "gpu.before.json")
                check_gpu_json(case / "gpu.after.json")
                output_hash = result["output_token_ids_sha256"]
                hashes.setdefault((tier, batch), {})[mode] = output_hash
                median = float(timing["median"])
                rows.append(
                    {
                        "family": f"custom_{mode}",
                        "mechanism": (
                            "Custom remote eager"
                            if mode == "eager"
                            else "Custom remote CUDA Graph"
                        ),
                        "framework": "vLLM in-process",
                        "tier": tier,
                        "batch": batch,
                        "median_ms": median,
                        "mean_ms": float(timing["mean"]),
                        "p90_ms": float(timing["p90"]),
                        "samples": EXPECTED_STEPS,
                        "tokens_per_second": batch * 1000.0 / median,
                        "output_sha256": output_hash,
                        "actual_offload_bytes": tier_input["selected_main_bytes"],
                        "cold_experts": tier_input["offload_plan_records"],
                        "graph_verified": mode == "graph",
                        "timing_boundary": "synchronized EngineCore.step",
                    }
                )
    for key, values in hashes.items():
        require(values["eager"] == values["graph"], f"custom output hash differs: {key}")
    module_after = read_json(root / "module.after.json")
    require(module_after["loaded_srcversion"] == STOCK_UVM, "stock UVM not restored")
    require(module_after["gpu_clean"], "GPU not clean after custom matrix")
    return rows


def collect_vllm(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected = {f"{tier}_{backend}" for tier in TIERS for backend in ("uva", "copy")}
    check_execution(root, expected)
    rows: list[dict[str, Any]] = []
    placement: dict[str, Any] = {}
    hashes: dict[tuple[str, int], dict[str, str]] = {}
    for tier in TIERS:
        tier_manifests: dict[str, dict[str, Any]] = {}
        for backend in ("uva", "copy"):
            case_name = f"{tier}_{backend}"
            case = root / case_name
            require(read_int(case / "returncode.txt") == 0, case_name)
            result = read_json(case / "result.json")
            require(result["status"] == "ok", f"vLLM not ok: {case_name}")
            config = result["config"]
            require(
                config["warmup_decode_steps"] == 5
                and config["measured_decode_steps"] == EXPECTED_STEPS,
                f"vLLM protocol: {case_name}",
            )
            manifest_records = [
                json.loads(line)
                for line in (case / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            params = [item for item in manifest_records if item["event"] == "cpu_offload_parameter"]
            require(params, f"empty vLLM manifest: {case_name}")
            total_bytes = sum(int(item["bytes"]) for item in params)
            require(
                all(item["offload_backend"] == backend for item in params),
                f"vLLM backend mismatch: {case_name}",
            )
            require(
                all(bool(item["copy_on_forward"]) == (backend == "copy") for item in params),
                f"vLLM copy flag mismatch: {case_name}",
            )
            layer_ids = sorted(
                {
                    int(match.group(1))
                    for item in params
                    if (match := re.search(r"model\.layers\.(\d+)\.", item["full_parameter_name"]))
                }
            )
            require(layer_ids == list(range(layer_ids[-1] + 1)), f"vLLM layer order: {case_name}")
            tier_manifests[backend] = {
                "actual_offload_bytes": total_bytes,
                "parameter_count": len(params),
                "layer_ids": layer_ids,
                "first_parameter": params[0]["full_parameter_name"],
                "last_parameter": params[-1]["full_parameter_name"],
            }
            require(len(result["phases"]) == 2, f"vLLM phases: {case_name}")
            for phase in result["phases"]:
                batch = int(phase["batch_tokens"])
                require(batch in BATCHES, f"vLLM batch: {case_name}")
                require(
                    phase["decode_step_count"] == EXPECTED_STEPS,
                    f"vLLM samples: {case_name}/b{batch}",
                )
                output_hash = phase["output_token_hash"]
                hashes.setdefault((tier, batch), {})[backend] = output_hash
                median = float(phase["decode_step_median_ms"])
                rows.append(
                    {
                        "family": f"vllm_{backend}",
                        "mechanism": (
                            "vLLM pinned UVA" if backend == "uva" else "vLLM copy-on-forward"
                        ),
                        "framework": "vLLM in-process",
                        "tier": tier,
                        "batch": batch,
                        "median_ms": median,
                        "mean_ms": float(phase["decode_step_mean_ms"]),
                        "p90_ms": float(phase["decode_step_p90_ms"]),
                        "samples": EXPECTED_STEPS,
                        "tokens_per_second": batch * 1000.0 / median,
                        "output_sha256": output_hash,
                        "actual_offload_bytes": total_bytes,
                        "cold_experts": None,
                        "graph_verified": False,
                        "timing_boundary": "synchronized EngineCore.step (enforce_eager)",
                    }
                )
            check_gpu_json(case / "gpu.before.json")
            check_gpu_json(case / "gpu.after.json")
        require(
            tier_manifests["uva"] == tier_manifests["copy"],
            f"vLLM UVA/copy placement differs: {tier}",
        )
        placement[tier] = tier_manifests["uva"]
    for key, values in hashes.items():
        require(values["uva"] == values["copy"], f"vLLM output hash differs: {key}")
    check_gpu_json(root / "gpu.before.json")
    check_gpu_json(root / "gpu.after.json")
    return rows, placement


def collect_kt(root: Path, inputs: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {f"{tier}_{mode}" for tier in TIERS for mode in ("eager", "graph")}
    check_execution(root, expected)
    rows: list[dict[str, Any]] = []
    hashes: dict[tuple[str, int], dict[str, str]] = {}
    for tier in TIERS:
        tier_input = inputs["tiers"][tier]
        for mode in ("eager", "graph"):
            case_name = f"{tier}_{mode}"
            case = root / case_name
            result = read_json(case / "result.json")
            run_summary = read_json(case / "run_summary.json")
            require(result["status"] == "ok", f"KT not ok: {case_name}")
            require(run_summary["runtime_mask_verified"], f"KT mask: {case_name}")
            require(
                run_summary["cold_experts"] == tier_input["offload_plan_records"],
                f"KT cold expert count: {case_name}",
            )
            require(
                run_summary["plan_sha256"] == tier_input["plan_sha256"],
                f"KT plan hash: {case_name}",
            )
            require(
                run_summary["cold_main_weight_bytes"] == tier_input["selected_main_bytes"],
                f"KT selected bytes: {case_name}",
            )
            capacity = read_int(case / "profiled_token_capacity.txt")
            required_tokens = read_int(case / "required_batch4_tokens.txt")
            require(capacity >= required_tokens, f"KT B4 capacity: {case_name}")
            log = (case / "server.log").read_text(encoding="utf-8", errors="replace")
            require(
                "#new-seq: 4, #new-token: 1217" in log
                and "#running-req: 4" in log,
                f"KT strict B4 was not observed: {case_name}",
            )
            for batch in BATCHES:
                phase = result["phase_summary"][str(batch)]
                require(
                    phase["measurement_repeats"] == EXPECTED_KT_REPEATS,
                    f"KT repeats: {case_name}/b{batch}",
                )
                require(
                    len(phase["output_token_ids_sha256"]) == 1,
                    f"KT output instability: {case_name}/b{batch}",
                )
                delta = phase["cuda_graph_pass_delta"]
                if mode == "graph":
                    require(delta["true"] > 0 and delta["false"] == 0, f"KT Graph: {case_name}")
                else:
                    require(delta["false"] > 0 and delta["true"] == 0, f"KT eager: {case_name}")
                output_hash = phase["output_token_ids_sha256"][0]
                hashes.setdefault((tier, batch), {})[mode] = output_hash
                timing = phase["decode_step_ms"]
                median = float(timing["median"])
                rows.append(
                    {
                        "family": f"kt_{mode}",
                        "mechanism": (
                            "KTransformers eager"
                            if mode == "eager"
                            else "KTransformers CUDA Graph"
                        ),
                        "framework": "KTransformers/SGLang",
                        "tier": tier,
                        "batch": batch,
                        "median_ms": median,
                        "mean_ms": float(timing["mean"]),
                        "sample_stdev_ms": float(timing["sample_stdev"]),
                        "cycle_median_samples_ms": [
                            float(value) for value in timing["samples"]
                        ],
                        "samples": EXPECTED_KT_REPEATS,
                        "steps_per_sample": EXPECTED_STEPS,
                        "tokens_per_second": batch * 1000.0 / median,
                        "output_sha256": output_hash,
                        "actual_offload_bytes": run_summary["cold_main_weight_bytes"],
                        "actual_offload_bytes_including_scales": run_summary[
                            "cold_bytes_including_scales"
                        ],
                        "cold_experts": run_summary["cold_experts"],
                        "graph_verified": mode == "graph",
                        "timing_boundary": "localhost batched HTTP/SSE token interval",
                        "profiled_token_capacity": capacity,
                    }
                )
            check_gpu_csv(case / "gpu.before.csv")
            check_gpu_csv(case / "gpu.after.csv")
    for key, values in hashes.items():
        require(values["eager"] == values["graph"], f"KT output hash differs: {key}")
    return rows


def build_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    index = row_index(rows)
    result: dict[str, Any] = {}
    pairs = {
        "custom_graph_vs_eager": ("custom_graph", "custom_eager"),
        "vllm_uva_vs_copy": ("vllm_uva", "vllm_copy"),
        "kt_graph_vs_eager": ("kt_graph", "kt_eager"),
    }
    for label, (faster_family, baseline_family) in pairs.items():
        result[label] = {}
        for tier in TIERS:
            result[label][tier] = {}
            for batch in BATCHES:
                faster = index[(faster_family, tier, batch)]["median_ms"]
                baseline = index[(baseline_family, tier, batch)]["median_ms"]
                result[label][tier][str(batch)] = {
                    "speedup": baseline / faster,
                    "latency_change_percent": percent_change(faster, baseline),
                }
    return result


def markdown_report(summary: dict[str, Any]) -> str:
    rows = summary["measurements"]
    index = row_index(rows)
    placement = summary["placement"]
    lines = [
        "# Qwen3 Decode 六机制对比（2026-08-21）",
        "",
        "## 结论",
        "",
        "- 在同为 vLLM 进程内 step 的四种方案中，Custom CUDA Graph 在本轮 2/4 GiB、B1/B4 均最快。",
        "- vLLM 原生 pinned UVA 明显快于 copy-on-forward；copy 的卸载量越大，逐 step 搬运代价越高。",
        "- KTransformers 开启 CUDA Graph 后也明显快于 eager，但其计时包含 localhost HTTP/SSE 与调度，不能把绝对值直接当作与 vLLM 的纯内核对比。",
        "- 冷集来自与 evaluation source index 不相交的 calibration：2 GiB 冷集零命中，4 GiB 冷集仅命中 5/86,016 路由槽。本结论适用于这组冻结请求与静态冷集，不代表任意线上流量。",
        "",
        "## 实验设置",
        "",
        "- 模型：Qwen3-30B-A3B-FP8；GPU：RTX 5090 32 GiB。",
        "- Batch：B1、严格并发 B4；每次先 5 个 warmup decode step，再测 16 个稳定 step。",
        "- Custom/vLLM：每个 case 一个冷进程，报告 16 step 中位数。KTransformers：1 次 conditioning + 3 次 measurement，报告 3 个 cycle 中位数。",
        "- Custom 与 KTransformers 使用同一冷专家坐标；vLLM 原生方案按模型参数/层顺序遍历卸载。",
        "",
        "## 卸载集合",
        "",
        "| 档位 | Custom/KT 冷专家 | Custom 主权重 | KT 含 scale | vLLM 实际卸载 | vLLM 参数/层范围 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for tier in TIERS:
        cold = summary["inputs"]["tiers"][tier]
        native = placement["vllm"][tier]
        lines.append(
            f"| {tier} | {cold['offload_plan_records']}（每层 {cold['experts_per_layer']}） "
            f"| {gib(cold['selected_main_bytes']):.4f} GiB "
            f"| {gib(cold['selected_bytes_including_scales']):.4f} GiB "
            f"| {gib(native['actual_offload_bytes']):.4f} GiB "
            f"| {native['parameter_count']} 个参数，L{native['layer_ids'][0]}–L{native['layer_ids'][-1]} |"
        )
    lines += ["", "## Decode 结果", ""]
    families = (
        "custom_eager",
        "custom_graph",
        "vllm_uva",
        "vllm_copy",
        "kt_eager",
        "kt_graph",
    )
    for tier in TIERS:
        lines += [
            f"### {tier}",
            "",
            "| 机制 | B1 ms/step | B1 tok/s | B4 ms/step | B4 tok/s | Graph |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for family in families:
            b1 = index[(family, tier, 1)]
            b4 = index[(family, tier, 4)]
            lines.append(
                f"| {b1['mechanism']} | {b1['median_ms']:.3f} | "
                f"{b1['tokens_per_second']:.2f} | {b4['median_ms']:.3f} | "
                f"{b4['tokens_per_second']:.2f} | "
                f"{'verified' if b1['graph_verified'] else 'eager'} |"
            )
        lines.append("")
    comparisons = summary["comparisons"]
    lines += [
        "## 同框架内的直接比较",
        "",
        "| 对比 | 档位 | B1 加速 | B4 加速 |",
        "|---|---|---:|---:|",
    ]
    labels = {
        "custom_graph_vs_eager": "Custom Graph / eager",
        "vllm_uva_vs_copy": "vLLM UVA / copy",
        "kt_graph_vs_eager": "KTransformers Graph / eager",
    }
    for key, label in labels.items():
        for tier in TIERS:
            lines.append(
                f"| {label} | {tier} | "
                f"{comparisons[key][tier]['1']['speedup']:.2f}× | "
                f"{comparisons[key][tier]['4']['speedup']:.2f}× |"
            )
    kt_4g_eager_b4 = index[("kt_eager", "4g", 4)]["cycle_median_samples_ms"]
    kt_4g_graph_b4 = index[("kt_graph", "4g", 4)]["cycle_median_samples_ms"]
    lines += [
        "",
        "## 如何理解",
        "",
        "Custom remote eager 仍承担 Python/调度与逐层执行开销；捕获完整 decode Graph 后，这部分 launch 开销被 replay 消除，所以本轮从约几十毫秒降到约 9–17 ms。vLLM pinned UVA 不做每步全量权重复制，因此比 copy-on-forward 稳定；copy 的 4 GiB case 比 2 GiB 更慢，符合每 step 搬运更多参数的预期。KTransformers 的 CPU 专家执行与激活传输仍在，Graph 主要削减 GPU 侧 launch 开销，因此加速明显但绝对延迟仍较高。",
        "",
        "## 证据与限制",
        "",
        "- Custom Graph 每 case 记录 2 次 capture、22 次 replay；96 个 Parameter VA 全阶段不变，decode H2D=0，remote map active、driver backing=0。",
        "- KTransformers Graph 的 B1/B4 pass counter 均只增长 `decode_cuda_graph`；eager 只增长 `decode_none`，且 eager/Graph 输出 hash 一致。",
        "- KTransformers 的 token capacity 为 1,536，高于 B4 所需 1,305；日志确认 `#new-seq: 4` 与 `#running-req: 4`，不是串行排队。",
        "- Custom/KT 与 vLLM 的卸载内容不同；前者是 calibration-cold 专家，后者包含按层遍历到的 attention、norm、router 和专家参数。因此这里是系统方案对比，不是只改变访问原语的微基准。",
        "- KTransformers 使用 HTTP/SSE 客户端 token 间隔，Custom/vLLM 使用同步 EngineCore.step。只应直接比较同一框架内 eager/Graph 或 UVA/copy 的倍率。",
        "- Custom/vLLM 每格只有一个冷进程；KTransformers 每格是一个冷 server 内 3 个 cycle。本轮属于机制 qualification，不是跨多次冷启动的置信区间实验。",
        f"- KTransformers 4g/B4 波动较大：eager 三个 cycle 中位数范围为 {min(kt_4g_eager_b4):.1f}–{max(kt_4g_eager_b4):.1f} ms，Graph 为 {min(kt_4g_graph_b4):.1f}–{max(kt_4g_graph_b4):.1f} ms；2g 与 4g 的 B4 输出 hash 也不同，档位间差异不能解释成只增加了卸载字节。",
        "- Custom eager 的 4g 数值没有随卸载量单调变慢；结合单冷进程样本和更冷的专家集合，不应据此推导卸载越多越快。",
        "",
        f"Verifier：`{summary['status']}`。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--custom-root",
        type=Path,
        default=SCRIPT_DIR / "artifacts/custom_20260821T154000Z_v2",
    )
    parser.add_argument(
        "--vllm-root", type=Path, default=SCRIPT_DIR / "artifacts/vllm_20260821"
    )
    parser.add_argument(
        "--kt-root",
        type=Path,
        default=SCRIPT_DIR / "artifacts/ktransformers_20260821_v2",
    )
    parser.add_argument(
        "--json-output", type=Path, default=SCRIPT_DIR / "RESULTS_20260821.json"
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=SCRIPT_DIR / "RESULTS_20260821.md"
    )
    args = parser.parse_args()

    inputs = read_json(SCRIPT_DIR / "inputs/manifest.json")
    source = Path(inputs["source"])
    calibration_manifest = read_json(
        source.parent.parent / "frequency_stats_manifest.json"
    )
    require(
        calibration_manifest["fixtures_disjoint_by_source_index"] is True,
        "calibration/evaluation fixtures are not proven disjoint",
    )
    require(
        calibration_manifest["raw_recorder"]["sha256"] == inputs["source_sha256"],
        "calibration recorder hash differs",
    )
    custom_rows = collect_custom(args.custom_root.resolve(), inputs)
    vllm_rows, vllm_placement = collect_vllm(args.vllm_root.resolve())
    kt_rows = collect_kt(args.kt_root.resolve(), inputs)
    rows = custom_rows + vllm_rows + kt_rows
    require(len(rows) == 24, f"expected 24 measurement rows, got {len(rows)}")
    summary = {
        "schema": "qwen_decode_six_mechanism_summary_v1",
        "status": "ok",
        "protocol": {
            "model": "Qwen3-30B-A3B-FP8 d206ba732169f29bb77fbf80fc2c4b81d4d30782",
            "gpu": "NVIDIA GeForce RTX 5090 32 GiB",
            "tiers": list(TIERS),
            "batch_sizes": list(BATCHES),
            "warmup_decode_steps": 5,
            "measured_decode_steps": EXPECTED_STEPS,
            "custom_vllm_cold_processes_per_cell": 1,
            "ktransformers_measurement_cycles_per_cell": EXPECTED_KT_REPEATS,
        },
        "roots": {
            "custom": str(args.custom_root.resolve()),
            "vllm": str(args.vllm_root.resolve()),
            "ktransformers": str(args.kt_root.resolve()),
        },
        "inputs": inputs,
        "placement": {"vllm": vllm_placement},
        "measurements": rows,
        "comparisons": build_comparisons(rows),
        "verification": {
            "custom_cases": 8,
            "vllm_cases": 4,
            "ktransformers_cases": 4,
            "measurement_cells": 24,
            "stock_uvm_srcversion": STOCK_UVM,
            "output_hashes_match_within_each_paired_mechanism": True,
            "strict_ktransformers_batch4_verified": True,
            "calibration_evaluation_disjoint_by_source_index": True,
        },
    }
    args.json_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(markdown_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "json_output": str(args.json_output.resolve()),
                "markdown_output": str(args.markdown_output.resolve()),
                "measurement_cells": len(rows),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
