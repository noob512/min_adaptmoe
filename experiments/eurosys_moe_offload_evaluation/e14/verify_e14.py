#!/usr/bin/env python3
"""Verify and summarize E14 eager pure remote-read UVA results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SCHEMA = "eurosys_e14_uva_layer_vs_cold_verification_v1"
ARMS = ("D", "W")
ARM_NAMES = {
    "D": "distributed_cold29",
    "W": "fixed_whole_layers_0_10",
}
BAD_KERNEL_MARKERS = (
    "xid",
    "fault_pte",
    "fault_pde",
    "mmu fault",
    "gpu page fault",
    "illegal address",
)


class VerificationError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def distribution(values: list[float]) -> dict[str, Any]:
    require(bool(values), "cannot summarize an empty distribution")
    return {
        "count": len(values),
        "samples": values,
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        require(isinstance(value, dict), f"{path}:{line_number}: non-object record")
        records.append(value)
    require(bool(records), f"{path}: empty manifest")
    return records


def check_uncaptured_steps(result: dict[str, Any], label: str) -> dict[str, int]:
    def require_uncaptured(delta: Any, phase: str) -> None:
        require(isinstance(delta, dict), f"{label}: {phase} graph audit absent")
        require(
            not any(
                int(value) != 0
                for key, value in delta.items()
                if key.endswith("_capture") or key.endswith("_replay")
            ),
            f"{label}: {phase} unexpectedly used a CUDA Graph: {delta}",
        )

    counters = {"prefill": 0, "decode": 0}
    cycles = result.get("cycles") or []
    require(bool(cycles), f"{label}: no cycles")
    for cycle in cycles:
        require_uncaptured(cycle.get("prefill_graph_wrapper_delta"), "Prefill")
        after_prefill = cycle.get("static_alias_diagnostics_after_prefill") or {}
        require(
            after_prefill.get("lifecycle_state") == "remote"
            and after_prefill.get("remote_map_active") is True,
            f"{label}: Prefill did not replay in the remote topology",
        )
        counters["prefill"] += 1
        for category in ("warmup_decode_steps", "measured_decode_steps"):
            for step in cycle.get(category) or []:
                require_uncaptured(step.get("graph_wrapper_delta"), "Decode")
                counters["decode"] += 1
        after_decode = cycle.get("static_alias_diagnostics_after_decode") or {}
        require(
            after_decode.get("lifecycle_state") == "remote"
            and after_decode.get("remote_map_active") is True,
            f"{label}: Decode did not replay in the remote topology",
        )
    return counters


def verify_case(
    case_dir: Path,
    symbol: str,
    protocol: dict[str, Any],
) -> tuple[
    dict[str, Any],
    dict[int, dict[str, float]],
    dict[int, list[str]],
    dict[str, int],
]:
    label = case_dir.name
    require((case_dir / "returncode.txt").read_text().strip() == "0", f"{label}: rc")
    result = load_json(case_dir / "result.json")
    require(result.get("status") == "ok", f"{label}: result status")
    config = result.get("config") or {}
    require(config.get("execution_mode") == "eager", f"{label}: mode")
    require(config.get("enforce_eager") is True, f"{label}: eager disabled")
    require(config.get("requested_compilation_mode") == "NONE", f"{label}: compile")
    require(config.get("realized_compilation_mode") == "NONE", f"{label}: realized compile")
    require(
        config.get("requested_cudagraph_mode") == "NONE",
        f"{label}: Graph request",
    )
    require(
        config.get("realized_cudagraph_mode") == "NONE",
        f"{label}: Graph realized",
    )
    require(
        sorted(config.get("realized_cudagraph_capture_sizes") or [])
        == protocol["capture_sizes"],
        f"{label}: capture sizes",
    )
    replay_counts = check_uncaptured_steps(result, label)
    require(
        result.get("compilation_counter_after_load", {}).get("num_cudagraph_captured") == 0
        and result.get("compilation_counter_final", {}).get("num_cudagraph_captured") == 0,
        f"{label}: a CUDA Graph was captured",
    )

    expected = protocol["arms"][symbol]
    expected_parameter_count = 96 if symbol == "D" else 22
    expected_module_count = int(expected["selected_layers"])
    expected_diagnostic_experts = expected["selected_expert_instances"]
    for where in ("static_alias_diagnostics_after_load", "static_alias_diagnostics_final"):
        diagnostics = result.get(where) or {}
        require(diagnostics.get("enabled") is True, f"{label}/{where}: UVA disabled")
        require(diagnostics.get("backend") == "uvm_original_va", f"{label}/{where}: backend")
        require(
            diagnostics.get("phase_mode") == "remote_only",
            f"{label}/{where}: phase",
        )
        require(diagnostics.get("lifecycle_state") == "remote", f"{label}/{where}: state")
        require(
            diagnostics.get("remote_map_active") is True,
            f"{label}/{where}: map",
        )
        require(diagnostics.get("remote_graph_captured") is False, f"{label}/{where}: remote Graph")
        require(diagnostics.get("capture_topology") is None, f"{label}/{where}: topology")
        require(diagnostics.get("alias_graph_captured") is False, f"{label}/{where}: alias Graph")
        require(int(diagnostics.get("capture_actual_h2d_bytes", -1)) == 0, f"{label}/{where}: H2D")
        require(int(diagnostics.get("active_driver_backing_bytes", -1)) == 0, f"{label}/{where}: backing live")
        require(int(diagnostics.get("arena_count", 0)) == 4, f"{label}/{where}: arenas")
        require(
            int(diagnostics.get("backing_bytes", -1))
            == int(expected["expected_alias_backing_bytes"]),
            f"{label}/{where}: backing geometry",
        )
        require(
            int(diagnostics.get("selected_parameter_count", -1)) == expected_parameter_count,
            f"{label}/{where}: parameter count",
        )
        require(
            int(diagnostics.get("selected_expert_count", -1)) == expected_diagnostic_experts,
            f"{label}/{where}: expert count",
        )
        require(
            int(diagnostics.get("selected_parameter_bytes_per_forward", -1))
            == int(expected["selected_main_bytes"]),
            f"{label}/{where}: selected bytes",
        )
        require(
            diagnostics.get("remote_graph_certificate") is None,
            f"{label}/{where}: unexpected Graph certificate",
        )
        require(
            len(diagnostics.get("original_vas") or []) == expected_parameter_count,
            f"{label}/{where}: original Parameter VAs",
        )
    require(
        result["static_alias_diagnostics_after_load"].get("original_vas")
        == result["static_alias_diagnostics_final"].get("original_vas"),
        f"{label}: Parameter VAs changed during uncaptured execution",
    )

    manifest = read_manifest(case_dir / "cpu_offload_manifest.jsonl")
    forwards = [item for item in manifest if item.get("event") == "cpu_offload_copy_forward"]
    require(not forwards, f"{label}: per-forward copy wrapper/trace was active")
    direct_forwards = [
        item
        for item in manifest
        if item.get("event") == "cpu_offload_uvm_original_va_direct_forward"
    ]
    require(
        len(direct_forwards) == expected_module_count,
        f"{label}: direct-forward module count",
    )
    require(
        all(
            item.get("phase_mode") == "remote_only"
            and item.get("module_forward_wrapped") is False
            for item in direct_forwards
        ),
        f"{label}: direct-forward certificate",
    )
    require(
        sum(int(item.get("selected_parameter_count", -1)) for item in direct_forwards)
        == expected_parameter_count,
        f"{label}: direct-forward parameter count",
    )
    require(
        sum(int(item.get("selected_parameter_bytes", -1)) for item in direct_forwards)
        == int(expected["selected_main_bytes"]),
        f"{label}: direct-forward selected bytes",
    )
    environment = (case_dir / "environment.txt").read_text(encoding="utf-8").splitlines()
    require("VLLM_UVM_REMOTE_ONLY_DIRECT_FORWARD=1" in environment, f"{label}: direct env")
    require("VLLM_CPU_OFFLOAD_COPY_TRACE=0" in environment, f"{label}: trace env")
    transitions = [
        item
        for item in manifest
        if item.get("event") == "cpu_offload_uvm_original_va_remote_transition"
    ]
    require(bool(transitions) and all(item.get("ok") is True for item in transitions), f"{label}: remote transition")

    gpu = load_json(case_dir / "gpu.after.json")
    require(gpu.get("gpu_clean") is True, f"{label}: GPU cleanup")
    kernel = (case_dir / "kernel.since.log").read_text(encoding="utf-8", errors="replace").lower()
    bad = [marker for marker in BAD_KERNEL_MARKERS if marker in kernel]
    require(not bad, f"{label}: kernel fault markers {bad}")

    metrics: dict[int, dict[str, float]] = {}
    hashes: dict[int, list[str]] = defaultdict(list)
    for batch_text, summary in (result.get("phase_summary") or {}).items():
        batch = int(batch_text)
        prefill_ms = float(summary["prefill_step_ms"]["median"])
        decode_ms = float(summary["decode_repeat_median_ms"]["median"])
        metrics[batch] = {
            "prefill_ms": prefill_ms,
            "prefill_tokens_per_second": float(summary["prompt_tokens"]) * 1_000.0 / prefill_ms,
            "decode_ms": decode_ms,
            "decode_tokens_per_second": batch * 1_000.0 / decode_ms,
        }
    for cycle in result.get("cycles") or []:
        if cycle.get("role") == "measurement":
            hashes[int(cycle["batch_size"])].append(str(cycle["output_token_ids_sha256"]))
    return result, metrics, dict(hashes), replay_counts


def build_report(root: Path) -> dict[str, Any]:
    protocol = load_json(root / "run_config.json")
    require(protocol.get("schema") == "eurosys_e14_uva_layer_vs_cold_protocol_v1", "bad protocol")
    rows = list(csv.DictReader((root / "execution_status.tsv").open(encoding="utf-8"), delimiter="\t"))
    require(bool(rows), "execution status is empty")
    require(all(row["returncode"] == "0" for row in rows), "a process returned non-zero")
    first_result = load_json(root / rows[0]["directory"] / "result.json")
    first_config = first_result.get("config") or {}
    actual_batches = sorted(int(value) for value in first_config.get("batch_sizes") or [])
    actual_capture_sizes = sorted(
        int(value) for value in first_config.get("realized_cudagraph_capture_sizes") or []
    )
    require(bool(actual_batches), "realized batch matrix is absent")
    require(actual_capture_sizes == [], "CUDA Graph capture sizes are not empty")
    protocol = dict(protocol)
    protocol["batch_sizes"] = actual_batches
    protocol["capture_sizes"] = actual_capture_sizes

    process_metrics: dict[str, dict[int, list[dict[str, float]]]] = {
        arm: defaultdict(list) for arm in ARMS
    }
    hashes: dict[int, list[str]] = defaultdict(list)
    process_receipts: list[dict[str, Any]] = []
    replay_totals = {"prefill": 0, "decode": 0}
    for row in rows:
        symbol = row["symbol"]
        require(symbol in ARMS, f"unknown arm {symbol}")
        case_dir = root / row["directory"]
        result, metrics, case_hashes, case_replays = verify_case(
            case_dir, symbol, protocol
        )
        for phase in replay_totals:
            replay_totals[phase] += int(case_replays[phase])
        for batch, values in metrics.items():
            process_metrics[symbol][batch].append(values)
        for batch, values in case_hashes.items():
            hashes[int(batch)].extend(values)
        process_receipts.append(
            {
                "ordinal": int(row["ordinal"]),
                "arm": symbol,
                "directory": row["directory"],
                "batch_order": [int(value) for value in row["batch_order"].split(",")],
                "load_seconds": float(result["load_seconds"]),
                "measurement_cycles": sum(
                    item.get("role") == "measurement" for item in result.get("cycles") or []
                ),
            }
        )

    batches = actual_batches
    for arm in ARMS:
        require(set(process_metrics[arm]) == set(batches), f"{arm}: batch coverage")
    require(all(len(set(hashes[batch])) == 1 for batch in batches), "cross-arm output hash mismatch")

    aggregate: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for batch in batches:
            records = process_metrics[arm][batch]
            aggregate[arm][str(batch)] = {
                metric: distribution([float(item[metric]) for item in records])
                for metric in (
                    "prefill_ms",
                    "prefill_tokens_per_second",
                    "decode_ms",
                    "decode_tokens_per_second",
                )
            }

    comparisons: list[dict[str, Any]] = []
    for batch in batches:
        d = aggregate["D"][str(batch)]
        w = aggregate["W"][str(batch)]
        comparisons.append(
            {
                "batch_size": batch,
                "distributed_over_whole_prefill_throughput": d["prefill_tokens_per_second"]["median"] / w["prefill_tokens_per_second"]["median"],
                "distributed_over_whole_decode_throughput": d["decode_tokens_per_second"]["median"] / w["decode_tokens_per_second"]["median"],
                "whole_over_distributed_prefill_latency": w["prefill_ms"]["median"] / d["prefill_ms"]["median"],
                "whole_over_distributed_decode_latency": w["decode_ms"]["median"] / d["decode_ms"]["median"],
            }
        )

    scaling: dict[str, dict[str, dict[str, float]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        prefill_b1 = aggregate[arm]["1"]["prefill_tokens_per_second"]["median"]
        decode_b1 = aggregate[arm]["1"]["decode_tokens_per_second"]["median"]
        for batch in batches:
            scaling[arm][str(batch)] = {
                "prefill_throughput_over_b1": aggregate[arm][str(batch)]["prefill_tokens_per_second"]["median"] / prefill_b1,
                "prefill_scaling_efficiency": aggregate[arm][str(batch)]["prefill_tokens_per_second"]["median"] / (batch * prefill_b1),
                "decode_throughput_over_b1": aggregate[arm][str(batch)]["decode_tokens_per_second"]["median"] / decode_b1,
                "decode_scaling_efficiency": aggregate[arm][str(batch)]["decode_tokens_per_second"]["median"] / (batch * decode_b1),
            }

    low = comparisons[0]
    high = comparisons[-1]
    d_high_scale = scaling["D"][str(batches[-1])]
    w_high_scale = scaling["W"][str(batches[-1])]
    claim_gate = {
        "low_concurrency_within_25_percent_both_phases": (
            0.75 <= low["distributed_over_whole_prefill_throughput"] <= 1.25
            and 0.75 <= low["distributed_over_whole_decode_throughput"] <= 1.25
        ),
        "distributed_b32_at_least_15_percent_faster_both_phases": (
            high["distributed_over_whole_prefill_throughput"] >= 1.15
            and high["distributed_over_whole_decode_throughput"] >= 1.15
        ),
        "distributed_b32_scaling_efficiency_at_least_90_percent_both_phases": (
            d_high_scale["prefill_scaling_efficiency"] >= 0.90
            and d_high_scale["decode_scaling_efficiency"] >= 0.90
        ),
        "whole_layer_b32_scaling_efficiency_at_most_75_percent_both_phases": (
            w_high_scale["prefill_scaling_efficiency"] <= 0.75
            and w_high_scale["decode_scaling_efficiency"] <= 0.75
        ),
        "distributed_b32_prefill_scaling_efficiency": d_high_scale["prefill_scaling_efficiency"],
        "distributed_b32_decode_scaling_efficiency": d_high_scale["decode_scaling_efficiency"],
        "whole_layer_b32_prefill_scaling_efficiency": w_high_scale["prefill_scaling_efficiency"],
        "whole_layer_b32_decode_scaling_efficiency": w_high_scale["decode_scaling_efficiency"],
        "prefill_d_over_w_ratio_increase_b1_to_b32": (
            high["distributed_over_whole_prefill_throughput"]
            - low["distributed_over_whole_prefill_throughput"]
        ),
        "decode_d_over_w_ratio_increase_b1_to_b32": (
            high["distributed_over_whole_decode_throughput"]
            - low["distributed_over_whole_decode_throughput"]
        ),
    }
    claim_gate["requested_pattern_observed"] = all(
        claim_gate[key]
        for key in (
            "low_concurrency_within_25_percent_both_phases",
            "distributed_b32_at_least_15_percent_faster_both_phases",
            "distributed_b32_scaling_efficiency_at_least_90_percent_both_phases",
            "whole_layer_b32_scaling_efficiency_at_most_75_percent_both_phases",
        )
    )

    arm_counts = {arm: sum(row["symbol"] == arm for row in rows) for arm in ARMS}
    formal_counts = protocol["formal"]["independent_processes_per_arm"]
    report = {
        "schema": SCHEMA,
        "status": "ok",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "run_root": str(root.resolve()),
        "protocol": {
            "batch_sizes": batches,
            "capture_sizes": protocol["capture_sizes"],
            "mechanism": protocol["execution"]["mechanism"],
            "phase_mode": protocol["execution"]["phase_mode"],
            "remote_only_direct_forward": protocol["execution"]["remote_only_direct_forward"],
            "per_forward_copy_trace": protocol["execution"]["per_forward_copy_trace"],
            "cudagraph_mode": protocol["execution"]["cudagraph_mode"],
            "enforce_eager": protocol["execution"]["enforce_eager"],
            "arm_process_counts": arm_counts,
            "formal_process_count_met": all(value == formal_counts for value in arm_counts.values()),
            "offload_gib": {
                arm: protocol["arms"][arm]["selected_main_gib"] for arm in ARMS
            },
            "offload_byte_difference_fraction": protocol["fairness"]["offload_byte_difference_fraction_of_distributed"],
        },
        "process_receipts": process_receipts,
        "uncaptured_step_totals": replay_totals,
        "output_hash_by_batch": {str(batch): hashes[batch][0] for batch in batches},
        "aggregate_process_medians": aggregate,
        "comparisons": comparisons,
        "scaling": scaling,
        "claim_gate": claim_gate,
        "cleanup": {
            "stock_uvm_restored": False,
            "final_gpu_clean": False,
        },
    }
    before = load_json(root / "module.before.json")
    after = load_json(root / "module.after.json")
    report["cleanup"] = {
        "stock_uvm_restored": (
            before.get("loaded_srcversion") == after.get("loaded_srcversion")
            and before.get("module_sha256") == after.get("module_sha256")
        ),
        "final_gpu_clean": after.get("gpu_clean") is True,
    }
    require(all(report["cleanup"].values()), "module/GPU cleanup failed")
    return report


def write_csv(root: Path, report: dict[str, Any]) -> None:
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "batch",
                "arm",
                "prefill_ms",
                "prefill_tok_s",
                "decode_ms",
                "decode_tok_s",
                "prefill_scaling_efficiency",
                "decode_scaling_efficiency",
            ]
        )
        for batch in report["protocol"]["batch_sizes"]:
            for arm in ARMS:
                metrics = report["aggregate_process_medians"][arm][str(batch)]
                scale = report["scaling"][arm][str(batch)]
                writer.writerow(
                    [
                        batch,
                        ARM_NAMES[arm],
                        f'{metrics["prefill_ms"]["median"]:.6f}',
                        f'{metrics["prefill_tokens_per_second"]["median"]:.6f}',
                        f'{metrics["decode_ms"]["median"]:.6f}',
                        f'{metrics["decode_tokens_per_second"]["median"]:.6f}',
                        f'{scale["prefill_scaling_efficiency"]:.6f}',
                        f'{scale["decode_scaling_efficiency"]:.6f}',
                    ]
                )


def write_markdown(root: Path, report: dict[str, Any]) -> None:
    table = [
        "| B | placement | Prefill ms | Prefill tok/s | Decode ms/step | Decode tok/s |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for batch in report["protocol"]["batch_sizes"]:
        for arm in ARMS:
            metrics = report["aggregate_process_medians"][arm][str(batch)]
            table.append(
                f'| {batch} | {ARM_NAMES[arm]} | '
                f'{metrics["prefill_ms"]["median"]:.2f} | '
                f'{metrics["prefill_tokens_per_second"]["median"]:.2f} | '
                f'{metrics["decode_ms"]["median"]:.2f} | '
                f'{metrics["decode_tokens_per_second"]["median"]:.2f} |'
            )
    ratios = [
        "| B | D/W Prefill throughput | D/W Decode throughput |",
        "|---:|---:|---:|",
    ]
    for item in report["comparisons"]:
        ratios.append(
            f'| {item["batch_size"]} | '
            f'{item["distributed_over_whole_prefill_throughput"]:.3f}× | '
            f'{item["distributed_over_whole_decode_throughput"]:.3f}× |'
        )
    scaling = [
        "| B | D Prefill efficiency | W Prefill efficiency | D Decode efficiency | W Decode efficiency |",
        "|---:|---:|---:|---:|---:|",
    ]
    for batch in report["protocol"]["batch_sizes"]:
        d = report["scaling"]["D"][str(batch)]
        w = report["scaling"]["W"][str(batch)]
        scaling.append(
            f'| {batch} | {d["prefill_scaling_efficiency"]:.3f} | '
            f'{w["prefill_scaling_efficiency"]:.3f} | '
            f'{d["decode_scaling_efficiency"]:.3f} | '
            f'{w["decode_scaling_efficiency"]:.3f} |'
        )
    gate = report["claim_gate"]
    text = "\n".join(
        [
            "# E14：纯 UVA 下逐层冷专家与固定整层卸载的并发扩展性",
            "",
            f'状态：`{report["status"]}`；目标趋势：`{gate["requested_pattern_observed"]}`。',
            "",
            "两臂均使用 Qwen3-30B-A3B-FP8、Short-128 冻结请求、original-VA managed UVA "
            "`remote_read`、`CUDAGraphMode.NONE` 和 `enforce_eager=True`。Prefill 和 "
            "Decode 均使用无 H2D 的 eager UVA 执行。逐层臂卸载 "
            "cold-29（6.117 GiB），整层臂卸载固定 L0--L10（6.188 GiB），字节差 1.15%。",
            "两臂均启用 remote-only direct forward，初始化完成后直接执行原始 "
            "`module.forward`；逐 forward copy wrapper/trace 已关闭。",
            "",
            "## 主结果",
            "",
            *table,
            "",
            "## 相对性能",
            "",
            *ratios,
            "",
            "低并发同量级门（B1 两阶段均在 ±25%）："
            f' `{gate["low_concurrency_within_25_percent_both_phases"]}`。',
            "",
            "高并发优势门（B32 两阶段逐层臂均至少快 15%）："
            f' `{gate["distributed_b32_at_least_15_percent_faster_both_phases"]}`。',
            "",
            "",
            "## 相对 B1 的线性扩展效率",
            "",
            *scaling,
            "",
            "B32 扩展效率门：逐层臂两阶段均不低于 90%："
            f' `{gate["distributed_b32_scaling_efficiency_at_least_90_percent_both_phases"]}`；'
            "整层臂两阶段均不高于 75%："
            f' `{gate["whole_layer_b32_scaling_efficiency_at_most_75_percent_both_phases"]}`。',
            "",
            "从 B1 到 B32，D/W 吞吐倍率的增量为："
            f'Prefill +{gate["prefill_d_over_w_ratio_increase_b1_to_b32"]:.3f} '
            f'（{100 * gate["prefill_d_over_w_ratio_increase_b1_to_b32"]:.1f} 个百分点），'
            f'Decode +{gate["decode_d_over_w_ratio_increase_b1_to_b32"]:.3f} '
            f'（{100 * gate["decode_d_over_w_ratio_increase_b1_to_b32"]:.1f} 个百分点）。',
            "",
            f'未捕获执行：Prefill {report["uncaptured_step_totals"]["prefill"]} 次，'
            f'Decode {report["uncaptured_step_totals"]["decode"]} 次；全部输出 hash 跨臂一致。',
            "`CUDAGraphMode.NONE` 且 `enforce_eager=True`；无服务期 H2D copy、"
            "CUDA Graph capture、Xid/MMU/PTE/PDE fault；"
            "结束后恢复 stock UVM 且 GPU 为 0 MiB。",
            "",
            "统计值为独立冷进程中位数的跨进程中位数；进程数见 `verification.json`。",
            "",
        ]
    )
    (root / "RESULTS_E14.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    root = args.run_root.resolve()
    report = build_report(root)
    if args.write_outputs:
        (root / "verification.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_csv(root, report)
        write_markdown(root, report)
    print(json.dumps({"status": report["status"], "claim_gate": report["claim_gate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
