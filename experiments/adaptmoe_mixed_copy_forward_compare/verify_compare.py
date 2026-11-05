#!/usr/bin/env python3
"""Verify and summarize the eager mixed-schedule copy-on-forward comparison."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


EXPECTED_BYTES = {
    "adaptmoe": 4_948_230_144,
    "native": 4_798_283_776,
}
EXPECTED_MODE = {
    "adaptmoe": "adaptmoe_copy_eager_continuous",
    "native": "native_copy_eager_continuous",
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} is not a JSON object")
        values.append(value)
    return values


def hashes(result: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["logical_request"]): str(item["output_token_ids_sha256"])
        for item in result.get("requests", [])
    }


def mean_metric(result: dict[str, Any], name: str) -> float:
    value = result["summary"][name]["mean"]
    return float(value)


def ratio(left: float, right: float) -> float:
    return left / right


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    config = load(args.run_root / "run_config.json")
    reference = load(args.reference)
    reference_hashes = hashes(reference)
    if reference.get("status") != "ok" or len(reference_hashes) != 8:
        errors.append("remote reference is not a complete successful 8-request run")

    status_rows = (args.run_root / "execution_status.tsv").read_text(
        encoding="utf-8"
    ).splitlines()
    cases: list[dict[str, Any]] = []
    for row in status_rows[1:]:
        if not row.strip():
            continue
        ordinal_text, sample_text, arm, directory, returncode_text = row.split("\t")
        case_dir = args.run_root / "arms" / directory
        case_errors: list[str] = []
        result_path = case_dir / "result.json"
        if not result_path.is_file():
            errors.append(f"{directory}: result.json is missing")
            continue
        result = load(result_path)
        manifest = load_jsonl(case_dir / "cpu_offload_manifest.jsonl")
        kernel_log = (case_dir / "kernel.since.log").read_text(
            encoding="utf-8", errors="replace"
        )
        summary = result.get("summary", {})

        if int(returncode_text) != 0:
            case_errors.append(f"runner return code is {returncode_text}")
        if result.get("status") != "ok":
            case_errors.append(f"runner status is {result.get('status')!r}")
        if result.get("mode") != EXPECTED_MODE.get(arm):
            case_errors.append(f"unexpected runner mode {result.get('mode')!r}")
        compilation = result.get("configuration", {}).get("compilation_config", {})
        if compilation != {"mode": "NONE", "cudagraph_mode": "NONE"}:
            case_errors.append("execution was not uniformly eager")
        if summary.get("successful_requests") != 8:
            case_errors.append("not all eight requests completed")
        if summary.get("generated_tokens") != 128:
            case_errors.append("generated token count is not 8 x 16")
        if summary.get("engine_steps") != 20:
            case_errors.append(f"expected 20 engine steps, got {summary.get('engine_steps')}")

        mixed = [
            step
            for step in result.get("scheduler_steps", [])
            if step.get("mixed_new_and_cached") is True
        ]
        if len(mixed) != 1:
            case_errors.append(f"expected one mixed step, got {len(mixed)}")
        elif not (
            len(mixed[0].get("scheduled_new_request_ids", [])) == 4
            and len(mixed[0].get("scheduled_cached_request_ids", [])) == 4
            and mixed[0].get("total_num_scheduled_tokens") == 516
        ):
            case_errors.append("mixed step is not 4x128 Prefill + 4x1 Decode")

        configs = [item for item in manifest if item.get("event") == "cpu_offload_config"]
        if len(configs) != 1:
            case_errors.append(f"expected one offload config record, got {len(configs)}")
            offload_config: dict[str, Any] = {}
        else:
            offload_config = configs[0]
        if arm == "adaptmoe":
            if not (
                offload_config.get("copy_exact_name_count") == 52
                and offload_config.get("graph_static_alias") is True
                and offload_config.get("graph_static_alias_backend") == "uvm_original_va"
                and offload_config.get("graph_static_alias_selected_expert_count") == 286
                and offload_config.get("uvm_original_va_phase_mode") == "off"
            ):
                case_errors.append("AdaptMoE manifest configuration differs")
            finalized = [
                item
                for item in manifest
                if item.get("event") == "cpu_offload_uvm_static_alias_finalized"
            ]
            if len(finalized) != 1 or not (
                finalized[0].get("ok") is True
                and finalized[0].get("state") == "sealed"
                and finalized[0].get("cold_residency_bytes")
                == EXPECTED_BYTES["adaptmoe"]
            ):
                case_errors.append("AdaptMoE sealed/cold-residency certificate differs")
        elif arm == "native":
            if not (
                offload_config.get("copy_exact_name_count") == 9
                and offload_config.get("graph_static_alias") is False
                and offload_config.get("uvm_original_va_phase_mode") == "off"
            ):
                case_errors.append("native copy-on-forward manifest configuration differs")
            parameters = [
                item for item in manifest if item.get("event") == "cpu_offload_parameter"
            ]
            native_parameter_bytes = sum(int(item.get("bytes", 0)) for item in parameters)
            if len(parameters) != 9 or native_parameter_bytes != EXPECTED_BYTES["native"]:
                case_errors.append(
                    "native offloaded-parameter set is not the frozen 9-Parameter plan"
                )
        else:
            case_errors.append(f"unknown arm {arm!r}")

        window = result.get("measurement_window", {})
        started = window.get("started_epoch")
        finished = window.get("finished_epoch")
        if not isinstance(started, (int, float)) or not isinstance(
            finished, (int, float)
        ):
            case_errors.append("runner did not record a measurement window")
            measured_forwards: list[dict[str, Any]] = []
        else:
            measured_forwards = [
                item
                for item in manifest
                if item.get("event") == "cpu_offload_copy_forward"
                and started <= float(item.get("timestamp", -1)) <= finished
            ]
        byte_field = (
            "graph_static_alias_actual_h2d_bytes"
            if arm == "adaptmoe"
            else "copy_bytes"
        )
        measured_bytes = sum(int(item.get(byte_field, 0)) for item in measured_forwards)
        expected_trace_bytes = int(summary.get("engine_steps", 0)) * EXPECTED_BYTES.get(
            arm, 0
        )
        if measured_bytes != expected_trace_bytes:
            case_errors.append(
                f"measured copy bytes differ: expected {expected_trace_bytes}, "
                f"got {measured_bytes}"
            )
        if not measured_forwards:
            case_errors.append("no measured copy-on-forward records")

        remote_events = [
            item
            for item in manifest
            if item.get("event")
            in {
                "cpu_offload_uvm_original_va_remote_transition",
                "cpu_offload_uvm_original_va_profile_all_remote_prepared",
                "cpu_offload_uvm_original_va_phase_poisoned",
            }
        ]
        if remote_events:
            case_errors.append("arm unexpectedly entered a remote/poisoned state")
        fault_tokens = ("Xid ", "FAULT_PDE", "FAULT_PTE", "BUG:", "Oops:")
        if any(token in kernel_log for token in fault_tokens):
            case_errors.append("kernel log contains a GPU/kernel fault")
        if load(case_dir / "gpu.after.json").get("gpu_clean") is not True:
            case_errors.append("GPU/UVM client state was not clean after the arm")

        case_hashes = hashes(result)
        matching_reference = sum(
            case_hashes.get(request) == digest
            for request, digest in reference_hashes.items()
        )
        if case_hashes != reference_hashes:
            warnings.append(
                f"{directory}: {matching_reference}/8 output hashes match remote reference"
            )
        errors.extend(f"{directory}: {message}" for message in case_errors)
        cases.append(
            {
                "ordinal": int(ordinal_text),
                "sample": int(sample_text),
                "arm": arm,
                "directory": directory,
                "status": "error" if case_errors else "ok",
                "errors": case_errors,
                "mixed_step_count": len(mixed),
                "engine_steps": summary.get("engine_steps"),
                "measured_forward_record_count": len(measured_forwards),
                "bytes_per_engine_step": (
                    measured_bytes / int(summary["engine_steps"])
                    if summary.get("engine_steps")
                    else None
                ),
                "measured_trace_bytes": measured_bytes,
                "request_throughput_per_second": summary.get(
                    "request_throughput_per_second"
                ),
                "mean_ttft_ms": mean_metric(result, "ttft_ms"),
                "mean_tpot_ms": mean_metric(result, "tpot_ms"),
                "mean_e2e_ms": mean_metric(result, "e2e_ms"),
                "output_hashes": case_hashes,
                "matching_remote_reference_hashes": matching_reference,
            }
        )

    expected_case_count = int(config.get("paired_samples", 0)) * 2
    if len(cases) != expected_case_count:
        errors.append(f"expected {expected_case_count} arm runs, found {len(cases)}")

    samples = sorted({case["sample"] for case in cases})
    paired_hash_matches: list[int] = []
    paired_ratios: dict[str, list[float]] = {
        "throughput": [],
        "ttft": [],
        "tpot": [],
        "e2e": [],
    }
    for sample in samples:
        by_arm = {case["arm"]: case for case in cases if case["sample"] == sample}
        if set(by_arm) != {"adaptmoe", "native"}:
            errors.append(f"sample {sample}: missing one paired arm")
            continue
        adapt_hashes = by_arm["adaptmoe"]["output_hashes"]
        native_hashes = by_arm["native"]["output_hashes"]
        matching = sum(
            adapt_hashes.get(request) == digest
            for request, digest in native_hashes.items()
        )
        paired_hash_matches.append(matching)
        if adapt_hashes != native_hashes:
            warnings.append(
                f"sample {sample}: AdaptMoE and native hashes match for {matching}/8 requests"
            )
        paired_ratios["throughput"].append(
            ratio(
                float(by_arm["adaptmoe"]["request_throughput_per_second"]),
                float(by_arm["native"]["request_throughput_per_second"]),
            )
        )
        for key in ("ttft", "tpot", "e2e"):
            paired_ratios[key].append(
                ratio(
                    float(by_arm["adaptmoe"][f"mean_{key}_ms"]),
                    float(by_arm["native"][f"mean_{key}_ms"]),
                )
            )

    aggregates: dict[str, dict[str, float]] = {}
    for arm in ("adaptmoe", "native"):
        arm_cases = [case for case in cases if case["arm"] == arm]
        aggregates[arm] = {
            "samples": len(arm_cases),
            "mean_request_throughput_per_second": statistics.fmean(
                float(case["request_throughput_per_second"]) for case in arm_cases
            ),
            "mean_ttft_ms": statistics.fmean(
                float(case["mean_ttft_ms"]) for case in arm_cases
            ),
            "mean_tpot_ms": statistics.fmean(
                float(case["mean_tpot_ms"]) for case in arm_cases
            ),
            "mean_e2e_ms": statistics.fmean(
                float(case["mean_e2e_ms"]) for case in arm_cases
            ),
        }

    comparison = {
        "adaptmoe_over_native_mean_paired_throughput_ratio": statistics.fmean(
            paired_ratios["throughput"]
        ),
        "adaptmoe_over_native_mean_paired_ttft_ratio": statistics.fmean(
            paired_ratios["ttft"]
        ),
        "adaptmoe_over_native_mean_paired_tpot_ratio": statistics.fmean(
            paired_ratios["tpot"]
        ),
        "adaptmoe_over_native_mean_paired_e2e_ratio": statistics.fmean(
            paired_ratios["e2e"]
        ),
        "paired_output_hash_matches": paired_hash_matches,
        "strict_paired_output_equivalence": bool(paired_hash_matches)
        and all(value == 8 for value in paired_hash_matches),
        "adaptmoe_bytes_per_forward": EXPECTED_BYTES["adaptmoe"],
        "native_bytes_per_forward": EXPECTED_BYTES["native"],
        "native_byte_shortfall_percent": (
            1.0 - EXPECTED_BYTES["native"] / EXPECTED_BYTES["adaptmoe"]
        )
        * 100.0,
    }

    module_after = load(args.run_root / "module.after.json")
    if module_after.get("gpu_clean") is not True:
        errors.append("final stock-module snapshot is not GPU-clean")

    status = "error" if errors else "ok_with_output_drift" if warnings else "ok"
    verification = {
        "schema": "adaptmoe_mixed_copy_forward_compare_verification_v1",
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "claim_scope": (
            "paired small-workload qualification; near-equal HBM/bytes, "
            "not equal expert coordinates"
        ),
        "cases": cases,
        "aggregate": aggregates,
        "comparison": comparison,
    }
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    throughput_ratio = comparison[
        "adaptmoe_over_native_mean_paired_throughput_ratio"
    ]
    report_lines = [
        "# Mixed continuous：AdaptMoE 搬运 vs vLLM copy-on-forward",
        "",
        f"状态：`{status}`。这是 {len(samples)} 组 AB/BA 配对的小负载资格实验。",
        "",
        "两臂均使用 continuous scheduler、eager 执行（CUDA Graph 关闭）、"
        "8 个请求（4 个初始 + 第 4 step 后注入 4 个）、128-token prompt、"
        "16-token output 和相同 KV cache。每次均实际产生一个 "
        "`4x128 Prefill + 4x1 Decode` mixed step。",
        "",
        "| Arm | 吞吐 req/s | TTFT ms | TPOT ms | E2E ms | 每 forward 搬运 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, label in (("adaptmoe", "AdaptMoE 固定搬运"), ("native", "vLLM 原生 copy-on-forward")):
        item = aggregates[arm]
        report_lines.append(
            f"| {label} | {item['mean_request_throughput_per_second']:.3f} | "
            f"{item['mean_ttft_ms']:.2f} | {item['mean_tpot_ms']:.2f} | "
            f"{item['mean_e2e_ms']:.2f} | {EXPECTED_BYTES[arm] / 1e9:.3f} GB |"
        )
    report_lines += [
        "",
        f"AdaptMoE/native 的配对平均吞吐比为 `{throughput_ratio:.3f}` "
        f"（{(throughput_ratio - 1.0) * 100.0:+.1f}%）；TPOT 比为 "
        f"`{comparison['adaptmoe_over_native_mean_paired_tpot_ratio']:.3f}x`。",
        "",
        f"逐组 AdaptMoE/native 输出 hash 匹配数：`{paired_hash_matches}`；"
        f"严格输出等价：`{comparison['strict_paired_output_equivalence']}`。",
        "",
        "公平性边界：AdaptMoE 选择分散在 26 层的 286 个 expert slices，"
        "每 forward 搬运 4,948,230,144 B；原生 vLLM 只能按完整 Parameter "
        "卸载，本实验选择 9 个 Parameters、4,798,283,776 B，少 3.03%。"
        "因此这是 near-equal HBM/bytes 对比，不是完全相同 expert 坐标，"
        "也不是 offered-load 扫描。",
        "",
    ]
    if warnings:
        report_lines.append("输出告警：")
        report_lines.append("")
        report_lines.extend(f"- {warning}" for warning in warnings)
        report_lines.append("")
    args.report.write_text("\n".join(report_lines), encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
