#!/usr/bin/env python3
"""Verify and summarize selected arms of the ShareGPT scheduler comparison."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


EXPECTED_MODE = {
    "adaptive": "adaptmoe_copy_eager_continuous",
    "native_uva": "native_uva_eager_continuous",
    "native_copy": "native_copy_eager_continuous",
}


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for value in [json.loads(line)]
        if isinstance(value, dict)
    ]


def hashes(result: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["logical_request"]): str(item["output_token_ids_sha256"])
        for item in result.get("requests", [])
    }


def mean_metric(summary: dict[str, Any], name: str) -> float:
    return float(summary[name]["mean"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    config = load(args.run_root / "run_config.json")
    expected_arms = config.get("arms", list(EXPECTED_MODE))
    if (
        not expected_arms
        or len(set(expected_arms)) != len(expected_arms)
        or any(arm not in EXPECTED_MODE for arm in expected_arms)
    ):
        raise SystemExit("invalid arm selection in run configuration")
    expected_samples = int(config["samples_per_arm"])
    expected_requests = int(config["requests"])
    expected_output_tokens = int(config["output_tokens_per_request"])
    expected_adaptive_bytes = int(config["adaptive_selected_bytes"])
    expected_native_bytes = int(config["native_expected_bytes"])
    errors: list[str] = []
    warnings: list[str] = []
    cases: list[dict[str, Any]] = []

    rows = (args.run_root / "execution_status.tsv").read_text(
        encoding="utf-8"
    ).splitlines()[1:]
    for row in rows:
        if not row.strip():
            continue
        ordinal_text, sample_text, arm, directory, rc_text = row.split("\t")
        case_dir = args.run_root / "arms" / directory
        result = load(case_dir / "result.json")
        manifest = load_jsonl(case_dir / "cpu_offload_manifest.jsonl")
        summary = result.get("summary", {})
        case_errors: list[str] = []

        if int(rc_text) != 0 or result.get("status") != "ok":
            case_errors.append("runner did not complete successfully")
        if result.get("mode") != EXPECTED_MODE.get(arm):
            case_errors.append(f"unexpected mode {result.get('mode')!r}")
        if arm not in expected_arms:
            case_errors.append(f"arm {arm!r} was not selected in the protocol")
        if result.get("configuration", {}).get("compilation_config") != {
            "mode": "NONE",
            "cudagraph_mode": "NONE",
        }:
            case_errors.append("CUDA Graph was not fully disabled")
        if summary.get("successful_requests") != expected_requests:
            case_errors.append("not all configured requests completed")
        if summary.get("generated_tokens") != expected_requests * expected_output_tokens:
            case_errors.append("generated token total differs from configuration")
        result_config = result.get("configuration", {})
        if result_config.get("arrival_qps") != config.get("arrival_qps"):
            case_errors.append("runner arrival QPS differs from protocol")
        if config.get("arrival_qps") is not None:
            if result_config.get("latency_origin") != "planned_arrival":
                case_errors.append("timed arrival latency omits admission delay")
            offsets = [item.get("planned_arrival_offset_seconds") for item in result.get("requests", [])]
            expected_offsets = config["planned_arrival_offsets_seconds"]
            if len(offsets) != len(expected_offsets) or any(
                actual is None or abs(actual - expected) > 1e-6
                for actual, expected in zip(offsets, expected_offsets)
            ):
                case_errors.append("planned request arrival offsets differ")
        result_to_protocol_key = {
            "initial_requests": "initial_requests",
            "late_requests": "late_requests",
            "late_after_engine_steps": "late_after_engine_steps",
            "output_tokens": "output_tokens_per_request",
            "kv_cache_memory_bytes": "kv_cache_memory_bytes",
            "max_model_len": "max_model_len",
            "max_num_batched_tokens": "max_num_batched_tokens",
            "max_num_seqs": "max_num_seqs",
        }
        for key, protocol_key in result_to_protocol_key.items():
            if int(result_config.get(key, -1)) != int(config[protocol_key]):
                case_errors.append(f"runner configuration differs for {key}")

        request_prompt_hashes = [
            str(item.get("prompt_token_ids_sha256"))
            for item in result.get("requests", [])
        ]
        if request_prompt_hashes != config["prompt_token_hashes"]:
            case_errors.append("request prompts differ from frozen ShareGPT input")

        mixed = [
            step
            for step in result.get("scheduler_steps", [])
            if step.get("mixed_new_and_cached") is True
        ]
        scheduled_totals = [
            int(step.get("total_num_scheduled_tokens", -1))
            for step in result.get("scheduler_steps", [])
        ]
        if not mixed and config.get("arrival_qps") is None:
            case_errors.append("continuous scheduler emitted no mixed step")
        if not scheduled_totals or any(
            total < 0 or total > int(config["max_num_batched_tokens"])
            for total in scheduled_totals
        ):
            case_errors.append("scheduled token totals exceed configured bucket")
        if not any(total >= int(config["resume_token_threshold"]) for total in scheduled_totals):
            case_errors.append("trace never entered the high-token regime")
        if not any(total <= int(config["park_token_threshold"]) for total in scheduled_totals):
            case_errors.append("trace never entered the low-token regime")

        expected_parks = 0
        expected_resumes = 0
        expected_remote = False
        for total in scheduled_totals:
            if total <= int(config["park_token_threshold"]) and not expected_remote:
                expected_parks += 1
                expected_remote = True
            elif total >= int(config["resume_token_threshold"]) and expected_remote:
                expected_resumes += 1
                expected_remote = False

        configs = [
            item for item in manifest if item.get("event") == "cpu_offload_config"
        ]
        offload_config = configs[0] if len(configs) == 1 else {}
        parameters = [
            item
            for item in manifest
            if item.get("event") == "cpu_offload_parameter"
        ]
        parameter_bytes = sum(int(item.get("bytes", 0)) for item in parameters)
        parameter_names = [str(item.get("full_parameter_name")) for item in parameters]
        parks: list[dict[str, Any]] = []
        resumes: list[dict[str, Any]] = []
        measured_copy_bytes = 0

        window = result.get("measurement_window", {})
        started = float(window.get("started_epoch", 0))
        finished = float(window.get("finished_epoch", 0))
        forwards = [
            item
            for item in manifest
            if item.get("event") == "cpu_offload_copy_forward"
            and started <= float(item.get("timestamp", -1)) <= finished
        ]

        if arm == "adaptive":
            finalized = [
                item
                for item in manifest
                if item.get("event") == "cpu_offload_uvm_static_alias_finalized"
            ]
            selected_bytes = (
                int(finalized[0].get("cold_residency_bytes", 0))
                if len(finalized) == 1
                else 0
            )
            if not (
                offload_config.get("copy_exact_name_count") == 52
                and offload_config.get("graph_static_alias") is True
                and offload_config.get("uvm_original_va_phase_mode")
                == "token_threshold"
                and offload_config.get("uvm_token_threshold_fast_transition")
                is True
                and int(
                    offload_config.get("uvm_original_va_token_threshold", -1)
                )
                == int(config["park_token_threshold"])
                and int(
                    offload_config.get(
                        "uvm_original_va_resume_token_threshold", -1
                    )
                )
                == int(config["resume_token_threshold"])
                and selected_bytes == expected_adaptive_bytes
            ):
                case_errors.append("adaptive configuration/byte certificate differs")
            parks = [
                item
                for item in manifest
                if item.get("event")
                == "cpu_offload_uvm_original_va_remote_transition"
                and item.get("reason") == "token_threshold_remote"
                and item.get("ok") is True
            ]
            resumes = [
                item
                for item in manifest
                if item.get("event")
                == "cpu_offload_uvm_original_va_alias_prefill_activated"
                and item.get("pre_state") == "remote_parked"
                and item.get("ok") is True
            ]
            runtime_gates = [
                item
                for item in manifest
                if item.get("event")
                == "cpu_offload_uvm_hot_cuda_read_gate_begin"
                and str(item.get("gate_stage", "")).startswith(
                    ("runtime_remote_", "runtime_prefill_")
                )
            ]
            if len(parks) != expected_parks or len(resumes) != expected_resumes:
                case_errors.append(
                    "adaptive topology transitions differ from token replay: "
                    f"expected {expected_parks} PARK/{expected_resumes} RESUME, "
                    f"got {len(parks)} PARK/{len(resumes)} RESUME"
                )
            if runtime_gates:
                case_errors.append("adaptive runtime read gate was not eliminated")
            if any(
                item.get("fast_runtime_transition") is not True
                or item.get("post_map_physical_audit_enabled") is not False
                for item in parks
            ):
                case_errors.append("adaptive PARK did not use the fast path")
            measured_copy_bytes = sum(
                int(item.get("graph_static_alias_actual_h2d_bytes", 0))
                for item in forwards
            )
        elif arm in {"native_uva", "native_copy"}:
            expected_backend = "uva" if arm == "native_uva" else "copy"
            if not (
                offload_config.get("copy_exact_name_count") == 9
                and offload_config.get("graph_static_alias") is False
                and parameter_bytes == expected_native_bytes
                and parameters
                and all(
                    item.get("offload_backend") == expected_backend
                    for item in parameters
                )
            ):
                case_errors.append(
                    f"{arm} parameter selection/backend certificate differs"
                )
            if arm == "native_copy":
                measured_copy_bytes = sum(
                    int(item.get("copy_bytes", 0)) for item in forwards
                )
                expected_copies = int(summary.get("engine_steps", 0)) * parameter_bytes
                if measured_copy_bytes != expected_copies:
                    case_errors.append(
                        "native copy measurement does not equal bytes x steps"
                    )
            elif forwards:
                case_errors.append("native UVA unexpectedly emitted copy-forward records")
        else:
            case_errors.append(f"unknown arm {arm!r}")

        kernel_log = (case_dir / "kernel.since.log").read_text(
            encoding="utf-8", errors="replace"
        )
        if any(
            token in kernel_log
            for token in ("Xid ", "FAULT_PDE", "FAULT_PTE", "BUG:", "Oops:")
        ):
            case_errors.append("kernel/GPU fault was recorded")
        gpu_after = load(case_dir / "gpu.after.json")
        if gpu_after.get("gpu_clean") is not True:
            recheck_path = case_dir / "gpu.after.recheck.json"
            recheck = load(recheck_path) if recheck_path.is_file() else {}
            if (
                gpu_after.get("memory_used_mib") == 0
                and gpu_after.get("compute_pids") == []
                and recheck.get("gpu_clean") is True
                and float(recheck.get("timestamp", 0)) >= float(gpu_after.get("timestamp", 0))
            ):
                warnings.append(f"{directory}: post-exit transient UVM clients {gpu_after.get('uvm_client_pids')}; retained original snapshot, clean recheck passed")
            else:
                case_errors.append("GPU was not clean after the arm")

        errors.extend(f"{directory}: {message}" for message in case_errors)
        cases.append(
            {
                "ordinal": int(ordinal_text),
                "sample": int(sample_text),
                "arm": arm,
                "directory": directory,
                "parameter_count": len(parameters),
                "offloaded_bytes": (
                    expected_adaptive_bytes if arm == "adaptive" else parameter_bytes
                ),
                "parameter_names": parameter_names,
                "measured_copy_bytes": measured_copy_bytes,
                "request_throughput_per_second": float(
                    summary.get("request_throughput_per_second", 0)
                ),
                "token_throughput_per_second": float(
                    summary.get("token_throughput_per_second", 0)
                ),
                "mean_ttft_ms": mean_metric(summary, "ttft_ms"),
                "mean_tpot_ms": mean_metric(summary, "tpot_ms"),
                "mean_e2e_ms": mean_metric(summary, "e2e_ms"),
                "mixed_steps": len(mixed),
                "engine_steps": summary.get("engine_steps"),
                "max_running_requests": max((len(step.get("running_before", [])) + len(step.get("scheduled_new_request_ids", [])) for step in result.get("scheduler_steps", [])), default=0),
                "mean_admission_delay_ms": summary.get("admission_delay_ms", {}).get("mean"),
                "output_hashes": hashes(result),
                "park_wall_ms": [
                    float(item["wall_ns"]) / 1e6 for item in parks
                ],
                "resume_wall_ms": [
                    float(item["wall_ns"]) / 1e6 for item in resumes
                ],
                "errors": case_errors,
            }
        )

    for arm in expected_arms:
        count = sum(case["arm"] == arm for case in cases)
        if count != expected_samples:
            errors.append(f"{arm}: expected {expected_samples} samples, got {count}")

    for sample in range(1, expected_samples + 1):
        group = [case for case in cases if case["sample"] == sample]
        if len(group) == len(expected_arms):
            reference = group[0]["output_hashes"]
            if any(case["output_hashes"] != reference for case in group[1:]):
                warnings.append(f"sample {sample}: cross-arm output hashes differ")
            native = [case for case in group if case["arm"].startswith("native_")]
            if len(native) == 2 and native[0]["parameter_names"] != native[1]["parameter_names"]:
                errors.append(f"sample {sample}: native parameter sets differ")

    aggregates: dict[str, dict[str, float | int]] = {}
    for arm in expected_arms:
        selected = [case for case in cases if case["arm"] == arm]
        aggregates[arm] = {
            "samples": len(selected),
            "offloaded_bytes": int(selected[0]["offloaded_bytes"]) if selected else 0,
            "request_throughput_per_second": statistics.fmean(
                case["request_throughput_per_second"] for case in selected
            ),
            "token_throughput_per_second": statistics.fmean(
                case["token_throughput_per_second"] for case in selected
            ),
            "mean_ttft_ms": statistics.fmean(
                case["mean_ttft_ms"] for case in selected
            ),
            "mean_tpot_ms": statistics.fmean(
                case["mean_tpot_ms"] for case in selected
            ),
            "mean_e2e_ms": statistics.fmean(
                case["mean_e2e_ms"] for case in selected
            ),
        }

    adaptive = aggregates.get("adaptive", {})
    comparisons: dict[str, dict[str, float]] = {}
    for native_arm in ("native_uva", "native_copy"):
        native = aggregates.get(native_arm, {})
        if adaptive and native:
            comparisons[f"adaptive_vs_{native_arm}"] = {
                "throughput_ratio": float(
                    adaptive["request_throughput_per_second"]
                )
                / float(native["request_throughput_per_second"]),
                "ttft_ratio": float(adaptive["mean_ttft_ms"])
                / float(native["mean_ttft_ms"]),
                "tpot_ratio": float(adaptive["mean_tpot_ms"])
                / float(native["mean_tpot_ms"]),
                "e2e_ratio": float(adaptive["mean_e2e_ms"])
                / float(native["mean_e2e_ms"]),
            }

    module_after = load(args.run_root / "module.after.json")
    if module_after.get("gpu_clean") is not True:
        errors.append("final stock UVM snapshot is not GPU-clean")
    status = "error" if errors else "ok_with_output_drift" if warnings else "ok"
    verification = {
        "schema": "adaptmoe_token_threshold_sharegpt_compare_verification_v1",
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "dataset": {
            "path": config["sharegpt_dataset"],
            "sha256": config["sharegpt_dataset_sha256"],
            "source_domain": config["source_domain"],
            "fixture_sha256": config["fixture_sha256"],
            "source_indices": config["source_indices"],
        },
        "protocol": config,
        "cases": cases,
        "aggregate": aggregates,
        "comparison": comparisons,
        "strict_cross_arm_output_equivalence": not any("output hashes differ" in warning for warning in warnings),
    }
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    labels = {
        "adaptive": "Adaptive PARK/RESUME",
        "native_uva": "vLLM native UVA",
        "native_copy": "vLLM copy-on-forward",
    }
    lines = [
        f"# {' vs '.join(labels[arm] for arm in expected_arms)}（ShareGPT）",
        "",
        f"状态：`{status}`。每臂 {expected_samples} 个独立进程；{config['arm_order']}。",
        "",
        "| 机制 | 卸载 GiB | req/s | token/s | TTFT ms | TPOT ms | E2E ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in expected_arms:
        item = aggregates[arm]
        lines.append(
            f"| {labels[arm]} | {int(item['offloaded_bytes']) / (1 << 30):.3f} | "
            f"{float(item['request_throughput_per_second']):.3f} | "
            f"{float(item['token_throughput_per_second']):.3f} | "
            f"{float(item['mean_ttft_ms']):.2f} | "
            f"{float(item['mean_tpot_ms']):.2f} | "
            f"{float(item['mean_e2e_ms']):.2f} |"
        )
    lines += [
        "",
        f"各机制使用完全相同的 {expected_requests} 个冻结 ShareGPT prompt"
        f"（每个 {int(config['prompt_tokens_per_request'])} token）、mixed continuous "
        "scheduler、KV 预算和 eager 执行。Adaptive 使用 286 个 expert slices；"
        "Native 使用固定完整 Parameter 集合。",
        "",
        "Adaptive 的 PARK/RESUME 仅在跨阈值时发生；native UVA 始终远程读取，"
        "native copy-on-forward 每轮重新搬运完整卸载参数。",
        "",
    ]
    if warnings:
        lines.extend(["验证告警：", "", *[f"- {item}" for item in warnings], ""])
    if config.get("arrival_qps") is not None:
        lines += [
            f"固定间隔到达：{config['arrival_qps']} req/s，offset=i/QPS，共 {expected_requests} 条请求。",
            "TTFT/E2E 从计划到达时刻计时，包含引擎 step 边界接收延迟；TPOT 为首 token 后的平均 token 间隔。",
            "实际 QPS=完成请求数/从首个计划到达到末请求完成的时间，包含尾部排空；有限 trace 不代表稳态容量。",
            "低负载不强制产生 mixed step；是否重叠由相同到达序列和各机制执行速度共同决定。",
            "",
        ]
    args.report.write_text("\n".join(lines), encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
