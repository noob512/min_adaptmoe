#!/usr/bin/env python3
"""Verify one tuned KTransformers round and compare current vLLM controls.

The historical filename is retained so existing artifact command lines remain valid.  The
current protocol deliberately accepts exactly one cold-process sample per coordinate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


EXPECTED = {
    ("short_dialogue", 1): (128, 32, "cold10", 118),
    ("short_dialogue", 16): (128, 32, "cold10", 118),
    ("ordinary_chat", 1): (512, 128, "cold10", 118),
    ("ordinary_chat", 16): (512, 128, "cold38", 90),
    ("medium_context", 1): (2048, 128, "cold10", 118),
    ("medium_context", 16): (2048, 128, "cold38", 90),
    ("rag_summary", 1): (8192, 128, "cold38", 90),
    ("rag_summary", 16): (8192, 128, "cold89", 39),
    ("long_generation", 1): (512, 512, "cold10", 118),
    ("long_generation", 16): (512, 512, "cold38", 90),
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def token_hash(token_ids: list[list[int]]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def dist(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def control_cells(path: Path, key: str) -> dict[tuple[str, int], dict[str, Any]]:
    document = load(path)
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for cell in document.get("cells", []):
        arm = cell.get(key)
        if arm and all(
            finite_positive(arm.get("metrics", {}).get(name))
            for name in ("prefill_ms", "decode_ms", "e2e_ms")
        ):
            result[(cell["workload"], int(cell["batch"]))] = {
                "metrics": arm["metrics"],
                "offload": cell.get("offload"),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kt-root", type=Path, action="append", required=True)
    parser.add_argument("--native-uva-summary", type=Path, required=True)
    parser.add_argument("--native-copy-summary", type=Path, required=True)
    parser.add_argument("--old-kt-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    expected_repeats = 1
    if len(args.kt_root) != expected_repeats:
        errors.append(
            f"expected {expected_repeats} KT root, got {len(args.kt_root)}"
        )
    samples: dict[tuple[str, int], list[dict[str, Any]]] = {
        coordinate: [] for coordinate in EXPECTED
    }
    for repeat, root in enumerate(args.kt_root, start=1):
        root_config_path = root / "configuration.json"
        plan_manifest_path = root / "provenance/kt_plans/manifest.json"
        if not root_config_path.is_file() or not plan_manifest_path.is_file():
            errors.append(f"repeat {repeat}: root configuration/plan manifest missing")
        else:
            root_config = load(root_config_path)
            if not (
                root_config.get("cpu_threads") == 7
                and root_config.get("process_cpuset") == "0-13"
                and root_config.get("repeats") == 1
                and root_config.get("cuda_graph") is True
                and root_config.get("mixed_chunk") is False
                and root_config.get("overlap_schedule_enabled") is True
            ):
                errors.append(f"repeat {repeat}: root protocol gate")
            plan_manifest = load(plan_manifest_path)
            plans = {
                Path(item.get("input_plan", "")).name: item
                for item in plan_manifest.get("plans", [])
            }
            plan_checks = []
            for name, cold, hot in (
                ("cold_decode_plan_2g.json", 480, 5664),
                ("cold_decode_plan_8g.json", 1824, 4320),
                ("cold89.json", 4272, 1872),
            ):
                item = plans.get(name, {})
                plan_checks.append(
                    item.get("cold_experts") == cold
                    and item.get("hot_experts") == hot
                    and item.get("kt_topk_mask_verified_equal_to_hot_plan") is True
                )
            if not (
                plan_manifest.get("num_moe_layers") == 48
                and plan_manifest.get("num_experts_per_layer") == 128
                and all(plan_checks)
            ):
                errors.append(f"repeat {repeat}: frequency-plan provenance gate")
        execution = root / "execution.tsv"
        if not execution.is_file():
            errors.append(f"repeat {repeat}: missing execution.tsv")
            continue
        rows = list(csv.DictReader(execution.open(encoding="utf-8"), delimiter="\t"))
        if len(rows) != len(EXPECTED):
            errors.append(f"repeat {repeat}: execution rows={len(rows)}")
        for row in rows:
            coordinate = (row["workload"], int(row["batch"]))
            if coordinate not in EXPECTED:
                errors.append(f"repeat {repeat}: unexpected coordinate {coordinate}")
                continue
            prompt, output_tokens, tier, gpu_experts = EXPECTED[coordinate]
            prefix = f"repeat {repeat} {coordinate[0]}/B{coordinate[1]}"
            if row["status"] != "ok" or int(row["returncode"]) != 0:
                errors.append(f"{prefix}: execution failed")
                continue
            if row["tier"] != tier or int(row["gpu_experts"]) != gpu_experts:
                errors.append(f"{prefix}: placement differs")
            case = root / "cases" / row["case_dir"]
            result_path = case / "result.json"
            if not result_path.is_file():
                errors.append(f"{prefix}: result missing")
                continue
            result = load(result_path)
            config = result.get("configuration", {})
            cells = result.get("cells", [])
            checks = [
                result.get("status") == "ok",
                result.get("schema") == "eurosys_e3_ktransformers_direct_batch_v3",
                config.get("prompt_tokens") == prompt,
                config.get("output_tokens") == output_tokens,
                config.get("batch_sizes") == [coordinate[1]],
                config.get("cpu_threads") == 7,
                config.get("gpu_prefill_token_threshold")
                == (
                    16384 if coordinate == ("medium_context", 16)
                    else 4096 if coordinate == ("rag_summary", 16)
                    else 0
                ),
                config.get("gpu_experts_per_layer") == gpu_experts,
                config.get("prefill_attention_backend")
                == ("triton" if coordinate == ("rag_summary", 16) else "flashinfer"),
                config.get("decode_attention_backend")
                == ("triton" if coordinate == ("rag_summary", 16) else "flashinfer"),
                config.get("cuda_graph_enabled") is True,
                config.get("cuda_graph_batch_sizes") == sorted({1, coordinate[1]}),
                config.get("overlap_schedule_enabled") is True,
                config.get("mixed_chunk_enabled") is False,
                config.get("prefill_covers_full_batch") is True,
                config.get("measurement_repeats") == 1,
                config.get("transport") == "none",
                len(cells) == 1,
            ]
            if not all(checks):
                errors.append(f"{prefix}: protocol/configuration gate")
                continue
            cell = cells[0]
            measured = cell.get("measured_decode_step_ms", {})
            if not (
                cell.get("successful_requests") == coordinate[1]
                and cell.get("request_count") == coordinate[1]
                and cell.get("batch_size") == coordinate[1]
                and cell.get("single_generate_call") is True
                and cell.get("all_requests_submitted_together") is True
                and cell.get("mixed_chunk_enabled") is False
                and cell.get("transport") == "none"
                and cell.get("p_to_d_switch_ms") is None
                and measured.get("count") == output_tokens - 7
                and finite_positive(cell.get("prefill_step_ms"))
                and finite_positive(measured.get("p50"))
                and finite_positive(cell.get("cycle_e2e_ms"))
                and finite_positive(cell.get("decode_tokens_per_second"))
            ):
                errors.append(f"{prefix}: measurement gate")
                continue
            output_ids = cell.get("output_token_ids", [])
            if not (
                len(output_ids) == coordinate[1]
                and all(len(request) == output_tokens for request in output_ids)
                and cell.get("output_token_ids_sha256") == token_hash(output_ids)
            ):
                errors.append(f"{prefix}: output length/hash gate")
                continue
            log = (case / "run.log").read_text(encoding="utf-8", errors="replace")
            if "Capture cuda graph begin" not in log or "cuda graph: True" not in log:
                errors.append(f"{prefix}: graph capture/replay evidence")
            if log.count(f"has {gpu_experts} GPU experts") != 48:
                errors.append(f"{prefix}: 48-layer placement audit")
            kernel_path = case / "kernel.since.log"
            kernel = kernel_path.read_text(encoding="utf-8", errors="replace") if kernel_path.is_file() else ""
            lowered = kernel.lower()
            if any(marker in lowered for marker in ("xid", "mmu fault", "fault_pde", "fault_pte")):
                errors.append(f"{prefix}: kernel fault")
            if (case / "gpu_cleanup_error.txt").exists():
                errors.append(f"{prefix}: GPU cleanup")
            samples[coordinate].append({
                "repeat": repeat,
                "prefill_ms": float(cell["prefill_step_ms"]),
                "decode_ms": float(measured["p50"]),
                "e2e_ms": float(cell["cycle_e2e_ms"]),
                "decode_tokens_per_second": float(cell["decode_tokens_per_second"]),
                "output_hash": cell["output_token_ids_sha256"],
            })

    native_uva = control_cells(args.native_uva_summary, "native")
    native_copy = control_cells(args.native_copy_summary, "native_copy")
    old_kt_document = load(args.old_kt_verification)
    old_kt = {
        (cell["workload"], int(cell["batch"])): cell["ktransformers"]
        for cell in old_kt_document.get("metrics", [])
    }
    cells_out: list[dict[str, Any]] = []
    for coordinate, expected in EXPECTED.items():
        values = samples[coordinate]
        if len(values) != expected_repeats:
            errors.append(f"{coordinate[0]}/B{coordinate[1]}: KT samples={len(values)}")
            continue
        point = {
            name: float(values[0][name])
            for name in ("prefill_ms", "decode_ms", "e2e_ms", "decode_tokens_per_second")
        }
        item: dict[str, Any] = {
            "workload": coordinate[0],
            "batch": coordinate[1],
            "prompt_tokens": expected[0],
            "output_tokens": expected[1],
            "tier": expected[2],
            "gpu_experts_per_layer": expected[3],
            "cpu_experts_per_layer": 128 - expected[3],
            "ktransformers": {
                "samples": values,
                "point_metrics": {
                    **point,
                    "prefill_tokens_per_second": (
                        expected[0] * coordinate[1] * 1000.0 / point["prefill_ms"]
                    ),
                },
                "prefill_ms": dist([float(x["prefill_ms"]) for x in values]),
                "decode_ms": dist([float(x["decode_ms"]) for x in values]),
                "e2e_ms": dist([float(x["e2e_ms"]) for x in values]),
                "decode_tokens_per_second": dist([
                    float(x["decode_tokens_per_second"]) for x in values
                ]),
                "output_hash_valid": True,
                "statistical_scope": "single cold-process point estimate",
            },
        }
        old = old_kt.get(coordinate)
        item["old_ktransformers_14t_triton"] = old
        item["tuned_speedup_over_old_ktransformers"] = (
            {
                "prefill": old["prefill_ms"] / point["prefill_ms"],
                "decode": old["steady_decode_ms"] / point["decode_ms"],
                "e2e": old["e2e_ms"] / point["e2e_ms"],
            }
            if old else None
        )
        for label, controls in (("vllm_native_uva", native_uva), ("vllm_copy", native_copy)):
            control = controls.get(coordinate)
            item[label] = control
            item[f"ktransformers_over_{label}_latency"] = (
                {
                    "prefill": point["prefill_ms"] / control["metrics"]["prefill_ms"],
                    "decode": point["decode_ms"] / control["metrics"]["decode_ms"],
                    "e2e": point["e2e_ms"] / control["metrics"]["e2e_ms"],
                }
                if control else None
            )
        cells_out.append(item)

    output = {
        "schema": "eurosys_e3_ktransformers_tuned_single_round_v1",
        "status": "ok" if not errors else "error",
        "errors": errors,
        "protocol": {
            "model": "Qwen3-30B-A3B-FP8",
            "same_frozen_request_token_ids": True,
            "ktransformers_repeats": 1,
            "ktransformers_statistical_scope": "single cold-process point estimate per cell",
            "ktransformers_cpu_expert_threads": 7,
            "ktransformers_process_cpuset": "0-13",
            "ktransformers_frequency_placement": True,
            "ktransformers_attention_policy": (
                "FlashInfer except RAG/B16, which uses TRITON for capacity"
            ),
            "ktransformers_gpu_prefill_policy": (
                "Medium/B16 threshold=16384; RAG/B16 threshold=4096; "
                "all other cells threshold=0"
            ),
            "ktransformers_decode_cuda_graph": True,
            "ktransformers_prefill_cuda_graph": False,
            "ktransformers_overlap_schedule": True,
            "ktransformers_mixed_chunk": False,
            "vllm_controls_repeats": 1,
            "vllm_controls_attention_backend": "TRITON_ATTN",
            "vllm_controls_cudagraph_mode": "FULL_SEPARATE",
            "comparison_boundary": (
                "System-level comparison on the same frozen requests. KT uses FlashInfer "
                "and exact per-expert CPU placement; vLLM controls use TRITON_ATTN and "
                "whole-Parameter offload with nearby realized bytes. Ratios are not a "
                "single-mechanism ablation, and vLLM controls have one cold process."
            ),
        },
        "roots": [str(path.resolve()) for path in args.kt_root],
        "controls": {
            "native_uva": str(args.native_uva_summary.resolve()),
            "copy": str(args.native_copy_summary.resolve()),
            "old_ktransformers": str(args.old_kt_verification.resolve()),
        },
        "cells": cells_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
