#!/usr/bin/env python3
"""Verify and summarize the D-aligned E3 KTransformers C1/C16 baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def token_hash(token_ids: list[list[int]]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def peak_hbm_mib(path: Path) -> float | None:
    values: list[float] = []
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) > 1:
            try:
                values.append(float(fields[1]))
            except ValueError:
                pass
    return max(values) if values else None


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def d_result_path(root: Path, workload: str, batch: int) -> Path:
    return (
        root
        / "ours"
        / f"{workload}_b{batch}_D"
        / "custom"
        / f"b0_{workload}_b{batch}_D"
        / "result.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--supplement-root", action="append", type=Path, default=[])
    parser.add_argument("--d-reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    roots = [args.root, *args.supplement_root]
    selected_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for root in roots:
        source_rows = list(csv.DictReader(
            (root / "execution.tsv").open(encoding="utf-8"), delimiter="\t"
        ))
        for row in source_rows:
            coordinate = (row["workload"], int(row["batch"]))
            row["_root"] = str(root)
            previous = selected_rows.get(coordinate)
            if previous is None:
                selected_rows[coordinate] = row
            elif previous["status"] != "ok" and row["status"] == "ok":
                selected_rows[coordinate] = row
            else:
                errors.append(f"duplicate successful coordinate: {coordinate}")
    rows = list(selected_rows.values())
    require(len(rows) == 10, f"selected execution row count is {len(rows)}, expected 10", errors)
    seen = {(row["workload"], int(row["batch"])) for row in rows}
    require(seen == set(EXPECTED), "execution coordinates differ", errors)

    plan_manifest = load_json(args.root / "provenance/kt_plans/manifest.json")
    plans = {Path(item["input_plan"]).name: item for item in plan_manifest["plans"]}
    for plan_name, cold, hot in (
        ("cold_decode_plan_2g.json", 480, 5664),
        ("cold_decode_plan_8g.json", 1824, 4320),
        ("cold89.json", 4272, 1872),
    ):
        item = plans.get(plan_name, {})
        require(item.get("cold_experts") == cold, f"{plan_name}: cold count", errors)
        require(item.get("hot_experts") == hot, f"{plan_name}: hot count", errors)
        require(
            item.get("kt_topk_mask_verified_equal_to_hot_plan") is True,
            f"{plan_name}: exact top-k mask was not verified",
            errors,
        )

    metrics: list[dict[str, Any]] = []
    for row in rows:
        workload = row["workload"]
        batch = int(row["batch"])
        prompt, output, tier, gpu_experts = EXPECTED[(workload, batch)]
        prefix = f"{workload}/B{batch}"
        require(row["status"] == "ok", f"{prefix}: execution status", errors)
        require(int(row["returncode"]) == 0, f"{prefix}: return code", errors)
        require(row["tier"] == tier, f"{prefix}: tier", errors)
        require(int(row["gpu_experts"]) == gpu_experts, f"{prefix}: gpu experts", errors)
        case = Path(row["_root"]) / "cases" / row["case_dir"]
        result_path = case / "result.json"
        if not result_path.is_file():
            errors.append(f"{prefix}: result is absent")
            continue
        result = load_json(result_path)
        config = result.get("configuration", {})
        cells = result.get("cells", [])
        require(result.get("status") == "ok", f"{prefix}: result status", errors)
        require(result.get("schema") == "eurosys_e3_ktransformers_direct_batch_v3", f"{prefix}: schema", errors)
        require(config.get("batch_sizes") == [batch], f"{prefix}: batch config", errors)
        require(config.get("prompt_tokens") == prompt, f"{prefix}: prompt config", errors)
        require(config.get("output_tokens") == output, f"{prefix}: output config", errors)
        require(config.get("gpu_experts_per_layer") == gpu_experts, f"{prefix}: placement config", errors)
        require(config.get("attention_backend") == "triton", f"{prefix}: backend", errors)
        require(config.get("cuda_graph_enabled") is True, f"{prefix}: graph mode", errors)
        require(
            config.get("cuda_graph_batch_sizes") == sorted({1, batch}),
            f"{prefix}: graph capture sizes",
            errors,
        )
        require(config.get("mixed_chunk_enabled") is False, f"{prefix}: mixed chunk", errors)
        require(config.get("prefill_covers_full_batch") is True, f"{prefix}: prefill budget", errors)
        if len(cells) != 1:
            errors.append(f"{prefix}: cell count")
            continue
        cell = cells[0]
        ids = cell.get("output_token_ids", [])
        require(cell.get("single_generate_call") is True, f"{prefix}: dispatch", errors)
        require(cell.get("all_requests_submitted_together") is True, f"{prefix}: admission", errors)
        require(cell.get("successful_requests") == batch, f"{prefix}: success count", errors)
        require(len(ids) == batch and all(len(item) == output for item in ids), f"{prefix}: output lengths", errors)
        require(cell.get("output_token_ids_sha256") == token_hash(ids), f"{prefix}: output hash", errors)
        require(finite_positive(cell.get("prefill_step_ms")), f"{prefix}: prefill timing", errors)
        require(cell.get("p_to_d_switch_ms") is None, f"{prefix}: P-to-D must be N/A", errors)
        require(finite_positive(cell.get("first_decode_step_ms")), f"{prefix}: first decode timing", errors)
        measured = cell.get("measured_decode_step_ms", {})
        require(measured.get("count") == output - 7, f"{prefix}: steady decode count", errors)
        require(finite_positive(measured.get("p50")), f"{prefix}: steady decode timing", errors)
        log = (case / "run.log").read_text(encoding="utf-8", errors="replace")
        marker = f"has {gpu_experts} GPU experts"
        require(log.count(marker) == 48, f"{prefix}: runtime placement audit", errors)
        require("Capture cuda graph begin" in log, f"{prefix}: graph capture marker", errors)
        require("cuda graph: True" in log, f"{prefix}: graph replay marker", errors)
        kernel = (case / "kernel.since.log").read_text(encoding="utf-8", errors="replace")
        lowered = kernel.lower()
        require("xid" not in lowered, f"{prefix}: Xid in kernel log", errors)
        require("mmu fault" not in lowered, f"{prefix}: MMU fault in kernel log", errors)
        require(not (case / "gpu_cleanup_error.txt").exists(), f"{prefix}: GPU cleanup", errors)

        d_path = d_result_path(args.d_reference, workload, batch)
        require(d_path.is_file(), f"{prefix}: D reference absent", errors)
        d_result = load_json(d_path) if d_path.is_file() else {}
        d_timing = d_result.get("timing", {})
        d_ids = [record["output_token_ids"] for record in d_result.get("outputs", [])]
        d_hash = d_result.get("output_token_ids_sha256")
        kt_hash = cell.get("output_token_ids_sha256")
        kt_prefill = float(cell["prefill_step_ms"])
        kt_decode = float(measured["p50"])
        kt_e2e = float(cell["cycle_e2e_ms"])
        d_prefill = d_timing.get("prefill_step_ms")
        d_decode = (d_timing.get("steady_decode_step_ms") or {}).get("median")
        d_e2e = d_timing.get("cycle_e2e_ms")
        metrics.append({
            "workload": workload,
            "batch": batch,
            "prompt_tokens": prompt,
            "output_tokens": output,
            "tier": tier,
            "gpu_experts_per_layer": gpu_experts,
            "cpu_experts_per_layer": 128 - gpu_experts,
            "attention_backend": "TRITON_ATTN",
            "ktransformers": {
                "prefill_ms": kt_prefill,
                "p_to_d_switch_ms": None,
                "first_decode_step_ms": cell["first_decode_step_ms"],
                "steady_decode_ms": kt_decode,
                "e2e_ms": kt_e2e,
                "client_ttft_ms": cell["client_ttft_batch_ms"],
                "peak_hbm_mib": peak_hbm_mib(case / "nvml.csv"),
                "output_hash": kt_hash,
            },
            "ours_d": {
                "prefill_ms": d_prefill,
                "p_to_d_switch_ms": d_timing.get("cutover_wall_ms"),
                "steady_decode_ms": d_decode,
                "e2e_ms": d_e2e,
                "output_hash": d_hash,
            },
            "ktransformers_over_ours_latency": {
                "prefill": kt_prefill / d_prefill if finite_positive(d_prefill) else None,
                "steady_decode": kt_decode / d_decode if finite_positive(d_decode) else None,
                "e2e": kt_e2e / d_e2e if finite_positive(d_e2e) else None,
            },
            "full_output_hash_equal_to_ours": kt_hash == d_hash,
            "first_token_equal_to_ours": bool(
                len(ids) == len(d_ids)
                and ids
                and all(
                    kt_request and d_request
                    and kt_request[0] == d_request[0]
                    for kt_request, d_request in zip(ids, d_ids)
                )
            ),
        })

    summary = {
        "schema": "eurosys_e3_ktransformers_c1_c16_verification_v2",
        "status": "ok" if not errors else "error",
        "errors": errors,
        "protocol": {
            "same_model": True,
            "same_frozen_prompt_token_ids": True,
            "same_prompt_output_lengths": True,
            "same_batch_coordinates": True,
            "same_attention_backend": "TRITON_ATTN",
            "same_exact_expert_coordinates_per_cell": True,
            "one_cold_process_per_cell": True,
            "repeats": 1,
            "ktransformers_cuda_graph": True,
            "ktransformers_p_to_d_switch": False,
        },
        "completed_cells": len(metrics),
        "expected_cells": 10,
        "evidence_roots": [str(root.resolve()) for root in roots],
        "metrics": metrics,
        "output_checks": {
            "full_hash_equal_cells": sum(item["full_output_hash_equal_to_ours"] for item in metrics),
            "first_token_equal_cells": sum(item["first_token_equal_to_ours"] for item in metrics),
        },
        "interpretation_boundary": (
            "KTransformers is a complete CUDA Graph framework baseline. The ratios include "
            "framework, CPU expert kernel, scheduling, memory layout, and "
            "page-table-switch differences and are not a pure mechanism ablation."
        ),
    }
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
