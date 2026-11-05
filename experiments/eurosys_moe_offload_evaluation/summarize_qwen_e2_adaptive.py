#!/usr/bin/env python3
"""Summarize the Qwen3 E2 adaptive-concurrency qualification artifact."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


EXPECTED_CELLS = 16
CELL_PATTERN = re.compile(r"^(?P<workload>.+)_c(?P<concurrency>\d+)$")
BLOCK_PATTERN = re.compile(r"cold(?P<width>\d+)$")
KV_PARTITION_PATTERN = re.compile(
    r"KV partition: candidates=(?P<candidates>\d+) "
    r"admitted=(?P<admitted>\d+) deferred=(?P<deferred>\d+) "
    r"required_blocks=(?P<required_blocks>\d+) "
    r"free_blocks=(?P<free_blocks>\d+) "
    r"reserve_blocks=(?P<reserve_blocks>\d+)"
)
FATAL_PATTERNS = {
    "xid": re.compile(r"\bXid\b", re.IGNORECASE),
    "fault_pte": re.compile(r"FAULT_PTE", re.IGNORECASE),
    "fault_pde": re.compile(r"FAULT_PDE", re.IGNORECASE),
    "illegal_memory_access": re.compile(
        r"illegal memory access", re.IGNORECASE
    ),
    "cuda_error": re.compile(r"CUDA error", re.IGNORECASE),
    "eager_fallback": re.compile(r"eager fallback", re.IGNORECASE),
}
MANIFEST_EVENTS = {
    "cpu_offload_uvm_original_va_graph_residency_reconciled",
    "cpu_offload_uvm_original_va_dual_graph_partition",
    "cpu_offload_uvm_original_va_sealed_hot_residency_reconciled",
    "cpu_offload_uvm_static_alias_destroyed",
}
CSV_FIELDS = (
    "block",
    "cold_experts_per_layer",
    "cell",
    "workload",
    "concurrency",
    "prompt_tokens",
    "output_tokens_per_request",
    "requests",
    "successful_requests",
    "success_rate",
    "slo_ttft_ms",
    "slo_tpot_ms",
    "slo_satisfied_requests",
    "ttft_mean_ms",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "client_tpot_mean_ms",
    "steady_decode_tpot_mean_ms",
    "steady_decode_tpot_p50_ms",
    "steady_decode_tpot_p95_ms",
    "first_post_cutover_itl_mean_ms",
    "e2e_mean_ms",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "service_elapsed_seconds",
    "goodput_requests_per_second",
    "output_tokens_per_second",
)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def atomic_write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def parse_nvml_peak(path: Path) -> int:
    peak = 0
    with path.open(encoding="utf-8") as handle:
        for row in csv.reader(handle, skipinitialspace=True):
            if len(row) >= 5:
                peak = max(peak, int(row[4]))
    return peak


def parse_manifest(path: Path) -> tuple[Counter[str], dict[str, dict[str, Any]]]:
    counts: Counter[str] = Counter()
    selected: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            event = record.get("event")
            if not isinstance(event, str):
                continue
            counts[event] += 1
            if event in MANIFEST_EVENTS:
                selected[event] = record
    return counts, selected


def parse_plan_manifest(env: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    plan = Path(
        env["overrides"][
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE"
        ]
    )
    manifest = plan.with_name("manifest.json")
    return plan, load_json(manifest)


def compact_residency(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.get(key)
        for key in (
            "ok",
            "queried_bytes",
            "gpu_bytes",
            "cpu_bytes",
            "unknown_bytes",
        )
    }


def summarize_cell(block: str, cold_width: int, client_path: Path) -> dict[str, Any]:
    cell = client_path.parent.name
    match = CELL_PATTERN.fullmatch(cell)
    if match is None:
        raise ValueError(f"invalid E2 cell directory: {cell}")
    client = load_json(client_path)
    if client.get("status") != "ok" or len(client.get("phases", [])) != 1:
        raise ValueError(f"incomplete client artifact: {client_path}")
    phase = client["phases"][0]
    requests = phase["requests"]
    if not requests or not all(request.get("success") for request in requests):
        raise ValueError(f"failed request in {client_path}")

    ttft = [float(request["ttft_ms"]) for request in requests]
    client_tpot = [float(request["tpot_ms"]) for request in requests]
    e2e = [float(request["e2e_ms"]) for request in requests]
    first_itl: list[float] = []
    request_steady_tpot: list[float] = []
    steady_itls: list[float] = []
    for request in requests:
        itls = [float(value) for value in request["itl_ms"]]
        if itls:
            first_itl.append(itls[0])
        remaining = itls[1:]
        if remaining:
            request_steady_tpot.append(fmean(remaining))
            steady_itls.extend(remaining)
    if not request_steady_tpot:
        raise ValueError(f"no steady Decode ITLs in {client_path}")

    metadata = client["fixture"]["metadata"]
    return {
        "block": block,
        "cold_experts_per_layer": cold_width,
        "cell": cell,
        "workload": match.group("workload"),
        "concurrency": int(match.group("concurrency")),
        "prompt_tokens": int(metadata["prompt_tokens"]),
        "output_tokens_per_request": int(metadata["output_tokens"]),
        "requests": int(phase["request_count"]),
        "successful_requests": int(phase["successful_requests"]),
        "success_rate": float(phase["success_rate"]),
        "slo_ttft_ms": float(phase["slo"]["ttft_ms"]),
        "slo_tpot_ms": float(phase["slo"]["tpot_ms"]),
        "slo_satisfied_requests": int(phase["slo_satisfied_requests"]),
        "ttft_mean_ms": fmean(ttft),
        "ttft_p50_ms": percentile(ttft, 0.50),
        "ttft_p95_ms": percentile(ttft, 0.95),
        "client_tpot_mean_ms": fmean(client_tpot),
        "steady_decode_tpot_mean_ms": fmean(request_steady_tpot),
        "steady_decode_tpot_p50_ms": percentile(steady_itls, 0.50),
        "steady_decode_tpot_p95_ms": percentile(steady_itls, 0.95),
        "first_post_cutover_itl_mean_ms": fmean(first_itl),
        "e2e_mean_ms": fmean(e2e),
        "e2e_p50_ms": percentile(e2e, 0.50),
        "e2e_p95_ms": percentile(e2e, 0.95),
        "service_elapsed_seconds": float(phase["service_elapsed_seconds"]),
        "goodput_requests_per_second": float(
            phase["goodput_requests_per_second"]
        ),
        "output_tokens_per_second": float(phase["output_tokens_per_second"]),
    }


def summarize_block(block_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    match = BLOCK_PATTERN.search(block_root.name)
    if match is None:
        raise ValueError(f"block name has no cold width: {block_root.name}")
    cold_width = int(match.group("width"))
    verification = load_json(block_root / "verification.json")
    completed = load_json(block_root / "COMPLETED.json")
    env = load_json(block_root / "env.json")
    plan_path, plan_manifest = parse_plan_manifest(env)
    plan_metadata = plan_manifest["metadata"]
    event_counts, events = parse_manifest(block_root / "offload_manifest.jsonl")

    clients = sorted(block_root.glob("*_c*/client.json"))
    cells = [
        summarize_cell(block_root.name, cold_width, client) for client in clients
    ]

    server_log = (block_root / "server.log").read_text(
        encoding="utf-8", errors="replace"
    )
    fatal_counts = {
        name: len(pattern.findall(server_log))
        for name, pattern in FATAL_PATTERNS.items()
    }
    kv_partitions = [
        {key: int(value) for key, value in match.groupdict().items()}
        for match in KV_PARTITION_PATTERN.finditer(server_log)
    ]

    reconciled = events[
        "cpu_offload_uvm_original_va_graph_residency_reconciled"
    ]
    graph_partition = events[
        "cpu_offload_uvm_original_va_dual_graph_partition"
    ]
    sealed_reconcile = events[
        "cpu_offload_uvm_original_va_sealed_hot_residency_reconciled"
    ]
    destroyed = events["cpu_offload_uvm_static_alias_destroyed"]
    destroy_result = destroyed["destroy"]
    block = {
        "block": block_root.name,
        "concurrencies": sorted({cell["concurrency"] for cell in cells}),
        "cell_count": len(cells),
        "verifier_status": verification.get("status"),
        "completed_status": completed.get("status"),
        "gpu_cleanup_verified": completed.get("gpu_cleanup_verified"),
        "server_returncode": completed.get("server_returncode"),
        "plan": {
            "path": str(plan_path),
            "sha256": plan_manifest["plan_sha256"],
            "cold_experts_per_layer": cold_width,
            "layers": int(plan_metadata["num_layers"]),
            "experts_per_layer_total": int(plan_metadata["num_experts"]),
            "cold_expert_instances": int(plan_metadata["offload_plan_records"]),
            "cold_main_bytes": int(plan_metadata["selected_main_bytes"]),
            "cold_bytes_including_scales": int(
                plan_metadata["selected_bytes_including_scales"]
            ),
            "calibration_decode_route_fraction": float(
                plan_metadata["selected_decode_route_fraction"]
            ),
        },
        "graphs": {
            "phase_mode": graph_partition["phase_mode"],
            "prefill_capture_sizes": [
                int(entry[0]) for entry in graph_partition["alias_entries"]
            ],
            "decode_capture_batch_sizes": [
                int(entry[0]) for entry in graph_partition["remote_entries"]
            ],
            "prefill_graph_count": int(graph_partition["alias_entry_count"]),
            "decode_graph_count": int(graph_partition["remote_entry_count"]),
            "allocator_cache_preserved": bool(
                reconciled["allocator_cache_preserved"]
            ),
            "allocator_reserved_before_bytes": int(
                reconciled["allocator_reserved_before"]
            ),
            "allocator_reserved_after_bytes": int(
                reconciled["allocator_reserved_after"]
            ),
        },
        "residency_at_serving_gate": {
            "ok": bool(reconciled["ok"]),
            "cold_bytes": int(reconciled["cold_bytes"]),
            "cold": compact_residency(reconciled["cold_physical_residency"]),
            "hot_bytes": int(reconciled["hot_bytes"]),
            "hot": compact_residency(reconciled["hot_physical_residency"]),
            "free_bytes": int(reconciled["free_bytes"]),
            "minimum_free_bytes": int(reconciled["minimum_free_bytes"]),
            "hot_restore_allocator_reserved_before_bytes": int(
                sealed_reconcile["allocator_reserved_before"]
            ),
            "hot_restore_allocator_reserved_after_bytes": int(
                sealed_reconcile["allocator_reserved_after"]
            ),
        },
        "nvml_peak_hbm_mib": parse_nvml_peak(block_root / "nvml.csv"),
        "phase_transitions": {
            "prefill_to_decode": event_counts[
                "cpu_offload_uvm_original_va_remote_transition"
            ],
            "decode_to_prefill": event_counts[
                "cpu_offload_uvm_original_va_alias_prefill_activated"
            ],
            "hot_read_gates_ok": event_counts[
                "cpu_offload_uvm_hot_cuda_read_gate_ok"
            ],
        },
        "kv_partitions": {
            "count": len(kv_partitions),
            "events": kv_partitions,
        },
        "fatal_log_pattern_counts": fatal_counts,
        "cleanup": {
            "ok": bool(destroyed["ok"] and destroy_result["ok"]),
            "remaining_alias_count": int(
                destroy_result["remaining_alias_count"]
            ),
            "remaining_alias_bytes": int(
                destroy_result["remaining_alias_bytes"]
            ),
            "remaining_backing_bytes": int(
                destroy_result["remaining_backing_bytes"]
            ),
        },
    }
    return block, cells


def write_csv(path: Path, cells: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(cells)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_root = args.artifact_root.resolve()
    output_dir = (args.output_dir or artifact_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    block_roots = sorted(
        path.parent for path in artifact_root.glob("*/COMPLETED.json")
    )
    if not block_roots:
        raise ValueError(f"no completed process blocks under {artifact_root}")

    blocks: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for block_root in block_roots:
        block, block_cells = summarize_block(block_root)
        blocks.append(block)
        cells.extend(block_cells)
    workload_order = {
        "short_dialogue": 0,
        "ordinary_chat": 1,
        "rag_summary": 2,
        "long_generation": 3,
    }
    cells.sort(key=lambda cell: (workload_order[cell["workload"]], cell["concurrency"]))

    all_fatal_zero = all(
        all(count == 0 for count in block["fatal_log_pattern_counts"].values())
        for block in blocks
    )
    qualification_ok = (
        len(cells) == EXPECTED_CELLS
        and all(block["verifier_status"] == "ok" for block in blocks)
        and all(block["gpu_cleanup_verified"] for block in blocks)
        and all(cell["success_rate"] == 1.0 for cell in cells)
        and all_fatal_zero
    )
    module_after = load_json(artifact_root / "module.after.json")
    total_requests = sum(cell["requests"] for cell in cells)
    total_slo = sum(cell["slo_satisfied_requests"] for cell in cells)
    summary = {
        "schema": "qwen3_e2_adaptive_concurrency_summary_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "status": "ok" if qualification_ok else "error",
        "scope": {
            "model": "Qwen/Qwen3-30B-A3B-FP8",
            "system": "ours",
            "protocol_id": "eurosys-pa-v2-20260824",
            "kv_cache_bytes": 2 * 1024**3,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 16384,
            "process_repeats": 1,
            "cross_system_comparison": False,
        },
        "metric_definitions": {
            "client_tpot_mean_ms": (
                "mean of all request ITLs, including the first ITL after the "
                "Prefill-to-Decode cutover"
            ),
            "steady_decode_tpot_mean_ms": (
                "mean of per-request ITL means after removing each request's "
                "first post-cutover ITL"
            ),
            "goodput_requests_per_second": (
                "requests satisfying the workload TTFT and client-TPOT SLOs "
                "divided by service elapsed time"
            ),
        },
        "qualification": {
            "expected_cells": EXPECTED_CELLS,
            "observed_cells": len(cells),
            "successful_cells": sum(
                1 for cell in cells if cell["success_rate"] == 1.0
            ),
            "total_requests": total_requests,
            "successful_requests": sum(
                cell["successful_requests"] for cell in cells
            ),
            "slo_satisfied_requests": total_slo,
            "fatal_log_patterns_absent": all_fatal_zero,
            "stock_uvm_restored": bool(
                module_after.get("gpu_clean")
                and module_after.get("manual_migration_fault_mode") is None
            ),
        },
        "adaptive_policy": [
            {
                "concurrencies": block["concurrencies"],
                "cold_experts_per_layer": block["plan"][
                    "cold_experts_per_layer"
                ],
                "maximum_prefill_graph_tokens": max(
                    block["graphs"]["prefill_capture_sizes"]
                ),
                "placement_frozen_within_process_block": True,
            }
            for block in blocks
        ],
        "blocks": blocks,
        "cells": cells,
        "limitations": [
            "This artifact is one 16-cell qualification pass, not five independent process repeats.",
            "It contains only ours/Qwen3 and is not an equal-HBM cross-system ranking.",
            "Client TPOT includes the first post-cutover ITL; steady Decode TPOT is reported separately.",
        ],
    }

    summary_path = output_dir / "summary.json"
    metrics_path = output_dir / "metrics.csv"
    atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        args.overwrite,
    )
    write_csv(metrics_path, cells, args.overwrite)
    print(f"status={summary['status']}")
    print(f"cells={len(cells)}/{EXPECTED_CELLS}")
    print(f"requests={total_requests}")
    print(f"summary={summary_path}")
    print(f"metrics={metrics_path}")
    return 0 if qualification_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
