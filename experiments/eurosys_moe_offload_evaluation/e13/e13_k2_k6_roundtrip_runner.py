#!/usr/bin/env python3
"""One-process steady-state K2 -> K6 -> K2 E13 qualification."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE_ROOT = REPO_ROOT / "experiments" / "vllm_p3_pinned_uva_phase_compare"
GRAPH_ROOT = REPO_ROOT / "experiments" / "vllm_p3_static_alias_graph_integration"
PHYSICAL_KV_ROOT = (
    REPO_ROOT / "experiments" / "vllm_uvm_prefill_alias_decode_remote_graph"
)
sys.path.insert(0, str(PHASE_ROOT))
sys.path.insert(0, str(GRAPH_ROOT))
sys.path.insert(0, str(PHYSICAL_KV_ROOT))

import benchmark_phase_steps as phase  # noqa: E402
from benchmark_phase_switch_steps import (  # noqa: E402
    kv_physical_snapshot,
    validate_kv_physical_snapshot,
)
from benchmark_full_graph_steps import GraphCallTracker  # noqa: E402

from e13_kv_adaptive_runner import (  # noqa: E402
    atomic_json,
    load_fixture,
    plan_snapshot,
    queue_snapshot,
    resolve_path,
    sha256_json,
    utc_now,
)
from e13_tier_steady_phase_runner import (  # noqa: E402
    CONTROLLED_TARGET_TOKEN_IDS,
)


MICRO_REPEATS = 3
LOW_REPEATS = 3
HIGH_REPEATS = 2
MICRO_BATCHES = (1, 8)
LOW_BATCH = 1
HIGH_BATCH = 6
MICRO_OUTPUT_TOKENS = 32
LOW_OUTPUT_TOKENS = 32
HIGH_OUTPUT_TOKENS = 16
RAG_PRESSURE_REPEATS = 2
RAG_PRESSURE_POINTS = (
    ("low", 1, "k2"),
    ("medium", 5, "k4"),
    ("high", 6, "k6"),
)
RAG_SWEEP_POINTS = (
    ("b1", 1, "k2"),
    ("b2", 2, "k2"),
    ("b3", 3, "k4"),
    ("b4", 4, "k4"),
    ("b5", 5, "k4"),
    ("b6", 6, "k6"),
    ("b7", 7, "k6"),
    ("b8", 8, "k6"),
)
SHORT_B6_BATCH = 6
SHORT_B64_BATCH = 64
SHORT_B128_BATCH = 128
SHORT_B6_REPEATS = 3
SHORT_MATRIX_BATCHES = (1, 16, 64)
SHORT_GRAPH_MATRIX_BATCHES = (64, 128)
SHORT_GRAPH_CAPTURE_SIZES = (64, 128, 8192)
MOTIVATION_LONG_PROMPT_TOKENS = 26_000
MOTIVATION_LONG_BATCH = 2
MOTIVATION_LONG_OUTPUT_TOKENS = 128
MOTIVATION_SHORT_PROMPT_TOKENS = 128
MOTIVATION_SHORT_BATCH = 128
MOTIVATION_SHORT_OUTPUT_TOKENS = 32
MOTIVATION_MAX_MODEL_LEN = 28_672
MOTIVATION_DECODE_WARMUP_STEPS = 8
MOTIVATION_DECODE_MEASURED_STEPS = 16
EXPERT_MAIN_BYTES = 4_718_592


class ContractError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ContractError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--arm",
        choices=("fixed_k2", "fixed_k4", "fixed_k6", "roundtrip"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], percent: float) -> float:
    require(bool(values), "empty percentile distribution")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percent / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict[str, float | int]:
    require(bool(values), "empty measurement distribution")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p50": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
        "min": min(values),
        "max": max(values),
    }


def model_runner_from_llm(llm: Any) -> Any:
    client = llm.llm_engine.engine_core
    core = getattr(client, "engine_core", None)
    require(core is not None, "runner requires in-process EngineCore")
    executor = core.model_executor
    wrapper = executor.driver_worker
    worker = wrapper.worker
    require(worker is not None and worker.model_runner is not None, "model runner absent")
    return worker.model_runner


def motivation_heat_metrics(raw: list[Any]) -> dict[str, Any]:
    """Reduce one diagnostic Decode step to routing/placement metrics."""
    require(len(raw) == 1, "single-worker heat snapshot required")
    worker = raw[0]
    require(
        isinstance(worker, dict) and worker.get("ok") is True,
        f"heat RPC failed: {worker!r}",
    )
    layers = worker.get("layers") or {}
    require(len(layers) == 48, f"expected 48 MoE layers, got {len(layers)}")
    total_assignments = 0
    remote_assignments = 0
    unique_active = 0
    unique_remote_active = 0
    route_rows: list[list[Any]] = []
    for layer_name, value in sorted(layers.items()):
        counts = [int(item) for item in value.get("expert_token_counts", ())]
        require(bool(counts), f"{layer_name}: empty expert counts")
        remote = {
            int(item)
            for key in ("remote_experts", "static_alias_owned_experts")
            for item in value.get(key, ())
        }
        active = [index for index, count in enumerate(counts) if count > 0]
        layer_total = sum(counts)
        layer_remote = sum(counts[index] for index in remote if index < len(counts))
        active_remote = [index for index in active if index in remote]
        total_assignments += layer_total
        remote_assignments += layer_remote
        unique_active += len(active)
        unique_remote_active += len(active_remote)
        route_rows.append([str(layer_name), counts])
    require(total_assignments > 0, "diagnostic Decode step has no route assignments")
    layer_count = len(layers)
    return {
        "layer_count": layer_count,
        "total_assignments": total_assignments,
        "remote_assignments": remote_assignments,
        "resident_hit_rate": 1.0 - remote_assignments / total_assignments,
        "unique_active_experts_per_layer": unique_active / layer_count,
        "unique_remote_experts_per_layer": unique_remote_active / layer_count,
        "logical_remote_bytes": unique_remote_active * EXPERT_MAIN_BYTES,
        "route_sha256": sha256_json(route_rows),
    }


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"output already exists: {args.output}")
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    require(
        protocol.get("schema") == "eurosys_e13_kv_adaptive_protocol_v1",
        "bad config schema",
    )
    runtime = protocol["runtime"]
    measurement_scope = os.environ.get(
        "E13_K2_K6_MEASUREMENT_SCOPE", "full"
    ).strip()
    full_separate_graph = os.environ.get(
        "E13_K2_K6_FULL_SEPARATE_GRAPH", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    require(
        measurement_scope
        in {
            "full",
            "short_b6",
            "short_b64",
            "short_b128",
            "short_b128_abba",
            "short_b128_shrink_isolation",
            "short_matrix",
            "short_graph_matrix",
            "short_graph_roundtrip",
            "rag_pressure_matrix",
            "rag_concurrency_sweep",
            "partition_motivation",
        },
        f"unknown measurement scope: {measurement_scope}",
    )
    require(
        measurement_scope
        in {
            "full",
            "short_b128_abba",
            "short_b128_shrink_isolation",
            "short_graph_roundtrip",
            "rag_pressure_matrix",
            "rag_concurrency_sweep",
            "partition_motivation",
        }
        or args.arm != "roundtrip",
        "short-batch scope only supports fixed arms",
    )
    require(
        args.arm != "fixed_k4"
        or measurement_scope
        in {"rag_pressure_matrix", "rag_concurrency_sweep", "partition_motivation"},
        "fixed_k4 is only supported by a RAG pressure experiment",
    )
    effective_max_num_seqs = (
        MOTIVATION_SHORT_BATCH
        if measurement_scope == "partition_motivation"
        else SHORT_B128_BATCH
        if measurement_scope
        in {
            "short_b128",
            "short_b128_abba",
            "short_b128_shrink_isolation",
            "short_graph_matrix",
            "short_graph_roundtrip",
        }
        else SHORT_B64_BATCH
        if measurement_scope in {"short_b64", "short_matrix"}
        else int(runtime["max_num_seqs"])
    )
    tiers = {str(item["name"]): item for item in protocol["tiers"]}
    require(list(tiers) == ["k2", "k4", "k6"], "unexpected tier order")
    repo_root = args.repo_root.resolve()
    motivation_workload = os.environ.get("E13_MOTIVATION_WORKLOAD", "").strip()
    motivation_diagnostic = os.environ.get(
        "E13_MOTIVATION_DIAGNOSTIC", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if measurement_scope == "partition_motivation":
        require(
            motivation_workload in {"long_c2", "short_b128"},
            f"bad motivation workload: {motivation_workload}",
        )
        require(args.arm != "roundtrip", "motivation scope supports fixed arms only")
    fixtures = {
        name: load_fixture(resolve_path(repo_root, protocol["fixtures"][name]))
        for name in ("short", "medium", "rag")
    }
    require(fixtures["short"][0]["prompt_tokens"] == 128, "bad Short fixture")
    require(
        all(item["prompt_tokens"] == 2048 for item in fixtures["medium"][:8]),
        "bad Medium fixture",
    )
    require(
        all(item["prompt_tokens"] == 8192 for item in fixtures["rag"][:6]),
        "bad RAG fixture",
    )

    fixture_hashes = {
        name: hashlib.sha256(
            resolve_path(repo_root, protocol["fixtures"][name]).read_bytes()
        ).hexdigest()
        for name in fixtures
    }
    result: dict[str, Any] = {
        "schema": "eurosys_e13_k2_k6_roundtrip_v1",
        "status": "loading",
        "started_at": utc_now(),
        "arm": args.arm,
        "seed": args.seed,
        "repeat_index": args.repeat_index,
        "configuration": {
            "model": protocol["model"],
            "runtime": runtime,
            "fixture_sha256": fixture_hashes,
            "controlled_tokens_sha256": sha256_json(
                list(CONTROLLED_TARGET_TOKEN_IDS)
            ),
            "micro_batches": list(MICRO_BATCHES),
            "micro_repeats": MICRO_REPEATS,
            "low_repeats": LOW_REPEATS,
            "high_repeats": HIGH_REPEATS,
            "rag_pressure_repeats": RAG_PRESSURE_REPEATS,
            "rag_pressure_points": [
                {"pressure": pressure, "batch_size": batch, "tier": tier}
                for pressure, batch, tier in RAG_PRESSURE_POINTS
            ],
            "rag_sweep_points": [
                {"pressure": pressure, "batch_size": batch, "tier": tier}
                for pressure, batch, tier in RAG_SWEEP_POINTS
            ],
            "switch_transaction_excluded": True,
            "post_switch_warmup_excluded": True,
            "single_cold_process_per_arm": True,
            "measurement_scope": measurement_scope,
            "short_b6_batch": SHORT_B6_BATCH,
            "short_b64_batch": SHORT_B64_BATCH,
            "short_b128_batch": SHORT_B128_BATCH,
            "short_b128_abba_order": ["a1_k2", "b1_k6", "b2_k6", "a2_k2"],
            "short_b128_shrink_isolation_order": [
                "a0_k2_initial",
                "b0_k4_grown",
                "a1_k2_after_k4",
                "c0_k6_grown",
                "b1_k4_after_k6",
                "a2_k2_after_k6",
            ],
            "short_b6_repeats": SHORT_B6_REPEATS,
            "short_matrix_batches": list(SHORT_MATRIX_BATCHES),
            "short_graph_matrix_batches": list(SHORT_GRAPH_MATRIX_BATCHES),
            "full_separate_graph": full_separate_graph,
            "graph_capture_sizes": (
                list(SHORT_GRAPH_CAPTURE_SIZES) if full_separate_graph else []
            ),
            "effective_max_num_seqs": effective_max_num_seqs,
            "effective_max_model_len": (
                MOTIVATION_MAX_MODEL_LEN
                if measurement_scope == "partition_motivation"
                else int(runtime["max_model_len"])
            ),
            "motivation_workload": (
                motivation_workload
                if measurement_scope == "partition_motivation"
                else None
            ),
            "motivation_diagnostic": motivation_diagnostic,
            "motivation_decode_warmup_steps": MOTIVATION_DECODE_WARMUP_STEPS,
            "motivation_decode_measured_steps": MOTIVATION_DECODE_MEASURED_STEPS,
            "short_b64_fixture_tiling": (
                "repeat_frozen_32_request_sequence_twice"
                if measurement_scope in {"short_b64", "short_matrix"}
                else None
            ),
        },
        "initialization": {},
        "transitions": [],
        "segments": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)

    llm: Any | None = None
    try:
        import torch
        from vllm import LLM, SamplingParams, TokensPrompt
        from vllm.compilation.counter import compilation_counter
        from vllm.config import CompilationMode, CUDAGraphMode
        from vllm.model_executor.models import utils
        from vllm.sampling_params import RequestOutputKind

        from e13_teacher_forced_logits import E13TeacherForcedLogitsProcessor

        graph_tracker = GraphCallTracker()
        graph_tracker.install()
        counter_before = dataclasses.asdict(compilation_counter.clone())
        compilation_config: dict[str, Any] = {
            "mode": "NONE",
            "cudagraph_mode": "NONE",
        }
        if full_separate_graph:
            compilation_config = {
                "mode": "NONE",
                "cudagraph_mode": "FULL_SEPARATE",
                "cudagraph_capture_sizes": list(SHORT_GRAPH_CAPTURE_SIZES),
                "max_cudagraph_capture_size": max(SHORT_GRAPH_CAPTURE_SIZES),
            }
        load_started = time.perf_counter()
        llm = LLM(
            model=protocol["model"],
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=float(runtime["gpu_memory_utilization"]),
            enforce_eager=not full_separate_graph,
            max_model_len=(
                MOTIVATION_MAX_MODEL_LEN
                if measurement_scope == "partition_motivation"
                else int(runtime["max_model_len"])
            ),
            max_num_seqs=effective_max_num_seqs,
            max_num_batched_tokens=int(runtime["max_num_batched_tokens"]),
            kv_cache_memory_bytes=int(runtime["kv_cache_memory_bytes"]),
            cpu_offload_gb=float(runtime["cpu_offload_gb"]),
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            enable_dbo=False,
            compilation_config=compilation_config,
            seed=args.seed,
            disable_log_stats=True,
            logits_processors=[E13TeacherForcedLogitsProcessor],
        )
        phase.enable_request_metrics_without_default_loggers(llm)
        torch.cuda.synchronize()
        realized = llm.llm_engine.vllm_config.compilation_config
        require(realized.mode == CompilationMode.NONE, "CompilationMode is not NONE")
        expected_graph_mode = (
            CUDAGraphMode.FULL_SEPARATE
            if full_separate_graph
            else CUDAGraphMode.NONE
        )
        require(
            realized.cudagraph_mode == expected_graph_mode,
            "realized CUDA Graph mode differs",
        )
        expected_capture_sizes = (
            list(SHORT_GRAPH_CAPTURE_SIZES) if full_separate_graph else []
        )
        require(
            list(realized.cudagraph_capture_sizes) == expected_capture_sizes,
            "realized CUDA Graph capture sizes differ",
        )
        counter_after_load = dataclasses.asdict(compilation_counter.clone())
        graph_calls_after_load = graph_tracker.snapshot()
        model_runner = model_runner_from_llm(llm)
        graph_class_diagnostics = (
            model_runner.model.cudagraph_class_diagnostics()
            if full_separate_graph
            and hasattr(model_runner.model, "cudagraph_class_diagnostics")
            else None
        )
        if full_separate_graph:
            capture_delta = int(counter_after_load["num_cudagraph_captured"]) - int(
                counter_before["num_cudagraph_captured"]
            )
            require(
                capture_delta >= len(SHORT_GRAPH_CAPTURE_SIZES),
                f"too few CUDA Graph captures: {capture_delta}",
            )
            require(
                int(graph_calls_after_load.get("full_capture", 0))
                >= len(SHORT_GRAPH_CAPTURE_SIZES),
                "Graph tracker saw too few FULL captures",
            )
            require(
                isinstance(graph_class_diagnostics, dict),
                "CUDA Graph class diagnostics absent",
            )

        def phase_state() -> dict[str, Any]:
            diagnostics = utils.get_cpu_offload_graph_static_alias_diagnostics()
            history = diagnostics.get("phase_transition_history") or []
            return {
                "phase_mode": diagnostics.get("phase_mode"),
                "lifecycle_state": diagnostics.get("lifecycle_state"),
                "park_resume_enabled": diagnostics.get("park_resume_enabled"),
                "remote_map_active": diagnostics.get("remote_map_active"),
                "alias_graph_captured": diagnostics.get("alias_graph_captured"),
                "remote_graph_captured": diagnostics.get("remote_graph_captured"),
                "active_plan_class": diagnostics.get("active_plan_class"),
                "phase_transition_count": len(history),
            }

        core = llm.llm_engine.engine_core.engine_core
        scheduler = core.scheduler
        policy = scheduler.kv_cache_manager.uvm_kv_runtime_policy
        require(policy.dynamic_pool_active(), "managed dynamic KV pool is absent")
        result["initialization"] = {
            "load_seconds": time.perf_counter() - load_started,
            "graph_counter_before": counter_before,
            "graph_counter_after": counter_after_load,
            "graph_wrapper_calls_after_load": graph_calls_after_load,
            "realized_compilation": {
                "mode": realized.mode.name,
                "cudagraph_mode": realized.cudagraph_mode.name,
                "cudagraph_capture_sizes": list(realized.cudagraph_capture_sizes),
            },
            "graph_class_diagnostics": graph_class_diagnostics,
            "phase": phase_state(),
            "kv": queue_snapshot(core),
            "plan": plan_snapshot(utils),
        }

        schedule_records: list[dict[str, Any]] = []
        original_schedule = scheduler.schedule

        def traced_schedule() -> Any:
            output = original_schedule()
            snapshot = phase.schedule_snapshot(output)
            snapshot["running_after"] = len(scheduler.running)
            snapshot["waiting_after"] = len(scheduler.waiting)
            snapshot["phase_epoch_request_count"] = len(
                getattr(scheduler, "_uvm_phase_epoch_request_ids", ())
            )
            schedule_records.append(snapshot)
            return output

        scheduler.schedule = traced_schedule
        request_serial = 0

        def apply_one_tier(tier_name: str, label: str) -> None:
            require(
                not llm.llm_engine.has_unfinished_requests(),
                f"{label}: switch requires drained measurement batch",
            )
            tier = tiers[tier_name]
            before_kv = queue_snapshot(core)
            before_plan = plan_snapshot(utils)
            before_graph_class = (
                model_runner.model.cudagraph_class_diagnostics()
                if full_separate_graph
                else None
            )
            old_target = int(before_kv["current_target_blocks"])
            new_target = int(tier["target_blocks"])
            before_kv_physical = kv_physical_snapshot(model_runner, old_target)
            validate_kv_physical_snapshot(before_kv_physical)
            torch.cuda.synchronize()
            started = time.perf_counter()
            value = core.apply_uvm_kv_dynamic_pool_target(
                new_target,
                reason=label,
                metadata={"release_expert_wave": False},
            )
            require(bool(value.get("ok")), f"{label}: tier application failed: {value!r}")
            core._note_uvm_kv_autoscale_resize(new_target)
            # Growth activates the matching expert Plan Class before making
            # the new KV suffix allocatable.  Do not activate that class a
            # second time after recording the committed KV tier.
            if isinstance(value.get("plan_class_before_grow"), dict):
                core._uvm_kv_autoscale_pending_plan_class = None
            pending_plan = core._apply_pending_uvm_expert_plan_class_if_ready()
            torch.cuda.synchronize()
            after_kv = queue_snapshot(core)
            after_plan = plan_snapshot(utils)
            after_kv_physical = kv_physical_snapshot(model_runner, new_target)
            validate_kv_physical_snapshot(after_kv_physical)
            after_graph_class = (
                model_runner.model.cudagraph_class_diagnostics()
                if full_separate_graph
                else None
            )
            require(
                pending_plan is None or bool(pending_plan.get("ok")),
                f"{label}: Plan Class application failed: {pending_plan!r}",
            )
            require(
                int(after_kv["current_target_blocks"]) == new_target,
                f"{label}: KV target differs",
            )
            require(
                after_plan["active_plan_class"] == tier["plan_class"],
                f"{label}: Plan Class differs",
            )
            if full_separate_graph:
                require(
                    after_graph_class.get("active_class") == tier["plan_class"],
                    f"{label}: CUDA Graph Class differs",
                )
            result["transitions"].append(
                {
                    "label": label,
                    "old_target_blocks": old_target,
                    "new_target_blocks": new_target,
                    "wall_ms_excluded": (time.perf_counter() - started) * 1000.0,
                    "result": value,
                    "pending_plan_result": pending_plan,
                    "before_kv": before_kv,
                    "after_kv": after_kv,
                    "before_kv_physical": before_kv_physical,
                    "after_kv_physical": after_kv_physical,
                    "before_plan": before_plan,
                    "after_plan": after_plan,
                    "before_graph_class": before_graph_class,
                    "after_graph_class": after_graph_class,
                }
            )

        ordered_tiers = ("k2", "k4", "k6")

        def apply_tier_path(target_name: str, label: str) -> None:
            current_target = int(queue_snapshot(core)["current_target_blocks"])
            current_name = next(
                name
                for name in ordered_tiers
                if int(tiers[name]["target_blocks"]) == current_target
            )
            current_index = ordered_tiers.index(current_name)
            target_index = ordered_tiers.index(target_name)
            if current_index == target_index:
                apply_one_tier(target_name, f"{label}_{target_name}_noop")
                return
            direction = 1 if target_index > current_index else -1
            for index in range(current_index + direction, target_index + direction, direction):
                apply_one_tier(
                    ordered_tiers[index],
                    f"{label}_{ordered_tiers[index]}",
                )

        def run_batch(
            *,
            label: str,
            workload: str,
            batch_size: int,
            output_tokens: int,
            role: str,
            repeat_index: int,
        ) -> dict[str, Any]:
            nonlocal request_serial
            require(not llm.llm_engine.has_unfinished_requests(), f"{label}: engine busy")
            if workload == "motivation_long":
                source_fixture = fixtures["rag"]
                selected = []
                for index in range(batch_size):
                    source = list(
                        source_fixture[index % len(source_fixture)]["prompt_token_ids"]
                    )
                    repeats = (MOTIVATION_LONG_PROMPT_TOKENS + len(source) - 1) // len(source)
                    tokens = (source * repeats)[:MOTIVATION_LONG_PROMPT_TOKENS]
                    selected.append(
                        {
                            "prompt_token_ids": tokens,
                            "prompt_tokens": len(tokens),
                            "prompt_sha256": hashlib.sha256(
                                ",".join(str(token) for token in tokens).encode("ascii")
                            ).hexdigest(),
                        }
                    )
            else:
                source_fixture = fixtures[workload]
                selected = [
                    source_fixture[index % len(source_fixture)]
                    for index in range(batch_size)
                ]
            prompts = [
                TokensPrompt(prompt_token_ids=list(item["prompt_token_ids"]))
                for item in selected
            ]
            targets = list(
                (CONTROLLED_TARGET_TOKEN_IDS * 4)[:output_tokens]
            )
            require(len(targets) == output_tokens, f"{label}: target length differs")
            samplings = []
            for index in range(batch_size):
                logical = f"{args.arm}/{label}/{role}/{repeat_index}/{index}"
                samplings.append(
                    SamplingParams(
                        max_tokens=output_tokens,
                        min_tokens=output_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        seed=args.seed,
                        ignore_eos=True,
                        output_kind=RequestOutputKind.FINAL_ONLY,
                        extra_args={
                            "e13_teacher_forced": {
                                "logical_request": logical,
                                "target_token_ids": targets,
                                "record_logits": False,
                            }
                        },
                    )
                )
            schedule_cursor = len(schedule_records)
            request_serial += batch_size
            phase_before = phase_state()
            graph_before_batch = graph_tracker.snapshot()
            torch.cuda.synchronize()
            started = time.perf_counter()
            llm._validate_and_add_requests(
                prompts=prompts,
                params=samplings,
                use_tqdm=False,
                lora_request=None,
                priority=None,
            )
            final_outputs: list[Any] = []
            step_ms: list[float] = []
            graph_step_deltas: list[dict[str, int]] = []
            route_steps: list[dict[str, Any]] = []
            decode_ordinal = 0
            stalled_steps = 0
            capacity_exceeded = False
            capacity_limited = False
            max_observed_active_blocks = 0
            if motivation_diagnostic:
                reset_raw = core.collective_rpc(
                    "get_uvm_dynamic_expert_heat", args=(True,)
                )
                require(
                    len(reset_raw) == 1
                    and isinstance(reset_raw[0], dict)
                    and reset_raw[0].get("ok") is True,
                    f"{label}: initial heat reset failed: {reset_raw!r}",
                )
            while llm.llm_engine.has_unfinished_requests():
                require(len(step_ms) < 1024, f"{label}: exceeded scheduler-step guard")
                torch.cuda.synchronize()
                graph_before_step = graph_tracker.snapshot()
                step_started = time.perf_counter()
                outputs = llm.llm_engine.step()
                torch.cuda.synchronize()
                step_ms.append((time.perf_counter() - step_started) * 1000.0)
                graph_step_deltas.append(
                    graph_tracker.delta(graph_before_step, graph_tracker.snapshot())
                )
                final_outputs.extend(output for output in outputs if output.finished)
                current_schedule = schedule_records[-1]
                if (
                    measurement_scope == "partition_motivation"
                    and workload == "motivation_long"
                    and int(current_schedule["phase_epoch_request_count"]) < batch_size
                ):
                    request_ids = [
                        str(request.request_id)
                        for request in list(scheduler.running) + list(scheduler.waiting)
                    ]
                    require(bool(request_ids), f"{label}: limited without request IDs")
                    llm.llm_engine.abort_request(request_ids)
                    capacity_limited = int(current_schedule["running_after"]) > 0
                    capacity_exceeded = not capacity_limited
                    break
                strict_decode_step = bool(
                    current_schedule["num_scheduled_tokens"]
                    and not current_schedule["new_request_ids"]
                    and all(
                        int(value) == 1
                        for value in current_schedule["num_scheduled_tokens"].values()
                    )
                )
                if strict_decode_step:
                    decode_ordinal += 1
                    if (
                        measurement_scope == "partition_motivation"
                        and decode_ordinal >= MOTIVATION_DECODE_WARMUP_STEPS
                        and (decode_ordinal - MOTIVATION_DECODE_WARMUP_STEPS) % 16 == 0
                    ):
                        kv_now = queue_snapshot(core)
                        max_observed_active_blocks = max(
                            max_observed_active_blocks,
                            int(kv_now.get("active_blocks") or 0),
                        )
                    if motivation_diagnostic:
                        if decode_ordinal == MOTIVATION_DECODE_WARMUP_STEPS:
                            reset_raw = core.collective_rpc(
                                "get_uvm_dynamic_expert_heat", args=(True,)
                            )
                            require(
                                len(reset_raw) == 1
                                and isinstance(reset_raw[0], dict)
                                and reset_raw[0].get("ok") is True,
                                f"{label}: warmup heat reset failed",
                            )
                        elif (
                            MOTIVATION_DECODE_WARMUP_STEPS
                            < decode_ordinal
                            <= MOTIVATION_DECODE_WARMUP_STEPS
                            + MOTIVATION_DECODE_MEASURED_STEPS
                        ):
                            route_steps.append(
                                motivation_heat_metrics(
                                    core.collective_rpc(
                                        "get_uvm_dynamic_expert_heat", args=(True,)
                                    )
                                )
                            )
                if (
                    int(current_schedule["total_num_scheduled_tokens"]) == 0
                    and int(current_schedule["running_after"]) == 0
                    and int(current_schedule["waiting_after"]) > 0
                ):
                    stalled_steps += 1
                else:
                    stalled_steps = 0
                if measurement_scope == "partition_motivation" and stalled_steps >= 3:
                    request_ids = [
                        str(request.request_id)
                        for request in list(scheduler.running) + list(scheduler.waiting)
                    ]
                    require(bool(request_ids), f"{label}: stalled without request IDs")
                    llm.llm_engine.abort_request(request_ids)
                    capacity_exceeded = True
                    break
            wall_ms = (time.perf_counter() - started) * 1000.0
            graph_after_batch = graph_tracker.snapshot()
            phase_after = phase_state()
            schedules = schedule_records[schedule_cursor:]
            require(len(schedules) == len(step_ms), f"{label}: schedule count differs")
            required_kv_blocks = batch_size * math.ceil(
                (int(selected[0]["prompt_tokens"]) + output_tokens)
                / int(runtime["kv_block_tokens"])
            )
            if capacity_exceeded or capacity_limited:
                return {
                    "label": label,
                    "workload": workload,
                    "role": role,
                    "repeat_index": repeat_index,
                    "status": (
                        "kv_capacity_limited"
                        if capacity_limited
                        else "kv_capacity_exceeded"
                    ),
                    "batch_size": batch_size,
                    "prompt_tokens_per_request": int(selected[0]["prompt_tokens"]),
                    "prompt_sha256": [str(item["prompt_sha256"]) for item in selected],
                    "output_tokens_per_request": output_tokens,
                    "required_kv_blocks": required_kv_blocks,
                    "required_kv_gib": (
                        required_kv_blocks * int(runtime["kv_block_bytes"]) / 2**30
                    ),
                    "wall_ms_before_capacity_classification": wall_ms,
                    "max_scheduler_running_requests": max(
                        int(item["running_after"]) for item in schedules
                    ),
                    "max_scheduler_waiting_requests": max(
                        int(item["waiting_after"]) for item in schedules
                    ),
                    "max_observed_active_blocks": max_observed_active_blocks,
                    "schedule_step_count": len(schedules),
                    "phase_before": phase_before,
                    "phase_after": phase_after,
                }
            require(
                all(not item["preempted_request_ids"] for item in schedules),
                f"{label}: preemption observed",
            )
            records = phase.output_records(final_outputs)
            require(len(records) == batch_size, f"{label}: output count differs")
            require(
                all(len(item["output_token_ids"]) == output_tokens for item in records),
                f"{label}: output length differs",
            )
            prefill_indices = [
                index
                for index, item in enumerate(schedules)
                if item["new_request_ids"]
                or any(int(value) > 1 for value in item["num_scheduled_tokens"].values())
            ]
            decode_indices = [
                index
                for index, item in enumerate(schedules)
                if item["num_scheduled_tokens"]
                and not item["new_request_ids"]
                and all(
                    int(value) == 1
                    for value in item["num_scheduled_tokens"].values()
                )
            ]
            prefill_values = [step_ms[index] for index in prefill_indices]
            decode_values = [step_ms[index] for index in decode_indices]
            require(prefill_values and decode_values, f"{label}: phase classification empty")
            steady_decode_indices = decode_indices[
                MOTIVATION_DECODE_WARMUP_STEPS:
            ]
            steady_decode_values = [step_ms[index] for index in steady_decode_indices]
            steady_decode_tokens = sum(
                int(schedules[index]["total_num_scheduled_tokens"])
                for index in steady_decode_indices
            )
            if measurement_scope == "partition_motivation":
                require(
                    len(steady_decode_values) >= MOTIVATION_DECODE_MEASURED_STEPS,
                    f"{label}: insufficient post-warmup Decode steps",
                )
                require(
                    not motivation_diagnostic
                    or len(route_steps) == MOTIVATION_DECODE_MEASURED_STEPS,
                    f"{label}: diagnostic route-step count differs",
                )
            if full_separate_graph:
                require(
                    len(graph_step_deltas) == len(step_ms)
                    and all(
                        int(item.get("full_replay", 0)) == 1
                        and int(item.get("full_capture", 0)) == 0
                        and int(item.get("none_passthrough", 0)) == 0
                        for item in graph_step_deltas
                    ),
                    f"{label}: a serving step did not use exactly one FULL Graph replay",
                )
                require(
                    phase_after.get("phase_mode") == "prefill_graph_remote_decode"
                    and phase_after.get("lifecycle_state") == "remote_parked"
                    and phase_after.get("park_resume_enabled") is True
                    and phase_after.get("remote_map_active") is True
                    and phase_after.get("alias_graph_captured") is True
                    and phase_after.get("remote_graph_captured") is True,
                    f"{label}: fast P/D topology contract differs",
                )
                require(
                    int(phase_after.get("phase_transition_count", 0))
                    > int(phase_before.get("phase_transition_count", 0)),
                    f"{label}: no P/D topology transition was recorded",
                )
            request_timings = phase.request_timing_summaries(records)
            max_running = max(int(item["running_after"]) for item in schedules)
            route_metrics = None
            if route_steps:
                route_metrics = {
                    "sampled_decode_steps": len(route_steps),
                    "unique_active_experts_per_layer_step": distribution(
                        [
                            float(item["unique_active_experts_per_layer"])
                            for item in route_steps
                        ]
                    ),
                    "resident_hit_rate": distribution(
                        [float(item["resident_hit_rate"]) for item in route_steps]
                    ),
                    "logical_remote_bytes_per_step": distribution(
                        [float(item["logical_remote_bytes"]) for item in route_steps]
                    ),
                    "total_assignments_per_step": distribution(
                        [float(item["total_assignments"]) for item in route_steps]
                    ),
                    "route_sha256": sha256_json(
                        [str(item["route_sha256"]) for item in route_steps]
                    ),
                }
            return {
                "label": label,
                "workload": workload,
                "role": role,
                "repeat_index": repeat_index,
                "status": (
                    "ok" if max_running == batch_size else "kv_capacity_limited"
                ),
                "batch_size": batch_size,
                "prompt_tokens_per_request": int(selected[0]["prompt_tokens"]),
                "prompt_sha256": [str(item["prompt_sha256"]) for item in selected],
                "output_tokens_per_request": output_tokens,
                "required_kv_blocks": required_kv_blocks,
                "required_kv_gib": (
                    required_kv_blocks * int(runtime["kv_block_bytes"]) / 2**30
                ),
                "max_observed_active_blocks": max_observed_active_blocks,
                "max_observed_active_kv_gib": (
                    max_observed_active_blocks
                    * int(runtime["kv_block_bytes"])
                    / 2**30
                ),
                "wall_ms": wall_ms,
                "request_throughput_per_second": batch_size * 1000.0 / wall_ms,
                "output_token_throughput_per_second": (
                    batch_size * output_tokens * 1000.0 / wall_ms
                ),
                "prefill_ms": sum(prefill_values),
                "prefill_step_count": len(prefill_values),
                "decode_step_ms": distribution(decode_values),
                "steady_decode_last16_ms": distribution(decode_values[-16:]),
                "steady_decode_after8_ms": distribution(steady_decode_values),
                "steady_decode_scheduled_tokens": steady_decode_tokens,
                "steady_decode_output_tokens_per_second": (
                    steady_decode_tokens * 1000.0 / sum(steady_decode_values)
                ),
                "route_metrics": route_metrics,
                "max_scheduler_running_requests": max_running,
                "max_scheduler_waiting_requests": max(
                    int(item["waiting_after"]) for item in schedules
                ),
                "request_timings": request_timings,
                "output_token_ids_sha256": phase.output_hash(records),
                "schedule_step_count": len(schedules),
                "prefill_step_values_ms": prefill_values,
                "decode_step_values_ms": decode_values,
                "pd_cutover_step_ms": step_ms[decode_indices[0]],
                "graph_wrapper_delta": graph_tracker.delta(
                    graph_before_batch, graph_after_batch
                ),
                "graph_step_deltas": graph_step_deltas,
                "phase_before": phase_before,
                "phase_after": phase_after,
            }

        def measure_segment(
            *,
            label: str,
            tier_name: str,
            include_low: bool,
            include_high: bool,
        ) -> None:
            before = {
                "kv": queue_snapshot(core),
                "plan": plan_snapshot(utils),
                "graph_class": (
                    model_runner.model.cudagraph_class_diagnostics()
                    if full_separate_graph
                    else None
                ),
            }
            tier = tiers[tier_name]
            require(
                int(before["kv"]["current_target_blocks"])
                == int(tier["target_blocks"]),
                f"{label}: wrong target",
            )
            require(
                before["plan"]["active_plan_class"] == tier["plan_class"],
                f"{label}: wrong Plan Class",
            )
            if full_separate_graph:
                require(
                    before["graph_class"].get("active_class")
                    == tier["plan_class"],
                    f"{label}: wrong CUDA Graph Class",
                )
            warmups = [
                run_batch(
                    label=f"{label}_target_warmup_b8",
                    workload="medium",
                    batch_size=8,
                    output_tokens=MICRO_OUTPUT_TOKENS,
                    role="warmup",
                    repeat_index=0,
                )
            ]
            measured: list[dict[str, Any]] = []
            for batch_size in MICRO_BATCHES:
                for repeat in range(MICRO_REPEATS):
                    measured.append(
                        run_batch(
                            label=f"{label}_micro_b{batch_size}",
                            workload="medium",
                            batch_size=batch_size,
                            output_tokens=MICRO_OUTPUT_TOKENS,
                            role="measured_micro",
                            repeat_index=repeat,
                        )
                    )
            if include_low:
                warmups.append(
                    run_batch(
                        label=f"{label}_low",
                        workload="short",
                        batch_size=LOW_BATCH,
                        output_tokens=LOW_OUTPUT_TOKENS,
                        role="warmup",
                        repeat_index=0,
                    )
                )
                for repeat in range(LOW_REPEATS):
                    measured.append(
                        run_batch(
                            label=f"{label}_low",
                            workload="short",
                            batch_size=LOW_BATCH,
                            output_tokens=LOW_OUTPUT_TOKENS,
                            role="measured_low",
                            repeat_index=repeat,
                        )
                    )
            if include_high:
                warmups.append(
                    run_batch(
                        label=f"{label}_high",
                        workload="rag",
                        batch_size=HIGH_BATCH,
                        output_tokens=HIGH_OUTPUT_TOKENS,
                        role="warmup",
                        repeat_index=0,
                    )
                )
                for repeat in range(HIGH_REPEATS):
                    measured.append(
                        run_batch(
                            label=f"{label}_high",
                            workload="rag",
                            batch_size=HIGH_BATCH,
                            output_tokens=HIGH_OUTPUT_TOKENS,
                            role="measured_high",
                            repeat_index=repeat,
                        )
                    )
            after = {
                "kv": queue_snapshot(core),
                "plan": plan_snapshot(utils),
                "graph_class": (
                    model_runner.model.cudagraph_class_diagnostics()
                    if full_separate_graph
                    else None
                ),
            }
            require(
                before["kv"]["current_target_blocks"]
                == after["kv"]["current_target_blocks"],
                f"{label}: KV target changed during measurement",
            )
            require(
                before["plan"]["active_plan_class"]
                == after["plan"]["active_plan_class"],
                f"{label}: Plan Class changed during measurement",
            )
            result["segments"].append(
                {
                    "label": label,
                    "tier": tier_name,
                    "warmups_excluded": warmups,
                    "measurements": measured,
                    "audit_before": before,
                    "audit_after": after,
                }
            )

        def measure_short_batch_segment(
            *, label: str, tier_name: str, batch_size: int
        ) -> None:
            before = {
                "kv": queue_snapshot(core),
                "plan": plan_snapshot(utils),
                "graph_class": (
                    model_runner.model.cudagraph_class_diagnostics()
                    if full_separate_graph
                    else None
                ),
            }
            tier = tiers[tier_name]
            require(
                int(before["kv"]["current_target_blocks"])
                == int(tier["target_blocks"]),
                f"{label}: wrong target",
            )
            require(
                before["plan"]["active_plan_class"] == tier["plan_class"],
                f"{label}: wrong Plan Class",
            )
            if full_separate_graph:
                require(
                    before["graph_class"].get("active_class")
                    == tier["plan_class"],
                    f"{label}: wrong CUDA Graph Class",
                )
            warmup = run_batch(
                label=label,
                workload="short",
                batch_size=batch_size,
                output_tokens=LOW_OUTPUT_TOKENS,
                role=f"warmup_short_b{batch_size}",
                repeat_index=0,
            )
            measured = [
                run_batch(
                    label=label,
                    workload="short",
                    batch_size=batch_size,
                    output_tokens=LOW_OUTPUT_TOKENS,
                    role=f"measured_short_b{batch_size}",
                    repeat_index=repeat,
                )
                for repeat in range(SHORT_B6_REPEATS)
            ]
            after = {
                "kv": queue_snapshot(core),
                "plan": plan_snapshot(utils),
                "graph_class": (
                    model_runner.model.cudagraph_class_diagnostics()
                    if full_separate_graph
                    else None
                ),
            }
            require(
                before["kv"]["current_target_blocks"]
                == after["kv"]["current_target_blocks"],
                f"{label}: KV target changed during measurement",
            )
            require(
                before["plan"]["active_plan_class"]
                == after["plan"]["active_plan_class"],
                f"{label}: Plan Class changed during measurement",
            )
            result["segments"].append(
                {
                    "label": label,
                    "tier": tier_name,
                    "warmups_excluded": [warmup],
                    "measurements": measured,
                    "audit_before": before,
                    "audit_after": after,
                }
            )

        def measure_rag_pressure_segment(
            *, label: str, tier_name: str, pressure: str, batch_size: int
        ) -> None:
            before = {
                "kv": queue_snapshot(core),
                "plan": plan_snapshot(utils),
                "graph_class": None,
            }
            tier = tiers[tier_name]
            require(
                int(before["kv"]["current_target_blocks"])
                == int(tier["target_blocks"]),
                f"{label}: wrong target",
            )
            require(
                before["plan"]["active_plan_class"] == tier["plan_class"],
                f"{label}: wrong Plan Class",
            )
            warmup = run_batch(
                label=label,
                workload="rag",
                batch_size=batch_size,
                output_tokens=HIGH_OUTPUT_TOKENS,
                role=f"warmup_pressure_{pressure}",
                repeat_index=0,
            )
            measured = [
                run_batch(
                    label=label,
                    workload="rag",
                    batch_size=batch_size,
                    output_tokens=HIGH_OUTPUT_TOKENS,
                    role=f"measured_pressure_{pressure}",
                    repeat_index=repeat,
                )
                for repeat in range(RAG_PRESSURE_REPEATS)
            ]
            after = {
                "kv": queue_snapshot(core),
                "plan": plan_snapshot(utils),
                "graph_class": None,
            }
            require(
                before["kv"]["current_target_blocks"]
                == after["kv"]["current_target_blocks"],
                f"{label}: KV target changed during measurement",
            )
            require(
                before["plan"]["active_plan_class"]
                == after["plan"]["active_plan_class"],
                f"{label}: Plan Class changed during measurement",
            )
            result["segments"].append(
                {
                    "label": label,
                    "tier": tier_name,
                    "pressure": pressure,
                    "warmups_excluded": [warmup],
                    "measurements": measured,
                    "audit_before": before,
                    "audit_after": after,
                }
            )

        def measure_partition_motivation_segment(
            *, label: str, tier_name: str, workload_name: str
        ) -> None:
            before = {
                "kv": queue_snapshot(core),
                "plan": plan_snapshot(utils),
                "graph_class": None,
            }
            tier = tiers[tier_name]
            require(
                int(before["kv"]["current_target_blocks"])
                == int(tier["target_blocks"]),
                f"{label}: wrong fixed KV target",
            )
            require(
                before["plan"]["active_plan_class"] == tier["plan_class"],
                f"{label}: wrong fixed Plan Class",
            )
            if workload_name == "long_c2":
                workload = "motivation_long"
                batch_size = MOTIVATION_LONG_BATCH
                output_tokens = MOTIVATION_LONG_OUTPUT_TOKENS
            else:
                workload = "short"
                batch_size = MOTIVATION_SHORT_BATCH
                output_tokens = MOTIVATION_SHORT_OUTPUT_TOKENS
            measurement = run_batch(
                label=label,
                workload=workload,
                batch_size=batch_size,
                output_tokens=output_tokens,
                role=("diagnostic" if motivation_diagnostic else "measured"),
                repeat_index=args.repeat_index,
            )
            after = {
                "kv": queue_snapshot(core),
                "plan": plan_snapshot(utils),
                "graph_class": None,
            }
            require(
                int(after["kv"]["current_target_blocks"])
                == int(tier["target_blocks"]),
                f"{label}: fixed KV target changed",
            )
            require(
                after["plan"]["active_plan_class"] == tier["plan_class"],
                f"{label}: fixed Plan Class changed",
            )
            result["segments"].append(
                {
                    "label": label,
                    "tier": tier_name,
                    "workload": workload_name,
                    "diagnostic": motivation_diagnostic,
                    "warmup_decode_steps_excluded": MOTIVATION_DECODE_WARMUP_STEPS,
                    "measurement": measurement,
                    "audit_before": before,
                    "audit_after": after,
                }
            )

        if measurement_scope == "partition_motivation":
            tier_name = args.arm.removeprefix("fixed_")
            apply_tier_path(tier_name, "fixed_initial")
            measure_partition_motivation_segment(
                label=f"fixed_{tier_name}_{motivation_workload}",
                tier_name=tier_name,
                workload_name=motivation_workload,
            )
        elif measurement_scope in {"rag_pressure_matrix", "rag_concurrency_sweep"}:
            pressure_points = (
                RAG_PRESSURE_POINTS
                if measurement_scope == "rag_pressure_matrix"
                else RAG_SWEEP_POINTS
            )
            if args.arm == "roundtrip":
                for pressure, batch_size, tier_name in pressure_points:
                    apply_tier_path(tier_name, f"adaptive_{pressure}_apply")
                    measure_rag_pressure_segment(
                        label=f"adaptive_{tier_name}_rag_b{batch_size}",
                        tier_name=tier_name,
                        pressure=pressure,
                        batch_size=batch_size,
                    )
            else:
                tier_name = args.arm.removeprefix("fixed_")
                apply_tier_path(tier_name, "fixed_initial")
                for pressure, batch_size, _ in pressure_points:
                    measure_rag_pressure_segment(
                        label=f"fixed_{tier_name}_rag_b{batch_size}",
                        tier_name=tier_name,
                        pressure=pressure,
                        batch_size=batch_size,
                    )
        elif measurement_scope == "short_b128_abba":
            require(args.arm == "roundtrip", "B128 ABBA requires roundtrip arm")
            for label, tier_name in (
                ("a1_k2", "k2"),
                ("b1_k6", "k6"),
                ("b2_k6", "k6"),
                ("a2_k2", "k2"),
            ):
                apply_tier_path(tier_name, f"{label}_apply")
                measure_short_batch_segment(
                    label=f"{label}_short_b128",
                    tier_name=tier_name,
                    batch_size=SHORT_B128_BATCH,
                )
        elif measurement_scope == "short_b128_shrink_isolation":
            require(
                args.arm == "roundtrip",
                "B128 shrink isolation requires roundtrip arm",
            )
            for label, tier_name in (
                ("a0_k2_initial", "k2"),
                ("b0_k4_grown", "k4"),
                ("a1_k2_after_k4", "k2"),
                ("c0_k6_grown", "k6"),
                ("b1_k4_after_k6", "k4"),
                ("a2_k2_after_k6", "k2"),
            ):
                apply_tier_path(tier_name, f"{label}_apply")
                measure_short_batch_segment(
                    label=f"{label}_short_b128",
                    tier_name=tier_name,
                    batch_size=SHORT_B128_BATCH,
                )
        elif measurement_scope in {"short_matrix", "short_graph_matrix"}:
            tier_name = "k2" if args.arm == "fixed_k2" else "k6"
            apply_tier_path(tier_name, "fixed_initial")
            batches = (
                SHORT_GRAPH_MATRIX_BATCHES
                if measurement_scope == "short_graph_matrix"
                else SHORT_MATRIX_BATCHES
            )
            for short_batch in batches:
                measure_short_batch_segment(
                    label=f"fixed_{tier_name}_short_b{short_batch}",
                    tier_name=tier_name,
                    batch_size=short_batch,
                )
        elif measurement_scope == "short_graph_roundtrip":
            if args.arm in {"fixed_k2", "fixed_k6"}:
                tier_name = "k2" if args.arm == "fixed_k2" else "k6"
                apply_tier_path(tier_name, "fixed_initial")
                for short_batch in SHORT_GRAPH_MATRIX_BATCHES:
                    measure_short_batch_segment(
                        label=f"fixed_{tier_name}_short_b{short_batch}",
                        tier_name=tier_name,
                        batch_size=short_batch,
                    )
            else:
                apply_tier_path("k2", "roundtrip_initial")
                for short_batch in SHORT_GRAPH_MATRIX_BATCHES:
                    measure_short_batch_segment(
                        label=f"adaptive_k2_initial_short_b{short_batch}",
                        tier_name="k2",
                        batch_size=short_batch,
                    )
                apply_tier_path("k6", "roundtrip_grow")
                for short_batch in SHORT_GRAPH_MATRIX_BATCHES:
                    measure_short_batch_segment(
                        label=f"adaptive_k6_after_grow_short_b{short_batch}",
                        tier_name="k6",
                        batch_size=short_batch,
                    )
                apply_tier_path("k2", "roundtrip_shrink")
                for short_batch in SHORT_GRAPH_MATRIX_BATCHES:
                    measure_short_batch_segment(
                        label=f"adaptive_k2_after_return_short_b{short_batch}",
                        tier_name="k2",
                        batch_size=short_batch,
                    )
        elif measurement_scope in {"short_b6", "short_b64", "short_b128"}:
            tier_name = "k2" if args.arm == "fixed_k2" else "k6"
            short_batch = {
                "short_b6": SHORT_B6_BATCH,
                "short_b64": SHORT_B64_BATCH,
                "short_b128": SHORT_B128_BATCH,
            }[measurement_scope]
            apply_tier_path(tier_name, "fixed_initial")
            measure_short_batch_segment(
                label=f"fixed_{tier_name}_short_b{short_batch}",
                tier_name=tier_name,
                batch_size=short_batch,
            )
        elif args.arm == "fixed_k2":
            apply_tier_path("k2", "fixed_initial")
            measure_segment(
                label="fixed_k2",
                tier_name="k2",
                include_low=True,
                include_high=True,
            )
        elif args.arm == "fixed_k6":
            apply_tier_path("k6", "fixed_initial")
            measure_segment(
                label="fixed_k6",
                tier_name="k6",
                include_low=True,
                include_high=True,
            )
        else:
            apply_tier_path("k2", "roundtrip_initial")
            measure_segment(
                label="adaptive_k2_initial",
                tier_name="k2",
                include_low=True,
                include_high=False,
            )
            apply_tier_path("k6", "roundtrip_grow")
            measure_segment(
                label="adaptive_k6",
                tier_name="k6",
                include_low=False,
                include_high=True,
            )
            apply_tier_path("k2", "roundtrip_shrink")
            measure_segment(
                label="adaptive_k2_return",
                tier_name="k2",
                include_low=True,
                include_high=False,
            )

        result["final_audit"] = {
            "kv": queue_snapshot(core),
            "plan": plan_snapshot(utils),
            "graph_counter": dataclasses.asdict(compilation_counter.clone()),
            "graph_class_diagnostics": (
                model_runner.model.cudagraph_class_diagnostics()
                if full_separate_graph
                else None
            ),
            "request_serial": request_serial,
        }
        result["status"] = "ok"
    except BaseException as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        result["finished_at"] = utc_now()
        atomic_json(args.output, result)
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
