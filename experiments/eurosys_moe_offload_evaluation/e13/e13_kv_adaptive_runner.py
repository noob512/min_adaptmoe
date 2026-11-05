#!/usr/bin/env python3
"""Run one cold-process E13 Adaptive or Fixed KV-tier trace."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import statistics
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


QUEUE_KEYS = (
    "total_allocatable_blocks",
    "applied_target_blocks",
    "backup_target_blocks",
    "gpu_free_blocks",
    "gpu_backup_blocks",
    "cpu_offline_blocks",
    "active_blocks",
    "active_block_ids_known",
    "max_active_block_id",
    "tail_free_start_block_id",
    "tail_free_block_count",
    "min_gpu_free_block_id",
    "null_blocks",
    "accounted_blocks",
    "duplicate_memberships",
    "membership_matches_queue_kind",
    "invariant_ok",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
        "p99": percentile(values, 99.0),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def compact(value: Any, depth: int = 0) -> Any:
    if depth > 7:
        return "<depth-limit>"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            text = str(key)
            if text.endswith("_block_ids") or text in {
                "consumer_vas",
                "original_vas",
                "source_vas",
                "expert_plan",
                "plan_class_registry",
            }:
                if isinstance(item, (list, tuple, dict)):
                    result[f"{text}_count"] = len(item)
                continue
            result[text] = compact(item, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        if len(value) > 32:
            return [compact(item, depth + 1) for item in value[:32]] + [
                f"<{len(value) - 32} more>"
            ]
        return [compact(item, depth + 1) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def selected_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(("VLLM_CPU_OFFLOAD_", "VLLM_UVM_"))
        or key
        in {
            "CUDA_VISIBLE_DEVICES",
            "VLLM_ATTENTION_BACKEND",
            "VLLM_ENABLE_V1_MULTIPROCESSING",
            "VLLM_FUSED_MOE_CHUNK_SIZE",
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("qualification", "formal", "capacity"),
        required=True,
    )
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--teacher-targets",
        type=Path,
        help="frozen continuations for the E13 numerical audit",
    )
    parser.add_argument(
        "--teacher-logits-output",
        type=Path,
        help="write recorded raw logits for the E13 numerical audit",
    )
    return parser.parse_args()


def load_teacher_targets(
    path: Path | None,
    *,
    profile: str,
    seed: int,
) -> tuple[dict[str, list[int]], set[str], dict[str, Any] | None]:
    if path is None:
        return {}, set(), None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "eurosys_e13_teacher_targets_v1":
        raise ValueError("unsupported E13 teacher-target schema")
    if payload.get("profile") != profile or int(payload.get("seed", -1)) != seed:
        raise ValueError("teacher targets do not match profile/seed")
    raw_targets = payload.get("targets")
    raw_record = payload.get("record_logical_requests")
    if not isinstance(raw_targets, dict) or not isinstance(raw_record, list):
        raise ValueError("teacher targets are incomplete")
    targets: dict[str, list[int]] = {}
    for logical, values in raw_targets.items():
        if not isinstance(logical, str) or not isinstance(values, list) or not values:
            raise ValueError("invalid teacher target entry")
        targets[logical] = [int(value) for value in values]
    record = {str(value) for value in raw_record}
    if not record <= set(targets):
        raise ValueError("record set is not a subset of teacher targets")
    return targets, record, payload


def load_protocol(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    if protocol.get("schema") != "eurosys_e13_kv_adaptive_protocol_v1":
        raise ValueError("unsupported E13 protocol schema")
    arms = {item["name"]: item for item in protocol["arms"]}
    if args.arm not in arms:
        raise ValueError(f"unknown arm: {args.arm}")
    if args.seed not in protocol["profiles"][args.profile]["seeds"]:
        raise ValueError("seed is not frozen in the selected profile")
    return protocol, arms[args.arm]


def load_fixture(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("requests") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"fixture has no requests: {path}")
    loaded: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        token_ids = record.get("prompt_token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(f"fixture record {index} has no prompt_token_ids")
        tokens = [int(token) for token in token_ids]
        digest = hashlib.sha256(
            ",".join(str(token) for token in tokens).encode("ascii")
        ).hexdigest()
        if int(record.get("prompt_tokens", -1)) != len(tokens):
            raise ValueError(f"fixture record {index} prompt length differs")
        if record.get("prompt_token_ids_sha256") != digest:
            raise ValueError(f"fixture record {index} prompt hash differs")
        loaded.append(
            {
                "fixture_index": index,
                "fixture_request_id": record.get("request", index),
                "prompt_token_ids": tokens,
                "prompt_tokens": len(tokens),
                "prompt_sha256": digest,
            }
        )
    return loaded


def queue_snapshot(core: Any) -> dict[str, Any]:
    snapshot = core.get_uvm_kv_autoscale_snapshot()
    queue = snapshot.get("queue_snapshot") or {}
    result = {key: queue.get(key) for key in QUEUE_KEYS}
    result.update(
        {
            "enabled": bool(snapshot.get("enabled")),
            "controller_step": int(snapshot.get("step", 0)),
            "low_streak": int(snapshot.get("low_streak", 0)),
            "predicted_blocks": int(snapshot.get("predicted_blocks", 0)),
            "prediction_error_blocks": float(
                snapshot.get("prediction_error_blocks", 0.0)
            ),
            "current_target_blocks": int(snapshot.get("current_target_blocks", -1)),
            "pending_plan_class": snapshot.get("pending_plan_class"),
            "last_decision": compact(snapshot.get("last_decision") or {}),
        }
    )
    return result


def plan_snapshot(utils: Any) -> dict[str, Any]:
    diagnostics = utils.get_cpu_offload_graph_static_alias_diagnostics()
    return {
        "lifecycle_state": diagnostics.get("lifecycle_state"),
        "active_plan_class": diagnostics.get("active_plan_class"),
        "initial_plan_class": diagnostics.get("initial_plan_class"),
        "selected_expert_count": diagnostics.get("selected_expert_count"),
        "selected_parameter_count": diagnostics.get("selected_parameter_count"),
        "selected_parameter_bytes_per_forward": diagnostics.get(
            "selected_parameter_bytes_per_forward"
        ),
        "full_parameter_bytes": diagnostics.get("full_parameter_bytes"),
    }


def summarize_requests(
    requests: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    resize_events: list[dict[str, Any]],
) -> dict[str, Any]:
    offered = len(requests)
    successful = [record for record in requests if record.get("success")]
    rejected = [record for record in requests if record.get("rejected")]
    failed = [
        record
        for record in requests
        if not record.get("success") and not record.get("rejected")
    ]
    arrivals = [float(record["arrival_elapsed_s"]) for record in requests]
    finishes = [
        float(record["finished_elapsed_s"])
        for record in successful
        if record.get("finished_elapsed_s") is not None
    ]
    elapsed = max(finishes) - min(arrivals) if finishes and arrivals else 0.0
    completion_tokens = sum(int(record["completion_tokens"]) for record in successful)
    targets = [
        int(step["kv"]["current_target_blocks"])
        for step in steps
        if int(step["kv"].get("current_target_blocks", -1)) >= 0
    ]
    active_blocks = [
        int(step["kv"].get("active_blocks") or 0) for step in steps
    ]
    changed_resize_events = [
        event
        for event in resize_events
        if event["direction"] in {"grow", "shrink"}
    ]
    failed_resize_events = [
        event
        for event in resize_events
        if not bool((event.get("result") or {}).get("ok"))
    ]
    switch_wall_ms = sum(
        float(event["wall_ms"]) for event in changed_resize_events
    )
    switch_response_ms = [
        float(event.get("response_ms", event["wall_ms"]))
        for event in changed_resize_events
    ]
    attempt_wall_ms = sum(float(event["wall_ms"]) for event in resize_events)
    return {
        "offered_requests": offered,
        "successful_requests": len(successful),
        "rejected_requests": len(rejected),
        "failed_requests": len(failed),
        "completion_tokens": completion_tokens,
        "elapsed_seconds": elapsed,
        "successful_request_throughput_per_second": (
            len(successful) / elapsed if elapsed > 0 else 0.0
        ),
        "output_token_throughput_per_second": (
            completion_tokens / elapsed if elapsed > 0 else 0.0
        ),
        "ttft_ms": distribution(
            [float(record["ttft_ms"]) for record in successful]
        ),
        "tpot_ms": distribution(
            [
                float(record["tpot_ms"])
                for record in successful
                if record.get("tpot_ms") is not None
            ]
        ),
        "e2e_ms": distribution(
            [float(record["e2e_ms"]) for record in successful]
        ),
        "queueing_ms": distribution(
            [float(record["queueing_ms"]) for record in successful]
        ),
        "maximum_prompt_tokens_completed": max(
            (int(record["prompt_tokens"]) for record in successful), default=0
        ),
        "maximum_context_tokens_completed": max(
            (
                int(record["prompt_tokens"]) + int(record["completion_tokens"])
                for record in successful
            ),
            default=0,
        ),
        "maximum_scheduler_running_requests": max(
            (int(step["running_request_count"]) for step in steps), default=0
        ),
        "maximum_scheduled_requests_per_step": max(
            (int(step["scheduled_request_count"]) for step in steps), default=0
        ),
        "maximum_active_kv_blocks": max(active_blocks, default=0),
        "minimum_target_blocks": min(targets, default=None),
        "maximum_target_blocks": max(targets, default=None),
        "mean_target_blocks": statistics.fmean(targets) if targets else None,
        "resize_count": len(changed_resize_events),
        "resize_attempt_count": len(resize_events),
        "failed_resize_attempt_count": len(failed_resize_events),
        "grow_count": sum(
            event["direction"] == "grow" for event in changed_resize_events
        ),
        "shrink_count": sum(
            event["direction"] == "shrink" for event in changed_resize_events
        ),
        "resize_wall_ms": switch_wall_ms,
        "resize_response_ms": distribution(switch_response_ms),
        "resize_attempt_wall_ms": attempt_wall_ms,
        "requested_kv_h2d_bytes": sum(
            int(event["requested_bytes"])
            for event in changed_resize_events
            if event["direction"] == "grow"
        ),
        "requested_kv_d2h_bytes": sum(
            int(event["requested_bytes"])
            for event in changed_resize_events
            if event["direction"] == "shrink"
        ),
        "queue_invariant_failures": sum(
            step["kv"].get("invariant_ok") is False for step in steps
        ),
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    protocol, arm = load_protocol(args)
    repo_root = args.repo_root.resolve()
    teacher_targets, teacher_record_logicals, teacher_payload = load_teacher_targets(
        args.teacher_targets,
        profile=args.profile,
        seed=args.seed,
    )
    if bool(args.teacher_targets) != bool(args.teacher_logits_output):
        raise ValueError(
            "--teacher-targets and --teacher-logits-output must be used together"
        )
    if args.teacher_logits_output is not None and args.teacher_logits_output.exists():
        raise SystemExit(
            f"teacher logits output already exists: {args.teacher_logits_output}"
        )
    runtime = protocol["runtime"]
    tier_by_name = {item["name"]: item for item in protocol["tiers"]}
    initial_tier = tier_by_name[arm["initial_tier"]]
    allowed_targets = {int(item["target_blocks"]) for item in protocol["tiers"]}
    scenarios = {
        item["name"]: item for item in protocol["scenarios"]
    }
    selected_scenarios = [
        scenarios[name]
        for name in protocol["profiles"][args.profile]["scenario_names"]
    ]

    fixture_paths = {
        name: resolve_path(repo_root, value)
        for name, value in protocol["fixtures"].items()
    }
    fixtures = {name: load_fixture(path) for name, path in fixture_paths.items()}
    fixture_cursors = {
        name: args.seed % len(records) for name, records in fixtures.items()
    }

    frozen_trace: list[dict[str, Any]] = []
    cursor_copy = dict(fixture_cursors)
    for scenario in selected_scenarios:
        for epoch in scenario["epochs"]:
            frozen_epoch = {
                "scenario": scenario["name"],
                "label": epoch["label"],
                "arrival_gap_ms": epoch["arrival_gap_ms"],
                "cohorts": [],
            }
            for cohort in epoch["cohorts"]:
                records = fixtures[cohort["workload"]]
                selected_indices = [
                    (cursor_copy[cohort["workload"]] + offset) % len(records)
                    for offset in range(int(cohort["batch_size"]))
                ]
                cursor_copy[cohort["workload"]] = (
                    cursor_copy[cohort["workload"]] + int(cohort["batch_size"])
                ) % len(records)
                frozen_epoch["cohorts"].append(
                    {
                        **cohort,
                        "fixture_indices": selected_indices,
                        "prompt_sha256": [
                            records[index]["prompt_sha256"] for index in selected_indices
                        ],
                    }
                )
            frozen_trace.append(frozen_epoch)
    trace_sha256 = sha256_json(frozen_trace)

    result: dict[str, Any] = {
        "schema": "eurosys_e13_kv_adaptive_run_v1",
        "status": "loading",
        "started_at": utc_now(),
        "arm": args.arm,
        "strategy": arm["strategy"],
        "profile": args.profile,
        "seed": args.seed,
        "repeat_index": args.repeat_index,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "trace_sha256": trace_sha256,
        "configuration": {
            "model": protocol["model"],
            "fixtures": {name: str(path) for name, path in fixture_paths.items()},
            "initial_tier": initial_tier,
            "allowed_target_blocks": sorted(allowed_targets),
            "runtime": runtime,
            "environment": selected_environment(),
            "frozen_trace": frozen_trace,
            "teacher_forced_numerical_audit": (
                {
                    "enabled": True,
                    "target_path": str(args.teacher_targets.resolve()),
                    "target_sha256": hashlib.sha256(
                        args.teacher_targets.read_bytes()
                    ).hexdigest(),
                    "source_run_root": teacher_payload.get("source_run_root"),
                    "record_logical_requests": sorted(teacher_record_logicals),
                    "timing_valid": False,
                }
                if teacher_payload is not None
                else {"enabled": False}
            ),
        },
        "initialization": {},
        "epochs": [],
        "requests": [],
        "time_series": [],
        "controller_events": [],
        "resize_events": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)

    llm: Any | None = None
    try:
        import torch
        from vllm import LLM, SamplingParams
        from vllm.compilation.counter import compilation_counter
        from vllm.model_executor.models import utils
        from vllm.sampling_params import RequestOutputKind

        logits_processors = None
        export_teacher_records = None
        if teacher_payload is not None:
            from e13_teacher_forced_logits import (
                E13TeacherForcedLogitsProcessor,
                export_records,
                reset_records,
            )

            reset_records()
            logits_processors = [E13TeacherForcedLogitsProcessor]
            export_teacher_records = export_records

        counter_before_load = dataclasses.asdict(compilation_counter.clone())
        load_started = time.perf_counter()
        llm = LLM(
            model=protocol["model"],
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=float(runtime["gpu_memory_utilization"]),
            enforce_eager=True,
            max_model_len=int(runtime["max_model_len"]),
            max_num_seqs=int(runtime["max_num_seqs"]),
            max_num_batched_tokens=int(runtime["max_num_batched_tokens"]),
            kv_cache_memory_bytes=int(runtime["kv_cache_memory_bytes"]),
            cpu_offload_gb=float(runtime["cpu_offload_gb"]),
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            enable_dbo=False,
            compilation_config={
                "mode": "NONE",
                "cudagraph_mode": "NONE",
            },
            seed=args.seed,
            disable_log_stats=True,
            logits_processors=logits_processors,
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        counter_after_load = dataclasses.asdict(compilation_counter.clone())

        core = llm.llm_engine.engine_core.engine_core
        scheduler = core.scheduler
        policy = scheduler.kv_cache_manager.uvm_kv_runtime_policy
        block_pool = scheduler.kv_cache_manager.block_pool
        if not policy.dynamic_pool_active():
            raise RuntimeError("E13 requires the managed dynamic KV pool")
        if int(policy.bytes_per_block) != int(runtime["kv_block_bytes"]):
            raise RuntimeError(
                f"KV block bytes differ: {policy.bytes_per_block} != "
                f"{runtime['kv_block_bytes']}"
            )
        if int(block_pool.hash_block_size) != int(runtime["kv_block_tokens"]):
            raise RuntimeError("KV block token size differs from frozen protocol")

        setup_before = queue_snapshot(core)
        setup_started = time.perf_counter()
        setup_old_target = int(setup_before["current_target_blocks"])
        setup_target = int(initial_tier["target_blocks"])
        if bool(runtime.get("synchronized_tier_steps")):
            ordered_targets = sorted(int(value) for value in allowed_targets)
            if setup_target >= setup_old_target:
                setup_targets = [
                    value
                    for value in ordered_targets
                    if setup_old_target < value <= setup_target
                ]
            else:
                setup_targets = [
                    value
                    for value in reversed(ordered_targets)
                    if setup_target <= value < setup_old_target
                ]
        else:
            setup_targets = [setup_target]
        if not setup_targets:
            setup_targets = [setup_target]
        setup_steps: list[dict[str, Any]] = []
        for step_target in setup_targets:
            step_before = queue_snapshot(core)
            step_result = core.apply_uvm_kv_dynamic_pool_target(
                int(step_target),
                reason=f"e13_initialization_{args.arm}",
                metadata={"release_expert_wave": False},
            )
            setup_steps.append(
                {
                    "old_target_blocks": int(
                        step_before["current_target_blocks"]
                    ),
                    "new_target_blocks": int(step_target),
                    "result": compact(step_result),
                }
            )
            if not step_result.get("ok"):
                break
        setup_result = {
            "ok": bool(setup_steps)
            and all(step["result"].get("ok") for step in setup_steps),
            "reason": "synchronized_initial_tier_steps",
            "tier_steps": setup_steps,
        }
        torch.cuda.synchronize()
        setup_wall_ms = (time.perf_counter() - setup_started) * 1000.0
        setup_after = queue_snapshot(core)
        setup_plan = plan_snapshot(utils)
        if not setup_result.get("ok"):
            raise RuntimeError(f"initial tier setup failed: {setup_result!r}")
        if setup_after["current_target_blocks"] != int(initial_tier["target_blocks"]):
            raise RuntimeError("initial physical KV target differs")
        if setup_plan["active_plan_class"] != initial_tier["plan_class"]:
            raise RuntimeError(
                "initial expert plan differs: "
                f"{setup_plan['active_plan_class']} != {initial_tier['plan_class']}"
            )
        result["initialization"] = {
            "load_seconds": load_seconds,
            "graph_counter_before_load": counter_before_load,
            "graph_counter_after_load": counter_after_load,
            "tier_setup_wall_ms": setup_wall_ms,
            "tier_setup_result": compact(setup_result),
            "kv_before": setup_before,
            "kv_after": setup_after,
            "plan_after": setup_plan,
        }

        trace_started_perf = time.perf_counter()
        trace_started_wall = time.time()
        result["trace_started_unix_s"] = trace_started_wall
        original_resize = core._resize_uvm_kv_target
        resize_context = {"scenario": "", "epoch": "", "source": "runtime"}
        pending_trigger_elapsed_s: dict[int, float] = {}

        def traced_resize(
            target_blocks: int,
            *,
            reason: str,
            release_expert_wave: bool,
            allow_active_prefix_shrink: bool = False,
        ) -> dict[str, Any]:
            before = queue_snapshot(core)
            started = time.perf_counter()
            trigger_elapsed_s = pending_trigger_elapsed_s.get(
                int(target_blocks), started - trace_started_perf
            )
            value = original_resize(
                target_blocks,
                reason=reason,
                release_expert_wave=release_expert_wave,
                allow_active_prefix_shrink=allow_active_prefix_shrink,
            )
            torch.cuda.synchronize()
            finished = time.perf_counter()
            after = queue_snapshot(core)
            old_target = int(before["current_target_blocks"])
            new_target = int(after["current_target_blocks"])
            direction = (
                "grow" if new_target > old_target else "shrink" if new_target < old_target else "hold"
            )
            result["resize_events"].append(
                {
                    "event_index": len(result["resize_events"]),
                    "elapsed_s": finished - trace_started_perf,
                    "scenario": resize_context["scenario"],
                    "epoch": resize_context["epoch"],
                    "source": resize_context["source"],
                    "reason": reason,
                    "requested_target_blocks": int(target_blocks),
                    "old_target_blocks": old_target,
                    "new_target_blocks": new_target,
                    "direction": direction,
                    "requested_bytes": abs(new_target - old_target)
                    * int(policy.bytes_per_block),
                    "wall_ms": (finished - started) * 1000.0,
                    "trigger_elapsed_s": trigger_elapsed_s,
                    "response_ms": (
                        finished - trace_started_perf - trigger_elapsed_s
                    )
                    * 1000.0,
                    "release_expert_wave": bool(release_expert_wave),
                    "allow_active_prefix_shrink": bool(allow_active_prefix_shrink),
                    "result": compact(value),
                    "kv_before": before,
                    "plan_after": plan_snapshot(utils),
                    "kv_after": after,
                }
            )
            if direction in {"grow", "shrink"}:
                pending_trigger_elapsed_s.pop(int(target_blocks), None)
            return value

        core._resize_uvm_kv_target = traced_resize
        original_observe = core._observe_uvm_kv_autoscale_pressure

        def traced_observe(scheduler_output: Any) -> dict[str, Any]:
            started = time.perf_counter()
            decision = original_observe(scheduler_output)
            finished = time.perf_counter()
            action = str(decision.get("action") or "")
            pending_target = decision.get("pending_shrink_target_blocks")
            elapsed_s = finished - trace_started_perf
            if action == "shrink_pending" and pending_target is not None:
                pending_trigger_elapsed_s.setdefault(
                    int(pending_target), elapsed_s
                )
            elif pending_target is None:
                pending_trigger_elapsed_s.clear()
            result["controller_events"].append(
                {
                    "elapsed_s": elapsed_s,
                    "scenario": resize_context["scenario"],
                    "epoch": resize_context["epoch"],
                    "wall_ms": (finished - started) * 1000.0,
                    "decision": compact(decision),
                }
            )
            return decision

        core._observe_uvm_kv_autoscale_pressure = traced_observe

        scheduler_records: list[dict[str, Any]] = []
        original_schedule = scheduler.schedule
        current_location = {"scenario": "", "epoch": ""}

        def traced_schedule() -> Any:
            waiting_before = [request.request_id for request in scheduler.waiting]
            running_before = [request.request_id for request in scheduler.running]
            started = time.perf_counter()
            output = original_schedule()
            finished = time.perf_counter()
            new_ids = [request.req_id for request in output.scheduled_new_reqs]
            cached_ids = list(output.scheduled_cached_reqs.req_ids)
            scheduler_records.append(
                {
                    "scenario": current_location["scenario"],
                    "epoch": current_location["epoch"],
                    "started_perf": started,
                    "finished_perf": finished,
                    "waiting_before": waiting_before,
                    "running_before": running_before,
                    "scheduled_new_request_ids": new_ids,
                    "scheduled_cached_request_ids": cached_ids,
                    "num_scheduled_tokens": {
                        str(key): int(value)
                        for key, value in output.num_scheduled_tokens.items()
                    },
                    "total_num_scheduled_tokens": int(
                        output.total_num_scheduled_tokens
                    ),
                    "preempted_request_ids": sorted(
                        str(value)
                        for value in (getattr(output, "preempted_req_ids", None) or ())
                    ),
                }
            )
            return output

        scheduler.schedule = traced_schedule
        request_state: dict[str, dict[str, Any]] = {}
        global_request_index = 0

        for scenario in selected_scenarios:
            scenario_name = str(scenario["name"])
            for epoch in scenario["epochs"]:
                epoch_label = str(epoch["label"])
                current_location.update(scenario=scenario_name, epoch=epoch_label)
                resize_context.update(
                    scenario=scenario_name, epoch=epoch_label, source="runtime"
                )
                gap_ms = float(epoch.get("arrival_gap_ms", 0))
                if gap_ms > 0:
                    time.sleep(gap_ms / 1000.0)
                if llm.llm_engine.has_unfinished_requests():
                    raise RuntimeError("epoch boundary is not idle")

                epoch_started_perf = time.perf_counter()
                epoch_started_wall = time.time()
                epoch_resize_cursor = len(result["resize_events"])
                initial_cohort = epoch["cohorts"][0]
                initial_records = fixtures[initial_cohort["workload"]]
                initial_indices = [
                    (fixture_cursors[initial_cohort["workload"]] + offset)
                    % len(initial_records)
                    for offset in range(int(initial_cohort["batch_size"]))
                ]
                predicted_prompt_counts = [
                    int(initial_records[index]["prompt_tokens"])
                    for index in initial_indices
                ]
                controller_before = queue_snapshot(core)
                controller_started = time.perf_counter()
                resize_context["source"] = "next_request_prediction"
                if arm["strategy"] == "adaptive":
                    controller_result = core.apply_uvm_kv_next_request_pressure(
                        predicted_prompt_counts,
                        int(initial_cohort["output_tokens"]),
                        reason=f"e13_{scenario_name}_{epoch_label}",
                    )
                else:
                    controller_result = {
                        "ok": True,
                        "reason": "fixed_initialization_only_no_runtime_decision",
                    }
                torch.cuda.synchronize()
                controller_finished = time.perf_counter()
                resize_context["source"] = "runtime_observation"
                controller_after = queue_snapshot(core)
                if not controller_result.get("ok"):
                    raise RuntimeError(
                        f"next-request controller failed: {controller_result!r}"
                    )
                if arm["strategy"] == "fixed" and (
                    controller_after["current_target_blocks"]
                    != int(initial_tier["target_blocks"])
                ):
                    raise RuntimeError("fixed arm changed target before admission")

                cohort_controller_events: list[dict[str, Any]] = [
                    {
                        "cohort_index": 0,
                        "after_engine_steps": int(
                            initial_cohort["after_engine_steps"]
                        ),
                        "late_arrival": False,
                        "arrival_elapsed_s": (
                            controller_started - trace_started_perf
                        ),
                        "controller_wall_ms": (
                            controller_finished - controller_started
                        )
                        * 1000.0,
                        "controller_before": controller_before,
                        "controller_after": controller_after,
                        "controller_result": compact(controller_result),
                        "deferred_growth": bool(
                            (controller_result.get("autoscale") or {}).get(
                                "pending"
                            )
                        ),
                    }
                ]

                epoch_request_ids: list[str] = []
                next_cohort = 0
                engine_steps = 0
                scheduler_cursor = len(scheduler_records)

                def admit_cohort(
                    cohort_index: int,
                    arrival_perf: float,
                    *,
                    deferred_for_kv_grow: bool = False,
                ) -> None:
                    nonlocal global_request_index
                    cohort = epoch["cohorts"][cohort_index]
                    workload = str(cohort["workload"])
                    records = fixtures[workload]
                    admitted_perf = time.perf_counter()
                    for cohort_offset in range(int(cohort["batch_size"])):
                        fixture_index = fixture_cursors[workload] % len(records)
                        fixture_cursors[workload] = (
                            fixture_cursors[workload] + 1
                        ) % len(records)
                        fixture = records[fixture_index]
                        request_id = f"r{global_request_index:04d}"
                        global_request_index += 1
                        logical_request = (
                            f"{scenario_name}/{epoch_label}/c{cohort_index}/"
                            f"{cohort_offset}"
                        )
                        target_token_ids = teacher_targets.get(logical_request)
                        if teacher_payload is not None:
                            if target_token_ids is None:
                                raise RuntimeError(
                                    "teacher target is missing for "
                                    f"{logical_request}"
                                )
                            if len(target_token_ids) != int(cohort["output_tokens"]):
                                raise RuntimeError(
                                    "teacher target length differs for "
                                    f"{logical_request}: {len(target_token_ids)} != "
                                    f"{cohort['output_tokens']}"
                                )
                        params = SamplingParams(
                            n=1,
                            temperature=0.0,
                            top_p=1.0,
                            max_tokens=int(cohort["output_tokens"]),
                            min_tokens=int(cohort["output_tokens"]),
                            ignore_eos=True,
                            seed=args.seed,
                            output_kind=RequestOutputKind.DELTA,
                            extra_args=(
                                {
                                    "e13_teacher_forced": {
                                        "logical_request": logical_request,
                                        "target_token_ids": target_token_ids,
                                        "record_logits": logical_request
                                        in teacher_record_logicals,
                                    }
                                }
                                if target_token_ids is not None
                                else None
                            ),
                        )
                        state = {
                            "request_id": request_id,
                            "logical_request": logical_request,
                            "scenario": scenario_name,
                            "epoch": epoch_label,
                            "cohort_index": cohort_index,
                            "cohort_after_engine_steps": int(
                                cohort["after_engine_steps"]
                            ),
                            "workload": workload,
                            "fixture_index": fixture_index,
                            "fixture_request_id": fixture["fixture_request_id"],
                            "prompt_tokens": fixture["prompt_tokens"],
                            "prompt_token_ids_sha256": fixture["prompt_sha256"],
                            "requested_output_tokens": int(cohort["output_tokens"]),
                            "arrival_perf": arrival_perf,
                            "arrival_elapsed_s": arrival_perf - trace_started_perf,
                            "arrival_unix_s": time.time(),
                            "admission_elapsed_s": (
                                admitted_perf - trace_started_perf
                            ),
                            "admission_delay_ms": (
                                admitted_perf - arrival_perf
                            )
                            * 1000.0,
                            "admission_deferred_for_kv_grow": bool(
                                deferred_for_kv_grow
                            ),
                            "first_scheduled_perf": None,
                            "token_times": [],
                            "output_token_ids": [],
                            "finished_perf": None,
                            "rejected": False,
                            "rejection_reason": None,
                        }
                        request_state[request_id] = state
                        epoch_request_ids.append(request_id)
                        try:
                            processed = llm.processor.process_inputs(
                                request_id,
                                {"prompt_token_ids": fixture["prompt_token_ids"]},
                                params,
                                arrival_time=time.time(),
                                priority=0,
                            )
                            llm.llm_engine.add_request(
                                request_id,
                                processed,
                                params,
                                priority=0,
                            )
                        except BaseException as exc:
                            state["rejected"] = True
                            state["rejection_reason"] = (
                                f"{type(exc).__name__}: {exc}"
                            )

                # The initial requests logically arrive before the online
                # controller runs, so their TTFT/E2E includes switching cost.
                admit_cohort(0, controller_started)
                next_cohort = 1
                deferred_cohort: dict[str, Any] | None = None

                while (
                    llm.llm_engine.has_unfinished_requests()
                    or next_cohort < len(epoch["cohorts"])
                    or deferred_cohort is not None
                ):
                    if deferred_cohort is not None and getattr(
                        core, "_uvm_pending_kv_grow_target_blocks", None
                    ) is None:
                        expected_target = int(deferred_cohort["target_blocks"])
                        actual_target = int(
                            queue_snapshot(core)["current_target_blocks"]
                        )
                        if actual_target != expected_target:
                            raise RuntimeError(
                                "deferred cohort target was not committed: "
                                f"expected={expected_target} actual={actual_target}"
                            )
                        admit_cohort(
                            int(deferred_cohort["cohort_index"]),
                            float(deferred_cohort["arrival_perf"]),
                            deferred_for_kv_grow=True,
                        )
                        deferred_cohort = None

                    while next_cohort < len(epoch["cohorts"]) and int(
                        epoch["cohorts"][next_cohort]["after_engine_steps"]
                    ) <= engine_steps:
                        if deferred_cohort is not None:
                            break
                        late_arrival_perf = time.perf_counter()
                        cohort = epoch["cohorts"][next_cohort]
                        workload = str(cohort["workload"])
                        records = fixtures[workload]
                        predicted_indices = [
                            (fixture_cursors[workload] + offset) % len(records)
                            for offset in range(int(cohort["batch_size"]))
                        ]
                        predicted_counts = [
                            int(records[index]["prompt_tokens"])
                            for index in predicted_indices
                        ]
                        late_before = queue_snapshot(core)
                        late_started = time.perf_counter()
                        resize_context["source"] = "late_cohort_prediction"
                        if arm["strategy"] == "adaptive":
                            late_result = core.apply_uvm_kv_next_request_pressure(
                                predicted_counts,
                                int(cohort["output_tokens"]),
                                reason=(
                                    f"e13_{scenario_name}_{epoch_label}_"
                                    f"cohort_{next_cohort}"
                                ),
                            )
                        else:
                            late_result = {
                                "ok": True,
                                "reason": (
                                    "fixed_initialization_only_no_runtime_decision"
                                ),
                            }
                        torch.cuda.synchronize()
                        late_finished = time.perf_counter()
                        resize_context["source"] = "runtime_observation"
                        late_after = queue_snapshot(core)
                        if not late_result.get("ok"):
                            raise RuntimeError(
                                "late-cohort controller failed: "
                                f"{late_result!r}"
                            )
                        growth_pending = bool(
                            (late_result.get("autoscale") or {}).get("pending")
                        )
                        cohort_controller_events.append(
                            {
                                "cohort_index": next_cohort,
                                "after_engine_steps": int(
                                    cohort["after_engine_steps"]
                                ),
                                "late_arrival": True,
                                "arrival_elapsed_s": (
                                    late_arrival_perf - trace_started_perf
                                ),
                                "controller_wall_ms": (
                                    late_finished - late_started
                                )
                                * 1000.0,
                                "controller_before": late_before,
                                "controller_after": late_after,
                                "controller_result": compact(late_result),
                                "deferred_growth": growth_pending,
                            }
                        )
                        if growth_pending:
                            target_blocks = int(
                                late_result["autoscale"]["new_target_blocks"]
                            )
                            if getattr(
                                core,
                                "_uvm_pending_kv_grow_target_blocks",
                                None,
                            ) != target_blocks:
                                raise RuntimeError(
                                    "late-cohort growth was not registered as pending"
                                )
                            deferred_cohort = {
                                "cohort_index": next_cohort,
                                "arrival_perf": late_arrival_perf,
                                "target_blocks": target_blocks,
                            }
                        else:
                            admit_cohort(next_cohort, late_arrival_perf)
                        next_cohort += 1
                        if deferred_cohort is not None:
                            break
                    if not llm.llm_engine.has_unfinished_requests():
                        if deferred_cohort is not None:
                            raise RuntimeError(
                                "pending KV grow was not committed when the "
                                "preceding cohort reached the idle boundary"
                            )
                        if next_cohort < len(epoch["cohorts"]):
                            raise RuntimeError(
                                "earlier cohort drained before the frozen late-arrival step"
                            )
                        break

                    schedule_index = len(scheduler_records)
                    step_started = time.perf_counter()
                    step_outputs = llm.llm_engine.step()
                    step_finished = time.perf_counter()
                    engine_steps += 1
                    if len(scheduler_records) <= schedule_index:
                        raise RuntimeError("engine step did not invoke scheduler")
                    schedule = scheduler_records[-1]
                    for request_id in (
                        schedule["scheduled_new_request_ids"]
                        + schedule["scheduled_cached_request_ids"]
                    ):
                        if (
                            request_id in request_state
                            and request_state[request_id]["first_scheduled_perf"] is None
                        ):
                            request_state[request_id]["first_scheduled_perf"] = schedule[
                                "finished_perf"
                            ]
                    emitted_tokens = 0
                    for output in step_outputs:
                        request_id = str(output.request_id)
                        state = request_state.get(request_id)
                        if state is None:
                            raise RuntimeError(f"unexpected request output: {request_id}")
                        if len(output.outputs) != 1:
                            raise RuntimeError("E13 requires one output choice")
                        delta = [int(token) for token in output.outputs[0].token_ids]
                        emitted_tokens += len(delta)
                        state["output_token_ids"].extend(delta)
                        state["token_times"].extend([step_finished] * len(delta))
                        if output.finished:
                            state["finished_perf"] = step_finished

                    kv = queue_snapshot(core)
                    plan = plan_snapshot(utils)
                    scheduled_ids = set(schedule["num_scheduled_tokens"])
                    result["time_series"].append(
                        {
                            "elapsed_s": step_finished - trace_started_perf,
                            "unix_s": time.time(),
                            "scenario": scenario_name,
                            "epoch": epoch_label,
                            "engine_step": engine_steps,
                            "step_wall_ms": (step_finished - step_started) * 1000.0,
                            "scheduler_wall_ms": (
                                float(schedule["finished_perf"])
                                - float(schedule["started_perf"])
                            )
                            * 1000.0,
                            "waiting_request_count": len(schedule["waiting_before"]),
                            "running_request_count": len(schedule["running_before"]),
                            "scheduled_request_count": len(scheduled_ids),
                            "scheduled_new_request_count": len(
                                schedule["scheduled_new_request_ids"]
                            ),
                            "scheduled_cached_request_count": len(
                                schedule["scheduled_cached_request_ids"]
                            ),
                            "scheduled_tokens": int(
                                schedule["total_num_scheduled_tokens"]
                            ),
                            "emitted_tokens": emitted_tokens,
                            "instant_output_tokens_per_second": (
                                emitted_tokens / (step_finished - step_started)
                                if step_finished > step_started
                                else 0.0
                            ),
                            "preempted_request_count": len(
                                schedule["preempted_request_ids"]
                            ),
                            "kv": kv,
                            "active_plan_class": plan["active_plan_class"],
                        }
                    )

                epoch_finished_perf = time.perf_counter()
                epoch_requests: list[dict[str, Any]] = []
                for request_id in epoch_request_ids:
                    state = request_state[request_id]
                    token_times = list(state.pop("token_times"))
                    token_ids = list(state.pop("output_token_ids"))
                    arrival_perf = float(state.pop("arrival_perf"))
                    first_scheduled_perf = state.pop("first_scheduled_perf")
                    finished_perf = state.pop("finished_perf")
                    inter_token_ms = [
                        (right - left) * 1000.0
                        for left, right in zip(token_times, token_times[1:])
                    ]
                    success = (
                        not state["rejected"]
                        and finished_perf is not None
                        and len(token_ids) == int(state["requested_output_tokens"])
                    )
                    record = {
                        **state,
                        "completion_tokens": len(token_ids),
                        "output_token_ids": token_ids,
                        "output_token_ids_sha256": sha256_json(token_ids),
                        "success": success,
                        "finished_elapsed_s": (
                            float(finished_perf) - trace_started_perf
                            if finished_perf is not None
                            else None
                        ),
                        "queueing_ms": (
                            (float(first_scheduled_perf) - arrival_perf) * 1000.0
                            if first_scheduled_perf is not None
                            else None
                        ),
                        "ttft_ms": (
                            (token_times[0] - arrival_perf) * 1000.0
                            if token_times
                            else None
                        ),
                        "tpot_ms": (
                            statistics.fmean(inter_token_ms)
                            if inter_token_ms
                            else None
                        ),
                        "e2e_ms": (
                            (float(finished_perf) - arrival_perf) * 1000.0
                            if finished_perf is not None
                            else None
                        ),
                        "itl_ms": inter_token_ms,
                    }
                    result["requests"].append(record)
                    epoch_requests.append(record)

                epoch_steps = result["time_series"][scheduler_cursor:]
                epoch_resizes = result["resize_events"][epoch_resize_cursor:]
                result["epochs"].append(
                    {
                        "scenario": scenario_name,
                        "label": epoch_label,
                        "arrival_gap_ms": gap_ms,
                        "started_elapsed_s": epoch_started_perf - trace_started_perf,
                        "started_unix_s": epoch_started_wall,
                        "finished_elapsed_s": epoch_finished_perf - trace_started_perf,
                        "engine_steps": engine_steps,
                        "controller_wall_ms": (
                            controller_finished - controller_started
                        )
                        * 1000.0,
                        "controller_before": controller_before,
                        "controller_after": controller_after,
                        "controller_result": compact(controller_result),
                        "cohort_controller_events": cohort_controller_events,
                        "request_ids": epoch_request_ids,
                        "summary": summarize_requests(
                            epoch_requests, epoch_steps, epoch_resizes
                        ),
                    }
                )
                atomic_json(args.output, result)

        trace_finished_perf = time.perf_counter()
        result["trace_finished_unix_s"] = time.time()
        result["trace_elapsed_seconds"] = trace_finished_perf - trace_started_perf
        result["final_kv"] = queue_snapshot(core)
        result["final_plan"] = plan_snapshot(utils)
        result["graph_counter_after_trace"] = dataclasses.asdict(
            compilation_counter.clone()
        )

        for request in result["requests"]:
            if request.get("success") and request.get("ttft_ms") is None:
                raise RuntimeError("successful request has no TTFT")
        scenario_summaries: dict[str, Any] = {}
        for scenario in selected_scenarios:
            name = scenario["name"]
            request_subset = [
                record for record in result["requests"] if record["scenario"] == name
            ]
            step_subset = [
                record for record in result["time_series"] if record["scenario"] == name
            ]
            resize_subset = [
                record for record in result["resize_events"] if record["scenario"] == name
            ]
            scenario_summaries[name] = summarize_requests(
                request_subset, step_subset, resize_subset
            )
        result["scenario_summaries"] = scenario_summaries
        result["summary"] = summarize_requests(
            result["requests"], result["time_series"], result["resize_events"]
        )
        result["summary"]["trace_elapsed_seconds"] = result[
            "trace_elapsed_seconds"
        ]
        result["summary"]["preempted_requests"] = sum(
            int(step["preempted_request_count"]) for step in result["time_series"]
        )
        result["summary"]["target_switches"] = [
            {
                key: event[key]
                for key in (
                    "elapsed_s",
                    "scenario",
                    "epoch",
                    "reason",
                    "old_target_blocks",
                    "new_target_blocks",
                    "direction",
                    "wall_ms",
                )
            }
            for event in result["resize_events"]
            if event["direction"] != "hold"
        ]
        result["output_hash_by_logical_request"] = {
            record["logical_request"]: record["output_token_ids_sha256"]
            for record in result["requests"]
            if record.get("success")
        }
        if any(
            step["kv"].get("invariant_ok") is False
            for step in result["time_series"]
        ):
            raise RuntimeError("KV queue invariant failed")
        observed_targets = {
            int(step["kv"]["current_target_blocks"])
            for step in result["time_series"]
        }
        if not observed_targets <= allowed_targets:
            raise RuntimeError(f"unregistered KV target observed: {observed_targets}")
        plan_class_by_target = {
            int(item["target_blocks"]): str(item["plan_class"])
            for item in protocol["tiers"]
        }
        mismatched_tier_steps = [
            {
                "engine_step": step["engine_step"],
                "target_blocks": step["kv"]["current_target_blocks"],
                "active_plan_class": step["active_plan_class"],
            }
            for step in result["time_series"]
            if step["active_plan_class"]
            != plan_class_by_target[int(step["kv"]["current_target_blocks"])]
        ]
        if mismatched_tier_steps:
            raise RuntimeError(
                f"KV/Plan Class tier mismatch: {mismatched_tier_steps[:3]}"
            )
        if arm["strategy"] == "fixed" and observed_targets != {
            int(initial_tier["target_blocks"])
        }:
            raise RuntimeError(f"fixed target changed: {observed_targets}")
        if arm["strategy"] == "adaptive" and observed_targets != allowed_targets:
            raise RuntimeError(
                f"adaptive tier coverage incomplete: {observed_targets}"
            )
        if any(
            not bool((event.get("result") or {}).get("ok"))
            for event in result["resize_events"]
        ):
            raise RuntimeError("one or more runtime resize attempts failed")
        final_target = int(result["final_kv"]["current_target_blocks"])
        if result["final_plan"]["active_plan_class"] != plan_class_by_target[
            final_target
        ]:
            raise RuntimeError("final KV target and Plan Class differ")
        if result["final_kv"].get("pending_plan_class") is not None:
            raise RuntimeError("a Plan Class activation remains pending")
        if teacher_payload is not None:
            if any(
                record["output_token_ids"]
                != teacher_targets[record["logical_request"]]
                for record in result["requests"]
            ):
                raise RuntimeError("teacher-forced output differs from frozen target")
            assert export_teacher_records is not None
            assert args.teacher_logits_output is not None
            result["teacher_forced_logits"] = export_teacher_records(
                args.teacher_logits_output
            )
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
