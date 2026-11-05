#!/usr/bin/env python3
"""Same-process ABBA diagnosis for E13 post-switch Decode state."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE_ROOT = REPO_ROOT / "experiments" / "vllm_p3_pinned_uva_phase_compare"
sys.path.insert(0, str(PHASE_ROOT))

import benchmark_phase_steps as phase  # noqa: E402

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
    DECODE_MEASURED_STEPS,
    DECODE_WARMUP_STEPS,
    OUTPUT_TOKENS,
    SOURCE_TIER,
    strict_decode,
)


STATS_PATH = Path("/proc/driver/nvidia-uvm/latency_stats")
COUNTER_RE = re.compile(r"^\s*([a-z][a-z0-9_]*)\s+(0x[0-9a-fA-F]+|\d+)\s*$")
COUNTER_TOKENS = (
    "cpu_chunk",
    "pte",
    "tlb",
    "slot_alias_dual",
    "immutable_shadow",
    "remote_map",
)
PRIMARY_ORDER = (
    ("a1_direct", "direct"),
    ("b1_switched", "switched"),
    ("b2_switched", "switched"),
    ("a2_direct", "direct"),
    ("s_switch_only", "switch_only"),
    ("c_target_condition", "target_condition"),
)


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
    parser.add_argument("--arm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def driver_counter_snapshot() -> dict[str, int]:
    try:
        text = STATS_PATH.read_text(encoding="utf-8")
    except OSError:
        return {}
    counters: dict[str, int] = {}
    for line in text.splitlines():
        match = COUNTER_RE.fullmatch(line)
        if not match:
            continue
        name, raw = match.groups()
        if any(token in name for token in COUNTER_TOKENS):
            counters[name] = int(raw, 0)
    return counters


def counter_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(set(before) | set(after))
        if int(after.get(key, 0)) != int(before.get(key, 0))
    }


def exact_plan_snapshot(utils: Any) -> dict[str, Any]:
    raw = utils.get_cpu_offload_graph_static_alias_diagnostics()
    expert_plan = [
        [int(layer), [int(expert) for expert in experts]]
        for layer, experts in raw.get("expert_plan", ())
    ]
    arenas = [
        {
            "layer_container": str(arena.get("layer_container")),
            "local_name": str(arena.get("local_name")),
            "bank_id": int(arena.get("bank_id", -1)),
            "pte_mode": arena.get("pte_mode"),
            "pte_alignment": int(arena.get("pte_alignment", 0)),
            "consumer_count": len(arena.get("consumer_vas", ())),
        }
        for arena in raw.get("arenas", ())
    ]
    pte_mode_counts: dict[str, int] = {}
    for arena in arenas:
        mode = str(arena.get("pte_mode"))
        pte_mode_counts[mode] = pte_mode_counts.get(mode, 0) + 1
    return {
        **plan_snapshot(utils),
        "expert_plan_sha256": digest(expert_plan),
        "expert_plan": expert_plan,
        "arena_pte_mode_counts": pte_mode_counts,
        "arenas_sha256": digest(arenas),
        "arenas": arenas,
        "dual_view_leaf_mode": "4K",
        "dual_view_parent_switch_granularity": "2M PDE",
        "dual_view_64k_leaf_count": 0,
    }


def block_state_snapshot(scheduler: Any, schedule: dict[str, Any]) -> dict[str, Any]:
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    raw = pool.snapshot_uvm_dynamic_queues()
    request_tables: dict[str, Any] = {}
    request_ids = sorted(str(value) for value in schedule["num_scheduled_tokens"])
    for request_id in request_ids:
        groups = [
            [int(block_id) for block_id in group]
            for group in manager.get_block_ids(request_id)
        ]
        request_tables[request_id] = {
            "groups": groups,
            "ordered_sha256": digest(groups),
            "block_count": sum(len(group) for group in groups),
        }
    queues = {}
    for name in (
        "gpu_free_block_ids",
        "gpu_backup_block_ids",
        "cpu_offline_block_ids",
        "active_block_ids",
    ):
        values = [int(value) for value in raw.get(name, ())]
        queues[name] = {
            "count": len(values),
            "ordered_sha256": digest(values),
            "head": values[:16],
            "tail": values[-16:],
        }
    return {
        "request_tables": request_tables,
        "request_tables_sha256": digest(
            {key: value["groups"] for key, value in request_tables.items()}
        ),
        "queues": queues,
        "queue_invariant_ok": bool(raw.get("invariant_ok")),
    }


def normalize_heat(raw: list[Any]) -> dict[str, Any]:
    require(len(raw) == 1, "single worker heat snapshot required")
    worker = raw[0]
    require(isinstance(worker, dict) and worker.get("ok") is True, "heat RPC failed")
    layers: dict[str, Any] = {}
    for layer_name, value in sorted(worker.get("layers", {}).items()):
        counts = [int(item) for item in value.get("expert_token_counts", ())]
        remote = sorted(
            {
                int(item)
                for key in ("remote_experts", "static_alias_owned_experts")
                for item in value.get(key, ())
            }
        )
        active = [[index, count] for index, count in enumerate(counts) if count]
        remote_set = set(remote)
        remote_active = [item for item in active if item[0] in remote_set]
        layers[str(layer_name)] = {
            "layer_id": value.get("layer_id"),
            "route_sha256": digest(counts),
            "active_expert_counts": active,
            "total_assignments": sum(counts),
            "remote_experts_sha256": digest(remote),
            "remote_expert_count": len(remote),
            "remote_assignments": remote_active,
            "remote_assignment_count": sum(item[1] for item in remote_active),
        }
    compact = {
        name: {
            "route_sha256": value["route_sha256"],
            "remote_experts_sha256": value["remote_experts_sha256"],
        }
        for name, value in layers.items()
    }
    return {
        "layer_count": len(layers),
        "route_sha256": digest(compact),
        "total_assignments": sum(item["total_assignments"] for item in layers.values()),
        "remote_assignment_count": sum(
            item["remote_assignment_count"] for item in layers.values()
        ),
        "layers": layers,
    }


def kernel_summary(profiler: Any) -> dict[str, Any]:
    rows: dict[str, dict[str, float | int]] = {}
    cuda_ranges: list[tuple[float, float]] = []
    try:
        import torch

        cuda_type = torch.autograd.DeviceType.CUDA
        for event in profiler.events():
            if getattr(event, "device_type", None) != cuda_type:
                continue
            name = str(getattr(event, "name", "unknown"))
            time_range = getattr(event, "time_range", None)
            elapsed_us = float(time_range.elapsed_us()) if time_range is not None else 0.0
            if time_range is not None:
                cuda_ranges.append((float(time_range.start), float(time_range.end)))
            row = rows.setdefault(name, {"count": 0, "total_us": 0.0})
            row["count"] = int(row["count"]) + 1
            row["total_us"] = float(row["total_us"]) + elapsed_us
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "kernels": []}
    kernels = [
        {"name": name, **value}
        for name, value in sorted(rows.items(), key=lambda item: -float(item[1]["total_us"]))
    ]
    categories = {"moe": 0.0, "attention": 0.0, "other": 0.0}
    for row in kernels:
        lowered = str(row["name"]).lower()
        category = "other"
        if any(token in lowered for token in ("moe", "expert", "grouped_gemm")):
            category = "moe"
        elif any(token in lowered for token in ("attention", "paged", "mla", "flash")):
            category = "attention"
        categories[category] += float(row["total_us"])
    total_us = sum(float(row["total_us"]) for row in kernels)
    span_us = (
        max(value[1] for value in cuda_ranges) - min(value[0] for value in cuda_ranges)
        if cuda_ranges
        else 0.0
    )
    return {
        "ok": True,
        "kernel_count": sum(int(row["count"]) for row in kernels),
        "category_total_us": categories,
        "cuda_kernel_total_us": total_us,
        "cuda_kernel_span_us": span_us,
        "cuda_inter_kernel_gap_us": max(0.0, span_us - total_us),
        "kernels": kernels,
    }


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"output already exists: {args.output}")
    mechanism, separator, tier_name = args.arm.partition("_")
    require(separator == "_" and mechanism == "direct", "diagnostic arm must be direct_TIER")
    require(tier_name in {"k4", "k6"}, "diagnostic tier must be k4 or k6")
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    runtime = protocol["runtime"]
    tiers = {item["name"]: item for item in protocol["tiers"]}
    target_tier = tiers[tier_name]
    source_tier = tiers[SOURCE_TIER[tier_name]]
    repo_root = args.repo_root.resolve()
    fixture_path = resolve_path(repo_root, protocol["fixtures"]["medium"])
    fixtures = load_fixture(fixture_path)
    require(len(fixtures) >= 24, "Medium fixture needs at least 24 records")
    heavy = os.environ.get("E13_DECODE_DIAGNOSTIC_HEAVY", "0") == "1"
    profile_step_indices = {
        int(value.strip())
        for value in os.environ.get("E13_DECODE_PROFILE_STEP_INDICES", "0").split(",")
        if value.strip()
    }
    selected_labels = {
        value.strip()
        for value in os.environ.get("E13_DECODE_DIAGNOSTIC_CONDITIONS", "").split(",")
        if value.strip()
    }
    condition_order = [
        item for item in PRIMARY_ORDER if not selected_labels or item[0] in selected_labels
    ]
    require(bool(condition_order), "no diagnostic conditions selected")
    require(
        not selected_labels or selected_labels == {item[0] for item in condition_order},
        "unknown diagnostic condition selected",
    )
    result: dict[str, Any] = {
        "schema": "eurosys_e13_decode_abba_diagnostic_v1",
        "status": "loading",
        "started_at": utc_now(),
        "arm": args.arm,
        "target_tier": tier_name,
        "source_tier": source_tier["name"],
        "seed": args.seed,
        "heavy_diagnostics": heavy,
        "configuration": {
            "fixture": str(fixture_path),
            "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "condition_order": condition_order,
            "controlled_target_token_ids_sha256": sha256_json(list(CONTROLLED_TARGET_TOKEN_IDS)),
            "decode_warmup_steps": DECODE_WARMUP_STEPS,
            "decode_measured_steps": DECODE_MEASURED_STEPS,
            "timed_switches": False,
            "execution_mode": "eager",
        },
        "initialization": {},
        "conditions": [],
        "step_diagnostics": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)

    llm: Any | None = None
    try:
        import torch
        from vllm import LLM, SamplingParams, TokensPrompt
        from vllm.compilation.counter import compilation_counter
        from vllm.model_executor.models import utils
        from e13_teacher_forced_logits import E13TeacherForcedLogitsProcessor

        if heavy:
            require(os.environ.get("VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE") == "1", "heavy pass requires expert heat counters")
        counter_before = dataclasses.asdict(compilation_counter.clone())
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
            compilation_config={"mode": "NONE", "cudagraph_mode": "NONE"},
            seed=args.seed,
            disable_log_stats=True,
            logits_processors=[E13TeacherForcedLogitsProcessor],
        )
        phase.enable_request_metrics_without_default_loggers(llm)
        torch.cuda.synchronize()
        core = llm.llm_engine.engine_core.engine_core
        scheduler = core.scheduler
        policy = scheduler.kv_cache_manager.uvm_kv_runtime_policy
        require(policy.dynamic_pool_active(), "managed dynamic KV pool is absent")
        transitions: list[dict[str, Any]] = []

        def apply_tier(tier: dict[str, Any], label: str) -> None:
            before_driver = driver_counter_snapshot()
            before_kv = queue_snapshot(core)
            before_plan = exact_plan_snapshot(utils)
            torch.cuda.synchronize()
            started = time.perf_counter()
            value = core.apply_uvm_kv_dynamic_pool_target(
                int(tier["target_blocks"]),
                reason=label,
                metadata={"release_expert_wave": False},
            )
            require(bool(value.get("ok")), f"tier application failed: {value!r}")
            core._note_uvm_kv_autoscale_resize(int(tier["target_blocks"]))
            if isinstance(value.get("plan_class_before_grow"), dict):
                core._uvm_kv_autoscale_pending_plan_class = None
            pending = core._apply_pending_uvm_expert_plan_class_if_ready()
            torch.cuda.synchronize()
            after_kv = queue_snapshot(core)
            after_plan = exact_plan_snapshot(utils)
            after_driver = driver_counter_snapshot()
            require(pending is None or bool(pending.get("ok")), f"pending plan failed: {pending!r}")
            require(int(after_kv["current_target_blocks"]) == int(tier["target_blocks"]), "KV target differs")
            require(after_plan["active_plan_class"] == tier["plan_class"], "Plan Class differs")
            transitions.append({
                "label": label,
                "wall_ms_excluded": (time.perf_counter() - started) * 1000.0,
                "result": value,
                "pending_plan_result": pending,
                "before_kv": before_kv,
                "after_kv": after_kv,
                "before_plan": before_plan,
                "after_plan": after_plan,
                "driver_counter_delta": counter_delta(before_driver, after_driver),
            })

        schedule_records: list[dict[str, Any]] = []
        original_schedule = scheduler.schedule

        def traced_schedule() -> Any:
            output = original_schedule()
            snapshot = phase.schedule_snapshot(output)
            if heavy:
                snapshot["kv_blocks"] = block_state_snapshot(scheduler, snapshot)
            schedule_records.append(snapshot)
            return output

        scheduler.schedule = traced_schedule
        if heavy:
            reset = core.collective_rpc("get_uvm_dynamic_expert_heat", args=(True,))
            normalize_heat(reset)

        current_condition = "initialization"
        current_phase = "initialization"
        current_stage = "initialization"
        current_stage_index = 0

        def run_step(*, profile_kernel: bool = False) -> dict[str, Any]:
            before_schedule = len(schedule_records)
            before_driver = driver_counter_snapshot() if heavy else {}
            unix_started = time.time()
            torch.cuda.synchronize()
            profiler = None
            if profile_kernel:
                profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                profiler.__enter__()
            torch.cuda.nvtx.range_push(f"e13/{tier_name}/{current_condition}/{current_phase}/{current_stage}/{current_stage_index}")
            started = time.perf_counter()
            try:
                outputs = llm.llm_engine.step()
                torch.cuda.synchronize()
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                torch.cuda.nvtx.range_pop()
                if profiler is not None:
                    profiler.__exit__(None, None, None)
            unix_finished = time.time()
            require(len(schedule_records) == before_schedule + 1, "scheduler call count differs")
            route = None
            after_driver: dict[str, int] = {}
            if heavy:
                route = normalize_heat(core.collective_rpc("get_uvm_dynamic_expert_heat", args=(True,)))
                after_driver = driver_counter_snapshot()
            record = {
                "global_step": len(result["step_diagnostics"]),
                "condition": current_condition,
                "phase": current_phase,
                "stage": current_stage,
                "stage_index": current_stage_index,
                "elapsed_ms": elapsed_ms,
                "unix_started": unix_started,
                "unix_finished": unix_finished,
                "schedule": schedule_records[-1],
            }
            if heavy:
                record["route"] = route
                record["driver_counter_delta"] = counter_delta(before_driver, after_driver)
                if profiler is not None:
                    record["kernel_profile"] = kernel_summary(profiler)
            result["step_diagnostics"].append(record)
            return {"elapsed_ms": elapsed_ms, "outputs": outputs, "schedule": schedule_records[-1], "diagnostic_step": record["global_step"]}

        def run_phase(label: str, fixture_indices: list[int], *, measured: bool) -> dict[str, Any]:
            nonlocal current_phase, current_stage, current_stage_index
            batch_size = len(fixture_indices)
            require(not llm.llm_engine.has_unfinished_requests(), f"{label}: engine busy")
            prompts = [TokensPrompt(prompt_token_ids=list(fixtures[index]["prompt_token_ids"])) for index in fixture_indices]
            sampling = [
                SamplingParams(
                    max_tokens=OUTPUT_TOKENS,
                    min_tokens=OUTPUT_TOKENS,
                    temperature=0.0,
                    seed=args.seed,
                    ignore_eos=True,
                    extra_args={"e13_teacher_forced": {
                        "logical_request": f"{tier_name}/{current_condition}/{label}/{index}",
                        "target_token_ids": list(CONTROLLED_TARGET_TOKEN_IDS),
                        "record_logits": False,
                    }},
                )
                for index in range(batch_size)
            ]
            llm._validate_and_add_requests(prompts=prompts, params=sampling, use_tqdm=False, lora_request=None, priority=None)
            current_phase = label
            prefill: list[dict[str, Any]] = []
            first_decode: dict[str, Any] | None = None
            for index in range(32):
                current_stage, current_stage_index = "prefill", index
                step = run_step()
                require(not step["schedule"]["preempted_request_ids"], f"{label}: preemption")
                require(not step["schedule"]["resumed_request_ids"], f"{label}: resume")
                require(not step["outputs"], f"{label}: finished during Prefill")
                if strict_decode(step["schedule"], batch_size):
                    first_decode = step
                    break
                prefill.append(step)
            require(first_decode is not None, f"{label}: strict Decode not reached")
            prefill_tokens = sum(int(item["schedule"]["total_num_scheduled_tokens"]) for item in prefill)
            require(prefill_tokens == 2048 * batch_size, f"{label}: Prefill tokens differ")
            warm = [float(first_decode["elapsed_ms"])]
            for index in range(1, DECODE_WARMUP_STEPS):
                current_stage, current_stage_index = "decode_warmup", index
                step = run_step()
                require(strict_decode(step["schedule"], batch_size), f"{label}: warm Decode differs")
                require(not step["outputs"], f"{label}: warm Decode finished early")
                warm.append(float(step["elapsed_ms"]))
            measured_values: list[float] = []
            measured_steps: list[int] = []
            window_started = time.time()
            for index in range(DECODE_MEASURED_STEPS):
                current_stage, current_stage_index = "decode_measured", index
                step = run_step(profile_kernel=heavy and index in profile_step_indices)
                require(strict_decode(step["schedule"], batch_size), f"{label}: measured Decode differs")
                require(not step["outputs"], f"{label}: measured Decode finished early")
                measured_values.append(float(step["elapsed_ms"]))
                measured_steps.append(int(step["diagnostic_step"]))
            window_finished = time.time()
            final_outputs: list[Any] = []
            for index in range(OUTPUT_TOKENS + 8):
                current_stage, current_stage_index = "drain", index
                step = run_step()
                if step["outputs"]:
                    final_outputs = step["outputs"]
                    break
            require(final_outputs and not llm.llm_engine.has_unfinished_requests(), f"{label}: drain failed")
            records = phase.output_records(final_outputs)
            require(len(records) == batch_size, f"{label}: output batch differs")
            require(all(len(record["output_token_ids"]) == OUTPUT_TOKENS for record in records), f"{label}: output length differs")
            prefill_ms = sum(float(item["elapsed_ms"]) for item in prefill)
            return {
                "label": label,
                "measured": measured,
                "fixture_indices": fixture_indices,
                "prompt_hashes": [fixtures[index]["prompt_sha256"] for index in fixture_indices],
                "batch_size": batch_size,
                "prefill_step_count": len(prefill),
                "prefill_scheduled_tokens": prefill_tokens,
                "prefill_ms": prefill_ms,
                "decode_warmup_step_ms": summarize(warm),
                "steady_decode_step_ms": summarize(measured_values),
                "measured_step_indices": measured_steps,
                "measurement_unix_started": window_started,
                "measurement_unix_finished": window_finished,
                "output_token_ids_sha256": phase.output_hash(records),
            }

        def run_condition(label: str, kind: str) -> dict[str, Any]:
            nonlocal current_condition
            current_condition = label
            source_run = None
            cuda_profiler_capture = os.environ.get("E13_NSYS_CUDA_PROFILER_CAPTURE", "0") == "1"
            if cuda_profiler_capture:
                torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(f"e13/{tier_name}/{label}/condition_window")
            try:
                if kind == "switched":
                    apply_tier(source_tier, f"{label}_source_{source_tier['name']}")
                    source_run = run_phase("source_condition", [0], measured=False)
                    apply_tier(target_tier, f"{label}_target_{tier_name}")
                elif kind == "switch_only":
                    apply_tier(source_tier, f"{label}_source_{source_tier['name']}")
                    apply_tier(target_tier, f"{label}_target_{tier_name}")
                elif kind == "target_condition":
                    apply_tier(target_tier, f"{label}_target_{tier_name}")
                    source_run = run_phase("source_condition_at_target", [0], measured=False)
                elif kind == "direct":
                    apply_tier(target_tier, f"{label}_target_{tier_name}")
                else:
                    raise AssertionError(kind)
                warmup = run_phase("common_target_warmup_b8", list(range(8)), measured=False)
                before_driver = driver_counter_snapshot()
                before = {"kv": queue_snapshot(core), "plan": exact_plan_snapshot(utils)}
                phases = [
                    run_phase("medium_b1", [8], measured=True),
                    run_phase("medium_b8", list(range(9, 17)), measured=True),
                ]
                after = {"kv": queue_snapshot(core), "plan": exact_plan_snapshot(utils)}
                after_driver = driver_counter_snapshot()
            finally:
                torch.cuda.nvtx.range_pop()
                if cuda_profiler_capture:
                    torch.cuda.cudart().cudaProfilerStop()
            require(before["plan"]["active_plan_class"] == target_tier["plan_class"], "bad plan before measurement")
            require(after["plan"]["active_plan_class"] == target_tier["plan_class"], "bad plan after measurement")
            return {
                "label": label,
                "kind": kind,
                "source_run": source_run,
                "target_warmup": warmup,
                "before": before,
                "phases": phases,
                "after": after,
                "measurement_driver_counter_delta": counter_delta(before_driver, after_driver),
            }

        apply_tier(target_tier, f"initial_target_{tier_name}")
        result["initialization"] = {
            "load_seconds": time.perf_counter() - load_started,
            "counter_before": counter_before,
            "counter_after": dataclasses.asdict(compilation_counter.clone()),
        }
        for label, kind in condition_order:
            result["conditions"].append(run_condition(label, kind))
            atomic_json(args.output, result)
        result["transitions"] = transitions
        result["final_driver_counters"] = driver_counter_snapshot()
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
