#!/usr/bin/env python3
"""One-shot E13 switched-tier versus direct-tier Prefill/Decode measurement."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
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


DECODE_WARMUP_STEPS = 5
DECODE_MEASURED_STEPS = 16
OUTPUT_TOKENS = 32
BATCH_SIZES = (1, 8)
SOURCE_TIER = {"k2": "k4", "k4": "k6", "k6": "k4"}
CONTROLLED_TARGET_TOKEN_IDS = (
    288,
    11,
    19438,
    4403,
    752,
    281,
    12487,
    1709,
    348,
    12589,
    264,
    94565,
    33335,
    409,
    5141,
    220,
    18,
    11,
    902,
    80815,
    11,
    27795,
    13526,
    902,
    66890,
    28742,
    409,
    6216,
    264,
    1187,
    3835,
    23737,
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


def summarize(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def strict_decode(schedule: dict[str, Any], batch_size: int) -> bool:
    scheduled = schedule["num_scheduled_tokens"]
    cached = schedule["cached_request_ids"]
    return (
        not schedule["new_request_ids"]
        and len(scheduled) == batch_size
        and len(cached) == batch_size
        and set(scheduled) == set(cached)
        and all(int(value) == 1 for value in scheduled.values())
        and int(schedule["total_num_scheduled_tokens"]) == batch_size
        and not schedule["resumed_request_ids"]
        and not schedule["preempted_request_ids"]
        and not schedule["finished_request_ids"]
    )


def main() -> int:
    args = parse_args()
    require(not args.output.exists(), f"output already exists: {args.output}")
    mechanism, separator, tier_name = args.arm.partition("_")
    require(separator == "_" and mechanism in {"direct", "switched"}, "invalid arm")
    require(tier_name in SOURCE_TIER, "invalid tier")
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    require(protocol.get("schema") == "eurosys_e13_kv_adaptive_protocol_v1", "bad config")
    runtime = protocol["runtime"]
    tiers = {item["name"]: item for item in protocol["tiers"]}
    target_tier = tiers[tier_name]
    source_tier = tiers[SOURCE_TIER[tier_name]]
    repo_root = args.repo_root.resolve()
    fixture_path = resolve_path(repo_root, protocol["fixtures"]["medium"])
    fixtures = load_fixture(fixture_path)
    require(len(fixtures) >= 24, "Medium fixture needs at least 24 records")
    require(all(item["prompt_tokens"] == 2048 for item in fixtures[:24]), "fixture length differs")

    result: dict[str, Any] = {
        "schema": "eurosys_e13_tier_steady_phase_v1",
        "status": "loading",
        "started_at": utc_now(),
        "arm": args.arm,
        "mechanism": mechanism,
        "target_tier": tier_name,
        "source_tier": source_tier["name"] if mechanism == "switched" else None,
        "seed": args.seed,
        "repeat_index": args.repeat_index,
        "configuration": {
            "model": protocol["model"],
            "fixture": str(fixture_path),
            "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
            "prompt_tokens_per_request": 2048,
            "batch_sizes": list(BATCH_SIZES),
            "output_tokens": OUTPUT_TOKENS,
            "decode_warmup_steps": DECODE_WARMUP_STEPS,
            "decode_measured_steps": DECODE_MEASURED_STEPS,
            "controlled_decode_tokens": True,
            "controlled_target_token_ids_sha256": sha256_json(
                list(CONTROLLED_TARGET_TOKEN_IDS)
            ),
            "controlled_token_logits_recording": False,
            "source_conditioning_excluded": True,
            "target_tier_warmup_excluded": True,
            "switch_transaction_excluded": True,
            "execution_mode": "eager",
            "runtime": runtime,
        },
        "initialization": {},
        "source_conditioning": None,
        "target_warmup": {},
        "phases": [],
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
            before_kv = queue_snapshot(core)
            before_plan = plan_snapshot(utils)
            torch.cuda.synchronize()
            started = time.perf_counter()
            value = core.apply_uvm_kv_dynamic_pool_target(
                int(tier["target_blocks"]),
                reason=label,
                metadata={"release_expert_wave": False},
            )
            require(bool(value.get("ok")), f"tier application failed: {value!r}")
            # A KV shrink commits the physical target first and defers the
            # matching expert Plan Class until an idle scheduler boundary.
            # This runner applies tiers only while the engine is idle, so
            # execute that deferred maintenance here and keep it inside the
            # excluded switch transaction.  Without this call a shrink such
            # as K4 -> K2 would measure K2 KV capacity with K4 placement.
            core._note_uvm_kv_autoscale_resize(int(tier["target_blocks"]))
            if isinstance(value.get("plan_class_before_grow"), dict):
                core._uvm_kv_autoscale_pending_plan_class = None
            pending_plan = core._apply_pending_uvm_expert_plan_class_if_ready()
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - started) * 1000.0
            after_kv = queue_snapshot(core)
            after_plan = plan_snapshot(utils)
            require(
                pending_plan is None or bool(pending_plan.get("ok")),
                f"pending Plan Class application failed: {pending_plan!r}",
            )
            require(
                int(after_kv["current_target_blocks"]) == int(tier["target_blocks"]),
                "physical KV target differs",
            )
            require(after_plan["active_plan_class"] == tier["plan_class"], "Plan Class differs")
            transitions.append(
                {
                    "label": label,
                    "wall_ms_excluded": wall_ms,
                    "result": value,
                    "pending_plan_result": pending_plan,
                    "before_kv": before_kv,
                    "after_kv": after_kv,
                    "before_plan": before_plan,
                    "after_plan": after_plan,
                }
            )

        if mechanism == "switched":
            apply_tier(source_tier, f"e13_phase_source_{source_tier['name']}")
            # Exercise the source tier as a real serving state before the
            # runtime switch.  Back-to-back pre-service Plan Class changes do
            # not establish the original-VA Prefill/Decode lifecycle and can
            # leave the first Decode transition without a valid remote view.
            # This conditioning request is fully drained and excluded.
            source_sampling = SamplingParams(
                max_tokens=OUTPUT_TOKENS,
                min_tokens=OUTPUT_TOKENS,
                temperature=0.0,
                seed=args.seed,
                ignore_eos=True,
                extra_args={
                    "e13_teacher_forced": {
                        "logical_request": f"{args.arm}/source_conditioning/0",
                        "target_token_ids": list(CONTROLLED_TARGET_TOKEN_IDS),
                        "record_logits": False,
                    }
                },
            )
            torch.cuda.synchronize()
            source_started = time.perf_counter()
            source_outputs = llm.generate(
                [TokensPrompt(prompt_token_ids=list(fixtures[0]["prompt_token_ids"]))],
                source_sampling,
                use_tqdm=False,
            )
            torch.cuda.synchronize()
            source_elapsed_ms = (time.perf_counter() - source_started) * 1000.0
            require(
                not llm.llm_engine.has_unfinished_requests(),
                "source conditioning did not drain",
            )
            source_records = phase.output_records(source_outputs)
            require(
                len(source_records) == 1
                and len(source_records[0]["output_token_ids"]) == OUTPUT_TOKENS,
                "source conditioning output differs",
            )
            result["source_conditioning"] = {
                "tier": source_tier["name"],
                "batch_size": 1,
                "prompt_tokens_per_request": 2048,
                "output_tokens_per_request": OUTPUT_TOKENS,
                "wall_ms_excluded": source_elapsed_ms,
                "output_token_ids_sha256": phase.output_hash(source_records),
            }
            apply_tier(target_tier, f"e13_phase_runtime_switch_{tier_name}")
        else:
            apply_tier(target_tier, f"e13_phase_direct_initial_{tier_name}")
        result["initialization"] = {
            "load_seconds": time.perf_counter() - load_started,
            "transitions": transitions,
            "counter_before": counter_before,
            "counter_after": dataclasses.asdict(compilation_counter.clone()),
            "measurement_started": False,
        }

        schedule_records: list[dict[str, Any]] = []
        original_schedule = scheduler.schedule

        def traced_schedule() -> Any:
            output = original_schedule()
            schedule_records.append(phase.schedule_snapshot(output))
            return output

        scheduler.schedule = traced_schedule

        def run_step() -> dict[str, Any]:
            before = len(schedule_records)
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = llm.llm_engine.step()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            require(len(schedule_records) == before + 1, "scheduler call count differs")
            return {
                "elapsed_ms": elapsed_ms,
                "outputs": outputs,
                "schedule": schedule_records[-1],
            }

        cursor = 0

        def run_phase(label: str, batch_size: int, output_tokens: int, measure: bool) -> dict[str, Any]:
            nonlocal cursor
            require(not llm.llm_engine.has_unfinished_requests(), f"{label}: engine busy")
            selected = [fixtures[(cursor + index) % len(fixtures)] for index in range(batch_size)]
            cursor = (cursor + batch_size) % len(fixtures)
            prompts = [
                TokensPrompt(prompt_token_ids=list(item["prompt_token_ids"]))
                for item in selected
            ]
            require(
                output_tokens == len(CONTROLLED_TARGET_TOKEN_IDS),
                f"{label}: controlled target length differs",
            )
            sampling = [
                SamplingParams(
                    max_tokens=output_tokens,
                    min_tokens=output_tokens,
                    temperature=0.0,
                    seed=args.seed,
                    ignore_eos=True,
                    extra_args={
                        "e13_teacher_forced": {
                            "logical_request": f"{args.arm}/{label}/{index}",
                            "target_token_ids": list(CONTROLLED_TARGET_TOKEN_IDS),
                            "record_logits": False,
                        }
                    },
                )
                for index in range(batch_size)
            ]
            llm._validate_and_add_requests(
                prompts=prompts,
                params=sampling,
                use_tqdm=False,
                lora_request=None,
                priority=None,
            )
            prefill: list[dict[str, Any]] = []
            first_decode: dict[str, Any] | None = None
            for _ in range(32):
                step = run_step()
                require(not step["schedule"]["preempted_request_ids"], f"{label}: preemption")
                require(not step["schedule"]["resumed_request_ids"], f"{label}: resume")
                require(not step["outputs"], f"{label}: finished during Prefill")
                if strict_decode(step["schedule"], batch_size):
                    first_decode = step
                    break
                prefill.append({key: value for key, value in step.items() if key != "outputs"})
            require(first_decode is not None, f"{label}: strict Decode not reached")
            prefill_scheduled = sum(
                int(item["schedule"]["total_num_scheduled_tokens"]) for item in prefill
            )
            require(prefill_scheduled == 2048 * batch_size, f"{label}: Prefill tokens differ")

            warmup_decode = [float(first_decode["elapsed_ms"])]
            for _ in range(DECODE_WARMUP_STEPS - 1):
                step = run_step()
                require(strict_decode(step["schedule"], batch_size), f"{label}: warm Decode differs")
                require(not step["outputs"], f"{label}: warm Decode finished early")
                warmup_decode.append(float(step["elapsed_ms"]))

            measured_decode: list[float] = []
            for _ in range(DECODE_MEASURED_STEPS):
                step = run_step()
                require(strict_decode(step["schedule"], batch_size), f"{label}: measured Decode differs")
                require(not step["outputs"], f"{label}: measured Decode finished early")
                measured_decode.append(float(step["elapsed_ms"]))

            final_outputs: list[Any] = []
            for _ in range(output_tokens + 8):
                step = run_step()
                if step["outputs"]:
                    final_outputs = step["outputs"]
                    break
            require(final_outputs and not llm.llm_engine.has_unfinished_requests(), f"{label}: drain failed")
            records = phase.output_records(final_outputs)
            require(len(records) == batch_size, f"{label}: output batch differs")
            require(
                all(len(record["output_token_ids"]) == output_tokens for record in records),
                f"{label}: output length differs",
            )
            prefill_ms = sum(float(item["elapsed_ms"]) for item in prefill)
            return {
                "label": label,
                "measured": measure,
                "batch_size": batch_size,
                "prompt_tokens_per_request": 2048,
                "output_tokens_per_request": output_tokens,
                "prefill_step_count": len(prefill),
                "prefill_scheduled_tokens": prefill_scheduled,
                "prefill_ms": prefill_ms,
                "prefill_tokens_per_second": prefill_scheduled * 1000.0 / prefill_ms,
                "decode_warmup_step_ms": summarize(warmup_decode),
                "steady_decode_step_ms": summarize(measured_decode),
                "steady_decode_tokens_per_second": (
                    batch_size * 1000.0 / statistics.median(measured_decode)
                ),
                "output_token_ids_sha256": phase.output_hash(records),
            }

        # Identical target-tier conditioning for both mechanisms.  Neither this
        # phase nor any transition above contributes to the measured timings.
        result["target_warmup"] = run_phase("target_warmup_b8", 8, OUTPUT_TOKENS, False)
        before_measurement = {
            "kv": queue_snapshot(core),
            "plan": plan_snapshot(utils),
        }
        result["initialization"]["measurement_started"] = True
        result["phases"] = [
            run_phase("medium_b1", 1, OUTPUT_TOKENS, True),
            run_phase("medium_b8", 8, OUTPUT_TOKENS, True),
        ]
        after_measurement = {
            "kv": queue_snapshot(core),
            "plan": plan_snapshot(utils),
        }
        require(before_measurement["kv"]["current_target_blocks"] == int(target_tier["target_blocks"]), "bad target before measurement")
        require(after_measurement["kv"]["current_target_blocks"] == int(target_tier["target_blocks"]), "target changed during measurement")
        require(before_measurement["plan"]["active_plan_class"] == target_tier["plan_class"], "bad plan before measurement")
        require(after_measurement["plan"]["active_plan_class"] == target_tier["plan_class"], "plan changed during measurement")
        result["measurement_audit"] = {
            "before": before_measurement,
            "after": after_measurement,
            "graph_counter_after": dataclasses.asdict(compilation_counter.clone()),
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
