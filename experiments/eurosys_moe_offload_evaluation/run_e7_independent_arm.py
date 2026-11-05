#!/usr/bin/env python3
"""Run exactly one E7 arm in one freshly started model process."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import statistics
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

from run_e7_workload_shift_sequence import (
    compact,
    digest_json,
    finalize_arm_metrics,
    initialize_dynamic_slots,
    load_fixture,
    load_plan,
    model_runner_from_llm,
    placement_snapshot,
    require,
    run_arm,
    selected_plan_bytes,
    sha256_file,
    state_target,
    write_result,
)


ARM_ORDER = (
    "frequency_cold",
    "random",
    "frozen_static",
    "gpu_heat_cpu_ewma",
    "oracle",
    "hot_adversarial",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARM_ORDER)
    parser.add_argument("--model", required=True)
    parser.add_argument("--shift-fixture", required=True, type=Path)
    parser.add_argument("--static-plan", required=True, type=Path)
    parser.add_argument("--superset-plan", required=True, type=Path)
    parser.add_argument("--frozen-targets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--ewma-alpha", type=float, default=0.25)
    parser.add_argument("--max-swaps-per-layer", type=int, default=1)
    parser.add_argument("--min-count-delta", type=int, default=1)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=268435456)
    parser.add_argument("--cpu-offload-gb", type=float, default=30.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def load_targets(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "frozen target payload is not an object")
    require(
        payload.get("schema") == "eurosys_e7_independent_cold_targets_v1",
        "frozen target schema differs",
    )
    require(payload.get("frozen_before_run") is True, "targets were not frozen")
    targets = payload.get("targets")
    require(isinstance(targets, dict), "frozen targets absent")
    require(tuple(targets) == ARM_ORDER, "frozen target order differs")
    for name, target in targets.items():
        require(isinstance(target, dict) and len(target) == 48, f"{name}: layer count")
        require(
            all(
                isinstance(values, list)
                and len(values) == 4
                and len(set(int(value) for value in values)) == 4
                for values in target.values()
            ),
            f"{name}: invalid four-slot target",
        )
    return payload


def adaptive_transfer_metrics(
    arm: dict[str, Any],
    initial_state: dict[str, Any],
) -> tuple[int, int, int]:
    swap_count = 0
    h2d_bytes = 0
    d2h_bytes = 0
    for window in arm["windows"]:
        swaps = ((window["maintenance"].get("plan") or {}).get("swaps_by_layer") or {})
        for layer_key, records in swaps.items():
            layer = initial_state["layers"][layer_key]
            for record in records:
                swap_count += 1
                h2d_bytes += int(
                    layer["expert_region_bytes"][str(int(record["promote_expert"]))]
                )
                d2h_bytes += int(
                    layer["expert_region_bytes"][str(int(record["demote_expert"]))]
                )
    return swap_count, h2d_bytes, d2h_bytes


def main() -> int:
    args = parse_args()
    require(0.0 <= args.ewma_alpha <= 1.0, "EWMA alpha outside [0, 1]")
    require(args.max_tokens > 0 and args.window_size > 0, "invalid request geometry")
    shift_path = args.shift_fixture.resolve()
    static_path = args.static_plan.resolve()
    superset_path = args.superset_plan.resolve()
    targets_path = args.frozen_targets.resolve()
    shift_prompts, shift_metadata = load_fixture(shift_path, "shift")
    require(len(shift_prompts) == 32, "E7 requires 32 shift requests")
    require(all(len(value) == 128 for value in shift_prompts), "prompt length differs")
    require(len(shift_prompts) % args.window_size == 0, "window does not divide trace")
    static_plan, static_metadata = load_plan(static_path)
    superset_plan, superset_metadata = load_plan(superset_path)
    require(set(static_plan) == set(superset_plan) == set(range(48)), "plan layers differ")
    dynamic_plan = {
        layer_id: sorted(superset_plan[layer_id] - static_plan[layer_id])
        for layer_id in static_plan
    }
    require(all(len(value) == 4 for value in dynamic_plan.values()), "four slots required")
    target_payload = load_targets(targets_path)
    selected_target = {
        str(key): [int(value) for value in values]
        for key, values in target_payload["targets"][args.arm].items()
    }
    oracle_target = {
        str(key): [int(value) for value in values]
        for key, values in target_payload["targets"]["oracle"].items()
    }
    require(
        Path(os.environ.get("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE", "")).resolve()
        == static_path,
        "configured static plan differs",
    )
    require(
        Path(os.environ.get("VLLM_UVM_INIT_WEIGHT_PLAN_FILE", "")).resolve()
        == superset_path,
        "configured superset plan differs",
    )
    require(os.environ.get("VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE") == "1", "swap disabled")
    require(
        os.environ.get("VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE") == "1",
        "incremental offload disabled",
    )
    os.environ["VLLM_UVM_DYNAMIC_EXPERT_HEAT_EWMA_ALPHA"] = str(args.ewma_alpha)
    os.environ["E7_MAX_SWAPS_PER_LAYER"] = str(args.max_swaps_per_layer)
    os.environ["E7_MIN_COUNT_DELTA"] = str(args.min_count_delta)

    result: dict[str, Any] = {
        "schema": "eurosys_e7_independent_cold_arm_v1",
        "status": "loading",
        "qualification_only": True,
        "statistical_repetitions": 1,
        "independent_cold_process": True,
        "model_load_count": 1,
        "arm_execution_count": 0,
        "process": {
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "start_time_unix": time.time(),
        },
        "arm_name": args.arm,
        "protocol": {
            "calibration_requests": 0,
            "shift_requests": len(shift_prompts),
            "prompt_tokens": 128,
            "output_tokens": args.max_tokens,
            "window_size": args.window_size,
            "ewma_alpha": args.ewma_alpha,
            "max_swaps_per_layer": args.max_swaps_per_layer,
            "min_count_delta": args.min_count_delta,
            "dynamic_slots_per_layer": 4,
            "measurement_scope": "one frozen-target shift trace only",
            "setup_and_model_load_excluded_from_request_latency": True,
            "maintenance_excluded_from_request_latency": True,
        },
        "inputs": {
            "model": args.model,
            "shift_fixture": str(shift_path),
            "shift_sha256": sha256_file(shift_path),
            "shift_metadata": shift_metadata,
            "static_plan": str(static_path),
            "static_plan_sha256": sha256_file(static_path),
            "static_plan_metadata": static_metadata,
            "superset_plan": str(superset_path),
            "superset_plan_sha256": sha256_file(superset_path),
            "superset_plan_metadata": superset_metadata,
            "frozen_targets": str(targets_path),
            "frozen_targets_sha256": sha256_file(targets_path),
            "target_source_result": target_payload["source_result"],
            "target_source_result_sha256": target_payload["source_result_sha256"],
            "oracle_future_knowledge": True,
        },
    }
    write_result(args.output, result)

    llm: Any = None
    model_runner: Any = None
    engine_client: Any = None
    error: BaseException | None = None
    cleanup_errors: list[str] = []
    try:
        import torch
        import vllm
        from vllm import SamplingParams, TokensPrompt
        from vllm.compilation.counter import compilation_counter
        from vllm.device_allocator.uvm_static_alias import get_static_alias_diagnostics
        from vllm.model_executor.models import utils as offload_utils

        capture_sizes = [1, 128]
        load_started = time.perf_counter_ns()
        llm = vllm.LLM(
            model=args.model,
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=False,
            max_model_len=512,
            max_num_seqs=1,
            max_num_batched_tokens=512,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            cpu_offload_gb=args.cpu_offload_gb,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enable_dbo=False,
            compilation_config={
                "mode": "NONE",
                "cudagraph_mode": "FULL_SEPARATE",
                "cudagraph_capture_sizes": capture_sizes,
                "max_cudagraph_capture_size": max(capture_sizes),
            },
            seed=args.seed,
            disable_log_stats=True,
        )
        torch.cuda.synchronize()
        result["load_ns"] = time.perf_counter_ns() - load_started
        model_runner = model_runner_from_llm(llm)
        engine_client = llm.llm_engine.engine_core
        core = engine_client.engine_core
        require(core is not None, "in-process core absent")
        graph_after_load = int(
            dataclasses.asdict(compilation_counter.clone())["num_cudagraph_captured"]
        )
        before_slots = placement_snapshot(model_runner)
        require(before_slots["layer_count"] == 48, "MoE layer count differs")
        require(before_slots["remote_expert_count"] == 0, "initial dynamic slots present")
        key_by_layer_id = {
            int(layer["layer_id"]): layer_key
            for layer_key, layer in before_slots["layers"].items()
        }
        frozen_static_target = {
            key_by_layer_id[layer_id]: experts
            for layer_id, experts in dynamic_plan.items()
        }
        require(set(selected_target) == set(frozen_static_target), "target layer keys differ")
        initialization = initialize_dynamic_slots(core, model_runner, frozen_static_target)
        runtime_remote_ranges = (
            model_runner._collect_uvm_runtime_remote_expert_ranges()
        )
        require(runtime_remote_ranges, "dynamic runtime-remote ranges absent")
        initialization_hot_reconcile = (
            offload_utils.reconcile_cpu_offload_uvm_original_va_sealed_hot_residency(
                preserve_allocator_cache=True,
                runtime_remote_ranges=runtime_remote_ranges,
            )
        )
        require(
            initialization_hot_reconcile.get("ok") is True,
            "dynamic slot initialization hot-residency reconciliation failed",
        )
        initialization_dual_refresh = None
        if os.environ.get("VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW") == "1":
            initialization_dual_refresh = (
                offload_utils.refresh_cpu_offload_uvm_original_va_full_dual_view(
                    nested_refresh=False
                )
            )
            require(
                initialization_dual_refresh.get("ok") is True,
                "dynamic slot initialization dual-view refresh failed",
            )
        initial_state = placement_snapshot(model_runner)
        require(initial_state["remote_expert_count"] == 192, "dynamic slot total differs")
        swappable = {
            key: set(int(value) for value in layer["swappable_experts"])
            for key, layer in initial_state["layers"].items()
        }
        require(
            all(set(values) <= swappable[key] for key, values in selected_target.items()),
            "selected target contains non-swappable expert",
        )
        require(
            all(set(values) <= swappable[key] for key, values in oracle_target.items()),
            "oracle target contains non-swappable expert",
        )
        result.update(
            {
                "status": "running_arm",
                "versions": {
                    "vllm": getattr(vllm, "__version__", None),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                },
                "graph_count_after_load": graph_after_load,
                "dynamic_slot_initialization": initialization,
                "dynamic_slot_initialization_hot_reconcile": compact(
                    initialization_hot_reconcile
                ),
                "dynamic_slot_initialization_dual_refresh": compact(
                    initialization_dual_refresh
                ),
                "initial_placement": compact(initial_state),
                "initial_dynamic_remote_region_bytes": int(initial_state["remote_region_bytes"]),
                "static_alias_selected_bytes": int(static_metadata.get("selected_main_bytes", 0)),
                "diagnostics_after_load": {
                    "model": compact(offload_utils.get_cpu_offload_graph_static_alias_diagnostics()),
                    "driver": compact(get_static_alias_diagnostics()),
                },
            }
        )
        write_result(args.output, result)

        sampling = SamplingParams(
            max_tokens=args.max_tokens,
            min_tokens=args.max_tokens,
            temperature=0.0,
            seed=args.seed,
            ignore_eos=True,
        )
        arm, _ = run_arm(
            name=args.arm,
            start_target=selected_target,
            adaptive=args.arm == "gpu_heat_cpu_ewma",
            llm=llm,
            torch=torch,
            TokensPrompt=TokensPrompt,
            sampling=sampling,
            shift_prompts=shift_prompts,
            window_size=args.window_size,
            core=core,
            model_runner=model_runner,
            compilation_counter=compilation_counter,
            reference_outputs={},
        )
        finalize_arm_metrics(arm, oracle_target)
        arm["dynamic_remote_plan_bytes"] = selected_plan_bytes(
            initial_state, arm["final_target"]
        )
        swaps, h2d_bytes, d2h_bytes = adaptive_transfer_metrics(arm, initial_state)
        arm["online_swap_count"] = swaps
        arm["online_maintenance_planned_h2d_bytes"] = h2d_bytes
        arm["online_maintenance_planned_d2h_bytes"] = d2h_bytes
        if arm["adaptive"]:
            require(swaps > 0 and h2d_bytes > 0 and d2h_bytes > 0, "adaptive swap absent")
        else:
            require(swaps == h2d_bytes == d2h_bytes == 0, "fixed arm swapped online")
        total_tokens = arm["request_count"] * arm["output_tokens_per_request"]
        arm["end_to_end_output_tokens_per_second"] = total_tokens / (
            (arm["request_wall_ns"] + arm["maintenance_ns"]) / 1e9
        )
        graph_final = int(
            dataclasses.asdict(compilation_counter.clone())["num_cudagraph_captured"]
        )
        require(graph_final == graph_after_load, "arm caused Graph recapture")
        result.update(
            {
                "status": "ok",
                "arm_execution_count": 1,
                "arm": arm,
                "graph_count_final": graph_final,
                "final_placement": compact(placement_snapshot(model_runner)),
                "diagnostics_final": {
                    "model": compact(offload_utils.get_cpu_offload_graph_static_alias_diagnostics()),
                    "driver": compact(get_static_alias_diagnostics()),
                },
            }
        )
    except BaseException as exc:
        error = exc
        result.update(
            {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        engine_shutdown = False
        if engine_client is None and llm is not None:
            engine_client = llm.llm_engine.engine_core
        if engine_client is not None:
            try:
                engine_client.shutdown()
                engine_shutdown = True
            except Exception as exc:
                cleanup_errors.append(repr(exc))
        try:
            from vllm.model_executor.models.utils import (
                release_cpu_offload_uvm_original_va_shutdown_references,
            )

            result["reference_release"] = compact(
                release_cpu_offload_uvm_original_va_shutdown_references()
            )
        except Exception as exc:
            cleanup_errors.append(repr(exc))
        model_runner = None
        if engine_shutdown and engine_client is not None and hasattr(
            engine_client, "engine_core"
        ):
            engine_client.engine_core = None
        llm = None
        engine_client = None
        gc.collect()
        try:
            from vllm.distributed.device_communicators.pynccl_allocator import (
                set_graph_pool_id,
            )
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

            set_graph_pool_id(None)
            cleanup_dist_env_and_memory()
            result["distributed_cleanup"] = True
        except Exception as exc:
            result["distributed_cleanup"] = False
            cleanup_errors.append(repr(exc))
        with suppress(Exception):
            import torch

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        result["cleanup_errors"] = cleanup_errors
        result["controlled_hard_exit_after_cleanup"] = (
            os.environ.get("VLLM_PHASE_SWITCH_CONTROLLED_HARD_EXIT_AFTER_CLEANUP") == "1"
        )
        if cleanup_errors and error is None:
            result.update(
                {
                    "status": "failed",
                    "error": f"cleanup failed: {cleanup_errors!r}",
                }
            )
        write_result(args.output, result)
    exit_code = 0 if result.get("status") == "ok" else 1
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "arm": args.arm,
                "pid": result["process"]["pid"],
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if exit_code == 0 and result.get("controlled_hard_exit_after_cleanup") is True:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
