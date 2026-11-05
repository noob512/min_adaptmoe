#!/usr/bin/env python3
"""Run Qwen E3 Prefill-only batches directly through the in-process engine.

Each whole batch is admitted before one timed EngineCore step.  ``max_tokens=1``
means that step performs Prompt Prefill plus first-token sampling and finishes
the requests; no Decode step, HTTP server, or client is involved.  Optional
conditioning and measurement repeats use complete real requests and an
untimed empty cleanup step between batches.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE_ROOT = REPO_ROOT / "experiments" / "vllm_p3_pinned_uva_phase_compare"
GRAPH_ROOT = REPO_ROOT / "experiments" / "vllm_p3_static_alias_graph_integration"
SWITCH_ROOT = REPO_ROOT / "experiments" / "vllm_uvm_prefill_alias_decode_remote_graph"
sys.path[:0] = [str(PHASE_ROOT), str(GRAPH_ROOT), str(SWITCH_ROOT)]

import benchmark_phase_steps as phase  # noqa: E402
from benchmark_full_graph_steps import GraphCallTracker  # noqa: E402
from benchmark_phase_switch_steps import (  # noqa: E402
    cuda_memory_snapshot,
    diagnostics_snapshot,
    parameter_snapshot,
    read_manifest,
    sha256_file,
    summarize_manifest,
)


ARMS = {
    "graph_copy": {
        "custom": True,
        "graph_prefill": True,
        "phase_mode": "prefill_graph_remote_decode",
        "prefill_topology": "sealed",
    },
    "remote_eager": {
        "custom": True,
        "graph_prefill": False,
        "phase_mode": "remote_only",
        "prefill_topology": "remote",
    },
    "alias_eager": {
        "custom": True,
        "graph_prefill": False,
        "phase_mode": "off",
        "prefill_topology": "sealed",
    },
    "native_copy_on_forward": {
        "custom": False,
        "graph_prefill": False,
        "phase_mode": "off",
        "prefill_topology": None,
    },
    "native_graph_copy_on_forward": {
        "custom": False,
        "graph_prefill": True,
        "phase_mode": "off",
        "prefill_topology": None,
    },
}


class ContractError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ContractError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--exact-names", required=True, type=Path)
    parser.add_argument("--expert-plan", type=Path)
    parser.add_argument("--offload-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--prompt-tokens", required=True, type=int)
    parser.add_argument("--batch-size", choices=(1, 8), required=True, type=int)
    parser.add_argument("--kv-cache-memory-bytes", required=True, type=int)
    parser.add_argument("--max-model-len", required=True, type=int)
    parser.add_argument("--max-num-batched-tokens", required=True, type=int)
    parser.add_argument("--cpu-offload-gb", required=True, type=float)
    parser.add_argument("--expected-parameter-count", required=True, type=int)
    parser.add_argument("--expected-full-parameter-bytes", required=True, type=int)
    parser.add_argument("--expected-selected-bytes", required=True, type=int)
    parser.add_argument("--expected-native-parameter-count", default=0, type=int)
    parser.add_argument("--expected-native-parameter-bytes", default=0, type=int)
    parser.add_argument("--conditioning-repeats", default=0, type=int)
    parser.add_argument("--measurement-repeats", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)
    return parser.parse_args()


def output_hash(records: list[dict[str, Any]]) -> str:
    payload = [record["output_token_ids"] for record in records]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main() -> int:
    args = parse_args()
    contract = ARMS[args.arm]
    require(not args.output.exists(), f"output already exists: {args.output}")
    require((args.model / "config.json").is_file(), "model snapshot is absent")
    require(args.fixture.is_file(), "fixture is absent")
    require(args.exact_names.is_file(), "exact-name file is absent")
    require(args.kv_cache_memory_bytes > 0, "KV size must be positive")
    require(args.max_num_batched_tokens >= args.prompt_tokens * args.batch_size,
            "Prefill exceeds max-num-batched-tokens")
    require(args.max_model_len >= args.prompt_tokens + 1,
            "max-model-len cannot hold Prefill plus sampled token")
    require(args.conditioning_repeats >= 0, "conditioning repeats must be nonnegative")
    require(args.measurement_repeats > 0, "measurement repeats must be positive")

    exact_names = [
        line.strip()
        for line in args.exact_names.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if contract["custom"]:
        require(len(exact_names) == args.expected_parameter_count,
                "exact-name count differs")
        require(args.expert_plan is not None and args.expert_plan.is_file(),
                "custom arm requires an expert plan")
        require(os.environ.get("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS") == "1",
                "custom arm requires static alias")
        require(
            os.environ.get("VLLM_UVM_ORIGINAL_VA_PHASE_MODE")
            == contract["phase_mode"],
            "phase-mode environment differs from arm",
        )
    else:
        require(args.expert_plan is None, "native arm must not receive an expert plan")
        require(os.environ.get("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", "0") == "0",
                "native arm unexpectedly enables static alias")
        require(os.environ.get("VLLM_CPU_OFFLOAD_BACKEND") == "copy",
                "native arm is not copy-on-forward")

    fixture_metadata, requests = phase.load_requests(args.fixture)
    require(len(requests) >= args.batch_size, "fixture has too few requests")
    selected = requests[: args.batch_size]
    require(all(item["prompt_tokens"] == args.prompt_tokens for item in selected),
            "fixture prompt length differs")
    expected_prefill_tokens = args.prompt_tokens * args.batch_size
    capture_sizes = sorted({args.batch_size, expected_prefill_tokens})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": "eurosys_e3_prefill_only_direct_batch_v1",
        "status": "loading",
        "arm": args.arm,
        "configuration": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "transport": "none",
            "http_server_started": False,
            "client_used": False,
            "output_tokens": 1,
            "decode_steps": 0,
            "all_requests_admitted_before_first_step": True,
            "graph_prefill": bool(contract["graph_prefill"]),
            "cudagraph_capture_sizes": capture_sizes if contract["graph_prefill"] else [],
            "coordinate_contract": (
                "exact expert-slice coordinates"
                if contract["custom"]
                else "nearest-byte full-Parameter auxiliary baseline"
            ),
        },
        "fixture": {
            "sha256": sha256_file(args.fixture),
            "metadata": fixture_metadata,
        },
        "exact_names_sha256": sha256_file(args.exact_names),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")

    llm: Any = None
    error: BaseException | None = None
    error_traceback = None
    tracker: GraphCallTracker | None = None
    shutdown: dict[str, Any] = {"cleanup_errors": []}
    try:
        import torch
        import vllm
        from vllm import SamplingParams, TokensPrompt
        from vllm.compilation.counter import compilation_counter
        from vllm.config import CompilationMode, CUDAGraphMode
        from vllm.device_allocator.uvm_static_alias import get_static_alias_diagnostics
        from vllm.model_executor.models import utils

        tracker = GraphCallTracker()
        tracker.install()
        counter_before = dataclasses.asdict(compilation_counter.clone())
        graph_prefill = bool(contract["graph_prefill"])
        compilation_config = (
            {
                "mode": "NONE",
                "cudagraph_mode": "FULL_SEPARATE",
                "cudagraph_capture_sizes": capture_sizes,
                "max_cudagraph_capture_size": max(capture_sizes),
            }
            if graph_prefill
            else {"mode": "NONE", "cudagraph_mode": "NONE"}
        )
        load_started = time.perf_counter()
        llm = vllm.LLM(
            model=str(args.model),
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=0.97,
            enforce_eager=not graph_prefill,
            max_model_len=args.max_model_len,
            max_num_seqs=args.batch_size,
            max_num_batched_tokens=args.max_num_batched_tokens,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            cpu_offload_gb=args.cpu_offload_gb,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enable_dbo=False,
            compilation_config=compilation_config,
            seed=args.seed,
            disable_log_stats=True,
        )
        phase.enable_request_metrics_without_default_loggers(llm)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        torch.cuda.reset_peak_memory_stats()

        realized = llm.llm_engine.vllm_config.compilation_config
        require(realized.mode == CompilationMode.NONE, "CompilationMode is not NONE")
        expected_mode = (
            CUDAGraphMode.FULL_SEPARATE if graph_prefill else CUDAGraphMode.NONE
        )
        require(realized.cudagraph_mode == expected_mode,
                "realized CUDA Graph mode differs")
        expected_sizes = capture_sizes if graph_prefill else []
        require(list(realized.cudagraph_capture_sizes) == expected_sizes,
                "realized capture sizes differ")

        counter_after_load = dataclasses.asdict(compilation_counter.clone())
        graph_after_load = tracker.snapshot()
        if graph_prefill:
            require(
                counter_after_load["num_cudagraph_captured"]
                - counter_before["num_cudagraph_captured"] >= 2,
                "Prefill/decode Graph partitions were not captured",
            )
            require(graph_after_load.get("full_capture", 0) >= 2,
                    "Graph tracker saw too few captures")

        diagnostics_after_load = diagnostics_snapshot(
            utils, get_static_alias_diagnostics
        )
        parameter_after_load = None
        if contract["custom"]:
            parameter_after_load = parameter_snapshot(
                llm,
                exact_names,
                expected_count=args.expected_parameter_count,
                expected_full_bytes=args.expected_full_parameter_bytes,
            )
            model_diag = diagnostics_after_load["model"]
            require(model_diag.get("lifecycle_state") == contract["prefill_topology"],
                    "custom load topology differs")
            require(int(model_diag.get("selected_parameter_bytes_per_forward", -1))
                    == args.expected_selected_bytes,
                    "selected expert bytes differ")

        manifest_before = read_manifest(args.offload_manifest)
        sampling = SamplingParams(
            max_tokens=1,
            min_tokens=1,
            temperature=0.0,
            seed=args.seed,
            ignore_eos=True,
        )
        core_client = llm.llm_engine.engine_core
        core = getattr(core_client, "engine_core", None)
        require(core is not None, "in-process EngineCore is unavailable")
        scheduler = core.scheduler
        original_schedule = scheduler.schedule
        schedules: list[dict[str, Any]] = []

        def traced_schedule():
            value = original_schedule()
            schedules.append(phase.schedule_snapshot(value))
            return value

        scheduler.schedule = traced_schedule
        graph_before = tracker.snapshot()
        counter_before_request = dataclasses.asdict(compilation_counter.clone())
        roles = ["conditioning"] * args.conditioning_repeats + [
            "measurement"
        ] * args.measurement_repeats
        role_indices = {"conditioning": 0, "measurement": 0}
        cycles: list[dict[str, Any]] = []
        for cycle_index, role in enumerate(roles):
            prompts = [
                TokensPrompt(prompt_token_ids=list(item["prompt_token_ids"]))
                for item in selected
            ]
            llm._validate_and_add_requests(
                prompts=prompts,
                params=sampling,
                use_tqdm=False,
                lora_request=None,
                priority=None,
            )

            schedule_count_before = len(schedules)
            cycle_graph_before = tracker.snapshot()
            torch.cuda.synchronize()
            profile_role = os.environ.get(
                "E3_NSYS_PREFILL_CAPTURE_ROLE", "measurement"
            )
            profile_role_index = int(os.environ.get(
                "E3_NSYS_PREFILL_CAPTURE_ROLE_INDEX", "0"
            ))
            profile_cycle = (
                os.environ.get("E3_NSYS_PREFILL_CAPTURE", "0") == "1"
                and role == profile_role
                and role_indices[role] == profile_role_index
            )
            if profile_cycle:
                status = int(torch.cuda.cudart().cudaProfilerStart())
                require(status == 0, f"cudaProfilerStart failed: {status}")
                torch.cuda.nvtx.range_push(
                    f"e3_prefill_step:{args.arm}:{role}:{role_indices[role]}"
                )
            try:
                started = time.perf_counter()
                outputs = llm.llm_engine.step()
                torch.cuda.synchronize()
            finally:
                if profile_cycle:
                    torch.cuda.nvtx.range_pop()
                    status = int(torch.cuda.cudart().cudaProfilerStop())
                    require(status == 0, f"cudaProfilerStop failed: {status}")
            prefill_ms = (time.perf_counter() - started) * 1000.0
            cycle_graph_delta = tracker.delta(
                cycle_graph_before, tracker.snapshot()
            )

            require(len(schedules) == schedule_count_before + 1,
                    "Prefill invoked scheduler more than once")
            schedule = schedules[-1]
            phase.validate_prefill(schedule, args.batch_size)
            require(schedule["total_num_scheduled_tokens"] == expected_prefill_tokens,
                    "scheduled Prefill token count differs")
            require(len(outputs) == args.batch_size, "Prefill output batch differs")
            records = phase.output_records(outputs)
            require(len(records) == args.batch_size, "output record count differs")
            require(all(record["finished"] for record in records),
                    "max_tokens=1 did not finish every request")
            require(all(len(record["output_token_ids"]) == 1 for record in records),
                    "a request did not produce exactly one sampled token")
            require(not llm.llm_engine.has_unfinished_requests(),
                    "requests remain after the Prefill step")
            expected_replays = 1 if graph_prefill else 0
            require(cycle_graph_delta.get("full_replay", 0) == expected_replays,
                    "Prefill Graph replay count differs")
            ttft_values = [
                float(record["ttft_ms"])
                for record in records
                if record.get("ttft_ms") is not None
            ]
            cycles.append(
                {
                    "cycle_index": cycle_index,
                    "role": role,
                    "role_index": role_indices[role],
                    "prefill_step_ms": prefill_ms,
                    "prompt_tokens_per_second": (
                        expected_prefill_tokens * 1000.0 / prefill_ms
                    ),
                    "request_ttft_ms_median": (
                        statistics.median(ttft_values) if ttft_values else None
                    ),
                    "graph_wrapper_delta_prefill": cycle_graph_delta,
                    "schedule": schedule,
                    "output_token_ids_sha256": output_hash(records),
                    "outputs": records,
                }
            )
            role_indices[role] += 1

            # V1 releases per-request scheduler state on the next step.  Keep
            # this empty cleanup outside the Prefill timing boundary so every
            # repeated batch starts from the same scheduler state.
            cleanup_outputs = llm.llm_engine.step()
            require(not cleanup_outputs,
                    "finished-request cleanup unexpectedly returned outputs")

        graph_delta = tracker.delta(graph_before, tracker.snapshot())
        counter_after_request = dataclasses.asdict(compilation_counter.clone())
        require(counter_after_request["num_cudagraph_captured"]
                == counter_before_request["num_cudagraph_captured"],
                "measurement caused an unexpected Graph recapture")

        measurement_cycles = [
            item for item in cycles if item["role"] == "measurement"
        ]
        conditioning_cycles = [
            item for item in cycles if item["role"] == "conditioning"
        ]
        measurement_samples = [
            float(item["prefill_step_ms"]) for item in measurement_cycles
        ]
        conditioning_samples = [
            float(item["prefill_step_ms"]) for item in conditioning_cycles
        ]
        require(len({item["output_token_ids_sha256"] for item in cycles}) == 1,
                "repeated batches produced different token hashes")
        final_cycle = cycles[-1]
        records = final_cycle["outputs"]
        schedule = final_cycle["schedule"]

        manifest_after = read_manifest(args.offload_manifest)
        serving_records = manifest_after[len(manifest_before):]
        copy_records = [
            item for item in serving_records
            if item.get("event") == "cpu_offload_copy_forward"
        ]
        parameter_records = [
            item for item in manifest_after
            if item.get("event") == "cpu_offload_parameter"
        ]
        if not contract["custom"]:
            if args.expected_native_parameter_count > 0:
                require(
                    len(parameter_records) == args.expected_native_parameter_count,
                    "native copied Parameter count differs",
                )
            if args.expected_native_parameter_bytes > 0:
                require(
                    sum(int(item.get("bytes", 0)) for item in parameter_records)
                    == args.expected_native_parameter_bytes,
                    "native copied Parameter bytes differ",
                )
            require(all(item.get("copy_on_forward") is True
                        for item in parameter_records),
                    "native manifest does not prove copy-on-forward")

        diagnostics_after_prefill = diagnostics_snapshot(
            utils, get_static_alias_diagnostics
        )
        result.update(
            {
                "status": "ok",
                "versions": {
                    "vllm": getattr(vllm, "__version__", None),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                },
                "load_seconds": load_seconds,
                "cuda_memory_after_prefill": cuda_memory_snapshot(torch),
                "compilation_counter_before_load": counter_before,
                "compilation_counter_after_load": counter_after_load,
                "compilation_counter_before_request": counter_before_request,
                "compilation_counter_after_request": counter_after_request,
                "graph_wrapper_calls_after_load": graph_after_load,
                "graph_wrapper_delta_prefill": graph_delta,
                "schedule": schedule,
                "timing": {
                    "prefill_step_ms": statistics.median(measurement_samples),
                    "first_request_prefill_step_ms": cycles[0]["prefill_step_ms"],
                    "conditioning_prefill_step_ms": (
                        phase.summarize(conditioning_samples)
                        if conditioning_samples else None
                    ),
                    "measurement_prefill_step_ms": phase.summarize(
                        measurement_samples
                    ),
                    "prompt_tokens_per_second": (
                        expected_prefill_tokens * 1000.0
                        / statistics.median(measurement_samples)
                    ),
                    "request_ttft_ms_median": final_cycle[
                        "request_ttft_ms_median"
                    ],
                },
                "cycles": cycles,
                "diagnostics": {
                    "after_load": diagnostics_after_load,
                    "after_prefill": diagnostics_after_prefill,
                },
                "parameter_va_after_load": parameter_after_load,
                "manifest": {
                    "all": summarize_manifest(manifest_after),
                    "serving_prefill": summarize_manifest(serving_records),
                    "serving_copy_forward_count": len(copy_records),
                    "serving_copy_forward_bytes": sum(
                        int(item.get("bytes", 0)) for item in copy_records
                    ),
                    "offloaded_parameter_count": len(parameter_records),
                    "offloaded_parameter_bytes": sum(
                        int(item.get("bytes", 0)) for item in parameter_records
                    ),
                },
                "output_token_ids_sha256": final_cycle[
                    "output_token_ids_sha256"
                ],
                "outputs": records,
            }
        )
    except BaseException as exc:
        error = exc
        error_traceback = exc.__traceback__
        result["status"] = "exception"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        try:
            if llm is not None:
                llm.llm_engine.engine_core.shutdown()
        except Exception as exc:
            shutdown["cleanup_errors"].append(repr(exc))
        try:
            from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

            set_graph_pool_id(None)
            cleanup_dist_env_and_memory()
            shutdown["distributed_cleanup"] = True
        except Exception as exc:
            shutdown["distributed_cleanup"] = False
            shutdown["cleanup_errors"].append(repr(exc))
        try:
            llm = None
            gc.collect()
            import torch

            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as exc:
            shutdown["cleanup_errors"].append(repr(exc))
        try:
            from vllm.device_allocator.uvm_static_alias import get_static_alias_diagnostics
            from vllm.model_executor.models import utils

            shutdown["diagnostics_after_shutdown"] = diagnostics_snapshot(
                utils, get_static_alias_diagnostics
            )
        except Exception as exc:
            shutdown["diagnostics_error"] = repr(exc)
        try:
            shutdown["manifest_final"] = summarize_manifest(
                read_manifest(args.offload_manifest)
            )
        except Exception as exc:
            shutdown["manifest_error"] = repr(exc)
        result["shutdown_contract"] = shutdown
        if shutdown["cleanup_errors"] and error is None:
            error = ContractError(f"shutdown failed: {shutdown['cleanup_errors']!r}")
            error_traceback = error.__traceback__
            result["status"] = "contract_failed"
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    if error is not None:
        raise error.with_traceback(error_traceback)
    print(json.dumps({
        "status": result["status"],
        "arm": args.arm,
        "batch_size": args.batch_size,
        "prefill_ms": result["timing"]["prefill_step_ms"],
        "output": str(args.output),
    }, sort_keys=True), flush=True)
    if os.environ.get("VLLM_PHASE_SWITCH_CONTROLLED_HARD_EXIT_AFTER_CLEANUP") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
