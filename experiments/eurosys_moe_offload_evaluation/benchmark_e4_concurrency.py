#!/usr/bin/env python3
"""Measure remote-only Qwen decode while scanning request concurrency C1--C32.

The model and frozen expert placement are loaded once per process block.  Each
phase admits a complete frozen request batch with chunked Prefill, waits for a
strict all-request Decode step, excludes that step plus a fixed warmup, and
then measures every remaining one-token Decode step until completion.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PHASE_HELPER = (
    REPO_ROOT
    / "experiments"
    / "vllm_p3_pinned_uva_phase_compare"
    / "benchmark_phase_steps.py"
)
STATIC_HELPER = (
    REPO_ROOT
    / "experiments"
    / "vllm_uvm_prefill_alias_decode_remote_graph"
    / "benchmark_phase_switch_steps.py"
)


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_positive_ints(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    require(bool(values), "concurrency sequence is empty")
    require(all(1 <= value <= 32 for value in values), "concurrency outside C1--C32")
    require(len(values) == len(set(values)), "concurrency sequence contains duplicates")
    return values


def summarize(values: list[float]) -> dict[str, Any]:
    require(bool(values), "cannot summarize empty timing samples")
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(values),
        "samples": values,
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "min": min(values),
        "max": max(values),
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_progress(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        stream.flush()


def compact_parameter_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    vas = [int(value) for value in snapshot["vas"]]
    encoded = json.dumps(vas, separators=(",", ":")).encode()
    return {
        "count": int(snapshot["count"]),
        "bytes": int(snapshot["bytes"]),
        "va_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--exact-names", required=True, type=Path)
    parser.add_argument("--expert-plan", required=True, type=Path)
    parser.add_argument("--offload-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--concurrencies", default=",".join(str(x) for x in range(1, 33)))
    parser.add_argument("--warm-concurrency", type=int, default=32)
    parser.add_argument("--warm-output-tokens", type=int, default=16)
    parser.add_argument("--skip-warmup-phase", action="store_true")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--warmup-decode-steps", type=int, default=5)
    parser.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    parser.add_argument("--cpu-offload-gb", type=float, default=30.0)
    parser.add_argument("--max-model-len", type=int, default=656)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--request-rotation", type=int, default=0)
    parser.add_argument(
        "--cuda-profiler-measured-steps",
        type=int,
        default=0,
        help="diagnostic only: bracket this many measured Decode steps with cudaProfilerStart/Stop",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-parameter-count", type=int, default=96)
    parser.add_argument("--expected-full-parameter-bytes", type=int, required=True)
    parser.add_argument("--expected-selected-bytes", type=int, required=True)
    args = parser.parse_args()

    concurrencies = parse_positive_ints(args.concurrencies)
    require(1 <= args.warm_concurrency <= 32, "invalid warm concurrency")
    require(args.warm_output_tokens >= args.warmup_decode_steps + 3, "warm output too short")
    require(args.output_tokens >= args.warmup_decode_steps + 3, "output too short")
    for path in (args.fixture, args.exact_names, args.expert_plan):
        require(path.is_file(), f"missing input: {path}")
    require(
        os.environ.get("VLLM_UVM_ORIGINAL_VA_PHASE_MODE", "").strip().lower()
        == "remote_only",
        "E4 requires remote_only phase mode",
    )
    configured_plan = os.environ.get(
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE", ""
    ).strip()
    require(
        configured_plan and Path(configured_plan).resolve() == args.expert_plan.resolve(),
        "expert plan argument differs from environment",
    )

    phase = load_module("e4_phase_helper", PHASE_HELPER)
    static = load_module("e4_static_helper", STATIC_HELPER)
    fixture_metadata, requests = phase.load_requests(args.fixture)
    require(len(requests) >= 32, "fixture must contain 32 frozen requests")
    rotation = args.request_rotation % len(requests)
    requests = requests[rotation:] + requests[:rotation]
    require(
        all(item["prompt_tokens"] + args.output_tokens <= args.max_model_len for item in requests[:32]),
        "prompt plus output exceeds max-model-len",
    )
    exact_names = static.load_exact_names(args.exact_names, args.expected_parameter_count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    args.offload_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.progress.write_text("", encoding="utf-8")
    args.offload_manifest.write_text("", encoding="utf-8")
    result: dict[str, Any] = {
        "schema": "eurosys_e4_remote_concurrency_v1",
        "status": "loading",
        "started_epoch_seconds": time.time(),
        "config": {
            **vars(args),
            "fixture": str(args.fixture.resolve()),
            "expert_plan": str(args.expert_plan.resolve()),
            "exact_names": str(args.exact_names.resolve()),
            "offload_manifest": str(args.offload_manifest.resolve()),
            "output": str(args.output.resolve()),
            "progress": str(args.progress.resolve()),
            "concurrencies": concurrencies,
            "fixture_sha256": sha256_file(args.fixture),
            "plan_sha256": sha256_file(args.expert_plan),
            "exact_names_sha256": sha256_file(args.exact_names),
            "fixture_metadata": fixture_metadata,
        },
        "environment": static.selected_environment(),
        "phases": [],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")

    live_llms: list[Any] = []
    shutdown_results: list[dict[str, Any]] = []
    llm: Any = None
    scheduler: Any = None
    original_schedule: Any = None
    original_llm_class: Any = None
    original_runner_shutdown: Any = None
    error: BaseException | None = None
    error_traceback = None

    try:
        import torch
        import vllm
        from vllm import SamplingParams, TokensPrompt
        from vllm.device_allocator.uvm_static_alias import get_static_alias_diagnostics
        from vllm.model_executor.models import utils
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        original_llm_class = vllm.LLM
        original_runner_shutdown = GPUModelRunner.shutdown_cpu_offload_graph_static_alias

        class TrackedLLM(original_llm_class):
            def __init__(self, *inner_args, **inner_kwargs):
                super().__init__(*inner_args, **inner_kwargs)
                live_llms.append(self)

        def tracked_shutdown(self):
            value = original_runner_shutdown(self)
            shutdown_results.append(dict(value))
            return value

        vllm.LLM = TrackedLLM
        GPUModelRunner.shutdown_cpu_offload_graph_static_alias = tracked_shutdown
        load_started = time.perf_counter()
        llm = vllm.LLM(
            model=args.model,
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
            max_model_len=args.max_model_len,
            max_num_seqs=32,
            max_num_batched_tokens=args.max_num_batched_tokens,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            cpu_offload_gb=args.cpu_offload_gb,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            enable_dbo=False,
            seed=args.seed,
            disable_log_stats=True,
        )
        phase.enable_request_metrics_without_default_loggers(llm)
        torch.cuda.synchronize()
        result["load_seconds"] = time.perf_counter() - load_started
        torch.cuda.reset_peak_memory_stats()

        diagnostics_after_load = static.diagnostics_snapshot(utils, get_static_alias_diagnostics)
        model_diag = diagnostics_after_load["model"]
        require(
            int(model_diag.get("selected_parameter_bytes_per_forward", -1))
            == args.expected_selected_bytes,
            "selected remote bytes differ from frozen plan",
        )
        require(
            int(model_diag.get("selected_expert_count", -1)) == 480,
            "selected expert count differs from frozen plan",
        )
        transition_history = model_diag.get("phase_transition_history", [])
        remote_transitions = [
            item for item in transition_history
            if isinstance(item, dict)
            and item.get("event") == "cpu_offload_uvm_original_va_remote_transition"
            and item.get("post_state") == "remote"
        ]
        require(remote_transitions, "remote-only load transition is absent")
        residency = remote_transitions[-1].get("physical_residency", {})
        require(
            int(residency.get("cpu_bytes", -1)) == args.expected_selected_bytes
            and int(residency.get("gpu_bytes", -1)) == 0
            and int(residency.get("unknown_bytes", -1)) == 0,
            "frozen remote ranges are not fully CPU resident",
        )
        parameter_after_load = static.parameter_snapshot(
            llm,
            exact_names,
            expected_count=args.expected_parameter_count,
            expected_full_bytes=args.expected_full_parameter_bytes,
        )
        result["placement_audit"] = {
            "selected_expert_count": int(model_diag["selected_expert_count"]),
            "selected_parameter_count": int(model_diag["selected_parameter_count"]),
            "selected_parameter_bytes_per_forward": int(
                model_diag["selected_parameter_bytes_per_forward"]
            ),
            "remote_physical_residency": residency,
            "parameter_va_after_load": compact_parameter_snapshot(parameter_after_load),
        }
        result["status"] = "running"
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")

        client = llm.llm_engine.engine_core
        core = getattr(client, "engine_core", None)
        require(core is not None, "in-process EngineCore is unavailable")
        scheduler = core.scheduler
        original_schedule = scheduler.schedule
        schedule_records: list[dict[str, Any]] = []

        def traced_schedule():
            output = original_schedule()
            schedule_records.append(phase.schedule_snapshot(output))
            return output

        scheduler.schedule = traced_schedule

        def run_step() -> dict[str, Any]:
            schedule_before = len(schedule_records)
            torch.cuda.synchronize()
            started_epoch = time.time()
            started = time.perf_counter()
            outputs = llm.llm_engine.step()
            torch.cuda.synchronize()
            finished = time.perf_counter()
            require(len(schedule_records) == schedule_before + 1, "scheduler call count differs")
            return {
                "started_epoch_seconds": started_epoch,
                "finished_epoch_seconds": time.time(),
                "elapsed_ms": (finished - started) * 1000.0,
                "outputs": outputs,
                "schedule": schedule_records[-1],
            }

        def strict_decode(schedule: dict[str, Any], concurrency: int) -> bool:
            scheduled = schedule["num_scheduled_tokens"]
            cached = schedule["cached_request_ids"]
            return (
                not schedule["new_request_ids"]
                and len(scheduled) == concurrency
                and len(cached) == concurrency
                and set(scheduled) == set(cached)
                and all(int(value) == 1 for value in scheduled.values())
                and int(schedule["total_num_scheduled_tokens"]) == concurrency
                and not schedule["resumed_request_ids"]
                and not schedule["preempted_request_ids"]
                and not schedule["finished_request_ids"]
            )

        def run_phase(label: str, concurrency: int, output_tokens: int) -> dict[str, Any]:
            require(not llm.llm_engine.has_unfinished_requests(), f"{label}: engine not idle")
            prompts = [
                TokensPrompt(prompt_token_ids=list(item["prompt_token_ids"]))
                for item in requests[:concurrency]
            ]
            sampling = SamplingParams(
                max_tokens=output_tokens,
                min_tokens=output_tokens,
                temperature=0.0,
                seed=args.seed,
                ignore_eos=True,
            )
            phase_started_epoch = time.time()
            append_progress(
                args.progress,
                {
                    "event": "phase_start",
                    "phase": label,
                    "concurrency": concurrency,
                    "output_tokens": output_tokens,
                    "epoch_seconds": phase_started_epoch,
                },
            )
            llm._validate_and_add_requests(
                prompts=prompts,
                params=sampling,
                use_tqdm=False,
                lora_request=None,
                priority=None,
            )

            ramp: list[dict[str, Any]] = []
            first_decode: dict[str, Any] | None = None
            for _ in range(1024):
                step = run_step()
                schedule = step["schedule"]
                require(not schedule["preempted_request_ids"], f"{label}: preemption during ramp")
                require(not schedule["resumed_request_ids"], f"{label}: resume during ramp")
                require(not step["outputs"], f"{label}: request finished before full concurrency")
                if strict_decode(schedule, concurrency):
                    first_decode = step
                    break
                ramp.append({key: value for key, value in step.items() if key != "outputs"})
            require(first_decode is not None, f"{label}: never reached strict C{concurrency} Decode")

            warmup: list[dict[str, Any]] = []
            for _ in range(args.warmup_decode_steps):
                step = run_step()
                phase.validate_decode(step["schedule"], concurrency)
                require(not step["outputs"], f"{label}: warmup finished early")
                warmup.append({key: value for key, value in step.items() if key != "outputs"})

            measured: list[dict[str, Any]] = []
            final_outputs: list[Any] = []
            profiler_active = False
            if args.cuda_profiler_measured_steps > 0:
                require(label.startswith("c"), "CUDA profiler is only valid for a C phase")
                status = int(torch.cuda.cudart().cudaProfilerStart())
                require(status == 0, f"cudaProfilerStart failed: {status}")
                profiler_active = True
            for measured_index in range(output_tokens + 4):
                step = run_step()
                phase.validate_decode(step["schedule"], concurrency)
                measured.append({key: value for key, value in step.items() if key != "outputs"})
                if (
                    profiler_active
                    and measured_index + 1 == args.cuda_profiler_measured_steps
                ):
                    status = int(torch.cuda.cudart().cudaProfilerStop())
                    require(status == 0, f"cudaProfilerStop failed: {status}")
                    profiler_active = False
                if step["outputs"]:
                    final_outputs = step["outputs"]
                    break
            if profiler_active:
                status = int(torch.cuda.cudart().cudaProfilerStop())
                require(status == 0, f"cudaProfilerStop failed: {status}")
            require(final_outputs, f"{label}: requests did not finish")
            require(not llm.llm_engine.has_unfinished_requests(), f"{label}: unfinished requests remain")
            records = phase.output_records(final_outputs)
            require(len(records) == concurrency, f"{label}: output batch differs")
            require(all(item["finished"] for item in records), f"{label}: unfinished output")
            require(
                all(len(item["output_token_ids"]) == output_tokens for item in records),
                f"{label}: output token count differs",
            )
            measured_ms = [float(item["elapsed_ms"]) for item in measured]
            warmup_ms = [float(item["elapsed_ms"]) for item in warmup]
            phase_finished_epoch = time.time()
            record = {
                "phase": label,
                "concurrency": concurrency,
                "prompt_tokens_per_request": int(requests[0]["prompt_tokens"]),
                "output_tokens_per_request": output_tokens,
                "phase_started_epoch_seconds": phase_started_epoch,
                "decode_measured_started_epoch_seconds": measured[0]["started_epoch_seconds"],
                "decode_measured_finished_epoch_seconds": measured[-1]["finished_epoch_seconds"],
                "phase_finished_epoch_seconds": phase_finished_epoch,
                "prefill_ramp_steps": len(ramp),
                "prefill_ramp_ms": sum(float(item["elapsed_ms"]) for item in ramp),
                "first_full_decode_ms": float(first_decode["elapsed_ms"]),
                "warmup_decode_step_ms": summarize(warmup_ms),
                "steady_decode_step_ms": summarize(measured_ms),
                "steady_decode_tokens_per_second": concurrency * 1000.0 / statistics.median(measured_ms),
                "request_timing": phase.request_timing_summaries(records),
                "output_token_ids_sha256": phase.output_hash(records),
                "request_ids": [item["request_id"] for item in records],
                "preemptions": 0,
                "resumes": 0,
            }
            append_progress(
                args.progress,
                {
                    "event": "phase_complete",
                    "phase": label,
                    "concurrency": concurrency,
                    "epoch_seconds": phase_finished_epoch,
                    "steady_decode_step_median_ms": record["steady_decode_step_ms"]["median"],
                    "request_tpot_median_ms": record["request_timing"]["request_tpot_ms"]["median"],
                },
            )
            return record

        if not args.skip_warmup_phase:
            warm = run_phase("warm_c32", args.warm_concurrency, args.warm_output_tokens)
            result["warmup_phase"] = warm
        for concurrency in concurrencies:
            record = run_phase(f"c{concurrency}", concurrency, args.output_tokens)
            result["phases"].append(record)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")

        diagnostics_final = static.diagnostics_snapshot(utils, get_static_alias_diagnostics)
        final_model_diag = diagnostics_final["model"]
        require(
            int(final_model_diag.get("selected_parameter_bytes_per_forward", -1))
            == args.expected_selected_bytes,
            "selected bytes changed during scan",
        )
        parameter_final = static.parameter_snapshot(
            llm,
            exact_names,
            expected_count=args.expected_parameter_count,
            expected_full_bytes=args.expected_full_parameter_bytes,
        )
        require(parameter_after_load["vas"] == parameter_final["vas"], "parameter VAs changed")
        manifest = static.summarize_manifest(static.read_manifest(args.offload_manifest))
        result["final_audit"] = {
            "parameter_va": compact_parameter_snapshot(parameter_final),
            "manifest": {key: value for key, value in manifest.items() if key != "records"},
            "diagnostics": {
                "selected_expert_count": int(final_model_diag.get("selected_expert_count", -1)),
                "selected_parameter_bytes_per_forward": int(
                    final_model_diag.get("selected_parameter_bytes_per_forward", -1)
                ),
                "phase_mode": final_model_diag.get("phase_mode"),
                "lifecycle_state": final_model_diag.get("lifecycle_state"),
            },
        }
        result["status"] = "ok"
        result["finished_epoch_seconds"] = time.time()
    except BaseException as exc:
        error = exc
        error_traceback = exc.__traceback__
        result["status"] = "exception"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        shutdown_contract: dict[str, Any] = {
            "model_runner_shutdown_results": [],
            "cleanup_errors": [],
        }
        try:
            if live_llms:
                live_llms[-1].llm_engine.engine_core.shutdown()
            gc.collect()
        except Exception as exc:
            shutdown_contract["cleanup_errors"].append(repr(exc))
        try:
            from vllm.distributed.device_communicators.pynccl_allocator import set_graph_pool_id
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

            set_graph_pool_id(None)
            cleanup_dist_env_and_memory()
            shutdown_contract["distributed_cleanup"] = True
        except Exception as exc:
            shutdown_contract["distributed_cleanup"] = False
            shutdown_contract["cleanup_errors"].append(repr(exc))
        try:
            import torch

            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
        except Exception as exc:
            shutdown_contract["cleanup_errors"].append(repr(exc))
        shutdown_contract["model_runner_shutdown_results"] = static._compact(shutdown_results)
        try:
            from vllm.device_allocator.uvm_static_alias import get_static_alias_diagnostics
            from vllm.model_executor.models import utils

            shutdown_contract["diagnostics_after_shutdown"] = static.diagnostics_snapshot(
                utils, get_static_alias_diagnostics
            )
        except Exception as exc:
            shutdown_contract["diagnostics_error"] = repr(exc)
        try:
            from vllm.model_executor.models.utils import (
                release_cpu_offload_uvm_original_va_shutdown_references,
            )

            shutdown_contract["reference_release"] = (
                release_cpu_offload_uvm_original_va_shutdown_references()
            )
        except Exception as exc:
            shutdown_contract["reference_release"] = {"ok": False, "error": repr(exc)}
            shutdown_contract["cleanup_errors"].append(repr(exc))
        try:
            if scheduler is not None and original_schedule is not None:
                scheduler.schedule = original_schedule
            live_llms.clear()
            scheduler = None
            llm = None
            gc.collect()
            import torch

            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            shutdown_contract["python_references_released"] = True
        except Exception as exc:
            shutdown_contract["python_references_released"] = False
            shutdown_contract["cleanup_errors"].append(repr(exc))
        if original_llm_class is not None:
            try:
                import vllm
                from vllm.v1.worker.gpu_model_runner import GPUModelRunner

                vllm.LLM = original_llm_class
                GPUModelRunner.shutdown_cpu_offload_graph_static_alias = original_runner_shutdown
            except Exception:
                pass
        result["shutdown_contract"] = shutdown_contract
        if shutdown_contract["cleanup_errors"] and error is None:
            error = ContractError(f"shutdown failed: {shutdown_contract['cleanup_errors']!r}")
            error_traceback = error.__traceback__
            result["status"] = "contract_failed"
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")

    if error is not None:
        raise error.with_traceback(error_traceback)
    print(
        json.dumps(
            {
                "status": result["status"],
                "phase_count": len(result["phases"]),
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
