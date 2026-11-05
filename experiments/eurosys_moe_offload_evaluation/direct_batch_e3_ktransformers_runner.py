#!/usr/bin/env python3
"""Run one E3 workload through KTransformers without HTTP.

Each measured B1/B8/B16 cell is submitted with one ``Engine.generate`` call whose
``input_ids`` value is a list of all prompts.  Mixed chunk scheduling is
disabled, and the prefill budget covers the complete batch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from direct_batch_phase_runner import atomic_json, distribution, load_fixture, utc_now


def parse_batches(raw: str) -> list[int]:
    values = [int(value) for value in raw.split(",") if value]
    if not values or any(value not in (1, 8, 16) for value in values):
        raise argparse.ArgumentTypeError(
            "batch sizes must be a non-empty subset of 1,8,16"
        )
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("batch sizes must be unique")
    return values


def token_hash(token_ids: list[list[int]]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sampling_params(output_tokens: int) -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": output_tokens,
        "min_new_tokens": output_tokens,
        "ignore_eos": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--frequency-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--prompt-tokens", required=True, type=int)
    parser.add_argument("--output-tokens", required=True, type=int)
    parser.add_argument("--batch-sizes", type=parse_batches, default=[1, 8])
    parser.add_argument("--gpu-experts", type=int, default=118)
    parser.add_argument("--cpu-threads", type=int, default=14)
    parser.add_argument("--gpu-prefill-token-threshold", type=int, default=0)
    parser.add_argument(
        "--attention-backend",
        choices=("flashinfer", "triton"),
        default="flashinfer",
    )
    parser.add_argument(
        "--decode-attention-backend",
        choices=("flashinfer", "triton"),
        default=None,
        help="optional decode-only backend override",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--warmup-decode-steps", type=int, default=5)
    parser.add_argument("--watchdog-timeout-seconds", type=int, default=7200)
    parser.add_argument(
        "--enable-cuda-graph",
        action="store_true",
        help="enable SGLang/KTransformers decode CUDA Graph replay",
    )
    parser.add_argument(
        "--disable-overlap-schedule",
        action="store_true",
        help="disable KTransformers CPU/GPU overlap scheduling",
    )
    return parser.parse_args()


def run_cell(
    engine: Any,
    prompts: list[list[int]],
    *,
    output_tokens: int,
    request_prefix: str,
    warmup_decode_steps: int,
) -> dict[str, Any]:
    batch = len(prompts)
    request_ids = [f"{request_prefix}-{index:02d}" for index in range(batch)]
    print(
        "E3_KT_DIRECT_DISPATCH "
        f"batch={batch} single_generate_call=true transport=none",
        flush=True,
    )
    started = time.perf_counter()
    stream = engine.generate(
        input_ids=prompts,
        sampling_params=[sampling_params(output_tokens) for _ in prompts],
        stream=True,
        rid=request_ids,
    )
    token_times: list[list[float]] = [[] for _ in prompts]
    output_ids: list[list[int]] = [[] for _ in prompts]
    completion_counts = [0] * batch
    final_meta: list[dict[str, Any]] = [{} for _ in prompts]
    for chunk in stream:
        observed = time.perf_counter()
        index = int(chunk.get("index", 0))
        if not 0 <= index < batch:
            raise RuntimeError(f"stream index outside batch: {index}")
        metadata = chunk.get("meta_info") or {}
        # SGLang emits populated prefill timing on the first streamed chunk,
        # then may emit the same keys with ``None`` on later decode chunks.
        # A placeholder must not erase the server-side measurement.
        final_meta[index].update(
            {key: value for key, value in metadata.items() if value is not None}
        )
        completion = int(metadata.get("completion_tokens", completion_counts[index]))
        new_tokens = max(0, completion - completion_counts[index])
        ids = [int(value) for value in chunk.get("output_ids", [])]
        if len(ids) == completion:
            output_ids[index] = ids
        elif new_tokens and len(ids) >= new_tokens:
            output_ids[index].extend(ids[-new_tokens:])
        token_times[index].extend([observed] * new_tokens)
        completion_counts[index] = completion
    finished = time.perf_counter()

    requests: list[dict[str, Any]] = []
    for index, (ids, times) in enumerate(zip(output_ids, token_times)):
        itl = [(right - left) * 1000.0 for left, right in zip(times, times[1:])]
        requests.append(
            {
                "index": index,
                "request_id": request_ids[index],
                "success": len(ids) == output_tokens and len(times) == output_tokens,
                "output_token_ids": ids,
                "completion_tokens": len(ids),
                "ttft_ms": (times[0] - started) * 1000.0 if times else None,
                "tpot_ms": statistics.fmean(itl) if itl else None,
                "e2e_ms": (times[-1] - started) * 1000.0 if times else None,
            }
        )
    successful = [record for record in requests if record["success"]]
    batch_token_times = [
        max(times[token_index] for times in token_times)
        for token_index in range(output_tokens)
    ] if successful and len(successful) == batch else []
    whole_batch_decode_steps_ms = [
        (right - left) * 1000.0
        for left, right in zip(batch_token_times, batch_token_times[1:])
    ]
    first_decode_step_ms = (
        whole_batch_decode_steps_ms[0] if whole_batch_decode_steps_ms else None
    )
    decode_warmup_ms = whole_batch_decode_steps_ms[
        1 : 1 + warmup_decode_steps
    ]
    measured_decode_ms = whole_batch_decode_steps_ms[
        1 + warmup_decode_steps :
    ]
    prefill_launch_ms = [
        float(metadata["prefill_launch_latency"]) * 1000.0
        for metadata in final_meta
        if isinstance(metadata.get("prefill_launch_latency"), (int, float))
        and float(metadata["prefill_launch_latency"]) > 0.0
    ]
    return {
        "batch_size": batch,
        "request_count": batch,
        "successful_requests": len(successful),
        "single_generate_call": True,
        "all_requests_submitted_together": True,
        "transport": "none",
        "mixed_chunk_enabled": False,
        "output_token_ids": output_ids,
        "output_token_ids_sha256": token_hash(output_ids),
        "client_ttft_batch_ms": max(
            (record["ttft_ms"] for record in successful), default=None
        ),
        "server_prefill_forward_ms": distribution(prefill_launch_ms),
        "prefill_step_ms": (
            statistics.median(prefill_launch_ms) if prefill_launch_ms else None
        ),
        "p_to_d_switch_ms": None,
        "p_to_d_switch_semantics": (
            "N/A: KTransformers keeps one eager address/placement view and "
            "does not perform the custom dual-page-table P-to-D switch"
        ),
        "first_decode_step_ms": first_decode_step_ms,
        "warmup_decode_step_ms": distribution(decode_warmup_ms),
        "measured_decode_step_ms": distribution(measured_decode_ms),
        "whole_batch_decode_steps_ms": whole_batch_decode_steps_ms,
        "steady_decode_median_ms": (
            statistics.median(measured_decode_ms) if measured_decode_ms else None
        ),
        "steady_decode_mean_ms": (
            statistics.fmean(measured_decode_ms) if measured_decode_ms else None
        ),
        "decode_tokens_per_second": (
            batch * 1000.0 / statistics.fmean(measured_decode_ms)
            if measured_decode_ms
            else None
        ),
        "cycle_e2e_ms": (finished - started) * 1000.0,
        "server_time_stats": final_meta,
        "requests": requests,
    }


def main() -> int:
    args = parse_args()
    # This opt-in affects only serialization of timestamps that the scheduler
    # already records; it does not add synchronization or alter execution.
    os.environ["SGLANG_E3_SERIALIZE_SCHEDULER_TIME_STATS"] = "1"
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    for path in (args.model / "config.json", args.fixture, args.frequency_plan):
        if not path.is_file():
            raise SystemExit(f"required file is missing: {path}")
    metadata, requests = load_fixture(args.fixture)
    if metadata.get("workload") != args.workload:
        raise SystemExit("fixture workload differs from --workload")
    max_batch = max(args.batch_sizes)
    if len(requests) < max_batch:
        raise SystemExit("fixture is smaller than maximum batch")
    if any(len(request.prompt_token_ids) != args.prompt_tokens for request in requests[:max_batch]):
        raise SystemExit("fixture prompt length differs")
    if args.output_tokens < args.warmup_decode_steps + 2:
        raise SystemExit(
            "output tokens must leave a first decode step, warmup steps, and "
            "at least one measured decode step"
        )
    if not 0 <= args.gpu_experts <= 128:
        raise SystemExit("--gpu-experts must be in [0, 128]")
    if args.gpu_prefill_token_threshold < 0:
        raise SystemExit("--gpu-prefill-token-threshold must be non-negative")

    context_length = args.prompt_tokens + args.output_tokens + 16
    max_prefill_tokens = max(args.prompt_tokens * max_batch, context_length)
    max_total_tokens = context_length * max_batch + 128
    cuda_graph_batch_sizes = sorted({1, *args.batch_sizes})
    result: dict[str, Any] = {
        "schema": "eurosys_e3_ktransformers_direct_batch_v3",
        "status": "running",
        "started_at": utc_now(),
        "fixture": {
            "path": str(args.fixture.resolve()),
            "metadata": metadata,
        },
        "configuration": {
            "model": str(args.model.resolve()),
            "frequency_plan": str(args.frequency_plan.resolve()),
            "frequency_plan_sha256": hashlib.sha256(
                args.frequency_plan.read_bytes()
            ).hexdigest(),
            "workload": args.workload,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "batch_sizes": args.batch_sizes,
            "gpu_experts_per_layer": args.gpu_experts,
            "cpu_experts_per_layer": 128 - args.gpu_experts,
            "cpu_threads": args.cpu_threads,
            "gpu_prefill_token_threshold": args.gpu_prefill_token_threshold,
            "attention_backend": args.attention_backend,
            "prefill_attention_backend": args.attention_backend,
            "decode_attention_backend": (
                args.decode_attention_backend or args.attention_backend
            ),
            "transport": "none",
            "http_server_started": False,
            "client_used": False,
            "measurement_repeats": 1,
            "mixed_chunk_enabled": False,
            "context_length": context_length,
            "max_total_tokens": max_total_tokens,
            "max_prefill_tokens": max_prefill_tokens,
            "prefill_covers_full_batch": max_prefill_tokens >= args.prompt_tokens * max_batch,
            "cuda_graph_enabled": args.enable_cuda_graph,
            "cuda_graph_batch_sizes": cuda_graph_batch_sizes,
            "overlap_schedule_enabled": not args.disable_overlap_schedule,
            "warmup_decode_steps": args.warmup_decode_steps,
            "watchdog_timeout_seconds": args.watchdog_timeout_seconds,
        },
        "warmup": None,
        "cells": [],
    }
    atomic_json(args.output, result)
    engine: Any | None = None
    try:
        import sglang as sgl

        engine = sgl.Engine(
            model_path=str(args.model),
            tp_size=1,
            context_length=context_length,
            max_running_requests=max_batch,
            mem_fraction_static=0.97,
            max_total_tokens=max_total_tokens,
            chunked_prefill_size=max_prefill_tokens,
            max_prefill_tokens=max_prefill_tokens,
            schedule_policy="fcfs",
            stream_interval=1,
            random_seed=args.seed,
            trust_remote_code=False,
            kt_weight_path=str(args.model),
            kt_method="FP8",
            kt_cpuinfer=args.cpu_threads,
            kt_threadpool_count=1,
            kt_numa_nodes=[0],
            kt_num_gpu_experts=args.gpu_experts,
            kt_expert_placement_strategy="frequency",
            init_expert_location=str(args.frequency_plan),
            kt_max_deferred_experts_per_token=0,
            kt_gpu_prefill_token_threshold=args.gpu_prefill_token_threshold,
            attention_backend=args.attention_backend,
            decode_attention_backend=args.decode_attention_backend,
            # ``--fp8-gemm-backend`` is the CLI spelling.  The in-process
            # Engine API receives the ServerArgs dataclass field name.
            fp8_gemm_runner_backend="triton",
            cuda_graph_max_bs=max(cuda_graph_batch_sizes),
            cuda_graph_bs=cuda_graph_batch_sizes,
            disable_cuda_graph=not args.enable_cuda_graph,
            disable_overlap_schedule=args.disable_overlap_schedule,
            enable_mixed_chunk=False,
            disable_shared_experts_fusion=True,
            disable_radix_cache=True,
            enable_metrics=True,
            decode_log_interval=1,
            # A full, unchunked 131,072-token RAG/B16 prefill can legitimately
            # exceed SGLang's 300 s default watchdog on the CPU-expert path.
            watchdog_timeout=args.watchdog_timeout_seconds,
            log_level="info",
        )
        warm_prompt = [requests[0].prompt_token_ids]
        warm_started = time.perf_counter()
        warm = engine.generate(
            input_ids=warm_prompt,
            sampling_params=[sampling_params(args.warmup_output_tokens)],
            stream=False,
            rid=["e3-kt-direct-warmup"],
        )
        result["warmup"] = {
            "request_count": 1,
            "output_tokens": args.warmup_output_tokens,
            "elapsed_ms": (time.perf_counter() - warm_started) * 1000.0,
            "successful": bool(warm),
        }
        for batch in args.batch_sizes:
            cell = run_cell(
                engine,
                [request.prompt_token_ids for request in requests[:batch]],
                output_tokens=args.output_tokens,
                request_prefix=f"e3-{args.workload}-b{batch}",
                warmup_decode_steps=args.warmup_decode_steps,
            )
            result["cells"].append(cell)
            atomic_json(args.output, result)
        result["status"] = "ok" if all(
            cell["successful_requests"] == cell["request_count"]
            and cell["single_generate_call"]
            and cell["transport"] == "none"
            and not cell["mixed_chunk_enabled"]
            for cell in result["cells"]
        ) else "error"
    except BaseException as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["finished_at"] = utc_now()
        atomic_json(args.output, result)
        if engine is not None:
            engine.shutdown()
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
