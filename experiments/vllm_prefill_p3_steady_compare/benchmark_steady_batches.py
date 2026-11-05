#!/usr/bin/env python3
"""Measure steady-session prefill batches without probes in the hot window.

The existing formal client intentionally measures one batch per process.  This
variant keeps the same model and request contract, but performs conditioning
and repeated measurements in one process.  Output inspection and serialization
are deferred until every timed batch has completed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER_ROOT = REPO_ROOT / "experiments" / "broad_offload_compare"
sys.path.insert(0, str(HELPER_ROOT))
import benchmark_custom_offload as base  # noqa: E402


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeated steady-session custom-vLLM prefill benchmark."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--evaluation-fixture", required=True, type=Path)
    parser.add_argument("--warmup-fixture", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-label", required=True)
    parser.add_argument("--mechanism-class", required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, default=544)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-prompt-token-ids", action="store_true")
    parser.add_argument(
        "--conditioning-batches",
        type=int,
        default=env_int("STEADY_CONDITIONING_BATCHES", 3),
    )
    parser.add_argument(
        "--measurement-batches",
        type=int,
        default=env_int("STEADY_MEASUREMENT_BATCHES", 5),
    )
    parser.add_argument(
        "--inter-batch-idle-ms",
        type=float,
        default=env_float("STEADY_INTER_BATCH_IDLE_MS", 0.0),
        help="Intentional idle before each batch after the initial warmup.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]
]:
    if args.concurrency <= 0 or args.output_tokens <= 0:
        raise SystemExit("concurrency and output tokens must be positive")
    if args.conditioning_batches < 0 or args.measurement_batches <= 0:
        raise SystemExit("conditioning batches must be non-negative and measurements positive")
    if not math.isfinite(args.inter_batch_idle_ms) or args.inter_batch_idle_ms < 0:
        raise SystemExit("inter-batch idle must be finite and non-negative")
    if not math.isfinite(args.cpu_offload_gb) or args.cpu_offload_gb < 0:
        raise SystemExit("cpu-offload-gb must be finite and non-negative")
    if args.warmup_requests <= 0 or args.warmup_fixture is None:
        raise SystemExit("steady mode requires a non-empty disjoint warmup fixture")
    if args.warmup_output_tokens <= 0:
        raise SystemExit("warmup output tokens must be positive")
    if args.max_num_seqs < max(args.concurrency, args.warmup_requests):
        raise SystemExit("max_num_seqs is smaller than a requested batch")

    eval_meta, evaluation = base.load_fixture(args.evaluation_fixture)
    warm_meta, warmup = base.load_fixture(args.warmup_fixture)
    if len(evaluation) < args.concurrency or len(warmup) < args.warmup_requests:
        raise SystemExit("a fixture is smaller than its requested batch")
    selected_eval = evaluation[: args.concurrency]
    selected_warmup = warmup[: args.warmup_requests]
    if any(
        item["prompt_tokens"] + args.output_tokens > args.max_model_len
        for item in selected_eval
    ) or any(
        item["prompt_tokens"] + args.warmup_output_tokens > args.max_model_len
        for item in selected_warmup
    ):
        raise SystemExit("prompt plus output exceeds max_model_len")
    if args.require_prompt_token_ids and any(
        item.get("prompt_token_ids") is None
        for item in [*selected_eval, *selected_warmup]
    ):
        raise SystemExit("formal mode requires frozen prompt_token_ids")
    eval_indices = base.source_indices(eval_meta)
    warm_indices = base.source_indices(warm_meta)
    if eval_indices is None or warm_indices is None:
        raise SystemExit("fixtures must expose metadata.source_indices")
    overlap = sorted(eval_indices & warm_indices)
    if overlap:
        raise SystemExit(f"warmup/evaluation fixture overlap: {overlap[:16]}")
    return eval_meta, selected_eval, warm_meta, selected_warmup


def main() -> int:
    args = parse_args()
    eval_meta, evaluation, warm_meta, warmup = validate_args(args)

    # CUDA-capable imports deliberately follow all static argument validation.
    import torch
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt

    def engine_inputs(records: list[dict[str, Any]]) -> list[str | TokensPrompt]:
        have_ids = [item.get("prompt_token_ids") is not None for item in records]
        if any(have_ids) and not all(have_ids):
            raise RuntimeError("a batch cannot mix text prompts and token-ID prompts")
        if all(have_ids):
            return [
                TokensPrompt(prompt_token_ids=list(item["prompt_token_ids"]))
                for item in records
            ]
        return [str(item["prompt"]) for item in records]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started_epoch = time.time()
    load_started = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        cpu_offload_gb=args.cpu_offload_gb,
        enable_prefix_caching=False,
        seed=args.seed,
        disable_log_stats=False,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    ready_torch = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    torch.cuda.reset_peak_memory_stats()

    warm_sampling = SamplingParams(
        temperature=0.0,
        seed=args.seed,
        ignore_eos=True,
        min_tokens=args.warmup_output_tokens,
        max_tokens=args.warmup_output_tokens,
    )
    eval_sampling = SamplingParams(
        temperature=0.0,
        seed=args.seed,
        ignore_eos=True,
        min_tokens=args.output_tokens,
        max_tokens=args.output_tokens,
    )

    raw_batches: list[dict[str, Any]] = []
    previous_end_epoch: float | None = None

    def run_batch(
        role: str,
        index: int,
        records: list[dict[str, Any]],
        sampling: Any,
        requested_idle_ms: float,
    ) -> dict[str, Any]:
        nonlocal previous_end_epoch
        if requested_idle_ms:
            time.sleep(requested_idle_ms / 1000.0)
        torch.cuda.synchronize()
        start_epoch = time.time()
        started = time.perf_counter()
        outputs = llm.generate(engine_inputs(records), sampling, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - started
        end_epoch = time.time()
        actual_idle_ms = (
            None
            if previous_end_epoch is None
            else max(0.0, (start_epoch - previous_end_epoch) * 1000.0)
        )
        previous_end_epoch = end_epoch
        return {
            "role": role,
            "index": index,
            "start_epoch": start_epoch,
            "end_epoch": end_epoch,
            "elapsed_seconds": elapsed_seconds,
            "requested_idle_ms": requested_idle_ms,
            "actual_idle_ms": actual_idle_ms,
            "expected": records,
            "outputs": outputs,
        }

    # No subprocess, NVML query, serialization, or output inspection occurs
    # from this point until all conditioning and measured batches complete.
    raw_warmup = run_batch("warmup", 0, warmup, warm_sampling, 0.0)
    for index in range(args.conditioning_batches):
        raw_batches.append(
            run_batch(
                "conditioning",
                index,
                evaluation,
                eval_sampling,
                args.inter_batch_idle_ms,
            )
        )
    for index in range(args.measurement_batches):
        raw_batches.append(
            run_batch(
                "measurement",
                index,
                evaluation,
                eval_sampling,
                args.inter_batch_idle_ms,
            )
        )

    def finalize(raw: dict[str, Any]) -> dict[str, Any]:
        outputs = raw["outputs"]
        expected = raw["expected"]
        if len(outputs) != len(expected):
            raise RuntimeError(
                f"{raw['role']}[{raw['index']}] completed {len(outputs)} "
                f"of {len(expected)} requests"
            )
        records = [
            base.request_record(output, wanted)
            for output, wanted in zip(outputs, expected)
        ]
        if any(
            item["prompt_tokens"] != item["expected_prompt_tokens"]
            or item["prompt_token_ids_exact"] is False
            for item in records
        ):
            raise RuntimeError(f"{raw['role']}[{raw['index']}] prompt mismatch")
        wanted_output_tokens = (
            args.warmup_output_tokens if raw["role"] == "warmup" else args.output_tokens
        )
        if any(item["output_tokens"] != wanted_output_tokens for item in records):
            raise RuntimeError(f"{raw['role']}[{raw['index']}] output count mismatch")
        ttfts = [item["ttft_ms"] for item in records]
        latencies = [item["request_latency_ms"] for item in records]
        if any(value is None or not math.isfinite(float(value)) for value in [*ttfts, *latencies]):
            raise RuntimeError(f"{raw['role']}[{raw['index']}] latency metric missing")
        aggregate_ids = [item["output_token_ids"] for item in records]
        return {
            key: raw[key]
            for key in (
                "role",
                "index",
                "start_epoch",
                "end_epoch",
                "elapsed_seconds",
                "requested_idle_ms",
                "actual_idle_ms",
            )
        } | {
            "completed_requests": len(records),
            "prompt_tokens": sum(int(item["prompt_tokens"]) for item in records),
            "output_tokens": sum(int(item["output_tokens"]) for item in records),
            "aggregate_output_token_ids_sha256": base.sha256_json(aggregate_ids),
            "ttft_ms": base.summarize([float(value) for value in ttfts]),
            "request_latency_ms": base.summarize(
                [float(value) for value in latencies]
            ),
            "requests": records,
        }

    warmup_result = finalize(raw_warmup)
    call_results = [finalize(raw) for raw in raw_batches]
    measured = [item for item in call_results if item["role"] == "measurement"]
    measured_elapsed_ms = [float(item["elapsed_seconds"]) * 1000.0 for item in measured]
    measured_ttft_ms = [float(item["ttft_ms"]["p50"]) for item in measured]

    config = {
        **vars(args),
        "evaluation_fixture": str(args.evaluation_fixture.resolve()),
        "warmup_fixture": str(args.warmup_fixture.resolve()),
        "output": str(args.output.resolve()),
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "ignore_eos": True,
        "temperature": 0.0,
    }
    result = {
        "schema": "vllm_prefill_three_arm_steady_batches_v1",
        "schema_version": 2,
        "status": "ok",
        "profile_label": args.profile_label,
        "mechanism_class": args.mechanism_class,
        "started_epoch": started_epoch,
        "config": config,
        "versions": {
            "vllm": getattr(vllm, "__version__", None),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "fixtures": {
            "evaluation_sha256": base.sha256_file(args.evaluation_fixture),
            "evaluation_metadata": eval_meta,
            "warmup_sha256": base.sha256_file(args.warmup_fixture),
            "warmup_metadata": warm_meta,
            "disjoint_source_indices": True,
        },
        "load_seconds": load_seconds,
        "ready_torch": ready_torch,
        "warmup": warmup_result,
        "calls": call_results,
        "measurement": {
            "sample_unit": "within_process_batch",
            "batch_count": len(measured),
            "elapsed_ms": {
                "samples": measured_elapsed_ms,
                "median": statistics.median(measured_elapsed_ms),
            },
            "ttft_p50_ms": {
                "samples": measured_ttft_ms,
                "median": statistics.median(measured_ttft_ms),
            },
        },
        "environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(
                ("VLLM_UVM_", "VLLM_CPU_OFFLOAD_", "STEADY_")
            )
            or key
            in {
                "VLLM_USE_UVM",
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "CUDA_VISIBLE_DEVICES",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            }
        },
        "final_torch": {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "process_memory": base.process_memory_snapshot(),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output),
                "measurement_batches": len(measured),
                "elapsed_ms_median": statistics.median(measured_elapsed_ms),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
