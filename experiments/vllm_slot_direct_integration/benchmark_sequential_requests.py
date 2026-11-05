#!/usr/bin/env python3
"""Run repeated requests through one LLM instance for slot-cache reuse tests."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_COMMON = (
    SCRIPT_DIR.parent / "broad_offload_compare" / "benchmark_custom_offload.py"
)


def load_common(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "vllm_slot_cache_common_benchmark", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--common-benchmark", type=Path, default=DEFAULT_COMMON)
    parser.add_argument("--model", required=True)
    parser.add_argument("--evaluation-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-label", required=True)
    parser.add_argument("--mechanism-class", required=True)
    parser.add_argument("--sequential-requests", type=int, default=2)
    parser.add_argument("--fixture-request-index", type=int, default=0)
    parser.add_argument("--output-tokens", type=int, default=4)
    parser.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, default=544)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sequential_requests < 2:
        raise SystemExit("sequential-requests must be at least 2")
    if args.fixture_request_index < 0:
        raise SystemExit("fixture-request-index must be non-negative")
    if args.output_tokens <= 0:
        raise SystemExit("output-tokens must be positive")
    if args.max_num_seqs <= 0:
        raise SystemExit("max-num-seqs must be positive")
    if not args.common_benchmark.is_file():
        raise SystemExit(f"common benchmark is missing: {args.common_benchmark}")

    common = load_common(args.common_benchmark)
    fixture_metadata, fixture_requests = common.load_fixture(
        args.evaluation_fixture
    )
    if args.fixture_request_index >= len(fixture_requests):
        raise SystemExit("fixture-request-index exceeds evaluation fixture")
    expected = fixture_requests[args.fixture_request_index]
    if expected.get("prompt_token_ids") is None:
        raise SystemExit("sequential cache test requires frozen prompt token IDs")
    if expected["prompt_tokens"] + args.output_tokens > args.max_model_len:
        raise SystemExit("prompt plus output exceeds max-model-len")

    # CUDA-initializing imports stay below all static argument validation.
    import torch
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt

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
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        cpu_offload_gb=0,
        enable_prefix_caching=False,
        seed=args.seed,
        disable_log_stats=False,
    )
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    ready_nvml = common.nvml_snapshot()
    ready_torch = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }

    sampling = SamplingParams(
        temperature=0.0,
        seed=args.seed,
        ignore_eos=True,
        min_tokens=args.output_tokens,
        max_tokens=args.output_tokens,
    )
    engine_input = TokensPrompt(
        prompt_token_ids=list(expected["prompt_token_ids"])
    )
    records: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    process_pid = os.getpid()
    llm_instance_id = id(llm)
    torch.cuda.reset_peak_memory_stats()
    measurement_start_epoch = time.time()
    measurement_started = time.perf_counter()
    for sequence_index in range(args.sequential_requests):
        call_start_epoch = time.time()
        call_started = time.perf_counter()
        outputs = llm.generate([engine_input], sampling, use_tqdm=False)
        torch.cuda.synchronize()
        call_elapsed = time.perf_counter() - call_started
        call_end_epoch = time.time()
        if len(outputs) != 1:
            raise RuntimeError(
                f"generate call {sequence_index} returned {len(outputs)} outputs"
            )
        record = common.request_record(outputs[0], expected)
        record.update(
            {
                "sequence_index": sequence_index,
                "process_pid": process_pid,
                "llm_instance_id": llm_instance_id,
                "generate_wall_ms": call_elapsed * 1000.0,
            }
        )
        if record["prompt_tokens"] != record["expected_prompt_tokens"]:
            raise RuntimeError(
                f"generate call {sequence_index} changed prompt token count"
            )
        if record["prompt_token_ids_exact"] is not True:
            raise RuntimeError(
                f"generate call {sequence_index} changed prompt token IDs"
            )
        if record["output_tokens"] != args.output_tokens:
            raise RuntimeError(
                f"generate call {sequence_index} returned a short completion"
            )
        records.append(record)
        calls.append(
            {
                "sequence_index": sequence_index,
                "process_pid": process_pid,
                "llm_instance_id": llm_instance_id,
                "request_id": record["request_id"],
                "start_epoch": call_start_epoch,
                "end_epoch": call_end_epoch,
                "wall_ms": call_elapsed * 1000.0,
                "prompt_token_ids_sha256": record["prompt_token_ids_sha256"],
                "output_token_ids_sha256": record["output_token_ids_sha256"],
            }
        )
    elapsed_seconds = time.perf_counter() - measurement_started
    measurement_end_epoch = time.time()
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        raise RuntimeError("measurement elapsed time is not positive and finite")

    ttfts = [float(record["ttft_ms"]) for record in records]
    tpots = [
        float(value)
        for record in records
        if (value := record["tpot_ms"]) is not None
    ]
    latencies = [float(record["request_latency_ms"]) for record in records]
    if any(
        not math.isfinite(value) or value < 0.0
        for value in [*ttfts, *tpots, *latencies]
    ):
        raise RuntimeError("a request latency is invalid")
    if args.output_tokens > 1 and len(tpots) != len(records):
        raise RuntimeError("a multi-token request lacks TPOT")
    output_count = args.sequential_requests * args.output_tokens
    aggregate_ids = [record["output_token_ids"] for record in records]
    after_measurement_nvml = common.nvml_snapshot()
    measurement_torch_peak = {
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    process_memory = common.process_memory_snapshot()

    # V1's in-process EngineCore client has no finalizer which calls
    # EngineCore.shutdown(). Close it explicitly so GPUWorker.shutdown() runs
    # while Python and CUDA are still fully alive. For slot_restore this is the
    # required mode-19 session-release boundary and produces the final trace
    # record checked by the integration verifier. Shutdown is deliberately
    # outside the measured request interval.
    shutdown_started_epoch = time.time()
    llm.llm_engine.engine_core.shutdown()
    shutdown_end_epoch = time.time()

    config = {
        "model": str(Path(args.model).resolve()),
        "evaluation_fixture": str(args.evaluation_fixture.resolve()),
        "output": str(args.output.resolve()),
        "profile_label": args.profile_label,
        "mechanism_class": args.mechanism_class,
        "concurrency": 1,
        "sequential_requests": args.sequential_requests,
        "fixture_request_index": args.fixture_request_index,
        "output_tokens": args.output_tokens,
        "warmup_requests": 0,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "seed": args.seed,
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "cpu_offload_gb": 0,
        "enable_prefix_caching": False,
        "ignore_eos": True,
        "temperature": 0.0,
    }
    result = {
        "schema_version": 2,
        "benchmark_schema": "vllm_slot_cache_sequential_v1",
        "status": "ok",
        "profile_label": args.profile_label,
        "mechanism_class": args.mechanism_class,
        "started_epoch": started_epoch,
        "measurement_start_epoch": measurement_start_epoch,
        "measurement_end_epoch": measurement_end_epoch,
        "config": config,
        "versions": {
            "vllm": getattr(vllm, "__version__", None),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "fixtures": {
            "evaluation_sha256": common.sha256_file(args.evaluation_fixture),
            "evaluation_metadata": fixture_metadata,
        },
        "environment": {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("VLLM_UVM_", "VLLM_CPU_OFFLOAD_"))
            or key
            in {
                "VLLM_USE_UVM",
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "CUDA_VISIBLE_DEVICES",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            }
        },
        "execution": {
            "single_llm_instance": True,
            "llm_load_count": 1,
            "generate_call_count": args.sequential_requests,
            "requests_per_generate": 1,
            "process_pid": process_pid,
            "llm_instance_id": llm_instance_id,
            "engine_shutdown_explicit": True,
            "shutdown_started_epoch": shutdown_started_epoch,
            "shutdown_end_epoch": shutdown_end_epoch,
            "calls": calls,
        },
        "load_seconds": load_seconds,
        "warmup_seconds": 0.0,
        "ready_nvml": ready_nvml,
        "ready_torch": ready_torch,
        "measurement": {
            "elapsed_seconds": elapsed_seconds,
            "completed_requests": len(records),
            "sequential_generate_calls": args.sequential_requests,
            "prompt_tokens": sum(int(record["prompt_tokens"]) for record in records),
            "output_tokens": output_count,
            "output_tokens_per_second": output_count / elapsed_seconds,
            "ttft_ms": common.summarize(ttfts),
            "tpot_ms": common.summarize(tpots),
            "request_latency_ms": common.summarize(latencies),
            "aggregate_output_token_ids_sha256": common.sha256_json(aggregate_ids),
            "requests": records,
        },
        "after_measurement_nvml": after_measurement_nvml,
        "measurement_torch_peak": measurement_torch_peak,
        "process_memory": process_memory,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
