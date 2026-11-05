#!/usr/bin/env python3
"""Run one E2 C1/C16 workload directly through the vLLM EngineCore.

The measurement requests are all admitted before the first engine step.  No
HTTP server, client, socket, or per-request submission task is involved.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from direct_batch_phase_runner import (
    atomic_json,
    load_fixture,
    run_batch,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=(
        "native_all_gpu",
        "native_pinned_uva",
        "native_copy_on_forward",
        "ours",
    ))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--warmup-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--concurrency", choices=(1, 16), default=16, type=int)
    parser.add_argument("--prompt-tokens", required=True, type=int)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--max-model-len", type=int, default=192)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=100663296)
    parser.add_argument("--cpu-offload-gb", type=float, required=True)
    parser.add_argument("--ttft-slo-ms", type=float, required=True)
    parser.add_argument("--tpot-slo-ms", type=float, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if not (args.model / "config.json").is_file():
        raise SystemExit(f"model snapshot is missing: {args.model}")
    eval_metadata, evaluation = load_fixture(args.fixture)
    warm_metadata, warmups = load_fixture(args.warmup_fixture)
    if eval_metadata.get("workload") != args.workload:
        raise SystemExit("evaluation fixture workload differs from --workload")
    if warm_metadata.get("workload") != args.workload:
        raise SystemExit("warmup fixture workload differs from --workload")
    if len(evaluation) < args.concurrency or len(warmups) < args.warmup_requests:
        raise SystemExit("fixture does not contain the required requests")
    if any(
        len(request.prompt_token_ids) != args.prompt_tokens
        for request in evaluation[:args.concurrency]
    ):
        raise SystemExit("the first 16 evaluation prompts have unexpected lengths")

    # Environment-dependent backends are imported only after argument and
    # fixture validation.  This is especially important for the UVM arm.
    from vllm import LLM

    ours = args.arm == "ours"
    force_cudagraph = os.environ.get("E2_C16_FORCE_CUDAGRAPH", "0") == "1"
    requested_native_mode = os.environ.get(
        "E2_C16_CUDAGRAPH_MODE",
        "FULL_SEPARATE" if force_cudagraph else "NONE",
    ).strip().upper()
    if requested_native_mode not in ("NONE", "FULL_DECODE_ONLY", "FULL_SEPARATE"):
        raise ValueError(
            f"unsupported E2_C16_CUDAGRAPH_MODE={requested_native_mode!r}"
        )
    cudagraph_mode = "FULL_SEPARATE" if ours else requested_native_mode
    use_cudagraph = cudagraph_mode != "NONE"
    pure_prefill_capture_requests: int | None = None
    if cudagraph_mode == "FULL_SEPARATE" and args.warmup_requests == 0:
        if args.max_num_batched_tokens % args.prompt_tokens != 0:
            raise ValueError(
                "A warmup-free FULL_SEPARATE run requires the formal Prefill "
                "Graph shape to contain an integral number of full prompts"
            )
        pure_prefill_capture_requests = min(
            args.concurrency,
            args.max_num_batched_tokens // args.prompt_tokens,
        )
        os.environ["VLLM_CUDAGRAPH_PURE_PREFILL_NUM_REQS"] = str(
            pure_prefill_capture_requests
        )
    compilation_config: dict[str, Any]
    if use_cudagraph:
        capture_sizes = (
            {1, args.concurrency}
            if cudagraph_mode == "FULL_DECODE_ONLY"
            # Warmups contain one full prompt, while the measurement admits
            # all 16 prompts before the first step.  Capture both Prefill
            # shapes: padding a one-request warmup directly to the full-batch
            # shape can replay stale/uninitialised input slots on Native UVA.
            else (
                {
                    1,
                    args.concurrency,
                    args.max_num_batched_tokens,
                }
                # BatchDescriptor does not encode the number of Prefill
                # requests or their individual query lengths.  A one-request
                # warmup therefore must not replay a graph captured for an
                # equal-token, max_num_seqs dummy batch.  When warmups are
                # disabled, capture only the formal all-request Prefill shape.
                if args.warmup_requests == 0
                else {
                    1,
                    args.concurrency,
                    min(args.prompt_tokens, args.max_num_batched_tokens),
                    args.max_num_batched_tokens,
                }
            )
        )
        compilation_config = {
            "mode": "NONE",
            "cudagraph_mode": cudagraph_mode,
            "cudagraph_capture_sizes": sorted(capture_sizes),
            "max_cudagraph_capture_size": max(capture_sizes),
        }
    else:
        compilation_config = {"mode": "NONE", "cudagraph_mode": "NONE"}

    result: dict[str, Any] = {
        "schema": "eurosys_e2_c16_direct_batch_vllm_v2",
        "status": "running",
        "started_at": utc_now(),
        "configuration": {
            "arm": args.arm,
            "workload": args.workload,
            "model": str(args.model.resolve()),
            "fixture": str(args.fixture.resolve()),
            "warmup_fixture": str(args.warmup_fixture.resolve()),
            "concurrency": args.concurrency,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "warmup_requests": args.warmup_requests,
            "warmup_output_tokens": args.warmup_output_tokens,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.concurrency,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "cpu_offload_gb": args.cpu_offload_gb,
            "compilation_config": compilation_config,
            "enforce_eager": not use_cudagraph,
            "force_cudagraph_diagnostic": force_cudagraph,
            "native_uva_cudagraph_mode": requested_native_mode,
            "pure_prefill_capture_requests": pure_prefill_capture_requests,
            "transport": "none",
            "http_server_started": False,
            "client_used": False,
            "measurement_admission": (
                f"{args.concurrency} add_request calls before first step"
            ),
            "measurement_request_batch_size": args.concurrency,
            "prefill_decode_policy": "phase-batched; no mixed step",
            "ttft_slo_ms": args.ttft_slo_ms,
            "tpot_slo_ms": args.tpot_slo_ms,
            "cpu_offload_backend": os.environ.get("VLLM_CPU_OFFLOAD_BACKEND"),
            "cpu_offload_exact_names_file": os.environ.get(
                "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE"
            ),
            "runtime_hot_audit": os.environ.get(
                "VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT"
            ),
        },
        "warmups": [],
        "measurement": None,
    }
    atomic_json(args.output, result)
    llm: Any | None = None
    try:
        llm = LLM(
            model=str(args.model),
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=not use_cudagraph,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=0.97,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            cpu_offload_gb=args.cpu_offload_gb,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.concurrency,
            enable_prefix_caching=False,
            generation_config="vllm",
            compilation_config=compilation_config,
            disable_log_stats=False,
        )
        for index in range(args.warmup_requests):
            print(f"DIRECT_C16_WARMUP_START index={index}", flush=True)
            warmup = run_batch(
                llm,
                [warmups[index]],
                output_tokens=args.warmup_output_tokens,
                seed=args.seed,
                request_prefix=f"c16-warmup-{index}",
                ttft_slo_ms=args.ttft_slo_ms,
                tpot_slo_ms=args.tpot_slo_ms,
            )
            result["warmups"].append(warmup)
            atomic_json(args.output, result)
            if warmup["successful_requests"] != 1:
                raise RuntimeError(f"warmup {index} failed")

        print(
            "DIRECT_BATCH_MEASUREMENT_ADMITTED "
            f"expected_requests={args.concurrency}",
            flush=True,
        )
        measurement = run_batch(
            llm,
            evaluation[:args.concurrency],
            output_tokens=args.output_tokens,
            seed=args.seed,
            request_prefix=f"c{args.concurrency}-measurement",
            ttft_slo_ms=args.ttft_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms,
        )
        result["measurement"] = measurement
        first_count = measurement["first_step_request_count"]
        print(
            "DIRECT_BATCH_FIRST_STEP "
            f"request_count={first_count} expected={args.concurrency} "
            f"single_batch={measurement['single_first_step_batch']}",
            flush=True,
        )
        result["status"] = "ok" if (
            measurement["successful_requests"] == args.concurrency
            and measurement["request_count"] == args.concurrency
            and measurement["admitted_request_count"] == args.concurrency
            and measurement["single_admission_batch"] is True
        ) else "error"
    except BaseException as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["finished_at"] = utc_now()
        atomic_json(args.output, result)
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
