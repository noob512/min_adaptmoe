#!/usr/bin/env python3
"""Run one E2 C16 workload in MoE-Infinity without HTTP.

The serving runtime is instantiated in-process.  Sixteen requests are added
to its waiting queue before the first scheduler/engine step.  The first
scheduler output is recorded directly, so a 1+15 cohort cannot pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from direct_batch_phase_runner import atomic_json, distribution, load_fixture, utc_now


def token_hash(token_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--offload-path", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--warmup-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--prompt-tokens", required=True, type=int)
    parser.add_argument("--device-memory-ratio", type=float, default=0.83)
    parser.add_argument("--kv-cache-ratio", type=float, default=0.014036)
    parser.add_argument("--expected-kv-blocks", type=int, default=192)
    parser.add_argument("--max-tokens-per-step", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, required=True)
    parser.add_argument("--ttft-slo-ms", type=float, required=True)
    parser.add_argument("--tpot-slo-ms", type=float, required=True)
    return parser.parse_args()


def run_batch(
    runtime: Any,
    fixture_requests: list[Any],
    *,
    output_tokens: int,
    prefix: str,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> dict[str, Any]:
    from moe_infinity.serving.sequence import SamplingParams

    params = SamplingParams(
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        max_tokens=output_tokens,
        min_tokens=output_tokens,
        ignore_eos=True,
    )
    admission_started = time.perf_counter()
    request_ids: list[str] = []
    for index, fixture_request in enumerate(fixture_requests):
        request_id = f"{prefix}-{index:02d}"
        runtime.add_request(
            request_id=request_id,
            prompt_token_ids=fixture_request.prompt_token_ids,
            sampling_params=params,
        )
        request_ids.append(request_id)
    admission_finished = time.perf_counter()
    started = admission_finished

    original_schedule = runtime.scheduler.schedule
    first_schedule: dict[str, Any] = {}
    schedule_history: list[dict[str, Any]] = []

    def traced_schedule() -> Any:
        output = original_schedule()
        entry = {
                "prefill_seq_ids": list(output.prefill_seq_ids),
                "decode_seq_ids": list(output.decode_seq_ids),
                "num_prefill_tokens": int(output.num_prefill_tokens),
                "num_decode_tokens": int(output.num_decode_tokens),
        }
        schedule_history.append(entry)
        if not first_schedule:
            first_schedule.update(entry)
        return output

    runtime.scheduler.schedule = traced_schedule
    token_times: dict[str, list[float]] = {request_id: [] for request_id in request_ids}
    output_ids: dict[str, list[int]] = {request_id: [] for request_id in request_ids}
    finished: set[str] = set()
    step_count = 0
    try:
        while runtime.has_pending_requests():
            outputs = runtime.step()
            observed = time.perf_counter()
            step_count += 1
            if not outputs:
                raise RuntimeError("engine made no progress with pending requests")
            for output in outputs:
                request_id = str(output.request_id)
                if request_id not in token_times:
                    raise RuntimeError(f"unexpected output request: {request_id}")
                output_ids[request_id].append(int(output.token_id))
                token_times[request_id].append(observed)
                if output.finished:
                    finished.add(request_id)
    finally:
        runtime.scheduler.schedule = original_schedule
    ended = time.perf_counter()

    records: list[dict[str, Any]] = []
    for request_id, fixture_request in zip(request_ids, fixture_requests):
        times = token_times[request_id]
        ids = output_ids[request_id]
        itl = [(right - left) * 1000.0 for left, right in zip(times, times[1:])]
        success = (
            request_id in finished
            and len(times) == output_tokens
            and len(ids) == output_tokens
        )
        records.append({
            "request_id": request_id,
            "fixture_index": fixture_request.fixture_index,
            "prompt_token_ids_sha256": fixture_request.prompt_token_ids_sha256,
            "completion_tokens": len(ids),
            "output_token_ids": ids,
            "output_token_ids_sha256": token_hash(ids),
            "success": success,
            "ttft_ms": (times[0] - started) * 1000.0 if times else None,
            "tpot_ms": statistics.fmean(itl) if itl else None,
            "itl_ms": itl,
            "e2e_ms": (times[-1] - started) * 1000.0 if times else None,
        })
    successful = [record for record in records if record["success"]]
    expected_first_tokens = sum(
        len(request.prompt_token_ids) for request in fixture_requests
    )
    first_prefill_count = len(first_schedule.get("prefill_seq_ids", []))
    admitted_prefill_ids: set[int] = set()
    first_decode_schedule_index: int | None = None
    for schedule_index, entry in enumerate(schedule_history):
        if entry["decode_seq_ids"] and first_decode_schedule_index is None:
            first_decode_schedule_index = schedule_index
        if first_decode_schedule_index is None:
            admitted_prefill_ids.update(entry["prefill_seq_ids"])
    return {
        "request_count": len(records),
        "successful_requests": len(successful),
        "failed_requests": len(records) - len(successful),
        "admission_duration_ms": (admission_finished - admission_started) * 1000.0,
        "timing_boundary": "after all add_request calls; before first engine step",
        "all_requests_added_before_first_step": True,
        "admitted_request_count": len(request_ids),
        "single_admission_batch": True,
        "first_scheduler_output": first_schedule,
        "scheduler_output_count": len(schedule_history),
        "first_decode_schedule_index": first_decode_schedule_index,
        "prefill_sequences_before_first_decode": len(admitted_prefill_ids),
        "pure_phase_no_mixed_scheduler_output": all(
            not (entry["prefill_seq_ids"] and entry["decode_seq_ids"])
            for entry in schedule_history
        ),
        "first_step_request_count": first_prefill_count,
        "single_first_step_batch": (
            first_prefill_count == len(records)
            and not first_schedule.get("decode_seq_ids")
            and first_schedule.get("num_prefill_tokens") == expected_first_tokens
        ),
        "engine_steps": step_count,
        "elapsed_seconds": ended - started,
        "ttft_ms": distribution([float(r["ttft_ms"]) for r in successful]),
        "tpot_ms": distribution([float(r["tpot_ms"]) for r in successful]),
        "e2e_ms": distribution([float(r["e2e_ms"]) for r in successful]),
        "slo_satisfied_requests": sum(
            r["ttft_ms"] <= ttft_slo_ms and r["tpot_ms"] <= tpot_slo_ms
            for r in successful
        ),
        "slo": {"ttft_ms": ttft_slo_ms, "tpot_ms": tpot_slo_ms},
        "requests": records,
    }


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    for path in (
        args.model / "config.json",
        args.offload_path,
        args.fixture,
        args.warmup_fixture,
    ):
        if not path.exists():
            raise SystemExit(f"required path is missing: {path}")
    eval_metadata, evaluation = load_fixture(args.fixture)
    warm_metadata, warmups = load_fixture(args.warmup_fixture)
    if eval_metadata.get("workload") != args.workload or \
            warm_metadata.get("workload") != args.workload:
        raise SystemExit("fixture workload differs from --workload")
    if len(evaluation) < 16 or len(warmups) < args.warmup_requests:
        raise SystemExit("fixture does not contain the required requests")
    if any(
        len(request.prompt_token_ids) != args.prompt_tokens
        for request in evaluation[:16]
    ):
        raise SystemExit("the first 16 evaluation prompts have unexpected lengths")

    from moe_infinity import MoE
    from moe_infinity.entrypoints.openai import api_server_v2

    result: dict[str, Any] = {
        "schema": "eurosys_e2_c16_direct_batch_moe_infinity_v2",
        "status": "running",
        "started_at": utc_now(),
        "configuration": {
            "arm": "moe_infinity",
            "workload": args.workload,
            "model": str(args.model.resolve()),
            "offload_path": str(args.offload_path.resolve()),
            "fixture": str(args.fixture.resolve()),
            "warmup_fixture": str(args.warmup_fixture.resolve()),
            "concurrency": 16,
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "warmup_requests": args.warmup_requests,
            "warmup_output_tokens": args.warmup_output_tokens,
            "device_memory_ratio": args.device_memory_ratio,
            "kv_cache_ratio": args.kv_cache_ratio,
            "expected_kv_blocks": args.expected_kv_blocks,
            "max_batch_size": 16,
            "max_tokens_per_step": args.max_tokens_per_step,
            "max_seq_length": args.max_seq_length,
            "transport": "none",
            "http_server_started": False,
            "client_used": False,
            "measurement_admission": "16 add_request calls before first step",
            "measurement_request_batch_size": 16,
            "prefill_decode_policy": "waiting-prefill first; full token budget prevents mixed step",
            "serving_logits_policy": (
                "project lm_head for the final hidden state only; preserve full "
                "KV cache and fall back to full causal-LM forward when unsupported"
            ),
            "kv_ownership": "non-paged Hugging Face past_key_values",
            "paged_kv_storage_materialized": False,
            "ttft_slo_ms": args.ttft_slo_ms,
            "tpot_slo_ms": args.tpot_slo_ms,
        },
        "warmups": [],
        "measurement": None,
    }
    atomic_json(args.output, result)
    runtime: Any | None = None
    try:
        model = MoE(
            str(args.model),
            {
                "offload_path": str(args.offload_path.resolve()),
                "device_memory_ratio": args.device_memory_ratio,
                "num_threads": 1,
                "use_native_engine": False,
                "prefetch": False,
                "speculative_prefetch": False,
                "speculative_prefetch_overlap": False,
            },
        )
        api_server_v2.initialize_with_model(
            moe_model=model,
            model_name=str(args.model),
            tok=getattr(model, "tokenizer", None),
            max_seq_length=args.max_seq_length,
            device_memory_ratio=args.device_memory_ratio,
            kv_cache_ratio=args.kv_cache_ratio,
            max_batch_size=16,
            max_tokens_per_step=args.max_tokens_per_step,
            enable_prefix_caching=False,
            materialize_paged_kv_cache=False,
        )
        runtime = api_server_v2.engine
        if runtime is None:
            raise RuntimeError("MoE-Infinity serving runtime was not initialized")
        stats = runtime.get_stats()
        result["initial_runtime_stats"] = stats
        atomic_json(args.output, result)
        actual_blocks = int(stats["kv_cache_num_blocks"])
        if actual_blocks < args.expected_kv_blocks:
            raise RuntimeError(
                f"KV pool has {actual_blocks} blocks; expected at least "
                f"{args.expected_kv_blocks}"
            )
        for index in range(args.warmup_requests):
            print(f"DIRECT_C16_WARMUP_START index={index}", flush=True)
            warmup = run_batch(
                runtime,
                [warmups[index]],
                output_tokens=args.warmup_output_tokens,
                prefix=f"c16-warmup-{index}",
                ttft_slo_ms=args.ttft_slo_ms,
                tpot_slo_ms=args.tpot_slo_ms,
            )
            result["warmups"].append(warmup)
            atomic_json(args.output, result)
            if warmup["successful_requests"] != 1:
                raise RuntimeError(f"warmup {index} failed")

        print("DIRECT_C16_MEASUREMENT_ADMITTED expected_requests=16", flush=True)
        measurement = run_batch(
            runtime,
            evaluation[:16],
            output_tokens=args.output_tokens,
            prefix="c16-measurement",
            ttft_slo_ms=args.ttft_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms,
        )
        result["measurement"] = measurement
        print(
            "DIRECT_C16_FIRST_STEP "
            f"request_count={measurement['first_step_request_count']} expected=16 "
            f"single_batch={measurement['single_first_step_batch']}",
            flush=True,
        )
        result["final_runtime_stats"] = runtime.get_stats()
        result["status"] = "ok" if (
            measurement["successful_requests"] == 16
            and measurement["request_count"] == 16
            and measurement["admitted_request_count"] == 16
            and measurement["single_admission_batch"] is True
        ) else "error"
    except BaseException as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["finished_at"] = utc_now()
        atomic_json(args.output, result)
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
