#!/usr/bin/env python3
"""Run one direct-engine E11 continuous-arrival trace.

Four requests are admitted at the idle boundary.  After their Prefill and a
few Decode scheduler steps, another four requests are admitted while the first
cohort is still decoding.  The runner records the actual SchedulerOutput for
every step, rather than inferring batch composition from request timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
        "p99": percentile(values, 99.0),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


@dataclass(frozen=True)
class FixtureRequest:
    index: int
    fixture_id: Any
    prompt_token_ids: list[int]
    prompt_sha256: str


def load_fixture(path: Path, count: int) -> list[FixtureRequest]:
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value.get("requests") if isinstance(value, dict) else None
    if not isinstance(records, list) or len(records) < count:
        raise ValueError(f"fixture must contain at least {count} requests")
    result: list[FixtureRequest] = []
    for index, record in enumerate(records[:count]):
        if not isinstance(record, dict):
            raise ValueError(f"fixture request {index} is not an object")
        token_ids = record.get("prompt_token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(f"fixture request {index} has no prompt tokens")
        if record.get("prompt_tokens") != len(token_ids):
            raise ValueError(f"fixture request {index} prompt length differs")
        digest = hashlib.sha256(
            ",".join(str(token) for token in token_ids).encode("ascii")
        ).hexdigest()
        if record.get("prompt_token_ids_sha256") != digest:
            raise ValueError(f"fixture request {index} prompt hash differs")
        result.append(
            FixtureRequest(
                index=index,
                fixture_id=record.get("request", index),
                prompt_token_ids=[int(token) for token in token_ids],
                prompt_sha256=digest,
            )
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "phase_epoch",
            "remote_phase_epoch",
            "remote_continuous",
            "hybrid_fail_closed",
            "always_copy_continuous",
            "adaptmoe_copy_eager_continuous",
            "native_copy_eager_continuous",
            "native_uva_eager_continuous",
        ),
    )
    parser.add_argument("--initial-requests", type=int, default=4)
    parser.add_argument("--late-requests", type=int, default=4)
    parser.add_argument("--late-after-engine-steps", type=int, default=4)
    parser.add_argument(
        "--arrival-qps", type=float,
        help="Replay all requests at offsets i/QPS instead of two step-based cohorts",
    )
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=2147483648)
    parser.add_argument("--cpu-offload-gb", type=float, default=30.0)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if not (args.model / "config.json").is_file():
        raise SystemExit(f"model snapshot is missing: {args.model}")
    if args.initial_requests <= 0 or args.late_requests <= 0:
        raise SystemExit("both request cohorts must be non-empty")
    if args.late_after_engine_steps < 2:
        raise SystemExit("late cohort must arrive after Prefill and Decode begin")
    if args.max_model_len <= args.output_tokens:
        raise SystemExit("max model length must exceed requested output length")
    if args.max_num_batched_tokens <= 0 or args.max_num_seqs <= 0:
        raise SystemExit("scheduler capacity values must be positive")
    if args.arrival_qps is not None and (
        not math.isfinite(args.arrival_qps) or args.arrival_qps <= 0
    ):
        raise SystemExit("arrival QPS must be finite and positive")

    total_requests = args.initial_requests + args.late_requests
    fixture = load_fixture(args.fixture, total_requests)
    expected_phase_batched = args.mode in {
        "phase_epoch",
        "remote_phase_epoch",
    }
    actual_phase_batched = os.environ.get(
        "VLLM_UVM_PHASE_BATCHED_SCHEDULING", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    cuda_profiler_capture = os.environ.get(
        "VLLM_E11_CUDA_PROFILER_CAPTURE", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if actual_phase_batched != expected_phase_batched:
        raise SystemExit(
            "VLLM_UVM_PHASE_BATCHED_SCHEDULING differs from the selected mode"
        )

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    # FULL_SEPARATE is the phase-specific Prefill/Decode mechanism. The
    # remote-only fallback and the paired copy-on-forward comparison use eager
    # execution. The legacy always-copy probe keeps eager Prefill/mixed work
    # but replays copy-bearing Graphs for uniform Decode.
    if args.mode in {
        "remote_phase_epoch",
        "remote_continuous",
        "adaptmoe_copy_eager_continuous",
        "native_copy_eager_continuous",
        "native_uva_eager_continuous",
    }:
        compilation_config = {
            "mode": "NONE",
            "cudagraph_mode": "NONE",
        }
    elif args.mode == "always_copy_continuous":
        compilation_config = {
            "mode": "NONE",
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": [1, 4, 8, 128, 512, 1024],
            "max_cudagraph_capture_size": 1024,
        }
    else:
        compilation_config = {
            "mode": "NONE",
            "cudagraph_mode": "FULL_SEPARATE",
            "cudagraph_capture_sizes": [1, 4, 8, 128, 512, 1024],
            "max_cudagraph_capture_size": 1024,
        }
    result: dict[str, Any] = {
        "schema": "eurosys_e11_continuous_arrival_direct_engine_v1",
        "status": "running",
        "started_at": utc_now(),
        "mode": args.mode,
        "configuration": {
            "model": str(args.model.resolve()),
            "fixture": str(args.fixture.resolve()),
            "initial_requests": args.initial_requests,
            "late_requests": args.late_requests,
            "late_after_engine_steps": args.late_after_engine_steps,
            "arrival_qps": args.arrival_qps,
            "arrival_policy": "fixed_interval" if args.arrival_qps else "two_cohort_steps",
            "latency_origin": "planned_arrival" if args.arrival_qps else "engine_admission",
            "output_tokens": args.output_tokens,
            "seed": args.seed,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "cpu_offload_gb": args.cpu_offload_gb,
            "max_model_len": args.max_model_len,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "phase_batched_scheduling": actual_phase_batched,
            "phase_mode": os.environ.get("VLLM_UVM_ORIGINAL_VA_PHASE_MODE"),
            "compilation_config": compilation_config,
            "cuda_profiler_capture": cuda_profiler_capture,
            "transport": (
                "static_alias_h2d_every_forward"
                if args.mode
                in {
                    "always_copy_continuous",
                    "adaptmoe_copy_eager_continuous",
                }
                else "native_parameter_copy_on_forward"
                if args.mode == "native_copy_eager_continuous"
                else "native_parameter_uva"
                if args.mode == "native_uva_eager_continuous"
                else "none"
            ),
        },
        "scheduler_steps": [],
        "requests": [],
    }
    atomic_json(args.output, result)

    llm: Any | None = None
    try:
        llm = LLM(
            model=str(args.model),
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=0.97,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            cpu_offload_gb=args.cpu_offload_gb,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            enable_prefix_caching=False,
            generation_config="vllm",
            compilation_config=compilation_config,
            disable_log_stats=False,
        )

        core = llm.llm_engine.engine_core.engine_core
        scheduler = core.scheduler
        original_schedule = scheduler.schedule
        scheduler_records: list[dict[str, Any]] = result["scheduler_steps"]
        scheduler_call = 0

        def traced_schedule() -> Any:
            nonlocal scheduler_call
            scheduler_call += 1
            before = time.perf_counter()
            waiting_before = [request.request_id for request in scheduler.waiting]
            running_before = [request.request_id for request in scheduler.running]
            output = original_schedule()
            after = time.perf_counter()
            new_ids = [request.req_id for request in output.scheduled_new_reqs]
            cached_ids = list(output.scheduled_cached_reqs.req_ids)
            scheduler_records.append(
                {
                    "scheduler_step": scheduler_call,
                    "started_perf_counter": before,
                    "finished_perf_counter": after,
                    "duration_ms": (after - before) * 1000.0,
                    "waiting_before": waiting_before,
                    "running_before": running_before,
                    "scheduled_new_request_ids": new_ids,
                    "scheduled_cached_request_ids": cached_ids,
                    "num_scheduled_tokens": {
                        str(key): int(value)
                        for key, value in output.num_scheduled_tokens.items()
                    },
                    "total_num_scheduled_tokens": int(
                        output.total_num_scheduled_tokens
                    ),
                    "mixed_new_and_cached": bool(new_ids and cached_ids),
                    "uvm_phase_epoch": getattr(
                        scheduler, "_uvm_phase_epoch", None
                    ),
                    "uvm_phase_epoch_request_ids": sorted(
                        getattr(scheduler, "_uvm_phase_epoch_request_ids", set())
                    ),
                }
            )
            return output

        scheduler.schedule = traced_schedule

        params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.output_tokens,
            min_tokens=args.output_tokens,
            ignore_eos=True,
            seed=args.seed,
            output_kind=RequestOutputKind.DELTA,
        )

        logical_by_id: dict[str, str] = {}
        fixture_by_id: dict[str, FixtureRequest] = {}
        admitted_at: dict[str, float] = {}
        arrival_at: dict[str, float] = {}
        admitted_before_scheduler_step: dict[str, int] = {}
        first_scheduled_at: dict[str, float] = {}
        first_scheduled_step: dict[str, int] = {}
        token_times: dict[str, list[float]] = {}
        output_ids: dict[str, list[int]] = {}
        finished_at: dict[str, float] = {}

        def admit(cohort: str, records: list[FixtureRequest]) -> list[str]:
            request_ids: list[str] = []
            for fixture_request in records:
                cohort_index = fixture_request.index - (
                    args.initial_requests if cohort == "late" else 0
                )
                request_id = f"{cohort}-{cohort_index:02d}"
                if request_id in admitted_at:
                    raise RuntimeError(f"duplicate admission {request_id}")
                planned = (
                    experiment_started + fixture_request.index / args.arrival_qps
                    if args.arrival_qps else None
                )
                arrival_epoch = (
                    experiment_started_epoch + fixture_request.index / args.arrival_qps
                    if args.arrival_qps else time.time()
                )
                prompt = {"prompt_token_ids": fixture_request.prompt_token_ids}
                engine_request = llm.processor.process_inputs(
                    request_id,
                    prompt,
                    params,
                    arrival_time=arrival_epoch,
                    priority=0,
                )
                llm.llm_engine.add_request(
                    request_id,
                    engine_request,
                    params,
                    priority=0,
                )
                observed = time.perf_counter()
                logical_by_id[request_id] = (
                    f"{cohort}-{fixture_request.index:02d}"
                )
                fixture_by_id[request_id] = fixture_request
                admitted_at[request_id] = observed
                arrival_at[request_id] = planned if planned is not None else observed
                admitted_before_scheduler_step[request_id] = scheduler_call
                token_times[request_id] = []
                output_ids[request_id] = []
                request_ids.append(request_id)
            return request_ids

        profiler_started = False
        if cuda_profiler_capture:
            import torch

            torch.cuda.synchronize()
            profiler_status = int(torch.cuda.cudart().cudaProfilerStart())
            if profiler_status != 0:
                raise RuntimeError(
                    f"cudaProfilerStart failed with status {profiler_status}"
                )
            profiler_started = True
        try:
            experiment_started_epoch = time.time()
            experiment_started = time.perf_counter()
            initial_ids = (
                [] if args.arrival_qps
                else admit("initial", fixture[: args.initial_requests])
            )
            late_ids: list[str] = []
            next_request = 0
            engine_steps = 0
            late_admitted_after_step: int | None = None

            while llm.llm_engine.has_unfinished_requests() or (
                next_request < total_requests if args.arrival_qps else not late_ids
            ):
                if args.arrival_qps:
                    while (
                        next_request < total_requests
                        and time.perf_counter() >= experiment_started + next_request / args.arrival_qps
                    ):
                        cohort = "initial" if next_request < args.initial_requests else "late"
                        ids = admit(cohort, [fixture[next_request]])
                        (initial_ids if cohort == "initial" else late_ids).extend(ids)
                        next_request += 1
                    if not llm.llm_engine.has_unfinished_requests():
                        if next_request < total_requests:
                            time.sleep(max(0.0, experiment_started + next_request / args.arrival_qps - time.perf_counter()))
                        continue
                elif not late_ids and engine_steps == args.late_after_engine_steps:
                    unfinished_initial = [
                        request_id
                        for request_id in initial_ids
                        if request_id not in finished_at
                    ]
                    if not unfinished_initial:
                        raise RuntimeError(
                            "initial cohort finished before the late arrival boundary"
                        )
                    late_ids = admit(
                        "late",
                        fixture[
                            args.initial_requests : args.initial_requests
                            + args.late_requests
                        ],
                    )
                    late_admitted_after_step = engine_steps

                step_index = engine_steps + 1
                step_started = time.perf_counter()
                if profiler_started:
                    torch.cuda.nvtx.range_push(f"e11_engine_step_{step_index}")
                try:
                    step_outputs = llm.llm_engine.step()
                finally:
                    if profiler_started:
                        torch.cuda.nvtx.range_pop()
                observed_at = time.perf_counter()
                engine_steps += 1
                if scheduler_records:
                    current_schedule = scheduler_records[-1]
                    current_schedule["engine_step_started_perf_counter"] = (
                        step_started
                    )
                    current_schedule["engine_step_finished_perf_counter"] = (
                        observed_at
                    )
                    current_schedule["engine_step_duration_ms"] = (
                        observed_at - step_started
                    ) * 1000.0
                    for request_id in (
                        current_schedule["scheduled_new_request_ids"]
                        + current_schedule["scheduled_cached_request_ids"]
                    ):
                        if request_id not in first_scheduled_at:
                            first_scheduled_at[request_id] = float(
                                current_schedule["finished_perf_counter"]
                            )
                            first_scheduled_step[request_id] = int(
                                current_schedule["scheduler_step"]
                            )
                for output in step_outputs:
                    request_id = str(output.request_id)
                    if request_id not in logical_by_id:
                        raise RuntimeError(f"unexpected request output {request_id}")
                    if len(output.outputs) != 1:
                        raise RuntimeError(
                            f"request {request_id} returned multiple choices"
                        )
                    delta = list(output.outputs[0].token_ids)
                    output_ids[request_id].extend(delta)
                    token_times[request_id].extend([observed_at] * len(delta))
                    if output.finished:
                        finished_at[request_id] = observed_at

            experiment_finished = time.perf_counter()
            experiment_finished_epoch = time.time()
        finally:
            if profiler_started:
                import torch

                torch.cuda.synchronize()
                profiler_status = int(torch.cuda.cudart().cudaProfilerStop())
                if profiler_status != 0:
                    raise RuntimeError(
                        f"cudaProfilerStop failed with status {profiler_status}"
                    )
        expected_ids = set(initial_ids + late_ids)
        if set(finished_at) != expected_ids:
            raise RuntimeError(
                f"finished IDs differ: expected={sorted(expected_ids)}, "
                f"actual={sorted(finished_at)}"
            )

        request_records: list[dict[str, Any]] = []
        for request_id in initial_ids + late_ids:
            times = token_times[request_id]
            ids = output_ids[request_id]
            fixture_request = fixture_by_id[request_id]
            itl_ms = [
                (right - left) * 1000.0 for left, right in zip(times, times[1:])
            ]
            request_records.append(
                {
                    "request_id": request_id,
                    "logical_request": logical_by_id[request_id],
                    "cohort": request_id.split("-", 1)[0],
                    "fixture_index": fixture_request.index,
                    "fixture_request_id": fixture_request.fixture_id,
                    "prompt_tokens": len(fixture_request.prompt_token_ids),
                    "prompt_token_ids_sha256": fixture_request.prompt_sha256,
                    "completion_tokens": len(ids),
                    "output_token_ids": ids,
                    "output_token_ids_sha256": hashlib.sha256(
                        json.dumps(ids, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                    "success": len(ids) == args.output_tokens,
                    "planned_arrival_offset_seconds": arrival_at[request_id] - experiment_started,
                    "admitted_offset_seconds": admitted_at[request_id] - experiment_started,
                    "admission_delay_ms": (admitted_at[request_id] - arrival_at[request_id]) * 1000.0,
                    "admitted_before_scheduler_step": (
                        admitted_before_scheduler_step[request_id]
                    ),
                    "first_scheduled_step": first_scheduled_step[request_id],
                    "deferred_scheduler_steps": max(
                        0,
                        first_scheduled_step[request_id]
                        - admitted_before_scheduler_step[request_id]
                        - 1,
                    ),
                    "queueing_ms": (
                        first_scheduled_at[request_id] - arrival_at[request_id]
                    )
                    * 1000.0,
                    "engine_queueing_ms": (first_scheduled_at[request_id] - admitted_at[request_id]) * 1000.0,
                    "ttft_ms": (times[0] - arrival_at[request_id]) * 1000.0,
                    "tpot_ms": statistics.fmean(itl_ms) if itl_ms else None,
                    "e2e_ms": (
                        finished_at[request_id] - arrival_at[request_id]
                    )
                    * 1000.0,
                    "itl_ms": itl_ms,
                }
            )
        result["requests"] = request_records

        successful = [record for record in request_records if record["success"]]
        ttft = [float(record["ttft_ms"]) for record in successful]
        tpot = [float(record["tpot_ms"]) for record in successful]
        e2e = [float(record["e2e_ms"]) for record in successful]
        queueing = [float(record["queueing_ms"]) for record in successful]
        slo_good = [
            record
            for record in successful
            if float(record["ttft_ms"]) <= 2000.0
            and float(record["tpot_ms"]) <= 100.0
        ]
        elapsed = experiment_finished - experiment_started
        mixed_steps = [
            record for record in scheduler_records if record["mixed_new_and_cached"]
        ]
        result["summary"] = {
            "request_count": len(request_records),
            "successful_requests": len(successful),
            "engine_steps": engine_steps,
            "late_admitted_after_engine_step": late_admitted_after_step,
            "scheduler_steps": len(scheduler_records),
            "mixed_scheduler_steps": len(mixed_steps),
            "mixed_scheduler_step_indices": [
                int(record["scheduler_step"]) for record in mixed_steps
            ],
            "empty_scheduler_steps": sum(
                int(record["total_num_scheduled_tokens"]) == 0
                for record in scheduler_records
            ),
            "elapsed_seconds": elapsed,
            "offered_qps": args.arrival_qps,
            "arrival_span_seconds": (total_requests - 1) / args.arrival_qps if args.arrival_qps else None,
            "admission_delay_ms": distribution([float(record["admission_delay_ms"]) for record in successful]),
            "generated_tokens": sum(
                int(record["completion_tokens"]) for record in successful
            ),
            "token_throughput_per_second": (
                sum(int(record["completion_tokens"]) for record in successful)
                / elapsed
            ),
            "request_throughput_per_second": len(successful) / elapsed,
            "slo_good_requests": len(slo_good),
            "goodput_requests_per_second": len(slo_good) / elapsed,
            "ttft_ms": distribution(ttft),
            "tpot_ms": distribution(tpot),
            "e2e_ms": distribution(e2e),
            "queueing_ms": distribution(queueing),
            "initial_queueing_ms": distribution(
                [
                    float(record["queueing_ms"])
                    for record in successful
                    if record["cohort"] == "initial"
                ]
            ),
            "late_queueing_ms": distribution(
                [
                    float(record["queueing_ms"])
                    for record in successful
                    if record["cohort"] == "late"
                ]
            ),
            "late_max_deferred_scheduler_steps": max(
                int(record["deferred_scheduler_steps"])
                for record in successful
                if record["cohort"] == "late"
            ),
        }
        result["measurement_window"] = {
            "started_epoch": experiment_started_epoch,
            "finished_epoch": experiment_finished_epoch,
        }
        if args.mode == "phase_epoch" and mixed_steps:
            raise RuntimeError("phase epoch emitted a mixed scheduler step")
        if (
            args.mode
            in {
                "remote_continuous",
                "always_copy_continuous",
                "adaptmoe_copy_eager_continuous",
                "native_copy_eager_continuous",
                "native_uva_eager_continuous",
            }
            and not mixed_steps
            and args.arrival_qps is None
        ):
            raise RuntimeError("default continuous scheduler did not emit a mixed step")
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
