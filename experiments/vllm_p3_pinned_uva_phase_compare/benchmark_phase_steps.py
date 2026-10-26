#!/usr/bin/env python3
"""Measure strict prefill and decode steps for one vLLM offload mechanism.

The process loads one engine, then runs conditioning and measured request
cycles for every requested batch size.  Each cycle consists of exactly one
complete prefill scheduler step followed by cached one-token decode steps.
There is no file I/O, NVML query, subprocess, or result inspection inside the
timed phase loop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any


def parse_positive_ints(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid positive integer list: {raw!r}")
    if len(values) != len(set(values)):
        raise ValueError(f"batch sizes contain duplicates: {raw!r}")
    return values


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "count": len(values),
        "samples": values,
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_requests(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("requests"), list):
        raise ValueError(f"fixture has no request list: {path}")
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"fixture has no metadata: {path}")
    requests: list[dict[str, Any]] = []
    for index, item in enumerate(value["requests"]):
        if not isinstance(item, dict):
            raise ValueError(f"fixture request {index} is not an object")
        token_ids = item.get("prompt_token_ids")
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(type(token) is not int or token < 0 for token in token_ids)
        ):
            raise ValueError(f"fixture request {index} lacks frozen token IDs")
        declared = item.get("prompt_tokens")
        if declared is not None and declared != len(token_ids):
            raise ValueError(f"fixture request {index} token count differs")
        requests.append(
            {
                "fixture_index": index,
                "prompt_token_ids": list(token_ids),
                "prompt_tokens": len(token_ids),
            }
        )
    if not requests:
        raise ValueError(f"fixture is empty: {path}")
    return metadata, requests


def schedule_snapshot(output: Any) -> dict[str, Any]:
    new_requests = list(output.scheduled_new_reqs)
    cached = output.scheduled_cached_reqs
    return {
        "total_num_scheduled_tokens": int(output.total_num_scheduled_tokens),
        "num_scheduled_tokens": {
            str(key): int(value)
            for key, value in output.num_scheduled_tokens.items()
        },
        "new_request_ids": [str(request.req_id) for request in new_requests],
        "new_request_prompt_tokens": {
            str(request.req_id): len(request.prompt_token_ids or [])
            for request in new_requests
        },
        "cached_request_ids": [str(req_id) for req_id in cached.req_ids],
        "resumed_request_ids": sorted(str(req_id) for req_id in cached.resumed_req_ids),
        "scheduled_spec_decode_request_ids": sorted(
            str(req_id) for req_id in output.scheduled_spec_decode_tokens
        ),
        "scheduled_encoder_request_ids": sorted(
            str(req_id) for req_id in output.scheduled_encoder_inputs
        ),
        "preempted_request_ids": sorted(
            str(req_id) for req_id in (output.preempted_req_ids or set())
        ),
        "finished_request_ids": sorted(str(req_id) for req_id in output.finished_req_ids),
    }


def validate_prefill(schedule: dict[str, Any], batch: int) -> None:
    prompt_tokens = schedule["new_request_prompt_tokens"]
    if len(schedule["new_request_ids"]) != batch:
        raise RuntimeError(f"prefill did not admit {batch} requests: {schedule!r}")
    if schedule["num_scheduled_tokens"] != prompt_tokens:
        raise RuntimeError(f"prefill was not one complete scheduler step: {schedule!r}")
    if schedule["total_num_scheduled_tokens"] != sum(prompt_tokens.values()):
        raise RuntimeError(f"prefill token total differs: {schedule!r}")
    if (
        schedule["cached_request_ids"]
        or schedule["resumed_request_ids"]
        or schedule["scheduled_spec_decode_request_ids"]
        or schedule["scheduled_encoder_request_ids"]
        or schedule["preempted_request_ids"]
    ):
        raise RuntimeError(f"prefill contains non-prefill scheduler work: {schedule!r}")


def validate_decode(schedule: dict[str, Any], batch: int) -> None:
    scheduled = schedule["num_scheduled_tokens"]
    cached_ids = [str(value) for value in schedule["cached_request_ids"]]
    if (
        schedule["new_request_ids"]
        or len(scheduled) != batch
        or any(value != 1 for value in scheduled.values())
        or schedule["total_num_scheduled_tokens"] != batch
        or len(cached_ids) != batch
        or set(scheduled) != set(cached_ids)
        or schedule["resumed_request_ids"]
        or schedule["scheduled_spec_decode_request_ids"]
        or schedule["scheduled_encoder_request_ids"]
        or schedule["preempted_request_ids"]
    ):
        raise RuntimeError(f"not a strict {batch}-token decode step: {schedule!r}")


def metric_time(metrics: Any, *names: str) -> float | None:
    for name in names:
        value = getattr(metrics, name, None)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def enable_request_metrics_without_default_loggers(llm: Any) -> None:
    """Collect RequestStateStats without enabling periodic engine loggers."""
    engine = getattr(llm, "llm_engine", None)
    output_processor = getattr(engine, "output_processor", None)
    if engine is None or output_processor is None:
        raise RuntimeError("V1 output processor is unavailable")
    if not hasattr(engine, "log_stats") or not hasattr(
        output_processor, "log_stats"
    ):
        raise RuntimeError("V1 request-stat controls are unavailable")

    # LLM(disable_log_stats=True) leaves logger_manager unset, so periodic
    # logging cannot enter a timed step. These switches still make step()
    # create IterationStats and future requests create RequestStateStats.
    engine.log_stats = True
    output_processor.log_stats = True


def output_records(outputs: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for output in outputs:
        completion = output.outputs[0] if output.outputs else None
        token_ids = list(completion.token_ids) if completion is not None else []
        metrics = getattr(output, "metrics", None)
        first = metric_time(metrics, "first_token_ts")
        last = metric_time(metrics, "last_token_ts")
        first_latency = metric_time(metrics, "first_token_latency")
        ttft_ms = (
            1000.0 * first_latency
            if first_latency is not None and first_latency >= 0.0
            else None
        )
        tpot_ms = (
            1000.0 * (last - first) / (len(token_ids) - 1)
            if first is not None
            and last is not None
            and last >= first
            and len(token_ids) > 1
            else None
        )
        request_latency_ms = (
            1000.0 * (first_latency + last - first)
            if first_latency is not None
            and first_latency >= 0.0
            and first is not None
            and last is not None
            and last >= first
            else None
        )
        records.append(
            {
                "request_id": str(output.request_id),
                "finished": bool(output.finished),
                "prompt_tokens": len(output.prompt_token_ids or []),
                "output_token_ids": token_ids,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "request_latency_ms": request_latency_ms,
            }
        )
    return sorted(
        records,
        key=lambda item: (
            (0, int(str(item["request_id"])))
            if str(item["request_id"]).isdigit()
            else (1, str(item["request_id"]))
        ),
    )


def request_timing_summaries(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    fields = {
        "request_ttft_ms": "ttft_ms",
        "request_tpot_ms": "tpot_ms",
        "request_latency_ms": "request_latency_ms",
    }
    result: dict[str, dict[str, Any]] = {}
    for output_key, record_key in fields.items():
        values = [record.get(record_key) for record in records]
        if any(
            value is None or not math.isfinite(float(value)) or float(value) <= 0.0
            for value in values
        ):
            raise RuntimeError(
                "standard request timing metrics are absent; RequestStateStats "
                "must be enabled"
            )
        result[output_key] = summarize([float(value) for value in values])
    return result


def output_hash(records: list[dict[str, Any]]) -> str:
    payload = [record["output_token_ids"] for record in records]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile-label", required=True)
    parser.add_argument("--mechanism-class", required=True)
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--conditioning-repeats", type=int, default=1)
    parser.add_argument("--measurement-repeats", type=int, default=3)
    parser.add_argument("--warmup-decode-steps", type=int, default=5)
    parser.add_argument("--measured-decode-steps", type=int, default=16)
    parser.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cudagraph-mode",
        choices=("NONE", "FULL_SEPARATE"),
        default="NONE",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batches = parse_positive_ints(args.batch_sizes)
    if args.conditioning_repeats < 0 or args.measurement_repeats <= 0:
        raise ValueError("invalid repeat counts")
    if args.warmup_decode_steps < 0 or args.measured_decode_steps <= 0:
        raise ValueError("invalid decode step counts")
    if max(batches) > args.max_num_seqs:
        raise ValueError("batch size exceeds max_num_seqs")
    if not math.isfinite(args.cpu_offload_gb) or args.cpu_offload_gb < 0:
        raise ValueError("cpu_offload_gb must be finite and non-negative")
    fixture_metadata, requests = load_requests(args.fixture)
    if max(batches) > len(requests):
        raise ValueError("batch size exceeds fixture request count")
    total_output_tokens = 1 + args.warmup_decode_steps + args.measured_decode_steps
    if any(
        request["prompt_tokens"] + total_output_tokens > args.max_model_len
        for request in requests[: max(batches)]
    ):
        raise ValueError("prompt plus generated tokens exceeds max_model_len")
    if any(
        sum(item["prompt_tokens"] for item in requests[:batch])
        > args.max_num_batched_tokens
        for batch in batches
    ):
        raise ValueError("a prefill batch exceeds max_num_batched_tokens")

    # CUDA-capable imports deliberately follow all static input validation.
    import torch
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt

    args.output.parent.mkdir(parents=True, exist_ok=True)
    use_cudagraph = args.cudagraph_mode != "NONE"
    capture_sizes = sorted({
        *batches,
        *(
            sum(item["prompt_tokens"] for item in requests[:batch])
            for batch in batches
        ),
    })
    compilation_config = (
        {
            "mode": "NONE",
            "cudagraph_mode": args.cudagraph_mode,
            "cudagraph_capture_sizes": capture_sizes,
            "max_cudagraph_capture_size": max(capture_sizes),
        }
        if use_cudagraph
        else {"mode": "NONE", "cudagraph_mode": "NONE"}
    )
    load_started = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not use_cudagraph,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        cpu_offload_gb=args.cpu_offload_gb,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        seed=args.seed,
        disable_log_stats=True,
        compilation_config=compilation_config,
    )
    enable_request_metrics_without_default_loggers(llm)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats()

    result: dict[str, Any] = {
        "schema": "vllm_p3_pinned_uva_phase_steps_v1",
        "status": "running",
        "profile_label": args.profile_label,
        "mechanism_class": args.mechanism_class,
        "config": {
            **vars(args),
            "fixture": str(args.fixture.resolve()),
            "output": str(args.output.resolve()),
            "batch_sizes": batches,
            "total_output_tokens": total_output_tokens,
            "tensor_parallel_size": 1,
            "enforce_eager": not use_cudagraph,
            "compilation_config": compilation_config,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "engine_core_multiprocessing": False,
            "temperature": 0.0,
            "ignore_eos": True,
            "frozen_prompt_token_ids": True,
            "disable_log_stats": True,
            "request_metrics_without_default_loggers": True,
        },
        "versions": {
            "vllm": getattr(vllm, "__version__", None),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "fixture": {
            "sha256": sha256_file(args.fixture),
            "metadata": fixture_metadata,
        },
        "load_seconds": load_seconds,
        "cycles": [],
    }

    try:
        client = llm.llm_engine.engine_core
        core = getattr(client, "engine_core", None)
        if core is None:
            raise RuntimeError(
                "strict phase benchmark requires VLLM_ENABLE_V1_MULTIPROCESSING=0"
            )
        direct_repeats = int(os.environ.get("E3_DIRECT_PREFILL_GRAPH_REPEATS", "0"))
        if direct_repeats < 0:
            raise ValueError("E3_DIRECT_PREFILL_GRAPH_REPEATS must be non-negative")
        if direct_repeats:
            executor = core.model_executor
            worker = executor.driver_worker.worker
            if worker is None or worker.model_runner is None:
                raise RuntimeError("model runner is unavailable for direct Graph replay")
            graph_entries = getattr(
                worker.model_runner.model, "concrete_cudagraph_entries", {}
            )
            direct_results: dict[str, Any] = {}
            for batch in batches:
                num_tokens = sum(
                    item["prompt_tokens"] for item in requests[:batch]
                )
                matching_entries = [
                    entry
                    for descriptor, entry in graph_entries.items()
                    if int(descriptor.num_tokens) == num_tokens
                    and not bool(descriptor.uniform_decode)
                    and entry.cudagraph is not None
                ]
                if len(matching_entries) != 1:
                    raise RuntimeError(
                        "direct Prefill Graph benchmark did not find one exact entry"
                    )
                samples: list[float] = []
                for _ in range(direct_repeats):
                    torch.cuda.synchronize()
                    direct_started = time.perf_counter()
                    matching_entries[0].cudagraph.replay()
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter() - direct_started) * 1000.0)
                direct_results[str(batch)] = {
                    "num_tokens": num_tokens,
                    "samples_ms": samples,
                    "steady_median_ms": (
                        statistics.median(samples[1:])
                        if len(samples) > 1
                        else samples[0]
                    ),
                }
            result["direct_prefill_graph_replay"] = direct_results
        scheduler = core.scheduler
        original_schedule = scheduler.schedule
        schedule_records: list[dict[str, Any]] = []

        def traced_schedule():
            schedule_output = original_schedule()
            schedule_records.append(schedule_snapshot(schedule_output))
            return schedule_output

        scheduler.schedule = traced_schedule

        sampling = SamplingParams(
            max_tokens=total_output_tokens,
            min_tokens=total_output_tokens,
            temperature=0.0,
            seed=args.seed,
            ignore_eos=True,
        )

        def run_step() -> tuple[float, list[Any], dict[str, Any]]:
            before = len(schedule_records)
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = llm.llm_engine.step()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if len(schedule_records) != before + 1:
                raise RuntimeError("expected exactly one scheduler call per engine step")
            return elapsed_ms, outputs, schedule_records[-1]

        repeat_roles = ["conditioning"] * args.conditioning_repeats + [
            "measurement"
        ] * args.measurement_repeats
        profile_prefill = os.environ.get("E3_NSYS_PREFILL_CAPTURE", "0") == "1"
        profiled_prefill = False
        for batch in batches:
            role_indices = {"conditioning": 0, "measurement": 0}
            selected = requests[:batch]
            prompt_inputs = [
                TokensPrompt(prompt_token_ids=list(item["prompt_token_ids"]))
                for item in selected
            ]
            expected_prompt_tokens = sum(item["prompt_tokens"] for item in selected)
            for role in repeat_roles:
                role_index = role_indices[role]
                role_indices[role] += 1
                cycle_started_epoch = time.time()
                cycle_started_perf = time.perf_counter()
                llm._validate_and_add_requests(
                    prompts=prompt_inputs,
                    params=sampling,
                    use_tqdm=False,
                    lora_request=None,
                    priority=None,
                )

                profile_this_prefill = (
                    profile_prefill and role == "measurement" and not profiled_prefill
                )
                if profile_this_prefill:
                    status = int(torch.cuda.cudart().cudaProfilerStart())
                    if status != 0:
                        raise RuntimeError(f"cudaProfilerStart failed: {status}")
                try:
                    prefill_ms, prefill_outputs, prefill_schedule = run_step()
                finally:
                    if profile_this_prefill:
                        status = int(torch.cuda.cudart().cudaProfilerStop())
                        profiled_prefill = True
                        if status != 0:
                            raise RuntimeError(f"cudaProfilerStop failed: {status}")
                validate_prefill(prefill_schedule, batch)
                if prefill_outputs:
                    raise RuntimeError("FINAL_ONLY prefill returned request outputs")
                if (
                    prefill_schedule["total_num_scheduled_tokens"]
                    != expected_prompt_tokens
                ):
                    raise RuntimeError("prefill used different frozen prompt tokens")

                warmup_steps: list[dict[str, Any]] = []
                for step_index in range(args.warmup_decode_steps):
                    elapsed_ms, outputs, schedule = run_step()
                    validate_decode(schedule, batch)
                    if outputs:
                        raise RuntimeError("decode warmup finished requests early")
                    warmup_steps.append(
                        {
                            "step": step_index + 1,
                            "elapsed_ms": elapsed_ms,
                            "schedule": schedule,
                        }
                    )

                measured_steps: list[dict[str, Any]] = []
                final_outputs: list[Any] = []
                for step_index in range(args.measured_decode_steps):
                    elapsed_ms, outputs, schedule = run_step()
                    validate_decode(schedule, batch)
                    last = step_index + 1 == args.measured_decode_steps
                    if outputs and not last:
                        raise RuntimeError("decode requests finished before final step")
                    if last:
                        final_outputs = outputs
                    measured_steps.append(
                        {
                            "step": step_index + 1,
                            "elapsed_ms": elapsed_ms,
                            "schedule": schedule,
                        }
                    )

                # End the E2E timing at synchronized GPU completion.  Output
                # validation and hashing below are deliberately excluded.
                cycle_elapsed_ms = (
                    time.perf_counter() - cycle_started_perf
                ) * 1000.0

                if llm.llm_engine.has_unfinished_requests():
                    raise RuntimeError("requests remain after a completed cycle")
                records = output_records(final_outputs)
                if len(records) != batch or not all(item["finished"] for item in records):
                    raise RuntimeError("final output set is incomplete")
                if any(
                    len(item["output_token_ids"]) != total_output_tokens
                    for item in records
                ):
                    raise RuntimeError("generated token count differs")
                if [item["prompt_tokens"] for item in records] != [
                    item["prompt_tokens"] for item in selected
                ]:
                    raise RuntimeError("output prompt lengths differ from fixture")
                request_timings = request_timing_summaries(records)
                step_samples = [float(item["elapsed_ms"]) for item in measured_steps]
                result["cycles"].append(
                    {
                        "role": role,
                        "role_index": role_index,
                        "batch_size": batch,
                        "prompt_tokens": expected_prompt_tokens,
                        "cycle_started_epoch": cycle_started_epoch,
                        "cycle_ended_epoch": time.time(),
                        "cycle_elapsed_ms": cycle_elapsed_ms,
                        "prefill_elapsed_ms": prefill_ms,
                        "prefill_schedule": prefill_schedule,
                        "warmup_decode_steps": warmup_steps,
                        "measured_decode_steps": measured_steps,
                        "decode_step_ms": summarize(step_samples),
                        "decode_tokens_per_second": (
                            batch * 1000.0 / statistics.median(step_samples)
                        ),
                        **request_timings,
                        "output_token_ids_sha256": output_hash(records),
                        "outputs": records,
                    }
                )

        measured = [item for item in result["cycles"] if item["role"] == "measurement"]
        phase_summary: dict[str, Any] = {}
        for batch in batches:
            selected_cycles = [item for item in measured if item["batch_size"] == batch]
            prefill_samples = [float(item["prefill_elapsed_ms"]) for item in selected_cycles]
            decode_repeat_medians = [
                float(item["decode_step_ms"]["median"]) for item in selected_cycles
            ]
            decode_steps = [
                float(step["elapsed_ms"])
                for item in selected_cycles
                for step in item["measured_decode_steps"]
            ]
            cycle_samples = [float(item["cycle_elapsed_ms"]) for item in selected_cycles]
            request_ttft = [
                float(item["request_ttft_ms"]["median"])
                for item in selected_cycles
            ]
            request_tpot = [
                float(item["request_tpot_ms"]["median"])
                for item in selected_cycles
            ]
            request_latency = [
                float(item["request_latency_ms"]["median"])
                for item in selected_cycles
            ]
            prompt_tokens = sum(item["prompt_tokens"] for item in requests[:batch])
            phase_summary[str(batch)] = {
                "batch_size": batch,
                "prompt_tokens": prompt_tokens,
                "measurement_repeats": len(selected_cycles),
                "prefill_step_ms": summarize(prefill_samples),
                "prefill_tokens_per_second": (
                    prompt_tokens * 1000.0 / statistics.median(prefill_samples)
                ),
                "decode_repeat_median_ms": summarize(decode_repeat_medians),
                "decode_all_step_ms": summarize(decode_steps),
                "decode_tokens_per_second": (
                    batch * 1000.0 / statistics.median(decode_repeat_medians)
                ),
                "cycle_e2e_ms": summarize(cycle_samples),
                "request_ttft_p50_ms": summarize(request_ttft),
                "request_tpot_p50_ms": summarize(request_tpot),
                "request_latency_p50_ms": summarize(request_latency),
            }
        result["phase_summary"] = phase_summary
        result["status"] = "ok"
        result["final_torch"] = {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        result["environment"] = {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith(("VLLM_UVM_", "VLLM_CPU_OFFLOAD_"))
            or key
            in {
                "VLLM_USE_UVM",
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "CUDA_VISIBLE_DEVICES",
            }
        }
    except Exception as exc:
        result["status"] = "exception"
        result["error"] = repr(exc)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise

    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "phase_summary": result["phase_summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
