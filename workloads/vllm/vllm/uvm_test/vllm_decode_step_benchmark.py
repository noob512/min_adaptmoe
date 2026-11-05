#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_ints(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"invalid positive integer list: {raw!r}")
    return values


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def gpu_snapshot() -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        parts = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
        return {
            "name": parts[0],
            "used_mib": int(parts[1]),
            "free_mib": int(parts[2]),
            "total_mib": int(parts[3]),
        }
    except Exception:
        return None


def load_requests(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    requests = value.get("requests") if isinstance(value, dict) else value
    if not isinstance(requests, list) or not requests:
        raise ValueError(f"fixture contains no requests: {path}")
    normalized = []
    for index, item in enumerate(requests):
        if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
            raise ValueError(f"invalid request {index}: {item!r}")
        normalized.append(
            {
                "request": int(item.get("request", index)),
                "prompt": item["prompt"],
                "fixture_prompt_tokens": int(item.get("prompt_tokens", -1)),
            }
        )
    return normalized


def schedule_snapshot(output: Any) -> dict[str, Any]:
    new_reqs = list(output.scheduled_new_reqs)
    cached = output.scheduled_cached_reqs
    return {
        "total_num_scheduled_tokens": int(output.total_num_scheduled_tokens),
        "num_scheduled_tokens": {
            str(key): int(value) for key, value in output.num_scheduled_tokens.items()
        },
        "new_request_ids": [str(request.req_id) for request in new_reqs],
        "new_request_prompt_tokens": {
            str(request.req_id): len(request.prompt_token_ids or [])
            for request in new_reqs
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


def output_records(outputs: list[Any]) -> list[dict[str, Any]]:
    records = []
    for output in outputs:
        completion = output.outputs[0] if output.outputs else None
        records.append(
            {
                "request_id": str(output.request_id),
                "finished": bool(output.finished),
                "prompt_tokens": len(output.prompt_token_ids or []),
                "output_token_ids": list(completion.token_ids)
                if completion is not None
                else [],
            }
        )
    return sorted(
        records,
        key=lambda record: (
            (0, int(record["request_id"]))
            if str(record["request_id"]).isdigit()
            else (1, str(record["request_id"]))
        ),
    )


def output_hash(records: list[dict[str, Any]]) -> str:
    payload = [record["output_token_ids"] for record in records]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()


def validate_prefill(schedule: dict[str, Any], batch: int) -> None:
    if len(schedule["new_request_ids"]) != batch:
        raise RuntimeError(f"prefill did not admit {batch} requests: {schedule!r}")
    prompt_tokens = schedule["new_request_prompt_tokens"]
    scheduled = schedule["num_scheduled_tokens"]
    if scheduled != prompt_tokens:
        raise RuntimeError(
            f"prefill was not scheduled in one complete step: {schedule!r}"
        )
    if schedule["total_num_scheduled_tokens"] != sum(prompt_tokens.values()):
        raise RuntimeError(f"prefill total mismatch: {schedule!r}")
    if (
        schedule["cached_request_ids"]
        or schedule["resumed_request_ids"]
        or schedule["scheduled_spec_decode_request_ids"]
        or schedule["scheduled_encoder_request_ids"]
        or schedule["preempted_request_ids"]
    ):
        raise RuntimeError(f"prefill contains unexpected scheduler state: {schedule!r}")


def validate_decode(schedule: dict[str, Any], batch: int) -> None:
    scheduled = schedule["num_scheduled_tokens"]
    if (
        schedule["new_request_ids"]
        or len(scheduled) != batch
        or any(value != 1 for value in scheduled.values())
        or schedule["total_num_scheduled_tokens"] != batch
        or len(schedule["cached_request_ids"]) != batch
        or schedule["resumed_request_ids"]
        or schedule["scheduled_spec_decode_request_ids"]
        or schedule["scheduled_encoder_request_ids"]
        or schedule["preempted_request_ids"]
    ):
        raise RuntimeError(f"not a strict {batch}-token decode step: {schedule!r}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure strict in-process vLLM decode steps."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64")
    parser.add_argument("--warmup-decode-steps", type=int, default=5)
    parser.add_argument("--measured-decode-steps", type=int, default=16)
    parser.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    args = parser.parse_args()

    batches = parse_ints(args.batch_sizes)
    if max(batches) > args.max_num_seqs:
        raise ValueError("batch size exceeds max_num_seqs")
    if args.warmup_decode_steps < 0 or args.measured_decode_steps <= 0:
        raise ValueError("invalid decode step counts")
    requests = load_requests(args.fixture)
    if max(batches) > len(requests):
        raise ValueError("batch size exceeds fixture request count")

    max_tokens = 1 + args.warmup_decode_steps + args.measured_decode_steps
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    args.progress.write_text("", encoding="utf-8")

    def progress(record: dict[str, Any]) -> None:
        with args.progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    from vllm import LLM, SamplingParams

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
        cpu_offload_gb=args.cpu_offload_gb,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        seed=0,
        disable_log_stats=True,
    )
    result: dict[str, Any] = {
        "status": "running",
        "config": vars(args)
        | {
            "fixture": str(args.fixture),
            "output": str(args.output),
            "progress": str(args.progress),
            "max_tokens_per_request": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "engine_core_multiprocessing": False,
        },
        "load_seconds": time.perf_counter() - load_started,
        "gpu_after_load": gpu_snapshot(),
        "phases": [],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    try:
        import torch

        client = llm.llm_engine.engine_core
        core = getattr(client, "engine_core", None)
        if core is None:
            raise RuntimeError(
                "strict benchmark requires VLLM_ENABLE_V1_MULTIPROCESSING=0"
            )
        scheduler = core.scheduler
        original_schedule = scheduler.schedule
        schedule_records: list[dict[str, Any]] = []

        def traced_schedule():
            schedule_output = original_schedule()
            schedule_records.append(schedule_snapshot(schedule_output))
            return schedule_output

        scheduler.schedule = traced_schedule

        for batch in batches:
            prompts = [request["prompt"] for request in requests[:batch]]
            sampling = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                ignore_eos=True,
            )
            phase_name = f"decode_b{batch}"
            progress(
                {
                    "action": "phase_begin",
                    "phase": phase_name,
                    "batch_tokens": batch,
                    "timestamp": time.time(),
                }
            )
            llm._validate_and_add_requests(
                prompts=prompts,
                params=sampling,
                use_tqdm=False,
                lora_request=None,
                priority=None,
            )

            def run_step() -> tuple[float, list[Any], dict[str, Any]]:
                before_records = len(schedule_records)
                torch.cuda.synchronize()
                started = time.perf_counter()
                outputs = llm.llm_engine.step()
                torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if len(schedule_records) != before_records + 1:
                    raise RuntimeError(
                        "expected exactly one scheduler invocation per engine step"
                    )
                return elapsed_ms, outputs, schedule_records[-1]

            prefill_ms, prefill_outputs, prefill_schedule = run_step()
            validate_prefill(prefill_schedule, batch)
            if prefill_outputs:
                raise RuntimeError("FINAL_ONLY prefill unexpectedly returned outputs")

            warmup_steps = []
            for step_index in range(args.warmup_decode_steps):
                elapsed_ms, outputs, schedule = run_step()
                validate_decode(schedule, batch)
                if outputs:
                    raise RuntimeError("warmup decode unexpectedly finished requests")
                warmup_steps.append(
                    {
                        "step": step_index + 1,
                        "elapsed_ms": elapsed_ms,
                        "schedule": schedule,
                    }
                )

            measured_steps = []
            final_outputs: list[Any] = []
            for step_index in range(args.measured_decode_steps):
                elapsed_ms, outputs, schedule = run_step()
                validate_decode(schedule, batch)
                is_last = step_index + 1 == args.measured_decode_steps
                if outputs and not is_last:
                    raise RuntimeError("requests finished before final measured step")
                if is_last:
                    final_outputs = outputs
                measured_steps.append(
                    {
                        "step": step_index + 1,
                        "elapsed_ms": elapsed_ms,
                        "schedule": schedule,
                    }
                )

            if llm.llm_engine.has_unfinished_requests():
                raise RuntimeError("requests remain after final measured step")
            records = output_records(final_outputs)
            if len(records) != batch or not all(record["finished"] for record in records):
                raise RuntimeError(f"invalid final outputs: {records!r}")
            if any(len(record["output_token_ids"]) != max_tokens for record in records):
                raise RuntimeError(f"invalid output lengths: {records!r}")

            step_ms = [float(step["elapsed_ms"]) for step in measured_steps]
            phase = {
                "phase": phase_name,
                "batch_tokens": batch,
                "request_count": batch,
                "prefill_elapsed_ms": prefill_ms,
                "prefill_schedule": prefill_schedule,
                "actual_prompt_tokens": prefill_schedule[
                    "total_num_scheduled_tokens"
                ],
                "warmup_decode_steps": warmup_steps,
                "measured_decode_steps": measured_steps,
                "decode_step_count": len(step_ms),
                "decode_step_mean_ms": statistics.mean(step_ms),
                "decode_step_median_ms": statistics.median(step_ms),
                "decode_step_p90_ms": percentile(step_ms, 90.0),
                "decode_step_min_ms": min(step_ms),
                "decode_step_max_ms": max(step_ms),
                "steady_decode_tokens_per_second": (
                    batch * 1000.0 / statistics.median(step_ms)
                ),
                "output_token_hash": output_hash(records),
                "outputs": records,
                "gpu_after_phase": gpu_snapshot(),
            }
            result["phases"].append(phase)
            progress(
                {
                    "action": "phase_end",
                    "phase": phase_name,
                    "batch_tokens": batch,
                    "decode_step_median_ms": phase["decode_step_median_ms"],
                    "timestamp": time.time(),
                }
            )
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    except Exception as exc:
        result["status"] = "exception"
        result["error"] = repr(exc)
        result["gpu_after_error"] = gpu_snapshot()
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise

    result["status"] = "ok"
    result["gpu_after_finish"] = gpu_snapshot()
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
