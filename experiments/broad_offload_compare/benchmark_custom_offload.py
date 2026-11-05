#!/usr/bin/env python3
"""Run one isolated custom-vLLM measurement phase.

This client intentionally has one measured phase.  Statistical repetition and
ABBA ordering are process-level responsibilities of ``run_custom_comparison.sh``.
Imports that may initialize CUDA occur only after argparse has accepted the
command, so ``--help`` is safe for static preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def nvml_snapshot() -> dict[str, Any] | None:
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.used,memory.free,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        parts = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
        return {
            "name": parts[0],
            "uuid": parts[1],
            "used_mib": int(parts[2]),
            "free_mib": int(parts[3]),
            "total_mib": int(parts[4]),
            "utilization_percent": int(parts[5]),
            "power_watts": float(parts[6]),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def process_memory_snapshot() -> dict[str, Any]:
    status: dict[str, str] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                if key in {"VmRSS", "VmHWM", "VmSize", "RssAnon", "RssFile"}:
                    status[key] = value.strip()
    except OSError:
        pass
    return {
        "proc_status": status,
        "ru_maxrss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }


def load_fixture(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("requests"), list):
        raise ValueError(f"{path}: expected object with requests list")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: metadata is missing")
    requests: list[dict[str, Any]] = []
    for index, raw in enumerate(payload["requests"]):
        if not isinstance(raw, dict) or not isinstance(raw.get("prompt"), str):
            raise ValueError(f"{path}: request {index} is invalid")
        prompt_tokens = raw.get("prompt_tokens")
        if isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int):
            raise ValueError(f"{path}: request {index} lacks integer prompt_tokens")
        prompt_token_ids = raw.get("prompt_token_ids")
        if prompt_token_ids is not None:
            if not isinstance(prompt_token_ids, list) or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in prompt_token_ids
            ):
                raise ValueError(f"{path}: request {index} has invalid prompt_token_ids")
            if len(prompt_token_ids) != prompt_tokens:
                raise ValueError(
                    f"{path}: request {index} prompt_token_ids length does not "
                    "match prompt_tokens"
                )
        requests.append(
            {
                "request": int(raw.get("request", index)),
                "prompt": raw["prompt"],
                "prompt_tokens": prompt_tokens,
                "prompt_token_ids": prompt_token_ids,
            }
        )
    if not requests:
        raise ValueError(f"{path}: fixture is empty")
    return metadata, requests


def source_indices(metadata: dict[str, Any]) -> set[int] | None:
    values = metadata.get("source_indices")
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        return None
    return set(values)


def metric_time(metrics: Any, *names: str) -> float | None:
    for name in names:
        value = getattr(metrics, name, None)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def request_record(output: Any, expected: dict[str, Any]) -> dict[str, Any]:
    if not output.outputs:
        raise RuntimeError(f"request {output.request_id} has no completion")
    completion = output.outputs[0]
    token_ids = [int(value) for value in completion.token_ids]
    prompt_ids = [int(value) for value in (output.prompt_token_ids or [])]
    metrics = getattr(output, "metrics", None)
    arrival = metric_time(metrics, "arrival_time")
    first = metric_time(metrics, "first_token_time", "first_token_ts")
    last = metric_time(metrics, "last_token_time", "last_token_ts", "finished_time")
    finished = metric_time(metrics, "finished_time")
    first_token_latency = metric_time(metrics, "first_token_latency")
    if first_token_latency is not None and first_token_latency >= 0:
        ttft_ms = 1000.0 * first_token_latency
    else:
        ttft_ms = (
            1000.0 * (first - arrival)
            if arrival is not None and first is not None and first >= arrival
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
    if (
        first_token_latency is not None
        and first_token_latency >= 0
        and first is not None
        and last is not None
        and last >= first
    ):
        latency_ms = 1000.0 * (first_token_latency + last - first)
    else:
        latency_ms = (
            1000.0 * (finished - arrival)
            if arrival is not None and finished is not None and finished >= arrival
            else None
        )
    queued = metric_time(metrics, "queued_ts")
    scheduled = metric_time(metrics, "scheduled_ts")
    queue_ms = (
        1000.0 * (scheduled - queued)
        if queued is not None and scheduled is not None and scheduled >= queued
        else None
    )
    if queue_ms is None and isinstance(
        getattr(metrics, "time_in_queue", None), (int, float)
    ):
        queue_ms = 1000.0 * float(getattr(metrics, "time_in_queue"))
    return {
        "request_id": str(output.request_id),
        "fixture_request": int(expected["request"]),
        "expected_prompt_tokens": int(expected["prompt_tokens"]),
        "expected_prompt_token_ids_sha256": (
            sha256_json(expected["prompt_token_ids"])
            if expected.get("prompt_token_ids") is not None
            else None
        ),
        "prompt_tokens": len(prompt_ids),
        "prompt_token_ids_sha256": sha256_json(prompt_ids),
        "prompt_token_ids_exact": (
            prompt_ids == expected["prompt_token_ids"]
            if expected.get("prompt_token_ids") is not None
            else None
        ),
        "output_tokens": len(token_ids),
        "output_token_ids": token_ids,
        "output_token_ids_sha256": sha256_json(token_ids),
        "decoded_text": str(getattr(completion, "text", "")),
        "finish_reason": getattr(completion, "finish_reason", None),
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "request_latency_ms": latency_ms,
        "queue_ms": queue_ms,
    }


def install_slot_preallocate_control_capture(
    pool_module: Any,
) -> tuple[Path | None, list[dict[str, Any]]]:
    """Capture only mode-8 control fields without enabling the full trace.

    The wrapper keeps its single result in memory while the measured request is
    running.  Serialization happens after measurement, so trace-off latency is
    not charged for JSON/file I/O or large per-region trace construction.
    """

    target_text = os.environ.get(
        "VLLM_UVM_SLOT_DIRECT_CONTROL_SUMMARY_FILE", ""
    ).strip()
    captures: list[dict[str, Any]] = []
    if not target_text:
        return None, captures

    original = pool_module.driver_slot_preallocate_ranges

    def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        captures.append(
            {
                "ok": result.get("ok"),
                "reason": result.get("reason"),
                "mode": result.get("mode"),
                "ioctl_calls": result.get("ioctl_calls"),
                "fd_candidate_attempts": result.get("fd_candidate_attempts"),
                "fd_candidate_ioctl_calls": result.get(
                    "fd_candidate_ioctl_calls"
                ),
                "requested_count": result.get("requested_count"),
                "requested_bytes": result.get("requested_bytes"),
                "success_count": result.get("success_count"),
                "accepted_bytes": result.get("accepted_bytes"),
            }
        )
        return result

    pool_module.driver_slot_preallocate_ranges = capture
    return Path(target_text), captures


def write_slot_preallocate_control_summary(
    target: Path | None,
    captures: list[dict[str, Any]],
    *,
    measurement_start_epoch: float,
    measurement_end_epoch: float,
) -> None:
    if target is None:
        return
    payload = {
        "schema": "vllm_slot_preallocate_control_v1",
        "measurement_start_epoch": measurement_start_epoch,
        "measurement_end_epoch": measurement_end_epoch,
        "preallocate_calls": len(captures),
        "logical_ioctl_calls": sum(int(item["ioctl_calls"]) for item in captures),
        "fd_candidate_attempts": sum(
            int(item["fd_candidate_attempts"]) for item in captures
        ),
        "fd_candidate_ioctl_calls": sum(
            int(item["fd_candidate_ioctl_calls"]) for item in captures
        ),
        "results": captures,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-process, one-phase benchmark for custom UVM offload."
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
    parser.add_argument(
        "--disable-chunked-prefill",
        action="store_true",
        help=(
            "Construct the engine with enable_chunked_prefill=False. This is "
            "required by phase-boundary experiments which must observe one "
            "complete prefill before the first decode step."
        ),
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--require-prompt-token-ids",
        action="store_true",
        help="Reject text-only fixtures; required for formal cross-framework runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.concurrency <= 0 or args.output_tokens <= 0:
        raise SystemExit("concurrency and output tokens must be positive")
    if not math.isfinite(args.cpu_offload_gb) or args.cpu_offload_gb < 0.0:
        raise SystemExit("cpu-offload-gb must be finite and non-negative")
    if args.warmup_requests < 0 or args.warmup_output_tokens <= 0:
        raise SystemExit("invalid warmup settings")
    if args.max_num_seqs < args.concurrency:
        raise SystemExit("max_num_seqs is smaller than concurrency")
    if args.warmup_requests and args.warmup_fixture is None:
        raise SystemExit("warmup requests require --warmup-fixture")

    eval_meta, evaluation = load_fixture(args.evaluation_fixture)
    if len(evaluation) < args.concurrency:
        raise SystemExit("evaluation fixture is smaller than concurrency")
    if any(
        item["prompt_tokens"] + args.output_tokens > args.max_model_len
        for item in evaluation[: args.concurrency]
    ):
        raise SystemExit("evaluation prompt+output exceeds max_model_len")
    if args.require_prompt_token_ids and any(
        item.get("prompt_token_ids") is None
        for item in evaluation[: args.concurrency]
    ):
        raise SystemExit("formal mode requires prompt_token_ids in evaluation fixture")
    warm_meta: dict[str, Any] | None = None
    warmup: list[dict[str, Any]] = []
    if args.warmup_requests:
        assert args.warmup_fixture is not None
        warm_meta, warmup = load_fixture(args.warmup_fixture)
        if len(warmup) < args.warmup_requests:
            raise SystemExit("warmup fixture is smaller than warmup request count")
        if any(
            item["prompt_tokens"] + args.warmup_output_tokens > args.max_model_len
            for item in warmup[: args.warmup_requests]
        ):
            raise SystemExit("warmup prompt+output exceeds max_model_len")
        if args.require_prompt_token_ids and any(
            item.get("prompt_token_ids") is None
            for item in warmup[: args.warmup_requests]
        ):
            raise SystemExit("formal mode requires prompt_token_ids in warmup fixture")
        eval_indices = source_indices(eval_meta)
        warm_indices = source_indices(warm_meta)
        if eval_indices is None or warm_indices is None:
            raise SystemExit("fixtures must expose metadata.source_indices")
        overlap = sorted(eval_indices & warm_indices)
        if overlap:
            raise SystemExit(f"warmup/evaluation fixture overlap: {overlap[:16]}")

    # Potentially CUDA-initializing imports are deliberately below validation.
    import torch
    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt

    slot_control_target: Path | None = None
    slot_control_captures: list[dict[str, Any]] = []
    if os.environ.get("VLLM_UVM_SLOT_DIRECT_CONTROL_SUMMARY_FILE", "").strip():
        from vllm.device_allocator import uvm_pool_residency

        slot_control_target, slot_control_captures = (
            install_slot_preallocate_control_capture(uvm_pool_residency)
        )

    def engine_inputs(records: list[dict[str, Any]]) -> list[str | TokensPrompt]:
        have_ids = [item.get("prompt_token_ids") is not None for item in records]
        if any(have_ids) and not all(have_ids):
            raise RuntimeError("a phase cannot mix text prompts and token-ID prompts")
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
    ready_nvml = nvml_snapshot()
    ready_torch = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }

    sampling_warmup = SamplingParams(
        temperature=0.0,
        seed=args.seed,
        ignore_eos=True,
        min_tokens=args.warmup_output_tokens,
        max_tokens=args.warmup_output_tokens,
    )
    warmup_seconds = 0.0
    if args.warmup_requests:
        warm_started = time.perf_counter()
        warm_outputs = llm.generate(
            engine_inputs(warmup[: args.warmup_requests]),
            sampling_warmup,
            use_tqdm=False,
        )
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - warm_started
        if len(warm_outputs) != args.warmup_requests or any(
            not output.outputs
            or len(list(output.outputs[0].token_ids)) != args.warmup_output_tokens
            for output in warm_outputs
        ):
            raise RuntimeError("warmup did not produce the exact requested token count")
    post_warmup_nvml = nvml_snapshot()
    post_warmup_torch = {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }

    measured_requests = evaluation[: args.concurrency]
    sampling = SamplingParams(
        temperature=0.0,
        seed=args.seed,
        ignore_eos=True,
        min_tokens=args.output_tokens,
        max_tokens=args.output_tokens,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    measurement_start_epoch = time.time()
    measurement_started = time.perf_counter()
    outputs = llm.generate(
        engine_inputs(measured_requests), sampling, use_tqdm=False
    )
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - measurement_started
    measurement_end_epoch = time.time()
    write_slot_preallocate_control_summary(
        slot_control_target,
        slot_control_captures,
        measurement_start_epoch=measurement_start_epoch,
        measurement_end_epoch=measurement_end_epoch,
    )
    if len(outputs) != args.concurrency:
        raise RuntimeError(f"completed {len(outputs)} of {args.concurrency} requests")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        raise RuntimeError("measurement elapsed time is not finite and positive")

    records = [
        request_record(output, expected)
        for output, expected in zip(outputs, measured_requests)
    ]
    if any(
        record["prompt_tokens"] != record["expected_prompt_tokens"] for record in records
    ):
        mismatches = [
            (record["request_id"], record["expected_prompt_tokens"], record["prompt_tokens"])
            for record in records
            if record["prompt_tokens"] != record["expected_prompt_tokens"]
        ]
        raise RuntimeError(f"prompt token count mismatch: {mismatches[:8]}")
    if any(record["prompt_token_ids_exact"] is False for record in records):
        raise RuntimeError("engine prompt token IDs differ from the frozen fixture")
    if any(record["output_tokens"] != args.output_tokens for record in records):
        raise RuntimeError("measured output token count is not exact")

    ttfts = [float(value) for record in records if (value := record["ttft_ms"]) is not None]
    tpots = [float(value) for record in records if (value := record["tpot_ms"]) is not None]
    latencies = [
        float(value)
        for record in records
        if (value := record["request_latency_ms"]) is not None
    ]
    if len(ttfts) != len(records) or len(latencies) != len(records):
        raise RuntimeError("TTFT/E2E metrics are incomplete")
    if args.output_tokens > 1 and len(tpots) != len(records):
        raise RuntimeError("TPOT metrics are incomplete")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in [*ttfts, *tpots, *latencies]
    ):
        raise RuntimeError("latency metrics must be finite and non-negative")
    output_token_count = sum(int(record["output_tokens"]) for record in records)
    aggregate_ids = [record["output_token_ids"] for record in records]
    result = {
        "schema_version": 1,
        "status": "ok",
        "profile_label": args.profile_label,
        "mechanism_class": args.mechanism_class,
        "started_epoch": started_epoch,
        "measurement_start_epoch": measurement_start_epoch,
        "measurement_end_epoch": measurement_end_epoch,
        "config": {
            **vars(args),
            "evaluation_fixture": str(args.evaluation_fixture.resolve()),
            "warmup_fixture": (
                str(args.warmup_fixture.resolve()) if args.warmup_fixture else None
            ),
            "output": str(args.output.resolve()),
            "tensor_parallel_size": 1,
            "enforce_eager": True,
            "cpu_offload_gb": args.cpu_offload_gb,
            "enable_prefix_caching": False,
            "ignore_eos": True,
            "temperature": 0.0,
        },
        "versions": {
            "vllm": getattr(vllm, "__version__", None),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "fixtures": {
            "evaluation_sha256": sha256_file(args.evaluation_fixture),
            "evaluation_metadata": eval_meta,
            "warmup_sha256": (
                sha256_file(args.warmup_fixture) if args.warmup_fixture else None
            ),
            "warmup_metadata": warm_meta,
            "disjoint_source_indices": bool(args.warmup_requests),
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
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "ready_nvml": ready_nvml,
        "ready_torch": ready_torch,
        "post_warmup_ready_nvml": post_warmup_nvml,
        "post_warmup_ready_torch": post_warmup_torch,
        "measurement": {
            "elapsed_seconds": elapsed_seconds,
            "completed_requests": len(records),
            "prompt_tokens": sum(int(record["prompt_tokens"]) for record in records),
            "output_tokens": output_token_count,
            "output_tokens_per_second": output_token_count / elapsed_seconds,
            "ttft_ms": summarize(ttfts),
            "tpot_ms": summarize(tpots),
            "request_latency_ms": summarize(latencies),
            "aggregate_output_token_ids_sha256": sha256_json(aggregate_ids),
            "requests": records,
        },
        "after_measurement_nvml": nvml_snapshot(),
        "measurement_torch_peak": {
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "process_memory": process_memory_snapshot(),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
