#!/usr/bin/env python3
"""Run E2 cells by admitting each batch before the first engine step.

This runner intentionally bypasses HTTP/SSE.  It preprocesses and inserts all
requests in a cell into the in-process EngineCore before calling ``step()``.
The timing boundary is the end of batch admission, so reported TTFT/TPOT/E2E
measure scheduler and model execution without client or transport overhead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SHORT_DIALOGUE_TTFT_SLO_MS = 2000.0
SHORT_DIALOGUE_TPOT_SLO_MS = 100.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _model_runner_from_llm(llm: Any) -> Any:
    core = llm.llm_engine.engine_core.engine_core
    wrapper = core.model_executor.driver_worker
    worker = getattr(wrapper, "worker", wrapper)
    return worker.model_runner


def _cuda_tensor_ranges(model_runner: Any) -> list[dict[str, Any]]:
    """Collect live named CUDA tensor storage ranges for address attribution."""
    import torch

    records: list[dict[str, Any]] = []
    seen_storages: set[tuple[int, int]] = set()

    def add(name: str, value: Any) -> None:
        if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
            return
        storage = value.untyped_storage()
        base = int(storage.data_ptr())
        size = int(storage.nbytes())
        key = (base, size)
        if base == 0 or size <= 0 or key in seen_storages:
            return
        seen_storages.add(key)
        records.append(
            {
                "name": name,
                "base": base,
                "end": base + size,
                "bytes": size,
                "tensor_data_ptr": int(value.data_ptr()),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        )

    model = model_runner.get_model()
    for name, value in model.named_parameters(recurse=True, remove_duplicate=False):
        add(f"model.parameter.{name}", value)
    for name, value in model.named_buffers(recurse=True, remove_duplicate=False):
        add(f"model.buffer.{name}", value)

    def walk(name: str, value: Any, depth: int) -> None:
        add(name, value)
        if depth <= 0:
            return
        if isinstance(value, dict):
            for key, item in value.items():
                walk(f"{name}[{key!r}]", item, depth - 1)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(f"{name}[{index}]", item, depth - 1)

    # These runner fields cover KV tensors and persistent Graph input buffers
    # without recursively walking the entire Python model/config object graph.
    for attribute, value in vars(model_runner).items():
        lowered = attribute.lower()
        if any(
            token in lowered
            for token in ("kv", "input", "buffer", "workspace", "graph")
        ):
            walk(f"model_runner.{attribute}", value, 3)
    return sorted(records, key=lambda item: (item["base"], item["name"]))


def _graph_entry_addresses(model_runner: Any) -> list[dict[str, Any]]:
    # ``get_model()`` intentionally unwraps CUDAGraphWrapper, which made the
    # original diagnostic report zero entries even when GraphExecs were live.
    # Inspect the runner-owned wrapper and every retained graph class instead.
    model = getattr(model_runner, "model", None)
    class_entries = getattr(model, "_cudagraph_class_entries", None)
    if not isinstance(class_entries, dict):
        entries = getattr(model, "concrete_cudagraph_entries", {})
        class_entries = {"active": entries} if isinstance(entries, dict) else {}

    def tensor_addresses(value: Any) -> list[dict[str, Any]]:
        import torch

        tensors: list[dict[str, Any]] = []

        def walk(item: Any, path: str, depth: int) -> None:
            if isinstance(item, torch.Tensor):
                storage = item.untyped_storage()
                base = int(storage.data_ptr())
                size = int(storage.nbytes())
                tensors.append(
                    {
                        "path": path,
                        "base": base,
                        "end": base + size,
                        "bytes": size,
                        "tensor_data_ptr": int(item.data_ptr()),
                        "shape": list(item.shape),
                        "dtype": str(item.dtype),
                    }
                )
                return
            if depth <= 0:
                return
            if isinstance(item, dict):
                for key, child in item.items():
                    walk(child, f"{path}[{key!r}]", depth - 1)
            elif isinstance(item, (list, tuple)):
                for index, child in enumerate(item):
                    walk(child, f"{path}[{index}]", depth - 1)

        walk(value, "output", 3)
        return tensors

    records: list[dict[str, Any]] = []
    seen_tables: dict[int, str] = {}
    for class_name, entries in sorted(class_entries.items()):
        if not isinstance(entries, dict):
            continue
        shared_with = seen_tables.get(id(entries))
        if shared_with is None:
            seen_tables[id(entries)] = str(class_name)
        for descriptor, entry in entries.items():
            records.append(
                {
                    "class_name": str(class_name),
                    "shared_entry_table_with": shared_with,
                    "active_class": str(
                        getattr(model, "_active_cudagraph_class", "")
                    ),
                    "descriptor": repr(descriptor),
                    "input_addresses": [
                        int(value) for value in (entry.input_addresses or [])
                    ],
                    "output_tensors": tensor_addresses(entry.output),
                    "captured": entry.cudagraph is not None,
                    "cudagraph_id": (
                        id(entry.cudagraph) if entry.cudagraph is not None else None
                    ),
                }
            )
    return records


def _capture_cuda_address_snapshot(
    llm: Any,
    stage: str,
    *,
    synchronize: bool = True,
    include_frames: bool = True,
) -> dict[str, Any]:
    import torch

    if synchronize:
        torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    raw = torch.cuda.memory._snapshot()
    segments: list[dict[str, Any]] = []
    for segment in raw.get("segments", []):
        blocks: list[dict[str, Any]] = []
        for block in segment.get("blocks", []):
            blocks.append(
                {
                    "address": int(block["address"]),
                    "end": int(block["address"]) + int(block["size"]),
                    "size": int(block["size"]),
                    "requested_size": int(block.get("requested_size", 0)),
                    "state": str(block["state"]),
                    "frames": [
                        {
                            "filename": str(frame.get("filename", "")),
                            "line": int(frame.get("line", 0)),
                            "name": str(frame.get("name", "")),
                        }
                        for frame in block.get("frames", [])
                    ] if include_frames else [],
                }
            )
        segments.append(
            {
                "address": int(segment["address"]),
                "end": int(segment["address"]) + int(segment["total_size"]),
                "total_size": int(segment["total_size"]),
                "allocated_size": int(segment["allocated_size"]),
                "active_size": int(segment["active_size"]),
                "stream": int(segment["stream"]),
                "segment_type": str(segment["segment_type"]),
                "segment_pool_id": segment.get("segment_pool_id"),
                "is_expandable": bool(segment.get("is_expandable", False)),
                "blocks": blocks,
            }
        )
    model_runner = _model_runner_from_llm(llm)
    return {
        "stage": stage,
        "synchronized": bool(synchronize),
        "memory_history_enabled": bool(include_frames),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "memory_allocated": int(torch.cuda.memory_allocated()),
        "memory_reserved": int(torch.cuda.memory_reserved()),
        "segments": segments,
        "named_tensor_ranges": _cuda_tensor_ranges(model_runner),
        "graph_entries": _graph_entry_addresses(model_runner),
    }


@dataclass(frozen=True)
class FixtureRequest:
    fixture_index: int
    fixture_request_id: Any
    prompt_token_ids: list[int]
    prompt_token_ids_sha256: str


def load_fixture(path: Path) -> tuple[dict[str, Any], list[FixtureRequest]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("metadata"), dict):
        raise ValueError("fixture must contain object-valued metadata")
    records = value.get("requests")
    if not isinstance(records, list) or not records:
        raise ValueError("fixture requests must be a non-empty list")
    requests: list[FixtureRequest] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"fixture request {index} is not an object")
        token_ids = record.get("prompt_token_ids")
        if not isinstance(token_ids, list) or not token_ids or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in token_ids
        ):
            raise ValueError(f"fixture request {index} has invalid tokens")
        if record.get("prompt_tokens") != len(token_ids):
            raise ValueError(f"fixture request {index} length metadata differs")
        digest = hashlib.sha256(
            ",".join(str(token) for token in token_ids).encode("ascii")
        ).hexdigest()
        if record.get("prompt_token_ids_sha256") != digest:
            raise ValueError(f"fixture request {index} hash differs")
        requests.append(
            FixtureRequest(
                fixture_index=index,
                fixture_request_id=record.get("request", index),
                prompt_token_ids=list(token_ids),
                prompt_token_ids_sha256=digest,
            )
        )
    return value["metadata"], requests


def parse_concurrencies(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("concurrencies must be integers") from exc
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("concurrencies must be unique and positive")
    return result


def add_batch(
    llm: Any,
    fixture_requests: list[FixtureRequest],
    sampling_params: Any,
    *,
    request_prefix: str,
) -> tuple[list[str], float, float]:
    """Add the full batch with one shared arrival timestamp and no engine step."""

    arrival_epoch = time.time()
    admission_started = time.perf_counter()
    request_ids: list[str] = []
    for index, fixture_request in enumerate(fixture_requests):
        request_id = f"{request_prefix}-{index:02d}"
        prompt = {"prompt_token_ids": fixture_request.prompt_token_ids}
        engine_request = llm.processor.process_inputs(
            request_id,
            prompt,
            sampling_params,
            arrival_time=arrival_epoch,
            priority=0,
        )
        llm.llm_engine.add_request(
            request_id,
            engine_request,
            sampling_params,
            priority=0,
        )
        request_ids.append(request_id)
    admission_finished = time.perf_counter()
    return request_ids, admission_started, admission_finished


def run_batch(
    llm: Any,
    fixture_requests: list[FixtureRequest],
    *,
    output_tokens: int,
    seed: int,
    request_prefix: str,
    ttft_slo_ms: float = SHORT_DIALOGUE_TTFT_SLO_MS,
    tpot_slo_ms: float = SHORT_DIALOGUE_TPOT_SLO_MS,
) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=output_tokens,
        min_tokens=output_tokens,
        ignore_eos=True,
        seed=seed,
        output_kind=RequestOutputKind.DELTA,
    )
    request_ids, admission_started, admission_finished = add_batch(
        llm,
        fixture_requests,
        params,
        request_prefix=request_prefix,
    )
    expected_ids = set(request_ids)
    started_at = admission_finished
    token_times: dict[str, list[float]] = {request_id: [] for request_id in request_ids}
    output_ids: dict[str, list[int]] = {request_id: [] for request_id in request_ids}
    finished: set[str] = set()
    first_step_ids: list[str] = []
    step_count = 0
    step_records: list[dict[str, Any]] = []
    all_prefills_complete_step_index: int | None = None
    requests_with_first_token: set[str] = set()

    while llm.llm_engine.has_unfinished_requests():
        step_started = time.perf_counter()
        step_outputs = llm.llm_engine.step()
        observed_at = time.perf_counter()
        step_count += 1
        step_ids: list[str] = []
        step_output_tokens = 0
        step_finished_requests = 0
        for output in step_outputs:
            request_id = str(output.request_id)
            if request_id not in expected_ids:
                raise RuntimeError(f"unexpected request output: {request_id}")
            step_ids.append(request_id)
            if len(output.outputs) != 1:
                raise RuntimeError(f"request {request_id} returned multiple choices")
            delta_ids = list(output.outputs[0].token_ids)
            step_output_tokens += len(delta_ids)
            if delta_ids and not output_ids[request_id]:
                requests_with_first_token.add(request_id)
            output_ids[request_id].extend(delta_ids)
            token_times[request_id].extend([observed_at] * len(delta_ids))
            if output.finished:
                finished.add(request_id)
                step_finished_requests += 1
        if step_count == 1:
            first_step_ids = step_ids
        if (
            requests_with_first_token == expected_ids
            and all_prefills_complete_step_index is None
        ):
            all_prefills_complete_step_index = step_count
        step_records.append(
            {
                "index": step_count,
                "elapsed_ms": (observed_at - step_started) * 1000.0,
                "output_request_count": len(step_ids),
                "output_token_count": step_output_tokens,
                "finished_request_count": step_finished_requests,
                "is_prefill_boundary": (
                    all_prefills_complete_step_index == step_count
                ),
            }
        )

    if finished != expected_ids:
        raise RuntimeError(
            f"finished request IDs differ: expected={sorted(expected_ids)}, "
            f"actual={sorted(finished)}"
        )
    finished_at = time.perf_counter()

    # V1 keeps IDs completed by the last token step until the following
    # scheduler step, where the worker releases its per-request cached state.
    # The direct harness used to add the next batch immediately, so that
    # cleanup and a new pure-Prefill Graph replay shared one step.  DeepSeek
    # Triton MLA can then replay the graph with stale request metadata.  Drive
    # the required empty cleanup step explicitly between independent batches;
    # keep it outside the request timing boundary.
    cleanup_outputs = llm.llm_engine.step()
    if cleanup_outputs:
        raise RuntimeError(
            "finished-request cleanup unexpectedly produced request outputs"
        )

    records: list[dict[str, Any]] = []
    for request_id, fixture_request in zip(request_ids, fixture_requests):
        times = token_times[request_id]
        ids = output_ids[request_id]
        itl_ms = [
            (right - left) * 1000.0 for left, right in zip(times, times[1:])
        ]
        ttft_ms = (times[0] - started_at) * 1000.0 if times else None
        e2e_ms = (times[-1] - started_at) * 1000.0 if times else None
        tpot_ms = statistics.fmean(itl_ms) if itl_ms else None
        success = len(ids) == output_tokens and len(times) == output_tokens
        records.append(
            {
                "request_id": request_id,
                "fixture_index": fixture_request.fixture_index,
                "fixture_request_id": fixture_request.fixture_request_id,
                "prompt_tokens": len(fixture_request.prompt_token_ids),
                "prompt_token_ids_sha256": fixture_request.prompt_token_ids_sha256,
                "success": success,
                "error": None if success else f"generated {len(ids)}/{output_tokens} tokens",
                "completion_tokens": len(ids),
                "output_token_ids": ids,
                "output_token_ids_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
                "itl_ms": itl_ms,
                "e2e_ms": e2e_ms,
            }
        )

    successful = [record for record in records if record["success"]]
    ttft_values = [float(record["ttft_ms"]) for record in successful]
    tpot_values = [
        float(record["tpot_ms"])
        for record in successful
        if record["tpot_ms"] is not None
    ]
    e2e_values = [float(record["e2e_ms"]) for record in successful]
    slo_satisfied = sum(
        record["ttft_ms"] <= ttft_slo_ms
        and (
            record["tpot_ms"] is None
            or record["tpot_ms"] <= tpot_slo_ms
        )
        for record in successful
    )
    if all_prefills_complete_step_index is None:
        raise RuntimeError("not every request emitted a first token")
    prefill_records = step_records[:all_prefills_complete_step_index]
    steady_decode_records = step_records[all_prefills_complete_step_index:]
    prefill_engine_ms = sum(float(item["elapsed_ms"]) for item in prefill_records)
    steady_decode_ms = sum(
        float(item["elapsed_ms"]) for item in steady_decode_records
    )
    steady_decode_tokens = sum(
        int(item["output_token_count"]) for item in steady_decode_records
    )
    return {
        "concurrency": len(fixture_requests),
        "request_count": len(records),
        "successful_requests": len(successful),
        "failed_requests": len(records) - len(successful),
        "slo_satisfied_requests": slo_satisfied,
        "slo": {
            "ttft_ms": ttft_slo_ms,
            "tpot_ms": tpot_slo_ms,
        },
        "timing_boundary": "after full batch admission; before first EngineCore step",
        "transport": "none",
        "admission_api": "Processor.process_inputs + LLMEngine.add_request",
        "admission_duration_ms": (admission_finished - admission_started) * 1000.0,
        "all_requests_added_before_first_step": True,
        "admitted_request_count": len(request_ids),
        "single_admission_batch": True,
        "first_step_request_ids": first_step_ids,
        "first_step_request_count": len(first_step_ids),
        "single_first_step_batch": set(first_step_ids) == expected_ids,
        "engine_steps": step_count,
        "engine_step_records": step_records,
        "phase_metrics": {
            "definition": (
                "Prefill includes EngineCore steps through the first token from "
                "every admitted request; steady Decode includes only later steps"
            ),
            "prefill_step_count": len(prefill_records),
            "prefill_engine_ms": prefill_engine_ms,
            "prefill_e2e_ms": max(
                (times[0] - started_at) * 1000.0
                for times in token_times.values()
            ),
            "steady_decode_step_count": len(steady_decode_records),
            "steady_decode_elapsed_ms": steady_decode_ms,
            "steady_decode_output_tokens": steady_decode_tokens,
            "steady_decode_tokens_per_second": (
                steady_decode_tokens * 1000.0 / steady_decode_ms
                if steady_decode_ms > 0.0
                else None
            ),
        },
        "elapsed_seconds": finished_at - started_at,
        "ttft_ms": distribution(ttft_values),
        "tpot_ms": distribution(tpot_values),
        "e2e_ms": distribution(e2e_values),
        "requests": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--warmup-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--concurrencies", type=parse_concurrencies, default=[1, 4, 8, 16])
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=2147483648)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    if not (args.model / "config.json").is_file():
        raise SystemExit(f"model snapshot is missing: {args.model}")
    eval_metadata, evaluation = load_fixture(args.fixture)
    warm_metadata, warmups = load_fixture(args.warmup_fixture)
    if len(evaluation) < max(args.concurrencies):
        raise SystemExit("evaluation fixture is smaller than maximum concurrency")
    if len(warmups) < args.warmup_requests:
        raise SystemExit("warmup fixture is too small")
    if eval_metadata.get("workload") != "short_dialogue":
        raise SystemExit("evaluation fixture is not short_dialogue")
    if warm_metadata.get("workload") != "short_dialogue":
        raise SystemExit("warmup fixture is not short_dialogue")

    address_diagnostic_path_raw = os.environ.get(
        "E2_CUDA_ADDRESS_DIAGNOSTIC_FILE", ""
    ).strip()
    address_diagnostic_mode = os.environ.get(
        "E2_CUDA_ADDRESS_DIAGNOSTIC_MODE", "full"
    ).strip().lower()
    if address_diagnostic_mode not in ("full", "lightweight"):
        raise SystemExit(
            "E2_CUDA_ADDRESS_DIAGNOSTIC_MODE must be full or lightweight"
        )

    # Import only after the launcher has installed the complete frozen runtime
    # environment.  In particular, fence-off must be visible before vLLM env
    # constants are evaluated.
    from vllm import LLM

    address_diagnostic_path = (
        Path(address_diagnostic_path_raw)
        if address_diagnostic_path_raw
        else None
    )
    address_snapshots: list[dict[str, Any]] = []
    if address_diagnostic_path is not None and address_diagnostic_mode == "full":
        import torch

        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="python",
            max_entries=200_000,
        )

    compilation_config = {
        "mode": "NONE",
        "cudagraph_mode": "FULL_SEPARATE",
        "cudagraph_capture_sizes": [
            1,
            4,
            8,
            16,
            128,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
        ],
        "max_cudagraph_capture_size": 16384,
    }
    result: dict[str, Any] = {
        "schema": "eurosys_e2_direct_batch_pure_phase_v1",
        "status": "running",
        "started_at": utc_now(),
        "configuration": {
            "model": str(args.model.resolve()),
            "fixture": str(args.fixture.resolve()),
            "warmup_fixture": str(args.warmup_fixture.resolve()),
            "concurrencies": args.concurrencies,
            "prompt_tokens": 128,
            "output_tokens": args.output_tokens,
            "warmup_requests_per_cell": args.warmup_requests,
            "warmup_output_tokens": args.warmup_output_tokens,
            "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
            "max_model_len": 8320,
            "max_num_batched_tokens": 16384,
            "max_num_seqs": 16,
            "compilation_config": compilation_config,
            "client_used": False,
            "http_server_started": False,
            "runtime_post_resume_cold_read_gate_present": (
                "VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE" in os.environ
            ),
            "cuda_address_diagnostic_mode": (
                address_diagnostic_mode
                if address_diagnostic_path is not None
                else "off"
            ),
        },
        "cells": [],
    }
    atomic_json(args.output, result)
    llm: Any | None = None
    try:
        llm = LLM(
            model=str(args.model),
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=8320,
            gpu_memory_utilization=0.97,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            cpu_offload_gb=30,
            max_num_batched_tokens=16384,
            max_num_seqs=16,
            enable_prefix_caching=False,
            generation_config="vllm",
            compilation_config=compilation_config,
            disable_log_stats=False,
        )
        if address_diagnostic_path is not None:
            address_snapshots.append(
                _capture_cuda_address_snapshot(
                    llm,
                    "after_llm_initialization",
                    synchronize=address_diagnostic_mode == "full",
                    include_frames=address_diagnostic_mode == "full",
                )
            )
            atomic_json(address_diagnostic_path, address_snapshots)
        for concurrency in args.concurrencies:
            print(f"DIRECT_BATCH_CELL_START concurrency={concurrency}", flush=True)
            warmup_records: list[dict[str, Any]] = []
            for warmup_index in range(args.warmup_requests):
                warmup = run_batch(
                    llm,
                    [warmups[warmup_index]],
                    output_tokens=args.warmup_output_tokens,
                    seed=args.seed,
                    request_prefix=f"c{concurrency}-warmup-{warmup_index}",
                )
                warmup_records.append(
                    {
                        "successful_requests": warmup["successful_requests"],
                        "single_first_step_batch": warmup["single_first_step_batch"],
                    }
                )
            if (
                address_diagnostic_path is not None
                and address_diagnostic_mode == "full"
            ):
                address_snapshots.append(
                    _capture_cuda_address_snapshot(
                        llm,
                        f"before_c{concurrency}_measurement",
                    )
                )
                atomic_json(address_diagnostic_path, address_snapshots)
            cell = run_batch(
                llm,
                evaluation[:concurrency],
                output_tokens=args.output_tokens,
                seed=args.seed,
                request_prefix=f"c{concurrency}-measurement",
            )
            cell["warmups"] = warmup_records
            result["cells"].append(cell)
            atomic_json(args.output, result)
            print(
                "DIRECT_BATCH_CELL_DONE "
                f"concurrency={concurrency} "
                f"success={cell['successful_requests']}/{cell['request_count']} "
                f"first_step={cell['first_step_request_count']}",
                flush=True,
            )
        result["status"] = "ok" if all(
            cell["successful_requests"] == cell["request_count"]
            and cell["single_first_step_batch"]
            for cell in result["cells"]
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
