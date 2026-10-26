#!/usr/bin/env python3
"""Measure strict vLLM eager or FULL-DECODE CUDA Graph steps.

This runner is intentionally separate from the published P3/pinned/UVA phase
benchmark.  That benchmark is frozen to eager execution.  Here we retain its
scheduler and output contracts, add an explicit FULL_DECODE_ONLY mode, and
record direct capture/replay evidence from vLLM's CUDAGraphWrapper.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PHASE_EXPERIMENT = REPO_ROOT / "experiments" / "vllm_p3_pinned_uva_phase_compare"
sys.path.insert(0, str(PHASE_EXPERIMENT))
import benchmark_phase_steps as phase  # noqa: E402


class GraphCallTracker:
    """Small, file-I/O-free tracer around vLLM's CUDA Graph wrapper."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def snapshot(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))

    @staticmethod
    def delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
        keys = set(before) | set(after)
        return {
            key: int(after.get(key, 0) - before.get(key, 0))
            for key in sorted(keys)
            if after.get(key, 0) != before.get(key, 0)
        }

    def install(self) -> None:
        from vllm.compilation.cuda_graph import CUDAGraphWrapper
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import get_forward_context

        tracker = self
        original_call = CUDAGraphWrapper.__call__
        if getattr(original_call, "_p3_alias_graph_tracker", False):
            raise RuntimeError("CUDAGraphWrapper tracker was installed twice")

        def traced_call(wrapper, *args, **kwargs):
            context = get_forward_context()
            runtime_mode = context.cudagraph_runtime_mode
            mode_name = getattr(runtime_mode, "name", str(runtime_mode)).lower()
            if (
                runtime_mode == CUDAGraphMode.NONE
                or runtime_mode != wrapper.runtime_mode
            ):
                action = "passthrough"
            else:
                entry = wrapper.concrete_cudagraph_entries.get(
                    context.batch_descriptor
                )
                action = (
                    "replay"
                    if entry is not None and entry.cudagraph is not None
                    else "capture"
                )
            tracker.counts[f"{mode_name}_{action}"] += 1
            return original_call(wrapper, *args, **kwargs)

        traced_call._p3_alias_graph_tracker = True  # type: ignore[attr-defined]
        CUDAGraphWrapper.__call__ = traced_call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument(
        "--prompt-token-limit",
        type=int,
        default=0,
        help="truncate every frozen request to this many prompt tokens",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--offload-manifest", required=True, type=Path)
    parser.add_argument("--profile-label", required=True)
    parser.add_argument("--mechanism-class", required=True)
    parser.add_argument(
        "--execution-mode",
        choices=("eager", "full_graph_decode", "full_graph_all"),
        required=True,
    )
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--cudagraph-capture-sizes", default="1,4")
    parser.add_argument("--conditioning-repeats", type=int, default=1)
    parser.add_argument("--measurement-repeats", type=int, default=3)
    parser.add_argument("--warmup-decode-steps", type=int, default=5)
    parser.add_argument("--measured-decode-steps", type=int, default=16)
    parser.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    parser.add_argument("--cpu-offload-gb", type=float, default=3.0)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--formal-e3",
        action="store_true",
        help="allow the frozen E3 B1/B4/B8 and workload-length matrix",
    )
    return parser.parse_args()


def cuda_memory_snapshot(torch: Any) -> dict[str, int]:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "used_bytes": int(total_bytes - free_bytes),
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def read_manifest_evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"offload manifest was not created: {path}")
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"invalid manifest JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise RuntimeError(
                f"manifest record at {path}:{line_number} is not an object"
            )
        records.append(record)

    event_counts = Counter(str(item.get("event", "<missing>")) for item in records)
    parameter_records = [
        item for item in records if item.get("event") == "cpu_offload_parameter"
    ]
    forward_records = [
        item for item in records if item.get("event") == "cpu_offload_copy_forward"
    ]
    alias_records = [item for item in records if item.get("graph_static_alias") is True]
    alias_parameter_records = [
        item
        for item in parameter_records
        if item.get("graph_static_alias") is True
    ]

    def unique_positive(field: str, selected: list[dict[str, Any]]) -> list[int]:
        return sorted(
            {
                int(item[field])
                for item in selected
                if isinstance(item.get(field), int) and int(item[field]) > 0
            }
        )

    upload_vas = unique_positive(
        "graph_static_alias_upload_va", alias_parameter_records
    )
    consumer_vas = unique_positive(
        "graph_static_alias_consumer_va", alias_parameter_records
    )
    pinned_source_vas = unique_positive(
        "graph_static_alias_source_ptr", alias_parameter_records
    )
    backing_by_upload: dict[int, int] = {}
    for item in alias_parameter_records:
        upload_va = item.get("graph_static_alias_upload_va")
        backing_bytes = item.get(
            "graph_static_alias_backing_bytes",
            item.get("graph_static_alias_aligned_bytes"),
        )
        if isinstance(upload_va, int) and upload_va > 0 and isinstance(
            backing_bytes, int
        ):
            previous = backing_by_upload.setdefault(upload_va, int(backing_bytes))
            if previous != int(backing_bytes):
                raise RuntimeError(
                    f"upload VA {upload_va:#x} has inconsistent backing sizes"
                )

    capture_values = Counter(
        str(bool(item.get("graph_static_alias_capture_active"))).lower()
        for item in forward_records
        if item.get("graph_static_alias") is True
        and "graph_static_alias_capture_active" in item
    )
    capture_phase_values = Counter(
        str(bool(item.get("graph_static_alias_capture_phase"))).lower()
        for item in forward_records
        if item.get("graph_static_alias") is True
        and "graph_static_alias_capture_phase" in item
    )
    stream_ids = sorted(
        {
            int(item["graph_static_alias_stream_id"])
            for item in forward_records
            if item.get("graph_static_alias") is True
            and isinstance(item.get("graph_static_alias_stream_id"), int)
        }
    )
    bank_ids = sorted(
        {
            int(item["graph_static_alias_bank_id"])
            for item in alias_parameter_records
            if isinstance(item.get("graph_static_alias_bank_id"), int)
        }
    )
    layer_indices = sorted(
        {
            int(item["graph_static_alias_layer_index"])
            for item in alias_parameter_records
            if isinstance(item.get("graph_static_alias_layer_index"), int)
        }
    )
    forward_indices = [
        int(item["graph_static_alias_forward_call_index"])
        for item in forward_records
        if item.get("graph_static_alias") is True
        and isinstance(item.get("graph_static_alias_forward_call_index"), int)
    ]
    static_bank_bytes = [
        int(item["copy_bank_bytes"])
        for item in records
        if isinstance(item.get("copy_bank_bytes"), int)
    ]
    static_bank_tensor_counts = [
        int(item["copy_bank_tensor_count"])
        for item in records
        if isinstance(item.get("copy_bank_tensor_count"), int)
    ]

    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "record_count": len(records),
        "event_counts": dict(sorted(event_counts.items())),
        "parameter_record_count": len(parameter_records),
        "parameter_logical_bytes": sum(
            int(item.get("bytes", 0)) for item in parameter_records
        ),
        "graph_static_alias_record_count": len(alias_records),
        "graph_static_alias_parameter_count": len(alias_parameter_records),
        "bank_ids": bank_ids,
        "layer_indices": layer_indices,
        "upload_vas": upload_vas,
        "consumer_vas": consumer_vas,
        "pinned_source_vas": pinned_source_vas,
        "unique_physical_backing_bytes": sum(backing_by_upload.values()),
        "forward_record_count": len(forward_records),
        "forward_capture_active_counts": dict(sorted(capture_values.items())),
        "forward_capture_phase_counts": dict(
            sorted(capture_phase_values.items())
        ),
        "forward_stream_ids": stream_ids,
        "max_forward_call_index": max(forward_indices, default=0),
        "upload_access_values": sorted(
            {
                str(item["graph_static_alias_upload_access"])
                for item in alias_parameter_records
                if item.get("graph_static_alias_upload_access") is not None
            }
        ),
        "consumer_access_values": sorted(
            {
                str(item["graph_static_alias_consumer_access"])
                for item in alias_parameter_records
                if item.get("graph_static_alias_consumer_access") is not None
            }
        ),
        "max_graph_static_alias_total_backing_bytes": max(
            (
                int(item["graph_static_alias_total_backing_bytes"])
                for item in alias_parameter_records
                if isinstance(
                    item.get("graph_static_alias_total_backing_bytes"), int
                )
            ),
            default=0,
        ),
        "max_static_bank_bytes": max(static_bank_bytes, default=0),
        "max_static_bank_tensor_count": max(static_bank_tensor_counts, default=0),
    }


def run() -> int:
    args = parse_args()
    batches = phase.parse_positive_ints(args.batch_sizes)
    capture_sizes = phase.parse_positive_ints(args.cudagraph_capture_sizes)
    if not args.formal_e3 and (len(batches) != 2 or set(batches) != {1, 4}):
        raise ValueError("qualification contract requires batches 1 and 4")
    if (
        not args.formal_e3
        and args.execution_mode in ("eager", "full_graph_decode")
        and capture_sizes != [1, 4]
    ):
        raise ValueError(
            "qualification contract requires --cudagraph-capture-sizes 1,4"
        )
    if args.conditioning_repeats < 0 or args.measurement_repeats <= 0:
        raise ValueError("invalid repeat counts")
    if args.warmup_decode_steps < 1 or args.measured_decode_steps <= 0:
        raise ValueError("at least one warmup and one measured decode are required")
    if not args.formal_e3 and args.max_num_seqs != 4:
        raise ValueError("qualification contract requires --max-num-seqs 4")
    if max(batches) > args.max_num_seqs:
        raise ValueError("batch size exceeds max_num_seqs")
    if not math.isfinite(args.cpu_offload_gb) or args.cpu_offload_gb < 0:
        raise ValueError("cpu_offload_gb must be finite and non-negative")

    fixture_metadata, requests = phase.load_requests(args.fixture)
    if args.prompt_token_limit < 0:
        raise ValueError("prompt token limit must be non-negative")
    if args.prompt_token_limit:
        if any(
            int(item["prompt_tokens"]) < args.prompt_token_limit
            for item in requests
        ):
            raise ValueError("fixture request is shorter than --prompt-token-limit")
        requests = [
            item
            | {
                "prompt_token_ids": list(item["prompt_token_ids"])[
                    : args.prompt_token_limit
                ],
                "prompt_tokens": args.prompt_token_limit,
            }
            for item in requests
        ]
    max_batch = max(batches)
    if len(requests) < max_batch:
        raise ValueError("fixture has fewer requests than the largest batch")
    expected_prefill_sizes = sorted(
        {sum(item["prompt_tokens"] for item in requests[:batch]) for batch in batches}
    )
    if args.execution_mode == "full_graph_all":
        expected_capture_sizes = sorted({*batches, *expected_prefill_sizes})
        if capture_sizes != expected_capture_sizes:
            raise ValueError(
                "full_graph_all requires exact decode and frozen-prefill capture "
                f"sizes {expected_capture_sizes}, got {capture_sizes}"
            )
    total_output_tokens = 1 + args.warmup_decode_steps + args.measured_decode_steps
    if any(
        request["prompt_tokens"] + total_output_tokens > args.max_model_len
        for request in requests[:max_batch]
    ):
        raise ValueError("prompt plus generated tokens exceeds max_model_len")
    if any(
        sum(item["prompt_tokens"] for item in requests[:batch])
        > args.max_num_batched_tokens
        for batch in batches
    ):
        raise ValueError("a prefill batch exceeds max_num_batched_tokens")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    initial_result: dict[str, Any] = {
        "schema": "vllm_p3_static_alias_graph_steps_v1",
        "status": "loading",
        "profile_label": args.profile_label,
        "mechanism_class": args.mechanism_class,
        "config": {
            **vars(args),
            "fixture": str(args.fixture.resolve()),
            "output": str(args.output.resolve()),
            "offload_manifest": str(args.offload_manifest.resolve()),
            "batch_sizes": batches,
            "cudagraph_capture_sizes": capture_sizes,
        },
    }
    args.output.write_text(
        json.dumps(initial_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = initial_result
    try:
        import torch
        import vllm
        from vllm import LLM, SamplingParams, TokensPrompt
        from vllm.compilation.counter import compilation_counter
        from vllm.config import CompilationMode, CUDAGraphMode
        from vllm.model_executor.models.utils import (
            get_cpu_offload_graph_static_alias_diagnostics,
        )

        graph_tracker = GraphCallTracker()
        graph_tracker.install()
        counter_before_load = dataclasses.asdict(compilation_counter.clone())

        enforce_eager = args.execution_mode == "eager"
        original_va_phase_mode = os.environ.get(
            "VLLM_UVM_ORIGINAL_VA_PHASE_MODE", "off"
        ).strip()
        full_graph_mode_name = (
            "FULL_SEPARATE"
            if original_va_phase_mode == "prefill_graph_remote_decode"
            else "FULL"
        )
        compilation_config: dict[str, Any] | None = None
        if not enforce_eager:
            requested_cudagraph_mode = (
                full_graph_mode_name
                if args.execution_mode == "full_graph_all"
                else "FULL_DECODE_ONLY"
            )
            compilation_config = {
                # FULL CUDA Graph capture does not require Dynamo/Inductor.
                # Keeping Python model execution here is intentional: the
                # offload hook emits the fixed H2D nodes during capture, while
                # replay bypasses Python entirely.
                "mode": "NONE",
                "cudagraph_mode": requested_cudagraph_mode,
                "cudagraph_capture_sizes": capture_sizes,
                "max_cudagraph_capture_size": max(capture_sizes),
            }

        load_started = time.perf_counter()
        llm = LLM(
            model=args.model,
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=enforce_eager,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            cpu_offload_gb=args.cpu_offload_gb,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enable_dbo=False,
            compilation_config=compilation_config,
            seed=args.seed,
            disable_log_stats=True,
        )
        phase.enable_request_metrics_without_default_loggers(llm)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - load_started
        torch.cuda.reset_peak_memory_stats()

        realized_compilation = llm.llm_engine.vllm_config.compilation_config
        realized_compilation_mode = realized_compilation.mode
        realized_mode = realized_compilation.cudagraph_mode
        if realized_compilation_mode != CompilationMode.NONE:
            raise RuntimeError(
                "qualification requires CompilationMode.NONE; realized "
                f"{realized_compilation_mode}"
            )
        if enforce_eager:
            if realized_mode != CUDAGraphMode.NONE:
                raise RuntimeError(f"eager arm realized unexpected mode {realized_mode}")
        else:
            expected_mode = (
                (
                    CUDAGraphMode.FULL_SEPARATE
                    if full_graph_mode_name == "FULL_SEPARATE"
                    else CUDAGraphMode.FULL
                )
                if args.execution_mode == "full_graph_all"
                else CUDAGraphMode.FULL_DECODE_ONLY
            )
            if realized_mode != expected_mode:
                raise RuntimeError(
                    "graph arm did not retain the requested mode; realized "
                    f"{realized_mode}, expected {expected_mode}"
                )
        realized_sizes = list(realized_compilation.cudagraph_capture_sizes)
        expected_realized_sizes = [] if enforce_eager else capture_sizes
        if realized_sizes != expected_realized_sizes:
            raise RuntimeError(
                "capture sizes differ: expected "
                f"{expected_realized_sizes}, realized {realized_sizes}"
            )

        counter_after_load = dataclasses.asdict(compilation_counter.clone())
        counter_load_delta = {
            key: int(counter_after_load[key] - counter_before_load[key])
            for key in sorted(counter_after_load)
        }
        graph_calls_after_load = graph_tracker.snapshot()
        alias_diagnostics_after_load = (
            get_cpu_offload_graph_static_alias_diagnostics()
        )
        alias_enabled = os.environ.get(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        expected_alias_arenas = int(
            os.environ.get(
                "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_ARENA_COUNT",
                "4",
            ),
            0,
        )
        expected_alias_backing_bytes = int(
            os.environ.get(
                "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPECTED_BACKING_BYTES",
                "1207959552",
            ),
            0,
        )
        if bool(alias_diagnostics_after_load.get("enabled")) != alias_enabled:
            raise RuntimeError(
                "static-alias diagnostics disagree with the requested mode: "
                f"{alias_diagnostics_after_load}"
            )
        if alias_enabled:
            if alias_diagnostics_after_load.get("capture_phase") is not False:
                raise RuntimeError("static-alias capture phase remained active")
            if (
                int(alias_diagnostics_after_load.get("arena_count", 0))
                != expected_alias_arenas
            ):
                raise RuntimeError(
                    "Static alias created an unexpected arena count: "
                    f"{alias_diagnostics_after_load}"
                )
            if (
                int(alias_diagnostics_after_load.get("backing_bytes", 0))
                != expected_alias_backing_bytes
            ):
                raise RuntimeError(
                    "Static alias created an unexpected backing size: "
                    f"{alias_diagnostics_after_load}"
                )
        if not enforce_eager:
            if counter_load_delta.get("num_cudagraph_captured", 0) < len(capture_sizes):
                raise RuntimeError(
                    "vLLM did not capture the requested full decode graphs: "
                    f"{counter_load_delta}"
                )
            if graph_calls_after_load.get("full_capture", 0) < len(capture_sizes):
                raise RuntimeError(
                    f"wrapper tracker saw too few FULL captures: {graph_calls_after_load}"
                )

        result = {
            **initial_result,
            "status": "running",
            "config": initial_result["config"]
            | {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "enable_dbo": False,
                "enforce_eager": enforce_eager,
                "requested_cudagraph_mode": (
                    "NONE"
                    if enforce_eager
                    else (
                        full_graph_mode_name
                        if args.execution_mode == "full_graph_all"
                        else "FULL_DECODE_ONLY"
                    )
                ),
                "requested_compilation_mode": "NONE",
                "realized_compilation_mode": realized_compilation_mode.name,
                "realized_cudagraph_mode": realized_mode.name,
                "realized_cudagraph_capture_sizes": realized_sizes,
                "total_output_tokens": total_output_tokens,
                "enable_prefix_caching": False,
                "enable_chunked_prefill": False,
                "engine_core_multiprocessing": False,
                "temperature": 0.0,
                "ignore_eos": True,
                "frozen_prompt_token_ids": True,
            },
            "versions": {
                "vllm": getattr(vllm, "__version__", None),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "fixture": {
                "sha256": phase.sha256_file(args.fixture),
                "metadata": fixture_metadata,
            },
            "load_seconds": load_seconds,
            "cuda_memory_after_load": cuda_memory_snapshot(torch),
            "compilation_counter_before_load": counter_before_load,
            "compilation_counter_after_load": counter_after_load,
            "compilation_counter_load_delta": counter_load_delta,
            "graph_wrapper_calls_after_load": graph_calls_after_load,
            "static_alias_diagnostics_after_load": alias_diagnostics_after_load,
            "cycles": [],
        }
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

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
            schedule_records.append(phase.schedule_snapshot(schedule_output))
            return schedule_output

        scheduler.schedule = traced_schedule
        sampling = SamplingParams(
            max_tokens=total_output_tokens,
            min_tokens=total_output_tokens,
            temperature=0.0,
            seed=args.seed,
            ignore_eos=True,
        )

        def run_step() -> tuple[float, list[Any], dict[str, Any], dict[str, int]]:
            schedule_before = len(schedule_records)
            torch.cuda.synchronize()
            graph_before = graph_tracker.snapshot()
            started = time.perf_counter()
            outputs = llm.llm_engine.step()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            graph_after = graph_tracker.snapshot()
            if len(schedule_records) != schedule_before + 1:
                raise RuntimeError("expected exactly one scheduler call per engine step")
            return (
                elapsed_ms,
                outputs,
                schedule_records[-1],
                graph_tracker.delta(graph_before, graph_after),
            )

        repeat_roles = ["conditioning"] * args.conditioning_repeats + [
            "measurement"
        ] * args.measurement_repeats
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

                prefill_ms, prefill_outputs, prefill_schedule, prefill_graph = run_step()
                phase.validate_prefill(prefill_schedule, batch)
                if prefill_outputs:
                    raise RuntimeError("FINAL_ONLY prefill returned request outputs")
                if prefill_schedule["total_num_scheduled_tokens"] != expected_prompt_tokens:
                    raise RuntimeError("prefill used different frozen prompt tokens")
                prefill_full_replays = int(prefill_graph.get("full_replay", 0))
                if args.execution_mode == "full_graph_all":
                    if prefill_full_replays != 1:
                        raise RuntimeError(
                            "full_graph_all prefill did not use exactly one FULL "
                            f"replay: {prefill_graph}"
                        )
                elif prefill_full_replays:
                    raise RuntimeError(
                        "large qualification prefill unexpectedly used a FULL graph"
                    )
                diagnostics_started = time.perf_counter()
                alias_diagnostics_after_prefill = (
                    get_cpu_offload_graph_static_alias_diagnostics()
                )
                diagnostics_overhead_seconds = (
                    time.perf_counter() - diagnostics_started
                )

                warmup_steps: list[dict[str, Any]] = []
                for step_index in range(args.warmup_decode_steps):
                    elapsed_ms, outputs, schedule, graph_delta = run_step()
                    phase.validate_decode(schedule, batch)
                    if outputs:
                        raise RuntimeError("decode warmup finished requests early")
                    warmup_steps.append(
                        {
                            "step": step_index + 1,
                            "elapsed_ms": elapsed_ms,
                            "schedule": schedule,
                            "graph_wrapper_delta": graph_delta,
                        }
                    )

                measured_steps: list[dict[str, Any]] = []
                final_outputs: list[Any] = []
                for step_index in range(args.measured_decode_steps):
                    elapsed_ms, outputs, schedule, graph_delta = run_step()
                    phase.validate_decode(schedule, batch)
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
                            "graph_wrapper_delta": graph_delta,
                        }
                    )

                cycle_elapsed_ms = (
                    time.perf_counter()
                    - cycle_started_perf
                    - diagnostics_overhead_seconds
                ) * 1000.0
                alias_diagnostics_after_decode = (
                    get_cpu_offload_graph_static_alias_diagnostics()
                )
                if alias_enabled and not enforce_eager and int(
                    alias_diagnostics_after_decode["forward_call_count"]
                ) != int(alias_diagnostics_after_prefill["forward_call_count"]):
                    raise RuntimeError(
                        "decode entered the Python offload hook instead of replaying "
                        "the captured H2D nodes"
                    )
                if llm.llm_engine.has_unfinished_requests():
                    raise RuntimeError("requests remain after a completed cycle")
                records = phase.output_records(final_outputs)
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

                step_samples = [float(item["elapsed_ms"]) for item in measured_steps]
                request_timings = phase.request_timing_summaries(records)
                result["cycles"].append(
                    {
                        "role": role,
                        "role_index": role_index,
                        "batch_size": batch,
                        "prompt_tokens": expected_prompt_tokens,
                        "cycle_started_epoch": cycle_started_epoch,
                        "cycle_ended_epoch": time.time(),
                        "cycle_elapsed_ms": cycle_elapsed_ms,
                        "excluded_diagnostics_overhead_ms": (
                            diagnostics_overhead_seconds * 1000.0
                        ),
                        "prefill_elapsed_ms": prefill_ms,
                        "prefill_schedule": prefill_schedule,
                        "prefill_graph_wrapper_delta": prefill_graph,
                        "static_alias_diagnostics_after_prefill": (
                            alias_diagnostics_after_prefill
                        ),
                        "static_alias_diagnostics_after_decode": (
                            alias_diagnostics_after_decode
                        ),
                        "warmup_decode_steps": warmup_steps,
                        "measured_decode_steps": measured_steps,
                        "decode_step_ms": phase.summarize(step_samples),
                        "decode_tokens_per_second": (
                            batch * 1000.0 / statistics.median(step_samples)
                        ),
                        **request_timings,
                        "output_token_ids_sha256": phase.output_hash(records),
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
            phase_summary[str(batch)] = {
                "batch_size": batch,
                "prompt_tokens": sum(item["prompt_tokens"] for item in requests[:batch]),
                "measurement_repeats": len(selected_cycles),
                "prefill_step_ms": phase.summarize(prefill_samples),
                "prefill_tokens_per_second": (
                    sum(item["prompt_tokens"] for item in requests[:batch])
                    * 1000.0
                    / statistics.median(prefill_samples)
                ),
                "decode_repeat_median_ms": phase.summarize(decode_repeat_medians),
                "decode_all_step_ms": phase.summarize(decode_steps),
                "decode_tokens_per_second": (
                    batch * 1000.0 / statistics.median(decode_repeat_medians)
                ),
                "cycle_e2e_ms": phase.summarize(cycle_samples),
            }

        result["phase_summary"] = phase_summary
        result["graph_wrapper_calls_final"] = graph_tracker.snapshot()
        result["static_alias_diagnostics_final"] = (
            get_cpu_offload_graph_static_alias_diagnostics()
        )
        result["compilation_counter_final"] = dataclasses.asdict(
            compilation_counter.clone()
        )
        result["offload_manifest_evidence"] = read_manifest_evidence(
            args.offload_manifest
        )
        result["cuda_memory_final"] = cuda_memory_snapshot(torch)
        result["environment"] = {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("VLLM_CPU_OFFLOAD_")
            or key
            in {
                "VLLM_USE_UVM",
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "CUDA_VISIBLE_DEVICES",
            }
        }
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "exception"
        result["error"] = repr(exc)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        raise

    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "phase_summary": result["phase_summary"],
                "graph_wrapper_calls_final": result["graph_wrapper_calls_final"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
