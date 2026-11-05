#!/usr/bin/env python3
"""Reproduce native-segment pressure on Qwen with a pure/mixed-shape A/B.

The process first completes one 7168-token pure Prefill.  It then schedules a
same-size mixed step containing 7167 Prefill tokens plus one Decode token.
Allocator segments, global free memory, and hot managed-expert residency are
sampled at every boundary.  The frozen Track-A expert plan remains unchanged.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PHASE_HELPER = (
    REPO_ROOT
    / "experiments"
    / "vllm_p3_pinned_uva_phase_compare"
    / "benchmark_phase_steps.py"
)
STATIC_HELPER = (
    REPO_ROOT
    / "experiments"
    / "vllm_uvm_prefill_alias_decode_remote_graph"
    / "benchmark_phase_switch_steps.py"
)


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        stream.flush()


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def expand_prompt(token_ids: list[int], length: int) -> list[int]:
    require(bool(token_ids), "source prompt is empty")
    repeats = (length + len(token_ids) - 1) // len(token_ids)
    return (token_ids * repeats)[:length]


def hot_ranges(utils: Any) -> list[tuple[int, int]]:
    bindings = getattr(utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS")
    range_fn = getattr(utils, "_uvm_original_alias_binding_hot_ranges")
    return [
        (int(base), int(length))
        for _, binding in sorted(bindings.items())
        for base, length in range_fn(binding)
    ]


def hot_residency(utils: Any, device: int = 0) -> dict[str, Any]:
    from vllm.device_allocator import uvm

    ranges = hot_ranges(utils)
    value = dict(
        uvm.driver_batch_query_ranges_residency_bytes(
            ranges,
            device=device,
            wait_for_trackers=False,
        )
    )
    value["range_count"] = len(ranges)
    value["requested_bytes"] = sum(length for _, length in ranges)
    return value


def compact_frames(entry: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "filename": str(frame.get("filename", "")),
            "line": int(frame.get("line", 0)),
            "name": str(frame.get("name", "")),
        }
        for frame in list(entry.get("frames", []))[:24]
    ]


class AllocatorTracker:
    def __init__(self, torch: Any, utils: Any, device: int = 0) -> None:
        self.torch = torch
        self.utils = utils
        self.device = device
        self.addresses: set[int] = set()
        self.trace_count = 0
        self.previous_memory: dict[str, int] = {}

    def enable(self) -> None:
        self.torch.cuda.memory._record_memory_history(
            enabled="all",
            context="alloc",
            stacks="python",
            max_entries=200000,
            device=self.device,
            clear_history=True,
        )

    def snapshot(self, label: str) -> dict[str, Any]:
        torch = self.torch
        torch.cuda.synchronize(self.device)
        free_bytes, total_bytes = (
            int(value) for value in torch.cuda.mem_get_info(self.device)
        )
        stats = torch.cuda.memory_stats(self.device)
        raw = torch.cuda.memory._snapshot(self.device)
        segments = [
            segment
            for segment in raw.get("segments", [])
            if int(segment.get("device", self.device)) == self.device
        ]
        current_addresses = {
            int(segment.get("address", 0)) for segment in segments
        }
        new_addresses = current_addresses - self.addresses
        new_segments = [
            {
                "address": int(segment.get("address", 0)),
                "total_size": int(segment.get("total_size", 0)),
                "allocated_size": int(segment.get("allocated_size", 0)),
                "active_size": int(segment.get("active_size", 0)),
                "segment_type": str(segment.get("segment_type", "unknown")),
                "stream": int(segment.get("stream", 0)),
            }
            for segment in segments
            if int(segment.get("address", 0)) in new_addresses
        ]
        new_segments.sort(key=lambda value: value["address"])

        device_traces = raw.get("device_traces", [])
        traces = device_traces[self.device] if self.device < len(device_traces) else []
        start = self.trace_count if self.trace_count <= len(traces) else 0
        new_trace_entries = traces[start:]
        segment_alloc_events = [
            {
                "action": "segment_alloc",
                "address": int(entry.get("addr", 0)),
                "bytes": int(entry.get("size", 0)),
                "stream": int(entry.get("stream", 0)),
                "frames": compact_frames(entry),
            }
            for entry in new_trace_entries
            if entry.get("action") == "segment_alloc"
        ]

        memory = {
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "allocated_bytes": int(torch.cuda.memory_allocated(self.device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(self.device)),
            "active_bytes": int(stats.get("active_bytes.all.current", 0)),
            "inactive_split_bytes": int(
                stats.get("inactive_split_bytes.all.current", 0)
            ),
            "segment_count": int(stats.get("segment.all.current", 0)),
            "allocation_count": int(stats.get("allocation.all.current", 0)),
            "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
            "num_ooms": int(stats.get("num_ooms", 0)),
        }
        delta = {
            key: int(value) - int(self.previous_memory.get(key, value))
            for key, value in memory.items()
            if key != "total_bytes"
        }
        residency = hot_residency(self.utils, self.device)
        record = {
            "label": label,
            "epoch_seconds": time.time(),
            "memory": memory,
            "delta": delta,
            "new_segment_bytes": sum(item["total_size"] for item in new_segments),
            "new_segments": new_segments,
            "segment_alloc_event_bytes": sum(
                item["bytes"] for item in segment_alloc_events
            ),
            "segment_alloc_events": segment_alloc_events,
            "hot_physical_residency": residency,
            "history_entry_count": len(traces),
        }
        self.addresses = current_addresses
        self.trace_count = len(traces)
        self.previous_memory = memory
        return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--exact-names", required=True, type=Path)
    parser.add_argument("--expert-plan", required=True, type=Path)
    parser.add_argument("--offload-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    parser.add_argument("--pure-prefill-tokens", type=int, default=7168)
    parser.add_argument("--mixed-prefill-tokens", type=int, default=7167)
    parser.add_argument("--lead-prefill-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=7172)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--cpu-offload-gb", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-selected-bytes", type=int, required=True)
    args = parser.parse_args()

    require(
        args.mixed_prefill_tokens + 1 == args.pure_prefill_tokens,
        "mixed step must equal the pure Prefill token count",
    )
    require(args.pure_prefill_tokens < 8192, "probe must remain below 8192 tokens")
    require(
        args.mixed_prefill_tokens + 2 <= args.max_model_len,
        "mixed prompt plus output exceeds max model length",
    )
    require(
        os.environ.get("VLLM_UVM_ORIGINAL_VA_PHASE_MODE", "").strip().lower()
        == "off",
        "probe requires the persistent sealed-alias phase",
    )
    for path in (args.fixture, args.exact_names, args.expert_plan):
        require(path.is_file(), f"missing input: {path}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    args.offload_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.progress.write_text("", encoding="utf-8")
    args.offload_manifest.write_text("", encoding="utf-8")

    phase = load_module("qwen_frag_phase_helper", PHASE_HELPER)
    static = load_module("qwen_frag_static_helper", STATIC_HELPER)
    fixture_metadata, requests = phase.load_requests(args.fixture)
    require(len(requests) >= 3, "fixture has fewer than three prompts")

    result: dict[str, Any] = {
        "schema": "qwen_native_segment_fragmentation_probe_v1",
        "status": "loading",
        "started_epoch_seconds": time.time(),
        "config": {
            **vars(args),
            "fixture": str(args.fixture.resolve()),
            "exact_names": str(args.exact_names.resolve()),
            "expert_plan": str(args.expert_plan.resolve()),
            "offload_manifest": str(args.offload_manifest.resolve()),
            "output": str(args.output.resolve()),
            "progress": str(args.progress.resolve()),
            "fixture_metadata": fixture_metadata,
        },
        "snapshots": [],
        "steps": [],
    }
    write_result(args.output, result)

    llm: Any = None
    scheduler: Any = None
    original_schedule: Any = None
    tracker: AllocatorTracker | None = None
    error: BaseException | None = None

    try:
        import torch
        import vllm
        from vllm import SamplingParams, TokensPrompt
        from vllm.device_allocator.uvm_static_alias import (
            get_static_alias_diagnostics,
        )
        from vllm.model_executor.models import utils

        load_started = time.perf_counter()
        llm = vllm.LLM(
            model=args.model,
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=True,
            max_model_len=args.max_model_len,
            max_num_seqs=2,
            max_num_batched_tokens=args.pure_prefill_tokens,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            cpu_offload_gb=args.cpu_offload_gb,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            enable_dbo=False,
            seed=args.seed,
            disable_log_stats=True,
        )
        phase.enable_request_metrics_without_default_loggers(llm)
        torch.cuda.synchronize()
        result["load_seconds"] = time.perf_counter() - load_started

        diagnostics = static.diagnostics_snapshot(utils, get_static_alias_diagnostics)
        model_diag = diagnostics["model"]
        require(
            int(model_diag.get("selected_parameter_bytes_per_forward", -1))
            == args.expected_selected_bytes,
            "selected bytes differ from the frozen plan",
        )
        require(model_diag.get("lifecycle_state") == "sealed", "alias state is not sealed")
        result["diagnostics_after_load"] = diagnostics

        client = llm.llm_engine.engine_core
        core = getattr(client, "engine_core", None)
        require(core is not None, "in-process EngineCore is unavailable")
        scheduler = core.scheduler
        original_schedule = scheduler.schedule
        schedule_records: list[dict[str, Any]] = []

        def traced_schedule():
            output = original_schedule()
            schedule_records.append(phase.schedule_snapshot(output))
            return output

        scheduler.schedule = traced_schedule

        tracker = AllocatorTracker(torch, utils)
        tracker.enable()

        def record_snapshot(label: str) -> dict[str, Any]:
            require(tracker is not None, "allocator tracker is absent")
            value = tracker.snapshot(label)
            result["snapshots"].append(value)
            append_jsonl(
                args.progress,
                {"event": "snapshot", **value},
            )
            write_result(args.output, result)
            return value

        def hot_read_gate(stage: str) -> dict[str, Any] | None:
            value = utils._uvm_original_alias_hot_cuda_read_gate(
                stage=stage,
                device=0,
            )
            append_jsonl(
                args.progress,
                {"event": "hot_read_gate", "stage": stage, "result": value},
            )
            return value

        def run_step(label: str) -> tuple[dict[str, Any], list[Any]]:
            before = record_snapshot(f"{label}:before")
            append_jsonl(
                args.progress,
                {"event": "step_begin", "label": label, "epoch_seconds": time.time()},
            )
            schedule_before = len(schedule_records)
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                outputs = llm.llm_engine.step()
                torch.cuda.synchronize()
            except BaseException as exc:
                append_jsonl(
                    args.progress,
                    {
                        "event": "step_exception",
                        "label": label,
                        "epoch_seconds": time.time(),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            require(
                len(schedule_records) == schedule_before + 1,
                f"{label}: scheduler call count differs",
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            after = record_snapshot(f"{label}:after")
            record = {
                "label": label,
                "elapsed_ms": elapsed_ms,
                "schedule": schedule_records[-1],
                "output_count": len(outputs),
                "before_snapshot": before["label"],
                "after_snapshot": after["label"],
            }
            result["steps"].append(record)
            append_jsonl(args.progress, {"event": "step_complete", **record})
            write_result(args.output, result)
            return record, outputs

        sampling = SamplingParams(
            max_tokens=2,
            min_tokens=2,
            temperature=0.0,
            seed=args.seed,
            ignore_eos=True,
        )

        hot_read_gate("runtime_qwen_probe_after_load")
        baseline = record_snapshot("after_load")
        hot = baseline["hot_physical_residency"]
        require(
            int(hot.get("gpu_bytes", -1)) == int(hot.get("requested_bytes", -2))
            and int(hot.get("cpu_bytes", -1)) == 0
            and int(hot.get("unknown_bytes", -1)) == 0,
            "hot experts are not fully GPU resident after load",
        )

        pure_prompt = TokensPrompt(
            prompt_token_ids=expand_prompt(
                list(requests[0]["prompt_token_ids"]),
                args.pure_prefill_tokens,
            )
        )
        llm._validate_and_add_requests(
            prompts=[pure_prompt],
            params=sampling,
            use_tqdm=False,
            lora_request=None,
            priority=None,
        )
        pure_prefill, outputs = run_step("pure_prefill")
        require(
            int(pure_prefill["schedule"]["total_num_scheduled_tokens"])
            == args.pure_prefill_tokens,
            "pure Prefill schedule differs",
        )
        for index in range(4):
            if outputs:
                break
            _, outputs = run_step(f"pure_finish_{index + 1}")
        require(outputs, "pure request did not finish")
        require(not llm.llm_engine.has_unfinished_requests(), "pure request remains")
        result["pure_control"] = {"status": "ok", "output_count": len(outputs)}

        # The generic fused-MoE audit is intentionally one-shot. Reset only
        # its diagnostic latch so the same instrumentation records the mixed
        # step as well; this does not alter allocation or model state.
        from vllm.model_executor.layers.fused_moe import fused_moe

        fused_moe._MOE_POST_WORKSPACE_UVM_AUDIT_DONE = False

        lead_prompt = TokensPrompt(
            prompt_token_ids=expand_prompt(
                list(requests[1]["prompt_token_ids"]),
                args.lead_prefill_tokens,
            )
        )
        llm._validate_and_add_requests(
            prompts=[lead_prompt],
            params=sampling,
            use_tqdm=False,
            lora_request=None,
            priority=None,
        )
        lead, lead_outputs = run_step("mixed_lead_prefill")
        require(not lead_outputs, "lead request finished during Prefill")
        require(
            int(lead["schedule"]["total_num_scheduled_tokens"])
            == args.lead_prefill_tokens,
            "lead Prefill schedule differs",
        )

        mixed_prompt = TokensPrompt(
            prompt_token_ids=expand_prompt(
                list(requests[2]["prompt_token_ids"]),
                args.mixed_prefill_tokens,
            )
        )
        llm._validate_and_add_requests(
            prompts=[mixed_prompt],
            params=sampling,
            use_tqdm=False,
            lora_request=None,
            priority=None,
        )
        hot_read_gate("runtime_qwen_probe_before_mixed_step")
        mixed, outputs = run_step("mixed_7167_prefill_plus_1_decode")
        schedule = mixed["schedule"]
        require(
            int(schedule["total_num_scheduled_tokens"])
            == args.pure_prefill_tokens,
            "mixed step token total differs from pure control",
        )
        require(
            len(schedule["new_request_ids"]) == 1
            and len(schedule["cached_request_ids"]) == 1,
            "scheduler did not form one-Prefill plus one-Decode mixed step",
        )

        for index in range(6):
            if not llm.llm_engine.has_unfinished_requests():
                break
            _, outputs = run_step(f"mixed_finish_{index + 1}")
        require(not llm.llm_engine.has_unfinished_requests(), "mixed requests remain")
        final = record_snapshot("completed")
        result["mixed_test"] = {"status": "ok"}
        result["final_hot_physical_residency"] = final["hot_physical_residency"]
        result["status"] = "ok"
        result["finished_epoch_seconds"] = time.time()
    except BaseException as exc:
        error = exc
        result["status"] = "exception"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        result["finished_epoch_seconds"] = time.time()
        append_jsonl(
            args.progress,
            {
                "event": "probe_exception",
                "epoch_seconds": time.time(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        if scheduler is not None and original_schedule is not None:
            try:
                scheduler.schedule = original_schedule
            except BaseException as exc:
                result.setdefault("cleanup_errors", []).append(repr(exc))
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
            except BaseException as exc:
                result.setdefault("cleanup_errors", []).append(repr(exc))
        llm = None
        gc.collect()
        try:
            import torch

            torch.cuda.memory._record_memory_history(enabled=None)
            torch.cuda.empty_cache()
        except BaseException as exc:
            result.setdefault("cleanup_errors", []).append(repr(exc))
        write_result(args.output, result)

    return 1 if error is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
