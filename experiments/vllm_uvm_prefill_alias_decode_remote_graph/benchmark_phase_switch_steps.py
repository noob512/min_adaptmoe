#!/usr/bin/env python3
"""Run one fail-closed Qwen original-VA phase-switch scheduler cycle.

Each process executes exactly one batch and one request cycle.  This is
intentional: once an H/G process changes its original parameter VAs from the
sealed alias topology to audited CPU-backed remote READ PTEs, a later prefill
would violate the serving contract.  The first decode step is kept separate
from warmup and steady-state decode so its one-time cutover is never folded
into the steady number.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PHASE_RUNNER = (
    REPO_ROOT
    / "experiments"
    / "vllm_p3_pinned_uva_phase_compare"
    / "benchmark_phase_steps.py"
)
GRAPH_RUNNER = (
    REPO_ROOT
    / "experiments"
    / "vllm_p3_static_alias_graph_integration"
    / "benchmark_full_graph_steps.py"
)

DUAL_BANK_BYTES = 1_207_959_552
LOGICAL_BYTES = 2_415_919_104
PARAMETER_COUNT = 8
ALIAS_COUNT = 6_912
ALIAS_BYTES = 3_623_878_656

ARM_CONTRACT = {
    "A": {
        "name": "alias_prefill_alias_eager_decode",
        "phase_mode": "off",
        "execution_mode": "eager",
        "graph": False,
        "prefill_topology": "sealed",
    },
    "B": {
        "name": "alias_prefill_alias_graph_decode",
        "phase_mode": "off",
        "execution_mode": "full_graph_decode",
        "graph": True,
        "prefill_topology": "sealed",
    },
    "H": {
        "name": "alias_prefill_remote_eager_decode",
        "phase_mode": "prefill_remote_decode",
        "execution_mode": "eager",
        "graph": False,
        "prefill_topology": "sealed",
    },
    "G": {
        "name": "alias_prefill_remote_graph_decode",
        "phase_mode": "prefill_remote_decode",
        "execution_mode": "full_graph_decode",
        "graph": True,
        "prefill_topology": "sealed",
    },
    "R": {
        "name": "remote_prefill_remote_eager_decode",
        "phase_mode": "remote_only",
        "execution_mode": "eager",
        "graph": False,
        "prefill_topology": "remote",
    },
    "C": {
        "name": "remote_prefill_remote_graph_decode",
        "phase_mode": "remote_only",
        "execution_mode": "full_graph_decode",
        "graph": True,
        "prefill_topology": "remote",
    },
    "D": {
        "name": "alias_graph_prefill_remote_graph_decode",
        "phase_mode": "prefill_graph_remote_decode",
        "execution_mode": "full_graph_separate",
        "graph": True,
        "prefill_graph": True,
        "prefill_topology": "sealed",
    },
    "E": {
        "name": "alias_graph_prefill_alias_graph_decode",
        "phase_mode": "off",
        "execution_mode": "full_graph_separate",
        "graph": True,
        "prefill_graph": True,
        "prefill_topology": "sealed",
    },
    "F": {
        "name": "remote_graph_prefill_remote_graph_decode",
        "phase_mode": "remote_only",
        "execution_mode": "full_graph_separate",
        "graph": True,
        "prefill_graph": True,
        "prefill_topology": "remote",
    },
}


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--exact-names", required=True, type=Path)
    parser.add_argument("--offload-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", choices=tuple(ARM_CONTRACT), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 4, 8, 16), required=True)
    parser.add_argument("--cudagraph-capture-sizes", default="1,4")
    parser.add_argument("--warmup-decode-steps", type=int, default=5)
    parser.add_argument("--measured-decode-steps", type=int, default=16)
    parser.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    parser.add_argument("--cpu-offload-gb", type=float, default=3.0)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--audit-trace", action="store_true")
    parser.add_argument(
        "--park-resume-roundtrip",
        action="store_true",
        help="run a second short request to prove remote->alias->remote reuse",
    )
    parser.add_argument(
        "--dynamic-plan-sequence",
        type=Path,
        help=(
            "JSON request-boundary plan sequence; each record supplies plan, "
            "selected bytes, batch size, and predicted KV pressure"
        ),
    )
    parser.add_argument(
        "--dynamic-kv-sequence",
        type=Path,
        help=(
            "JSON next-request workload sequence; EngineCore derives each "
            "physical KV target while every KV tensor VA remains fixed"
        ),
    )
    parser.add_argument(
        "--allow-explicit-dynamic-kv-targets",
        action="store_true",
        help=(
            "qualification-only mode: allow each dynamic-KV record to name an "
            "exact physical target; the default automatic-pressure contract "
            "continues to reject target_blocks"
        ),
    )
    parser.add_argument(
        "--incremental-expert-wave-plan",
        type=Path,
        help=(
            "qualification-only ordered per-layer expert waves used by an "
            "explicit dynamic-KV sequence"
        ),
    )
    parser.add_argument(
        "--physical-kv-reuse-probe-bytes",
        type=int,
        default=0,
        help=(
            "after the final physical-KV resize, allocate and touch this many "
            "ordinary CUDA bytes to prove released HBM is reusable"
        ),
    )
    parser.add_argument(
        "--kv-autoscale-roundtrip",
        action="store_true",
        help=(
            "after the main request, run a short Prefill/Decode request that "
            "must drive the unified KV controller through its low-pressure "
            "shrink path"
        ),
    )
    parser.add_argument(
        "--kv-multilevel-pressure-2-4-6",
        action="store_true",
        help=(
            "qualification-only mode: measure the initial 2 GiB tier, then "
            "run two real backup-pressure requests that must rebalance to "
            "the frozen 4 and 6 GiB KV/expert tiers"
        ),
    )
    parser.add_argument(
        "--kv-multilevel-trigger-mode",
        choices=("pressure", "direct"),
        default="pressure",
        help=(
            "for the 2/4/6 GiB qualification, either let real backup-queue "
            "pressure trigger each grow inside Prefill, or apply the same "
            "frozen tier transaction directly at the idle request boundary"
        ),
    )
    parser.add_argument(
        "--kv-multilevel-tier-specs",
        default="",
        help=(
            "optional ordered gib:target_blocks:plan_class records separated "
            "by commas; default is 2:1364:cold70,4:2730:cold80,6:4095:cold89"
        ),
    )
    parser.add_argument(
        "--kv-autoscale-low-prompt-tokens",
        type=int,
        default=16,
        help="prompt-token count for the low-pressure autoscale request",
    )
    parser.add_argument(
        "--kv-autoscale-low-output-tokens",
        type=int,
        default=5,
        help="generated-token count for the low-pressure autoscale request",
    )
    parser.add_argument(
        "--kv-path-direct-high-preset",
        action="store_true",
        help=(
            "before the conditioning request, place one cold expert per layer "
            "in the runtime-remote set without changing the KV target"
        ),
    )
    parser.add_argument(
        "--kv-path-high-performance-requests",
        type=int,
        default=0,
        help="number of post-conditioning requests measured in the high state",
    )
    parser.add_argument(
        "--kv-path-low-performance-requests",
        type=int,
        default=0,
        help="number of requests measured in the final low state",
    )
    parser.add_argument(
        "--kv-path-performance-output-tokens",
        type=int,
        default=23,
        help="fixed generated-token count for each path-parity request",
    )
    parser.add_argument(
        "--kv-path-performance-warmup-decode-steps",
        type=int,
        default=5,
        help="decode steps excluded before path-parity steady samples",
    )
    parser.add_argument("--expert-plan", type=Path)
    parser.add_argument("--expected-parameter-count", type=int, default=PARAMETER_COUNT)
    parser.add_argument(
        "--expected-full-parameter-bytes", type=int, default=LOGICAL_BYTES
    )
    parser.add_argument("--expected-selected-bytes", type=int, default=LOGICAL_BYTES)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_multilevel_tier_specs(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    if not text:
        text = "2:1364:cold70,4:2730:cold80,6:4095:cold89"
    result: list[dict[str, Any]] = []
    for item in text.split(","):
        fields = item.strip().split(":")
        if len(fields) != 3:
            raise ValueError(f"invalid multilevel tier specification: {item!r}")
        gib, target = int(fields[0]), int(fields[1])
        class_name = fields[2].strip()
        if gib <= 0 or target <= 0 or not class_name:
            raise ValueError(f"invalid multilevel tier specification: {item!r}")
        result.append(
            {"tier_gib": gib, "target_blocks": target, "plan_class": class_name}
        )
    if len(result) < 2:
        raise ValueError("multilevel tier sequence requires at least two tiers")
    if [item["tier_gib"] for item in result] != sorted(
        {item["tier_gib"] for item in result}
    ):
        raise ValueError("multilevel GiB tiers must be unique and increasing")
    if [item["target_blocks"] for item in result] != sorted(
        {item["target_blocks"] for item in result}
    ):
        raise ValueError("multilevel targets must be unique and increasing")
    if len({item["plan_class"] for item in result}) != len(result):
        raise ValueError("multilevel plan classes must be unique")
    return result


def cold_class_width(class_name: str) -> int:
    if not class_name.startswith("cold") or not class_name[4:].isdigit():
        raise ValueError(f"plan class has no cold width: {class_name!r}")
    return int(class_name[4:])


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


def kv_physical_snapshot(model_runner: Any, target_blocks: int) -> dict[str, Any]:
    """Audit reserved KV VAs and their current GPU/CPU physical residency."""
    from vllm.device_allocator.uvm import (
        driver_batch_query_ranges_residency_bytes,
    )
    from vllm.v1.core.uvm_kv_runtime_policy import (
        snapshot_uvm_kv_cache_physical_spans,
    )

    raw_spans = snapshot_uvm_kv_cache_physical_spans()
    require(raw_spans, "no managed KV physical spans are registered")
    unique: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    for span in raw_spans:
        key = (
            int(span["ptr"]),
            int(span["size_bytes"]),
            int(span["page_size"]),
            int(span["max_block_id"]),
        )
        unique.setdefault(key, dict(span))
    spans = sorted(unique.values(), key=lambda item: int(item["ptr"]))
    max_blocks = {int(item["max_block_id"]) for item in spans}
    require(len(max_blocks) == 1, "managed KV spans disagree on block capacity")
    total_blocks = next(iter(max_blocks))
    target = min(max(int(target_blocks), 0), total_blocks)

    full_ranges: list[tuple[int, int]] = []
    head_ranges: list[tuple[int, int]] = []
    tail_ranges: list[tuple[int, int]] = []
    records: list[dict[str, Any]] = []
    for item in spans:
        ptr = int(item["ptr"])
        size_bytes = int(item["size_bytes"])
        page_size = int(item["page_size"])
        covered_bytes = min(size_bytes, (total_blocks + 1) * page_size)
        head_bytes = min(covered_bytes, (target + 1) * page_size)
        tail_bytes = covered_bytes - head_bytes
        full_ranges.append((ptr, covered_bytes))
        if head_bytes:
            head_ranges.append((ptr, head_bytes))
        if tail_bytes:
            tail_ranges.append((ptr + head_bytes, tail_bytes))
        records.append(
            {
                "span_id": str(item["span_id"]),
                "ptr": ptr,
                "size_bytes": size_bytes,
                "page_size": page_size,
                "max_block_id": total_blocks,
                "head_bytes": head_bytes,
                "tail_bytes": tail_bytes,
            }
        )

    def query(ranges: list[tuple[int, int]]) -> dict[str, Any]:
        value = dict(
            driver_batch_query_ranges_residency_bytes(
                ranges,
                device=0,
                wait_for_trackers=False,
            )
        )
        value["range_count"] = len(ranges)
        value["requested_bytes"] = sum(length for _, length in ranges)
        return value

    va_payload = json.dumps(
        [(int(item["ptr"]), int(item["size_bytes"])) for item in records],
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": "vllm_uvm_kv_physical_snapshot_v1",
        "span_count": len(records),
        "spans": records,
        "va_sha256": hashlib.sha256(va_payload).hexdigest(),
        "reserved_va_bytes": sum(int(item["size_bytes"]) for item in records),
        "covered_bytes": sum(length for _, length in full_ranges),
        "total_blocks": total_blocks,
        "target_blocks": target,
        "full": query(full_ranges),
        "head": query(head_ranges),
        "tail": query(tail_ranges),
    }


def validate_kv_physical_snapshot(snapshot: dict[str, Any]) -> None:
    """Require the active KV prefix on GPU and the inactive tail on CPU."""
    full = snapshot["full"]
    head = snapshot["head"]
    tail = snapshot["tail"]
    for label, record in (("full", full), ("head", head), ("tail", tail)):
        require(record.get("ok") is True, f"KV {label} residency query failed")
        require(
            int(record.get("unknown_bytes", -1)) == 0,
            f"KV {label} residency contains unknown bytes",
        )
    require(
        int(head.get("gpu_bytes", -1)) == int(head.get("requested_bytes", -2))
        and int(head.get("cpu_bytes", -1)) == 0,
        "active KV prefix is not completely GPU resident",
    )
    require(
        int(tail.get("cpu_bytes", -1)) == int(tail.get("requested_bytes", -2))
        and int(tail.get("gpu_bytes", -1)) == 0,
        "inactive KV tail is not completely CPU resident",
    )
    require(
        int(full.get("gpu_bytes", -1)) + int(full.get("cpu_bytes", -1))
        == int(full.get("requested_bytes", -2)),
        "complete KV VA reservation is not physically accounted for",
    )


def load_exact_names(path: Path, expected_count: int) -> tuple[str, ...]:
    names = tuple(line.strip() for line in path.read_text().splitlines() if line.strip())
    require(
        len(names) == expected_count,
        f"exact-name input must contain {expected_count} names",
    )
    require(len(set(names)) == len(names), "exact-name input contains duplicates")
    return tuple(sorted(names))


def load_dynamic_plan_sequence(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), "dynamic plan sequence is absent")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid dynamic plan sequence JSON: {exc}") from exc
    require(isinstance(payload, list) and payload, "dynamic plan sequence is empty")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        require(isinstance(raw, dict), f"dynamic plan record {index} is not an object")
        label = raw.get("label")
        plan_value = raw.get("plan")
        expected_bytes = raw.get("expected_selected_bytes")
        batch_size = raw.get("batch_size")
        max_tokens = raw.get("max_tokens", 2)
        threshold = raw.get("pressure_threshold_tokens")
        expected_class = raw.get("expected_pressure_class")
        require(isinstance(label, str) and label, f"dynamic plan {index} has no label")
        require(isinstance(plan_value, str) and plan_value, f"dynamic plan {index} has no path")
        plan = Path(plan_value)
        if not plan.is_absolute():
            plan = path.parent / plan
        plan = plan.resolve()
        require(plan.is_file(), f"dynamic plan {index} file is absent: {plan}")
        require(
            isinstance(expected_bytes, int)
            and not isinstance(expected_bytes, bool)
            and expected_bytes > 0,
            f"dynamic plan {index} has invalid expected bytes",
        )
        require(batch_size in (1, 4, 8, 16), f"dynamic plan {index} has invalid batch")
        require(
            isinstance(max_tokens, int)
            and not isinstance(max_tokens, bool)
            and max_tokens >= 2,
            f"dynamic plan {index} has invalid max_tokens",
        )
        require(
            isinstance(threshold, int)
            and not isinstance(threshold, bool)
            and threshold > 0,
            f"dynamic plan {index} has invalid pressure threshold",
        )
        require(
            expected_class in ("low", "high"),
            f"dynamic plan {index} has invalid expected pressure class",
        )
        records.append(
            {
                "label": label,
                "plan": plan,
                "expected_selected_bytes": expected_bytes,
                "batch_size": batch_size,
                "max_tokens": max_tokens,
                "pressure_threshold_tokens": threshold,
                "expected_pressure_class": expected_class,
            }
        )
    return records


def load_dynamic_kv_sequence(
    path: Path,
    *,
    allow_explicit_targets: bool = False,
) -> list[dict[str, Any]]:
    """Load next-request workloads and optional qualification-only targets."""
    require(path.is_file(), "dynamic KV sequence is absent")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid dynamic KV sequence JSON: {exc}") from exc
    require(isinstance(payload, list) and payload, "dynamic KV sequence is empty")
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        require(isinstance(raw, dict), f"dynamic KV record {index} is not an object")
        label = raw.get("label")
        batch_size = raw.get("batch_size")
        max_tokens = raw.get("max_tokens", 23)
        expected_class = raw.get("expected_pressure_class")
        warmup_steps = raw.get("decode_warmup_steps", 5)
        target_blocks = raw.get("target_blocks")
        expert_waves = raw.get("expert_waves", 0)
        require(isinstance(label, str) and label, f"dynamic KV {index} has no label")
        require(batch_size in (1, 4, 8, 16), f"dynamic KV {index} has invalid batch")
        require(
            isinstance(max_tokens, int)
            and not isinstance(max_tokens, bool)
            and max_tokens >= 3,
            f"dynamic KV {index} has invalid max_tokens",
        )
        if allow_explicit_targets:
            require(
                isinstance(target_blocks, int)
                and not isinstance(target_blocks, bool)
                and target_blocks >= 0,
                f"dynamic KV {index} has invalid explicit target_blocks",
            )
            require(
                isinstance(expert_waves, int)
                and not isinstance(expert_waves, bool)
                and expert_waves >= 0,
                f"dynamic KV {index} has invalid expert_waves",
            )
        else:
            require(
                "target_blocks" not in raw,
                f"dynamic KV {index} must not prescribe target_blocks",
            )
            require(
                "expert_waves" not in raw,
                f"dynamic KV {index} must not prescribe expert_waves",
            )
        require(
            expected_class in (
                ("explicit",) if allow_explicit_targets else ("low", "high")
            ),
            f"dynamic KV {index} has invalid expected pressure class",
        )
        require(
            isinstance(warmup_steps, int)
            and not isinstance(warmup_steps, bool)
            and warmup_steps >= 0
            and max_tokens >= warmup_steps + 3,
            f"dynamic KV {index} leaves no measured decode steps",
        )
        records.append(
            {
                "label": label,
                "batch_size": batch_size,
                "max_tokens": max_tokens,
                "expected_pressure_class": expected_class,
                "decode_warmup_steps": warmup_steps,
                "target_blocks": (
                    int(target_blocks) if allow_explicit_targets else None
                ),
                "expert_waves": int(expert_waves),
            }
        )
    return records


def load_incremental_expert_waves(path: Path) -> list[dict[str, int]]:
    """Load deterministic all-layer waves for joint KV/expert qualification."""
    require(path.is_file(), "incremental expert wave plan is absent")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid incremental expert wave JSON: {exc}") from exc
    require(
        isinstance(payload, dict)
        and payload.get("schema") == "qwen_large_kv_incremental_expert_waves_v1",
        "incremental expert wave schema differs",
    )
    raw_waves = payload.get("waves")
    require(isinstance(raw_waves, list) and raw_waves, "expert wave list is empty")
    waves: list[dict[str, int]] = []
    expected_layers: set[str] | None = None
    seen_by_layer: dict[str, set[int]] = {}
    for index, raw in enumerate(raw_waves, 1):
        require(
            isinstance(raw, dict) and int(raw.get("wave", -1)) == index,
            f"expert wave {index} has invalid generation",
        )
        layer_experts = raw.get("layer_experts")
        require(isinstance(layer_experts, dict), f"expert wave {index} is invalid")
        normalized = {
            str(layer): int(expert) for layer, expert in layer_experts.items()
        }
        require(len(normalized) == 48, f"expert wave {index} is not all-layer")
        if expected_layers is None:
            expected_layers = set(normalized)
        require(
            set(normalized) == expected_layers,
            f"expert wave {index} changes the layer set",
        )
        for layer, expert in normalized.items():
            require(0 <= expert < 128, f"expert wave {index} has invalid expert id")
            prior = seen_by_layer.setdefault(layer, set())
            require(expert not in prior, f"expert wave {index} repeats an expert")
            prior.add(expert)
        waves.append(normalized)
    return waves


def model_runner_from_llm(llm: Any) -> Any:
    client = llm.llm_engine.engine_core
    core = getattr(client, "engine_core", None)
    require(core is not None, "benchmark requires VLLM_ENABLE_V1_MULTIPROCESSING=0")
    executor = core.model_executor
    wrapper = executor.driver_worker
    worker = wrapper.worker
    require(worker is not None and worker.model_runner is not None, "model runner absent")
    return worker.model_runner


def incremental_expert_state_snapshot(model_runner: Any) -> dict[str, Any]:
    """Summarize Stage-M expert placement without serializing every expert."""
    layers: dict[str, dict[str, Any]] = {}
    remote_regions: set[str] = set()
    offloaded_regions: set[str] = set()
    for module in model_runner.get_model().modules():
        records = getattr(module, "_uvm_expert_records", None)
        if not isinstance(records, dict):
            continue
        layer_key_fn = getattr(
            model_runner,
            "_uvm_incremental_expert_layer_key",
            None,
        )
        layer_key = (
            str(layer_key_fn(module))
            if callable(layer_key_fn)
            else str(getattr(module, "layer_name", ""))
        )
        remote_ids = sorted(
            int(value)
            for value in getattr(
                module,
                "_uvm_remote_read_global_expert_ids",
                set(),
            )
        )
        offloaded_ids = sorted(
            int(value)
            for value in getattr(module, "_uvm_offloaded_global_expert_ids", set())
        )
        for (layer_id, expert_id), record in records.items():
            del layer_id
            region_ids = {
                str(value)
                for value in getattr(record, "envelope_region_ids", ())
            }
            if int(expert_id) in remote_ids:
                remote_regions.update(region_ids)
            if int(expert_id) in offloaded_ids:
                offloaded_regions.update(region_ids)
        layers[layer_key] = {
            "remote_expert_ids": remote_ids,
            "offloaded_expert_ids": offloaded_ids,
            "registered_expert_count": len(records),
            "access_mode": str(
                getattr(module, "uvm_expert_offload_access_mode", "")
            ),
        }

    def region_bytes(region_ids: set[str]) -> int:
        total = 0
        for region_id in region_ids:
            try:
                total += int(region_id.rsplit(":", 1)[-1], 16)
            except ValueError:
                continue
        return total

    return {
        "schema": "vllm_uvm_incremental_expert_state_v1",
        "layer_count": len(layers),
        "remote_expert_count": sum(
            len(record["remote_expert_ids"]) for record in layers.values()
        ),
        "offloaded_expert_count": sum(
            len(record["offloaded_expert_ids"]) for record in layers.values()
        ),
        "remote_region_count": len(remote_regions),
        "offloaded_region_count": len(offloaded_regions),
        "remote_region_bytes": region_bytes(remote_regions),
        "offloaded_region_bytes": region_bytes(offloaded_regions),
        "layers": layers,
    }


def parameter_snapshot(
    llm: Any,
    exact_names: tuple[str, ...],
    *,
    expected_count: int,
    expected_full_bytes: int,
) -> dict[str, Any]:
    model = model_runner_from_llm(llm).get_model()
    parameters = dict(model.named_parameters())
    missing = sorted(set(exact_names) - set(parameters))
    require(not missing, f"selected parameters disappeared: {missing!r}")
    records: list[dict[str, Any]] = []
    for name in exact_names:
        value = parameters[name]
        records.append(
            {
                "name": name,
                "data_ptr": int(value.data_ptr()),
                "bytes": int(value.numel() * value.element_size()),
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        )
    require(
        sum(item["bytes"] for item in records) == expected_full_bytes,
        "full parameter byte total differs",
    )
    pointers = [item["data_ptr"] for item in records]
    require(len(set(pointers)) == expected_count, "parameter VAs are not unique")
    return {
        "count": len(records),
        "bytes": sum(item["bytes"] for item in records),
        "vas": pointers,
        "records": records,
    }


def _compact(value: Any, *, depth: int = 0) -> Any:
    """Keep audit data while replacing huge per-PTE status arrays."""
    if depth > 12:
        return "<max-depth>"
    if isinstance(value, dict):
        return {
            str(key): _compact(item, depth=depth + 1)
            for key, item in value.items()
            if str(key) not in {"entry_statuses", "page_statuses"}
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 128:
            encoded = json.dumps(value, sort_keys=True, default=str).encode()
            return {
                "compacted_sequence": True,
                "count": len(value),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        return [_compact(item, depth=depth + 1) for item in value]
    return value


def diagnostics_snapshot(utils: Any, driver_getter: Any) -> dict[str, Any]:
    return {
        "model": _compact(utils.get_cpu_offload_graph_static_alias_diagnostics()),
        "driver": _compact(driver_getter()),
    }


def plan_class_state_snapshot(utils: Any) -> dict[str, Any]:
    """Record the small plan-class state needed by autoscale qualification."""
    diagnostics = utils.get_cpu_offload_graph_static_alias_diagnostics()
    return {
        "active_plan_class": diagnostics.get("active_plan_class"),
        "initial_plan_class": diagnostics.get("initial_plan_class"),
        "plan_generation": diagnostics.get("plan_generation"),
        "selected_parameter_bytes_per_forward": diagnostics.get(
            "selected_parameter_bytes_per_forward"
        ),
        "backing_bytes": diagnostics.get("backing_bytes"),
        "stable_envelope_bytes": diagnostics.get("stable_envelope_bytes"),
        "lifecycle_state": diagnostics.get("lifecycle_state"),
        "session_id": diagnostics.get("session_id"),
    }


def read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid manifest JSON at line {line_number}: {exc}") from exc
        require(isinstance(value, dict), f"manifest line {line_number} is not an object")
        records.append(value)
    return records


def summarize_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    forwards = [item for item in records if item.get("event") == "cpu_offload_copy_forward"]
    with_actual = [
        item for item in forwards if isinstance(item.get("graph_static_alias_actual_h2d_bytes"), int)
    ]
    return {
        "record_count": len(records),
        "event_counts": dict(sorted(Counter(str(x.get("event")) for x in records).items())),
        "forward_count": len(forwards),
        "actual_h2d_field_count": len(with_actual),
        "actual_h2d_bytes": sum(
            int(item["graph_static_alias_actual_h2d_bytes"]) for item in with_actual
        ),
        "logical_copy_bytes": sum(int(item.get("copy_bytes", 0)) for item in forwards),
        "copy_submitted_counts": dict(
            sorted(
                Counter(
                    str(bool(item.get("graph_static_alias_copy_submitted"))).lower()
                    for item in with_actual
                ).items()
            )
        ),
        "lifecycle_state_counts": dict(
            sorted(Counter(str(item.get("graph_static_alias_lifecycle_state")) for item in forwards).items())
        ),
        "capture_phase_counts": dict(
            sorted(Counter(str(bool(item.get("graph_static_alias_capture_phase"))).lower() for item in forwards).items())
        ),
        "capture_active_counts": dict(
            sorted(Counter(str(bool(item.get("graph_static_alias_capture_active"))).lower() for item in forwards).items())
        ),
        "module_prefixes": sorted({str(item.get("module_prefix")) for item in forwards}),
        "transition_reasons": [
            str(item.get("reason"))
            for item in records
            if item.get("event") == "cpu_offload_uvm_original_va_remote_transition"
        ],
        "records": _compact(records),
    }


def transition_delta(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    before_history = before["model"].get("phase_transition_history", [])
    after_history = after["model"].get("phase_transition_history", [])
    require(isinstance(before_history, list), "before transition history is invalid")
    require(isinstance(after_history, list), "after transition history is invalid")
    require(len(after_history) >= len(before_history), "transition history moved backwards")
    return [dict(item) for item in after_history[len(before_history) :] if isinstance(item, dict)]


def summarize(values: list[float]) -> dict[str, float | int]:
    require(bool(values), "cannot summarize an empty sample")
    require(all(math.isfinite(x) and x > 0 for x in values), "invalid timing sample")
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": ordered[min(len(ordered) - 1, math.ceil(0.90 * len(ordered)) - 1)],
    }


def selected_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(("VLLM_CPU_OFFLOAD_", "VLLM_UVM_"))
        or key
        in {
            "CUDA_VISIBLE_DEVICES",
            "VLLM_ATTENTION_BACKEND",
            "VLLM_ENABLE_V1_MULTIPROCESSING",
            "VLLM_FUSED_MOE_CHUNK_SIZE",
            "VLLM_FLASHINFER_AUTOTUNE_MAX_TOKENS",
        }
    }


def main() -> int:
    args = parse_args()
    kv_path_performance_enabled = bool(
        args.kv_path_direct_high_preset
        or args.kv_path_high_performance_requests
        or args.kv_path_low_performance_requests
    )
    three_queue_qualification = (
        os.environ.get("VLLM_UVM_THREE_QUEUE_QUALIFICATION", "0")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )
    tiered_plan_class_qualification = bool(
        three_queue_qualification
        and os.environ.get(
            "VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_BEFORE_GROW", "0"
        )
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )
    multilevel_pressure_qualification = bool(
        args.kv_multilevel_pressure_2_4_6
    )
    multilevel_trigger_mode = str(args.kv_multilevel_trigger_mode)
    multilevel_tier_specs = parse_multilevel_tier_specs(
        str(args.kv_multilevel_tier_specs)
    )
    dynamic_specs = (
        load_dynamic_plan_sequence(args.dynamic_plan_sequence)
        if args.dynamic_plan_sequence is not None
        else []
    )
    dynamic_kv_specs = (
        load_dynamic_kv_sequence(
            args.dynamic_kv_sequence,
            allow_explicit_targets=args.allow_explicit_dynamic_kv_targets,
        )
        if args.dynamic_kv_sequence is not None
        else []
    )
    incremental_expert_waves = (
        load_incremental_expert_waves(args.incremental_expert_wave_plan)
        if args.incremental_expert_wave_plan is not None
        else []
    )
    require(
        not args.allow_explicit_dynamic_kv_targets or bool(dynamic_kv_specs),
        "explicit physical-KV targets require a dynamic-KV sequence",
    )
    require(
        multilevel_pressure_qualification
        or (
            multilevel_trigger_mode == "pressure"
            and not str(args.kv_multilevel_tier_specs).strip()
        ),
        "multilevel options are only valid with the multilevel qualification",
    )
    require(
        not incremental_expert_waves
        or (args.allow_explicit_dynamic_kv_targets and bool(dynamic_kv_specs)),
        "incremental expert waves require an explicit dynamic-KV sequence",
    )
    require(
        not dynamic_kv_specs
        or max(int(spec["expert_waves"]) for spec in dynamic_kv_specs)
        <= len(incremental_expert_waves),
        "dynamic-KV sequence exceeds the expert wave plan",
    )
    require(
        incremental_expert_waves
        or all(int(spec["expert_waves"]) == 0 for spec in dynamic_kv_specs),
        "dynamic-KV sequence requests expert waves without a wave plan",
    )
    require(
        not (dynamic_specs and dynamic_kv_specs),
        "expert-plan and physical-KV sequences are mutually exclusive",
    )
    require(
        not (three_queue_qualification and (dynamic_specs or dynamic_kv_specs)),
        "three-queue qualification requires one isolated request cycle",
    )
    require(
        not (
            multilevel_pressure_qualification
            and (
                dynamic_specs
                or dynamic_kv_specs
                or args.kv_autoscale_roundtrip
                or args.park_resume_roundtrip
                or kv_path_performance_enabled
            )
        ),
        "multilevel pressure qualification requires an isolated request sequence",
    )
    phase = load_module("_phase_switch_phase_helper", PHASE_RUNNER)
    graph = load_module("_phase_switch_graph_helper", GRAPH_RUNNER)
    contract = ARM_CONTRACT[args.arm]
    capture_sizes = phase.parse_positive_ints(args.cudagraph_capture_sizes)
    require(args.batch_size in capture_sizes, "decode batch has no FULL Graph size")
    require(args.warmup_decode_steps >= 1, "at least one decode warmup is required")
    require(args.measured_decode_steps >= 1, "at least one measured decode is required")
    require(args.max_num_seqs >= args.batch_size, "max-num-seqs is below batch size")
    require(args.expected_parameter_count > 0, "bad expected parameter count")
    require(args.expected_full_parameter_bytes > 0, "bad full parameter bytes")
    require(args.expected_selected_bytes > 0, "bad selected bytes")
    require(
        args.physical_kv_reuse_probe_bytes >= 0,
        "bad physical KV reuse probe size",
    )
    require(
        args.kv_autoscale_low_prompt_tokens > 0,
        "bad autoscale low-pressure prompt size",
    )
    require(
        args.kv_autoscale_low_output_tokens >= 4,
        "autoscale low-pressure request needs at least four output tokens",
    )
    require(
        args.kv_path_high_performance_requests >= 0
        and args.kv_path_low_performance_requests >= 0,
        "KV path performance request counts must be non-negative",
    )
    require(
        args.kv_path_performance_warmup_decode_steps >= 0,
        "KV path performance warmup count must be non-negative",
    )
    require(
        args.kv_path_performance_output_tokens
        >= 3 + args.kv_path_performance_warmup_decode_steps,
        "KV path performance request has no measured Decode step",
    )
    if kv_path_performance_enabled:
        require(args.arm == "D", "KV path performance requires arm D")
        require(
            not (dynamic_specs or dynamic_kv_specs),
            "KV path performance requires an isolated fixed-plan run",
        )
        require(
            os.environ.get("VLLM_UVM_KV_CACHE_MANAGED", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
            and os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_ENABLE", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            "KV path performance requires the managed dynamic pool",
        )
    if args.kv_path_direct_high_preset:
        require(
            os.environ.get("VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            "direct-high preset requires incremental expert offload",
        )
    if args.physical_kv_reuse_probe_bytes:
        require(
            bool(dynamic_kv_specs),
            "physical KV reuse probe requires a dynamic KV sequence",
        )
    require(math.isfinite(args.cpu_offload_gb) and args.cpu_offload_gb > 0, "bad offload budget")
    if args.park_resume_roundtrip:
        require(args.arm == "D", "park/resume roundtrip requires arm D")
        require(
            not dynamic_kv_specs,
            "physical-KV sequence and fixed-plan roundtrip are mutually exclusive",
        )
        require(
            os.environ.get(
                "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME", "0"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            "park/resume roundtrip requires the reversible topology switch",
        )
    if args.kv_autoscale_roundtrip:
        require(args.arm == "D", "KV autoscale roundtrip requires arm D")
        require(
            not (dynamic_specs or dynamic_kv_specs),
            "KV autoscale roundtrip requires an isolated fixed-plan run",
        )
        require(
            os.environ.get("VLLM_UVM_KV_CACHE_MANAGED", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            "KV autoscale roundtrip requires managed KV allocations",
        )
        require(
            os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_ENABLE", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            "KV autoscale roundtrip requires the dynamic pool",
        )
        require(
            os.environ.get(
                "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME",
                "0",
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            "KV autoscale roundtrip requires reversible topology switching",
        )
    if multilevel_pressure_qualification:
        require(args.arm == "D", "multilevel pressure qualification requires arm D")
        require(args.batch_size == 1, "multilevel pressure qualification requires batch 1")
        require(
            os.environ.get("VLLM_UVM_KV_CACHE_MANAGED", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
            and os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_ENABLE", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            "multilevel pressure qualification requires the managed dynamic pool",
        )
        require(
            os.environ.get("VLLM_UVM_KV_AUTO_SCALE_ENABLE", "0")
            .strip()
            .lower()
            not in {"1", "true", "yes", "on"},
            "initial 2 GiB measurement must start with autoscale disabled",
        )
        require(
            os.environ.get(
                "VLLM_UVM_KV_BACKUP_PRESSURE_STANDALONE_GROW_ENABLE", "1"
            )
            .strip()
            .lower()
            not in {"1", "true", "yes", "on"},
            "initial 2 GiB measurement must defer standalone backup growth",
        )
        require(
            os.environ.get("VLLM_UVM_KV_AUTO_SCALE_BACKUP_ONLY", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
            and os.environ.get(
                "VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_BEFORE_GROW", "0"
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
            and os.environ.get(
                "VLLM_UVM_KV_AUTO_SCALE_RESUME_PREFILL_BEFORE_PLAN_CLASS", "0"
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            "multilevel pressure qualification requires backup-only "
            "resume-before-class-before-KV growth",
        )
        require(
            bool(
                os.environ.get(
                    "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE", ""
                ).strip()
            )
            and bool(
                os.environ.get(
                    "VLLM_UVM_KV_BACKUP_RESERVE_TARGET_MAP", ""
                ).strip()
            ),
            "multilevel pressure qualification requires frozen class/reserve maps",
        )
    if dynamic_specs:
        require(args.arm == "D", "dynamic plan rebuild requires arm D")
        require(
            not args.park_resume_roundtrip,
            "dynamic plan sequence and fixed-plan roundtrip are mutually exclusive",
        )
        require(
            os.environ.get(
                "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME", "0"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            "dynamic plan rebuild requires the reversible topology switch",
        )
        require(
            all(spec["batch_size"] in capture_sizes for spec in dynamic_specs),
            "a dynamic decode batch has no FULL Graph size",
        )
        require(
            all(spec["batch_size"] <= args.max_num_seqs for spec in dynamic_specs),
            "a dynamic batch exceeds max-num-seqs",
        )
    if dynamic_kv_specs:
        require(args.arm == "D", "physical KV resize requires arm D")
        require(
            os.environ.get("VLLM_UVM_KV_CACHE_MANAGED", "0").strip().lower()
            in {"1", "true", "yes", "on"},
            "physical KV resize requires managed KV allocations",
        )
        require(
            os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_ENABLE", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            "physical KV resize requires the dynamic pool",
        )
        require(
            os.environ.get(
                "VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_ENABLE", "1"
            )
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            "physical KV resize requires physical migration",
        )
        require(
            all(spec["batch_size"] in capture_sizes for spec in dynamic_kv_specs),
            "a physical-KV decode batch has no FULL Graph size",
        )
        require(
            all(spec["batch_size"] <= args.max_num_seqs for spec in dynamic_kv_specs),
            "a physical-KV batch exceeds max-num-seqs",
        )
        if incremental_expert_waves:
            require(
                os.environ.get("VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE", "0")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"},
                "expert-wave qualification requires incremental expert offload",
            )
    if three_queue_qualification:
        require(args.arm == "D", "three-queue qualification requires arm D")
        require(
            os.environ.get("VLLM_UVM_KV_CACHE_MANAGED", "0").strip().lower()
            in {"1", "true", "yes", "on"},
            "three-queue qualification requires managed KV allocations",
        )
        require(
            os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_ENABLE", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"},
            "three-queue qualification requires the dynamic pool",
        )
        require(
            int(os.environ.get("VLLM_UVM_KV_BACKUP_RESERVE_BLOCKS", "0")) > 0,
            "three-queue qualification requires a non-empty backup reserve",
        )
        require(
            int(os.environ.get("VLLM_UVM_KV_BACKUP_GROW_BLOCKS", "0")) > 0,
            "three-queue qualification requires backup-triggered growth",
        )
        incremental_enabled = (
            os.environ.get("VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE", "0")
            .strip()
            .lower()
            in {"1", "true", "yes", "on"}
        )
        if tiered_plan_class_qualification:
            require(
                not incremental_enabled,
                "tiered qualification must disable legacy incremental waves",
            )
            require(
                os.environ.get("VLLM_UVM_KV_AUTO_SCALE_BACKUP_ONLY", "0")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"},
                "tiered qualification requires backup-only growth",
            )
            require(
                bool(
                    os.environ.get(
                        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE",
                        "",
                    ).strip()
                ),
                "tiered qualification requires frozen plan classes",
            )
        else:
            require(
                incremental_enabled,
                "three-queue qualification requires incremental expert offload",
            )
    require(args.exact_names.is_file(), "exact-name file is absent")
    exact_names = load_exact_names(args.exact_names, args.expected_parameter_count)
    configured_plan = os.environ.get(
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE", ""
    ).strip()
    if args.expert_plan is None:
        require(not configured_plan, "unexpected expert-plan environment")
    else:
        require(args.expert_plan.is_file(), "expert-plan file is absent")
        require(
            bool(configured_plan)
            and Path(configured_plan).resolve() == args.expert_plan.resolve(),
            "expert-plan environment differs from argument",
        )
    expected_phase_mode = str(contract["phase_mode"])
    require(
        os.environ.get("VLLM_UVM_ORIGINAL_VA_PHASE_MODE", "").strip().lower()
        == expected_phase_mode,
        "phase-mode environment differs from arm",
    )
    alias_eager = os.environ.get(
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    require(
        alias_eager == (args.arm in ("A", "B", "E")),
        "alias-eager environment differs from arm",
    )
    trace_env = os.environ.get("VLLM_CPU_OFFLOAD_COPY_TRACE", "0").strip().lower()
    require((trace_env in {"1", "true", "yes", "on"}) == args.audit_trace, "trace contract differs")

    fixture_metadata, requests = phase.load_requests(args.fixture)
    require(len(requests) >= args.batch_size, "fixture has too few requests")
    selected = requests[: args.batch_size]
    total_output_tokens = 2 + args.warmup_decode_steps + args.measured_decode_steps
    require(
        all(item["prompt_tokens"] + total_output_tokens <= args.max_model_len for item in selected),
        "prompt plus generated tokens exceeds max-model-len",
    )
    expected_prompt_tokens = sum(item["prompt_tokens"] for item in selected)
    require(expected_prompt_tokens <= args.max_num_batched_tokens, "prefill exceeds token budget")
    if bool(contract.get("prefill_graph", False)):
        require(
            expected_prompt_tokens in capture_sizes,
            "dual-topology prefill token count has no dedicated alias Graph",
        )
    if args.kv_autoscale_roundtrip:
        require(
            len(selected[0]["prompt_token_ids"])
            >= args.kv_autoscale_low_prompt_tokens,
            "fixture prompt is shorter than the autoscale low-pressure prompt",
        )
        require(
            args.kv_autoscale_low_prompt_tokens in capture_sizes,
            "autoscale low-pressure Prefill has no dedicated alias Graph",
        )
        require(
            args.kv_autoscale_low_prompt_tokens
            + args.kv_autoscale_low_output_tokens
            <= args.max_model_len,
            "autoscale low-pressure request exceeds max-model-len",
        )
    if args.kv_path_low_performance_requests:
        require(
            len(selected[0]["prompt_token_ids"])
            >= args.kv_autoscale_low_prompt_tokens,
            "fixture prompt is shorter than the low-state performance prompt",
        )
        require(
            args.kv_autoscale_low_prompt_tokens in capture_sizes,
            "low-state performance Prefill has no dedicated alias Graph",
        )
        require(
            args.kv_autoscale_low_prompt_tokens
            + args.kv_path_performance_output_tokens
            <= args.max_model_len,
            "low-state performance request exceeds max-model-len",
        )
    if args.kv_path_high_performance_requests:
        require(
            all(
                item["prompt_tokens"] + args.kv_path_performance_output_tokens
                <= args.max_model_len
                for item in selected
            ),
            "high-state performance request exceeds max-model-len",
        )
    for spec in dynamic_specs:
        dynamic_selected = requests[: spec["batch_size"]]
        dynamic_prompt_tokens = sum(
            item["prompt_tokens"] for item in dynamic_selected
        )
        require(
            len(dynamic_selected) == spec["batch_size"],
            "fixture has too few requests for a dynamic batch",
        )
        require(
            dynamic_prompt_tokens in capture_sizes,
            "dynamic prefill token count has no dedicated alias Graph",
        )
        require(
            dynamic_prompt_tokens <= args.max_num_batched_tokens,
            "dynamic prefill exceeds token budget",
        )
        require(
            all(
                item["prompt_tokens"] + spec["max_tokens"]
                <= args.max_model_len
                for item in dynamic_selected
            ),
            "dynamic request exceeds max-model-len",
        )
    for spec in dynamic_kv_specs:
        dynamic_selected = requests[: spec["batch_size"]]
        dynamic_prompt_tokens = sum(
            item["prompt_tokens"] for item in dynamic_selected
        )
        require(
            len(dynamic_selected) == spec["batch_size"],
            "fixture has too few requests for a physical-KV batch",
        )
        require(
            dynamic_prompt_tokens in capture_sizes,
            "physical-KV prefill token count has no dedicated alias Graph",
        )
        require(
            dynamic_prompt_tokens <= args.max_num_batched_tokens,
            "physical-KV prefill exceeds token budget",
        )
        require(
            all(
                item["prompt_tokens"] + spec["max_tokens"]
                <= args.max_model_len
                for item in dynamic_selected
            ),
            "physical-KV request exceeds max-model-len",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema": "vllm_uvm_prefill_alias_decode_remote_steps_v1",
        "status": "loading",
        "arm": args.arm,
        "arm_name": contract["name"],
        "config": {
            **vars(args),
            "fixture": str(args.fixture.resolve()),
            "exact_names": str(args.exact_names.resolve()),
            "offload_manifest": str(args.offload_manifest.resolve()),
            "output": str(args.output.resolve()),
            "cudagraph_capture_sizes": capture_sizes,
            "total_output_tokens": total_output_tokens,
            "phase_mode": expected_phase_mode,
            "execution_mode": contract["execution_mode"],
            "graph_decode": bool(contract["graph"]),
            "graph_prefill": bool(contract.get("prefill_graph", False)),
            "dynamic_plan_sequence": (
                str(args.dynamic_plan_sequence.resolve())
                if args.dynamic_plan_sequence is not None
                else None
            ),
            "dynamic_plan_records": [
                {**spec, "plan": str(spec["plan"])} for spec in dynamic_specs
            ],
            "dynamic_kv_sequence": (
                str(args.dynamic_kv_sequence.resolve())
                if args.dynamic_kv_sequence is not None
                else None
            ),
            "dynamic_kv_records": list(dynamic_kv_specs),
            "incremental_expert_wave_plan": (
                str(args.incremental_expert_wave_plan.resolve())
                if args.incremental_expert_wave_plan is not None
                else None
            ),
            "incremental_expert_wave_plan_sha256": (
                sha256_file(args.incremental_expert_wave_plan)
                if args.incremental_expert_wave_plan is not None
                else None
            ),
            "incremental_expert_wave_count": len(incremental_expert_waves),
            "three_queue_qualification": three_queue_qualification,
            "tiered_plan_class_qualification": (
                tiered_plan_class_qualification
            ),
            "multilevel_pressure_qualification": (
                multilevel_pressure_qualification
            ),
            "multilevel_trigger_mode": multilevel_trigger_mode,
            "multilevel_tier_specs": list(multilevel_tier_specs),
        },
        "environment": selected_environment(),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")

    live_llms: list[Any] = []
    shutdown_results: list[dict[str, Any]] = []
    error: BaseException | None = None
    error_traceback = None
    manifest_cursor = 0
    llm: Any = None
    client: Any = None
    core: Any = None
    scheduler: Any = None
    original_schedule: Any = None
    original_uvm_single_worker_rpc: Any = None
    active_model_runner: Any = None
    kv_manager: Any = None
    block_pool: Any = None
    kv_policy: Any = None
    original_llm_class: Any = None
    original_runner_shutdown: Any = None
    uvm_rpc_records: list[dict[str, Any]] = []
    three_queue_record: dict[str, Any] | None = None
    kv_autoscale_roundtrip: dict[str, Any] | None = None
    kv_multilevel_pressure: dict[str, Any] | None = None
    kv_path_direct_high_preset: dict[str, Any] | None = None
    kv_path_performance: dict[str, Any] | None = None

    def manifest_segment(label: str) -> dict[str, Any]:
        nonlocal manifest_cursor
        records = read_manifest(args.offload_manifest)
        require(len(records) >= manifest_cursor, "manifest was truncated")
        segment = summarize_manifest(records[manifest_cursor:])
        segment["label"] = label
        segment["start_record"] = manifest_cursor
        segment["end_record"] = len(records)
        manifest_cursor = len(records)
        return segment

    try:
        import torch
        import vllm
        from vllm import SamplingParams, TokensPrompt
        from vllm.compilation.counter import compilation_counter
        from vllm.config import CompilationMode, CUDAGraphMode
        from vllm.device_allocator.uvm_static_alias import get_static_alias_diagnostics
        from vllm.model_executor.models import utils
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        original_llm_class = vllm.LLM
        original_runner_shutdown = GPUModelRunner.shutdown_cpu_offload_graph_static_alias

        class TrackedLLM(original_llm_class):
            def __init__(self, *inner_args, **inner_kwargs):
                super().__init__(*inner_args, **inner_kwargs)
                live_llms.append(self)

        def tracked_shutdown(self):
            value = original_runner_shutdown(self)
            shutdown_results.append(dict(value))
            return value

        vllm.LLM = TrackedLLM
        GPUModelRunner.shutdown_cpu_offload_graph_static_alias = tracked_shutdown

        tracker = graph.GraphCallTracker()
        tracker.install()
        counter_before = dataclasses.asdict(compilation_counter.clone())
        enforce_eager = not bool(contract["graph"])
        compilation_config = None
        if not enforce_eager:
            compilation_config = {
                "mode": "NONE",
                "cudagraph_mode": (
                    "FULL_SEPARATE"
                    if bool(contract.get("prefill_graph", False))
                    else "FULL_DECODE_ONLY"
                ),
                "cudagraph_capture_sizes": capture_sizes,
                "max_cudagraph_capture_size": max(capture_sizes),
            }

        load_started = time.perf_counter()
        llm = vllm.LLM(
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

        realized = llm.llm_engine.vllm_config.compilation_config
        require(realized.mode == CompilationMode.NONE, "CompilationMode is not NONE")
        expected_mode = (
            CUDAGraphMode.NONE
            if enforce_eager
            else (
                CUDAGraphMode.FULL_SEPARATE
                if bool(contract.get("prefill_graph", False))
                else CUDAGraphMode.FULL_DECODE_ONLY
            )
        )
        require(realized.cudagraph_mode == expected_mode, "realized CUDA Graph mode differs")
        expected_sizes = [] if enforce_eager else capture_sizes
        require(list(realized.cudagraph_capture_sizes) == expected_sizes, "capture sizes differ")
        counter_after = dataclasses.asdict(compilation_counter.clone())
        graph_after_load = tracker.snapshot()
        if contract["graph"]:
            # FULL_SEPARATE uses disjoint shape sets: uniform decode sizes
            # (remote topology) and non-uniform prefill sizes (sealed alias
            # topology).  ``capture_sizes`` is their union, so multiplying
            # its length by two incorrectly expects every shape in both
            # partitions.  Count the actual partition entries instead.
            if bool(contract.get("prefill_graph", False)):
                remote_capture_count = sum(
                    1
                    for size in capture_sizes
                    if size >= 1 and size <= args.max_num_seqs
                )
                alias_capture_count = len(capture_sizes) - remote_capture_count
                minimum_captures = remote_capture_count + alias_capture_count
            else:
                minimum_captures = len(capture_sizes)
            counter_delta = (
                counter_after["num_cudagraph_captured"]
                - counter_before["num_cudagraph_captured"]
            )
            require(
                counter_delta >= minimum_captures,
                f"too few Graph captures: got {counter_delta}, expected {minimum_captures}",
            )
            require(
                graph_after_load.get("full_capture", 0) >= minimum_captures,
                f"tracker saw too few full captures: got {graph_after_load.get('full_capture', 0)}, expected {minimum_captures}",
            )

        diagnostics_after_load = diagnostics_snapshot(utils, get_static_alias_diagnostics)
        expected_load_state = str(contract["prefill_topology"])
        require(diagnostics_after_load["model"].get("lifecycle_state") == expected_load_state, "load topology differs")
        require(diagnostics_after_load["model"].get("phase_mode") == expected_phase_mode, "diagnostic phase mode differs")
        parameter_after_load = parameter_snapshot(
            llm,
            exact_names,
            expected_count=args.expected_parameter_count,
            expected_full_bytes=args.expected_full_parameter_bytes,
        )
        model_diagnostics = diagnostics_after_load["model"]
        require(
            int(model_diagnostics.get("selected_parameter_count", -1))
            == args.expected_parameter_count,
            "selected parameter count differs",
        )
        require(
            int(model_diagnostics.get("selected_parameter_bytes_per_forward", -1))
            == args.expected_selected_bytes,
            "selected expert bytes differ",
        )
        require(
            int(model_diagnostics.get("full_parameter_bytes", -1))
            == args.expected_full_parameter_bytes,
            "diagnostic full parameter bytes differ",
        )
        load_manifest = manifest_segment("load_and_capture")

        result.update(
            {
                "status": "running",
                "versions": {
                    "vllm": getattr(vllm, "__version__", None),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                },
                "fixture": {"sha256": sha256_file(args.fixture), "metadata": fixture_metadata},
                "exact_names_sha256": sha256_file(args.exact_names),
                "load_seconds": load_seconds,
                "cuda_memory_after_load": cuda_memory_snapshot(torch),
                "compilation_counter_before_load": counter_before,
                "compilation_counter_after_load": counter_after,
                "graph_wrapper_calls_after_load": graph_after_load,
                "diagnostics": {"after_load": diagnostics_after_load},
                "parameter_va_snapshots": {"after_load": parameter_after_load},
                "manifest_segments": {"load_and_capture": load_manifest},
            }
        )
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")

        client = llm.llm_engine.engine_core
        core = getattr(client, "engine_core", None)
        require(core is not None, "in-process EngineCore is unavailable")
        scheduler = core.scheduler
        original_schedule = scheduler.schedule
        schedule_records: list[dict[str, Any]] = []

        def traced_schedule():
            output = original_schedule()
            schedule_record = phase.schedule_snapshot(output)
            if three_queue_qualification or multilevel_pressure_qualification:
                require(block_pool is not None, "three-queue block pool is absent")
                require(kv_policy is not None, "three-queue policy is absent")
                schedule_record["uvm_three_queue_after_schedule"] = (
                    block_pool.snapshot_uvm_dynamic_queues()
                )
                schedule_record["uvm_backup_pressure_after_schedule"] = (
                    kv_policy.snapshot_backup_activation()
                )
            schedule_records.append(schedule_record)
            return output

        scheduler.schedule = traced_schedule

        def run_step() -> dict[str, Any]:
            schedule_before = len(schedule_records)
            torch.cuda.synchronize()
            graph_before = tracker.snapshot()
            started = time.perf_counter()
            outputs = llm.llm_engine.step()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            require(len(schedule_records) == schedule_before + 1, "scheduler call count differs")
            return {
                "elapsed_ms": elapsed_ms,
                "outputs": outputs,
                "schedule": schedule_records[-1],
                "graph_wrapper_delta": tracker.delta(graph_before, tracker.snapshot()),
            }

        initial_kv_physical: dict[str, Any] | None = None
        if (
            dynamic_kv_specs
            or three_queue_qualification
            or multilevel_pressure_qualification
            or args.kv_autoscale_roundtrip
            or kv_path_performance_enabled
        ):
            active_model_runner = model_runner_from_llm(llm)
            kv_manager = scheduler.kv_cache_manager
            block_pool = kv_manager.block_pool
            kv_policy = kv_manager.uvm_kv_runtime_policy
            require(kv_policy is not None, "managed KV runtime policy is absent")
            require(
                kv_policy.dynamic_pool_active(),
                "managed KV dynamic pool is inactive",
            )
            initial_target = int(kv_policy._dynamic_pool_applied_target_blocks)
            initial_kv_physical = kv_physical_snapshot(
                active_model_runner,
                initial_target,
            )
            # Persist the raw snapshot even when strict validation fails so a
            # qualification artifact retains the exact GPU/CPU/unknown split.
            result["initial_kv_physical"] = initial_kv_physical
            validate_kv_physical_snapshot(initial_kv_physical)
            require(
                int(initial_kv_physical["total_blocks"])
                == int(kv_policy.total_blocks),
                "scheduler and worker disagree on maximum KV block capacity",
            )

        if args.kv_path_direct_high_preset:
            require(active_model_runner is not None, "direct-high model runner absent")
            require(block_pool is not None, "direct-high block pool absent")
            require(kv_policy is not None, "direct-high KV policy absent")
            require(
                not llm.llm_engine.has_unfinished_requests()
                and not scheduler.has_unfinished_requests(),
                "direct-high preset requires an idle scheduler",
            )
            preset_target = int(kv_policy._dynamic_pool_applied_target_blocks)
            preset_expert_before = incremental_expert_state_snapshot(
                active_model_runner
            )
            preset_queue_before = block_pool.snapshot_uvm_dynamic_queues()
            preset_physical_before = kv_physical_snapshot(
                active_model_runner,
                preset_target,
            )
            validate_kv_physical_snapshot(preset_physical_before)
            preset_parameter_before = parameter_snapshot(
                llm,
                exact_names,
                expected_count=args.expected_parameter_count,
                expected_full_bytes=args.expected_full_parameter_bytes,
            )
            preset_graph_before = tracker.snapshot()
            preset_counter_before = dataclasses.asdict(
                compilation_counter.clone()
            )
            preset_started = time.perf_counter()
            preset_candidates = core._uvm_single_worker_rpc(
                "get_uvm_incremental_expert_offload_candidates",
                args=("cold",),
            )
            require(
                isinstance(preset_candidates, dict)
                and preset_candidates.get("ok") is True,
                "direct-high expert candidate RPC failed",
            )
            selected_experts = preset_candidates.get("candidates") or {}
            require(
                isinstance(selected_experts, dict)
                and len(selected_experts) == 48,
                "direct-high preset did not select every MoE layer",
            )
            preset_release = core._uvm_single_worker_rpc(
                "run_uvm_incremental_expert_offload",
                args=(False, selected_experts),
            )
            torch.cuda.synchronize()
            preset_elapsed_ms = (
                time.perf_counter() - preset_started
            ) * 1000.0
            require(
                isinstance(preset_release, dict)
                and preset_release.get("ok") is True
                and int(preset_release.get("released_expert_count", -1)) == 48,
                "direct-high expert release RPC failed",
            )
            preset_expert_after = incremental_expert_state_snapshot(
                active_model_runner
            )
            require(
                int(preset_expert_after["remote_expert_count"])
                - int(preset_expert_before["remote_expert_count"])
                == 48,
                "direct-high preset remote expert count differs",
            )
            preset_queue_after = block_pool.snapshot_uvm_dynamic_queues()
            preset_physical_after = kv_physical_snapshot(
                active_model_runner,
                preset_target,
            )
            validate_kv_physical_snapshot(preset_physical_after)
            preset_parameter_after = parameter_snapshot(
                llm,
                exact_names,
                expected_count=args.expected_parameter_count,
                expected_full_bytes=args.expected_full_parameter_bytes,
            )
            preset_graph_after = tracker.snapshot()
            preset_counter_after = dataclasses.asdict(
                compilation_counter.clone()
            )
            require(
                preset_parameter_before["vas"] == preset_parameter_after["vas"],
                "direct-high preset changed Parameter VAs",
            )
            require(
                preset_physical_before["va_sha256"]
                == preset_physical_after["va_sha256"],
                "direct-high preset changed KV VAs",
            )
            require(
                preset_graph_before.get("full_capture", 0)
                == preset_graph_after.get("full_capture", 0)
                and preset_counter_before["num_cudagraph_captured"]
                == preset_counter_after["num_cudagraph_captured"],
                "direct-high preset recaptured a CUDA Graph",
            )
            kv_path_direct_high_preset = {
                "schema": "vllm_uvm_kv_path_direct_high_preset_v1",
                "target_blocks": preset_target,
                "elapsed_ms": preset_elapsed_ms,
                "selected_experts": {
                    str(key): int(value)
                    for key, value in selected_experts.items()
                },
                "candidates": _compact(preset_candidates),
                "release": _compact(preset_release),
                "expert_before": preset_expert_before,
                "expert_after": preset_expert_after,
                "queue_before": preset_queue_before,
                "queue_after": preset_queue_after,
                "physical_before": preset_physical_before,
                "physical_after": preset_physical_after,
                "parameter_va_before": preset_parameter_before,
                "parameter_va_after": preset_parameter_after,
                "graph_wrapper_calls_before": preset_graph_before,
                "graph_wrapper_calls_after": preset_graph_after,
                "compilation_counter_before": preset_counter_before,
                "compilation_counter_after": preset_counter_after,
            }

        if three_queue_qualification or multilevel_pressure_qualification:
            original_uvm_single_worker_rpc = core._uvm_single_worker_rpc

            def traced_uvm_single_worker_rpc(
                method: str,
                *,
                args: tuple[Any, ...],
            ) -> Any:
                started_ns = time.perf_counter_ns()
                value = original_uvm_single_worker_rpc(method, args=args)
                uvm_rpc_records.append(
                    {
                        "method": str(method),
                        "args": _compact(args),
                        "result": _compact(value),
                        "wall_ns": time.perf_counter_ns() - started_ns,
                    }
                )
                return value

            core._uvm_single_worker_rpc = traced_uvm_single_worker_rpc

            queue_before = block_pool.snapshot_uvm_dynamic_queues()
            pressure_before = kv_policy.snapshot_backup_activation()
            expert_before = incremental_expert_state_snapshot(active_model_runner)
            request_required_blocks = sum(
                math.ceil(
                    (int(item["prompt_tokens"]) + total_output_tokens)
                    / int(block_pool.hash_block_size)
                )
                for item in selected
            )
            require(queue_before.get("invariant_ok") is True, "initial queue invariant")
            require(
                int(queue_before["gpu_backup_blocks"]) > 0,
                "initial backup reserve is empty",
            )
            require(
                request_required_blocks
                > int(queue_before["gpu_free_blocks"]),
                "qualification request does not reach backup reserve",
            )
            require(
                request_required_blocks
                <= int(queue_before["gpu_free_blocks"])
                + int(queue_before["gpu_backup_blocks"]),
                "qualification request exceeds free plus backup capacity",
            )
            three_queue_record = {
                "schema": (
                    "vllm_uvm_kv_three_queue_e2e_v2"
                    if tiered_plan_class_qualification
                    else "vllm_uvm_kv_three_queue_e2e_v1"
                ),
                "expert_transition_mode": (
                    "grouped_plan_class"
                    if tiered_plan_class_qualification
                    else "incremental_per_layer"
                ),
                "request_required_blocks": request_required_blocks,
                "queue_before_request": queue_before,
                "pressure_before_request": pressure_before,
                "expert_before_request": expert_before,
                "plan_class_before_request": plan_class_state_snapshot(utils),
                "autoscale_before_request": (
                    core.get_uvm_kv_autoscale_snapshot()
                ),
                "physical_before_request": initial_kv_physical,
                "graph_wrapper_calls_before_request": tracker.snapshot(),
                "compilation_counter_before_request": dataclasses.asdict(
                    compilation_counter.clone()
                ),
            }

        direct_repeats = int(os.environ.get("E3_DIRECT_PREFILL_GRAPH_REPEATS", "0"))
        if (
            not direct_repeats
            and os.environ.get("E3_DIRECT_CONDITION_PREFILL_GRAPH", "0") == "1"
        ):
            direct_repeats = 1
        require(direct_repeats >= 0, "direct Prefill Graph repeats must be non-negative")
        if direct_repeats:
            active_model_runner = model_runner_from_llm(llm)
            graph_entries = getattr(
                active_model_runner.model, "concrete_cudagraph_entries", {}
            )
            matching_entries = [
                entry
                for descriptor, entry in graph_entries.items()
                if int(descriptor.num_tokens) == expected_prompt_tokens
                and not bool(descriptor.uniform_decode)
                and entry.cudagraph is not None
            ]
            require(
                len(matching_entries) == 1,
                "direct Prefill Graph conditioning did not find one exact entry",
            )
            direct_samples: list[float] = []
            for _ in range(direct_repeats):
                torch.cuda.synchronize()
                conditioning_started = time.perf_counter()
                matching_entries[0].cudagraph.replay()
                torch.cuda.synchronize()
                direct_samples.append(
                    (time.perf_counter() - conditioning_started) * 1000.0
                )
            result["prefill_graph_direct_conditioning"] = {
                "enabled": True,
                "num_tokens": expected_prompt_tokens,
                "samples_ms": direct_samples,
                "elapsed_ms": direct_samples[0],
                "steady_median_ms": (
                    statistics.median(direct_samples[1:])
                    if len(direct_samples) > 1
                    else direct_samples[0]
                ),
            }

        sampling = SamplingParams(
            max_tokens=total_output_tokens,
            min_tokens=total_output_tokens,
            temperature=0.0,
            seed=args.seed,
            ignore_eos=True,
        )
        prompts = [TokensPrompt(prompt_token_ids=list(item["prompt_token_ids"])) for item in selected]
        cycle_started = time.perf_counter()
        llm._validate_and_add_requests(
            prompts=prompts,
            params=sampling,
            use_tqdm=False,
            lora_request=None,
            priority=None,
        )

        profile_prefill = os.environ.get("E3_NSYS_PREFILL_CAPTURE", "0") == "1"
        if profile_prefill:
            status = int(torch.cuda.cudart().cudaProfilerStart())
            require(status == 0, f"cudaProfilerStart failed: {status}")
        try:
            prefill = run_step()
        finally:
            if profile_prefill:
                status = int(torch.cuda.cudart().cudaProfilerStop())
                require(status == 0, f"cudaProfilerStop failed: {status}")
        phase.validate_prefill(prefill["schedule"], args.batch_size)
        require(not prefill["outputs"], "prefill unexpectedly returned final outputs")
        require(prefill["schedule"]["total_num_scheduled_tokens"] == expected_prompt_tokens, "prompt token count differs")
        expected_prefill_replays = (
            1 if bool(contract.get("prefill_graph", False)) else 0
        )
        require(
            prefill["graph_wrapper_delta"].get("full_replay", 0)
            == expected_prefill_replays,
            "prefill FULL Graph replay count differs",
        )
        diagnostics_after_prefill = diagnostics_snapshot(utils, get_static_alias_diagnostics)
        parameter_after_prefill = parameter_snapshot(
            llm,
            exact_names,
            expected_count=args.expected_parameter_count,
            expected_full_bytes=args.expected_full_parameter_bytes,
        )
        prefill_manifest = manifest_segment("serving_prefill")
        if three_queue_qualification:
            require(three_queue_record is not None, "three-queue record is absent")
            target_after_prefill = int(
                kv_policy._dynamic_pool_applied_target_blocks
            )
            physical_after_prefill = kv_physical_snapshot(
                active_model_runner,
                target_after_prefill,
            )
            validate_kv_physical_snapshot(physical_after_prefill)
            three_queue_record.update(
                {
                    "prefill_schedule": prefill["schedule"],
                    "queue_after_prefill": (
                        block_pool.snapshot_uvm_dynamic_queues()
                    ),
                    "pressure_after_prefill": (
                        kv_policy.snapshot_backup_activation()
                    ),
                    "expert_after_prefill": incremental_expert_state_snapshot(
                        active_model_runner
                    ),
                    "plan_class_after_prefill": plan_class_state_snapshot(
                        utils
                    ),
                    "autoscale_after_prefill": (
                        core.get_uvm_kv_autoscale_snapshot()
                    ),
                    "physical_after_prefill": physical_after_prefill,
                    "target_after_prefill_blocks": target_after_prefill,
                    "graph_wrapper_calls_after_prefill": tracker.snapshot(),
                    "compilation_counter_after_prefill": dataclasses.asdict(
                        compilation_counter.clone()
                    ),
                    "uvm_rpc_records_after_prefill": _compact(uvm_rpc_records),
                }
            )

        first_decode = run_step()
        phase.validate_decode(first_decode["schedule"], args.batch_size)
        require(not first_decode["outputs"], "first decode finished requests early")
        diagnostics_after_first = diagnostics_snapshot(utils, get_static_alias_diagnostics)
        parameter_after_first = parameter_snapshot(
            llm,
            exact_names,
            expected_count=args.expected_parameter_count,
            expected_full_bytes=args.expected_full_parameter_bytes,
        )
        first_manifest = manifest_segment("first_decode")
        new_transitions = transition_delta(diagnostics_after_prefill, diagnostics_after_first)
        cutovers = [
            item
            for item in new_transitions
            if item.get("event") == "cpu_offload_uvm_original_va_remote_transition"
            and item.get("post_state") in ("remote", "remote_parked")
        ]
        if args.arm in ("H", "G", "D"):
            require(len(cutovers) == 1 and cutovers[0].get("ok") is True, "first decode did not perform exactly one cutover")
            cutover_transition: dict[str, Any] | None = cutovers[0]
        else:
            require(not cutovers, "remote-only arm unexpectedly cut over during decode")
            cutover_transition = None

        expected_decode_replays = 1 if bool(contract["graph"]) else 0
        require(
            first_decode["graph_wrapper_delta"].get("full_replay", 0)
            == expected_decode_replays,
            "first decode FULL Graph replay count differs",
        )

        warmup: list[dict[str, Any]] = []
        for index in range(args.warmup_decode_steps):
            step = run_step()
            phase.validate_decode(step["schedule"], args.batch_size)
            require(not step["outputs"], "decode warmup finished requests early")
            require(
                step["graph_wrapper_delta"].get("full_replay", 0)
                == expected_decode_replays,
                "warmup decode FULL Graph replay count differs",
            )
            step["step"] = index + 1
            step.pop("outputs")
            warmup.append(step)

        measured: list[dict[str, Any]] = []
        final_outputs: list[Any] = []
        for index in range(args.measured_decode_steps):
            step = run_step()
            phase.validate_decode(step["schedule"], args.batch_size)
            last = index + 1 == args.measured_decode_steps
            if step["outputs"] and not last:
                raise ContractError("measured decode finished requests early")
            if last:
                final_outputs = step["outputs"]
            require(
                step["graph_wrapper_delta"].get("full_replay", 0)
                == expected_decode_replays,
                "measured decode FULL Graph replay count differs",
            )
            step["step"] = index + 1
            step.pop("outputs")
            measured.append(step)

        cycle_elapsed_ms = (time.perf_counter() - cycle_started) * 1000.0
        diagnostics_final = diagnostics_snapshot(utils, get_static_alias_diagnostics)
        parameter_final = parameter_snapshot(
            llm,
            exact_names,
            expected_count=args.expected_parameter_count,
            expected_full_bytes=args.expected_full_parameter_bytes,
        )
        remainder_manifest = manifest_segment("remaining_decode")
        require(not llm.llm_engine.has_unfinished_requests(), "requests remain unfinished")
        output_records = phase.output_records(final_outputs)
        require(len(output_records) == args.batch_size, "final output batch differs")
        require(all(item["finished"] for item in output_records), "one or more requests unfinished")
        require(all(len(item["output_token_ids"]) == total_output_tokens for item in output_records), "output token count differs")
        require([item["prompt_tokens"] for item in output_records] == [item["prompt_tokens"] for item in selected], "prompt lengths differ")

        if three_queue_qualification:
            require(three_queue_record is not None, "three-queue record is absent")
            queue_after_request = block_pool.snapshot_uvm_dynamic_queues()
            pressure_after_request = kv_policy.snapshot_backup_activation()
            expert_after_request = incremental_expert_state_snapshot(
                active_model_runner
            )
            final_target = int(kv_policy._dynamic_pool_applied_target_blocks)
            physical_after_request = kv_physical_snapshot(
                active_model_runner,
                final_target,
            )
            validate_kv_physical_snapshot(physical_after_request)
            three_queue_record.update(
                {
                    "queue_after_request": queue_after_request,
                    "pressure_after_request": pressure_after_request,
                    "expert_after_request": expert_after_request,
                    "plan_class_after_request": plan_class_state_snapshot(
                        utils
                    ),
                    "autoscale_after_request": (
                        core.get_uvm_kv_autoscale_snapshot()
                    ),
                    "physical_after_request": physical_after_request,
                    "final_target_blocks": final_target,
                    "uvm_rpc_records": _compact(uvm_rpc_records),
                    "graph_wrapper_calls_after_request": tracker.snapshot(),
                    "compilation_counter_after_request": dataclasses.asdict(
                        compilation_counter.clone()
                    ),
                    "output_token_ids_sha256": phase.output_hash(output_records),
                }
            )

            schedule_queue = prefill["schedule"][
                "uvm_three_queue_after_schedule"
            ]
            schedule_pressure = prefill["schedule"][
                "uvm_backup_pressure_after_schedule"
            ]
            queue_before = three_queue_record["queue_before_request"]
            pressure_after_prefill = three_queue_record["pressure_after_prefill"]
            expert_before = three_queue_record["expert_before_request"]
            expert_after_prefill = three_queue_record["expert_after_prefill"]
            autoscale_after_prefill = three_queue_record[
                "autoscale_after_prefill"
            ]
            activated_blocks = int(schedule_pressure["pending_blocks"])
            require(activated_blocks > 0, "backup reserve was not activated")
            require(
                int(schedule_pressure["pending_events"]) > 0,
                "backup pressure event was not published",
            )
            require(
                int(schedule_queue["gpu_backup_blocks"])
                < int(queue_before["gpu_backup_blocks"]),
                "scheduler did not consume backup blocks",
            )
            require(
                int(pressure_after_prefill["pending_blocks"]) == 0
                and int(pressure_after_prefill["pending_events"]) == 0,
                "backup pressure event was not consumed",
            )
            require(
                int(pressure_after_prefill["total_consumed_blocks"])
                == activated_blocks,
                "consumed backup pressure differs from activation",
            )
            require(
                int(three_queue_record["target_after_prefill_blocks"])
                - int(queue_before["applied_target_blocks"])
                >= activated_blocks,
                "KV target did not grow by the consumed reserve",
            )
            if tiered_plan_class_qualification:
                plan_before = three_queue_record["plan_class_before_request"]
                plan_after = three_queue_record["plan_class_after_prefill"]
                decision = autoscale_after_prefill.get("last_decision") or {}
                require(
                    plan_before.get("active_plan_class") == "cold29"
                    and plan_after.get("active_plan_class") == "cold30",
                    "backup pressure did not switch cold29 -> cold30",
                )
                require(
                    int(
                        plan_after.get(
                            "selected_parameter_bytes_per_forward", -1
                        )
                    )
                    > int(
                        plan_before.get(
                            "selected_parameter_bytes_per_forward", -1
                        )
                    ),
                    "cold30 did not select more cold expert bytes",
                )
                require(
                    decision.get("growth_trigger_policy")
                    == "backup_threshold_only"
                    and int(decision.get("predicted_blocks", -1)) == 0
                    and int(decision.get("prediction_error_blocks", -1)) == 0
                    and decision.get("backup_pressure_threshold_reached") is True
                    and int(decision.get("backup_pressure_level_blocks", -1))
                    == int(decision.get("backup_pressure_threshold_blocks", -2))
                    == activated_blocks,
                    "grow decision was not caused solely by the frozen backup threshold",
                )
            else:
                require(
                    int(expert_after_prefill["remote_expert_count"])
                    > int(expert_before["remote_expert_count"]),
                    "expert wave did not release any expert",
                )
            require(
                queue_after_request.get("invariant_ok") is True,
                "final queue invariant failed",
            )
            require(
                int(queue_after_request["gpu_backup_blocks"])
                == int(queue_after_request["backup_target_blocks"]),
                "backup reserve was not refilled",
            )
            rpc_methods = [record["method"] for record in uvm_rpc_records]
            if tiered_plan_class_qualification:
                require(
                    rpc_methods.count("activate_uvm_expert_graph_class") == 1
                    and rpc_methods.count("resize_uvm_kv_cache_target") == 1
                    and rpc_methods.index("activate_uvm_expert_graph_class")
                    < rpc_methods.index("resize_uvm_kv_cache_target"),
                    "grouped cold30 activation did not precede physical grow",
                )
            else:
                require(
                    "get_uvm_incremental_expert_offload_candidates"
                    in rpc_methods
                    and "run_uvm_incremental_expert_offload" in rpc_methods
                    and "resize_uvm_kv_cache_target" in rpc_methods,
                    "expert-wave or physical-grow RPC is absent",
                )

        if multilevel_pressure_qualification:
            require(active_model_runner is not None, "multilevel model runner absent")
            require(block_pool is not None, "multilevel block pool absent")
            require(kv_policy is not None, "multilevel KV policy absent")
            require(initial_kv_physical is not None, "multilevel initial KV audit absent")
            require(
                not llm.llm_engine.has_unfinished_requests()
                and not scheduler.has_unfinished_requests(),
                "multilevel tier boundary is not idle",
            )

            tier_gib_values = tuple(
                int(item["tier_gib"]) for item in multilevel_tier_specs
            )
            tier_targets = tuple(
                int(item["target_blocks"]) for item in multilevel_tier_specs
            )
            tier_classes = tuple(
                str(item["plan_class"]) for item in multilevel_tier_specs
            )
            initial_target = int(kv_policy._dynamic_pool_applied_target_blocks)
            initial_plan = plan_class_state_snapshot(utils)
            require(initial_target == tier_targets[0], "initial KV tier is not 2 GiB")
            require(
                initial_plan.get("active_plan_class") == tier_classes[0],
                "initial expert class is not cold70",
            )
            initial_pending_pressure = kv_policy.snapshot_backup_activation()
            cleared_initial_pressure = kv_policy.consume_backup_activation()
            require(
                int(cleared_initial_pressure.get("blocks", 0)) > 0,
                "2 GiB measurement did not exercise the backup queue",
            )
            tier2_queue = block_pool.snapshot_uvm_dynamic_queues()
            tier2_physical = kv_physical_snapshot(
                active_model_runner,
                tier_targets[0],
            )
            validate_kv_physical_snapshot(tier2_physical)
            tier2_expert = plan_class_state_snapshot(utils)
            tier2_parameter = parameter_snapshot(
                llm,
                exact_names,
                expected_count=args.expected_parameter_count,
                expected_full_bytes=args.expected_full_parameter_bytes,
            )
            main_decode_ms = [float(item["elapsed_ms"]) for item in measured]
            require(bool(main_decode_ms), "2 GiB steady Decode sample is empty")
            tier_records: list[dict[str, Any]] = [
                {
                    "schema": "vllm_uvm_kv_multilevel_pressure_tier_v1",
                    "tier_gib": tier_gib_values[0],
                    "target_blocks": tier_targets[0],
                    "plan_class": tier_classes[0],
                    "trigger": "initial_measurement_autoscale_disabled",
                    "trigger_mode": multilevel_trigger_mode,
                    "pressure_before_clear": initial_pending_pressure,
                    "cleared_pressure": cleared_initial_pressure,
                    "queue_after_request": tier2_queue,
                    "physical_after_request": tier2_physical,
                    "plan_class_after_request": tier2_expert,
                    "parameter_va_after_request": tier2_parameter,
                    "prefill": {
                        key: value for key, value in prefill.items() if key != "outputs"
                    },
                    "first_decode": {
                        key: value
                        for key, value in first_decode.items()
                        if key != "outputs"
                    },
                    "warmup_decode": warmup,
                    "measured_decode": measured,
                    "timing": {
                        "prefill_end_to_end_ms": float(prefill["elapsed_ms"]),
                        "rebalance_rpc_wall_ms": 0.0,
                        "prefill_topology_resume_ms": 0.0,
                        "prefill_compute_excluding_rebalance_and_resume_ms": float(
                            prefill["elapsed_ms"]
                        ),
                        "tier_transition_wall_ms": 0.0,
                        "tier_transition_placement": "initial_tier",
                        "first_decode_end_to_end_ms": float(
                            first_decode["elapsed_ms"]
                        ),
                        "first_decode_topology_park_ms": float(
                            (cutover_transition or {}).get("wall_ns", 0)
                        )
                        / 1e6,
                        "steady_decode_step_ms": summarize(main_decode_ms),
                        "steady_decode_tokens_per_second": (
                            args.batch_size * 1000.0
                            / statistics.median(main_decode_ms)
                        ),
                        "request_end_to_end_ms": cycle_elapsed_ms,
                        "request_plus_tier_transition_ms": cycle_elapsed_ms,
                    },
                    "output_token_ids_sha256": phase.output_hash(output_records),
                    "outputs": output_records,
                }
            ]

            previous_autoscale = os.environ.get("VLLM_UVM_KV_AUTO_SCALE_ENABLE")
            os.environ["VLLM_UVM_KV_AUTO_SCALE_ENABLE"] = (
                "1" if multilevel_trigger_mode == "pressure" else "0"
            )
            try:
                for tier_gib, expected_target, expected_class in zip(
                    tier_gib_values[1:],
                    tier_targets[1:],
                    tier_classes[1:],
                    strict=True,
                ):
                    require(
                        not llm.llm_engine.has_unfinished_requests()
                        and not scheduler.has_unfinished_requests(),
                        f"{tier_gib} GiB pressure request did not start idle",
                    )
                    target_before = int(
                        kv_policy._dynamic_pool_applied_target_blocks
                    )
                    queue_before = block_pool.snapshot_uvm_dynamic_queues()
                    pressure_before = kv_policy.snapshot_backup_activation()
                    plan_before = plan_class_state_snapshot(utils)
                    physical_before = kv_physical_snapshot(
                        active_model_runner,
                        target_before,
                    )
                    validate_kv_physical_snapshot(physical_before)
                    diagnostics_before = diagnostics_snapshot(
                        utils,
                        get_static_alias_diagnostics,
                    )
                    parameter_before = parameter_snapshot(
                        llm,
                        exact_names,
                        expected_count=args.expected_parameter_count,
                        expected_full_bytes=args.expected_full_parameter_bytes,
                    )
                    graph_before = tracker.snapshot()
                    counter_before_tier = dataclasses.asdict(
                        compilation_counter.clone()
                    )
                    rpc_cursor = len(uvm_rpc_records)

                    direct_transition: dict[str, Any] | None = None
                    direct_transition_ms = 0.0
                    if multilevel_trigger_mode == "direct":
                        direct_started = time.perf_counter()
                        direct_transition = core.apply_uvm_kv_dynamic_pool_target(
                            expected_target,
                            reason=(
                                "request_boundary_direct_multilevel_"
                                f"{tier_gib}g"
                            ),
                            metadata={"release_expert_wave": False},
                        )
                        torch.cuda.synchronize()
                        direct_transition_ms = (
                            time.perf_counter() - direct_started
                        ) * 1000.0
                        require(
                            isinstance(direct_transition, dict)
                            and direct_transition.get("ok") is True,
                            f"{tier_gib} GiB direct tier transition failed: "
                            f"{direct_transition!r}",
                        )
                        require(
                            int(kv_policy._dynamic_pool_applied_target_blocks)
                            == expected_target
                            and plan_class_state_snapshot(utils).get(
                                "active_plan_class"
                            )
                            == expected_class,
                            f"{tier_gib} GiB direct tier did not commit before request",
                        )

                    pressure_sampling = SamplingParams(
                        max_tokens=total_output_tokens,
                        min_tokens=total_output_tokens,
                        temperature=0.0,
                        seed=args.seed,
                        ignore_eos=True,
                    )
                    llm._validate_and_add_requests(
                        prompts=[
                            TokensPrompt(
                                prompt_token_ids=list(
                                    selected[0]["prompt_token_ids"]
                                )
                            )
                        ],
                        params=pressure_sampling,
                        use_tqdm=False,
                        lora_request=None,
                        priority=None,
                    )
                    request_started = time.perf_counter()
                    # The pressure arm opens the controller for this Prefill
                    # boundary only. The direct arm has already committed the
                    # identical frozen tier while idle and keeps autoscale off.
                    os.environ["VLLM_UVM_KV_AUTO_SCALE_ENABLE"] = (
                        "1" if multilevel_trigger_mode == "pressure" else "0"
                    )
                    tier_prefill = run_step()
                    os.environ["VLLM_UVM_KV_AUTO_SCALE_ENABLE"] = "0"
                    phase.validate_prefill(tier_prefill["schedule"], 1)
                    require(
                        not tier_prefill["outputs"]
                        and tier_prefill["graph_wrapper_delta"].get(
                            "full_replay", 0
                        )
                        == 1,
                        f"{tier_gib} GiB Prefill Graph replay failed",
                    )
                    target_after_prefill = int(
                        kv_policy._dynamic_pool_applied_target_blocks
                    )
                    queue_after_prefill = (
                        block_pool.snapshot_uvm_dynamic_queues()
                    )
                    pressure_after_prefill = (
                        kv_policy.snapshot_backup_activation()
                    )
                    plan_after_prefill = plan_class_state_snapshot(utils)
                    physical_after_prefill = kv_physical_snapshot(
                        active_model_runner,
                        target_after_prefill,
                    )
                    validate_kv_physical_snapshot(physical_after_prefill)
                    autoscale_after_prefill = (
                        core.get_uvm_kv_autoscale_snapshot()
                    )
                    diagnostics_after_prefill_tier = diagnostics_snapshot(
                        utils,
                        get_static_alias_diagnostics,
                    )

                    tier_first_decode = run_step()
                    phase.validate_decode(tier_first_decode["schedule"], 1)
                    require(
                        not tier_first_decode["outputs"]
                        and tier_first_decode["graph_wrapper_delta"].get(
                            "full_replay", 0
                        )
                        == 1,
                        f"{tier_gib} GiB first Decode Graph replay failed",
                    )
                    diagnostics_after_first_tier = diagnostics_snapshot(
                        utils,
                        get_static_alias_diagnostics,
                    )

                    tier_warmup: list[dict[str, Any]] = []
                    for decode_index in range(args.warmup_decode_steps):
                        decode_step = run_step()
                        phase.validate_decode(decode_step["schedule"], 1)
                        require(
                            not decode_step["outputs"]
                            and decode_step["graph_wrapper_delta"].get(
                                "full_replay", 0
                            )
                            == 1,
                            f"{tier_gib} GiB warmup Decode failed",
                        )
                        decode_step["step"] = decode_index + 1
                        decode_step.pop("outputs")
                        tier_warmup.append(decode_step)

                    tier_measured: list[dict[str, Any]] = []
                    tier_final_outputs: list[Any] = []
                    for decode_index in range(args.measured_decode_steps):
                        decode_step = run_step()
                        phase.validate_decode(decode_step["schedule"], 1)
                        is_final = decode_index + 1 == args.measured_decode_steps
                        if is_final:
                            tier_final_outputs = decode_step["outputs"]
                        else:
                            require(
                                not decode_step["outputs"],
                                f"{tier_gib} GiB request finished early",
                            )
                        require(
                            decode_step["graph_wrapper_delta"].get(
                                "full_replay", 0
                            )
                            == 1,
                            f"{tier_gib} GiB measured Decode failed",
                        )
                        decode_step["step"] = decode_index + 1
                        decode_step.pop("outputs")
                        tier_measured.append(decode_step)

                    request_elapsed_ms = (
                        time.perf_counter() - request_started
                    ) * 1000.0
                    require(
                        not llm.llm_engine.has_unfinished_requests()
                        and not scheduler.has_unfinished_requests(),
                        f"{tier_gib} GiB request remains unfinished",
                    )
                    tier_outputs = phase.output_records(tier_final_outputs)
                    require(
                        len(tier_outputs) == 1
                        and tier_outputs[0]["finished"]
                        and len(tier_outputs[0]["output_token_ids"])
                        == total_output_tokens,
                        f"{tier_gib} GiB output is incomplete",
                    )
                    require(
                        phase.output_hash(tier_outputs)
                        == phase.output_hash(output_records),
                        f"{tier_gib} GiB output changed across tiers",
                    )

                    queue_after_request = (
                        block_pool.snapshot_uvm_dynamic_queues()
                    )
                    pressure_after_request = (
                        kv_policy.snapshot_backup_activation()
                    )
                    cleared_post_request_pressure = (
                        kv_policy.consume_backup_activation()
                    )
                    plan_after_request = plan_class_state_snapshot(utils)
                    physical_after_request = kv_physical_snapshot(
                        active_model_runner,
                        expected_target,
                    )
                    validate_kv_physical_snapshot(physical_after_request)
                    diagnostics_after_request = diagnostics_snapshot(
                        utils,
                        get_static_alias_diagnostics,
                    )
                    parameter_after = parameter_snapshot(
                        llm,
                        exact_names,
                        expected_count=args.expected_parameter_count,
                        expected_full_bytes=args.expected_full_parameter_bytes,
                    )
                    graph_after = tracker.snapshot()
                    counter_after_tier = dataclasses.asdict(
                        compilation_counter.clone()
                    )
                    tier_rpcs = uvm_rpc_records[rpc_cursor:]
                    rpc_methods = [item["method"] for item in tier_rpcs]
                    require(
                        target_after_prefill == expected_target
                        and plan_after_prefill.get("active_plan_class")
                        == expected_class
                        and plan_after_request.get("active_plan_class")
                        == expected_class,
                        f"{tier_gib} GiB KV/expert tier did not commit",
                    )
                    schedule_pressure = tier_prefill["schedule"][
                        "uvm_backup_pressure_after_schedule"
                    ]
                    decision = autoscale_after_prefill.get(
                        "last_decision"
                    ) or {}
                    require(
                        int(schedule_pressure.get("pending_blocks", 0)) > 0,
                        f"{tier_gib} GiB request did not consume backup blocks",
                    )
                    if multilevel_trigger_mode == "pressure":
                        require(
                            decision.get("growth_trigger_policy")
                            == "backup_threshold_only"
                            and decision.get(
                                "backup_pressure_threshold_reached"
                            )
                            is True
                            and int(decision.get("predicted_blocks", -1)) == 0,
                            f"{tier_gib} GiB growth was not caused by backup pressure",
                        )
                    else:
                        require(
                            direct_transition is not None
                            and direct_transition.get("ok") is True,
                            f"{tier_gib} GiB direct transition record is absent",
                        )
                    require(
                        rpc_methods.count("activate_uvm_expert_graph_class")
                        == 1
                        and rpc_methods.count("resize_uvm_kv_cache_target")
                        == 1
                        and rpc_methods.index(
                            "activate_uvm_expert_graph_class"
                        )
                        < rpc_methods.index("resize_uvm_kv_cache_target"),
                        f"{tier_gib} GiB expert-first rebalance order differs",
                    )
                    activation_rpc = next(
                        item
                        for item in tier_rpcs
                        if item.get("method")
                        == "activate_uvm_expert_graph_class"
                    )
                    activation_result = activation_rpc.get("result") or {}
                    early_resume = activation_result.get(
                        "prefill_topology_resume_before_plan_switch"
                    ) or {}
                    plan_switch = activation_result.get("plan_switch") or {}
                    delta_result = plan_switch.get("rebind") or {}
                    old_width = cold_class_width(
                        str(plan_before.get("active_plan_class"))
                    )
                    new_width = cold_class_width(expected_class)
                    expected_delta_entries = (new_width - old_width) * 450
                    require(
                        early_resume.get("ok") is True
                        and early_resume.get("pre_state") == "remote_parked"
                        and early_resume.get("post_state") == "sealed"
                        and plan_switch.get("pre_state") == "sealed"
                        and plan_switch.get("post_state") == "sealed"
                        and int(
                            plan_switch.get("quiesced_delta_add_count", -1)
                        )
                        == expected_delta_entries
                        and int(
                            plan_switch.get("quiesced_delta_remove_count", -1)
                        )
                        == 0,
                        f"{tier_gib} GiB plan-class append geometry differs",
                    )
                    if expected_delta_entries <= 4608:
                        require(
                            plan_switch.get("quiesced_delta_eligible") is True
                            and plan_switch.get("quiesced_delta_used") is True
                            and plan_switch.get("park") is None
                            and plan_switch.get("resume") is None
                            and delta_result.get("operation") == "map"
                            and int(delta_result.get("requested_count", -1))
                            == expected_delta_entries
                            and int(delta_result.get("ioctl_calls", -1)) == 1,
                            f"{tier_gib} GiB did not use one sealed delta append",
                        )
                    else:
                        require(
                            plan_switch.get("quiesced_delta_used") is False
                            and isinstance(plan_switch.get("rebind"), dict)
                            and plan_switch.get("rebind", {}).get("ok") is True,
                            f"{tier_gib} GiB large append fallback did not commit",
                        )
                    require(
                        int(queue_after_request["gpu_backup_blocks"])
                        == int(queue_after_request["backup_target_blocks"])
                        and queue_after_request.get("invariant_ok") is True,
                        f"{tier_gib} GiB queue did not refill",
                    )
                    require(
                        physical_before["va_sha256"]
                        == physical_after_prefill["va_sha256"]
                        == physical_after_request["va_sha256"]
                        == initial_kv_physical["va_sha256"]
                        and parameter_before["vas"]
                        == parameter_after["vas"]
                        == parameter_after_load["vas"],
                        f"{tier_gib} GiB VA identity changed",
                    )
                    require(
                        graph_before.get("full_capture", 0)
                        == graph_after.get("full_capture", 0)
                        and counter_before_tier["num_cudagraph_captured"]
                        == counter_after_tier["num_cudagraph_captured"],
                        f"{tier_gib} GiB rebalance recaptured a Graph",
                    )

                    prefill_transitions = transition_delta(
                        diagnostics_before,
                        diagnostics_after_prefill_tier,
                    )
                    first_decode_transitions = transition_delta(
                        diagnostics_after_prefill_tier,
                        diagnostics_after_first_tier,
                    )
                    resumes = [
                        item
                        for item in prefill_transitions
                        if item.get("event")
                        == "cpu_offload_uvm_original_va_alias_prefill_activated"
                        and item.get("ok") is True
                    ]
                    parks = [
                        item
                        for item in first_decode_transitions
                        if item.get("event")
                        == "cpu_offload_uvm_original_va_remote_transition"
                        and item.get("post_state") == "remote_parked"
                        and item.get("ok") is True
                    ]
                    require(
                        len(resumes) == 1 and len(parks) == 1,
                        f"{tier_gib} GiB topology transition count differs",
                    )
                    resume_ms = float(resumes[0].get("wall_ns", 0)) / 1e6
                    rebalance_including_resume_ms = sum(
                        float(item.get("wall_ns", 0)) / 1e6
                        for item in tier_rpcs
                        if item.get("method")
                        in {
                            "activate_uvm_expert_graph_class",
                            "resize_uvm_kv_cache_target",
                        }
                    )
                    rebalance_ms = max(
                        0.0,
                        rebalance_including_resume_ms - resume_ms,
                    )
                    tier_decode_ms = [
                        float(item["elapsed_ms"]) for item in tier_measured
                    ]
                    tier_records.append(
                        {
                            "schema": "vllm_uvm_kv_multilevel_pressure_tier_v1",
                            "tier_gib": tier_gib,
                            "target_before_blocks": target_before,
                            "target_blocks": expected_target,
                            "plan_class_before": plan_before,
                            "plan_class": expected_class,
                            "trigger": (
                                "real_scheduler_backup_queue_pressure"
                                if multilevel_trigger_mode == "pressure"
                                else "direct_idle_request_boundary"
                            ),
                            "trigger_mode": multilevel_trigger_mode,
                            "direct_transition": _compact(direct_transition),
                            "queue_before_request": queue_before,
                            "queue_after_prefill": queue_after_prefill,
                            "queue_after_request": queue_after_request,
                            "pressure_before_request": pressure_before,
                            "pressure_after_prefill": pressure_after_prefill,
                            "pressure_after_request": pressure_after_request,
                            "cleared_post_request_pressure": (
                                cleared_post_request_pressure
                            ),
                            "schedule_pressure": schedule_pressure,
                            "autoscale_after_prefill": autoscale_after_prefill,
                            "physical_before_request": physical_before,
                            "physical_after_prefill": physical_after_prefill,
                            "physical_after_request": physical_after_request,
                            "plan_class_after_prefill": plan_after_prefill,
                            "plan_class_after_request": plan_after_request,
                            "parameter_va_before_request": parameter_before,
                            "parameter_va_after_request": parameter_after,
                            "prefill": {
                                key: value
                                for key, value in tier_prefill.items()
                                if key != "outputs"
                            },
                            "first_decode": {
                                key: value
                                for key, value in tier_first_decode.items()
                                if key != "outputs"
                            },
                            "warmup_decode": tier_warmup,
                            "measured_decode": tier_measured,
                            "rpc_records": _compact(tier_rpcs),
                            "prefill_transitions": _compact(
                                prefill_transitions
                            ),
                            "first_decode_transitions": _compact(
                                first_decode_transitions
                            ),
                            "timing": {
                                "prefill_end_to_end_ms": float(
                                    tier_prefill["elapsed_ms"]
                                ),
                                "rebalance_rpc_wall_ms": rebalance_ms,
                                "rebalance_rpc_including_prefill_resume_ms": (
                                    rebalance_including_resume_ms
                                ),
                                "prefill_topology_resume_ms": resume_ms,
                                "prefill_compute_excluding_rebalance_and_resume_ms": max(
                                    0.0,
                                    float(tier_prefill["elapsed_ms"])
                                    - (
                                        rebalance_ms + resume_ms
                                        if multilevel_trigger_mode == "pressure"
                                        else 0.0
                                    ),
                                ),
                                "tier_transition_wall_ms": (
                                    rebalance_including_resume_ms
                                    if multilevel_trigger_mode == "pressure"
                                    else direct_transition_ms
                                ),
                                "tier_transition_placement": (
                                    "inside_prefill_schedule"
                                    if multilevel_trigger_mode == "pressure"
                                    else "before_request_while_idle"
                                ),
                                "first_decode_end_to_end_ms": float(
                                    tier_first_decode["elapsed_ms"]
                                ),
                                "first_decode_topology_park_ms": float(
                                    parks[0].get("wall_ns", 0)
                                )
                                / 1e6,
                                "steady_decode_step_ms": summarize(
                                    tier_decode_ms
                                ),
                                "steady_decode_tokens_per_second": (
                                    1000.0
                                    / statistics.median(tier_decode_ms)
                                ),
                                "request_end_to_end_ms": request_elapsed_ms,
                                "request_plus_tier_transition_ms": (
                                    request_elapsed_ms
                                    if multilevel_trigger_mode == "pressure"
                                    else request_elapsed_ms
                                    + direct_transition_ms
                                ),
                            },
                            "graph_wrapper_calls_before": graph_before,
                            "graph_wrapper_calls_after": graph_after,
                            "compilation_counter_before": counter_before_tier,
                            "compilation_counter_after": counter_after_tier,
                            "output_token_ids_sha256": phase.output_hash(
                                tier_outputs
                            ),
                            "outputs": tier_outputs,
                            "manifest": manifest_segment(
                                f"kv_multilevel_pressure_{tier_gib}g"
                            ),
                        }
                    )
            finally:
                if previous_autoscale is None:
                    os.environ.pop("VLLM_UVM_KV_AUTO_SCALE_ENABLE", None)
                else:
                    os.environ[
                        "VLLM_UVM_KV_AUTO_SCALE_ENABLE"
                    ] = previous_autoscale

            kv_multilevel_pressure = {
                "schema": (
                    "vllm_uvm_kv_multilevel_pressure_2_4_6_v1"
                    if tier_gib_values == (2, 4, 6)
                    else "vllm_uvm_kv_multilevel_pressure_trace_v2"
                ),
                "trigger_mode": multilevel_trigger_mode,
                "pressure_source": (
                    "real_scheduler_backup_queue_consumption"
                    if multilevel_trigger_mode == "pressure"
                    else "direct_idle_request_boundary"
                ),
                "prediction_enabled": False,
                "target_sequence_blocks": list(tier_targets),
                "plan_class_sequence": list(tier_classes),
                "initial_reserved_va_bytes": int(
                    initial_kv_physical["reserved_va_bytes"]
                ),
                "initial_physical": initial_kv_physical,
                "tiers": tier_records,
            }

        if kv_path_performance_enabled:
            require(active_model_runner is not None, "path performance runner absent")
            require(block_pool is not None, "path performance block pool absent")
            require(kv_policy is not None, "path performance KV policy absent")

            def kv_path_state_snapshot() -> dict[str, Any]:
                target = int(kv_policy._dynamic_pool_applied_target_blocks)
                physical = kv_physical_snapshot(active_model_runner, target)
                validate_kv_physical_snapshot(physical)
                diagnostics = diagnostics_snapshot(
                    utils,
                    get_static_alias_diagnostics,
                )
                model_state = diagnostics.get("model") or {}
                driver_state = diagnostics.get("driver") or {}
                parameter = parameter_snapshot(
                    llm,
                    exact_names,
                    expected_count=args.expected_parameter_count,
                    expected_full_bytes=args.expected_full_parameter_bytes,
                )
                autoscale = core.get_uvm_kv_autoscale_snapshot()
                return {
                    "target_blocks": target,
                    "queue": block_pool.snapshot_uvm_dynamic_queues(),
                    "physical": physical,
                    "expert": incremental_expert_state_snapshot(
                        active_model_runner
                    ),
                    "parameter_vas": parameter,
                    "model": {
                        "lifecycle_state": model_state.get("lifecycle_state"),
                        "sealed": model_state.get("sealed"),
                        "parked": model_state.get("parked"),
                        "poisoned": model_state.get("poisoned"),
                        "session_id": model_state.get("session_id"),
                        "selected_parameter_bytes_per_forward": model_state.get(
                            "selected_parameter_bytes_per_forward"
                        ),
                        "expert_plan_file": model_state.get("expert_plan_file"),
                        "active_plan_class": model_state.get("active_plan_class"),
                        "plan_generation": model_state.get("plan_generation"),
                        "active_driver_backing_bytes": model_state.get(
                            "active_driver_backing_bytes"
                        ),
                    },
                    "driver": {
                        "session_count": driver_state.get("session_count"),
                        "poisoned_session_count": driver_state.get(
                            "poisoned_session_count"
                        ),
                    },
                    "autoscale": {
                        "enabled": autoscale.get("enabled"),
                        "current_target_blocks": autoscale.get(
                            "current_target_blocks"
                        ),
                        "pending_grow_target_blocks": autoscale.get(
                            "pending_grow_target_blocks"
                        ),
                        "pending_shrink_target_blocks": autoscale.get(
                            "pending_shrink_target_blocks"
                        ),
                        "expert_restore_pending": autoscale.get(
                            "expert_restore_pending"
                        ),
                        "pending_plan_class": autoscale.get(
                            "pending_plan_class"
                        ),
                    },
                    "graph_wrapper_calls": tracker.snapshot(),
                    "compilation_counter": dataclasses.asdict(
                        compilation_counter.clone()
                    ),
                    "cuda_memory": cuda_memory_snapshot(torch),
                }

            def run_kv_path_requests(
                *,
                state_label: str,
                request_prompt_ids: list[list[int]],
                request_count: int,
            ) -> list[dict[str, Any]]:
                previous_autoscale = os.environ.get(
                    "VLLM_UVM_KV_AUTO_SCALE_ENABLE"
                )
                os.environ["VLLM_UVM_KV_AUTO_SCALE_ENABLE"] = "0"
                records: list[dict[str, Any]] = []
                try:
                    for request_index in range(request_count):
                        require(
                            not llm.llm_engine.has_unfinished_requests()
                            and not scheduler.has_unfinished_requests(),
                            "path performance request requires an idle boundary",
                        )
                        state_before = kv_path_state_snapshot()
                        diagnostics_before = diagnostics_snapshot(
                            utils,
                            get_static_alias_diagnostics,
                        )
                        path_sampling = SamplingParams(
                            max_tokens=args.kv_path_performance_output_tokens,
                            min_tokens=args.kv_path_performance_output_tokens,
                            temperature=0.0,
                            seed=args.seed,
                            ignore_eos=True,
                        )
                        llm._validate_and_add_requests(
                            prompts=[
                                TokensPrompt(prompt_token_ids=list(token_ids))
                                for token_ids in request_prompt_ids
                            ],
                            params=path_sampling,
                            use_tqdm=False,
                            lora_request=None,
                            priority=None,
                        )
                        request_started = time.perf_counter()
                        path_prefill = run_step()
                        phase.validate_prefill(
                            path_prefill["schedule"],
                            len(request_prompt_ids),
                        )
                        require(
                            not path_prefill["outputs"]
                            and path_prefill["graph_wrapper_delta"].get(
                                "full_replay", 0
                            )
                            == 1,
                            "path performance Prefill Graph replay failed",
                        )
                        diagnostics_after_prefill = diagnostics_snapshot(
                            utils,
                            get_static_alias_diagnostics,
                        )
                        path_first_decode = run_step()
                        phase.validate_decode(
                            path_first_decode["schedule"],
                            len(request_prompt_ids),
                        )
                        require(
                            not path_first_decode["outputs"]
                            and path_first_decode["graph_wrapper_delta"].get(
                                "full_replay", 0
                            )
                            == 1,
                            "path performance first Decode Graph replay failed",
                        )
                        diagnostics_after_first_decode = diagnostics_snapshot(
                            utils,
                            get_static_alias_diagnostics,
                        )

                        decode_tail: list[dict[str, Any]] = []
                        final_outputs: list[Any] = []
                        remaining_steps = (
                            args.kv_path_performance_output_tokens - 2
                        )
                        for decode_index in range(remaining_steps):
                            decode_step = run_step()
                            phase.validate_decode(
                                decode_step["schedule"],
                                len(request_prompt_ids),
                            )
                            is_final = decode_index + 1 == remaining_steps
                            if is_final:
                                final_outputs = decode_step["outputs"]
                            else:
                                require(
                                    not decode_step["outputs"],
                                    "path performance request finished early",
                                )
                            require(
                                decode_step["graph_wrapper_delta"].get(
                                    "full_replay", 0
                                )
                                == 1,
                                "path performance Decode Graph replay failed",
                            )
                            decode_tail.append(
                                {
                                    key: value
                                    for key, value in decode_step.items()
                                    if key != "outputs"
                                }
                            )
                        request_elapsed_ms = (
                            time.perf_counter() - request_started
                        ) * 1000.0
                        require(
                            not llm.llm_engine.has_unfinished_requests()
                            and not scheduler.has_unfinished_requests(),
                            "path performance request remains unfinished",
                        )
                        output_records = phase.output_records(final_outputs)
                        require(
                            len(output_records) == len(request_prompt_ids)
                            and all(item["finished"] for item in output_records)
                            and all(
                                len(item["output_token_ids"])
                                == args.kv_path_performance_output_tokens
                                for item in output_records
                            ),
                            "path performance output is incomplete",
                        )
                        state_after = kv_path_state_snapshot()
                        resume_transitions = transition_delta(
                            diagnostics_before,
                            diagnostics_after_prefill,
                        )
                        park_transitions = transition_delta(
                            diagnostics_after_prefill,
                            diagnostics_after_first_decode,
                        )
                        resumes = [
                            item
                            for item in resume_transitions
                            if item.get("event")
                            == "cpu_offload_uvm_original_va_alias_prefill_activated"
                            and item.get("ok") is True
                        ]
                        parks = [
                            item
                            for item in park_transitions
                            if item.get("event")
                            == "cpu_offload_uvm_original_va_remote_transition"
                            and item.get("post_state") == "remote_parked"
                            and item.get("ok") is True
                        ]
                        require(
                            len(resumes) == 1 and len(parks) == 1,
                            "path performance topology transition count differs",
                        )
                        resume_ms = float(resumes[0].get("wall_ns", 0)) / 1e6
                        park_ms = float(parks[0].get("wall_ns", 0)) / 1e6
                        prefill_compute_ms = (
                            float(path_prefill["elapsed_ms"]) - resume_ms
                        )
                        first_decode_compute_ms = (
                            float(path_first_decode["elapsed_ms"]) - park_ms
                        )
                        require(
                            prefill_compute_ms >= -0.1
                            and first_decode_compute_ms >= -0.1,
                            "path performance transition exceeds step wall time",
                        )
                        warmup_count = int(
                            args.kv_path_performance_warmup_decode_steps
                        )
                        warmup_decode = decode_tail[:warmup_count]
                        measured_decode = decode_tail[warmup_count:]
                        measured_ms = [
                            float(item["elapsed_ms"])
                            for item in measured_decode
                        ]
                        require(
                            bool(measured_ms),
                            "path performance steady Decode sample is empty",
                        )
                        records.append(
                            {
                                "schema": "vllm_uvm_kv_path_performance_request_v1",
                                "state_label": state_label,
                                "request_index": request_index,
                                "batch_size": len(request_prompt_ids),
                                "prompt_tokens": sum(
                                    len(value) for value in request_prompt_ids
                                ),
                                "output_tokens_per_request": (
                                    args.kv_path_performance_output_tokens
                                ),
                                "request_elapsed_ms": request_elapsed_ms,
                                "prefill": {
                                    key: value
                                    for key, value in path_prefill.items()
                                    if key != "outputs"
                                },
                                "first_decode": {
                                    key: value
                                    for key, value in path_first_decode.items()
                                    if key != "outputs"
                                },
                                "warmup_decode": warmup_decode,
                                "measured_decode": measured_decode,
                                "timing": {
                                    "prefill_end_to_end_ms": float(
                                        path_prefill["elapsed_ms"]
                                    ),
                                    "prefill_topology_resume_ms": resume_ms,
                                    "prefill_excluding_topology_resume_ms": max(
                                        0.0,
                                        prefill_compute_ms,
                                    ),
                                    "first_decode_end_to_end_ms": float(
                                        path_first_decode["elapsed_ms"]
                                    ),
                                    "first_decode_topology_park_ms": park_ms,
                                    "first_decode_excluding_topology_park_ms": max(
                                        0.0,
                                        first_decode_compute_ms,
                                    ),
                                    "steady_decode_step_ms": summarize(
                                        measured_ms
                                    ),
                                    "steady_decode_tokens_per_second": (
                                        len(request_prompt_ids)
                                        * 1000.0
                                        / statistics.median(measured_ms)
                                    ),
                                },
                                "state_before": state_before,
                                "state_after": state_after,
                                "resume_transitions": _compact(
                                    resume_transitions
                                ),
                                "park_transitions": _compact(
                                    park_transitions
                                ),
                                "output_token_ids_sha256": phase.output_hash(
                                    output_records
                                ),
                                "outputs": output_records,
                                "manifest": manifest_segment(
                                    f"kv_path_{state_label}_{request_index}"
                                ),
                            }
                        )
                finally:
                    if previous_autoscale is None:
                        os.environ.pop("VLLM_UVM_KV_AUTO_SCALE_ENABLE", None)
                    else:
                        os.environ[
                            "VLLM_UVM_KV_AUTO_SCALE_ENABLE"
                        ] = previous_autoscale
                return records

            kv_path_performance = {
                "schema": "vllm_uvm_kv_expert_path_performance_v1",
                "high": [],
                "low": [],
            }
            if args.kv_path_high_performance_requests:
                kv_path_performance["high"] = run_kv_path_requests(
                    state_label="high",
                    request_prompt_ids=[
                        list(item["prompt_token_ids"])
                        for item in selected
                    ],
                    request_count=args.kv_path_high_performance_requests,
                )

        if args.kv_autoscale_roundtrip:
            require(active_model_runner is not None, "autoscale model runner absent")
            require(block_pool is not None, "autoscale block pool absent")
            require(kv_policy is not None, "autoscale KV policy absent")
            require(
                not llm.llm_engine.has_unfinished_requests()
                and not scheduler.has_unfinished_requests(),
                "autoscale roundtrip requires an idle request boundary",
            )
            autoscale_enabled = (
                os.environ.get("VLLM_UVM_KV_AUTO_SCALE_ENABLE", "0")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            )
            low_prompt_ids = list(selected[0]["prompt_token_ids"])[
                : args.kv_autoscale_low_prompt_tokens
            ]
            low_prompt = TokensPrompt(prompt_token_ids=low_prompt_ids)
            low_sampling = SamplingParams(
                max_tokens=args.kv_autoscale_low_output_tokens,
                min_tokens=args.kv_autoscale_low_output_tokens,
                temperature=0.0,
                seed=args.seed,
                ignore_eos=True,
            )
            target_before_low = int(
                kv_policy._dynamic_pool_applied_target_blocks
            )
            queue_before_low = block_pool.snapshot_uvm_dynamic_queues()
            physical_before_low = kv_physical_snapshot(
                active_model_runner,
                target_before_low,
            )
            validate_kv_physical_snapshot(physical_before_low)
            expert_before_low = incremental_expert_state_snapshot(
                active_model_runner
            )
            plan_class_before_low = plan_class_state_snapshot(utils)
            diagnostics_before_low = diagnostics_snapshot(
                utils,
                get_static_alias_diagnostics,
            )
            parameter_before_low = parameter_snapshot(
                llm,
                exact_names,
                expected_count=args.expected_parameter_count,
                expected_full_bytes=args.expected_full_parameter_bytes,
            )
            autoscale_before_low = core.get_uvm_kv_autoscale_snapshot()
            graph_before_low = tracker.snapshot()
            counter_before_low = dataclasses.asdict(
                compilation_counter.clone()
            )
            rpc_cursor = len(uvm_rpc_records)

            llm._validate_and_add_requests(
                prompts=[low_prompt],
                params=low_sampling,
                use_tqdm=False,
                lora_request=None,
                priority=None,
            )
            low_steps: list[dict[str, Any]] = []
            low_final_outputs: list[Any] = []
            diagnostics_after_low_prefill: dict[str, Any] | None = None
            diagnostics_after_low_first_decode: dict[str, Any] | None = None
            parameter_after_low_prefill: dict[str, Any] | None = None
            parameter_after_low_first_decode: dict[str, Any] | None = None
            shrink_observation: dict[str, Any] | None = None
            previous_target = target_before_low
            low_started = time.perf_counter()
            for step_index in range(args.kv_autoscale_low_output_tokens + 2):
                low_step = run_step()
                if step_index == 0:
                    phase.validate_prefill(low_step["schedule"], 1)
                    require(
                        int(low_step["schedule"]["total_num_scheduled_tokens"])
                        == args.kv_autoscale_low_prompt_tokens,
                        "autoscale low-pressure Prefill token count differs",
                    )
                else:
                    phase.validate_decode(low_step["schedule"], 1)
                require(
                    low_step["graph_wrapper_delta"].get("full_replay", 0)
                    == 1,
                    "autoscale roundtrip did not replay a FULL CUDA Graph",
                )

                queue_after_step = block_pool.snapshot_uvm_dynamic_queues()
                autoscale_after_step = core.get_uvm_kv_autoscale_snapshot()
                expert_after_step = incremental_expert_state_snapshot(
                    active_model_runner
                )
                plan_class_after_step = plan_class_state_snapshot(utils)
                target_after_step = int(
                    kv_policy._dynamic_pool_applied_target_blocks
                )
                step_outputs = low_step.pop("outputs")
                low_steps.append(
                    {
                        **low_step,
                        "step": step_index,
                        "phase": "prefill" if step_index == 0 else "decode",
                        "queue_after_step": queue_after_step,
                        "autoscale_after_step": autoscale_after_step,
                        "expert_after_step": expert_after_step,
                        "plan_class_after_step": plan_class_after_step,
                        "target_after_step_blocks": target_after_step,
                    }
                )
                if target_after_step < previous_target:
                    require(
                        shrink_observation is None,
                        "autoscale roundtrip performed more than one shrink",
                    )
                    physical_after_shrink = kv_physical_snapshot(
                        active_model_runner,
                        target_after_step,
                    )
                    validate_kv_physical_snapshot(physical_after_shrink)
                    shrink_observation = {
                        "step": step_index,
                        "old_target_blocks": previous_target,
                        "new_target_blocks": target_after_step,
                        "queue": queue_after_step,
                        "physical": physical_after_shrink,
                        "autoscale": autoscale_after_step,
                        "expert": expert_after_step,
                    }
                previous_target = target_after_step

                if step_index == 0:
                    diagnostics_after_low_prefill = diagnostics_snapshot(
                        utils,
                        get_static_alias_diagnostics,
                    )
                    parameter_after_low_prefill = parameter_snapshot(
                        llm,
                        exact_names,
                        expected_count=args.expected_parameter_count,
                        expected_full_bytes=args.expected_full_parameter_bytes,
                    )
                elif step_index == 1:
                    diagnostics_after_low_first_decode = diagnostics_snapshot(
                        utils,
                        get_static_alias_diagnostics,
                    )
                    parameter_after_low_first_decode = parameter_snapshot(
                        llm,
                        exact_names,
                        expected_count=args.expected_parameter_count,
                        expected_full_bytes=args.expected_full_parameter_bytes,
                    )
                if step_outputs:
                    low_final_outputs = step_outputs
                    break

            require(
                not llm.llm_engine.has_unfinished_requests(),
                "autoscale low-pressure request remains unfinished",
            )
            require(low_final_outputs, "autoscale request returned no final output")
            low_output_records = phase.output_records(low_final_outputs)
            require(
                len(low_output_records) == 1
                and low_output_records[0]["finished"] is True
                and len(low_output_records[0]["output_token_ids"])
                == args.kv_autoscale_low_output_tokens,
                "autoscale low-pressure output is incomplete",
            )
            require(
                diagnostics_after_low_prefill is not None
                and diagnostics_after_low_first_decode is not None
                and parameter_after_low_prefill is not None
                and parameter_after_low_first_decode is not None,
                "autoscale phase snapshots are incomplete",
            )
            diagnostics_after_low = diagnostics_snapshot(
                utils,
                get_static_alias_diagnostics,
            )
            parameter_after_low = parameter_snapshot(
                llm,
                exact_names,
                expected_count=args.expected_parameter_count,
                expected_full_bytes=args.expected_full_parameter_bytes,
            )
            target_after_low = int(
                kv_policy._dynamic_pool_applied_target_blocks
            )
            queue_after_low = block_pool.snapshot_uvm_dynamic_queues()
            physical_after_low = kv_physical_snapshot(
                active_model_runner,
                target_after_low,
            )
            validate_kv_physical_snapshot(physical_after_low)
            expert_after_low = incremental_expert_state_snapshot(
                active_model_runner
            )
            plan_class_after_low = plan_class_state_snapshot(utils)
            autoscale_after_low = core.get_uvm_kv_autoscale_snapshot()
            graph_after_low = tracker.snapshot()
            counter_after_low = dataclasses.asdict(compilation_counter.clone())
            resume_transitions = transition_delta(
                diagnostics_before_low,
                diagnostics_after_low_prefill,
            )
            repark_transitions = transition_delta(
                diagnostics_after_low_prefill,
                diagnostics_after_low_first_decode,
            )
            require(
                len(
                    [
                        item
                        for item in resume_transitions
                        if item.get("event")
                        == "cpu_offload_uvm_original_va_alias_prefill_activated"
                        and item.get("ok") is True
                    ]
                )
                == 1,
                "autoscale Prefill did not resume the alias topology",
            )
            require(
                len(
                    [
                        item
                        for item in repark_transitions
                        if item.get("event")
                        == "cpu_offload_uvm_original_va_remote_transition"
                        and item.get("post_state") == "remote_parked"
                        and item.get("ok") is True
                    ]
                )
                == 1,
                "autoscale Decode did not re-park the remote topology",
            )
            if autoscale_enabled:
                require(
                    shrink_observation is not None
                    and target_after_low < target_before_low,
                    "enabled autoscale roundtrip did not shrink KV",
                )
                if tiered_plan_class_qualification:
                    require(
                        plan_class_before_low.get("active_plan_class")
                        == "cold30"
                        and plan_class_after_low.get("active_plan_class")
                        == "cold29",
                        "enabled autoscale did not switch cold30 -> cold29",
                    )
                else:
                    require(
                        int(expert_after_low["remote_expert_count"])
                        < int(expert_before_low["remote_expert_count"]),
                        "enabled autoscale roundtrip did not restore experts",
                    )
                require(
                    autoscale_after_low.get("expert_restore_pending") is False,
                    "expert restoration remains pending after idle",
                )
            else:
                require(
                    shrink_observation is None
                    and target_after_low == target_before_low,
                    "disabled baseline unexpectedly changed KV target",
                )
            require(
                counter_after_low["num_cudagraph_captured"]
                == counter_before_low["num_cudagraph_captured"]
                and graph_after_low.get("full_capture", 0)
                == graph_before_low.get("full_capture", 0),
                "autoscale roundtrip unexpectedly recaptured a CUDA Graph",
            )
            low_manifest = manifest_segment("kv_autoscale_roundtrip")
            kv_autoscale_roundtrip = {
                "schema": "vllm_uvm_kv_autoscale_roundtrip_v1",
                "enabled": autoscale_enabled,
                "elapsed_ms": (time.perf_counter() - low_started) * 1000.0,
                "prompt_tokens": args.kv_autoscale_low_prompt_tokens,
                "output_tokens": args.kv_autoscale_low_output_tokens,
                "target_before_blocks": target_before_low,
                "target_after_blocks": target_after_low,
                "queue_before": queue_before_low,
                "queue_after": queue_after_low,
                "physical_before": physical_before_low,
                "physical_after": physical_after_low,
                "expert_before": expert_before_low,
                "expert_after": expert_after_low,
                "plan_class_before": plan_class_before_low,
                "plan_class_after": plan_class_after_low,
                "autoscale_before": autoscale_before_low,
                "autoscale_after": autoscale_after_low,
                "steps": low_steps,
                "shrink_observation": shrink_observation,
                "diagnostics_before": diagnostics_before_low,
                "diagnostics_after_prefill": diagnostics_after_low_prefill,
                "diagnostics_after_first_decode": (
                    diagnostics_after_low_first_decode
                ),
                "diagnostics_after": diagnostics_after_low,
                "resume_transitions": _compact(resume_transitions),
                "repark_transitions": _compact(repark_transitions),
                "parameter_va_snapshots": {
                    "before": parameter_before_low,
                    "after_prefill": parameter_after_low_prefill,
                    "after_first_decode": parameter_after_low_first_decode,
                    "after": parameter_after_low,
                },
                "graph_wrapper_calls_before": graph_before_low,
                "graph_wrapper_calls_after": graph_after_low,
                "compilation_counter_before": counter_before_low,
                "compilation_counter_after": counter_after_low,
                "rpc_records": _compact(uvm_rpc_records[rpc_cursor:]),
                "output_token_ids_sha256": phase.output_hash(
                    low_output_records
                ),
                "outputs": low_output_records,
                "manifest": low_manifest,
            }

        if args.kv_path_low_performance_requests:
            require(
                kv_path_performance is not None
                and callable(run_kv_path_requests),
                "low-state path performance helper is unavailable",
            )
            low_prompt_ids = list(selected[0]["prompt_token_ids"])[
                : args.kv_autoscale_low_prompt_tokens
            ]
            kv_path_performance["low"] = run_kv_path_requests(
                state_label="low",
                request_prompt_ids=[low_prompt_ids],
                request_count=args.kv_path_low_performance_requests,
            )

        dynamic_rebuilds: list[dict[str, Any]] = []
        if dynamic_specs:
            shared_output_prefix = output_records[0]["output_token_ids"]
            active_model_runner = model_runner_from_llm(llm)
            kv_manager = scheduler.kv_cache_manager
            block_pool = kv_manager.block_pool
            kv_policy = kv_manager.uvm_kv_runtime_policy
            for generation, spec in enumerate(dynamic_specs, 1):
                require(
                    not llm.llm_engine.has_unfinished_requests(),
                    "dynamic rebuild boundary still has unfinished requests",
                )
                dynamic_selected = requests[: spec["batch_size"]]
                dynamic_prompts = [
                    TokensPrompt(prompt_token_ids=list(item["prompt_token_ids"]))
                    for item in dynamic_selected
                ]
                predicted_kv_tokens = sum(
                    item["prompt_tokens"] + spec["max_tokens"]
                    for item in dynamic_selected
                )
                predicted_blocks = sum(
                    math.ceil(
                        (item["prompt_tokens"] + spec["max_tokens"])
                        / block_pool.hash_block_size
                    )
                    for item in dynamic_selected
                )
                pressure_class = (
                    "high"
                    if predicted_kv_tokens >= spec["pressure_threshold_tokens"]
                    else "low"
                )
                require(
                    pressure_class == spec["expected_pressure_class"],
                    "dynamic pressure class differs from the frozen policy",
                )
                logical_pressure = kv_policy._pressure_snapshot(
                    block_pool,
                    predicted_blocks,
                )
                pressure_record = {
                    "schema": "vllm_uvm_next_request_kv_pressure_v1",
                    "previous_request_finished": True,
                    "scheduler_unfinished_request_count": 0,
                    "next_batch_size": int(spec["batch_size"]),
                    "next_prompt_tokens": sum(
                        item["prompt_tokens"] for item in dynamic_selected
                    ),
                    "next_max_output_tokens_per_request": int(
                        spec["max_tokens"]
                    ),
                    "predicted_kv_tokens": int(predicted_kv_tokens),
                    "predicted_kv_blocks": int(predicted_blocks),
                    "kv_block_size_tokens": int(block_pool.hash_block_size),
                    "pressure_threshold_tokens": int(
                        spec["pressure_threshold_tokens"]
                    ),
                    "pressure_class": pressure_class,
                    "selected_plan_label": str(spec["label"]),
                    "selected_offload_bytes": int(
                        spec["expected_selected_bytes"]
                    ),
                    "logical_pool_usage_before": float(kv_manager.usage),
                    "logical_policy_snapshot": logical_pressure,
                    "physical_kv_allocation_bytes": int(
                        args.kv_cache_memory_bytes
                    ),
                    "physical_kv_allocation_changes_at_boundary": False,
                    "cuda_memory_before": cuda_memory_snapshot(torch),
                }

                diagnostics_before_rebuild = diagnostics_snapshot(
                    utils, get_static_alias_diagnostics
                )
                parameter_before_rebuild = parameter_snapshot(
                    llm,
                    exact_names,
                    expected_count=args.expected_parameter_count,
                    expected_full_bytes=args.expected_full_parameter_bytes,
                )
                graph_before_rebuild = tracker.snapshot()
                counter_before_rebuild = dataclasses.asdict(
                    compilation_counter.clone()
                )
                rebuild_started = time.perf_counter()
                rebuild_result = (
                    active_model_runner.rebuild_cpu_offload_uvm_expert_plan_at_idle_boundary(
                        str(spec["plan"]),
                        reason=(
                            f"request_boundary_generation_{generation}_"
                            f"{pressure_class}_kv_pressure"
                        ),
                        scheduler_idle=True,
                        expected_selected_bytes=int(
                            spec["expected_selected_bytes"]
                        ),
                        pressure=pressure_record,
                    )
                )
                rebuild_elapsed_ms = (
                    time.perf_counter() - rebuild_started
                ) * 1000.0
                graph_after_rebuild = tracker.snapshot()
                counter_after_rebuild = dataclasses.asdict(
                    compilation_counter.clone()
                )
                diagnostics_after_rebuild = diagnostics_snapshot(
                    utils, get_static_alias_diagnostics
                )
                parameter_after_rebuild = parameter_snapshot(
                    llm,
                    exact_names,
                    expected_count=args.expected_parameter_count,
                    expected_full_bytes=args.expected_full_parameter_bytes,
                )
                rebuilt_model = diagnostics_after_rebuild["model"]
                require(rebuild_result.get("ok") is True, "dynamic rebuild failed")
                require(
                    int(rebuilt_model.get("plan_generation", -1)) == generation,
                    "dynamic plan generation differs",
                )
                require(
                    int(
                        rebuilt_model.get(
                            "selected_parameter_bytes_per_forward", -1
                        )
                    )
                    == int(spec["expected_selected_bytes"]),
                    "dynamic selected bytes differ after rebuild",
                )
                require(
                    rebuilt_model.get("lifecycle_state") == "sealed"
                    and rebuilt_model.get("remote_graph_captured") is True
                    and rebuilt_model.get("alias_graph_captured") is True,
                    "dynamic recapture did not restore sealed dual Graphs",
                )
                require(
                    parameter_before_rebuild["vas"]
                    == parameter_after_rebuild["vas"]
                    == parameter_after_load["vas"],
                    "Parameter VAs changed across dynamic rebuild",
                )
                graph_rebuild_delta = tracker.delta(
                    graph_before_rebuild, graph_after_rebuild
                )
                require(
                    graph_rebuild_delta.get("full_capture", 0)
                    >= len(capture_sizes),
                    "dynamic rebuild captured too few FULL Graphs",
                )
                require(
                    counter_after_rebuild["num_cudagraph_captured"]
                    - counter_before_rebuild["num_cudagraph_captured"]
                    >= len(capture_sizes),
                    "dynamic rebuild compilation counter captured too few Graphs",
                )

                dynamic_sampling = SamplingParams(
                    max_tokens=spec["max_tokens"],
                    min_tokens=spec["max_tokens"],
                    temperature=0.0,
                    seed=args.seed,
                    ignore_eos=True,
                )
                llm._validate_and_add_requests(
                    prompts=dynamic_prompts,
                    params=dynamic_sampling,
                    use_tqdm=False,
                    lora_request=None,
                    priority=None,
                )
                request_started = time.perf_counter()
                dynamic_prefill = run_step()
                phase.validate_prefill(
                    dynamic_prefill["schedule"], spec["batch_size"]
                )
                require(
                    not dynamic_prefill["outputs"],
                    "dynamic prefill completed requests early",
                )
                require(
                    dynamic_prefill["graph_wrapper_delta"].get(
                        "full_replay", 0
                    )
                    == 1,
                    "dynamic prefill did not replay its recaptured alias Graph",
                )
                diagnostics_after_dynamic_prefill = diagnostics_snapshot(
                    utils, get_static_alias_diagnostics
                )

                dynamic_decode_steps: list[dict[str, Any]] = []
                dynamic_final_outputs: list[Any] = []
                for decode_index in range(spec["max_tokens"] - 1):
                    decode_step = run_step()
                    phase.validate_decode(
                        decode_step["schedule"], spec["batch_size"]
                    )
                    final_decode = decode_index == spec["max_tokens"] - 2
                    if final_decode:
                        dynamic_final_outputs = decode_step["outputs"]
                    else:
                        require(
                            not decode_step["outputs"],
                            "dynamic decode completed requests early",
                        )
                    require(
                        decode_step["graph_wrapper_delta"].get(
                            "full_replay", 0
                        )
                        == 1,
                        "dynamic decode did not replay its remote Graph",
                    )
                    dynamic_decode_steps.append(
                        {
                            key: value
                            for key, value in decode_step.items()
                            if key != "outputs"
                        }
                    )
                request_elapsed_ms = (
                    time.perf_counter() - request_started
                ) * 1000.0
                require(
                    not llm.llm_engine.has_unfinished_requests(),
                    "dynamic request remains unfinished",
                )
                dynamic_outputs = phase.output_records(dynamic_final_outputs)
                require(
                    len(dynamic_outputs) == spec["batch_size"]
                    and all(item["finished"] for item in dynamic_outputs),
                    "dynamic output batch is incomplete",
                )
                require(
                    all(
                        len(item["output_token_ids"]) == spec["max_tokens"]
                        for item in dynamic_outputs
                    ),
                    "dynamic output token count differs",
                )
                require(
                    dynamic_outputs[0]["output_token_ids"]
                    == shared_output_prefix[: spec["max_tokens"]],
                    "shared prompt output changed after dynamic plan rebuild",
                )
                diagnostics_after_dynamic_request = diagnostics_snapshot(
                    utils, get_static_alias_diagnostics
                )
                parameter_after_dynamic_request = parameter_snapshot(
                    llm,
                    exact_names,
                    expected_count=args.expected_parameter_count,
                    expected_full_bytes=args.expected_full_parameter_bytes,
                )
                require(
                    diagnostics_after_dynamic_request["model"].get(
                        "lifecycle_state"
                    )
                    == "remote_parked",
                    "dynamic request did not cut over to remote_parked",
                )
                require(
                    parameter_after_dynamic_request["vas"]
                    == parameter_after_load["vas"],
                    "Parameter VAs changed while serving a dynamic request",
                )
                dynamic_transitions = transition_delta(
                    diagnostics_after_rebuild,
                    diagnostics_after_dynamic_request,
                )
                dynamic_cutovers = [
                    item
                    for item in dynamic_transitions
                    if item.get("event")
                    == "cpu_offload_uvm_original_va_remote_transition"
                    and item.get("post_state") == "remote_parked"
                    and item.get("ok") is True
                ]
                require(
                    len(dynamic_cutovers) == 1,
                    "dynamic request did not perform exactly one cutover",
                )
                pressure_record["logical_pool_usage_after"] = float(
                    kv_manager.usage
                )
                pressure_record["cuda_memory_after"] = cuda_memory_snapshot(
                    torch
                )
                dynamic_manifest = manifest_segment(
                    f"dynamic_generation_{generation}_{spec['label']}"
                )
                dynamic_rebuilds.append(
                    {
                        "generation": generation,
                        "label": spec["label"],
                        "plan": str(spec["plan"]),
                        "expected_selected_bytes": int(
                            spec["expected_selected_bytes"]
                        ),
                        "pressure": pressure_record,
                        "rebuild_elapsed_ms": rebuild_elapsed_ms,
                        "request_elapsed_ms": request_elapsed_ms,
                        "rebuild": _compact(rebuild_result),
                        "graph_wrapper_delta": graph_rebuild_delta,
                        "compilation_counter_before": counter_before_rebuild,
                        "compilation_counter_after": counter_after_rebuild,
                        "diagnostics_before_rebuild": diagnostics_before_rebuild,
                        "diagnostics_after_rebuild": diagnostics_after_rebuild,
                        "diagnostics_after_prefill": (
                            diagnostics_after_dynamic_prefill
                        ),
                        "diagnostics_after_request": (
                            diagnostics_after_dynamic_request
                        ),
                        "parameter_va_snapshots": {
                            "before_rebuild": parameter_before_rebuild,
                            "after_rebuild": parameter_after_rebuild,
                            "after_request": parameter_after_dynamic_request,
                        },
                        "prefill": {
                            key: value
                            for key, value in dynamic_prefill.items()
                            if key != "outputs"
                        },
                        "decode": dynamic_decode_steps,
                        "cutovers": _compact(dynamic_cutovers),
                        "output_token_ids_sha256": phase.output_hash(
                            dynamic_outputs
                        ),
                        "outputs": dynamic_outputs,
                        "manifest": dynamic_manifest,
                    }
                )

        dynamic_kv_resizes: list[dict[str, Any]] = []
        physical_kv_reuse_probe: dict[str, Any] | None = None
        if dynamic_kv_specs:
            require(active_model_runner is not None, "physical KV model runner absent")
            require(kv_manager is not None, "physical KV manager absent")
            require(block_pool is not None, "physical KV block pool absent")
            require(kv_policy is not None, "physical KV policy absent")
            require(initial_kv_physical is not None, "initial KV audit absent")
            shared_output_prefix = output_records[0]["output_token_ids"]
            kv_va_sha256 = str(initial_kv_physical["va_sha256"])
            kv_reserved_va_bytes = int(initial_kv_physical["reserved_va_bytes"])

            def expected_incremental_experts(
                wave_count: int,
            ) -> dict[str, tuple[int, ...]]:
                expected: dict[str, list[int]] = {}
                for wave in incremental_expert_waves[: int(wave_count)]:
                    for layer, expert in wave.items():
                        expected.setdefault(layer, []).append(int(expert))
                return {
                    layer: tuple(sorted(experts))
                    for layer, experts in expected.items()
                }

            def validate_incremental_expert_waves(
                snapshot: dict[str, Any],
                wave_count: int,
                label: str,
            ) -> None:
                expected = expected_incremental_experts(wave_count)
                layers = snapshot.get("layers") or {}
                if incremental_expert_waves:
                    require(
                        set(layers) == set(incremental_expert_waves[0]),
                        f"{label}: incremental expert layer set differs",
                    )
                require(
                    int(snapshot.get("remote_expert_count", -1))
                    == int(wave_count) * len(layers),
                    f"{label}: incremental remote expert count differs",
                )
                require(
                    int(snapshot.get("offloaded_expert_count", -1)) == 0,
                    f"{label}: non-remote incremental experts are present",
                )
                for layer, state in layers.items():
                    actual = tuple(
                        sorted(int(value) for value in state["remote_expert_ids"])
                    )
                    require(
                        actual == expected.get(layer, ()),
                        f"{label}: incremental experts differ for {layer}",
                    )

            def transition_incremental_expert_waves(
                current_count: int,
                target_count: int,
            ) -> list[dict[str, Any]]:
                transitions: list[dict[str, Any]] = []
                if target_count > current_count:
                    wave_indexes = range(current_count, target_count)
                    operation = "release"
                else:
                    wave_indexes = range(current_count - 1, target_count - 1, -1)
                    operation = "restore"
                for wave_index in wave_indexes:
                    selected_experts = incremental_expert_waves[wave_index]
                    started_ns = time.perf_counter_ns()
                    if operation == "release":
                        value = core._uvm_single_worker_rpc(
                            "run_uvm_incremental_expert_offload",
                            args=(False, selected_experts),
                        )
                        next_count = wave_index + 1
                        expected_count = len(selected_experts)
                        actual_count = int(
                            (value or {}).get("released_expert_count", -1)
                        )
                    else:
                        value = core._uvm_single_worker_rpc(
                            "run_uvm_incremental_expert_restore",
                            args=(selected_experts,),
                        )
                        next_count = wave_index
                        expected_count = len(selected_experts)
                        actual_count = int(
                            (value or {}).get("restored_expert_count", -1)
                        )
                    require(
                        isinstance(value, dict)
                        and value.get("ok") is True
                        and actual_count == expected_count,
                        f"incremental expert wave {operation} failed: "
                        f"wave={wave_index + 1} result={value!r}",
                    )
                    torch.cuda.synchronize()
                    snapshot = incremental_expert_state_snapshot(
                        active_model_runner
                    )
                    validate_incremental_expert_waves(
                        snapshot,
                        next_count,
                        f"after expert wave {operation} {wave_index + 1}",
                    )
                    transitions.append(
                        {
                            "operation": operation,
                            "wave": wave_index + 1,
                            "selected_experts": dict(selected_experts),
                            "elapsed_ms": (
                                time.perf_counter_ns() - started_ns
                            )
                            / 1_000_000.0,
                            "result": _compact(value),
                            "state_after": snapshot,
                        }
                    )
                return transitions

            initial_expert_state = incremental_expert_state_snapshot(
                active_model_runner
            )
            validate_incremental_expert_waves(
                initial_expert_state,
                0,
                "initial dynamic-KV expert state",
            )
            current_expert_wave_count = 0

            for generation, spec in enumerate(dynamic_kv_specs, 1):
                require(
                    not llm.llm_engine.has_unfinished_requests(),
                    "physical KV resize boundary still has unfinished requests",
                )
                require(
                    not scheduler.has_unfinished_requests(),
                    "scheduler is not idle at physical KV resize boundary",
                )
                dynamic_selected = requests[: spec["batch_size"]]
                dynamic_prompts = [
                    TokensPrompt(prompt_token_ids=list(item["prompt_token_ids"]))
                    for item in dynamic_selected
                ]
                old_target = int(kv_policy._dynamic_pool_applied_target_blocks)
                before_physical = kv_physical_snapshot(
                    active_model_runner,
                    old_target,
                )
                validate_kv_physical_snapshot(before_physical)
                require(
                    before_physical["va_sha256"] == kv_va_sha256
                    and int(before_physical["reserved_va_bytes"])
                    == kv_reserved_va_bytes,
                    "KV VA reservation changed before physical resize",
                )
                target_expert_wave_count = int(spec["expert_waves"])
                expert_before_resize = incremental_expert_state_snapshot(
                    active_model_runner
                )
                validate_incremental_expert_waves(
                    expert_before_resize,
                    current_expert_wave_count,
                    f"generation {generation} before resize",
                )

                graph_before_resize = tracker.snapshot()
                counter_before_resize = dataclasses.asdict(
                    compilation_counter.clone()
                )
                cuda_before_resize = cuda_memory_snapshot(torch)
                expert_transitions_before_kv: list[dict[str, Any]] = []
                expert_transitions_after_kv: list[dict[str, Any]] = []
                if target_expert_wave_count > current_expert_wave_count:
                    expert_transitions_before_kv = (
                        transition_incremental_expert_waves(
                            current_expert_wave_count,
                            target_expert_wave_count,
                        )
                    )
                    current_expert_wave_count = target_expert_wave_count
                resize_started = time.perf_counter()
                resize_reason = (
                    f"request_boundary_physical_kv_generation_{generation}_"
                    f"{spec['label']}"
                )
                explicit_target = spec.get("target_blocks")
                if explicit_target is None:
                    resize_result = core.apply_uvm_kv_next_request_pressure(
                        [
                            int(item["prompt_tokens"])
                            for item in dynamic_selected
                        ],
                        int(spec["max_tokens"]),
                        reason=resize_reason,
                    )
                else:
                    explicit_result = core.apply_uvm_kv_dynamic_pool_target(
                        int(explicit_target),
                        reason=resize_reason,
                        metadata={"release_expert_wave": False},
                    )
                    predicted_blocks = sum(
                        math.ceil(
                            (
                                int(item["prompt_tokens"])
                                + int(spec["max_tokens"])
                            )
                            / int(block_pool.hash_block_size)
                        )
                        for item in dynamic_selected
                    )
                    pressure = {
                        "schema": "vllm_uvm_explicit_physical_kv_target_v1",
                        "decision_source": (
                            "explicit_multilevel_physical_kv_qualification"
                        ),
                        "previous_request_finished": True,
                        "scheduler_unfinished_request_count": 0,
                        "queued_batch_count": 0,
                        "old_target_blocks": old_target,
                        "new_target_blocks": int(explicit_target),
                        "physical_target_changed": (
                            int(explicit_target) != old_target
                        ),
                        "maximum_reserved_blocks": int(kv_policy.total_blocks),
                        "next_batch_size": int(spec["batch_size"]),
                        "next_prompt_tokens": sum(
                            int(item["prompt_tokens"])
                            for item in dynamic_selected
                        ),
                        "next_max_output_tokens_per_request": int(
                            spec["max_tokens"]
                        ),
                        "kv_block_size_tokens": int(block_pool.hash_block_size),
                        "predicted_kv_blocks": predicted_blocks,
                        "predicted_kv_tokens": sum(
                            int(item["prompt_tokens"])
                            + int(spec["max_tokens"])
                            for item in dynamic_selected
                        ),
                        "pressure_class": "explicit",
                    }
                    resize_result = {
                        **dict(explicit_result),
                        "pressure": pressure,
                    }
                torch.cuda.synchronize()
                resize_elapsed_ms = (
                    time.perf_counter() - resize_started
                ) * 1000.0
                require(resize_result.get("ok") is True, "physical KV resize failed")
                pressure_record = dict(resize_result.get("pressure") or {})
                expected_decision_source = (
                    "explicit_multilevel_physical_kv_qualification"
                    if explicit_target is not None
                    else "engine_next_request_pressure_policy"
                )
                require(
                    pressure_record.get("decision_source")
                    == expected_decision_source,
                    "physical KV target decision source differs",
                )
                pressure_class = str(pressure_record.get("pressure_class") or "")
                require(
                    pressure_class == spec["expected_pressure_class"],
                    "physical KV pressure class differs from expectation",
                )
                expected_predicted_blocks = sum(
                    math.ceil(
                        (int(item["prompt_tokens"]) + int(spec["max_tokens"]))
                        / int(block_pool.hash_block_size)
                    )
                    for item in dynamic_selected
                )
                require(
                    int(pressure_record.get("predicted_kv_blocks", -1))
                    == expected_predicted_blocks,
                    "EngineCore physical KV pressure calculation differs",
                )
                requested_target = int(
                    pressure_record.get("new_target_blocks", -1)
                )
                if explicit_target is not None:
                    require(
                        requested_target == int(explicit_target),
                        "explicit physical KV target differs",
                    )
                pressure_record["maximum_reserved_va_bytes"] = kv_reserved_va_bytes
                require(
                    int(kv_policy._dynamic_pool_applied_target_blocks)
                    == requested_target,
                    "scheduler did not commit the requested physical KV target",
                )
                if target_expert_wave_count < current_expert_wave_count:
                    expert_transitions_after_kv = (
                        transition_incremental_expert_waves(
                            current_expert_wave_count,
                            target_expert_wave_count,
                        )
                    )
                    current_expert_wave_count = target_expert_wave_count
                expert_after_resize = incremental_expert_state_snapshot(
                    active_model_runner
                )
                validate_incremental_expert_waves(
                    expert_after_resize,
                    target_expert_wave_count,
                    f"generation {generation} after resize",
                )
                after_resize_physical = kv_physical_snapshot(
                    active_model_runner,
                    requested_target,
                )
                cuda_after_resize = cuda_memory_snapshot(torch)
                validate_kv_physical_snapshot(after_resize_physical)
                require(
                    after_resize_physical["va_sha256"] == kv_va_sha256
                    and int(after_resize_physical["reserved_va_bytes"])
                    == kv_reserved_va_bytes,
                    "KV VA reservation changed during physical resize",
                )
                graph_after_resize = tracker.snapshot()
                counter_after_resize = dataclasses.asdict(
                    compilation_counter.clone()
                )
                require(
                    graph_after_resize.get("full_capture", 0)
                    == graph_before_resize.get("full_capture", 0),
                    "physical KV resize unexpectedly captured a CUDA Graph",
                )
                require(
                    counter_after_resize["num_cudagraph_captured"]
                    == counter_before_resize["num_cudagraph_captured"],
                    "physical KV resize changed the CUDA Graph capture counter",
                )

                diagnostics_before_request = diagnostics_snapshot(
                    utils, get_static_alias_diagnostics
                )
                parameter_before_request = parameter_snapshot(
                    llm,
                    exact_names,
                    expected_count=args.expected_parameter_count,
                    expected_full_bytes=args.expected_full_parameter_bytes,
                )
                request_graph_before = tracker.snapshot()
                request_counter_before = dataclasses.asdict(
                    compilation_counter.clone()
                )
                dynamic_sampling = SamplingParams(
                    max_tokens=spec["max_tokens"],
                    min_tokens=spec["max_tokens"],
                    temperature=0.0,
                    seed=args.seed,
                    ignore_eos=True,
                )
                llm._validate_and_add_requests(
                    prompts=dynamic_prompts,
                    params=dynamic_sampling,
                    use_tqdm=False,
                    lora_request=None,
                    priority=None,
                )
                request_started = time.perf_counter()
                dynamic_prefill = run_step()
                phase.validate_prefill(
                    dynamic_prefill["schedule"], spec["batch_size"]
                )
                require(
                    not dynamic_prefill["outputs"],
                    "physical-KV prefill completed requests early",
                )
                require(
                    dynamic_prefill["graph_wrapper_delta"].get("full_replay", 0)
                    == 1,
                    "physical-KV prefill did not replay the retained alias Graph",
                )

                dynamic_decode_steps: list[dict[str, Any]] = []
                dynamic_final_outputs: list[Any] = []
                for decode_index in range(spec["max_tokens"] - 1):
                    decode_step = run_step()
                    phase.validate_decode(
                        decode_step["schedule"], spec["batch_size"]
                    )
                    final_decode = decode_index == spec["max_tokens"] - 2
                    if final_decode:
                        dynamic_final_outputs = decode_step["outputs"]
                    else:
                        require(
                            not decode_step["outputs"],
                            "physical-KV decode completed requests early",
                        )
                    require(
                        decode_step["graph_wrapper_delta"].get("full_replay", 0)
                        == 1,
                        "physical-KV decode did not replay the retained remote Graph",
                    )
                    dynamic_decode_steps.append(
                        {
                            key: value
                            for key, value in decode_step.items()
                            if key != "outputs"
                        }
                    )

                request_elapsed_ms = (
                    time.perf_counter() - request_started
                ) * 1000.0
                require(
                    not llm.llm_engine.has_unfinished_requests(),
                    "physical-KV request remains unfinished",
                )
                dynamic_outputs = phase.output_records(dynamic_final_outputs)
                require(
                    len(dynamic_outputs) == spec["batch_size"]
                    and all(item["finished"] for item in dynamic_outputs),
                    "physical-KV output batch is incomplete",
                )
                require(
                    all(
                        len(item["output_token_ids"]) == spec["max_tokens"]
                        for item in dynamic_outputs
                    ),
                    "physical-KV output token count differs",
                )
                shared_prompt_matches_initial_batch = (
                    dynamic_outputs[0]["output_token_ids"]
                    == shared_output_prefix[: spec["max_tokens"]]
                )
                diagnostics_after_request = diagnostics_snapshot(
                    utils, get_static_alias_diagnostics
                )
                expert_after_request = incremental_expert_state_snapshot(
                    active_model_runner
                )
                validate_incremental_expert_waves(
                    expert_after_request,
                    target_expert_wave_count,
                    f"generation {generation} after request",
                )
                parameter_after_request = parameter_snapshot(
                    llm,
                    exact_names,
                    expected_count=args.expected_parameter_count,
                    expected_full_bytes=args.expected_full_parameter_bytes,
                )
                require(
                    diagnostics_after_request["model"].get("lifecycle_state")
                    == "remote_parked",
                    "physical-KV request did not return to remote_parked",
                )
                require(
                    parameter_before_request["vas"]
                    == parameter_after_request["vas"]
                    == parameter_after_load["vas"],
                    "expert Parameter VAs changed during physical KV serving",
                )
                after_request_physical = kv_physical_snapshot(
                    active_model_runner,
                    requested_target,
                )
                validate_kv_physical_snapshot(after_request_physical)
                require(
                    after_request_physical["va_sha256"] == kv_va_sha256
                    and int(after_request_physical["reserved_va_bytes"])
                    == kv_reserved_va_bytes,
                    "KV VAs changed while serving after physical resize",
                )
                request_graph_after = tracker.snapshot()
                request_counter_after = dataclasses.asdict(
                    compilation_counter.clone()
                )
                require(
                    request_graph_after.get("full_capture", 0)
                    == request_graph_before.get("full_capture", 0),
                    "physical-KV request unexpectedly recaptured a CUDA Graph",
                )
                require(
                    request_counter_after["num_cudagraph_captured"]
                    == request_counter_before["num_cudagraph_captured"],
                    "physical-KV request changed the Graph capture counter",
                )

                warmup_count = int(spec["decode_warmup_steps"])
                # Keep this request-local decode sample separate from the
                # qualification cycle's initial first_decode.  The latter is
                # paired with cutover_transition below; overwriting it here
                # would compare timing records from two different requests.
                dynamic_first_decode = dynamic_decode_steps[0]
                warmup_decode = dynamic_decode_steps[1 : 1 + warmup_count]
                measured_decode = dynamic_decode_steps[1 + warmup_count :]
                measured_ms = [
                    float(item["elapsed_ms"]) for item in measured_decode
                ]
                require(bool(measured_ms), "physical-KV decode sample is empty")
                dynamic_manifest = manifest_segment(
                    f"physical_kv_generation_{generation}_{spec['label']}"
                )
                dynamic_kv_resizes.append(
                    {
                        "generation": generation,
                        "label": spec["label"],
                        "pressure": pressure_record,
                        "resize": _compact(resize_result),
                        "resize_elapsed_ms": resize_elapsed_ms,
                        "target_expert_waves": target_expert_wave_count,
                        "expert_before_resize": expert_before_resize,
                        "expert_transitions_before_kv": (
                            expert_transitions_before_kv
                        ),
                        "expert_transitions_after_kv": (
                            expert_transitions_after_kv
                        ),
                        "expert_after_resize": expert_after_resize,
                        "expert_after_request": expert_after_request,
                        "request_elapsed_ms": request_elapsed_ms,
                        "cuda_memory_before_resize": cuda_before_resize,
                        "cuda_memory_after_resize": cuda_after_resize,
                        "cuda_memory_after_request": cuda_memory_snapshot(torch),
                        "physical_before_resize": before_physical,
                        "physical_after_resize": after_resize_physical,
                        "physical_after_request": after_request_physical,
                        "prefill": {
                            key: value
                            for key, value in dynamic_prefill.items()
                            if key != "outputs"
                        },
                        "first_decode": dynamic_first_decode,
                        "warmup_decode": warmup_decode,
                        "measured_decode": measured_decode,
                        "steady_decode_step_ms": summarize(measured_ms),
                        "steady_decode_tokens_per_second": (
                            int(spec["batch_size"])
                            * 1000.0
                            / statistics.median(measured_ms)
                        ),
                        "diagnostics_before_request": diagnostics_before_request,
                        "diagnostics_after_request": diagnostics_after_request,
                        "parameter_va_snapshots": {
                            "before_request": parameter_before_request,
                            "after_request": parameter_after_request,
                        },
                        "graph_wrapper_calls_before_resize": graph_before_resize,
                        "graph_wrapper_calls_after_resize": graph_after_resize,
                        "graph_wrapper_calls_after_request": request_graph_after,
                        "compilation_counter_before_resize": counter_before_resize,
                        "compilation_counter_after_resize": counter_after_resize,
                        "compilation_counter_after_request": request_counter_after,
                        "output_token_ids_sha256": phase.output_hash(dynamic_outputs),
                        "shared_prompt_matches_initial_batch": (
                            shared_prompt_matches_initial_batch
                        ),
                        "outputs": dynamic_outputs,
                        "manifest": dynamic_manifest,
                    }
                )

        if args.physical_kv_reuse_probe_bytes:
            from vllm.device_allocator.uvm import probe_driver_capable_range

            require(active_model_runner is not None, "reuse probe model runner absent")
            require(kv_policy is not None, "reuse probe KV policy absent")
            require(
                not llm.llm_engine.has_unfinished_requests()
                and not scheduler.has_unfinished_requests(),
                "physical KV reuse probe requires an idle request boundary",
            )
            probe_target = int(kv_policy._dynamic_pool_applied_target_blocks)
            require(
                probe_target < int(kv_policy.total_blocks),
                "physical KV reuse probe requires a shrunken pool",
            )
            physical_before_probe = kv_physical_snapshot(
                active_model_runner,
                probe_target,
            )
            validate_kv_physical_snapshot(physical_before_probe)
            graph_before_probe = tracker.snapshot()
            counter_before_probe = dataclasses.asdict(compilation_counter.clone())
            cuda_before_probe = cuda_memory_snapshot(torch)
            allocated_before = int(torch.cuda.memory_allocated())
            probe_tensor: Any | None = None
            probe_ptr = 0
            probe_marker = -1
            probe_is_managed: bool | None = None
            probe_pointer_kind = "not_probed"
            probe_pointer_status = -1
            allocator_backend = str(torch.cuda.memory.get_allocator_backend())
            cuda_during_probe: dict[str, int] = {}
            physical_during_probe: dict[str, Any] = {}
            try:
                probe_tensor = torch.empty(
                    int(args.physical_kv_reuse_probe_bytes),
                    dtype=torch.uint8,
                    device="cuda",
                )
                probe_ptr = int(probe_tensor.data_ptr())
                (
                    probe_is_managed,
                    probe_pointer_kind,
                    probe_pointer_status,
                ) = probe_driver_capable_range(
                    probe_ptr,
                    int(args.physical_kv_reuse_probe_bytes),
                )
                require(
                    probe_is_managed is False
                    and probe_pointer_kind == "not_cuda_managed"
                    and probe_pointer_status == 0,
                    "physical KV reuse probe is not an ordinary CUDA allocation",
                )
                probe_tensor.fill_(90)
                torch.cuda.synchronize()
                probe_marker = int(probe_tensor[0].item()) + int(
                    probe_tensor[-1].item()
                )
                cuda_during_probe = cuda_memory_snapshot(torch)
                physical_during_probe = kv_physical_snapshot(
                    active_model_runner,
                    probe_target,
                )
            finally:
                probe_tensor = None
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            restore_result = active_model_runner.rebalance_uvm_kv_cache_after_warmup(
                probe_target
            )
            require(
                bool(restore_result.get("offload_complete")),
                "physical KV residency restore failed after reuse probe",
            )
            physical_after_probe = kv_physical_snapshot(
                active_model_runner,
                probe_target,
            )
            validate_kv_physical_snapshot(physical_after_probe)
            graph_after_probe = tracker.snapshot()
            counter_after_probe = dataclasses.asdict(compilation_counter.clone())
            require(
                counter_after_probe["num_cudagraph_captured"]
                == counter_before_probe["num_cudagraph_captured"]
                and graph_after_probe.get("full_capture", 0)
                == graph_before_probe.get("full_capture", 0),
                "physical KV reuse probe unexpectedly recaptured a CUDA Graph",
            )
            require(
                physical_before_probe["va_sha256"]
                == physical_after_probe["va_sha256"],
                "physical KV reuse probe changed the KV VA reservation",
            )
            physical_kv_reuse_probe = {
                "schema": "vllm_uvm_physical_kv_hbm_reuse_probe_v1",
                "ok": True,
                "requested_bytes": int(args.physical_kv_reuse_probe_bytes),
                "device_allocation_ptr": probe_ptr,
                "allocator_backend": allocator_backend,
                "cuda_pointer_is_managed": probe_is_managed,
                "cuda_pointer_probe_reason": probe_pointer_kind,
                "cuda_pointer_probe_status": probe_pointer_status,
                "fill_marker": probe_marker,
                "allocation_touched": probe_marker == 180,
                "allocator_allocated_delta_bytes": (
                    int(cuda_during_probe.get("allocated_bytes", 0))
                    - allocated_before
                ),
                "cuda_memory_before": cuda_before_probe,
                "cuda_memory_during": cuda_during_probe,
                "cuda_memory_after_release_and_restore": cuda_memory_snapshot(
                    torch
                ),
                "physical_before": physical_before_probe,
                "physical_during": physical_during_probe,
                "physical_after_restore": physical_after_probe,
                "restore_result": _compact(restore_result),
                "graph_wrapper_calls_before": graph_before_probe,
                "graph_wrapper_calls_after": graph_after_probe,
                "compilation_counter_before": counter_before_probe,
                "compilation_counter_after": counter_after_probe,
            }

        roundtrip: dict[str, Any] | None = None
        if args.park_resume_roundtrip:
            graph_before_roundtrip = tracker.snapshot()
            counter_before_roundtrip = dataclasses.asdict(compilation_counter.clone())
            diagnostics_before_roundtrip = diagnostics_snapshot(
                utils, get_static_alias_diagnostics
            )
            parameter_before_roundtrip = parameter_snapshot(
                llm,
                exact_names,
                expected_count=args.expected_parameter_count,
                expected_full_bytes=args.expected_full_parameter_bytes,
            )
            second_sampling = SamplingParams(
                max_tokens=2,
                min_tokens=2,
                temperature=0.0,
                seed=args.seed,
                ignore_eos=True,
            )
            llm._validate_and_add_requests(
                prompts=prompts,
                params=second_sampling,
                use_tqdm=False,
                lora_request=None,
                priority=None,
            )

            roundtrip_started = time.perf_counter()
            second_prefill = run_step()
            phase.validate_prefill(second_prefill["schedule"], args.batch_size)
            require(
                not second_prefill["outputs"],
                "roundtrip prefill unexpectedly completed requests",
            )
            require(
                second_prefill["graph_wrapper_delta"].get("full_replay", 0) == 1,
                "roundtrip prefill did not replay the retained alias Graph",
            )
            diagnostics_after_resume = diagnostics_snapshot(
                utils, get_static_alias_diagnostics
            )
            parameter_after_resume = parameter_snapshot(
                llm,
                exact_names,
                expected_count=args.expected_parameter_count,
                expected_full_bytes=args.expected_full_parameter_bytes,
            )
            resume_transitions = transition_delta(
                diagnostics_before_roundtrip, diagnostics_after_resume
            )
            resume_events = [
                item
                for item in resume_transitions
                if item.get("event")
                == "cpu_offload_uvm_original_va_alias_prefill_activated"
            ]
            require(
                len(resume_events) == 1
                and resume_events[0].get("ok") is True
                and resume_events[0].get("post_state") == "sealed",
                "roundtrip prefill did not perform exactly one successful resume",
            )

            second_decode = run_step()
            phase.validate_decode(second_decode["schedule"], args.batch_size)
            require(
                second_decode["graph_wrapper_delta"].get("full_replay", 0) == 1,
                "roundtrip decode did not replay the retained remote Graph",
            )
            diagnostics_after_repark = diagnostics_snapshot(
                utils, get_static_alias_diagnostics
            )
            parameter_after_repark = parameter_snapshot(
                llm,
                exact_names,
                expected_count=args.expected_parameter_count,
                expected_full_bytes=args.expected_full_parameter_bytes,
            )
            repark_transitions = transition_delta(
                diagnostics_after_resume, diagnostics_after_repark
            )
            repark_events = [
                item
                for item in repark_transitions
                if item.get("event")
                == "cpu_offload_uvm_original_va_remote_transition"
                and item.get("post_state") == "remote_parked"
            ]
            require(
                len(repark_events) == 1 and repark_events[0].get("ok") is True,
                "roundtrip decode did not perform exactly one successful re-park",
            )
            require(
                not llm.llm_engine.has_unfinished_requests(),
                "roundtrip requests remain unfinished",
            )
            second_output_records = phase.output_records(second_decode["outputs"])
            require(
                len(second_output_records) == args.batch_size
                and all(item["finished"] for item in second_output_records),
                "roundtrip output batch is incomplete",
            )
            require(
                all(len(item["output_token_ids"]) == 2 for item in second_output_records),
                "roundtrip output token count differs",
            )
            require(
                [item["output_token_ids"] for item in second_output_records]
                == [item["output_token_ids"][:2] for item in output_records],
                "roundtrip output prefix differs from the first cycle",
            )
            graph_after_roundtrip = tracker.snapshot()
            counter_after_roundtrip = dataclasses.asdict(compilation_counter.clone())
            require(
                counter_after_roundtrip["num_cudagraph_captured"]
                == counter_before_roundtrip["num_cudagraph_captured"],
                "park/resume roundtrip unexpectedly recaptured a CUDA Graph",
            )
            roundtrip_manifest = manifest_segment("park_resume_roundtrip")
            roundtrip = {
                "elapsed_ms": (time.perf_counter() - roundtrip_started) * 1000.0,
                "prefill": {
                    key: value
                    for key, value in second_prefill.items()
                    if key != "outputs"
                },
                "decode": {
                    key: value
                    for key, value in second_decode.items()
                    if key != "outputs"
                },
                "diagnostics_before": diagnostics_before_roundtrip,
                "diagnostics_after_resume": diagnostics_after_resume,
                "diagnostics_after_repark": diagnostics_after_repark,
                "parameter_va_snapshots": {
                    "before": parameter_before_roundtrip,
                    "after_resume": parameter_after_resume,
                    "after_repark": parameter_after_repark,
                },
                "resume_transitions": _compact(resume_transitions),
                "repark_transitions": _compact(repark_transitions),
                "output_token_ids_sha256": phase.output_hash(
                    second_output_records
                ),
                "outputs": second_output_records,
                "graph_wrapper_calls_before": graph_before_roundtrip,
                "graph_wrapper_calls_after": graph_after_roundtrip,
                "compilation_counter_before": counter_before_roundtrip,
                "compilation_counter_after": counter_after_roundtrip,
                "manifest": roundtrip_manifest,
            }

        measured_ms = [float(item["elapsed_ms"]) for item in measured]
        warmup_ms = [float(item["elapsed_ms"]) for item in warmup]
        cutover_wall_ms = (
            float(cutover_transition.get("wall_ns", 0)) / 1e6
            if cutover_transition is not None
            else 0.0
        )
        first_excluding_cutover = float(first_decode["elapsed_ms"]) - cutover_wall_ms
        require(first_excluding_cutover >= -0.1, "cutover time exceeds first-decode wall time")

        result["diagnostics"].update(
            {
                "after_prefill": diagnostics_after_prefill,
                "after_first_decode": diagnostics_after_first,
                "final": diagnostics_final,
            }
        )
        result["parameter_va_snapshots"].update(
            {
                "after_prefill": parameter_after_prefill,
                "after_first_decode": parameter_after_first,
                "final": parameter_final,
            }
        )
        result["manifest_segments"].update(
            {
                "serving_prefill": prefill_manifest,
                "first_decode": first_manifest,
                "remaining_decode": remainder_manifest,
            }
        )
        result["steps"] = {
            "prefill": {key: value for key, value in prefill.items() if key != "outputs"},
            "first_decode": {key: value for key, value in first_decode.items() if key != "outputs"},
            "warmup_decode": warmup,
            "measured_decode": measured,
            "park_resume_roundtrip": roundtrip,
        }
        result["dynamic_plan_rebuilds"] = dynamic_rebuilds
        result["dynamic_kv_resizes"] = dynamic_kv_resizes
        result["physical_kv_reuse_probe"] = physical_kv_reuse_probe
        result["three_queue_qualification"] = three_queue_record
        result["kv_autoscale_roundtrip"] = kv_autoscale_roundtrip
        result["kv_multilevel_pressure"] = kv_multilevel_pressure
        result["kv_path_direct_high_preset"] = kv_path_direct_high_preset
        result["kv_path_performance"] = kv_path_performance
        result["timing"] = {
            "prefill_step_ms": float(prefill["elapsed_ms"]),
            "first_decode_step_ms": float(first_decode["elapsed_ms"]),
            "cutover_wall_ms": cutover_wall_ms,
            "first_decode_excluding_cutover_ms": max(0.0, first_excluding_cutover),
            "warmup_decode_step_ms": summarize(warmup_ms),
            "steady_decode_step_ms": summarize(measured_ms),
            "steady_decode_tokens_per_second": args.batch_size * 1000.0 / statistics.median(measured_ms),
            "cycle_e2e_ms": cycle_elapsed_ms,
        }
        result["cutover_transition"] = _compact(cutover_transition)
        result["new_first_decode_transitions"] = _compact(new_transitions)
        result["output_token_ids_sha256"] = phase.output_hash(output_records)
        result["outputs"] = output_records
        result["graph_wrapper_calls_final"] = tracker.snapshot()
        result["compilation_counter_final"] = dataclasses.asdict(compilation_counter.clone())
        result["cuda_memory_final"] = cuda_memory_snapshot(torch)
        result["status"] = "ok"
    except BaseException as exc:
        error = exc
        error_traceback = exc.__traceback__
        result["status"] = "exception"
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        shutdown_contract: dict[str, Any] = {
            "model_runner_shutdown_results": [],
            "cleanup_errors": [],
        }
        try:
            if live_llms:
                live_llms[-1].llm_engine.engine_core.shutdown()
            gc.collect()
        except Exception as exc:
            shutdown_contract["cleanup_errors"].append(repr(exc))
        try:
            from vllm.distributed.device_communicators.pynccl_allocator import (
                set_graph_pool_id,
            )
            from vllm.distributed.parallel_state import (
                cleanup_dist_env_and_memory,
            )

            # Dynamic recapture deliberately retires graph-pool ids. Release
            # the final NCCL pool reference and process groups while CUDA is
            # still fully initialized instead of relying on interpreter
            # module-destruction order.
            set_graph_pool_id(None)
            cleanup_dist_env_and_memory()
            shutdown_contract["distributed_cleanup"] = True
        except Exception as exc:
            shutdown_contract["distributed_cleanup"] = False
            shutdown_contract["cleanup_errors"].append(repr(exc))
        try:
            import torch

            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
        except Exception as exc:
            shutdown_contract["cleanup_errors"].append(repr(exc))
        shutdown_contract["model_runner_shutdown_results"] = _compact(shutdown_results)
        try:
            from vllm.device_allocator.uvm_static_alias import get_static_alias_diagnostics
            from vllm.model_executor.models import utils

            shutdown_contract["diagnostics_after_shutdown"] = diagnostics_snapshot(
                utils, get_static_alias_diagnostics
            )
        except Exception as exc:
            shutdown_contract["diagnostics_error"] = repr(exc)
        try:
            final_records = read_manifest(args.offload_manifest)
            shutdown_contract["manifest_final"] = summarize_manifest(final_records)
        except Exception as exc:
            shutdown_contract["manifest_error"] = repr(exc)
        try:
            from vllm.model_executor.models.utils import (
                release_cpu_offload_uvm_original_va_shutdown_references,
            )

            shutdown_contract["reference_release"] = (
                release_cpu_offload_uvm_original_va_shutdown_references()
            )
        except Exception as exc:
            shutdown_contract["reference_release"] = {
                "ok": False,
                "error": repr(exc),
            }
            shutdown_contract["cleanup_errors"].append(repr(exc))
        try:
            if scheduler is not None and original_schedule is not None:
                scheduler.schedule = original_schedule
            live_llms.clear()
            active_model_runner = None
            kv_policy = None
            block_pool = None
            kv_manager = None
            original_schedule = None
            scheduler = None
            core = None
            client = None
            llm = None
            gc.collect()
            import torch

            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            shutdown_contract["python_references_released"] = True
        except Exception as exc:
            shutdown_contract["python_references_released"] = False
            shutdown_contract["cleanup_errors"].append(repr(exc))
        result["shutdown_contract"] = shutdown_contract
        shutdown_contract["controlled_hard_exit_after_cleanup"] = (
            os.environ.get(
                "VLLM_PHASE_SWITCH_CONTROLLED_HARD_EXIT_AFTER_CLEANUP", "0"
            )
            == "1"
        )
        if original_llm_class is not None:
            try:
                import vllm
                from vllm.v1.worker.gpu_model_runner import GPUModelRunner

                vllm.LLM = original_llm_class
                GPUModelRunner.shutdown_cpu_offload_graph_static_alias = original_runner_shutdown
            except Exception:
                pass
        if shutdown_contract["cleanup_errors"] and error is None:
            error = ContractError(f"shutdown failed: {shutdown_contract['cleanup_errors']!r}")
            error_traceback = error.__traceback__
            result["status"] = "contract_failed"
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    if error is not None:
        raise error.with_traceback(error_traceback)
    print(
        json.dumps(
            {
                "status": result["status"],
                "arm": args.arm,
                "batch_size": args.batch_size,
                "timing": result["timing"],
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if result["shutdown_contract"].get(
        "controlled_hard_exit_after_cleanup", False
    ):
        # PyTorch 2.9 may segfault during interpreter finalization after
        # several shared CUDA graph pools have been retired. All model Graphs,
        # UVM sessions/backing, distributed groups, and process-global tensor
        # owners were explicitly released and audited above. Skip only the
        # unordered interpreter/module destructor phase; runtime failures never
        # reach this branch.
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
