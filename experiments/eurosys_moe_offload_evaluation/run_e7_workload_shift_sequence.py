#!/usr/bin/env python3
"""Run one six-arm E7 workload-shift end-to-end qualification."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import random
import statistics
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any


ARM_ORDER = (
    "frequency_cold",
    "random",
    "frozen_static",
    "gpu_heat_cpu_ewma",
    "oracle",
    "hot_adversarial",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pre-shift-fixture", required=True, type=Path)
    parser.add_argument("--shift-fixture", required=True, type=Path)
    parser.add_argument("--static-plan", required=True, type=Path)
    parser.add_argument("--superset-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--ewma-alpha", type=float, default=0.25)
    parser.add_argument("--max-swaps-per-layer", type=int, default=1)
    parser.add_argument("--min-count-delta", type=int, default=1)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=268435456)
    parser.add_argument("--cpu-offload-gb", type=float, default=30.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compact(value: Any, depth: int = 0) -> Any:
    if depth > 12:
        return "<max-depth>"
    if isinstance(value, dict):
        return {
            str(key): compact(item, depth + 1)
            for key, item in value.items()
            if str(key) not in {"entry_statuses", "page_statuses", "statuses"}
        }
    if isinstance(value, (tuple, list)):
        if len(value) > 256:
            return {
                "compacted": True,
                "count": len(value),
                "sha256": digest_json(value),
            }
        return [compact(item, depth + 1) for item in value]
    return value


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def load_fixture(path: Path, expected_role: str) -> tuple[list[list[int]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{path}: fixture is not an object")
    metadata = payload.get("metadata")
    requests = payload.get("requests")
    require(isinstance(metadata, dict), f"{path}: metadata absent")
    require(metadata.get("fixture_role") == expected_role, f"{path}: role differs")
    require(metadata.get("frozen_prompt_token_ids") is True, f"{path}: tokens not frozen")
    require(isinstance(requests, list) and requests, f"{path}: requests absent")
    prompts: list[list[int]] = []
    for index, record in enumerate(requests):
        tokens = record.get("prompt_token_ids") if isinstance(record, dict) else None
        require(
            isinstance(tokens, list)
            and tokens
            and all(isinstance(token, int) for token in tokens),
            f"{path}: invalid request {index}",
        )
        prompts.append([int(token) for token in tokens])
    return prompts, metadata


def load_plan(path: Path) -> tuple[dict[int, set[int]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("offload_plan") if isinstance(payload, dict) else None
    require(isinstance(records, list) and records, f"{path}: offload plan absent")
    layers: dict[int, set[int]] = {}
    for record in records:
        require(isinstance(record, dict), f"{path}: malformed plan record")
        layer_id = int(record["layer_id"])
        expert_id = int(record["expert_id"])
        require(0 <= expert_id < 128, f"{path}: invalid expert id")
        layers.setdefault(layer_id, set()).add(expert_id)
    return layers, dict(payload.get("metadata") or {})


def model_runner_from_llm(llm: Any) -> Any:
    core = getattr(llm.llm_engine.engine_core, "engine_core", None)
    require(core is not None, "E7 requires VLLM_ENABLE_V1_MULTIPROCESSING=0")
    worker = core.model_executor.driver_worker.worker
    require(worker is not None and worker.model_runner is not None, "model runner absent")
    return worker.model_runner


def region_bytes(region_ids: Any) -> int:
    total = 0
    for region_id in set(str(value) for value in region_ids):
        try:
            total += int(region_id.rsplit(":", 1)[-1], 16)
        except ValueError as exc:
            raise RuntimeError(f"invalid expert region id: {region_id}") from exc
    return total


def placement_snapshot(model_runner: Any) -> dict[str, Any]:
    layers: dict[str, dict[str, Any]] = {}
    all_remote_regions: set[str] = set()
    for module in model_runner.get_model().modules():
        records = getattr(module, "_uvm_expert_records", None)
        layer_id_fn = getattr(module, "_uvm_layer_id", None)
        if not isinstance(records, dict) or not callable(layer_id_fn):
            continue
        layer_id = layer_id_fn()
        if layer_id is None:
            continue
        layer_key_fn = getattr(model_runner, "_uvm_incremental_expert_layer_key", None)
        layer_key = (
            str(layer_key_fn(module))
            if callable(layer_key_fn)
            else str(getattr(module, "layer_name", layer_id))
        )
        remote = sorted(
            int(value)
            for value in getattr(module, "_uvm_remote_read_global_expert_ids", set())
        )
        static_owned = sorted(
            int(value)
            for value in getattr(
                module,
                "_uvm_static_alias_owned_global_expert_ids",
                set(),
            )
        )
        expert_region_bytes: dict[str, int] = {}
        for (record_layer_id, expert_id), record in records.items():
            if int(record_layer_id) != int(layer_id):
                continue
            ids = set(str(value) for value in record.envelope_region_ids)
            expert_region_bytes[str(int(expert_id))] = region_bytes(ids)
            if int(expert_id) in remote:
                all_remote_regions.update(ids)
        swappable = sorted(set(int(value) for value in expert_region_bytes) - set(static_owned))
        layers[layer_key] = {
            "layer_id": int(layer_id),
            "remote_experts": remote,
            "remote_count": len(remote),
            "static_alias_owned_experts": static_owned,
            "swappable_experts": swappable,
            "expert_region_bytes": expert_region_bytes,
        }
    return {
        "layer_count": len(layers),
        "remote_expert_count": sum(layer["remote_count"] for layer in layers.values()),
        "remote_region_bytes": region_bytes(all_remote_regions),
        "layers": dict(sorted(layers.items())),
    }


def normalize_heat(worker_results: list[Any]) -> dict[str, dict[str, Any]]:
    require(len(worker_results) == 1, "single-worker E7 heat result required")
    worker = worker_results[0]
    require(isinstance(worker, dict) and worker.get("ok") is True, "heat RPC failed")
    raw_layers = worker.get("layers")
    require(isinstance(raw_layers, dict) and raw_layers, "heat layers absent")
    normalized: dict[str, dict[str, Any]] = {}
    for layer_key, raw in raw_layers.items():
        require(isinstance(raw, dict), f"invalid heat layer {layer_key}")
        counts = raw.get("expert_token_counts")
        require(isinstance(counts, list) and len(counts) == 128, "heat width differs")
        normalized[str(layer_key)] = {
            **raw,
            "expert_token_counts": [int(value) for value in counts],
            "remote_experts": [int(value) for value in raw.get("remote_experts", ())],
            "swappable_experts": [int(value) for value in raw.get("swappable_experts", ())],
            "static_alias_owned_experts": [
                int(value) for value in raw.get("static_alias_owned_experts", ())
            ],
        }
    return dict(sorted(normalized.items()))


def collect_heat(core: Any, *, reset: bool = True) -> tuple[list[Any], int]:
    started = time.perf_counter_ns()
    raw = core.collective_rpc("get_uvm_dynamic_expert_heat", args=(bool(reset),))
    elapsed = time.perf_counter_ns() - started
    normalize_heat(raw)
    return raw, elapsed


def add_counts(
    accumulator: dict[str, list[int]],
    layers: dict[str, dict[str, Any]],
) -> None:
    for layer_key, layer in layers.items():
        counts = layer["expert_token_counts"]
        prior = accumulator.setdefault(layer_key, [0] * len(counts))
        require(len(prior) == len(counts), "heat accumulator width differs")
        accumulator[layer_key] = [left + right for left, right in zip(prior, counts)]


def rank_targets(
    counts: dict[str, list[int]],
    state: dict[str, Any],
    *,
    hottest: bool,
) -> dict[str, list[int]]:
    targets: dict[str, list[int]] = {}
    for layer_key, layer in state["layers"].items():
        candidates = [int(value) for value in layer["swappable_experts"]]
        slot_count = int(layer["remote_count"])
        require(slot_count > 0 and slot_count <= len(candidates), "invalid dynamic slot count")
        layer_counts = counts[layer_key]
        ranked = sorted(
            candidates,
            key=(
                (lambda expert: (-layer_counts[expert], expert))
                if hottest
                else (lambda expert: (layer_counts[expert], expert))
            ),
        )
        targets[layer_key] = sorted(ranked[:slot_count])
    return targets


def state_target(state: dict[str, Any]) -> dict[str, list[int]]:
    return {
        layer_key: [int(value) for value in layer["remote_experts"]]
        for layer_key, layer in state["layers"].items()
    }


def random_target(state: dict[str, Any], seed: int) -> dict[str, list[int]]:
    targets: dict[str, list[int]] = {}
    for layer_key, layer in state["layers"].items():
        generator = random.Random(seed + 1009 * int(layer["layer_id"]))
        targets[layer_key] = sorted(
            generator.sample(
                [int(value) for value in layer["swappable_experts"]],
                int(layer["remote_count"]),
            )
        )
    return targets


def selected_plan_bytes(state: dict[str, Any], targets: dict[str, list[int]]) -> int:
    return sum(
        int(state["layers"][layer_key]["expert_region_bytes"][str(expert_id)])
        for layer_key, experts in targets.items()
        for expert_id in experts
    )


def apply_target(core: Any, model_runner: Any, targets: dict[str, list[int]]) -> dict[str, Any]:
    before = placement_snapshot(model_runner)
    swaps: dict[str, list[dict[str, object]]] = {}
    planned_h2d_bytes = 0
    planned_d2h_bytes = 0
    for layer_key, desired_values in targets.items():
        layer = before["layers"][layer_key]
        current = set(int(value) for value in layer["remote_experts"])
        desired = set(int(value) for value in desired_values)
        require(len(current) == len(desired), f"{layer_key}: slot-count drift")
        require(desired <= set(layer["swappable_experts"]), f"{layer_key}: target not swappable")
        promote = sorted(current - desired)
        demote = sorted(desired - current)
        require(len(promote) == len(demote), f"{layer_key}: unbalanced target")
        if promote:
            swaps[layer_key] = [
                {"promote_expert": left, "demote_expert": right}
                for left, right in zip(promote, demote)
            ]
            for expert_id in promote:
                planned_h2d_bytes += int(layer["expert_region_bytes"][str(expert_id)])
            for expert_id in demote:
                planned_d2h_bytes += int(layer["expert_region_bytes"][str(expert_id)])
    started = time.perf_counter_ns()
    worker_results: list[Any] = []
    if swaps:
        worker_results = core.collective_rpc(
            "run_uvm_dynamic_expert_swaps",
            args=(swaps,),
        )
    elapsed = time.perf_counter_ns() - started
    ok = not swaps or (
        bool(worker_results)
        and all(isinstance(value, dict) and value.get("ok") is True for value in worker_results)
    )
    require(ok, f"target transition failed: {worker_results!r}")
    after = placement_snapshot(model_runner)
    require(state_target(after) == targets, "target transition placement differs")
    return {
        "ok": True,
        "elapsed_ns": elapsed,
        "selected_layers": len(swaps),
        "selected_swaps": sum(len(value) for value in swaps.values()),
        "planned_h2d_bytes": planned_h2d_bytes,
        "planned_d2h_bytes": planned_d2h_bytes,
        "worker_results": compact(worker_results),
        "placement_before": state_target(before),
        "placement_after": state_target(after),
    }


def initialize_dynamic_slots(
    core: Any,
    model_runner: Any,
    targets: dict[str, list[int]],
) -> dict[str, Any]:
    require(placement_snapshot(model_runner)["remote_expert_count"] == 0, "dynamic slots already exist")
    slot_count = len(next(iter(targets.values())))
    rounds: list[dict[str, Any]] = []
    for slot_index in range(slot_count):
        selected = {
            layer_key: int(experts[slot_index])
            for layer_key, experts in targets.items()
        }
        started = time.perf_counter_ns()
        raw = core._uvm_single_worker_rpc(
            "run_uvm_incremental_expert_offload",
            args=(False, selected),
        )
        elapsed = time.perf_counter_ns() - started
        require(
            isinstance(raw, dict)
            and raw.get("ok") is True
            and int(raw.get("released_expert_count", -1)) == len(targets),
            f"dynamic slot initialization round {slot_index} failed: {raw!r}",
        )
        rounds.append({"slot_index": slot_index, "elapsed_ns": elapsed, "result": compact(raw)})
    after = placement_snapshot(model_runner)
    require(state_target(after) == targets, "initialized dynamic placement differs")
    return {
        "ok": True,
        "rounds": rounds,
        "elapsed_ns": sum(int(item["elapsed_ns"]) for item in rounds),
        "placement_after": state_target(after),
        "remote_region_bytes": int(after["remote_region_bytes"]),
    }


def output_tokens(outputs: list[Any]) -> list[int]:
    require(len(outputs) == 1 and len(outputs[0].outputs) == 1, "output cardinality differs")
    return [int(value) for value in outputs[0].outputs[0].token_ids]


def record_output_digest(
    *,
    arm_name: str,
    request_index: int,
    tokens: list[int],
    reference_outputs: dict[int, str],
    output_digests: dict[int, str],
) -> str:
    token_hash = digest_json(tokens)
    if reference_outputs:
        require(
            reference_outputs[request_index] == token_hash,
            f"{arm_name}: greedy output differs for request {request_index}",
        )
    output_digests[request_index] = token_hash
    return token_hash


def run_request(
    llm: Any,
    torch: Any,
    TokensPrompt: Any,
    sampling: Any,
    prompt_tokens: list[int],
) -> tuple[list[int], int]:
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_tokens)],
        sampling,
        use_tqdm=False,
    )
    torch.cuda.synchronize()
    return output_tokens(outputs), time.perf_counter_ns() - started


def heat_window_metrics(
    layers: dict[str, dict[str, Any]],
    placement: dict[str, list[int]],
) -> dict[str, Any]:
    assignments = 0
    dynamic_remote = 0
    static_remote = 0
    for layer_key, layer in layers.items():
        counts = layer["expert_token_counts"]
        assignments += sum(counts)
        dynamic_remote += sum(counts[index] for index in placement[layer_key])
        static_remote += sum(
            counts[index] for index in layer["static_alias_owned_experts"]
        )
    return {
        "expert_assignments": assignments,
        "dynamic_remote_assignments": dynamic_remote,
        "static_remote_assignments": static_remote,
        "total_remote_assignments": dynamic_remote + static_remote,
    }


def run_arm(
    *,
    name: str,
    start_target: dict[str, list[int]],
    adaptive: bool,
    llm: Any,
    torch: Any,
    TokensPrompt: Any,
    sampling: Any,
    shift_prompts: list[list[int]],
    window_size: int,
    core: Any,
    model_runner: Any,
    compilation_counter: Any,
    reference_outputs: dict[int, str],
) -> tuple[dict[str, Any], dict[int, str]]:
    setup_transition = apply_target(core, model_runner, start_target)
    reset = core.reset_uvm_dynamic_expert_heat()
    require(reset.get("ok") is True, f"{name}: heat reset failed")
    if adaptive:
        core._uvm_dynamic_expert_heat_ewma = {}
    counter_before = int(
        dataclasses.asdict(compilation_counter.clone())["num_cudagraph_captured"]
    )
    windows: list[dict[str, Any]] = []
    output_digests: dict[int, str] = {}
    aggregate_counts: dict[str, list[int]] = {}
    all_latencies: list[int] = []
    for window_index, start in enumerate(range(0, len(shift_prompts), window_size)):
        placement_before = state_target(placement_snapshot(model_runner))
        requests: list[dict[str, Any]] = []
        for request_index in range(start, min(start + window_size, len(shift_prompts))):
            tokens, elapsed_ns = run_request(
                llm,
                torch,
                TokensPrompt,
                sampling,
                shift_prompts[request_index],
            )
            require(len(tokens) == int(sampling.max_tokens), f"{name}: short output")
            token_hash = record_output_digest(
                arm_name=name,
                request_index=request_index,
                tokens=tokens,
                reference_outputs=reference_outputs,
                output_digests=output_digests,
            )
            all_latencies.append(elapsed_ns)
            requests.append(
                {
                    "request_index": request_index,
                    "latency_ns": elapsed_ns,
                    "output_sha256": token_hash,
                }
            )
        maintenance_started = time.perf_counter_ns()
        if adaptive:
            maintenance = core.rebalance_uvm_remote_experts(
                max_swaps_per_layer=int(os.environ["E7_MAX_SWAPS_PER_LAYER"]),
                min_count_delta=int(os.environ["E7_MIN_COUNT_DELTA"]),
                reset_heat=True,
            )
            raw_heat = maintenance.get("snapshot_results")
            require(maintenance.get("ok") is True, f"{name}: rebalance failed: {maintenance!r}")
            require(isinstance(raw_heat, list), f"{name}: raw adaptive heat absent")
        else:
            raw_heat = core.collective_rpc("get_uvm_dynamic_expert_heat", args=(True,))
            maintenance = {
                "ok": all(
                    isinstance(value, dict) and value.get("ok") is True
                    for value in raw_heat
                ),
                "reason": "synchronous_heat_snapshot_only",
                "snapshot_results": raw_heat,
            }
            require(maintenance["ok"] is True, f"{name}: heat snapshot failed")
        torch.cuda.synchronize()
        maintenance_ns = time.perf_counter_ns() - maintenance_started
        layers = normalize_heat(raw_heat)
        require(
            all(
                layer.get("counter_mode") == "graph_stable_synchronous"
                for layer in layers.values()
            ),
            f"{name}: heat counter is not Graph-stable synchronous mode",
        )
        add_counts(aggregate_counts, layers)
        placement_after = state_target(placement_snapshot(model_runner))
        if not adaptive:
            require(placement_after == placement_before, f"{name}: fixed placement drifted")
        metrics = heat_window_metrics(layers, placement_before)
        windows.append(
            {
                "window_index": window_index,
                "request_start": start,
                "request_count": len(requests),
                "requests": requests,
                "placement_before": placement_before,
                "placement_after": placement_after,
                "heat_epoch": {
                    layer_key: int(layer["heat_epoch"])
                    for layer_key, layer in layers.items()
                },
                "heat_sha256": digest_json(
                    {key: value["expert_token_counts"] for key, value in layers.items()}
                ),
                "maintenance_ns": maintenance_ns,
                "maintenance": compact(maintenance),
                **metrics,
            }
        )
    counter_after = int(
        dataclasses.asdict(compilation_counter.clone())["num_cudagraph_captured"]
    )
    require(counter_after == counter_before, f"{name}: request caused Graph recapture")
    totals = {
        key: sum(int(window[key]) for window in windows)
        for key in (
            "expert_assignments",
            "dynamic_remote_assignments",
            "static_remote_assignments",
            "total_remote_assignments",
        )
    }
    arm = {
        "name": name,
        "adaptive": adaptive,
        "setup_transition": setup_transition,
        "start_target": start_target,
        "final_target": state_target(placement_snapshot(model_runner)),
        "request_count": len(shift_prompts),
        "output_tokens_per_request": int(sampling.max_tokens),
        "window_count": len(windows),
        "windows": windows,
        "request_latency_ns": all_latencies,
        "request_latency_median_ns": int(statistics.median(all_latencies)),
        "request_latency_mean_ns": int(statistics.mean(all_latencies)),
        "request_wall_ns": sum(all_latencies),
        "maintenance_ns": sum(int(window["maintenance_ns"]) for window in windows),
        "output_sha256": digest_json(output_digests),
        "route_counts_sha256": digest_json(aggregate_counts),
        "aggregate_expert_token_counts": aggregate_counts,
        "graph_count_before": counter_before,
        "graph_count_after": counter_after,
        **totals,
    }
    return arm, output_digests


def overlap_metrics(
    target: dict[str, list[int]],
    oracle: dict[str, list[int]],
) -> dict[str, Any]:
    intersections = 0
    slots = 0
    jaccards: dict[str, float] = {}
    for layer_key, oracle_values in oracle.items():
        left = set(target[layer_key])
        right = set(oracle_values)
        intersections += len(left & right)
        slots += len(right)
        jaccards[layer_key] = len(left & right) / len(left | right)
    return {
        "plan_hit_rate": intersections / slots,
        "mean_layer_jaccard": statistics.mean(jaccards.values()),
        "per_layer_jaccard": jaccards,
        "matching_slots": intersections,
        "slot_count": slots,
    }


def finalize_arm_metrics(
    arm: dict[str, Any],
    oracle: dict[str, list[int]],
) -> None:
    arm["start_overlap_with_oracle"] = overlap_metrics(arm["start_target"], oracle)
    arm["final_overlap_with_oracle"] = overlap_metrics(arm["final_target"], oracle)
    arm["dynamic_remote_assignment_rate"] = (
        arm["dynamic_remote_assignments"] / arm["expert_assignments"]
    )
    arm["total_remote_assignment_rate"] = (
        arm["total_remote_assignments"] / arm["expert_assignments"]
    )
    arm["output_tokens_per_second"] = (
        arm["request_count"]
        * arm["output_tokens_per_request"]
        / (arm["request_wall_ns"] / 1e9)
    )
    if arm["adaptive"]:
        convergence_window = None
        for window in arm["windows"]:
            score = overlap_metrics(window["placement_after"], oracle)
            window["overlap_after_with_oracle"] = score
            if convergence_window is None and score["plan_hit_rate"] >= 0.95:
                convergence_window = int(window["window_index"])
        arm["ewma_convergence_window"] = convergence_window
        arm["ewma_convergence_requests"] = (
            sum(
                int(window["request_count"])
                for window in arm["windows"][: convergence_window + 1]
            )
            if convergence_window is not None
            else None
        )


def main() -> int:
    args = parse_args()
    require(0.0 <= args.ewma_alpha <= 1.0, "EWMA alpha outside [0, 1]")
    require(args.max_tokens > 0 and args.window_size > 0, "invalid request geometry")
    pre_prompts, pre_metadata = load_fixture(args.pre_shift_fixture.resolve(), "evaluation")
    shift_prompts, shift_metadata = load_fixture(args.shift_fixture.resolve(), "shift")
    require(len(pre_prompts) == len(shift_prompts) == 32, "E7 requires 32+32 requests")
    require(all(len(value) == 128 for value in pre_prompts + shift_prompts), "E7 requires 128-token prompts")
    require(len(shift_prompts) % args.window_size == 0, "window does not divide shift fixture")
    static_plan, static_metadata = load_plan(args.static_plan.resolve())
    superset_plan, superset_metadata = load_plan(args.superset_plan.resolve())
    require(set(static_plan) == set(superset_plan) == set(range(48)), "plan layer set differs")
    dynamic_plan = {
        layer_id: sorted(superset_plan[layer_id] - static_plan[layer_id])
        for layer_id in static_plan
    }
    require(
        all(len(value) == 4 for value in dynamic_plan.values()),
        "E7 requires four dynamic slots per layer",
    )
    require(
        Path(os.environ.get("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE", "")).resolve()
        == args.static_plan.resolve(),
        "configured static plan differs",
    )
    require(
        Path(os.environ.get("VLLM_UVM_INIT_WEIGHT_PLAN_FILE", "")).resolve()
        == args.superset_plan.resolve(),
        "configured superset plan differs",
    )
    require(os.environ.get("VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE") == "1", "dynamic swap disabled")
    require(os.environ.get("VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE") == "1", "incremental offload disabled")
    os.environ["VLLM_UVM_DYNAMIC_EXPERT_HEAT_EWMA_ALPHA"] = str(args.ewma_alpha)
    os.environ["E7_MAX_SWAPS_PER_LAYER"] = str(args.max_swaps_per_layer)
    os.environ["E7_MIN_COUNT_DELTA"] = str(args.min_count_delta)

    result: dict[str, Any] = {
        "schema": "eurosys_e7_workload_shift_qualification_v1",
        "status": "loading",
        "qualification_only": True,
        "statistical_repetitions": 1,
        "arm_order": ARM_ORDER,
        "protocol": {
            "pre_shift_requests": len(pre_prompts),
            "shift_requests_per_arm": len(shift_prompts),
            "prompt_tokens": 128,
            "output_tokens": args.max_tokens,
            "window_size": args.window_size,
            "ewma_alpha": args.ewma_alpha,
            "max_swaps_per_layer": args.max_swaps_per_layer,
            "min_count_delta": args.min_count_delta,
            "dynamic_slots_per_layer": 4,
            "plan_hit_rate_definition": "fraction of dynamic remote slots matching the full-shift offline oracle",
            "remote_assignment_rate_definition": "routed expert assignments landing in remote experts",
            "maintenance_mode": "synchronous GPU-counter D2H and synchronous swap RPC",
            "arm_setup_transitions_excluded_from_request_latency": True,
        },
        "inputs": {
            "model": args.model,
            "pre_shift_fixture": str(args.pre_shift_fixture.resolve()),
            "pre_shift_sha256": sha256_file(args.pre_shift_fixture.resolve()),
            "pre_shift_metadata": pre_metadata,
            "shift_fixture": str(args.shift_fixture.resolve()),
            "shift_sha256": sha256_file(args.shift_fixture.resolve()),
            "shift_metadata": shift_metadata,
            "static_plan": str(args.static_plan.resolve()),
            "static_plan_sha256": sha256_file(args.static_plan.resolve()),
            "static_plan_metadata": static_metadata,
            "superset_plan": str(args.superset_plan.resolve()),
            "superset_plan_sha256": sha256_file(args.superset_plan.resolve()),
            "superset_plan_metadata": superset_metadata,
        },
        "byte_evidence": {
            "per_arm_remote_demand_bytes": {
                "available": False,
                "reason": "current UVM counters do not attribute demand-read bytes to an E7 arm",
            },
            "maintenance_h2d_bytes": "exact requested expert-region geometry",
            "maintenance_d2h_bytes": "exact requested expert-region geometry",
        },
    }
    write_result(args.output, result)
    llm: Any = None
    model_runner: Any = None
    engine_client: Any = None
    error: BaseException | None = None
    cleanup_errors: list[str] = []
    try:
        import torch
        import vllm
        from vllm import SamplingParams, TokensPrompt
        from vllm.compilation.counter import compilation_counter
        from vllm.device_allocator.uvm_static_alias import get_static_alias_diagnostics
        from vllm.model_executor.models import utils as offload_utils

        capture_sizes = [1, 128]
        load_started = time.perf_counter_ns()
        llm = vllm.LLM(
            model=args.model,
            tensor_parallel_size=1,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=False,
            max_model_len=512,
            max_num_seqs=1,
            max_num_batched_tokens=512,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            cpu_offload_gb=args.cpu_offload_gb,
            enable_prefix_caching=False,
            enable_chunked_prefill=False,
            enable_dbo=False,
            compilation_config={
                "mode": "NONE",
                "cudagraph_mode": "FULL_SEPARATE",
                "cudagraph_capture_sizes": capture_sizes,
                "max_cudagraph_capture_size": max(capture_sizes),
            },
            seed=args.seed,
            disable_log_stats=True,
        )
        torch.cuda.synchronize()
        load_ns = time.perf_counter_ns() - load_started
        model_runner = model_runner_from_llm(llm)
        engine_client = llm.llm_engine.engine_core
        core = engine_client.engine_core
        require(core is not None, "in-process core absent")
        graph_count_after_load = int(
            dataclasses.asdict(compilation_counter.clone())["num_cudagraph_captured"]
        )
        before_slots = placement_snapshot(model_runner)
        require(before_slots["layer_count"] == 48, "MoE layer count differs")
        require(before_slots["remote_expert_count"] == 0, "unexpected initial dynamic slots")
        key_by_layer_id = {
            int(layer["layer_id"]): layer_key
            for layer_key, layer in before_slots["layers"].items()
        }
        frozen_target = {
            key_by_layer_id[layer_id]: experts
            for layer_id, experts in dynamic_plan.items()
        }
        initialization = initialize_dynamic_slots(core, model_runner, frozen_target)
        initial_state = placement_snapshot(model_runner)
        require(initial_state["remote_expert_count"] == 48 * 4, "dynamic slot total differs")
        require(
            all(
                not (
                    set(layer["remote_experts"])
                    & set(layer["static_alias_owned_experts"])
                )
                for layer in initial_state["layers"].values()
            ),
            "dynamic and static plans overlap",
        )
        result.update(
            {
                "status": "calibrating",
                "versions": {
                    "vllm": getattr(vllm, "__version__", None),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                },
                "load_ns": load_ns,
                "graph_count_after_load": graph_count_after_load,
                "dynamic_slot_initialization": initialization,
                "initial_placement": compact(initial_state),
                "initial_dynamic_remote_region_bytes": int(initial_state["remote_region_bytes"]),
                "static_alias_selected_bytes": int(static_metadata.get("selected_main_bytes", 0)),
                "diagnostics_after_load": {
                    "model": compact(offload_utils.get_cpu_offload_graph_static_alias_diagnostics()),
                    "driver": compact(get_static_alias_diagnostics()),
                },
            }
        )
        write_result(args.output, result)

        sampling = SamplingParams(
            max_tokens=args.max_tokens,
            min_tokens=args.max_tokens,
            temperature=0.0,
            seed=args.seed,
            ignore_eos=True,
        )
        reset = core.reset_uvm_dynamic_expert_heat()
        require(reset.get("ok") is True, "pre-calibration heat reset failed")
        calibration_latencies: list[int] = []
        calibration_outputs: list[str] = []
        for prompt in pre_prompts:
            tokens, latency_ns = run_request(
                llm, torch, TokensPrompt, sampling, prompt
            )
            require(len(tokens) == args.max_tokens, "short calibration output")
            calibration_latencies.append(latency_ns)
            calibration_outputs.append(digest_json(tokens))
        calibration_raw, calibration_rpc_ns = collect_heat(core, reset=True)
        calibration_layers = normalize_heat(calibration_raw)
        require(
            all(
                layer.get("counter_mode") == "graph_stable_synchronous"
                for layer in calibration_layers.values()
            ),
            "calibration heat counter is not Graph-stable synchronous mode",
        )
        calibration_counts = {
            key: value["expert_token_counts"] for key, value in calibration_layers.items()
        }
        frequency_target = rank_targets(
            calibration_counts,
            initial_state,
            hottest=False,
        )
        seeded_random_target = random_target(initial_state, args.seed)
        result["calibration"] = {
            "request_count": len(pre_prompts),
            "request_latency_ns": calibration_latencies,
            "request_latency_median_ns": int(statistics.median(calibration_latencies)),
            "output_sha256": digest_json(calibration_outputs),
            "heat_rpc_ns": calibration_rpc_ns,
            "heat": compact(calibration_raw),
            "route_counts_sha256": digest_json(calibration_counts),
            "frequency_cold_target": frequency_target,
        }
        result["status"] = "running_arms"
        write_result(args.output, result)

        arms: list[dict[str, Any]] = []
        reference_outputs: dict[int, str] = {}
        frequency_arm, reference_outputs = run_arm(
            name="frequency_cold",
            start_target=frequency_target,
            adaptive=False,
            llm=llm,
            torch=torch,
            TokensPrompt=TokensPrompt,
            sampling=sampling,
            shift_prompts=shift_prompts,
            window_size=args.window_size,
            core=core,
            model_runner=model_runner,
            compilation_counter=compilation_counter,
            reference_outputs=reference_outputs,
        )
        arms.append(frequency_arm)
        shift_counts = frequency_arm["aggregate_expert_token_counts"]
        oracle_target = rank_targets(shift_counts, initial_state, hottest=False)
        adversarial_target = rank_targets(shift_counts, initial_state, hottest=True)
        result["oracle"] = {
            "provenance": "offline full-shift route observed in frequency_cold arm",
            "future_knowledge": True,
            "upper_bound_only": True,
            "target": oracle_target,
            "hot_adversarial_target": adversarial_target,
            "route_counts_sha256": digest_json(shift_counts),
        }
        result["arms"] = arms
        write_result(args.output, result)

        remaining = (
            ("random", seeded_random_target, False),
            ("frozen_static", frozen_target, False),
            ("gpu_heat_cpu_ewma", frequency_target, True),
            ("oracle", oracle_target, False),
            ("hot_adversarial", adversarial_target, False),
        )
        for name, target, adaptive in remaining:
            arm, outputs = run_arm(
                name=name,
                start_target=target,
                adaptive=adaptive,
                llm=llm,
                torch=torch,
                TokensPrompt=TokensPrompt,
                sampling=sampling,
                shift_prompts=shift_prompts,
                window_size=args.window_size,
                core=core,
                model_runner=model_runner,
                compilation_counter=compilation_counter,
                reference_outputs=reference_outputs,
            )
            require(outputs == reference_outputs, f"{name}: output map differs")
            require(
                arm["route_counts_sha256"] == frequency_arm["route_counts_sha256"],
                f"{name}: route trace differs despite greedy-output parity",
            )
            arms.append(arm)
            result["arms"] = arms
            write_result(args.output, result)

        for arm in arms:
            finalize_arm_metrics(arm, oracle_target)
            arm["dynamic_remote_plan_bytes"] = selected_plan_bytes(
                initial_state,
                arm["final_target"],
            )
            arm["online_maintenance_planned_h2d_bytes"] = 0
            arm["online_maintenance_planned_d2h_bytes"] = 0
        adaptive_arm = next(arm for arm in arms if arm["name"] == "gpu_heat_cpu_ewma")
        adaptive_h2d = 0
        adaptive_d2h = 0
        adaptive_swap_count = 0
        for window in adaptive_arm["windows"]:
            maintenance = window["maintenance"]
            plan = maintenance.get("plan") or {}
            swaps = plan.get("swaps_by_layer") or {}
            placement = window["placement_before"]
            for layer_key, records in swaps.items():
                layer = initial_state["layers"][layer_key]
                for record in records:
                    adaptive_swap_count += 1
                    adaptive_h2d += int(
                        layer["expert_region_bytes"][str(int(record["promote_expert"]))]
                    )
                    adaptive_d2h += int(
                        layer["expert_region_bytes"][str(int(record["demote_expert"]))]
                    )
                require(set(placement[layer_key]) <= set(layer["swappable_experts"]), "invalid adaptive placement")
        adaptive_arm["online_swap_count"] = adaptive_swap_count
        adaptive_arm["online_maintenance_planned_h2d_bytes"] = adaptive_h2d
        adaptive_arm["online_maintenance_planned_d2h_bytes"] = adaptive_d2h
        require(adaptive_swap_count > 0, "adaptive arm executed no swaps")
        adaptive_window_medians = [
            int(statistics.median(request["latency_ns"] for request in window["requests"]))
            for window in adaptive_arm["windows"]
        ]
        frequency_window_medians = [
            int(statistics.median(request["latency_ns"] for request in window["requests"]))
            for window in frequency_arm["windows"]
        ]
        convergence_window = adaptive_arm["ewma_convergence_window"]
        recovery_window = (
            int(convergence_window)
            if convergence_window is not None
            else len(adaptive_window_medians) - 1
        )
        adaptive_arm["performance_recovery"] = {
            "first_window_request_latency_median_ns": adaptive_window_medians[0],
            "recovery_window": recovery_window,
            "recovery_window_request_latency_median_ns": adaptive_window_medians[
                recovery_window
            ],
            "first_to_recovery_speed_ratio": (
                adaptive_window_medians[0] / adaptive_window_medians[recovery_window]
            ),
            "frequency_cold_same_window_latency_median_ns": frequency_window_medians[
                recovery_window
            ],
            "adaptive_to_frequency_latency_ratio_at_recovery": (
                adaptive_window_medians[recovery_window]
                / frequency_window_medians[recovery_window]
            ),
            "uses_last_window_when_not_converged": convergence_window is None,
        }
        oracle_arm = next(arm for arm in arms if arm["name"] == "oracle")
        adversarial_arm = next(arm for arm in arms if arm["name"] == "hot_adversarial")
        require(
            oracle_arm["dynamic_remote_assignments"]
            <= min(
                arm["dynamic_remote_assignments"]
                for arm in arms
                if arm["name"] != "gpu_heat_cpu_ewma"
            ),
            "oracle is not the minimum fixed remote-assignment plan",
        )
        require(
            adversarial_arm["dynamic_remote_assignments"]
            >= max(
                arm["dynamic_remote_assignments"]
                for arm in arms
                if arm["name"] != "gpu_heat_cpu_ewma"
            ),
            "adversarial plan is not the maximum fixed remote-assignment plan",
        )
        graph_count_final = int(
            dataclasses.asdict(compilation_counter.clone())["num_cudagraph_captured"]
        )
        require(graph_count_final == graph_count_after_load, "E7 caused Graph recapture")
        result.update(
            {
                "status": "ok",
                "arms": arms,
                "graph_count_final": graph_count_final,
                "reference_output_sha256": frequency_arm["output_sha256"],
                "reference_route_counts_sha256": frequency_arm["route_counts_sha256"],
                "final_placement": compact(placement_snapshot(model_runner)),
                "diagnostics_final": {
                    "model": compact(offload_utils.get_cpu_offload_graph_static_alias_diagnostics()),
                    "driver": compact(get_static_alias_diagnostics()),
                },
            }
        )
    except BaseException as exc:
        error = exc
        result.update(
            {
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        engine_shutdown = False
        if engine_client is None and llm is not None:
            engine_client = llm.llm_engine.engine_core
        if engine_client is not None:
            try:
                engine_client.shutdown()
                engine_shutdown = True
            except Exception as exc:
                cleanup_errors.append(repr(exc))
        try:
            from vllm.model_executor.models.utils import (
                release_cpu_offload_uvm_original_va_shutdown_references,
            )

            result["reference_release"] = compact(
                release_cpu_offload_uvm_original_va_shutdown_references()
            )
        except Exception as exc:
            cleanup_errors.append(repr(exc))
        model_runner = None
        if engine_shutdown and engine_client is not None and hasattr(
            engine_client, "engine_core"
        ):
            engine_client.engine_core = None
        llm = None
        engine_client = None
        gc.collect()
        try:
            from vllm.distributed.device_communicators.pynccl_allocator import (
                set_graph_pool_id,
            )
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

            set_graph_pool_id(None)
            cleanup_dist_env_and_memory()
            result["distributed_cleanup"] = True
        except Exception as exc:
            result["distributed_cleanup"] = False
            cleanup_errors.append(repr(exc))
        with suppress(Exception):
            import torch

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        result["cleanup_errors"] = cleanup_errors
        result["controlled_hard_exit_after_cleanup"] = (
            os.environ.get("VLLM_PHASE_SWITCH_CONTROLLED_HARD_EXIT_AFTER_CLEANUP") == "1"
        )
        if cleanup_errors and error is None:
            result.update(
                {
                    "status": "failed",
                    "error": f"cleanup failed: {cleanup_errors!r}",
                }
            )
        write_result(args.output, result)
    exit_code = 0 if result.get("status") == "ok" else 1
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "arm_count": len(result.get("arms", ())),
                "output": str(args.output),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if exit_code == 0 and result.get("controlled_hard_exit_after_cleanup") is True:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
