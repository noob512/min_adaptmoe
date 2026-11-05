#!/usr/bin/env python3
"""Verify the sparse Qwen P3-prefill/remote-decode three-arm experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


LAYERS = tuple(range(48))
EXPERTS = 128
COLD_EXPERTS_PER_LAYER = 12
REGIONS_PER_EXPERT = 9
REGIONS_PER_LAYER = COLD_EXPERTS_PER_LAYER * REGIONS_PER_EXPERT
EXPERT_BYTES = 4_718_592
LAYER_BYTES = COLD_EXPERTS_PER_LAYER * EXPERT_BYTES
TOTAL_EXPERTS = len(LAYERS) * COLD_EXPERTS_PER_LAYER
TOTAL_REGIONS = len(LAYERS) * REGIONS_PER_LAYER
TOTAL_BYTES = len(LAYERS) * LAYER_BYTES
OUTPUT_TOKENS = 8
PREFILL_TRANSITIONS = len(LAYERS)
PURE_P3_TRANSITIONS = OUTPUT_TOKENS * len(LAYERS)
REMOTE_BATCH_MAX_RANGES = 64
REMOTE_BATCH_IOCTL_CALLS = (TOTAL_REGIONS + REMOTE_BATCH_MAX_RANGES - 1) // REMOTE_BATCH_MAX_RANGES
ARM_NAMES = {"H": "hybrid", "P": "p3", "U": "remote"}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL artifact: {path}")
    result: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        require(isinstance(value, dict), f"{path}:{line_number}: not an object")
        result.append(value)
    return result


def load_env(path: Path) -> dict[str, str]:
    require(path.is_file(), f"missing environment artifact: {path}")
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw:
            continue
        require("=" in raw, f"{path}:{line_number}: malformed environment line")
        key, value = raw.split("=", 1)
        require(bool(key) and key not in result, f"{path}:{line_number}: bad key")
        result[key] = value
    return result


def load_stats(path: Path) -> dict[str, int]:
    require(path.is_file(), f"missing stats artifact: {path}")
    result: dict[str, int] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.split()
        if len(fields) != 2:
            continue
        try:
            result[fields[0]] = int(fields[1], 0)
        except ValueError:
            pass
    return result


def stat_delta(case_dir: Path, name: str) -> int:
    before = load_stats(case_dir / "latency_stats.before.txt")
    after = load_stats(case_dir / "latency_stats.after.txt")
    require(name in before and name in after, f"stats lack {name}")
    value = after[name] - before[name]
    require(value >= 0, f"counter moved backwards: {name}")
    return value


def finite(value: Any, label: str, *, positive: bool = False) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} is not finite",
    )
    result = float(value)
    require(result > 0.0 if positive else result >= 0.0, f"{label} is out of range")
    return result


def verify_plan(path: Path) -> dict[int, tuple[int, ...]]:
    plan = load_json(path)

    def coordinates(value: Any, label: str) -> set[tuple[int, int]]:
        require(isinstance(value, list), f"{label} is not a list")
        result: list[tuple[int, int]] = []
        for index, item in enumerate(value):
            require(isinstance(item, dict), f"{label}[{index}] is not an object")
            layer = item.get("layer_id")
            expert = item.get("expert_id")
            require(type(layer) is int and type(expert) is int, f"{label}[{index}] is invalid")
            require(layer in LAYERS and 0 <= expert < EXPERTS, f"{label}[{index}] out of range")
            result.append((layer, expert))
        require(len(result) == len(set(result)), f"{label} contains duplicates")
        return set(result)

    cold = coordinates(plan.get("offload_plan"), "offload_plan")
    hot = coordinates(plan.get("hot_plan"), "hot_plan")
    universe = {(layer, expert) for layer in LAYERS for expert in range(EXPERTS)}
    require(not cold & hot and cold | hot == universe, "cold/hot plan is not a partition")
    selected: dict[int, tuple[int, ...]] = {}
    for layer in LAYERS:
        values = tuple(sorted(expert for candidate, expert in cold if candidate == layer))
        require(len(values) == COLD_EXPERTS_PER_LAYER, f"layer {layer} is not sparse-12")
        selected[layer] = values

    metadata = plan.get("metadata")
    require(isinstance(metadata, dict), "plan metadata is absent")
    require(metadata.get("envelope_bytes") == 524_288, "plan is not 512-KiB region based")
    require(metadata.get("regions_per_expert") == REGIONS_PER_EXPERT, "region geometry differs")
    require(metadata.get("cold_experts") == TOTAL_EXPERTS, "cold expert total differs")
    require(metadata.get("cold_regions") == TOTAL_REGIONS, "cold region total differs")
    require(metadata.get("actual_offload_main_bytes") == TOTAL_BYTES, "cold byte total differs")
    require(metadata.get("shared_hbm_bank_bytes") == LAYER_BYTES, "bank bytes differ")
    require(metadata.get("shared_hbm_bank_slots") == REGIONS_PER_LAYER, "bank slots differ")
    return selected


def verify_result(path: Path) -> tuple[dict[str, Any], dict[str, float], str, int]:
    result = load_json(path)
    require(result.get("status") == "ok", f"{path}: benchmark did not succeed")
    config = result.get("config")
    measurement = result.get("measurement")
    require(isinstance(config, dict) and isinstance(measurement, dict), f"{path}: sections absent")
    expected_config = {
        "concurrency": 1,
        "output_tokens": OUTPUT_TOKENS,
        "warmup_requests": 0,
        "max_num_seqs": 1,
        "disable_chunked_prefill": True,
        "enforce_eager": True,
        "require_prompt_token_ids": True,
    }
    for key, expected in expected_config.items():
        require(config.get(key) == expected, f"{path}: config mismatch: {key}")
    require(measurement.get("completed_requests") == 1, f"{path}: completion count differs")
    require(measurement.get("output_tokens") == OUTPUT_TOKENS, f"{path}: output count differs")
    metrics: dict[str, float] = {}
    for key in ("ttft_ms", "tpot_ms", "request_latency_ms"):
        summary = measurement.get(key)
        require(isinstance(summary, dict), f"{path}: {key} summary absent")
        metrics[key] = finite(summary.get("p50"), f"{path}: {key}.p50", positive=True)
    output_hash = measurement.get("aggregate_output_token_ids_sha256")
    require(isinstance(output_hash, str) and len(output_hash) == 64, f"{path}: output hash absent")
    prompt_tokens = measurement.get("prompt_tokens")
    require(type(prompt_tokens) is int and prompt_tokens > 0, f"{path}: prompt count absent")
    return result, metrics, output_hash, prompt_tokens


def verify_p3_environment(env: dict[str, str], *, hybrid: bool) -> None:
    expected = {
        "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE": "slot_restore",
        "VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD": "1",
        "VLLM_UVM_SLOT_DIRECT_INIT_BANK": "1",
        "VLLM_UVM_SLOT_DIRECT_REUSE_BANK": "1",
        "VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES": "1",
        "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION": "1",
        "VLLM_UVM_SLOT_DIRECT_SHARED_PUSH": "1",
        "VLLM_UVM_SLOT_DIRECT_64K_PTE": "1",
        "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN": "1",
        "VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT": "1",
        "VLLM_UVM_SLOT_DIRECT_CUDA_FENCE": "1",
        "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES": str(2 * LAYER_BYTES),
        "VLLM_UVM_SLOT_DIRECT_MAX_SLOTS": str(2 * REGIONS_PER_LAYER),
        "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ": "1" if hybrid else "0",
        "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_BATCH": "1" if hybrid else "0",
    }
    for key, expected_value in expected.items():
        require(env.get(key) == expected_value, f"P3 environment mismatch: {key}")


def verify_restore_trace(
    case_dir: Path,
    selected: dict[int, tuple[int, ...]],
    *,
    transitions: int,
) -> dict[str, Any]:
    records = load_jsonl(case_dir / "slot_trace.jsonl")
    registrations = [item for item in records if item.get("operation") == "register_layer_plans"]
    restores = [item for item in records if item.get("operation") == "restore_layer"]
    require(len(registrations) == 1, "expected one registered-plan setup")
    registration = registrations[0]
    require(registration.get("ok") is True, "registered-plan setup failed")
    require(registration.get("layer_count") == len(LAYERS), "registered layer count differs")
    require(registration.get("lane_count") == REGIONS_PER_LAYER, "registered lane count differs")
    require(registration.get("bank_count") == 2, "registered plan is not dual bank")
    require(len(restores) == transitions, f"expected {transitions} P3 transitions")
    wall_ms: list[float] = []
    for index, record in enumerate(restores):
        expected_layer = LAYERS[index % len(LAYERS)]
        require(record.get("ok") is True, f"P3 transition {index} failed")
        require(record.get("layer_id") == expected_layer, f"P3 transition {index} layer differs")
        if "expert_ids" in record:
            require(record["expert_ids"] == list(selected[expected_layer]), "P3 expert set differs")
        low = record.get("restore")
        require(isinstance(low, dict) and low.get("ok") is True, f"P3 ioctl {index} failed")
        expected_low = {
            "plan_execute": True,
            "requested_bytes": LAYER_BYTES,
            "accepted_bytes": LAYER_BYTES,
            "success_count": REGIONS_PER_LAYER,
            "remap_count": REGIONS_PER_LAYER,
            "failure_count": 0,
            "conflict_count": 0,
            "poisoned": False,
            "cuda_fence_accepted": True,
            "cuda_completion_signaled": True,
            "cuda_stream_wait_enqueued": True,
        }
        for key, expected_value in expected_low.items():
            require(low.get(key) == expected_value, f"P3 ioctl {index} mismatch: {key}")
        wall_ns = low.get("ioctl_wall_ns")
        require(type(wall_ns) is int and wall_ns > 0, f"P3 ioctl {index} timing absent")
        wall_ms.append(wall_ns / 1_000_000.0)
    require(
        stat_delta(case_dir, "slot_plan_execute_calls") == transitions,
        "execute counter differs",
    )
    require(stat_delta(case_dir, "slot_direct_errors") == 0, "P3 kernel error observed")
    return {
        "transitions": transitions,
        "transition_cycles": transitions // len(LAYERS),
        "transition_bytes": transitions * LAYER_BYTES,
        "ioctl_wall_ms_median": statistics.median(wall_ms),
    }


def verify_hybrid(
    case_dir: Path,
    env: dict[str, str],
    selected: dict[int, tuple[int, ...]],
) -> dict[str, Any]:
    verify_p3_environment(env, hybrid=True)
    evidence = verify_restore_trace(
        case_dir,
        selected,
        transitions=PREFILL_TRANSITIONS,
    )
    transition = load_json(case_dir / "transition_result.json")
    expected = {
        "schema": "vllm_prefill_p3_decode_remote_transition_v1",
        "ok": True,
        "reason": "ok",
        "module_count": len(LAYERS),
        "expert_count": TOTAL_EXPERTS,
        "region_count": TOTAL_REGIONS,
        "close_contract_verified": True,
        "remote_batch": True,
        "remote_batch_ioctl_calls": REMOTE_BATCH_IOCTL_CALLS,
        "remote_mapped_bytes": TOTAL_BYTES,
    }
    for key, expected_value in expected.items():
        require(transition.get(key) == expected_value, f"hybrid transition mismatch: {key}")
    total_ioctl_calls = transition.get("remote_batch_total_ioctl_calls")
    probe_ioctl_calls = transition.get("remote_batch_fd_probe_ioctl_calls")
    fd_candidate_attempts = transition.get("remote_batch_fd_candidate_attempts")
    require(
        type(total_ioctl_calls) is int
        and total_ioctl_calls >= REMOTE_BATCH_IOCTL_CALLS,
        "hybrid total ioctl count is invalid",
    )
    require(
        type(probe_ioctl_calls) is int
        and probe_ioctl_calls == total_ioctl_calls - REMOTE_BATCH_IOCTL_CALLS,
        "hybrid fd probe ioctl count is invalid",
    )
    require(
        type(fd_candidate_attempts) is int and fd_candidate_attempts >= 1,
        "hybrid fd candidate count is invalid",
    )
    for key in ("sync_ns", "close_ns", "remote_map_ns", "wall_ns"):
        require(
            type(transition.get(key)) is int and transition[key] >= 0,
            f"hybrid timing absent: {key}",
        )
    require(transition["wall_ns"] > 0, "hybrid cutover wall time is not positive")
    trace = load_jsonl(case_dir / "slot_trace.jsonl")
    session_releases = [
        item
        for item in trace
        if item.get("operation") == "release" and item.get("scope") == "session"
    ]
    require(len(session_releases) == 1, "hybrid did not close exactly one P3 session")
    require(session_releases[0].get("ok") is True, "hybrid P3 session close failed")
    return {
        **evidence,
        "decode_mode": "custom_remote_read",
        "cutover_wall_ms": transition["wall_ns"] / 1_000_000.0,
        "cutover_sync_ms": transition["sync_ns"] / 1_000_000.0,
        "cutover_close_ms": transition["close_ns"] / 1_000_000.0,
        "cutover_remote_map_ms": transition["remote_map_ns"] / 1_000_000.0,
        "remote_modules": transition["module_count"],
        "remote_experts": transition["expert_count"],
        "remote_regions": transition["region_count"],
        "remote_batch_ioctl_calls": transition["remote_batch_ioctl_calls"],
        "remote_batch_total_ioctl_calls": total_ioctl_calls,
        "remote_batch_fd_probe_ioctl_calls": probe_ioctl_calls,
        "remote_batch_fd_candidate_attempts": fd_candidate_attempts,
        "remote_mapped_bytes": transition["remote_mapped_bytes"],
    }


def verify_pure_p3(
    case_dir: Path,
    env: dict[str, str],
    selected: dict[int, tuple[int, ...]],
) -> dict[str, Any]:
    verify_p3_environment(env, hybrid=False)
    require(not (case_dir / "transition_result.json").exists(), "pure P3 wrote a hybrid result")
    return verify_restore_trace(case_dir, selected, transitions=PURE_P3_TRANSITIONS)


def verify_remote(case_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    require(env.get("VLLM_CPU_OFFLOAD_BACKEND") == "uva", "remote backend differs")
    require(env.get("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE") == "remote_read", "remote mode differs")
    require(env.get("VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE") == "1", "remote Stage M disabled")
    require(env.get("VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ") == "0", "remote arm enabled hybrid")
    require(
        env.get("VLLM_UVM_PREFILL_P3_DECODE_REMOTE_BATCH") == "0",
        "remote arm enabled batch cutover",
    )
    trace = load_jsonl(case_dir / "slot_trace.jsonl")
    require(
        not any(
            item.get("operation") in {"register_layer_plans", "restore_layer"}
            for item in trace
        ),
        "remote arm unexpectedly used P3",
    )
    require(stat_delta(case_dir, "slot_plan_execute_calls") == 0, "remote arm executed a slot plan")
    placement = load_json(case_dir / "runtime_remote_placement_manifest.json")
    expected = {
        "status": "ok",
        "complete": True,
        "remote_expert_count": TOTAL_EXPERTS,
        "remote_region_count": TOTAL_REGIONS,
        "remote_region_bytes": TOTAL_BYTES,
        "offloaded_expert_count": 0,
        "release_time_remote_mapping_verified": True,
    }
    for key, expected_value in expected.items():
        require(placement.get(key) == expected_value, f"remote placement mismatch: {key}")
    return {
        "remote_experts": TOTAL_EXPERTS,
        "remote_regions": TOTAL_REGIONS,
        "remote_bytes": TOTAL_BYTES,
    }


def comparison(candidate: float, baseline: float) -> dict[str, float]:
    return {
        "delta_ms": candidate - baseline,
        "ratio": candidate / baseline,
        "percent_change": (candidate / baseline - 1.0) * 100.0,
    }


def verify(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    selected = verify_plan(run_root / "inputs" / "exact_l0_l3_plan.json")
    config = load_json(run_root / "run_config.json")
    order = config.get("order")
    require(isinstance(order, str) and order, "run order is absent")
    require(set(order) == set(ARM_NAMES), "run order must contain H, P, and U")
    expected_run_config = {
        "schema": "vllm_prefill_p3_decode_remote_run_v1",
        "layers": len(LAYERS),
        "cold_experts_per_layer": COLD_EXPERTS_PER_LAYER,
        "regions_per_layer": REGIONS_PER_LAYER,
        "bytes_per_layer": LAYER_BYTES,
        "total_regions": TOTAL_REGIONS,
        "total_bytes": TOTAL_BYTES,
        "concurrency": 1,
        "output_tokens": OUTPUT_TOKENS,
        "max_num_seqs": 1,
        "chunked_prefill": False,
        "enforce_eager": True,
        "hybrid_remote_batch_max_ranges": REMOTE_BATCH_MAX_RANGES,
        "hybrid_remote_batch_ioctl_calls": REMOTE_BATCH_IOCTL_CALLS,
        "hybrid_remote_mapped_bytes": TOTAL_BYTES,
    }
    for key, expected_value in expected_run_config.items():
        require(config.get(key) == expected_value, f"run config mismatch: {key}")

    with (run_root / "execution_status.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(len(rows) == len(order), "execution row count differs")

    samples: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in ARM_NAMES.values()
    }
    output_hashes: set[str] = set()
    prompt_counts: set[int] = set()
    details: list[dict[str, Any]] = []
    for ordinal, (symbol, row) in enumerate(zip(order, rows), 1):
        arm = ARM_NAMES[symbol]
        require(int(row.get("ordinal", -1)) == ordinal, f"row {ordinal}: ordinal differs")
        require(row.get("arm") == arm, f"row {ordinal}: arm differs")
        require(int(row.get("returncode", -1)) == 0, f"row {ordinal}: arm failed")
        case_dir = run_root / str(row.get("directory"))
        result, metrics, output_hash, prompt_tokens = verify_result(case_dir / "result.json")
        require(
            result.get("mechanism_class") == config["mechanism_classes"][arm],
            f"{arm}: mechanism class differs",
        )
        env = load_env(case_dir / "environment.txt")
        if arm == "hybrid":
            evidence = verify_hybrid(case_dir, env, selected)
        elif arm == "p3":
            evidence = verify_pure_p3(case_dir, env, selected)
        else:
            evidence = verify_remote(case_dir, env)
        for key, value in metrics.items():
            samples[arm][key].append(value)
        output_hashes.add(output_hash)
        prompt_counts.add(prompt_tokens)
        details.append(
            {
                "ordinal": ordinal,
                "arm": arm,
                "directory": str(case_dir),
                "metrics": metrics,
                "output_token_ids_sha256": output_hash,
                "evidence": evidence,
            }
        )

    require(len(output_hashes) == 1, "output token hashes differ across arms")
    require(len(prompt_counts) == 1, "prompt token counts differ across arms")
    medians = {
        arm: {metric: statistics.median(values) for metric, values in arm_samples.items()}
        for arm, arm_samples in samples.items()
    }
    for arm in ARM_NAMES.values():
        require(
            set(medians[arm]) == {"ttft_ms", "tpot_ms", "request_latency_ms"},
            f"{arm}: metrics absent",
        )

    return {
        "schema": "vllm_prefill_p3_decode_remote_compare_v1",
        "status": "pass",
        "run_root": str(run_root),
        "order": order,
        "sparse_weight_contract": {
            "layers": len(LAYERS),
            "cold_experts_per_layer": COLD_EXPERTS_PER_LAYER,
            "regions_per_expert": REGIONS_PER_EXPERT,
            "regions_per_layer": REGIONS_PER_LAYER,
            "bytes_per_layer": LAYER_BYTES,
            "total_experts": TOTAL_EXPERTS,
            "total_regions": TOTAL_REGIONS,
            "total_bytes": TOTAL_BYTES,
        },
        "measurement_contract": {
            "concurrency": 1,
            "output_tokens": OUTPUT_TOKENS,
            "max_num_seqs": 1,
            "chunked_prefill": False,
            "enforce_eager": True,
            "prompt_tokens": next(iter(prompt_counts)),
        },
        "hybrid_remote_cutover_contract": {
            "batch": True,
            "max_ranges_per_ioctl": REMOTE_BATCH_MAX_RANGES,
            "ioctl_calls": REMOTE_BATCH_IOCTL_CALLS,
            "module_count": len(LAYERS),
            "expert_count": TOTAL_EXPERTS,
            "region_count": TOTAL_REGIONS,
            "remote_mapped_bytes": TOTAL_BYTES,
        },
        "metric_medians_ms": medians,
        "comparison": {
            metric: {
                "hybrid_vs_p3": comparison(medians["hybrid"][metric], medians["p3"][metric]),
                "hybrid_vs_remote": comparison(
                    medians["hybrid"][metric], medians["remote"][metric]
                ),
                "p3_vs_remote": comparison(medians["p3"][metric], medians["remote"][metric]),
            }
            for metric in ("ttft_ms", "tpot_ms", "request_latency_ms")
        },
        "output_token_ids_sha256": next(iter(output_hashes)),
        "cases": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        summary = verify(args.run_root)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, VerificationError) as exc:
        print(f"verification failed: {exc}")
        return 1
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
