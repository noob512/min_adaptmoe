#!/usr/bin/env python3
"""Verify and summarize the repeated steady-session three-arm comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OLD_ROOT = REPO_ROOT / "experiments" / "vllm_prefill_p3_pinned_uva_compare"
sys.path.insert(0, str(OLD_ROOT))
import verify_compare as old  # noqa: E402


LAYERS = tuple(range(4))
LAYER_BYTES = 603_979_776
REGIONS_PER_LAYER = 1_152
BLOCKS_PER_LAYER = 288
ARM_NAMES = {"P": "p3", "C": "pinned", "U": "uva"}
MECHANISMS = {
    "p3": "vllm_prefill_p3_registered_512k_dual_bank_fence",
    "pinned": "vllm_prefill_pinned_copy_exact_l0_l3",
    "uva": "custom_prefill_managed_uva_remote_read_exact_l0_l3",
}


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
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"{path}:{number}: expected object")
        records.append(value)
    return records


def finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label}: expected finite number",
    )
    return float(value)


def percentile(values: list[float], fraction: float) -> float:
    require(bool(values), "cannot summarize an empty sample")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, Any]:
    q1 = percentile(values, 0.25)
    q3 = percentile(values, 0.75)
    return {
        "count": len(values),
        "samples": values,
        "median": statistics.median(values),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": min(values),
        "max": max(values),
    }


def comparison(candidate: float, baseline: float) -> dict[str, float]:
    return {
        "delta_ms": candidate - baseline,
        "ratio": candidate / baseline,
        "percent_slower": (candidate / baseline - 1.0) * 100.0,
    }


def parse_plain_stats(path: Path) -> dict[str, int]:
    require(path.is_file(), f"missing latency stats: {path}")
    result: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].replace("_", "").isalnum():
            try:
                value = int(fields[1], 0)
            except ValueError:
                continue
            result[fields[0]] = value
    return result


def stats_delta(case_dir: Path) -> dict[str, int]:
    before = parse_plain_stats(case_dir / "latency_stats.before.txt")
    after = parse_plain_stats(case_dir / "latency_stats.after.txt")
    keys = set(before) | set(after)
    return {key: after.get(key, 0) - before.get(key, 0) for key in keys}


def exact_counter(delta: dict[str, int], key: str, wanted: int, label: str) -> None:
    require(key in delta, f"{label}: missing latency counter {key}")
    require(delta[key] == wanted, f"{label}: {key}={delta[key]}, expected {wanted}")


def verify_result(
    path: Path,
    *,
    concurrency: int,
    warmup_requests: int,
    conditioning_batches: int,
    measurement_batches: int,
    disable_chunked_prefill: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = load_json(path)
    label = str(path)
    require(result.get("status") == "ok", f"{label}: status is not ok")
    require(
        result.get("schema") == "vllm_prefill_three_arm_steady_batches_v1",
        f"{label}: schema differs",
    )
    config = result.get("config")
    require(isinstance(config, dict), f"{label}: config absent")
    expected_config = {
        "concurrency": concurrency,
        "warmup_requests": warmup_requests,
        "conditioning_batches": conditioning_batches,
        "measurement_batches": measurement_batches,
        "disable_chunked_prefill": disable_chunked_prefill,
        "output_tokens": 1,
        "warmup_output_tokens": 1,
        "require_prompt_token_ids": True,
    }
    for key, wanted in expected_config.items():
        require(config.get(key) == wanted, f"{label}: config {key} differs")
    require(
        finite(config.get("inter_batch_idle_ms"), f"{label}: idle") == 0.0,
        f"{label}: formal run must request zero inter-batch idle",
    )

    warmup = result.get("warmup")
    calls = result.get("calls")
    require(isinstance(warmup, dict) and isinstance(calls, list), f"{label}: batches absent")
    require(warmup.get("role") == "warmup" and warmup.get("index") == 0, f"{label}: warmup differs")
    expected_roles = ["conditioning"] * conditioning_batches + [
        "measurement"
    ] * measurement_batches
    require([item.get("role") for item in calls] == expected_roles, f"{label}: call roles differ")
    require(
        [item.get("index") for item in calls[:conditioning_batches]]
        == list(range(conditioning_batches)),
        f"{label}: conditioning indices differ",
    )
    require(
        [item.get("index") for item in calls[conditioning_batches:]]
        == list(range(measurement_batches)),
        f"{label}: measurement indices differ",
    )

    all_batches = [warmup, *calls]
    previous_end: float | None = None
    for ordinal, batch in enumerate(all_batches):
        role = str(batch.get("role"))
        batch_label = f"{label}:{role}[{batch.get('index')}]"
        start = finite(batch.get("start_epoch"), f"{batch_label}: start")
        end = finite(batch.get("end_epoch"), f"{batch_label}: end")
        elapsed = finite(batch.get("elapsed_seconds"), f"{batch_label}: elapsed")
        require(end > start and elapsed > 0.0, f"{batch_label}: invalid time interval")
        require(batch.get("completed_requests") == concurrency, f"{batch_label}: request count differs")
        expected_prompt_tokens = 1226 if role == "warmup" else 1217
        require(
            batch.get("prompt_tokens") == expected_prompt_tokens,
            f"{batch_label}: prompt tokens differ",
        )
        require(batch.get("output_tokens") == concurrency, f"{batch_label}: output count differs")
        output_hash = batch.get("aggregate_output_token_ids_sha256")
        require(isinstance(output_hash, str) and len(output_hash) == 64, f"{batch_label}: output hash absent")
        if ordinal:
            require(
                finite(batch.get("requested_idle_ms"), f"{batch_label}: requested idle") == 0.0,
                f"{batch_label}: intentional idle is non-zero",
            )
            actual_idle = finite(batch.get("actual_idle_ms"), f"{batch_label}: actual idle")
            require(actual_idle >= 0.0, f"{batch_label}: actual idle is negative")
            assert previous_end is not None
            require(start >= previous_end, f"{batch_label}: batches overlap")
        previous_end = end

    evaluation_hashes = {
        str(item["aggregate_output_token_ids_sha256"]) for item in calls
    }
    require(len(evaluation_hashes) == 1, f"{label}: output changes across repeated eval batches")
    measurement = [item for item in calls if item["role"] == "measurement"]
    elapsed_ms = [finite(item["elapsed_seconds"], label) * 1000.0 for item in measurement]
    ttft_ms = [finite(item["ttft_ms"]["p50"], label) for item in measurement]
    actual_idles = [finite(item["actual_idle_ms"], label) for item in calls]
    summary = {
        "elapsed_ms_batches": elapsed_ms,
        "elapsed_ms_process_median": statistics.median(elapsed_ms),
        "ttft_ms_batches": ttft_ms,
        "ttft_ms_process_median": statistics.median(ttft_ms),
        "evaluation_output_hash": next(iter(evaluation_hashes)),
        "warmup_output_hash": warmup["aggregate_output_token_ids_sha256"],
        "max_actual_idle_ms": max(actual_idles),
        "batch_ranges": [
            {
                "role": item["role"],
                "index": item["index"],
                "start_epoch": item["start_epoch"],
                "end_epoch": item["end_epoch"],
            }
            for item in all_batches
        ],
    }
    return result, summary


def verify_p3(
    case_dir: Path,
    env: dict[str, str],
    summary: dict[str, Any],
    *,
    total_batches: int,
) -> dict[str, Any]:
    expected_env = {
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
    }
    for key, wanted in expected_env.items():
        require(env.get(key) == wanted, f"{case_dir}: P3 environment mismatch {key}")

    trace = load_jsonl(case_dir / "slot_trace.jsonl")
    registrations = [item for item in trace if item.get("operation") == "register_layer_plans"]
    restores = [item for item in trace if item.get("operation") == "restore_layer"]
    require(len(registrations) == 1 and registrations[0].get("ok") is True, f"{case_dir}: registration failed")
    transitions = total_batches * len(LAYERS)
    require(len(restores) == transitions, f"{case_dir}: transition count differs")
    require(
        [item.get("layer_id") for item in restores] == list(LAYERS) * total_batches,
        f"{case_dir}: transition layer order differs",
    )
    ioctl_ms: list[float] = []
    for item in restores:
        low = item.get("restore")
        require(item.get("ok") is True and item.get("registered_plan_used") is True, f"{case_dir}: registered restore failed")
        require(item.get("poisoned") is False, f"{case_dir}: userspace session poisoned")
        require(isinstance(low, dict) and low.get("ok") is True, f"{case_dir}: ioctl failed")
        expected_low = {
            "plan_execute": True,
            "requested_bytes": LAYER_BYTES,
            "accepted_bytes": LAYER_BYTES,
            "cuda_fence_accepted": True,
            "cuda_completion_signaled": True,
            "cuda_stream_wait_enqueued": True,
            "success_contract_ok": True,
            "success_count": REGIONS_PER_LAYER,
            "failure_count": 0,
            "copy_push_count": 1,
            "tlb_invalidate_count": BLOCKS_PER_LAYER,
            "poisoned": False,
            "kernel_poisoned": False,
        }
        for key, wanted in expected_low.items():
            require(low.get(key) == wanted, f"{case_dir}: transition field {key} differs")
        wall_ns = low.get("ioctl_wall_ns")
        require(isinstance(wall_ns, int) and wall_ns > 0, f"{case_dir}: ioctl timing absent")
        ioctl_ms.append(wall_ns / 1_000_000.0)

    for batch_range in summary["batch_ranges"]:
        contained = [
            item
            for item in restores
            if finite(item.get("timestamp"), f"{case_dir}: restore timestamp")
            >= float(batch_range["start_epoch"])
            and float(item["timestamp"]) <= float(batch_range["end_epoch"])
        ]
        require(
            [item.get("layer_id") for item in contained] == list(LAYERS),
            f"{case_dir}: {batch_range['role']}[{batch_range['index']}] does not contain L0-L3",
        )

    delta = stats_delta(case_dir)
    exact = {
        "slot_direct_ioctl_calls": transitions,
        "slot_direct_shared_calls": transitions,
        "slot_direct_successes": transitions,
        "slot_direct_fallbacks": 0,
        "slot_direct_errors": 0,
        "slot_direct_requested_ranges": transitions * REGIONS_PER_LAYER,
        "slot_direct_completed_ranges": transitions * REGIONS_PER_LAYER,
        "slot_direct_pte_4k_blocks": 0,
        "slot_direct_pte_64k_blocks": transitions * BLOCKS_PER_LAYER,
        "slot_direct_pte_2m_blocks": 0,
        "slot_direct_copy_pushes": transitions,
        "slot_direct_memcopy_methods": transitions * REGIONS_PER_LAYER,
        "slot_direct_requested_bytes": transitions * LAYER_BYTES,
        "slot_direct_completed_bytes": transitions * LAYER_BYTES,
        "slot_direct_tracker_waits": transitions,
        "slot_direct_transition_group_clear_fallbacks": 0,
        "slot_direct_copy_run_cache_hits": transitions,
        "slot_direct_metadata_recipe_cache_attempts": transitions,
        "slot_direct_target_recipe_cache_hits": transitions,
        "slot_direct_clear_recipe_cache_hits": transitions - 2,
        "slot_plan_register_calls": 5,
        "slot_plan_pte_prewarm_calls": 5,
        "slot_plan_pte_prewarm_successes": 5,
        "slot_plan_pte_prewarm_failures": 0,
        "slot_plan_copy_cache_calls": 5,
        "slot_plan_copy_cache_successes": 5,
        "slot_plan_copy_cache_failures": 0,
        "slot_plan_execute_calls": transitions,
        "slot_legacy_transition_calls": 0,
    }
    for key, wanted in exact.items():
        exact_counter(delta, key, wanted, str(case_dir))
    for key in (
        "slot_direct_dual_bank_copy_first_calls",
        "slot_direct_shadow_pte_prefill_calls",
        "slot_direct_transition_fast_index_calls",
        "slot_direct_transition_group_clear_calls",
        "slot_direct_transition_direct_cutover_calls",
    ):
        exact_counter(delta, key, transitions - 1, str(case_dir))

    measured_ranges = [
        item for item in summary["batch_ranges"] if item["role"] == "measurement"
    ]
    measured_ioctl = [
        int(item["restore"]["ioctl_wall_ns"]) / 1_000_000.0
        for item in restores
        if any(
            float(window["start_epoch"])
            <= float(item["timestamp"])
            <= float(window["end_epoch"])
            for window in measured_ranges
        )
    ]
    require(len(measured_ioctl) == len(measured_ranges) * len(LAYERS), f"{case_dir}: measured transition count differs")
    return {
        "transitions": transitions,
        "measurement_transitions": len(measured_ioctl),
        "ioctl_ms_measurement_median": statistics.median(measured_ioctl),
        "ioctl_ms_all_median": statistics.median(ioctl_ms),
        "copy_cache_hits": delta["slot_direct_copy_run_cache_hits"],
        "target_recipe_hits": delta["slot_direct_target_recipe_cache_hits"],
        "clear_recipe_hits": delta["slot_direct_clear_recipe_cache_hits"],
        "fallbacks": delta["slot_direct_fallbacks"],
        "errors": delta["slot_direct_errors"],
    }


def verify(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    old.verify_plan(run_root)
    config = load_json(run_root / "run_config.json")
    order = config.get("order")
    require(isinstance(order, str) and len(order) % 3 == 0, "order must contain complete three-process blocks")
    blocks = [order[index : index + 3] for index in range(0, len(order), 3)]
    require(all(set(block) == set(ARM_NAMES) for block in blocks), "each process block must be a permutation of PCU")
    concurrency = int(config.get("concurrency", 0))
    warmup_requests = int(config.get("warmup_requests", 0))
    disable_chunked_prefill = bool(config.get("disable_chunked_prefill", False))
    conditioning_batches = int(config.get("steady_conditioning_batches", 3))
    measurement_batches = int(config.get("steady_measurement_batches", 5))
    require(concurrency == 4 and warmup_requests == 4, "formal run requires four-request batches")
    require(disable_chunked_prefill, "formal run requires chunked prefill disabled")
    require(conditioning_batches == 3 and measurement_batches == 5, "formal repetition contract differs")

    status_path = run_root / "execution_status.tsv"
    require(status_path.is_file(), "execution status absent")
    with status_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(len(rows) == len(order), "execution row count differs")

    cases: list[dict[str, Any]] = []
    process_elapsed: dict[str, list[float]] = defaultdict(list)
    process_ttft: dict[str, list[float]] = defaultdict(list)
    output_hashes: set[str] = set()
    warmup_hashes: set[str] = set()
    engine_contracts: set[tuple[int, int, int]] = set()
    for position, (symbol, row) in enumerate(zip(order, rows), 1):
        arm = ARM_NAMES[symbol]
        require(row.get("arm") == arm and int(row.get("returncode", -1)) == 0, f"process {position}: execution failed")
        case_dir = run_root / str(row.get("directory"))
        require(case_dir.is_dir(), f"process {position}: case directory absent")
        env = old.load_env(case_dir / "environment.txt")
        result, summary = verify_result(
            case_dir / "result.json",
            concurrency=concurrency,
            warmup_requests=warmup_requests,
            conditioning_batches=conditioning_batches,
            measurement_batches=measurement_batches,
            disable_chunked_prefill=disable_chunked_prefill,
        )
        require(result.get("mechanism_class") == MECHANISMS[arm], f"{case_dir}: mechanism class differs")
        result_config = result["config"]
        engine_contracts.add(
            (
                int(result_config.get("max_model_len", 0)),
                int(result_config.get("max_num_batched_tokens", 0)),
                int(result_config.get("max_num_seqs", 0)),
            )
        )
        if arm == "p3":
            evidence = verify_p3(
                case_dir,
                env,
                summary,
                total_batches=1 + conditioning_batches + measurement_batches,
            )
        elif arm == "pinned":
            evidence = old.verify_pinned(case_dir, env)
        else:
            evidence = old.verify_uva(case_dir, env)
        process_elapsed[arm].append(summary["elapsed_ms_process_median"])
        process_ttft[arm].append(summary["ttft_ms_process_median"])
        output_hashes.add(summary["evaluation_output_hash"])
        warmup_hashes.add(summary["warmup_output_hash"])
        cases.append(
            {
                "ordinal": position,
                "block": (position - 1) // 3 + 1,
                "arm": arm,
                "directory": str(case_dir),
                **summary,
                "evidence": evidence,
            }
        )

    require(len(output_hashes) == 1, "evaluation output hashes differ across arms/processes")
    require(len(warmup_hashes) == 1, "warmup output hashes differ across arms/processes")
    require(len(engine_contracts) == 1, "engine geometry differs across arms/processes")
    max_model_len, max_num_batched_tokens, max_num_seqs = next(
        iter(engine_contracts)
    )
    process_count = len(blocks)
    require(all(len(process_elapsed[arm]) == process_count for arm in ARM_NAMES.values()), "arm process counts differ")

    elapsed_distributions = {
        arm: distribution(process_elapsed[arm]) for arm in ARM_NAMES.values()
    }
    ttft_distributions = {
        arm: distribution(process_ttft[arm]) for arm in ARM_NAMES.values()
    }
    arm_medians = {arm: elapsed_distributions[arm]["median"] for arm in ARM_NAMES.values()}
    comparisons = {
        "p3_vs_pinned": comparison(arm_medians["p3"], arm_medians["pinned"]),
        "p3_vs_uva": comparison(arm_medians["p3"], arm_medians["uva"]),
        "pinned_vs_uva": comparison(arm_medians["pinned"], arm_medians["uva"]),
    }

    paired: dict[str, list[dict[str, float | int]]] = defaultdict(list)
    for block_index in range(1, process_count + 1):
        values = {
            item["arm"]: float(item["elapsed_ms_process_median"])
            for item in cases
            if item["block"] == block_index
        }
        for name, candidate, baseline in (
            ("p3_vs_pinned", "p3", "pinned"),
            ("p3_vs_uva", "p3", "uva"),
            ("pinned_vs_uva", "pinned", "uva"),
        ):
            paired[name].append(
                {
                    "block": block_index,
                    "delta_ms": values[candidate] - values[baseline],
                    "ratio": values[candidate] / values[baseline],
                }
            )
    paired_summary = {
        name: {
            "pairs": values,
            "delta_ms_median": statistics.median(
                [float(item["delta_ms"]) for item in values]
            ),
            "ratio_median": statistics.median(
                [float(item["ratio"]) for item in values]
            ),
        }
        for name, values in paired.items()
    }
    return {
        "schema": "vllm_prefill_p3_pinned_uva_steady_compare_v1",
        "status": "pass",
        "run_root": str(run_root),
        "order": order,
        "process_blocks": blocks,
        "sample_unit": "independent_process_within_process_measurement_median",
        "measurement_contract": {
            "concurrency": concurrency,
            "prompt_tokens": 1217,
            "output_tokens": 1,
            "initial_warmup_batches": 1,
            "conditioning_batches": conditioning_batches,
            "measurement_batches": measurement_batches,
            "intentional_inter_batch_idle_ms": 0.0,
            "independent_processes_per_arm": process_count,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "max_num_seqs": max_num_seqs,
            "legacy_engine_geometry_match": (
                max_model_len == 512
                and max_num_batched_tokens == 2048
                and max_num_seqs == 4
            ),
        },
        "elapsed_ms": elapsed_distributions,
        "ttft_ms": ttft_distributions,
        "comparisons": comparisons,
        "block_paired": paired_summary,
        "output_token_ids_sha256": next(iter(output_hashes)),
        "warmup_output_token_ids_sha256": next(iter(warmup_hashes)),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.run_root)
    except (VerificationError, old.VerificationError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
