#!/usr/bin/env python3
"""Verify and summarize the strict P3/pinned/managed-UVA phase comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OLD_VERIFY_ROOT = REPO_ROOT / "experiments" / "vllm_prefill_p3_pinned_uva_compare"
sys.path.insert(0, str(OLD_VERIFY_ROOT))
import verify_compare as old  # noqa: E402


LAYERS = tuple(range(4))
EXPERTS = 128
REGIONS_PER_EXPERT = 9
REGIONS_PER_LAYER = EXPERTS * REGIONS_PER_EXPERT
BLOCKS_PER_LAYER = 288
LAYER_BYTES = 603_979_776
TOTAL_BYTES = len(LAYERS) * LAYER_BYTES
ARM_NAMES = {"P": "p3", "C": "pinned", "U": "uva"}
MECHANISMS = {
    "p3": "sync_cmd96_p3_registered_512k_dual_bank",
    "pinned": "vllm_pinned_copy_exact_l0_l3",
    "uva": "custom_managed_uva_remote_read_exact_l0_l3",
}
COMPARISONS = (
    ("p3_vs_pinned", "p3", "pinned"),
    ("p3_vs_uva", "p3", "uva"),
    ("pinned_vs_uva", "pinned", "uva"),
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL artifact: {path}")
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        require(isinstance(value, dict), f"{path}:{line_number}: expected object")
        records.append(value)
    return records


def finite(value: Any, label: str, *, positive: bool = False) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label}: expected finite number",
    )
    result = float(value)
    require(result > 0.0 if positive else result >= 0.0, f"{label}: out of range")
    return result


def percentile(values: list[float], fraction: float) -> float:
    require(bool(values), "cannot summarize an empty distribution")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, Any]:
    require(bool(values), "cannot summarize an empty distribution")
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
        "percent_change": (candidate / baseline - 1.0) * 100.0,
    }


def parse_stats(path: Path) -> dict[str, int]:
    require(path.is_file(), f"missing latency stats: {path}")
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


def stats_delta(case_dir: Path) -> dict[str, int]:
    before = parse_stats(case_dir / "latency_stats.before.txt")
    after = parse_stats(case_dir / "latency_stats.after.txt")
    return {key: after.get(key, 0) - value for key, value in before.items()}


def exact_counter(delta: dict[str, int], key: str, wanted: int, label: str) -> None:
    require(key in delta, f"{label}: missing counter {key}")
    require(delta[key] == wanted, f"{label}: {key}={delta[key]}, expected {wanted}")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixture_contract(path: Path, batches: list[int]) -> dict[int, list[int]]:
    fixture = load_json(path)
    requests = fixture.get("requests")
    require(isinstance(requests, list), "fixture request list is absent")
    require(len(requests) >= max(batches), "fixture is smaller than the largest batch")
    lengths: list[int] = []
    for index, request in enumerate(requests[: max(batches)]):
        require(isinstance(request, dict), f"fixture request {index} is invalid")
        token_ids = request.get("prompt_token_ids")
        require(
            isinstance(token_ids, list)
            and token_ids
            and all(type(token) is int and token >= 0 for token in token_ids),
            f"fixture request {index} lacks frozen token IDs",
        )
        require(
            request.get("prompt_tokens", len(token_ids)) == len(token_ids),
            f"fixture request {index} token count differs",
        )
        lengths.append(len(token_ids))
    return {batch: lengths[:batch] for batch in batches}


def verify_prefill_schedule(
    schedule: dict[str, Any], batch: int, prompt_lengths: list[int], label: str
) -> None:
    require(len(schedule.get("new_request_ids", [])) == batch, f"{label}: new count")
    prompt_tokens = schedule.get("new_request_prompt_tokens")
    scheduled = schedule.get("num_scheduled_tokens")
    require(isinstance(prompt_tokens, dict) and scheduled == prompt_tokens, f"{label}: split prefill")
    require(
        sorted(int(value) for value in prompt_tokens.values()) == sorted(prompt_lengths),
        f"{label}: frozen prompt lengths differ",
    )
    require(
        schedule.get("total_num_scheduled_tokens") == sum(prompt_lengths),
        f"{label}: prefill token total differs",
    )
    for key in (
        "cached_request_ids",
        "resumed_request_ids",
        "scheduled_spec_decode_request_ids",
        "scheduled_encoder_request_ids",
        "preempted_request_ids",
    ):
        require(not schedule.get(key), f"{label}: unexpected {key}")


def verify_decode_schedule(schedule: dict[str, Any], batch: int, label: str) -> None:
    scheduled = schedule.get("num_scheduled_tokens")
    require(isinstance(scheduled, dict), f"{label}: scheduled map absent")
    require(not schedule.get("new_request_ids"), f"{label}: contains new request")
    require(len(scheduled) == batch, f"{label}: scheduled request count differs")
    require(all(value == 1 for value in scheduled.values()), f"{label}: not one token/request")
    require(schedule.get("total_num_scheduled_tokens") == batch, f"{label}: token total differs")
    cached_ids = [str(value) for value in schedule.get("cached_request_ids", [])]
    require(len(cached_ids) == batch, f"{label}: cached count differs")
    require(set(scheduled) == set(cached_ids), f"{label}: scheduled/cached IDs differ")
    for key in (
        "resumed_request_ids",
        "scheduled_spec_decode_request_ids",
        "scheduled_encoder_request_ids",
        "preempted_request_ids",
    ):
        require(not schedule.get(key), f"{label}: unexpected {key}")


def verify_result(
    path: Path,
    *,
    mechanism: str,
    batch_order: list[int],
    prompt_lengths: dict[int, list[int]],
    conditioning_repeats: int,
    measurement_repeats: int,
    warmup_decode_steps: int,
    measured_decode_steps: int,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    expected_model: Path,
    expected_fixture: Path,
    expected_fixture_sha256: str,
    expected_cpu_offload_gb: float,
) -> tuple[dict[str, Any], dict[int, dict[str, float]], dict[int, str]]:
    result = load_json(path)
    label = str(path)
    require(result.get("schema") == "vllm_p3_pinned_uva_phase_steps_v1", f"{label}: schema")
    require(result.get("status") == "ok", f"{label}: status")
    require(result.get("mechanism_class") == mechanism, f"{label}: mechanism")
    config = result.get("config")
    require(isinstance(config, dict), f"{label}: config absent")
    expected_config = {
        "batch_sizes": batch_order,
        "conditioning_repeats": conditioning_repeats,
        "measurement_repeats": measurement_repeats,
        "warmup_decode_steps": warmup_decode_steps,
        "measured_decode_steps": measured_decode_steps,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "engine_core_multiprocessing": False,
        "frozen_prompt_token_ids": True,
        "disable_log_stats": True,
        "request_metrics_without_default_loggers": True,
        "seed": 0,
    }
    for key, wanted in expected_config.items():
        require(config.get(key) == wanted, f"{label}: config {key} differs")
    require(
        Path(str(config.get("model", ""))).resolve() == expected_model.resolve(),
        f"{label}: model differs",
    )
    require(
        Path(str(config.get("fixture", ""))).resolve() == expected_fixture.resolve(),
        f"{label}: fixture differs",
    )
    require(
        math.isclose(
            finite(config.get("cpu_offload_gb"), f"{label}: cpu_offload_gb"),
            expected_cpu_offload_gb,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        f"{label}: cpu_offload_gb differs",
    )
    fixture_section = result.get("fixture")
    require(isinstance(fixture_section, dict), f"{label}: fixture section absent")
    require(
        fixture_section.get("sha256") == expected_fixture_sha256,
        f"{label}: fixture SHA-256 differs",
    )

    cycles = result.get("cycles")
    require(isinstance(cycles, list), f"{label}: cycles absent")
    repeats = conditioning_repeats + measurement_repeats
    require(len(cycles) == len(batch_order) * repeats, f"{label}: cycle count differs")
    output_hashes: dict[int, set[str]] = defaultdict(set)
    summaries: dict[int, dict[str, float]] = {}
    cursor = 0
    for batch in batch_order:
        expected_roles = ["conditioning"] * conditioning_repeats + [
            "measurement"
        ] * measurement_repeats
        role_indices = {"conditioning": 0, "measurement": 0}
        measured_prefill: list[float] = []
        measured_decode_repeat: list[float] = []
        measured_e2e: list[float] = []
        measured_ttft: list[float] = []
        measured_tpot: list[float] = []
        measured_latency: list[float] = []
        for role in expected_roles:
            cycle = cycles[cursor]
            cycle_label = f"{label}:cycle[{cursor}]"
            cursor += 1
            require(cycle.get("role") == role, f"{cycle_label}: role differs")
            require(cycle.get("role_index") == role_indices[role], f"{cycle_label}: role index")
            role_indices[role] += 1
            require(cycle.get("batch_size") == batch, f"{cycle_label}: batch differs")
            require(cycle.get("prompt_tokens") == sum(prompt_lengths[batch]), f"{cycle_label}: prompts")
            verify_prefill_schedule(
                cycle.get("prefill_schedule"), batch, prompt_lengths[batch], cycle_label
            )
            prefill_ms = finite(cycle.get("prefill_elapsed_ms"), f"{cycle_label}: prefill", positive=True)
            warmup = cycle.get("warmup_decode_steps")
            measured = cycle.get("measured_decode_steps")
            require(isinstance(warmup, list) and len(warmup) == warmup_decode_steps, f"{cycle_label}: warmup")
            require(isinstance(measured, list) and len(measured) == measured_decode_steps, f"{cycle_label}: measured")
            for kind, steps in (("warmup", warmup), ("measured", measured)):
                for step_index, step in enumerate(steps, 1):
                    require(step.get("step") == step_index, f"{cycle_label}: {kind} index")
                    finite(step.get("elapsed_ms"), f"{cycle_label}: {kind} timing", positive=True)
                    verify_decode_schedule(step.get("schedule"), batch, f"{cycle_label}:{kind}[{step_index}]")
            outputs = cycle.get("outputs")
            require(isinstance(outputs, list) and len(outputs) == batch, f"{cycle_label}: outputs")
            total_output_tokens = 1 + warmup_decode_steps + measured_decode_steps
            require(
                all(
                    item.get("finished") is True
                    and len(item.get("output_token_ids", [])) == total_output_tokens
                    for item in outputs
                ),
                f"{cycle_label}: output contract differs",
            )
            expected_hash = sha256_json([item["output_token_ids"] for item in outputs])
            require(cycle.get("output_token_ids_sha256") == expected_hash, f"{cycle_label}: hash")
            output_hashes[batch].add(expected_hash)
            e2e_ms = finite(cycle.get("cycle_elapsed_ms"), f"{cycle_label}: E2E", positive=True)
            ttft_ms = finite(
                cycle.get("request_ttft_ms", {}).get("median"),
                f"{cycle_label}: TTFT",
                positive=True,
            )
            tpot_ms = finite(
                cycle.get("request_tpot_ms", {}).get("median"),
                f"{cycle_label}: TPOT",
                positive=True,
            )
            latency_ms = finite(
                cycle.get("request_latency_ms", {}).get("median"),
                f"{cycle_label}: request latency",
                positive=True,
            )
            decode_values = [float(item["elapsed_ms"]) for item in measured]
            decode_median = statistics.median(decode_values)
            require(
                math.isclose(
                    finite(cycle.get("decode_step_ms", {}).get("median"), f"{cycle_label}: decode median"),
                    decode_median,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ),
                f"{cycle_label}: decode summary differs",
            )
            if role == "measurement":
                measured_prefill.append(prefill_ms)
                measured_decode_repeat.append(decode_median)
                measured_e2e.append(e2e_ms)
                measured_ttft.append(ttft_ms)
                measured_tpot.append(tpot_ms)
                measured_latency.append(latency_ms)
        summaries[batch] = {
            "prefill_step_ms": statistics.median(measured_prefill),
            "decode_step_ms": statistics.median(measured_decode_repeat),
            "cycle_e2e_ms": statistics.median(measured_e2e),
            "request_ttft_p50_ms": statistics.median(measured_ttft),
            "request_tpot_p50_ms": statistics.median(measured_tpot),
            "request_latency_p50_ms": statistics.median(measured_latency),
        }

    phase_summary = result.get("phase_summary")
    require(isinstance(phase_summary, dict), f"{label}: phase summary absent")
    for batch, summary in summaries.items():
        stored = phase_summary.get(str(batch))
        require(isinstance(stored, dict), f"{label}: batch {batch} summary absent")
        for result_key, stored_key in (
            ("prefill_step_ms", "prefill_step_ms"),
            ("decode_step_ms", "decode_repeat_median_ms"),
            ("cycle_e2e_ms", "cycle_e2e_ms"),
            ("request_ttft_p50_ms", "request_ttft_p50_ms"),
            ("request_tpot_p50_ms", "request_tpot_p50_ms"),
            ("request_latency_p50_ms", "request_latency_p50_ms"),
        ):
            stored_section = stored.get(stored_key)
            require(isinstance(stored_section, dict), f"{label}: {stored_key} absent")
            require(
                math.isclose(
                    finite(stored_section.get("median"), f"{label}: {stored_key}"),
                    summary[result_key],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ),
                f"{label}: {stored_key} differs from cycles",
            )
    require(all(len(values) == 1 for values in output_hashes.values()), f"{label}: output changes between repeats")
    return result, summaries, {batch: next(iter(values)) for batch, values in output_hashes.items()}


def verify_p3(case_dir: Path, env: dict[str, str], transitions: int) -> dict[str, Any]:
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
        "VLLM_UVM_SLOT_DIRECT_TRACE_MODE": "compact",
        "VLLM_UVM_SLOT_DIRECT_BANK_LEVEL_COMMIT": "1",
    }
    for key, wanted in expected_env.items():
        require(env.get(key) == wanted, f"{case_dir}: P3 environment {key}")
    trace = load_jsonl(case_dir / "slot_trace.jsonl")
    registrations = [item for item in trace if item.get("operation") == "register_layer_plans"]
    restores = [item for item in trace if item.get("operation") == "restore_layer"]
    require(len(registrations) == 1 and registrations[0].get("ok") is True, f"{case_dir}: registration")
    require(registrations[0].get("bank_level_commit") is True, f"{case_dir}: registration commit mode")
    require(len(restores) == transitions, f"{case_dir}: transition count differs")
    require(
        [item.get("layer_id") for item in restores] == list(LAYERS) * (transitions // len(LAYERS)),
        f"{case_dir}: transition layer order differs",
    )
    ioctl_ms: list[float] = []
    for index, item in enumerate(restores):
        require(item.get("trace_schema") == "slot_direct_compact_v1", f"{case_dir}: trace mode")
        require(item.get("ok") is True and item.get("registered_plan_used") is True, f"{case_dir}: restore {index}")
        require(item.get("bank_level_commit") is True, f"{case_dir}: bank commit {index}")
        require(item.get("userspace_commit_slots") == 0, f"{case_dir}: O(1) commit {index}")
        require(item.get("poisoned") is False, f"{case_dir}: poison {index}")
        low = item.get("restore")
        require(isinstance(low, dict) and low.get("ok") is True, f"{case_dir}: ioctl {index}")
        expected_low = {
            "plan_execute": True,
            "requested_bytes": LAYER_BYTES,
            "accepted_bytes": LAYER_BYTES,
            "requested_count": REGIONS_PER_LAYER,
            "success_count": REGIONS_PER_LAYER,
            "failure_count": 0,
            "copy_push_count": 1,
            "tlb_invalidate_count": BLOCKS_PER_LAYER,
            "cuda_fence_accepted": True,
            "cuda_completion_signaled": True,
            "cuda_stream_wait_enqueued": True,
            "success_contract_ok": True,
            "entry_statuses_elided": True,
            "poisoned": False,
            "kernel_poisoned": False,
        }
        for key, wanted in expected_low.items():
            require(low.get(key) == wanted, f"{case_dir}: restore {index} field {key}")
        ioctl_ms.append(finite(low.get("ioctl_wall_ns"), f"{case_dir}: ioctl timing", positive=True) / 1_000_000.0)

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
        "slot_session_abort_calls": 1,
        "slot_session_abort_successes": 1,
        "slot_session_abort_failures": 0,
        "slot_session_abort_released_slots": 2 * REGIONS_PER_LAYER,
        "slot_session_abort_released_bytes": 2 * LAYER_BYTES,
        "slot_session_abort_remaining_slots": 0,
        "slot_legacy_transition_calls": 0,
    }
    for key, wanted in exact.items():
        exact_counter(delta, key, wanted, str(case_dir))
    require(
        "slot_plan_destroy_calls" in delta,
        f"{case_dir}: missing counter slot_plan_destroy_calls",
    )
    registered_plans = delta["slot_plan_register_calls"]
    explicitly_destroyed_plans = delta["slot_plan_destroy_calls"]
    require(
        registered_plans - explicitly_destroyed_plans == 1,
        f"{case_dir}: registered plans ({registered_plans}) minus explicitly "
        f"destroyed plans ({explicitly_destroyed_plans}) must leave exactly one "
        "resident plan for session abort",
    )
    for key in (
        "slot_direct_dual_bank_copy_first_calls",
        "slot_direct_shadow_pte_prefill_calls",
        "slot_direct_transition_fast_index_calls",
        "slot_direct_transition_group_clear_calls",
        "slot_direct_transition_direct_cutover_calls",
    ):
        exact_counter(delta, key, transitions - 1, str(case_dir))
    return {
        "executor": "synchronous_fenced_cmd96",
        "transitions": transitions,
        "transition_bytes": transitions * LAYER_BYTES,
        "ioctl_wall_ms_median": statistics.median(ioctl_ms),
        "copy_cache_hits": delta["slot_direct_copy_run_cache_hits"],
        "target_recipe_hits": delta["slot_direct_target_recipe_cache_hits"],
        "clear_recipe_hits": delta["slot_direct_clear_recipe_cache_hits"],
        "fallbacks": delta["slot_direct_fallbacks"],
        "errors": delta["slot_direct_errors"],
        "registered_plans": registered_plans,
        "plan_destroy_attempts_before_abort": explicitly_destroyed_plans,
        "resident_plans_before_abort": registered_plans - explicitly_destroyed_plans,
        "abort_successes": delta["slot_session_abort_successes"],
        "abort_released_slots": delta["slot_session_abort_released_slots"],
        "abort_released_bytes": delta["slot_session_abort_released_bytes"],
        "abort_remaining_slots": delta["slot_session_abort_remaining_slots"],
    }


def verify(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    old.verify_plan(run_root)
    config = load_json(run_root / "run_config.json")
    require(config.get("schema") == "vllm_p3_pinned_uva_phase_compare_run_v1", "run schema differs")
    expected_static = {
        "layers": list(LAYERS),
        "experts_per_layer": EXPERTS,
        "regions_per_layer": REGIONS_PER_LAYER,
        "bytes_per_layer": LAYER_BYTES,
        "total_bytes": TOTAL_BYTES,
        "p3_executor": "synchronous_fenced_cmd96",
        "bank_level_commit": True,
        "trace_mode": "compact",
        "epoch_validation": True,
        "enforce_eager": True,
        "chunked_prefill": False,
    }
    for key, wanted in expected_static.items():
        require(config.get(key) == wanted, f"run config {key} differs")
    order = config.get("order")
    require(isinstance(order, str) and len(order) % 3 == 0, "order is not blocked")
    blocks = [order[index : index + 3] for index in range(0, len(order), 3)]
    require(all(set(block) == set(ARM_NAMES) for block in blocks), "order block is not PCU permutation")
    batches = config.get("batch_sizes")
    require(
        isinstance(batches, list)
        and all(type(value) is int and value > 0 for value in batches)
        and len(batches) == len(set(batches))
        and {1, 4}.issubset(batches),
        "formal batch sizes must include 1 and 4",
    )
    prompt_lengths = fixture_contract(Path(config["fixture"]), batches)
    fixture_path = Path(config["fixture"]).resolve()
    model_path = Path(config["model"]).resolve()
    fixture_sha256 = sha256_file(fixture_path)
    conditioning_repeats = int(config.get("conditioning_repeats", -1))
    measurement_repeats = int(config.get("measurement_repeats", 0))
    warmup_decode_steps = int(config.get("warmup_decode_steps", -1))
    measured_decode_steps = int(config.get("measured_decode_steps", 0))
    require(conditioning_repeats >= 1, "at least one conditioning repeat is required")
    require(measurement_repeats >= 3, "at least three measurement repeats are required")
    require(warmup_decode_steps >= 1 and measured_decode_steps >= 1, "decode step contract differs")

    with (run_root / "execution_status.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(len(rows) == len(order), "execution row count differs")
    process_values: dict[str, dict[int, dict[str, list[float]]]] = {
        phase: {
            batch: defaultdict(list) for batch in batches
        }
        for phase in (
            "prefill_step_ms",
            "decode_step_ms",
            "cycle_e2e_ms",
            "request_ttft_p50_ms",
            "request_tpot_p50_ms",
            "request_latency_p50_ms",
        )
    }
    output_hashes: dict[int, set[str]] = defaultdict(set)
    cases: list[dict[str, Any]] = []
    cycles_per_case = len(batches) * (conditioning_repeats + measurement_repeats)
    forward_steps_per_cycle = 1 + warmup_decode_steps + measured_decode_steps
    p3_transitions = cycles_per_case * forward_steps_per_cycle * len(LAYERS)
    for ordinal, (symbol, row) in enumerate(zip(order, rows), 1):
        arm = ARM_NAMES[symbol]
        block = (ordinal - 1) // 3 + 1
        require(int(row.get("ordinal", -1)) == ordinal, f"row {ordinal}: ordinal")
        require(int(row.get("block", -1)) == block, f"row {ordinal}: block")
        require(row.get("symbol") == symbol and row.get("arm") == arm, f"row {ordinal}: arm")
        require(int(row.get("returncode", -1)) == 0, f"row {ordinal}: failed")
        expected_batch_order = batches if block % 2 == 1 else list(reversed(batches))
        row_batch_order = [int(item) for item in str(row.get("batch_order", "")).split(",") if item]
        require(row_batch_order == expected_batch_order, f"row {ordinal}: batch order")
        case_dir = run_root / str(row.get("directory"))
        require(case_dir.is_dir(), f"row {ordinal}: directory absent")
        env = old.load_env(case_dir / "environment.txt")
        for key, wanted in {
            "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ": "0",
            "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_BATCH": "0",
            "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_RESULT_FILE": "",
        }.items():
            require(env.get(key) == wanted, f"{case_dir}: inherited hybrid gate {key}")
        _result, summaries, hashes = verify_result(
            case_dir / "result.json",
            mechanism=MECHANISMS[arm],
            batch_order=expected_batch_order,
            prompt_lengths=prompt_lengths,
            conditioning_repeats=conditioning_repeats,
            measurement_repeats=measurement_repeats,
            warmup_decode_steps=warmup_decode_steps,
            measured_decode_steps=measured_decode_steps,
            max_model_len=int(config["max_model_len"]),
            max_num_seqs=int(config["max_num_seqs"]),
            max_num_batched_tokens=int(config["max_num_batched_tokens"]),
            expected_model=model_path,
            expected_fixture=fixture_path,
            expected_fixture_sha256=fixture_sha256,
            expected_cpu_offload_gb=3.0 if arm == "pinned" else 0.0,
        )
        if arm == "p3":
            evidence = verify_p3(case_dir, env, p3_transitions)
        elif arm == "pinned":
            evidence = old.verify_pinned(case_dir, env)
            exact_counter(stats_delta(case_dir), "slot_plan_execute_calls", 0, str(case_dir))
        else:
            evidence = old.verify_uva(case_dir, env)
            exact_counter(stats_delta(case_dir), "slot_plan_execute_calls", 0, str(case_dir))
        for batch, summary in summaries.items():
            for phase, value in summary.items():
                process_values[phase][batch][arm].append(value)
            output_hashes[batch].add(hashes[batch])
        cases.append(
            {
                "ordinal": ordinal,
                "block": block,
                "arm": arm,
                "directory": str(case_dir),
                "batch_order": expected_batch_order,
                "process_medians_ms": summaries,
                "evidence": evidence,
            }
        )

    require(all(len(values) == 1 for values in output_hashes.values()), "outputs differ across mechanisms")
    phase_metrics: dict[str, Any] = {}
    phase_comparisons: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for phase in (
        "prefill_step_ms",
        "decode_step_ms",
        "cycle_e2e_ms",
        "request_ttft_p50_ms",
        "request_tpot_p50_ms",
        "request_latency_p50_ms",
    ):
        phase_metrics[phase] = {}
        phase_comparisons[phase] = {}
        paired[phase] = {}
        for batch in batches:
            distributions = {
                arm: distribution(list(process_values[phase][batch][arm]))
                for arm in ARM_NAMES.values()
            }
            require(
                all(item["count"] == len(blocks) for item in distributions.values()),
                f"{phase} batch {batch}: process counts differ",
            )
            phase_metrics[phase][str(batch)] = distributions
            phase_comparisons[phase][str(batch)] = {
                name: comparison(
                    distributions[candidate]["median"],
                    distributions[baseline]["median"],
                )
                for name, candidate, baseline in COMPARISONS
            }
            paired[phase][str(batch)] = {}
            for name, candidate, baseline in COMPARISONS:
                pairs: list[dict[str, float | int]] = []
                for block in range(1, len(blocks) + 1):
                    values = {
                        case["arm"]: case["process_medians_ms"][batch][phase]
                        for case in cases
                        if case["block"] == block
                    }
                    pairs.append(
                        {
                            "block": block,
                            "delta_ms": values[candidate] - values[baseline],
                            "ratio": values[candidate] / values[baseline],
                        }
                    )
                paired[phase][str(batch)][name] = {
                    "pairs": pairs,
                    "delta_ms_median": statistics.median(
                        [float(item["delta_ms"]) for item in pairs]
                    ),
                    "ratio_median": statistics.median(
                        [float(item["ratio"]) for item in pairs]
                    ),
                }

    throughput = {
        str(batch): {
            arm: {
                "prefill_tokens_per_second": (
                    sum(prompt_lengths[batch])
                    * 1000.0
                    / phase_metrics["prefill_step_ms"][str(batch)][arm]["median"]
                ),
                "decode_tokens_per_second": (
                    batch
                    * 1000.0
                    / phase_metrics["decode_step_ms"][str(batch)][arm]["median"]
                ),
            }
            for arm in ARM_NAMES.values()
        }
        for batch in batches
    }
    return {
        "schema": "vllm_p3_pinned_uva_phase_compare_verification_v1",
        "status": "pass",
        "run_root": str(run_root),
        "order": order,
        "process_blocks": blocks,
        "sample_unit": "independent_process_median",
        "weight_contract": {
            "layers": list(LAYERS),
            "experts_per_layer": EXPERTS,
            "regions_per_layer": REGIONS_PER_LAYER,
            "bytes_per_layer": LAYER_BYTES,
            "total_bytes": TOTAL_BYTES,
            "total_gib": TOTAL_BYTES / 1024**3,
            "traffic_note": "P3/pinned copy all selected bytes; managed-UVA reads are demand-driven",
        },
        "measurement_contract": {
            "batch_sizes": batches,
            "prompt_tokens": {
                str(batch): sum(prompt_lengths[batch]) for batch in batches
            },
            "conditioning_repeats": conditioning_repeats,
            "measurement_repeats": measurement_repeats,
            "warmup_decode_steps": warmup_decode_steps,
            "measured_decode_steps": measured_decode_steps,
            "processes_per_arm": len(blocks),
            "prefill_metric": "synchronized strict complete prefill engine-step wall time",
            "decode_metric": "synchronized cached one-token-per-request engine-step wall time",
            "e2e_metric": "one prefill plus warmup and measured decode steps per request cycle",
            "standard_request_metrics": (
                "vLLM request TTFT, average TPOT, and request latency; reported "
                "separately from strict synchronized phase steps"
            ),
        },
        "phase_metrics_ms": phase_metrics,
        "throughput": throughput,
        "comparisons": phase_comparisons,
        "block_paired": paired,
        "output_token_ids_sha256": {
            str(batch): next(iter(values)) for batch, values in output_hashes.items()
        },
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.run_root)
    except (
        VerificationError,
        old.VerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        json.JSONDecodeError,
    ) as exc:
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
