#!/usr/bin/env python3
"""Validate and summarize paired dual-bank shadow-epoch experiments.

This module is deliberately GPU-free.  It consumes JSON artifacts emitted by
uvm_slot_dual_bank_event_probe and compares paired rounds after rejecting any
artifact which does not satisfy the safety and cleanup contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
REQUIRED_ROUNDS = 3
STEADY_INDICES = (2, 3, 4)
EXPECTED_EXECUTION_ORDER = (0, 1, 2, 3, 0)
EXPECTED_BANK_ORDER = (0, 1, 0, 1, 0)
EXPECTED_SHAPE = {
    "blocks": 288,
    "entries_per_bank": 1152,
    "bytes_per_bank": 603_979_776,
    "total_bank_bytes": 1_207_959_552,
}
REQUIRED_MODULE_PARAMETERS = {
    "dual_bank_event": 1,
    "copy_run_cache": 1,
    "pte_prewarm": 1,
    "legacy_overlap": 0,
    "fast_index": 1,
    "direct_cutover": 1,
}

# The names intentionally distinguish host submission from device completion.
# In this probe contract copy_complete/map_complete are zero and the joined
# device tail is represented by wait_tail_ns.
METRICS: Mapping[str, tuple[str, str]] = {
    "ioctl_wall_ns": ("execution", "ioctl_wall_ns"),
    "worker_wall_ns": ("execution", "wall_ns"),
    "kernel_total_ns": ("profile", "slot_direct_profile.total.ns"),
    "prepare_ns": ("profile", "slot_direct_profile.prepare.ns"),
    "old_clear_ns": (
        "profile",
        "slot_direct_profile.transition_clear_rebind.ns",
    ),
    "target_unmap_submit_ns": ("profile", "slot_direct_profile.unmap.ns"),
    "copy_submit_ns": ("profile", "slot_direct_profile.copy.ns"),
    "map_submit_ns": ("profile", "slot_direct_profile.map.ns"),
    "wait_tail_ns": ("profile", "slot_direct_profile.wait.ns"),
    "flatten_sort_ns": ("profile", "slot_direct_profile.flatten_sort.ns"),
    "slot_validate_ns": ("profile", "slot_direct_profile.slot_validate.ns"),
    "transition_preflight_ns": (
        "profile",
        "slot_direct_profile.transition_preflight.ns",
    ),
}

# These stages are exported by the shadow-PTE implementation. They are kept
# separate from METRICS so archived artifacts produced before that telemetry
# existed remain readable. A single arm may not mix old and new telemetry.
SHADOW_METRICS: Mapping[str, tuple[str, str]] = {
    "shadow_pte_prefill_ns": (
        "profile",
        "slot_direct_profile.shadow_pte_prefill.ns",
    ),
    "shadow_pte_publish_ns": (
        "profile",
        "slot_direct_profile.shadow_pte_publish.ns",
    ),
}
SHADOW_PREFILL_CALLS = "slot_direct_shadow_pte_prefill_calls"
SHADOW_PREFILL_BLOCKS = "slot_direct_shadow_pte_prefill_blocks"
SHADOW_PROFILE_STAGES = ("shadow_pte_prefill", "shadow_pte_publish")


class ValidationError(ValueError):
    """Raised when an input artifact cannot support a trustworthy comparison."""


def _fail(source: str, message: str) -> None:
    raise ValidationError(f"{source}: {message}")


def _require(condition: bool, source: str, message: str) -> None:
    if not condition:
        _fail(source, message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _nonnegative_number(value: Any, source: str, field: str) -> float | int:
    _require(_is_number(value), source, f"{field} must be numeric")
    _require(value >= 0, source, f"{field} must be non-negative")
    return value


def _nested_get(root: Mapping[str, Any], dotted_path: str, source: str) -> Any:
    current: Any = root
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            _fail(source, f"missing expected field {dotted_path!r}")
        current = current[component]
    return current


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_expectations(items: Iterable[str]) -> dict[str, Any]:
    """Parse repeated key=value arguments, using JSON for scalar values."""

    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValidationError(
                f"expectation {item!r} must use key=value syntax"
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValidationError(f"expectation {item!r} has an empty key")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value
        if key in parsed and parsed[key] != value:
            raise ValidationError(f"conflicting expectations for {key!r}")
        parsed[key] = value
    return parsed


def validate_artifact(
    artifact: Mapping[str, Any],
    source: str = "artifact",
    extra_expectations: Mapping[str, Any] | None = None,
) -> None:
    """Validate safety, workload, telemetry, and cleanup invariants."""

    _require(artifact.get("status") == "pass", source, "status is not pass")
    _require(
        artifact.get("compute_copy_overlap_allowed") is False,
        source,
        "compute/copy overlap must be disabled",
    )
    _require(
        artifact.get("compute_event_waited_before_each_transition") is True,
        source,
        "compute event was not waited before every transition",
    )
    _require(
        artifact.get("worker_returncode") == 0,
        source,
        "worker_returncode is not zero",
    )
    _require(
        artifact.get("gpu_memory_final_mib") == 0,
        source,
        "final GPU memory is not zero",
    )

    for field, expected in EXPECTED_SHAPE.items():
        _require(
            artifact.get(field) == expected,
            source,
            f"{field} must be {expected}, got {artifact.get(field)!r}",
        )

    _require(
        tuple(artifact.get("execution_order", ())) == EXPECTED_EXECUTION_ORDER,
        source,
        f"execution_order must be {list(EXPECTED_EXECUTION_ORDER)}",
    )
    _require(
        tuple(artifact.get("bank_order", ())) == EXPECTED_BANK_ORDER,
        source,
        f"bank_order must be {list(EXPECTED_BANK_ORDER)}",
    )

    srcversion = artifact.get("module_srcversion")
    _require(
        isinstance(srcversion, str) and bool(srcversion),
        source,
        "module_srcversion must be a non-empty string",
    )

    parameters = artifact.get("module_parameters")
    _require(
        isinstance(parameters, Mapping), source, "module_parameters is missing"
    )
    for key, expected in REQUIRED_MODULE_PARAMETERS.items():
        _require(
            parameters.get(key) == expected,
            source,
            f"module_parameters.{key} must be {expected}, "
            f"got {parameters.get(key)!r}",
        )
    budget = parameters.get("vidmem_budget_bytes")
    _require(
        _is_number(budget) and budget >= EXPECTED_SHAPE["total_bank_bytes"],
        source,
        "module_parameters.vidmem_budget_bytes is below the two-bank backing",
    )

    feature_keys = ("fast_validation", "shadow_pte_prefill")
    feature_instrumented = any(key in parameters for key in feature_keys)
    if feature_instrumented:
        for key in feature_keys:
            _require(
                parameters.get(key) in (0, 1),
                source,
                f"module_parameters.{key} must be 0 or 1",
            )

    for key, expected in (extra_expectations or {}).items():
        dotted_path = key if "." in key else f"module_parameters.{key}"
        actual = _nested_get(artifact, dotted_path, source)
        _require(
            actual == expected,
            source,
            f"{dotted_path} must be {expected!r}, got {actual!r}",
        )

    executions = artifact.get("executions")
    profiles = artifact.get("execute_profile_deltas")
    barriers = artifact.get("event_barriers")
    expected_count = len(EXPECTED_EXECUTION_ORDER)
    _require(
        isinstance(executions, list) and len(executions) == expected_count,
        source,
        f"executions must contain {expected_count} items",
    )
    _require(
        isinstance(profiles, list) and len(profiles) == expected_count,
        source,
        f"execute_profile_deltas must contain {expected_count} items",
    )
    _require(
        isinstance(barriers, list) and len(barriers) == expected_count,
        source,
        f"event_barriers must contain {expected_count} items",
    )

    for index, (execution, profile, barrier) in enumerate(
        zip(executions, profiles, barriers)
    ):
        item_source = f"{source}: execution[{index}]"
        _require(isinstance(execution, Mapping), item_source, "not an object")
        _require(isinstance(profile, Mapping), item_source, "profile is not an object")
        _require(isinstance(barrier, Mapping), item_source, "barrier is not an object")
        _require(execution.get("ok") is True, item_source, "execution failed")
        _require(execution.get("ioctl_ret") == 0, item_source, "ioctl_ret is not zero")
        _require(execution.get("errno") == 0, item_source, "errno is not zero")
        _require(execution.get("reason") == "ok", item_source, "reason is not ok")
        _require(
            execution.get("layer_id") == EXPECTED_EXECUTION_ORDER[index],
            item_source,
            "layer_id does not match execution_order",
        )
        _require(
            execution.get("bank_id") == EXPECTED_BANK_ORDER[index],
            item_source,
            "bank_id does not match bank_order",
        )
        _require(
            execution.get("compute_event_completed_before_execute") is True,
            item_source,
            "kernel-visible compute event was incomplete",
        )
        _require(
            execution.get("success_count") == EXPECTED_SHAPE["entries_per_bank"],
            item_source,
            "success_count does not cover the complete bank",
        )
        _require(
            execution.get("failure_count") == 0,
            item_source,
            "failure_count is not zero",
        )
        _require(
            execution.get("accepted_bytes") == EXPECTED_SHAPE["bytes_per_bank"],
            item_source,
            "accepted_bytes does not cover the complete bank",
        )

        _nonnegative_number(
            execution.get("ioctl_wall_ns"), item_source, "ioctl_wall_ns"
        )
        _nonnegative_number(execution.get("wall_ns"), item_source, "wall_ns")
        for metric_name, (location, field) in METRICS.items():
            if location != "profile":
                continue
            _nonnegative_number(profile.get(field), item_source, field)

        if feature_instrumented:
            for metric_name, (location, field) in SHADOW_METRICS.items():
                _require(location == "profile", item_source, "invalid metric schema")
                _nonnegative_number(profile.get(field), item_source, field)
            prefill_calls = _nonnegative_number(
                profile.get(SHADOW_PREFILL_CALLS),
                item_source,
                SHADOW_PREFILL_CALLS,
            )
            prefill_blocks = _nonnegative_number(
                profile.get(SHADOW_PREFILL_BLOCKS),
                item_source,
                SHADOW_PREFILL_BLOCKS,
            )
            stage_values: dict[str, tuple[float | int, float | int]] = {}
            for stage in SHADOW_PROFILE_STAGES:
                stage_calls_key = f"slot_direct_profile.{stage}.calls"
                stage_ns_key = f"slot_direct_profile.{stage}.ns"
                stage_values[stage] = (
                    _nonnegative_number(
                        profile.get(stage_calls_key), item_source, stage_calls_key
                    ),
                    _nonnegative_number(
                        profile.get(stage_ns_key), item_source, stage_ns_key
                    ),
                )

            shadow_expected = parameters["shadow_pte_prefill"] == 1 and index > 0
            if shadow_expected:
                _require(
                    prefill_calls == 1,
                    item_source,
                    "shadow-PTE prefill did not hit exactly once",
                )
                _require(
                    prefill_blocks == EXPECTED_SHAPE["blocks"],
                    item_source,
                    "shadow-PTE prefill did not cover every target block",
                )
                for stage, (calls, ns) in stage_values.items():
                    _require(
                        calls == 1,
                        item_source,
                        f"{stage} did not execute exactly once",
                    )
                    _require(ns > 0, item_source, f"{stage} timing is not positive")
            else:
                _require(
                    prefill_calls == 0 and prefill_blocks == 0,
                    item_source,
                    "unexpected shadow-PTE prefill telemetry",
                )
                for stage, (calls, ns) in stage_values.items():
                    _require(
                        calls == 0 and ns == 0,
                        item_source,
                        f"unexpected {stage} telemetry",
                    )

        for completion_name in ("copy_complete", "map_complete"):
            ns_key = f"slot_direct_profile.{completion_name}.ns"
            calls_key = f"slot_direct_profile.{completion_name}.calls"
            _require(
                profile.get(ns_key) == 0 and profile.get(calls_key) == 0,
                item_source,
                f"{completion_name} must remain zero; copy/map are submission "
                "timings and wait is the joined device tail",
            )
        _require(
            profile.get("slot_direct_copy_run_cache_hits") == 1,
            item_source,
            "registered copy-run cache did not hit exactly once",
        )

        _require(
            barrier.get("sequence_index") == index,
            item_source,
            "barrier sequence_index is inconsistent",
        )
        _require(
            barrier.get("layer_id") == EXPECTED_EXECUTION_ORDER[index],
            item_source,
            "barrier layer_id is inconsistent",
        )
        _require(
            barrier.get("bank_id") == EXPECTED_BANK_ORDER[index],
            item_source,
            "barrier bank_id is inconsistent",
        )
        _require(
            barrier.get("checksum") == barrier.get("expected"),
            item_source,
            "checksum does not match expected data",
        )
        completed_ns = _nonnegative_number(
            barrier.get("completed_ns"), item_source, "completed_ns"
        )
        execute_end_ns = _nonnegative_number(
            barrier.get("execute_end_ns"), item_source, "execute_end_ns"
        )
        _require(
            completed_ns >= execute_end_ns,
            item_source,
            "compute completion precedes transition completion",
        )

    generation_start = executions[0].get("generation")
    _require(
        isinstance(generation_start, int)
        and not isinstance(generation_start, bool)
        and generation_start > 0,
        source,
        "initial execute generation must be positive",
    )
    previous_new_generation: int | None = None
    for index, execution in enumerate(executions):
        generation = execution.get("generation")
        new_generation = execution.get("new_generation")
        generation_source = f"{source}: execution[{index}] generation"
        _require(
            isinstance(generation, int) and not isinstance(generation, bool),
            generation_source,
            "input generation must be an integer",
        )
        _require(
            isinstance(new_generation, int) and not isinstance(new_generation, bool),
            generation_source,
            "new generation must be an integer",
        )
        _require(
            new_generation == generation + 1,
            generation_source,
            "successful transition did not advance generation exactly once",
        )
        if previous_new_generation is not None:
            _require(
                generation == previous_new_generation,
                generation_source,
                "generation chain is not contiguous",
            )
        previous_new_generation = new_generation

    registrations = artifact.get("registrations")
    _require(
        isinstance(registrations, list) and len(registrations) == expected_count,
        source,
        f"registrations must contain {expected_count} items",
    )
    registered_plan_ids: set[int] = set()
    for index, registration in enumerate(registrations):
        registration_source = f"{source}: registrations[{index}]"
        _require(isinstance(registration, Mapping), registration_source, "not an object")
        _require(registration.get("ok") is True, registration_source, "registration failed")
        _require(
            registration.get("generation") == generation_start,
            registration_source,
            "registration generation does not match initial execution",
        )
        plan_id = registration.get("plan_id")
        _require(
            isinstance(plan_id, int)
            and not isinstance(plan_id, bool)
            and plan_id > 0,
            registration_source,
            "plan_id must be positive",
        )
        _require(
            plan_id not in registered_plan_ids,
            registration_source,
            "duplicate plan_id",
        )
        registered_plan_ids.add(plan_id)
    for index, execution in enumerate(executions):
        _require(
            execution.get("plan_id") in registered_plan_ids,
            f"{source}: execution[{index}]",
            "execution used an unregistered plan",
        )

    destroy_results = artifact.get("destroy_results")
    _require(
        isinstance(destroy_results, list) and len(destroy_results) == expected_count,
        source,
        f"destroy_results must contain {expected_count} items",
    )
    for index, result in enumerate(destroy_results):
        destroy_source = f"{source}: destroy_results[{index}]"
        _require(isinstance(result, Mapping), destroy_source, "not an object")
        _require(result.get("ok") is True, destroy_source, "destroy failed")
        _require(result.get("destroyed") is True, destroy_source, "plan survived")
        _require(result.get("ioctl_ret") == 0, destroy_source, "ioctl_ret is not zero")
        _require(result.get("errno") == 0, destroy_source, "errno is not zero")
        _require(
            result.get("plan_id") in registered_plan_ids,
            destroy_source,
            "destroy used an unregistered plan",
        )
        _require(
            result.get("generation") == previous_new_generation,
            destroy_source,
            "destroy generation does not match the committed generation",
        )
        expected_current_generation = (
            0 if index == len(destroy_results) - 1 else previous_new_generation
        )
        _require(
            result.get("current_generation") == expected_current_generation,
            destroy_source,
            "destroy current_generation does not prove final session teardown",
        )
    _require(
        {result.get("plan_id") for result in destroy_results} == registered_plan_ids,
        source,
        "destroy_results do not cover every registered plan exactly once",
    )

    close = artifact.get("close")
    _require(isinstance(close, Mapping), source, "close result is missing")
    _require(close.get("ok") is True, source, "session close failed")
    _require(close.get("reason") == "closed", source, "session was not closed")

    pre_close = artifact.get("pre_close_state")
    post_close = artifact.get("post_close_state")
    _require(isinstance(pre_close, Mapping), source, "pre_close_state is missing")
    _require(isinstance(post_close, Mapping), source, "post_close_state is missing")
    _require(
        pre_close.get("allocated_slots") == 2 * EXPECTED_SHAPE["entries_per_bank"],
        source,
        "pre-close slot count does not contain two banks",
    )
    _require(pre_close.get("plan_count") == 0, source, "plans remain before close")
    _require(
        pre_close.get("session_open") is True,
        source,
        "session unexpectedly closed before close check",
    )
    _require(post_close.get("allocated_slots") == 0, source, "slots leaked after close")
    _require(post_close.get("plan_count") == 0, source, "plans leaked after close")
    _require(
        post_close.get("session_open") is False,
        source,
        "session remains open after close",
    )


def _feature_instrumented(artifact: Mapping[str, Any]) -> bool:
    parameters = artifact["module_parameters"]
    return "fast_validation" in parameters or "shadow_pte_prefill" in parameters


def _available_metrics(artifact: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    metrics = dict(METRICS)
    if _feature_instrumented(artifact):
        metrics.update(SHADOW_METRICS)
    return metrics


def _metric_samples(
    artifact: Mapping[str, Any],
    metric: str,
    available_metrics: Mapping[str, tuple[str, str]],
) -> list[float | int]:
    location, field = available_metrics[metric]
    executions = artifact["executions"]
    profiles = artifact["execute_profile_deltas"]
    source = executions if location == "execution" else profiles
    return [source[index][field] for index in STEADY_INDICES]


def _load_round(
    path: Path, extra_expectations: Mapping[str, Any]
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            artifact = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValidationError(f"{path}: cannot read artifact: {error}") from error
    _require(isinstance(artifact, Mapping), str(path), "top level must be an object")
    validate_artifact(artifact, str(path), extra_expectations)
    available_metrics = _available_metrics(artifact)
    samples = {
        metric: _metric_samples(artifact, metric, available_metrics)
        for metric in available_metrics
    }
    parameters = artifact["module_parameters"]
    feature_instrumented = _feature_instrumented(artifact)
    shadow_enabled = feature_instrumented and parameters["shadow_pte_prefill"] == 1
    round_summary = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "module_srcversion": artifact["module_srcversion"],
        "steady_indices": list(STEADY_INDICES),
        "steady_samples_ns": samples,
        "steady_median_ns": {
            metric: statistics.median(values) for metric, values in samples.items()
        },
        "feature_evidence": {
            "telemetry_instrumented": feature_instrumented,
            "fast_validation_enabled": (
                parameters["fast_validation"] == 1 if feature_instrumented else None
            ),
            "fast_validation_successful_executions": (
                len(artifact["executions"])
                if feature_instrumented and parameters["fast_validation"] == 1
                else 0
            ),
            "shadow_pte_prefill_enabled": (
                shadow_enabled if feature_instrumented else None
            ),
            "shadow_prefill_verified_steady_executions": (
                len(STEADY_INDICES) if shadow_enabled else 0
            ),
            "shadow_prefill_blocks_per_steady_execution": (
                EXPECTED_SHAPE["blocks"] if shadow_enabled else 0
            ),
        },
    }
    return artifact, round_summary


def summarize_arm(
    label: str,
    paths: Sequence[Path],
    extra_expectations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(bool(label), f"arm {label!r}", "label must be non-empty")
    _require(
        len(paths) == REQUIRED_ROUNDS,
        f"arm {label}",
        f"exactly {REQUIRED_ROUNDS} rounds are required, got {len(paths)}",
    )
    summaries: list[dict[str, Any]] = []
    for path in paths:
        _, summary = _load_round(path, extra_expectations or {})
        summaries.append(summary)
    srcversions = {item["module_srcversion"] for item in summaries}
    _require(
        len(srcversions) == 1,
        f"arm {label}",
        f"module_srcversion changed within arm: {sorted(srcversions)}",
    )
    metric_sets = {tuple(item["steady_median_ns"]) for item in summaries}
    _require(
        len(metric_sets) == 1,
        f"arm {label}",
        "rounds mix legacy and shadow telemetry schemas",
    )
    metric_names = list(next(iter(metric_sets)))
    feature_configs = {
        (
            item["feature_evidence"]["telemetry_instrumented"],
            item["feature_evidence"]["fast_validation_enabled"],
            item["feature_evidence"]["shadow_pte_prefill_enabled"],
        )
        for item in summaries
    }
    _require(
        len(feature_configs) == 1,
        f"arm {label}",
        "fast-validation/shadow configuration changed within arm",
    )
    feature_config = next(iter(feature_configs))

    pooled: dict[str, list[float | int]] = {}
    for metric in metric_names:
        pooled[metric] = [
            value
            for summary in summaries
            for value in summary["steady_samples_ns"][metric]
        ]
    return {
        "label": label,
        "module_srcversion": next(iter(srcversions)),
        "round_count": len(summaries),
        "steady_sample_count": len(summaries) * len(STEADY_INDICES),
        "metrics": metric_names,
        "feature_configuration": {
            "telemetry_instrumented": feature_config[0],
            "fast_validation_enabled": feature_config[1],
            "shadow_pte_prefill_enabled": feature_config[2],
        },
        "rounds": summaries,
        "median_of_round_medians_ns": {
            metric: statistics.median(
                [item["steady_median_ns"][metric] for item in summaries]
            )
            for metric in metric_names
        },
        "pooled_steady_median_ns": {
            metric: statistics.median(values) for metric, values in pooled.items()
        },
    }


def _median_or_none(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def compare_arms(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        reference["round_count"] == candidate["round_count"],
        "comparison",
        "reference and candidate round counts differ",
    )
    _require(
        reference["metrics"] == candidate["metrics"],
        "comparison",
        "reference and candidate telemetry schemas differ",
    )
    metric_results: dict[str, Any] = {}
    for metric in reference["metrics"]:
        pairs: list[dict[str, Any]] = []
        ratios: list[float] = []
        wins = ties = losses = 0
        for round_index, (reference_round, candidate_round) in enumerate(
            zip(reference["rounds"], candidate["rounds"]), start=1
        ):
            reference_value = reference_round["steady_median_ns"][metric]
            candidate_value = candidate_round["steady_median_ns"][metric]
            delta = reference_value - candidate_value
            speedup = reference_value / candidate_value if candidate_value else None
            if speedup is not None:
                ratios.append(speedup)
            if delta > 0:
                winner = "candidate"
                wins += 1
            elif delta < 0:
                winner = "reference"
                losses += 1
            else:
                winner = "tie"
                ties += 1
            pairs.append(
                {
                    "round": round_index,
                    "reference_ns": reference_value,
                    "candidate_ns": candidate_value,
                    "delta_reference_minus_candidate_ns": delta,
                    "speedup_reference_over_candidate": speedup,
                    "winner": winner,
                }
            )
        metric_results[metric] = {
            "paired_rounds": pairs,
            "paired_delta_median_ns": statistics.median(
                [pair["delta_reference_minus_candidate_ns"] for pair in pairs]
            ),
            "paired_speedup_median": _median_or_none(ratios),
            "candidate_wins": wins,
            "ties": ties,
            "candidate_losses": losses,
            "reference_pooled_steady_median_ns": reference[
                "pooled_steady_median_ns"
            ][metric],
            "candidate_pooled_steady_median_ns": candidate[
                "pooled_steady_median_ns"
            ][metric],
        }
    return {
        "reference_label": reference["label"],
        "candidate_label": candidate["label"],
        "delta_semantics": "reference - candidate; positive means candidate is faster",
        "speedup_semantics": "reference / candidate; greater than 1 means candidate is faster",
        "metrics": metric_results,
    }


def build_summary(
    arm_a_label: str,
    arm_a_paths: Sequence[Path],
    arm_b_label: str,
    arm_b_paths: Sequence[Path],
    expect_a: Mapping[str, Any] | None = None,
    expect_b: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(
        arm_a_label != arm_b_label,
        "comparison",
        "arm labels must be distinct",
    )
    _require(
        len(arm_a_paths) == len(arm_b_paths),
        "comparison",
        "reference and candidate round counts differ",
    )
    reference = summarize_arm(arm_a_label, arm_a_paths, expect_a)
    candidate = summarize_arm(arm_b_label, arm_b_paths, expect_b)
    return {
        "schema_version": SCHEMA_VERSION,
        "workload_contract": {
            **EXPECTED_SHAPE,
            "execution_order": list(EXPECTED_EXECUTION_ORDER),
            "bank_order": list(EXPECTED_BANK_ORDER),
            "steady_indices": list(STEADY_INDICES),
            "excluded_indices": [0, 1],
            "excluded_reasons": ["bootstrap", "first fill of empty Bank B"],
            "compute_copy_overlap_allowed": False,
        },
        "telemetry_contract": {
            "copy_submit_ns": "host-side CE copy submission, not device completion",
            "map_submit_ns": "host-side PTE map submission, not device completion",
            "wait_tail_ns": "joined device completion tail for copy/map/tracker work",
            "copy_complete_and_map_complete_must_be_zero": True,
            "shadow_pte_prefill_ns": (
                "host time to submit unreachable leaf-PTE prefill; positive "
                "timing plus full-block hit counters are required when enabled"
            ),
            "shadow_pte_publish_ns": (
                "host time to publish the parent PDE after copy dependencies"
            ),
        },
        "statistics_scope": (
            "Three paired rounds; medians are descriptive only and do not imply "
            "statistical significance. Per-round steady medians are the primary "
            "comparison; pooled medians are secondary diagnostics."
        ),
        "arms": {"reference": reference, "candidate": candidate},
        "comparison": compare_arms(reference, candidate),
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and summarize three paired 576 MiB dual-bank rounds. "
            "Arm A is the reference and arm B is the candidate."
        )
    )
    parser.add_argument("--arm-a-label", required=True)
    parser.add_argument("--arm-a", type=Path, nargs="+", required=True)
    parser.add_argument("--arm-b-label", required=True)
    parser.add_argument("--arm-b", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--expect-a",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "additional arm-A expectation; an undotted key addresses "
            "module_parameters, while a dotted key addresses the JSON root"
        ),
    )
    parser.add_argument(
        "--expect-b",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="additional arm-B expectation (repeatable)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON here instead of stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        summary = build_summary(
            args.arm_a_label,
            args.arm_a,
            args.arm_b_label,
            args.arm_b,
            parse_expectations(args.expect_a),
            parse_expectations(args.expect_b),
        )
        rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except ValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
