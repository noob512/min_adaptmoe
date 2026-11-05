#!/usr/bin/env python3
"""Strict verifier for a cold-process generic-restore/slot-restore pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping


RUN_SCHEMA = "vllm_slot_direct_integration_run_v1"
SUMMARY_SCHEMA = "vllm_slot_direct_integration_summary_v1"
MODE_PREALLOCATE = 8
MODE_RESTORE = 10
MODE_RELEASE = 19
GIB = 1024**3

_SECTION_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*):\s*$")
_PROFILE_RE = re.compile(
    r"^\s+([a-z][a-z0-9_]*)\s+"
    r"calls=(\d+)\s+ns=(\d+)\s+avg_ns=(\d+)\s+"
    r"pages=(\d+)\s+bytes=(\d+)\s*$"
)
_COUNTER_RE = re.compile(
    r"^\s*([a-z][a-z0-9_]*)\s+(0x[0-9a-fA-F]+|\d+)\s*$"
)


class VerificationError(RuntimeError):
    """An artifact failed a required correctness or accounting gate."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path}: expected a JSON object")
    return value


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


def finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label}: expected a number",
    )
    result = float(value)
    require(math.isfinite(result), f"{label}: value is not finite")
    if positive:
        require(result > 0.0, f"{label}: value must be positive")
    else:
        require(result >= 0.0, f"{label}: value must be non-negative")
    return result


def parse_latency_stats(text: str, label: str) -> dict[str, int]:
    values: dict[str, int] = {}
    section: str | None = None
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        section_match = _SECTION_RE.fullmatch(line)
        if section_match:
            section = section_match.group(1)
            continue
        profile_match = _PROFILE_RE.fullmatch(line)
        if profile_match:
            require(
                section is not None,
                f"{label}:{line_number}: profile row precedes a section",
            )
            name, calls, nanoseconds, _average, pages, bytes_ = (
                profile_match.groups()
            )
            prefix = f"{section}.{name}"
            require(
                f"{prefix}.calls" not in values,
                f"{label}:{line_number}: duplicate profile row {prefix}",
            )
            values[f"{prefix}.calls"] = int(calls)
            values[f"{prefix}.ns"] = int(nanoseconds)
            values[f"{prefix}.pages"] = int(pages)
            values[f"{prefix}.bytes"] = int(bytes_)
            continue
        counter_match = _COUNTER_RE.fullmatch(line)
        if counter_match:
            name, raw_value = counter_match.groups()
            require(
                name not in values,
                f"{label}:{line_number}: duplicate counter {name}",
            )
            values[name] = int(raw_value, 0)
            section = None
            continue
        raise VerificationError(
            f"{label}:{line_number}: unrecognized latency_stats row {line!r}"
        )
    require(values, f"{label}: latency_stats is empty")
    return values


def read_latency_stats(path: Path) -> dict[str, int]:
    require(path.is_file(), f"missing latency_stats snapshot: {path}")
    return parse_latency_stats(path.read_text(encoding="utf-8"), str(path))


def profile_delta(
    before: Mapping[str, int], after: Mapping[str, int], label: str
) -> dict[str, int]:
    result: dict[str, int] = {}
    for key in sorted(set(before) | set(after)):
        value = int(after.get(key, 0)) - int(before.get(key, 0))
        require(value >= 0, f"{label}: counter moved backwards: {key}={value}")
        result[key] = value
    return result


def parse_environment(path: Path) -> dict[str, str]:
    require(path.is_file(), f"missing environment artifact: {path}")
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        require("=" in line, f"{path}:{line_number}: malformed environment row")
        key, value = line.split("=", 1)
        require(key and key not in result, f"{path}:{line_number}: duplicate key {key!r}")
        result[key] = value
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL artifact: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerificationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        require(isinstance(value, dict), f"{path}:{line_number}: expected object")
        records.append(value)
    return records


def _expected_qwen_plan(plan: dict[str, Any]) -> tuple[list[list[int]], list[list[int]]]:
    cold = plan.get("offload_plan")
    hot = plan.get("hot_plan", plan.get("prefetch_plan"))
    require(isinstance(cold, list) and isinstance(hot, list), "plan lacks cold/hot lists")

    def coordinates(records: list[Any], label: str) -> list[list[int]]:
        parsed: list[list[int]] = []
        for index, record in enumerate(records):
            require(isinstance(record, dict), f"{label}[{index}] is not an object")
            layer = record.get("layer_id")
            expert = record.get("expert_id")
            require(
                isinstance(layer, int)
                and not isinstance(layer, bool)
                and 0 <= layer < 48,
                f"{label}[{index}] has invalid layer_id",
            )
            require(
                isinstance(expert, int)
                and not isinstance(expert, bool)
                and 0 <= expert < 128,
                f"{label}[{index}] has invalid expert_id",
            )
            parsed.append([layer, expert])
        require(len(parsed) == len({tuple(item) for item in parsed}), f"{label} has duplicates")
        return sorted(parsed)

    cold_coordinates = coordinates(cold, "offload_plan")
    hot_coordinates = coordinates(hot, "hot_plan")
    cold_set = {tuple(item) for item in cold_coordinates}
    hot_set = {tuple(item) for item in hot_coordinates}
    require(not cold_set & hot_set, "plan cold and hot coordinates overlap")
    expected = {(layer, expert) for layer in range(48) for expert in range(128)}
    require(cold_set | hot_set == expected, "plan does not cover Qwen 48x128 topology")
    require(cold_coordinates, "plan has no cold experts")
    return cold_coordinates, hot_coordinates


def verify_benchmark_result(
    path: Path,
    *,
    arm: str,
    expected_concurrency: int,
    expected_requests: int,
    expected_output_tokens: int,
    fixture_sha256: str,
    sequential: bool,
) -> tuple[dict[str, Any], dict[str, float]]:
    result = load_json(path)
    prefix = str(path)
    require(result.get("status") == "ok", f"{prefix}: benchmark status is not ok")
    expected_profile = "generic_restore" if arm == "generic" else "slot_restore"
    require(result.get("profile_label") == expected_profile, f"{prefix}: profile label mismatch")
    if sequential:
        expected_mechanism = (
            "vllm_generic_uvm_restore_sequential"
            if arm == "generic"
            else "vllm_bounded_slot_cache_mode8_mode10_mode19"
        )
    else:
        expected_mechanism = (
            "vllm_generic_uvm_restore"
            if arm == "generic"
            else "vllm_slot_direct_mode8_mode10_restore"
        )
    require(
        result.get("mechanism_class") == expected_mechanism,
        f"{prefix}: mechanism class mismatch",
    )
    config = result.get("config")
    require(isinstance(config, dict), f"{prefix}: config is missing")
    require(config.get("concurrency") == expected_concurrency, f"{prefix}: concurrency mismatch")
    require(config.get("output_tokens") == expected_output_tokens, f"{prefix}: output token config mismatch")
    require(config.get("warmup_requests") == 0, f"{prefix}: run was not cold/no-warmup")
    if sequential:
        require(
            result.get("benchmark_schema") == "vllm_slot_cache_sequential_v1",
            f"{prefix}: sequential benchmark schema mismatch",
        )
        require(
            config.get("sequential_requests") == expected_requests,
            f"{prefix}: sequential request count mismatch",
        )
        execution = result.get("execution")
        require(isinstance(execution, dict), f"{prefix}: execution proof is missing")
        require(execution.get("single_llm_instance") is True, f"{prefix}: LLM instance was not retained")
        require(execution.get("llm_load_count") == 1, f"{prefix}: LLM was loaded more than once")
        require(execution.get("generate_call_count") == expected_requests, f"{prefix}: generate call count mismatch")
        require(execution.get("requests_per_generate") == 1, f"{prefix}: requests were not sequential")
        require(execution.get("engine_shutdown_explicit") is True, f"{prefix}: EngineCore shutdown was not explicit")
        shutdown_started = finite_number(
            execution.get("shutdown_started_epoch"),
            f"{prefix}: shutdown start",
            positive=True,
        )
        shutdown_ended = finite_number(
            execution.get("shutdown_end_epoch"),
            f"{prefix}: shutdown end",
            positive=True,
        )
        require(shutdown_ended >= shutdown_started, f"{prefix}: shutdown interval is invalid")
        measurement_ended = finite_number(
            result.get("measurement_end_epoch"),
            f"{prefix}: measurement end",
            positive=True,
        )
        require(
            shutdown_started >= measurement_ended,
            f"{prefix}: EngineCore shutdown overlapped the measured requests",
        )
        calls = execution.get("calls")
        require(isinstance(calls, list) and len(calls) == expected_requests, f"{prefix}: generate call records incomplete")
        expected_indices = list(range(expected_requests))
        require([call.get("sequence_index") for call in calls] == expected_indices, f"{prefix}: generate call sequence is not contiguous")
        process_pid = execution.get("process_pid")
        instance_id = execution.get("llm_instance_id")
        require(isinstance(process_pid, int) and process_pid > 0, f"{prefix}: invalid process PID")
        require(isinstance(instance_id, int) and instance_id > 0, f"{prefix}: invalid LLM instance id")
        require(all(call.get("process_pid") == process_pid for call in calls), f"{prefix}: generate calls crossed processes")
        require(all(call.get("llm_instance_id") == instance_id for call in calls), f"{prefix}: generate calls crossed LLM instances")
    fixtures = result.get("fixtures")
    require(isinstance(fixtures, dict), f"{prefix}: fixture provenance is missing")
    require(fixtures.get("evaluation_sha256") == fixture_sha256, f"{prefix}: fixture SHA mismatch")
    environment = result.get("environment")
    require(isinstance(environment, dict), f"{prefix}: captured environment is missing")
    expected_access = "restore" if arm == "generic" else "slot_restore"
    require(
        environment.get("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE") == expected_access,
        f"{prefix}: access mode mismatch",
    )

    measurement = result.get("measurement")
    require(isinstance(measurement, dict), f"{prefix}: measurement is missing")
    require(
        measurement.get("completed_requests") == expected_requests,
        f"{prefix}: completed request count mismatch",
    )
    require(
        measurement.get("output_tokens") == expected_requests * expected_output_tokens,
        f"{prefix}: aggregate output token count mismatch",
    )
    requests = measurement.get("requests")
    require(isinstance(requests, list) and len(requests) == expected_requests, f"{prefix}: request records incomplete")
    for index, record in enumerate(requests):
        require(isinstance(record, dict), f"{prefix}: request {index} is not an object")
        require(record.get("prompt_tokens") == record.get("expected_prompt_tokens"), f"{prefix}: request {index} prompt count mismatch")
        require(record.get("prompt_token_ids_exact") is True, f"{prefix}: request {index} prompt IDs are not exact")
        require(
            isinstance(record.get("expected_prompt_token_ids_sha256"), str)
            and record.get("prompt_token_ids_sha256")
            == record.get("expected_prompt_token_ids_sha256"),
            f"{prefix}: request {index} prompt token hash mismatch",
        )
        require(record.get("output_tokens") == expected_output_tokens, f"{prefix}: request {index} output count mismatch")
        require(isinstance(record.get("output_token_ids"), list), f"{prefix}: request {index} lacks output token IDs")
        require(
            record.get("output_token_ids_sha256")
            == sha256_json(record["output_token_ids"]),
            f"{prefix}: request {index} output token hash is invalid",
        )
        finite_number(record.get("ttft_ms"), f"{prefix}: request {index} TTFT")
        finite_number(record.get("request_latency_ms"), f"{prefix}: request {index} E2E", positive=True)
        if expected_output_tokens > 1:
            finite_number(record.get("tpot_ms"), f"{prefix}: request {index} TPOT")
        if sequential:
            execution = result["execution"]
            require(record.get("sequence_index") == index, f"{prefix}: request {index} sequence index mismatch")
            require(record.get("process_pid") == execution["process_pid"], f"{prefix}: request {index} process mismatch")
            require(record.get("llm_instance_id") == execution["llm_instance_id"], f"{prefix}: request {index} LLM instance mismatch")

    elapsed = finite_number(measurement.get("elapsed_seconds"), f"{prefix}: elapsed", positive=True)
    tokens_per_second = finite_number(
        measurement.get("output_tokens_per_second"),
        f"{prefix}: output tokens/s",
        positive=True,
    )
    require(
        math.isclose(
            tokens_per_second,
            expected_requests * expected_output_tokens / elapsed,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ),
        f"{prefix}: output tokens/s does not close against tokens and elapsed",
    )
    require(
        measurement.get("aggregate_output_token_ids_sha256")
        == sha256_json([record["output_token_ids"] for record in requests]),
        f"{prefix}: aggregate output token hash is invalid",
    )
    ttft = measurement.get("ttft_ms")
    e2e = measurement.get("request_latency_ms")
    require(isinstance(ttft, dict) and ttft.get("count") == expected_requests, f"{prefix}: TTFT summary incomplete")
    require(isinstance(e2e, dict) and e2e.get("count") == expected_requests, f"{prefix}: E2E summary incomplete")
    ttft_p50 = finite_number(ttft.get("p50"), f"{prefix}: TTFT p50")
    e2e_p50 = finite_number(e2e.get("p50"), f"{prefix}: E2E p50", positive=True)
    return result, {
        "ttft_ms": ttft_p50,
        "e2e_ms": e2e_p50,
        "measurement_wall_ms": elapsed * 1000.0,
        "output_tokens_per_second": tokens_per_second,
    }


def compare_tokens(generic: dict[str, Any], slot: dict[str, Any]) -> None:
    generic_measurement = generic["measurement"]
    slot_measurement = slot["measurement"]
    require(
        generic_measurement.get("aggregate_output_token_ids_sha256")
        == slot_measurement.get("aggregate_output_token_ids_sha256"),
        "generic/slot aggregate output token hashes differ",
    )
    generic_requests = generic_measurement["requests"]
    slot_requests = slot_measurement["requests"]
    require(len(generic_requests) == len(slot_requests), "generic/slot request counts differ")
    prompt_fields = (
        "fixture_request",
        "expected_prompt_tokens",
        "expected_prompt_token_ids_sha256",
        "prompt_tokens",
        "prompt_token_ids_sha256",
    )
    output_fields = (
        "output_tokens",
        "output_token_ids",
        "output_token_ids_sha256",
        "decoded_text",
        "finish_reason",
    )
    for index, (left, right) in enumerate(zip(generic_requests, slot_requests)):
        for field in prompt_fields + output_fields:
            require(
                left.get(field) == right.get(field),
                f"generic/slot request {index} differs in {field}",
            )


def verify_sequential_requests(result: dict[str, Any], label: str) -> None:
    requests = result["measurement"]["requests"]
    require(len(requests) >= 2, f"{label}: fewer than two sequential requests")
    require(
        [record.get("sequence_index") for record in requests]
        == list(range(len(requests))),
        f"{label}: request sequence is not contiguous",
    )
    first_prompt = (
        requests[0].get("prompt_tokens"),
        requests[0].get("prompt_token_ids_sha256"),
    )
    first_output = requests[0].get("output_token_ids")
    require(
        all(
            (record.get("prompt_tokens"), record.get("prompt_token_ids_sha256"))
            == first_prompt
            for record in requests
        ),
        f"{label}: sequential calls did not repeat the same frozen prompt",
    )
    require(
        all(record.get("output_token_ids") == first_output for record in requests),
        f"{label}: repeated deterministic requests produced different output tokens",
    )
    calls = result["execution"]["calls"]
    for previous, current in zip(calls, calls[1:]):
        previous_end = finite_number(previous.get("end_epoch"), f"{label}: call end", positive=True)
        current_start = finite_number(current.get("start_epoch"), f"{label}: call start", positive=True)
        require(
            current_start >= previous_end,
            f"{label}: generate calls overlapped instead of running sequentially",
        )


def _verify_no_failure_markers(value: Any, label: str) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lower = str(key).lower()
            if (
                lower == "error"
                or "fallback" in lower
                or ("evict" in lower and lower != "evictions")
            ):
                if isinstance(nested, bool):
                    require(not nested, f"{label}: nonzero failure marker {key}={nested!r}")
                elif isinstance(nested, (int, float)):
                    require(nested == 0, f"{label}: nonzero failure marker {key}={nested!r}")
                elif isinstance(nested, str):
                    require(not nested, f"{label}: nonempty failure marker {key}={nested!r}")
                elif nested is not None:
                    require(not nested, f"{label}: nonempty failure marker {key}")
            _verify_no_failure_markers(nested, f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _verify_no_failure_markers(nested, f"{label}[{index}]")


def verify_mode_result(
    value: Any,
    *,
    expected_mode: int,
    expected_bytes: int,
    label: str,
) -> dict[str, int]:
    require(isinstance(value, dict), f"{label}: result is not an object")
    required_fields = {
        "ok",
        "reason",
        "mode",
        "requested_count",
        "success_count",
        "failure_count",
        "conflict_count",
        "requested_bytes",
        "accepted_bytes",
        "remap_count",
        "tlb_invalidate_count",
        "copy_push_count",
        "ioctl_calls",
        "ioctl_wall_ns",
        "wall_ns",
        "rm_status",
        "entry_statuses",
        "batches",
    }
    require(required_fields <= set(value), f"{label}: missing fields {sorted(required_fields - set(value))}")
    require(value.get("ok") is True and value.get("reason") == "ok", f"{label}: result is not successful")
    require(value.get("mode") == expected_mode, f"{label}: expected mode {expected_mode}")
    count = value.get("requested_count")
    require(isinstance(count, int) and not isinstance(count, bool) and count > 0, f"{label}: invalid requested_count")
    require(value.get("success_count") == count, f"{label}: success count mismatch")
    require(value.get("failure_count") == 0, f"{label}: failure_count is nonzero")
    require(value.get("conflict_count") == 0, f"{label}: conflict_count is nonzero")
    require(value.get("requested_bytes") == expected_bytes, f"{label}: requested byte mismatch")
    require(value.get("accepted_bytes") == expected_bytes, f"{label}: accepted byte mismatch")
    require(value.get("rm_status") == 0, f"{label}: rm_status is nonzero")
    require(value.get("errno", 0) == 0, f"{label}: errno is nonzero")
    statuses = value.get("entry_statuses")
    require(isinstance(statuses, list) and len(statuses) == count, f"{label}: entry statuses incomplete")
    require(all(status == 0 for status in statuses), f"{label}: entry status is nonzero")
    batches = value.get("batches")
    ioctl_calls = value.get("ioctl_calls")
    require(isinstance(batches, list) and batches, f"{label}: batch list is empty")
    require(ioctl_calls == len(batches), f"{label}: ioctl/batch count mismatch")
    ioctl_wall_ns = value.get("ioctl_wall_ns")
    wall_ns = value.get("wall_ns")
    require(isinstance(ioctl_wall_ns, int) and ioctl_wall_ns > 0, f"{label}: invalid ioctl wall time")
    require(isinstance(wall_ns, int) and wall_ns >= ioctl_wall_ns, f"{label}: invalid complete wall time")

    aggregate_fields = (
        "count",
        "success_count",
        "failure_count",
        "conflict_count",
        "requested_bytes",
        "accepted_bytes",
        "remap_count",
        "tlb_invalidate_count",
        "copy_push_count",
    )
    sums = {field: 0 for field in aggregate_fields}
    batch_indices: list[int] = []
    batch_wall_sum = 0
    for batch_index, batch in enumerate(batches):
        batch_label = f"{label}.batches[{batch_index}]"
        require(isinstance(batch, dict), f"{batch_label}: batch is not an object")
        require(batch.get("ok") is True and batch.get("reason") == "ok", f"{batch_label}: batch failed")
        require(batch.get("problems") == [], f"{batch_label}: problems is not empty")
        require(batch.get("ioctl_ret") == 0, f"{batch_label}: ioctl return is nonzero")
        require(batch.get("errno") == 0, f"{batch_label}: errno is nonzero")
        require(batch.get("rm_status") == 0, f"{batch_label}: rm_status is nonzero")
        batch_count = batch.get("count")
        require(isinstance(batch_count, int) and batch_count > 0, f"{batch_label}: invalid count")
        indices = batch.get("input_indices")
        batch_statuses = batch.get("entry_statuses")
        require(isinstance(indices, list) and len(indices) == batch_count, f"{batch_label}: input indices incomplete")
        require(isinstance(batch_statuses, list) and len(batch_statuses) == batch_count, f"{batch_label}: statuses incomplete")
        require(all(status == 0 for status in batch_statuses), f"{batch_label}: entry failure")
        batch_indices.extend(indices)
        batch_wall = batch.get("wall_ns")
        require(isinstance(batch_wall, int) and batch_wall > 0, f"{batch_label}: invalid wall time")
        batch_wall_sum += batch_wall
        for field in aggregate_fields:
            field_value = batch.get(field)
            require(isinstance(field_value, int) and field_value >= 0, f"{batch_label}: invalid {field}")
            sums[field] += field_value
        require(batch.get("success_count") == batch_count, f"{batch_label}: short success")
        require(batch.get("failure_count") == 0, f"{batch_label}: failure_count is nonzero")
        require(batch.get("conflict_count") == 0, f"{batch_label}: conflict_count is nonzero")
        require(batch.get("requested_bytes") == batch.get("accepted_bytes"), f"{batch_label}: byte acceptance mismatch")
        if expected_mode == MODE_PREALLOCATE:
            require(batch.get("remap_count") == 0, f"{batch_label}: mode 8 remapped PTEs")
            require(batch.get("tlb_invalidate_count") == 0, f"{batch_label}: mode 8 invalidated TLB")
            require(batch.get("copy_push_count") == 0, f"{batch_label}: mode 8 issued copy")
        else:
            require(batch.get("remap_count") == batch_count, f"{batch_label}: mode 10 short remap")
            require(batch.get("tlb_invalidate_count", 0) > 0, f"{batch_label}: mode 10 lacks TLB invalidate")
            require(batch.get("copy_push_count", 0) > 0, f"{batch_label}: mode 10 lacks copy push")

    require(sorted(batch_indices) == list(range(count)), f"{label}: batches do not cover every input exactly once")
    require(batch_wall_sum == ioctl_wall_ns, f"{label}: batch/ioctl time does not close")
    top_mapping = {
        "requested_count": "count",
        "success_count": "success_count",
        "failure_count": "failure_count",
        "conflict_count": "conflict_count",
        "requested_bytes": "requested_bytes",
        "accepted_bytes": "accepted_bytes",
        "remap_count": "remap_count",
        "tlb_invalidate_count": "tlb_invalidate_count",
        "copy_push_count": "copy_push_count",
    }
    for top_field, sum_field in top_mapping.items():
        require(value.get(top_field) == sums[sum_field], f"{label}: {top_field} does not close across batches")

    if expected_mode == MODE_PREALLOCATE:
        require(value.get("remap_count") == 0, f"{label}: mode 8 remap count is nonzero")
        require(value.get("tlb_invalidate_count") == 0, f"{label}: mode 8 TLB count is nonzero")
        require(value.get("copy_push_count") == 0, f"{label}: mode 8 copy count is nonzero")
    else:
        require(value.get("remap_count") == count, f"{label}: mode 10 remap count mismatch")
        require(value.get("tlb_invalidate_count", 0) > 0, f"{label}: mode 10 TLB count is zero")
        require(value.get("copy_push_count", 0) > 0, f"{label}: mode 10 copy count is zero")
    _verify_no_failure_markers(value, label)
    return {
        "entries": count,
        "bytes": expected_bytes,
        "ioctl_calls": int(ioctl_calls),
        "wall_ns": int(wall_ns),
        "ioctl_wall_ns": int(ioctl_wall_ns),
        "remap_count": int(value["remap_count"]),
        "tlb_invalidate_count": int(value["tlb_invalidate_count"]),
        "copy_push_count": int(value["copy_push_count"]),
    }


def verify_release_result(
    value: Any,
    *,
    expected_count: int,
    label: str,
) -> dict[str, int]:
    require(isinstance(value, dict), f"{label}: release result is not an object")
    required_fields = {
        "ok",
        "reason",
        "mode",
        "requested_count",
        "success_count",
        "failure_count",
        "conflict_count",
        "requested_bytes",
        "accepted_bytes",
        "remap_count",
        "tlb_invalidate_count",
        "copy_push_count",
        "ioctl_calls",
        "ioctl_wall_ns",
        "wall_ns",
        "rm_status",
        "entry_statuses",
        "batches",
    }
    require(
        required_fields <= set(value),
        f"{label}: missing fields {sorted(required_fields - set(value))}",
    )
    require(value.get("ok") is True and value.get("reason") == "ok", f"{label}: release failed")
    require(value.get("mode") == MODE_RELEASE, f"{label}: expected mode 19")
    require(value.get("requested_count") == expected_count and expected_count > 0, f"{label}: release count mismatch")
    require(value.get("success_count") == expected_count, f"{label}: short release")
    require(value.get("failure_count") == 0, f"{label}: failure_count is nonzero")
    require(value.get("conflict_count") == 0, f"{label}: conflict_count is nonzero")
    require(value.get("requested_bytes") == 0, f"{label}: identity-only mode19 requested bytes must be zero")
    accepted_bytes = value.get("accepted_bytes")
    require(isinstance(accepted_bytes, int) and accepted_bytes > 0, f"{label}: no backing bytes were reclaimed")
    require(value.get("remap_count") == expected_count, f"{label}: release remap count mismatch")
    tlb_count = value.get("tlb_invalidate_count")
    require(isinstance(tlb_count, int) and 0 <= tlb_count <= expected_count, f"{label}: invalid release TLB count")
    require(value.get("copy_push_count") == 0, f"{label}: release unexpectedly issued a copy")
    require(value.get("rm_status") == 0 and value.get("errno", 0) == 0, f"{label}: driver status is nonzero")
    statuses = value.get("entry_statuses")
    require(isinstance(statuses, list) and len(statuses) == expected_count, f"{label}: release statuses incomplete")
    require(all(status == 0 for status in statuses), f"{label}: a release entry failed")
    batches = value.get("batches")
    require(isinstance(batches, list) and batches, f"{label}: release batches are empty")
    require(value.get("ioctl_calls") == len(batches), f"{label}: release ioctl/batch mismatch")
    ioctl_wall_ns = value.get("ioctl_wall_ns")
    wall_ns = value.get("wall_ns")
    require(isinstance(ioctl_wall_ns, int) and ioctl_wall_ns > 0, f"{label}: invalid ioctl wall")
    require(isinstance(wall_ns, int) and wall_ns >= ioctl_wall_ns, f"{label}: invalid complete wall")

    total_count = 0
    total_success = 0
    total_accepted = 0
    total_remaps = 0
    total_tlb = 0
    total_ioctl_wall = 0
    indices: list[int] = []
    for batch_index, batch in enumerate(batches):
        batch_label = f"{label}.batches[{batch_index}]"
        require(isinstance(batch, dict), f"{batch_label}: batch is not an object")
        require(batch.get("ok") is True and batch.get("reason") == "ok", f"{batch_label}: batch failed")
        require(batch.get("problems") == [], f"{batch_label}: problems is nonempty")
        require(batch.get("ioctl_ret") == 0 and batch.get("errno") == 0 and batch.get("rm_status") == 0, f"{batch_label}: driver status is nonzero")
        count = batch.get("count")
        require(isinstance(count, int) and count > 0, f"{batch_label}: invalid count")
        batch_indices = batch.get("input_indices")
        batch_statuses = batch.get("entry_statuses")
        require(isinstance(batch_indices, list) and len(batch_indices) == count, f"{batch_label}: indices incomplete")
        require(isinstance(batch_statuses, list) and len(batch_statuses) == count, f"{batch_label}: statuses incomplete")
        require(all(status == 0 for status in batch_statuses), f"{batch_label}: entry failed")
        require(batch.get("success_count") == count, f"{batch_label}: short success")
        require(batch.get("failure_count") == 0 and batch.get("conflict_count") == 0, f"{batch_label}: failure/conflict is nonzero")
        require(batch.get("requested_bytes") == 0, f"{batch_label}: requested bytes is nonzero")
        batch_accepted = batch.get("accepted_bytes")
        require(isinstance(batch_accepted, int) and batch_accepted > 0, f"{batch_label}: reclaimed bytes is zero")
        require(batch.get("remap_count") == count, f"{batch_label}: release count mismatch")
        batch_tlb = batch.get("tlb_invalidate_count")
        require(isinstance(batch_tlb, int) and 0 <= batch_tlb <= count, f"{batch_label}: invalid TLB count")
        require(batch.get("copy_push_count") == 0, f"{batch_label}: copy push is nonzero")
        batch_wall = batch.get("wall_ns")
        require(isinstance(batch_wall, int) and batch_wall > 0, f"{batch_label}: invalid wall")
        indices.extend(batch_indices)
        total_count += count
        total_success += int(batch["success_count"])
        total_accepted += batch_accepted
        total_remaps += int(batch["remap_count"])
        total_tlb += batch_tlb
        total_ioctl_wall += batch_wall

    require(sorted(indices) == list(range(expected_count)), f"{label}: batches do not cover entries exactly once")
    require(total_count == expected_count and total_success == expected_count, f"{label}: batch counts do not close")
    require(total_accepted == accepted_bytes, f"{label}: reclaimed bytes do not close")
    require(total_remaps == value["remap_count"], f"{label}: remap counts do not close")
    require(total_tlb == tlb_count, f"{label}: TLB counts do not close")
    require(total_ioctl_wall == ioctl_wall_ns, f"{label}: ioctl time does not close")
    _verify_no_failure_markers(value, label)
    return {
        "entries": expected_count,
        "accepted_bytes": accepted_bytes,
        "ioctl_calls": len(batches),
        "wall_ns": wall_ns,
    }


def _cache_snapshot(
    value: Any,
    *,
    expected_bytes_limit: int,
    expected_slot_limit: int,
    label: str,
) -> tuple[int, int]:
    require(isinstance(value, dict), f"{label}: cache snapshot is not an object")
    bytes_ = value.get("bytes")
    slots = value.get("slots")
    require(isinstance(bytes_, int) and bytes_ >= 0, f"{label}: invalid bytes")
    require(isinstance(slots, int) and slots >= 0, f"{label}: invalid slots")
    require(bytes_ <= expected_bytes_limit, f"{label}: byte limit exceeded")
    require(slots <= expected_slot_limit, f"{label}: slot limit exceeded")
    return bytes_, slots


def _cache_limits(
    value: Any,
    *,
    expected_bytes: int,
    expected_slots: int,
    label: str,
) -> None:
    require(isinstance(value, dict), f"{label}: limits is not an object")
    require(value.get("bytes") == expected_bytes, f"{label}: byte limit mismatch")
    require(value.get("max_slots") == expected_slots, f"{label}: slot limit mismatch")


def _slot_identity(value: Mapping[str, Any], label: str) -> tuple[Any, ...]:
    layer = value.get("layer_id")
    expert = value.get("expert_id")
    regions = value.get("region_ids")
    slots = value.get("slot_ids")
    require(isinstance(layer, int) and 0 <= layer < 48, f"{label}: invalid layer")
    require(isinstance(expert, int) and 0 <= expert < 128, f"{label}: invalid expert")
    if regions is None and isinstance(slots, dict):
        regions = list(slots)
    require(isinstance(regions, list) and regions and all(isinstance(item, str) and item for item in regions), f"{label}: invalid region_ids")
    require(len(regions) == len(set(regions)), f"{label}: duplicate region_ids")
    require(isinstance(slots, dict) and set(slots) == set(regions), f"{label}: slot_ids do not cover regions")
    require(all(isinstance(slot_id, int) and slot_id >= 0 for slot_id in slots.values()), f"{label}: invalid slot id")
    require(len(set(slots.values())) == len(slots), f"{label}: duplicate slot id")
    return (
        layer,
        expert,
        tuple(sorted((region, int(slots[region])) for region in regions)),
    )


def verify_slot_cache_trace(
    records: list[dict[str, Any]],
    result: dict[str, Any],
    *,
    cache_bytes: int,
    cache_max_slots: int,
) -> dict[str, Any]:
    require(records, "slot cache arm produced no trace records")
    measurement_start = finite_number(result.get("measurement_start_epoch"), "cache measurement start", positive=True)
    measurement_end = finite_number(result.get("measurement_end_epoch"), "cache measurement end", positive=True)
    require(measurement_end >= measurement_start, "cache measurement interval is invalid")
    totals: dict[str, Any] = {
        "records": len(records),
        "restore_records": 0,
        "cache_hit_records": 0,
        "release_records": 0,
        "capacity_releases": 0,
        "explicit_releases": 0,
        "requested_bytes": 0,
        "placement_wall_ns": 0,
        "mode8_wall_ns": 0,
        "mode8_ioctl_wall_ns": 0,
        "mode8_ioctl_calls": 0,
        "mode10_wall_ns": 0,
        "mode10_ioctl_wall_ns": 0,
        "mode10_ioctl_calls": 0,
        "mode10_entries": 0,
        "mode10_remap_count": 0,
        "mode10_tlb_invalidate_count": 0,
        "mode10_copy_push_count": 0,
        "mode19_wall_ns": 0,
        "mode19_ioctl_calls": 0,
        "mode19_reclaimed_bytes": 0,
    }
    restore_events: list[tuple[int, tuple[Any, ...], dict[str, Any]]] = []
    release_events: list[tuple[int, tuple[Any, ...], dict[str, Any]]] = []
    last_timestamp = 0.0
    eviction_references: list[tuple[int, tuple[Any, ...]]] = []

    for index, record in enumerate(records):
        label = f"slot_cache_trace[{index}]"
        operation = record.get("operation")
        require(operation in ("restore", "release"), f"{label}: unexpected operation")
        require(record.get("ok") is True, f"{label}: operation failed")
        timestamp = finite_number(record.get("timestamp"), f"{label}: timestamp", positive=True)
        require(timestamp >= last_timestamp, f"{label}: timestamps moved backwards")
        last_timestamp = timestamp
        _cache_limits(
            record.get("limits"),
            expected_bytes=cache_bytes,
            expected_slots=cache_max_slots,
            label=f"{label}.limits",
        )
        before_bytes, before_slots = _cache_snapshot(
            record.get("cache_before"),
            expected_bytes_limit=cache_bytes,
            expected_slot_limit=cache_max_slots,
            label=f"{label}.cache_before",
        )
        after_bytes, after_slots = _cache_snapshot(
            record.get("cache_after"),
            expected_bytes_limit=cache_bytes,
            expected_slot_limit=cache_max_slots,
            label=f"{label}.cache_after",
        )
        session_release = operation == "release" and record.get("scope") == "session"
        identity = None if session_release else _slot_identity(record, label)

        if operation == "restore":
            assert identity is not None
            require(measurement_start <= timestamp <= measurement_end, f"{label}: restore lies outside measured requests")
            reason = record.get("reason")
            require(reason in ("ok", "already_slot_ready"), f"{label}: unexpected restore reason")
            require(record.get("failed_region_ids") == [], f"{label}: failed regions is nonempty")
            require(record.get("logical_placement") == "slot_gpu", f"{label}: logical placement mismatch")
            require(record.get("ordinary_uvm_placement") == "cpu", f"{label}: CPU residency was not retained")
            statuses = record.get("statuses")
            require(isinstance(statuses, dict) and statuses and all(value is True for value in statuses.values()), f"{label}: restore statuses failed")
            regions = list(record["slot_ids"])
            require(set(statuses) == set(regions), f"{label}: status/region sets differ")
            requested_bytes = record.get("requested_bytes")
            require(isinstance(requested_bytes, int) and requested_bytes > 0, f"{label}: invalid requested bytes")
            wall_ns = record.get("wall_ns")
            require(isinstance(wall_ns, int) and wall_ns > 0, f"{label}: invalid wall")
            evictions = record.get("evictions")
            require(isinstance(evictions, list), f"{label}: evictions is not a list")
            if reason == "already_slot_ready":
                require(record.get("preallocate") is None and record.get("restore") is None, f"{label}: cache hit unexpectedly issued mode8/10")
                require(not evictions, f"{label}: cache hit unexpectedly evicted a victim")
                require(
                    (before_bytes, before_slots) == (after_bytes, after_slots),
                    f"{label}: cache hit changed occupancy",
                )
                totals["cache_hit_records"] += 1
                _verify_no_failure_markers(record, label)
                continue
            mode8 = verify_mode_result(
                record.get("preallocate"),
                expected_mode=MODE_PREALLOCATE,
                expected_bytes=requested_bytes,
                label=f"{label}.preallocate",
            )
            mode10 = verify_mode_result(
                record.get("restore"),
                expected_mode=MODE_RESTORE,
                expected_bytes=requested_bytes,
                label=f"{label}.restore",
            )
            require(wall_ns >= mode8["wall_ns"] + mode10["wall_ns"], f"{label}: outer wall does not contain mode8+10")
            for eviction_index, eviction in enumerate(evictions):
                require(isinstance(eviction, dict), f"{label}.evictions[{eviction_index}]: not an object")
                eviction_references.append(
                    (index, _slot_identity(eviction, f"{label}.evictions[{eviction_index}]"))
                )
            require(after_bytes >= requested_bytes, f"{label}: restored bytes absent from cache accounting")
            require(after_slots >= len(record["slot_ids"]), f"{label}: restored slots absent from cache accounting")
            restore_events.append((index, identity, record))
            totals["restore_records"] += 1
            totals["requested_bytes"] += requested_bytes
            totals["placement_wall_ns"] += wall_ns
            totals["mode8_wall_ns"] += mode8["wall_ns"]
            totals["mode8_ioctl_wall_ns"] += mode8["ioctl_wall_ns"]
            totals["mode8_ioctl_calls"] += mode8["ioctl_calls"]
            totals["mode10_wall_ns"] += mode10["wall_ns"]
            totals["mode10_ioctl_wall_ns"] += mode10["ioctl_wall_ns"]
            totals["mode10_ioctl_calls"] += mode10["ioctl_calls"]
            totals["mode10_entries"] += mode10["entries"]
            totals["mode10_remap_count"] += mode10["remap_count"]
            totals["mode10_tlb_invalidate_count"] += mode10["tlb_invalidate_count"]
            totals["mode10_copy_push_count"] += mode10["copy_push_count"]
        else:
            trigger = record.get("trigger")
            require(trigger in ("capacity", "explicit"), f"{label}: invalid release trigger")
            require(after_bytes < before_bytes and after_slots < before_slots, f"{label}: release did not reduce cache occupancy")
            require(record.get("poisoned", False) is False, f"{label}: slot policy was poisoned")
            if session_release:
                require(trigger == "explicit", f"{label}: session close is not explicit")
                require(record.get("layer_id") == -1 and record.get("expert_id") == -1, f"{label}: invalid session-close identity")
                close_result = record.get("result")
                require(isinstance(close_result, dict), f"{label}: close result is missing")
                require(close_result.get("ok") is True and close_result.get("reason") == "closed", f"{label}: low-level session did not close")
                require(close_result.get("closed") is True, f"{label}: duplicated UVM fd remained open")
                release = verify_release_result(
                    close_result.get("release"),
                    expected_count=len(record["slot_ids"]),
                    label=f"{label}.result.release",
                )
                close_wall = close_result.get("wall_ns")
                require(isinstance(close_wall, int) and close_wall >= release["wall_ns"], f"{label}: close wall does not contain mode19")
            else:
                assert identity is not None
                require(record.get("reason") == "ok", f"{label}: release reason is not ok")
                release = verify_release_result(
                    record.get("result"),
                    expected_count=len(record["slot_ids"]),
                    label=f"{label}.result",
                )
                release_events.append((index, identity, record))
            totals["release_records"] += 1
            totals[f"{trigger}_releases"] += 1
            totals["mode19_wall_ns"] += release["wall_ns"]
            totals["mode19_ioctl_calls"] += release["ioctl_calls"]
            totals["mode19_reclaimed_bytes"] += release["accepted_bytes"]
        _verify_no_failure_markers(record, label)

    require(totals["restore_records"] >= 2, "cache trace has fewer than two restores")
    require(totals["capacity_releases"] > 0, "cache limit did not trigger a capacity release")
    require(totals["explicit_releases"] > 0, "cache did not emit explicit shutdown release")
    require(release_events, "cache trace has no successful mode19 release")
    final = records[-1]
    require(final.get("operation") == "release" and final.get("trigger") == "explicit", "final cache event is not explicit release")
    require(final.get("cache_after") == {"bytes": 0, "slots": 0}, "final userspace cache is not empty")

    capacity_by_identity = {
        identity: release_index
        for release_index, identity, release in release_events
        if release.get("trigger") == "capacity"
    }
    require(capacity_by_identity, "no capacity victim identity was recorded")
    reuse_pairs = [
        (release_index, restore_index, identity)
        for identity, release_index in capacity_by_identity.items()
        for restore_index, restore_identity, _restore in restore_events
        if restore_index > release_index and restore_identity == identity
    ]
    require(reuse_pairs, "no capacity-released logical slot identity was restored later")
    for restore_index, identity in eviction_references:
        require(
            any(
                release_index < restore_index
                and release_identity == identity
                and release.get("trigger") == "capacity"
                for release_index, release_identity, release in release_events
            ),
            "restore eviction reference lacks a preceding successful capacity release",
        )
    require(eviction_references, "restore records do not report any capacity evictions")

    totals["placement_ms"] = totals["placement_wall_ns"] / 1e6
    totals["mode8_ms"] = totals["mode8_wall_ns"] / 1e6
    totals["mode10_ms"] = totals["mode10_wall_ns"] / 1e6
    totals["mode19_ms"] = totals["mode19_wall_ns"] / 1e6
    totals["placement_effective_gib_s"] = (
        totals["requested_bytes"] * 1e9
        / (GIB * totals["placement_wall_ns"])
    )
    totals["mode10_effective_gib_s"] = (
        totals["requested_bytes"] * 1e9 / (GIB * totals["mode10_wall_ns"])
    )
    totals["cache_bytes_limit"] = cache_bytes
    totals["cache_max_slots"] = cache_max_slots
    totals["bounded"] = True
    totals["release_then_restore_proven"] = True
    totals["final_cache_empty"] = True
    totals["mode8_mode10_mode19_complete"] = True
    return totals


def verify_slot_trace(
    records: list[dict[str, Any]], result: dict[str, Any]
) -> dict[str, Any]:
    require(records, "slot arm produced no slot-direct trace records")
    measurement_start = finite_number(result.get("measurement_start_epoch"), "slot measurement start", positive=True)
    measurement_end = finite_number(result.get("measurement_end_epoch"), "slot measurement end", positive=True)
    require(measurement_end >= measurement_start, "slot measurement epoch interval is invalid")
    seen_experts: set[tuple[int, int]] = set()
    totals = {
        "records": 0,
        "requested_bytes": 0,
        "placement_wall_ns": 0,
        "mode8_wall_ns": 0,
        "mode8_ioctl_wall_ns": 0,
        "mode8_ioctl_calls": 0,
        "mode10_wall_ns": 0,
        "mode10_ioctl_wall_ns": 0,
        "mode10_ioctl_calls": 0,
        "mode10_entries": 0,
        "mode10_remap_count": 0,
        "mode10_tlb_invalidate_count": 0,
        "mode10_copy_push_count": 0,
    }
    for index, record in enumerate(records):
        label = f"slot_trace[{index}]"
        require(record.get("operation") == "restore", f"{label}: unexpected operation")
        require(record.get("ok") is True and record.get("reason") == "ok", f"{label}: restore failed")
        require(record.get("failed_region_ids") == [], f"{label}: failed region list is nonempty")
        require(record.get("logical_placement") == "slot_gpu", f"{label}: logical placement mismatch")
        require(record.get("ordinary_uvm_placement") == "cpu", f"{label}: CPU residency was not retained")
        statuses = record.get("statuses")
        require(isinstance(statuses, dict) and statuses, f"{label}: statuses are missing")
        require(all(value is True for value in statuses.values()), f"{label}: region status failed")
        layer = record.get("layer_id")
        expert = record.get("expert_id")
        require(isinstance(layer, int) and 0 <= layer < 48, f"{label}: invalid layer")
        require(isinstance(expert, int) and 0 <= expert < 128, f"{label}: invalid expert")
        key = (layer, expert)
        require(key not in seen_experts, f"{label}: expert restored more than once")
        seen_experts.add(key)
        timestamp = finite_number(record.get("timestamp"), f"{label}: timestamp", positive=True)
        require(measurement_start <= timestamp <= measurement_end, f"{label}: restore lies outside measured generate")
        requested_bytes = record.get("requested_bytes")
        require(isinstance(requested_bytes, int) and requested_bytes > 0, f"{label}: invalid requested bytes")
        wall_ns = record.get("wall_ns")
        require(isinstance(wall_ns, int) and wall_ns > 0, f"{label}: invalid placement wall")
        slot_ids = record.get("slot_ids")
        require(isinstance(slot_ids, dict) and set(slot_ids) == set(statuses), f"{label}: slot IDs do not cover regions")
        require(len(set(slot_ids.values())) == len(slot_ids), f"{label}: slot IDs collide within restore")
        mode8 = verify_mode_result(
            record.get("preallocate"),
            expected_mode=MODE_PREALLOCATE,
            expected_bytes=requested_bytes,
            label=f"{label}.preallocate",
        )
        mode10 = verify_mode_result(
            record.get("restore"),
            expected_mode=MODE_RESTORE,
            expected_bytes=requested_bytes,
            label=f"{label}.restore",
        )
        require(wall_ns >= mode8["wall_ns"] + mode10["wall_ns"], f"{label}: outer wall does not contain mode 8+10")
        _verify_no_failure_markers(record, label)
        totals["records"] += 1
        totals["requested_bytes"] += requested_bytes
        totals["placement_wall_ns"] += wall_ns
        totals["mode8_wall_ns"] += mode8["wall_ns"]
        totals["mode8_ioctl_wall_ns"] += mode8["ioctl_wall_ns"]
        totals["mode8_ioctl_calls"] += mode8["ioctl_calls"]
        totals["mode10_wall_ns"] += mode10["wall_ns"]
        totals["mode10_ioctl_wall_ns"] += mode10["ioctl_wall_ns"]
        totals["mode10_ioctl_calls"] += mode10["ioctl_calls"]
        totals["mode10_entries"] += mode10["entries"]
        totals["mode10_remap_count"] += mode10["remap_count"]
        totals["mode10_tlb_invalidate_count"] += mode10["tlb_invalidate_count"]
        totals["mode10_copy_push_count"] += mode10["copy_push_count"]

    totals["placement_ms"] = totals["placement_wall_ns"] / 1e6
    totals["mode8_ms"] = totals["mode8_wall_ns"] / 1e6
    totals["mode10_ms"] = totals["mode10_wall_ns"] / 1e6
    totals["placement_effective_gib_s"] = (
        totals["requested_bytes"] * 1e9 / (GIB * totals["placement_wall_ns"])
    )
    totals["mode10_effective_gib_s"] = (
        totals["requested_bytes"] * 1e9 / (GIB * totals["mode10_wall_ns"])
    )
    totals["mode8_mode10_complete"] = True
    return totals


def verify_slot_counters(delta: Mapping[str, int], placement: Mapping[str, Any]) -> dict[str, int]:
    required = (
        "slot_direct_ioctl_calls",
        "slot_direct_per_block_calls",
        "slot_direct_successes",
        "slot_direct_requested_ranges",
        "slot_direct_completed_ranges",
        "slot_direct_requested_bytes",
        "slot_direct_completed_bytes",
        "slot_direct_copy_pushes",
        "slot_direct_fallbacks",
        "slot_direct_errors",
        "slot_direct_vidmem_allocations",
        "slot_direct_vidmem_evictions",
    )
    for key in required:
        require(key in delta, f"slot latency_stats lacks required counter {key}")
    expected_calls = int(placement["mode10_ioctl_calls"])
    expected_entries = int(placement["mode10_entries"])
    expected_bytes = int(placement["requested_bytes"])
    expected_pushes = int(placement["mode10_copy_push_count"])
    exact = {
        "slot_direct_ioctl_calls": expected_calls,
        "slot_direct_per_block_calls": expected_calls,
        "slot_direct_successes": expected_calls,
        "slot_direct_requested_ranges": expected_entries,
        "slot_direct_completed_ranges": expected_entries,
        "slot_direct_requested_bytes": expected_bytes,
        "slot_direct_completed_bytes": expected_bytes,
        "slot_direct_copy_pushes": expected_pushes,
        "slot_direct_fallbacks": 0,
        "slot_direct_errors": 0,
        "slot_direct_vidmem_allocations": 0,
        "slot_direct_vidmem_evictions": 0,
    }
    for key, expected in exact.items():
        require(delta[key] == expected, f"slot counter mismatch: {key}={delta[key]}, expected {expected}")
    if "slot_direct_pte_64k_blocks" in delta:
        require(delta["slot_direct_pte_64k_blocks"] == 0, "slot path unexpectedly installed 64 KiB PTEs")
    if "slot_direct_pte_2m_blocks" in delta:
        require(delta["slot_direct_pte_2m_blocks"] == 0, "slot path unexpectedly installed 2 MiB PTEs")
    return {key: int(delta[key]) for key in exact}


def verify_generic_counters(delta: Mapping[str, int]) -> None:
    for key in (
        "slot_direct_ioctl_calls",
        "slot_direct_successes",
        "slot_direct_requested_ranges",
        "slot_direct_requested_bytes",
        "slot_direct_fallbacks",
        "slot_direct_errors",
        "slot_direct_vidmem_allocations",
        "slot_direct_vidmem_evictions",
    ):
        require(key in delta, f"generic latency_stats lacks required counter {key}")
        require(delta[key] == 0, f"generic arm unexpectedly changed {key} by {delta[key]}")


def safe_ratio(numerator: float, denominator: float, label: str) -> float:
    require(denominator > 0.0, f"cannot compute {label}: denominator is zero")
    return numerator / denominator


def verify_run(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    config = load_json(run_root / "run_config.json")
    require(config.get("schema") == RUN_SCHEMA, "run_config schema mismatch")
    order = config.get("order")
    require(order in ("AB", "BA"), "run order must be AB or BA")
    experiment_kind = config.get("experiment_kind", "cold_pair")
    require(
        experiment_kind in ("cold_pair", "cache_reuse"),
        "unknown integration experiment kind",
    )
    module = config.get("module")
    require(isinstance(module, dict), "run_config module section is missing")
    require(
        isinstance(module.get("custom_srcversion"), str)
        and module.get("loaded_srcversion") == module.get("custom_srcversion"),
        "recorded loaded/custom srcversions differ",
    )
    require(module.get("slot_enable") == 1, "recorded slot enable is not 1")
    require(
        isinstance(module.get("slot_budget_bytes"), int)
        and module["slot_budget_bytes"] >= 5 * GIB // 2,
        "recorded slot budget is below 2.5 GiB",
    )
    arm_dirs = config.get("arm_dirs")
    require(isinstance(arm_dirs, dict) and set(arm_dirs) == {"generic", "slot"}, "run_config arm_dirs is invalid")
    benchmark = config.get("benchmark")
    require(isinstance(benchmark, dict), "run_config benchmark section is missing")
    concurrency = benchmark.get("concurrency")
    output_tokens = benchmark.get("output_tokens")
    require(isinstance(concurrency, int) and concurrency > 0, "invalid configured concurrency")
    require(isinstance(output_tokens, int) and output_tokens > 0, "invalid configured output tokens")
    require(benchmark.get("warmup_requests") == 0, "configured warmup must be zero")
    sequential = experiment_kind == "cache_reuse"
    expected_requests = concurrency
    cache_bytes = 0
    cache_max_slots = 0
    if sequential:
        require(concurrency == 1, "cache reuse benchmark must issue one request per generate")
        expected_requests = benchmark.get("sequential_requests")
        require(
            isinstance(expected_requests, int) and expected_requests >= 2,
            "cache reuse benchmark requires at least two sequential requests",
        )
        cache_config = config.get("slot_cache")
        require(isinstance(cache_config, dict), "run_config slot_cache section is missing")
        cache_bytes = cache_config.get("bytes")
        cache_max_slots = cache_config.get("max_slots")
        require(isinstance(cache_bytes, int) and cache_bytes > 0, "invalid userspace cache byte limit")
        require(isinstance(cache_max_slots, int) and cache_max_slots > 0, "invalid userspace cache slot limit")
    fixture_path = Path(str(benchmark.get("evaluation_fixture", "")))
    plan_path = Path(str(benchmark.get("plan", "")))
    require(fixture_path.is_file(), f"evaluation fixture is unavailable: {fixture_path}")
    require(plan_path.is_file(), f"plan is unavailable: {plan_path}")
    fixture_sha = sha256_file(fixture_path)
    plan_sha = sha256_file(plan_path)
    inputs = config.get("inputs")
    require(isinstance(inputs, dict), "run_config inputs section is missing")
    require(inputs.get("evaluation_fixture_sha256") == fixture_sha, "run_config fixture SHA mismatch")
    require(inputs.get("plan_sha256") == plan_sha, "run_config plan SHA mismatch")
    plan = load_json(plan_path)
    cold, hot = _expected_qwen_plan(plan)
    remote_bytes = plan.get("metadata", {}).get("actual_offload_main_bytes")
    require(isinstance(remote_bytes, int) and remote_bytes > 0, "plan lacks actual_offload_main_bytes")

    results: dict[str, dict[str, Any]] = {}
    metrics: dict[str, dict[str, float]] = {}
    deltas: dict[str, dict[str, int]] = {}
    traces: dict[str, list[dict[str, Any]]] = {}
    for arm in ("generic", "slot"):
        arm_dir = run_root / str(arm_dirs[arm])
        require(arm_dir.is_dir(), f"missing arm directory: {arm_dir}")
        for artifact_name in (
            "command.txt",
            "run.log",
            "execution.env",
            "gpu.before.csv",
            "gpu.after.csv",
        ):
            require(
                (arm_dir / artifact_name).is_file(),
                f"{arm}: missing required artifact {artifact_name}",
            )
        returncode_path = arm_dir / "returncode.txt"
        require(returncode_path.is_file(), f"{arm}: return code artifact is missing")
        try:
            returncode = int(returncode_path.read_text(encoding="utf-8").strip())
        except ValueError as exc:
            raise VerificationError(f"{arm}: invalid return code artifact") from exc
        require(returncode == 0, f"{arm}: benchmark returned {returncode}")
        environment = parse_environment(arm_dir / "environment.txt")
        expected_access = "restore" if arm == "generic" else "slot_restore"
        require(environment.get("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE") == expected_access, f"{arm}: environment access mode mismatch")
        require(environment.get("VLLM_UVM_SLOT_DIRECT_TRACE_FILE") == str((arm_dir / "slot_trace.jsonl").resolve()), f"{arm}: trace path mismatch")
        require(
            environment.get("VLLM_UVM_INIT_WEIGHT_PLAN_FILE")
            == str(plan_path.resolve()),
            f"{arm}: frozen plan path mismatch",
        )
        require(
            environment.get("VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST") == "",
            f"{arm}: incompatible remote-only placement manifest was enabled",
        )
        if sequential:
            require(
                environment.get("VLLM_UVM_SLOT_DIRECT_CACHE_BYTES")
                == str(cache_bytes),
                f"{arm}: userspace cache byte limit mismatch",
            )
            require(
                environment.get("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS")
                == str(cache_max_slots),
                f"{arm}: userspace cache slot limit mismatch",
            )
        results[arm], metrics[arm] = verify_benchmark_result(
            arm_dir / "result.json",
            arm=arm,
            expected_concurrency=concurrency,
            expected_requests=expected_requests,
            expected_output_tokens=output_tokens,
            fixture_sha256=fixture_sha,
            sequential=sequential,
        )
        captured_environment = results[arm]["environment"]
        for key in (
            "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE",
            "VLLM_UVM_SLOT_DIRECT_TRACE_FILE",
            "VLLM_UVM_INIT_WEIGHT_PLAN_FILE",
            "VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST",
        ):
            require(
                captured_environment.get(key) == environment.get(key),
                f"{arm}: benchmark/environment artifact differs for {key}",
            )
        if sequential:
            for key in (
                "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES",
                "VLLM_UVM_SLOT_DIRECT_MAX_SLOTS",
            ):
                require(
                    captured_environment.get(key) == environment.get(key),
                    f"{arm}: benchmark/environment artifact differs for {key}",
                )
        before = read_latency_stats(arm_dir / "latency_stats.before.txt")
        after = read_latency_stats(arm_dir / "latency_stats.after.txt")
        deltas[arm] = profile_delta(before, after, arm)
        traces[arm] = load_jsonl(arm_dir / "slot_trace.jsonl")

    compare_tokens(results["generic"], results["slot"])
    if sequential:
        verify_sequential_requests(results["generic"], "generic")
        verify_sequential_requests(results["slot"], "slot")
    require(not traces["generic"], "generic arm unexpectedly emitted slot-direct trace records")
    verify_generic_counters(deltas["generic"])
    if sequential:
        placement = verify_slot_cache_trace(
            traces["slot"],
            results["slot"],
            cache_bytes=cache_bytes,
            cache_max_slots=cache_max_slots,
        )
    else:
        placement = verify_slot_trace(traces["slot"], results["slot"])
    slot_counter_summary = verify_slot_counters(deltas["slot"], placement)

    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "pass",
        "run_root": str(run_root),
        "order": order,
        "experiment_kind": experiment_kind,
        "correctness": {
            "both_status_ok": True,
            "cold_process_per_arm": True,
            "warmup_requests": 0,
            "prompt_tokens_exact": True,
            "output_tokens_exact": True,
            "frozen_initial_offload_plan_equal": True,
            "slot_trace_all_ok": True,
            "mode8_mode10_complete": True,
            "sequential_same_llm_requests": sequential,
            "successful_release_then_restore": sequential,
            "userspace_cache_bounded": sequential,
            "final_resources_closed": sequential,
            "zero_fallback_evict_error": True,
        },
        "arms": metrics,
        "comparison": {
            "slot_over_generic_ttft_ratio": safe_ratio(metrics["slot"]["ttft_ms"], metrics["generic"]["ttft_ms"], "TTFT ratio") if metrics["generic"]["ttft_ms"] > 0 else None,
            "generic_over_slot_ttft_speedup": safe_ratio(metrics["generic"]["ttft_ms"], metrics["slot"]["ttft_ms"], "TTFT speedup") if metrics["slot"]["ttft_ms"] > 0 else None,
            "slot_over_generic_e2e_ratio": safe_ratio(metrics["slot"]["e2e_ms"], metrics["generic"]["e2e_ms"], "E2E ratio"),
            "generic_over_slot_e2e_speedup": safe_ratio(metrics["generic"]["e2e_ms"], metrics["slot"]["e2e_ms"], "E2E speedup"),
            "slot_over_generic_tokens_per_second": safe_ratio(metrics["slot"]["output_tokens_per_second"], metrics["generic"]["output_tokens_per_second"], "throughput ratio"),
        },
        "slot_placement": placement,
        "slot_driver_counters": slot_counter_summary,
        "provenance": {
            "evaluation_fixture_sha256": fixture_sha,
            "plan_sha256": plan_sha,
            "cold_experts": len(cold),
            "hot_experts": len(hot),
            "planned_offload_bytes": remote_bytes,
        },
    }
    return summary


def write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    arms = summary["arms"]
    placement = summary["slot_placement"]
    comparison = summary["comparison"]
    cache_reuse = summary.get("experiment_kind") == "cache_reuse"
    lines = [
        (
            "# vLLM bounded slot-cache reuse"
            if cache_reuse
            else "# vLLM generic restore vs slot restore"
        ),
        "",
        f"Status: **{summary['status']}**; cold-process order: `{summary['order']}`.",
        "",
        "| arm | TTFT (ms) | E2E (ms) | output tok/s |",
        "|---|---:|---:|---:|",
    ]
    for arm in ("generic", "slot"):
        row = arms[arm]
        lines.append(
            f"| {arm} | {row['ttft_ms']:.3f} | {row['e2e_ms']:.3f} | "
            f"{row['output_tokens_per_second']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Slot/generic E2E ratio: `{comparison['slot_over_generic_e2e_ratio']:.4f}x`; "
            f"slot/generic throughput: `{comparison['slot_over_generic_tokens_per_second']:.4f}x`.",
            "",
            "| slot placement | value |",
            "|---|---:|",
            f"| restore events | {placement.get('restore_records', placement['records'])} |",
            f"| restored bytes | {placement['requested_bytes']} |",
            f"| total placement | {placement['placement_ms']:.3f} ms |",
            f"| total placement bandwidth | {placement['placement_effective_gib_s']:.3f} GiB/s |",
            f"| mode 8 preallocate | {placement['mode8_ms']:.3f} ms |",
            f"| mode 10 copy+map | {placement['mode10_ms']:.3f} ms |",
            f"| mode 10 effective bandwidth | {placement['mode10_effective_gib_s']:.3f} GiB/s |",
        ]
    )
    if cache_reuse:
        lines.extend(
            [
                f"| capacity releases | {placement['capacity_releases']} |",
                f"| explicit releases | {placement['explicit_releases']} |",
                f"| mode 19 release | {placement['mode19_ms']:.3f} ms |",
                f"| mode 19 reclaimed bytes | {placement['mode19_reclaimed_bytes']} |",
                f"| cache byte/slot limits | {placement['cache_bytes_limit']} / {placement['cache_max_slots']} |",
                "",
                "The same LLM instance handled sequential repeated requests. Token IDs "
                "match the generic arm and each other; a capacity-released identity was "
                "later restored through mode 8/10, all occupancy snapshots remained "
                "bounded, and explicit mode 19 cleanup ended at zero bytes/slots.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Prompt/output token IDs are identical. Every slot trace completed mode 8 and "
                "mode 10 with exact byte/count closure and zero fallback, eviction, timed "
                "allocation, or error counters.",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or args.run_root / "summary.json"
    markdown = args.markdown or args.run_root / "SUMMARY.md"
    try:
        summary = verify_run(args.run_root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_markdown(summary, markdown)
    except (OSError, VerificationError) as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        return 1
    placement = summary["slot_placement"]
    print(
        "VERIFIED "
        f"kind={summary['experiment_kind']} "
        f"generic_e2e_ms={summary['arms']['generic']['e2e_ms']:.3f} "
        f"slot_e2e_ms={summary['arms']['slot']['e2e_ms']:.3f} "
        f"slot_placement_ms={placement['placement_ms']:.3f} "
        f"slot_placement_gib_s={placement['placement_effective_gib_s']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
