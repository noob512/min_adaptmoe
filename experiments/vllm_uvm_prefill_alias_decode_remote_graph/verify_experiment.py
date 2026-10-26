#!/usr/bin/env python3
"""Fail-closed verifier and performance reducer for the H/G/R/C experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DUAL_BANK_BYTES = 1_207_959_552
LOGICAL_BYTES = 2_415_919_104
PARAMETER_COUNT = 8
ALIAS_COUNT = 6_912
ALIAS_BYTES = 3_623_878_656
ARM_NAMES = {
    "H": "alias_prefill_remote_eager_decode",
    "G": "alias_prefill_remote_graph_decode",
    "R": "remote_prefill_remote_eager_decode",
    "C": "remote_prefill_remote_graph_decode",
}
GRAPH_ARMS = {"G", "C"}
HYBRID_ARMS = {"H", "G"}
PHASE_MODE = {
    "H": "prefill_remote_decode",
    "G": "prefill_remote_decode",
    "R": "remote_only",
    "C": "remote_only",
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def load_env(path: Path) -> dict[str, str]:
    require(path.is_file(), f"missing environment: {path}")
    result: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line:
            continue
        require("=" in line, f"{path}:{line_number}: malformed line")
        key, value = line.split("=", 1)
        require(bool(key) and key not in result, f"{path}:{line_number}: duplicate key")
        result[key] = value
    return result


def read_tsv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing TSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]


def validate_input_hashes(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing input hash manifest: {path}")
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64, f"hash line {line_number} malformed")
        expected, raw_path = fields
        candidate = Path(raw_path.lstrip("*"))
        require(candidate.is_file(), f"hashed input disappeared: {candidate}")
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        require(actual == expected, f"hashed input changed: {candidate}")
        records.append({"path": str(candidate), "sha256": actual})
    require(bool(records), "input hash manifest is empty")
    return records


def finite(value: Any, label: str, *, positive: bool = False) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label}: non-finite value",
    )
    number = float(value)
    require(number > 0 if positive else number >= 0, f"{label}: out of range")
    return number


def output_digest(outputs: Any) -> str:
    require(isinstance(outputs, list) and bool(outputs), "outputs are absent")
    payload: list[list[int]] = []
    for index, item in enumerate(outputs):
        require(isinstance(item, dict), f"output {index} is not an object")
        require(item.get("finished") is True, f"output {index} is unfinished")
        tokens = item.get("output_token_ids")
        require(isinstance(tokens, list) and all(type(x) is int for x in tokens), "bad tokens")
        payload.append(tokens)
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def validate_parameter_snapshots(result: dict[str, Any], label: str) -> tuple[int, ...]:
    snapshots = result.get("parameter_va_snapshots")
    require(isinstance(snapshots, dict), f"{label}: VA snapshots absent")
    stages = ("after_load", "after_prefill", "after_first_decode", "final")
    reference: tuple[int, ...] | None = None
    names: tuple[str, ...] | None = None
    for stage in stages:
        value = snapshots.get(stage)
        require(isinstance(value, dict), f"{label}: {stage} snapshot absent")
        require(value.get("count") == PARAMETER_COUNT, f"{label}: {stage} count differs")
        require(value.get("bytes") == LOGICAL_BYTES, f"{label}: {stage} bytes differ")
        records = value.get("records")
        require(isinstance(records, list) and len(records) == PARAMETER_COUNT, f"{label}: bad records")
        current_names = tuple(str(item.get("name")) for item in records)
        current_vas = tuple(int(item.get("data_ptr", 0)) for item in records)
        require(len(set(current_vas)) == PARAMETER_COUNT and min(current_vas) > 0, f"{label}: bad VAs")
        require(sum(int(item.get("bytes", 0)) for item in records) == LOGICAL_BYTES, f"{label}: record byte total differs")
        require(all(str(item.get("device", "")).startswith("cuda") for item in records), f"{label}: selected parameter not CUDA-visible")
        if reference is None:
            reference = current_vas
            names = current_names
        else:
            require(current_vas == reference, f"{label}: Parameter VA changed at {stage}")
            require(current_names == names, f"{label}: parameter order changed at {stage}")
    assert reference is not None
    return reference


def validate_remote_transition(
    transition: dict[str, Any],
    label: str,
    *,
    preserved_graph: bool | None = None,
) -> dict[str, float]:
    require(transition.get("event") == "cpu_offload_uvm_original_va_remote_transition", f"{label}: wrong transition event")
    require(transition.get("ok") is True, f"{label}: transition failed")
    require(transition.get("pre_state") == "sealed", f"{label}: transition did not start sealed")
    require(transition.get("post_state") == "remote", f"{label}: transition did not end remote")
    if preserved_graph is not None:
        require(transition.get("preserve_remote_graphs") is preserved_graph, f"{label}: Graph preservation differs")
    before = transition.get("parameter_vas_before")
    after = transition.get("parameter_vas_after")
    require(isinstance(before, list) and len(before) == PARAMETER_COUNT, f"{label}: before VAs absent")
    require(before == after, f"{label}: VAs changed across transition")

    remote = transition.get("remote_map")
    require(isinstance(remote, dict), f"{label}: remote audit absent")
    expected_remote = {
        "ok": True,
        "reason": "ok",
        "requested_count": PARAMETER_COUNT,
        "requested_bytes": LOGICAL_BYTES,
        "success_count": PARAMETER_COUNT,
        "failure_count": 0,
        "unknown_bytes": 0,
        "first_failed_index": -1,
        "ioctl_ret": 0,
        "rm_status": 0,
        "errno": 0,
    }
    for key, expected in expected_remote.items():
        require(remote.get(key) == expected, f"{label}: remote {key} differs")
    for key in ("cpu_resident_bytes", "remote_mapped_bytes", "covered_bytes"):
        require(int(remote.get(key, -1)) >= LOGICAL_BYTES, f"{label}: remote {key} incomplete")
    require(int(remote.get("ioctl_calls", 0)) >= 1, f"{label}: no remote-map ioctl")

    physical = transition.get("physical_residency")
    require(isinstance(physical, dict) and physical.get("ok") is True, f"{label}: physical audit failed")
    require(int(physical.get("queried_bytes", -1)) == LOGICAL_BYTES, f"{label}: queried bytes differ")
    require(int(physical.get("cpu_bytes", -1)) == LOGICAL_BYTES, f"{label}: CPU residency differs")
    require(int(physical.get("gpu_bytes", -1)) == 0, f"{label}: GPU physical bytes remain")
    require(int(physical.get("any_bytes", -1)) == LOGICAL_BYTES, f"{label}: coverage differs")
    require(int(physical.get("unknown_bytes", -1)) == 0, f"{label}: unknown residency")

    destroy = transition.get("destroy")
    require(isinstance(destroy, dict) and destroy.get("ok") is True, f"{label}: cmd99 wrapper failed")
    raw = destroy.get("destroy")
    require(isinstance(raw, dict) and raw.get("ok") is True, f"{label}: raw cmd99 failed")
    expected_destroy = {
        "cache_cleared": True,
        "session_cleared": True,
        "released_alias_count": ALIAS_COUNT,
        "released_alias_bytes": ALIAS_BYTES,
        "released_backing_bytes": DUAL_BANK_BYTES,
        "remaining_alias_count": 0,
        "remaining_alias_bytes": 0,
        "remaining_backing_bytes": 0,
    }
    for key, expected in expected_destroy.items():
        require(raw.get(key) == expected, f"{label}: cmd99 {key} differs")

    timings = {}
    for key in ("sync_ns", "destroy_ns", "remote_map_ns", "physical_audit_ns", "wall_ns"):
        timings[key.removesuffix("_ns") + "_ms"] = finite(transition.get(key), f"{label}: {key}") / 1e6
    component_ns = sum(int(transition.get(key, 0)) for key in ("sync_ns", "destroy_ns", "remote_map_ns", "physical_audit_ns"))
    require(abs(component_ns - int(transition["wall_ns"])) <= 1_000_000, f"{label}: timing components do not close")
    return timings


def _state(snapshot: dict[str, Any], label: str, expected: str, phase_mode: str, vas: tuple[int, ...]) -> None:
    model = snapshot.get("model")
    driver = snapshot.get("driver")
    require(isinstance(model, dict) and isinstance(driver, dict), f"{label}: diagnostics absent")
    require(model.get("enabled") is True and model.get("backend") == "uvm_original_va", f"{label}: backend differs")
    require(model.get("phase_mode") == phase_mode, f"{label}: phase mode differs")
    require(model.get("lifecycle_state") == expected, f"{label}: lifecycle differs")
    require(tuple(model.get("original_vas", [])) == vas, f"{label}: diagnostic VAs differ")
    require(model.get("cleanup_pending") is False and model.get("cleanup_unresolved") is False, f"{label}: cleanup unresolved")
    if expected == "sealed":
        require(model.get("sealed") is True and int(model.get("session_id", 0)) > 0, f"{label}: not sealed")
        require(model.get("remote_map_active") is False, f"{label}: stale remote map active")
        require(int(model.get("active_driver_backing_bytes", -1)) == DUAL_BANK_BYTES, f"{label}: active backing differs")
        require(driver.get("session_count") == 1, f"{label}: driver session absent")
    else:
        require(model.get("sealed") is False and int(model.get("session_id", -1)) == 0, f"{label}: alias session remains")
        require(model.get("remote_map_active") is True, f"{label}: remote map not active")
        require(int(model.get("active_driver_backing_bytes", -1)) == 0, f"{label}: backing still active")
        require(driver.get("session_count") == 0, f"{label}: userspace alias session remains")


def _segment_records(segment: dict[str, Any], label: str) -> list[dict[str, Any]]:
    records = segment.get("records")
    require(isinstance(records, list), f"{label}: records were compacted unexpectedly")
    return [item for item in records if isinstance(item, dict)]


def validate_manifest_segments(result: dict[str, Any], arm: str, label: str) -> None:
    segments = result.get("manifest_segments")
    require(isinstance(segments, dict), f"{label}: manifest segments absent")
    audit = result.get("config", {}).get("audit_trace") is True
    expected_prefixes = {f"model.layers.{index}" for index in range(4)}
    for name in ("load_and_capture", "serving_prefill", "first_decode", "remaining_decode"):
        segment = segments.get(name)
        require(isinstance(segment, dict), f"{label}: segment {name} absent")
        if audit:
            require(segment.get("actual_h2d_field_count") == segment.get("forward_count"), f"{label}: {name} lacks actual-H2D evidence")
        else:
            require(segment.get("forward_count") == 0, f"{label}: performance process enabled copy trace")
    if not audit:
        return

    load_records = _segment_records(segments["load_and_capture"], f"{label}/load")
    captured = [
        item
        for item in load_records
        if item.get("event") == "cpu_offload_copy_forward"
        and item.get("graph_static_alias_capture_phase") is True
    ]
    if arm in GRAPH_ARMS:
        require(len(captured) >= 8 and len(captured) % 4 == 0, f"{label}: capture forward count differs")
        require(all(item.get("graph_static_alias_lifecycle_state") == "remote" for item in captured), f"{label}: Graph captured outside remote topology")
        require(all(item.get("graph_static_alias_actual_h2d_bytes") == 0 for item in captured), f"{label}: captured Graph contains H2D")
        require(all(item.get("graph_static_alias_copy_submitted") is False for item in captured), f"{label}: captured copy node detected")
    else:
        require(not captured, f"{label}: eager arm captured a Graph")

    prefill = segments["serving_prefill"]
    require(prefill.get("forward_count") == 4, f"{label}: prefill did not visit four selected layers")
    require(set(prefill.get("module_prefixes", [])) == expected_prefixes, f"{label}: prefill layer set differs")
    expected_bytes = LOGICAL_BYTES if arm in HYBRID_ARMS else 0
    require(prefill.get("actual_h2d_bytes") == expected_bytes, f"{label}: prefill actual H2D differs")
    expected_state = "sealed" if arm in HYBRID_ARMS else "remote"
    require(prefill.get("lifecycle_state_counts") == {expected_state: 4}, f"{label}: prefill state differs")
    expected_copy = {"true": 4} if arm in HYBRID_ARMS else {"false": 4}
    require(prefill.get("copy_submitted_counts") == expected_copy, f"{label}: prefill copy flags differ")

    first = segments["first_decode"]
    remainder = segments["remaining_decode"]
    if arm in GRAPH_ARMS:
        require(first.get("forward_count") == 0 and remainder.get("forward_count") == 0, f"{label}: Graph replay re-entered Python hook")
    else:
        warmup = int(result["config"]["warmup_decode_steps"])
        measured = int(result["config"]["measured_decode_steps"])
        require(first.get("forward_count") == 4, f"{label}: eager first decode layer count differs")
        require(remainder.get("forward_count") == 4 * (warmup + measured), f"{label}: eager steady layer count differs")
        for name, segment in (("first", first), ("remainder", remainder)):
            require(segment.get("actual_h2d_bytes") == 0, f"{label}: {name} remote decode copied H2D")
            require(segment.get("lifecycle_state_counts") == {"remote": int(segment["forward_count"])}, f"{label}: {name} state differs")
            require(segment.get("copy_submitted_counts") == {"false": int(segment["forward_count"])}, f"{label}: {name} copy submitted")


def validate_graph_steps(result: dict[str, Any], arm: str, label: str) -> int:
    steps = result.get("steps")
    require(isinstance(steps, dict), f"{label}: steps absent")
    all_decode = [steps.get("first_decode")]
    all_decode += list(steps.get("warmup_decode", []))
    all_decode += list(steps.get("measured_decode", []))
    require(all(isinstance(item, dict) for item in all_decode), f"{label}: bad decode step")
    replay_count = 0
    for index, step in enumerate(all_decode):
        delta = step.get("graph_wrapper_delta", {})
        require(isinstance(delta, dict), f"{label}: step {index} Graph delta absent")
        full = int(delta.get("full_replay", 0))
        other = sum(int(value) for key, value in delta.items() if str(key).endswith("_replay") and key != "full_replay")
        if arm in GRAPH_ARMS:
            require(full == 1 and other == 0, f"{label}: step {index} did not use one FULL replay")
        else:
            require(full == 0 and other == 0, f"{label}: eager step replayed a Graph")
        replay_count += full
    return replay_count


def validate_allocator_log(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"{label}: UVM allocator log absent")
    contents = path.read_text(errors="replace")
    require(contents.count("vLLM UVM Allocator Session Started") == 1, f"{label}: allocator session count differs")
    allocations = [
        (int(match.group(1)), int(match.group(4)))
        for match in re.finditer(r"TRACE_ALLOC alloc_id=(\d+) ptr=(0x[0-9a-f]+) end=(0x[0-9a-f]+) size_bytes=(\d+)", contents)
    ]
    require(len(allocations) == 12, f"{label}: expected eight weights and four uploads")
    require([item[0] for item in allocations] == list(range(1, 13)), f"{label}: allocator ids differ")
    return {"allocation_count": len(allocations), "allocation_bytes": sum(item[1] for item in allocations)}


def validate_case(
    directory: Path,
    row: dict[str, str],
    run_config: dict[str, Any],
) -> dict[str, Any]:
    label = directory.name
    arm = str(row["symbol"])
    batch = int(row["batch"])
    kind = str(row["kind"])
    require(arm in ARM_NAMES, f"{label}: unknown arm")
    result = load_json(directory / "result.json")
    require(result.get("schema") == "vllm_uvm_prefill_alias_decode_remote_steps_v1", f"{label}: schema differs")
    require(result.get("status") == "ok", f"{label}: benchmark failed")
    require(result.get("arm") == arm and result.get("arm_name") == ARM_NAMES[arm], f"{label}: arm differs")
    config = result.get("config")
    require(isinstance(config, dict), f"{label}: config absent")
    require(config.get("batch_size") == batch, f"{label}: batch differs")
    require(config.get("phase_mode") == PHASE_MODE[arm], f"{label}: phase mode differs")
    require(config.get("audit_trace") is (kind == "audit"), f"{label}: trace/kind differs")
    require(config.get("graph_decode") is (arm in GRAPH_ARMS), f"{label}: Graph flag differs")
    require(config.get("cudagraph_capture_sizes") == [1, 4], f"{label}: capture sizes differ")
    require(config.get("max_num_seqs") == 4 and config.get("max_num_batched_tokens") == 2048, f"{label}: scheduler geometry differs")

    env = load_env(directory / "environment.txt")
    expected_env = {
        "VLLM_CPU_OFFLOAD_BACKEND": "copy",
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS": "1",
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND": "uvm_original_va",
        "VLLM_UVM_EXPERT_WEIGHTS_MANAGED": "1",
        "VLLM_UVM_ORIGINAL_VA_PHASE_MODE": PHASE_MODE[arm],
        "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE": "0",
        "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE": "0",
        "VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE": "0",
        "VLLM_CPU_OFFLOAD_COPY_TRACE": "1" if kind == "audit" else "0",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    }
    for key, expected in expected_env.items():
        require(env.get(key) == expected, f"{label}: environment {key} differs")
    require(
        env.get("VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE")
        == str((directory / "phase_transition_result.json").resolve()),
        f"{label}: transition result path differs",
    )

    for state_file in ("gpu.before.json", "gpu.after.json"):
        state = load_json(directory / state_file)
        require(state.get("gpu_clean") is True, f"{label}: {state_file} is dirty")
    kernel_log = directory / "kernel.since.log"
    require(kernel_log.is_file(), f"{label}: kernel log absent")
    require(
        (directory / "kernel_log_returncode.txt").read_text().strip() == "0",
        f"{label}: kernel log collection failed",
    )
    require("manual_migration_fault" not in kernel_log.read_text(errors="replace"), f"{label}: mode-2 fault observed")

    vas = validate_parameter_snapshots(result, label)
    diagnostics = result.get("diagnostics")
    require(isinstance(diagnostics, dict), f"{label}: diagnostics absent")
    initial_state = "sealed" if arm in HYBRID_ARMS else "remote"
    _state(diagnostics["after_load"], f"{label}/load", initial_state, PHASE_MODE[arm], vas)
    _state(diagnostics["after_prefill"], f"{label}/prefill", initial_state, PHASE_MODE[arm], vas)
    _state(diagnostics["after_first_decode"], f"{label}/first", "remote", PHASE_MODE[arm], vas)
    _state(diagnostics["final"], f"{label}/final", "remote", PHASE_MODE[arm], vas)

    final_model = diagnostics["final"]["model"]
    transitions = [
        item
        for item in final_model.get("phase_transition_history", [])
        if isinstance(item, dict)
        and item.get("event") == "cpu_offload_uvm_original_va_remote_transition"
    ]
    expected_count = 2 if arm == "G" else 1
    require(len(transitions) == expected_count, f"{label}: remote transition count differs")
    transition_timings = [
        validate_remote_transition(
            item,
            f"{label}/transition{index}",
            preserved_graph=(True if arm == "G" and index == len(transitions) - 1 else False),
        )
        for index, item in enumerate(transitions)
    ]
    if arm in GRAPH_ARMS:
        require(final_model.get("capture_topology") == "remote", f"{label}: Graph topology differs")
        require(final_model.get("remote_graph_captured") is True, f"{label}: remote Graph evidence absent")
        require(int(final_model.get("capture_success_count", 0)) >= 1, f"{label}: capture count absent")
        certificate = final_model.get("remote_graph_certificate")
        require(isinstance(certificate, dict), f"{label}: remote Graph certificate absent")
        require(
            certificate.get("schema")
            == "uvm_original_va_remote_graph_certificate_v1"
            and certificate.get("topology") == "remote"
            and certificate.get("phase_mode") == PHASE_MODE[arm],
            f"{label}: remote Graph certificate identity differs",
        )
        require(tuple(certificate.get("parameter_vas", [])) == vas, f"{label}: captured VA fingerprint differs")
        require(int(certificate.get("capture_forward_count", 0)) > 0, f"{label}: certificate has no selected forward")
        require(int(certificate.get("actual_h2d_bytes", -1)) == 0, f"{label}: certificate contains H2D")
        require(int(certificate.get("remote_map_requested_bytes", -1)) == LOGICAL_BYTES, f"{label}: certificate remote byte total differs")
        require(int(final_model.get("capture_actual_h2d_bytes", -1)) == 0, f"{label}: capture counter contains H2D")
    else:
        require(final_model.get("remote_graph_captured") is False, f"{label}: eager arm reports Graph")

    cutover = result.get("cutover_transition")
    if arm in HYBRID_ARMS:
        require(isinstance(cutover, dict), f"{label}: serving cutover absent")
        validate_remote_transition(cutover, f"{label}/serving_cutover", preserved_graph=(arm == "G"))
        phase_result = load_json(directory / "phase_transition_result.json")
        require(
            phase_result.get("schema")
            == "vllm_uvm_original_va_phase_transition_v1"
            and phase_result.get("ok") is True
            and phase_result.get("phase_mode") == "prefill_remote_decode"
            and phase_result.get("remote_graph_preserved") is (arm == "G"),
            f"{label}: external phase-transition result differs",
        )
        require(
            phase_result.get("transition", {}).get("parameter_vas_before")
            == phase_result.get("transition", {}).get("parameter_vas_after"),
            f"{label}: external transition reports a VA change",
        )
    else:
        require(cutover is None, f"{label}: remote-only arm has serving cutover")
        require(
            not (directory / "phase_transition_result.json").exists(),
            f"{label}: remote-only arm emitted a serving cutover result",
        )

    replay_count = validate_graph_steps(result, arm, label)
    validate_manifest_segments(result, arm, label)
    digest = output_digest(result.get("outputs"))
    require(digest == result.get("output_token_ids_sha256"), f"{label}: output digest differs")

    timing = result.get("timing")
    require(isinstance(timing, dict), f"{label}: timing absent")
    values = {
        "prefill_step_ms": finite(timing.get("prefill_step_ms"), f"{label}: prefill", positive=True),
        "first_decode_step_ms": finite(timing.get("first_decode_step_ms"), f"{label}: first", positive=True),
        "cutover_wall_ms": finite(timing.get("cutover_wall_ms"), f"{label}: cutover"),
        "first_decode_excluding_cutover_ms": finite(timing.get("first_decode_excluding_cutover_ms"), f"{label}: first-minus-cutover"),
        "steady_decode_step_ms": finite(timing.get("steady_decode_step_ms", {}).get("median"), f"{label}: steady", positive=True),
        "cycle_e2e_ms": finite(timing.get("cycle_e2e_ms"), f"{label}: e2e", positive=True),
    }
    if arm in HYBRID_ARMS:
        require(values["cutover_wall_ms"] > 0, f"{label}: cutover timing absent")
    else:
        require(values["cutover_wall_ms"] == 0, f"{label}: remote-only cutover timing nonzero")

    shutdown = result.get("shutdown_contract")
    require(isinstance(shutdown, dict) and not shutdown.get("cleanup_errors"), f"{label}: shutdown failed")
    shutdown_results = shutdown.get("model_runner_shutdown_results")
    require(isinstance(shutdown_results, list) and bool(shutdown_results), f"{label}: model-runner shutdown absent")
    expected_graphs = 2 if arm in GRAPH_ARMS else 0
    require(int(shutdown_results[-1].get("destroyed_graph_count", -1)) == expected_graphs, f"{label}: destroyed Graph count differs")
    post = shutdown.get("diagnostics_after_shutdown", {})
    require(post.get("driver", {}).get("session_count") == 0, f"{label}: driver session survived")
    require(post.get("model", {}).get("session_id") == 0, f"{label}: model session survived")
    require(not any(item.get("event") == "cpu_offload_uvm_original_va_phase_poisoned" for item in shutdown.get("manifest_final", {}).get("records", []) if isinstance(item, dict)), f"{label}: phase was poisoned")

    allocator = validate_allocator_log(directory / "uvm_allocations.log", label)
    return {
        "arm": arm,
        "batch": batch,
        "kind": kind,
        "sample": int(row["sample"]),
        "digest": digest,
        "timing": values,
        "replay_count": replay_count,
        "transition_timings": transition_timings,
        "allocator": allocator,
    }


def validate_module_cycle(run_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    before = load_json(run_root / "module.before.json")
    custom = load_json(run_root / "module.custom.json")
    after = load_json(run_root / "module.after.json")
    require(before.get("loaded_srcversion") == config.get("stock_module_srcversion"), "initial module is not stock")
    require(custom.get("loaded_srcversion") == config.get("custom_module_srcversion"), "custom module not loaded")
    require(after.get("loaded_srcversion") == config.get("stock_module_srcversion"), "stock module not restored")
    require(all(item.get("gpu_clean") is True for item in (before, custom, after)), "module cycle GPU state dirty")
    require(
        before.get("module_sha256") == config.get("stock_module_sha256")
        and after.get("module_sha256") == config.get("stock_module_sha256"),
        "stock module hash changed",
    )
    require(
        custom.get("module_sha256") == config.get("custom_module_sha256"),
        "custom module hash differs",
    )
    require(custom.get("manual_migration_fault_mode") == 2, "manual migration mode is not fail-fast 2")
    require(int(custom.get("slot_budget_bytes", 0)) >= DUAL_BANK_BYTES, "slot budget too small")
    return {"before": before, "custom": custom, "after": after}


def aggregate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[(case["arm"], case["batch"])].append(case)
    result: dict[str, Any] = {}
    for (arm, batch), values in sorted(grouped.items()):
        target = result.setdefault(arm, {}).setdefault(str(batch), {})
        for metric in (
            "prefill_step_ms",
            "first_decode_step_ms",
            "cutover_wall_ms",
            "first_decode_excluding_cutover_ms",
            "steady_decode_step_ms",
            "cycle_e2e_ms",
        ):
            samples = [float(item["timing"][metric]) for item in values]
            target[metric] = {
                "process_count": len(samples),
                "samples": samples,
                "median": statistics.median(samples),
                "mean": statistics.fmean(samples),
                "min": min(samples),
                "max": max(samples),
            }
    return result


def ratio(aggregate: dict[str, Any], left: str, right: str, batch: str, metric: str) -> dict[str, float]:
    left_ms = float(aggregate[left][batch][metric]["median"])
    right_ms = float(aggregate[right][batch][metric]["median"])
    return {
        f"{left}_ms": left_ms,
        f"{right}_ms": right_ms,
        f"{left}_over_{right}_ratio": left_ms / right_ms,
        f"{left}_vs_{right}_change_percent": 100.0 * (left_ms - right_ms) / right_ms,
    }


def verify(run_root: Path) -> dict[str, Any]:
    input_hashes = validate_input_hashes(run_root / "input_sha256.txt")
    config = load_json(run_root / "run_config.json")
    require(config.get("schema") == "vllm_uvm_prefill_alias_decode_remote_graph_run_v1", "run schema differs")
    module_cycle = validate_module_cycle(run_root, config)
    status = read_tsv(run_root / "execution_status.tsv")
    plan = read_tsv(run_root / "case_plan.tsv")
    require(len(status) == len(plan) and bool(status), "case/status count differs")
    cases: list[dict[str, Any]] = []
    hashes: dict[int, set[str]] = defaultdict(set)
    counts: Counter[tuple[str, int, str]] = Counter()
    for index, (row, expected) in enumerate(zip(status, plan), 1):
        for key in ("ordinal", "kind", "sample", "batch", "symbol"):
            require(row.get(key) == expected.get(key), f"row {index}: plan differs at {key}")
        require(row.get("returncode") == "0", f"row {index}: process failed")
        directory = run_root / str(row["directory"])
        for filename in ("command.txt", "environment.txt", "result.json", "gpu.before.json", "gpu.after.json"):
            require((directory / filename).is_file(), f"{directory.name}: missing {filename}")
        case = validate_case(directory, row, config)
        cases.append(case)
        hashes[case["batch"]].add(case["digest"])
        counts[(case["arm"], case["batch"], case["kind"])] += 1
    require(all(len(values) == 1 for values in hashes.values()), "output hashes differ across arms/processes")

    expected_batches = {int(value) for value in config.get("batch_sizes", [])}
    for arm in ARM_NAMES:
        for batch in expected_batches:
            require(counts[(arm, batch, "audit")] >= 1, f"missing audit case {arm}{batch}")
            expected_perf = int(config.get("performance_repeats", 0))
            require(counts[(arm, batch, "performance")] == expected_perf, f"performance count differs for {arm}{batch}")

    performance = [case for case in cases if case["kind"] == "performance"]
    audit = [case for case in cases if case["kind"] == "audit"]
    aggregate_audit = aggregate_cases(audit)
    aggregate_performance = aggregate_cases(performance) if performance else {}
    comparisons: dict[str, Any] = {}
    if aggregate_performance:
        for batch in sorted({str(value) for value in expected_batches}):
            comparisons[batch] = {
                "prefill_alias_vs_remote_eager": ratio(aggregate_performance, "H", "R", batch, "prefill_step_ms"),
                "prefill_alias_vs_remote_graph_config": ratio(aggregate_performance, "G", "C", batch, "prefill_step_ms"),
                "hybrid_remote_decode_graph_vs_eager": ratio(aggregate_performance, "G", "H", batch, "steady_decode_step_ms"),
                "remote_only_decode_graph_vs_eager": ratio(aggregate_performance, "C", "R", batch, "steady_decode_step_ms"),
                "hybrid_first_decode_graph_vs_eager_raw": ratio(aggregate_performance, "G", "H", batch, "first_decode_step_ms"),
                "hybrid_first_decode_graph_vs_eager_without_cutover": ratio(aggregate_performance, "G", "H", batch, "first_decode_excluding_cutover_ms"),
                "remote_only_first_decode_graph_vs_eager": ratio(aggregate_performance, "C", "R", batch, "first_decode_step_ms"),
            }
    return {
        "schema": "vllm_uvm_prefill_alias_decode_remote_graph_verification_v1",
        "status": "ok",
        "run_root": str(run_root.resolve()),
        "mode": config.get("mode"),
        "case_count": len(cases),
        "counts": {f"{arm}{batch}/{kind}": value for (arm, batch, kind), value in sorted(counts.items())},
        "output_hashes": {str(batch): next(iter(values)) for batch, values in sorted(hashes.items())},
        "module_cycle": module_cycle,
        "input_hashes": input_hashes,
        "aggregate_audit": aggregate_audit,
        "aggregate_performance": aggregate_performance,
        "comparisons": comparisons,
        "performance_is_acceptance_gate": False,
        "correctness_is_acceptance_gate": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_root / "verification.json"
    try:
        result = verify(args.run_root)
    except Exception as exc:
        result = {
            "schema": "vllm_uvm_prefill_alias_decode_remote_graph_verification_v1",
            "status": "failed",
            "run_root": str(args.run_root.resolve()),
            "error": repr(exc),
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
