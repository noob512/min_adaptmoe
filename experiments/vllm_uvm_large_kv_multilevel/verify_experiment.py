#!/usr/bin/env python3
"""Verify real-HBM 2/4/8/10 GiB fixed-VA KV path qualification."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


BLOCK_BYTES = 1_572_864
MAX_BYTES = 10_737_942_528
TARGETS = {
    "2g": 1_364,
    "4g": 2_730,
    "8g": 5_460,
    "10g": 6_826,
}
EXPERT_WAVES = {
    "2g": 0,
    "4g": 10,
    "8g": 29,
    "10g": 38,
}
BASE_EXPERTS_PER_LAYER = 70
EXPERT_MAIN_BYTES = 4_718_592
EXPERT_LAYER_COUNT = 48
FULL_PARAMETER_BYTES = 28_991_029_248
SELECTED_PARAMETER_BYTES = 15_854_469_120
HOT_PARAMETER_BYTES = FULL_PARAMETER_BYTES - SELECTED_PARAMETER_BYTES
TOTAL_MODEL_BYTES = 32_444_792_832
RESIDENT_MODEL_BYTES = TOTAL_MODEL_BYTES - SELECTED_PARAMETER_BYTES
EXPECTED_BACKING_BYTES = 660_602_880
EXPECTED_RELEASE_REGION_COUNT = EXPERT_LAYER_COUNT * BASE_EXPERTS_PER_LAYER * 2
GRAPH_WORKSPACE_RESERVE_BYTES = 512 * 2**20
MIN_SAFETY_HEADROOM_BYTES = 2 * 2**30
PREFILL_TOLERANCE_PERCENT = 15.0
DECODE_TOLERANCE_PERCENT = 10.0
DECODE_P95_TOLERANCE_PERCENT = 15.0
BAD_KERNEL = re.compile(
    r"NVRM:\s*Xid|FAULT_PDE|manual[_ -]migration[_ -]fault|"
    r"\bOops\b|\bBUG:|kernel panic|general protection fault|Call Trace",
    re.IGNORECASE,
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def verify_shutdown(result: dict[str, Any], label: str) -> None:
    shutdown = result.get("shutdown_contract") or {}
    require(not shutdown.get("cleanup_errors"), f"{label}: cleanup errors")
    require(
        shutdown.get("controlled_hard_exit_after_cleanup") is True,
        f"{label}: controlled cleanup",
    )
    diagnostics = shutdown.get("diagnostics_after_shutdown") or {}
    model = diagnostics.get("model") or {}
    driver = diagnostics.get("driver") or {}
    require(int(model.get("session_id", -1)) == 0, f"{label}: model session")
    require(
        int(model.get("active_driver_backing_bytes", -1)) == 0,
        f"{label}: model backing",
    )
    require(int(driver.get("session_count", -1)) == 0, f"{label}: driver session")


def expected_gpu_bytes(target: int) -> int:
    # target excludes vLLM's permanently reserved null block.
    return (int(target) + 1) * BLOCK_BYTES


def physical_ok(snapshot: dict[str, Any], target: int, label: str) -> None:
    require(int(snapshot.get("reserved_va_bytes", -1)) == MAX_BYTES, f"{label}: VA bytes")
    require(int(snapshot.get("total_blocks", -1)) == TARGETS["10g"], f"{label}: max blocks")
    require(int(snapshot.get("target_blocks", -1)) == target, f"{label}: target")
    for part in ("full", "head", "tail"):
        value = snapshot[part]
        require(value.get("ok") is True, f"{label}: {part} query")
        require(int(value.get("unknown_bytes", -1)) == 0, f"{label}: {part} unknown")
    full = snapshot["full"]
    head = snapshot["head"]
    tail = snapshot["tail"]
    gpu_bytes = expected_gpu_bytes(target)
    require(
        int(snapshot.get("covered_bytes", -1)) == MAX_BYTES
        and int(snapshot.get("span_count", -1)) > 0
        and bool(snapshot.get("va_sha256")),
        f"{label}: incomplete fixed-VA inventory",
    )
    require(int(full["requested_bytes"]) == MAX_BYTES, f"{label}: full bytes")
    require(
        int(full["gpu_bytes"]) == gpu_bytes
        and int(full["cpu_bytes"]) == MAX_BYTES - gpu_bytes,
        f"{label}: full placement",
    )
    require(
        int(full.get("queried_bytes", -1)) == MAX_BYTES
        and int(full.get("any_bytes", -1)) == MAX_BYTES,
        f"{label}: incomplete full residency audit",
    )
    require(
        int(head["requested_bytes"]) == gpu_bytes
        and int(head["gpu_bytes"]) == gpu_bytes
        and int(head["cpu_bytes"]) == 0,
        f"{label}: GPU prefix",
    )
    require(
        int(head.get("queried_bytes", -1)) == gpu_bytes
        and int(head.get("any_bytes", -1)) == gpu_bytes,
        f"{label}: incomplete GPU-prefix audit",
    )
    require(
        int(tail["requested_bytes"]) == MAX_BYTES - gpu_bytes
        and int(tail["cpu_bytes"]) == MAX_BYTES - gpu_bytes
        and int(tail["gpu_bytes"]) == 0,
        f"{label}: CPU tail",
    )
    require(
        int(tail.get("queried_bytes", -1)) == MAX_BYTES - gpu_bytes
        and int(tail.get("any_bytes", -1)) == MAX_BYTES - gpu_bytes,
        f"{label}: incomplete CPU-tail audit",
    )


def request_remote_transition(record: dict[str, Any], label: str) -> dict[str, Any]:
    before = (
        record.get("diagnostics_before_request", {})
        .get("model", {})
        .get("phase_transition_history", [])
    )
    after = (
        record.get("diagnostics_after_request", {})
        .get("model", {})
        .get("phase_transition_history", [])
    )
    require(
        isinstance(before, list)
        and isinstance(after, list)
        and len(after) >= len(before)
        and after[: len(before)] == before,
        f"{label}: phase-transition history is not append-only",
    )
    transitions = after[len(before) :]
    resumes = [
        item
        for item in transitions
        if item.get("event")
        == "cpu_offload_uvm_original_va_alias_prefill_activated"
        and item.get("post_state") == "sealed"
        and item.get("ok") is True
    ]
    parks = [
        item
        for item in transitions
        if item.get("event") == "cpu_offload_uvm_original_va_remote_transition"
        and item.get("post_state") == "remote_parked"
        and item.get("ok") is True
    ]
    require(len(resumes) == 1, f"{label}: Prefill RESUME count")
    require(len(parks) == 1, f"{label}: Decode PARK count")
    require(
        transitions.index(resumes[0]) < transitions.index(parks[0]),
        f"{label}: PARK preceded Prefill RESUME",
    )
    return parks[0]


def runtime_remote_bytes(expert_waves: int) -> int:
    return int(expert_waves) * EXPERT_LAYER_COUNT * EXPERT_MAIN_BYTES


def verify_incremental_expert_state(
    snapshot: dict[str, Any],
    *,
    expert_waves: int,
    label: str,
) -> None:
    require(
        snapshot.get("schema") == "vllm_uvm_incremental_expert_state_v1",
        f"{label}: incremental expert-state schema",
    )
    require(
        int(snapshot.get("layer_count", -1)) == EXPERT_LAYER_COUNT,
        f"{label}: incremental layer count",
    )
    require(
        int(snapshot.get("remote_expert_count", -1))
        == int(expert_waves) * EXPERT_LAYER_COUNT
        and int(snapshot.get("offloaded_expert_count", -1)) == 0,
        f"{label}: incremental expert count",
    )
    require(
        int(snapshot.get("remote_region_bytes", -1))
        == runtime_remote_bytes(expert_waves)
        and int(snapshot.get("offloaded_region_bytes", -1)) == 0,
        f"{label}: incremental expert bytes",
    )
    layers = snapshot.get("layers") or {}
    require(len(layers) == EXPERT_LAYER_COUNT, f"{label}: layer inventory")
    for layer_name, state in layers.items():
        registered_count = int(state.get("registered_expert_count", -1))
        require(
            len(state.get("remote_expert_ids") or []) == int(expert_waves)
            and not (state.get("offloaded_expert_ids") or [])
            and (
                registered_count == 128
                or (int(expert_waves) == 0 and registered_count == 0)
            )
            and state.get("access_mode") == "remote_read",
            f"{label}: unbalanced layer state {layer_name}",
        )


def verify_expert_residency(
    record: dict[str, Any],
    label: str,
    *,
    expert_waves: int,
) -> None:
    model = record["diagnostics_after_request"]["model"]
    require(model.get("lifecycle_state") == "remote_parked", f"{label}: lifecycle")
    require(
        int(model.get("selected_parameter_bytes_per_forward", -1))
        == SELECTED_PARAMETER_BYTES,
        f"{label}: selected expert bytes",
    )
    transition = request_remote_transition(record, label)
    cold = transition.get("physical_residency") or {}
    hot = transition.get("hot_physical_residency") or {}
    incremental_bytes = runtime_remote_bytes(expert_waves)
    require(
        cold.get("ok") is True
        and int(cold.get("queried_bytes", -1)) == SELECTED_PARAMETER_BYTES
        and int(cold.get("cpu_bytes", -1)) == SELECTED_PARAMETER_BYTES
        and int(cold.get("gpu_bytes", -1)) == 0
        and int(cold.get("unknown_bytes", -1)) == 0,
        f"{label}: cold expert residency",
    )
    require(
        hot.get("ok") is True
        and int(hot.get("queried_bytes", -1)) == HOT_PARAMETER_BYTES
        and int(hot.get("gpu_bytes", -1))
        == HOT_PARAMETER_BYTES - incremental_bytes
        and int(hot.get("cpu_bytes", -1)) == incremental_bytes
        and int(hot.get("unknown_bytes", -1)) == 0,
        f"{label}: tier-specific hot/runtime-remote residency",
    )
    require(
        int(transition.get("runtime_remote_bytes", -1)) == incremental_bytes
        and int(transition.get("hot_runtime_remote_bytes", -1))
        == incremental_bytes
        and int(transition.get("hot_expected_gpu_bytes", -1))
        == HOT_PARAMETER_BYTES - incremental_bytes,
        f"{label}: runtime remote transition inventory",
    )


def verify_record(
    record: dict[str, Any],
    *,
    label: str,
    old_target: int,
    target: int,
    expected_va: str,
    expected_parameter_vas: list[int],
    expected_source_vas: list[int],
    old_expert_waves: int,
    expert_waves: int,
) -> None:
    require(record.get("label") == label, f"{label}: record label")
    pressure = record.get("pressure") or {}
    require(
        pressure.get("schema") == "vllm_uvm_explicit_physical_kv_target_v1"
        and pressure.get("decision_source")
        == "explicit_multilevel_physical_kv_qualification",
        f"{label}: explicit decision source",
    )
    require(
        pressure.get("previous_request_finished") is True
        and int(pressure.get("scheduler_unfinished_request_count", -1)) == 0
        and int(pressure.get("queued_batch_count", -1)) == 0,
        f"{label}: unsafe resize boundary",
    )
    require(
        int(pressure.get("old_target_blocks", -1)) == old_target
        and int(pressure.get("new_target_blocks", -1)) == target
        and pressure.get("physical_target_changed") is (target != old_target),
        f"{label}: target transition",
    )
    require(record.get("resize", {}).get("ok") is True, f"{label}: resize")
    expected_resize_reason = "applied" if target != old_target else "already_at_target"
    require(
        record.get("resize", {}).get("reason") == expected_resize_reason,
        f"{label}: resize result reason",
    )
    physical_ok(record["physical_before_resize"], old_target, f"{label}: before")
    physical_ok(record["physical_after_resize"], target, f"{label}: resized")
    physical_ok(record["physical_after_request"], target, f"{label}: served")
    for key in ("physical_before_resize", "physical_after_resize", "physical_after_request"):
        require(record[key].get("va_sha256") == expected_va, f"{label}: KV VA")

    graph_before = record["graph_wrapper_calls_before_resize"]
    graph_resize = record["graph_wrapper_calls_after_resize"]
    graph_request = record["graph_wrapper_calls_after_request"]
    require(
        int(graph_before.get("full_capture", -1))
        == int(graph_resize.get("full_capture", -2))
        == int(graph_request.get("full_capture", -3)),
        f"{label}: Graph recapture",
    )
    counter_before = record["compilation_counter_before_resize"]
    counter_resize = record["compilation_counter_after_resize"]
    counter_request = record["compilation_counter_after_request"]
    require(
        int(counter_before["num_cudagraph_captured"])
        == int(counter_resize["num_cudagraph_captured"])
        == int(counter_request["num_cudagraph_captured"]),
        f"{label}: capture counter",
    )
    require(
        int(record["prefill"]["graph_wrapper_delta"].get("full_replay", 0)) == 1,
        f"{label}: Prefill Graph replay",
    )
    require(
        int(record["first_decode"]["graph_wrapper_delta"].get("full_replay", 0)) == 1,
        f"{label}: first Decode Graph replay",
    )
    require(len(record["warmup_decode"]) == 5, f"{label}: warmup count")
    require(len(record["measured_decode"]) == 16, f"{label}: measured count")
    require(
        all(
            int(step["graph_wrapper_delta"].get("full_replay", 0)) == 1
            for step in record["warmup_decode"] + record["measured_decode"]
        ),
        f"{label}: Decode Graph replay",
    )
    parameter = record["parameter_va_snapshots"]
    require(
        parameter["before_request"]["vas"] == parameter["after_request"]["vas"],
        f"{label}: Parameter VA",
    )
    require(
        parameter["before_request"]["vas"] == expected_parameter_vas,
        f"{label}: Parameter VA changed from load-time identity",
    )
    model_before = record["diagnostics_before_request"]["model"]
    model_after = record["diagnostics_after_request"]["model"]
    require(
        int(model_before.get("session_id", 0)) > 0
        and model_before.get("session_id") == model_after.get("session_id"),
        f"{label}: alias session identity changed",
    )
    require(
        model_before.get("source_vas") == model_after.get("source_vas")
        == expected_source_vas
        and model_before.get("original_vas") == model_after.get("original_vas")
        == expected_parameter_vas
        and model_before.get("alias_graph_captured") is True
        and model_before.get("remote_graph_captured") is True
        and model_after.get("alias_graph_captured") is True
        and model_after.get("remote_graph_captured") is True
        and model_after.get("alias_graphs_destroyed") is False,
        f"{label}: stable Graph/VA certificate changed",
    )
    require(
        int(record.get("target_expert_waves", -1)) == int(expert_waves),
        f"{label}: expert wave target",
    )
    verify_incremental_expert_state(
        record.get("expert_before_resize") or {},
        expert_waves=old_expert_waves,
        label=f"{label}: expert before resize",
    )
    verify_incremental_expert_state(
        record.get("expert_after_resize") or {},
        expert_waves=expert_waves,
        label=f"{label}: expert after resize",
    )
    verify_incremental_expert_state(
        record.get("expert_after_request") or {},
        expert_waves=expert_waves,
        label=f"{label}: expert after request",
    )
    before_transitions = record.get("expert_transitions_before_kv") or []
    after_transitions = record.get("expert_transitions_after_kv") or []
    difference = abs(int(expert_waves) - int(old_expert_waves))
    if expert_waves > old_expert_waves:
        require(
            len(before_transitions) == difference and not after_transitions,
            f"{label}: grow must release experts before KV migration",
        )
        expected_operation = "release"
        transitions = before_transitions
    elif expert_waves < old_expert_waves:
        require(
            not before_transitions and len(after_transitions) == difference,
            f"{label}: shrink must restore experts after KV migration",
        )
        expected_operation = "restore"
        transitions = after_transitions
    else:
        require(
            not before_transitions and not after_transitions,
            f"{label}: unchanged tier unexpectedly moved experts",
        )
        expected_operation = ""
        transitions = []
    require(
        all(
            item.get("operation") == expected_operation
            and int((item.get("result") or {}).get(
                "released_expert_count"
                if expected_operation == "release"
                else "restored_expert_count",
                -1,
            ))
            == EXPERT_LAYER_COUNT
            for item in transitions
        ),
        f"{label}: incomplete all-layer expert wave",
    )
    verify_expert_residency(record, label, expert_waves=expert_waves)
    outputs = record.get("outputs") or []
    expected_output_tokens = int(pressure.get("next_max_output_tokens_per_request", -1))
    require(
        len(outputs) == 1
        and outputs[0].get("finished") is True
        and len(outputs[0].get("output_token_ids") or []) == expected_output_tokens
        and int(outputs[0].get("prompt_tokens", -1))
        == int(pressure.get("next_prompt_tokens", -2)),
        f"{label}: output",
    )
    require(
        record.get("shared_prompt_matches_initial_batch") is True,
        f"{label}: shared-prompt output changed",
    )
    resize_ms = float(record.get("resize_elapsed_ms", -1))
    require(math.isfinite(resize_ms) and resize_ms >= 0, f"{label}: resize timing")


def prefill_compute_ms(record: dict[str, Any]) -> float:
    before = (
        record.get("diagnostics_before_request", {})
        .get("model", {})
        .get("phase_transition_history", [])
    )
    after = (
        record.get("diagnostics_after_request", {})
        .get("model", {})
        .get("phase_transition_history", [])
    )
    require(len(after) >= len(before) and after[: len(before)] == before, "transition prefix")
    resumes = [
        item
        for item in after[len(before) :]
        if item.get("event") == "cpu_offload_uvm_original_va_alias_prefill_activated"
        and item.get("ok") is True
    ]
    require(len(resumes) == 1, "prefill RESUME count")
    value = float(record["prefill"]["elapsed_ms"]) - float(resumes[0]["wall_ns"]) / 1e6
    require(math.isfinite(value) and value > 0, "prefill compute timing")
    return value


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    require(bool(ordered), "empty percentile")
    return ordered[max(math.ceil(len(ordered) * quantile) - 1, 0)]


def timing(records: list[dict[str, Any]]) -> dict[str, float | int]:
    prefill = [prefill_compute_ms(record) for record in records]
    decode = [
        float(step["elapsed_ms"])
        for record in records
        for step in record["measured_decode"]
    ]
    require(len(prefill) == 3 and len(decode) == 48, "timing cardinality")
    require(
        all(math.isfinite(value) and value > 0 for value in prefill + decode),
        "invalid timing sample",
    )
    decode_median = statistics.median(decode)
    return {
        "prefill_samples": len(prefill),
        "decode_samples": len(decode),
        "prefill_excluding_resume_median_ms": statistics.median(prefill),
        "decode_median_ms": decode_median,
        "decode_p95_ms": percentile(decode, 0.95),
        "decode_tokens_per_second": 1000.0 / decode_median,
    }


def delta_percent(path: float, direct: float) -> float:
    return (path / direct - 1.0) * 100.0


def verify_graph_contract(result: dict[str, Any], label: str) -> None:
    config = result.get("config") or {}
    require(
        config.get("graph_prefill") is True
        and config.get("graph_decode") is True
        and config.get("execution_mode") == "full_graph_separate",
        f"{label}: FULL Prefill/Decode Graph mode",
    )
    capture_sizes = {int(value) for value in config.get("cudagraph_capture_sizes", [])}
    require(1 in capture_sizes, f"{label}: Decode capture size")
    after_load = result.get("graph_wrapper_calls_after_load") or {}
    counter = result.get("compilation_counter_after_load") or {}
    require(
        int(after_load.get("full_capture", 0)) >= 2
        and int(counter.get("num_cudagraph_captured", 0)) >= 2,
        f"{label}: both FULL Graphs were not captured",
    )
    model = result["diagnostics"]["after_load"]["model"]
    certificate = model.get("dual_graph_partition_certificate") or {}
    require(
        model.get("alias_graph_captured") is True
        and model.get("remote_graph_captured") is True
        and model.get("alias_graphs_destroyed") is False
        and int(certificate.get("alias_entry_count", 0)) >= 1
        and int(certificate.get("remote_entry_count", 0)) >= 1,
        f"{label}: dual-Graph certificate",
    )
    parameter_vas = result["parameter_va_snapshots"]["after_load"]["vas"]
    require(
        model.get("original_vas") == parameter_vas
        and certificate.get("parameter_vas") == parameter_vas
        and len(model.get("source_vas") or []) == len(parameter_vas),
        f"{label}: captured Graph VA certificate",
    )


def transition_summary(
    record: dict[str, Any], old_target: int, target: int
) -> dict[str, float | int | str]:
    moved = abs(expected_gpu_bytes(target) - expected_gpu_bytes(old_target))
    elapsed_ms = float(record["resize_elapsed_ms"])
    return {
        "label": str(record["label"]),
        "old_target_blocks": old_target,
        "new_target_blocks": target,
        "moved_bytes": moved,
        "resize_elapsed_ms": elapsed_ms,
        "effective_gib_per_second": (
            moved / 2**30 / (elapsed_ms / 1000.0) if moved and elapsed_ms > 0 else 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_root.resolve()
    labels = ("stepped", "direct_2g", "direct_4g", "direct_8g", "direct_10g")
    results = {label: load_json(root / label / "result.json") for label in labels}

    hardware = (root / "hardware.csv").read_text(encoding="utf-8").strip().split(",")
    require(len(hardware) == 3, "hardware record")
    device_name = hardware[0].strip()
    nominal_device_memory_bytes = int(hardware[1].strip()) * 2**20
    driver_version = hardware[2].strip()

    torch_total_bytes = {
        int(result.get("cuda_memory_after_load", {}).get("total_bytes", -1))
        for result in results.values()
    }
    require(
        len(torch_total_bytes) == 1 and next(iter(torch_total_bytes)) > 0,
        "Torch usable device capacity differs across cases",
    )
    usable_device_memory_bytes = next(iter(torch_total_bytes))
    backing_bytes = {
        int(
            result.get("diagnostics", {})
            .get("after_load", {})
            .get("model", {})
            .get("backing_bytes", -1)
        )
        for result in results.values()
    }
    require(
        backing_bytes == {EXPECTED_BACKING_BYTES},
        "static alias backing differs from the uniform70 base budget",
    )
    budget_headroom = (
        usable_device_memory_bytes
        - RESIDENT_MODEL_BYTES
        - MAX_BYTES
        - EXPECTED_BACKING_BYTES
        - GRAPH_WORKSPACE_RESERVE_BYTES
    )
    require(
        budget_headroom >= MIN_SAFETY_HEADROOM_BYTES,
        "10 GiB tier violates the conservative Torch HBM budget",
    )

    for label, result in results.items():
        require(
            (root / label / "returncode.txt").read_text().strip() == "0",
            f"{label}: runner return code",
        )
        require(result.get("status") == "ok", f"{label}: runner status")
        require(result.get("arm") == "D", f"{label}: Graph arm")
        config = result.get("config") or {}
        require(int(config.get("kv_cache_memory_bytes", -1)) == MAX_BYTES, f"{label}: KV bytes")
        require(config.get("allow_explicit_dynamic_kv_targets") is True, f"{label}: explicit flag")
        require(
            int(config.get("expected_selected_bytes", -1)) == SELECTED_PARAMETER_BYTES,
            f"{label}: plan bytes",
        )
        require(
            int(config.get("incremental_expert_wave_count", -1)) == 38
            and bool(config.get("incremental_expert_wave_plan_sha256")),
            f"{label}: incremental wave plan",
        )
        environment = result.get("environment") or {}
        require(
            str(environment.get("VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE")) == "0"
            and str(
                environment.get("VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE")
            )
            == "1"
            and environment.get("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE")
            == "remote_read",
            f"{label}: Stage-M/runtime-wave separation",
        )
        original_release = (
            result.get("diagnostics", {})
            .get("after_load", {})
            .get("model", {})
            .get("original_release_result", {})
        )
        require(
            original_release.get("ok") is True
            and original_release.get("required") is True
            and original_release.get("reason") == "released",
            f"{label}: original cold GPU chunks were not released before MAP",
        )
        require(
            int(original_release.get("region_count", -1))
            == EXPECTED_RELEASE_REGION_COUNT
            and int(original_release.get("requested_bytes", -1))
            == SELECTED_PARAMETER_BYTES
            and int(original_release.get("released_bytes", -1)) > 0
            and len(str(original_release.get("geometry_sha256", ""))) == 64,
            f"{label}: original cold release audit mismatch",
        )
        verify_graph_contract(result, label)
        verify_shutdown(result, label)
        physical_ok(result["initial_kv_physical"], TARGETS["2g"], f"{label}: initial")
        require(
            int(result["compilation_counter_after_load"]["num_cudagraph_captured"])
            == int(result["compilation_counter_final"]["num_cudagraph_captured"]),
            f"{label}: global Graph recapture",
        )

    expected_expert_plan = results["stepped"]["diagnostics"]["after_load"][
        "model"
    ].get("expert_plan")
    require(
        isinstance(expected_expert_plan, list)
        and len(expected_expert_plan) == 48
        and all(
            isinstance(item, list)
            and len(item) == 2
            and len(item[1]) == BASE_EXPERTS_PER_LAYER
            for item in expected_expert_plan
        ),
        "stepped: base expert plan is not uniform70",
    )
    for label, result in results.items():
        model = result["diagnostics"]["after_load"]["model"]
        require(model.get("expert_plan") == expected_expert_plan, f"{label}: expert plan")
        require(
            int(model.get("selected_expert_count", -1))
            == EXPERT_LAYER_COUNT * BASE_EXPERTS_PER_LAYER,
            f"{label}: selected expert count",
        )

    stepped = results["stepped"]["dynamic_kv_resizes"]
    require(len(stepped) == 18, "stepped record count")
    direct = {
        tier: results[f"direct_{tier}"]["dynamic_kv_resizes"]
        for tier in TARGETS
    }
    require(all(len(records) == 3 for records in direct.values()), "direct record count")

    stepped_groups: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    old_target = TARGETS["2g"]
    old_expert_waves = EXPERT_WAVES["2g"]
    transitions: list[dict[str, float | int | str]] = []
    for direction, tier in (
        ("ascending", "4g"),
        ("ascending", "8g"),
        ("ascending", "10g"),
        ("descending", "8g"),
        ("descending", "4g"),
        ("descending", "2g"),
    ):
        records = stepped[cursor : cursor + 3]
        cursor += 3
        target = TARGETS[tier]
        target_expert_waves = EXPERT_WAVES[tier]
        expected_va = results["stepped"]["initial_kv_physical"]["va_sha256"]
        expected_parameter_vas = results["stepped"]["parameter_va_snapshots"][
            "after_load"
        ]["vas"]
        expected_source_vas = results["stepped"]["diagnostics"]["after_load"][
            "model"
        ]["source_vas"]
        for index, record in enumerate(records, 1):
            expected_old = old_target if index == 1 else target
            expected_old_expert_waves = (
                old_expert_waves if index == 1 else target_expert_waves
            )
            verify_record(
                record,
                label=f"{direction}_{tier}_{index}",
                old_target=expected_old,
                target=target,
                expected_va=expected_va,
                expected_parameter_vas=expected_parameter_vas,
                expected_source_vas=expected_source_vas,
                old_expert_waves=expected_old_expert_waves,
                expert_waves=target_expert_waves,
            )
        transitions.append(transition_summary(records[0], old_target, target))
        old_target = target
        old_expert_waves = target_expert_waves
        stepped_groups[f"{direction}_{tier}"] = records

    for tier, records in direct.items():
        target = TARGETS[tier]
        target_expert_waves = EXPERT_WAVES[tier]
        expected_va = results[f"direct_{tier}"]["initial_kv_physical"]["va_sha256"]
        expected_parameter_vas = results[f"direct_{tier}"][
            "parameter_va_snapshots"
        ]["after_load"]["vas"]
        expected_source_vas = results[f"direct_{tier}"]["diagnostics"][
            "after_load"
        ]["model"]["source_vas"]
        for index, record in enumerate(records, 1):
            old = TARGETS["2g"] if index == 1 else target
            old_waves = EXPERT_WAVES["2g"] if index == 1 else target_expert_waves
            verify_record(
                record,
                label=f"direct_{tier}_{index}",
                old_target=old,
                target=target,
                expected_va=expected_va,
                expected_parameter_vas=expected_parameter_vas,
                expected_source_vas=expected_source_vas,
                old_expert_waves=old_waves,
                expert_waves=target_expert_waves,
            )

    comparisons: dict[str, dict[str, Any]] = {}
    timing_groups: dict[str, dict[str, float | int]] = {
        f"direct_{tier}": timing(records) for tier, records in direct.items()
    }
    for group, records in stepped_groups.items():
        tier = group.rsplit("_", 1)[1]
        path_timing = timing(records)
        direct_timing = timing_groups[f"direct_{tier}"]
        prefill_delta = delta_percent(
            float(path_timing["prefill_excluding_resume_median_ms"]),
            float(direct_timing["prefill_excluding_resume_median_ms"]),
        )
        decode_delta = delta_percent(
            float(path_timing["decode_median_ms"]),
            float(direct_timing["decode_median_ms"]),
        )
        decode_p95_delta = delta_percent(
            float(path_timing["decode_p95_ms"]),
            float(direct_timing["decode_p95_ms"]),
        )
        comparisons[group] = {
            "tier": tier,
            "path_timing": path_timing,
            "direct_timing": direct_timing,
            "prefill_delta_percent": prefill_delta,
            "decode_delta_percent": decode_delta,
            "decode_p95_delta_percent": decode_p95_delta,
            "within_tolerance": (
                abs(prefill_delta) <= PREFILL_TOLERANCE_PERCENT
                and abs(decode_delta) <= DECODE_TOLERANCE_PERCENT
                and abs(decode_p95_delta) <= DECODE_P95_TOLERANCE_PERCENT
            ),
        }
        timing_groups[group] = path_timing
        for path_record, direct_record in zip(records, direct[tier]):
            require(
                path_record["output_token_ids_sha256"]
                == direct_record["output_token_ids_sha256"],
                f"{group}: output differs from direct {tier}",
            )

    require(
        all(bool(value["within_tolerance"]) for value in comparisons.values()),
        "stepped/direct steady-state performance exceeds tolerance",
    )

    all_hashes = {
        record["output_token_ids_sha256"]
        for result in results.values()
        for record in result["dynamic_kv_resizes"]
    }
    require(len(all_hashes) == 1, "outputs differ across physical tiers")

    kernel_text = (root / "kernel.since.log").read_text(encoding="utf-8")
    require(BAD_KERNEL.search(kernel_text) is None, "kernel fault signature")
    for label in labels:
        gpu_after = load_json(root / label / "gpu.after.json")
        require(gpu_after.get("gpu_clean") is True, f"{label}: GPU not clean")
        require(not gpu_after.get("compute_pids"), f"{label}: compute clients")
        require(not gpu_after.get("uvm_client_pids"), f"{label}: UVM clients")
        uvm_log = root / label / "uvm.log"
        if uvm_log.is_file():
            require(
                BAD_KERNEL.search(uvm_log.read_text(encoding="utf-8")) is None,
                f"{label}: UVM fault signature",
            )
    before_module = load_json(root / "module.before.json")
    after_module = load_json(root / "module.after.json")
    require(
        after_module.get("loaded_srcversion") == before_module.get("loaded_srcversion")
        and after_module.get("module_sha256") == before_module.get("module_sha256")
        and after_module.get("gpu_clean") is True,
        "stock nvidia_uvm was not restored",
    )

    verification = {
        "schema": "vllm_uvm_large_kv_multilevel_verification_v1",
        "status": "ok",
        "device": {
            "name": device_name,
            "driver_version": driver_version,
            "nominal_memory_bytes": nominal_device_memory_bytes,
            "torch_usable_memory_bytes": usable_device_memory_bytes,
        },
        "hbm_budget": {
            "total_model_bytes": TOTAL_MODEL_BYTES,
            "full_parameter_bytes": FULL_PARAMETER_BYTES,
            "cold_parameter_bytes": SELECTED_PARAMETER_BYTES,
            "hot_parameter_bytes": HOT_PARAMETER_BYTES,
            "resident_model_bytes": RESIDENT_MODEL_BYTES,
            "maximum_kv_bytes": MAX_BYTES,
            "retained_alias_backing_bytes": EXPECTED_BACKING_BYTES,
            "graph_workspace_reserve_bytes": GRAPH_WORKSPACE_RESERVE_BYTES,
            "minimum_safety_headroom_bytes": MIN_SAFETY_HEADROOM_BYTES,
            "budget_headroom_bytes": budget_headroom,
        },
        "tiers": {
            tier: {
                "target_blocks": target,
                "gpu_resident_bytes_including_null": expected_gpu_bytes(target),
            }
            for tier, target in TARGETS.items()
        },
        "stepped_sequence": [
            "2g",
            "4g",
            "8g",
            "10g",
            "8g",
            "4g",
            "2g",
        ],
        "transitions": transitions,
        "timing": timing_groups,
        "path_parity": {
            "prefill_tolerance_percent": PREFILL_TOLERANCE_PERCENT,
            "decode_tolerance_percent": DECODE_TOLERANCE_PERCENT,
            "decode_p95_tolerance_percent": DECODE_P95_TOLERANCE_PERCENT,
            "comparisons": comparisons,
            "within_tolerance": all(
                bool(value["within_tolerance"])
                for value in comparisons.values()
            ),
        },
        "outputs_match": True,
        "base_expert_plan_uniform_per_layer": BASE_EXPERTS_PER_LAYER,
        "tier_expert_waves": EXPERT_WAVES,
        "tier_remote_experts_per_layer": {
            tier: BASE_EXPERTS_PER_LAYER + waves
            for tier, waves in EXPERT_WAVES.items()
        },
        "kv_va_stable_within_each_process": True,
        "parameter_va_stable_within_each_process": True,
        "source_va_stable_within_each_process": True,
        "prefill_decode_full_graph_replayed": True,
        "prefill_resume_decode_park_validated": True,
        "non_runtime_remote_hot_weights_remained_gpu_resident": True,
        "cuda_graph_recaptured": False,
        "faults_detected": False,
        "resources_released": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
