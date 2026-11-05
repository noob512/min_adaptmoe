#!/usr/bin/env python3
"""Verify fixed-VA physical KV low/high resize and direct-state parity."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


MAX_BYTES = 1_610_612_736
LOW_BLOCKS = 105
HIGH_BLOCKS = 1023
EXPECTED_SELECTED_BYTES = 6_568_280_064
BAD_KERNEL = re.compile(
    r"Xid|FAULT_PDE|manual_migration_fault|\bOops\b|\bBUG:",
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


def physical_ok(snapshot: dict[str, Any], target: int) -> None:
    require(int(snapshot.get("reserved_va_bytes", -1)) == MAX_BYTES, "KV VA bytes")
    require(int(snapshot.get("total_blocks", -1)) == HIGH_BLOCKS, "KV max blocks")
    require(int(snapshot.get("target_blocks", -1)) == target, "KV target")
    for label in ("full", "head", "tail"):
        value = snapshot[label]
        require(value.get("ok") is True, f"{label}: query status")
        require(int(value.get("unknown_bytes", -1)) == 0, f"{label}: unknown")
    head = snapshot["head"]
    tail = snapshot["tail"]
    full = snapshot["full"]
    require(
        int(head["gpu_bytes"]) == int(head["requested_bytes"])
        and int(head["cpu_bytes"]) == 0,
        "active KV prefix placement",
    )
    require(
        int(tail["cpu_bytes"]) == int(tail["requested_bytes"])
        and int(tail["gpu_bytes"]) == 0,
        "inactive KV tail placement",
    )
    require(
        int(full["gpu_bytes"]) + int(full["cpu_bytes"])
        == int(full["requested_bytes"]),
        "full KV physical accounting",
    )


def verify_shutdown(result: dict[str, Any], label: str) -> None:
    shutdown = result.get("shutdown_contract", {})
    require(not shutdown.get("cleanup_errors"), f"{label}: cleanup errors")
    require(
        shutdown.get("controlled_hard_exit_after_cleanup") is True,
        f"{label}: controlled cleanup",
    )
    diagnostics = shutdown.get("diagnostics_after_shutdown", {})
    model = diagnostics.get("model", {})
    driver = diagnostics.get("driver", {})
    require(
        int(model.get("session_id", -1)) == 0
        and int(model.get("active_driver_backing_bytes", -1)) == 0,
        f"{label}: model resources",
    )
    require(
        int(driver.get("session_count", -1)) == 0,
        f"{label}: driver resources",
    )


def verify_record(
    record: dict[str, Any],
    *,
    label: str,
    target: int,
    batch: int,
    pressure_class: str,
    expected_va: str,
) -> None:
    require(record.get("label") == label, f"{label}: record label")
    pressure = record.get("pressure", {})
    require(
        pressure.get("previous_request_finished") is True
        and int(pressure.get("scheduler_unfinished_request_count", -1)) == 0,
        f"{label}: idle boundary",
    )
    require(
        pressure.get("decision_source")
        == "engine_next_request_pressure_policy",
        f"{label}: target was not selected by EngineCore pressure policy",
    )
    require(
        int(pressure.get("new_target_blocks", -1)) == target
        and int(pressure.get("next_batch_size", -1)) == batch
        and pressure.get("pressure_class") == pressure_class,
        f"{label}: policy decision",
    )
    physical_ok(record["physical_after_resize"], target)
    physical_ok(record["physical_after_request"], target)
    for key in ("physical_before_resize", "physical_after_resize", "physical_after_request"):
        require(record[key].get("va_sha256") == expected_va, f"{label}: KV VA")
    require(record.get("resize", {}).get("ok") is True, f"{label}: resize")

    graph_before = record["graph_wrapper_calls_before_resize"]
    graph_after_resize = record["graph_wrapper_calls_after_resize"]
    graph_after_request = record["graph_wrapper_calls_after_request"]
    require(
        int(graph_before.get("full_capture", 0))
        == int(graph_after_resize.get("full_capture", -1))
        == int(graph_after_request.get("full_capture", -2)),
        f"{label}: Graph recapture",
    )
    counter_before = record["compilation_counter_before_resize"]
    counter_after_resize = record["compilation_counter_after_resize"]
    counter_after_request = record["compilation_counter_after_request"]
    require(
        int(counter_before["num_cudagraph_captured"])
        == int(counter_after_resize["num_cudagraph_captured"])
        == int(counter_after_request["num_cudagraph_captured"]),
        f"{label}: capture counter",
    )
    require(
        record["prefill"]["graph_wrapper_delta"].get("full_replay") == 1,
        f"{label}: prefill replay",
    )
    require(
        all(
            step["graph_wrapper_delta"].get("full_replay") == 1
            for step in record["measured_decode"]
        ),
        f"{label}: decode replay",
    )
    require(len(record["measured_decode"]) == 16, f"{label}: decode samples")
    parameter = record["parameter_va_snapshots"]
    require(
        parameter["before_request"]["vas"] == parameter["after_request"]["vas"],
        f"{label}: expert Parameter VAs",
    )
    require(
        record["diagnostics_after_request"]["model"].get("lifecycle_state")
        == "remote_parked",
        f"{label}: expert topology",
    )
    require(
        len(record.get("outputs", [])) == batch
        and all(item.get("finished") is True for item in record["outputs"]),
        f"{label}: outputs",
    )


def prefill_timing(record: dict[str, Any]) -> tuple[float, float, float]:
    """Separate physical-tier compute from the unrelated alias-resume wall."""
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
    require(isinstance(before, list) and isinstance(after, list), "transition history")
    require(
        len(after) >= len(before) and after[: len(before)] == before,
        "transition history prefix",
    )
    resumes = [
        transition
        for transition in after[len(before) :]
        if transition.get("event")
        == "cpu_offload_uvm_original_va_alias_prefill_activated"
    ]
    require(
        len(resumes) == 1 and resumes[0].get("ok") is True,
        "prefill alias-resume transition",
    )
    end_to_end_ms = float(record["prefill"]["elapsed_ms"])
    resume_ms = float(resumes[0].get("wall_ns", 0)) / 1e6
    compute_ms = end_to_end_ms - resume_ms
    require(
        all(
            math.isfinite(value) and value > 0
            for value in (end_to_end_ms, resume_ms, compute_ms)
        ),
        "invalid prefill timing decomposition",
    )
    return end_to_end_ms, resume_ms, compute_ms


def timing(records: list[dict[str, Any]]) -> dict[str, float | int]:
    prefill_parts = [prefill_timing(record) for record in records]
    prefill_end_to_end = [value[0] for value in prefill_parts]
    prefill_resume = [value[1] for value in prefill_parts]
    prefill_compute = [value[2] for value in prefill_parts]
    decode = [
        float(step["elapsed_ms"])
        for record in records
        for step in record["measured_decode"]
    ]
    require(
        len(prefill_compute) == 3 and len(decode) == 48,
        "timing sample cardinality",
    )
    require(
        all(
            math.isfinite(value) and value > 0
            for value in prefill_compute + decode
        ),
        "invalid timing sample",
    )
    batch = int(records[0]["pressure"]["next_batch_size"])
    decode_median = statistics.median(decode)
    return {
        "prefill_samples": len(prefill_compute),
        "decode_samples": len(decode),
        "prefill_end_to_end_median_ms": statistics.median(prefill_end_to_end),
        "prefill_topology_resume_median_ms": statistics.median(prefill_resume),
        "prefill_excluding_topology_resume_median_ms": statistics.median(
            prefill_compute
        ),
        "decode_median_ms": decode_median,
        "decode_tokens_per_second": batch * 1000.0 / decode_median,
    }


def delta_percent(dynamic: float, direct: float) -> float:
    return (dynamic / direct - 1.0) * 100.0


def parse_args() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_root.resolve()
    results = {
        label: load_json(root / label / "result.json")
        for label in ("dynamic", "static_low", "static_high")
    }
    for label, result in results.items():
        require(
            (root / label / "returncode.txt").read_text().strip() == "0",
            f"{label}: runner rc",
        )
        require(result.get("status") == "ok", f"{label}: runner status")
        require(result.get("arm") == "D", f"{label}: arm")
        require(
            int(result.get("config", {}).get("kv_cache_memory_bytes", -1))
            == MAX_BYTES,
            f"{label}: maximum KV reservation",
        )
        require(
            int(
                result.get("diagnostics", {})
                .get("after_load", {})
                .get("model", {})
                .get("selected_parameter_bytes_per_forward", -1)
            )
            == EXPECTED_SELECTED_BYTES,
            f"{label}: fixed expert plan",
        )
        verify_shutdown(result, label)
        sequence = json.loads(
            (root / label / "physical_kv_sequence.json").read_text(
                encoding="utf-8"
            )
        )
        require(
            isinstance(sequence, list)
            and sequence
            and all(
                isinstance(record, dict) and "target_blocks" not in record
                for record in sequence
            ),
            f"{label}: workload sequence prescribes a physical KV target",
        )

    dynamic = results["dynamic"]["dynamic_kv_resizes"]
    static_low = results["static_low"]["dynamic_kv_resizes"]
    static_high = results["static_high"]["dynamic_kv_resizes"]
    require(len(dynamic) == 6, "dynamic generation count")
    require(len(static_low) == 3, "static-low sample count")
    require(len(static_high) == 3, "static-high sample count")

    initial_targets = {
        "dynamic": LOW_BLOCKS,
        "static_low": LOW_BLOCKS,
        "static_high": LOW_BLOCKS,
    }
    for label, target in initial_targets.items():
        physical_ok(results[label]["initial_kv_physical"], target)

    dynamic_va = results["dynamic"]["initial_kv_physical"]["va_sha256"]
    static_low_va = results["static_low"]["initial_kv_physical"]["va_sha256"]
    static_high_va = results["static_high"]["initial_kv_physical"]["va_sha256"]
    for index, record in enumerate(dynamic[:3], 1):
        verify_record(
            record,
            label=f"dynamic_high_{index}",
            target=HIGH_BLOCKS,
            batch=8,
            pressure_class="high",
            expected_va=dynamic_va,
        )
    for index, record in enumerate(dynamic[3:], 1):
        verify_record(
            record,
            label=f"dynamic_low_{index}",
            target=LOW_BLOCKS,
            batch=1,
            pressure_class="low",
            expected_va=dynamic_va,
        )
    for index, record in enumerate(static_low, 1):
        verify_record(
            record,
            label=f"static_low_{index}",
            target=LOW_BLOCKS,
            batch=1,
            pressure_class="low",
            expected_va=static_low_va,
        )
    for index, record in enumerate(static_high, 1):
        verify_record(
            record,
            label=f"static_high_{index}",
            target=HIGH_BLOCKS,
            batch=8,
            pressure_class="high",
            expected_va=static_high_va,
        )

    require(
        dynamic[0]["pressure"]["physical_target_changed"] is True
        and int(dynamic[0]["pressure"]["old_target_blocks"]) == LOW_BLOCKS,
        "dynamic grow transition",
    )
    require(
        dynamic[3]["pressure"]["physical_target_changed"] is True
        and int(dynamic[3]["pressure"]["old_target_blocks"]) == HIGH_BLOCKS,
        "dynamic shrink transition",
    )
    require(
        static_high[0]["pressure"]["physical_target_changed"] is True
        and int(static_high[0]["pressure"]["old_target_blocks"]) == LOW_BLOCKS,
        "direct-high grow transition",
    )
    require(
        all(
            record["pressure"]["physical_target_changed"] is False
            for record in (
                dynamic[1:3] + dynamic[4:] + static_low + static_high[1:]
            )
        ),
        "unexpected extra physical transition",
    )

    for dynamic_record, direct_record in zip(dynamic[:3], static_high):
        require(
            dynamic_record["output_token_ids_sha256"]
            == direct_record["output_token_ids_sha256"],
            "high-tier output differs from direct baseline",
        )
    for dynamic_record, direct_record in zip(dynamic[3:], static_low):
        require(
            dynamic_record["output_token_ids_sha256"]
            == direct_record["output_token_ids_sha256"],
            "low-tier output differs from direct baseline",
        )

    timing_dynamic_high = timing(dynamic[:3])
    timing_dynamic_low = timing(dynamic[3:])
    timing_static_high = timing(static_high)
    timing_static_low = timing(static_low)
    high_prefill_compute_delta = delta_percent(
        float(timing_dynamic_high["prefill_excluding_topology_resume_median_ms"]),
        float(timing_static_high["prefill_excluding_topology_resume_median_ms"]),
    )
    high_decode_delta = delta_percent(
        float(timing_dynamic_high["decode_median_ms"]),
        float(timing_static_high["decode_median_ms"]),
    )
    low_prefill_compute_delta = delta_percent(
        float(timing_dynamic_low["prefill_excluding_topology_resume_median_ms"]),
        float(timing_static_low["prefill_excluding_topology_resume_median_ms"]),
    )
    low_decode_delta = delta_percent(
        float(timing_dynamic_low["decode_median_ms"]),
        float(timing_static_low["decode_median_ms"]),
    )
    high_prefill_end_to_end_delta = delta_percent(
        float(timing_dynamic_high["prefill_end_to_end_median_ms"]),
        float(timing_static_high["prefill_end_to_end_median_ms"]),
    )
    low_prefill_end_to_end_delta = delta_percent(
        float(timing_dynamic_low["prefill_end_to_end_median_ms"]),
        float(timing_static_low["prefill_end_to_end_median_ms"]),
    )
    parity = {
        "scope": (
            "steady physical-tier compute; excludes physical resize and "
            "remote_parked_to_sealed topology resume"
        ),
        "prefill_tolerance_percent": 15.0,
        "decode_tolerance_percent": 10.0,
        "high_prefill_compute_delta_percent": high_prefill_compute_delta,
        "high_decode_delta_percent": high_decode_delta,
        "low_prefill_compute_delta_percent": low_prefill_compute_delta,
        "low_decode_delta_percent": low_decode_delta,
        "diagnostic_high_prefill_end_to_end_delta_percent": (
            high_prefill_end_to_end_delta
        ),
        "diagnostic_low_prefill_end_to_end_delta_percent": (
            low_prefill_end_to_end_delta
        ),
        "within_tolerance": (
            abs(high_prefill_compute_delta) <= 15.0
            and abs(low_prefill_compute_delta) <= 15.0
            and abs(high_decode_delta) <= 10.0
            and abs(low_decode_delta) <= 10.0
        ),
    }
    require(
        parity["within_tolerance"] is True,
        "dynamic/direct serving performance exceeds qualification tolerance",
    )

    reuse_probe = results["dynamic"].get("physical_kv_reuse_probe") or {}
    require(reuse_probe.get("ok") is True, "HBM reuse probe status")
    require(
        int(reuse_probe.get("requested_bytes", 0)) >= 512 * 1024 * 1024,
        "HBM reuse probe is too small",
    )
    require(
        reuse_probe.get("allocation_touched") is True
        and int(reuse_probe.get("fill_marker", -1)) == 180,
        "HBM reuse probe was not fully submitted and touched",
    )
    require(
        int(reuse_probe.get("allocator_allocated_delta_bytes", 0))
        >= int(reuse_probe.get("requested_bytes", 1)),
        "HBM reuse probe did not create the requested CUDA allocation",
    )
    require(
        reuse_probe.get("allocator_backend") == "native"
        and reuse_probe.get("cuda_pointer_is_managed") is False
        and reuse_probe.get("cuda_pointer_probe_reason") == "not_cuda_managed"
        and int(reuse_probe.get("cuda_pointer_probe_status", -1)) == 0,
        "HBM reuse probe is not proven to be native device memory",
    )
    physical_ok(reuse_probe["physical_before"], LOW_BLOCKS)
    physical_ok(reuse_probe["physical_after_restore"], LOW_BLOCKS)
    require(
        reuse_probe["physical_before"]["va_sha256"]
        == reuse_probe["physical_after_restore"]["va_sha256"],
        "HBM reuse probe changed KV VAs",
    )
    require(
        reuse_probe["compilation_counter_before"]["num_cudagraph_captured"]
        == reuse_probe["compilation_counter_after"]["num_cudagraph_captured"],
        "HBM reuse probe recaptured a CUDA Graph",
    )

    kernel_text = (root / "kernel.since.log").read_text(encoding="utf-8")
    require(BAD_KERNEL.search(kernel_text) is None, "kernel fault signature")
    for label in results:
        uvm_log = root / label / "uvm.log"
        if uvm_log.is_file():
            require(
                BAD_KERNEL.search(uvm_log.read_text(encoding="utf-8")) is None,
                f"{label}: UVM fault signature",
            )

    verification = {
        "schema": "vllm_uvm_physical_kv_pool_resize_verification_v3",
        "status": "ok",
        "cases": 3,
        "maximum_reserved_va_bytes": MAX_BYTES,
        "low_target_blocks": LOW_BLOCKS,
        "high_target_blocks": HIGH_BLOCKS,
        "dynamic_grow_ms": float(dynamic[0]["resize_elapsed_ms"]),
        "direct_high_grow_ms": float(static_high[0]["resize_elapsed_ms"]),
        "dynamic_shrink_ms": float(dynamic[3]["resize_elapsed_ms"]),
        "direct_high_baseline": (
            "first_scheduler_idle_boundary_after_common_low_startup"
        ),
        "low_cpu_tail_bytes": int(
            dynamic[3]["physical_after_resize"]["tail"]["cpu_bytes"]
        ),
        "high_gpu_bytes": int(
            dynamic[0]["physical_after_resize"]["full"]["gpu_bytes"]
        ),
        "timing": {
            "dynamic_high": timing_dynamic_high,
            "direct_high": timing_static_high,
            "dynamic_low": timing_dynamic_low,
            "direct_low": timing_static_low,
        },
        "performance_parity": parity,
        "kv_va_stable_within_each_process": True,
        "cuda_graph_recaptured_during_resize": False,
        "outputs_match_direct_baselines": True,
        "pressure_target_selected_by_engine_policy": True,
        "hbm_reuse_probe_bytes": int(reuse_probe["requested_bytes"]),
        "hbm_reuse_probe_touched": True,
        "hbm_reuse_probe_native_device_memory": True,
        "faults_detected": False,
        "resources_released": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
