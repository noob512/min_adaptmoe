"""CPU-only contract tests for the four-arm fail-closed verifier."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import pytest


HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFY = load("phase_switch_verify_test", HERE / "verify_experiment.py")
BENCH = load("phase_switch_benchmark_test", HERE / "benchmark_phase_switch_steps.py")


def remote_transition(*, preserve: bool = False) -> dict[str, Any]:
    vas = [0x100000 + index * 0x100000 for index in range(VERIFY.PARAMETER_COUNT)]
    return {
        "event": "cpu_offload_uvm_original_va_remote_transition",
        "ok": True,
        "reason": "test",
        "pre_state": "sealed",
        "post_state": "remote",
        "parameter_vas_before": vas,
        "parameter_vas_after": vas,
        "preserve_remote_graphs": preserve,
        "sync_ns": 1_000_000,
        "destroy_ns": 2_000_000,
        "remote_map_ns": 3_000_000,
        "physical_audit_ns": 4_000_000,
        "wall_ns": 10_000_000,
        "remote_map": {
            "ok": True,
            "reason": "ok",
            "requested_count": VERIFY.PARAMETER_COUNT,
            "requested_bytes": VERIFY.LOGICAL_BYTES,
            "success_count": VERIFY.PARAMETER_COUNT,
            "failure_count": 0,
            "ioctl_calls": 1,
            "cpu_resident_bytes": VERIFY.LOGICAL_BYTES,
            "remote_mapped_bytes": VERIFY.LOGICAL_BYTES,
            "covered_bytes": VERIFY.LOGICAL_BYTES,
            "unknown_bytes": 0,
            "first_failed_index": -1,
            "ioctl_ret": 0,
            "rm_status": 0,
            "errno": 0,
        },
        "physical_residency": {
            "ok": True,
            "queried_bytes": VERIFY.LOGICAL_BYTES,
            "cpu_bytes": VERIFY.LOGICAL_BYTES,
            "gpu_bytes": 0,
            "any_bytes": VERIFY.LOGICAL_BYTES,
            "unknown_bytes": 0,
        },
        "destroy": {
            "ok": True,
            "destroy": {
                "ok": True,
                "cache_cleared": True,
                "session_cleared": True,
                "released_alias_count": VERIFY.ALIAS_COUNT,
                "released_alias_bytes": VERIFY.ALIAS_BYTES,
                "released_backing_bytes": VERIFY.DUAL_BANK_BYTES,
                "remaining_alias_count": 0,
                "remaining_alias_bytes": 0,
                "remaining_backing_bytes": 0,
            },
        },
    }


def parameter_result() -> dict[str, Any]:
    names = [f"model.layers.{index // 2}.weight{index % 2}" for index in range(8)]
    sizes = [VERIFY.LOGICAL_BYTES // 8] * 8
    vas = [0x1000000 + index * 0x2000000 for index in range(8)]
    snapshot = {
        "count": 8,
        "bytes": VERIFY.LOGICAL_BYTES,
        "vas": vas,
        "records": [
            {
                "name": name,
                "data_ptr": va,
                "bytes": size,
                "device": "cuda:0",
            }
            for name, va, size in zip(names, vas, sizes, strict=True)
        ],
    }
    return {
        "parameter_va_snapshots": {
            stage: copy.deepcopy(snapshot)
            for stage in ("after_load", "after_prefill", "after_first_decode", "final")
        }
    }


def forward(layer: int, *, state: str, actual: int, capture: bool = False) -> dict[str, Any]:
    return {
        "event": "cpu_offload_copy_forward",
        "module_prefix": f"model.layers.{layer}",
        "copy_bytes": VERIFY.LOGICAL_BYTES // 4,
        "graph_static_alias_actual_h2d_bytes": actual,
        "graph_static_alias_actual_h2d_tensor_count": 2 if actual else 0,
        "graph_static_alias_copy_submitted": bool(actual),
        "graph_static_alias_lifecycle_state": state,
        "graph_static_alias_capture_phase": capture,
        "graph_static_alias_capture_active": capture,
    }


def manifest_result(arm: str) -> dict[str, Any]:
    graph = arm in VERIFY.GRAPH_ARMS
    hybrid = arm in VERIFY.HYBRID_ARMS
    load_records = (
        [forward(layer, state="remote", actual=0, capture=True) for _ in range(2) for layer in range(4)]
        if graph
        else []
    )
    per_layer = VERIFY.LOGICAL_BYTES // 4 if hybrid else 0
    prefill_records = [
        forward(layer, state="sealed" if hybrid else "remote", actual=per_layer)
        for layer in range(4)
    ]
    first_records = [] if graph else [forward(layer, state="remote", actual=0) for layer in range(4)]
    remainder_records = (
        []
        if graph
        else [forward(layer, state="remote", actual=0) for _ in range(3) for layer in range(4)]
    )
    return {
        "config": {"audit_trace": True, "warmup_decode_steps": 1, "measured_decode_steps": 2},
        "manifest_segments": {
            "load_and_capture": BENCH.summarize_manifest(load_records),
            "serving_prefill": BENCH.summarize_manifest(prefill_records),
            "first_decode": BENCH.summarize_manifest(first_records),
            "remaining_decode": BENCH.summarize_manifest(remainder_records),
        },
    }


def test_remote_transition_contract() -> None:
    timings = VERIFY.validate_remote_transition(remote_transition(preserve=True), "case", preserved_graph=True)
    assert timings["wall_ms"] == 10.0
    assert timings["remote_map_ms"] == 3.0


def test_remote_transition_rejects_gpu_residency() -> None:
    value = remote_transition()
    value["physical_residency"]["gpu_bytes"] = 65536
    with pytest.raises(VERIFY.VerificationError, match="GPU physical bytes remain"):
        VERIFY.validate_remote_transition(value, "case")


def test_parameter_va_change_is_rejected() -> None:
    result = parameter_result()
    result["parameter_va_snapshots"]["final"]["records"][0]["data_ptr"] += 4096
    with pytest.raises(VERIFY.VerificationError, match="Parameter VA changed"):
        VERIFY.validate_parameter_snapshots(result, "case")


@pytest.mark.parametrize("arm", tuple(VERIFY.ARM_NAMES))
def test_manifest_contract_for_all_arms(arm: str) -> None:
    VERIFY.validate_manifest_segments(manifest_result(arm), arm, f"case-{arm}")


def test_remote_capture_h2d_is_rejected() -> None:
    result = manifest_result("G")
    record = result["manifest_segments"]["load_and_capture"]["records"][0]
    record["graph_static_alias_actual_h2d_bytes"] = 1024
    with pytest.raises(VERIFY.VerificationError, match="captured Graph contains H2D"):
        VERIFY.validate_manifest_segments(result, "G", "case-G")
