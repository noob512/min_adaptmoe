#!/usr/bin/env python3
"""Wrap the shared FULL-Graph benchmark for cold-5 versus layer offload.

The outer harness selects one fresh-process arm with
``VLLM_LAYER_COMPARE_ARM``:

* Z: all-GPU FULL Graph control;
* C: five frozen cold experts in every one of the 48 MoE layers;
* B: whole expert Parameters in layers 0--1;
* L: whole expert Parameters in layers 0--3.

This wrapper does not choose the environment for an arm.  It audits the
realized topology, Graph replay, manifest H2D traffic, original Parameter VAs,
and ordered Graph/alias teardown around the shared timing runner.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
BASE_RUNNER = (
    REPO_ROOT
    / "experiments"
    / "vllm_p3_static_alias_graph_integration"
    / "benchmark_full_graph_steps.py"
)
VALID_ARMS = frozenset({"Z", "C", "B", "L"})
ALIAS_ARMS = frozenset({"C", "B", "L"})
MIB = 1024**2
GIB = 1024**3
CAPTURE_SIZES = (1, 4, 314, 1217)
ARENA_COUNT = 4
PTE_ALIGNMENT = 4096


@dataclass(frozen=True)
class AliasGeometry:
    layer_ids: tuple[int, ...]
    parameter_count: int
    selected_expert_count: int
    selected_bytes: int
    full_bytes: int
    consumer_range_count: int
    backing_bytes: int
    alias_count: int
    alias_bytes: int
    layer_copy_bytes: int
    experts_per_parameter: int | None


GEOMETRIES = {
    "C": AliasGeometry(
        layer_ids=tuple(range(48)),
        parameter_count=96,
        selected_expert_count=240,
        selected_bytes=1080 * MIB,
        full_bytes=27 * GIB,
        consumer_range_count=480,
        backing_bytes=45 * MIB,
        alias_count=2250,
        alias_bytes=1125 * MIB,
        layer_copy_bytes=45 * MIB // 2,
        experts_per_parameter=5,
    ),
    "B": AliasGeometry(
        layer_ids=(0, 1),
        parameter_count=4,
        selected_expert_count=0,
        selected_bytes=1152 * MIB,
        full_bytes=1152 * MIB,
        consumer_range_count=4,
        backing_bytes=1152 * MIB,
        alias_count=4608,
        alias_bytes=2304 * MIB,
        layer_copy_bytes=576 * MIB,
        experts_per_parameter=None,
    ),
    "L": AliasGeometry(
        layer_ids=(0, 1, 2, 3),
        parameter_count=8,
        selected_expert_count=0,
        selected_bytes=2304 * MIB,
        full_bytes=2304 * MIB,
        consumer_range_count=8,
        backing_bytes=1152 * MIB,
        alias_count=6912,
        alias_bytes=3456 * MIB,
        layer_copy_bytes=576 * MIB,
        experts_per_parameter=None,
    ),
}


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def output_path_from_argv() -> Path:
    try:
        return Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise RuntimeError("--output PATH is required") from exc


def read_result(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "wrapper_exception", "error": repr(exc)}
    if not isinstance(payload, dict):
        return {"status": "wrapper_exception", "error": "result is not an object"}
    return payload


def manifest_path_from_result(result: dict[str, Any]) -> Path:
    raw = result.get("config", {}).get(
        "offload_manifest",
        os.environ.get("VLLM_CPU_OFFLOAD_MANIFEST_FILE", ""),
    )
    require(isinstance(raw, str) and bool(raw), "offload manifest path is absent")
    return Path(raw).resolve()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"offload manifest was not created: {path}")
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"invalid manifest JSON at {path}:{line_number}: {exc}"
            ) from exc
        require(isinstance(value, dict), f"manifest line {line_number} is not an object")
        records.append(value)
    require(records, "offload manifest is empty")
    return records


def records_for(records: list[dict[str, Any]], event: str) -> list[dict[str, Any]]:
    return [item for item in records if item.get("event") == event]


def expected_parameter_names(geometry: AliasGeometry) -> set[str]:
    return {
        f"model.layers.{layer}.mlp.experts.{local_name}"
        for layer in geometry.layer_ids
        for local_name in ("w13_weight", "w2_weight")
    }


def validate_full_graph_contract(result: dict[str, Any]) -> dict[str, Any]:
    config = result.get("config", {})
    require(config.get("execution_mode") == "full_graph_all", "execution is not FULL Graph")
    require(config.get("enforce_eager") is False, "FULL Graph arm realized eager mode")
    require(config.get("requested_compilation_mode") == "NONE", "compilation mode differs")
    require(config.get("realized_compilation_mode") == "NONE", "realized compilation differs")
    require(config.get("requested_cudagraph_mode") == "FULL", "requested Graph mode differs")
    require(config.get("realized_cudagraph_mode") == "FULL", "realized Graph mode differs")
    require(
        tuple(config.get("realized_cudagraph_capture_sizes", ())) == CAPTURE_SIZES,
        "FULL Graph capture sizes differ",
    )

    cycles = [item for item in result.get("cycles", []) if isinstance(item, dict)]
    require(cycles, "no benchmark cycles were recorded")
    prefill_count = 0
    decode_count = 0
    prompt_tokens_by_batch: dict[int, set[int]] = {}
    for cycle in cycles:
        batch = int(cycle.get("batch_size", 0))
        prompt_tokens = int(cycle.get("prompt_tokens", 0))
        prompt_tokens_by_batch.setdefault(batch, set()).add(prompt_tokens)
        delta = cycle.get("prefill_graph_wrapper_delta", {})
        require(delta == {"full_replay": 1}, f"prefill was not one FULL replay: {delta}")
        prefill_count += 1
        for category in ("warmup_decode_steps", "measured_decode_steps"):
            steps = cycle.get(category, [])
            require(isinstance(steps, list), f"{category} is not a list")
            for step in steps:
                delta = step.get("graph_wrapper_delta", {})
                require(delta == {"full_replay": 1}, f"decode was not one FULL replay: {delta}")
                decode_count += 1
    require(prompt_tokens_by_batch == {1: {314}, 4: {1217}}, "frozen prefill sizes differ")
    require(decode_count > 0, "no decode Graph replay evidence")
    return {
        "capture_sizes": list(CAPTURE_SIZES),
        "prefill_step_count": prefill_count,
        "prefill_full_replay_count": prefill_count,
        "decode_step_count": decode_count,
        "decode_full_replay_count": decode_count,
        "prompt_tokens_by_batch": {
            str(key): sorted(value) for key, value in sorted(prompt_tokens_by_batch.items())
        },
    }


def collect_va_snapshots(
    result: dict[str, Any], active: dict[str, Any]
) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []

    def append(label: str, value: Any) -> None:
        if not isinstance(value, dict):
            return
        snapshots.append(
            {
                "label": label,
                "vas": [int(item) for item in value.get("original_va_snapshot", [])],
                "state": value.get("lifecycle_state"),
                "session_id": int(value.get("session_id", 0)),
            }
        )

    append("after_load", result.get("static_alias_diagnostics_after_load"))
    for index, cycle in enumerate(result.get("cycles", [])):
        if not isinstance(cycle, dict):
            continue
        append(f"cycle_{index}_after_prefill", cycle.get("static_alias_diagnostics_after_prefill"))
        append(f"cycle_{index}_after_decode", cycle.get("static_alias_diagnostics_after_decode"))
    append("final", result.get("static_alias_diagnostics_final"))
    append("before_shutdown", active)
    return snapshots


def validate_residency(
    value: Any,
    *,
    expected_bytes: int,
    cpu_bytes: int,
    gpu_bytes: int,
    label: str,
) -> None:
    require(isinstance(value, dict) and value.get("ok") is True, f"{label} absent")
    require(int(value.get("queried_bytes", -1)) == expected_bytes, f"{label} queried bytes")
    require(int(value.get("cpu_bytes", -1)) == cpu_bytes, f"{label} CPU bytes")
    require(int(value.get("gpu_bytes", -1)) == gpu_bytes, f"{label} GPU bytes")
    require(int(value.get("any_bytes", -1)) == expected_bytes, f"{label} any bytes")
    require(int(value.get("unknown_bytes", -1)) == 0, f"{label} unknown bytes")


def validate_manifest_parameters(
    records: list[dict[str, Any]], geometry: AliasGeometry
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    parameters = records_for(records, "cpu_offload_parameter")
    require(len(parameters) == geometry.parameter_count, "manifest Parameter count differs")
    require(
        {str(item.get("full_parameter_name")) for item in parameters}
        == expected_parameter_names(geometry),
        "manifest selected a different layer/Parameter set",
    )
    require(
        sum(int(item.get("bytes", 0)) for item in parameters) == geometry.selected_bytes,
        "manifest selected bytes differ",
    )
    require(
        sum(int(item.get("parameter_bytes", 0)) for item in parameters)
        == geometry.full_bytes,
        "manifest full Parameter bytes differ",
    )
    require(
        all(
            item.get("graph_static_alias") is True
            and item.get("graph_static_alias_backend") == "uvm_original_va"
            and item.get("uvm_offloading") is True
            and item.get("uvm_original_parameter_va_retained") is True
            for item in parameters
        ),
        "manifest did not retain every original managed Parameter VA",
    )
    expected_experts = geometry.experts_per_parameter
    if expected_experts is None:
        require(
            all(not (item.get("graph_static_alias_expert_ids") or []) for item in parameters),
            "whole-Parameter manifest unexpectedly contains expert ids",
        )
    else:
        require(
            all(
                len(item.get("graph_static_alias_expert_ids") or []) == expected_experts
                for item in parameters
            ),
            "cold-5 manifest expert selection differs",
        )
    consumer_vas = tuple(
        int(item.get("uvm_original_parameter_va", 0)) for item in parameters
    )
    require(
        all(value > 0 for value in consumer_vas)
        and len(set(consumer_vas)) == geometry.parameter_count,
        "manifest original Parameter VAs are absent or duplicated",
    )
    return parameters, consumer_vas


def validate_manifest_forwards(
    records: list[dict[str, Any]], geometry: AliasGeometry
) -> dict[str, Any]:
    forwards = records_for(records, "cpu_offload_copy_forward")
    require(forwards, "copy-forward H2D trace is absent")
    # The hook is installed on the decoder-layer module, while the selected
    # Parameter names below it are ``mlp.experts.{w13,w2}_weight``.
    expected_prefixes = {f"model.layers.{layer}" for layer in geometry.layer_ids}
    prefixes = [str(item.get("module_prefix")) for item in forwards]
    require(set(prefixes) == expected_prefixes, "forward trace covered different layers")
    per_layer_counts = {prefix: prefixes.count(prefix) for prefix in expected_prefixes}
    require(len(set(per_layer_counts.values())) == 1, "forward trace is unbalanced by layer")
    require(
        all(
            int(item.get("copy_bytes", 0)) == geometry.layer_copy_bytes
            and int(item.get("graph_static_alias_actual_h2d_bytes", 0))
            == geometry.layer_copy_bytes
            and int(item.get("graph_static_alias_actual_h2d_tensor_count", 0)) == 2
            and item.get("graph_static_alias") is True
            and item.get("graph_static_alias_backend") == "uvm_original_va"
            and item.get("graph_static_alias_lifecycle_state") == "sealed"
            and int(item.get("copy_exact_name_count", 0)) == 2
            for item in forwards
        ),
        "one selected layer submitted an unexpected H2D copy",
    )
    captured = [
        item for item in forwards if item.get("graph_static_alias_capture_active") is True
    ]
    expected_captured = len(CAPTURE_SIZES) * len(geometry.layer_ids)
    require(len(captured) == expected_captured, "Graph capture H2D trace count differs")
    require(
        all(item.get("graph_static_alias_capture_phase") is True for item in captured),
        "captured H2D record was outside the capture phase",
    )
    captured_h2d = sum(
        int(item.get("graph_static_alias_actual_h2d_bytes", 0)) for item in captured
    )
    require(
        captured_h2d == len(CAPTURE_SIZES) * geometry.selected_bytes,
        "captured manifest H2D bytes differ",
    )
    return {
        "forward_trace_count": len(forwards),
        "forward_trace_h2d_bytes": sum(
            int(item.get("graph_static_alias_actual_h2d_bytes", 0))
            for item in forwards
        ),
        "capture_forward_trace_count": len(captured),
        "capture_forward_h2d_bytes": captured_h2d,
        "per_layer_forward_trace_count": dict(sorted(per_layer_counts.items())),
    }


def validate_alias_contract(
    *,
    result: dict[str, Any],
    diagnostics: dict[str, Any],
    driver: dict[str, Any],
    records: list[dict[str, Any]],
    geometry: AliasGeometry,
) -> dict[str, Any]:
    require(diagnostics.get("enabled") is True, "alias diagnostics disabled")
    require(diagnostics.get("backend") == "uvm_original_va", "alias backend differs")
    require(diagnostics.get("lifecycle_state") == "sealed", "alias is not sealed")
    require(diagnostics.get("sealed") is True, "sealed diagnostic is false")
    require(diagnostics.get("capture_phase") is False, "capture phase remained active")
    require(int(diagnostics.get("session_id", 0)) > 0, "alias session is absent")
    require(int(diagnostics.get("arena_count", 0)) == ARENA_COUNT, "arena count differs")
    require(int(diagnostics.get("backing_bytes", 0)) == geometry.backing_bytes, "backing differs")
    require(
        int(diagnostics.get("active_driver_backing_bytes", 0)) == geometry.backing_bytes,
        "active driver backing differs",
    )
    require(
        int(diagnostics.get("selected_expert_count", -1))
        == geometry.selected_expert_count,
        "selected expert diagnostic differs",
    )
    require(
        int(diagnostics.get("selected_parameter_count", 0))
        == geometry.parameter_count,
        "selected Parameter diagnostic differs",
    )
    require(
        int(diagnostics.get("selected_parameter_bytes_per_forward", 0))
        == geometry.selected_bytes,
        "selected H2D bytes per forward differ",
    )
    require(
        int(diagnostics.get("full_parameter_bytes", 0)) == geometry.full_bytes,
        "full Parameter geometry differs",
    )
    require(
        int(diagnostics.get("consumer_range_count", 0))
        == geometry.consumer_range_count,
        "consumer range count differs",
    )

    expert_plan = diagnostics.get("expert_plan", [])
    if geometry.experts_per_parameter is None:
        require(not diagnostics.get("expert_plan_file"), "whole-Parameter arm loaded an expert plan")
        require(not expert_plan, "whole-Parameter diagnostics contain an expert plan")
        require(diagnostics.get("cold_residency_result") is None, "whole-Parameter cold residency should be absent")
        require(diagnostics.get("hot_residency_result") is None, "whole-Parameter hot residency should be absent")
    else:
        require(bool(diagnostics.get("expert_plan_file")), "cold-5 expert plan file is absent")
        require(len(expert_plan) == len(geometry.layer_ids), "cold-5 expert plan layer count differs")
        require(
            {int(item[0]) for item in expert_plan} == set(geometry.layer_ids)
            and all(len(item[1]) == geometry.experts_per_parameter for item in expert_plan),
            "cold-5 diagnostic expert plan differs",
        )
        validate_residency(
            diagnostics.get("cold_residency_result"),
            expected_bytes=geometry.selected_bytes,
            cpu_bytes=geometry.selected_bytes,
            gpu_bytes=0,
            label="cold residency",
        )
        hot_bytes = geometry.full_bytes - geometry.selected_bytes
        validate_residency(
            diagnostics.get("hot_residency_result"),
            expected_bytes=hot_bytes,
            cpu_bytes=0,
            gpu_bytes=hot_bytes,
            label="hot residency",
        )

    arenas = diagnostics.get("arenas", [])
    require(isinstance(arenas, list) and len(arenas) == ARENA_COUNT, "arena details differ")
    require(
        all(
            arena.get("pte_mode") == "4K"
            and int(arena.get("pte_alignment", 0)) == PTE_ALIGNMENT
            for arena in arenas
        ),
        "one or more aliases did not use 4K PTEs",
    )
    require(
        sum(int(arena.get("backing_bytes", 0)) for arena in arenas)
        == geometry.backing_bytes,
        "arena backing sum differs",
    )
    arena_consumer_count = sum(len(arena.get("consumer_vas", [])) for arena in arenas)
    require(arena_consumer_count == geometry.parameter_count, "arena consumer count differs")
    nested_expert_ids = [
        value for arena in arenas for value in arena.get("expert_ids", [])
    ]
    if geometry.experts_per_parameter is None:
        require(
            len(nested_expert_ids) == geometry.parameter_count
            and all(value is None for value in nested_expert_ids),
            "whole-Parameter arena contains expert slices",
        )
    else:
        require(
            len(nested_expert_ids) == geometry.parameter_count
            and all(len(value) == geometry.experts_per_parameter for value in nested_expert_ids),
            "cold-5 arena expert geometry differs",
        )

    preallocate = diagnostics.get("preallocate_result", {})
    require(preallocate.get("ok") is True, "slot preallocation failed")
    require(int(preallocate.get("requested_bytes", 0)) == geometry.backing_bytes, "preallocation requested bytes differ")
    require(int(preallocate.get("accepted_bytes", 0)) == geometry.backing_bytes, "preallocation accepted bytes differ")
    maps = diagnostics.get("map_results", [])
    require(isinstance(maps, (list, tuple)) and maps, "MAP evidence is absent")
    require(all(item.get("ok") is True for item in maps), "one or more MAP batches failed")
    seal = diagnostics.get("seal_result", {})
    require(seal.get("ok") is True and seal.get("sealed") is True, "SEAL failed")
    require(int(seal.get("alias_count", 0)) == geometry.alias_count, "alias count differs")
    require(int(seal.get("alias_bytes", 0)) == geometry.alias_bytes, "alias bytes differ")
    require(int(seal.get("backing_bytes", 0)) == geometry.backing_bytes, "SEAL backing differs")

    require(driver.get("session_count") == 1, "driver session count differs")
    sessions = driver.get("sessions", [])
    require(isinstance(sessions, list) and len(sessions) == 1, "driver session is absent")
    session = sessions[0]
    require(session.get("sealed") is True and session.get("poisoned") is False, "driver session state differs")
    require(int(session.get("session_id", 0)) == int(diagnostics["session_id"]), "session ids differ")
    require(int(session.get("alias_count", 0)) == geometry.alias_count, "driver alias count differs")
    require(int(session.get("alias_bytes", 0)) == geometry.alias_bytes, "driver alias bytes differ")
    require(int(session.get("backing_bytes", 0)) == geometry.backing_bytes, "driver backing differs")

    _, manifest_vas = validate_manifest_parameters(records, geometry)
    forward_evidence = validate_manifest_forwards(records, geometry)
    finalized = records_for(records, "cpu_offload_uvm_static_alias_finalized")
    require(len(finalized) == 1, "manifest finalize record count differs")
    finalize = finalized[0]
    require(finalize.get("ok") is True and finalize.get("sealed") is True, "manifest finalize failed")
    require(finalize.get("pte_mode") == "4K", "manifest PTE mode differs")
    require(int(finalize.get("session_id", 0)) == int(diagnostics["session_id"]), "manifest session differs")
    require(int(finalize.get("backing_bytes", 0)) == geometry.backing_bytes, "manifest finalize backing differs")
    require(int(finalize.get("alias_count", 0)) == geometry.alias_count, "manifest finalize alias count differs")
    require(int(finalize.get("alias_bytes", 0)) == geometry.alias_bytes, "manifest finalize alias bytes differ")

    snapshots = collect_va_snapshots(result, diagnostics)
    fingerprints = [tuple(item["vas"]) for item in snapshots]
    require(fingerprints, "original Parameter VA snapshots are absent")
    require(
        all(
            len(value) == geometry.parameter_count
            and all(va > 0 for va in value)
            and len(set(value)) == geometry.parameter_count
            for value in fingerprints
        ),
        "original Parameter VA snapshot geometry differs",
    )
    require(len(set(fingerprints)) == 1, "an original Parameter VA changed")
    require(all(item["state"] == "sealed" for item in snapshots), "alias lifecycle changed during execution")
    require(
        all(item["session_id"] == int(diagnostics["session_id"]) for item in snapshots),
        "alias session changed during execution",
    )
    fingerprint = fingerprints[0]
    require(set(manifest_vas) == set(fingerprint), "manifest and diagnostic Parameter VAs differ")
    require(
        {int(value) for value in finalize.get("original_parameter_vas", [])}
        == set(fingerprint),
        "finalize and diagnostic Parameter VAs differ",
    )
    certificate = diagnostics.get("alias_graph_certificate", {})
    require(
        diagnostics.get("capture_topology") == "sealed"
        and diagnostics.get("alias_graph_captured") is True
        and certificate.get("topology") == "sealed",
        "sealed Graph capture certificate is absent",
    )
    require(
        int(certificate.get("capture_forward_count", 0))
        == len(CAPTURE_SIZES) * len(geometry.layer_ids),
        "capture certificate forward count differs",
    )
    require(
        int(certificate.get("actual_h2d_bytes", 0))
        == len(CAPTURE_SIZES) * geometry.selected_bytes,
        "capture certificate H2D bytes differ",
    )
    require(
        tuple(int(value) for value in certificate.get("parameter_vas", []))
        == fingerprint,
        "capture certificate Parameter VAs differ",
    )
    return {
        "geometry": {
            "layer_ids": list(geometry.layer_ids),
            "parameter_count": geometry.parameter_count,
            "selected_expert_count": geometry.selected_expert_count,
            "selected_bytes": geometry.selected_bytes,
            "full_bytes": geometry.full_bytes,
            "consumer_range_count": geometry.consumer_range_count,
            "backing_bytes": geometry.backing_bytes,
            "alias_count": geometry.alias_count,
            "alias_bytes": geometry.alias_bytes,
            "layer_copy_bytes": geometry.layer_copy_bytes,
        },
        "parameter_va_snapshots": snapshots,
        "parameter_va_fingerprint": list(fingerprint),
        **forward_evidence,
    }


def validate_control_contract(
    diagnostics: dict[str, Any],
    driver: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    require(diagnostics.get("enabled") is False, "all-GPU control enabled aliasing")
    require(diagnostics.get("backend") is None, "all-GPU control has an alias backend")
    require(int(diagnostics.get("session_id", 0)) == 0, "all-GPU control has a session")
    require(int(diagnostics.get("arena_count", 0)) == 0, "all-GPU control has arenas")
    require(int(diagnostics.get("backing_bytes", 0)) == 0, "all-GPU control has backing")
    require(driver.get("session_count") == 0, "all-GPU control retained a driver session")
    require(not records_for(records, "cpu_offload_parameter"), "all-GPU control offloaded Parameters")
    require(not records_for(records, "cpu_offload_copy_forward"), "all-GPU control submitted offload copies")
    require(not records_for(records, "cpu_offload_uvm_static_alias_finalized"), "all-GPU control finalized aliases")
    return {"parameter_count": 0, "forward_trace_count": 0}


def validate_shutdown_contract(
    *,
    arm: str,
    geometry: AliasGeometry | None,
    shutdown_results: list[dict[str, Any]],
    post: dict[str, Any],
    driver_post: dict[str, Any],
    records_after: list[dict[str, Any]],
) -> dict[str, Any]:
    require(len(shutdown_results) == 1, "model-runner shutdown result count differs")
    shutdown = shutdown_results[0]
    require(shutdown.get("ok") is True, "model-runner shutdown failed")
    require(int(post.get("session_id", -1)) == 0, "session survived shutdown")
    require(driver_post.get("session_count") == 0, "driver session survived shutdown")
    if arm == "Z":
        require(shutdown.get("reason") == "disabled", "control shutdown reason differs")
        require(post.get("enabled") is False, "control alias state became enabled")
        require(not records_for(records_after, "cpu_offload_uvm_static_alias_destroyed"), "control emitted alias destroy")
        return {"destroyed_alias_count": 0, "destroyed_alias_bytes": 0}

    assert geometry is not None
    require(shutdown.get("backend") == "uvm_original_va", "shutdown backend differs")
    require(int(shutdown.get("destroyed_graph_count", 0)) == len(CAPTURE_SIZES), "Graph teardown count differs")
    require(shutdown.get("state") == "remote", "shutdown did not reach remote state")
    destroy = shutdown.get("destroy", {})
    require(destroy.get("ok") is True, "cmd99 destroy failed")
    require(destroy.get("cache_cleared") is True, "driver cache was not cleared")
    require(destroy.get("session_cleared") is True, "driver session was not cleared")
    require(int(destroy.get("released_alias_count", 0)) == geometry.alias_count, "released alias count differs")
    require(int(destroy.get("released_alias_bytes", 0)) == geometry.alias_bytes, "released alias bytes differ")
    require(int(destroy.get("released_backing_bytes", 0)) == geometry.backing_bytes, "released backing differs")
    for field in ("remaining_alias_count", "remaining_alias_bytes", "remaining_backing_bytes"):
        require(int(destroy.get(field, -1)) == 0, f"{field} is nonzero after destroy")
    slot_close = shutdown.get("slot_close", {})
    require(slot_close.get("ok") is True and slot_close.get("closed") is True, "slot session close failed")
    require(post.get("enabled") is True, "post-shutdown alias configuration disappeared")
    require(post.get("backend") == "uvm_original_va", "post-shutdown backend differs")
    require(post.get("lifecycle_state") == "remote", "post-shutdown state differs")
    require(post.get("sealed") is False, "post-shutdown diagnostics remained sealed")
    require(int(post.get("active_driver_backing_bytes", -1)) == 0, "active backing survived shutdown")
    require(post.get("cleanup_pending") is False, "alias cleanup remains pending")
    require(post.get("cleanup_unresolved") is False, "alias cleanup is unresolved")
    destroyed = records_for(records_after, "cpu_offload_uvm_static_alias_destroyed")
    require(len(destroyed) == 1, "manifest destroy record count differs")
    manifest_destroy = destroyed[0]
    require(manifest_destroy.get("ok") is True and manifest_destroy.get("state") == "remote", "manifest destroy failed")
    require(
        int(manifest_destroy.get("destroy", {}).get("released_alias_count", 0))
        == geometry.alias_count,
        "manifest released alias count differs",
    )
    return {
        "destroyed_graph_count": int(shutdown.get("destroyed_graph_count", 0)),
        "destroyed_alias_count": int(destroy.get("released_alias_count", 0)),
        "destroyed_alias_bytes": int(destroy.get("released_alias_bytes", 0)),
        "released_backing_bytes": int(destroy.get("released_backing_bytes", 0)),
    }


def summarize_manifest(records: list[dict[str, Any]]) -> dict[str, Any]:
    event_counts: dict[str, int] = {}
    for item in records:
        event = str(item.get("event", "<missing>"))
        event_counts[event] = event_counts.get(event, 0) + 1
    parameters = records_for(records, "cpu_offload_parameter")
    forwards = records_for(records, "cpu_offload_copy_forward")
    return {
        "record_count": len(records),
        "event_counts": dict(sorted(event_counts.items())),
        "parameter_count": len(parameters),
        "selected_parameter_bytes": sum(int(item.get("bytes", 0)) for item in parameters),
        "full_parameter_bytes": sum(int(item.get("parameter_bytes", 0)) for item in parameters),
        "forward_trace_count": len(forwards),
        "forward_actual_h2d_bytes": sum(
            int(item.get("graph_static_alias_actual_h2d_bytes", 0)) for item in forwards
        ),
    }


def selected_environment() -> dict[str, str]:
    prefixes = (
        "VLLM_CPU_OFFLOAD_",
        "VLLM_UVM_",
        "VLLM_LAYER_COMPARE_",
        "VLLM_MOE_TOKEN_EXPERT_TRACE",
    )
    exact = {
        "CUDA_VISIBLE_DEVICES",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_USE_UVM",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
    }
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(prefixes) or key in exact
    }


def main() -> int:
    arm = os.environ.get("VLLM_LAYER_COMPARE_ARM", "").strip().upper()
    require(arm in VALID_ARMS, f"invalid VLLM_LAYER_COMPARE_ARM={arm!r}")
    output_path = output_path_from_argv()
    base = load_module("_layer_vs_cold5_prefill_base", BASE_RUNNER)

    import vllm
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    live_llms: list[Any] = []
    shutdown_results: list[dict[str, Any]] = []
    original_llm = vllm.LLM
    original_shutdown = GPUModelRunner.shutdown_cpu_offload_graph_static_alias

    class TrackedLLM(original_llm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            live_llms.append(self)

    def tracked_shutdown(self):
        value = original_shutdown(self)
        shutdown_results.append(dict(value))
        return value

    vllm.LLM = TrackedLLM
    GPUModelRunner.shutdown_cpu_offload_graph_static_alias = tracked_shutdown
    base_error: BaseException | None = None
    base_traceback = None
    return_code = 1
    try:
        return_code = int(base.run())
    except BaseException as exc:
        base_error = exc
        base_traceback = exc.__traceback__

    result = read_result(output_path)
    active: dict[str, Any] = {}
    driver_active: dict[str, Any] = {}
    post: dict[str, Any] = {}
    driver_post: dict[str, Any] = {}
    records_before: list[dict[str, Any]] = []
    records_after: list[dict[str, Any]] = []
    graph_evidence: dict[str, Any] = {}
    alias_evidence: dict[str, Any] = {}
    shutdown_evidence: dict[str, Any] = {}
    contract_errors: list[str] = []

    def record_error(label: str, exc: BaseException) -> None:
        contract_errors.append(f"{label}: {exc!r}")
        contract_errors.append(traceback.format_exc())

    try:
        from vllm.device_allocator.uvm_static_alias import get_static_alias_diagnostics
        from vllm.model_executor.models.utils import (
            get_cpu_offload_graph_static_alias_diagnostics,
        )

        try:
            active = get_cpu_offload_graph_static_alias_diagnostics()
            driver_active = get_static_alias_diagnostics()
            records_before = read_manifest(manifest_path_from_result(result))
            if base_error is None:
                require(result.get("status") == "ok", "base benchmark did not finish")
                graph_evidence = validate_full_graph_contract(result)
                if arm in ALIAS_ARMS:
                    alias_evidence = validate_alias_contract(
                        result=result,
                        diagnostics=active,
                        driver=driver_active,
                        records=records_before,
                        geometry=GEOMETRIES[arm],
                    )
                else:
                    alias_evidence = validate_control_contract(
                        active, driver_active, records_before
                    )
        except Exception as exc:
            record_error("active_contract", exc)

        try:
            require(bool(live_llms), "no tracked LLM is available for explicit shutdown")
            live_llms[-1].llm_engine.engine_core.shutdown()
            gc.collect()
            try:
                import torch

                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
            except Exception as exc:
                record_error("post_shutdown_sync", exc)
        except Exception as exc:
            record_error("explicit_shutdown", exc)

        try:
            post = get_cpu_offload_graph_static_alias_diagnostics()
            driver_post = get_static_alias_diagnostics()
            records_after = read_manifest(manifest_path_from_result(result))
            if base_error is None:
                shutdown_evidence = validate_shutdown_contract(
                    arm=arm,
                    geometry=GEOMETRIES.get(arm),
                    shutdown_results=shutdown_results,
                    post=post,
                    driver_post=driver_post,
                    records_after=records_after,
                )
        except Exception as exc:
            record_error("shutdown_contract", exc)
    except Exception as exc:
        record_error("wrapper_contract", exc)
    finally:
        GPUModelRunner.shutdown_cpu_offload_graph_static_alias = original_shutdown
        vllm.LLM = original_llm

    result["base_schema"] = result.get("schema")
    result["schema"] = "vllm_uvm_layer_vs_cold5_prefill_steps_v1"
    result["arm"] = arm
    result["graph_contract"] = graph_evidence
    result["alias_contract"] = alias_evidence
    result["mechanism_contract"] = {
        "diagnostics_before_shutdown": active,
        "driver_diagnostics_before_shutdown": driver_active,
        "manifest_before_shutdown": summarize_manifest(records_before),
    }
    result["shutdown_contract"] = {
        "model_runner_shutdown_results": shutdown_results,
        "diagnostics_after_shutdown": post,
        "driver_diagnostics_after_shutdown": driver_post,
        "manifest_after_shutdown": summarize_manifest(records_after),
        "verified_release": shutdown_evidence,
    }
    result["reproducibility_environment"] = selected_environment()
    result["wrapper_contract_errors"] = contract_errors
    if contract_errors and base_error is None:
        result["status"] = "contract_failed"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    if base_error is not None:
        raise base_error.with_traceback(base_traceback)
    if contract_errors:
        raise ContractError("; ".join(contract_errors))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
