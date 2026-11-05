#!/usr/bin/env python3
"""Wrap the shared vLLM step benchmark with cold-5 alias contracts."""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import traceback
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
VALID_ARMS = frozenset({"T", "Z", "A", "G"})
PARAMETER_COUNT = 96
SELECTED_EXPERT_COUNT = 240
CONSUMER_RANGE_COUNT = 480
FULL_PARAMETER_BYTES = 27 * 1024**3
SELECTED_BYTES = 1080 * 1024**2
HOT_BYTES = FULL_PARAMETER_BYTES - SELECTED_BYTES
BACKING_BYTES = 45 * 1024**2
ALIAS_COUNT = 2250
ALIAS_BYTES = 1125 * 1024**2
LAYER_COPY_BYTES = 225 * 1024**2 // 10


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
    return payload if isinstance(payload, dict) else {"status": "wrapper_exception"}


def read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        require(isinstance(value, dict), f"manifest line {line_number} is not an object")
        records.append(value)
    return records


def validate_graph_contract(result: dict[str, Any], arm: str) -> dict[str, int]:
    expected_graph = arm in {"Z", "G"}
    prefill_deltas = [
        cycle.get("prefill_graph_wrapper_delta", {})
        for cycle in result.get("cycles", [])
        if isinstance(cycle, dict)
    ]
    require(prefill_deltas, "no prefill steps were recorded")
    for delta in prefill_deltas:
        require(
            int(delta.get("full_replay", 0)) == int(expected_graph),
            f"prefill Graph replay differs for arm {arm}: {delta}",
        )
    decode_deltas = [
        step.get("graph_wrapper_delta", {})
        for cycle in result.get("cycles", [])
        if isinstance(cycle, dict)
        for step in cycle.get("measured_decode_steps", [])
        if isinstance(step, dict)
    ]
    require(decode_deltas, "no measured decode steps were recorded")
    for delta in decode_deltas:
        require(
            int(delta.get("full_replay", 0)) == int(expected_graph),
            f"decode Graph replay differs for arm {arm}: {delta}",
        )
    return {
        "prefill_step_count": len(prefill_deltas),
        "prefill_full_replay_count": sum(
            int(delta.get("full_replay", 0)) for delta in prefill_deltas
        ),
        "decode_step_count": len(decode_deltas),
        "decode_full_replay_count": sum(
            int(delta.get("full_replay", 0)) for delta in decode_deltas
        ),
    }


def collect_va_snapshots(result: dict[str, Any]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []

    def append(label: str, value: Any) -> None:
        if not isinstance(value, dict):
            return
        snapshots.append(
            {
                "label": label,
                "vas": [int(item) for item in value.get("original_va_snapshot", [])],
                "state": value.get("lifecycle_state"),
            }
        )

    append("after_load", result.get("static_alias_diagnostics_after_load"))
    for index, cycle in enumerate(result.get("cycles", [])):
        if not isinstance(cycle, dict):
            continue
        append(
            f"cycle_{index}_after_prefill",
            cycle.get("static_alias_diagnostics_after_prefill"),
        )
        append(
            f"cycle_{index}_after_decode",
            cycle.get("static_alias_diagnostics_after_decode"),
        )
    return snapshots


def validate_residency(
    result: dict[str, Any] | None,
    *,
    expected_bytes: int,
    cpu_bytes: int,
    gpu_bytes: int,
    label: str,
) -> None:
    require(isinstance(result, dict) and result.get("ok") is True, f"{label} absent")
    require(int(result.get("queried_bytes", -1)) == expected_bytes, f"{label} queried")
    require(int(result.get("cpu_bytes", -1)) == cpu_bytes, f"{label} CPU bytes")
    require(int(result.get("gpu_bytes", -1)) == gpu_bytes, f"{label} GPU bytes")
    require(int(result.get("any_bytes", -1)) == expected_bytes, f"{label} any bytes")
    require(int(result.get("unknown_bytes", -1)) == 0, f"{label} unknown bytes")


def validate_alias_contract(
    result: dict[str, Any],
    diagnostics: dict[str, Any],
    driver: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    require(diagnostics.get("enabled") is True, "alias diagnostics disabled")
    require(diagnostics.get("backend") == "uvm_original_va", "backend differs")
    require(diagnostics.get("lifecycle_state") == "sealed", "alias is not sealed")
    require(int(diagnostics.get("session_id", 0)) > 0, "alias session absent")
    require(int(diagnostics.get("arena_count", 0)) == 4, "arena count differs")
    require(int(diagnostics.get("backing_bytes", 0)) == BACKING_BYTES, "backing differs")
    require(
        int(diagnostics.get("selected_expert_count", 0)) == SELECTED_EXPERT_COUNT,
        "selected expert count differs",
    )
    require(
        int(diagnostics.get("selected_parameter_count", 0)) == PARAMETER_COUNT,
        "selected Parameter count differs",
    )
    require(
        int(diagnostics.get("selected_parameter_bytes_per_forward", 0))
        == SELECTED_BYTES,
        "H2D bytes per forward differ",
    )
    require(
        int(diagnostics.get("full_parameter_bytes", 0)) == FULL_PARAMETER_BYTES,
        "full Parameter geometry differs",
    )
    require(
        int(diagnostics.get("consumer_range_count", 0)) == CONSUMER_RANGE_COUNT,
        "cold consumer range count differs",
    )
    require(
        len(diagnostics.get("original_va_snapshot", [])) == PARAMETER_COUNT,
        "original Parameter VA count differs",
    )
    require(
        all(arena.get("pte_mode") == "4K" for arena in diagnostics.get("arenas", [])),
        "partial aliases did not use 4K PTEs",
    )
    validate_residency(
        diagnostics.get("cold_residency_result"),
        expected_bytes=SELECTED_BYTES,
        cpu_bytes=SELECTED_BYTES,
        gpu_bytes=0,
        label="cold residency",
    )
    validate_residency(
        diagnostics.get("hot_residency_result"),
        expected_bytes=HOT_BYTES,
        cpu_bytes=0,
        gpu_bytes=HOT_BYTES,
        label="hot residency",
    )

    seal = diagnostics.get("seal_result", {})
    require(seal.get("ok") is True and seal.get("sealed") is True, "SEAL failed")
    require(int(seal.get("alias_count", 0)) == ALIAS_COUNT, "alias count differs")
    require(int(seal.get("alias_bytes", 0)) == ALIAS_BYTES, "alias bytes differ")
    require(int(seal.get("backing_bytes", 0)) == BACKING_BYTES, "driver backing differs")
    require(driver.get("session_count") == 1, "driver session count differs")
    sessions = driver.get("sessions", [])
    require(len(sessions) == 1 and sessions[0].get("sealed") is True, "driver session")

    copy_trace_enabled = os.environ.get("VLLM_CPU_OFFLOAD_COPY_TRACE") == "1"
    parameter_records = [
        item for item in records if item.get("event") == "cpu_offload_parameter"
    ]
    forward_records = [
        item for item in records if item.get("event") == "cpu_offload_copy_forward"
    ]
    # Parameter records are one-time initialization certificates and remain
    # enabled when per-forward copy tracing is disabled.
    require(len(parameter_records) == PARAMETER_COUNT, "manifest Parameter count differs")
    require(
        sum(int(item.get("bytes", 0)) for item in parameter_records) == SELECTED_BYTES,
        "manifest selected bytes differ",
    )
    require(
        sum(int(item.get("parameter_bytes", 0)) for item in parameter_records)
        == FULL_PARAMETER_BYTES,
        "manifest full bytes differ",
    )
    require(
        all(len(item.get("graph_static_alias_expert_ids") or []) == 5 for item in parameter_records),
        "manifest expert selection differs",
    )
    if copy_trace_enabled:
        require(forward_records, "copy-forward trace absent")
        require(
            all(int(item.get("copy_bytes", 0)) == LAYER_COPY_BYTES for item in forward_records),
            "one layer copied bytes outside its five cold experts",
        )
        require(
            all(
                int(item.get("graph_static_alias_actual_h2d_bytes", 0))
                == LAYER_COPY_BYTES
                for item in forward_records
            ),
            "one alias forward submitted an unexpected H2D size",
        )
    else:
        require(not forward_records, "forward trace present with copy trace disabled")

    snapshots = collect_va_snapshots(result)
    fingerprints = [tuple(item["vas"]) for item in snapshots]
    require(fingerprints and all(len(value) == PARAMETER_COUNT for value in fingerprints), "VA evidence absent")
    require(len(set(fingerprints)) == 1, "a Parameter VA changed during execution")
    require(all(item["state"] == "sealed" for item in snapshots), "alias state changed")
    return {
        "parameter_va_snapshots": snapshots,
        "parameter_va_fingerprint": list(fingerprints[0]),
        "copy_trace_enabled": copy_trace_enabled,
        "forward_trace_count": len(forward_records),
    }


def selected_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith("VLLM_CPU_OFFLOAD_")
        or key.startswith("VLLM_UVM_")
        or key.startswith("VLLM_MOE_TOKEN_EXPERT_TRACE")
        or key in {"CUDA_VISIBLE_DEVICES", "VLLM_ATTENTION_BACKEND"}
    }


def main() -> int:
    arm = os.environ.get("VLLM_COLD5_ARM", "").strip().upper()
    require(arm in VALID_ARMS, f"invalid VLLM_COLD5_ARM={arm!r}")
    output_path = output_path_from_argv()
    base = load_module("_cold5_prefill_base", BASE_RUNNER)

    import torch
    import vllm
    from vllm.device_allocator.uvm_static_alias import get_static_alias_diagnostics
    from vllm.model_executor.models.utils import (
        get_cpu_offload_graph_static_alias_diagnostics,
    )
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
    contract_errors: list[str] = []
    alias_evidence: dict[str, Any] = {}
    graph_evidence: dict[str, int] = {}
    try:
        active = get_cpu_offload_graph_static_alias_diagnostics()
        driver_active = get_static_alias_diagnostics()
        records = read_manifest(
            Path(result.get("config", {}).get("offload_manifest", ""))
        )
        if base_error is None:
            require(result.get("status") == "ok", "base benchmark did not finish")
            graph_evidence = validate_graph_contract(result, arm)
            if arm in {"A", "G"}:
                alias_evidence = validate_alias_contract(
                    result, active, driver_active, records
                )
            else:
                require(active.get("enabled") is False, "control enabled alias")
                require(driver_active.get("session_count") == 0, "control has session")

        if live_llms:
            live_llms[-1].llm_engine.engine_core.shutdown()
        elif arm in {"A", "G"} and int(active.get("session_id", 0)) > 0:
            contract_errors.append("no tracked LLM owns the active alias session")
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        post = get_cpu_offload_graph_static_alias_diagnostics()
        driver_post = get_static_alias_diagnostics()
        require(int(post.get("session_id", 0)) == 0, "session survived shutdown")
        require(driver_post.get("session_count") == 0, "driver session survived shutdown")
        if arm in {"A", "G"}:
            require(shutdown_results, "alias shutdown result absent")
            destroy = shutdown_results[-1].get("destroy", {})
            require(destroy.get("ok") is True, "cmd99 destroy failed")
            require(int(destroy.get("released_alias_count", 0)) == ALIAS_COUNT, "released aliases differ")
            require(int(destroy.get("released_backing_bytes", 0)) == BACKING_BYTES, "released backing differs")
    except Exception as exc:
        contract_errors.append(repr(exc))
        contract_errors.append(traceback.format_exc())

    result["base_schema"] = result.get("schema")
    result["schema"] = "vllm_uvm_cold5_alias_prefill_steps_v1"
    result["arm"] = arm
    result["graph_contract"] = graph_evidence
    result["alias_contract"] = alias_evidence
    result["mechanism_contract"] = {
        "diagnostics_before_shutdown": active,
        "driver_diagnostics_before_shutdown": driver_active,
    }
    result["shutdown_contract"] = {
        "model_runner_shutdown_results": shutdown_results,
        "diagnostics_after_shutdown": post,
        "driver_diagnostics_after_shutdown": driver_post,
    }
    result["reproducibility_environment"] = selected_environment()
    result["wrapper_contract_errors"] = contract_errors
    if contract_errors and base_error is None:
        result["status"] = "contract_failed"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    GPUModelRunner.shutdown_cpu_offload_graph_static_alias = original_shutdown
    vllm.LLM = original_llm
    if base_error is not None:
        raise base_error.with_traceback(base_traceback)
    if contract_errors:
        raise ContractError("; ".join(contract_errors))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
