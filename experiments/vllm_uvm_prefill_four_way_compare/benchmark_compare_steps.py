#!/usr/bin/env python3
"""Run the shared vLLM phase benchmark with five explicit data paths.

The four prefill mechanisms are native copy-on-forward (N), original-VA
static-alias eager (A), original-VA static-alias FULL Graph (G), and audited
original-VA CPU remote READ eager (R).  C adds the requested remote-decode
FULL Graph control. F is a native copy-on-forward FULL-Graph correctness
control for G. The outer harness selects exactly one arm per process.
"""

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
SHARED_WRAPPER = (
    REPO_ROOT
    / "experiments"
    / "vllm_uvm_original_va_graph_integration"
    / "benchmark_original_va_graph_steps.py"
)
DUAL_BANK_BYTES = 1_207_959_552
LOGICAL_BYTES = 2_415_919_104
PARAMETER_COUNT = 8
VALID_ARMS = frozenset({"N", "A", "G", "R", "C", "F"})


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


def load_result(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "wrapper_exception", "error": repr(exc)}
    return value if isinstance(value, dict) else {"status": "wrapper_exception"}


def graph_contract(result: dict[str, Any], arm: str) -> dict[str, Any]:
    prefill_deltas: list[dict[str, int]] = []
    decode_deltas: list[dict[str, int]] = []
    for cycle in result.get("cycles", []):
        if not isinstance(cycle, dict):
            continue
        prefill_deltas.append(dict(cycle.get("prefill_graph_wrapper_delta", {})))
        for category in ("warmup_decode_steps", "measured_decode_steps"):
            for step in cycle.get(category, []):
                decode_deltas.append(dict(step.get("graph_wrapper_delta", {})))

    expected_prefill_graph = arm in {"G", "F"}
    expected_decode_graph = arm in {"G", "C", "F"}
    require(prefill_deltas, "no prefill Graph evidence")
    require(decode_deltas, "no decode Graph evidence")
    for delta in prefill_deltas:
        replays = int(delta.get("full_replay", 0))
        require(
            replays == int(expected_prefill_graph),
            f"prefill Graph replay differs for arm {arm}: {delta}",
        )
    for delta in decode_deltas:
        replays = int(delta.get("full_replay", 0))
        require(
            replays == int(expected_decode_graph),
            f"decode Graph replay differs for arm {arm}: {delta}",
        )
    return {
        "expected_prefill_graph": expected_prefill_graph,
        "expected_decode_graph": expected_decode_graph,
        "prefill_step_count": len(prefill_deltas),
        "prefill_full_replay_count": sum(
            int(item.get("full_replay", 0)) for item in prefill_deltas
        ),
        "decode_step_count": len(decode_deltas),
        "decode_full_replay_count": sum(
            int(item.get("full_replay", 0)) for item in decode_deltas
        ),
    }


def parameter_va_snapshots(result: dict[str, Any]) -> dict[str, Any]:
    snapshots: list[dict[str, Any]] = []

    def append(label: str, diagnostics: Any) -> None:
        if not isinstance(diagnostics, dict):
            return
        vas = diagnostics.get("original_va_snapshot", [])
        snapshots.append(
            {
                "label": label,
                "vas": [int(value) for value in vas],
                "lifecycle_state": diagnostics.get("lifecycle_state"),
                "session_id": int(diagnostics.get("session_id", 0)),
                "active_driver_backing_bytes": int(
                    diagnostics.get("active_driver_backing_bytes", 0)
                ),
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
    return {"snapshots": snapshots}


def validate_active_contract(
    *,
    arm: str,
    result: dict[str, Any],
    diagnostics: dict[str, Any],
    driver: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    require(result.get("status") == "ok", "base benchmark did not finish")
    graph_contract(result, arm)
    require(
        manifest.get("parameter_count") == PARAMETER_COUNT
        and manifest.get("parameter_logical_bytes") == LOGICAL_BYTES,
        "selected parameter geometry differs",
    )

    if arm in {"N", "F"}:
        require(diagnostics.get("enabled") is False, "native arm enabled alias")
        require(driver.get("session_count") == 0, "native arm retained UVM session")
        return

    require(diagnostics.get("enabled") is True, "custom arm did not enable alias")
    require(
        diagnostics.get("backend") == "uvm_original_va",
        "custom arm did not use original-VA UVM backend",
    )
    require(
        manifest.get("uvm_original_parameter_count") == PARAMETER_COUNT,
        "not every selected parameter used original-VA UVM",
    )
    require(
        manifest.get("original_va_retained_count") == PARAMETER_COUNT,
        "one or more original Parameter VAs were replaced",
    )
    require(
        len(manifest.get("original_parameter_vas", [])) == PARAMETER_COUNT,
        "original Parameter VA fingerprint differs",
    )

    if arm in {"A", "G"}:
        require(diagnostics.get("lifecycle_state") == "sealed", "alias not sealed")
        require(int(diagnostics.get("session_id", 0)) > 0, "alias session absent")
        require(
            int(diagnostics.get("active_driver_backing_bytes", 0))
            == DUAL_BANK_BYTES,
            "active alias backing differs",
        )
        require(driver.get("session_count") == 1, "driver session differs")
    else:
        require(diagnostics.get("lifecycle_state") == "remote", "remote state absent")
        require(int(diagnostics.get("session_id", 0)) == 0, "remote retained session")
        require(
            int(diagnostics.get("active_driver_backing_bytes", 0)) == 0,
            "remote retained active backing",
        )
        require(diagnostics.get("remote_map_active") is True, "remote map inactive")
        remote = diagnostics.get("remote_map_result", {})
        require(remote.get("ok") is True, "audited remote map failed")
        require(int(remote.get("success_count", 0)) == PARAMETER_COUNT, "remote count")
        for field in (
            "requested_bytes",
            "cpu_resident_bytes",
            "remote_mapped_bytes",
            "covered_bytes",
        ):
            require(int(remote.get(field, 0)) == LOGICAL_BYTES, f"remote {field}")
        require(int(remote.get("unknown_bytes", -1)) == 0, "remote unknown bytes")
        require(driver.get("session_count") == 0, "remote retained driver session")


def selected_environment() -> dict[str, str]:
    prefixes = ("VLLM_CPU_OFFLOAD_", "VLLM_UVM_", "VLLM_PREFILL_COMPARE_")
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
    arm = os.environ.get("VLLM_PREFILL_COMPARE_ARM", "").strip().upper()
    require(arm in VALID_ARMS, f"invalid VLLM_PREFILL_COMPARE_ARM={arm!r}")
    output_path = output_path_from_argv()
    base = load_module("_prefill_compare_base", BASE_RUNNER)
    shared = load_module("_prefill_compare_shared", SHARED_WRAPPER)

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

    result = load_result(output_path)
    active: dict[str, Any] = {}
    driver_active: dict[str, Any] = {}
    post: dict[str, Any] = {}
    driver_post: dict[str, Any] = {}
    before_manifest: dict[str, Any] = {}
    after_manifest: dict[str, Any] = {}
    contract_errors: list[str] = []

    try:
        from vllm.device_allocator.uvm_static_alias import get_static_alias_diagnostics
        from vllm.model_executor.models.utils import (
            get_cpu_offload_graph_static_alias_diagnostics,
        )

        active = get_cpu_offload_graph_static_alias_diagnostics()
        driver_active = get_static_alias_diagnostics()
        manifest_path = Path(
            result.get("config", {}).get(
                "offload_manifest",
                os.environ.get("VLLM_CPU_OFFLOAD_MANIFEST_FILE", ""),
            )
        )
        records = shared.load_manifest(manifest_path)
        before_manifest = shared.manifest_contract(records)
        if base_error is None:
            try:
                validate_active_contract(
                    arm=arm,
                    result=result,
                    diagnostics=active,
                    driver=driver_active,
                    manifest=before_manifest,
                )
            except Exception as exc:
                contract_errors.append(f"active_contract: {exc!r}")

        if live_llms:
            live_llms[-1].llm_engine.engine_core.shutdown()
        elif arm != "N" and int(active.get("session_id", 0)) > 0:
            contract_errors.append("no tracked LLM owns the active UVM session")
        gc.collect()
        try:
            import torch

            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
        except Exception as exc:
            contract_errors.append(f"post_shutdown_sync: {exc!r}")

        post = get_cpu_offload_graph_static_alias_diagnostics()
        driver_post = get_static_alias_diagnostics()
        records = shared.load_manifest(manifest_path)
        after_manifest = shared.manifest_contract(records)
        if arm != "N":
            require(int(post.get("session_id", 0)) == 0, "session survived shutdown")
            require(
                int(post.get("active_driver_backing_bytes", 0)) == 0,
                "backing survived shutdown",
            )
            require(driver_post.get("session_count") == 0, "driver cache survived")
    except Exception as exc:
        contract_errors.append(f"wrapper_cleanup: {exc!r}")
        contract_errors.append(traceback.format_exc())

    result["base_schema"] = result.get("schema")
    result["schema"] = "vllm_uvm_prefill_four_way_steps_v1"
    result["arm"] = arm
    result["graph_contract"] = (
        graph_contract(result, arm) if base_error is None else {}
    )
    result["parameter_va_snapshots"] = parameter_va_snapshots(result)
    result["mechanism_contract"] = {
        "diagnostics_before_shutdown": active,
        "driver_diagnostics_before_shutdown": driver_active,
        "manifest_before_shutdown": before_manifest,
    }
    result["shutdown_contract"] = {
        "model_runner_shutdown_results": shutdown_results,
        "diagnostics_after_shutdown": post,
        "driver_diagnostics_after_shutdown": driver_post,
        "manifest_after_shutdown": after_manifest,
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
