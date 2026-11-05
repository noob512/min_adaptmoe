#!/usr/bin/env python3
"""Run the existing strict vLLM step benchmark with original-VA audits.

The data-plane benchmark deliberately stays in
``vllm_p3_static_alias_graph_integration/benchmark_full_graph_steps.py`` so
that the VMM and UVM experiments use the same scheduler, fixture, and timing
code.  This wrapper adds the contracts which are specific to the custom UVM
backend and performs cmd99 teardown in the benchmark process after the vLLM
engine (and therefore its CUDA Graphs) has been released.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import traceback
from collections import Counter
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
DUAL_BANK_BYTES = 1_207_959_552
LOGICAL_BYTES = 2_415_919_104
PARAMETER_COUNT = 8


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def env_true(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_base_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_vllm_static_alias_base_benchmark", BASE_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base runner: {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def output_path_from_argv() -> Path:
    try:
        index = sys.argv.index("--output")
        value = sys.argv[index + 1]
    except (ValueError, IndexError) as exc:
        raise RuntimeError("the wrapped benchmark requires --output PATH") from exc
    return Path(value).resolve()


def load_result(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "wrapper_exception",
            "error": f"could not read base result: {exc!r}",
        }
    if not isinstance(value, dict):
        return {
            "status": "wrapper_exception",
            "error": "base result is not a JSON object",
        }
    return value


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(
                f"invalid manifest JSON at {path}:{line_number}: {exc}"
            ) from exc
        require(isinstance(value, dict), f"manifest line {line_number} is not an object")
        records.append(value)
    return records


def manifest_contract(records: list[dict[str, Any]]) -> dict[str, Any]:
    event_counts = Counter(str(record.get("event")) for record in records)
    parameters = [
        record
        for record in records
        if record.get("event") == "cpu_offload_parameter"
    ]
    selected = [
        record
        for record in parameters
        if record.get("graph_static_alias_backend") == "uvm_original_va"
    ]
    finalized = [
        record
        for record in records
        if record.get("event") == "cpu_offload_uvm_static_alias_finalized"
    ]
    destroyed = [
        record
        for record in records
        if record.get("event") == "cpu_offload_uvm_static_alias_destroyed"
    ]
    original_vas = sorted(
        {
            int(record["uvm_original_parameter_va"])
            for record in selected
            if isinstance(record.get("uvm_original_parameter_va"), int)
        }
    )
    consumer_vas = sorted(
        {
            int(record["graph_static_alias_consumer_va"])
            for record in selected
            if isinstance(record.get("graph_static_alias_consumer_va"), int)
        }
    )
    upload_vas = sorted(
        {
            int(record["graph_static_alias_upload_va"])
            for record in selected
            if isinstance(record.get("graph_static_alias_upload_va"), int)
        }
    )
    return {
        "record_count": len(records),
        "event_counts": dict(sorted(event_counts.items())),
        "parameter_count": len(parameters),
        "parameter_logical_bytes": sum(int(item.get("bytes", 0)) for item in parameters),
        "uvm_original_parameter_count": len(selected),
        "original_parameter_vas": original_vas,
        "consumer_vas": consumer_vas,
        "upload_vas": upload_vas,
        "original_va_retained_count": sum(
            record.get("uvm_original_parameter_va_retained") is True
            and record.get("uvm_original_parameter_va")
            == record.get("graph_static_alias_consumer_va")
            for record in selected
        ),
        "managed_parameter_count": sum(
            record.get("uvm_offloading") is True for record in selected
        ),
        "backend_values": sorted(
            {str(record.get("graph_static_alias_backend")) for record in selected}
        ),
        "finalize_records": finalized,
        "destroy_records": destroyed,
    }


def replay_contract(result: dict[str, Any]) -> dict[str, Any]:
    replay_count = 0
    decode_step_count = 0
    output_hashes: list[dict[str, Any]] = []
    for cycle in result.get("cycles", []):
        if not isinstance(cycle, dict):
            continue
        digest = cycle.get("output_token_ids_sha256")
        output_hashes.append(
            {
                "batch_size": cycle.get("batch_size"),
                "role": cycle.get("role"),
                "role_index": cycle.get("role_index"),
                "sha256": digest,
            }
        )
        for category in ("warmup_decode_steps", "measured_decode_steps"):
            for step in cycle.get(category, []):
                decode_step_count += 1
                delta = step.get("graph_wrapper_delta", {})
                replay_count += int(delta.get("full_replay", 0))
    return {
        "decode_step_count": decode_step_count,
        "full_graph_replay_count": replay_count,
        "output_hashes": output_hashes,
    }


def validate_active_contract(
    result: dict[str, Any],
    diagnostics: dict[str, Any],
    driver_diagnostics: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    require(result.get("status") == "ok", "base vLLM benchmark did not finish")
    require(diagnostics.get("enabled") is True, "static alias is not enabled")
    require(
        diagnostics.get("backend") == "uvm_original_va",
        f"wrong backend: {diagnostics.get('backend')!r}",
    )
    require(diagnostics.get("lifecycle_state") == "sealed", "topology is not sealed")
    require(diagnostics.get("sealed") is True, "sealed diagnostic is false")
    require(int(diagnostics.get("session_id", 0)) > 0, "session id is missing")
    require(int(diagnostics.get("arena_count", 0)) == 4, "expected four arenas")
    require(
        int(diagnostics.get("backing_bytes", 0)) == DUAL_BANK_BYTES,
        "backing is not exactly 1.125 GiB",
    )
    arenas = diagnostics.get("arenas")
    require(isinstance(arenas, list) and len(arenas) == 4, "arena list differs")
    consumer_vas = [
        int(va) for arena in arenas for va in arena.get("consumer_vas", [])
    ]
    upload_vas = [int(arena["upload_va"]) for arena in arenas]
    require(len(consumer_vas) == 8, "expected eight consumer/original VAs")
    require(len(set(consumer_vas)) == 8, "consumer/original VAs are not unique")
    require(len(set(upload_vas)) == 4, "upload VAs are not unique")
    require(not (set(consumer_vas) & set(upload_vas)), "upload and consumer VAs overlap")
    require(
        manifest.get("parameter_count") == PARAMETER_COUNT
        and manifest.get("parameter_logical_bytes") == LOGICAL_BYTES,
        "exact Qwen parameter selection differs",
    )
    require(
        manifest.get("uvm_original_parameter_count") == PARAMETER_COUNT,
        "not every selected parameter used the UVM original-VA backend",
    )
    require(
        manifest.get("original_va_retained_count") == PARAMETER_COUNT,
        "at least one parameter allocation VA was replaced",
    )
    require(
        manifest.get("managed_parameter_count") == PARAMETER_COUNT,
        "at least one original parameter is not managed",
    )
    require(
        manifest.get("original_parameter_vas") == sorted(consumer_vas),
        "manifest original VAs differ from live consumer VAs",
    )
    require(
        manifest.get("upload_vas") == sorted(upload_vas),
        "manifest upload VAs differ from live upload VAs",
    )
    require(len(manifest.get("finalize_records", [])) == 1, "missing finalize record")
    finalization = manifest["finalize_records"][0]
    require(finalization.get("ok") is True, "finalization was not successful")
    require(finalization.get("sealed") is True, "finalization did not seal")
    require(
        int(finalization.get("session_id", 0)) == int(diagnostics["session_id"]),
        "manifest and live session ids differ",
    )
    require(
        int(finalization.get("backing_bytes", 0)) == DUAL_BANK_BYTES,
        "finalize backing differs",
    )
    require(
        diagnostics.get("preallocate_result", {}).get("ok") is True,
        "slot preallocation did not succeed",
    )
    maps = diagnostics.get("map_results")
    require(isinstance(maps, (list, tuple)) and bool(maps), "MAP evidence is absent")
    require(all(item.get("ok") is True for item in maps), "one or more MAP batches failed")
    require(
        diagnostics.get("seal_result", {}).get("ok") is True,
        "SEAL evidence is absent or failed",
    )
    sessions = driver_diagnostics.get("sessions")
    require(
        driver_diagnostics.get("session_count") == 1
        and isinstance(sessions, list)
        and len(sessions) == 1,
        "userspace driver session cache differs",
    )
    session = sessions[0]
    require(session.get("sealed") is True, "userspace session is not sealed")
    require(
        int(session.get("session_id", 0)) == int(diagnostics["session_id"]),
        "userspace and model session ids differ",
    )
    replay = replay_contract(result)
    require(replay["decode_step_count"] > 0, "no decode steps were measured")
    require(
        replay["full_graph_replay_count"] == replay["decode_step_count"],
        "not every decode step used one FULL Graph replay",
    )
    require(
        all(
            isinstance(item["sha256"], str) and len(item["sha256"]) == 64
            for item in replay["output_hashes"]
        ),
        "one or more output hashes are missing",
    )


def validate_shutdown_contract(
    cleanup: dict[str, Any],
    post: dict[str, Any],
    driver_post: dict[str, Any],
) -> None:
    require(cleanup.get("ok") is True, f"cmd99 cleanup failed: {cleanup!r}")
    require(post.get("session_id") == 0, "model diagnostics retained a session")
    require(post.get("sealed") is False, "model diagnostics remained sealed")
    require(post.get("lifecycle_state") == "remote", "post-destroy state is not remote")
    require(driver_post.get("session_count") == 0, "userspace session cache was not cleared")


def selected_environment() -> dict[str, str]:
    prefixes = ("VLLM_CPU_OFFLOAD_", "VLLM_UVM_")
    exact = {
        "CUDA_VISIBLE_DEVICES",
        "VLLM_USE_UVM",
        "VLLM_ENABLE_V1_MULTIPROCESSING",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    }
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(prefixes) or key in exact
    }


def main() -> int:
    output_path = output_path_from_argv()
    base = load_base_runner()
    # Hold the exact LLM instance created inside the shared benchmark.  Python
    # garbage collection is not a CUDA Graph lifetime boundary; explicit
    # EngineCore shutdown below is what reaches Worker -> GPUModelRunner and
    # orders Graph.reset() before cmd99.
    import vllm
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    live_llms: list[Any] = []
    graph_shutdown_results: list[dict[str, Any]] = []
    original_llm_class = vllm.LLM
    original_runner_shutdown = (
        GPUModelRunner.shutdown_cpu_offload_graph_static_alias
    )

    class TrackedLLM(original_llm_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            live_llms.append(self)

    def tracked_runner_shutdown(self):
        value = original_runner_shutdown(self)
        graph_shutdown_results.append(dict(value))
        return value

    vllm.LLM = TrackedLLM
    GPUModelRunner.shutdown_cpu_offload_graph_static_alias = (
        tracked_runner_shutdown
    )
    base_error: BaseException | None = None
    base_traceback = None
    return_code = 1
    try:
        return_code = int(base.run())
    except BaseException as exc:  # cleanup is required even on load/capture failure
        base_error = exc
        base_traceback = exc.__traceback__

    result = load_result(output_path)
    requested = env_true("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS")
    requested_backend = os.environ.get(
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND", "vmm"
    ).strip().lower()
    uvm_requested = requested and requested_backend in {
        "uvm",
        "original_va",
        "uvm_original_va",
    }

    active: dict[str, Any] = {}
    driver_active: dict[str, Any] = {}
    cleanup: dict[str, Any] = {
        "ok": True,
        "operation": "destroy",
        "reason": "backend_not_active",
    }
    post: dict[str, Any] = {}
    driver_post: dict[str, Any] = {}
    contract_errors: list[str] = []

    try:
        from vllm.device_allocator.uvm_static_alias import (
            get_static_alias_diagnostics,
        )
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
        records = load_manifest(manifest_path) if str(manifest_path) else []
        before_manifest = manifest_contract(records)

        if uvm_requested and base_error is None:
            try:
                validate_active_contract(result, active, driver_active, before_manifest)
            except Exception as exc:
                contract_errors.append(f"active_contract: {exc!r}")

        # Explicitly invoke the in-process EngineCore shutdown. This reaches
        # GPUModelRunner while it still owns concrete Graph entries; its
        # teardown synchronizes, resets every CUDAGraph, then calls cmd99.
        if live_llms:
            live_llms[-1].llm_engine.engine_core.shutdown()
        elif uvm_requested and int(active.get("session_id", 0)) > 0:
            contract_errors.append(
                "explicit_shutdown: no tracked LLM owns the active session"
            )
        gc.collect()
        try:
            import torch

            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
        except Exception as exc:
            contract_errors.append(f"pre_destroy_synchronize: {exc!r}")

        if graph_shutdown_results:
            cleanup = graph_shutdown_results[-1]
        post = get_cpu_offload_graph_static_alias_diagnostics()
        driver_post = get_static_alias_diagnostics()
        records = load_manifest(manifest_path) if str(manifest_path) else []
        after_manifest = manifest_contract(records)
        if uvm_requested and int(active.get("session_id", 0)) > 0:
            try:
                validate_shutdown_contract(cleanup, post, driver_post)
                require(
                    len(after_manifest.get("destroy_records", [])) == 1,
                    "manifest does not contain exactly one destroy record",
                )
            except Exception as exc:
                contract_errors.append(f"shutdown_contract: {exc!r}")
    except Exception as exc:
        before_manifest = {}
        after_manifest = {}
        contract_errors.append(f"wrapper_cleanup: {exc!r}")
        contract_errors.append(traceback.format_exc())

    result["base_schema"] = result.get("schema")
    result["schema"] = "vllm_uvm_original_va_graph_steps_v1"
    result["original_va_contract"] = {
        "requested": uvm_requested,
        "expected_backing_bytes": DUAL_BANK_BYTES,
        "expected_logical_bytes": LOGICAL_BYTES,
        "diagnostics_before_destroy": active,
        "driver_diagnostics_before_destroy": driver_active,
        "manifest_before_destroy": before_manifest,
        "graph_replay": replay_contract(result),
    }
    result["shutdown_contract"] = {
        "graphs_destroyed_before_cmd99": bool(
            not uvm_requested
            or (
                graph_shutdown_results
                and int(
                    graph_shutdown_results[-1].get(
                        "destroyed_graph_count", 0
                    )
                )
                > 0
            )
        ),
        "model_runner_shutdown_results": graph_shutdown_results,
        "cleanup_result": cleanup,
        "diagnostics_after_destroy": post,
        "driver_diagnostics_after_destroy": driver_post,
        "manifest_after_destroy": after_manifest,
        "process_gpu_clean_checked_by_outer_harness": True,
        "stock_module_restore_checked_by_outer_harness": True,
    }
    result["reproducibility_environment"] = selected_environment()
    result["wrapper_contract_errors"] = contract_errors
    if contract_errors and base_error is None:
        result["status"] = "contract_failed"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    GPUModelRunner.shutdown_cpu_offload_graph_static_alias = (
        original_runner_shutdown
    )
    vllm.LLM = original_llm_class

    if base_error is not None:
        raise base_error.with_traceback(base_traceback)
    if contract_errors:
        raise ContractError("; ".join(contract_errors))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
