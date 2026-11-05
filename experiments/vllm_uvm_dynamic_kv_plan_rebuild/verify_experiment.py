#!/usr/bin/env python3
"""Verify the same-process 2 GiB -> 4 GiB -> 2 GiB plan rebuild."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_EXPECTED = (
    ("4g_high_pressure", "high", 4, 4_303_355_904),
    ("2g_low_pressure", "low", 1, 2_264_924_160),
)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_root.resolve()
    result = load_json(root / "result.json")
    config_path = root / "run_config.json"
    run_config = load_json(config_path) if config_path.is_file() else {
        "initial_selected_bytes": 2_264_924_160,
        "expected_parameter_count": 96,
        "expected_full_parameter_bytes": 28_991_029_248,
        "physical_kv_allocation_bytes": 167_772_160,
        "dynamic_records": [
            {
                "label": label,
                "expected_selected_bytes": selected_bytes,
                "batch_size": batch,
                "expected_pressure_class": pressure,
            }
            for label, pressure, batch, selected_bytes in DEFAULT_EXPECTED
        ],
    }
    expected_records = tuple(
        (
            str(record["label"]),
            str(record["expected_pressure_class"]),
            int(record["batch_size"]),
            int(record["expected_selected_bytes"]),
        )
        for record in run_config["dynamic_records"]
    )
    initial_selected_bytes = int(run_config["initial_selected_bytes"])
    expected_parameter_count = int(run_config["expected_parameter_count"])
    physical_kv_bytes = int(run_config["physical_kv_allocation_bytes"])
    parameter_va_mode = str(
        run_config.get("parameter_va_mode", "stable_rebuild_recapture")
    )
    require(
        parameter_va_mode
        in ("stable_rebuild_recapture", "nonstable_reallocate_recapture"),
        "unknown Parameter VA mode",
    )
    require((root / "returncode.txt").read_text().strip() == "0", "runner rc")
    require(result.get("status") == "ok", "runner status is not ok")
    require(result.get("arm") == "D", "experiment did not use arm D")
    require(
        result.get("config", {}).get("phase_mode")
        == "prefill_graph_remote_decode",
        "phase mode differs",
    )
    require(
        int(result.get("config", {}).get("expected_selected_bytes", -1))
        == initial_selected_bytes,
        "initial selected bytes differ from run contract",
    )

    initial_vas = result["parameter_va_snapshots"]["after_load"]["vas"]
    require(
        len(initial_vas) == expected_parameter_count,
        "initial Parameter VA count differs",
    )
    dynamic = result.get("dynamic_plan_rebuilds")
    require(
        isinstance(dynamic, list) and len(dynamic) == len(expected_records),
        "dynamic generations",
    )
    initial_prefix = result["outputs"][0]["output_token_ids"][:2]
    summaries: list[dict[str, Any]] = []
    expected_generation_vas = initial_vas

    for generation, (record, expected) in enumerate(zip(dynamic, expected_records), 1):
        label, pressure_class, batch, selected_bytes = expected
        require(record.get("generation") == generation, f"g{generation}: generation")
        require(record.get("label") == label, f"g{generation}: label")
        require(
            int(record.get("expected_selected_bytes", -1)) == selected_bytes,
            f"g{generation}: selected bytes",
        )
        pressure = record.get("pressure", {})
        require(
            pressure.get("previous_request_finished") is True
            and int(pressure.get("scheduler_unfinished_request_count", -1)) == 0,
            f"g{generation}: request boundary was not idle",
        )
        require(
            pressure.get("pressure_class") == pressure_class
            and int(pressure.get("next_batch_size", -1)) == batch,
            f"g{generation}: pressure decision",
        )
        predicted_tokens = int(pressure.get("predicted_kv_tokens", -1))
        threshold = int(pressure.get("pressure_threshold_tokens", -1))
        require(
            (predicted_tokens >= threshold) == (pressure_class == "high"),
            f"g{generation}: pressure threshold contract",
        )
        require(
            int(pressure.get("physical_kv_allocation_bytes", -1)) == physical_kv_bytes
            and pressure.get("physical_kv_allocation_changes_at_boundary") is False,
            f"g{generation}: physical/logical KV distinction",
        )
        require(
            0.0 <= float(pressure.get("logical_pool_usage_before", -1.0)) <= 1.0,
            f"g{generation}: logical usage before",
        )
        logical = pressure.get("logical_policy_snapshot", {})
        require(
            int(logical.get("projected_used_blocks", -1))
            >= int(pressure.get("predicted_kv_blocks", -2)),
            f"g{generation}: projected KV blocks",
        )

        rebuild = record.get("rebuild", {})
        low_level = rebuild.get("rebuild", {})
        require(
            rebuild.get("ok") is True
            and low_level.get("ok") is True
            and int(low_level.get("generation", -1)) == generation,
            f"g{generation}: rebuild status",
        )
        require(
            int(low_level.get("selected_bytes", -1)) == selected_bytes,
            f"g{generation}: rebuilt byte count",
        )
        low_before = low_level.get("parameter_vas_before")
        low_after = low_level.get("parameter_vas_after")
        if parameter_va_mode == "stable_rebuild_recapture":
            require(
                rebuild.get("parameter_va_mode") == parameter_va_mode
                and low_before == low_after == initial_vas,
                f"g{generation}: low-level Parameter VAs",
            )
        else:
            require(
                rebuild.get("parameter_va_mode") == parameter_va_mode
                and isinstance(low_before, list)
                and isinstance(low_after, list)
                and len(low_before) == len(low_after) == expected_parameter_count
                and low_before == expected_generation_vas
                and all(before != after for before, after in zip(low_before, low_after))
                and low_level.get("all_parameter_vas_changed") is True,
                f"g{generation}: non-stable low-level Parameter VAs",
            )
            expected_generation_vas = low_after
        require(
            int(rebuild.get("destroyed_graph_count", -1)) == 4,
            f"g{generation}: destroyed Graph count",
        )
        require(
            int(record.get("graph_wrapper_delta", {}).get("full_capture", 0)) >= 4,
            f"g{generation}: Graph recapture count",
        )

        after_rebuild = record["diagnostics_after_rebuild"]["model"]
        after_request = record["diagnostics_after_request"]["model"]
        require(
            after_rebuild.get("lifecycle_state") == "sealed"
            and after_rebuild.get("remote_graph_captured") is True
            and after_rebuild.get("alias_graph_captured") is True
            and isinstance(
                after_rebuild.get("dual_graph_partition_certificate"), dict
            ),
            f"g{generation}: recaptured dual topology",
        )
        require(
            int(after_rebuild.get("selected_parameter_bytes_per_forward", -1))
            == selected_bytes,
            f"g{generation}: diagnostic selected bytes",
        )
        require(
            after_request.get("lifecycle_state") == "remote_parked"
            and after_request.get("remote_map_active") is True,
            f"g{generation}: post-request remote topology",
        )
        remote_map = after_request.get("remote_map_result", {})
        require(
            remote_map.get("ok") is True
            and int(remote_map.get("requested_bytes", -1)) == selected_bytes
            and int(remote_map.get("cpu_resident_bytes", -1)) >= selected_bytes
            and int(remote_map.get("unknown_bytes", -1)) == 0,
            f"g{generation}: remote map coverage",
        )
        snapshots = record["parameter_va_snapshots"]
        if parameter_va_mode == "stable_rebuild_recapture":
            require(
                snapshots["before_rebuild"]["vas"]
                == snapshots["after_rebuild"]["vas"]
                == snapshots["after_request"]["vas"]
                == initial_vas,
                f"g{generation}: Parameter VA snapshots",
            )
        else:
            # The model-registry snapshot intentionally retains the original
            # handles owned by vLLM's wrapper.  The active original-VA binding
            # is authoritative for this baseline and must expose the new VAs
            # through both post-rebuild and post-request diagnostics.
            require(
                after_rebuild.get("original_vas") == low_after
                and after_request.get("original_vas") == low_after,
                f"g{generation}: active non-stable Parameter VA diagnostics",
            )
        require(
            record["outputs"][0]["output_token_ids"] == initial_prefix,
            f"g{generation}: shared prompt output",
        )
        require(len(record.get("cutovers", [])) == 1, f"g{generation}: cutover")
        summaries.append(
            {
                "generation": generation,
                "label": label,
                "pressure_class": pressure_class,
                "batch_size": batch,
                "predicted_kv_tokens": predicted_tokens,
                "selected_bytes": selected_bytes,
                "rebuild_elapsed_ms": float(record["rebuild_elapsed_ms"]),
                "graph_reset_ms": float(rebuild["graph_reset_ns"]) / 1e6,
                "plan_rebuild_ms": float(rebuild["plan_rebuild_ns"]) / 1e6,
                "graph_capture_ms": float(rebuild["graph_capture_ns"]) / 1e6,
                "request_elapsed_ms": float(record["request_elapsed_ms"]),
                "output_sha256": record["output_token_ids_sha256"],
            }
        )

    final_diag = dynamic[-1]["diagnostics_after_request"]["model"]
    history = final_diag.get("plan_rebuild_history")
    require(isinstance(history, list) and len(history) == 2, "rebuild history")
    require(all(item.get("ok") is True for item in history), "rebuild history status")

    shutdown = result.get("shutdown_contract", {})
    shutdown_diag = shutdown.get("diagnostics_after_shutdown", {})
    model_shutdown = shutdown_diag.get("model", {})
    driver_shutdown = shutdown_diag.get("driver", {})
    require(not shutdown.get("cleanup_errors"), "shutdown cleanup errors")
    require(
        shutdown.get("controlled_hard_exit_after_cleanup") is True,
        "controlled post-cleanup exit contract is absent",
    )
    require(
        int(model_shutdown.get("session_id", -1)) == 0
        and int(model_shutdown.get("active_driver_backing_bytes", -1)) == 0,
        "model resources remain active after shutdown",
    )
    require(
        int(driver_shutdown.get("session_count", -1)) == 0,
        "driver sessions remain after shutdown",
    )

    kernel_text = (root / "kernel.since.log").read_text(encoding="utf-8")
    uvm_log = (root / "uvm.log").read_text() if (root / "uvm.log").is_file() else ""
    require(BAD_KERNEL.search(kernel_text) is None, "kernel fault signature detected")
    require(BAD_KERNEL.search(uvm_log) is None, "UVM fault signature detected")

    verification = {
        "schema": "vllm_uvm_dynamic_kv_plan_rebuild_verification_v1",
        "status": "ok",
        "cases": 1,
        "generations": summaries,
        "initial_selected_bytes": initial_selected_bytes,
        "parameter_va_count": len(initial_vas),
        "shared_output_prefix": initial_prefix,
        "physical_kv_allocation_bytes": physical_kv_bytes,
        "parameter_va_mode": parameter_va_mode,
        "physical_kv_allocation_dynamic": False,
        "logical_next_request_pressure_dynamic": True,
        "resources_released": True,
        "faults_detected": False,
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
