#!/usr/bin/env python3
"""Verify path-independent steady performance for joint KV/expert states."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


HIGH_TARGET = 40
LOW_TARGET = 20
BACKUP_BLOCKS = 8
REMOTE_EXPERTS = 48
REMOTE_BYTES = 226_492_416
STATIC_SELECTED_BYTES = 6_568_280_064
PREFILL_TOLERANCE_PERCENT = 15.0
DECODE_TOLERANCE_PERCENT = 10.0
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


def verify_queue(snapshot: dict[str, Any], target: int, label: str) -> None:
    require(snapshot.get("invariant_ok") is True, f"{label}: queue invariant")
    groups = [
        snapshot["gpu_free_block_ids"],
        snapshot["gpu_backup_block_ids"],
        snapshot["cpu_offline_block_ids"],
        snapshot["active_block_ids"],
        snapshot["null_block_ids"],
    ]
    flattened = [int(value) for group in groups for value in group]
    total = int(snapshot["total_blocks_including_null"])
    require(len(flattened) == total, f"{label}: queue conservation")
    require(len(flattened) == len(set(flattened)), f"{label}: queue overlap")
    require(set(flattened) == set(range(total)), f"{label}: missing blocks")
    require(snapshot["null_block_ids"] == [0], f"{label}: null block")
    require(
        int(snapshot["applied_target_blocks"]) == target,
        f"{label}: queue target",
    )
    require(
        int(snapshot["gpu_backup_blocks"]) == BACKUP_BLOCKS
        and int(snapshot["backup_target_blocks"]) == BACKUP_BLOCKS,
        f"{label}: backup reserve",
    )
    require(int(snapshot["active_blocks"]) == 0, f"{label}: active blocks")
    require(
        int(snapshot["gpu_free_blocks"]) == target - BACKUP_BLOCKS,
        f"{label}: GPU free count",
    )
    require(
        all(int(value) > target for value in snapshot["cpu_offline_block_ids"]),
        f"{label}: offline block in GPU prefix",
    )


def verify_physical(snapshot: dict[str, Any], target: int, label: str) -> None:
    require(int(snapshot["target_blocks"]) == target, f"{label}: physical target")
    for part in ("full", "head", "tail"):
        record = snapshot[part]
        require(record.get("ok") is True, f"{label}: {part} query")
        require(int(record.get("unknown_bytes", -1)) == 0, f"{label}: {part} unknown")
    head = snapshot["head"]
    tail = snapshot["tail"]
    full = snapshot["full"]
    require(
        int(head["gpu_bytes"]) == int(head["requested_bytes"])
        and int(head["cpu_bytes"]) == 0,
        f"{label}: GPU head residency",
    )
    require(
        int(tail["cpu_bytes"]) == int(tail["requested_bytes"])
        and int(tail["gpu_bytes"]) == 0,
        f"{label}: CPU tail residency",
    )
    require(
        int(full["gpu_bytes"]) + int(full["cpu_bytes"])
        == int(full["requested_bytes"]),
        f"{label}: physical accounting",
    )


def expert_ids(snapshot: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    return {
        str(layer): tuple(int(value) for value in record["remote_expert_ids"])
        for layer, record in snapshot["layers"].items()
    }


def state_signature(state: dict[str, Any]) -> dict[str, Any]:
    queue = state["queue"]
    physical = state["physical"]
    expert = state["expert"]
    model = state["model"]
    return {
        "target_blocks": int(state["target_blocks"]),
        "queue_counts": (
            int(queue["gpu_free_blocks"]),
            int(queue["gpu_backup_blocks"]),
            int(queue["cpu_offline_blocks"]),
            int(queue["active_blocks"]),
        ),
        "physical": (
            int(physical["full"]["gpu_bytes"]),
            int(physical["full"]["cpu_bytes"]),
            int(physical["full"]["unknown_bytes"]),
        ),
        "remote_expert_count": int(expert["remote_expert_count"]),
        "remote_region_bytes": int(expert["remote_region_bytes"]),
        "expert_ids": expert_ids(expert),
        "static_selected_bytes": int(
            model["selected_parameter_bytes_per_forward"]
        ),
        "lifecycle_state": model["lifecycle_state"],
        "active_plan_class": model.get("active_plan_class"),
    }


def verify_state(
    state: dict[str, Any],
    *,
    target: int,
    remote_count: int,
    label: str,
) -> None:
    verify_queue(state["queue"], target, label)
    verify_physical(state["physical"], target, label)
    expert = state["expert"]
    require(
        int(expert["remote_expert_count"]) == remote_count,
        f"{label}: remote expert count",
    )
    expected_bytes = REMOTE_BYTES if remote_count else 0
    require(
        int(expert["remote_region_bytes"]) == expected_bytes,
        f"{label}: remote expert bytes",
    )
    ids = expert_ids(expert)
    require(len(ids) == REMOTE_EXPERTS, f"{label}: expert layer count")
    require(
        all(len(values) == (1 if remote_count else 0) for values in ids.values()),
        f"{label}: expert placement is not uniform by layer",
    )
    model = state["model"]
    require(model.get("lifecycle_state") == "remote_parked", f"{label}: lifecycle")
    require(model.get("poisoned") is not True, f"{label}: poisoned")
    require(
        int(model.get("selected_parameter_bytes_per_forward", -1))
        == STATIC_SELECTED_BYTES,
        f"{label}: static plan bytes",
    )
    require(model.get("active_plan_class") is None, f"{label}: plan class")
    autoscale = state["autoscale"]
    require(autoscale.get("enabled") is False, f"{label}: autoscale not frozen")
    require(
        autoscale.get("pending_grow_target_blocks") is None
        and autoscale.get("pending_shrink_target_blocks") is None
        and autoscale.get("expert_restore_pending") is False
        and autoscale.get("pending_plan_class") is None,
        f"{label}: pending controller work",
    )


def verify_record(
    record: dict[str, Any],
    *,
    state_label: str,
    target: int,
    remote_count: int,
    batch_size: int,
    label: str,
) -> None:
    require(
        record.get("schema") == "vllm_uvm_kv_path_performance_request_v1",
        f"{label}: schema",
    )
    require(record.get("state_label") == state_label, f"{label}: state label")
    require(int(record["batch_size"]) == batch_size, f"{label}: batch size")
    require(len(record["warmup_decode"]) == 5, f"{label}: warmup samples")
    require(len(record["measured_decode"]) == 16, f"{label}: steady samples")
    for step_name in ("prefill", "first_decode"):
        require(
            int(record[step_name]["graph_wrapper_delta"].get("full_replay", 0))
            == 1,
            f"{label}: {step_name} Graph replay",
        )
    require(
        all(
            int(step["graph_wrapper_delta"].get("full_replay", 0)) == 1
            for step in record["warmup_decode"] + record["measured_decode"]
        ),
        f"{label}: Decode Graph replay",
    )
    before = record["state_before"]
    after = record["state_after"]
    verify_state(before, target=target, remote_count=remote_count, label=f"{label}: before")
    verify_state(after, target=target, remote_count=remote_count, label=f"{label}: after")
    require(state_signature(before) == state_signature(after), f"{label}: state drift")
    require(
        before["parameter_vas"]["vas"] == after["parameter_vas"]["vas"],
        f"{label}: Parameter VA changed",
    )
    require(
        before["physical"]["va_sha256"] == after["physical"]["va_sha256"],
        f"{label}: KV VA changed",
    )
    require(
        int(before["graph_wrapper_calls"].get("full_capture", -1))
        == int(after["graph_wrapper_calls"].get("full_capture", -2))
        and int(before["compilation_counter"]["num_cudagraph_captured"])
        == int(after["compilation_counter"]["num_cudagraph_captured"]),
        f"{label}: Graph recapture",
    )
    require(
        len(record.get("resume_transitions") or []) == 1
        and len(record.get("park_transitions") or []) == 1,
        f"{label}: topology transition count",
    )
    require(
        all(item.get("finished") is True for item in record["outputs"]),
        f"{label}: output completion",
    )


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    require(bool(ordered), "empty percentile sample")
    index = max(math.ceil(len(ordered) * quantile) - 1, 0)
    return ordered[index]


def timing(records: list[dict[str, Any]]) -> dict[str, float | int]:
    prefill = [
        float(record["timing"]["prefill_excluding_topology_resume_ms"])
        for record in records
    ]
    prefill_e2e = [
        float(record["timing"]["prefill_end_to_end_ms"])
        for record in records
    ]
    first_decode = [
        float(record["timing"]["first_decode_excluding_topology_park_ms"])
        for record in records
    ]
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
    batch_size = int(records[0]["batch_size"])
    decode_median = statistics.median(decode)
    return {
        "prefill_samples": len(prefill),
        "decode_samples": len(decode),
        "prefill_excluding_resume_median_ms": statistics.median(prefill),
        "prefill_end_to_end_median_ms": statistics.median(prefill_e2e),
        "first_decode_excluding_park_median_ms": statistics.median(first_decode),
        "decode_median_ms": decode_median,
        "decode_p95_ms": percentile(decode, 0.95),
        "decode_tokens_per_second": batch_size * 1000.0 / decode_median,
    }


def delta_percent(dynamic: float, direct: float) -> float:
    return (dynamic / direct - 1.0) * 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.run_root.resolve()
    results = {
        label: load_json(root / label / "result.json")
        for label in ("dynamic", "direct_high", "direct_low")
    }
    for label, result in results.items():
        require(
            (root / label / "returncode.txt").read_text().strip() == "0",
            f"{label}: runner return code",
        )
        require(result.get("status") == "ok", f"{label}: runner status")
        require(result.get("arm") == "D", f"{label}: Graph arm")
        verify_shutdown(result, label)
        require(
            int(result["compilation_counter_after_load"]["num_cudagraph_captured"])
            == int(result["compilation_counter_final"]["num_cudagraph_captured"]),
            f"{label}: global Graph recapture",
        )

    dynamic = results["dynamic"]
    direct_high = results["direct_high"]
    direct_low = results["direct_low"]
    dynamic_perf = dynamic["kv_path_performance"]
    direct_high_perf = direct_high["kv_path_performance"]
    direct_low_perf = direct_low["kv_path_performance"]
    for label, payload in (
        ("dynamic", dynamic_perf),
        ("direct_high", direct_high_perf),
        ("direct_low", direct_low_perf),
    ):
        require(
            payload.get("schema") == "vllm_uvm_kv_expert_path_performance_v1",
            f"{label}: performance schema",
        )
    require(
        len(dynamic_perf["high"]) == len(dynamic_perf["low"]) == 3,
        "dynamic performance sample count",
    )
    require(
        len(direct_high_perf["high"]) == 3 and not direct_high_perf["low"],
        "direct-high performance sample count",
    )
    require(
        len(direct_low_perf["low"]) == 3 and not direct_low_perf["high"],
        "direct-low performance sample count",
    )

    preset = direct_high.get("kv_path_direct_high_preset") or {}
    require(
        preset.get("schema") == "vllm_uvm_kv_path_direct_high_preset_v1"
        and int(preset.get("target_blocks", -1)) == HIGH_TARGET
        and int(preset["release"].get("released_expert_count", -1))
        == REMOTE_EXPERTS,
        "direct-high preset",
    )
    roundtrip = dynamic.get("kv_autoscale_roundtrip") or {}
    require(
        int(roundtrip.get("target_before_blocks", -1)) == HIGH_TARGET
        and int(roundtrip.get("target_after_blocks", -1)) == LOW_TARGET,
        "dynamic high-to-low roundtrip",
    )

    groups = (
        ("dynamic_high", dynamic_perf["high"], "high", HIGH_TARGET, REMOTE_EXPERTS, 8),
        ("direct_high", direct_high_perf["high"], "high", HIGH_TARGET, REMOTE_EXPERTS, 8),
        ("dynamic_low", dynamic_perf["low"], "low", LOW_TARGET, 0, 1),
        ("direct_low", direct_low_perf["low"], "low", LOW_TARGET, 0, 1),
    )
    for group_label, records, state_label, target, remote_count, batch_size in groups:
        for index, record in enumerate(records):
            verify_record(
                record,
                state_label=state_label,
                target=target,
                remote_count=remote_count,
                batch_size=batch_size,
                label=f"{group_label}[{index}]",
            )

    dynamic_high_signature = state_signature(dynamic_perf["high"][0]["state_before"])
    direct_high_signature = state_signature(direct_high_perf["high"][0]["state_before"])
    dynamic_low_signature = state_signature(dynamic_perf["low"][0]["state_before"])
    direct_low_signature = state_signature(direct_low_perf["low"][0]["state_before"])
    require(dynamic_high_signature == direct_high_signature, "high states differ")
    require(dynamic_low_signature == direct_low_signature, "low states differ")

    for dynamic_record, direct_record in zip(
        dynamic_perf["high"], direct_high_perf["high"]
    ):
        require(
            dynamic_record["output_token_ids_sha256"]
            == direct_record["output_token_ids_sha256"],
            "high-state output differs",
        )
    for dynamic_record, direct_record in zip(
        dynamic_perf["low"], direct_low_perf["low"]
    ):
        require(
            dynamic_record["output_token_ids_sha256"]
            == direct_record["output_token_ids_sha256"],
            "low-state output differs",
        )

    timing_dynamic_high = timing(dynamic_perf["high"])
    timing_direct_high = timing(direct_high_perf["high"])
    timing_dynamic_low = timing(dynamic_perf["low"])
    timing_direct_low = timing(direct_low_perf["low"])
    parity = {
        "scope": (
            "steady identical KV/expert states; transition, RESUME, and PARK "
            "wall times excluded from primary parity metrics"
        ),
        "prefill_tolerance_percent": PREFILL_TOLERANCE_PERCENT,
        "decode_tolerance_percent": DECODE_TOLERANCE_PERCENT,
        "high_prefill_delta_percent": delta_percent(
            float(timing_dynamic_high["prefill_excluding_resume_median_ms"]),
            float(timing_direct_high["prefill_excluding_resume_median_ms"]),
        ),
        "high_decode_delta_percent": delta_percent(
            float(timing_dynamic_high["decode_median_ms"]),
            float(timing_direct_high["decode_median_ms"]),
        ),
        "low_prefill_delta_percent": delta_percent(
            float(timing_dynamic_low["prefill_excluding_resume_median_ms"]),
            float(timing_direct_low["prefill_excluding_resume_median_ms"]),
        ),
        "low_decode_delta_percent": delta_percent(
            float(timing_dynamic_low["decode_median_ms"]),
            float(timing_direct_low["decode_median_ms"]),
        ),
    }
    parity["within_tolerance"] = (
        abs(float(parity["high_prefill_delta_percent"]))
        <= PREFILL_TOLERANCE_PERCENT
        and abs(float(parity["low_prefill_delta_percent"]))
        <= PREFILL_TOLERANCE_PERCENT
        and abs(float(parity["high_decode_delta_percent"]))
        <= DECODE_TOLERANCE_PERCENT
        and abs(float(parity["low_decode_delta_percent"]))
        <= DECODE_TOLERANCE_PERCENT
    )

    kernel_text = (root / "kernel.since.log").read_text(encoding="utf-8")
    require(BAD_KERNEL.search(kernel_text) is None, "kernel fault signature")
    for label in results:
        state = load_json(root / label / "gpu.after.json")
        require(state.get("gpu_clean") is True, f"{label}: GPU not clean")
        require(not state.get("compute_pids"), f"{label}: compute clients remain")
        require(not state.get("uvm_client_pids"), f"{label}: UVM clients remain")
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
        "schema": "vllm_uvm_kv_expert_path_parity_verification_v1",
        "status": "ok",
        "model": dynamic["config"]["model"],
        "high_state": {
            "target_blocks": HIGH_TARGET,
            "remote_experts": REMOTE_EXPERTS,
            "remote_expert_bytes": REMOTE_BYTES,
            "batch_size": 8,
        },
        "low_state": {
            "target_blocks": LOW_TARGET,
            "remote_experts": 0,
            "batch_size": 1,
        },
        "timing": {
            "dynamic_high": timing_dynamic_high,
            "direct_high": timing_direct_high,
            "dynamic_low": timing_dynamic_low,
            "direct_low": timing_direct_low,
        },
        "performance_parity": parity,
        "outputs_match": True,
        "state_equivalence_verified": True,
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
