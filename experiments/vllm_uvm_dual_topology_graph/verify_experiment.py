#!/usr/bin/env python3
"""Fail-closed verifier for the same-process dual-topology Graph experiment."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


LOGICAL_BYTES = 2_415_919_104
BACKING_BYTES = 1_207_959_552
ALIAS_COUNT = 6_912
ALIAS_BYTES = 3_623_878_656
REMOTE_ENTRIES = ((1, True, False), (4, True, False))
ALIAS_ENTRIES = ((314, False, False), (1217, False, False))


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def percentile(values: list[float], quantile: float) -> float:
    require(bool(values), "percentile requires at least one value")
    require(0.0 <= quantile <= 1.0, "percentile quantile out of range")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_median_ci(
    values: list[float], *, seed: int, resamples: int = 10_000
) -> list[float]:
    require(bool(values), "bootstrap requires at least one value")
    require(resamples > 0, "bootstrap resample count must be positive")
    rng = random.Random(seed)
    count = len(values)
    medians = [
        statistics.median(values[rng.randrange(count)] for _ in range(count))
        for _ in range(resamples)
    ]
    return [percentile(medians, 0.025), percentile(medians, 0.975)]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path}: JSON root is not an object")
    return value


def normalized_entries(value: Any) -> tuple[tuple[int, bool, bool], ...]:
    require(isinstance(value, (list, tuple)), "Graph entry list is absent")
    result = []
    for item in value:
        require(isinstance(item, (list, tuple)) and len(item) == 3, "bad Graph entry")
        result.append((int(item[0]), bool(item[1]), bool(item[2])))
    return tuple(sorted(result))


def validate_va_snapshots(result: dict[str, Any], label: str) -> list[int]:
    snapshots = result.get("parameter_va_snapshots", {})
    required = ("after_load", "after_prefill", "after_first_decode", "final")
    values: list[list[int]] = []
    for phase in required:
        snapshot = snapshots.get(phase, {})
        require(snapshot.get("count") == 8, f"{label}: {phase} parameter count")
        require(snapshot.get("bytes") == LOGICAL_BYTES, f"{label}: {phase} bytes")
        vas = snapshot.get("vas")
        require(isinstance(vas, list) and len(set(vas)) == 8, f"{label}: {phase} VAs")
        values.append([int(value) for value in vas])
    require(all(value == values[0] for value in values[1:]), f"{label}: Parameter VA changed")
    return values[0]


def model_diag(result: dict[str, Any], phase: str) -> dict[str, Any]:
    value = result.get("diagnostics", {}).get(phase, {}).get("model")
    require(isinstance(value, dict), f"diagnostics {phase} absent")
    return value


def validate_batched_transition(
    transition: dict[str, Any],
    label: str,
    *,
    required: bool,
    transition_mode: str = "grouped",
    expected_grouped_transactions: int = 1,
) -> None:
    if not required:
        return
    processed = int(transition.get("processed_alias_count", 0))
    batches = int(transition.get("transition_batch_count", 0))
    require(transition.get("batch_count_reported") is True, f"{label}: batch count absent")
    require(transition.get("batch_count_valid") is True, f"{label}: invalid batch count")
    require(transition.get("ioctl_calls") == 1, f"{label}: transition ioctl count")
    require(processed == 4_608, f"{label}: transition consumer count")
    expected_transactions = (
        4_608 if transition_mode == "sequential" else expected_grouped_transactions
    )
    require(
        batches == expected_transactions,
        f"{label}: expected {expected_transactions} {transition_mode} transactions, got {batches}",
    )
    requested_mode = transition.get("transition_mode")
    if transition_mode == "sequential":
        require(requested_mode == "sequential", f"{label}: sequential mode absent")
        require(
            transition.get("sequential_requested") is True,
            f"{label}: sequential flag absent",
        )
    else:
        require(
            requested_mode in (None, "grouped"),
            f"{label}: unexpected transition mode {requested_mode!r}",
        )
    require(int(transition.get("tlb_invalidate_count", 0)) > 0, f"{label}: no PTE/TLB work")


def validate_prebuild_certificate(
    transition: dict[str, Any], label: str, *, enabled: bool
) -> None:
    expected_blocks = 1_152 if enabled else 0
    require(
        transition.get("dual_view_block_count") == expected_blocks,
        f"{label}: dual-view block count",
    )
    require(
        transition.get("dual_view_full_coverage") is enabled,
        f"{label}: dual-view coverage certificate",
    )
    if enabled:
        require(
            transition.get("tlb_invalidate_count") == 18,
            f"{label}: expected 18 aggregated PDE/TLB batches",
        )


def validate_capture(result: dict[str, Any], label: str, vas: list[int]) -> None:
    load = model_diag(result, "after_load")
    require(load.get("phase_mode") == "prefill_graph_remote_decode", f"{label}: phase mode")
    require(load.get("park_resume_enabled") is True, f"{label}: park/resume disabled")
    require(load.get("lifecycle_state") == "sealed", f"{label}: load not sealed")
    require(int(load.get("session_id", 0)) > 0, f"{label}: alias session absent")
    require(load.get("active_driver_backing_bytes") == BACKING_BYTES, f"{label}: backing")
    require(load.get("arena_count") == 4, f"{label}: arena count")
    require(load.get("owner_thread_id") is not None, f"{label}: owner thread")
    require(load.get("owner_device_index") == 0, f"{label}: owner device")
    require(load.get("capture_success_count") == 2, f"{label}: capture count")
    require(load.get("remote_graph_captured") is True, f"{label}: remote capture absent")
    require(load.get("alias_graph_captured") is True, f"{label}: alias capture absent")
    require(load.get("alias_graphs_destroyed") is False, f"{label}: alias prematurely destroyed")
    capture_history = load.get("remote_map_history", ())
    require(
        isinstance(capture_history, (list, tuple)) and capture_history,
        f"{label}: capture remote-map history absent",
    )
    for index, item in enumerate(capture_history):
        require(
            isinstance(item, dict)
            and item.get("ok") is True
            and item.get("requested_bytes") == LOGICAL_BYTES
            and item.get("remote_mapped_bytes", 0) >= LOGICAL_BYTES
            and item.get("covered_bytes", 0) >= LOGICAL_BYTES
            and item.get("unknown_bytes") == 0,
            f"{label}: invalid capture remote-map record {index}",
        )

    remote = load.get("remote_graph_certificate")
    alias = load.get("alias_graph_certificate")
    partition = load.get("dual_graph_partition_certificate")
    require(isinstance(remote, dict), f"{label}: remote certificate")
    require(isinstance(alias, dict), f"{label}: alias certificate")
    require(isinstance(partition, dict), f"{label}: partition certificate")
    require(remote.get("topology") == "remote", f"{label}: remote topology cert")
    require(remote.get("actual_h2d_bytes") == 0, f"{label}: remote capture H2D")
    require(remote.get("capture_forward_count") == 8, f"{label}: remote capture forwards")
    require(alias.get("topology") == "sealed", f"{label}: alias topology cert")
    require(alias.get("actual_h2d_bytes") == 2 * LOGICAL_BYTES, f"{label}: alias capture H2D")
    require(alias.get("capture_forward_count") == 8, f"{label}: alias capture forwards")
    require(alias.get("backing_bytes") == BACKING_BYTES, f"{label}: alias cert backing")
    require(tuple(remote.get("parameter_vas", ())) == tuple(vas), f"{label}: remote cert VAs")
    require(tuple(alias.get("parameter_vas", ())) == tuple(vas), f"{label}: alias cert VAs")
    require(normalized_entries(partition.get("remote_entries")) == REMOTE_ENTRIES, f"{label}: remote partition")
    require(normalized_entries(partition.get("alias_entries")) == ALIAS_ENTRIES, f"{label}: alias partition")
    driver = result.get("diagnostics", {}).get("after_load", {}).get("driver", {})
    require(driver.get("session_count") == 1, f"{label}: driver session count")
    sessions = driver.get("sessions")
    require(isinstance(sessions, list) and len(sessions) == 1, f"{label}: driver session")
    session = sessions[0]
    require(session.get("sealed") is True and session.get("poisoned") is False, f"{label}: driver session state")
    require(session.get("parked") is False, f"{label}: driver session remained parked")
    require(session.get("alias_count") == ALIAS_COUNT, f"{label}: driver alias count")
    require(session.get("alias_bytes") == ALIAS_BYTES, f"{label}: driver alias bytes")
    require(session.get("backing_bytes") == BACKING_BYTES, f"{label}: driver backing")


def validate_replays(result: dict[str, Any], label: str) -> None:
    steps = result.get("steps", {})
    prefill = steps.get("prefill", {})
    require(prefill.get("graph_wrapper_delta", {}).get("full_replay") == 1, f"{label}: prefill replay")
    decode_steps = [steps.get("first_decode", {})]
    decode_steps.extend(steps.get("warmup_decode", []))
    decode_steps.extend(steps.get("measured_decode", []))
    require(bool(decode_steps), f"{label}: decode steps absent")
    for index, step in enumerate(decode_steps):
        require(step.get("graph_wrapper_delta", {}).get("full_replay") == 1, f"{label}: decode replay {index}")
    final_calls = result.get("graph_wrapper_calls_final", {})
    require(final_calls.get("full_capture") == 4, f"{label}: full capture wrapper count")
    require(
        final_calls.get("full_replay") == 3 + len(decode_steps),
        f"{label}: full replay total",
    )


def validate_cutover(
    case_dir: Path,
    result: dict[str, Any],
    label: str,
    vas: list[int],
    *,
    require_park_batch: bool,
    transition_mode: str,
    expected_grouped_transactions: int,
    expected_remote_map_operation: str,
    prebuild_enabled: bool,
) -> None:
    prefill = model_diag(result, "after_prefill")
    after = model_diag(result, "after_first_decode")
    final = model_diag(result, "final")
    require(prefill.get("lifecycle_state") == "sealed", f"{label}: prefill topology")
    require(prefill.get("alias_graphs_destroyed") is False, f"{label}: prefill alias Graph gone")
    session_id = int(prefill.get("session_id", 0))
    require(session_id > 0, f"{label}: prefill session absent")
    prefill_history = prefill.get("remote_map_history", ())
    require(
        isinstance(prefill_history, (list, tuple)) and prefill_history,
        f"{label}: prefill remote-map history",
    )
    for name, diag in (("first", after), ("final", final)):
        require(diag.get("lifecycle_state") == "remote_parked", f"{label}: {name} topology")
        require(diag.get("parked") is True, f"{label}: {name} parked bit")
        require(diag.get("session_id") == session_id, f"{label}: {name} session")
        require(diag.get("active_driver_backing_bytes") == BACKING_BYTES, f"{label}: {name} backing")
        require(diag.get("remote_map_active") is True, f"{label}: {name} remote map")
        require(diag.get("alias_graphs_destroyed") is False, f"{label}: {name} alias Graph destroyed")
        require(diag.get("alias_graph_captured") is True, f"{label}: {name} alias Graph absent")
        require(diag.get("remote_graph_captured") is True, f"{label}: {name} remote Graph absent")
        history = diag.get("remote_map_history", ())
        require(
            isinstance(history, (list, tuple))
            and len(history) > len(prefill_history),
            f"{label}: {name} remote-map history did not advance",
        )
        for index, item in enumerate(history):
            require(
                isinstance(item, dict)
                and item.get("ok") is True
                and item.get("requested_bytes") == LOGICAL_BYTES
                and item.get("remote_mapped_bytes", 0) >= LOGICAL_BYTES
                and item.get("covered_bytes", 0) >= LOGICAL_BYTES
                and item.get("unknown_bytes") == 0,
                f"{label}: invalid {name} remote-map record {index}",
            )
    for phase in ("after_first_decode", "final"):
        driver = result.get("diagnostics", {}).get(phase, {}).get("driver", {})
        sessions = driver.get("sessions")
        require(driver.get("session_count") == 1, f"{label}: {phase} driver session")
        require(isinstance(sessions, list) and len(sessions) == 1, f"{label}: {phase} sessions")
        session = sessions[0]
        require(session.get("session_id") == session_id, f"{label}: {phase} session id")
        require(session.get("parked") is True, f"{label}: {phase} not parked")
        require(session.get("poisoned") is False, f"{label}: {phase} poisoned")
        require(session.get("alias_count") == ALIAS_COUNT, f"{label}: {phase} aliases")
        require(session.get("backing_bytes") == BACKING_BYTES, f"{label}: {phase} backing")

    phase_result = load_json(case_dir / "phase_transition_result.json")
    require(phase_result.get("park_resume") is True, f"{label}: phase result mode")
    require(phase_result.get("alias_graph_teardown") is None, f"{label}: alias Graph teardown occurred")

    cutover = result.get("cutover_transition")
    require(isinstance(cutover, dict) and cutover.get("ok") is True, f"{label}: cutover")
    require(cutover.get("pre_state") == "sealed" and cutover.get("post_state") == "remote_parked", f"{label}: cutover states")
    require(cutover.get("preserve_remote_graphs") is True, f"{label}: remote Graph not preserved")
    require(tuple(cutover.get("parameter_vas_before", ())) == tuple(vas), f"{label}: cutover VAs before")
    require(tuple(cutover.get("parameter_vas_after", ())) == tuple(vas), f"{label}: cutover VAs after")
    require(cutover.get("destroy") is None, f"{label}: cmd99 ran during cutover")
    require(
        cutover.get("remote_map_operation") == expected_remote_map_operation,
        f"{label}: expected remote-map operation {expected_remote_map_operation}",
    )
    parked = cutover.get("park", {})
    require(parked.get("ok") is True and parked.get("parked") is True, f"{label}: park")
    require(parked.get("processed_alias_count") == 4_608, f"{label}: parked consumers")
    require(parked.get("upload_alias_count") == 2_304, f"{label}: retained uploads")
    require(parked.get("processed_alias_bytes") == LOGICAL_BYTES, f"{label}: parked bytes")
    require(parked.get("backing_bytes") == BACKING_BYTES, f"{label}: retained backing")
    validate_batched_transition(
        parked,
        f"{label}: initial park",
        required=require_park_batch,
        transition_mode=transition_mode,
        expected_grouped_transactions=expected_grouped_transactions,
    )
    validate_prebuild_certificate(
        parked, f"{label}: initial park", enabled=prebuild_enabled
    )
    remote_map = cutover.get("remote_map", {})
    require(remote_map.get("ok") is True and remote_map.get("requested_bytes") == LOGICAL_BYTES, f"{label}: remote map")
    require(remote_map.get("remote_mapped_bytes", 0) >= LOGICAL_BYTES, f"{label}: mapped bytes")
    physical = cutover.get("physical_residency", {})
    require(physical.get("cpu_bytes") == LOGICAL_BYTES, f"{label}: CPU residency")
    require(physical.get("gpu_bytes") == 0 and physical.get("unknown_bytes") == 0, f"{label}: physical residency")


def validate_roundtrip(
    result: dict[str, Any],
    label: str,
    vas: list[int],
    *,
    require_park_batch: bool,
    transition_mode: str,
    expected_grouped_transactions: int,
    expected_remote_map_operation: str,
    prebuild_enabled: bool,
) -> dict[str, float]:
    roundtrip = result.get("steps", {}).get("park_resume_roundtrip")
    require(isinstance(roundtrip, dict), f"{label}: roundtrip absent")
    require(roundtrip.get("prefill", {}).get("graph_wrapper_delta", {}).get("full_replay") == 1, f"{label}: resumed prefill replay")
    require(roundtrip.get("decode", {}).get("graph_wrapper_delta", {}).get("full_replay") == 1, f"{label}: repark decode replay")
    before = roundtrip.get("diagnostics_before", {}).get("model", {})
    resumed = roundtrip.get("diagnostics_after_resume", {}).get("model", {})
    reparked = roundtrip.get("diagnostics_after_repark", {}).get("model", {})
    session_id = int(before.get("session_id", 0))
    require(before.get("lifecycle_state") == "remote_parked", f"{label}: roundtrip start")
    require(resumed.get("lifecycle_state") == "sealed", f"{label}: resume state")
    require(resumed.get("remote_map_active") is False, f"{label}: remote map remained active")
    require(resumed.get("session_id") == session_id, f"{label}: resume session changed")
    require(resumed.get("active_driver_backing_bytes") == BACKING_BYTES, f"{label}: resume backing")
    require(reparked.get("lifecycle_state") == "remote_parked", f"{label}: repark state")
    require(reparked.get("remote_map_active") is True, f"{label}: repark remote map")
    require(reparked.get("session_id") == session_id, f"{label}: repark session changed")
    for phase in ("before", "after_resume", "after_repark"):
        snapshot = roundtrip.get("parameter_va_snapshots", {}).get(phase, {})
        require(tuple(snapshot.get("vas", ())) == tuple(vas), f"{label}: roundtrip {phase} VAs")
    resume_events = roundtrip.get("resume_transitions", [])
    require(isinstance(resume_events, list) and len(resume_events) == 1, f"{label}: resume transition count")
    resume = resume_events[0]
    resume_ioctl = resume.get("resume", {})
    require(resume.get("post_state") == "sealed" and resume_ioctl.get("ok") is True, f"{label}: resume transition")
    require(resume_ioctl.get("processed_alias_count") == 4_608, f"{label}: resumed consumers")
    require(resume_ioctl.get("processed_alias_bytes") == LOGICAL_BYTES, f"{label}: resumed bytes")
    validate_batched_transition(
        resume_ioctl,
        f"{label}: resume",
        required=require_park_batch,
        transition_mode=transition_mode,
        expected_grouped_transactions=expected_grouped_transactions,
    )
    validate_prebuild_certificate(
        resume_ioctl, f"{label}: resume", enabled=prebuild_enabled
    )
    repark_events = roundtrip.get("repark_transitions", [])
    require(isinstance(repark_events, list) and len(repark_events) == 1, f"{label}: repark transition count")
    repark = repark_events[0]
    require(repark.get("post_state") == "remote_parked", f"{label}: repark transition")
    require(
        repark.get("remote_map_operation") == expected_remote_map_operation,
        f"{label}: repark remote-map operation",
    )
    require(repark.get("park", {}).get("processed_alias_count") == 4_608, f"{label}: repark consumers")
    validate_batched_transition(
        repark.get("park", {}),
        f"{label}: repark",
        required=require_park_batch,
        transition_mode=transition_mode,
        expected_grouped_transactions=expected_grouped_transactions,
    )
    validate_prebuild_certificate(
        repark.get("park", {}), f"{label}: repark", enabled=prebuild_enabled
    )
    before_counter = roundtrip.get("compilation_counter_before", {})
    after_counter = roundtrip.get("compilation_counter_after", {})
    require(before_counter.get("num_cudagraph_captured") == after_counter.get("num_cudagraph_captured"), f"{label}: Graph recaptured")
    before_calls = roundtrip.get("graph_wrapper_calls_before", {})
    after_calls = roundtrip.get("graph_wrapper_calls_after", {})
    require(after_calls.get("full_capture") == before_calls.get("full_capture"), f"{label}: wrapper recaptured")
    require(after_calls.get("full_replay") == before_calls.get("full_replay", 0) + 2, f"{label}: roundtrip replay delta")
    resume_wall_ms = float(resume.get("wall_ns", 0)) / 1e6
    resume_ioctl_ms = float(resume_ioctl.get("ioctl_wall_ns", 0)) / 1e6
    repark_wall_ms = float(repark.get("wall_ns", 0)) / 1e6
    repark_ioctl_ms = float(repark.get("park", {}).get("ioctl_wall_ns", 0)) / 1e6
    require(all(value > 0 and math.isfinite(value) for value in (resume_wall_ms, resume_ioctl_ms, repark_wall_ms, repark_ioctl_ms)), f"{label}: roundtrip timings")
    return {
        "resume_wall_ms": resume_wall_ms,
        "resume_ioctl_ms": resume_ioctl_ms,
        "repark_wall_ms": repark_wall_ms,
        "repark_ioctl_ms": repark_ioctl_ms,
    }


def validate_shutdown(result: dict[str, Any], label: str) -> None:
    shutdown = result.get("shutdown_contract", {})
    require(shutdown.get("cleanup_errors") == [], f"{label}: cleanup errors")
    items = shutdown.get("model_runner_shutdown_results")
    require(isinstance(items, list) and len(items) == 1, f"{label}: shutdown result")
    require(items[0].get("ok") is True, f"{label}: shutdown failed")
    require(items[0].get("destroyed_graph_count") == 4, f"{label}: retained Graph reset count")
    diag = shutdown.get("diagnostics_after_shutdown", {}).get("model", {})
    require(diag.get("session_id") == 0, f"{label}: shutdown session")
    require(diag.get("active_driver_backing_bytes") == 0, f"{label}: shutdown backing")


def validate_case(
    case_dir: Path,
    *,
    require_park_batch: bool = False,
    transition_mode: str = "grouped",
    expected_grouped_transactions: int = 1,
    expected_remote_map_operation: str = "prebuilt_full_dual_view",
    prebuild_enabled: bool = True,
) -> dict[str, Any]:
    label = case_dir.name
    result = load_json(case_dir / "result.json")
    require(result.get("status") == "ok", f"{label}: status")
    require(result.get("arm") == "D", f"{label}: arm")
    config = result.get("config", {})
    require(config.get("phase_mode") == "prefill_graph_remote_decode", f"{label}: config phase")
    require(config.get("graph_prefill") is True, f"{label}: prefill Graph config")
    require(config.get("graph_decode") is True, f"{label}: decode Graph config")
    require(config.get("park_resume_roundtrip") is True, f"{label}: roundtrip config")
    require(config.get("cudagraph_capture_sizes") == [1, 4, 314, 1217], f"{label}: capture sizes")
    require(result.get("environment", {}).get("VLLM_UVM_ORIGINAL_VA_PHASE_MODE") == "prefill_graph_remote_decode", f"{label}: environment phase")
    require(result.get("environment", {}).get("VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME") == "1", f"{label}: environment park/resume")
    recorded_backend = result.get("environment", {}).get(
        "VLLM_ATTENTION_BACKEND"
    )
    environment_lines = set((case_dir / "environment.txt").read_text().splitlines())
    require(
        recorded_backend == "TRITON_ATTN"
        or "VLLM_ATTENTION_BACKEND=TRITON_ATTN" in environment_lines,
        f"{label}: attention backend",
    )
    vas = validate_va_snapshots(result, label)
    validate_capture(result, label, vas)
    validate_replays(result, label)
    validate_cutover(
        case_dir,
        result,
        label,
        vas,
        require_park_batch=require_park_batch,
        transition_mode=transition_mode,
        expected_grouped_transactions=expected_grouped_transactions,
        expected_remote_map_operation=expected_remote_map_operation,
        prebuild_enabled=prebuild_enabled,
    )
    roundtrip_timing = validate_roundtrip(
        result,
        label,
        vas,
        require_park_batch=require_park_batch,
        transition_mode=transition_mode,
        expected_grouped_transactions=expected_grouped_transactions,
        expected_remote_map_operation=expected_remote_map_operation,
        prebuild_enabled=prebuild_enabled,
    )
    validate_shutdown(result, label)
    kernel_log = (case_dir / "kernel.since.log").read_text(errors="replace")
    for forbidden in ("manual_migration_fault", "NVRM: Xid", "BUG:", "Oops:"):
        require(forbidden not in kernel_log, f"{label}: forbidden kernel record {forbidden}")
    timing = result.get("timing", {})
    for key in ("prefill_step_ms", "first_decode_step_ms", "cutover_wall_ms", "cycle_e2e_ms"):
        value = timing.get(key)
        require(isinstance(value, (int, float)) and math.isfinite(value) and value > 0, f"{label}: timing {key}")
    steady = timing.get("steady_decode_step_ms", {}).get("median")
    require(isinstance(steady, (int, float)) and steady > 0, f"{label}: steady decode")
    return {
        "case": label,
        "batch_size": int(config["batch_size"]),
        "audit_trace": bool(config["audit_trace"]),
        "transition_mode": transition_mode,
        "output_hash": result.get("output_token_ids_sha256"),
        "prefill_step_ms": float(timing["prefill_step_ms"]),
        "steady_decode_step_ms": float(steady),
        "first_decode_step_ms": float(timing["first_decode_step_ms"]),
        "cutover_wall_ms": float(timing["cutover_wall_ms"]),
        "cycle_e2e_ms": float(timing["cycle_e2e_ms"]),
        "cuda_used_after_load_bytes": int(result.get("cuda_memory_after_load", {}).get("used_bytes", 0)),
        "cuda_max_allocated_bytes": int(result.get("cuda_memory_final", {}).get("max_allocated_bytes", 0)),
        **roundtrip_timing,
    }


def aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    batches = sorted({int(item["batch_size"]) for item in samples})
    require(bool(batches) and set(batches) <= {1, 4}, "unexpected batch set")
    for batch in batches:
        members = [item for item in samples if item["batch_size"] == batch]
        require(bool(members), f"batch {batch} absent")
        hashes = {item["output_hash"] for item in members}
        require(len(hashes) == 1 and None not in hashes, f"batch {batch}: output hash drift")
        performance = [item for item in members if not item["audit_trace"]] or members
        metrics: dict[str, Any] = {"sample_count": len(performance), "output_hash": next(iter(hashes))}
        for metric_index, key in enumerate((
            "prefill_step_ms",
            "steady_decode_step_ms",
            "first_decode_step_ms",
            "cutover_wall_ms",
            "resume_wall_ms",
            "resume_ioctl_ms",
            "repark_wall_ms",
            "repark_ioctl_ms",
            "cycle_e2e_ms",
            "cuda_used_after_load_bytes",
            "cuda_max_allocated_bytes",
        )):
            values = [float(item[key]) for item in performance]
            bootstrap_seed = 20260828 + batch * 100 + metric_index
            metrics[key] = {
                "median": statistics.median(values),
                "p95": percentile(values, 0.95),
                "p99": percentile(values, 0.99),
                "median_bootstrap_95_ci": bootstrap_median_ci(
                    values, seed=bootstrap_seed
                ),
                "bootstrap_resamples": 10_000,
                "bootstrap_seed": bootstrap_seed,
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
            }
        output[str(batch)] = metrics
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    run_config = load_json(root / "run_config.json")
    require(run_config.get("attention_backend") == "TRITON_ATTN", "run config attention backend")
    require(run_config.get("cudagraph_mode") == "FULL_SEPARATE", "run config Graph mode")
    require(run_config.get("cudagraph_capture_sizes") == [1, 4, 314, 1217], "run config capture sizes")
    require(run_config.get("park_resume") is True, "run config park/resume")
    require(run_config.get("roundtrip_cycles") == 2, "run config roundtrip cycles")
    require_park_batch = run_config.get("park_batch_required") is True
    transition_mode = str(run_config.get("transition_mode", "grouped"))
    require(
        transition_mode in ("grouped", "sequential"),
        "run config transition mode",
    )
    module_config = load_json(root / "module.custom.json")
    dual_view_enabled = module_config.get("static_alias_dual_view_enable") == 1
    require(
        module_config.get("static_alias_dual_view_require_full")
        == (1 if dual_view_enabled else 0),
        "module dual-view guard differs from mode",
    )
    expected_grouped_transactions = 1
    expected_remote_map_operation = (
        "prebuilt_full_dual_view" if dual_view_enabled else "audited_range_install"
    )
    status = root / "execution_status.tsv"
    require(status.is_file(), "execution_status.tsv absent")
    rows = [line.split("\t") for line in status.read_text().splitlines()[1:] if line]
    require(bool(rows), "no cases")
    require(all(row[-1] == "0" for row in rows), "one or more case return codes failed")
    cases = [root / row[-2] for row in rows]
    samples = [
        validate_case(
            path,
            require_park_batch=require_park_batch,
            transition_mode=transition_mode,
            expected_grouped_transactions=expected_grouped_transactions,
            expected_remote_map_operation=expected_remote_map_operation,
            prebuild_enabled=dual_view_enabled,
        )
        for path in cases
    ]
    verification = {
        "schema": (
            "vllm_uvm_dual_topology_graph_park_resume_verification_v3"
            if require_park_batch
            else "vllm_uvm_dual_topology_graph_park_resume_verification_v2"
        ),
        "status": "ok",
        "run_root": str(root),
        "run_config": run_config,
        "page_table_prebuild": {
            "enabled": dual_view_enabled,
            "expected_grouped_transactions": expected_grouped_transactions,
            "expected_remote_map_operation": expected_remote_map_operation,
        },
        "case_count": len(samples),
        "samples": samples,
        "aggregate": aggregate(samples),
    }
    output = args.output or (root / "verification.json")
    output.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ok", "case_count": len(samples), "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
