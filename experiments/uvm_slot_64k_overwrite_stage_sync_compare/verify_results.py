#!/usr/bin/env python3
"""Strictly verify saved existing-64K overwrite/stage-sync artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_benchmark as runner


MIB = 1024 * 1024
GIB = 1024 * MIB
VA_BLOCK_BYTES = 2 * MIB
PTE_64K_BYTES = 64 * 1024
MODES = runner.MODES
OVERWRITE_MODES = frozenset(
    ("overwrite_64k_e2e", "overwrite_64k_stage_sync")
)
STAGE_SYNC_MODES = frozenset(
    ("normal_64k_stage_sync", "overwrite_64k_stage_sync")
)
GENERIC_COUNTERS = (
    "prefetch_gpu_batch_fast_ioctl_calls",
    "prefetch_gpu_batch_fast_core_calls",
    "prefetch_gpu_batch_fast_requested_ranges",
    "prefetch_gpu_batch_fast_completed_ranges",
    "prefetch_gpu_batch_fast_requested_bytes",
    "prefetch_gpu_batch_fast_completed_bytes",
    "prefetch_gpu_batch_fast_waits",
    "prefetch_gpu_batch_fast_fallbacks",
    "prefetch_gpu_batch_fast_errors",
)
SLOT_COUNTERS = (
    "slot_direct_ioctl_calls",
    "slot_direct_per_block_calls",
    "slot_direct_shared_calls",
    "slot_direct_successes",
    "slot_direct_requested_ranges",
    "slot_direct_completed_ranges",
    "slot_direct_requested_bytes",
    "slot_direct_completed_bytes",
    "slot_direct_blocks",
    "slot_direct_copy_pushes",
    "slot_direct_memcopy_methods",
    "slot_direct_tracker_waits",
    "slot_direct_vidmem_allocations",
    "slot_direct_vidmem_evictions",
    "slot_direct_pte_map_ranges",
    "slot_direct_pte_map_bytes",
    "slot_direct_pte_4k_blocks",
    "slot_direct_pte_64k_blocks",
    "slot_direct_pte_2m_blocks",
    "slot_direct_pte_write_entries",
    "slot_direct_skipped_unmap_blocks",
    "slot_direct_stage_sync_calls",
    "slot_direct_fallbacks",
    "slot_direct_errors",
)
LEGACY_SLOT_PROFILE_STAGES = (
    "total",
    "prepare",
    "unmap",
    "copy",
    "map",
    "wait",
    "prepare_complete",
    "copy_complete",
    "map_complete",
)
COMPLETE_PROFILE_STAGES = ("prepare_complete", "copy_complete", "map_complete")
PARENT_PROFILE_STAGES = (
    "pre_helper_total",
    "helper_total",
    "post_helper_total",
)
PRE_HELPER_PROFILE_STAGES = (
    "gate",
    "state_alloc",
    "gpu_retain",
    "outer_lock_acquire",
    "batch_bookkeeping",
    "slot_validate",
    "flatten_sort",
    "slot_tracker_init",
)
HELPER_RESIDUAL_PROFILE_STAGES = (
    "helper_context",
    "block_lock_acquire",
    "locked_revalidate",
    "copy_to_map",
    "block_unlock",
    "helper_context_free",
)
POST_HELPER_PROFILE_STAGES = (
    "result_check",
    "pte_metadata",
    "status_resolve",
    "slot_state_commit",
    "telemetry_cleanup",
    "outer_lock_release",
    "resource_teardown",
)
RESIDUAL_PROFILE_STAGES = (
    PRE_HELPER_PROFILE_STAGES
    + HELPER_RESIDUAL_PROFILE_STAGES
    + POST_HELPER_PROFILE_STAGES
)
DETAILED_PROFILE_STAGES = PARENT_PROFILE_STAGES + RESIDUAL_PROFILE_STAGES
SLOT_PROFILE_STAGES_V2 = LEGACY_SLOT_PROFILE_STAGES + DETAILED_PROFILE_STAGES
CLOSURE_MEASUREMENT_FIELDS = (
    "driver_accounted_ms",
    "driver_residual_ms",
    "driver_closure_fraction",
    "host_outer_ms",
    "parent_partition_residual_ms",
    "pre_helper_residual_ms",
    "helper_residual_ms",
    "post_helper_residual_ms",
)
DRIVER_RESIDUAL_ABS_LIMIT_NS = 100_000
DRIVER_RESIDUAL_REL_LIMIT = 0.005
BOUNDARY_FIELDS = (
    "migrate_profile.api_total.calls",
    "prefetch_gpu_batch_fast_ioctl_calls",
    "slot_direct_ioctl_calls",
)
EXPECTED_ABI = {
    "generic_ioctl": 90,
    "generic_max_ranges": 128,
    "generic_params_size": 9280,
    "phase_f_ioctl": 81,
    "phase_f_max_ranges": 64,
    "phase_f_params_size": 4656,
    "phase_f_action_prepare_remote_read": 5,
    "phase_f_action_force_remote_read_64k": 20,
    "slot_ioctl": 91,
    "slot_max_entries": 128,
    "slot_params_size": 6224,
    "slot_mode_prealloc": 8,
    "slot_mode_normal_64k_e2e": 12,
    "slot_mode_overwrite_64k_e2e": 17,
    "slot_mode_normal_64k_stage_sync": 16,
    "slot_mode_overwrite_64k_stage_sync": 18,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument(
        "--pinned-result",
        type=Path,
        help="optional matching 2GiB/512-range pinned-copy sidecar result",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_mapping(value: Any, message: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), message)
    return value


def schema_version(value: Mapping[str, Any], context: str) -> int:
    raw = value.get("schema_version")
    require(
        isinstance(raw, int) and not isinstance(raw, bool) and raw in (1, 2),
        f"{context}: supported integer schema_version 1 or 2",
    )
    return raw


def raw_integer(value: Any, context: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{context}: raw profile value is an integer",
    )
    return value


def finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def close(observed: Any, expected: Any, *, rel_tol: float = 1e-12) -> bool:
    try:
        left = float(observed)
        right = float(expected)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(left)
        and math.isfinite(right)
        and math.isclose(left, right, rel_tol=rel_tol, abs_tol=1e-12)
    )


def exact_ms(observed: Any, nanoseconds: int) -> bool:
    """Check a JSON millisecond projection of an authoritative integer ns value."""

    return close(observed, nanoseconds / 1_000_000.0)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load {path}: {error}") from error


def load(requested: Path) -> tuple[Path, dict[str, Any]]:
    path = requested / "summary.json" if requested.is_dir() else requested
    document = read_json(path)
    require(isinstance(document, dict), f"{path}: top-level JSON is not an object")
    return path, document


def verify_raw_artifacts(path: Path, document: Mapping[str, Any]) -> None:
    """Ensure summary data is an exact copy of all separately saved raw records."""

    root = path.parent
    config_path = root / "config.json"
    jsonl_path = root / "samples.jsonl"
    samples_root = root / "samples"
    require(config_path.is_file(), f"{path}: missing raw config.json")
    require(jsonl_path.is_file(), f"{path}: missing raw samples.jsonl")
    require(samples_root.is_dir(), f"{path}: missing raw samples directory")
    require(
        read_json(config_path) == document["config"],
        f"{path}: config.json differs from summary config",
    )

    try:
        raw_lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"cannot load {jsonl_path}: {error}") from error
    require(raw_lines and all(line.strip() for line in raw_lines), f"{path}: empty JSONL row")
    raw_samples: list[Any] = []
    for line_number, line in enumerate(raw_lines, 1):
        try:
            raw_samples.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"{jsonl_path}:{line_number}: malformed sample: {error}"
            ) from error
    require(
        raw_samples == document["samples"],
        f"{path}: samples.jsonl differs from summary raw samples",
    )

    sample_paths = sorted(samples_root.glob("*/sample.json"))
    require(
        len(sample_paths) == len(raw_samples),
        f"{path}: per-process sample.json count mismatch",
    )
    per_process = [read_json(sample_path) for sample_path in sample_paths]
    try:
        per_process.sort(key=lambda sample: int(sample["schedule"]["global_index"]))
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"{path}: malformed per-process schedule index") from error
    require(
        per_process == raw_samples,
        f"{path}: per-process sample.json files differ from samples.jsonl",
    )


def verify_pinned_result(path: Path) -> float:
    """Verify the existing pinned-copy sidecar without conflating its VA semantics."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"cannot load pinned result {path}: {error}") from error
    result_lines = [line for line in lines if line.startswith("RESULT ")]
    require(len(result_lines) == 1, f"{path}: exactly one filtered RESULT row")
    fields: dict[str, str] = {}
    for token in result_lines[0].split()[1:]:
        require("=" in token, f"{path}: malformed RESULT token {token!r}")
        name, value = token.split("=", 1)
        fields[name] = value
    try:
        contract_ok = (
            fields.get("layout") == "multi_alloc"
            and fields.get("backend") == "pinned_copy"
            and fields.get("mode") == "tensor_batch"
            and int(fields["bytes"]) == 2 * GIB
            and int(fields["logical_ranges"]) == 512
            and int(fields["submitted_ranges"]) == 512
            and int(fields["functional_ok"]) == 1
            and int(fields["in_place_va"]) == 0
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"{path}: incomplete pinned-copy RESULT row") from error
    require(contract_ok, f"{path}: matching pinned-copy sidecar contract")
    require(
        any(
            line.startswith("benchmark_result ")
            and "functional_success=1" in line.split()
            and "filtered=1" in line.split()
            for line in lines
        ),
        f"{path}: filtered pinned benchmark completed successfully",
    )
    migration_ms = float(fields["migration_ms_p50"])
    migration_gib_s = float(fields["migration_gib_s"])
    require(
        finite_positive(migration_ms) and finite_positive(migration_gib_s),
        f"{path}: positive pinned-copy timing and bandwidth",
    )
    require(
        close(migration_gib_s, 2.0 / (migration_ms / 1000.0), rel_tol=5e-5),
        f"{path}: pinned bandwidth is consistent with 2GiB/time",
    )
    return migration_gib_s


def residency_equals(
    value: Mapping[str, Any], *, ranges: int, cpu: int, gpu: int, any_: int
) -> bool:
    try:
        return (
            int(value["ranges"]) == ranges
            and int(value["cpu_bytes"]) == cpu
            and int(value["gpu_bytes"]) == gpu
            and int(value["any_bytes"]) == any_
            and int(value["unknown_bytes"]) == 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def verify_cpu_reset(
    value: Mapping[str, Any], allocations: int, logical_bytes: int, context: str
) -> None:
    require(
        int(value["accepted"]) == allocations
        and int(value["last_prefetch_cpu_ranges"]) == allocations
        and bool(value["va_stable"])
        and residency_equals(
            require_mapping(value["source_residency"], f"{context}: source residency"),
            ranges=allocations,
            cpu=logical_bytes,
            gpu=0,
            any_=logical_bytes,
        ),
        f"{context}: complete CPU residency and stable VA",
    )


def verify_outcome(
    value: Mapping[str, Any], allocations: int, logical_bytes: int, *, transfer: bool
) -> None:
    require(
        int(value["calls"]) == 4
        and int(value["success"]) == allocations
        and int(value["failure"]) == 0
        and int(value["conflict"]) == 0
        and int(value["requested_bytes"]) == logical_bytes
        and int(value["accepted_bytes"]) == logical_bytes
        and int(value["bad_entries"]) == 0,
        "four synchronous ioctl batches completed every range",
    )
    if not transfer:
        require(
            int(value["remap"]) == 0
            and int(value["tlb_invalidate"]) == 0
            and int(value["copy_pushes"]) == 0,
            "mode-8 preallocation did not copy or map",
        )


def expected_slot_values(
    mode: str, allocations: int, logical_bytes: int
) -> dict[str, int]:
    blocks = logical_bytes // VA_BLOCK_BYTES
    overwrite = mode in OVERWRITE_MODES
    stage_sync = mode in STAGE_SYNC_MODES
    return {
        "slot_direct_ioctl_calls": 4,
        "slot_direct_per_block_calls": 4,
        "slot_direct_shared_calls": 0,
        "slot_direct_successes": 4,
        "slot_direct_requested_ranges": allocations,
        "slot_direct_completed_ranges": allocations,
        "slot_direct_requested_bytes": logical_bytes,
        "slot_direct_completed_bytes": logical_bytes,
        "slot_direct_blocks": blocks,
        "slot_direct_copy_pushes": blocks,
        "slot_direct_memcopy_methods": blocks,
        "slot_direct_tracker_waits": 16 if stage_sync else 4,
        "slot_direct_vidmem_allocations": 0,
        "slot_direct_vidmem_evictions": 0,
        "slot_direct_pte_map_ranges": allocations,
        "slot_direct_pte_map_bytes": logical_bytes,
        "slot_direct_pte_4k_blocks": 0,
        "slot_direct_pte_64k_blocks": blocks,
        "slot_direct_pte_2m_blocks": 0,
        "slot_direct_pte_write_entries": blocks * (VA_BLOCK_BYTES // PTE_64K_BYTES),
        "slot_direct_skipped_unmap_blocks": blocks if overwrite else 0,
        "slot_direct_stage_sync_calls": 4 if stage_sync else 0,
        "slot_direct_fallbacks": 0,
        "slot_direct_errors": 0,
    }


def expected_required_profile(
    mode: str, allocations: int, logical_bytes: int, sample_schema: int
) -> dict[str, int]:
    pages = logical_bytes // 4096
    stage_sync = mode in STAGE_SYNC_MODES
    overwrite = mode in OVERWRITE_MODES
    expected = {name: 0 for name in GENERIC_COUNTERS}
    expected["migrate_profile.api_total.calls"] = 0
    expected.update(expected_slot_values(mode, allocations, logical_bytes))
    stages = (
        LEGACY_SLOT_PROFILE_STAGES
        if sample_schema == 1
        else SLOT_PROFILE_STAGES_V2
    )
    for stage in stages:
        complete = stage in COMPLETE_PROFILE_STAGES
        enabled = not complete or stage_sync
        prefix = f"slot_direct_profile.{stage}"
        expected[f"{prefix}.calls"] = 4 if enabled else 0
        expected[f"{prefix}.pages"] = pages if enabled else 0
        expected[f"{prefix}.bytes"] = logical_bytes if enabled else 0
        if not enabled or (stage == "unmap" and overwrite):
            expected[f"{prefix}.ns"] = 0
    return expected


def verify_precondition(
    value: Mapping[str, Any], allocations: int, logical_bytes: int
) -> None:
    require(
        value["order"]
        == ["cpu_reset", "prepare_remote_read_action_5", "force_64k_action_20"],
        "precondition order is exactly CPU reset -> action 5 -> action 20",
    )
    verify_cpu_reset(
        require_mapping(value["cpu_reset"], "precondition CPU reset"),
        allocations,
        logical_bytes,
        "precondition CPU reset",
    )
    ioctl_calls = math.ceil(allocations / 64)
    action5 = require_mapping(value["prepare_remote_read"], "action-5 result")
    require(
        int(action5["action"]) == 5
        and int(action5["ioctl_calls"]) == ioctl_calls
        and int(action5["success_ranges"]) == allocations
        and 0 <= int(action5["newly_mapped_bytes"]) <= logical_bytes
        and int(action5["cpu_candidate_bytes"]) == logical_bytes
        and int(action5["covered_bytes"]) == logical_bytes
        and int(action5["unknown_bytes"]) == 0,
        "action 5 covered the complete CPU source set",
    )
    action20 = require_mapping(value["force_remote_read_64k"], "action-20 result")
    split = int(action20["split_2m_to_64k_bytes"])
    already = int(action20["already_64k_bytes"])
    require(
        int(action20["action"]) == 20
        and int(action20["ioctl_calls"]) == ioctl_calls
        and int(action20["success_ranges"]) == allocations
        and split >= 0
        and already >= 0
        and split + already == logical_bytes
        and split % PTE_64K_BYTES == 0
        and already % PTE_64K_BYTES == 0
        and int(action20["validated_64k_bytes"]) == logical_bytes
        and int(action20["unknown_bytes"]) == 0,
        "action 20 split-or-reused and strictly validated every byte as 64K",
    )
    require(
        residency_equals(
            require_mapping(value["post_action_residency"], "post-action residency"),
            ranges=allocations,
            cpu=logical_bytes,
            gpu=0,
            any_=logical_bytes,
        )
        and bool(value["va_stable"]),
        "action 20 preserved ordinary CPU residency and managed VA",
    )


def verify_measurement(
    sample: Mapping[str, Any],
    allocations: int,
    logical_bytes: int,
    sample_schema: int,
) -> None:
    mode = str(sample["mode"])
    blocks = logical_bytes // VA_BLOCK_BYTES
    measurement = require_mapping(sample["measurement"], "measurement")
    outcome = require_mapping(measurement["outcome"], "timed outcome")
    verify_outcome(outcome, allocations, logical_bytes, transfer=True)
    require(
        int(outcome["remap"]) == allocations
        and int(outcome["tlb_invalidate"]) == blocks
        and int(outcome["copy_pushes"]) == blocks,
        "timed path remapped every range with one copy push per VA block",
    )
    wall_ns = int(measurement["wall_ns"])
    wall_ms = float(measurement["wall_ms"])
    expected_gib_s = (logical_bytes / GIB) / (wall_ns / 1_000_000_000.0)
    require(
        bool(
            measurement[
                "cpu_reset_action5_action20_immediately_before_profile_and_timing"
            ]
        )
        and bool(measurement["host_monotonic_to_synchronous_ioctl_completion"])
        and wall_ns > 0
        and close(wall_ms, wall_ns / 1_000_000.0)
        and close(measurement["gib_s"], expected_gib_s)
        and measurement["component_source"] == "slot_direct_profile"
        and bool(measurement["submit_stages_are_not_ce_completion"]),
        "e2e and submit-stage timings have explicit, internally consistent semantics",
    )
    for field in (
        "driver_profile_total_ms",
        "prepare_ms",
        "copy_ms",
        "map_ms",
        "wait_ms",
    ):
        require(finite_positive(measurement[field]), f"positive {field}")
    overwrite = mode in OVERWRITE_MODES
    require(
        close(measurement["unmap_ms"], 0.0)
        if overwrite
        else finite_positive(measurement["unmap_ms"]),
        "separate unmap is skipped only by overwrite modes",
    )
    stage_sync = mode in STAGE_SYNC_MODES
    if stage_sync:
        require(
            bool(measurement["stage_sync_completion_boundaries"])
            and all(finite_positive(measurement[field]) for field in (
                "prepare_complete_ms",
                "copy_complete_ms",
                "map_complete_ms",
            )),
            "stage-sync mode exposes all positive completion boundaries",
        )
        copy_complete = float(measurement["copy_complete_ms"])
        copy_submit = float(measurement["copy_ms"])
        tail = float(measurement["copy_complete_minus_submit_ms"])
        copy_gib_s = float(measurement["copy_complete_gib_s"])
        require(
            tail >= 0
            and close(tail, copy_complete - copy_submit)
            and close(copy_gib_s, (logical_bytes / GIB) / (copy_complete / 1000.0)),
            "copy completion tail and bandwidth are internally consistent",
        )
    else:
        require(
            not bool(measurement["stage_sync_completion_boundaries"])
            and all(
                measurement[field] is None
                for field in (
                    "prepare_complete_ms",
                    "copy_complete_ms",
                    "map_complete_ms",
                    "copy_complete_gib_s",
                    "copy_complete_minus_submit_ms",
                )
            ),
            "non-stage-sync mode does not fabricate completion diagnostics",
        )

    if sample_schema == 2:
        for stage in DETAILED_PROFILE_STAGES:
            require(
                finite_positive(measurement[f"{stage}_ms"]),
                f"schema-2 measurement has positive {stage}_ms",
            )
        for field in CLOSURE_MEASUREMENT_FIELDS:
            try:
                value = float(measurement[field])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"schema-2 measurement has numeric {field}"
                ) from error
            require(math.isfinite(value), f"schema-2 measurement has finite {field}")
        require(
            float(measurement["driver_accounted_ms"]) > 0
            and float(measurement["driver_residual_ms"]) >= 0
            and 0 < float(measurement["driver_closure_fraction"]) <= 1
            and float(measurement["host_outer_ms"]) >= 0
            and float(measurement["parent_partition_residual_ms"]) == 0
            and float(measurement["pre_helper_residual_ms"]) >= 0
            and float(measurement["helper_residual_ms"]) >= 0
            and float(measurement["post_helper_residual_ms"]) >= 0,
            "schema-2 closure diagnostics have valid signs and bounds",
        )


def verify_validation(
    sample: Mapping[str, Any], allocations: int, logical_bytes: int
) -> None:
    validation = require_mapping(sample["validation"], "validation")
    require(
        residency_equals(
            require_mapping(validation["post_residency"], "post residency"),
            ranges=allocations,
            cpu=logical_bytes,
            gpu=0,
            any_=logical_bytes,
        )
        and residency_equals(
            require_mapping(
                validation["post_checksum_residency"], "post-checksum residency"
            ),
            ranges=allocations,
            cpu=logical_bytes,
            gpu=0,
            any_=logical_bytes,
        ),
        "slot mappings preserve ordinary CPU UVM residency",
    )
    require(
        bool(validation["slot_mapping_is_not_uvm_gpu_residency"])
        and int(validation["cpu_last_prefetch_after"]) == allocations
        and int(validation["checksum"]) == int(validation["expected_checksum"])
        and bool(validation["full_gpu_checksum"])
        and finite_positive(validation["checksum_ms"])
        and bool(validation["va_stable"]),
        "full checksum, CPU LastPrefetchLocation, and managed VA are valid",
    )


def verify_v2_closure_from_raw_ns(
    sample: Mapping[str, Any], delta: Mapping[str, int]
) -> None:
    """Independently reconstruct every schema-2 closure from raw profile ns."""

    mode = str(sample["mode"])
    measurement = require_mapping(sample["measurement"], "measurement")

    def stage_ns(stage: str) -> int:
        name = f"slot_direct_profile.{stage}.ns"
        require(name in delta, f"schema-2 raw profile contains {name}")
        return raw_integer(delta[name], name)

    total_ns = stage_ns("total")
    wait_ns = stage_ns("wait")
    helper_core_stages = (
        COMPLETE_PROFILE_STAGES
        if mode in STAGE_SYNC_MODES
        else ("prepare", "copy", "map")
    )
    helper_core_ns = sum(stage_ns(stage) for stage in helper_core_stages)
    core_ns = helper_core_ns + wait_ns
    leaf_ns = sum(stage_ns(stage) for stage in RESIDUAL_PROFILE_STAGES)
    driver_accounted_ns = core_ns + leaf_ns
    driver_residual_ns = total_ns - driver_accounted_ns

    pre_helper_total_ns = stage_ns("pre_helper_total")
    helper_total_ns = stage_ns("helper_total")
    post_helper_total_ns = stage_ns("post_helper_total")
    parent_partition_residual_ns = total_ns - (
        pre_helper_total_ns + helper_total_ns + post_helper_total_ns
    )
    pre_helper_residual_ns = pre_helper_total_ns - sum(
        stage_ns(stage) for stage in PRE_HELPER_PROFILE_STAGES
    )
    helper_residual_ns = helper_total_ns - (
        helper_core_ns
        + sum(stage_ns(stage) for stage in HELPER_RESIDUAL_PROFILE_STAGES)
    )
    post_helper_residual_ns = post_helper_total_ns - (
        wait_ns + sum(stage_ns(stage) for stage in POST_HELPER_PROFILE_STAGES)
    )

    wall_ns = raw_integer(measurement["wall_ns"], "measurement.wall_ns")
    host_outer_ns = wall_ns - total_ns
    residual_limit_ns = max(
        DRIVER_RESIDUAL_ABS_LIMIT_NS,
        math.ceil(total_ns * DRIVER_RESIDUAL_REL_LIMIT),
    )
    require(total_ns > 0, "schema-2 raw driver total is positive")
    require(
        parent_partition_residual_ns == 0,
        "schema-2 pre/helper/post parents exactly partition raw driver total",
    )
    require(
        pre_helper_residual_ns >= 0
        and helper_residual_ns >= 0
        and post_helper_residual_ns >= 0,
        "schema-2 raw pre/helper/post residuals are non-negative",
    )
    require(
        driver_residual_ns
        == (
            parent_partition_residual_ns
            + pre_helper_residual_ns
            + helper_residual_ns
            + post_helper_residual_ns
        ),
        "schema-2 raw parent and leaf closures reconstruct driver residual",
    )
    require(
        0 <= driver_residual_ns <= residual_limit_ns,
        "schema-2 unaccounted driver time is within max(0.1ms, 0.5%)",
    )
    require(host_outer_ns >= 0, "schema-2 wall time contains raw driver total")

    derived_ms = {
        "driver_accounted_ms": driver_accounted_ns,
        "driver_residual_ms": driver_residual_ns,
        "host_outer_ms": host_outer_ns,
        "parent_partition_residual_ms": parent_partition_residual_ns,
        "pre_helper_residual_ms": pre_helper_residual_ns,
        "helper_residual_ms": helper_residual_ns,
        "post_helper_residual_ms": post_helper_residual_ns,
    }
    for field, nanoseconds in derived_ms.items():
        require(
            exact_ms(measurement[field], nanoseconds),
            f"schema-2 measurement.{field} matches independent raw-ns result",
        )
    require(
        close(
            measurement["driver_closure_fraction"],
            driver_accounted_ns / total_ns,
        ),
        "schema-2 measurement.driver_closure_fraction matches raw ns",
    )


def verify_profile(
    sample: Mapping[str, Any],
    allocations: int,
    logical_bytes: int,
    sample_schema: int,
) -> None:
    mode = str(sample["mode"])
    pages = logical_bytes // 4096
    profile = require_mapping(sample["profile"], "profile")
    first_boundary = require_mapping(profile["first_boundary"], "first boundary")
    timing_boundary = require_mapping(profile["timing_boundary"], "timing boundary")
    require(
        bool(profile["snapshot_after_cpu_reset_action5_action20"])
        and set(first_boundary) == set(BOUNDARY_FIELDS)
        and first_boundary == timing_boundary,
        "unchanged profile baseline was captured after action 20",
    )
    required_source = require_mapping(profile["required"], "required profile")
    delta_source = require_mapping(profile["delta"], "profile delta")
    if sample_schema == 2:
        required = {
            name: raw_integer(value, f"required profile {name}")
            for name, value in required_source.items()
        }
        delta = {
            name: raw_integer(value, f"profile delta {name}")
            for name, value in delta_source.items()
        }
    else:
        required = {name: int(value) for name, value in required_source.items()}
        delta = {name: int(value) for name, value in delta_source.items()}
    expected_required = expected_required_profile(
        mode, allocations, logical_bytes, sample_schema
    )
    require(required == expected_required, "worker required-profile contract is exact")
    require(
        all(name in delta and delta[name] == value for name, value in required.items()),
        "raw driver profile satisfies every exact required field",
    )
    require(
        delta["migrate_profile.api_total.calls"] == 0
        and all(delta[name] == 0 for name in GENERIC_COUNTERS),
        "timed path used neither generic migrate nor ioctl-90 batch prefetch",
    )
    require(
        all(name in delta for name in SLOT_COUNTERS),
        "raw profile contains every slot telemetry counter",
    )
    expected_slot = expected_slot_values(mode, allocations, logical_bytes)
    require(
        all(delta[name] == value for name, value in expected_slot.items()),
        "strict-64K slot counters match the selected mode exactly",
    )

    stage_sync = mode in STAGE_SYNC_MODES
    overwrite = mode in OVERWRITE_MODES
    measurement = require_mapping(sample["measurement"], "measurement")
    measurement_fields = {
        "total": "driver_profile_total_ms",
        "prepare": "prepare_ms",
        "unmap": "unmap_ms",
        "copy": "copy_ms",
        "map": "map_ms",
        "wait": "wait_ms",
        "prepare_complete": "prepare_complete_ms",
        "copy_complete": "copy_complete_ms",
        "map_complete": "map_complete_ms",
    }
    measurement_fields.update(
        {stage: f"{stage}_ms" for stage in DETAILED_PROFILE_STAGES}
    )
    stages = (
        LEGACY_SLOT_PROFILE_STAGES
        if sample_schema == 1
        else SLOT_PROFILE_STAGES_V2
    )
    for stage in stages:
        complete = stage in COMPLETE_PROFILE_STAGES
        enabled = not complete or stage_sync
        prefix = f"slot_direct_profile.{stage}"
        require(
            delta[f"{prefix}.calls"] == (4 if enabled else 0)
            and delta[f"{prefix}.pages"] == (pages if enabled else 0)
            and delta[f"{prefix}.bytes"] == (logical_bytes if enabled else 0),
            f"exact calls/pages/bytes profile contract for {stage}",
        )
        nanoseconds = delta[f"{prefix}.ns"]
        if not enabled or (stage == "unmap" and overwrite):
            require(nanoseconds == 0, f"zero profile time for disabled {stage}")
        else:
            require(nanoseconds > 0, f"positive profile time for enabled {stage}")
        measured = measurement[measurement_fields[stage]]
        if enabled:
            require(
                measured is not None and close(measured, nanoseconds / 1_000_000.0),
                f"measurement/profile consistency for {stage}",
            )
        else:
            require(measured is None, f"disabled {stage} has null measurement")

    if sample_schema == 2:
        verify_v2_closure_from_raw_ns(sample, delta)


def verify_sample(sample: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    mode = str(sample["mode"])
    sample_schema = schema_version(sample, "sample")
    require(
        sample["status"] == "ok" and mode in MODES,
        "sample schema/status/mode",
    )
    allocations = int(config["allocations"])
    allocation_bytes = int(config["allocation_bytes"])
    logical_bytes = int(config["logical_bytes"])
    blocks = int(config["expected_va_blocks"])
    require(
        sample["config"]
        == {
            "device": config["device"],
            "allocations": allocations,
            "allocation_bytes": allocation_bytes,
            "guard_bytes": config["guard_bytes"],
            "batch_size": config["batch_size"],
            "batches": 4,
            "logical_bytes": logical_bytes,
            "expected_va_blocks": blocks,
            "warmups": config["worker_warmups"],
            "slot_layer_id": config["slot_layer_id"],
            "slot_expert_id": config["slot_expert_id"],
        },
        "sample geometry matches its run",
    )
    abi = require_mapping(sample["abi"], "ABI")
    require(
        {name: int(value) for name, value in abi.items()} == EXPECTED_ABI,
        "ioctl 90/81/91, actions 5/20, and modes 8/12/16/17/18 ABI",
    )

    layout = require_mapping(sample["layout"], "layout")
    try:
        va_snapshot = [int(value, 0) for value in layout["va_snapshot"]]
    except (TypeError, ValueError) as error:
        raise RuntimeError("layout contains malformed managed VA") from error
    ordered = sorted(va_snapshot)
    gaps = [
        second - (first + allocation_bytes)
        for first, second in zip(ordered, ordered[1:])
    ]
    require(
        len(va_snapshot) == allocations
        and len(set(va_snapshot)) == allocations
        and all(value > 0 and value % VA_BLOCK_BYTES == 0 for value in va_snapshot)
        and all(gap >= int(config["guard_bytes"]) for gap in gaps)
        and int(layout["adjacent_test_ranges"]) == sum(gap == 0 for gap in gaps) == 0
        and int(layout["min_gap_bytes"]) == min(gaps)
        and bool(layout["all_regions_2m_aligned"]),
        "all regions are unique, guard-separated, and 2MiB aligned",
    )

    setup = require_mapping(sample["setup"], "setup")
    require(
        bool(setup["slot_vidmem_preallocated_before_timing"]),
        "mode-8 slot preallocation precedes timing",
    )
    verify_cpu_reset(
        require_mapping(
            setup["initial_cpu_reset_after_context_probe"], "initial CPU reset"
        ),
        allocations,
        logical_bytes,
        "initial CPU reset after context probe",
    )
    verify_outcome(
        require_mapping(setup["prealloc"], "preallocation outcome"),
        allocations,
        logical_bytes,
        transfer=False,
    )
    verify_precondition(
        require_mapping(sample["precondition"], "measured precondition"),
        allocations,
        logical_bytes,
    )
    verify_measurement(sample, allocations, logical_bytes, sample_schema)
    verify_validation(sample, allocations, logical_bytes)
    verify_profile(sample, allocations, logical_bytes, sample_schema)


def compare_tree(observed: Any, expected: Any, path: str = "aggregate") -> None:
    if isinstance(expected, dict):
        require(isinstance(observed, dict), f"{path}: expected object")
        require(set(observed) == set(expected), f"{path}: key set mismatch")
        for key in expected:
            compare_tree(observed[key], expected[key], f"{path}.{key}")
        return
    if expected is None:
        require(observed is None, f"{path}: expected null")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(
            isinstance(observed, (int, float)) and close(observed, expected),
            f"{path}: {observed}/{expected}",
        )
        return
    require(observed == expected, f"{path}: {observed!r}/{expected!r}")


def verify_schedule(samples: Sequence[Mapping[str, Any]], path: Path) -> None:
    require(
        hasattr(runner, "matched_abba_schedule"),
        f"{path}: runner lacks matched ABBA schedule contract",
    )
    global_indices = sorted(
        int(sample["schedule"]["global_index"]) for sample in samples
    )
    require(
        global_indices == list(range(32)),
        f"{path}: contiguous unique global sample indices 0..31",
    )
    for block_index in range(4):
        selected = sorted(
            (
                sample
                for sample in samples
                if int(sample["schedule"]["schedule_block_index"]) == block_index
            ),
            key=lambda sample: int(sample["schedule"]["global_index"]),
        )
        observed: list[tuple[str, int, str]] = []
        for sample in selected:
            schedule = require_mapping(sample["schedule"], "sample schedule")
            require(
                set(schedule)
                == {"schedule_block_index", "pair", "pair_slot", "global_index"},
                f"{path}: exact sample schedule fields",
            )
            observed.append(
                (str(schedule["pair"]), int(schedule["pair_slot"]), str(sample["mode"]))
            )
        expected = [
            (str(pair), int(slot), str(mode))
            for pair, slot, mode in runner.matched_abba_schedule(block_index)
        ]
        require(
            observed == expected,
            f"{path}: schedule block {block_index} is exact matched ABBA/BAAB",
        )


def verify_run(
    path: Path, document: dict[str, Any], require_full: bool
) -> dict[str, float]:
    run_schema = schema_version(document, str(path))
    require(
        document.get("status") == "ok",
        f"{path}: run schema/status",
    )
    config = require_mapping(document["config"], f"{path}: config")
    require(
        schema_version(config, f"{path}: config") == run_schema,
        f"{path}: summary and config schema versions match",
    )
    samples_value = document["samples"]
    require(isinstance(samples_value, list), f"{path}: samples is not a list")
    samples = [require_mapping(sample, f"{path}: raw sample") for sample in samples_value]
    profile = str(config["profile"])
    allocations = int(config["allocations"])
    allocation_bytes = int(config["allocation_bytes"])
    guard_bytes = int(config["guard_bytes"])
    batch_size = int(config["batch_size"])
    logical_bytes = int(config["logical_bytes"])
    expected_va_blocks = logical_bytes // VA_BLOCK_BYTES
    expected_geometry = (
        (4, 4 * MIB, 4 * MIB, 1, 16 * MIB, 8)
        if profile == "smoke"
        else (512, 4 * MIB, 4 * MIB, 128, 2 * GIB, 1024)
    )
    require(profile in ("smoke", "full"), f"{path}: known profile")
    require(
        (
            allocations,
            allocation_bytes,
            guard_bytes,
            batch_size,
            logical_bytes,
            expected_va_blocks,
        )
        == expected_geometry,
        f"{path}: exact {profile} geometry",
    )
    if require_full:
        require(profile == "full", f"{path}: full profile required")
    require(
        int(config["batches"]) == 4
        and int(config["expected_va_blocks"]) == expected_va_blocks
        and int(config["schedule_blocks"]) == 4
        and int(config["pairs_per_block"]) == 2
        and int(config["samples_per_block"]) == 8
        and int(config["samples_per_mode"]) == 8
        and config["schedule"] == "matched_pair_abba_independent_processes"
        and tuple(config["modes"]) == MODES,
        f"{path}: four-batch/four-block matched-ABBA config",
    )
    if run_schema == 2:
        require(
            int(config["sample_schema_version"]) == 2
            and config["profile_contract"] == runner.PROFILE_CONTRACT
            and config["timing_fields"] == list(runner.TIMING_FIELDS)
            and config["closure_timing_fields"]
            == list(runner.CLOSURE_TIMING_FIELDS),
            f"{path}: exact schema-2 sample/timing/profile contract",
        )
    require(
        int(config["slot_vidmem_budget_bytes"]) == 0
        or int(config["slot_vidmem_budget_bytes"]) >= logical_bytes,
        f"{path}: slot budget covers the complete working set",
    )
    require(
        len(samples) == 32
        and int(document["sample_count"]) == 32
        and int(document["unique_process_count"]) == 32,
        f"{path}: exactly 32 raw samples",
    )
    require(
        len({int(sample["pid"]) for sample in samples}) == 32,
        f"{path}: every sample ran in a fresh process",
    )
    require(
        len({str(sample["sample_id"]) for sample in samples}) == 32,
        f"{path}: unique sample identifiers",
    )
    require(
        all(schema_version(sample, f"{path}: sample") == run_schema for sample in samples),
        f"{path}: every sample schema matches summary/config schema",
    )
    verify_schedule(samples, path)
    for mode in MODES:
        require(
            sum(sample["mode"] == mode for sample in samples) == 8,
            f"{path}: exactly eight {mode} samples",
        )
    for sample in samples:
        verify_sample(sample, config)
    require(
        len({int(sample["validation"]["expected_checksum"]) for sample in samples})
        == 1,
        f"{path}: identical checksum contract across fresh processes",
    )
    verify_raw_artifacts(path, document)
    expected_aggregate = runner.aggregate(list(samples_value))
    compare_tree(document["aggregate"], expected_aggregate)
    return {
        name: float(value)
        for name, value in expected_aggregate["comparisons"].items()
    }


def main() -> int:
    args = parse_args()
    comparisons: list[dict[str, float]] = []
    documents: list[dict[str, Any]] = []
    try:
        for requested in args.runs:
            path, document = load(requested)
            comparisons.append(verify_run(path, document, args.require_full))
            documents.append(document)
        keys = sorted(comparisons[0])
        require(
            all(sorted(row) == keys for row in comparisons),
            "comparison key sets differ across runs",
        )
        rendered = " ".join(
            f"{key}={statistics.median(row[key] for row in comparisons):.4f}"
            for key in keys
        )
        extra = ""
        if args.pinned_result:
            require(
                all(int(document["config"]["logical_bytes"]) == 2 * GIB for document in documents),
                "--pinned-result requires only matching 2GiB runs",
            )
            pinned_gib_s = verify_pinned_result(args.pinned_result)
            overwrite_stage_gib_s = statistics.median(
                float(
                    document["aggregate"]["modes"]["overwrite_64k_stage_sync"]
                    ["copy_complete_gib_s_p50"]
                )
                for document in documents
            )
            extra = (
                f" pinned_copy_gib_s={pinned_gib_s:.3f}"
                f" overwrite_stage_copy_vs_pinned="
                f"{overwrite_stage_gib_s / pinned_gib_s:.4f}x"
            )
        print(f"VERIFIED runs={len(comparisons)} {rendered}{extra}")
        return 0
    except Exception as error:  # noqa: BLE001 - verification boundary.
        print(f"FAIL {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
