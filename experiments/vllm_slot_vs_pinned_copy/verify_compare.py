#!/usr/bin/env python3
"""Strictly verify and summarize exact-manifest slot vs pinned-copy runs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


EXPECTED_BYTES = 2_415_919_104
EXPECTED_GIB = EXPECTED_BYTES / 1024**3
EXPECTED_EXPERTS = 512
EXPECTED_REGIONS = 4_608
EXPECTED_BANK_BYTES = 603_979_776
EXPECTED_BANK_SLOTS = 1_152
EXPECTED_LAYERS = 4
EXPECTED_EXPERTS_PER_LAYER = 128
EXPECTED_BLOCKS_PER_LAYER = 288
EXPECTED_64K_PTES_PER_BLOCK = (2 * 1024 * 1024) // (64 * 1024)
EXPECTED_LAYER_TRANSITIONS = EXPECTED_LAYERS - 1
EXPECTED_REGISTERED_PLAN_CALLS = EXPECTED_LAYERS + 1
EXPECTED_REGISTERED_PLAN_EXECUTES = EXPECTED_LAYERS
EXPECTED_REGISTERED_PLAN_DESTROYS = EXPECTED_REGISTERED_PLAN_CALLS
EXPECTED_IDENTITY_RELEASE_IOCTLS = 1
EXPECTED_REGISTERED_PLAN_ABORT_CALLS = 1
EXPECTED_DIRECT_CUTOVER_PAGES = EXPECTED_LAYER_TRANSITIONS * EXPECTED_BANK_BYTES // 4096
MIN_SHARED_PUSH_REDUCTION = 4
MAX_U64 = (1 << 64) - 1
PLAN_GENERATION_CONTRACT = "module_lifetime_monotonic_nonzero_v1"
DEFAULT_PINNED_ANCHOR_MAX_DRIFT_PCT = 3.0

MODE_VIDMEM_PREALLOC = 8
MODE_SLOT_DIRECT_PER_BLOCK = 10
MODE_SLOT_DIRECT_SHARED_PUSH = 11
MODE_SLOT_DIRECT_64K_PTE = 12
MODE_SLOT_DIRECT_LAYER_TRANSITION = 20
MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE = 21


def registered_plan_specs() -> list[tuple[int, int, int]]:
    """Return ``(plan_id, layer_id, mode)`` in registration order."""

    return [
        (1, 0, MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE),
        *[
            ((layer_id << 2) | 2, layer_id, MODE_SLOT_DIRECT_LAYER_TRANSITION)
            for layer_id in range(EXPECTED_LAYERS)
        ],
    ]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"{path}:{number}: JSON object required")
        records.append(value)
    return records


def stats_snapshot(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    section = ""
    profile_re = re.compile(r"^\s{2}([a-zA-Z0-9_]+)\s+calls=(\d+)\s+ns=(\d+)")
    counter_re = re.compile(r"^(slot_[a-zA-Z0-9_.]+)\s+(\d+)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line[0].isspace() and line.endswith(":"):
            section = line[:-1]
            continue
        match = profile_re.match(line)
        if match and section == "slot_direct_profile":
            name, calls, ns = match.groups()
            values[f"slot_direct_profile.{name}.calls"] = int(calls)
            values[f"slot_direct_profile.{name}.ns"] = int(ns)
            continue
        match = counter_re.match(line)
        if match:
            values[match.group(1)] = int(match.group(2))
    return values


def stats_delta(before: Path, after: Path) -> dict[str, int]:
    left = stats_snapshot(before)
    right = stats_snapshot(after)
    return {key: right.get(key, 0) - left.get(key, 0) for key in right | left}


def verify_direct_cutover_counters(
    delta: dict[str, int],
    direct_cutover: bool | None,
    label: object,
) -> dict[str, int | bool | None]:
    """Verify the mode-20 no-CPU-restore path when the run declared it.

    ``None`` preserves verification of artifacts produced before the runner
    recorded the direct-cutover module parameter. New runs always declare a
    boolean and must therefore expose both driver counters, including exact
    zeroes for the legacy path.
    """

    if direct_cutover is None:
        return {
            "enabled": None,
            "calls": None,
            "pages": None,
        }

    expected_calls = EXPECTED_LAYER_TRANSITIONS if direct_cutover else 0
    expected_pages = EXPECTED_DIRECT_CUTOVER_PAGES if direct_cutover else 0
    calls_name = "slot_direct_transition_direct_cutover_calls"
    pages_name = "slot_direct_transition_direct_cutover_pages"
    require(
        delta.get(calls_name) == expected_calls,
        f"{label}: {calls_name}={delta.get(calls_name)!r}, expected {expected_calls}",
    )
    require(
        delta.get(pages_name) == expected_pages,
        f"{label}: {pages_name}={delta.get(pages_name)!r}, expected {expected_pages}",
    )
    return {
        "enabled": direct_cutover,
        "calls": expected_calls,
        "pages": expected_pages,
    }


def verify_registered_plan_counters(
    delta: dict[str, int],
    *,
    registered_plan: bool,
    declared: bool,
    reuse_bank: bool,
    layer_transition: bool,
    expected_legacy_ioctl_calls: int | None,
    label: object,
) -> dict[str, int | None]:
    """Verify the complete per-process slot control plane from driver deltas."""

    if not (registered_plan or declared):
        return {
            "register_calls": None,
            "execute_calls": None,
            "destroy_calls": None,
            "legacy_ioctl_calls": None,
            "legacy_mode20_calls": None,
            "abort_calls": None,
            "abort_successes": None,
            "abort_failures": None,
            "abort_released_slots": None,
            "abort_released_bytes": None,
            "abort_remaining_slots": None,
        }

    expected_register = EXPECTED_REGISTERED_PLAN_CALLS if registered_plan else 0
    expected_execute = EXPECTED_REGISTERED_PLAN_EXECUTES if registered_plan else 0
    expected_destroy = EXPECTED_REGISTERED_PLAN_DESTROYS if registered_plan else 0
    require(
        expected_legacy_ioctl_calls is not None,
        f"{label}: legacy ioctl expectation is unavailable",
    )
    expected_legacy_mode20 = (
        0
        if registered_plan
        else EXPECTED_LAYER_TRANSITIONS
        if reuse_bank and layer_transition
        else 0
    )
    values = {
        "register_calls": delta.get("slot_plan_register_calls"),
        "execute_calls": delta.get("slot_plan_execute_calls"),
        "destroy_calls": delta.get("slot_plan_destroy_calls"),
        "legacy_ioctl_calls": delta.get("slot_legacy_ioctl_calls"),
        "legacy_mode20_calls": delta.get("slot_legacy_transition_calls"),
        "abort_calls": delta.get("slot_session_abort_calls"),
        "abort_successes": delta.get("slot_session_abort_successes"),
        "abort_failures": delta.get("slot_session_abort_failures"),
        "abort_released_slots": delta.get("slot_session_abort_released_slots"),
        "abort_released_bytes": delta.get("slot_session_abort_released_bytes"),
        "abort_remaining_slots": delta.get("slot_session_abort_remaining_slots"),
    }
    for name, expected in (
        ("register_calls", expected_register),
        ("execute_calls", expected_execute),
        ("destroy_calls", expected_destroy),
        ("legacy_ioctl_calls", expected_legacy_ioctl_calls),
        ("legacy_mode20_calls", expected_legacy_mode20),
        (
            "abort_calls",
            EXPECTED_REGISTERED_PLAN_ABORT_CALLS if registered_plan else 0,
        ),
        (
            "abort_successes",
            EXPECTED_REGISTERED_PLAN_ABORT_CALLS if registered_plan else 0,
        ),
        ("abort_failures", 0),
        ("abort_released_slots", EXPECTED_BANK_SLOTS if registered_plan else 0),
        ("abort_released_bytes", EXPECTED_BANK_BYTES if registered_plan else 0),
        ("abort_remaining_slots", 0),
    ):
        require(
            values[name] == expected,
            f"{label}: driver {name}={values[name]!r}, expected {expected}",
        )
    return values


def expected_layer_mode8_ioctl_counts(
    *,
    init_bank: bool,
    reuse_bank: bool,
    layer_transition: bool,
) -> tuple[int, int]:
    """Return request-period and whole-process mode-8 ioctl counts.

    With an initialized fixed bank, the one allocation happens during model
    initialization and L0 only performs copy/map during the measured request.
    Otherwise L0 allocates the bank on demand.  L1-L3 use mode 20 only when
    fixed-bank transition is active.
    """

    request_calls = (
        0 if init_bank else 1 if reuse_bank and layer_transition else EXPECTED_LAYERS
    )
    return request_calls, request_calls + int(init_bank)


def verify_mode8_fd_candidate_ioctl_calls(
    details: list[dict[str, Any]],
    *,
    trace_enabled: bool,
    expected_logical_ioctl_calls: int,
    control_ioctl_calls: int | None,
    declared_ioctl_calls: int | None,
    label: object,
) -> int:
    """Resolve actual mode-8 kernel entries independently of driver totals.

    ``ioctl_calls`` only describes the successful fd candidate returned by the
    wrapper.  Before that success the wrapper may have issued the same mode-8
    request on other candidate UVM fds.  Those kernel entries are included in
    ``slot_legacy_ioctl_calls`` and are reported by
    ``fd_candidate_ioctl_calls``.

    Trace-on runs prove the count from userspace records.  Trace-off runs must
    predeclare the count established by a trace-on diagnostic; deriving it from
    ``slot_legacy_ioctl_calls`` would make the later driver comparison circular.
    """

    require(
        expected_logical_ioctl_calls >= 0,
        f"{label}: logical mode8 ioctl expectation is invalid",
    )
    if declared_ioctl_calls is not None:
        require(
            declared_ioctl_calls >= expected_logical_ioctl_calls,
            f"{label}: declared mode8 fd-candidate ioctl count is too small",
        )
    if control_ioctl_calls is not None:
        require(
            control_ioctl_calls >= expected_logical_ioctl_calls,
            f"{label}: control-summary mode8 fd-candidate ioctl count is too small",
        )
        if declared_ioctl_calls is not None:
            require(
                control_ioctl_calls == declared_ioctl_calls,
                f"{label}: control/declaration mode8 fd-candidate ioctl counts differ",
            )

    if not trace_enabled:
        require(
            control_ioctl_calls is not None,
            f"{label}: trace-off mode8 control summary is missing",
        )
        assert control_ioctl_calls is not None
        return control_ioctl_calls

    logical_ioctl_calls = 0
    candidate_ioctl_calls = 0
    for index, detail in enumerate(details):
        logical_calls = detail.get("ioctl_calls")
        candidate_attempts = detail.get("fd_candidate_attempts")
        candidate_calls = detail.get("fd_candidate_ioctl_calls")
        require(
            isinstance(logical_calls, int)
            and not isinstance(logical_calls, bool)
            and logical_calls > 0,
            f"{label}: mode8 trace {index} logical ioctl count is invalid",
        )
        require(
            isinstance(candidate_attempts, int)
            and not isinstance(candidate_attempts, bool)
            and candidate_attempts > 0,
            f"{label}: mode8 trace {index} fd-candidate attempts are missing",
        )
        require(
            isinstance(candidate_calls, int)
            and not isinstance(candidate_calls, bool)
            and candidate_calls >= logical_calls
            and candidate_calls >= candidate_attempts,
            f"{label}: mode8 trace {index} fd-candidate ioctl calls are invalid",
        )
        logical_ioctl_calls += logical_calls
        candidate_ioctl_calls += candidate_calls

    require(
        logical_ioctl_calls == expected_logical_ioctl_calls,
        f"{label}: traced logical mode8 ioctl count mismatch",
    )
    if declared_ioctl_calls is not None:
        require(
            candidate_ioctl_calls == declared_ioctl_calls,
            f"{label}: traced/declared mode8 fd-candidate ioctl counts differ",
        )
    if control_ioctl_calls is not None:
        require(
            candidate_ioctl_calls == control_ioctl_calls,
            f"{label}: trace/control mode8 fd-candidate ioctl counts differ",
        )
    return candidate_ioctl_calls


def verify_mode8_control_summary(
    path: Path,
    *,
    expected_logical_ioctl_calls: int,
    required: bool,
    label: object,
) -> int | None:
    """Validate the post-measurement, constant-overhead mode-8 summary."""

    if not path.is_file():
        require(not required, f"{label}: mode8 control summary is missing")
        return None
    payload = load_json(path)
    require(
        payload.get("schema") == "vllm_slot_preallocate_control_v1",
        f"{label}: mode8 control-summary schema mismatch",
    )
    results = payload.get("results")
    require(isinstance(results, list), f"{label}: mode8 control results are missing")
    assert isinstance(results, list)
    require(
        int(payload.get("preallocate_calls", -1)) == len(results),
        f"{label}: mode8 control call count mismatch",
    )
    logical_calls = 0
    candidate_attempts = 0
    candidate_calls = 0
    for index, result in enumerate(results):
        require(
            isinstance(result, dict)
            and result.get("ok") is True
            and int(result.get("mode", -1)) == MODE_VIDMEM_PREALLOC,
            f"{label}: mode8 control result {index} did not succeed",
        )
        assert isinstance(result, dict)
        logical = result.get("ioctl_calls")
        attempts = result.get("fd_candidate_attempts")
        calls = result.get("fd_candidate_ioctl_calls")
        require(
            isinstance(logical, int)
            and not isinstance(logical, bool)
            and logical > 0,
            f"{label}: mode8 control result {index} logical count is invalid",
        )
        require(
            isinstance(attempts, int)
            and not isinstance(attempts, bool)
            and attempts > 0,
            f"{label}: mode8 control result {index} candidate attempts are invalid",
        )
        require(
            isinstance(calls, int)
            and not isinstance(calls, bool)
            and calls >= logical
            and calls >= attempts,
            f"{label}: mode8 control result {index} candidate calls are invalid",
        )
        logical_calls += logical
        candidate_attempts += attempts
        candidate_calls += calls
    require(
        logical_calls == expected_logical_ioctl_calls
        and int(payload.get("logical_ioctl_calls", -1)) == logical_calls
        and int(payload.get("fd_candidate_attempts", -1)) == candidate_attempts
        and int(payload.get("fd_candidate_ioctl_calls", -1)) == candidate_calls,
        f"{label}: mode8 control-summary totals mismatch",
    )
    require(
        0
        < float(payload.get("measurement_start_epoch", 0))
        <= float(payload.get("measurement_end_epoch", 0)),
        f"{label}: mode8 control-summary measurement boundary is invalid",
    )
    return candidate_calls


def expected_slot_legacy_ioctl_calls(
    mode8_fd_candidate_ioctl_calls: int,
    *,
    registered_plan: bool,
) -> int:
    """Return all legacy kernel entries for one slot worker process."""

    require(
        mode8_fd_candidate_ioctl_calls >= 0,
        "mode8 fd-candidate ioctl count is invalid",
    )
    return (
        mode8_fd_candidate_ioctl_calls
        + (0 if registered_plan else EXPECTED_LAYERS)
        + (0 if registered_plan else EXPECTED_IDENTITY_RELEASE_IOCTLS)
    )


def verify_init_bank_trace(
    records: list[dict[str, Any]],
    *,
    init_bank: bool,
    precomputed_lanes: bool,
    label: object,
) -> dict[str, Any] | None:
    """Verify the initialization-only L0 bank reservation trace."""

    initializations = [
        record for record in records if record.get("operation") == "preallocate_layer"
    ]
    require(
        len(initializations) == int(init_bank),
        f"{label}: init-bank trace count mismatch",
    )
    if not init_bank:
        return None

    record = initializations[0]
    expected_experts = list(range(EXPECTED_EXPERTS_PER_LAYER))
    require(
        record.get("ok") is True
        and record.get("reason") == "preallocated_only"
        and record.get("preallocate_only") is True,
        f"{label}: init-bank reservation did not succeed",
    )
    require(
        int(record.get("layer_id", -1)) == 0
        and int(record.get("expert_count", -1)) == EXPECTED_EXPERTS_PER_LAYER
        and [int(value) for value in record.get("expert_ids", [])] == expected_experts,
        f"{label}: init-bank did not cover L0 experts",
    )
    require(
        int(record.get("requested_bytes", -1)) == EXPECTED_BANK_BYTES
        and len(record.get("statuses", {})) == EXPECTED_BANK_SLOTS
        and len(record.get("slot_ids", {})) == EXPECTED_BANK_SLOTS,
        f"{label}: init-bank region coverage mismatch",
    )
    require(
        record.get("logical_placement") == "slot_reserved"
        and record.get("ordinary_uvm_placement") == "cpu"
        and record.get("restore") is None
        and record.get("transition") is None
        and record.get("layer_transition_used") is False,
        f"{label}: init-bank made weights visible or transitioned an owner",
    )
    require(
        record.get("bank_reuse_enabled") is True
        and int(record.get("bank_rebind_slots", -1)) == 0
        and not record.get("evictions", []),
        f"{label}: init-bank did not reserve an empty fixed bank",
    )
    require(
        record.get("cache_before")
        == {
            "bytes": 0,
            "slots": 0,
        }
        and record.get("cache_after")
        == {
            "bytes": EXPECTED_BANK_BYTES,
            "slots": EXPECTED_BANK_SLOTS,
        },
        f"{label}: init-bank cache geometry mismatch",
    )
    require(
        int(record.get("bank_lane_plan_cache_hits", -1)) == 0
        and int(record.get("bank_lane_plan_builds", -1)) == int(precomputed_lanes),
        f"{label}: init-bank lane-plan accounting mismatch",
    )

    detail = record.get("preallocate")
    require(
        isinstance(detail, dict) and detail.get("ok") is True,
        f"{label}: init-bank mode8 failed",
    )
    assert isinstance(detail, dict)
    require(
        int(detail.get("mode", -1)) == MODE_VIDMEM_PREALLOC
        and int(detail.get("requested_count", -1)) == EXPECTED_BANK_SLOTS
        and int(detail.get("success_count", -1)) == EXPECTED_BANK_SLOTS
        and int(detail.get("failure_count", -1)) == 0,
        f"{label}: init-bank mode8 entry counts mismatch",
    )
    require(
        int(detail.get("requested_bytes", -1)) == EXPECTED_BANK_BYTES
        and int(detail.get("accepted_bytes", -1)) == EXPECTED_BANK_BYTES
        and int(detail.get("backing_reuse_count", -1)) == 0,
        f"{label}: init-bank mode8 byte/reuse accounting mismatch",
    )
    require(
        int(detail.get("ioctl_calls", -1)) == 1 and len(detail.get("batches", [])) == 1,
        f"{label}: init-bank was not exactly one mode8 ioctl",
    )
    return record


def verify_registered_plan_registration_trace(
    records: list[dict[str, Any]],
    *,
    registered_plan: bool,
    label: object,
) -> dict[str, Any] | None:
    """Verify the five initialization-time immutable plan registrations."""

    registrations = [
        record
        for record in records
        if record.get("operation") == "register_layer_plans"
    ]
    require(
        len(registrations) == int(registered_plan),
        f"{label}: registered-plan trace count mismatch",
    )
    if not registered_plan:
        return None

    record = registrations[0]
    require(
        record.get("ok") is True and record.get("reason") == "ok",
        f"{label}: registered-plan initialization failed",
    )
    generation = int(record.get("generation", -1))
    require(
        int(record.get("initial_layer", -1)) == 0
        and int(record.get("initial_plan_id", -1)) == 1
        and 0 < generation <= MAX_U64 - EXPECTED_REGISTERED_PLAN_EXECUTES,
        f"{label}: registered-plan initial owner/generation mismatch",
    )
    require(
        int(record.get("layer_count", -1)) == EXPECTED_LAYERS
        and int(record.get("lane_count", -1)) == EXPECTED_BANK_SLOTS
        and record.get("requested_bytes_per_layer")
        == {str(layer_id): EXPECTED_BANK_BYTES for layer_id in range(EXPECTED_LAYERS)},
        f"{label}: registered-plan layer geometry mismatch",
    )
    require(
        int(record.get("wall_ns", 0)) > 0,
        f"{label}: registered-plan wall time is missing",
    )

    results = record.get("register_results")
    require(
        isinstance(results, list) and len(results) == EXPECTED_REGISTERED_PLAN_CALLS,
        f"{label}: expected exactly five plan-register ioctls",
    )
    assert isinstance(results, list)
    for result, (plan_id, layer_id, mode) in zip(results, registered_plan_specs()):
        require(
            isinstance(result, dict)
            and result.get("ok") is True
            and result.get("reason") == "ok",
            f"{label}: plan registration {plan_id} failed",
        )
        assert isinstance(result, dict)
        require(
            int(result.get("plan_id", -1)) == plan_id
            and int(result.get("layer_id", -1)) == layer_id
            and int(result.get("mode", -1)) == mode
            and int(result.get("generation", -1)) == generation,
            f"{label}: plan registration {plan_id} identity mismatch",
        )
        require(
            int(result.get("requested_count", -1)) == EXPECTED_BANK_SLOTS
            and int(result.get("requested_bytes", -1)) == EXPECTED_BANK_BYTES,
            f"{label}: plan registration {plan_id} geometry mismatch",
        )
        require(
            int(result.get("ioctl_ret", -1)) == 0
            and int(result.get("errno", -1)) == 0
            and int(result.get("rm_status", -1)) == 0
            and int(result.get("ioctl_wall_ns", 0)) > 0
            and int(result.get("wall_ns", 0)) > 0,
            f"{label}: plan registration {plan_id} ioctl was not clean",
        )
    return record


def verify_registered_plan_execute_trace(
    record: dict[str, Any],
    *,
    layer_id: int,
    generation_start: int,
    label: object,
) -> dict[str, Any]:
    """Verify one request-time compact plan execute and its generation."""

    initial = layer_id == 0
    expected_plan_id = 1 if initial else (layer_id << 2) | 2
    expected_mode = (
        MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE
        if initial
        else MODE_SLOT_DIRECT_LAYER_TRANSITION
    )
    require(
        0 < generation_start <= MAX_U64 - EXPECTED_REGISTERED_PLAN_EXECUTES,
        f"{label}: registered-plan generation start is invalid",
    )
    expected_generation = generation_start + layer_id
    expected_new_generation = expected_generation + 1
    expected_owner_cutover_count = 0 if initial else EXPECTED_BANK_SLOTS

    require(
        record.get("registered_plan_used") is True
        and int(record.get("plan_id", -1)) == expected_plan_id
        and int(record.get("plan_generation", -1)) == expected_generation
        and int(record.get("plan_new_generation", -1)) == expected_new_generation,
        f"{label}: L{layer_id} compact-plan identity/generation mismatch",
    )
    require(
        record.get("preallocate") is None
        and record.get("bank_rebind_reason") == "registered_plan"
        and int(record.get("bank_rebind_slots", -1)) == expected_owner_cutover_count
        and not record.get("evictions", []),
        f"{label}: L{layer_id} compact plan used legacy rebind planning",
    )
    require(
        record.get("layer_transition_used") is (not initial)
        and record.get("layer_transition_reason")
        == ("initial_registered_plan" if initial else "ok"),
        f"{label}: L{layer_id} compact transition selection mismatch",
    )
    require(
        record.get("transition") == (None if initial else record.get("restore")),
        f"{label}: L{layer_id} compact transition trace is inconsistent",
    )
    require(
        int(record.get("bank_lane_plan_cache_hits", -1)) == 1
        and int(record.get("bank_lane_plan_builds", -1)) == 0
        and record.get("poisoned") is False,
        f"{label}: L{layer_id} compact plan cache/poison state mismatch",
    )
    expected_cache = {
        "bytes": EXPECTED_BANK_BYTES,
        "slots": EXPECTED_BANK_SLOTS,
    }
    require(
        record.get("cache_before") == expected_cache
        and record.get("cache_after") == expected_cache,
        f"{label}: L{layer_id} compact plan changed fixed-bank geometry",
    )

    detail = record.get("restore")
    require(
        isinstance(detail, dict)
        and detail.get("ok") is True
        and detail.get("reason") == "ok"
        and detail.get("plan_execute") is True,
        f"{label}: L{layer_id} compact execute failed",
    )
    assert isinstance(detail, dict)
    require(
        int(detail.get("plan_id", -1)) == expected_plan_id
        and int(detail.get("layer_id", -1)) == layer_id
        and int(detail.get("generation", -1)) == expected_generation
        and int(detail.get("new_generation", -1)) == expected_new_generation
        and int(detail.get("mode", -1)) == expected_mode,
        f"{label}: L{layer_id} compact execute result identity mismatch",
    )
    require(
        int(detail.get("requested_count", -1)) == EXPECTED_BANK_SLOTS
        and int(detail.get("success_count", -1)) == EXPECTED_BANK_SLOTS
        and int(detail.get("failure_count", -1)) == 0
        and int(detail.get("conflict_count", -1)) == 0,
        f"{label}: L{layer_id} compact execute entry counts mismatch",
    )
    require(
        int(detail.get("requested_bytes", -1)) == EXPECTED_BANK_BYTES
        and int(detail.get("accepted_bytes", -1)) == EXPECTED_BANK_BYTES
        and int(detail.get("remap_count", -1)) == EXPECTED_BANK_SLOTS
        and int(detail.get("tlb_invalidate_count", -1)) == EXPECTED_BLOCKS_PER_LAYER,
        f"{label}: L{layer_id} compact execute byte/PTE counts mismatch",
    )
    require(
        detail.get("owner_cutover") is (not initial)
        and int(detail.get("owner_cutover_count", -1)) == expected_owner_cutover_count
        and detail.get("owner_cutover_trusted") is True
        and detail.get("owner_state_indeterminate") is False
        and detail.get("poisoned") is False,
        f"{label}: L{layer_id} compact owner cutover is unproven",
    )
    require(
        int(detail.get("ioctl_calls", -1)) == 1
        and detail.get("batches") == []
        and int(detail.get("ioctl_ret", -1)) == 0
        and int(detail.get("errno", -1)) == 0
        and int(detail.get("rm_status", -1)) == 0
        and int(detail.get("ioctl_wall_ns", 0)) > 0
        and int(detail.get("wall_ns", 0)) > 0,
        f"{label}: L{layer_id} compact execute ioctl was not clean",
    )
    entry_statuses = detail.get("entry_statuses")
    require(
        isinstance(entry_statuses, list)
        and len(entry_statuses) == EXPECTED_BANK_SLOTS
        and all(type(status) is int and status == 0 for status in entry_statuses),
        f"{label}: L{layer_id} compact execute statuses are incomplete",
    )
    copy_pushes = int(detail.get("copy_push_count", -1))
    require(
        0 < copy_pushes <= EXPECTED_BANK_SLOTS // MIN_SHARED_PUSH_REDUCTION,
        f"{label}: L{layer_id} compact execute did not use shared pushes",
    )
    return detail


def verify_proof_mode(
    *,
    trace_enabled: bool,
    require_combined: bool,
    require_abba: bool,
    order: str | None,
) -> str:
    """Classify trace-on correctness and trace-off ABBA performance runs."""

    if require_abba:
        require(require_combined, "ABBA proof requires the combined mechanism")
        require(
            order in ("CSSC", "SCCS"),
            "ABBA proof has an invalid arm order",
        )
        return (
            "correctness_trace_on_abba"
            if trace_enabled
            else "performance_trace_off_abba"
        )
    return "correctness_trace_on" if trace_enabled else "counter_only_trace_off"


def result_metrics(value: dict[str, Any]) -> dict[str, Any]:
    measurement = value["measurement"]
    return {
        "elapsed_ms": float(measurement["elapsed_seconds"]) * 1000.0,
        "ttft_ms": float(measurement["ttft_ms"]["p50"]),
        "e2e_ms": float(measurement["request_latency_ms"]["p50"]),
        "output_sha256": measurement["aggregate_output_token_ids_sha256"],
        "after_used_mib": int(value["after_measurement_nvml"]["used_mib"]),
        "ready_used_mib": int(value["ready_nvml"]["used_mib"]),
        "torch_peak_reserved_bytes": int(
            value["measurement_torch_peak"]["max_reserved_bytes"]
        ),
    }


def expected_restore_mode(shared_push: bool, pte_64k: bool) -> int:
    if shared_push and pte_64k:
        return MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE
    if shared_push:
        return MODE_SLOT_DIRECT_SHARED_PUSH
    if pte_64k:
        return MODE_SLOT_DIRECT_64K_PTE
    return MODE_SLOT_DIRECT_PER_BLOCK


def verify_copy(
    arm_dir: Path,
    expected_names: set[str],
    measurement_start: float,
    measurement_end: float,
    trace_enabled: bool,
    copy_sync: bool,
) -> dict[str, Any]:
    require(
        not (arm_dir / "slot_control_summary.json").exists(),
        f"{arm_dir}: copy arm unexpectedly emitted a mode8 control summary",
    )
    records = load_jsonl(arm_dir / "cpu_offload_manifest.jsonl")
    params = [
        record for record in records if record.get("event") == "cpu_offload_parameter"
    ]
    copies = [
        record
        for record in records
        if record.get("event") == "cpu_offload_copy_forward"
        and measurement_start <= float(record.get("timestamp", -1.0)) <= measurement_end
    ]
    names = {str(record.get("full_parameter_name")) for record in params}
    require(names == expected_names, f"{arm_dir}: exact copy manifest mismatch")
    require(len(params) == 8, f"{arm_dir}: expected eight offloaded parameters")
    require(
        sum(int(record["bytes"]) for record in params) == EXPECTED_BYTES,
        f"{arm_dir}: copy parameter bytes mismatch",
    )
    require(
        all(record.get("cpu_data_pinned") is True for record in params),
        f"{arm_dir}: every copy source must be pinned",
    )
    require(
        all(record.get("copy_on_forward") is True for record in params),
        f"{arm_dir}: manifest does not prove copy-on-forward",
    )
    nonzero = [record for record in copies if int(record.get("copy_bytes", 0)) > 0]
    if trace_enabled:
        require(len(nonzero) == 4, f"{arm_dir}: expected four copied decoder layers")
        require(
            sum(int(record["copy_bytes"]) for record in nonzero) == EXPECTED_BYTES,
            f"{arm_dir}: forward copy bytes mismatch",
        )
        require(
            all(int(record.get("copy_tensor_count", -1)) == 2 for record in nonzero),
            f"{arm_dir}: each layer must copy w13 and w2 only",
        )
        require(
            all(bool(record.get("copy_sync")) is copy_sync for record in nonzero),
            f"{arm_dir}: copy synchronization mode mismatch",
        )
    else:
        require(not nonzero, f"{arm_dir}: copy trace was not disabled")
    submit_ms = (
        sum(float(record["copy_submit_us"]) for record in nonzero) / 1000.0
        if nonzero
        else None
    )
    cuda_ms = (
        sum(float(record["copy_cuda_ms"]) for record in nonzero)
        if copy_sync and nonzero
        else None
    )
    return {
        "parameter_count": len(params),
        "trace_enabled": trace_enabled,
        "copy_sync": copy_sync,
        "forward_count": len(nonzero) if trace_enabled else None,
        "copy_bytes": EXPECTED_BYTES,
        "copy_tensor_calls": (
            sum(int(record["copy_tensor_count"]) for record in nonzero)
            if trace_enabled
            else None
        ),
        "host_submit_ms": submit_ms,
        "copy_cuda_ms": cuda_ms,
        "effective_copy_gib_s": (
            EXPECTED_GIB / (cuda_ms / 1000.0) if cuda_ms is not None else None
        ),
    }


def verify_slot(
    arm_dir: Path,
    trace_enabled: bool,
    require_combined: bool,
    reuse_bank: bool,
    layer_batch: bool,
    precomputed_lanes: bool,
    layer_transition: bool,
    init_bank: bool,
    registered_plan: bool,
    registered_plan_declared: bool,
    declared_mode8_fd_candidate_ioctl_calls: int | None,
    shared_push: bool,
    pte_64k: bool,
    direct_cutover: bool | None,
) -> dict[str, Any]:
    records = load_jsonl(arm_dir / "slot_trace.jsonl")
    expert_restores = [
        record for record in records if record.get("operation") == "restore"
    ]
    layer_restores = [
        record for record in records if record.get("operation") == "restore_layer"
    ]
    if trace_enabled:
        init_preallocate = verify_init_bank_trace(
            records,
            init_bank=init_bank,
            precomputed_lanes=precomputed_lanes,
            label=arm_dir,
        )
        plan_registration = verify_registered_plan_registration_trace(
            records,
            registered_plan=registered_plan,
            label=arm_dir,
        )
        plan_generation_start = (
            int(plan_registration["generation"])
            if plan_registration is not None
            else None
        )
        if registered_plan:
            require(
                [record.get("operation") for record in records]
                == [
                    "preallocate_layer",
                    "register_layer_plans",
                    *("restore_layer" for _ in range(EXPECTED_LAYERS)),
                ],
                f"{arm_dir}: registered-plan trace ordering mismatch",
            )
        require(
            not (expert_restores and layer_restores),
            f"{arm_dir}: mixed expert/layer restore trace",
        )
        require(
            bool(layer_restores) is layer_batch,
            f"{arm_dir}: layer-batch config/trace mismatch",
        )
        restores = layer_restores if layer_batch else expert_restores
        expected_restore_records = EXPECTED_LAYERS if layer_batch else EXPECTED_EXPERTS
        require(
            len(restores) == expected_restore_records,
            f"{arm_dir}: restore count mismatch",
        )
        require(
            all(record.get("ok") is True for record in restores),
            f"{arm_dir}: failed restore",
        )
        require(
            sum(int(record["requested_bytes"]) for record in restores)
            == EXPECTED_BYTES,
            f"{arm_dir}: slot logical bytes mismatch",
        )
        if layer_batch:
            require(
                {int(record["layer_id"]) for record in restores}
                == set(range(EXPECTED_LAYERS)),
                f"{arm_dir}: layer restore did not cover L0-L3",
            )
            if registered_plan:
                require(
                    [int(record["layer_id"]) for record in restores]
                    == list(range(EXPECTED_LAYERS)),
                    f"{arm_dir}: registered-plan execution order is not L0-L3",
                )
            for record in restores:
                layer_id = int(record["layer_id"])
                transition_used = bool(record.get("layer_transition_used", False))
                expected_experts = list(range(EXPECTED_EXPERTS_PER_LAYER))
                require(
                    int(record.get("expert_count", -1)) == EXPECTED_EXPERTS_PER_LAYER
                    and [int(value) for value in record.get("expert_ids", [])]
                    == expected_experts,
                    f"{arm_dir}: L{layer_id} expert coverage mismatch",
                )
                require(
                    int(record["requested_bytes"]) == EXPECTED_BANK_BYTES,
                    f"{arm_dir}: L{layer_id} byte count mismatch",
                )
                require(
                    len(record.get("statuses", {})) == EXPECTED_BANK_SLOTS
                    and all(record["statuses"].values()),
                    f"{arm_dir}: L{layer_id} region statuses are incomplete",
                )
                if registered_plan:
                    require(
                        record.get("slot_ids") in (None, {}),
                        f"{arm_dir}: L{layer_id} compact trace leaked descriptors",
                    )
                else:
                    require(
                        len(record.get("slot_ids", {})) == EXPECTED_BANK_SLOTS,
                        f"{arm_dir}: L{layer_id} slot identities are incomplete",
                    )
                expected_transition = layer_transition and reuse_bank and layer_id > 0
                require(
                    transition_used is expected_transition,
                    f"{arm_dir}: L{layer_id} layer-transition selection mismatch",
                )
                expected_mode = (
                    MODE_SLOT_DIRECT_LAYER_TRANSITION
                    if expected_transition
                    else expected_restore_mode(shared_push, pte_64k)
                )
                operations = (
                    ()
                    if registered_plan
                    else (("transition", record.get("transition")),)
                    if transition_used
                    else tuple(
                        operation
                        for operation in (
                            (
                                "preallocate",
                                record.get("preallocate"),
                            ),
                            ("restore", record.get("restore")),
                        )
                        if not (
                            init_bank
                            and layer_id == 0
                            and operation[0] == "preallocate"
                        )
                    )
                )
                if registered_plan:
                    assert plan_generation_start is not None
                    verify_registered_plan_execute_trace(
                        record,
                        layer_id=layer_id,
                        generation_start=plan_generation_start,
                        label=arm_dir,
                    )
                else:
                    require(
                        record.get("registered_plan_used") in (None, False),
                        f"{arm_dir}: L{layer_id} unexpectedly used a registered plan",
                    )
                if init_bank and layer_id == 0:
                    require(
                        record.get("preallocate") is None,
                        f"{arm_dir}: L0 repeated mode8 after init-bank",
                    )
                for operation, detail in operations:
                    require(
                        isinstance(detail, dict) and detail.get("ok") is True,
                        f"{arm_dir}: L{layer_id} {operation} failed",
                    )
                    require(
                        int(detail.get("requested_count", -1)) == EXPECTED_BANK_SLOTS
                        and int(detail.get("success_count", -1)) == EXPECTED_BANK_SLOTS
                        and int(detail.get("failure_count", -1)) == 0,
                        f"{arm_dir}: L{layer_id} {operation} entry counts mismatch",
                    )
                    require(
                        int(detail.get("requested_bytes", -1)) == EXPECTED_BANK_BYTES
                        and int(detail.get("accepted_bytes", -1))
                        == EXPECTED_BANK_BYTES,
                        f"{arm_dir}: L{layer_id} {operation} bytes mismatch",
                    )
                    require(
                        int(detail.get("ioctl_calls", -1)) == 1
                        and len(detail.get("batches", [])) == 1,
                        f"{arm_dir}: L{layer_id} {operation} was not one ioctl",
                    )
                    require(
                        int(detail.get("mode", -1))
                        == (
                            MODE_VIDMEM_PREALLOC
                            if operation == "preallocate"
                            else expected_mode
                        ),
                        f"{arm_dir}: L{layer_id} {operation} mode mismatch",
                    )
                if transition_used:
                    transition_detail = record["transition"]
                    require(
                        record.get("preallocate") is None
                        and record.get("restore") == transition_detail,
                        f"{arm_dir}: L{layer_id} transition trace is inconsistent",
                    )
                    require(
                        transition_detail.get("owner_cutover") is True
                        and transition_detail.get("owner_cutover_trusted") is True
                        and int(transition_detail.get("owner_cutover_count", -1))
                        == EXPECTED_BANK_SLOTS,
                        f"{arm_dir}: L{layer_id} owner cutover is unproven",
                    )
                    require(
                        transition_detail.get("owner_state_indeterminate") is False,
                        f"{arm_dir}: L{layer_id} owner state is indeterminate",
                    )
                else:
                    restore_detail = record["restore"]
                    if require_combined or "owner_cutover" in restore_detail:
                        require(
                            restore_detail.get("owner_cutover") is False
                            and int(restore_detail.get("owner_cutover_count", -1)) == 0
                            and restore_detail.get("owner_state_indeterminate")
                            is False,
                            f"{arm_dir}: L{layer_id} non-transition owner state is invalid",
                        )
                require(
                    int(record["restore"].get("remap_count", -1))
                    == EXPECTED_BANK_SLOTS,
                    f"{arm_dir}: L{layer_id} remap count mismatch",
                )
                require(
                    int(record["restore"].get("tlb_invalidate_count", -1))
                    == (EXPECTED_BLOCKS_PER_LAYER if pte_64k else EXPECTED_BANK_SLOTS),
                    f"{arm_dir}: L{layer_id} TLB invalidate count mismatch",
                )
                layer_pushes = int(record["restore"].get("copy_push_count", -1))
                if shared_push:
                    require(
                        0
                        < layer_pushes
                        <= EXPECTED_BANK_SLOTS // MIN_SHARED_PUSH_REDUCTION,
                        f"{arm_dir}: L{layer_id} shared push reduction is insufficient",
                    )
                else:
                    require(
                        layer_pushes == EXPECTED_BANK_SLOTS,
                        f"{arm_dir}: L{layer_id} per-range push count mismatch",
                    )
        else:
            require(
                {
                    (int(record["layer_id"]), int(record["expert_id"]))
                    for record in restores
                }
                == {
                    (layer, expert)
                    for layer in range(EXPECTED_LAYERS)
                    for expert in range(EXPECTED_EXPERTS_PER_LAYER)
                },
                f"{arm_dir}: formal restore did not cover L0-L3 all experts",
            )
            require(
                all(
                    int(record["restore"]["requested_count"]) == 9
                    for record in restores
                ),
                f"{arm_dir}: slot expert does not contain nine envelopes",
            )
        require(
            all(
                int(record["cache_after"]["bytes"]) <= EXPECTED_BANK_BYTES
                and int(record["cache_after"]["slots"]) <= EXPECTED_BANK_SLOTS
                for record in restores
            ),
            f"{arm_dir}: slot cache exceeded the shared one-layer bank",
        )
        rebind_slots = sum(
            int(record.get("bank_rebind_slots", 0)) for record in restores
        )
        backing_reuses = sum(
            int(record["preallocate"].get("backing_reuse_count", 0))
            for record in restores
            if isinstance(record.get("preallocate"), dict)
        )
        transition_cutovers = sum(
            int(record["transition"].get("owner_cutover_count", 0))
            for record in restores
            if isinstance(record.get("transition"), dict)
        )
        lane_plan_cache_hits = sum(
            int(record.get("bank_lane_plan_cache_hits", 0)) for record in restores
        )
        request_lane_plan_builds = sum(
            int(record.get("bank_lane_plan_builds", 0)) for record in restores
        )
        init_lane_plan_builds = (
            int(init_preallocate.get("bank_lane_plan_builds", 0))
            if init_preallocate is not None
            else 0
        )
        lane_plan_builds = request_lane_plan_builds + init_lane_plan_builds
        capacity_records = [
            eviction for record in restores for eviction in record.get("evictions", [])
        ]
        if reuse_bank:
            require(
                rebind_slots == 3 * EXPECTED_BANK_SLOTS,
                f"{arm_dir}: fixed bank did not rebind every L1-L3 slot",
            )
            require(
                backing_reuses + transition_cutovers == rebind_slots,
                f"{arm_dir}: kernel reuse acknowledgements are incomplete",
            )
            if registered_plan:
                require(
                    not capacity_records,
                    f"{arm_dir}: compact plan repeated userspace eviction planning",
                )
            else:
                require(
                    len(capacity_records) == 3 * 128,
                    f"{arm_dir}: fixed-bank capacity transition count mismatch",
                )
                require(
                    all(
                        eviction.get("mechanism") == "bank_rebind"
                        and eviction.get("release") is None
                        for eviction in capacity_records
                    ),
                    f"{arm_dir}: a capacity transition used mode19",
                )
            if precomputed_lanes:
                require(
                    lane_plan_builds == 1
                    and lane_plan_cache_hits
                    == (EXPECTED_LAYERS if registered_plan else EXPECTED_LAYERS - 1),
                    f"{arm_dir}: fixed-bank lane plan was not built once and reused",
                )
            else:
                require(
                    lane_plan_builds == 0 and lane_plan_cache_hits == 0,
                    f"{arm_dir}: disabled lane-plan cache was used",
                )
        else:
            require(rebind_slots == 0, f"{arm_dir}: unexpected bank rebind")
    else:
        require(not records, f"{arm_dir}: slot trace was not disabled")
        init_preallocate = None
        plan_registration = None
        plan_generation_start = None
        restores = []
        rebind_slots = None
        backing_reuses = None
        transition_cutovers = None
        lane_plan_cache_hits = None
        lane_plan_builds = None
        request_lane_plan_builds = None
    plan_register_calls = (
        len(plan_registration.get("register_results", []))
        if plan_registration is not None
        else 0
        if trace_enabled
        else None
    )
    compact_execute_calls = (
        sum(record.get("registered_plan_used") is True for record in restores)
        if trace_enabled
        else None
    )
    legacy_mode20_calls = (
        sum(
            bool(record.get("layer_transition_used"))
            and record.get("registered_plan_used") is not True
            for record in restores
        )
        if trace_enabled
        else None
    )
    expected_plan_register_calls = (
        EXPECTED_REGISTERED_PLAN_CALLS if registered_plan else 0
    )
    expected_compact_execute_calls = (
        EXPECTED_REGISTERED_PLAN_EXECUTES if registered_plan else 0
    )
    expected_legacy_mode20_calls = (
        0
        if registered_plan
        else EXPECTED_LAYER_TRANSITIONS
        if reuse_bank and layer_transition
        else 0
    )
    if trace_enabled:
        require(
            plan_register_calls == expected_plan_register_calls,
            f"{arm_dir}: plan-register call count mismatch",
        )
        require(
            compact_execute_calls == expected_compact_execute_calls,
            f"{arm_dir}: compact execute count mismatch",
        )
        require(
            legacy_mode20_calls == expected_legacy_mode20_calls,
            f"{arm_dir}: legacy mode20 count mismatch",
        )
    evictions = (
        sum(len(record.get("evictions", [])) for record in restores)
        if trace_enabled
        else None
    )
    expected_request_mode8_calls: int | None = None
    expected_process_mode8_calls: int | None = None
    if layer_batch:
        expected_request_mode8_calls, expected_process_mode8_calls = (
            expected_layer_mode8_ioctl_counts(
                init_bank=init_bank,
                reuse_bank=reuse_bank,
                layer_transition=layer_transition,
            )
        )
    request_mode8_ioctl_calls = (
        sum(
            int(record["preallocate"]["ioctl_calls"])
            for record in restores
            if isinstance(record.get("preallocate"), dict)
        )
        if trace_enabled
        else expected_request_mode8_calls
        if layer_batch
        else None
    )
    init_mode8_ioctl_calls = (
        int(init_preallocate["preallocate"]["ioctl_calls"])
        if init_preallocate is not None
        else int(init_bank)
        if not trace_enabled and layer_batch
        else 0
    )
    process_mode8_ioctl_calls = (
        request_mode8_ioctl_calls + init_mode8_ioctl_calls
        if request_mode8_ioctl_calls is not None
        else None
    )
    if layer_batch:
        require(
            request_mode8_ioctl_calls == expected_request_mode8_calls,
            f"{arm_dir}: request-period mode8 ioctl count mismatch",
        )
        require(
            process_mode8_ioctl_calls == expected_process_mode8_calls,
            f"{arm_dir}: whole-process mode8 ioctl count mismatch",
        )
    mode8_details = [
        record["preallocate"]
        for record in restores
        if isinstance(record.get("preallocate"), dict)
    ]
    if init_preallocate is not None:
        mode8_details.insert(0, init_preallocate["preallocate"])
    mode8_fd_candidate_ioctl_calls: int | None = None
    mode8_control_ioctl_calls: int | None = None
    if registered_plan or registered_plan_declared:
        require(
            expected_process_mode8_calls is not None,
            f"{arm_dir}: logical mode8 ioctl expectation is unavailable",
        )
        assert expected_process_mode8_calls is not None
        mode8_control_ioctl_calls = verify_mode8_control_summary(
            arm_dir / "slot_control_summary.json",
            expected_logical_ioctl_calls=expected_process_mode8_calls,
            required=not trace_enabled,
            label=arm_dir,
        )
        mode8_fd_candidate_ioctl_calls = verify_mode8_fd_candidate_ioctl_calls(
            mode8_details,
            trace_enabled=trace_enabled,
            expected_logical_ioctl_calls=expected_process_mode8_calls,
            control_ioctl_calls=mode8_control_ioctl_calls,
            declared_ioctl_calls=declared_mode8_fd_candidate_ioctl_calls,
            label=arm_dir,
        )
    expected_legacy_ioctl_calls = (
        expected_slot_legacy_ioctl_calls(
            mode8_fd_candidate_ioctl_calls,
            registered_plan=registered_plan,
        )
        if mode8_fd_candidate_ioctl_calls is not None
        else None
    )
    mode8_wall_ms = (
        sum(
            float(record["preallocate"]["wall_ns"])
            for record in restores
            if isinstance(record.get("preallocate"), dict)
        )
        / 1e6
        if trace_enabled
        else None
    )
    init_mode8_wall_ms = (
        float(init_preallocate["preallocate"]["wall_ns"]) / 1e6
        if init_preallocate is not None
        else None
    )
    mode10_wall_ms = (
        sum(float(record["restore"]["wall_ns"]) for record in restores) / 1e6
        if trace_enabled
        else None
    )
    mode10_ioctl_ms = (
        sum(float(record["restore"]["ioctl_wall_ns"]) for record in restores) / 1e6
        if trace_enabled
        else None
    )
    outer_wall_ms = (
        sum(float(record["wall_ns"]) for record in restores) / 1e6
        if trace_enabled
        else None
    )
    delta = stats_delta(
        arm_dir / "latency_stats.before.txt",
        arm_dir / "latency_stats.after.txt",
    )
    registered_counters = verify_registered_plan_counters(
        delta,
        registered_plan=registered_plan,
        declared=registered_plan_declared,
        reuse_bank=reuse_bank,
        layer_transition=layer_transition,
        expected_legacy_ioctl_calls=expected_legacy_ioctl_calls,
        label=arm_dir,
    )
    driver_plan_register_calls = registered_counters["register_calls"]
    driver_plan_execute_calls = registered_counters["execute_calls"]
    driver_plan_destroy_calls = registered_counters["destroy_calls"]
    driver_legacy_ioctl_calls = registered_counters["legacy_ioctl_calls"]
    driver_legacy_mode20_calls = registered_counters["legacy_mode20_calls"]
    driver_abort_calls = registered_counters["abort_calls"]
    driver_abort_successes = registered_counters["abort_successes"]
    driver_abort_failures = registered_counters["abort_failures"]
    driver_abort_released_slots = registered_counters["abort_released_slots"]
    driver_abort_released_bytes = registered_counters["abort_released_bytes"]
    driver_abort_remaining_slots = registered_counters["abort_remaining_slots"]
    if registered_plan or registered_plan_declared:
        if trace_enabled:
            require(
                plan_register_calls == driver_plan_register_calls
                and compact_execute_calls == driver_plan_execute_calls
                and legacy_mode20_calls == driver_legacy_mode20_calls,
                f"{arm_dir}: registered-plan trace/driver counts differ",
            )
        else:
            plan_register_calls = driver_plan_register_calls
            compact_execute_calls = driver_plan_execute_calls
            legacy_mode20_calls = driver_legacy_mode20_calls
    for name in ("slot_direct_fallbacks", "slot_direct_errors"):
        require(delta.get(name, 0) == 0, f"{arm_dir}: {name} is nonzero")
    require(
        delta.get("slot_direct_requested_bytes") == EXPECTED_BYTES,
        f"{arm_dir}: driver slot byte delta mismatch",
    )
    require(
        delta.get("slot_direct_requested_ranges") == EXPECTED_REGIONS,
        f"{arm_dir}: driver slot region delta mismatch",
    )
    require(
        delta.get("slot_direct_completed_bytes") == EXPECTED_BYTES
        and delta.get("slot_direct_pte_map_bytes") == EXPECTED_BYTES,
        f"{arm_dir}: driver completed/PTE-map byte delta mismatch",
    )
    require(
        delta.get("slot_direct_completed_ranges") == EXPECTED_REGIONS
        and delta.get("slot_direct_pte_map_ranges") == EXPECTED_REGIONS,
        f"{arm_dir}: driver completed/PTE-map range delta mismatch",
    )
    require(
        delta.get("slot_direct_memcopy_methods", 0) >= EXPECTED_REGIONS,
        f"{arm_dir}: driver submitted too few CE memcopy methods",
    )
    require(
        delta.get("slot_direct_vidmem_allocations") == 0
        and delta.get("slot_direct_vidmem_evictions") == 0,
        f"{arm_dir}: slot-direct unexpectedly allocated or evicted backing",
    )
    copy_pushes = int(delta.get("slot_direct_copy_pushes", -1))
    if shared_push:
        require(
            0 < copy_pushes <= EXPECTED_REGIONS // MIN_SHARED_PUSH_REDUCTION,
            f"{arm_dir}: shared push did not reduce copy-push count by at least "
            f"{MIN_SHARED_PUSH_REDUCTION}x",
        )
    else:
        require(
            copy_pushes == EXPECTED_REGIONS,
            f"{arm_dir}: driver slot copy-push delta mismatch",
        )
    expected_pte_writes = (
        EXPECTED_LAYERS * EXPECTED_BLOCKS_PER_LAYER * EXPECTED_64K_PTES_PER_BLOCK
        if pte_64k
        else EXPECTED_BYTES // 4096
    )
    require(
        delta.get("slot_direct_pte_write_entries") == expected_pte_writes,
        f"{arm_dir}: driver PTE count mismatch",
    )
    expected_64k_blocks = EXPECTED_LAYERS * EXPECTED_BLOCKS_PER_LAYER if pte_64k else 0
    expected_4k_blocks = 0 if pte_64k else EXPECTED_REGIONS
    require(
        delta.get("slot_direct_pte_64k_blocks") == expected_64k_blocks,
        f"{arm_dir}: driver 64K block count mismatch",
    )
    require(
        delta.get("slot_direct_pte_4k_blocks") == expected_4k_blocks,
        f"{arm_dir}: driver 4K block count mismatch",
    )
    require(
        delta.get("slot_direct_pte_2m_blocks") == 0,
        f"{arm_dir}: unexpected 2M PTE block",
    )
    require(
        delta.get("slot_direct_blocks")
        == (expected_64k_blocks if pte_64k else EXPECTED_REGIONS),
        f"{arm_dir}: driver mapped-block count mismatch",
    )
    direct_calls = int(delta.get("slot_direct_ioctl_calls", -1))
    expected_direct_calls = EXPECTED_LAYERS if layer_batch else direct_calls
    expected_shared_calls = expected_direct_calls if shared_push else 0
    expected_per_block_calls = 0 if shared_push else expected_direct_calls
    require(
        delta.get("slot_direct_shared_calls") == expected_shared_calls,
        f"{arm_dir}: shared transaction count mismatch",
    )
    require(
        delta.get("slot_direct_per_block_calls") == expected_per_block_calls,
        f"{arm_dir}: per-block transaction count mismatch",
    )
    if trace_enabled and layer_batch:
        traced_pushes = sum(
            int(record["restore"].get("copy_push_count", -1)) for record in restores
        )
        require(
            traced_pushes == copy_pushes,
            f"{arm_dir}: trace/driver copy-push count mismatch",
        )
    if layer_batch:
        require(
            delta.get("slot_direct_ioctl_calls") == EXPECTED_LAYERS,
            f"{arm_dir}: mode10 did not use one ioctl per layer",
        )
        require(
            delta.get("slot_direct_successes") == EXPECTED_LAYERS,
            f"{arm_dir}: not every layer transaction succeeded",
        )
        require(
            delta.get("slot_direct_profile.total.calls") == EXPECTED_LAYERS,
            f"{arm_dir}: mode10 core transaction count mismatch",
        )
        for counter_name in (
            "slot_preallocate_ioctl_calls",
            "slot_mode8_ioctl_calls",
            "slot_direct_mode8_ioctl_calls",
        ):
            if counter_name in delta:
                assert expected_process_mode8_calls is not None
                require(
                    delta[counter_name] == expected_process_mode8_calls,
                    f"{arm_dir}: whole-process mode8 ioctl count mismatch",
                )
    if require_combined:
        expected_transition_ranges = (EXPECTED_LAYERS - 1) * EXPECTED_BANK_SLOTS
        expected_transition_blocks = (EXPECTED_LAYERS - 1) * EXPECTED_BLOCKS_PER_LAYER
        transition_counters = {
            "slot_direct_transition_fast_index_calls": EXPECTED_LAYERS - 1,
            "slot_direct_transition_group_clear_calls": EXPECTED_LAYERS - 1,
            "slot_direct_transition_group_clear_fallbacks": 0,
            "slot_direct_transition_clear_ranges": expected_transition_ranges,
            "slot_direct_transition_clear_unique_blocks": expected_transition_blocks,
        }
        for counter_name, expected_value in transition_counters.items():
            require(
                delta.get(counter_name) == expected_value,
                f"{arm_dir}: {counter_name} does not prove the combined transition",
            )
    direct_cutover_proof = verify_direct_cutover_counters(
        delta,
        direct_cutover,
        arm_dir,
    )
    mode8_ioctl_calls = request_mode8_ioctl_calls if layer_batch else None
    mode10_ioctl_calls = (
        sum(int(record["restore"]["ioctl_calls"]) for record in restores)
        if trace_enabled
        else delta.get("slot_direct_ioctl_calls")
    )
    driver_total_ms = delta.get("slot_direct_profile.total.ns", 0) / 1e6
    driver_copy_ms = delta.get("slot_direct_profile.copy.ns", 0) / 1e6
    driver_map_ms = delta.get("slot_direct_profile.map.ns", 0) / 1e6
    transition_preflight_ms = (
        delta.get("slot_direct_profile.transition_preflight.ns", 0) / 1e6
    )
    transition_clear_rebind_ms = (
        delta.get("slot_direct_profile.transition_clear_rebind.ns", 0) / 1e6
    )
    return {
        "trace_enabled": trace_enabled,
        "trace_schema": "layer" if layer_batch else "expert",
        "restore_layers": len(restores) if trace_enabled and layer_batch else None,
        "restore_experts": (
            sum(int(record.get("expert_count", 0)) for record in restores)
            if trace_enabled and layer_batch
            else len(restores)
            if trace_enabled
            else EXPECTED_EXPERTS
        ),
        "regions": EXPECTED_REGIONS,
        "bytes": EXPECTED_BYTES,
        "capacity_evictions": evictions,
        "reuse_bank": reuse_bank,
        "precomputed_lanes": precomputed_lanes,
        "layer_transition": layer_transition,
        "init_bank": init_bank,
        "registered_plan": registered_plan,
        "shared_push": shared_push,
        "pte_64k": pte_64k,
        "direct_cutover": direct_cutover,
        "direct_cutover_calls": direct_cutover_proof["calls"],
        "direct_cutover_pages": direct_cutover_proof["pages"],
        "restore_mode": expected_restore_mode(shared_push, pte_64k),
        "transition_mode": (
            MODE_SLOT_DIRECT_LAYER_TRANSITION
            if layer_transition and reuse_bank
            else None
        ),
        "copy_pushes": copy_pushes,
        "pte_write_entries": expected_pte_writes,
        "bank_rebind_slots": rebind_slots,
        "backing_reuse_count": backing_reuses,
        "transition_owner_cutover_count": transition_cutovers,
        "lane_plan_cache_hits": lane_plan_cache_hits,
        "lane_plan_builds": lane_plan_builds,
        "request_lane_plan_builds": request_lane_plan_builds,
        "plan_register_calls": plan_register_calls,
        "compact_execute_calls": compact_execute_calls,
        "legacy_mode20_calls": legacy_mode20_calls,
        "expected_plan_register_calls": expected_plan_register_calls,
        "expected_compact_execute_calls": expected_compact_execute_calls,
        "expected_plan_destroy_calls": (
            EXPECTED_REGISTERED_PLAN_DESTROYS if registered_plan else 0
        ),
        "expected_legacy_ioctl_calls": expected_legacy_ioctl_calls,
        "declared_mode8_fd_candidate_ioctl_calls": (
            declared_mode8_fd_candidate_ioctl_calls
        ),
        "mode8_fd_candidate_ioctl_calls": mode8_fd_candidate_ioctl_calls,
        "mode8_control_ioctl_calls": mode8_control_ioctl_calls,
        "expected_legacy_mode20_calls": expected_legacy_mode20_calls,
        "driver_plan_register_calls": driver_plan_register_calls,
        "driver_plan_execute_calls": driver_plan_execute_calls,
        "driver_plan_destroy_calls": driver_plan_destroy_calls,
        "driver_legacy_ioctl_calls": driver_legacy_ioctl_calls,
        "driver_legacy_mode20_calls": driver_legacy_mode20_calls,
        "driver_abort_calls": driver_abort_calls,
        "driver_abort_successes": driver_abort_successes,
        "driver_abort_failures": driver_abort_failures,
        "driver_abort_released_slots": driver_abort_released_slots,
        "driver_abort_released_bytes": driver_abort_released_bytes,
        "driver_abort_remaining_slots": driver_abort_remaining_slots,
        "plan_generation_start": plan_generation_start,
        "plan_generation_end": (
            plan_generation_start + EXPECTED_REGISTERED_PLAN_EXECUTES
            if plan_generation_start is not None
            else None
        ),
        "capacity_mode19_calls": (
            sum(
                eviction.get("release") is not None
                for record in restores
                for eviction in record.get("evictions", [])
            )
            if trace_enabled
            else None
        ),
        "mode8_wall_ms": mode8_wall_ms,
        "init_mode8_wall_ms": init_mode8_wall_ms,
        "mode10_wall_ms": mode10_wall_ms,
        "mode8_ioctl_calls": mode8_ioctl_calls,
        "process_mode8_ioctl_calls": process_mode8_ioctl_calls,
        "mode10_ioctl_calls": mode10_ioctl_calls,
        "mode10_ioctl_ms": mode10_ioctl_ms,
        "restore_outer_wall_ms": outer_wall_ms,
        "effective_mode10_gib_s": (
            EXPECTED_GIB / (mode10_wall_ms / 1000.0)
            if mode10_wall_ms is not None
            else None
        ),
        "driver_profile_total_ms": driver_total_ms,
        "driver_profile_copy_ms": driver_copy_ms,
        "driver_profile_map_ms": driver_map_ms,
        "driver_profile_transition_preflight_ms": transition_preflight_ms,
        "driver_profile_transition_clear_rebind_ms": transition_clear_rebind_ms,
        "driver_delta": delta,
    }


def median(values: list[float]) -> float:
    require(bool(values), "cannot summarize an empty arm")
    return float(statistics.median(values))


def verify_plan_generation_contract(value: str | None) -> str | None:
    """Accept the current contract marker and legacy artifacts without one."""

    require(
        value in (None, PLAN_GENERATION_CONTRACT),
        "slot plan-generation contract is invalid",
    )
    return value


def parse_pinned_anchor_threshold(value: str | None) -> float | None:
    """Parse a threshold recorded before a run; missing means legacy artifact."""

    if value is None:
        return None
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise AssertionError("pinned-anchor drift threshold is invalid") from None
    require(
        math.isfinite(threshold) and threshold >= 0.0,
        "pinned-anchor drift threshold is invalid",
    )
    return threshold


def compare_pinned_anchor_summaries(
    baseline: dict[str, Any],
    optimized: dict[str, Any],
) -> dict[str, Any]:
    """Apply the pre-registered cross-run pinned-anchor drift gate."""

    for label, result, registered_plan in (
        ("baseline", baseline, False),
        ("optimized", optimized, True),
    ):
        require(result.get("status") == "pass", f"{label} summary did not pass")
        require(
            result.get("output_token_ids_equal") is True,
            f"{label} output tokens differ",
        )
        require(
            result.get("trace_enabled") is False,
            f"{label} is not a trace-off performance run",
        )
        require(
            result.get("require_abba") is True
            and result.get("proof_mode") == "performance_trace_off_abba",
            f"{label} is not a verified ABBA performance proof",
        )
        require(
            result.get("slot_registered_plan") is registered_plan,
            f"{label} registered-plan mode is wrong",
        )
        require(
            result.get("slot_init_bank") is registered_plan,
            f"{label} init-bank mode is wrong",
        )
        for arm in ("copy", "slot"):
            aggregate = result.get("aggregate", {}).get(arm, {})
            require(
                int(aggregate.get("count", -1)) == 2,
                f"{label} {arm} arm is not an ABBA pair",
            )

    baseline_threshold = parse_pinned_anchor_threshold(
        baseline.get("pinned_anchor_max_drift_pct")
    )
    optimized_threshold = parse_pinned_anchor_threshold(
        optimized.get("pinned_anchor_max_drift_pct")
    )
    require(
        baseline_threshold is not None and optimized_threshold is not None,
        "pinned-anchor drift threshold was not pre-registered in both runs",
    )
    require(
        baseline_threshold == optimized_threshold,
        "baseline and optimized pinned-anchor thresholds differ",
    )

    baseline_copy = float(baseline["aggregate"]["copy"]["e2e_ms_median"])
    optimized_copy = float(optimized["aggregate"]["copy"]["e2e_ms_median"])
    baseline_slot = float(baseline["aggregate"]["slot"]["e2e_ms_median"])
    optimized_slot = float(optimized["aggregate"]["slot"]["e2e_ms_median"])
    require(
        all(
            math.isfinite(value) and value > 0.0
            for value in (
                baseline_copy,
                optimized_copy,
                baseline_slot,
                optimized_slot,
            )
        ),
        "anchor comparison contains invalid latency",
    )
    drift_pct = abs(optimized_copy / baseline_copy - 1.0) * 100.0
    anchor_pass = drift_pct <= baseline_threshold
    return {
        "schema": "vllm_slot_registered_plan_anchor_compare_v1",
        "status": "pass" if anchor_pass else "fail",
        "pinned_anchor_pass": anchor_pass,
        "pinned_anchor_max_drift_pct": baseline_threshold,
        "pinned_anchor_drift_pct": drift_pct,
        "baseline_pinned_e2e_ms": baseline_copy,
        "optimized_pinned_e2e_ms": optimized_copy,
        "baseline_slot_e2e_ms": baseline_slot,
        "optimized_registered_plan_slot_e2e_ms": optimized_slot,
        "registered_plan_slot_speedup": baseline_slot / optimized_slot,
        "failure_reason": (
            None
            if anchor_pass
            else "pinned anchor drift exceeded the pre-registered threshold"
        ),
    }


def compare_stage1_summaries(
    baseline: dict[str, Any],
    init_bank_only: dict[str, Any],
    registered_plan: dict[str, Any],
) -> dict[str, Any]:
    """Verify the three-arm phase-1 decomposition and quantify both gains."""

    arms = (
        ("baseline", baseline, False, False, "SCCS"),
        ("init_bank_only", init_bank_only, True, False, "CSSC"),
        ("registered_plan", registered_plan, True, True, "SCCS"),
    )
    thresholds: list[float] = []
    pinned_ms: dict[str, float] = {}
    slot_ms: dict[str, float] = {}
    for label, result, init_bank, uses_plan, expected_order in arms:
        require(result.get("status") == "pass", f"{label} summary did not pass")
        require(
            result.get("output_token_ids_equal") is True,
            f"{label} output tokens differ",
        )
        require(
            result.get("trace_enabled") is False
            and result.get("require_abba") is True
            and result.get("proof_mode") == "performance_trace_off_abba",
            f"{label} is not a trace-off ABBA performance proof",
        )
        require(
            result.get("order") == expected_order,
            f"{label} ABBA order is not the pre-registered design",
        )
        require(
            result.get("slot_init_bank") is init_bank
            and result.get("slot_registered_plan") is uses_plan,
            f"{label} mechanism flags are wrong",
        )
        expected_invariants = {
            "copy_sync": False,
            "slot_reuse_bank": True,
            "slot_layer_batch": True,
            "slot_precomputed_lanes": True,
            "slot_layer_transition": True,
            "slot_shared_push": True,
            "slot_64k_pte": True,
            "slot_direct_cutover": True,
            "slot_require_combined": True,
            "slot_plan_generation_contract": PLAN_GENERATION_CONTRACT,
            "logical_bytes": EXPECTED_BYTES,
            "shared_bank_bytes": EXPECTED_BANK_BYTES,
        }
        require(
            all(
                result.get(name) == value
                for name, value in expected_invariants.items()
            ),
            f"{label} changed a controlled experiment invariant",
        )
        for mechanism in ("copy", "slot"):
            aggregate = result.get("aggregate", {}).get(mechanism, {})
            require(
                int(aggregate.get("count", -1)) == 2,
                f"{label} {mechanism} arm is not an ABBA pair",
            )
        threshold = parse_pinned_anchor_threshold(
            result.get("pinned_anchor_max_drift_pct")
        )
        require(
            threshold is not None,
            f"{label} pinned-anchor threshold was not pre-registered",
        )
        thresholds.append(threshold)
        pinned_ms[label] = float(result["aggregate"]["copy"]["e2e_ms_median"])
        slot_ms[label] = float(result["aggregate"]["slot"]["e2e_ms_median"])

    require(
        len(set(thresholds)) == 1,
        "phase-1 pinned-anchor thresholds differ",
    )
    require(
        all(
            math.isfinite(value) and value > 0.0
            for value in (*pinned_ms.values(), *slot_ms.values())
        ),
        "phase-1 comparison contains invalid latency",
    )
    threshold = thresholds[0]
    anchor_spread_pct = (
        max(pinned_ms.values()) / min(pinned_ms.values()) - 1.0
    ) * 100.0
    anchor_pass = anchor_spread_pct <= threshold
    baseline_slot = slot_ms["baseline"]
    init_slot = slot_ms["init_bank_only"]
    plan_slot = slot_ms["registered_plan"]
    return {
        "schema": "vllm_slot_registered_plan_stage1_compare_v1",
        "status": "pass" if anchor_pass else "fail",
        "pinned_anchor_pass": anchor_pass,
        "pinned_anchor_max_drift_pct": threshold,
        "pinned_anchor_spread_pct": anchor_spread_pct,
        "pinned_anchor_e2e_ms": pinned_ms,
        "slot_e2e_ms": slot_ms,
        "init_bank_speedup": baseline_slot / init_slot,
        "init_bank_latency_reduction_ms": baseline_slot - init_slot,
        "init_bank_latency_reduction_pct": (1.0 - init_slot / baseline_slot) * 100.0,
        "registered_plan_incremental_speedup": init_slot / plan_slot,
        "registered_plan_incremental_latency_reduction_ms": init_slot - plan_slot,
        "registered_plan_incremental_latency_reduction_pct": (
            1.0 - plan_slot / init_slot
        )
        * 100.0,
        "combined_speedup": baseline_slot / plan_slot,
        "failure_reason": (
            None
            if anchor_pass
            else "three-arm pinned anchor spread exceeded the pre-registered threshold"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, nargs="?")
    comparison_group = parser.add_mutually_exclusive_group()
    comparison_group.add_argument(
        "--compare-pinned-anchors",
        nargs=2,
        type=Path,
        metavar=("BASELINE_SUMMARY", "OPTIMIZED_SUMMARY"),
    )
    comparison_group.add_argument(
        "--compare-stage1-arms",
        nargs=3,
        type=Path,
        metavar=(
            "BASELINE_SUMMARY",
            "INIT_BANK_SUMMARY",
            "REGISTERED_PLAN_SUMMARY",
        ),
    )
    parser.add_argument("--anchor-output", type=Path)
    args = parser.parse_args(argv)
    if (
        args.compare_pinned_anchors is not None
        or args.compare_stage1_arms is not None
    ):
        require(args.run_root is None, "run_root cannot accompany anchor comparison")
        if args.compare_stage1_arms is not None:
            baseline_path, init_bank_path, registered_plan_path = (
                args.compare_stage1_arms
            )
            comparison = compare_stage1_summaries(
                load_json(baseline_path),
                load_json(init_bank_path),
                load_json(registered_plan_path),
            )
        else:
            assert args.compare_pinned_anchors is not None
            baseline_path, optimized_path = args.compare_pinned_anchors
            comparison = compare_pinned_anchor_summaries(
                load_json(baseline_path),
                load_json(optimized_path),
            )
        if args.anchor_output is not None:
            args.anchor_output.parent.mkdir(parents=True, exist_ok=True)
            args.anchor_output.write_text(
                json.dumps(comparison, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(comparison, indent=2, sort_keys=True))
        return 0 if comparison["pinned_anchor_pass"] else 1

    require(args.run_root is not None, "run_root is required")
    require(args.anchor_output is None, "--anchor-output requires anchor comparison")
    assert args.run_root is not None
    root = args.run_root.resolve()
    inputs = load_json(root / "inputs" / "input_summary.json")
    require(int(inputs["logical_bytes"]) == EXPECTED_BYTES, "input byte count mismatch")
    expected_names = {
        line.strip()
        for line in (root / "inputs" / "exact_parameter_names.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    require(len(expected_names) == 8, "exact-name input must contain eight names")
    run_config = {
        key: value
        for line in (root / "run_config.env").read_text(encoding="utf-8").splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }
    trace_text = run_config.get("trace_enable", "1")
    require(trace_text in ("0", "1"), "trace mode is invalid")
    trace_enabled = trace_text == "1"
    copy_sync_text = run_config.get("copy_sync", "0")
    require(copy_sync_text in ("0", "1"), "copy sync mode is invalid")
    copy_sync = copy_sync_text == "1"
    reuse_text = run_config.get("slot_reuse_bank", "0")
    require(reuse_text in ("0", "1"), "slot reuse mode is invalid")
    reuse_bank = reuse_text == "1"
    layer_batch_text = run_config.get("slot_layer_batch", "0")
    require(layer_batch_text in ("0", "1"), "slot layer-batch mode is invalid")
    layer_batch = layer_batch_text == "1"
    precomputed_lanes_text = run_config.get("slot_precomputed_lanes", "0")
    require(
        precomputed_lanes_text in ("0", "1"),
        "slot precomputed-lanes mode is invalid",
    )
    precomputed_lanes = precomputed_lanes_text == "1"
    layer_transition_text = run_config.get("slot_layer_transition", "0")
    require(
        layer_transition_text in ("0", "1"),
        "slot layer-transition mode is invalid",
    )
    layer_transition = layer_transition_text == "1"
    init_bank_text = run_config.get("slot_init_bank", "0")
    require(init_bank_text in ("0", "1"), "slot init-bank mode is invalid")
    init_bank = init_bank_text == "1"
    if init_bank:
        require(
            layer_batch and reuse_bank and layer_transition,
            "slot init-bank requires layer batching, fixed-bank reuse, and "
            "layer transition",
        )
    registered_plan_declared = "slot_registered_plan" in run_config
    registered_plan_text = run_config.get("slot_registered_plan", "0")
    require(
        registered_plan_text in ("0", "1"),
        "slot registered-plan mode is invalid",
    )
    registered_plan = registered_plan_text == "1"
    mode8_candidate_text = run_config.get(
        "slot_expected_mode8_fd_candidate_ioctl_calls"
    )
    declared_mode8_fd_candidate_ioctl_calls: int | None = None
    if mode8_candidate_text is not None:
        require(
            mode8_candidate_text.isdigit(),
            "slot expected mode8 fd-candidate ioctl count is invalid",
        )
        declared_mode8_fd_candidate_ioctl_calls = int(mode8_candidate_text)
    plan_generation_contract = verify_plan_generation_contract(
        run_config.get("slot_plan_generation_contract")
    )
    pinned_anchor_max_drift_pct = parse_pinned_anchor_threshold(
        run_config.get("pinned_anchor_max_drift_pct")
    )
    shared_push_text = run_config.get("slot_shared_push", "0")
    require(shared_push_text in ("0", "1"), "slot shared-push mode is invalid")
    shared_push = shared_push_text == "1"
    pte_64k_text = run_config.get("slot_64k_pte", "0")
    require(pte_64k_text in ("0", "1"), "slot 64K-PTE mode is invalid")
    pte_64k = pte_64k_text == "1"
    if registered_plan:
        require(
            init_bank
            and layer_batch
            and reuse_bank
            and precomputed_lanes
            and layer_transition
            and shared_push
            and pte_64k,
            "slot registered-plan mode requires init-bank and the complete "
            "fixed-bank transition path",
        )
    direct_cutover_requested_text = run_config.get("slot_direct_cutover_requested")
    direct_cutover: bool | None
    direct_cutover_text = run_config.get("slot_direct_cutover", "unavailable")
    direct_cutover_final_text = run_config.get(
        "slot_direct_cutover_final", direct_cutover_text
    )
    if direct_cutover_requested_text is None:
        # Backward compatibility for artifacts captured before this parameter
        # and its driver counters were part of the comparison contract.
        direct_cutover = None
    else:
        require(
            direct_cutover_requested_text in ("0", "1"),
            "requested slot direct-cutover mode is invalid",
        )
        require(
            direct_cutover_text in ("0", "1")
            and direct_cutover_final_text in ("0", "1"),
            "slot direct-cutover parameter was unavailable or invalid",
        )
        require(
            direct_cutover_text
            == direct_cutover_requested_text
            == direct_cutover_final_text,
            "slot direct-cutover parameter changed or did not match the request",
        )
        direct_cutover = direct_cutover_requested_text == "1"
        if direct_cutover:
            require(
                reuse_bank and layer_transition,
                "direct cutover requires fixed-bank reuse and layer transition",
            )
    transition_fast_index_text = run_config.get(
        "slot_transition_fast_index", "unavailable"
    )
    require(
        transition_fast_index_text in ("0", "1", "unavailable"),
        "slot transition fast-index mode is invalid",
    )
    transition_fast_index_final_text = run_config.get(
        "slot_transition_fast_index_final", transition_fast_index_text
    )
    require(
        transition_fast_index_final_text in ("0", "1", "unavailable"),
        "final slot transition fast-index mode is invalid",
    )
    require_combined_text = run_config.get("slot_require_combined", "0")
    require(
        require_combined_text in ("0", "1"),
        "combined-mechanism requirement is invalid",
    )
    require_combined = require_combined_text == "1"
    require_abba_text = run_config.get("require_abba", "0")
    require(require_abba_text in ("0", "1"), "ABBA requirement is invalid")
    require_abba = require_abba_text == "1"
    if require_combined:
        require(not copy_sync, "combined performance proof requires COPY_SYNC=0")
        require(
            reuse_bank
            and layer_batch
            and precomputed_lanes
            and layer_transition
            and shared_push
            and pte_64k,
            "combined-mechanism proof is missing an optimization",
        )
        require(
            "slot_transition_fast_index_final" in run_config
            and transition_fast_index_text == "1"
            and transition_fast_index_final_text == "1",
            "combined-mechanism proof requires stable fast-index=1",
        )
        module_versions = (
            run_config.get("loaded_srcversion", ""),
            run_config.get("custom_srcversion", ""),
            run_config.get("loaded_srcversion_final", ""),
        )
        require(
            all(re.fullmatch(r"[0-9A-F]+", value) for value in module_versions)
            and len(set(module_versions)) == 1,
            "custom UVM module identity changed during the run",
        )
    proof_mode = verify_proof_mode(
        trace_enabled=trace_enabled,
        require_combined=require_combined,
        require_abba=require_abba,
        order=run_config.get("order"),
    )

    status_lines = (
        (root / "execution_status.tsv").read_text(encoding="utf-8").splitlines()
    )
    require(len(status_lines) >= 3, "comparison must contain both arms")
    runs: list[dict[str, Any]] = []
    output_hashes: set[str] = set()
    for line in status_lines[1:]:
        ordinal_text, arm, directory, rc_text = line.split("\t")
        require(int(rc_text) == 0, f"{directory}: nonzero return code")
        arm_dir = root / directory
        result = load_json(arm_dir / "result.json")
        require(result.get("status") == "ok", f"{directory}: result is not ok")
        require(
            int(result["measurement"]["completed_requests"]) == 1,
            f"{directory}: request count",
        )
        require(
            int(result["measurement"]["prompt_tokens"]) == 314,
            f"{directory}: prompt count",
        )
        require(
            int(result["measurement"]["output_tokens"]) == 1,
            f"{directory}: output count",
        )
        metrics = result_metrics(result)
        output_hashes.add(str(metrics["output_sha256"]))
        mechanism = (
            verify_copy(
                arm_dir,
                expected_names,
                float(result["measurement_start_epoch"]),
                float(result["measurement_end_epoch"]),
                trace_enabled,
                copy_sync,
            )
            if arm == "copy"
            else verify_slot(
                arm_dir,
                trace_enabled,
                require_combined,
                reuse_bank,
                layer_batch,
                precomputed_lanes,
                layer_transition,
                init_bank,
                registered_plan,
                registered_plan_declared,
                declared_mode8_fd_candidate_ioctl_calls,
                shared_push,
                pte_64k,
                direct_cutover,
            )
        )
        runs.append(
            {
                "ordinal": int(ordinal_text),
                "arm": arm,
                "directory": directory,
                "metrics": metrics,
                "mechanism": mechanism,
            }
        )
    require({run["arm"] for run in runs} == {"copy", "slot"}, "both arms required")
    if require_abba:
        actual_order = "".join("C" if run["arm"] == "copy" else "S" for run in runs)
        require(
            actual_order == run_config.get("order")
            and actual_order in ("CSSC", "SCCS"),
            "execution status does not prove the requested ABBA order",
        )
        require(
            sum(run["arm"] == "copy" for run in runs) == 2
            and sum(run["arm"] == "slot" for run in runs) == 2,
            "ABBA proof requires two independent runs per arm",
        )
        require(
            [run["ordinal"] for run in runs] == [1, 2, 3, 4]
            and len({run["directory"] for run in runs}) == 4,
            "ABBA proof requires unique ordered process directories",
        )
    require(len(output_hashes) == 1, "slot and copy output token IDs differ")

    by_arm: dict[str, dict[str, float | int]] = {}
    for arm in ("copy", "slot"):
        selected = [run for run in runs if run["arm"] == arm]
        by_arm[arm] = {
            "count": len(selected),
            "elapsed_ms_median": median(
                [run["metrics"]["elapsed_ms"] for run in selected]
            ),
            "ttft_ms_median": median([run["metrics"]["ttft_ms"] for run in selected]),
            "e2e_ms_median": median([run["metrics"]["e2e_ms"] for run in selected]),
            "after_used_mib_median": median(
                [run["metrics"]["after_used_mib"] for run in selected]
            ),
        }
    e2e_ratio = float(by_arm["slot"]["e2e_ms_median"]) / float(
        by_arm["copy"]["e2e_ms_median"]
    )
    summary = {
        "status": "pass",
        "schema": "vllm_slot_vs_pinned_copy_exact_v2",
        "run_root": str(root),
        "order": run_config.get("order"),
        "logical_bytes": EXPECTED_BYTES,
        "logical_gib": EXPECTED_GIB,
        "shared_bank_bytes": EXPECTED_BANK_BYTES,
        "shared_bank_slots": EXPECTED_BANK_SLOTS,
        "trace_enabled": trace_enabled,
        "copy_sync": copy_sync,
        "slot_reuse_bank": reuse_bank,
        "slot_layer_batch": layer_batch,
        "slot_precomputed_lanes": precomputed_lanes,
        "slot_layer_transition": layer_transition,
        "slot_init_bank": init_bank,
        "slot_registered_plan": registered_plan,
        "slot_registered_plan_declared": registered_plan_declared,
        "slot_expected_mode8_fd_candidate_ioctl_calls": (
            declared_mode8_fd_candidate_ioctl_calls
        ),
        "slot_plan_generation_contract": plan_generation_contract,
        "pinned_anchor_max_drift_pct": pinned_anchor_max_drift_pct,
        "slot_shared_push": shared_push,
        "slot_64k_pte": pte_64k,
        "slot_direct_cutover": direct_cutover,
        "slot_direct_cutover_requested": direct_cutover_requested_text,
        "slot_direct_cutover_parameter": direct_cutover_text,
        "slot_direct_cutover_parameter_final": direct_cutover_final_text,
        "slot_expected_restore_mode": expected_restore_mode(shared_push, pte_64k),
        "slot_expected_transition_mode": (
            MODE_SLOT_DIRECT_LAYER_TRANSITION
            if layer_transition and reuse_bank
            else None
        ),
        "slot_expected_pte_write_entries": (
            EXPECTED_LAYERS * EXPECTED_BLOCKS_PER_LAYER * EXPECTED_64K_PTES_PER_BLOCK
            if pte_64k
            else EXPECTED_BYTES // 4096
        ),
        "slot_transition_fast_index": transition_fast_index_text,
        "slot_transition_fast_index_final": transition_fast_index_final_text,
        "slot_require_combined": require_combined,
        "require_abba": require_abba,
        "proof_mode": proof_mode,
        "output_token_ids_equal": True,
        "runs": runs,
        "aggregate": by_arm,
        "slot_over_copy_e2e_ratio": e2e_ratio,
        "slot_significantly_slower_10pct": e2e_ratio > 1.10,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = [
        "# Exact slot vs pinned copy-on-forward",
        "",
        "| arm | runs | median elapsed | median TTFT | median E2E |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ("copy", "slot"):
        item = by_arm[arm]
        rows.append(
            f"| {arm} | {item['count']} | {item['elapsed_ms_median']:.3f} ms | "
            f"{item['ttft_ms_median']:.3f} ms | {item['e2e_ms_median']:.3f} ms |"
        )
    rows.extend(
        [
            "",
            f"Slot/copy E2E ratio: `{e2e_ratio:.4f}x`.",
            "",
            f"Payload: `{EXPECTED_BYTES}` bytes ({EXPECTED_GIB:.2f} GiB); "
            f"shared HBM bank: `{EXPECTED_BANK_BYTES}` bytes.",
            "",
            f"Slot restore mode: `{expected_restore_mode(shared_push, pte_64k)}`; "
            f"layer transition mode: "
            f"`{MODE_SLOT_DIRECT_LAYER_TRANSITION if layer_transition and reuse_bank else 'off'}`; "
            f"shared push: `{int(shared_push)}`; 64K PTE: `{int(pte_64k)}`.",
            "Direct cutover: "
            f"`{'legacy-artifact' if direct_cutover is None else int(direct_cutover)}`.",
        ]
    )
    (root / "SUMMARY.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
