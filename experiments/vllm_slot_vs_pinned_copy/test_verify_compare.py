"""Unit tests for direct-cutover proof in the comparison verifier."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).with_name("verify_compare.py")
RUNBOOK_PATH = MODULE_PATH.parents[1] / "vllm_slot_registered_plan_probe" / "RUNBOOK.md"
SPEC = importlib.util.spec_from_file_location("vllm_slot_verify_compare", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFY_COMPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY_COMPARE)
BENCHMARK_PATH = MODULE_PATH.parents[1] / "broad_offload_compare" / "benchmark_custom_offload.py"
BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "slot_control_benchmark", BENCHMARK_PATH
)
assert BENCHMARK_SPEC is not None and BENCHMARK_SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(BENCHMARK_SPEC)
BENCHMARK_SPEC.loader.exec_module(BENCHMARK)

CALLS = "slot_direct_transition_direct_cutover_calls"
PAGES = "slot_direct_transition_direct_cutover_pages"
PLAN_REGISTER = "slot_plan_register_calls"
PLAN_EXECUTE = "slot_plan_execute_calls"
PLAN_DESTROY = "slot_plan_destroy_calls"
LEGACY_IOCTL = "slot_legacy_ioctl_calls"
LEGACY_MODE20 = "slot_legacy_transition_calls"
ABORT_CALLS = "slot_session_abort_calls"
ABORT_SUCCESSES = "slot_session_abort_successes"
ABORT_FAILURES = "slot_session_abort_failures"
ABORT_RELEASED_SLOTS = "slot_session_abort_released_slots"
ABORT_RELEASED_BYTES = "slot_session_abort_released_bytes"
ABORT_REMAINING_SLOTS = "slot_session_abort_remaining_slots"


def valid_abort_delta(enabled: bool) -> dict[str, int]:
    return {
        ABORT_CALLS: int(enabled),
        ABORT_SUCCESSES: int(enabled),
        ABORT_FAILURES: 0,
        ABORT_RELEASED_SLOTS: (
            VERIFY_COMPARE.EXPECTED_BANK_SLOTS if enabled else 0
        ),
        ABORT_RELEASED_BYTES: (
            VERIFY_COMPARE.EXPECTED_BANK_BYTES if enabled else 0
        ),
        ABORT_REMAINING_SLOTS: 0,
    }


def valid_init_bank_record() -> dict[str, object]:
    return {
        "operation": "preallocate_layer",
        "ok": True,
        "reason": "preallocated_only",
        "preallocate_only": True,
        "layer_id": 0,
        "expert_count": VERIFY_COMPARE.EXPECTED_EXPERTS_PER_LAYER,
        "expert_ids": list(range(VERIFY_COMPARE.EXPECTED_EXPERTS_PER_LAYER)),
        "requested_bytes": VERIFY_COMPARE.EXPECTED_BANK_BYTES,
        "statuses": {
            f"region-{index}": False
            for index in range(VERIFY_COMPARE.EXPECTED_BANK_SLOTS)
        },
        "slot_ids": {
            f"region-{index}": index + 1
            for index in range(VERIFY_COMPARE.EXPECTED_BANK_SLOTS)
        },
        "logical_placement": "slot_reserved",
        "ordinary_uvm_placement": "cpu",
        "restore": None,
        "transition": None,
        "layer_transition_used": False,
        "bank_reuse_enabled": True,
        "bank_rebind_slots": 0,
        "evictions": [],
        "cache_before": {"bytes": 0, "slots": 0},
        "cache_after": {
            "bytes": VERIFY_COMPARE.EXPECTED_BANK_BYTES,
            "slots": VERIFY_COMPARE.EXPECTED_BANK_SLOTS,
        },
        "bank_lane_plan_cache_hits": 0,
        "bank_lane_plan_builds": 1,
        "preallocate": {
            "ok": True,
            "mode": VERIFY_COMPARE.MODE_VIDMEM_PREALLOC,
            "requested_count": VERIFY_COMPARE.EXPECTED_BANK_SLOTS,
            "success_count": VERIFY_COMPARE.EXPECTED_BANK_SLOTS,
            "failure_count": 0,
            "requested_bytes": VERIFY_COMPARE.EXPECTED_BANK_BYTES,
            "accepted_bytes": VERIFY_COMPARE.EXPECTED_BANK_BYTES,
            "backing_reuse_count": 0,
            "ioctl_calls": 1,
            "fd_candidate_attempts": 2,
            "fd_candidate_ioctl_calls": 2,
            "batches": [{}],
            "wall_ns": 1,
        },
    }


def valid_mode8_control_summary() -> dict[str, object]:
    return {
        "schema": "vllm_slot_preallocate_control_v1",
        "measurement_start_epoch": 10.0,
        "measurement_end_epoch": 11.0,
        "preallocate_calls": 1,
        "logical_ioctl_calls": 1,
        "fd_candidate_attempts": 2,
        "fd_candidate_ioctl_calls": 2,
        "results": [
            {
                "ok": True,
                "reason": "ok",
                "mode": VERIFY_COMPARE.MODE_VIDMEM_PREALLOC,
                "ioctl_calls": 1,
                "fd_candidate_attempts": 2,
                "fd_candidate_ioctl_calls": 2,
            }
        ],
    }


def valid_registered_plan_record(generation: int = 1) -> dict[str, object]:
    register_results = []
    for plan_id, layer_id, mode in VERIFY_COMPARE.registered_plan_specs():
        register_results.append(
            {
                "ok": True,
                "reason": "ok",
                "plan_id": plan_id,
                "layer_id": layer_id,
                "mode": mode,
                "generation": generation,
                "requested_count": VERIFY_COMPARE.EXPECTED_BANK_SLOTS,
                "requested_bytes": VERIFY_COMPARE.EXPECTED_BANK_BYTES,
                "ioctl_ret": 0,
                "errno": 0,
                "rm_status": 0,
                "ioctl_wall_ns": 1,
                "wall_ns": 2,
            }
        )
    return {
        "operation": "register_layer_plans",
        "ok": True,
        "reason": "ok",
        "initial_layer": 0,
        "initial_plan_id": 1,
        "generation": generation,
        "layer_count": VERIFY_COMPARE.EXPECTED_LAYERS,
        "lane_count": VERIFY_COMPARE.EXPECTED_BANK_SLOTS,
        "requested_bytes_per_layer": {
            str(layer_id): VERIFY_COMPARE.EXPECTED_BANK_BYTES
            for layer_id in range(VERIFY_COMPARE.EXPECTED_LAYERS)
        },
        "register_results": register_results,
        "wall_ns": 10,
    }


def valid_registered_execute_record(
    layer_id: int,
    generation_start: int = 1,
) -> dict[str, object]:
    initial = layer_id == 0
    plan_id = 1 if initial else (layer_id << 2) | 2
    generation = generation_start + layer_id
    cutover_count = 0 if initial else VERIFY_COMPARE.EXPECTED_BANK_SLOTS
    detail = {
        "ok": True,
        "reason": "ok",
        "plan_execute": True,
        "plan_id": plan_id,
        "layer_id": layer_id,
        "generation": generation,
        "new_generation": generation + 1,
        "mode": (
            VERIFY_COMPARE.MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE
            if initial
            else VERIFY_COMPARE.MODE_SLOT_DIRECT_LAYER_TRANSITION
        ),
        "requested_count": VERIFY_COMPARE.EXPECTED_BANK_SLOTS,
        "success_count": VERIFY_COMPARE.EXPECTED_BANK_SLOTS,
        "failure_count": 0,
        "conflict_count": 0,
        "requested_bytes": VERIFY_COMPARE.EXPECTED_BANK_BYTES,
        "accepted_bytes": VERIFY_COMPARE.EXPECTED_BANK_BYTES,
        "remap_count": VERIFY_COMPARE.EXPECTED_BANK_SLOTS,
        "tlb_invalidate_count": VERIFY_COMPARE.EXPECTED_BLOCKS_PER_LAYER,
        "copy_push_count": 1,
        "owner_cutover": not initial,
        "owner_cutover_count": cutover_count,
        "owner_cutover_trusted": True,
        "owner_state_indeterminate": False,
        "poisoned": False,
        "ioctl_calls": 1,
        "batches": [],
        "ioctl_ret": 0,
        "errno": 0,
        "rm_status": 0,
        "ioctl_wall_ns": 1,
        "wall_ns": 2,
        "entry_statuses": [0] * VERIFY_COMPARE.EXPECTED_BANK_SLOTS,
    }
    cache = {
        "bytes": VERIFY_COMPARE.EXPECTED_BANK_BYTES,
        "slots": VERIFY_COMPARE.EXPECTED_BANK_SLOTS,
    }
    return {
        "registered_plan_used": True,
        "plan_id": plan_id,
        "plan_generation": generation,
        "plan_new_generation": generation + 1,
        "preallocate": None,
        "restore": detail,
        "transition": None if initial else detail,
        "bank_rebind_reason": "registered_plan",
        "bank_rebind_slots": cutover_count,
        "evictions": [],
        "layer_transition_used": not initial,
        "layer_transition_reason": "initial_registered_plan" if initial else "ok",
        "bank_lane_plan_cache_hits": 1,
        "bank_lane_plan_builds": 0,
        "poisoned": False,
        "cache_before": cache,
        "cache_after": cache,
    }


def valid_anchor_summary(
    *,
    registered_plan: bool,
    copy_ms: float,
    slot_ms: float,
    threshold_pct: float | None = 3.0,
    init_bank: bool | None = None,
    order: str = "SCCS",
) -> dict[str, object]:
    if init_bank is None:
        init_bank = registered_plan
    result: dict[str, object] = {
        "status": "pass",
        "output_token_ids_equal": True,
        "trace_enabled": False,
        "require_abba": True,
        "proof_mode": "performance_trace_off_abba",
        "order": order,
        "slot_registered_plan": registered_plan,
        "slot_init_bank": init_bank,
        "copy_sync": False,
        "slot_reuse_bank": True,
        "slot_layer_batch": True,
        "slot_precomputed_lanes": True,
        "slot_layer_transition": True,
        "slot_shared_push": True,
        "slot_64k_pte": True,
        "slot_direct_cutover": True,
        "slot_require_combined": True,
        "slot_plan_generation_contract": VERIFY_COMPARE.PLAN_GENERATION_CONTRACT,
        "logical_bytes": VERIFY_COMPARE.EXPECTED_BYTES,
        "shared_bank_bytes": VERIFY_COMPARE.EXPECTED_BANK_BYTES,
        "aggregate": {
            "copy": {"count": 2, "e2e_ms_median": copy_ms},
            "slot": {"count": 2, "e2e_ms_median": slot_ms},
        },
    }
    if threshold_pct is not None:
        result["pinned_anchor_max_drift_pct"] = threshold_pct
    return result


def test_legacy_artifact_does_not_require_direct_cutover_counters() -> None:
    assert VERIFY_COMPARE.verify_direct_cutover_counters({}, None, "legacy") == {
        "enabled": None,
        "calls": None,
        "pages": None,
    }


def test_disabled_direct_cutover_requires_exact_zero_counters() -> None:
    assert VERIFY_COMPARE.verify_direct_cutover_counters(
        {CALLS: 0, PAGES: 0}, False, "legacy-path"
    ) == {
        "enabled": False,
        "calls": 0,
        "pages": 0,
    }


def test_enabled_direct_cutover_requires_exact_transition_counts() -> None:
    expected_calls = VERIFY_COMPARE.EXPECTED_LAYER_TRANSITIONS
    expected_pages = VERIFY_COMPARE.EXPECTED_DIRECT_CUTOVER_PAGES

    assert VERIFY_COMPARE.verify_direct_cutover_counters(
        {CALLS: expected_calls, PAGES: expected_pages}, True, "direct-path"
    ) == {
        "enabled": True,
        "calls": expected_calls,
        "pages": expected_pages,
    }


@pytest.mark.parametrize(
    ("direct_cutover", "delta"),
    [
        (False, {}),
        (False, {CALLS: 0}),
        (False, {PAGES: 0}),
        (False, {CALLS: 1, PAGES: 0}),
        (False, {CALLS: 0, PAGES: 1}),
        (True, {}),
        (True, {CALLS: VERIFY_COMPARE.EXPECTED_LAYER_TRANSITIONS}),
        (True, {PAGES: VERIFY_COMPARE.EXPECTED_DIRECT_CUTOVER_PAGES}),
        (
            True,
            {
                CALLS: VERIFY_COMPARE.EXPECTED_LAYER_TRANSITIONS - 1,
                PAGES: VERIFY_COMPARE.EXPECTED_DIRECT_CUTOVER_PAGES,
            },
        ),
        (
            True,
            {
                CALLS: VERIFY_COMPARE.EXPECTED_LAYER_TRANSITIONS,
                PAGES: VERIFY_COMPARE.EXPECTED_DIRECT_CUTOVER_PAGES - 1,
            },
        ),
    ],
)
def test_missing_or_inexact_direct_cutover_counters_fail(
    direct_cutover: bool,
    delta: dict[str, int],
) -> None:
    with pytest.raises(AssertionError):
        VERIFY_COMPARE.verify_direct_cutover_counters(
            delta, direct_cutover, "bad-proof"
        )


def test_registered_plan_driver_counters_are_exact() -> None:
    assert VERIFY_COMPARE.verify_registered_plan_counters(
        {
            PLAN_REGISTER: VERIFY_COMPARE.EXPECTED_REGISTERED_PLAN_CALLS,
            PLAN_EXECUTE: VERIFY_COMPARE.EXPECTED_REGISTERED_PLAN_EXECUTES,
            PLAN_DESTROY: VERIFY_COMPARE.EXPECTED_REGISTERED_PLAN_DESTROYS,
            LEGACY_IOCTL: 2,
            LEGACY_MODE20: 0,
            **valid_abort_delta(True),
        },
        registered_plan=True,
        declared=True,
        reuse_bank=True,
        layer_transition=True,
        expected_legacy_ioctl_calls=2,
        label="compact",
    ) == {
        "register_calls": 5,
        "execute_calls": 4,
        "destroy_calls": 5,
        "legacy_ioctl_calls": 2,
        "legacy_mode20_calls": 0,
        "abort_calls": 1,
        "abort_successes": 1,
        "abort_failures": 0,
        "abort_released_slots": VERIFY_COMPARE.EXPECTED_BANK_SLOTS,
        "abort_released_bytes": VERIFY_COMPARE.EXPECTED_BANK_BYTES,
        "abort_remaining_slots": 0,
    }


def test_declared_legacy_path_requires_zero_plan_and_three_mode20_calls() -> None:
    assert VERIFY_COMPARE.verify_registered_plan_counters(
        {
            PLAN_REGISTER: 0,
            PLAN_EXECUTE: 0,
            PLAN_DESTROY: 0,
            LEGACY_IOCTL: 7,
            LEGACY_MODE20: 3,
            **valid_abort_delta(False),
        },
        registered_plan=False,
        declared=True,
        reuse_bank=True,
        layer_transition=True,
        expected_legacy_ioctl_calls=7,
        label="legacy",
    ) == {
        "register_calls": 0,
        "execute_calls": 0,
        "destroy_calls": 0,
        "legacy_ioctl_calls": 7,
        "legacy_mode20_calls": 3,
        "abort_calls": 0,
        "abort_successes": 0,
        "abort_failures": 0,
        "abort_released_slots": 0,
        "abort_released_bytes": 0,
        "abort_remaining_slots": 0,
    }


def test_historical_artifact_without_plan_declaration_keeps_compatibility() -> None:
    assert VERIFY_COMPARE.verify_registered_plan_counters(
        {},
        registered_plan=False,
        declared=False,
        reuse_bank=True,
        layer_transition=True,
        expected_legacy_ioctl_calls=None,
        label="history",
    ) == {
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


@pytest.mark.parametrize(
    "delta",
    [
        {},
        {
            PLAN_REGISTER: 4,
            PLAN_EXECUTE: 4,
            PLAN_DESTROY: 5,
            LEGACY_IOCTL: 3,
            LEGACY_MODE20: 0,
        },
        {
            PLAN_REGISTER: 5,
            PLAN_EXECUTE: 3,
            PLAN_DESTROY: 5,
            LEGACY_IOCTL: 3,
            LEGACY_MODE20: 0,
        },
        {
            PLAN_REGISTER: 5,
            PLAN_EXECUTE: 4,
            PLAN_DESTROY: 0,
            LEGACY_IOCTL: 3,
            LEGACY_MODE20: 0,
        },
        {
            PLAN_REGISTER: 5,
            PLAN_EXECUTE: 4,
            PLAN_DESTROY: 5,
            LEGACY_IOCTL: 2,
            LEGACY_MODE20: 0,
        },
        {
            PLAN_REGISTER: 5,
            PLAN_EXECUTE: 4,
            PLAN_DESTROY: 5,
            LEGACY_IOCTL: 4,
            LEGACY_MODE20: 0,
        },
        {
            PLAN_REGISTER: 5,
            PLAN_EXECUTE: 4,
            PLAN_DESTROY: 5,
            LEGACY_IOCTL: 3,
            LEGACY_MODE20: 1,
        },
    ],
)
def test_registered_plan_driver_counter_mismatch_fails(
    delta: dict[str, int],
) -> None:
    with pytest.raises(AssertionError):
        VERIFY_COMPARE.verify_registered_plan_counters(
            delta,
            registered_plan=True,
            declared=True,
            reuse_bank=True,
            layer_transition=True,
            expected_legacy_ioctl_calls=2,
            label="bad-compact",
        )


@pytest.mark.parametrize(
    ("counter", "bad_value"),
    [
        (ABORT_CALLS, 0),
        (ABORT_CALLS, 2),
        (ABORT_SUCCESSES, 0),
        (ABORT_FAILURES, 1),
        (ABORT_RELEASED_SLOTS, VERIFY_COMPARE.EXPECTED_BANK_SLOTS - 1),
        (ABORT_RELEASED_BYTES, VERIFY_COMPARE.EXPECTED_BANK_BYTES - 64 * 1024),
        (ABORT_REMAINING_SLOTS, 1),
    ],
)
def test_registered_plan_abort_counter_mismatch_fails(
    counter: str,
    bad_value: int,
) -> None:
    delta = {
        PLAN_REGISTER: VERIFY_COMPARE.EXPECTED_REGISTERED_PLAN_CALLS,
        PLAN_EXECUTE: VERIFY_COMPARE.EXPECTED_REGISTERED_PLAN_EXECUTES,
        PLAN_DESTROY: VERIFY_COMPARE.EXPECTED_REGISTERED_PLAN_DESTROYS,
        LEGACY_IOCTL: 2,
        LEGACY_MODE20: 0,
        **valid_abort_delta(True),
    }
    delta[counter] = bad_value
    with pytest.raises(AssertionError):
        VERIFY_COMPARE.verify_registered_plan_counters(
            delta,
            registered_plan=True,
            declared=True,
            reuse_bank=True,
            layer_transition=True,
            expected_legacy_ioctl_calls=2,
            label="bad-abort",
        )


@pytest.mark.parametrize(
    ("init_bank", "reuse_bank", "layer_transition", "expected"),
    [
        (True, True, True, (0, 1)),
        (False, True, True, (1, 1)),
        (False, True, False, (4, 4)),
        (False, False, True, (4, 4)),
    ],
)
def test_expected_layer_mode8_ioctl_counts(
    init_bank: bool,
    reuse_bank: bool,
    layer_transition: bool,
    expected: tuple[int, int],
) -> None:
    assert (
        VERIFY_COMPARE.expected_layer_mode8_ioctl_counts(
            init_bank=init_bank,
            reuse_bank=reuse_bank,
            layer_transition=layer_transition,
        )
        == expected
    )


def test_trace_on_uses_all_mode8_fd_candidate_ioctl_calls() -> None:
    detail = valid_init_bank_record()["preallocate"]
    assert isinstance(detail, dict)
    assert (
        VERIFY_COMPARE.verify_mode8_fd_candidate_ioctl_calls(
            [detail],
            trace_enabled=True,
            expected_logical_ioctl_calls=1,
            control_ioctl_calls=2,
            declared_ioctl_calls=2,
            label="trace-on",
        )
        == 2
    )


def test_trace_off_requires_independent_mode8_control_summary() -> None:
    with pytest.raises(AssertionError, match="control summary is missing"):
        VERIFY_COMPARE.verify_mode8_fd_candidate_ioctl_calls(
            [],
            trace_enabled=False,
            expected_logical_ioctl_calls=1,
            control_ioctl_calls=None,
            declared_ioctl_calls=2,
            label="trace-off",
        )
    assert (
        VERIFY_COMPARE.verify_mode8_fd_candidate_ioctl_calls(
            [],
            trace_enabled=False,
            expected_logical_ioctl_calls=1,
            control_ioctl_calls=2,
            declared_ioctl_calls=2,
            label="trace-off",
        )
        == 2
    )


def test_mode8_control_summary_proves_dynamic_candidate_calls(
    tmp_path: Path,
) -> None:
    path = tmp_path / "slot_control_summary.json"
    path.write_text(json.dumps(valid_mode8_control_summary()), encoding="utf-8")
    assert (
        VERIFY_COMPARE.verify_mode8_control_summary(
            path,
            expected_logical_ioctl_calls=1,
            required=True,
            label="control",
        )
        == 2
    )


def test_benchmark_writes_control_summary_only_after_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "slot_control_summary.json"
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CONTROL_SUMMARY_FILE", str(path))
    pool = SimpleNamespace(
        driver_slot_preallocate_ranges=lambda *_args, **_kwargs: {
            "ok": True,
            "reason": "ok",
            "mode": VERIFY_COMPARE.MODE_VIDMEM_PREALLOC,
            "ioctl_calls": 1,
            "fd_candidate_attempts": 2,
            "fd_candidate_ioctl_calls": 2,
            "requested_count": VERIFY_COMPARE.EXPECTED_BANK_SLOTS,
            "requested_bytes": VERIFY_COMPARE.EXPECTED_BANK_BYTES,
            "success_count": VERIFY_COMPARE.EXPECTED_BANK_SLOTS,
            "accepted_bytes": VERIFY_COMPARE.EXPECTED_BANK_BYTES,
        }
    )
    target, captures = BENCHMARK.install_slot_preallocate_control_capture(pool)
    pool.driver_slot_preallocate_ranges([], device=0)
    assert target == path and len(captures) == 1 and not path.exists()
    BENCHMARK.write_slot_preallocate_control_summary(
        target,
        captures,
        measurement_start_epoch=10.0,
        measurement_end_epoch=11.0,
    )
    assert path.is_file()
    assert (
        VERIFY_COMPARE.verify_mode8_control_summary(
            path,
            expected_logical_ioctl_calls=1,
            required=True,
            label="benchmark-control",
        )
        == 2
    )


@pytest.mark.parametrize("bad_total", [1, 3])
def test_mode8_control_summary_rejects_fewer_or_more_candidate_calls(
    tmp_path: Path,
    bad_total: int,
) -> None:
    payload = valid_mode8_control_summary()
    payload["fd_candidate_ioctl_calls"] = bad_total
    path = tmp_path / "slot_control_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="totals mismatch"):
        VERIFY_COMPARE.verify_mode8_control_summary(
            path,
            expected_logical_ioctl_calls=1,
            required=True,
            label="bad-control",
        )


def test_mode8_control_summary_is_fail_closed_when_missing(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="summary is missing"):
        VERIFY_COMPARE.verify_mode8_control_summary(
            tmp_path / "missing.json",
            expected_logical_ioctl_calls=1,
            required=True,
            label="missing-control",
        )


@pytest.mark.parametrize("declared", [1, 3])
def test_trace_on_rejects_fewer_or_more_declared_candidate_calls(
    declared: int,
) -> None:
    detail = valid_init_bank_record()["preallocate"]
    assert isinstance(detail, dict)
    with pytest.raises(AssertionError, match="traced/declared"):
        VERIFY_COMPARE.verify_mode8_fd_candidate_ioctl_calls(
            [detail],
            trace_enabled=True,
            expected_logical_ioctl_calls=1,
            control_ioctl_calls=None,
            declared_ioctl_calls=declared,
            label="candidate-mismatch",
        )


@pytest.mark.parametrize(
    "missing_field",
    ["fd_candidate_attempts", "fd_candidate_ioctl_calls"],
)
def test_trace_on_rejects_missing_candidate_control_fields(
    missing_field: str,
) -> None:
    detail = copy.deepcopy(valid_init_bank_record()["preallocate"])
    assert isinstance(detail, dict)
    detail.pop(missing_field)
    with pytest.raises(AssertionError):
        VERIFY_COMPARE.verify_mode8_fd_candidate_ioctl_calls(
            [detail],
            trace_enabled=True,
            expected_logical_ioctl_calls=1,
            control_ioctl_calls=None,
            declared_ioctl_calls=None,
            label="missing-candidate-field",
        )


@pytest.mark.parametrize(
    ("registered_plan", "expected"),
    [(False, 7), (True, 2)],
)
def test_candidate_derived_legacy_ioctl_totals(
    registered_plan: bool,
    expected: int,
) -> None:
    assert (
        VERIFY_COMPARE.expected_slot_legacy_ioctl_calls(
            2,
            registered_plan=registered_plan,
        )
        == expected
    )


def test_init_bank_trace_proves_one_initialization_mode8() -> None:
    record = valid_init_bank_record()
    assert (
        VERIFY_COMPARE.verify_init_bank_trace(
            [record],
            init_bank=True,
            precomputed_lanes=True,
            label="init-bank",
        )
        is record
    )


def test_disabled_init_bank_rejects_initialization_trace() -> None:
    assert (
        VERIFY_COMPARE.verify_init_bank_trace(
            [],
            init_bank=False,
            precomputed_lanes=True,
            label="legacy",
        )
        is None
    )
    with pytest.raises(AssertionError, match="trace count"):
        VERIFY_COMPARE.verify_init_bank_trace(
            [valid_init_bank_record()],
            init_bank=False,
            precomputed_lanes=True,
            label="unexpected-init",
        )


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("layer_id",), 1),
        (("bank_rebind_slots",), 1),
        (("preallocate", "ioctl_calls"), 2),
        (("preallocate", "backing_reuse_count"), 1),
    ],
)
def test_init_bank_trace_rejects_inexact_contract(
    path: tuple[str, ...],
    bad_value: object,
) -> None:
    record = copy.deepcopy(valid_init_bank_record())
    target = record
    for key in path[:-1]:
        value = target[key]
        assert isinstance(value, dict)
        target = value
    target[path[-1]] = bad_value
    with pytest.raises(AssertionError):
        VERIFY_COMPARE.verify_init_bank_trace(
            [record],
            init_bank=True,
            precomputed_lanes=True,
            label="bad-init",
        )


def test_registered_plan_trace_proves_five_initialization_ioctls() -> None:
    record = valid_registered_plan_record()
    assert (
        VERIFY_COMPARE.verify_registered_plan_registration_trace(
            [record],
            registered_plan=True,
            label="registered-plan",
        )
        is record
    )


def test_registered_plan_trace_accepts_noninitial_module_generation() -> None:
    record = valid_registered_plan_record(generation=41)
    assert (
        VERIFY_COMPARE.verify_registered_plan_registration_trace(
            [record],
            registered_plan=True,
            label="registered-plan-generation-41",
        )
        is record
    )


@pytest.mark.parametrize("generation", [0, -1, VERIFY_COMPARE.MAX_U64 - 3])
def test_registered_plan_trace_rejects_invalid_generation(generation: int) -> None:
    with pytest.raises(AssertionError, match="owner/generation"):
        VERIFY_COMPARE.verify_registered_plan_registration_trace(
            [valid_registered_plan_record(generation=generation)],
            registered_plan=True,
            label="invalid-generation",
        )


def test_registered_plan_trace_requires_one_generation_for_all_plans() -> None:
    record = valid_registered_plan_record(generation=41)
    results = record["register_results"]
    assert isinstance(results, list)
    assert isinstance(results[-1], dict)
    results[-1]["generation"] = 42
    with pytest.raises(AssertionError, match="identity mismatch"):
        VERIFY_COMPARE.verify_registered_plan_registration_trace(
            [record],
            registered_plan=True,
            label="mixed-registration-generation",
        )


def test_disabled_registered_plan_rejects_registration_trace() -> None:
    assert (
        VERIFY_COMPARE.verify_registered_plan_registration_trace(
            [],
            registered_plan=False,
            label="legacy",
        )
        is None
    )
    with pytest.raises(AssertionError, match="trace count"):
        VERIFY_COMPARE.verify_registered_plan_registration_trace(
            [valid_registered_plan_record()],
            registered_plan=False,
            label="unexpected-plan",
        )


@pytest.mark.parametrize("layer_id", range(VERIFY_COMPARE.EXPECTED_LAYERS))
def test_registered_plan_execute_proves_compact_generation_chain(
    layer_id: int,
) -> None:
    generation_start = 41
    record = valid_registered_execute_record(layer_id, generation_start)
    assert (
        VERIFY_COMPARE.verify_registered_plan_execute_trace(
            record,
            layer_id=layer_id,
            generation_start=generation_start,
            label="compact-execute",
        )
        is record["restore"]
    )


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("plan_generation",), 7),
        (("bank_rebind_reason",), "legacy"),
        (("restore", "plan_execute"), False),
        (("restore", "new_generation"), 9),
        (("restore", "batches"), [{}]),
        (("restore", "poisoned"), True),
    ],
)
def test_registered_plan_execute_rejects_inexact_contract(
    path: tuple[str, ...],
    bad_value: object,
) -> None:
    record = copy.deepcopy(valid_registered_execute_record(1))
    target = record
    for key in path[:-1]:
        value = target[key]
        assert isinstance(value, dict)
        target = value
    target[path[-1]] = bad_value
    with pytest.raises(AssertionError):
        VERIFY_COMPARE.verify_registered_plan_execute_trace(
            record,
            layer_id=1,
            generation_start=1,
            label="bad-compact-execute",
        )


def test_registered_plan_execute_rejects_wrong_generation_start() -> None:
    record = valid_registered_execute_record(2, generation_start=41)
    with pytest.raises(AssertionError, match="identity/generation"):
        VERIFY_COMPARE.verify_registered_plan_execute_trace(
            record,
            layer_id=2,
            generation_start=42,
            label="wrong-generation-start",
        )


def test_plan_generation_contract_keeps_legacy_artifacts_compatible() -> None:
    assert VERIFY_COMPARE.verify_plan_generation_contract(None) is None
    assert (
        VERIFY_COMPARE.verify_plan_generation_contract(
            VERIFY_COMPARE.PLAN_GENERATION_CONTRACT
        )
        == VERIFY_COMPARE.PLAN_GENERATION_CONTRACT
    )
    with pytest.raises(AssertionError, match="generation contract"):
        VERIFY_COMPARE.verify_plan_generation_contract("fixed-at-one")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("3", 3.0), ("0.0", 0.0), ("2.5", 2.5)],
)
def test_parse_pinned_anchor_threshold(
    value: str | None,
    expected: float | None,
) -> None:
    assert VERIFY_COMPARE.parse_pinned_anchor_threshold(value) == expected


@pytest.mark.parametrize("value", ["", "nan", "inf", "-0.1", "three"])
def test_parse_pinned_anchor_threshold_rejects_invalid_value(value: str) -> None:
    with pytest.raises(AssertionError, match="threshold"):
        VERIFY_COMPARE.parse_pinned_anchor_threshold(value)


def test_pinned_anchor_comparison_passes_under_preregistered_three_percent() -> None:
    result = VERIFY_COMPARE.compare_pinned_anchor_summaries(
        valid_anchor_summary(
            registered_plan=False,
            copy_ms=100.0,
            slot_ms=600.0,
        ),
        valid_anchor_summary(
            registered_plan=True,
            copy_ms=102.9,
            slot_ms=400.0,
        ),
    )
    assert result["status"] == "pass"
    assert result["pinned_anchor_pass"] is True
    assert result["pinned_anchor_max_drift_pct"] == 3.0
    assert result["pinned_anchor_drift_pct"] == pytest.approx(2.9)
    assert result["registered_plan_slot_speedup"] == pytest.approx(1.5)


def test_pinned_anchor_comparison_reports_fail_above_threshold() -> None:
    result = VERIFY_COMPARE.compare_pinned_anchor_summaries(
        valid_anchor_summary(
            registered_plan=False,
            copy_ms=100.0,
            slot_ms=600.0,
        ),
        valid_anchor_summary(
            registered_plan=True,
            copy_ms=103.1,
            slot_ms=400.0,
        ),
    )
    assert result["status"] == "fail"
    assert result["pinned_anchor_pass"] is False
    assert result["pinned_anchor_drift_pct"] == pytest.approx(3.1)
    assert result["failure_reason"] is not None


@pytest.mark.parametrize("missing_side", ["baseline", "optimized"])
def test_pinned_anchor_comparison_requires_preregistered_thresholds(
    missing_side: str,
) -> None:
    baseline = valid_anchor_summary(
        registered_plan=False,
        copy_ms=100.0,
        slot_ms=600.0,
        threshold_pct=None if missing_side == "baseline" else 3.0,
    )
    optimized = valid_anchor_summary(
        registered_plan=True,
        copy_ms=101.0,
        slot_ms=400.0,
        threshold_pct=None if missing_side == "optimized" else 3.0,
    )
    with pytest.raises(AssertionError, match="pre-registered"):
        VERIFY_COMPARE.compare_pinned_anchor_summaries(baseline, optimized)


def test_pinned_anchor_comparison_requires_matching_thresholds() -> None:
    with pytest.raises(AssertionError, match="thresholds differ"):
        VERIFY_COMPARE.compare_pinned_anchor_summaries(
            valid_anchor_summary(
                registered_plan=False,
                copy_ms=100.0,
                slot_ms=600.0,
                threshold_pct=3.0,
            ),
            valid_anchor_summary(
                registered_plan=True,
                copy_ms=101.0,
                slot_ms=400.0,
                threshold_pct=5.0,
            ),
        )


def test_pinned_anchor_cli_writes_fail_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_path = tmp_path / "baseline.json"
    optimized_path = tmp_path / "optimized.json"
    output_path = tmp_path / "anchor.json"
    baseline_path.write_text(
        json.dumps(
            valid_anchor_summary(
                registered_plan=False,
                copy_ms=100.0,
                slot_ms=600.0,
            )
        ),
        encoding="utf-8",
    )
    optimized_path.write_text(
        json.dumps(
            valid_anchor_summary(
                registered_plan=True,
                copy_ms=104.0,
                slot_ms=400.0,
            )
        ),
        encoding="utf-8",
    )
    assert (
        VERIFY_COMPARE.main(
            [
                "--compare-pinned-anchors",
                str(baseline_path),
                str(optimized_path),
                "--anchor-output",
                str(output_path),
            ]
        )
        == 1
    )
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "fail"
    assert json.loads(capsys.readouterr().out)["pinned_anchor_pass"] is False


def test_stage1_three_arm_comparison_quantifies_both_gains() -> None:
    result = VERIFY_COMPARE.compare_stage1_summaries(
        valid_anchor_summary(
            registered_plan=False,
            init_bank=False,
            order="SCCS",
            copy_ms=100.0,
            slot_ms=600.0,
        ),
        valid_anchor_summary(
            registered_plan=False,
            init_bank=True,
            order="CSSC",
            copy_ms=101.0,
            slot_ms=500.0,
        ),
        valid_anchor_summary(
            registered_plan=True,
            init_bank=True,
            order="SCCS",
            copy_ms=102.0,
            slot_ms=400.0,
        ),
    )
    assert result["status"] == "pass"
    assert result["pinned_anchor_spread_pct"] == pytest.approx(2.0)
    assert result["init_bank_speedup"] == pytest.approx(1.2)
    assert result["init_bank_latency_reduction_ms"] == pytest.approx(100.0)
    assert result["registered_plan_incremental_speedup"] == pytest.approx(1.25)
    assert result["registered_plan_incremental_latency_reduction_ms"] == pytest.approx(
        100.0
    )
    assert result["combined_speedup"] == pytest.approx(1.5)


def test_stage1_three_arm_comparison_fails_anchor_spread() -> None:
    result = VERIFY_COMPARE.compare_stage1_summaries(
        valid_anchor_summary(
            registered_plan=False,
            init_bank=False,
            order="SCCS",
            copy_ms=100.0,
            slot_ms=600.0,
        ),
        valid_anchor_summary(
            registered_plan=False,
            init_bank=True,
            order="CSSC",
            copy_ms=101.0,
            slot_ms=500.0,
        ),
        valid_anchor_summary(
            registered_plan=True,
            init_bank=True,
            order="SCCS",
            copy_ms=104.0,
            slot_ms=400.0,
        ),
    )
    assert result["status"] == "fail"
    assert result["pinned_anchor_pass"] is False
    assert result["pinned_anchor_spread_pct"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("bad_arm", "field", "value"),
    [
        ("baseline", "slot_init_bank", True),
        ("init", "slot_registered_plan", True),
        ("init", "order", "SCCS"),
        ("plan", "order", "CSSC"),
        ("baseline", "slot_shared_push", False),
    ],
)
def test_stage1_three_arm_comparison_rejects_wrong_design(
    bad_arm: str,
    field: str,
    value: object,
) -> None:
    summaries = {
        "baseline": valid_anchor_summary(
            registered_plan=False,
            init_bank=False,
            order="SCCS",
            copy_ms=100.0,
            slot_ms=600.0,
        ),
        "init": valid_anchor_summary(
            registered_plan=False,
            init_bank=True,
            order="CSSC",
            copy_ms=101.0,
            slot_ms=500.0,
        ),
        "plan": valid_anchor_summary(
            registered_plan=True,
            init_bank=True,
            order="SCCS",
            copy_ms=102.0,
            slot_ms=400.0,
        ),
    }
    summaries[bad_arm][field] = value
    with pytest.raises(AssertionError):
        VERIFY_COMPARE.compare_stage1_summaries(
            summaries["baseline"],
            summaries["init"],
            summaries["plan"],
        )


def test_stage1_three_arm_cli_writes_pass_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inputs = [
        valid_anchor_summary(
            registered_plan=False,
            init_bank=False,
            order="SCCS",
            copy_ms=100.0,
            slot_ms=600.0,
        ),
        valid_anchor_summary(
            registered_plan=False,
            init_bank=True,
            order="CSSC",
            copy_ms=101.0,
            slot_ms=500.0,
        ),
        valid_anchor_summary(
            registered_plan=True,
            init_bank=True,
            order="SCCS",
            copy_ms=102.0,
            slot_ms=400.0,
        ),
    ]
    paths: list[Path] = []
    for index, payload in enumerate(inputs):
        path = tmp_path / f"arm-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)
    output_path = tmp_path / "stage1.json"
    assert (
        VERIFY_COMPARE.main(
            [
                "--compare-stage1-arms",
                *(str(path) for path in paths),
                "--anchor-output",
                str(output_path),
            ]
        )
        == 0
    )
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "pass"
    assert json.loads(capsys.readouterr().out)["init_bank_speedup"] == pytest.approx(
        1.2
    )


def test_stage1_runbook_preregisters_three_arms_and_recovery_traps() -> None:
    text = RUNBOOK_PATH.read_text(encoding="utf-8")
    assert "20260816T_phase1_baseline_trace_off_SCCS" in text
    assert "20260816T_phase1_init_only_trace_off_CSSC" in text
    assert "20260816T_phase1_init_plan_trace_off_SCCS" in text
    assert text.count("PINNED_ANCHOR_MAX_DRIFT_PCT=3.0") == 3
    assert text.count("SLOT_EXPECTED_MODE8_FD_CANDIDATE_IOCTL_CALLS=2") == 4
    assert text.count("legacy ioctl=7") == 2
    assert text.count("legacy ioctl=2") == 2
    assert "calls/successes/failures=1/1/0" in text
    assert "sudo modprobe nvidia\nsudo insmod" in text
    assert "--compare-stage1-arms" in text
    assert "stage1_restore_environment()" in text
    assert "trap 'stage1_err_trap" in text
    assert "trap stage1_exit_trap EXIT" in text
    assert "trap 'exit 130' INT" in text
    assert "trap 'exit 143' TERM" in text


def test_trace_off_abba_is_a_valid_performance_proof_mode() -> None:
    assert (
        VERIFY_COMPARE.verify_proof_mode(
            trace_enabled=False,
            require_combined=True,
            require_abba=True,
            order="CSSC",
        )
        == "performance_trace_off_abba"
    )
    assert (
        VERIFY_COMPARE.verify_proof_mode(
            trace_enabled=True,
            require_combined=True,
            require_abba=False,
            order="CS",
        )
        == "correctness_trace_on"
    )


@pytest.mark.parametrize(
    ("require_combined", "order"),
    [(False, "CSSC"), (True, "CS")],
)
def test_abba_still_requires_combined_mechanism_and_exact_order(
    require_combined: bool,
    order: str,
) -> None:
    with pytest.raises(AssertionError):
        VERIFY_COMPARE.verify_proof_mode(
            trace_enabled=False,
            require_combined=require_combined,
            require_abba=True,
            order=order,
        )
