#!/usr/bin/env python3
"""CPU-only contract tests for the deterministic SIGTERM lifecycle probe."""

from __future__ import annotations

import importlib.util
import signal
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("run_lifecycle.py")
MODULES_BEFORE = set(sys.modules)
SPEC = importlib.util.spec_from_file_location("slot_lifecycle_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
lifecycle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lifecycle)
MODULES_ADDED = set(sys.modules) - MODULES_BEFORE


def state(generation: int, *, ready_slots: int = lifecycle.BANK_SLOTS) -> dict[str, object]:
    return {
        "cache": {
            "allocated_bytes": lifecycle.BANK_BYTES,
            "allocated_slots": lifecycle.BANK_SLOTS,
            "ready_slots": ready_slots,
            "resident_groups": 128,
            "cache_bytes_limit": lifecycle.BANK_BYTES,
            "cache_slots_limit": lifecycle.BANK_SLOTS,
        },
        "policy_plan_count": 4,
        "policy_generation": generation,
        "policy_active_layer": None,
        "driver_allocated_slots": lifecycle.BANK_SLOTS,
        "driver_plan_count": 5,
        "driver_generation": generation,
        "session_open": True,
        "poison": None,
    }


def mode8_capture(candidate_calls: int = 2) -> list[dict[str, object]]:
    return [{
        "ok": True,
        "reason": "ok",
        "mode": 8,
        "ioctl_calls": 1,
        "fd_candidate_attempts": candidate_calls,
        "fd_candidate_ioctl_calls": candidate_calls,
        "requested_count": lifecycle.BANK_SLOTS,
        "success_count": lifecycle.BANK_SLOTS,
        "requested_bytes": lifecycle.BANK_BYTES,
        "accepted_bytes": lifecycle.BANK_BYTES,
    }]


def route_record(*, slots: int, bytes_: int, failed: int, total: int) -> dict[str, object]:
    return {
        "timestamp_ns": 1_700_000_000_100_000_000 + total,
        "va_space": f"0x{total:x}",
        "reclaimed": slots,
        "bytes": bytes_,
        "failed": failed,
        "total_reclaimed": total,
        "total_bytes": total * 4096,
        "total_failed": failed,
        "owner_mismatches": 0,
    }


def valid_evidence() -> dict[str, object]:
    nonce = "0123456789abcdef"
    pid = 321
    initial = 100
    final = initial + lifecycle.LAYERS
    expected = lifecycle.expected_sigterm_delta(1, 2)
    written_ns = 1_700_000_000_000_000_000
    return {
        "ready": {
            "nonce": nonce,
            "pid": pid,
            "generation": initial,
            "sigterm_handler": "SIG_DFL",
            "state": state(initial, ready_slots=0),
            "mode8_captures": mode8_capture(),
        },
        "round": {
            "nonce": nonce,
            "pid": pid,
            "written_epoch_ns": written_ns,
            "completed_rounds": 1,
            "generation": final,
            "prompt_token_ids_exact": True,
            "output_token_ids": [4710],
            "state": state(final),
        },
        "child": {
            "nonce": nonce,
            "pid": pid,
            "status": "waiting_for_sigterm",
        },
        "returncode": -signal.SIGTERM,
        "exit_state_exists": False,
        "escalated_to_sigkill": False,
        "signal_sent_ns": written_ns + 1,
        "process_exit_ns": written_ns + 2,
        "stats_delta": expected,
        "route_records": [
            route_record(
                slots=lifecycle.BANK_SLOTS // 2,
                bytes_=lifecycle.BANK_BYTES // 2,
                failed=0,
                total=lifecycle.BANK_SLOTS // 2,
            ),
            route_record(
                slots=lifecycle.BANK_SLOTS // 2,
                bytes_=lifecycle.BANK_BYTES // 2,
                failed=0,
                total=lifecycle.BANK_SLOTS,
            ),
        ],
        "gpu_after": {"used_mib": 0, "compute_pids": []},
    }


def test_import_is_gpu_free() -> None:
    assert "torch" not in MODULES_ADDED
    assert "vllm" not in MODULES_ADDED
    assert not any(name.startswith("vllm.") for name in MODULES_ADDED)


def test_latency_stats_parser_and_delta() -> None:
    before_text = """\
slot_direct_profile:
  total                    calls=4 ns=10 avg_ns=2 pages=8 bytes=32
slot_direct_ioctl_calls         4
stage_a_last_fault_addr         0x10
"""
    after_text = """\
slot_direct_profile:
  total                    calls=8 ns=30 avg_ns=3 pages=16 bytes=64
slot_direct_ioctl_calls         8
stage_a_last_fault_addr         0x20
"""
    before = lifecycle.parse_latency_stats(before_text)
    after = lifecycle.parse_latency_stats(after_text)
    assert before["slot_direct_profile.total.calls"] == 4
    assert before["stage_a_last_fault_addr"] == 16
    delta = lifecycle.stats_delta(
        before, after, required=frozenset({"slot_direct_ioctl_calls"})
    )
    assert delta["slot_direct_profile.total.ns"] == 20
    assert delta["slot_direct_ioctl_calls"] == 4


@pytest.mark.parametrize(
    "text, message",
    [
        ("  total calls=1 ns=1 avg_ns=1 pages=1 bytes=1\n", "before section"),
        ("slot_direct_ioctl_calls 1\nslot_direct_ioctl_calls 2\n", "duplicate"),
        ("slot_direct_ioctl_calls -1\n", "unrecognized"),
    ],
)
def test_latency_stats_parser_rejects_bad_input(text: str, message: str) -> None:
    with pytest.raises(lifecycle.LifecycleError, match=message):
        lifecycle.parse_latency_stats(text)


def test_stats_delta_rejects_reset_and_missing_required() -> None:
    with pytest.raises(lifecycle.LifecycleError, match="moved backwards"):
        lifecycle.stats_delta({"a": 2}, {"a": 1})
    with pytest.raises(lifecycle.LifecycleError, match="before snapshot lacks"):
        lifecycle.stats_delta({}, {"a": 1}, required=frozenset({"a"}))


def test_exact_formula_one_and_eight_requests() -> None:
    one = lifecycle.expected_sigterm_delta(1, 2)
    assert one["slot_plan_execute_calls"] == 4
    assert one["slot_direct_requested_ranges"] == 4608
    assert one["slot_direct_requested_bytes"] == 2_415_919_104
    assert one["slot_direct_blocks"] == 1152
    assert one["slot_direct_pte_write_entries"] == 36_864
    assert one["slot_direct_transition_fast_index_calls"] == 3
    assert one["slot_direct_transition_clear_ranges"] == 3456
    assert one["slot_direct_transition_clear_unique_blocks"] == 864
    assert one["slot_direct_transition_direct_cutover_pages"] == 442_368
    assert one["slot_plan_destroy_calls"] == 0
    assert one["slot_session_abort_calls"] == 0
    assert one["slot_legacy_ioctl_calls"] == 2

    eight = lifecycle.expected_sigterm_delta(8, 3)
    assert eight["slot_plan_execute_calls"] == 32
    assert eight["slot_direct_requested_ranges"] == 36_864
    assert eight["slot_direct_requested_bytes"] == 19_327_352_832
    assert eight["slot_direct_transition_fast_index_calls"] == 31
    assert eight["slot_legacy_ioctl_calls"] == 3


def test_formula_rejects_bool_zero_and_negative() -> None:
    for requests, candidates in ((True, 2), (0, 2), (1, 0), (1, -1)):
        with pytest.raises(lifecycle.LifecycleError):
            lifecycle.expected_sigterm_delta(requests, candidates)


def test_mode8_uses_candidate_not_logical_ioctl_count() -> None:
    assert lifecycle.verify_mode8_captures(mode8_capture(7)) == 7
    bad = mode8_capture()
    bad[0]["ioctl_calls"] = 2
    with pytest.raises(lifecycle.LifecycleError, match="logical ioctl"):
        lifecycle.verify_mode8_captures(bad)


def test_parse_and_aggregate_multiple_close_records() -> None:
    text = """\
[1700000000.000000001] unrelated value=1
[1700000000.100000000] moe_slot_pool_close_reclaim va_space=0x1 reclaimed=500 bytes=262144000 failed=0 total_reclaimed=500 total_bytes=262144000 total_failed=0 owner_mismatches=0 extra=ok
[1700000000.200000000] moe_slot_pool_close_reclaim va_space=(____ptrval____) reclaimed=652 bytes=341835776 failed=0 total_reclaimed=1152 total_bytes=603979776 total_failed=0 owner_mismatches=0
"""
    records = lifecycle.parse_close_reclaim_records(
        text,
        start_ns=1_700_000_000_050_000_000,
        end_ns=1_700_000_000_250_000_000,
    )
    aggregate = lifecycle.verify_close_reclaim(records)
    assert aggregate == {
        "records": 2,
        "nonzero_records": 2,
        "reclaimed": lifecycle.BANK_SLOTS,
        "bytes": lifecycle.BANK_BYTES,
        "failed": 0,
    }


def test_close_aggregate_uses_local_not_cumulative_values() -> None:
    records = [
        route_record(slots=512, bytes_=1, failed=0, total=10_000),
        route_record(slots=640, bytes_=lifecycle.BANK_BYTES - 1, failed=0, total=10_640),
    ]
    aggregate = lifecycle.aggregate_close_reclaim(records)
    assert aggregate["reclaimed"] == lifecycle.BANK_SLOTS
    assert aggregate["bytes"] == lifecycle.BANK_BYTES


def test_close_reclaim_rejects_partial_or_failed_cleanup() -> None:
    partial = [route_record(slots=1151, bytes_=lifecycle.BANK_BYTES, failed=0, total=1151)]
    with pytest.raises(lifecycle.LifecycleError, match="slot mismatch"):
        lifecycle.verify_close_reclaim(partial)
    failed = [route_record(slots=1152, bytes_=lifecycle.BANK_BYTES, failed=1, total=1152)]
    with pytest.raises(lifecycle.LifecycleError, match="failures"):
        lifecycle.verify_close_reclaim(failed)


def test_wait_close_reclaim_handles_deferred_file_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = tmp_path / "route.log"
    route.write_text("", encoding="utf-8")
    complete = (
        "[1700000000.100000000] moe_slot_pool_close_reclaim "
        "va_space=0x1 reclaimed=1152 bytes=603979776 failed=0 "
        "total_reclaimed=1152 total_bytes=603979776 total_failed=0 "
        "owner_mismatches=0\n"
    )
    monkeypatch.setattr(lifecycle.time, "monotonic", lambda: 0.0)
    monkeypatch.setattr(
        lifecycle.time,
        "sleep",
        lambda _seconds: route.write_text(complete, encoding="utf-8"),
    )
    text, records = lifecycle._wait_close_reclaim_log(
        route,
        start_ns=1_700_000_000_000_000_000,
        timeout=1.0,
    )
    assert text == complete
    assert lifecycle.verify_close_reclaim(records)["reclaimed"] == 1152


def test_complete_sigterm_contract_passes() -> None:
    result = lifecycle.verify_sigterm_evidence(valid_evidence())
    assert result["initial_generation"] == 100
    assert result["final_generation"] == 104
    assert result["candidate_ioctl_calls"] == 2


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value.update(returncode=0), "did not exit by SIGTERM"),
        (lambda value: value.update(exit_state_exists=True), "unexpectedly ran atexit"),
        (lambda value: value.update(escalated_to_sigkill=True), "escalated"),
        (lambda value: value["stats_delta"].update(slot_plan_destroy_calls=5), "slot_plan_destroy_calls"),
        (lambda value: value["stats_delta"].update(slot_session_abort_calls=1), "slot_session_abort_calls"),
        (lambda value: value["round"].update(output_token_ids=[1]), "output token"),
        (lambda value: value["ready"].update(sigterm_handler="custom"), "installed a SIGTERM"),
    ],
)
def test_sigterm_contract_rejects_false_evidence(mutation, message: str) -> None:
    value = valid_evidence()
    mutation(value)
    with pytest.raises(lifecycle.LifecycleError, match=message):
        lifecycle.verify_sigterm_evidence(value)


def test_generation_recovery_must_be_strictly_newer() -> None:
    assert lifecycle.verify_recovery_generation(
        {"status": "pass", "first_generation": 105}, 104
    ) == 105
    for generation in (104, 103, 0):
        with pytest.raises(lifecycle.LifecycleError, match="generation"):
            lifecycle.verify_recovery_generation(
                {"status": "pass", "first_generation": generation}, 104
            )


def test_child_command_and_environment_are_static_and_gpu_free(tmp_path: Path) -> None:
    environment = lifecycle.build_slot_environment(
        plan=tmp_path / "plan.json",
        case_dir=tmp_path / "case",
        gpu_index=0,
        cpu_threads=14,
    )
    assert environment["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert environment["VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN"] == "1"
    assert environment["VLLM_UVM_SLOT_DIRECT_INIT_BANK"] == "1"
    assert environment["VLLM_UVM_SLOT_DIRECT_SHARED_PUSH"] == "1"
    assert environment["VLLM_UVM_SLOT_DIRECT_64K_PTE"] == "1"
    assert environment["VLLM_UVM_SLOT_DIRECT_CONTROL_SUMMARY_FILE"]
    command = lifecycle.build_child_command(
        python=Path("/venv/python"), script=MODULE_PATH, cpuset="0-13",
        work_dir=tmp_path / "case", model=tmp_path / "model",
        fixture=tmp_path / "fixture.json", nonce="abc",
    )
    assert command[:3] == ["taskset", "-c", "0-13"]
    assert command[-2:] == ["--nonce", "abc"]
    assert "torch" not in (set(sys.modules) - MODULES_BEFORE)


def test_live_gate_fails_before_any_gpu_action() -> None:
    with pytest.raises(lifecycle.LifecycleError, match="ALLOW_GPU_RUN"):
        lifecycle.require_live_gate({})


def test_extract_json_returns_outer_recovery_payload() -> None:
    text = 'log line\n{"status":"pass","first_generation":9,"finish":{"ok":true}}\n'
    assert lifecycle._extract_json(text) == {
        "status": "pass",
        "first_generation": 9,
        "finish": {"ok": True},
    }
