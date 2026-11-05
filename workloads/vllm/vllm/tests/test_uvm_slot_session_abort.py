# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
import threading

import pytest

from vllm.device_allocator import uvm as uvm_module


DEVICE = 0
FD = 123
UUID_BYTES = bytes(range(16))


@pytest.fixture(autouse=True)
def _isolated_abort_session(monkeypatch: pytest.MonkeyPatch):
    for mapping in (
        uvm_module._uvm_slot_direct_sessions,
        uvm_module._uvm_slot_direct_owners,
        uvm_module._uvm_slot_direct_plans,
        uvm_module._uvm_slot_direct_plan_generations,
        uvm_module._uvm_slot_direct_active_plan_ids,
        uvm_module._uvm_slot_direct_session_pids,
        uvm_module._uvm_slot_direct_poisoned,
    ):
        mapping.clear()
    monkeypatch.setattr(
        uvm_module,
        "_run_uvm_slot_direct_policy_cleanup",
        lambda _device: None,
    )
    yield
    for mapping in (
        uvm_module._uvm_slot_direct_sessions,
        uvm_module._uvm_slot_direct_owners,
        uvm_module._uvm_slot_direct_plans,
        uvm_module._uvm_slot_direct_plan_generations,
        uvm_module._uvm_slot_direct_active_plan_ids,
        uvm_module._uvm_slot_direct_session_pids,
        uvm_module._uvm_slot_direct_poisoned,
    ):
        mapping.clear()


def _install_session(*, pid: int | None = None) -> None:
    uvm_module._uvm_slot_direct_sessions[DEVICE] = (FD, UUID_BYTES)
    uvm_module._uvm_slot_direct_session_pids[DEVICE] = (
        os.getpid() if pid is None else int(pid)
    )
    uvm_module._uvm_slot_direct_owners[DEVICE] = {(3, 5): 7}
    uvm_module._uvm_slot_direct_plans[DEVICE] = {
        11: {"layer_id": 3, "mode": 20, "entries": ()}
    }
    uvm_module._uvm_slot_direct_plan_generations[DEVICE] = 41


def _complete_abort(
    params: object,
    *,
    had_session: int = 1,
    generation: int = 41,
    destroyed_plans: int = 1,
    released_slots: int = 1,
    released_bytes: int = 4096,
) -> None:
    params.generation_before = generation
    params.released_bytes = released_bytes
    params.remaining_bytes = 0
    params.destroyed_plan_count = destroyed_plans
    params.released_slot_count = released_slots
    params.remaining_slot_count = 0
    params.had_registered_session = had_session
    params.session_cleared = 1
    params.poisoned = 0
    params.rmStatus = 0


def test_abort_ioctl_abi_matches_kernel_header() -> None:
    params = uvm_module._MoeSlotPlanAbortSessionParams
    assert uvm_module._UVM_MOE_SLOT_PLAN_ABORT_SESSION_IOCTL == 95
    assert ctypes.sizeof(params) == 64
    assert params.flags.offset == 0
    assert params.reserved.offset == 4
    assert params.generation_before.offset == 8
    assert params.released_bytes.offset == 16
    assert params.remaining_bytes.offset == 24
    assert params.destroyed_plan_count.offset == 32
    assert params.released_slot_count.offset == 36
    assert params.remaining_slot_count.offset == 40
    assert params.had_registered_session.offset == 44
    assert params.session_cleared.offset == 48
    assert params.poisoned.offset == 52
    assert params.rmStatus.offset == 56


def test_abort_empty_session_is_idempotent_without_ioctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        uvm_module,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected ioctl")),
    )

    first = uvm_module.driver_slot_abort_session(device=DEVICE)
    second = uvm_module.driver_slot_abort_session(device=DEVICE)

    assert first["ok"] is second["ok"] is True
    assert first["reason"] == second["reason"] == "already_aborted"
    assert first["session_cleared"] is True
    assert first["ioctl_calls"] == second["ioctl_calls"] == 0


def test_abort_success_uses_driver_counts_and_only_then_clears_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_session()
    calls: list[tuple[int, int]] = []

    def ioctl(fd: int, request: int, params: object) -> int:
        calls.append((fd, request))
        # The driver knows about more resources than userspace. These counts
        # must be reported, not compared for equality with local dictionaries.
        _complete_abort(
            params,
            destroyed_plans=3,
            released_slots=9,
            released_bytes=9 * 4096,
        )
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module.driver_slot_abort_session(device=DEVICE)

    assert result["ok"] is True
    assert result["destroyed_plan_count"] == 3
    assert result["released_slot_count"] == 9
    assert result["released_bytes"] == 9 * 4096
    assert calls == [(FD, 95)]
    assert DEVICE in uvm_module._uvm_slot_direct_sessions
    assert DEVICE in uvm_module._uvm_slot_direct_session_pids
    assert DEVICE not in uvm_module._uvm_slot_direct_owners
    assert DEVICE not in uvm_module._uvm_slot_direct_plans
    assert DEVICE not in uvm_module._uvm_slot_direct_plan_generations
    assert DEVICE not in uvm_module._uvm_slot_direct_poisoned


@pytest.mark.parametrize(
    "mutate",
    [
        lambda params: setattr(params, "session_cleared", 0),
        lambda params: setattr(params, "remaining_slot_count", 1),
        lambda params: setattr(params, "remaining_bytes", 4096),
        lambda params: setattr(params, "had_registered_session", 2),
        lambda params: setattr(params, "released_slot_count", 0),
        lambda params: (
            setattr(params, "had_registered_session", 0),
            setattr(params, "generation_before", 1),
        ),
    ],
)
def test_abort_rejects_incomplete_or_malformed_contract_without_forgetting_state(
    monkeypatch: pytest.MonkeyPatch,
    mutate,
) -> None:
    _install_session()

    def ioctl(_fd: int, _request: int, params: object) -> int:
        _complete_abort(params)
        mutate(params)
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module.driver_slot_abort_session(device=DEVICE)

    assert result["ok"] is False
    assert result["session_poisoned"] is True
    assert DEVICE in uvm_module._uvm_slot_direct_sessions
    assert DEVICE in uvm_module._uvm_slot_direct_owners
    assert DEVICE in uvm_module._uvm_slot_direct_plans
    assert DEVICE in uvm_module._uvm_slot_direct_plan_generations
    assert DEVICE in uvm_module._uvm_slot_direct_poisoned


def test_abort_partial_failure_is_poisoned_and_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_session()
    attempts = 0

    def ioctl(_fd: int, request: int, params: object) -> int:
        nonlocal attempts
        assert request == 95
        attempts += 1
        if attempts == 1:
            params.generation_before = 41
            # Kernel invariant residues intentionally report a retry count
            # even when no trustworthy remaining-byte total is available.
            params.remaining_bytes = 0
            params.remaining_slot_count = 1
            params.had_registered_session = 1
            params.session_cleared = 0
            params.poisoned = 1
            return 0
        _complete_abort(params)
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    first = uvm_module.driver_slot_abort_session(device=DEVICE)
    second = uvm_module.driver_slot_abort_session(device=DEVICE)

    assert first["ok"] is False
    assert first["contract_ok"] is True
    assert first["reason"] == "slot_session_abort_incomplete"
    assert second["ok"] is True
    assert attempts == 2
    assert DEVICE not in uvm_module._uvm_slot_direct_poisoned


def test_close_unknown_generation_aborts_instead_of_treating_dup_close_as_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_session()
    uvm_module._uvm_slot_direct_plan_generations.pop(DEVICE)
    calls: list[int] = []
    closed: list[int] = []

    def ioctl(_fd: int, request: int, params: object) -> int:
        calls.append(request)
        assert request == 95
        _complete_abort(params, generation=41)
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    monkeypatch.setattr(uvm_module.os, "close", lambda fd: closed.append(fd))
    result = uvm_module.close_uvm_slot_direct_session(device=DEVICE)

    assert result["ok"] is True
    assert result["abort"]["ok"] is True
    assert result["cleanup_trigger"] == "untrusted_plan_generation"
    assert calls == [95]
    assert closed == [FD]
    assert DEVICE not in uvm_module._uvm_slot_direct_sessions


def test_close_abort_failure_keeps_fd_state_and_poison_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_session()
    uvm_module._uvm_slot_direct_plan_generations.pop(DEVICE)
    closed: list[int] = []
    attempts = 0

    def ioctl(_fd: int, request: int, params: object) -> int:
        nonlocal attempts
        assert request == 95
        attempts += 1
        if attempts == 1:
            params.generation_before = 41
            params.remaining_bytes = 4096
            params.remaining_slot_count = 1
            params.had_registered_session = 1
            params.session_cleared = 0
            params.poisoned = 1
        else:
            _complete_abort(params)
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    monkeypatch.setattr(uvm_module.os, "close", lambda fd: closed.append(fd))

    first = uvm_module.close_uvm_slot_direct_session(device=DEVICE)
    assert first["ok"] is False
    assert first["reason"] == "slot_session_abort_failed"
    assert closed == []
    assert DEVICE in uvm_module._uvm_slot_direct_sessions
    assert DEVICE in uvm_module._uvm_slot_direct_plans
    assert DEVICE in uvm_module._uvm_slot_direct_poisoned

    second = uvm_module.close_uvm_slot_direct_session(device=DEVICE)
    assert second["ok"] is True
    assert closed == [FD]
    assert attempts == 2


def test_registered_plan_close_holds_one_lock_across_destroy_abort_and_fd_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_session()
    calls: list[int] = []
    lock_was_held: list[bool] = []

    def observe_lock() -> None:
        acquired: list[bool] = []

        def contender() -> None:
            got = uvm_module._uvm_slot_direct_lock.acquire(timeout=0.02)
            acquired.append(got)
            if got:
                uvm_module._uvm_slot_direct_lock.release()

        thread = threading.Thread(target=contender)
        thread.start()
        thread.join()
        lock_was_held.append(acquired == [False])

    def ioctl(_fd: int, request: int, params: object) -> int:
        calls.append(request)
        observe_lock()
        if request == 94:
            params.current_generation = 41
            params.destroyed = 1
        else:
            assert request == 95
            _complete_abort(params, destroyed_plans=0)
        return 0

    closed: list[int] = []

    def close_fd(fd: int) -> None:
        observe_lock()
        closed.append(fd)

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    monkeypatch.setattr(uvm_module.os, "close", close_fd)
    result = uvm_module.close_uvm_slot_direct_session(device=DEVICE)

    assert result["ok"] is True
    assert [item["plan_id"] for item in result["plan_destroy"]] == [11]
    assert calls == [94, 95]
    assert closed == [FD]
    assert lock_was_held == [True, True, True]


def test_registered_plan_close_aborts_when_resident_recipe_rejects_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_session()
    calls: list[int] = []

    def ioctl(_fd: int, request: int, params: object) -> int:
        calls.append(request)
        if request == 94:
            # The kernel retains resident/failed-target plans so cmd95 can
            # still sweep their target PTEs during generation-free teardown.
            params.current_generation = 41
            params.destroyed = 0
            params.rmStatus = 5
            return -1
        assert request == 95
        _complete_abort(params, destroyed_plans=1)
        return 0

    closed: list[int] = []
    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    monkeypatch.setattr(uvm_module.os, "close", lambda fd: closed.append(fd))

    result = uvm_module.close_uvm_slot_direct_session(device=DEVICE)

    assert result["ok"] is True
    assert result["cleanup_trigger"] == "plan_destroy_failed"
    assert result["plan_destroy"][0]["ok"] is False
    assert result["abort"]["ok"] is True
    assert calls == [94, 95]
    assert closed == [FD]


def test_fork_child_entry_fails_closed_and_close_only_drops_child_dup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_pid = 5000
    child_pid = 5001
    _install_session(pid=parent_pid)
    ioctl_calls: list[int] = []
    closed: list[int] = []
    monkeypatch.setattr(uvm_module.os, "getpid", lambda: child_pid)
    monkeypatch.setattr(
        uvm_module,
        "_phase_f_gpu_uuid",
        lambda _device: (_ for _ in ()).throw(
            AssertionError("fork child must fail before CUDA UUID lookup")
        ),
    )
    monkeypatch.setattr(
        uvm_module,
        "_libc_ioctl",
        lambda _fd, request, _params: ioctl_calls.append(request),
    )
    monkeypatch.setattr(uvm_module.os, "close", lambda fd: closed.append(fd))

    restore = uvm_module.driver_slot_restore_ranges(
        [(0x40000000, 4096, 3, 7, 5, 0)],
        device=DEVICE,
    )
    cleanup = uvm_module.close_uvm_slot_direct_session(device=DEVICE)

    assert restore["ok"] is False
    assert restore["reason"] == "fork_inherited_slot_session"
    assert cleanup["ok"] is True
    assert cleanup["reason"] == "fork_child_fd_closed"
    assert cleanup["owner_pid"] == parent_pid
    assert cleanup["current_pid"] == child_pid
    assert ioctl_calls == []
    assert closed == [FD]
    assert DEVICE not in uvm_module._uvm_slot_direct_sessions


def test_atexit_cleanup_synchronizes_before_closing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_session()
    events: list[tuple[str, int]] = []
    monkeypatch.setattr(
        uvm_module.torch.cuda,
        "synchronize",
        lambda device: events.append(("sync", int(device))),
    )
    monkeypatch.setattr(
        uvm_module,
        "close_uvm_slot_direct_session",
        lambda *, device: events.append(("close", int(device))) or {"ok": True},
    )

    uvm_module._close_uvm_slot_direct_sessions()

    assert events == [("sync", DEVICE), ("close", DEVICE)]


def test_atexit_sync_failure_defers_driver_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_session()
    monkeypatch.setattr(
        uvm_module.torch.cuda,
        "synchronize",
        lambda _device: (_ for _ in ()).throw(RuntimeError("CUDA unavailable")),
    )
    monkeypatch.setattr(
        uvm_module,
        "close_uvm_slot_direct_session",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe cleanup must be deferred")
        ),
    )

    uvm_module._close_uvm_slot_direct_sessions()

    assert DEVICE in uvm_module._uvm_slot_direct_sessions
    assert DEVICE in uvm_module._uvm_slot_direct_plans


def test_plan_register_postcondition_failure_immediately_aborts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_session()
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_LAYER_TRANSITION_ENV, "1")
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_SHARED_PUSH_ENV, "1")
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_64K_PTE_ENV, "1")
    monkeypatch.setattr(
        uvm_module,
        "_phase_f_gpu_uuid",
        lambda _device: uvm_module._PhaseFUuid(
            (ctypes.c_uint8 * 16).from_buffer_copy(UUID_BYTES)
        ),
    )
    calls: list[int] = []

    def ioctl(_fd: int, request: int, params: object) -> int:
        calls.append(request)
        if request == 92:
            params.generation = 42
            params.requested_bytes = 2 * 1024 * 1024
        else:
            assert request == 95
            _complete_abort(params, generation=42, destroyed_plans=2)
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module.driver_slot_register_plan(
        12,
        4,
        [
            (
                0x40000000,
                2 * 1024 * 1024,
                4,
                7,
                5,
                0,
                uvm_module.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT
                | uvm_module.UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED,
            )
        ],
        transition=True,
        device=DEVICE,
    )

    assert result["ok"] is False
    assert result["reason"] == "plan_generation_mismatch"
    assert result["abort"]["ok"] is True
    assert calls == [92, 95]
    assert DEVICE not in uvm_module._uvm_slot_direct_plans
    assert DEVICE not in uvm_module._uvm_slot_direct_plan_generations


def test_failed_execute_with_copy_side_effect_immediately_aborts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_session()
    calls: list[int] = []

    def ioctl(_fd: int, request: int, params: object) -> int:
        calls.append(request)
        if request == 93:
            params.new_generation = 41
            params.copy_push_count = 1
            params.poisoned = 0
            return -1
        assert request == 95
        _complete_abort(params, generation=41)
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module.driver_slot_execute_plan(
        11,
        3,
        41,
        device=DEVICE,
    )

    assert result["ok"] is False
    assert result["side_effect_observed"] is True
    assert result["contract_violation"] is True
    assert result["poisoned"] is True
    assert result["session_aborted"] is True
    assert calls == [93, 95]
    assert DEVICE not in uvm_module._uvm_slot_direct_plans
