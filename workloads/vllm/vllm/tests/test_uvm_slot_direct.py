# SPDX-License-Identifier: Apache-2.0

import ctypes
from collections.abc import Callable

import pytest

from vllm.device_allocator import uvm as uvm_module

UUID_BYTES = bytes(range(16))
PAGE_BYTES = 4096
ENVELOPE_BYTES = 512 * 1024
LAYER_EXPERTS = 128
REGIONS_PER_EXPERT = 9
LAYER_REGIONS = LAYER_EXPERTS * REGIONS_PER_EXPERT


def _gpu_uuid() -> uvm_module._PhaseFUuid:
    return uvm_module._PhaseFUuid(
        (ctypes.c_uint8 * len(UUID_BYTES)).from_buffer_copy(UUID_BYTES)
    )


@pytest.fixture(autouse=True)
def _isolated_slot_session(monkeypatch: pytest.MonkeyPatch):
    session_maps = (
        uvm_module._uvm_slot_direct_sessions,
        uvm_module._uvm_slot_direct_owners,
        uvm_module._uvm_slot_direct_plans,
        uvm_module._uvm_slot_direct_plan_generations,
        uvm_module._uvm_slot_direct_active_plan_ids,
        uvm_module._uvm_slot_direct_session_pids,
        uvm_module._uvm_slot_direct_poisoned,
        uvm_module._uvm_slot_direct_cuda_fences,
    )
    for mapping in session_maps:
        mapping.clear()
    uvm_module._uvm_slot_direct_cuda_fence_cleanup_deferred.clear()
    monkeypatch.delenv("VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION", raising=False)
    monkeypatch.delenv(uvm_module._UVM_SLOT_DIRECT_SHARED_PUSH_ENV, raising=False)
    monkeypatch.delenv(uvm_module._UVM_SLOT_DIRECT_64K_PTE_ENV, raising=False)
    monkeypatch.delenv(uvm_module._UVM_SLOT_DIRECT_CUDA_FENCE_ENV, raising=False)
    monkeypatch.delenv(
        uvm_module._UVM_SLOT_DIRECT_BANK_LEVEL_COMMIT_ENV, raising=False
    )
    monkeypatch.setattr(uvm_module, "_phase_f_gpu_uuid", lambda _device: _gpu_uuid())
    monkeypatch.setattr(
        uvm_module,
        "_open_initialized_uvm_fd",
        lambda: (_ for _ in ()).throw(AssertionError("new UVM fd is forbidden")),
    )
    yield
    for mapping in session_maps:
        mapping.clear()
    uvm_module._uvm_slot_direct_cuda_fence_cleanup_deferred.clear()


def _ranges(count: int, *, base: int = 0x10000000):
    return [
        (base + index * PAGE_BYTES, PAGE_BYTES, 3, index, index, 0)
        for index in range(count)
    ]


def _layer_ranges(*, base: int = 0x40000000):
    """Return one Qwen-sized layer with four 512 KiB slices per VA block."""
    return [
        (
            base + index * ENVELOPE_BYTES,
            ENVELOPE_BYTES,
            3,
            index // REGIONS_PER_EXPERT,
            index + 1,
            0,
        )
        for index in range(LAYER_REGIONS)
    ]


def _fragmented_envelope_ranges(
    count: int = 4, *, base: int = 0x50000000
):
    """Return one 512 KiB envelope in each otherwise-unused 2 MiB block."""
    return [
        (
            base + index * (2 * 1024 * 1024),
            ENVELOPE_BYTES,
            3,
            index,
            index + 1,
            0,
        )
        for index in range(count)
    ]


def _install_existing_fd(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sources: list[int] | None = None,
) -> None:
    monkeypatch.setattr(
        uvm_module,
        "_iter_existing_uvm_fd_sources",
        lambda: list(sources if sources is not None else [5]),
    )
    monkeypatch.setattr(
        uvm_module,
        "_initialize_duplicated_uvm_fd",
        lambda source_fd: source_fd + 100,
    )


def _complete_ioctl(
    calls: list[dict[str, object]],
) -> Callable[[int, int, object], int]:
    def ioctl(fd: int, request: int, params: object) -> int:
        entries = []
        requested_bytes = 0
        for index in range(int(params.count)):
            entry = params.entries[index]
            requested_bytes += int(entry.bytes)
            entries.append(
                {
                    "layer_id": int(entry.layer_id),
                    "expert_id": int(entry.expert_id),
                    "slot_id": int(entry.slot_id),
                    "va_base": int(entry.va_base),
                    "bytes": int(entry.bytes),
                    "slot_offset": int(entry.slot_offset),
                    "flags": int(entry.flags),
                }
            )
        calls.append(
            {
                "fd": fd,
                "request": request,
                "mode": int(params.mode),
                "count": int(params.count),
                "validate_managed": int(params.validate_managed),
                "uuid": bytes(params.gpu_uuid.uuid),
                "entries": entries,
            }
        )
        params.success_count = params.count
        params.requested_bytes = requested_bytes
        params.accepted_bytes = requested_bytes
        if int(params.mode) in (10, 11, 12, 20, 21):
            params.remap_count = params.count
            params.tlb_invalidate_count = params.count
            uses_shared_push = int(params.mode) in (11, 21) or (
                int(params.mode) == 20
                and bool(
                    int(params.entries[0].flags)
                    & uvm_module.UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_SHARED_PUSH
                )
            )
            params.reserved = 1 if uses_shared_push else params.count
        elif int(params.mode) == 8:
            params.reserved = sum(
                bool(
                    int(params.entries[index].flags)
                    & uvm_module.UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
                )
                for index in range(int(params.count))
            )
        elif int(params.mode) == 19:
            params.accepted_bytes = int(params.count) * PAGE_BYTES
            params.remap_count = params.count
            params.tlb_invalidate_count = params.count
        return 0

    return ioctl


def test_slot_ioctl_abi_matches_kernel_header() -> None:
    assert ctypes.sizeof(uvm_module._MoeSlotRemapEntry) == 48
    assert uvm_module._MoeSlotRemapEntry.va_base.offset == 16
    assert uvm_module._MoeSlotRemapEntry.status.offset == 44
    assert uvm_module._UVM_MOE_SLOT_REMAP_MAX_ENTRIES == 4608
    assert uvm_module._UVM_MOE_SLOT_REMAP_MAX_ENTRIES >= LAYER_REGIONS
    assert ctypes.sizeof(uvm_module._MoeSlotRemapBatchParams) == 221264
    assert uvm_module._MoeSlotRemapBatchParams.requested_bytes.offset == 56
    assert uvm_module._MoeSlotRemapBatchParams.entries.offset == 72
    assert uvm_module._MoeSlotRemapBatchParams.rmStatus.offset == 221256
    assert (
        uvm_module._UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_SHARED_PUSH_64K_PTE
        == 21
    )
    assert (
        uvm_module.UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_SHARED_PUSH
        == 0x4
    )
    assert uvm_module.UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_64K_PTE == 0x8
    assert uvm_module._UVM_MOE_SLOT_PLAN_EXECUTE_FENCED_IOCTL == 96
    assert ctypes.sizeof(uvm_module._MoeSlotPlanExecuteFencedParams) == 128
    assert uvm_module._MoeSlotPlanExecuteFencedParams.fence_flags.offset == 84
    assert (
        uvm_module._MoeSlotPlanExecuteFencedParams.compute_semaphore_address.offset
        == 88
    )
    assert uvm_module._MoeSlotPlanExecuteFencedParams.rmStatus.offset == 124


def test_preallocate_then_restore_reuses_existing_va_space_fd_and_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))
    entry = (0x20000000, PAGE_BYTES, 7, 11, 13, 0)

    preallocate = uvm_module.driver_slot_preallocate_ranges([entry])
    restore = uvm_module.driver_slot_restore_ranges([entry])

    assert preallocate["ok"] is True
    assert restore["ok"] is True
    assert preallocate["wall_ns"] > 0
    assert restore["wall_ns"] > 0
    assert preallocate["ioctl_calls"] == restore["ioctl_calls"] == 1
    assert preallocate["remap_count"] == 0
    assert preallocate["tlb_invalidate_count"] == 0
    assert preallocate["copy_push_count"] == 0
    assert preallocate["backing_reuse_count"] == 0
    assert restore["remap_count"] == 1
    assert restore["tlb_invalidate_count"] == 1
    assert restore["copy_push_count"] == 1
    assert [call["mode"] for call in calls] == [8, 10]
    assert [call["fd"] for call in calls] == [105, 105]
    assert all(call["request"] == 91 for call in calls)
    assert all(call["validate_managed"] == 1 for call in calls)
    assert all(call["uuid"] == UUID_BYTES for call in calls)
    assert all(call["entries"][0]["flags"] == 1 for call in calls)


@pytest.mark.parametrize("operation", ["preallocate", "restore", "transition", "register"])
def test_slot_operations_recheck_poison_after_gpu_uuid_lookup(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    device = 0
    uvm_module._uvm_slot_direct_sessions[device] = (105, UUID_BYTES)
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION", "1")

    def poison_during_uuid(_device: int) -> uvm_module._PhaseFUuid:
        uvm_module._uvm_slot_direct_poisoned[device] = "concurrent failure"
        return _gpu_uuid()

    monkeypatch.setattr(uvm_module, "_phase_f_gpu_uuid", poison_during_uuid)
    monkeypatch.setattr(
        uvm_module,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("poisoned operation must not issue an ioctl")
        ),
    )
    entry = (0x20000000, PAGE_BYTES, 7, 11, 13, 0)
    transition_entry = (
        *entry,
        uvm_module.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT
        | uvm_module.UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED,
    )

    if operation == "preallocate":
        result = uvm_module.driver_slot_preallocate_ranges([entry], device=device)
    elif operation == "restore":
        result = uvm_module.driver_slot_restore_ranges([entry], device=device)
    elif operation == "transition":
        result = uvm_module.driver_slot_transition_ranges(
            [transition_entry], device=device
        )
    else:
        result = uvm_module.driver_slot_register_plan(
            17,
            7,
            [transition_entry],
            transition=True,
            device=device,
        )

    assert result["ok"] is False
    assert result["reason"] == "slot_session_poisoned"


def test_preallocate_reuse_required_is_audited_separately_from_copy_pushes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))
    entry = (
        0x21000000,
        PAGE_BYTES,
        0xFFFFFFFF,
        123,
        13,
        0,
        uvm_module.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT
        | uvm_module.UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED,
    )

    result = uvm_module.driver_slot_preallocate_ranges([entry])

    assert result["ok"] is True
    assert result["backing_reuse_count"] == 1
    assert result["copy_push_count"] == 0
    assert calls[0]["mode"] == 8
    assert calls[0]["entries"][0]["flags"] == 3


def test_preallocate_syscall_failure_cannot_commit_zero_initialized_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uvm_module._uvm_slot_direct_sessions[0] = (105, UUID_BYTES)
    uvm_module._uvm_slot_direct_owners[0] = {(7, 13): 11}

    def ioctl(_fd: int, _request: int, params: object) -> int:
        # Even apparently successful aggregate output cannot prove an entry
        # commit when the syscall transport itself failed.
        requested_bytes = int(params.entries[0].bytes)
        params.success_count = params.count
        params.requested_bytes = requested_bytes
        params.accepted_bytes = requested_bytes
        ctypes.set_errno(5)
        return -1

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module.driver_slot_preallocate_ranges(
        [(0x21800000, PAGE_BYTES, 7, 12, 13, 0)]
    )

    assert result["ok"] is False
    assert result["reason"] == "ioctl_ret"
    assert result["entry_statuses"] == [None]
    assert uvm_module._uvm_slot_direct_owners[0] == {(7, 13): 11}


def test_preallocate_keeps_a_complete_1152_region_layer_in_one_ioctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))

    result = uvm_module.driver_slot_preallocate_ranges(_layer_ranges())

    assert result["ok"] is True
    assert result["ioctl_calls"] == 1
    assert [call["mode"] for call in calls] == [8]
    assert [call["count"] for call in calls] == [LAYER_REGIONS]
    assert result["success_count"] == LAYER_REGIONS
    assert result["accepted_bytes"] == LAYER_REGIONS * ENVELOPE_BYTES


def test_preallocate_splits_only_after_a_complete_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))

    batch_entries = uvm_module._UVM_MOE_SLOT_REMAP_MAX_ENTRIES
    assert batch_entries % LAYER_REGIONS == 0
    result = uvm_module.driver_slot_preallocate_ranges(
        _ranges(batch_entries + 1)
    )

    assert result["ok"] is True
    assert result["ioctl_calls"] == 2
    assert [call["count"] for call in calls] == [batch_entries, 1]
    assert result["success_count"] == batch_entries + 1
    assert result["accepted_bytes"] == (batch_entries + 1) * PAGE_BYTES


def test_restore_keeps_four_nonoverlapping_slices_of_one_va_block_in_one_ioctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))
    base = 0x30000000
    entries = [
        (base + index * ENVELOPE_BYTES, ENVELOPE_BYTES, 1, index, index + 1, 0)
        for index in range(4)
    ]
    assert uvm_module.driver_slot_preallocate_ranges(entries)["ok"] is True
    calls.clear()

    result = uvm_module.driver_slot_restore_ranges(entries)

    assert result["ok"] is True
    assert result["ioctl_calls"] == 1
    assert [batch["input_indices"] for batch in result["batches"]] == [
        [0, 1, 2, 3]
    ]
    assert [call["mode"] for call in calls] == [10]
    assert [call["count"] for call in calls] == [4]
    assert [entry["va_base"] for entry in calls[0]["entries"]] == [
        base + index * ENVELOPE_BYTES for index in range(4)
    ]


def test_complete_layer_uses_exactly_one_mode8_and_one_mode10_ioctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))
    entries = _layer_ranges()

    preallocate = uvm_module.driver_slot_preallocate_ranges(entries)
    restore = uvm_module.driver_slot_restore_ranges(entries)

    expected_bytes = LAYER_REGIONS * ENVELOPE_BYTES
    assert preallocate["ok"] is True
    assert restore["ok"] is True
    assert preallocate["ioctl_calls"] == 1
    assert restore["ioctl_calls"] == 1
    assert preallocate["requested_count"] == restore["requested_count"] == LAYER_REGIONS
    assert preallocate["accepted_bytes"] == restore["accepted_bytes"] == expected_bytes
    assert preallocate["entry_statuses"] == [0] * LAYER_REGIONS
    assert restore["entry_statuses"] == [0] * LAYER_REGIONS
    assert [(call["mode"], call["count"]) for call in calls] == [
        (8, LAYER_REGIONS),
        (10, LAYER_REGIONS),
    ]
    assert len({entry["expert_id"] for entry in calls[1]["entries"]}) == LAYER_EXPERTS
    assert len({entry["slot_id"] for entry in calls[1]["entries"]}) == LAYER_REGIONS


def test_layer_transition_rebinds_owner_and_restores_in_one_ioctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION", "1")
    old_entry = (0x43000000, PAGE_BYTES, 0xFFFFFFFF, 11, 13, 0)
    new_entry = (
        0x44000000,
        PAGE_BYTES,
        0xFFFFFFFF,
        12,
        13,
        0,
        uvm_module.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT
        | uvm_module.UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED,
    )
    assert uvm_module.driver_slot_preallocate_ranges([old_entry])["ok"] is True
    calls.clear()

    result = uvm_module.driver_slot_transition_ranges([new_entry])

    assert result["ok"] is True
    assert result["mode"] == 20
    assert result["ioctl_calls"] == 1
    assert result["owner_cutover"] is True
    assert result["owner_cutover_count"] == 1
    assert result["owner_cutover_trusted"] is True
    assert result["owner_state_indeterminate"] is False
    assert [(call["mode"], call["count"]) for call in calls] == [(20, 1)]
    assert calls[0]["entries"][0]["flags"] == 3
    assert uvm_module._uvm_slot_direct_owners[0] == {(0xFFFFFFFF, 13): 12}


def test_layer_transition_is_disabled_by_default_without_ioctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION", raising=False)
    monkeypatch.setattr(
        uvm_module,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected ioctl")),
    )
    entry = (0x45000000, PAGE_BYTES, 0xFFFFFFFF, 12, 13, 0, 3)

    result = uvm_module.driver_slot_transition_ranges([entry])

    assert result["ok"] is False
    assert result["reason"] == "layer_transition_disabled"
    assert result["ioctl_calls"] == 0


@pytest.mark.parametrize(
    (
        "cutover_count",
        "expected_cutover",
        "expected_trusted",
        "expected_indeterminate",
    ),
    [(0, False, True, False), (1, None, False, True)],
)
def test_failed_layer_transition_only_trusts_proven_no_cutover(
    monkeypatch: pytest.MonkeyPatch,
    cutover_count: int,
    expected_cutover: bool | None,
    expected_trusted: bool,
    expected_indeterminate: bool,
) -> None:
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION", "1")
    uvm_module._uvm_slot_direct_sessions[0] = (105, UUID_BYTES)
    uvm_module._uvm_slot_direct_owners[0] = {(0xFFFFFFFF, 13): 11}

    def ioctl(_fd: int, _request: int, params: object) -> int:
        params.requested_bytes = int(params.entries[0].bytes)
        params.success_count = 0
        params.failure_count = 1
        params.remap_count = cutover_count
        params.rmStatus = 5
        params.entries[0].status = 7
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    entry = (0x46000000, PAGE_BYTES, 0xFFFFFFFF, 12, 13, 0, 3)

    result = uvm_module.driver_slot_transition_ranges([entry])

    assert result["ok"] is False
    assert result["owner_cutover"] is expected_cutover
    assert result["owner_cutover_trusted"] is expected_trusted
    assert result["owner_state_indeterminate"] is expected_indeterminate
    # A malformed failure report must never advance userspace ownership.
    assert uvm_module._uvm_slot_direct_owners[0][(0xFFFFFFFF, 13)] == 11


def test_layer_transition_syscall_failure_makes_owner_indeterminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION", "1")
    uvm_module._uvm_slot_direct_sessions[0] = (105, UUID_BYTES)
    uvm_module._uvm_slot_direct_owners[0] = {(0xFFFFFFFF, 13): 11}

    def ioctl(_fd: int, _request: int, params: object) -> int:
        params.requested_bytes = int(params.entries[0].bytes)
        params.remap_count = 1
        ctypes.set_errno(5)
        return -1

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_slot_transition_ranges(
        [(0x47000000, PAGE_BYTES, 0xFFFFFFFF, 12, 13, 0, 3)]
    )

    assert result["ok"] is False
    assert result["owner_cutover"] is None
    assert result["owner_cutover_trusted"] is False
    assert result["owner_state_indeterminate"] is True
    assert uvm_module._uvm_slot_direct_owners[0][(0xFFFFFFFF, 13)] == 11


def test_layer_transition_64k_policy_keeps_partial_block_in_64k_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION", "1")
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_64K_PTE_ENV, "1")
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))
    old_entry = _layer_ranges(base=0x47600000)[0]
    assert uvm_module.driver_slot_preallocate_ranges([old_entry])["ok"] is True
    calls.clear()
    entry = (*_layer_ranges(base=0x47800000)[0], 3)

    result = uvm_module.driver_slot_transition_ranges([entry])

    assert result["ok"] is True
    assert result["mode"] == 20
    assert result["ioctl_calls"] == 1
    assert [(call["mode"], call["count"]) for call in calls] == [(20, 1)]
    assert calls[0]["entries"][0]["flags"] == (
        uvm_module.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT
        | uvm_module.UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
        | uvm_module.UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_64K_PTE
    )


def test_complete_layer_can_select_mode11_shared_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_SHARED_PUSH_ENV, "1")
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))
    entries = _layer_ranges()

    assert uvm_module.driver_slot_preallocate_ranges(entries)["ok"] is True
    result = uvm_module.driver_slot_restore_ranges(entries)

    assert result["ok"] is True
    assert result["mode"] == 11
    assert result["ioctl_calls"] == 1
    assert result["copy_push_count"] == 1
    assert [(call["mode"], call["count"]) for call in calls] == [
        (8, LAYER_REGIONS),
        (11, LAYER_REGIONS),
    ]


def test_four_512k_slices_can_select_mode12_64k_pte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_64K_PTE_ENV, "1")
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))
    entries = _layer_ranges()[:4]

    assert uvm_module.driver_slot_preallocate_ranges(entries)["ok"] is True
    result = uvm_module.driver_slot_restore_ranges(entries)

    assert result["ok"] is True
    assert result["mode"] == 12
    assert result["ioctl_calls"] == 1
    assert [(call["mode"], call["count"]) for call in calls] == [
        (8, 4),
        (12, 4),
    ]


def test_four_512k_slices_can_select_mode21_shared_push_and_64k_pte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_SHARED_PUSH_ENV, "1")
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_64K_PTE_ENV, "1")
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))
    entries = _layer_ranges()[:4]

    assert uvm_module.driver_slot_preallocate_ranges(entries)["ok"] is True
    result = uvm_module.driver_slot_restore_ranges(entries)

    assert result["ok"] is True
    assert result["mode"] == 21
    assert result["ioctl_calls"] == 1
    assert result["copy_push_count"] == 1
    assert [(call["mode"], call["count"]) for call in calls] == [
        (8, 4),
        (21, 4),
    ]


def test_fragmented_registered_bootstrap_selects_mode11(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    uvm_module._uvm_slot_direct_sessions[0] = (105, UUID_BYTES)
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_SHARED_PUSH_ENV, "1")
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_64K_PTE_ENV, "0")

    def ioctl(fd: int, request: int, params: object) -> int:
        calls.append(
            {
                "fd": fd,
                "request": request,
                "mode": int(params.mode),
                "count": int(params.count),
                "flags": [
                    int(params.entries[index].flags)
                    for index in range(int(params.count))
                ],
            }
        )
        params.generation = 7
        params.requested_bytes = sum(
            int(params.entries[index].bytes)
            for index in range(int(params.count))
        )
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module.driver_slot_register_plan(
        11,
        3,
        _fragmented_envelope_ranges(),
        transition=False,
    )

    assert result["ok"] is True
    assert result["mode"] == 11
    assert calls == [
        {
            "fd": 105,
            "request": uvm_module._UVM_MOE_SLOT_PLAN_REGISTER_IOCTL,
            "mode": 11,
            "count": 4,
            "flags": [uvm_module.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT] * 4,
        }
    ]
    assert uvm_module._uvm_slot_direct_plans[0][11]["mode"] == 11


def test_fragmented_registered_bootstrap_selects_sparse_mode21(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    uvm_module._uvm_slot_direct_sessions[0] = (105, UUID_BYTES)
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_SHARED_PUSH_ENV, "1")
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_64K_PTE_ENV, "1")

    def ioctl(fd: int, request: int, params: object) -> int:
        calls.append(
            {
                "fd": fd,
                "request": request,
                "mode": int(params.mode),
                "count": int(params.count),
            }
        )
        params.generation = 7
        params.requested_bytes = sum(
            int(params.entries[index].bytes)
            for index in range(int(params.count))
        )
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_slot_register_plan(
        21,
        3,
        _fragmented_envelope_ranges(),
        transition=False,
    )

    assert result["ok"] is True
    assert result["mode"] == 21
    assert calls == [
        {
            "fd": 105,
            "request": uvm_module._UVM_MOE_SLOT_PLAN_REGISTER_IOCTL,
            "mode": 21,
            "count": 4,
        }
    ]
    assert uvm_module._uvm_slot_direct_plans[0][21]["mode"] == 21


@pytest.mark.parametrize("shared_push", [False, True])
def test_64k_restore_modes_keep_partial_2m_block_in_explicit_64k_mode(
    monkeypatch: pytest.MonkeyPatch,
    shared_push: bool,
) -> None:
    calls: list[dict[str, object]] = []
    if shared_push:
        monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_SHARED_PUSH_ENV, "1")
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_64K_PTE_ENV, "1")
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))
    entry = _layer_ranges()[:1]
    assert uvm_module.driver_slot_preallocate_ranges(entry)["ok"] is True
    calls.clear()

    result = uvm_module.driver_slot_restore_ranges(entry)

    assert result["ok"] is True
    assert result["mode"] == (21 if shared_push else 12)
    assert result["ioctl_calls"] == 1
    assert [(call["mode"], call["count"]) for call in calls] == [
        (21 if shared_push else 12, 1)
    ]


def test_sparse_64k_layout_preserves_holes_for_ordinary_hot_leaves() -> None:
    """A registered cold plan may leave non-overlapping hot leaves untouched."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    block_base = 0x52000000
    GPUModelRunner._validate_uvm_slot_64k_layout(
        [
            (block_base, ENVELOPE_BYTES),
            (block_base + 2 * ENVELOPE_BYTES, ENVELOPE_BYTES),
        ],
        layer_id=3,
    )


def test_registered_sparse_64k_layout_rejects_overlapping_cold_leaves() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    block_base = 0x52200000
    with pytest.raises(RuntimeError, match="64 KiB ranges overlap"):
        GPUModelRunner._validate_uvm_slot_64k_layout(
            [
                (block_base, ENVELOPE_BYTES),
                (block_base + ENVELOPE_BYTES // 2, ENVELOPE_BYTES),
            ],
            layer_id=3,
        )


@pytest.mark.parametrize(
    "entry",
    [
        (0x52401000, ENVELOPE_BYTES, 3, 0, 1, 0),
        (0x52600000, ENVELOPE_BYTES - PAGE_BYTES, 3, 0, 1, 0),
        (0x52800000, ENVELOPE_BYTES, 3, 0, 1, PAGE_BYTES),
    ],
)
def test_sparse_64k_mode_rejects_non_64k_geometry_before_ioctl(
    monkeypatch: pytest.MonkeyPatch,
    entry: tuple[int, int, int, int, int, int],
) -> None:
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_SHARED_PUSH_ENV, "1")
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_64K_PTE_ENV, "1")
    monkeypatch.setattr(
        uvm_module,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected ioctl")),
    )

    result = uvm_module.driver_slot_restore_ranges([entry])

    assert result["ok"] is False
    assert result["reason"] == "validation_failed"
    assert result["mode"] == 21
    assert "64 KiB aligned" in result["error"]
    assert result["ioctl_calls"] == 0


def test_sparse_64k_mode_rejects_overlapping_slices_before_ioctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_SHARED_PUSH_ENV, "1")
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_64K_PTE_ENV, "1")
    monkeypatch.setattr(
        uvm_module,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected ioctl")),
    )
    block_base = 0x52A00000
    entries = [
        (block_base, ENVELOPE_BYTES, 3, 0, 1, 0),
        (block_base + ENVELOPE_BYTES // 2, ENVELOPE_BYTES, 3, 1, 2, 0),
    ]

    result = uvm_module.driver_slot_restore_ranges(entries)

    assert result["ok"] is False
    assert result["reason"] == "validation_failed"
    assert result["mode"] == 21
    assert result["error"] == "source managed ranges overlap"
    assert result["ioctl_calls"] == 0


def test_restore_mode_gates_are_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uvm_module._uvm_slot_direct_sessions[0] = (105, UUID_BYTES)
    monkeypatch.setattr(
        uvm_module,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected ioctl")),
    )
    entries = _layer_ranges()[:4]

    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_SHARED_PUSH_ENV, "yes")
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_64K_PTE_ENV, "0")
    invalid = uvm_module.driver_slot_restore_ranges(entries)
    assert invalid["ok"] is False
    assert invalid["reason"] == "validation_failed"
    assert "must be 0 or 1" in invalid["error"]


def test_layer_transition_adds_one_consistent_combined_policy_to_all_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION", "1")
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_SHARED_PUSH_ENV, "1")
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_64K_PTE_ENV, "1")
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))
    old_entries = _layer_ranges(base=0x48000000)[:4]
    new_entries = [
        (
            *entry[:2],
            0xFFFFFFFF,
            entry[3] + 100,
            entry[4],
            entry[5],
            uvm_module.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT
            | uvm_module.UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED,
        )
        for entry in _layer_ranges(base=0x4A000000)[:4]
    ]
    old_entries = [
        (*entry[:2], 0xFFFFFFFF, entry[3], entry[4], entry[5])
        for entry in old_entries
    ]
    assert uvm_module.driver_slot_preallocate_ranges(old_entries)["ok"] is True
    calls.clear()

    result = uvm_module.driver_slot_transition_ranges(new_entries)

    assert result["ok"] is True
    assert result["mode"] == 20
    assert result["copy_push_count"] == 1
    assert [(call["mode"], call["count"]) for call in calls] == [(20, 4)]
    expected_flags = (
        uvm_module.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT
        | uvm_module.UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
        | uvm_module.UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_SHARED_PUSH
        | uvm_module.UVM_MOE_SLOT_REMAP_FLAG_LAYER_TRANSITION_64K_PTE
    )
    assert {entry["flags"] for entry in calls[0]["entries"]} == {
        expected_flags
    }


def test_partial_batch_failure_stops_before_unsubmitted_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    _install_existing_fd(monkeypatch)

    def ioctl(_fd: int, _request: int, params: object) -> int:
        nonlocal calls
        calls += 1
        requested_bytes = sum(
            int(params.entries[index].bytes) for index in range(int(params.count))
        )
        params.requested_bytes = requested_bytes
        params.accepted_bytes = requested_bytes
        params.success_count = params.count
        if calls == 2:
            params.success_count = int(params.count) - 1
            params.failure_count = 1
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    batch_entries = uvm_module._UVM_MOE_SLOT_REMAP_MAX_ENTRIES
    result = uvm_module.driver_slot_preallocate_ranges(
        _ranges(2 * batch_entries + 1)
    )

    assert result["ok"] is False
    assert result["reason"] == "short_success"
    assert result["ioctl_calls"] == 2
    assert calls == 2
    assert result["entry_statuses"][-1] is None


def test_no_existing_uvm_fd_fails_closed_without_open_or_ioctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(uvm_module, "_iter_existing_uvm_fd_sources", lambda: [])
    monkeypatch.setattr(
        uvm_module,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected ioctl")),
    )

    result = uvm_module.driver_slot_preallocate_ranges(_ranges(1))

    assert result["ok"] is False
    assert result["reason"] == "no_existing_uvm_fd"
    assert result["ioctl_calls"] == 0


def test_restore_requires_successful_preallocation_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        uvm_module,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected ioctl")),
    )

    result = uvm_module.driver_slot_restore_ranges(_ranges(1))

    assert result["ok"] is False
    assert result["reason"] == "slot_preallocation_required"


def test_release_uses_mode19_identity_only_and_closes_audited_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))
    entry = (0x50000000, PAGE_BYTES, 7, 11, 13, 0)
    assert uvm_module.driver_slot_preallocate_ranges([entry])["ok"] is True

    result = uvm_module.driver_slot_release_ranges([(7, 11, 13)])

    assert result["ok"] is True
    assert result["mode"] == 19
    assert result["success_count"] == 1
    assert result["requested_bytes"] == 0
    assert result["accepted_bytes"] == PAGE_BYTES
    assert result["remap_count"] == 1
    assert result["tlb_invalidate_count"] == 1
    assert result["copy_push_count"] == 0
    release_call = calls[-1]
    assert release_call["mode"] == 19
    assert release_call["validate_managed"] == 0
    assert release_call["entries"] == [
        {
            "layer_id": 7,
            "expert_id": 11,
            "slot_id": 13,
            "va_base": 0,
            "bytes": 0,
            "slot_offset": 0,
            "flags": 0,
        }
    ]
    assert uvm_module._uvm_slot_direct_owners[0] == {}


def test_close_session_releases_all_slots_before_closing_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    closed: list[int] = []
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))
    monkeypatch.setattr(uvm_module.os, "close", lambda fd: closed.append(fd))
    entries = [
        (0x51000000, PAGE_BYTES, 7, 11, 13, 0),
        (0x51200000, PAGE_BYTES, 7, 12, 14, 0),
    ]
    assert uvm_module.driver_slot_preallocate_ranges(entries)["ok"] is True

    result = uvm_module.close_uvm_slot_direct_session(device=0)

    assert result["ok"] is True
    assert result["closed"] is True
    assert result["release"]["ok"] is True
    assert [call["mode"] for call in calls] == [8, 19]
    assert closed == [105]
    assert 0 not in uvm_module._uvm_slot_direct_sessions
    assert 0 not in uvm_module._uvm_slot_direct_owners


def test_close_fd_failure_keeps_session_for_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    close_attempts: list[int] = []
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))

    def close_once_then_succeed(fd: int) -> None:
        close_attempts.append(fd)
        if len(close_attempts) == 1:
            raise OSError("synthetic close failure")

    monkeypatch.setattr(uvm_module.os, "close", close_once_then_succeed)
    entry = (0x51800000, PAGE_BYTES, 7, 11, 13, 0)
    assert uvm_module.driver_slot_preallocate_ranges([entry])["ok"] is True

    first = uvm_module.close_uvm_slot_direct_session(device=0)
    assert first["ok"] is False
    assert first["reason"] == "fd_close_failed"
    assert first["release"]["ok"] is True
    assert 0 in uvm_module._uvm_slot_direct_sessions
    assert uvm_module._uvm_slot_direct_owners[0] == {}

    second = uvm_module.close_uvm_slot_direct_session(device=0)
    assert second["ok"] is True
    assert second["closed"] is True
    assert second["release"] is None
    assert close_attempts == [105, 105]
    assert [call["mode"] for call in calls] == [8, 19]
    assert 0 not in uvm_module._uvm_slot_direct_sessions


def test_release_rejects_untracked_or_wrong_owner_without_ioctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    _install_existing_fd(monkeypatch)
    monkeypatch.setattr(uvm_module, "_libc_ioctl", _complete_ioctl(calls))
    entry = (0x52000000, PAGE_BYTES, 7, 11, 13, 0)
    assert uvm_module.driver_slot_preallocate_ranges([entry])["ok"] is True
    calls.clear()

    result = uvm_module.driver_slot_release_ranges([(7, 99, 13)])

    assert result["ok"] is False
    assert result["reason"] == "slot_owner_mismatch"
    assert calls == []


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [(0x40000001, PAGE_BYTES, 1, 1, 1, 0)],
        [(0x40000000, 4 * 1024 * 1024 + PAGE_BYTES, 1, 1, 1, 0)],
        [
            (0x40000000, 2 * PAGE_BYTES, 1, 1, 1, 0),
            (0x40001000, PAGE_BYTES, 1, 2, 2, 0),
        ],
        [
            (0x40000000, PAGE_BYTES, 1, 1, 1, 0),
            (0x40001000, PAGE_BYTES, 1, 1, 1, 0),
        ],
        [(0x40000000, PAGE_BYTES, 1, 1, 1, 0, -1)],
        [(0x40000000, PAGE_BYTES, 1, 1, 1, 0, 4)],
    ],
)
def test_invalid_ranges_are_rejected_before_fd_or_ioctl(
    monkeypatch: pytest.MonkeyPatch,
    entries: list[tuple[int, ...]],
) -> None:
    monkeypatch.setattr(
        uvm_module,
        "_iter_existing_uvm_fd_sources",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected fd scan")),
    )
    monkeypatch.setattr(
        uvm_module,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected ioctl")),
    )

    result = uvm_module.driver_slot_preallocate_ranges(entries)

    assert result["ok"] is False
    assert result["reason"] == "validation_failed"
    assert result["ioctl_calls"] == 0


@pytest.mark.parametrize(
    ("failure", "expected_reason"),
    [
        ("syscall", "ioctl_ret"),
        ("rm_status", "rm_status"),
        ("entry_status", "entry_status"),
        ("short_success", "short_success"),
        ("short_bytes", "accepted_bytes"),
    ],
)
def test_restore_contract_failures_never_fallback_to_prefetch(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected_reason: str,
) -> None:
    uvm_module._uvm_slot_direct_sessions[0] = (105, UUID_BYTES)
    monkeypatch.setattr(
        uvm_module,
        "batch_prefetch_ranges_to_device",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prefetch fallback is forbidden")
        ),
    )

    def ioctl(_fd: int, _request: int, params: object) -> int:
        requested_bytes = int(params.entries[0].bytes)
        params.success_count = params.count
        params.requested_bytes = requested_bytes
        params.accepted_bytes = requested_bytes
        params.remap_count = params.count
        params.tlb_invalidate_count = params.count
        params.reserved = params.count
        if failure == "syscall":
            ctypes.set_errno(5)
            return -1
        if failure == "rm_status":
            params.rmStatus = 5
        elif failure == "entry_status":
            params.entries[0].status = 7
        elif failure == "short_success":
            params.success_count = 0
        elif failure == "short_bytes":
            params.accepted_bytes = requested_bytes - 1
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_slot_restore_ranges(_ranges(1))

    assert result["ok"] is False
    assert result["reason"] == expected_reason
    assert result["ioctl_calls"] == 1
