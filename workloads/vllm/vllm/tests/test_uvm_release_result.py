# SPDX-License-Identifier: Apache-2.0

import ctypes
import inspect
import os
from types import SimpleNamespace

import pytest

from vllm.device_allocator import uvm as uvm_module


def test_cuda_fence_ticket_exhaustion_never_wraps_to_a_stale_value() -> None:
    fence = uvm_module._UvmSlotCudaFence(
        fd=123,
        base=0x1000,
        device_base=0x2000,
        length=4096,
    )
    fence.next_ticket = uvm_module._UVM_MOE_SLOT_UINT32_MAX - 1

    assert fence.reserve_ticket() == uvm_module._UVM_MOE_SLOT_UINT32_MAX - 1
    assert fence.reserve_ticket() == uvm_module._UVM_MOE_SLOT_UINT32_MAX
    with pytest.raises(RuntimeError, match="ticket space exhausted"):
        fence.reserve_ticket()


def test_registered_p3_cuda_fence_requires_raw_uva_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        uvm_module._UVM_SLOT_DIRECT_CUDA_FENCE_KERNEL_LIB_ENV,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="bounded raw-UVA"):
        uvm_module._uvm_slot_cuda_fence_create_locked(device=0, fd=123)


def test_region_state_ioctl_is_available_for_local_uvm_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []
    runtime = SimpleNamespace(
        uvm_driver_update_region_state=lambda region_id, state: (
            calls.append((int(region_id.value), int(state.value))) or 1
        ),
        uvm_driver_query_region=lambda *_args: 1,
        uvm_driver_query_region_by_va=lambda *_args: 1,
    )
    monkeypatch.setattr(uvm_module, "_uvm_enabled", False)
    monkeypatch.setattr(uvm_module, "_uvm_lib", runtime)

    assert uvm_module.has_region_state_ioctl() is True
    assert uvm_module.driver_update_region_state(region_id=7, state=4) == 1
    assert calls == [(7, 4)]


def _release_params(*, remote_read: bool = False):
    params = uvm_module._ExpertOffloadReleaseParams()
    params.region_id = 7
    params.base = 0x100000
    params.length = 0x80000
    if remote_read:
        params.flags = uvm_module.UVM_EXPERT_OFFLOAD_RELEASE_FLAG_REMOTE_READ
    return params


def _patch_driver_fds(
    monkeypatch: pytest.MonkeyPatch,
    *,
    existing_sources: list[int] | None = None,
    new_fd: int = 200,
) -> list[int]:
    closed: list[int] = []
    monkeypatch.setattr(
        uvm_module,
        "_phase_f_gpu_uuid",
        lambda _device: uvm_module._PhaseFUuid(),
    )
    monkeypatch.setattr(
        uvm_module,
        "_iter_existing_uvm_fd_sources",
        lambda: list(existing_sources or []),
    )
    monkeypatch.setattr(
        uvm_module,
        "_initialize_duplicated_uvm_fd",
        lambda source_fd: source_fd + 100,
    )
    monkeypatch.setattr(uvm_module, "_open_initialized_uvm_fd", lambda: new_fd)
    monkeypatch.setattr(
        uvm_module.os,
        "close",
        lambda fd: closed.append(fd),
    )
    return closed


def test_immutable_cpu_shadow_cutover_batches_and_audits_every_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranges = [
        (0x100000 + index * 0x10000, 0x10000) for index in range(65)
    ]
    closed = _patch_driver_fds(monkeypatch, existing_sources=[10])
    batch_sizes: list[int] = []

    def ioctl(fd: int, request: int, params: object) -> int:
        assert fd == 110
        assert request == uvm_module._UVM_PHASE_F_BATCH_EXECUTOR_IOCTL
        assert isinstance(params, uvm_module._PhaseFBatchExecutorParams)
        assert params.action == (
            uvm_module.UVM_PHASE_F_ACTION_IMMUTABLE_CPU_SHADOW_CUTOVER
        )
        assert params.async_ == 0
        batch_sizes.append(int(params.count))
        params.success_count = params.count
        params.failure_count = 0
        params.rmStatus = 0
        for index in range(int(params.count)):
            length = int(params.ranges[index].length)
            result = params.results[index]
            result.status = 0
            result.resident_gpu_bytes = length
            result.resident_cpu_bytes = length
            result.resident_any_bytes = length
            result.resident_unknown_bytes = 0
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_cutover_ranges_to_immutable_cpu_shadow(ranges)

    assert uvm_module.UVM_PHASE_F_ACTION_IMMUTABLE_CPU_SHADOW_CUTOVER == 22
    assert result["ok"] is True
    assert result["range_count"] == 65
    assert result["success_count"] == 65
    assert result["failure_count"] == 0
    assert result["requested_bytes"] == 65 * 0x10000
    assert result["released_gpu_bytes"] == 65 * 0x10000
    assert result["retained_cpu_shadow_bytes"] == 65 * 0x10000
    assert result["remote_mapped_bytes"] == 65 * 0x10000
    assert result["selected_fd_source"] == 10
    assert batch_sizes == [64, 1]
    assert closed == [110]


def test_read_duplicate_fast_promotion_uses_phase_f_action_18(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranges = ((0x100000, 0x10000), (0x200000, 0x20000))
    closed = _patch_driver_fds(monkeypatch, existing_sources=[10])
    calls: list[tuple[int, int, int]] = []

    def ioctl(fd: int, request: int, params: object) -> int:
        assert isinstance(params, uvm_module._PhaseFBatchExecutorParams)
        calls.append((int(fd), int(request), int(params.action)))
        params.success_count = params.count
        params.failure_count = 0
        params.rmStatus = 0
        for index in range(int(params.count)):
            length = int(params.ranges[index].length)
            result = params.results[index]
            result.status = 0
            result.resident_gpu_bytes = length
            result.resident_cpu_bytes = length
            result.resident_any_bytes = length
            result.resident_unknown_bytes = 0
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_promote_ranges_to_read_duplicate_fast(
        ranges, device=2
    )

    assert uvm_module.UVM_PHASE_F_ACTION_EXPERT_READ_DUP_FAST == 18
    assert result["ok"] is True
    assert result["action"] == "expert_read_duplicate_fast"
    assert result["range_count"] == 2
    assert result["success_count"] == 2
    assert result["failure_count"] == 0
    assert result["requested_bytes"] == 0x30000
    assert result["gpu_read_duplicate_bytes"] == 0x30000
    assert result["cpu_shadow_bytes"] == 0x30000
    assert result["selected_fd_source"] == 10
    assert calls == [(110, uvm_module._UVM_PHASE_F_BATCH_EXECUTOR_IOCTL, 18)]
    assert closed == [110]


def test_read_duplicate_fast_promotion_skips_nonowning_uvm_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranges = ((0x100000, 0x10000),)
    closed = _patch_driver_fds(monkeypatch, existing_sources=[10, 20])

    def ioctl(fd: int, _request: int, params: object) -> int:
        assert isinstance(params, uvm_module._PhaseFBatchExecutorParams)
        result = params.results[0]
        length = int(params.ranges[0].length)
        if fd == 110:
            params.success_count = 0
            params.failure_count = 1
            params.rmStatus = uvm_module._NV_ERR_INVALID_DEVICE
            result.status = uvm_module._NV_ERR_INVALID_DEVICE
            result.resident_unknown_bytes = length
        else:
            assert fd == 120
            params.success_count = 1
            params.failure_count = 0
            params.rmStatus = 0
            result.status = 0
            result.resident_gpu_bytes = length
            result.resident_cpu_bytes = length
            result.resident_any_bytes = length
            result.resident_unknown_bytes = 0
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_promote_ranges_to_read_duplicate_fast(ranges)

    assert result["ok"] is True
    assert result["selected_fd_source"] == 20
    assert result["fd_candidate_index"] == 2
    assert closed == [110, 120]


def test_read_duplicate_fast_promotion_accepts_idempotent_metadata_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranges = ((0x100000, 0x10000),)
    _patch_driver_fds(monkeypatch, existing_sources=[10])

    def ioctl(_fd: int, _request: int, params: object) -> int:
        assert isinstance(params, uvm_module._PhaseFBatchExecutorParams)
        params.success_count = 1
        params.failure_count = 0
        params.rmStatus = 0
        result = params.results[0]
        result.status = 0
        # No data copy is needed when CPU/GPU residency already exists. The
        # action still repairs read_duplicated_pages and the GPU READ PTE.
        result.resident_gpu_bytes = 0
        result.resident_cpu_bytes = 0
        result.resident_any_bytes = int(params.ranges[0].length)
        result.resident_unknown_bytes = 0
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_promote_ranges_to_read_duplicate_fast(ranges)

    assert result["ok"] is True
    assert result["success_count"] == 1
    assert result["gpu_read_duplicate_bytes"] == 0
    assert result["cpu_shadow_bytes"] == 0


def test_region_release_accepts_complete_release() -> None:
    params = _release_params()
    params.block_count = 1
    params.attempted_blocks = 1
    params.released_bytes = params.length

    assert uvm_module._region_release_outcome(params, ret=0) == (True, True)


def test_region_release_accepts_zero_count_cpu_idempotency() -> None:
    params = _release_params()

    assert uvm_module._region_release_outcome(params, ret=0) == (True, False)


def test_region_release_rejects_zero_count_without_valid_region() -> None:
    params = uvm_module._ExpertOffloadReleaseParams()

    assert uvm_module._region_release_outcome(params, ret=0) == (False, False)


def test_remote_release_requires_complete_verified_mapping() -> None:
    params = _release_params(remote_read=True)
    params.block_count = 1
    params.cpu_resident_bytes = params.length
    params.remote_mapped_bytes = params.length - 1

    assert uvm_module._region_release_outcome(params, ret=0) == (False, False)

    params.remote_mapped_bytes = params.length
    assert uvm_module._region_release_outcome(params, ret=0) == (True, False)


def test_new_remote_mapping_counts_as_progress() -> None:
    params = _release_params(remote_read=True)
    params.block_count = 1
    params.cpu_resident_bytes = params.length
    params.remote_mapped_bytes = params.length
    params.newly_remote_mapped_bytes = params.length

    assert uvm_module._region_release_outcome(params, ret=0) == (True, True)


def test_driver_release_optional_audit_preserves_verified_remote_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_driver_fds(monkeypatch)

    def ioctl(_fd: int, _request: int, params: object) -> int:
        params.base = 0x100000
        params.length = 0x80000
        params.block_count = 1
        params.cpu_resident_bytes = params.length
        params.remote_mapped_bytes = params.length
        params.newly_remote_mapped_bytes = params.length
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_release_regions(
        [11], remote_read=True, prefer_existing=False, include_audit=True
    )

    assert result["ok"] is True
    assert result["statuses"] == {11: True}
    assert result["audits"][11]["ok"] is True
    assert result["audits"][11]["length"] == 0x80000
    assert result["audits"][11]["cpu_resident_bytes"] == 0x80000
    assert result["audits"][11]["remote_mapped_bytes"] == 0x80000


def test_driver_release_returns_only_per_region_statuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_driver_fds(monkeypatch)

    def ioctl(_fd: int, _request: int, params: object) -> int:
        params.length = 0x80000
        params.block_count = 1
        if params.region_id == 11:
            params.attempted_blocks = 1
            params.released_bytes = params.length
            return 0
        params.rmStatus = 5
        return -1

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_release_regions([11, 12], prefer_existing=False)

    assert result == {"ok": False, "statuses": {11: True, 12: False}}


def test_existing_fd_without_progress_falls_back_to_new_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _patch_driver_fds(monkeypatch, existing_sources=[5], new_fd=200)
    ioctl_fds: list[int] = []

    def ioctl(fd: int, _request: int, params: object) -> int:
        ioctl_fds.append(fd)
        params.length = 0x80000
        if fd == 105:
            params.rmStatus = 5
            return -1
        params.block_count = 1
        params.attempted_blocks = 1
        params.released_bytes = params.length
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_release_regions([11])

    assert result == {"ok": True, "statuses": {11: True}}
    assert ioctl_fds == [105, 200]
    assert closed == [105, 200]


def test_existing_fd_with_partial_progress_does_not_retry_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_driver_fds(monkeypatch, existing_sources=[5])
    opened_new = False

    def open_new() -> int:
        nonlocal opened_new
        opened_new = True
        return 200

    monkeypatch.setattr(uvm_module, "_open_initialized_uvm_fd", open_new)

    def ioctl(_fd: int, _request: int, params: object) -> int:
        params.length = 0x80000
        if params.region_id == 11:
            params.block_count = 1
            params.attempted_blocks = 1
            params.released_bytes = params.length
            return 0
        params.rmStatus = 5
        return -1

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_release_regions([11, 12])

    assert result == {"ok": False, "statuses": {11: True, 12: False}}
    assert opened_new is False


def _install_registered_transition_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, int, int]:
    device = 0
    plan_id = 17
    generation = 41
    entry = (0x200000, 0x80000, 9, 3, 5, 0, 3)
    monkeypatch.setattr(
        uvm_module, "_uvm_slot_direct_sessions", {device: (123, bytes(16))}
    )
    monkeypatch.setattr(
        uvm_module,
        "_uvm_slot_direct_plans",
        {
            device: {
                plan_id: {
                    "layer_id": 2,
                    "mode": (
                        uvm_module._UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION
                    ),
                    "entries": (entry,),
                }
            }
        },
    )
    monkeypatch.setattr(
        uvm_module, "_uvm_slot_direct_plan_generations", {device: generation}
    )
    monkeypatch.setattr(uvm_module, "_uvm_slot_direct_owners", {device: {}})
    monkeypatch.setattr(
        uvm_module, "_uvm_slot_direct_session_pids", {device: os.getpid()}
    )
    monkeypatch.setattr(uvm_module, "_uvm_slot_direct_poisoned", {})
    return device, plan_id, generation


def _complete_registered_transition(params: object) -> None:
    params.mode = uvm_module._UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION
    params.count = 1
    params.new_generation = 42
    params.success_count = 1
    params.failure_count = 0
    params.conflict_count = 0
    params.requested_bytes = 0x80000
    params.accepted_bytes = 0x80000
    params.remap_count = 1
    params.tlb_invalidate_count = 1
    params.copy_push_count = 1
    params.owner_cutover = 1
    params.poisoned = 0


def _enable_registered_cuda_fence(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device: int,
    fd: int = 123,
) -> uvm_module._UvmSlotCudaFence:
    fence = uvm_module._UvmSlotCudaFence(
        fd=fd,
        base=0x70000000,
        device_base=0x90000000,
        length=4096,
        transport="raw_kernel",
        kernel_library_path="/tmp/fence.so",
    )
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_CUDA_FENCE_ENV, "1")
    monkeypatch.setattr(
        uvm_module,
        "_uvm_slot_direct_cuda_fences",
        {int(device): fence},
    )
    monkeypatch.setattr(
        uvm_module,
        "_uvm_slot_cuda_fence_current_stream",
        lambda requested_device: 0xABC000 + int(requested_device),
    )
    monkeypatch.setattr(
        uvm_module.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    return fence


def test_registered_plan_execute_accepts_only_complete_plan_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device, plan_id, generation = _install_registered_transition_plan(monkeypatch)

    def ioctl(_fd: int, _request: int, params: object) -> int:
        _complete_registered_transition(params)
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module._driver_slot_execute_plan_impl(
        plan_id, 2, generation, device=device
    )

    assert result["ok"] is True
    assert result["success_contract_ok"] is True
    assert result["failure_contract_ok"] is False
    assert result["contract_violation"] is False
    assert result["expected_count"] == 1
    assert result["expected_requested_bytes"] == 0x80000
    assert result["expected_copy_push_count_min"] == 1
    assert result["expected_copy_push_count_max"] == 128
    assert result["new_generation_trusted"] is True


def test_registered_plan_bank_commit_selects_plan_without_owner_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device, plan_id, generation = _install_registered_transition_plan(monkeypatch)
    plan = uvm_module._uvm_slot_direct_plans[device][plan_id]
    entries = plan["entries"]
    plan["execute_contract"] = uvm_module._build_registered_plan_execute_contract(
        entries,
        int(plan["mode"]),
    )
    original_owners = {(99, 100): 101}
    monkeypatch.setattr(
        uvm_module, "_uvm_slot_direct_owners", {device: dict(original_owners)}
    )
    monkeypatch.setattr(uvm_module, "_uvm_slot_direct_active_plan_ids", {})
    monkeypatch.setenv(uvm_module._UVM_SLOT_DIRECT_BANK_LEVEL_COMMIT_ENV, "1")

    def ioctl(_fd: int, _request: int, params: object) -> int:
        _complete_registered_transition(params)
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module._driver_slot_execute_plan_impl(
        plan_id, 2, generation, device=device
    )

    assert result["ok"] is True
    assert result["entry_statuses"] == []
    assert result["entry_statuses_elided"] is True
    assert uvm_module._uvm_slot_direct_active_plan_ids == {device: plan_id}
    assert uvm_module._uvm_slot_direct_owners[device] == original_owners


def test_registered_plan_fenced_execute_orders_stream_write_ioctl_and_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device, plan_id, generation = _install_registered_transition_plan(monkeypatch)
    fence = _enable_registered_cuda_fence(monkeypatch, device=device)
    events: list[tuple[object, ...]] = []

    def write_compute(*, stream: int, address: int, ticket: int) -> None:
        events.append(("write", stream, address, ticket))

    def ioctl(fd: int, request: int, params: object) -> int:
        events.append(("ioctl", fd, request))
        assert request == uvm_module._UVM_MOE_SLOT_PLAN_EXECUTE_FENCED_IOCTL
        assert isinstance(params, uvm_module._MoeSlotPlanExecuteFencedParams)
        assert int(params.fence_flags) == uvm_module._UVM_MOE_SLOT_PLAN_FENCE_FLAGS_ALL
        assert int(params.compute_semaphore_address) == fence.compute_address
        assert int(params.completion_semaphore_address) == fence.completion_address
        assert int(params.compute_payload) == 1
        assert int(params.completion_payload) == 1
        assert int(params.reserved) == 0
        _complete_registered_transition(params)
        params.fence_accepted = 1
        params.completion_signaled = 1
        return 0

    def wait_completion(*, stream: int, address: int, ticket: int) -> None:
        events.append(("wait", stream, address, ticket))

    monkeypatch.setattr(uvm_module, "_uvm_slot_cuda_fence_write_compute", write_compute)
    monkeypatch.setattr(
        uvm_module, "_uvm_slot_cuda_fence_wait_completion", wait_completion
    )
    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module._driver_slot_execute_plan_impl(
        plan_id, 2, generation, device=device
    )

    stream = 0xABC000 + device
    assert result["ok"] is True
    assert result["cuda_fence_used"] is True
    assert result["cuda_fence_ticket"] == 1
    assert result["cuda_fence_accepted"] is True
    assert result["cuda_completion_signaled"] is True
    assert result["cuda_stream_wait_enqueued"] is True
    assert events == [
        ("write", stream, fence.compute_device_address, 1),
        ("ioctl", 123, uvm_module._UVM_MOE_SLOT_PLAN_EXECUTE_FENCED_IOCTL),
        ("wait", stream, fence.completion_device_address, 1),
    ]


def test_registered_plan_fenced_execute_rejects_cuda_graph_capture_before_ticket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device, plan_id, generation = _install_registered_transition_plan(monkeypatch)
    fence = _enable_registered_cuda_fence(monkeypatch, device=device)
    monkeypatch.setattr(
        uvm_module.torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )
    monkeypatch.setattr(
        uvm_module,
        "_uvm_slot_cuda_fence_write_compute",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("capture rejection must precede the stream signal")
        ),
    )
    monkeypatch.setattr(
        uvm_module,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("capture rejection must precede the ioctl")
        ),
    )

    result = uvm_module._driver_slot_execute_plan_impl(
        plan_id, 2, generation, device=device
    )

    assert result["ok"] is False
    assert result["reason"] == "cuda_graph_capture_unsupported"
    assert result["poisoned"] is False
    assert fence.next_ticket == 1


def test_registered_plan_fenced_execute_never_waits_on_unsignaled_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device, plan_id, generation = _install_registered_transition_plan(monkeypatch)
    _enable_registered_cuda_fence(monkeypatch, device=device)
    wait_calls: list[int] = []
    abort_calls: list[int] = []
    synchronize_calls: list[int] = []

    monkeypatch.setattr(
        uvm_module,
        "_uvm_slot_cuda_fence_write_compute",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        uvm_module,
        "_uvm_slot_cuda_fence_wait_completion",
        lambda **kwargs: wait_calls.append(int(kwargs["ticket"])),
    )

    def ioctl(_fd: int, request: int, params: object) -> int:
        assert request == uvm_module._UVM_MOE_SLOT_PLAN_EXECUTE_FENCED_IOCTL
        _complete_registered_transition(params)
        params.fence_accepted = 1
        params.completion_signaled = 0
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    monkeypatch.setattr(
        uvm_module.torch.cuda,
        "synchronize",
        lambda requested_device: synchronize_calls.append(int(requested_device)),
    )
    monkeypatch.setattr(
        uvm_module,
        "_driver_slot_abort_session_impl",
        lambda *, device: abort_calls.append(int(device)) or {"ok": True},
    )

    result = uvm_module._driver_slot_execute_plan_impl(
        plan_id, 2, generation, device=device
    )

    assert result["ok"] is False
    assert result["reason"] == "plan_execute_contract_mismatch"
    assert result["cuda_fence_accepted"] is True
    assert result["cuda_completion_signaled"] is False
    assert wait_calls == []
    assert synchronize_calls == [device]
    assert abort_calls == [device]


def test_registered_plan_fenced_execute_compute_write_failure_aborts_before_ioctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device, plan_id, generation = _install_registered_transition_plan(monkeypatch)
    _enable_registered_cuda_fence(monkeypatch, device=device)
    abort_calls: list[int] = []
    synchronize_calls: list[int] = []
    monkeypatch.setattr(
        uvm_module,
        "_uvm_slot_cuda_fence_write_compute",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("write failed")),
    )
    monkeypatch.setattr(
        uvm_module,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("ioctl must not run after compute-write failure")
        ),
    )
    monkeypatch.setattr(
        uvm_module,
        "_driver_slot_abort_session_impl",
        lambda *, device: abort_calls.append(int(device)) or {"ok": True},
    )
    monkeypatch.setattr(
        uvm_module.torch.cuda,
        "synchronize",
        lambda requested_device: synchronize_calls.append(int(requested_device)),
    )

    result = uvm_module._driver_slot_execute_plan_impl(
        plan_id, 2, generation, device=device
    )

    assert result["ok"] is False
    assert result["reason"] == "cuda_compute_fence_enqueue_failed"
    assert result["poisoned"] is True
    assert result["deferred_to_process_teardown"] is False
    assert synchronize_calls == [device]
    assert abort_calls == [device]


def test_registered_plan_fence_failure_defers_abort_when_sync_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device, plan_id, generation = _install_registered_transition_plan(monkeypatch)
    _enable_registered_cuda_fence(monkeypatch, device=device)
    abort_calls: list[int] = []
    monkeypatch.setattr(
        uvm_module,
        "_uvm_slot_cuda_fence_write_compute",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("write failed")),
    )
    monkeypatch.setattr(
        uvm_module.torch.cuda,
        "synchronize",
        lambda _device: (_ for _ in ()).throw(RuntimeError("sync failed")),
    )
    monkeypatch.setattr(
        uvm_module,
        "_driver_slot_abort_session_impl",
        lambda *, device: abort_calls.append(int(device)) or {"ok": True},
    )

    result = uvm_module._driver_slot_execute_plan_impl(
        plan_id, 2, generation, device=device
    )

    assert result["ok"] is False
    assert result["reason"] == "cuda_compute_fence_enqueue_failed"
    assert result["abort"] is None
    assert result["session_aborted"] is False
    assert result["deferred_to_process_teardown"] is True
    assert "sync failed" in str(result["failure_boundary_error"])
    assert abort_calls == []


def test_registered_plan_fenced_execute_completion_wait_failure_aborts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device, plan_id, generation = _install_registered_transition_plan(monkeypatch)
    _enable_registered_cuda_fence(monkeypatch, device=device)
    abort_calls: list[int] = []
    synchronize_calls: list[int] = []
    monkeypatch.setattr(
        uvm_module,
        "_uvm_slot_cuda_fence_write_compute",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        uvm_module,
        "_uvm_slot_cuda_fence_wait_completion",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("wait failed")),
    )

    def ioctl(_fd: int, request: int, params: object) -> int:
        assert request == uvm_module._UVM_MOE_SLOT_PLAN_EXECUTE_FENCED_IOCTL
        _complete_registered_transition(params)
        params.fence_accepted = 1
        params.completion_signaled = 1
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    monkeypatch.setattr(
        uvm_module,
        "_driver_slot_abort_session_impl",
        lambda *, device: abort_calls.append(int(device)) or {"ok": True},
    )
    monkeypatch.setattr(
        uvm_module.torch.cuda,
        "synchronize",
        lambda requested_device: synchronize_calls.append(int(requested_device)),
    )

    result = uvm_module._driver_slot_execute_plan_impl(
        plan_id, 2, generation, device=device
    )

    assert result["ok"] is False
    assert result["reason"] == "cuda_completion_wait_enqueue_failed"
    assert result["poisoned"] is True
    assert uvm_module._uvm_slot_direct_plan_generations[device] == generation
    assert synchronize_calls == [device]
    assert abort_calls == [device]


def test_cuda_fence_pool_cleanup_orders_unregister_unmap_and_uvm_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = 0
    fd = 123
    fence = uvm_module._UvmSlotCudaFence(
        fd=fd,
        base=0x70000000,
        device_base=0x90000000,
        length=4096,
    )
    monkeypatch.setattr(
        uvm_module, "_uvm_slot_direct_cuda_fences", {device: fence}
    )
    events: list[tuple[str, int]] = []

    def unregister(address: ctypes.c_void_p) -> int:
        events.append(("unregister", int(address.value or 0)))
        return 0

    monkeypatch.setattr(
        uvm_module,
        "_load_cuda_host_mapping_ops",
        lambda: (object(), object(), unregister),
    )

    def munmap(address: ctypes.c_void_p, length: ctypes.c_size_t) -> int:
        events.append(("munmap", int(address.value or 0)))
        assert int(length.value) == fence.length
        return 0

    monkeypatch.setattr(uvm_module._LIBC, "munmap", munmap)

    def ioctl(request_fd: int, request: int, params: object) -> int:
        assert request_fd == fd
        assert request == uvm_module._UVM_FREE_IOCTL
        events.append(("free", int(params.base)))
        params.rmStatus = 0
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    uvm_module._uvm_slot_cuda_fence_destroy_locked(device=device, fd=fd)

    assert events == [
        ("unregister", fence.base),
        ("munmap", fence.base),
        ("free", fence.base),
    ]
    assert fence.device_base == 0
    assert device not in uvm_module._uvm_slot_direct_cuda_fences


def test_raw_cuda_fence_pool_cleanup_skips_host_unregister(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = 0
    fd = 123
    fence = uvm_module._UvmSlotCudaFence(
        fd=fd,
        base=0x70000000,
        device_base=0x70000000,
        length=4096,
        transport="raw_kernel",
        cuda_host_registered=False,
        kernel_library_path="/tmp/fence.so",
    )
    monkeypatch.setattr(
        uvm_module, "_uvm_slot_direct_cuda_fences", {device: fence}
    )
    monkeypatch.setattr(
        uvm_module,
        "_load_cuda_host_mapping_ops",
        lambda: (_ for _ in ()).throw(
            AssertionError("raw fence must not unregister host memory")
        ),
    )
    events: list[tuple[str, int]] = []
    monkeypatch.setattr(
        uvm_module._LIBC,
        "munmap",
        lambda address, _length: events.append(
            ("munmap", int(address.value or 0))
        )
        or 0,
    )

    def ioctl(_fd: int, request: int, params: object) -> int:
        assert request == uvm_module._UVM_FREE_IOCTL
        events.append(("free", int(params.base)))
        params.rmStatus = 0
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    uvm_module._uvm_slot_cuda_fence_destroy_locked(device=device, fd=fd)

    assert events == [("munmap", fence.base), ("free", fence.base)]


def test_raw_cuda_fence_timeout_close_is_process_teardown_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = 0
    monkeypatch.setattr(
        uvm_module, "_uvm_slot_direct_sessions", {device: (123, b"uuid")}
    )
    monkeypatch.setattr(
        uvm_module, "_uvm_slot_direct_session_pids", {device: os.getpid()}
    )
    monkeypatch.setattr(
        uvm_module, "_uvm_slot_direct_cuda_fence_cleanup_deferred", {device}
    )
    monkeypatch.setattr(
        uvm_module.torch.cuda,
        "synchronize",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("timeout cleanup must not synchronize")
        ),
    )

    result = uvm_module.close_uvm_slot_direct_session(device=device)

    assert result["ok"] is False
    assert result["reason"] == "cuda_fence_cleanup_process_teardown_only"
    assert result["deferred_to_process_teardown"] is True
    assert uvm_module._uvm_slot_direct_sessions[device][0] == 123


def test_cuda_fence_pool_cleanup_failure_keeps_session_resource_tracked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = 0
    fd = 123
    fence = uvm_module._UvmSlotCudaFence(
        fd=fd,
        base=0x70000000,
        device_base=0x90000000,
        length=4096,
    )
    monkeypatch.setattr(
        uvm_module, "_uvm_slot_direct_cuda_fences", {device: fence}
    )
    monkeypatch.setattr(
        uvm_module,
        "_load_cuda_host_mapping_ops",
        lambda: (object(), object(), lambda _address: 7),
    )
    monkeypatch.setattr(
        uvm_module._LIBC,
        "munmap",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("failed unregister must not unmap the pool")
        ),
    )
    monkeypatch.setattr(
        uvm_module,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("failed unregister must not free the pool")
        ),
    )

    with pytest.raises(RuntimeError, match="cuMemHostUnregister"):
        uvm_module._uvm_slot_cuda_fence_destroy_locked(device=device, fd=fd)

    assert uvm_module._uvm_slot_direct_cuda_fences[device] is fence
    assert fence.device_base == 0x90000000


def test_registered_plan_failure_keeps_owner_and_generation_then_aborts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device, plan_id, generation = _install_registered_transition_plan(monkeypatch)
    old_owners = {(9, 5): 99}
    uvm_module._uvm_slot_direct_owners[device] = dict(old_owners)
    abort_calls: list[int] = []

    def ioctl(_fd: int, _request: int, params: object) -> int:
        params.mode = (
            uvm_module._UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION
        )
        params.count = 1
        params.new_generation = generation
        params.requested_bytes = 0x80000
        params.failure_count = 1
        # Pre-commit CE work is allowed, but no PTE/owner publication is.
        params.copy_push_count = 1
        params.remap_count = 0
        params.owner_cutover = 0
        params.poisoned = 1
        params.rmStatus = 5
        return 0

    def abort(*, device: int = 0) -> dict[str, object]:
        abort_calls.append(device)
        return {"ok": True, "reason": "synthetic_abort"}

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    monkeypatch.setattr(uvm_module, "_driver_slot_abort_session_impl", abort)
    result = uvm_module._driver_slot_execute_plan_impl(
        plan_id, 2, generation, device=device
    )

    assert result["ok"] is False
    assert result["reason"] == "plan_execute_ioctl_failed"
    assert result["failure_contract_ok"] is True
    assert result["contract_violation"] is False
    assert result["kernel_poisoned"] is True
    assert result["poisoned"] is True
    assert result["publication_observed"] is False
    assert result["precommit_work_observed"] is True
    assert result["owner_cutover"] is False
    assert result["owner_cutover_trusted"] is True
    assert result["owner_state_indeterminate"] is False
    assert result["new_generation"] == generation
    assert result["new_generation_trusted"] is True
    assert uvm_module._uvm_slot_direct_plan_generations[device] == generation
    assert uvm_module._uvm_slot_direct_owners[device] == old_owners
    assert device in uvm_module._uvm_slot_direct_poisoned
    assert result["session_aborted"] is True
    assert abort_calls == [device]


def test_registered_plan_preflight_rejection_is_trusted_then_aborted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device, plan_id, generation = _install_registered_transition_plan(monkeypatch)
    old_owners = {(9, 5): 99}
    uvm_module._uvm_slot_direct_owners[device] = dict(old_owners)
    abort_calls: list[int] = []

    def ioctl(_fd: int, _request: int, params: object) -> int:
        params.mode = (
            uvm_module._UVM_MOE_SLOT_REMAP_MODE_SLOT_DIRECT_LAYER_TRANSITION
        )
        params.count = 1
        params.new_generation = generation
        params.requested_bytes = 0x80000
        params.failure_count = 1
        params.remap_count = 0
        params.owner_cutover = 0
        params.poisoned = 0
        params.rmStatus = 5
        return 0

    def abort(*, device: int = 0) -> dict[str, object]:
        abort_calls.append(device)
        return {"ok": True, "reason": "synthetic_abort"}

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    monkeypatch.setattr(uvm_module, "_driver_slot_abort_session_impl", abort)
    result = uvm_module._driver_slot_execute_plan_impl(
        plan_id, 2, generation, device=device
    )

    assert result["ok"] is False
    assert result["reason"] == "plan_execute_preflight_rejected"
    assert result["preflight_rejection_ok"] is True
    assert result["failure_contract_ok"] is False
    assert result["contract_violation"] is False
    assert result["kernel_poisoned"] is False
    assert result["poisoned"] is True
    assert result["side_effect_observed"] is False
    assert result["owner_cutover_trusted"] is True
    assert result["owner_state_indeterminate"] is False
    assert result["new_generation_trusted"] is True
    assert uvm_module._uvm_slot_direct_owners[device] == old_owners
    assert result["session_aborted"] is True
    assert abort_calls == [device]


def test_registered_plan_execute_accepts_capacity_split_shared_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device, plan_id, generation = _install_registered_transition_plan(monkeypatch)
    plans = uvm_module._uvm_slot_direct_plans[device]
    plans[plan_id]["entries"] = (
        (0x200000, 0x80000, 9, 3, 5, 0, 3),
        (0x280000, 0x80000, 9, 4, 6, 0, 3),
    )

    def ioctl(_fd: int, _request: int, params: object) -> int:
        _complete_registered_transition(params)
        params.count = 2
        params.success_count = 2
        params.requested_bytes = 0x100000
        params.accepted_bytes = 0x100000
        params.remap_count = 2
        # This synthetic mode-20 plan does not set the 64 KiB grouping flag,
        # so its two independently mapped entries each publish a TLB update.
        params.tlb_invalidate_count = 2
        params.copy_push_count = 2
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module._driver_slot_execute_plan_impl(
        plan_id, 2, generation, device=device
    )

    assert result["ok"] is True
    assert result["copy_push_count"] == 2
    assert result["expected_copy_push_count_max"] == 256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", 21),
        ("count", 0),
        ("success_count", 0),
        ("conflict_count", 1),
        ("requested_bytes", 0x40000),
        ("accepted_bytes", 0x40000),
        ("remap_count", 0),
        ("tlb_invalidate_count", 0),
        ("copy_push_count", 0),
        ("copy_push_count", 129),
        ("owner_cutover", 0),
        ("new_generation", 43),
    ],
)
def test_registered_plan_execute_poisoned_on_short_kernel_contract(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: int,
) -> None:
    device, plan_id, generation = _install_registered_transition_plan(monkeypatch)

    def ioctl(_fd: int, _request: int, params: object) -> int:
        _complete_registered_transition(params)
        setattr(params, field, value)
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module._driver_slot_execute_plan_impl(
        plan_id, 2, generation, device=device
    )

    assert result["ok"] is False
    assert result["reason"] == "plan_execute_contract_mismatch"
    assert result["contract_violation"] is True
    assert result["poisoned"] is True


def test_registered_plan_execute_poisoned_on_failed_ioctl_with_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device, plan_id, generation = _install_registered_transition_plan(monkeypatch)

    def ioctl(_fd: int, _request: int, params: object) -> int:
        _complete_registered_transition(params)
        params.rmStatus = 5
        return -1

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module._driver_slot_execute_plan_impl(
        plan_id, 2, generation, device=device
    )

    assert result["ok"] is False
    assert result["side_effect_observed"] is True
    assert result["contract_violation"] is True
    assert result["poisoned"] is True


def test_malformed_plan_generation_requires_successful_abort_before_fd_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device, plan_id, generation = _install_registered_transition_plan(monkeypatch)

    def ioctl(_fd: int, _request: int, params: object) -> int:
        _complete_registered_transition(params)
        params.new_generation = generation + 2
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module._driver_slot_execute_plan_impl(
        plan_id, 2, generation, device=device
    )

    assert result["ok"] is False
    assert result["new_generation_trusted"] is False
    assert result["generation_cleanup_requires_fd_close"] is True
    assert device not in uvm_module._uvm_slot_direct_plan_generations

    closed: list[int] = []
    policy_cleanup: list[int] = []
    monkeypatch.setattr(uvm_module.os, "close", lambda fd: closed.append(int(fd)))
    monkeypatch.setattr(
        uvm_module,
        "_run_uvm_slot_direct_policy_cleanup",
        lambda cleanup_device: policy_cleanup.append(int(cleanup_device)),
    )
    cleanup = uvm_module.close_uvm_slot_direct_session(device=device)

    assert cleanup["ok"] is False
    assert cleanup["reason"] == "slot_session_abort_failed"
    assert cleanup["closed"] is False
    assert cleanup["plan_destroy"] == []
    assert closed == []
    assert policy_cleanup == []

    def complete_abort(_fd: int, request: int, params: object) -> int:
        assert request == uvm_module._UVM_MOE_SLOT_PLAN_ABORT_SESSION_IOCTL
        params.generation_before = generation
        params.released_bytes = 0x80000
        params.remaining_bytes = 0
        params.destroyed_plan_count = 1
        params.released_slot_count = 1
        params.remaining_slot_count = 0
        params.had_registered_session = 1
        params.session_cleared = 1
        params.poisoned = 0
        params.rmStatus = 0
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", complete_abort)
    cleanup = uvm_module.close_uvm_slot_direct_session(device=device)

    assert cleanup["ok"] is True
    assert cleanup["closed"] is True
    assert cleanup["abort"]["ok"] is True
    assert cleanup["forced_fd_cleanup"] is False
    assert cleanup["plan_destroy"] == []
    assert closed == [123]
    assert policy_cleanup == [device]


@pytest.mark.parametrize("failure_stage", ["uuid", "open", "ioctl"])
def test_driver_release_exceptions_fail_every_requested_region(
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    _patch_driver_fds(monkeypatch)
    if failure_stage == "uuid":
        monkeypatch.setattr(
            uvm_module,
            "_phase_f_gpu_uuid",
            lambda _device: (_ for _ in ()).throw(RuntimeError("uuid")),
        )
    elif failure_stage == "open":
        monkeypatch.setattr(
            uvm_module,
            "_open_initialized_uvm_fd",
            lambda: (_ for _ in ()).throw(OSError("open")),
        )
    else:
        monkeypatch.setattr(
            uvm_module,
            "_libc_ioctl",
            lambda *_args: (_ for _ in ()).throw(OSError("ioctl")),
        )

    result = uvm_module.driver_release_regions([11, 12], prefer_existing=False)

    assert result == {"ok": False, "statuses": {11: False, 12: False}}


def test_empty_release_is_idempotent_and_api_has_no_partial_limit() -> None:
    assert uvm_module.driver_release_regions([]) == {"ok": True, "statuses": {}}
    parameters = inspect.signature(uvm_module.driver_release_regions).parameters
    assert "max_blocks" not in parameters
    assert "return_entries" not in parameters
    assert "max_blocks" not in dict(uvm_module._ExpertOffloadReleaseParams._fields_)
    assert uvm_module._ExpertOffloadReleaseParams.base.offset == 32
    assert uvm_module._ExpertOffloadReleaseParams.rmStatus.offset == 140
