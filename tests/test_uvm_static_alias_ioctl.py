from __future__ import annotations

import ctypes
import errno
import os
from collections.abc import Callable

import pytest

from vllm.device_allocator import uvm_static_alias as static_alias

UUID_BYTES = bytes(range(16))
PAGE_BYTES = 4096


@pytest.fixture(autouse=True)
def reset_static_alias_state(monkeypatch: pytest.MonkeyPatch):
    static_alias._static_alias_sessions.clear()
    static_alias._uvm._uvm_slot_direct_owners.clear()
    monkeypatch.setattr(
        static_alias.torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )
    yield
    static_alias._static_alias_sessions.clear()
    static_alias._uvm._uvm_slot_direct_owners.clear()


def _uuid() -> static_alias._StaticAliasUuid:
    value = static_alias._StaticAliasUuid()
    value.uuid[:] = UUID_BYTES
    return value


def _install_fd_candidates(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_fds: tuple[int, ...] = (10,),
    duplicate_fds: tuple[int, ...] = (110,),
) -> list[int]:
    assert len(source_fds) == len(duplicate_fds)
    mapping = dict(zip(source_fds, duplicate_fds))
    closed: list[int] = []
    monkeypatch.setattr(static_alias, "_phase_f_gpu_uuid", lambda _device: _uuid())
    monkeypatch.setattr(
        static_alias,
        "_iter_existing_uvm_fd_sources",
        lambda: list(source_fds),
    )
    monkeypatch.setattr(
        static_alias,
        "_initialize_duplicated_uvm_fd",
        lambda source_fd: mapping[source_fd],
    )
    monkeypatch.setattr(static_alias.os, "close", lambda fd: closed.append(int(fd)))
    return closed


def _entries() -> list[tuple[int, int, int, int, int, int, int]]:
    return [
        (
            0,
            11,
            7,
            static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW,
            0x20000000,
            PAGE_BYTES,
            0,
        ),
        (0, 11, 7, 0, 0x30000000, PAGE_BYTES, 0),
    ]


def _complete_map(
    params: static_alias._StaticAliasMapParams,
    *,
    session_id: int = 41,
    prior_alias_bytes: int = 0,
    backing_bytes: int = PAGE_BYTES,
) -> None:
    requested = sum(int(params.entries[index].bytes) for index in range(params.count))
    params.session_id = session_id
    params.requested_bytes = requested
    params.mapped_bytes = requested
    params.total_alias_bytes = prior_alias_bytes + requested
    params.backing_bytes = backing_bytes
    params.success_count = params.count
    params.failure_count = 0
    params.alias_count = params.count
    params.tlb_invalidate_count = params.count
    params.poisoned = 0
    for index in range(params.count):
        params.entries[index].status = 0
    params.rmStatus = 0


def _create_session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ioctl: Callable[[int, int, object], int] | None = None,
) -> dict[str, object]:
    _install_fd_candidates(monkeypatch)

    if ioctl is None:

        def ioctl(_fd: int, _request: int, raw: object) -> int:
            assert isinstance(raw, static_alias._StaticAliasMapParams)
            _complete_map(raw)
            return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    return static_alias.driver_static_alias_map(_entries())


def test_static_alias_ioctl_abi_matches_kernel_header() -> None:
    assert static_alias.UVM_MOE_STATIC_ALIAS_ABI_VERSION == 1
    assert static_alias.UVM_MOE_STATIC_ALIAS_MAX_ENTRIES == 4608
    assert static_alias._UVM_MOE_STATIC_ALIAS_MAP_IOCTL == 97
    assert static_alias._UVM_MOE_STATIC_ALIAS_SEAL_IOCTL == 98
    assert static_alias._UVM_MOE_STATIC_ALIAS_DESTROY_IOCTL == 99
    assert static_alias._UVM_MOE_STATIC_ALIAS_PARK_IOCTL == 101
    assert static_alias._UVM_MOE_STATIC_ALIAS_RESUME_IOCTL == 102
    assert static_alias._UVM_MOE_STATIC_ALIAS_REBIND_IOCTL == 103
    assert static_alias._UVM_MOE_STATIC_ALIAS_QUIESCED_REMOVE_IOCTL == 104
    assert static_alias.UVM_MOE_STATIC_ALIAS_MAP_FLAG_QUIESCED_APPEND == 1
    assert static_alias.UVM_MOE_STATIC_ALIAS_MAP_FLAG_DEFER_DUAL_REFRESH == 2
    assert static_alias.UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_DEFER_CPU_RESTORE == 2
    assert static_alias.UVM_MOE_STATIC_ALIAS_REBIND_FLAG_RESIZE == 8

    assert ctypes.sizeof(static_alias._StaticAliasEntry) == 48
    assert static_alias._StaticAliasEntry.va_base.offset == 16
    assert static_alias._StaticAliasEntry.status.offset == 40

    assert ctypes.sizeof(static_alias._StaticAliasMapParams) == 221288
    assert static_alias._StaticAliasMapParams.session_id.offset == 16
    assert static_alias._StaticAliasMapParams.requested_bytes.offset == 40
    assert static_alias._StaticAliasMapParams.count.offset == 72
    assert static_alias._StaticAliasMapParams.entries.offset == 96
    assert static_alias._StaticAliasMapParams.rmStatus.offset == 221280

    assert ctypes.sizeof(static_alias._StaticAliasSealParams) == 72
    assert static_alias._StaticAliasSealParams.expected_alias_count.offset == 48
    assert static_alias._StaticAliasSealParams.rmStatus.offset == 64

    assert ctypes.sizeof(static_alias._StaticAliasDestroyParams) == 80
    assert static_alias._StaticAliasDestroyParams.released_alias_count.offset == 56
    assert static_alias._StaticAliasDestroyParams.rmStatus.offset == 72

    assert ctypes.sizeof(static_alias._StaticAliasTransitionParams) == 152
    assert static_alias._StaticAliasTransitionParams.reserved.offset == 12
    assert static_alias._StaticAliasTransitionParams.session_id.offset == 16
    assert (
        static_alias._StaticAliasTransitionParams.processed_alias_count.offset == 48
    )
    assert static_alias._StaticAliasTransitionParams.rmStatus.offset == 76
    assert (
        static_alias._StaticAliasTransitionParams.dual_view_full_coverage.offset
        == 144
    )
    assert static_alias._StaticAliasTransitionParams.dual_view_block_count.offset == 148

    assert ctypes.sizeof(static_alias._StaticAliasRebindParams) == 221288
    assert static_alias._StaticAliasRebindParams.session_id.offset == 16
    assert static_alias._StaticAliasRebindParams.rebind_id.offset == 24
    assert static_alias._StaticAliasRebindParams.count.offset == 64
    assert static_alias._StaticAliasRebindParams.entries.offset == 96
    assert static_alias._StaticAliasRebindParams.rmStatus.offset == 221280

    assert ctypes.sizeof(static_alias._StaticAliasQuiescedRemoveParams) == 221288
    assert static_alias._StaticAliasQuiescedRemoveParams.session_id.offset == 16
    assert static_alias._StaticAliasQuiescedRemoveParams.count.offset == 72
    assert static_alias._StaticAliasQuiescedRemoveParams.entries.offset == 96
    assert static_alias._StaticAliasQuiescedRemoveParams.rmStatus.offset == 221280


def _parked_rebind_session() -> static_alias._StaticAliasSession:
    upload = (
        0,
        11,
        7,
        static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW,
        0x20000000,
        PAGE_BYTES,
        0,
    )
    consumer = (0, 11, 7, 0, 0x30000000, PAGE_BYTES, 0)
    return static_alias._StaticAliasSession(
        fd=110,
        uuid_bytes=UUID_BYTES,
        session_id=41,
        owner_pid=os.getpid(),
        sealed=True,
        parked=True,
        alias_count=2,
        alias_bytes=2 * PAGE_BYTES,
        backing_bytes=PAGE_BYTES,
        target_ranges=[
            (upload[4], upload[4] + upload[5]),
            (consumer[4], consumer[4] + consumer[5]),
        ],
        backing_identities={(0, 11, 7)},
        alias_entries=[upload, consumer],
    )


def _sealed_two_identity_session() -> static_alias._StaticAliasSession:
    records = [
        (
            0,
            11,
            7,
            static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW,
            0x20000000,
            PAGE_BYTES,
            0,
        ),
        (0, 11, 7, 0, 0x30000000, PAGE_BYTES, 0),
        (
            0,
            12,
            8,
            static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW,
            0x40000000,
            PAGE_BYTES,
            0,
        ),
        (0, 12, 8, 0, 0x50000000, PAGE_BYTES, 0),
    ]
    return static_alias._StaticAliasSession(
        fd=110,
        uuid_bytes=UUID_BYTES,
        session_id=41,
        owner_pid=os.getpid(),
        sealed=True,
        parked=False,
        alias_count=len(records),
        alias_bytes=len(records) * PAGE_BYTES,
        backing_bytes=2 * PAGE_BYTES,
        target_ranges=[
            (record[4], record[4] + record[5]) for record in records
        ],
        backing_identities={(0, 11, 7), (0, 12, 8)},
        alias_entries=records,
    )


def test_rebind_streams_large_consumer_set_and_updates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _parked_rebind_session()
    static_alias._static_alias_sessions[0] = session
    consumers = [
        (0, 11, 7, 0, 0x40000000 + index * PAGE_BYTES, PAGE_BYTES, 0)
        for index in range(static_alias.UVM_MOE_STATIC_ALIAS_MAX_ENTRIES + 1)
    ]
    calls: list[tuple[int, int, int]] = []
    staged_count = 0
    staged_bytes = 0

    def ioctl(fd: int, request: int, raw: object) -> int:
        nonlocal staged_count, staged_bytes
        assert fd == 110
        assert request == 103
        assert isinstance(raw, static_alias._StaticAliasRebindParams)
        calls.append((int(raw.flags), int(raw.rebind_id), int(raw.count)))
        if raw.flags & static_alias.UVM_MOE_STATIC_ALIAS_REBIND_FLAG_BEGIN:
            assert raw.rebind_id == 0
            raw.rebind_id = 77
        else:
            assert raw.rebind_id == 77
        chunk_bytes = sum(int(raw.entries[index].bytes) for index in range(raw.count))
        staged_count += int(raw.count)
        staged_bytes += chunk_bytes
        raw.accepted_count = raw.count
        raw.upload_alias_count = 1
        raw.backing_bytes = PAGE_BYTES
        for index in range(raw.count):
            raw.entries[index].status = 0
        if raw.flags & static_alias.UVM_MOE_STATIC_ALIAS_REBIND_FLAG_COMMIT:
            assert staged_count == len(consumers)
            raw.committed = 1
            raw.rebind_id = 0
            raw.staged_consumer_alias_count = 0
            raw.staged_consumer_alias_bytes = 0
            raw.total_alias_bytes = PAGE_BYTES + staged_bytes
        else:
            raw.staged_consumer_alias_count = staged_count
            raw.staged_consumer_alias_bytes = staged_bytes
            raw.total_alias_bytes = session.alias_bytes
        raw.poisoned = 0
        raw.rmStatus = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_rebind(41, consumers)

    assert result["ok"] is True
    assert result["batch_count"] == 2
    assert calls == [
        (
            static_alias.UVM_MOE_STATIC_ALIAS_REBIND_FLAG_BEGIN,
            0,
            static_alias.UVM_MOE_STATIC_ALIAS_MAX_ENTRIES,
        ),
        (static_alias.UVM_MOE_STATIC_ALIAS_REBIND_FLAG_COMMIT, 77, 1),
    ]
    assert session.alias_count == static_alias.UVM_MOE_STATIC_ALIAS_MAX_ENTRIES + 2
    assert (
        session.alias_bytes
        == (static_alias.UVM_MOE_STATIC_ALIAS_MAX_ENTRIES + 2) * PAGE_BYTES
    )
    assert session.alias_entries[0][3] & static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW
    assert session.alias_entries[1:] == consumers
    assert session.parked is True
    assert session.poisoned is False


def test_rebind_failed_append_aborts_and_preserves_committed_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _parked_rebind_session()
    static_alias._static_alias_sessions[0] = session
    original_entries = list(session.alias_entries)
    consumers = [
        (0, 11, 7, 0, 0x50000000 + index * PAGE_BYTES, PAGE_BYTES, 0)
        for index in range(static_alias.UVM_MOE_STATIC_ALIAS_MAX_ENTRIES + 1)
    ]
    calls: list[int] = []

    def ioctl(_fd: int, request: int, raw: object) -> int:
        assert request == 103
        assert isinstance(raw, static_alias._StaticAliasRebindParams)
        calls.append(int(raw.flags))
        if raw.flags == static_alias.UVM_MOE_STATIC_ALIAS_REBIND_FLAG_ABORT:
            raw.rebind_id = 0
            raw.rmStatus = 0
            return 0
        if raw.flags & static_alias.UVM_MOE_STATIC_ALIAS_REBIND_FLAG_BEGIN:
            raw.rebind_id = 91
            raw.accepted_count = raw.count
            raw.staged_consumer_alias_count = raw.count
            raw.staged_consumer_alias_bytes = raw.count * PAGE_BYTES
            raw.upload_alias_count = 1
            raw.backing_bytes = PAGE_BYTES
            for index in range(raw.count):
                raw.entries[index].status = 0
            raw.rmStatus = 0
            return 0
        raw.rebind_id = 91
        raw.rmStatus = 5
        return -1

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_rebind(41, consumers)

    assert result["ok"] is False
    assert result["reason"] == "rebind_ioctl_failed"
    assert result["abort_result"]["ok"] is True
    assert calls[-1] == static_alias.UVM_MOE_STATIC_ALIAS_REBIND_FLAG_ABORT
    assert session.alias_entries == original_entries
    assert session.alias_count == 2
    assert session.alias_bytes == 2 * PAGE_BYTES
    assert session.poisoned is False


def test_resize_rebind_reuses_session_and_updates_backing_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _parked_rebind_session()
    static_alias._static_alias_sessions[0] = session
    old_identity = (0, 11, 7)
    added_identity = (0, 12, 8)
    static_alias._uvm._uvm_slot_direct_owners[0] = {
        (old_identity[0], old_identity[2]): old_identity[1],
        (added_identity[0], added_identity[2]): added_identity[1],
    }
    target = [
        session.alias_entries[0],
        (
            *added_identity,
            static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW,
            0x21000000,
            PAGE_BYTES,
            0,
        ),
        (0, 11, 7, 0, 0x31000000, PAGE_BYTES, 0),
        (0, 12, 8, 0, 0x32000000, PAGE_BYTES, 0),
    ]
    calls: list[int] = []

    def ioctl(_fd: int, request: int, raw: object) -> int:
        assert request == 103
        assert isinstance(raw, static_alias._StaticAliasRebindParams)
        calls.append(int(raw.flags))
        assert raw.flags == (
            static_alias.UVM_MOE_STATIC_ALIAS_REBIND_FLAG_BEGIN
            | static_alias.UVM_MOE_STATIC_ALIAS_REBIND_FLAG_COMMIT
            | static_alias.UVM_MOE_STATIC_ALIAS_REBIND_FLAG_RESIZE
        )
        assert raw.expected_consumer_alias_count == len(target)
        assert raw.expected_consumer_alias_bytes == len(target) * PAGE_BYTES
        raw.accepted_count = raw.count
        raw.staged_consumer_alias_count = 0
        raw.staged_consumer_alias_bytes = 0
        raw.upload_alias_count = 2
        raw.backing_bytes = 2 * PAGE_BYTES
        raw.total_alias_bytes = len(target) * PAGE_BYTES
        raw.committed = 1
        raw.rebind_id = 0
        raw.poisoned = 0
        for index in range(raw.count):
            raw.entries[index].status = 0
        raw.rmStatus = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_rebind(
        41,
        target,
        resize=True,
        expected_backing_bytes=2 * PAGE_BYTES,
    )

    assert result["ok"] is True
    assert result["resized"] is True
    assert result["retained_backing_count"] == 1
    assert result["added_backing_count"] == 1
    assert result["removed_backing_count"] == 0
    assert session.session_id == 41
    assert session.alias_entries == target
    assert session.alias_count == 4
    assert session.backing_bytes == 2 * PAGE_BYTES
    assert calls == [11]


def test_map_create_marshals_entries_and_caches_independent_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fd_candidates(monkeypatch)
    calls: list[tuple[int, int]] = []

    def ioctl(fd: int, request: int, params: object) -> int:
        assert isinstance(params, static_alias._StaticAliasMapParams)
        calls.append((fd, request))
        assert params.abi_version == 1
        assert params.struct_size == 221288
        assert params.flags == 0
        assert params.reserved == 0
        assert params.session_id == 0
        assert bytes(params.gpu_uuid.uuid) == UUID_BYTES
        assert params.count == 2
        first = params.entries[0]
        assert (
            first.backing_layer_id,
            first.backing_expert_id,
            first.backing_slot_id,
            first.flags,
            first.va_base,
            first.bytes,
            first.slot_offset,
        ) == (0, 11, 7, 1, 0x20000000, PAGE_BYTES, 0)
        _complete_map(params)
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_map(_entries())

    assert result["ok"] is True
    assert result["session_id"] == 41
    assert result["requested_bytes"] == 2 * PAGE_BYTES
    assert result["mapped_bytes"] == 2 * PAGE_BYTES
    assert result["total_alias_bytes"] == 2 * PAGE_BYTES
    assert result["backing_bytes"] == PAGE_BYTES
    assert result["entry_statuses"] == [0, 0]
    assert calls == [(110, 97)]
    diagnostics = static_alias.get_static_alias_diagnostics()
    assert diagnostics["session_count"] == 1
    assert diagnostics["sessions"][0] == {
        "device": 0,
        "session_id": 41,
        "owner_pid": os.getpid(),
        "sealed": False,
        "parked": False,
        "poisoned": False,
        "alias_count": 2,
        "alias_bytes": 2 * PAGE_BYTES,
        "backing_bytes": PAGE_BYTES,
    }


def test_park_resume_preserves_cached_session_and_backing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = static_alias._StaticAliasSession(
        fd=110,
        uuid_bytes=UUID_BYTES,
        session_id=41,
        owner_pid=os.getpid(),
        sealed=True,
        alias_count=2,
        alias_bytes=2 * PAGE_BYTES,
        backing_bytes=PAGE_BYTES,
    )
    static_alias._static_alias_sessions[0] = session
    calls: list[int] = []

    def ioctl(_fd: int, request: int, raw: object) -> int:
        assert isinstance(raw, static_alias._StaticAliasTransitionParams)
        assert raw.struct_size == 152
        assert raw.reserved == 0
        assert raw.session_id == 41
        assert raw.flags == 0
        calls.append(request)
        raw.processed_alias_count = 1
        raw.consumer_alias_count = 1
        raw.upload_alias_count = 1
        raw.processed_alias_bytes = PAGE_BYTES
        raw.consumer_alias_bytes = PAGE_BYTES
        raw.backing_bytes = PAGE_BYTES
        raw.tlb_invalidate_count = 1
        raw.reserved = 1
        raw.parked = int(request == 101)
        raw.sealed = 1
        raw.poisoned = 0
        raw.rmStatus = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    parked = static_alias.driver_static_alias_park(41)
    assert parked["ok"] is True
    assert parked["parked"] is True
    assert parked["transition_batch_count"] == 1
    assert parked["batch_count_reported"] is True
    assert session.parked is True
    assert session.backing_bytes == PAGE_BYTES

    resumed = static_alias.driver_static_alias_resume(41)
    assert resumed["ok"] is True
    assert resumed["parked"] is False
    assert resumed["transition_batch_count"] == 1
    assert session.parked is False
    assert session.backing_bytes == PAGE_BYTES
    assert calls == [101, 102]


def test_transition_sequential_flag_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = static_alias._StaticAliasSession(
        fd=110,
        uuid_bytes=UUID_BYTES,
        session_id=41,
        owner_pid=os.getpid(),
        sealed=True,
        alias_count=2,
        alias_bytes=2 * PAGE_BYTES,
        backing_bytes=PAGE_BYTES,
    )
    static_alias._static_alias_sessions[0] = session
    flags: list[int] = []

    def ioctl(_fd: int, request: int, raw: object) -> int:
        assert isinstance(raw, static_alias._StaticAliasTransitionParams)
        flags.append(int(raw.flags))
        raw.processed_alias_count = 1
        raw.consumer_alias_count = 1
        raw.upload_alias_count = 1
        raw.processed_alias_bytes = PAGE_BYTES
        raw.consumer_alias_bytes = PAGE_BYTES
        raw.backing_bytes = PAGE_BYTES
        raw.tlb_invalidate_count = 1
        raw.reserved = 1
        raw.parked = int(request == 101)
        raw.sealed = 1
        raw.poisoned = 0
        raw.rmStatus = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    parked = static_alias.driver_static_alias_park(
        41, sequential=True, defer_cpu_restore=True
    )
    assert parked["ok"] is True
    assert parked["transition_mode"] == "sequential"
    assert parked["sequential_requested"] is True

    monkeypatch.setenv(
        "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_TRANSITION_MODE", "sequential"
    )
    resumed = static_alias.driver_static_alias_resume(41)
    assert resumed["ok"] is True
    assert resumed["transition_mode"] == "sequential"
    assert flags == [
        static_alias.UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_FORCE_SEQUENTIAL
        | static_alias.UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_DEFER_CPU_RESTORE,
        static_alias.UVM_MOE_STATIC_ALIAS_TRANSITION_FLAG_FORCE_SEQUENTIAL,
    ]


def test_transition_rejects_impossible_batch_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = static_alias._StaticAliasSession(
        fd=110,
        uuid_bytes=UUID_BYTES,
        session_id=41,
        owner_pid=os.getpid(),
        sealed=True,
        alias_count=2,
        alias_bytes=2 * PAGE_BYTES,
        backing_bytes=PAGE_BYTES,
    )
    static_alias._static_alias_sessions[0] = session

    def ioctl(_fd: int, request: int, raw: object) -> int:
        assert request == 101
        assert isinstance(raw, static_alias._StaticAliasTransitionParams)
        raw.processed_alias_count = 1
        raw.consumer_alias_count = 1
        raw.upload_alias_count = 1
        raw.processed_alias_bytes = PAGE_BYTES
        raw.consumer_alias_bytes = PAGE_BYTES
        raw.backing_bytes = PAGE_BYTES
        raw.tlb_invalidate_count = 1
        raw.reserved = 2
        raw.parked = 1
        raw.sealed = 1
        raw.poisoned = 0
        raw.rmStatus = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_park(41)

    assert result["ok"] is False
    assert result["reason"] == "park_contract_mismatch"
    assert result["batch_count_valid"] is False
    assert session.poisoned is True


def test_map_create_probes_zero_progress_fd_then_keeps_matching_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _install_fd_candidates(
        monkeypatch,
        source_fds=(10, 11),
        duplicate_fds=(110, 111),
    )
    calls: list[int] = []

    def ioctl(fd: int, request: int, params: object) -> int:
        assert request == 97
        assert isinstance(params, static_alias._StaticAliasMapParams)
        calls.append(fd)
        if fd == 110:
            ctypes.set_errno(errno.EINVAL)
            return -1
        _complete_map(params)
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_map(_entries())

    assert result["ok"] is True
    assert result["fd_candidate_attempts"] == 2
    assert result["fd_candidate_ioctl_calls"] == 2
    assert calls == [110, 111]
    assert closed == [110]
    assert static_alias._static_alias_sessions[0].fd == 111


def test_map_create_retries_bare_failed_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _install_fd_candidates(
        monkeypatch,
        source_fds=(10, 11),
        duplicate_fds=(110, 111),
    )
    calls: list[int] = []

    def ioctl(fd: int, request: int, params: object) -> int:
        assert request == 97
        assert isinstance(params, static_alias._StaticAliasMapParams)
        calls.append(fd)
        if fd == 110:
            params.session_id = 7
            params.failure_count = params.count
            params.rmStatus = 0x26
            return 0
        _complete_map(params)
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_map(_entries())

    assert result["ok"] is True
    assert result["fd_candidate_attempts"] == 2
    assert calls == [110, 111]
    assert closed == [110]
    assert static_alias._static_alias_sessions[0].fd == 111


def test_map_create_prefers_slot_preallocation_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _install_fd_candidates(
        monkeypatch,
        source_fds=(10, 11),
        duplicate_fds=(110, 111),
    )
    static_alias._uvm._uvm_slot_direct_sessions[0] = (11, UUID_BYTES)
    calls: list[int] = []

    def ioctl(fd: int, request: int, params: object) -> int:
        assert request == 97
        assert isinstance(params, static_alias._StaticAliasMapParams)
        calls.append(fd)
        _complete_map(params)
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    try:
        result = static_alias.driver_static_alias_map(_entries())
    finally:
        static_alias._uvm._uvm_slot_direct_sessions.pop(0, None)

    assert result["ok"] is True
    assert calls == [111]
    assert closed == []
    assert static_alias._static_alias_sessions[0].fd == 111


def test_map_append_uses_cached_fd_and_updates_total_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _create_session(monkeypatch)
    assert created["ok"] is True
    calls: list[tuple[int, int, int]] = []

    def ioctl(fd: int, request: int, params: object) -> int:
        assert isinstance(params, static_alias._StaticAliasMapParams)
        calls.append((fd, request, int(params.session_id)))
        _complete_map(
            params,
            session_id=41,
            prior_alias_bytes=2 * PAGE_BYTES,
        )
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_map(
        [(0, 11, 7, 0, 0x40000000, PAGE_BYTES, 0)],
        session_id=41,
    )

    assert result["ok"] is True
    assert result["total_alias_bytes"] == 3 * PAGE_BYTES
    assert result["total_alias_count"] == 3
    assert calls == [(110, 97, 41)]
    assert static_alias._static_alias_sessions[0].alias_count == 3


def test_map_append_failure_after_topology_commit_syncs_destroy_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _create_session(monkeypatch)
    assert created["ok"] is True
    appended = [(0, 12, 8, 0, 0x40000000, PAGE_BYTES, 0)]

    def ioctl(_fd: int, request: int, params: object) -> int:
        assert request == 97
        assert isinstance(params, static_alias._StaticAliasMapParams)
        _complete_map(
            params,
            session_id=41,
            prior_alias_bytes=2 * PAGE_BYTES,
            backing_bytes=2 * PAGE_BYTES,
        )
        # This mirrors the driver path where the complete alias topology is
        # committed before a later PTE operation fails and poisons the session.
        params.mapped_bytes = 0
        params.success_count = 0
        params.failure_count = params.count
        params.poisoned = 1
        params.entries[0].status = 8
        params.rmStatus = 0x56
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_map(appended, session_id=41)

    assert result["ok"] is False
    assert result["made_progress"] is True
    assert result["alias_count"] == 1
    assert result["total_alias_count"] == 3
    cached = static_alias._static_alias_sessions[0]
    assert cached.poisoned is True
    assert cached.alias_count == 3
    assert cached.alias_bytes == 3 * PAGE_BYTES
    assert cached.backing_bytes == 2 * PAGE_BYTES
    assert cached.target_ranges[-1] == (0x40000000, 0x40000000 + PAGE_BYTES)
    assert cached.backing_identities == {(0, 11, 7), (0, 12, 8)}


@pytest.mark.parametrize(
    "entries,error_fragment",
    [
        (
            [
                (0, 0, 0, 0, 0x20000000, 2 * PAGE_BYTES, 0),
                (0, 0, 0, 0, 0x20001000, PAGE_BYTES, 0),
            ],
            "overlap",
        ),
        (
            [
                (
                    0,
                    0,
                    0,
                    static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K,
                    0x20001000,
                    PAGE_BYTES,
                    0,
                )
            ],
            "65536-byte aligned",
        ),
        ([(0, 0, 0, 4, 0x20000000, PAGE_BYTES, 0)], "unknown flags"),
        (
            [
                (0, 0, 0, 0, 0x20000000, PAGE_BYTES, 0),
                (
                    0,
                    0,
                    0,
                    static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K,
                    0x30000000,
                    64 * 1024,
                    0,
                ),
            ],
            "mix 4K and 64K",
        ),
    ],
)
def test_map_validation_rejects_before_fd_or_ioctl(
    monkeypatch: pytest.MonkeyPatch,
    entries: list[tuple[int, ...]],
    error_fragment: str,
) -> None:
    monkeypatch.setattr(
        static_alias,
        "_phase_f_gpu_uuid",
        lambda _device: (_ for _ in ()).throw(AssertionError("UUID must not run")),
    )
    monkeypatch.setattr(
        static_alias,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("ioctl must not run")),
    )

    result = static_alias.driver_static_alias_map(entries)

    assert result["ok"] is False
    assert result["reason"] == "validation_failed"
    assert error_fragment in str(result["error"])


@pytest.mark.parametrize("operation", ["map", "seal", "destroy"])
def test_all_static_alias_ioctls_reject_cuda_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    monkeypatch.setattr(
        static_alias.torch.cuda,
        "is_current_stream_capturing",
        lambda: True,
    )
    monkeypatch.setattr(
        static_alias,
        "_phase_f_gpu_uuid",
        lambda _device: (_ for _ in ()).throw(AssertionError("UUID must not run")),
    )
    monkeypatch.setattr(
        static_alias,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("ioctl must not run")),
    )
    if operation == "map":
        result = static_alias.driver_static_alias_map(_entries())
    else:
        static_alias._static_alias_sessions[0] = static_alias._StaticAliasSession(
            fd=110,
            uuid_bytes=UUID_BYTES,
            session_id=41,
            owner_pid=os.getpid(),
            alias_count=2,
            alias_bytes=2 * PAGE_BYTES,
            backing_bytes=PAGE_BYTES,
        )
        if operation == "seal":
            result = static_alias.driver_static_alias_seal(41)
        else:
            result = static_alias.driver_static_alias_destroy(41)

    assert result["ok"] is False
    assert result["reason"] == "cuda_graph_capture_unsupported"


def test_seal_checks_kernel_counts_and_freezes_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _create_session(monkeypatch)
    assert created["ok"] is True
    calls: list[tuple[int, int]] = []

    def ioctl(fd: int, request: int, params: object) -> int:
        assert isinstance(params, static_alias._StaticAliasSealParams)
        calls.append((fd, request))
        assert params.abi_version == 1
        assert params.struct_size == 72
        assert params.flags == params.reserved == 0
        assert params.session_id == 41
        assert params.expected_alias_count == 2
        assert params.expected_alias_bytes == 2 * PAGE_BYTES
        params.sealed = 1
        params.alias_count = 2
        params.alias_bytes = 2 * PAGE_BYTES
        params.backing_bytes = PAGE_BYTES
        params.poisoned = 0
        params.rmStatus = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_seal(41)

    assert result["ok"] is True
    assert result["sealed"] is True
    assert calls == [(110, 98)]
    append = static_alias.driver_static_alias_map(
        [(0, 11, 7, 0, 0x40000000, PAGE_BYTES, 0)],
        session_id=41,
    )
    assert append["ok"] is False
    assert append["reason"] == "static_alias_session_sealed"


def test_sealed_session_can_refresh_dual_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _create_session(monkeypatch)
    assert created["ok"] is True
    session = static_alias._static_alias_sessions[0]
    session.sealed = True
    calls: list[tuple[int, int, int]] = []

    def ioctl(fd: int, request: int, params: object) -> int:
        assert isinstance(params, static_alias._StaticAliasSealParams)
        calls.append((fd, request, int(params.flags)))
        assert params.flags == (
            static_alias.UVM_MOE_STATIC_ALIAS_SEAL_FLAG_REFRESH_DUAL
        )
        params.sealed = 1
        params.alias_count = session.alias_count
        params.alias_bytes = session.alias_bytes
        params.backing_bytes = session.backing_bytes
        params.poisoned = 0
        params.rmStatus = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_seal(41, refresh_dual=True)

    assert result["ok"] is True
    assert result["refresh_dual"] is True
    assert calls == [
        (
            110,
            98,
            static_alias.UVM_MOE_STATIC_ALIAS_SEAL_FLAG_REFRESH_DUAL,
        )
    ]


def test_sealed_session_can_refresh_nested_dual_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _create_session(monkeypatch)
    assert created["ok"] is True
    session = static_alias._static_alias_sessions[0]
    session.sealed = True

    def ioctl(fd: int, request: int, params: object) -> int:
        assert isinstance(params, static_alias._StaticAliasSealParams)
        assert params.flags == (
            static_alias.UVM_MOE_STATIC_ALIAS_SEAL_FLAG_REFRESH_DUAL
            | static_alias.UVM_MOE_STATIC_ALIAS_SEAL_FLAG_REFRESH_DUAL_NESTED
        )
        params.sealed = 1
        params.alias_count = session.alias_count
        params.alias_bytes = session.alias_bytes
        params.backing_bytes = session.backing_bytes
        params.poisoned = 0
        params.rmStatus = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_seal(
        41,
        refresh_dual=True,
        nested_refresh=True,
    )

    assert result["ok"] is True
    assert result["refresh_dual"] is True
    assert result["nested_refresh"] is True


def test_nested_dual_refresh_requires_refresh_flag() -> None:
    result = static_alias.driver_static_alias_seal(41, nested_refresh=True)

    assert result["ok"] is False
    assert result["reason"] == "validation_failed"


def test_quiesced_append_updates_a_sealed_session_without_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _create_session(monkeypatch)
    assert created["ok"] is True
    session = static_alias._static_alias_sessions[0]
    session.sealed = True
    appended = [
        (
            0,
            12,
            8,
            static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW,
            0x40000000,
            PAGE_BYTES,
            0,
        ),
        (0, 12, 8, 0, 0x50000000, PAGE_BYTES, 0),
    ]

    def ioctl(fd: int, request: int, params: object) -> int:
        assert fd == 110 and request == 97
        assert isinstance(params, static_alias._StaticAliasMapParams)
        assert params.flags == (
            static_alias.UVM_MOE_STATIC_ALIAS_MAP_FLAG_QUIESCED_APPEND
        )
        assert params.session_id == 41
        _complete_map(
            params,
            prior_alias_bytes=2 * PAGE_BYTES,
            backing_bytes=2 * PAGE_BYTES,
        )
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_map(
        appended,
        session_id=41,
        flags=static_alias.UVM_MOE_STATIC_ALIAS_MAP_FLAG_QUIESCED_APPEND,
    )

    assert result["ok"] is True
    assert result["quiesced_append"] is True
    assert session.sealed is True
    assert session.parked is False
    assert session.alias_count == 4
    assert session.alias_bytes == 4 * PAGE_BYTES
    assert session.backing_bytes == 2 * PAGE_BYTES
    assert session.alias_entries[-2:] == appended


def test_quiesced_append_can_defer_dual_view_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _create_session(monkeypatch)
    assert created["ok"] is True
    session = static_alias._static_alias_sessions[0]
    session.sealed = True
    appended = [(0, 12, 8, 0, 0x50000000, PAGE_BYTES, 0)]
    expected_flags = (
        static_alias.UVM_MOE_STATIC_ALIAS_MAP_FLAG_QUIESCED_APPEND
        | static_alias.UVM_MOE_STATIC_ALIAS_MAP_FLAG_DEFER_DUAL_REFRESH
    )

    def ioctl(fd: int, request: int, params: object) -> int:
        assert fd == 110 and request == 97
        assert isinstance(params, static_alias._StaticAliasMapParams)
        assert params.flags == expected_flags
        _complete_map(
            params,
            prior_alias_bytes=2 * PAGE_BYTES,
            backing_bytes=2 * PAGE_BYTES,
        )
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_map(
        appended,
        session_id=41,
        flags=expected_flags,
    )

    assert result["ok"] is True
    assert result["quiesced_append"] is True
    assert result["dual_view_refresh_deferred"] is True


def test_deferred_dual_view_refresh_requires_quiesced_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        static_alias,
        "_phase_f_gpu_uuid",
        lambda _device: (_ for _ in ()).throw(AssertionError("UUID must not run")),
    )

    result = static_alias.driver_static_alias_map(
        _entries(),
        flags=static_alias.UVM_MOE_STATIC_ALIAS_MAP_FLAG_DEFER_DUAL_REFRESH,
    )

    assert result["ok"] is False
    assert result["reason"] == "validation_failed"
    assert "requires a quiesced append" in str(result["error"])


def test_quiesced_append_accepts_e10_4500_entry_delta_in_one_ioctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _create_session(monkeypatch)
    assert created["ok"] is True
    session = static_alias._static_alias_sessions[0]
    session.sealed = True
    appended = [
        (
            index // 100,
            index + 100,
            index + 100,
            0,
            0x60000000 + index * PAGE_BYTES,
            PAGE_BYTES,
            0,
        )
        for index in range(4500)
    ]
    ioctl_calls = 0

    def ioctl(fd: int, request: int, params: object) -> int:
        nonlocal ioctl_calls
        assert fd == 110 and request == 97
        assert isinstance(params, static_alias._StaticAliasMapParams)
        assert params.flags == (
            static_alias.UVM_MOE_STATIC_ALIAS_MAP_FLAG_QUIESCED_APPEND
        )
        assert params.count == 4500
        ioctl_calls += 1
        _complete_map(
            params,
            prior_alias_bytes=2 * PAGE_BYTES,
            backing_bytes=4502 * PAGE_BYTES,
        )
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_map(
        appended,
        session_id=41,
        flags=static_alias.UVM_MOE_STATIC_ALIAS_MAP_FLAG_QUIESCED_APPEND,
    )

    assert result["ok"] is True
    assert result["requested_count"] == 4500
    assert result["ioctl_calls"] == 1
    assert ioctl_calls == 1
    assert session.alias_count == 4502
    assert session.alias_entries[-4500:] == appended


def test_quiesced_append_delegates_existing_overlap_check_to_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _create_session(monkeypatch)
    assert created["ok"] is True
    session = static_alias._static_alias_sessions[0]
    session.sealed = True
    ioctl_calls = 0

    def ioctl(_fd: int, request: int, params: object) -> int:
        nonlocal ioctl_calls
        assert request == static_alias._UVM_MOE_STATIC_ALIAS_MAP_IOCTL
        assert isinstance(params, static_alias._StaticAliasMapParams)
        assert params.entries[0].va_base == session.alias_entries[0][4]
        ioctl_calls += 1
        ctypes.set_errno(errno.EBUSY)
        return -1

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_map(
        [session.alias_entries[0]],
        session_id=session.session_id,
        flags=static_alias.UVM_MOE_STATIC_ALIAS_MAP_FLAG_QUIESCED_APPEND,
    )

    assert ioctl_calls == 1
    assert result["ok"] is False
    assert result["reason"] == "map_ioctl_failed"
    assert result["made_progress"] is False
    assert session.poisoned is False


@pytest.mark.parametrize("parked", [False, True])
def test_quiesced_remove_releases_one_complete_identity_and_updates_cache(
    monkeypatch: pytest.MonkeyPatch,
    parked: bool,
) -> None:
    session = _sealed_two_identity_session()
    session.parked = parked
    static_alias._static_alias_sessions[0] = session
    static_alias._uvm._uvm_slot_direct_owners[0] = {(0, 7): 11, (0, 8): 12}
    monkeypatch.setattr(static_alias, "_phase_f_gpu_uuid", lambda _device: _uuid())
    removed = session.alias_entries[-2:]

    def ioctl(fd: int, request: int, params: object) -> int:
        assert fd == 110 and request == 104
        assert isinstance(params, static_alias._StaticAliasQuiescedRemoveParams)
        assert params.flags == (
            static_alias.UVM_MOE_STATIC_ALIAS_QUIESCED_REMOVE_FLAG_RELEASE_BACKING
        )
        assert params.session_id == 41
        assert params.count == 2
        params.requested_bytes = 2 * PAGE_BYTES
        params.released_alias_bytes = 2 * PAGE_BYTES
        params.total_alias_bytes = 2 * PAGE_BYTES
        params.backing_bytes = PAGE_BYTES
        params.success_count = 2
        params.failure_count = 0
        params.remaining_alias_count = 2
        params.tlb_invalidate_count = 2
        params.poisoned = 0
        for index in range(params.count):
            params.entries[index].status = 0
        params.rmStatus = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_quiesced_remove(41, removed)

    assert result["ok"] is True
    assert result["released_backing_bytes"] == PAGE_BYTES
    assert session.sealed is True
    assert session.parked is parked
    assert session.poisoned is False
    assert session.alias_count == 2
    assert session.alias_bytes == 2 * PAGE_BYTES
    assert session.backing_bytes == PAGE_BYTES
    assert session.backing_identities == {(0, 11, 7)}
    assert static_alias._uvm._uvm_slot_direct_owners[0] == {(0, 7): 11}


def test_quiesced_remove_rejects_an_incomplete_backing_before_ioctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _sealed_two_identity_session()
    static_alias._static_alias_sessions[0] = session
    monkeypatch.setattr(static_alias, "_phase_f_gpu_uuid", lambda _device: _uuid())
    monkeypatch.setattr(
        static_alias,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("ioctl must not run")),
    )

    result = static_alias.driver_static_alias_quiesced_remove(
        41, [session.alias_entries[-1]]
    )

    assert result["ok"] is False
    assert result["reason"] == "validation_failed"
    assert "every alias" in str(result["error"])
    assert session.poisoned is False


def test_quiesced_remove_contract_mismatch_poisons_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _sealed_two_identity_session()
    static_alias._static_alias_sessions[0] = session
    monkeypatch.setattr(static_alias, "_phase_f_gpu_uuid", lambda _device: _uuid())
    removed = session.alias_entries[-2:]

    def ioctl(_fd: int, request: int, params: object) -> int:
        assert request == 104
        assert isinstance(params, static_alias._StaticAliasQuiescedRemoveParams)
        params.requested_bytes = 2 * PAGE_BYTES
        params.released_alias_bytes = 2 * PAGE_BYTES
        params.total_alias_bytes = 3 * PAGE_BYTES
        params.backing_bytes = PAGE_BYTES
        params.success_count = 2
        params.failure_count = 0
        params.remaining_alias_count = 2
        params.poisoned = 0
        for index in range(params.count):
            params.entries[index].status = 0
        params.rmStatus = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_quiesced_remove(41, removed)

    assert result["ok"] is False
    assert result["reason"] == "quiesced_remove_contract_mismatch"
    assert result["contract_violation"] is True
    assert session.poisoned is True
    assert session.alias_count == 4


def test_map_contract_mismatch_poison_is_destroy_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fd_candidates(monkeypatch)

    def ioctl(_fd: int, request: int, params: object) -> int:
        assert request == 97
        assert isinstance(params, static_alias._StaticAliasMapParams)
        _complete_map(params, session_id=77)
        params.mapped_bytes -= PAGE_BYTES
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", ioctl)
    result = static_alias.driver_static_alias_map(_entries())

    assert result["ok"] is False
    assert result["reason"] == "map_contract_mismatch"
    assert result["contract_violation"] is True
    assert static_alias._static_alias_sessions[0].poisoned is True
    assert static_alias._static_alias_sessions[0].backing_identities == {
        (0, 11, 7)
    }
    append = static_alias.driver_static_alias_map(
        [(0, 11, 7, 0, 0x40000000, PAGE_BYTES, 0)],
        session_id=77,
    )
    assert append["reason"] == "static_alias_session_poisoned"


def test_destroy_release_backing_closes_fd_and_clears_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _install_fd_candidates(monkeypatch)

    def create_ioctl(_fd: int, _request: int, params: object) -> int:
        assert isinstance(params, static_alias._StaticAliasMapParams)
        _complete_map(params)
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", create_ioctl)
    assert static_alias.driver_static_alias_map(_entries())["ok"] is True
    static_alias._uvm._uvm_slot_direct_owners[0] = {(0, 7): 11}

    def destroy_ioctl(fd: int, request: int, params: object) -> int:
        assert fd == 110 and request == 99
        assert isinstance(params, static_alias._StaticAliasDestroyParams)
        assert params.abi_version == 1
        assert params.struct_size == 80
        assert params.flags == 1
        assert params.reserved == 0
        assert params.session_id == 41
        params.released_alias_count = 2
        params.remaining_alias_count = 0
        params.released_alias_bytes = 2 * PAGE_BYTES
        params.remaining_alias_bytes = 0
        params.released_backing_bytes = PAGE_BYTES
        params.remaining_backing_bytes = 0
        params.session_cleared = 1
        params.poisoned = 0
        params.rmStatus = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", destroy_ioctl)
    result = static_alias.driver_static_alias_destroy(41)

    assert result["ok"] is True
    assert result["session_cleared"] is True
    assert static_alias.get_static_alias_diagnostics()["session_count"] == 0
    assert static_alias._uvm._uvm_slot_direct_owners[0] == {}
    assert closed == [110]


def test_destroy_can_defer_cpu_restore_for_immediate_remote_cutover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _install_fd_candidates(monkeypatch)

    def create_ioctl(_fd: int, _request: int, params: object) -> int:
        assert isinstance(params, static_alias._StaticAliasMapParams)
        _complete_map(params)
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", create_ioctl)
    assert static_alias.driver_static_alias_map(_entries())["ok"] is True

    def destroy_ioctl(fd: int, request: int, params: object) -> int:
        assert fd == 110 and request == 99
        assert isinstance(params, static_alias._StaticAliasDestroyParams)
        assert params.flags == (
            static_alias.UVM_MOE_STATIC_ALIAS_DESTROY_FLAG_RELEASE_BACKING
            | static_alias.UVM_MOE_STATIC_ALIAS_DESTROY_FLAG_DEFER_CPU_RESTORE
        )
        params.released_alias_count = 2
        params.remaining_alias_count = 0
        params.released_alias_bytes = 2 * PAGE_BYTES
        params.remaining_alias_bytes = 0
        params.released_backing_bytes = PAGE_BYTES
        params.remaining_backing_bytes = 0
        params.session_cleared = 1
        params.poisoned = 0
        params.rmStatus = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", destroy_ioctl)
    result = static_alias.driver_static_alias_destroy(
        41,
        defer_cpu_restore=True,
    )

    assert result["ok"] is True
    assert result["defer_cpu_restore"] is True
    assert closed == [110]


def test_destroy_deferred_cpu_restore_requires_backing_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        static_alias,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("invalid flags must not reach ioctl")
        ),
    )

    result = static_alias.driver_static_alias_destroy(
        41,
        release_backing=False,
        defer_cpu_restore=True,
    )

    assert result["ok"] is False
    assert result["reason"] == "validation_failed"


def test_destroy_kernel_cleared_retires_stale_userspace_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _install_fd_candidates(monkeypatch)

    def create_ioctl(_fd: int, _request: int, params: object) -> int:
        assert isinstance(params, static_alias._StaticAliasMapParams)
        _complete_map(params)
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", create_ioctl)
    assert static_alias.driver_static_alias_map(_entries())["ok"] is True
    static_alias._uvm._uvm_slot_direct_owners[0] = {(0, 7): 11}

    def destroy_ioctl(_fd: int, request: int, params: object) -> int:
        assert request == 99
        assert isinstance(params, static_alias._StaticAliasDestroyParams)
        # Model an older stale wrapper cache after a committed append: the
        # kernel releases more aliases/backing than userspace expected, but
        # authoritatively reports that nothing remains.
        params.released_alias_count = 3
        params.released_alias_bytes = 3 * PAGE_BYTES
        params.released_backing_bytes = 2 * PAGE_BYTES
        params.remaining_alias_count = 0
        params.remaining_alias_bytes = 0
        params.remaining_backing_bytes = 0
        params.session_cleared = 1
        params.poisoned = 0
        params.rmStatus = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", destroy_ioctl)
    result = static_alias.driver_static_alias_destroy(41)

    assert result["ok"] is False
    assert result["reason"] == "destroy_contract_mismatch"
    assert result["contract_violation"] is True
    assert result["session_cleared"] is True
    assert result["cache_cleared"] is True
    assert static_alias.get_static_alias_diagnostics()["session_count"] == 0
    assert static_alias._uvm._uvm_slot_direct_owners[0] == {}
    assert closed == [110]


def test_destroy_can_unmap_then_release_retained_backing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _install_fd_candidates(monkeypatch)

    def create_ioctl(_fd: int, _request: int, params: object) -> int:
        assert isinstance(params, static_alias._StaticAliasMapParams)
        _complete_map(params)
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", create_ioctl)
    assert static_alias.driver_static_alias_map(_entries())["ok"] is True

    def unmap_ioctl(_fd: int, request: int, params: object) -> int:
        assert request == 99
        assert isinstance(params, static_alias._StaticAliasDestroyParams)
        assert params.flags == 0
        params.released_alias_count = 2
        params.released_alias_bytes = 2 * PAGE_BYTES
        params.remaining_backing_bytes = PAGE_BYTES
        params.session_cleared = 0
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", unmap_ioctl)
    first = static_alias.driver_static_alias_destroy(41, release_backing=False)
    assert first["ok"] is True
    assert static_alias._static_alias_sessions[0].alias_count == 0
    assert closed == []

    def release_ioctl(_fd: int, request: int, params: object) -> int:
        assert request == 99
        assert isinstance(params, static_alias._StaticAliasDestroyParams)
        assert params.flags == 1
        params.released_backing_bytes = PAGE_BYTES
        params.session_cleared = 1
        return 0

    monkeypatch.setattr(static_alias, "_libc_ioctl", release_ioctl)
    second = static_alias.driver_static_alias_destroy(41, release_backing=True)
    assert second["ok"] is True
    assert closed == [110]


def test_fork_inherited_session_cannot_seal_or_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = _create_session(monkeypatch)
    assert created["ok"] is True
    owner_pid = os.getpid()
    monkeypatch.setattr(static_alias.os, "getpid", lambda: owner_pid + 1)
    monkeypatch.setattr(
        static_alias,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("ioctl must not run")),
    )

    sealed = static_alias.driver_static_alias_seal(41)
    destroyed = static_alias.driver_static_alias_destroy(41)

    assert sealed["reason"] == "fork_inherited_static_alias_session"
    assert destroyed["reason"] == "fork_inherited_static_alias_session"
