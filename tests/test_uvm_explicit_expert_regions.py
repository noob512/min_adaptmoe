# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm.device_allocator import uvm_pool_residency as residency_module
from vllm.device_allocator.uvm_region import (
    ExpertRecord,
    ExplicitRegionRegistry,
    KernelRegion,
    RegionState,
    compile_expert_envelopes,
)
from vllm.model_executor.layers.fused_moe import layer as layer_module
from vllm.model_executor.layers.fused_moe.layer import FusedMoE


def test_compile_expert_envelopes_records_only_explicit_ids() -> None:
    owners: dict[str, tuple[int, int]] = {}
    record = compile_expert_envelopes(
        layer_id=7,
        global_expert_id=3,
        slices=[(0x100000, 0x80000), (0x200000, 0x80000)],
        envelope_bytes=0x80000,
        existing_owners=owners,
    )

    assert record == ExpertRecord(
        layer_id=7,
        global_expert_id=3,
        aggregate_state=RegionState.UNKNOWN,
        envelope_region_ids=[
            "weights:7:3:envelope:100000:80000",
            "weights:7:3:envelope:200000:80000",
        ],
    )
    assert owners == {
        "100000:80000": (7, 3),
        "200000:80000": (7, 3),
    }


def test_compile_expert_envelopes_rejects_shared_ownership() -> None:
    owners: dict[str, tuple[int, int]] = {}
    compile_expert_envelopes(
        layer_id=7,
        global_expert_id=3,
        slices=[(0x100000, 0x80000)],
        envelope_bytes=0x80000,
        existing_owners=owners,
    )

    with pytest.raises(ValueError, match="shared by experts"):
        compile_expert_envelopes(
            layer_id=7,
            global_expert_id=4,
            slices=[(0x140000, 0x1000)],
            envelope_bytes=0x80000,
            existing_owners=owners,
        )


def test_registry_unknown_requires_conservative_executor() -> None:
    registry = ExplicitRegionRegistry()
    region = registry.register_region("unknown", 0x100000, 0x80000)
    calls: list[tuple[int, int]] = []

    assert registry.release_regions([region.region_id])["ok"] is False
    assert region.state == RegionState.UNKNOWN

    result = registry.release_regions(
        [region.region_id],
        lambda base, length: calls.append((base, length)) or True,
    )

    assert result["ok"] is True
    assert calls == [(region.base, region.length)]
    assert region.state == RegionState.CPU


def test_pool_unknown_release_and_restore_issue_physical_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        residency_module,
        "_EXPLICIT_REGION_REGISTRY",
        ExplicitRegionRegistry(),
    )
    monkeypatch.setattr(residency_module, "driver_register_region", lambda **_: 0)
    monkeypatch.setattr(residency_module, "driver_update_region_state", lambda **_: 0)
    release_calls: list[dict[str, object]] = []

    def release(
        region_ids: list[int],
        **kwargs: object,
    ) -> dict[str, object]:
        release_calls.append({"region_ids": list(region_ids), **kwargs})
        return {
            "ok": True,
            "statuses": {region_id: True for region_id in region_ids},
        }

    monkeypatch.setattr(residency_module, "driver_release_regions", release)
    region = residency_module.register_uvm_region(
        "unknown-pool-region", 0x100000, 0x80000
    )

    released = residency_module.release_uvm_regions(
        [region.region_id], device=2, remote_read=True
    )

    assert released["ok"] is True
    assert release_calls[0]["device"] == 2
    assert release_calls[0]["remote_read"] is True
    assert "return_entries" not in release_calls[0]
    assert region.state == RegionState.CPU

    # CPU residency alone does not prove that remote-read PTEs still exist, so
    # a remote-read request must reach the driver again. A plain CPU release is
    # placement-idempotent and can still skip it.
    released_again = residency_module.release_uvm_regions(
        [region.region_id], device=2, remote_read=True
    )
    assert released_again["ok"] is True
    assert len(release_calls) == 2

    plain_release = residency_module.release_uvm_regions(
        [region.region_id], device=2, remote_read=False
    )
    assert plain_release["ok"] is True
    assert len(release_calls) == 2

    region.state = RegionState.UNKNOWN
    prefetch_calls: list[tuple[list[tuple[int, int]], int, str]] = []

    def prefetch(
        ranges: list[tuple[int, int]],
        device: int,
        *,
        executor: str,
    ) -> tuple[int, str, int]:
        prefetch_calls.append((list(ranges), device, executor))
        return len(ranges), executor, 0

    monkeypatch.setattr(
        residency_module,
        "batch_prefetch_ranges_to_device_with_executor",
        prefetch,
    )
    restored = residency_module.restore_uvm_regions(
        [region.region_id], device=3, executor="cpp_batch"
    )

    assert restored["ok"] is True
    assert prefetch_calls == [([(region.base, region.length)], 3, "cpp_batch")]
    assert region.state == RegionState.GPU


def test_pool_release_commits_each_driver_status_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        residency_module,
        "_EXPLICIT_REGION_REGISTRY",
        ExplicitRegionRegistry(),
    )
    monkeypatch.setattr(residency_module, "driver_register_region", lambda **_: 0)
    monkeypatch.setattr(residency_module, "driver_update_region_state", lambda **_: 0)
    first = residency_module.register_uvm_region("release-first", 0x100000, 0x80000)
    second = residency_module.register_uvm_region("release-second", 0x180000, 0x80000)

    def release(
        region_ids: list[int],
        **_kwargs: object,
    ) -> dict[str, object]:
        return {"ok": False, "statuses": {region_ids[0]: True}}

    monkeypatch.setattr(residency_module, "driver_release_regions", release)

    result = residency_module.release_uvm_regions([first.region_id, second.region_id])

    assert result["ok"] is False
    assert result["statuses"] == {
        first.region_id: True,
        second.region_id: False,
    }
    assert first.state == RegionState.CPU
    assert second.state == RegionState.UNKNOWN


def test_re_registered_cpu_region_reconciles_before_duplicate_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Destroy/re-register must not release an already CPU-resident VA twice."""
    registry = ExplicitRegionRegistry()
    monkeypatch.setattr(residency_module, "_EXPLICIT_REGION_REGISTRY", registry)
    monkeypatch.setattr(residency_module, "driver_register_region", lambda **_: 0)
    monkeypatch.setattr(residency_module, "driver_unregister_region", lambda *_: 0)
    update_states: list[int] = []
    monkeypatch.setattr(
        residency_module,
        "driver_update_region_state",
        lambda *, state, **_: update_states.append(int(state)) or 0,
    )

    physical_queries = iter(
        [
            {
                "ok": True,
                "status": 1,
                "queried_bytes": 0x80000,
                "gpu_bytes": 0x80000,
                "cpu_bytes": 0,
                "any_bytes": 0x80000,
                "unknown_bytes": 0,
            },
            {
                "ok": True,
                "status": 1,
                "queried_bytes": 0x80000,
                "gpu_bytes": 0,
                "cpu_bytes": 0x80000,
                "any_bytes": 0x80000,
                "unknown_bytes": 0,
            },
        ]
    )
    monkeypatch.setattr(
        residency_module,
        "driver_batch_query_ranges_residency_bytes",
        lambda *args, **kwargs: next(physical_queries),
    )
    release_calls: list[list[int]] = []

    def release(region_ids: list[int], **_kwargs: object) -> dict[str, object]:
        release_calls.append(list(region_ids))
        return {
            "ok": True,
            "statuses": {region_id: True for region_id in region_ids},
        }

    monkeypatch.setattr(residency_module, "driver_release_regions", release)

    first = residency_module.register_uvm_region("stable-va", 0x100000, 0x80000)
    first_reconcile = (
        residency_module.reconcile_uvm_regions_cpu_state_from_physical_residency(
            [first.region_id]
        )
    )
    assert first_reconcile["physical_cpu_only"] is False
    assert residency_module.release_uvm_regions([first.region_id])["ok"] is True
    assert len(release_calls) == 1
    assert residency_module.unregister_uvm_regions([first.region_id])["ok"] is True

    second = residency_module.register_uvm_region("stable-va", 0x100000, 0x80000)
    assert second.state == RegionState.UNKNOWN
    second_reconcile = (
        residency_module.reconcile_uvm_regions_cpu_state_from_physical_residency(
            [second.region_id]
        )
    )
    assert second_reconcile["ok"] is True
    assert second_reconcile["physical_cpu_only"] is True
    assert second_reconcile["reconciled"] is True
    assert second.state == RegionState.CPU
    assert residency_module.release_uvm_regions([second.region_id])["ok"] is True
    assert len(release_calls) == 1
    assert update_states == [1, 1]


def test_driver_region_registration_failure_is_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ExplicitRegionRegistry()
    update_calls: list[dict[str, int]] = []
    monkeypatch.setattr(residency_module, "_EXPLICIT_REGION_REGISTRY", registry)
    monkeypatch.setattr(residency_module, "driver_register_region", lambda **_: -37)
    monkeypatch.setattr(
        residency_module,
        "driver_update_region_state",
        lambda **kwargs: update_calls.append(kwargs),
    )

    with pytest.raises(
        RuntimeError,
        match=r"driver region registration failed: .*status=-37",
    ):
        residency_module.register_uvm_region(
            "failed-registration",
            0x100000,
            0x80000,
        )

    assert registry.regions == {}
    assert update_calls == []


@pytest.mark.parametrize("register_status", [None, 0, 3])
def test_driver_region_registration_forces_unknown_without_state_update(
    monkeypatch: pytest.MonkeyPatch,
    register_status: int | None,
) -> None:
    registry = ExplicitRegionRegistry()
    update_calls: list[dict[str, int]] = []
    monkeypatch.setattr(residency_module, "_EXPLICIT_REGION_REGISTRY", registry)
    monkeypatch.setattr(
        residency_module,
        "driver_register_region",
        lambda **_: register_status,
    )
    monkeypatch.setattr(
        residency_module,
        "driver_update_region_state",
        lambda **kwargs: update_calls.append(kwargs),
    )

    region = residency_module.register_uvm_region(
        "fixed-unknown-registration",
        0x100000,
        0x80000,
        state="gpu",
    )

    assert region.state == RegionState.UNKNOWN
    assert update_calls == []


def test_layer_propagates_driver_region_registration_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = ExplicitRegionRegistry()
    monkeypatch.setattr(residency_module, "_EXPLICIT_REGION_REGISTRY", registry)
    monkeypatch.setattr(residency_module, "driver_register_region", lambda **_: -6)
    layer = SimpleNamespace(
        uvm_expert_envelope_bytes=0x80000,
        _uvm_all_local_expert_pairs=lambda: [(3, 0)],
        _uvm_static_alias_owned_experts=lambda _layer_id: set(),
        _uvm_expert_weight_slices=lambda _local_id: [
            (0x100000, 0x80000, (0x100000, 0x180000))
        ],
    )

    with pytest.raises(
        RuntimeError,
        match=r"Failed to register explicit UVM Expert envelope: layer_id=7",
    ) as exc_info:
        FusedMoE._uvm_compile_expert_records(layer, 7)

    assert "status=-6" in str(exc_info.value.__cause__)
    assert not hasattr(layer, "_uvm_expert_records")
    assert registry.regions == {}


def test_hot_prepare_preserves_adjacent_allocation_boundaries_until_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        residency_module,
        "_EXPLICIT_REGION_REGISTRY",
        ExplicitRegionRegistry(),
    )
    monkeypatch.setattr(residency_module, "driver_register_region", lambda **_: 0)
    monkeypatch.setattr(residency_module, "driver_update_region_state", lambda **_: 0)
    first = residency_module.register_uvm_region("hot-first", 0x100000, 0x80000)
    second = residency_module.register_uvm_region("hot-second", 0x180000, 0x80000)
    batches: list[list[tuple[int, int]]] = []

    def prefetch(
        ranges: list[tuple[int, int]],
        _device: int,
        *,
        executor: str,
    ) -> tuple[int, str, int]:
        batches.append(list(ranges))
        return len(ranges), executor, 0

    monkeypatch.setattr(
        residency_module,
        "batch_prefetch_ranges_to_device_with_executor",
        prefetch,
    )

    prepared = residency_module.prepare_uvm_regions_for_gpu_placement(
        [first.region_id, second.region_id],
        device=1,
        executor="cpp_batch",
    )

    assert prepared == {"ok": True, "reason": "prefetch_submitted"}
    assert batches == [[(0x100000, 0x80000), (0x180000, 0x80000)]]
    assert residency_module.query_uvm_region_states(
        [first.region_id, second.region_id]
    ) == {
        first.region_id: RegionState.PREFETCHING_GPU.value,
        second.region_id: RegionState.PREFETCHING_GPU.value,
    }

    committed = residency_module.commit_uvm_regions_gpu_placement(
        [first.region_id, second.region_id], synchronized=True
    )

    assert committed == {"ok": True, "reason": "committed"}
    assert residency_module.query_uvm_region_states(
        [first.region_id, second.region_id]
    ) == {
        first.region_id: RegionState.GPU.value,
        second.region_id: RegionState.GPU.value,
    }


def test_hot_prepare_partial_batch_marks_every_region_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        residency_module,
        "_EXPLICIT_REGION_REGISTRY",
        ExplicitRegionRegistry(),
    )
    monkeypatch.setattr(residency_module, "driver_register_region", lambda **_: 0)
    monkeypatch.setattr(residency_module, "driver_update_region_state", lambda **_: 0)
    first = residency_module.register_uvm_region("partial-first", 0x100000, 0x80000)
    second = residency_module.register_uvm_region("partial-second", 0x300000, 0x80000)
    monkeypatch.setattr(
        residency_module,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda ranges, _device, *, executor: (1, executor, 0),
    )

    result = residency_module.prepare_uvm_regions_for_gpu_placement(
        [first.region_id, second.region_id], executor="cpp_batch"
    )

    assert result == {"ok": False, "reason": "prefetch_incomplete"}
    assert residency_module.query_uvm_region_states(
        [first.region_id, second.region_id]
    ) == {
        first.region_id: RegionState.UNKNOWN.value,
        second.region_id: RegionState.UNKNOWN.value,
    }


def test_hot_prepare_driver_state_failure_rolls_every_region_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        residency_module,
        "_EXPLICIT_REGION_REGISTRY",
        ExplicitRegionRegistry(),
    )
    monkeypatch.setattr(residency_module, "driver_register_region", lambda **_: 0)
    first = residency_module.register_uvm_region("prepare-first", 0x100000, 0x80000)
    second = residency_module.register_uvm_region("prepare-second", 0x300000, 0x80000)
    update_states: list[int] = []

    def update(*, state: int, **_: object) -> int:
        update_states.append(state)
        if state == 4 and update_states.count(4) == 2:
            return -17
        return 0

    prefetch_calls: list[list[tuple[int, int]]] = []
    monkeypatch.setattr(residency_module, "driver_update_region_state", update)
    monkeypatch.setattr(
        residency_module,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda ranges, *_args, **_kwargs: (
            prefetch_calls.append(list(ranges)) or len(ranges),
            "cpp_batch",
            0,
        ),
    )

    result = residency_module.prepare_uvm_regions_for_gpu_placement(
        [first.region_id, second.region_id]
    )

    assert result == {"ok": False, "reason": "driver_state_update_failed"}
    assert prefetch_calls == []
    assert update_states == [4, 4, 0, 0]
    assert first.state == RegionState.UNKNOWN
    assert second.state == RegionState.UNKNOWN


def test_hot_commit_driver_state_exception_rolls_every_region_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        residency_module,
        "_EXPLICIT_REGION_REGISTRY",
        ExplicitRegionRegistry(),
    )
    monkeypatch.setattr(residency_module, "driver_register_region", lambda **_: None)
    first = residency_module.register_uvm_region("commit-first", 0x100000, 0x80000)
    second = residency_module.register_uvm_region("commit-second", 0x300000, 0x80000)
    update_states: list[int] = []

    def update(*, state: int, **_: object) -> int | None:
        update_states.append(state)
        if state == 4:
            return None
        if state == 3 and update_states.count(3) == 2:
            raise RuntimeError("mock state update failure")
        return 2

    monkeypatch.setattr(residency_module, "driver_update_region_state", update)
    monkeypatch.setattr(
        residency_module,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda ranges, *_args, **_kwargs: (len(ranges), "cpp_batch", 0),
    )

    prepared = residency_module.prepare_uvm_regions_for_gpu_placement(
        [first.region_id, second.region_id]
    )
    committed = residency_module.commit_uvm_regions_gpu_placement(
        [first.region_id, second.region_id],
        synchronized=True,
    )

    assert prepared == {"ok": True, "reason": "prefetch_submitted"}
    assert committed == {"ok": False, "reason": "driver_state_update_failed"}
    assert update_states == [4, 4, 3, 3, 0, 0]
    assert first.state == RegionState.UNKNOWN
    assert second.state == RegionState.UNKNOWN


def test_restore_driver_state_failure_returns_failure_and_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        residency_module,
        "_EXPLICIT_REGION_REGISTRY",
        ExplicitRegionRegistry(),
    )
    monkeypatch.setattr(residency_module, "driver_register_region", lambda **_: 1)
    region = residency_module.register_uvm_region("restore-failure", 0x100000, 0x80000)
    update_states: list[int] = []

    def update(*, state: int, **_: object) -> int:
        update_states.append(state)
        return -9 if state == 3 else 1

    monkeypatch.setattr(residency_module, "driver_update_region_state", update)
    monkeypatch.setattr(
        residency_module,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda ranges, *_args, **_kwargs: (len(ranges), "cpp_batch", 0),
    )

    result = residency_module.restore_uvm_regions([region.region_id])

    assert result["ok"] is False
    assert result["statuses"] == {region.region_id: False}
    assert result["failed_region_ids"] == [region.region_id]
    assert update_states == [3, 0]
    assert region.state == RegionState.UNKNOWN


def test_layer_compiles_expert_records_with_unknown_initial_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registrations: list[tuple[str, int, int, str]] = []

    def register(
        region_id: str,
        base: int,
        length: int,
        *,
        state: str,
    ) -> KernelRegion:
        registrations.append((region_id, base, length, state))
        return KernelRegion(region_id, base, length, state)

    monkeypatch.setattr(layer_module, "register_uvm_region", register)
    layer = SimpleNamespace(
        uvm_expert_envelope_bytes=0x80000,
        _uvm_all_local_expert_pairs=lambda: [(3, 0)],
        _uvm_static_alias_owned_experts=lambda _layer_id: set(),
        _uvm_expert_weight_slices=lambda _local_id: [
            (0x100000, 0x80000, (0x100000, 0x180000))
        ],
    )

    FusedMoE._uvm_compile_expert_records(layer, 7)

    record = layer._uvm_expert_records[(7, 3)]
    assert record.aggregate_state == RegionState.UNKNOWN
    assert registrations == [
        (record.envelope_region_ids[0], 0x100000, 0x80000, "unknown")
    ]


def test_layer_hot_set_excludes_released_experts() -> None:
    cold = ExpertRecord(7, 1, RegionState.CPU, ["cold"])
    remote = ExpertRecord(7, 2, RegionState.CPU, ["remote"])
    survivor = ExpertRecord(7, 3, RegionState.UNKNOWN, ["hot-a", "hot-b"])
    layer = SimpleNamespace(
        uvm_init_weight_offload_mode="driver_release",
        _uvm_offloaded_global_expert_ids={1},
        _uvm_remote_read_global_expert_ids={2},
        _uvm_expert_records={
            cold.key: cold,
            remote.key: remote,
            survivor.key: survivor,
        },
        _uvm_init_hot_expert_records=lambda: FusedMoE._uvm_init_hot_expert_records(
            layer
        ),
    )

    assert FusedMoE.get_uvm_init_hot_expert_region_ids(layer) == [
        "hot-a",
        "hot-b",
    ]
    assert FusedMoE.set_uvm_init_hot_expert_state(layer, RegionState.GPU) == 1
    assert survivor.aggregate_state == RegionState.GPU
    assert cold.aggregate_state == RegionState.CPU
    assert remote.aggregate_state == RegionState.CPU


def test_layer_hot_set_excludes_static_alias_owned_experts() -> None:
    alias_owned = ExpertRecord(7, 2, RegionState.UNKNOWN, ["alias-owned"])
    survivor = ExpertRecord(7, 3, RegionState.UNKNOWN, ["hot"])
    layer = SimpleNamespace(
        uvm_init_weight_offload_mode="driver_release",
        _uvm_offloaded_global_expert_ids=set(),
        _uvm_remote_read_global_expert_ids=set(),
        _uvm_static_alias_owned_global_expert_ids={2},
        _uvm_expert_records={
            alias_owned.key: alias_owned,
            survivor.key: survivor,
        },
        _uvm_init_hot_expert_records=lambda: FusedMoE._uvm_init_hot_expert_records(
            layer
        ),
    )

    assert FusedMoE.get_uvm_init_hot_expert_region_ids(layer) == ["hot"]
    assert FusedMoE.set_uvm_init_hot_expert_state(layer, RegionState.GPU) == 1
    assert alias_owned.aggregate_state == RegionState.UNKNOWN
    assert survivor.aggregate_state == RegionState.GPU


def test_dynamic_swap_uses_explicit_expert_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    records = {
        (7, 3): ExpertRecord(7, 3, RegionState.CPU, ["remote"]),
        (7, 4): ExpertRecord(7, 4, RegionState.GPU, ["local"]),
    }

    def release(*, global_expert_id: int, **_kwargs: object) -> dict[str, object]:
        calls.append(("release", global_expert_id))
        records[(7, global_expert_id)].aggregate_state = RegionState.CPU
        return {"ok": True}

    def restore(*, global_expert_id: int, **_kwargs: object) -> dict[str, object]:
        calls.append(("restore", global_expert_id))
        records[(7, global_expert_id)].aggregate_state = RegionState.GPU
        return {"ok": True}

    layer = SimpleNamespace(
        uvm_dynamic_expert_swap_enable=True,
        _uvm_stage_m_uses_remote_read=lambda: True,
        _uvm_layer_id=lambda: 7,
        _uvm_remote_read_global_expert_ids={3},
        _uvm_incremental_offloaded_global_expert_ids={3},
        _uvm_expert_records=records,
        _uvm_release_expert_record=release,
        _uvm_restore_expert_record=restore,
    )
    monkeypatch.setattr(layer_module.torch.cuda, "is_available", lambda: False)

    result = FusedMoE.run_uvm_dynamic_remote_read_swaps(
        layer,
        [{"promote_expert": 3, "demote_expert": 4}],
    )

    assert result["ok"] is True
    assert calls == [("release", 4), ("restore", 3)]
    assert layer._uvm_remote_read_global_expert_ids == {4}
    assert layer._uvm_incremental_offloaded_global_expert_ids == {4}
    assert records[(7, 3)].aggregate_state == RegionState.GPU
    assert records[(7, 4)].aggregate_state == RegionState.CPU


def test_dynamic_swap_finishes_all_demotions_before_promotions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    records = {
        (7, 3): ExpertRecord(7, 3, RegionState.CPU, ["remote-3"]),
        (7, 4): ExpertRecord(7, 4, RegionState.GPU, ["local-4"]),
        (7, 5): ExpertRecord(7, 5, RegionState.CPU, ["remote-5"]),
        (7, 6): ExpertRecord(7, 6, RegionState.GPU, ["local-6"]),
    }

    def release(*, global_expert_id: int, **_kwargs: object) -> dict[str, object]:
        calls.append(("release", global_expert_id))
        records[(7, global_expert_id)].aggregate_state = RegionState.CPU
        return {"ok": True}

    def restore(*, global_expert_id: int, **_kwargs: object) -> dict[str, object]:
        calls.append(("restore", global_expert_id))
        records[(7, global_expert_id)].aggregate_state = RegionState.GPU
        return {"ok": True}

    layer = SimpleNamespace(
        uvm_dynamic_expert_swap_enable=True,
        _uvm_stage_m_uses_remote_read=lambda: True,
        _uvm_layer_id=lambda: 7,
        _uvm_remote_read_global_expert_ids={3, 5},
        _uvm_incremental_offloaded_global_expert_ids={3, 5},
        _uvm_expert_records=records,
        _uvm_release_expert_record=release,
        _uvm_restore_expert_record=restore,
    )
    monkeypatch.setattr(layer_module.torch.cuda, "is_available", lambda: False)

    result = FusedMoE.run_uvm_dynamic_remote_read_swaps(
        layer,
        [
            {"promote_expert": 3, "demote_expert": 4},
            {"promote_expert": 5, "demote_expert": 6},
        ],
    )

    assert result["ok"] is True
    assert result["mutation_order"] == "all_demotions_then_all_promotions"
    assert calls == [
        ("release", 4),
        ("release", 6),
        ("restore", 3),
        ("restore", 5),
    ]
    assert layer._uvm_remote_read_global_expert_ids == {4, 6}
    assert layer._uvm_incremental_offloaded_global_expert_ids == {4, 6}


def test_expert_release_preserves_remote_read_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW", "1")
    record = ExpertRecord(7, 3, RegionState.GPU, ["first", "second"])
    calls: list[dict[str, object]] = []

    def release(
        region_ids: list[str],
        **kwargs: object,
    ) -> dict[str, object]:
        calls.append({"region_ids": list(region_ids), **kwargs})
        return {"ok": True}

    layer = SimpleNamespace(
        _uvm_expert_records={(7, 3): record},
        _uvm_stage_m_uses_remote_read=lambda: True,
        _uvm_record_expert_state=lambda _layer_id, _expert_id, state: setattr(
            record,
            "aggregate_state",
            state,
        ),
    )
    monkeypatch.setattr(layer_module, "release_uvm_regions", release)

    result = FusedMoE._uvm_release_expert_record(
        layer,
        layer_id=7,
        global_expert_id=3,
        device=1,
    )

    assert result["ok"] is True
    assert calls == [
        {
            "region_ids": ["first", "second"],
            "device": 1,
            "remote_read": True,
            "immutable_cpu_shadow": True,
        }
    ]
    assert record.aggregate_state == RegionState.CPU
