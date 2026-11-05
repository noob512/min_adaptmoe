# SPDX-License-Identifier: Apache-2.0

import json
import threading
from types import SimpleNamespace

import pytest

from vllm.device_allocator import uvm_pool_residency as residency_module
from vllm.device_allocator.uvm_region import (
    ExpertRecord,
    ExplicitRegionRegistry,
    RegionState,
)
from vllm.model_executor.layers.fused_moe import layer as layer_module
from vllm.model_executor.layers.fused_moe.layer import FusedMoE


PAGE_BYTES = 4096
ENVELOPE_BYTES = 512 * 1024
LAYER_EXPERTS = 128
REGIONS_PER_EXPERT = 9
LAYER_REGIONS = LAYER_EXPERTS * REGIONS_PER_EXPERT


@pytest.fixture(autouse=True)
def _isolated_slot_policy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        residency_module,
        "_EXPLICIT_REGION_REGISTRY",
        ExplicitRegionRegistry(),
    )
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_BINDINGS", {})
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_DRIVER_LAYERS", {})
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_SLOT_OWNERS", {})
    monkeypatch.setattr(
        residency_module, "_SLOT_DIRECT_DRIVER_OWNER_TOKENS", {}
    )
    monkeypatch.setattr(
        residency_module, "_SLOT_DIRECT_DRIVER_TOKEN_OWNERS", {}
    )
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_PREALLOCATED", set())
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_READY", set())
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_ALLOCATION_BYTES", {})
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_GROUP_LAST_USED", {})
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_PROTECTED_GROUPS", {})
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_USE_EPOCH", 0)
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_ACTIVE_DEVICE", None)
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_OWNER_THREAD", None)
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_POISON_REASON", None)
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_CACHE_LIMITS", None)
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_REUSE_BANK_ACTIVE", None)
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_FIXED_BANK_LANES", None)
    monkeypatch.setattr(
        residency_module, "_SLOT_DIRECT_PRECOMPUTED_LANES_ACTIVE", None
    )
    monkeypatch.setattr(
        residency_module, "_SLOT_DIRECT_LAYER_TRANSITION_ACTIVE", None
    )
    monkeypatch.setattr(
        residency_module, "_SLOT_DIRECT_REGISTERED_PLAN_ACTIVE", None
    )
    monkeypatch.setattr(
        residency_module, "_SLOT_DIRECT_DUAL_BANK_EVENT_ACTIVE", None
    )
    monkeypatch.setattr(
        residency_module, "_SLOT_DIRECT_BANK_LEVEL_COMMIT_ACTIVE", None
    )
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_REGISTERED_PLANS", {})
    monkeypatch.setattr(
        residency_module, "_SLOT_DIRECT_REGISTERED_BANK_LANES", None
    )
    monkeypatch.setattr(
        residency_module, "_SLOT_DIRECT_REGISTERED_BANK_KEYS", None
    )
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_INITIAL_PLAN_ID", None)
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_PLAN_GENERATION", None)
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_PLAN_ACTIVE_LAYER", None)
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_REGISTERED_RUNTIME", None)
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_SESSION_CONTRACT", None)
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_FIXED_BANK_PLAN_HITS", 0)
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_FIXED_BANK_PLAN_BUILDS", 0)
    monkeypatch.setattr(
        residency_module, "_SLOT_DIRECT_FIXED_BANK_PLAN_BUILD_NS", 0
    )
    monkeypatch.delenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", raising=False)
    monkeypatch.delenv(
        "VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES", raising=False
    )
    monkeypatch.delenv(
        "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION", raising=False
    )
    monkeypatch.delenv(
        "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", raising=False
    )
    monkeypatch.delenv(
        "VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT", raising=False
    )
    monkeypatch.delenv("VLLM_UVM_SLOT_DIRECT_CUDA_FENCE", raising=False)
    monkeypatch.delenv(
        "VLLM_UVM_SLOT_DIRECT_BANK_LEVEL_COMMIT", raising=False
    )
    monkeypatch.delenv("VLLM_UVM_SLOT_DIRECT_TRACE_MODE", raising=False)


def _register_cpu_regions(*region_ids: str) -> None:
    for index, region_id in enumerate(region_ids):
        residency_module._EXPLICIT_REGION_REGISTRY.register_region(
            region_id,
            0x40000000 + index * PAGE_BYTES,
            PAGE_BYTES,
            RegionState.CPU,
        )


def test_slot_direct_compact_trace_bounds_success_and_preserves_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    trace_path = tmp_path / "slot_trace.jsonl"
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_TRACE_FILE", str(trace_path))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_TRACE_MODE", "compact")
    large_status = {f"region-{index}": True for index in range(LAYER_REGIONS)}
    low_level = {
        "ok": True,
        "reason": "ok",
        "plan_execute": True,
        "requested_bytes": LAYER_REGIONS * ENVELOPE_BYTES,
        "accepted_bytes": LAYER_REGIONS * ENVELOPE_BYTES,
        "success_count": LAYER_REGIONS,
        "failure_count": 0,
        "entry_statuses": [0] * LAYER_REGIONS,
        "ioctl_wall_ns": 1234,
    }
    residency_module._write_slot_direct_trace(
        {
            "operation": "restore_layer",
            "ok": True,
            "reason": "ok",
            "layer_id": 3,
            "registered_plan_used": True,
            "statuses": large_status,
            "restore": low_level,
            "transition": low_level,
        }
    )
    success_raw = trace_path.read_text().splitlines()[0]
    success = json.loads(success_raw)
    assert len(success_raw) < 2048
    assert success["trace_schema"] == "slot_direct_compact_v1"
    assert "statuses" not in success and "transition" not in success
    assert "entry_statuses" not in success["restore"]
    assert success["restore"]["success_count"] == LAYER_REGIONS

    residency_module._write_slot_direct_trace(
        {
            "operation": "restore_layer",
            "ok": False,
            "reason": "injected_failure",
            "statuses": {"region-0": False},
            "restore": {"ok": False, "entry_statuses": [None]},
        }
    )
    failure = json.loads(trace_path.read_text().splitlines()[1])
    assert failure["statuses"] == {"region-0": False}
    assert failure["restore"]["entry_statuses"] == [None]


def test_slot_direct_compact_trace_preserves_session_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    trace_path = tmp_path / "slot_trace.jsonl"
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_TRACE_FILE", str(trace_path))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_TRACE_MODE", "compact")

    residency_module._write_slot_direct_trace(
        {
            "operation": "release",
            "ok": True,
            "reason": "closed",
            "scope": "session",
            "trigger": "explicit",
            "layer_id": -1,
            "expert_id": -1,
            "closed": True,
        }
    )

    release = json.loads(trace_path.read_text())
    assert release["trace_schema"] == "slot_direct_compact_v1"
    assert release["scope"] == "session"
    assert release["trigger"] == "explicit"
    assert release["expert_id"] == -1


def test_slot_direct_errors_trace_captures_restore_early_return(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    trace_path = tmp_path / "slot_trace.jsonl"
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_TRACE_FILE", str(trace_path))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_TRACE_MODE", "errors")

    result = residency_module.restore_uvm_registered_layer_plan(
        layer_id=7,
        expert_ids=[1, 2],
    )

    assert result == {"ok": False, "reason": "registered_plan_missing"}
    failure = json.loads(trace_path.read_text())
    assert failure["operation"] == "restore_layer"
    assert failure["ok"] is False
    assert failure["reason"] == "registered_plan_missing"
    assert failure["wall_ns"] > 0
    assert "trace_schema" not in failure


def _register_complete_layer(
    *,
    layer_id: int = 3,
    base: int = 0x80000000,
) -> list[tuple[int, list[str]]]:
    groups: list[tuple[int, list[str]]] = []
    for expert_id in range(LAYER_EXPERTS):
        region_ids: list[str] = []
        for envelope_index in range(REGIONS_PER_EXPERT):
            flat_index = expert_id * REGIONS_PER_EXPERT + envelope_index
            region_id = f"layer:{layer_id}:expert:{expert_id}:region:{envelope_index}"
            residency_module._EXPLICIT_REGION_REGISTRY.register_region(
                region_id,
                base + flat_index * ENVELOPE_BYTES,
                ENVELOPE_BYTES,
                RegionState.CPU,
            )
            region_ids.append(region_id)
        groups.append((expert_id, region_ids))
    return groups


def test_complete_layer_policy_submits_mode8_and_mode10_once_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = _register_complete_layer()
    monkeypatch.setenv(
        "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", str(LAYER_REGIONS * ENVELOPE_BYTES)
    )
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", str(LAYER_REGIONS))
    residency_module._SLOT_DIRECT_CACHE_LIMITS = None
    calls: list[tuple[str, list[tuple[int, ...]], int]] = []

    def complete(
        operation: str,
        entries: list[tuple[int, ...]],
        device: int,
    ) -> dict[str, object]:
        calls.append((operation, list(entries), device))
        accepted_bytes = sum(entry[1] for entry in entries)
        return {
            "ok": True,
            "requested_count": len(entries),
            "success_count": len(entries),
            "failure_count": 0,
            "requested_bytes": accepted_bytes,
            "accepted_bytes": accepted_bytes,
            "entry_statuses": [0] * len(entries),
            "ioctl_calls": 1,
            "batches": [{"count": len(entries)}],
            "backing_reuse_count": 0,
            "remap_count": len(entries) if operation == "restore" else 0,
        }

    monkeypatch.setattr(
        residency_module,
        "driver_slot_preallocate_ranges",
        lambda entries, *, device: complete("preallocate", entries, device),
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_restore_ranges",
        lambda entries, *, device: complete("restore", entries, device),
    )

    result = residency_module.restore_uvm_expert_groups_to_slots(
        groups,
        layer_id=3,
        device=0,
    )

    expected_bytes = LAYER_REGIONS * ENVELOPE_BYTES
    assert result["ok"] is True
    assert result["expert_ids"] == list(range(LAYER_EXPERTS))
    assert result["expert_count"] == LAYER_EXPERTS
    assert result["requested_bytes"] == expected_bytes
    assert len(result["statuses"]) == LAYER_REGIONS
    assert all(result["statuses"].values())
    assert [(name, len(entries), device) for name, entries, device in calls] == [
        ("preallocate", LAYER_REGIONS, 0),
        ("restore", LAYER_REGIONS, 0),
    ]
    preallocate_entries = calls[0][1]
    restore_entries = calls[1][1]
    assert preallocate_entries == restore_entries
    assert len({entry[4] for entry in restore_entries}) == LAYER_REGIONS
    assert {entry[3] for entry in restore_entries} == set(range(LAYER_EXPERTS))

    # The first four envelopes deliberately occupy distinct quarters of one
    # 2 MiB VA block. They must remain in the same layer transaction.
    first_four = restore_entries[:4]
    assert {entry[0] // (2 * 1024 * 1024) for entry in first_four} == {
        first_four[0][0] // (2 * 1024 * 1024)
    }
    assert [entry[0] for entry in first_four] == [
        0x80000000 + index * ENVELOPE_BYTES for index in range(4)
    ]
    assert all(entry[1] == ENVELOPE_BYTES for entry in first_four)

    # A second lookup is a logical hit and must not create hidden ioctls.
    again = residency_module.restore_uvm_expert_groups_to_slots(
        groups,
        layer_id=3,
        device=0,
    )
    assert again["ok"] is True
    assert again["reason"] == "already_layer_slot_ready"
    assert len(calls) == 2


def test_init_bank_preallocation_moves_mode8_before_first_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = _register_complete_layer(layer_id=0)
    monkeypatch.setenv(
        "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES",
        str(LAYER_REGIONS * ENVELOPE_BYTES),
    )
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", str(LAYER_REGIONS))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "1")
    residency_module._SLOT_DIRECT_CACHE_LIMITS = None
    calls: list[tuple[str, list[tuple[int, ...]]]] = []

    def complete(operation: str, entries: list[tuple[int, ...]]):
        copied = list(entries)
        calls.append((operation, copied))
        return {
            "ok": True,
            "entry_statuses": [0] * len(copied),
            "backing_reuse_count": 0,
        }

    monkeypatch.setattr(
        residency_module,
        "driver_slot_preallocate_ranges",
        lambda entries, **_kwargs: complete("preallocate", entries),
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_restore_ranges",
        lambda entries, **_kwargs: complete("restore", entries),
    )

    initialized = residency_module.preallocate_uvm_expert_groups_to_slots(
        groups,
        layer_id=0,
    )

    assert initialized["ok"] is True
    assert initialized["reason"] == "preallocated_only"
    assert initialized["preallocate_only"] is True
    assert initialized["logical_placement"] == "slot_reserved"
    assert [name for name, _entries in calls] == ["preallocate"]
    assert len(residency_module._SLOT_DIRECT_PREALLOCATED) == LAYER_REGIONS
    assert residency_module._SLOT_DIRECT_READY == set()
    assert len(residency_module._SLOT_DIRECT_FIXED_BANK_LANES or ()) == LAYER_REGIONS

    first_request = residency_module.restore_uvm_expert_groups_to_slots(
        groups,
        layer_id=0,
    )

    assert first_request["ok"] is True
    assert first_request["preallocate"] is None
    assert first_request["preallocate_only"] is False
    assert [name for name, _entries in calls] == ["preallocate", "restore"]
    assert residency_module._SLOT_DIRECT_READY == (
        residency_module._SLOT_DIRECT_PREALLOCATED
    )


def test_init_bank_repeated_identical_geometry_is_zero_ioctl_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    groups = _register_complete_layer(layer_id=0)
    monkeypatch.setenv(
        "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES",
        str(LAYER_REGIONS * ENVELOPE_BYTES),
    )
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", str(LAYER_REGIONS))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "1")
    calls: list[list[tuple[int, ...]]] = []

    def preallocate(entries, **_kwargs):
        copied = list(entries)
        calls.append(copied)
        return {
            "ok": True,
            "entry_statuses": [0] * len(copied),
            "backing_reuse_count": 0,
        }

    monkeypatch.setattr(
        residency_module, "driver_slot_preallocate_ranges", preallocate
    )

    first = residency_module.preallocate_uvm_expert_groups_to_slots(
        groups, layer_id=0
    )
    lanes = residency_module._SLOT_DIRECT_FIXED_BANK_LANES
    second = residency_module.preallocate_uvm_expert_groups_to_slots(
        groups, layer_id=0
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["reason"] == "already_layer_slot_preallocated"
    assert second["preallocate"] is None
    assert len(calls) == 1
    assert residency_module._SLOT_DIRECT_FIXED_BANK_LANES is lanes
    assert residency_module._SLOT_DIRECT_READY == set()


@pytest.mark.parametrize(
    ("tamper", "error"),
    [
        (
            lambda: residency_module._SLOT_DIRECT_DRIVER_TOKEN_OWNERS.update(
                {
                    next(
                        iter(residency_module._SLOT_DIRECT_DRIVER_TOKEN_OWNERS)
                    ): (99, 99)
                }
            ),
            "reverse owner-token set changed",
        ),
        (
            lambda: setattr(
                residency_module,
                "_SLOT_DIRECT_FIXED_BANK_LANES",
                (
                    residency_module._SLOT_DIRECT_FIXED_BANK_LANES[0],
                    residency_module._SLOT_DIRECT_FIXED_BANK_LANES[0],
                ),
            ),
            "physical lane geometry changed",
        ),
    ],
)
def test_init_bank_idempotence_rejects_corrupt_owner_or_lane_bijection(
    monkeypatch: pytest.MonkeyPatch,
    tamper,
    error: str,
) -> None:
    _register_cpu_regions("region-a", "region-b")
    groups = [(0, ["region-a", "region-b"])]
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", str(2 * PAGE_BYTES))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", "2")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "1")
    calls = 0

    def preallocate(entries, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "ok": True,
            "entry_statuses": [0] * len(entries),
            "backing_reuse_count": 0,
        }

    monkeypatch.setattr(
        residency_module, "driver_slot_preallocate_ranges", preallocate
    )
    assert residency_module.preallocate_uvm_expert_groups_to_slots(
        groups, layer_id=0
    )["ok"]
    assert len(residency_module._SLOT_DIRECT_FIXED_BANK_LANES or ()) == 2

    tamper()
    conflict = residency_module.preallocate_uvm_expert_groups_to_slots(
        groups, layer_id=0
    )

    assert conflict["ok"] is False
    assert conflict["reason"] == "preallocate_only_state_conflict"
    assert error in conflict["error"]
    assert calls == 1


def test_init_bank_idempotence_rejects_coherent_duplicate_physical_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_cpu_regions("region-a", "region-b")
    groups = [(0, ["region-a", "region-b"])]
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", str(2 * PAGE_BYTES))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", "2")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "1")
    calls = 0

    def preallocate(entries, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "ok": True,
            "entry_statuses": [0] * len(entries),
            "backing_reuse_count": 0,
        }

    monkeypatch.setattr(
        residency_module, "driver_slot_preallocate_ranges", preallocate
    )
    assert residency_module.preallocate_uvm_expert_groups_to_slots(
        groups, layer_id=0
    )["ok"]

    first_key, second_key = sorted(residency_module._SLOT_DIRECT_BINDINGS)
    driver_layer = residency_module._SLOT_DIRECT_DRIVER_LAYERS[first_key]
    slot_id = residency_module._SLOT_DIRECT_BINDINGS[first_key]
    residency_module._SLOT_DIRECT_BINDINGS[second_key] = slot_id
    residency_module._SLOT_DIRECT_DRIVER_LAYERS[second_key] = driver_layer
    residency_module._SLOT_DIRECT_SLOT_OWNERS.clear()
    residency_module._SLOT_DIRECT_SLOT_OWNERS[(driver_layer, slot_id)] = second_key
    residency_module._SLOT_DIRECT_FIXED_BANK_LANES = (
        (driver_layer, slot_id, PAGE_BYTES),
    )

    conflict = residency_module.preallocate_uvm_expert_groups_to_slots(
        groups, layer_id=0
    )

    assert conflict["ok"] is False
    assert conflict["reason"] == "preallocate_only_state_conflict"
    assert "physical lane" in conflict["error"]
    assert calls == 1


def test_init_bank_conflicting_geometry_is_rejected_without_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_groups = _register_complete_layer(layer_id=0, base=0x80000000)
    second_groups = _register_complete_layer(layer_id=1, base=0xC0000000)
    monkeypatch.setenv(
        "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES",
        str(LAYER_REGIONS * ENVELOPE_BYTES),
    )
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", str(LAYER_REGIONS))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "1")
    calls = 0

    def preallocate(entries, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "ok": True,
            "entry_statuses": [0] * len(entries),
            "backing_reuse_count": 0,
        }

    monkeypatch.setattr(
        residency_module, "driver_slot_preallocate_ranges", preallocate
    )
    monkeypatch.setattr(
        residency_module,
        "release_uvm_regions_from_slots",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("conflicting init must not release a live bank")
        ),
    )
    assert residency_module.preallocate_uvm_expert_groups_to_slots(
        first_groups, layer_id=0
    )["ok"]
    before = (
        dict(residency_module._SLOT_DIRECT_BINDINGS),
        dict(residency_module._SLOT_DIRECT_DRIVER_LAYERS),
        dict(residency_module._SLOT_DIRECT_SLOT_OWNERS),
        set(residency_module._SLOT_DIRECT_PREALLOCATED),
        dict(residency_module._SLOT_DIRECT_ALLOCATION_BYTES),
        residency_module._SLOT_DIRECT_FIXED_BANK_LANES,
    )

    conflict = residency_module.preallocate_uvm_expert_groups_to_slots(
        second_groups, layer_id=1
    )
    after = (
        dict(residency_module._SLOT_DIRECT_BINDINGS),
        dict(residency_module._SLOT_DIRECT_DRIVER_LAYERS),
        dict(residency_module._SLOT_DIRECT_SLOT_OWNERS),
        set(residency_module._SLOT_DIRECT_PREALLOCATED),
        dict(residency_module._SLOT_DIRECT_ALLOCATION_BYTES),
        residency_module._SLOT_DIRECT_FIXED_BANK_LANES,
    )

    assert conflict["ok"] is False
    assert conflict["reason"] == "preallocate_only_state_conflict"
    assert calls == 1
    assert after == before


def test_registered_layer_plan_executes_and_commits_synchronously_after_init_bank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layers = [
        _register_complete_layer(
            layer_id=layer_id,
            base=0x80000000 + layer_id * 0x40000000,
        )
        for layer_id in range(2)
    ]
    monkeypatch.setenv(
        "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES",
        str(LAYER_REGIONS * ENVELOPE_BYTES),
    )
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", str(LAYER_REGIONS))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_SHARED_PUSH", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_64K_PTE", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", "1")
    residency_module._SLOT_DIRECT_CACHE_LIMITS = None

    monkeypatch.setattr(
        residency_module,
        "driver_slot_preallocate_ranges",
        lambda entries, **_kwargs: {
            "ok": True,
            "entry_statuses": [0] * len(entries),
            "backing_reuse_count": 0,
        },
    )
    legacy_calls: list[str] = []
    monkeypatch.setattr(
        residency_module,
        "driver_slot_restore_ranges",
        lambda *_args, **_kwargs: legacy_calls.append("restore") or {"ok": False},
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_transition_ranges",
        lambda *_args, **_kwargs: legacy_calls.append("transition")
        or {"ok": False},
    )

    registered: list[tuple[int, int, bool, list[tuple[int, ...]]]] = []

    def register_plan(plan_id, layer_id, entries, *, transition, device):
        registered.append((plan_id, layer_id, transition, list(entries)))
        return {"ok": True, "generation": 1, "plan_id": plan_id}

    generation = 1
    executed: list[tuple[int, int, int]] = []

    def execute_plan(plan_id, layer_id, expected_generation, *, device):
        nonlocal generation
        assert expected_generation == generation
        executed.append((plan_id, layer_id, expected_generation))
        generation += 1
        return {
            "ok": True,
            "reason": "ok",
            "new_generation": generation,
            "mode": 21 if len(executed) == 1 else 20,
            "requested_count": LAYER_REGIONS,
            "success_count": LAYER_REGIONS,
            "failure_count": 0,
            "requested_bytes": LAYER_REGIONS * ENVELOPE_BYTES,
            "accepted_bytes": LAYER_REGIONS * ENVELOPE_BYTES,
            "remap_count": LAYER_REGIONS,
            "tlb_invalidate_count": LAYER_REGIONS // 4,
            "copy_push_count": 1,
            "owner_cutover": len(executed) > 1,
            "owner_cutover_trusted": True,
            "poisoned": False,
            "plan_execute": True,
        }

    monkeypatch.setattr(
        residency_module, "driver_slot_register_plan", register_plan
    )
    monkeypatch.setattr(
        residency_module, "driver_slot_execute_plan", execute_plan
    )

    initialized = residency_module.preallocate_uvm_expert_groups_to_slots(
        layers[0], layer_id=0
    )
    plans = residency_module.register_uvm_slot_direct_layer_plans(
        [(0, layers[0]), (1, layers[1])]
    )
    first = residency_module.restore_uvm_registered_layer_plan(
        layer_id=0,
        expert_ids=range(LAYER_EXPERTS),
    )
    # The cmd96 call has completed before restore returns: the next layer uses
    # the new generation directly, with no submit/drain queue in between.
    assert first["ok"] is True
    assert residency_module._SLOT_DIRECT_PLAN_GENERATION == 2
    assert residency_module._SLOT_DIRECT_PLAN_ACTIVE_LAYER == 0
    second = residency_module.restore_uvm_registered_layer_plan(
        layer_id=1,
        expert_ids=range(LAYER_EXPERTS),
    )

    assert initialized["ok"] and plans["ok"] and second["ok"]
    assert len(registered) == 3  # L0 initial plus L0/L1 transition targets.
    assert all(len(entries) == LAYER_REGIONS for *_prefix, entries in registered)
    assert all(
        all(entry[5] == 0 for entry in entries)
        for *_prefix, entries in registered
    )
    assert executed == [(1, 0, 1), (6, 1, 2)]
    assert legacy_calls == []
    assert first["registered_plan_used"] is True
    assert first["layer_transition_used"] is False
    assert second["layer_transition_used"] is True
    assert residency_module._SLOT_DIRECT_PLAN_GENERATION == 3
    assert residency_module._SLOT_DIRECT_PLAN_ACTIVE_LAYER == 1
    assert len(residency_module._SLOT_DIRECT_READY) == LAYER_REGIONS
    assert {key[0] for key in residency_module._SLOT_DIRECT_READY} == {1}


@pytest.mark.parametrize("bank_level_commit", [False, True])
def test_registered_dual_bank_plans_use_disjoint_alternating_backing(
    monkeypatch: pytest.MonkeyPatch,
    bank_level_commit: bool,
) -> None:
    layers = [
        _register_complete_layer(
            layer_id=layer_id,
            base=0x280000000 + layer_id * 0x40000000,
        )
        for layer_id in range(4)
    ]
    bank_bytes = LAYER_REGIONS * ENVELOPE_BYTES
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", str(2 * bank_bytes))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", str(2 * LAYER_REGIONS))
    for name in (
        "VLLM_UVM_SLOT_DIRECT_REUSE_BANK",
        "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION",
        "VLLM_UVM_SLOT_DIRECT_SHARED_PUSH",
        "VLLM_UVM_SLOT_DIRECT_64K_PTE",
        "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN",
        "VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv(
        "VLLM_UVM_SLOT_DIRECT_BANK_LEVEL_COMMIT",
        "1" if bank_level_commit else "0",
    )
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_TRACE_MODE", "compact")
    residency_module._SLOT_DIRECT_CACHE_LIMITS = None

    allocations: list[list[tuple[int, ...]]] = []

    def preallocate(entries, **_kwargs):
        copied = list(entries)
        allocations.append(copied)
        return {
            "ok": True,
            "entry_statuses": [0] * len(copied),
            "backing_reuse_count": 0,
        }

    monkeypatch.setattr(
        residency_module, "driver_slot_preallocate_ranges", preallocate
    )
    registered: list[tuple[int, int, bool, list[tuple[int, ...]]]] = []

    def register(plan_id, layer_id, entries, *, transition, device):
        copied = list(entries)
        registered.append((int(plan_id), int(layer_id), bool(transition), copied))
        return {"ok": True, "generation": 17, "plan_id": int(plan_id)}

    monkeypatch.setattr(residency_module, "driver_slot_register_plan", register)

    first = residency_module.preallocate_uvm_expert_groups_to_slots(
        layers[0], layer_id=0
    )
    second = residency_module.preallocate_uvm_registered_plan_second_bank(
        layers[1], layer_id=1
    )
    plans = residency_module.register_uvm_slot_direct_layer_plans(
        [(layer_id, layers[layer_id]) for layer_id in range(4)]
    )

    assert first["ok"] and second["ok"] and plans["ok"]
    assert len(allocations) == 2
    assert len(residency_module._SLOT_DIRECT_PREALLOCATED) == 2 * LAYER_REGIONS
    assert plans["bank_count"] == 2
    assert plans["dual_bank_event"] is True
    assert [(item[0], item[1], item[2]) for item in registered] == [
        (1, 0, False),
        (2, 0, True),
        (6, 1, True),
        (10, 2, True),
        (14, 3, True),
    ]

    identities = [
        {(int(entry[2]), int(entry[4])) for entry in item[3]}
        for item in registered
    ]
    assert identities[0] == identities[1] == identities[3]
    assert identities[2] == identities[4]
    assert identities[1].isdisjoint(identities[2])
    assert {entry[2] for entry in allocations[0]} == {
        residency_module._SLOT_DIRECT_REUSE_BANK_LAYER
    }
    assert {entry[2] for entry in allocations[1]} == {
        residency_module._SLOT_DIRECT_SECOND_BANK_LAYER
    }

    generation = 17
    executions: list[int] = []

    def execute(_plan_id, layer_id, expected_generation, **_kwargs):
        nonlocal generation
        assert expected_generation == generation
        executions.append(int(layer_id))
        generation += 1
        return {"ok": True, "new_generation": generation, "poisoned": False}

    monkeypatch.setattr(residency_module, "driver_slot_execute_plan", execute)
    rejected_initial = residency_module.restore_uvm_registered_layer_plan(
        layer_id=1, expert_ids=range(LAYER_EXPERTS)
    )
    restored_zero = residency_module.restore_uvm_registered_layer_plan(
        layer_id=0, expert_ids=range(LAYER_EXPERTS)
    )
    rejected_skip = residency_module.restore_uvm_registered_layer_plan(
        layer_id=2, expert_ids=range(LAYER_EXPERTS)
    )
    restored_one = residency_module.restore_uvm_registered_layer_plan(
        layer_id=1, expert_ids=range(LAYER_EXPERTS)
    )

    assert rejected_initial["reason"] == "dual_bank_layer_order_mismatch"
    assert rejected_skip["reason"] == "dual_bank_layer_order_mismatch"
    assert restored_zero["ok"] and restored_one["ok"]
    assert executions == [0, 1]
    assert len(residency_module._SLOT_DIRECT_PREALLOCATED) == 2 * LAYER_REGIONS
    if bank_level_commit:
        runtime = residency_module._SLOT_DIRECT_REGISTERED_RUNTIME
        assert runtime is not None
        assert runtime["active_bank"] == 1
        assert runtime["active_layer"] == 1
        assert runtime["generation"] == generation
        assert runtime["bank_owner_layers"] == [0, 1]
        assert runtime["ready_slots"] == LAYER_REGIONS
        # Per-slot dictionaries remain the immutable registration snapshot.
        assert residency_module._SLOT_DIRECT_READY == set()
        assert restored_one["userspace_commit_slots"] == 0
        assert restored_one["expert_statuses"] == {}
        assert restored_one["statuses"] == {}
    else:
        assert len(residency_module._SLOT_DIRECT_READY) == LAYER_REGIONS
        assert {key[0] for key in residency_module._SLOT_DIRECT_READY} == {1}
        assert len(restored_one["expert_statuses"]) == LAYER_EXPERTS
        assert all(restored_one["expert_statuses"].values())
        assert len(restored_one["statuses"]) == LAYER_REGIONS
        assert all(restored_one["statuses"].values())

    def fail_execute(_plan_id, layer_id, expected_generation, **_kwargs):
        assert int(layer_id) == 2
        assert expected_generation == generation
        return {
            "ok": False,
            "reason": "injected_failure",
            "new_generation": generation,
            "new_generation_trusted": True,
            "poisoned": False,
        }

    monkeypatch.setattr(
        residency_module, "driver_slot_execute_plan", fail_execute
    )
    failed_two = residency_module.restore_uvm_registered_layer_plan(
        layer_id=2, expert_ids=range(LAYER_EXPERTS)
    )
    assert failed_two["ok"] is False
    assert failed_two["error"] == "injected_failure"
    assert failed_two["failed_expert_ids"] == list(range(LAYER_EXPERTS))
    assert len(failed_two["expert_statuses"]) == LAYER_EXPERTS
    assert not any(failed_two["expert_statuses"].values())
    assert len(failed_two["failed_region_ids"]) == LAYER_REGIONS
    assert len(failed_two["statuses"]) == LAYER_REGIONS
    assert not any(failed_two["statuses"].values())


@pytest.mark.parametrize(
    ("layer_ids", "reason"),
    [
        ([0, 1, 2], "dual_bank_requires_even_layer_count"),
        ([0, 2], "dual_bank_requires_contiguous_layers"),
    ],
)
def test_registered_dual_bank_rejects_unsafe_layer_topology_before_driver(
    monkeypatch: pytest.MonkeyPatch,
    layer_ids: list[int],
    reason: str,
) -> None:
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT", "1")
    monkeypatch.setattr(
        residency_module,
        "driver_slot_register_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe topology must not reach the driver")
        ),
    )

    result = residency_module.register_uvm_slot_direct_layer_plans(
        [(layer_id, []) for layer_id in layer_ids]
    )

    assert result["ok"] is False
    assert result["reason"] == reason


def test_registered_layer_plan_rolls_back_partial_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layers = [
        _register_complete_layer(
            layer_id=layer_id,
            base=0x180000000 + layer_id * 0x40000000,
        )
        for layer_id in range(2)
    ]
    monkeypatch.setenv(
        "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES",
        str(LAYER_REGIONS * ENVELOPE_BYTES),
    )
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", str(LAYER_REGIONS))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_SHARED_PUSH", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_64K_PTE", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", "1")
    residency_module._SLOT_DIRECT_CACHE_LIMITS = None
    monkeypatch.setattr(
        residency_module,
        "driver_slot_preallocate_ranges",
        lambda entries, **_kwargs: {
            "ok": True,
            "entry_statuses": [0] * len(entries),
            "backing_reuse_count": 0,
        },
    )

    accepted: list[int] = []

    def register(plan_id, _layer_id, _entries, **_kwargs):
        ok = int(plan_id) != 6
        if ok:
            accepted.append(int(plan_id))
        return {"ok": ok, "generation": 41, "plan_id": int(plan_id)}

    destroyed: list[tuple[int, int]] = []
    monkeypatch.setattr(residency_module, "driver_slot_register_plan", register)
    monkeypatch.setattr(
        residency_module,
        "driver_slot_destroy_plan",
        lambda plan_id, generation, **_kwargs: (
            destroyed.append((int(plan_id), int(generation)))
            or {"ok": True, "plan_id": int(plan_id)}
        ),
    )

    initialized = residency_module.preallocate_uvm_expert_groups_to_slots(
        layers[0], layer_id=0
    )
    result = residency_module.register_uvm_slot_direct_layer_plans(
        [(0, layers[0]), (1, layers[1])]
    )

    assert initialized["ok"] is True
    assert result["ok"] is False
    assert result["reason"] == "transition_plan_register_failed"
    assert accepted == [1, 2]
    assert destroyed == [(2, 41), (1, 41)]
    assert len(result["rollback_results"]) == 2
    assert residency_module._SLOT_DIRECT_REGISTERED_PLANS == {}
    assert residency_module._SLOT_DIRECT_PLAN_GENERATION is None


def test_registered_plan_api_rejects_duplicate_and_limit_before_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", "1")
    monkeypatch.setattr(
        residency_module,
        "driver_slot_register_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid plan set must not reach the driver")
        ),
    )

    duplicate = residency_module.register_uvm_slot_direct_layer_plans(
        [(7, []), (7, [])]
    )
    assert duplicate["ok"] is False
    assert duplicate["reason"] == "registered_plan_layer_id_duplicate"

    too_many = residency_module.register_uvm_slot_direct_layer_plans(
        [(layer_id, []) for layer_id in range(64)]
    )
    assert too_many["ok"] is False
    assert too_many["reason"] == "registered_plan_limit_exceeded"
    assert too_many["total_plan_count"] == 65
    assert too_many["max_plan_count"] == 64


def test_complete_layer_reuses_session_fixed_bank_lane_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layers = [
        _register_complete_layer(
            layer_id=layer_id,
            base=0x80000000 + layer_id * 0x40000000,
        )
        for layer_id in range(3)
    ]
    monkeypatch.setenv(
        "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", str(LAYER_REGIONS * ENVELOPE_BYTES)
    )
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", str(LAYER_REGIONS))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "1")
    residency_module._SLOT_DIRECT_CACHE_LIMITS = None
    calls: list[tuple[str, list[tuple[int, ...]]]] = []
    slot_id_allocations: list[tuple[int, int, str]] = []
    original_slot_id = residency_module._slot_direct_slot_id_locked

    def counted_slot_id(key: tuple[int, int, str]) -> int:
        if key not in residency_module._SLOT_DIRECT_BINDINGS:
            slot_id_allocations.append(key)
        return original_slot_id(key)

    monkeypatch.setattr(
        residency_module, "_slot_direct_slot_id_locked", counted_slot_id
    )

    def complete(
        operation: str,
        entries: list[tuple[int, ...]],
    ) -> dict[str, object]:
        copied = list(entries)
        calls.append((operation, copied))
        return {
            "ok": True,
            "entry_statuses": [0] * len(copied),
            "backing_reuse_count": sum(
                bool(
                    entry[6]
                    & residency_module.UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
                )
                for entry in copied
            ),
        }

    monkeypatch.setattr(
        residency_module,
        "driver_slot_preallocate_ranges",
        lambda entries, *, device: complete("preallocate", entries),
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_restore_ranges",
        lambda entries, *, device: complete("restore", entries),
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_release_ranges",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a compatible fixed bank must not use mode19")
        ),
    )

    first = residency_module.restore_uvm_expert_groups_to_slots(
        layers[0], layer_id=0
    )
    first_plan = residency_module._SLOT_DIRECT_FIXED_BANK_LANES
    second = residency_module.restore_uvm_expert_groups_to_slots(
        layers[1], layer_id=1
    )
    learned_plan = residency_module._SLOT_DIRECT_FIXED_BANK_LANES
    third = residency_module.restore_uvm_expert_groups_to_slots(
        layers[2], layer_id=2
    )

    assert first["ok"] and second["ok"] and third["ok"]
    assert first["bank_lane_plan_builds"] == 1
    assert first["bank_lane_plan_cache_hits"] == 0
    assert second["bank_lane_plan_cache_hits"] == 1
    assert third["bank_lane_plan_cache_hits"] == 1
    assert second["bank_rebind_slots"] == third["bank_rebind_slots"] == LAYER_REGIONS
    assert first_plan is not None
    assert learned_plan is not None
    assert len(learned_plan) == LAYER_REGIONS
    # A cache hit retains the immutable lane tuple and only changes owners in
    # the authoritative lease tables.
    assert learned_plan is first_plan
    assert residency_module._SLOT_DIRECT_FIXED_BANK_LANES is first_plan
    # L1/L2 bind directly to retained lanes. Only L0 needs hashed slot-id
    # allocation; no 2 * 1,152 provisional ids are created and discarded.
    assert len(slot_id_allocations) == LAYER_REGIONS
    assert set(first["slot_ids"].values()) == set(second["slot_ids"].values())
    assert set(second["slot_ids"].values()) == set(third["slot_ids"].values())
    assert [name for name, _entries in calls] == [
        "preallocate",
        "restore",
        "preallocate",
        "restore",
        "preallocate",
        "restore",
    ]
    assert all(
        not (
            entry[6]
            & residency_module.UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
        )
        for entry in calls[0][1]
    )
    assert all(
        entry[6]
        & residency_module.UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
        for entry in calls[2][1] + calls[4][1]
    )

    residency_module._clear_uvm_slot_direct_policy_state(0)
    assert residency_module._SLOT_DIRECT_FIXED_BANK_LANES is None
    assert residency_module._SLOT_DIRECT_CACHE_LIMITS is None


def test_complete_fixed_bank_uses_mode20_with_combined_policy_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_groups = _register_complete_layer(layer_id=0, base=0x80000000)
    second_groups = _register_complete_layer(layer_id=1, base=0xC0000000)
    monkeypatch.setenv(
        "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES",
        str(LAYER_REGIONS * ENVELOPE_BYTES),
    )
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", str(LAYER_REGIONS))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_SHARED_PUSH", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_64K_PTE", "1")
    residency_module._SLOT_DIRECT_CACHE_LIMITS = None
    calls: list[tuple[str, list[tuple[int, ...]]]] = []

    def complete(operation: str, entries: list[tuple[int, ...]]):
        copied = list(entries)
        calls.append((operation, copied))
        result: dict[str, object] = {
            "ok": True,
            "entry_statuses": [0] * len(copied),
        }
        if operation == "transition":
            result.update(
                owner_cutover=True,
                owner_cutover_count=len(copied),
                owner_cutover_trusted=True,
                owner_state_indeterminate=False,
            )
        return result

    monkeypatch.setattr(
        residency_module,
        "driver_slot_preallocate_ranges",
        lambda entries, **_kwargs: complete("preallocate", entries),
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_restore_ranges",
        lambda entries, **_kwargs: complete("restore", entries),
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_transition_ranges",
        lambda entries, **_kwargs: complete("transition", entries),
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_release_ranges",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a complete fixed bank must not be released")
        ),
    )

    first = residency_module.restore_uvm_expert_groups_to_slots(
        first_groups, layer_id=0
    )
    second = residency_module.restore_uvm_expert_groups_to_slots(
        second_groups, layer_id=1
    )

    assert first["ok"] is True
    assert first["layer_transition_used"] is False
    assert first["layer_transition_reason"] == "initial_bank_allocation"
    assert second["ok"] is True
    assert second["layer_transition_used"] is True
    assert second["layer_transition_reason"] == "ok"
    assert second["transition"]["owner_cutover"] is True
    assert [name for name, _entries in calls] == [
        "preallocate",
        "restore",
        "transition",
    ]
    transition_entries = calls[-1][1]
    assert len(transition_entries) == LAYER_REGIONS
    # The policy supplies the ownership contract. The low-level mode-20
    # wrapper derives and consistently adds shared-push/64K policy bits.
    assert all(entry[6] == 3 for entry in transition_entries)
    assert not any(key[0] == 0 for key in residency_module._SLOT_DIRECT_READY)
    assert sum(key[0] == 1 for key in residency_module._SLOT_DIRECT_READY) == LAYER_REGIONS


@pytest.mark.parametrize(
    (
        "owner_cutover",
        "owner_cutover_trusted",
        "expected_reason",
        "expected_owner_layer",
    ),
    [
        (False, True, "pre_cutover_failure", 0),
        (True, True, "post_cutover_failure", 1),
        (None, False, "indeterminate_owner", 0),
    ],
)
def test_mode20_failure_poisoning_preserves_auditable_owner_state(
    monkeypatch: pytest.MonkeyPatch,
    owner_cutover: bool | None,
    owner_cutover_trusted: bool,
    expected_reason: str,
    expected_owner_layer: int,
) -> None:
    _register_cpu_regions("old", "new")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", str(PAGE_BYTES))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION", "1")
    residency_module._SLOT_DIRECT_CACHE_LIMITS = None
    monkeypatch.setattr(
        residency_module,
        "driver_slot_preallocate_ranges",
        lambda entries, **_kwargs: {
            "ok": True,
            "entry_statuses": [0] * len(entries),
        },
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_restore_ranges",
        lambda entries, **_kwargs: {
            "ok": True,
            "entry_statuses": [0] * len(entries),
        },
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_transition_ranges",
        lambda entries, **_kwargs: {
            "ok": False,
            "reason": "synthetic_failure",
            "entry_statuses": [7] * len(entries),
            "owner_cutover": owner_cutover,
            "owner_cutover_count": len(entries) if owner_cutover else 0,
            "owner_cutover_trusted": owner_cutover_trusted,
            "owner_state_indeterminate": owner_cutover is None,
        },
    )

    first = residency_module.restore_uvm_expert_groups_to_slots(
        [(1, ["old"])], layer_id=0
    )
    second = residency_module.restore_uvm_expert_groups_to_slots(
        [(1, ["new"])], layer_id=1
    )

    assert first["ok"] is True
    assert second["ok"] is False
    assert second["reason"] == "slot_layer_transition_failed"
    assert second["layer_transition_used"] is True
    assert second["layer_transition_reason"] == expected_reason
    assert second["poisoned"] is True
    owner_layers = {key[0] for key in residency_module._SLOT_DIRECT_PREALLOCATED}
    assert owner_layers == {expected_owner_layer}
    assert residency_module._SLOT_DIRECT_READY == set()


def test_precomputed_lane_plan_ablation_keeps_provisional_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_cpu_regions("old", "new")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", str(PAGE_BYTES))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES", "0")
    residency_module._SLOT_DIRECT_CACHE_LIMITS = None
    allocations: list[tuple[int, int, str]] = []
    original_slot_id = residency_module._slot_direct_slot_id_locked

    def counted_slot_id(key: tuple[int, int, str]) -> int:
        if key not in residency_module._SLOT_DIRECT_BINDINGS:
            allocations.append(key)
        return original_slot_id(key)

    monkeypatch.setattr(
        residency_module, "_slot_direct_slot_id_locked", counted_slot_id
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_preallocate_ranges",
        lambda entries, **_kwargs: {
            "ok": True,
            "entry_statuses": [0] * len(entries),
        },
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_restore_ranges",
        lambda entries, **_kwargs: {
            "ok": True,
            "entry_statuses": [0] * len(entries),
        },
    )

    first = residency_module.restore_uvm_expert_groups_to_slots(
        [(1, ["old"])], layer_id=0
    )
    second = residency_module.restore_uvm_expert_groups_to_slots(
        [(1, ["new"])], layer_id=1
    )

    assert first["ok"] and second["ok"]
    assert len(allocations) == 2
    assert residency_module._SLOT_DIRECT_FIXED_BANK_LANES is None
    assert first["bank_lane_plan_builds"] == 0
    assert second["bank_lane_plan_cache_hits"] == 0


def test_precomputed_lane_plan_mode_is_validated_and_session_locked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES", "invalid")
    with pytest.raises(RuntimeError, match="must be 0 or 1"):
        residency_module._slot_direct_precomputed_lanes_enabled()

    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES", "1")
    assert residency_module._slot_direct_precomputed_lanes_enabled() is True
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES", "0")
    with pytest.raises(RuntimeError, match="cannot change during a slot session"):
        residency_module._slot_direct_precomputed_lanes_enabled()


def test_slot_restore_success_deduplicates_and_preserves_cpu_residency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_cpu_regions("first", "second")
    calls: list[tuple[str, list[tuple[int, ...]], int]] = []

    def preallocate(entries: list[tuple[int, ...]], *, device: int):
        calls.append(("preallocate", list(entries), device))
        return {"ok": True}

    def restore(entries: list[tuple[int, ...]], *, device: int):
        calls.append(("restore", list(entries), device))
        return {"ok": True}

    monkeypatch.setattr(
        residency_module, "driver_slot_preallocate_ranges", preallocate
    )
    monkeypatch.setattr(residency_module, "driver_slot_restore_ranges", restore)

    result = residency_module.restore_uvm_regions_to_slots(
        ["first", "first", "second"],
        layer_id=7,
        expert_id=11,
        device=2,
    )

    assert result["ok"] is True
    assert result["reason"] == "ok"
    assert result["statuses"] == {"first": True, "second": True}
    assert result["requested_bytes"] == 2 * PAGE_BYTES
    assert result["logical_placement"] == "slot_gpu"
    assert result["ordinary_uvm_placement"] == "cpu"
    assert [call[0] for call in calls] == ["preallocate", "restore"]
    assert all(call[2] == 2 for call in calls)
    assert calls[0][1] == calls[1][1]
    entries = calls[0][1]
    assert len(entries) == 2
    assert all(entry[2:4] == (7, 11) for entry in entries)
    assert all(entry[5] == 0 for entry in entries)
    assert all(
        entry[6] == residency_module._SLOT_DIRECT_NO_EVICT_FLAG
        for entry in entries
    )
    assert len({entry[4] for entry in entries}) == 2
    assert all(
        region.state == RegionState.CPU
        for region in residency_module._EXPLICIT_REGION_REGISTRY.regions.values()
    )
    assert len(residency_module._SLOT_DIRECT_PREALLOCATED) == 2
    assert residency_module._SLOT_DIRECT_READY == (
        residency_module._SLOT_DIRECT_PREALLOCATED
    )


def test_slot_preallocate_failure_does_not_restore_or_commit_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_cpu_regions("weight")
    captured: list[tuple[int, ...]] = []

    def preallocate(entries: list[tuple[int, ...]], *, device: int):
        assert device == 0
        captured.extend(entries)
        return {"ok": False, "reason": "synthetic_preallocate_failure"}

    monkeypatch.setattr(
        residency_module, "driver_slot_preallocate_ranges", preallocate
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_restore_ranges",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("restore must not follow failed preallocation")
        ),
    )

    result = residency_module.restore_uvm_regions_to_slots(
        ["weight"], layer_id=3, expert_id=5
    )

    assert result["ok"] is False
    assert result["reason"] == "slot_preallocate_failed"
    assert result["restore"] is None
    assert captured[0][6] == residency_module._SLOT_DIRECT_NO_EVICT_FLAG
    assert residency_module._SLOT_DIRECT_PREALLOCATED == set()
    assert residency_module._SLOT_DIRECT_READY == set()
    assert (
        residency_module._EXPLICIT_REGION_REGISTRY.regions["weight"].state
        == RegionState.CPU
    )


def test_restore_failure_rolls_back_preallocation_and_never_marks_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_cpu_regions("weight")
    preallocate_calls = 0
    restore_calls = 0
    release_calls = 0

    def preallocate(_entries: list[tuple[int, ...]], *, device: int):
        nonlocal preallocate_calls
        preallocate_calls += 1
        return {"ok": True}

    def restore(_entries: list[tuple[int, ...]], *, device: int):
        nonlocal restore_calls
        restore_calls += 1
        return {"ok": False, "reason": "synthetic_restore_failure"}

    def release(entries: list[tuple[int, ...]], *, device: int):
        nonlocal release_calls
        release_calls += 1
        return {"ok": True, "entry_statuses": [0] * len(entries), "mode": 19}

    monkeypatch.setattr(
        residency_module, "driver_slot_preallocate_ranges", preallocate
    )
    monkeypatch.setattr(residency_module, "driver_slot_restore_ranges", restore)
    monkeypatch.setattr(residency_module, "driver_slot_release_ranges", release)

    first = residency_module.restore_uvm_regions_to_slots(
        ["weight"], layer_id=3, expert_id=5
    )
    second = residency_module.restore_uvm_regions_to_slots(
        ["weight"], layer_id=3, expert_id=5
    )

    assert first["reason"] == second["reason"] == "slot_restore_failed"
    assert first["rollback"]["ok"] is True
    assert second["rollback"]["ok"] is True
    assert preallocate_calls == 2
    assert restore_calls == 2
    assert release_calls == 2
    assert residency_module._SLOT_DIRECT_PREALLOCATED == set()
    assert residency_module._SLOT_DIRECT_READY == set()
    assert (
        residency_module._EXPLICIT_REGION_REGISTRY.regions["weight"].state
        == RegionState.CPU
    )


def test_already_ready_skips_driver_and_fused_moe_uses_slot_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_cpu_regions("weight")
    policy_calls = 0

    def policy_restore(
        region_ids: list[str],
        *,
        layer_id: int,
        expert_id: int,
        device: int,
    ) -> dict[str, object]:
        nonlocal policy_calls
        policy_calls += 1
        return residency_module.restore_uvm_regions_to_slots(
            region_ids,
            layer_id=layer_id,
            expert_id=expert_id,
            device=device,
        )

    low_level_calls: list[str] = []
    monkeypatch.setattr(
        residency_module,
        "driver_slot_preallocate_ranges",
        lambda _entries, **_kwargs: low_level_calls.append("preallocate")
        or {"ok": True},
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_restore_ranges",
        lambda _entries, **_kwargs: low_level_calls.append("restore")
        or {"ok": True},
    )
    monkeypatch.setattr(layer_module, "restore_uvm_regions_to_slots", policy_restore)
    monkeypatch.setattr(
        layer_module,
        "validate_uvm_slot_direct_session_contract",
        lambda **_kwargs: {"registered_plan": False},
    )
    monkeypatch.setattr(
        layer_module,
        "restore_uvm_regions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ordinary UVM restore is forbidden in slot mode")
        ),
    )
    record = ExpertRecord(7, 11, RegionState.CPU, ["weight"])
    layer = SimpleNamespace(
        uvm_expert_offload_access_mode="slot_restore",
        _uvm_expert_records={(7, 11): record},
        _uvm_record_expert_state=lambda _layer, _expert, state: setattr(
            record, "aggregate_state", state
        ),
    )

    first = FusedMoE._uvm_restore_expert_record(
        layer, layer_id=7, global_expert_id=11, device=2
    )
    second = FusedMoE._uvm_restore_expert_record(
        layer, layer_id=7, global_expert_id=11, device=2
    )

    assert first["ok"] is True
    assert second["reason"] == "already_slot_ready"
    assert policy_calls == 2
    assert low_level_calls == ["preallocate", "restore"]
    # Logical slot accessibility is separate from ordinary UVM residency.
    assert record.aggregate_state == RegionState.CPU
    assert (
        residency_module._EXPLICIT_REGION_REGISTRY.regions["weight"].state
        == RegionState.CPU
    )
    assert second["logical_placement"] == "slot_gpu"
    assert second["ordinary_uvm_placement"] == "cpu"


@pytest.mark.parametrize(
    ("drift_name", "drift_value"),
    [
        ("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", "1"),
        ("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "restore"),
        ("VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", str(2 * PAGE_BYTES)),
        ("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", "2"),
        ("VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT", "1"),
        ("VLLM_UVM_SLOT_DIRECT_CUDA_FENCE", "1"),
    ],
)
def test_slot_forward_rejects_environment_drift_before_restore(
    monkeypatch: pytest.MonkeyPatch,
    drift_name: str,
    drift_value: str,
) -> None:
    monkeypatch.setenv("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "slot_restore")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", "0")
    residency_module.freeze_uvm_slot_direct_session_contract(
        registered_plan=False,
        access_mode="slot_restore",
        device=0,
    )
    monkeypatch.setenv(drift_name, drift_value)
    monkeypatch.setattr(
        layer_module,
        "restore_uvm_regions_to_slots",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("drifted forward must not restore")
        ),
    )
    record = ExpertRecord(0, 0, RegionState.CPU, ["weight"])
    layer = SimpleNamespace(
        uvm_expert_offload_access_mode="slot_restore",
        _uvm_expert_records={(0, 0): record},
        _uvm_record_expert_state=lambda *_args: None,
    )

    with pytest.raises(RuntimeError, match="environment changed"):
        FusedMoE._uvm_restore_expert_record(
            layer, layer_id=0, global_expert_id=0, device=0
        )


def test_slot_policy_cleanup_resets_cache_limits_for_next_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", str(2 * PAGE_BYTES))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", "2")
    assert residency_module._slot_direct_cache_limits_locked() == (
        2 * PAGE_BYTES,
        2,
    )

    residency_module._clear_uvm_slot_direct_policy_state(0)
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", str(4 * PAGE_BYTES))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", "4")

    assert residency_module._slot_direct_cache_limits_locked() == (
        4 * PAGE_BYTES,
        4,
    )


def test_bounded_cache_releases_lru_and_restores_evicted_expert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_cpu_regions("first", "second")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", str(PAGE_BYTES))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", "1")
    residency_module._SLOT_DIRECT_CACHE_LIMITS = None
    calls: list[tuple[str, list[tuple[int, ...]]]] = []

    def preallocate(entries: list[tuple[int, ...]], *, device: int):
        calls.append(("preallocate", list(entries)))
        return {"ok": True, "entry_statuses": [0] * len(entries)}

    def restore(entries: list[tuple[int, ...]], *, device: int):
        calls.append(("restore", list(entries)))
        return {"ok": True}

    def release(entries: list[tuple[int, ...]], *, device: int):
        calls.append(("release", list(entries)))
        return {"ok": True, "entry_statuses": [0] * len(entries), "mode": 19}

    monkeypatch.setattr(
        residency_module, "driver_slot_preallocate_ranges", preallocate
    )
    monkeypatch.setattr(residency_module, "driver_slot_restore_ranges", restore)
    monkeypatch.setattr(residency_module, "driver_slot_release_ranges", release)

    first = residency_module.restore_uvm_regions_to_slots(
        ["first"], layer_id=7, expert_id=11
    )
    second = residency_module.restore_uvm_regions_to_slots(
        ["second"], layer_id=7, expert_id=12
    )
    first_again = residency_module.restore_uvm_regions_to_slots(
        ["first"], layer_id=7, expert_id=11
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert first_again["ok"] is True
    assert second["evictions"][0]["expert_id"] == 11
    assert first_again["evictions"][0]["expert_id"] == 12
    assert [name for name, _entries in calls] == [
        "preallocate",
        "restore",
        "release",
        "preallocate",
        "restore",
        "release",
        "preallocate",
        "restore",
    ]
    assert first_again["slot_ids"]["first"] == first["slot_ids"]["first"]
    assert residency_module._SLOT_DIRECT_READY == {(7, 11, "first")}
    assert residency_module._SLOT_DIRECT_PREALLOCATED == {(7, 11, "first")}
    assert first_again["cache_after"] == {"bytes": PAGE_BYTES, "slots": 1}


def test_fixed_bank_rebind_reuses_identity_without_capacity_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_cpu_regions("first", "second")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", str(PAGE_BYTES))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "1")
    residency_module._SLOT_DIRECT_CACHE_LIMITS = None
    calls: list[tuple[str, list[tuple[int, ...]]]] = []

    def preallocate(entries: list[tuple[int, ...]], *, device: int):
        calls.append(("preallocate", list(entries)))
        reuse_count = sum(
            bool(
                entry[6]
                & residency_module.UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
            )
            for entry in entries
        )
        return {
            "ok": True,
            "entry_statuses": [0] * len(entries),
            "backing_reuse_count": reuse_count,
        }

    def restore(entries: list[tuple[int, ...]], *, device: int):
        calls.append(("restore", list(entries)))
        return {"ok": True}

    monkeypatch.setattr(
        residency_module, "driver_slot_preallocate_ranges", preallocate
    )
    monkeypatch.setattr(residency_module, "driver_slot_restore_ranges", restore)
    monkeypatch.setattr(
        residency_module,
        "driver_slot_release_ranges",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("capacity mode19 is forbidden for a compatible bank")
        ),
    )

    first = residency_module.restore_uvm_regions_to_slots(
        ["first"], layer_id=0, expert_id=11
    )
    second = residency_module.restore_uvm_regions_to_slots(
        ["second"], layer_id=1, expert_id=11
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["bank_reuse_enabled"] is True
    assert second["bank_rebind_slots"] == 1
    assert second["bank_rebind_reason"] == "ok"
    assert second["preallocate"]["backing_reuse_count"] == 1
    assert second["evictions"][0]["mechanism"] == "bank_rebind"
    assert second["evictions"][0]["release"] is None
    assert first["slot_ids"]["first"] == second["slot_ids"]["second"]
    assert [name for name, _entries in calls] == [
        "preallocate",
        "restore",
        "preallocate",
        "restore",
    ]

    first_preallocate = calls[0][1][0]
    second_preallocate = calls[2][1][0]
    second_restore = calls[3][1][0]
    assert first_preallocate[2] == second_preallocate[2] == (1 << 32) - 1
    assert first_preallocate[4] == second_preallocate[4]
    assert first_preallocate[3] != second_preallocate[3]
    assert second_preallocate[6] == (
        residency_module._SLOT_DIRECT_NO_EVICT_FLAG
        | residency_module.UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
    )
    assert second_restore[6] == residency_module._SLOT_DIRECT_NO_EVICT_FLAG
    assert residency_module._SLOT_DIRECT_READY == {(1, 11, "second")}
    assert residency_module._SLOT_DIRECT_PREALLOCATED == {(1, 11, "second")}
    assert second["cache_after"] == {"bytes": PAGE_BYTES, "slots": 1}


def test_layer_restore_submits_all_experts_in_one_mode8_and_mode10_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_cpu_regions("e11_a", "e11_b", "e12_a", "e12_b")
    calls: list[tuple[str, list[tuple[int, ...]]]] = []

    def preallocate(entries: list[tuple[int, ...]], *, device: int):
        assert device == 0
        calls.append(("preallocate", list(entries)))
        return {"ok": True, "entry_statuses": [0] * len(entries)}

    def restore(entries: list[tuple[int, ...]], *, device: int):
        assert device == 0
        calls.append(("restore", list(entries)))
        return {"ok": True, "entry_statuses": [0] * len(entries)}

    monkeypatch.setattr(
        residency_module, "driver_slot_preallocate_ranges", preallocate
    )
    monkeypatch.setattr(residency_module, "driver_slot_restore_ranges", restore)

    result = residency_module.restore_uvm_expert_groups_to_slots(
        [(11, ["e11_a", "e11_b"]), (12, ["e12_a", "e12_b"])],
        layer_id=7,
    )

    assert result["ok"] is True
    assert result["expert_statuses"] == {"11": True, "12": True}
    assert [name for name, _entries in calls] == ["preallocate", "restore"]
    assert len(calls[0][1]) == len(calls[1][1]) == 4
    assert calls[0][1] == calls[1][1]
    assert {entry[3] for entry in calls[0][1]} == {11, 12}
    assert len({entry[4] for entry in calls[0][1]}) == 4
    assert residency_module._SLOT_DIRECT_READY == (
        residency_module._SLOT_DIRECT_PREALLOCATED
    )


def test_partial_layer_mode8_rebind_commits_exact_statuses_and_poisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_cpu_regions("old_a", "old_b", "new_a", "new_b")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", str(2 * PAGE_BYTES))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", "2")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "1")
    residency_module._SLOT_DIRECT_CACHE_LIMITS = None
    preallocate_calls = 0

    def preallocate(entries: list[tuple[int, ...]], *, device: int):
        nonlocal preallocate_calls
        preallocate_calls += 1
        if preallocate_calls == 1:
            return {"ok": True, "entry_statuses": [0, 0]}
        assert all(
            entry[6]
            == (
                residency_module._SLOT_DIRECT_NO_EVICT_FLAG
                | residency_module.UVM_MOE_SLOT_REMAP_FLAG_REUSE_BACKING_REQUIRED
            )
            for entry in entries
        )
        return {
            "ok": False,
            "reason": "synthetic_partial_layer_rebind",
            "entry_statuses": [0, 7],
            "backing_reuse_count": 1,
        }

    restore_calls = 0

    def restore(entries: list[tuple[int, ...]], *, device: int):
        nonlocal restore_calls
        restore_calls += 1
        return {"ok": True, "entry_statuses": [0] * len(entries)}

    monkeypatch.setattr(
        residency_module, "driver_slot_preallocate_ranges", preallocate
    )
    monkeypatch.setattr(residency_module, "driver_slot_restore_ranges", restore)
    monkeypatch.setattr(
        residency_module,
        "driver_slot_release_ranges",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("partial layer rebind must retain the physical bank")
        ),
    )

    first = residency_module.restore_uvm_expert_groups_to_slots(
        [(1, ["old_a"]), (2, ["old_b"])],
        layer_id=0,
    )
    second = residency_module.restore_uvm_expert_groups_to_slots(
        [(1, ["new_a"]), (2, ["new_b"])],
        layer_id=1,
    )

    assert first["ok"] is True
    assert second["ok"] is False
    assert second["reason"] == "slot_preallocate_failed"
    assert second["poisoned"] is True
    assert second["bank_rebind_slots"] == 2
    assert restore_calls == 1
    assert len(residency_module._SLOT_DIRECT_PREALLOCATED) == 2
    assert len(residency_module._SLOT_DIRECT_READY) == 1
    assert any(key[0] == 1 for key in residency_module._SLOT_DIRECT_PREALLOCATED)
    assert any(key[0] == 0 for key in residency_module._SLOT_DIRECT_PREALLOCATED)


def test_partial_fixed_bank_rebind_commits_only_success_and_poisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_cpu_regions("old_a", "old_b", "new_a", "new_b")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", str(2 * PAGE_BYTES))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", "2")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "1")
    residency_module._SLOT_DIRECT_CACHE_LIMITS = None
    preallocate_calls = 0

    def preallocate(entries: list[tuple[int, ...]], *, device: int):
        nonlocal preallocate_calls
        preallocate_calls += 1
        if preallocate_calls == 1:
            return {"ok": True, "entry_statuses": [0, 0]}
        return {
            "ok": False,
            "reason": "synthetic_partial_rebind",
            "entry_statuses": [0, 7],
            "backing_reuse_count": 1,
        }

    restore_calls = 0

    def restore(_entries: list[tuple[int, ...]], *, device: int):
        nonlocal restore_calls
        restore_calls += 1
        return {"ok": True}

    monkeypatch.setattr(
        residency_module, "driver_slot_preallocate_ranges", preallocate
    )
    monkeypatch.setattr(residency_module, "driver_slot_restore_ranges", restore)
    monkeypatch.setattr(
        residency_module,
        "driver_slot_release_ranges",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("partial rebind must not destroy retained backing")
        ),
    )

    assert residency_module.restore_uvm_regions_to_slots(
        ["old_a", "old_b"], layer_id=0, expert_id=1
    )["ok"]
    result = residency_module.restore_uvm_regions_to_slots(
        ["new_a", "new_b"], layer_id=1, expert_id=1
    )

    assert result["ok"] is False
    assert result["reason"] == "slot_preallocate_failed"
    assert result["rollback"] is None
    assert restore_calls == 1
    assert residency_module._SLOT_DIRECT_POISON_REASON is not None
    assert (1, 1, "new_a") in residency_module._SLOT_DIRECT_PREALLOCATED
    assert (1, 1, "new_b") not in residency_module._SLOT_DIRECT_PREALLOCATED
    assert (0, 1, "old_a") not in residency_module._SLOT_DIRECT_PREALLOCATED
    assert (0, 1, "old_b") in residency_module._SLOT_DIRECT_PREALLOCATED
    assert residency_module._SLOT_DIRECT_READY == {(0, 1, "old_b")}


def test_mode10_failure_after_rebind_retains_bank_and_poisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_cpu_regions("old", "new")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", str(PAGE_BYTES))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "1")
    residency_module._SLOT_DIRECT_CACHE_LIMITS = None
    monkeypatch.setattr(
        residency_module,
        "driver_slot_preallocate_ranges",
        lambda entries, **_kwargs: {
            "ok": True,
            "entry_statuses": [0] * len(entries),
            "backing_reuse_count": sum(entry[6] == 3 for entry in entries),
        },
    )
    restore_calls = 0

    def restore(_entries: list[tuple[int, ...]], *, device: int):
        nonlocal restore_calls
        restore_calls += 1
        return {
            "ok": restore_calls == 1,
            "reason": "ok" if restore_calls == 1 else "synthetic_mode10_failure",
        }

    monkeypatch.setattr(residency_module, "driver_slot_restore_ranges", restore)
    monkeypatch.setattr(
        residency_module,
        "driver_slot_release_ranges",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("mode10 failure must retain the physical bank")
        ),
    )

    assert residency_module.restore_uvm_regions_to_slots(
        ["old"], layer_id=0, expert_id=1
    )["ok"]
    result = residency_module.restore_uvm_regions_to_slots(
        ["new"], layer_id=1, expert_id=1
    )

    assert result["ok"] is False
    assert result["reason"] == "slot_restore_failed"
    assert result["rollback"] is None
    assert residency_module._SLOT_DIRECT_PREALLOCATED == {(1, 1, "new")}
    assert residency_module._SLOT_DIRECT_READY == set()
    assert residency_module._SLOT_DIRECT_POISON_REASON is not None


def test_fixed_bank_mode_is_locked_for_the_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_cpu_regions("weight")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "1")
    monkeypatch.setattr(
        residency_module,
        "driver_slot_preallocate_ranges",
        lambda entries, **_kwargs: {
            "ok": True,
            "entry_statuses": [0] * len(entries),
        },
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_restore_ranges",
        lambda *_args, **_kwargs: {"ok": True},
    )
    assert residency_module.restore_uvm_regions_to_slots(
        ["weight"], layer_id=0, expert_id=1
    )["ok"]

    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REUSE_BANK", "0")
    with pytest.raises(RuntimeError, match="cannot change during a slot session"):
        residency_module.restore_uvm_regions_to_slots(
            ["weight"], layer_id=0, expert_id=1
        )


def test_protected_working_set_fails_before_evicting_unconsumed_expert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_cpu_regions("first", "second")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CACHE_BYTES", str(PAGE_BYTES))
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_MAX_SLOTS", "1")
    residency_module._SLOT_DIRECT_CACHE_LIMITS = None
    monkeypatch.setattr(
        residency_module,
        "driver_slot_preallocate_ranges",
        lambda entries, **_kwargs: {
            "ok": True,
            "entry_statuses": [0] * len(entries),
        },
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_restore_ranges",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_release_ranges",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a protected expert must not be released")
        ),
    )
    assert residency_module.restore_uvm_regions_to_slots(
        ["first"], layer_id=7, expert_id=11
    )["ok"]

    with residency_module.protect_uvm_slot_direct_groups([(7, 11), (7, 12)]):
        second = residency_module.restore_uvm_regions_to_slots(
            ["second"], layer_id=7, expert_id=12
        )

    assert second["ok"] is False
    assert second["reason"] == "slot_cache_capacity_failed"
    assert residency_module._SLOT_DIRECT_READY == {(7, 11, "first")}


def test_partial_preallocation_is_tracked_and_can_be_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_cpu_regions("first", "second")
    monkeypatch.setattr(
        residency_module,
        "driver_slot_preallocate_ranges",
        lambda entries, **_kwargs: {
            "ok": False,
            "entry_statuses": [0, None],
            "reason": "synthetic_partial_failure",
        },
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_restore_ranges",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("restore must not follow partial preallocation")
        ),
    )
    released: list[list[tuple[int, ...]]] = []
    monkeypatch.setattr(
        residency_module,
        "driver_slot_release_ranges",
        lambda entries, **_kwargs: released.append(list(entries))
        or {"ok": True, "entry_statuses": [0], "mode": 19},
    )

    restore = residency_module.restore_uvm_regions_to_slots(
        ["first", "second"], layer_id=7, expert_id=11
    )
    assert restore["ok"] is False
    assert restore["rollback"]["ok"] is True
    assert len(released) == 1
    assert len(released[0]) == 1
    assert residency_module._SLOT_DIRECT_PREALLOCATED == set()
    assert residency_module._SLOT_DIRECT_BINDINGS == {}
    assert residency_module._SLOT_DIRECT_SLOT_OWNERS == {}


def test_slot_active_restore_keeps_ordinary_offloaded_membership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restored: list[tuple[int, int, int]] = []
    layer = SimpleNamespace(
        uvm_expert_offload_access_mode="slot_restore",
        _uvm_offloaded_global_expert_ids={11, 12},
        _uvm_layer_id=lambda: 7,
        _uvm_active_expert_pairs=lambda _ids: [(11, 0)],
        _uvm_restore_expert_records_to_slots=lambda **kwargs: restored.extend(
            (
                kwargs["layer_id"],
                global_expert_id,
                kwargs["device"],
            )
            for global_expert_id in kwargs["global_expert_ids"]
        )
        or {"ok": True},
    )
    monkeypatch.setattr(layer_module.torch.cuda, "is_available", lambda: False)

    FusedMoE._uvm_restore_active_experts(layer, layer_module.torch.tensor([11]))

    assert restored == [(7, 11, 0)]
    assert layer._uvm_offloaded_global_expert_ids == {11, 12}


def test_slot_active_restore_uses_stream_fence_only_for_complete_safe_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restored: list[int] = []
    synchronize_calls: list[int] = []
    layer = SimpleNamespace(
        uvm_expert_offload_access_mode="slot_restore",
        _uvm_offloaded_global_expert_ids={11},
        _uvm_layer_id=lambda: 7,
        _uvm_active_expert_pairs=lambda _ids: [(11, 0)],
        _uvm_restore_expert_records_to_slots=lambda **kwargs: restored.extend(
            int(value) for value in kwargs["global_expert_ids"]
        )
        or {"ok": True},
    )
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CUDA_FENCE", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT", "1")
    monkeypatch.setattr(layer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(layer_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        layer_module.torch.cuda,
        "synchronize",
        lambda device: synchronize_calls.append(int(device)),
    )

    FusedMoE._uvm_restore_active_experts(layer, layer_module.torch.tensor([11]))

    assert restored == [11]
    assert synchronize_calls == []


@pytest.mark.parametrize(
    "disabled_flag",
    [
        "VLLM_UVM_SLOT_DIRECT_CUDA_FENCE",
        "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN",
        "VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT",
    ],
)
def test_slot_active_restore_keeps_device_sync_when_fence_bundle_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    disabled_flag: str,
) -> None:
    layer = SimpleNamespace(
        uvm_expert_offload_access_mode="slot_restore",
        _uvm_offloaded_global_expert_ids={11},
        _uvm_layer_id=lambda: 7,
        _uvm_active_expert_pairs=lambda _ids: [(11, 0)],
        _uvm_restore_expert_records_to_slots=lambda **_kwargs: {"ok": True},
    )
    for name in (
        "VLLM_UVM_SLOT_DIRECT_CUDA_FENCE",
        "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN",
        "VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT",
    ):
        monkeypatch.setenv(name, "0" if name == disabled_flag else "1")
    synchronize_calls: list[int] = []
    monkeypatch.setattr(layer_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(layer_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        layer_module.torch.cuda,
        "synchronize",
        lambda device: synchronize_calls.append(int(device)),
    )

    FusedMoE._uvm_restore_active_experts(layer, layer_module.torch.tensor([11]))

    # The legacy path establishes an old-compute boundary before transition
    # and a new-mapping completion boundary afterwards.
    assert synchronize_calls == [0, 0]


def test_slot_formal_restore_selects_every_offloaded_cold_expert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restored: list[tuple[int, int, int]] = []
    layer = SimpleNamespace(
        uvm_expert_offload_access_mode="slot_restore",
        uvm_slot_formal_restore_all_cold=True,
        _uvm_offloaded_global_expert_ids={13, 11, 12},
        _uvm_layer_id=lambda: 7,
        _uvm_active_expert_pairs=lambda _ids: (_ for _ in ()).throw(
            AssertionError("formal selection must not consult routed active IDs")
        ),
        _uvm_restore_expert_records_to_slots=lambda **kwargs: restored.extend(
            (
                kwargs["layer_id"],
                global_expert_id,
                kwargs["device"],
            )
            for global_expert_id in kwargs["global_expert_ids"]
        )
        or {"ok": True},
    )
    monkeypatch.setattr(layer_module.torch.cuda, "is_available", lambda: False)

    FusedMoE._uvm_restore_active_experts(layer, layer_module.torch.tensor([11]))

    assert restored == [(7, 11, 0), (7, 12, 0), (7, 13, 0)]
    assert layer._uvm_offloaded_global_expert_ids == {11, 12, 13}


def test_slot_formal_restore_env_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        parallel_config=SimpleNamespace(
            enable_dbo=False,
            world_size_across_dp=1,
        ),
    )
    monkeypatch.delenv("VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD", raising=False)
    assert (
        layer_module._uvm_slot_formal_restore_all_cold_enabled(
            "restore", config
        )
        is False
    )

    monkeypatch.setenv("VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD", "1")
    with pytest.raises(RuntimeError, match="requires.*slot_restore"):
        layer_module._uvm_slot_formal_restore_all_cold_enabled("restore", config)
    assert (
        layer_module._uvm_slot_formal_restore_all_cold_enabled(
            "slot_restore", config
        )
        is True
    )

    config.model_config.enforce_eager = False
    with pytest.raises(RuntimeError, match="CUDA graph execution"):
        layer_module._uvm_slot_formal_restore_all_cold_enabled(
            "slot_restore", config
        )


def test_slot_formal_restore_is_identifiable_in_routing_trace(tmp_path) -> None:
    formal_trace = tmp_path / "formal.jsonl"
    layer = SimpleNamespace(
        uvm_moe_trace_enabled=True,
        uvm_moe_trace_file=str(formal_trace),
        _uvm_moe_trace_step=3,
        layer_name="model.layers.7.mlp.experts",
        top_k=1,
        global_num_experts=4,
        logical_num_experts=4,
        local_num_experts=4,
        enable_eplb=False,
        uvm_slot_formal_restore_all_cold=True,
        _uvm_offloaded_global_expert_ids={0, 2, 3},
    )
    topk_ids = layer_module.torch.tensor([[2]])
    topk_weights = layer_module.torch.tensor([[1.0]])
    hidden_states = layer_module.torch.zeros((1, 4))
    router_logits = layer_module.torch.zeros((1, 4))

    FusedMoE._log_uvm_moe_routing_trace(
        layer,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        hidden_states=hidden_states,
        router_logits=router_logits,
    )
    formal = json.loads(formal_trace.read_text(encoding="utf-8"))
    assert formal["slot_formal_restore_all_cold"] is True
    assert formal["slot_formal_restore_expert_count"] == 3

    default_trace = tmp_path / "default.jsonl"
    layer.uvm_moe_trace_file = str(default_trace)
    layer.uvm_slot_formal_restore_all_cold = False
    FusedMoE._log_uvm_moe_routing_trace(
        layer,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        hidden_states=hidden_states,
        router_logits=router_logits,
    )
    default = json.loads(default_trace.read_text(encoding="utf-8"))
    assert "slot_formal_restore_all_cold" not in default
    assert "slot_formal_restore_expert_count" not in default


def test_cross_thread_release_is_rejected_before_mode19(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_cpu_regions("weight")
    monkeypatch.setattr(
        residency_module,
        "driver_slot_preallocate_ranges",
        lambda entries, **_kwargs: {
            "ok": True,
            "entry_statuses": [0] * len(entries),
        },
    )
    monkeypatch.setattr(
        residency_module,
        "driver_slot_restore_ranges",
        lambda *_args, **_kwargs: {"ok": True},
    )
    release_calls: list[list[tuple[int, ...]]] = []
    monkeypatch.setattr(
        residency_module,
        "driver_slot_release_ranges",
        lambda entries, **_kwargs: release_calls.append(list(entries))
        or {"ok": True, "entry_statuses": [0] * len(entries), "mode": 19},
    )
    assert residency_module.restore_uvm_regions_to_slots(
        ["weight"], layer_id=7, expert_id=11
    )["ok"]

    results: list[dict[str, object]] = []

    def release_from_another_thread() -> None:
        results.append(
            residency_module.release_uvm_regions_from_slots(
                ["weight"], layer_id=7, expert_id=11
            )
        )

    thread = threading.Thread(target=release_from_another_thread)
    thread.start()
    thread.join()

    assert results[0]["ok"] is False
    assert results[0]["reason"] == "slot_direct_single_execution_thread_only"
    assert release_calls == []
    assert residency_module._SLOT_DIRECT_READY == {(7, 11, "weight")}

    cleanup = residency_module.release_uvm_regions_from_slots(
        ["weight"], layer_id=7, expert_id=11
    )
    assert cleanup["ok"] is True
    assert len(release_calls) == 1


def test_slot_restore_execution_contract_accepts_safe_config() -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        parallel_config=SimpleNamespace(
            enable_dbo=False,
            world_size_across_dp=1,
        ),
    )
    layer_module._validate_uvm_slot_restore_execution(config)


@pytest.mark.parametrize(
    ("enforce_eager", "enable_dbo", "world_size", "marker"),
    [
        (False, False, 1, "CUDA graph execution"),
        (True, True, 1, "dual-batch overlap"),
        (True, False, 2, "multi-GPU execution"),
    ],
)
def test_slot_restore_execution_contract_rejects_unsafe_config(
    enforce_eager: bool,
    enable_dbo: bool,
    world_size: int,
    marker: str,
) -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
        parallel_config=SimpleNamespace(
            enable_dbo=enable_dbo,
            world_size_across_dp=world_size,
        ),
    )
    with pytest.raises(RuntimeError, match=marker):
        layer_module._validate_uvm_slot_restore_execution(config)
