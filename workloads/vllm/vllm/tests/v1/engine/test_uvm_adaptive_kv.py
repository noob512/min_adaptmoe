# SPDX-License-Identifier: Apache-2.0

from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.device_allocator.uvm_region import RegionState
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.uvm_kv_runtime_policy import UvmKvRuntimePolicy
from vllm.v1.engine.core import EngineCore


class _FakeScheduler:
    def __init__(self, policy: Any) -> None:
        self.kv_cache_manager = SimpleNamespace(
            uvm_kv_runtime_policy=policy,
            block_pool=object(),
        )
        self.unfinished = False
        self.running: list[object] = []

    def has_unfinished_requests(self) -> bool:
        return bool(self.unfinished)


def _make_core(*, applied_target: int = 8, total_blocks: int = 32) -> EngineCore:
    core = object.__new__(EngineCore)
    policy = SimpleNamespace(
        _dynamic_pool_applied_target_blocks=applied_target,
        _dynamic_pool_reconciliation_block_ids=set(),
        _dynamic_pool_external_reconciliation_required=False,
        total_blocks=total_blocks,
        backup_reserve_blocks=0,
        dynamic_pool_initial_blocks=0,
        dynamic_pool_physical_migration_enable=True,
        dynamic_pool_active=lambda: True,
    )
    policy.get_dynamic_pool_reconciliation_block_ids = lambda: sorted(
        policy._dynamic_pool_reconciliation_block_ids
    )
    core.scheduler = _FakeScheduler(policy)  # type: ignore[assignment]
    policy.consume_backup_activation = lambda: {"blocks": 0, "events": 0}
    block_pool = SimpleNamespace(hash_block_size=16, active_blocks=0)

    def snapshot_uvm_dynamic_queues() -> dict[str, object]:
        active_ids = list(range(1, int(block_pool.active_blocks) + 1))
        max_active = max(active_ids, default=0)
        target = int(policy._dynamic_pool_applied_target_blocks)
        return {
            "schema": "test_uvm_kv_three_queue_snapshot_v1",
            "applied_target_blocks": target,
            "active_blocks": block_pool.active_blocks,
            "active_block_ids": active_ids,
            "active_block_ids_known": True,
            "max_active_block_id": max_active,
            "tail_free_start_block_id": min(max_active + 1, target + 1),
            "tail_free_block_count": max(target - max_active, 0),
            "gpu_free_blocks": max(target - block_pool.active_blocks, 0),
            "gpu_backup_blocks": 0,
            "cpu_offline_blocks": max(policy.total_blocks - target, 0),
            "invariant_ok": True,
        }

    block_pool.snapshot_uvm_dynamic_queues = snapshot_uvm_dynamic_queues
    core.scheduler.kv_cache_manager.block_pool = block_pool
    core.batch_queue = None
    core._uvm_pending_kv_grow_target_blocks = None
    core._uvm_pending_kv_grow_reason = ""
    return core


def _make_real_policy_core() -> tuple[EngineCore, UvmKvRuntimePolicy, BlockPool]:
    policy = object.__new__(UvmKvRuntimePolicy)
    policy.dynamic_pool_enable = True
    policy.dynamic_pool_target_blocks = 6
    policy.dynamic_pool_physical_migration_enable = True
    policy.dynamic_pool_physical_migration_sync = True
    policy.dynamic_pool_physical_migration_executor = "auto"
    policy.dynamic_pool_physical_migration_device = 0
    policy.num_kv_cache_groups = 1
    policy.total_blocks = 6
    policy._dynamic_pool_applied_target_blocks = 6
    policy._dynamic_pool_generation = 0
    policy._dynamic_pool_saved_lifecycle = {}
    policy._dynamic_pool_reconciliation_needed = False
    policy._dynamic_pool_reconciliation_block_ids = set()
    policy._dynamic_pool_external_reconciliation_required = False
    policy._logical_kv_blocks = {}
    pool = BlockPool(
        num_gpu_blocks=7,
        enable_caching=False,
        hash_block_size=16,
        uvm_kv_runtime_policy=policy,
    )
    policy.register_stage_l5_kv_blocks(pool)
    core = object.__new__(EngineCore)
    core.scheduler = _FakeScheduler(policy)  # type: ignore[assignment]
    core.scheduler.kv_cache_manager.block_pool = pool
    core.batch_queue = None
    core._uvm_pending_kv_grow_target_blocks = None
    core._uvm_pending_kv_grow_reason = ""
    return core, policy, pool


def _seed_partial_shrink(
    policy: UvmKvRuntimePolicy,
    pool: BlockPool,
) -> None:
    pool.apply_uvm_dynamic_pool_target(3, old_target_blocks=6)
    for block_id in (4, 5, 6):
        policy._mark_dynamic_pool_block_unknown(block=pool.blocks[block_id])
    policy.dynamic_pool_target_blocks = 3
    policy._dynamic_pool_reconciliation_needed = True
    policy._dynamic_pool_reconciliation_block_ids = {4, 5, 6}


def _worker_resize_result(
    *,
    ok: bool,
    made_progress: bool,
) -> dict[str, object]:
    return {"ok": ok, "made_progress": made_progress}


def _queue_ids(queue: Any) -> list[int]:
    return [block.block_id for block in queue.get_all_free_blocks()]


def _make_policy_from_env() -> UvmKvRuntimePolicy:
    kv_cache_config = SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(size=4096)],
        num_blocks=7,
        kv_cache_groups=[object()],
    )
    return UvmKvRuntimePolicy(kv_cache_config)  # type: ignore[arg-type]


def test_backup_pressure_snapshot_tracks_publish_and_consume() -> None:
    policy = _make_policy_from_env()

    policy.on_backup_blocks_activated(3, reason="qualification")
    pending = policy.snapshot_backup_activation()
    assert pending["pending_blocks"] == 3
    assert pending["pending_events"] == 1
    assert pending["total_activated_blocks"] == 3
    assert pending["total_consumed_blocks"] == 0

    assert policy.consume_backup_activation() == {"blocks": 3, "events": 1}
    consumed = policy.snapshot_backup_activation()
    assert consumed["pending_blocks"] == 0
    assert consumed["pending_events"] == 0
    assert consumed["total_activated_blocks"] == 3
    assert consumed["total_consumed_blocks"] == 3


def test_single_gpu_validation_catches_automatic_post_warmup_pool(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("VLLM_UVM_KV_RUNTIME_ENABLE", raising=False)
    monkeypatch.delenv("VLLM_UVM_KV_DYNAMIC_POOL_ENABLE", raising=False)
    monkeypatch.setenv("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "3")
    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE", "post_warmup")
    policy = _make_policy_from_env()
    config = SimpleNamespace(parallel_config=SimpleNamespace(world_size=2))

    assert policy.dynamic_pool_active()
    with pytest.raises(RuntimeError, match="supports one GPU only"):
        EngineCore._validate_uvm_dynamic_pool_single_gpu(config, policy)  # type: ignore[arg-type]


def test_single_gpu_validation_allows_runtime_disabled_policy(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "3")
    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE", "post_warmup")
    monkeypatch.setenv("VLLM_UVM_KV_RUNTIME_ENABLE", "0")
    monkeypatch.setenv("VLLM_UVM_KV_DYNAMIC_POOL_ENABLE", "1")
    policy = _make_policy_from_env()
    config = SimpleNamespace(parallel_config=SimpleNamespace(world_size=2))

    assert not policy.dynamic_pool_active()
    EngineCore._validate_uvm_dynamic_pool_single_gpu(config, policy)  # type: ignore[arg-type]


def test_preemption_grows_per_preempted_request(
    monkeypatch: Any,
) -> None:
    core = _make_core()
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE", "1")
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_GROW_BLOCKS_PER_PREEMPTION", "3")
    monkeypatch.setenv("VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE", "1")
    calls: list[tuple[int, str, bool]] = []

    def fake_resize(
        target_blocks: int,
        *,
        reason: str,
        release_expert_wave: bool,
    ) -> dict[str, object]:
        calls.append((target_blocks, reason, release_expert_wave))
        return {"ok": True}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", fake_resize)
    output = SimpleNamespace(preempted_req_ids={"r1", "r2"})

    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]

    assert calls == [(14, "scheduler_preemption_pressure", True)]


def test_backup_activation_grows_kv_and_releases_expert_wave(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=8)
    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
    policy.backup_grow_blocks = 6
    policy.consume_backup_activation = lambda: {"blocks": 2, "events": 1}
    calls: list[tuple[int, str, bool]] = []

    def fake_resize(
        target_blocks: int,
        *,
        reason: str,
        release_expert_wave: bool,
    ) -> dict[str, object]:
        calls.append((target_blocks, reason, release_expert_wave))
        return {"ok": True}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", fake_resize)
    output = SimpleNamespace(preempted_req_ids=set())

    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]

    assert calls == [(14, "backup_reserve_activation", True)]


def test_standalone_backup_growth_can_be_deferred_until_autoscale_is_enabled(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=8)
    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
    monkeypatch.setenv(
        "VLLM_UVM_KV_BACKUP_PRESSURE_STANDALONE_GROW_ENABLE", "0"
    )
    consume_calls = 0

    def consume_backup_activation() -> dict[str, int]:
        nonlocal consume_calls
        consume_calls += 1
        return {"blocks": 2, "events": 1}

    policy.consume_backup_activation = consume_backup_activation
    monkeypatch.setattr(
        core,
        "_resize_uvm_kv_target",
        lambda *_args, **_kwargs: pytest.fail(
            "standalone backup growth must remain deferred"
        ),
    )

    core._maybe_handle_uvm_kv_preemption_pressure(  # type: ignore[arg-type]
        SimpleNamespace(preempted_req_ids=set())
    )

    assert consume_calls == 0


def test_preemption_waits_for_batch_queue_to_drain(
    monkeypatch: Any,
) -> None:
    core = _make_core()
    core.batch_queue = deque([(object(), object())])  # type: ignore[assignment]
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE", "1")
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_GROW_BLOCKS_PER_PREEMPTION", "4")
    monkeypatch.setenv("VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE", "1")
    calls: list[tuple[int, str, bool]] = []

    def fake_resize(
        target_blocks: int,
        *,
        reason: str,
        release_expert_wave: bool,
    ) -> dict[str, object]:
        calls.append((target_blocks, reason, release_expert_wave))
        return {"ok": True}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", fake_resize)
    output = SimpleNamespace(preempted_req_ids={"r1"})

    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]
    assert calls == []
    assert core._uvm_pending_kv_grow_target_blocks == 12
    assert core._apply_pending_uvm_kv_growth_if_ready() is None

    core.batch_queue.clear()
    result = core._apply_pending_uvm_kv_growth_if_ready()
    assert result == {"ok": True}
    assert calls == [
        (
            12,
            "scheduler_preemption_pressure_after_batch_queue_drain",
            True,
        )
    ]
    assert core._uvm_pending_kv_grow_target_blocks is None


@pytest.mark.parametrize(
    ("prompt_tokens", "max_tokens", "expected_class", "expected_target"),
    [
        ([10], 6, "low", 6),
        ([80, 80], 16, "high", 24),
    ],
)
def test_next_request_pressure_dynamic_pool_selects_target(
    monkeypatch: Any,
    prompt_tokens: list[int],
    max_tokens: int,
    expected_class: str,
    expected_target: int,
) -> None:
    core = _make_core(applied_target=8, total_blocks=32)
    monkeypatch.setenv("VLLM_UVM_KV_NEXT_REQUEST_LOW_TARGET_BLOCKS", "6")
    monkeypatch.setenv("VLLM_UVM_KV_NEXT_REQUEST_HIGH_TARGET_BLOCKS", "24")
    monkeypatch.setenv(
        "VLLM_UVM_KV_NEXT_REQUEST_PRESSURE_THRESHOLD_BLOCKS", "10"
    )
    calls: list[tuple[int, str, bool]] = []

    def resize(
        target_blocks: int,
        *,
        reason: str,
        release_expert_wave: bool,
    ) -> dict[str, object]:
        calls.append((target_blocks, reason, release_expert_wave))
        return {"ok": True, "reason": "applied"}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", resize)

    result = core.apply_uvm_kv_next_request_pressure(
        prompt_tokens,
        max_tokens,
    )

    assert result["ok"] is True
    pressure = result["pressure"]
    assert pressure["decision_source"] == "engine_next_request_pressure_policy"
    assert pressure["pressure_class"] == expected_class
    assert pressure["new_target_blocks"] == expected_target
    assert pressure["next_prompt_tokens"] == sum(prompt_tokens)
    assert calls == [
        (expected_target, "next_request_pressure_policy", False)
    ]


def test_legacy_next_request_pressure_requires_idle_boundary(
    monkeypatch: Any,
) -> None:
    core = _make_core()
    monkeypatch.setattr(core.scheduler, "has_unfinished_requests", lambda: True)
    monkeypatch.setattr(
        core,
        "_resize_uvm_kv_target",
        lambda *_args, **_kwargs: pytest.fail("non-idle resize was submitted"),
    )

    assert core.apply_uvm_kv_next_request_pressure([32], 8) == {
        "ok": False,
        "reason": "request_boundary_not_idle",
    }


def test_autoscale_late_request_pressure_defers_growth_until_idle(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=8, total_blocks=32)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS",
        "8,16,32",
    )
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "8")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "32")
    core.scheduler.unfinished = True
    core.scheduler.running.append(object())
    core._ensure_uvm_kv_autoscale_state()
    core._uvm_pending_kv_shrink_target_blocks = 4
    core._uvm_pending_kv_shrink_reason = "stale_low_pressure"
    resize_calls: list[int] = []

    def resize(target_blocks: int, **_kwargs: object) -> dict[str, object]:
        resize_calls.append(target_blocks)
        return {"ok": True, "reason": "applied"}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", resize)

    result = core.apply_uvm_kv_next_request_pressure([160, 160], 16)

    assert result["ok"] is True
    assert result["pressure"]["scheduler_unfinished"] is True
    assert result["autoscale"]["pending"] is True
    assert result["autoscale"]["reason"] == (
        "growth_deferred_until_request_boundary"
    )
    assert core._uvm_pending_kv_grow_target_blocks == 32
    assert core._uvm_pending_kv_shrink_target_blocks is None
    assert core._apply_pending_uvm_kv_growth_if_ready() is None
    assert resize_calls == []

    core.scheduler.running.clear()
    core.scheduler.unfinished = False
    committed = core._apply_pending_uvm_kv_growth_if_ready()

    assert committed is not None and committed["ok"] is True
    assert resize_calls == [32]
    assert core._uvm_pending_kv_grow_target_blocks is None


def test_autoscale_late_request_pressure_grows_at_scheduler_boundary(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=8, total_blocks=32)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_KV_SCHEDULER_BOUNDARY_RESIZE_ENABLE", "1")
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS",
        "8,16,32",
    )
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "8")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "32")
    core.scheduler.unfinished = True
    core.scheduler.running.append(object())
    resize_calls: list[int] = []

    def resize(target_blocks: int, **_kwargs: object) -> dict[str, object]:
        resize_calls.append(target_blocks)
        return {"ok": True, "reason": "applied"}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", resize)
    result = core.apply_uvm_kv_next_request_pressure([160, 160], 16)

    assert result["ok"] is True
    assert result["autoscale"]["pending"] is False
    assert resize_calls == [32]
    assert core._uvm_pending_kv_grow_target_blocks is None


def test_pending_growth_waits_for_inflight_model_future(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=8, total_blocks=32)
    monkeypatch.setenv("VLLM_UVM_KV_SCHEDULER_BOUNDARY_RESIZE_ENABLE", "1")
    core.scheduler.unfinished = True
    core.scheduler.running.append(object())
    core._uvm_pending_kv_grow_target_blocks = 16
    core._uvm_pending_kv_grow_reason = "test_online_growth"
    core._uvm_model_execution_inflight = True
    resize_calls: list[int] = []
    monkeypatch.setattr(
        core,
        "_resize_uvm_kv_target",
        lambda target_blocks, **_kwargs: (
            resize_calls.append(target_blocks) or {"ok": True}
        ),
    )

    assert core._apply_pending_uvm_kv_growth_if_ready() is None
    assert resize_calls == []

    core._uvm_model_execution_inflight = False
    result = core._apply_pending_uvm_kv_growth_if_ready()
    assert result == {"ok": True}
    assert resize_calls == [16]


def test_autoscale_allowed_tier_uses_smallest_capacity_fitting_target(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=1364, total_blocks=4095)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS",
        "1364,2730,4095",
    )
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "1364")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "4095")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_HIGH_WATERMARK", "0.8")
    targets: list[int] = []
    monkeypatch.setattr(
        core,
        "_resize_uvm_kv_target",
        lambda target_blocks, **_kwargs: (
            targets.append(target_blocks) or {"ok": True, "reason": "applied"}
        ),
    )

    result = core._request_uvm_kv_autoscale_growth(
        required_blocks=2597,
        reason="next_request_autoscale_prediction",
        urgent=True,
    )

    assert result["ok"] is True
    assert result["new_target_blocks"] == 2730
    assert targets == [2730]


def test_synchronized_growth_commits_each_adjacent_kv_expert_tier(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=1364, total_blocks=4095)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS",
        "1364,2730,4095",
    )
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "1364")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "4095")
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_SYNCHRONIZED_TIER_STEPS",
        "1",
    )
    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
    targets: list[int] = []

    def resize(target_blocks: int, **_kwargs: object) -> dict[str, object]:
        targets.append(target_blocks)
        old_target = int(policy._dynamic_pool_applied_target_blocks)
        policy._dynamic_pool_applied_target_blocks = target_blocks
        return {
            "ok": True,
            "reason": "applied",
            "old_target_blocks": old_target,
            "plan_class_before_grow": {"ok": True},
        }

    monkeypatch.setattr(core, "_resize_uvm_kv_target", resize)
    result = core._request_uvm_kv_autoscale_growth(
        required_blocks=3000,
        reason="synchronized_kv_expert_growth",
        urgent=True,
    )

    assert result["ok"] is True
    assert result["reason"] == "applied_synchronized_tier_steps"
    assert result["new_target_blocks"] == 4095
    assert result["applied_target_blocks"] == 4095
    assert targets == [2730, 4095]
    assert [
        (step["old_target_blocks"], step["new_target_blocks"])
        for step in result["tier_steps"]
    ] == [(1364, 2730), (2730, 4095)]
    assert core._uvm_kv_autoscale_pending_plan_class is None


def test_synchronized_tier_guard_rejects_direct_two_tier_jump(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=1364, total_blocks=4095)
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS",
        "1364,2730,4095",
    )
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "1364")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "4095")
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_SYNCHRONIZED_TIER_STEPS",
        "1",
    )

    result = core._resize_uvm_kv_target(
        4095,
        reason="invalid_direct_jump",
        release_expert_wave=False,
    )

    assert result["ok"] is False
    assert result["reason"] == "non_adjacent_synchronized_tier_resize"
    assert result["old_target_blocks"] == 1364
    assert result["new_target_blocks"] == 4095


def test_autoscale_nonurgent_active_pressure_does_not_queue_stale_growth(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=2730, total_blocks=4095)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS",
        "1364,2730,4095",
    )
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "1364")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "4095")
    core.scheduler.unfinished = True
    core.scheduler.running.append(object())
    core._uvm_kv_autoscale_step = 20
    core._uvm_kv_autoscale_last_resize_step = 1
    monkeypatch.setattr(
        core,
        "_resize_uvm_kv_target",
        lambda *_args, **_kwargs: pytest.fail("stale grow must not resize"),
    )

    result = core._request_uvm_kv_autoscale_growth(
        required_blocks=2597,
        reason="autoscale_high_watermark_growth",
        urgent=False,
    )

    assert result["ok"] is True
    assert result["reason"] == "active_cohort_fits_current_target"
    assert result["new_target_blocks"] == 2730
    assert result["pending"] is False
    assert core._uvm_pending_kv_grow_target_blocks is None


@pytest.mark.parametrize("inflight", [False, True])
def test_proactive_high_watermark_growth_obeys_safe_boundary(
    monkeypatch: Any,
    inflight: bool,
) -> None:
    core = _make_core(applied_target=2730, total_blocks=4095)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS", "1364,2730,4095")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "1364")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "4095")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_ACTIVE_HIGH_WATERMARK_GROW", "1")
    monkeypatch.setenv("VLLM_UVM_KV_SCHEDULER_BOUNDARY_RESIZE_ENABLE", "1")
    core.scheduler.unfinished = True
    core.scheduler.running.append(object())
    core._uvm_model_execution_inflight = inflight
    core._uvm_kv_autoscale_step = 20
    core._uvm_kv_autoscale_last_resize_step = 1
    targets: list[int] = []
    monkeypatch.setattr(core, "_resize_uvm_kv_target",
                        lambda target, **_kwargs: targets.append(target) or {"ok": True})

    result = core._request_uvm_kv_autoscale_growth(
        required_blocks=2200,
        reason="autoscale_high_watermark_growth",
        urgent=False,
    )

    assert result["ok"] is True
    assert result["new_target_blocks"] == 4095
    assert result["pending"] is inflight
    assert targets == ([] if inflight else [4095])


def test_proactive_actual_pressure_grows_k2_k4_k6_separately(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=1364, total_blocks=4095)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS", "1364,2730,4095")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "1364")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "4095")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_HIGH_WATERMARK", "0.8")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_ACTIVE_HIGH_WATERMARK_GROW", "1")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_SYNCHRONIZED_TIER_STEPS", "1")
    monkeypatch.setenv("VLLM_UVM_KV_SCHEDULER_BOUNDARY_RESIZE_ENABLE", "1")
    core.scheduler.unfinished = True
    core.scheduler.running.append(object())
    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
    pool = core.scheduler.kv_cache_manager.block_pool
    targets: list[int] = []

    def resize(target: int, **_kwargs: object) -> dict[str, object]:
        targets.append(target)
        policy._dynamic_pool_applied_target_blocks = target
        return {"ok": True, "plan_class_before_grow": {"ok": True}}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", resize)
    for used_blocks, expected in [(1000, []), (1100, [2730]),
                                  (2000, [2730]), (2200, [2730, 4095])]:
        pool.active_blocks = used_blocks
        core._observe_uvm_kv_autoscale_pressure(_autoscale_scheduler_output())
        assert targets == expected
    assert policy._dynamic_pool_applied_target_blocks == 4095


def _autoscale_scheduler_output(
    *,
    scheduled_tokens: int = 1,
    preemptions: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        total_num_scheduled_tokens=scheduled_tokens,
        preempted_req_ids={f"r{index}" for index in range(preemptions)},
    )


def _enable_test_autoscale(monkeypatch: Any) -> None:
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_ENABLE", "1")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "4")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "64")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_LOW_PRESSURE_STREAK", "3")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_GROW_FACTOR", "2")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_SHRINK_DIVISOR", "2")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_RESIDENCY_STEPS", "0")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_SAFETY_BLOCKS", "0")


def test_unified_autoscale_requires_three_low_rounds_then_halves(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=16, total_blocks=64)
    _enable_test_autoscale(monkeypatch)
    block_pool = core.scheduler.kv_cache_manager.block_pool
    block_pool.active_blocks = 4
    resize_calls: list[tuple[int, str, bool, bool]] = []

    def resize(
        target_blocks: int,
        *,
        reason: str,
        release_expert_wave: bool,
        allow_active_prefix_shrink: bool = False,
    ) -> dict[str, object]:
        resize_calls.append(
            (
                target_blocks,
                reason,
                release_expert_wave,
                allow_active_prefix_shrink,
            )
        )
        return {"ok": True, "reason": "applied"}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", resize)
    output = _autoscale_scheduler_output()

    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]
    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]
    assert core._uvm_kv_autoscale_low_streak == 2
    assert core._uvm_pending_kv_shrink_target_blocks is None
    assert resize_calls == []

    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]
    assert core._uvm_pending_kv_shrink_target_blocks == 8
    assert resize_calls == []

    core.scheduler.unfinished = True
    committed = core._apply_pending_uvm_kv_shrink_if_ready()
    assert committed is not None and committed["ok"] is True
    assert committed["new_target_blocks"] == 8
    assert resize_calls == [
        (8, "autoscale_sustained_low_pressure", False, True)
    ]
    assert core._uvm_kv_autoscale_low_streak == 0
    assert core._uvm_pending_kv_shrink_target_blocks is None


def test_minimum_free_id_does_not_prove_tail_is_free(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=6, total_blocks=12)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "3")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "12")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_LOW_PRESSURE_STREAK", "1")
    block_pool = core.scheduler.kv_cache_manager.block_pool
    block_pool.snapshot_uvm_dynamic_queues = lambda: {
        "schema": "test_sparse_active_ids",
        "applied_target_blocks": 6,
        "active_blocks": 2,
        "active_block_ids": [1, 6],
        "active_block_ids_known": True,
        "max_active_block_id": 6,
        "tail_free_start_block_id": 7,
        "tail_free_block_count": 0,
        "min_gpu_free_block_id": 2,
        "gpu_free_blocks": 4,
        "gpu_backup_blocks": 0,
        "cpu_offline_blocks": 6,
        "invariant_ok": True,
    }

    decision = core._observe_uvm_kv_autoscale_pressure(
        _autoscale_scheduler_output()  # type: ignore[arg-type]
    )

    assert decision["candidate_shrink_target_blocks"] == 3
    assert decision["queue_snapshot"]["min_gpu_free_block_id"] == 2
    assert decision["candidate_tail_all_free"] is False
    assert decision["action"] == "hold"
    assert core._uvm_pending_kv_shrink_target_blocks is None


def test_pending_shrink_is_cancelled_if_tail_becomes_active(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=16, total_blocks=64)
    _enable_test_autoscale(monkeypatch)
    core._ensure_uvm_kv_autoscale_state()
    core._uvm_pending_kv_shrink_target_blocks = 8
    core._uvm_pending_kv_shrink_reason = "stale_low_pressure"
    block_pool = core.scheduler.kv_cache_manager.block_pool
    block_pool.snapshot_uvm_dynamic_queues = lambda: {
        "schema": "test_tail_reallocated",
        "applied_target_blocks": 16,
        "active_blocks": 2,
        "active_block_ids": [1, 12],
        "active_block_ids_known": True,
        "max_active_block_id": 12,
        "tail_free_start_block_id": 13,
        "tail_free_block_count": 4,
        "gpu_free_blocks": 14,
        "gpu_backup_blocks": 0,
        "cpu_offline_blocks": 48,
        "invariant_ok": True,
    }
    monkeypatch.setattr(
        core,
        "_resize_uvm_kv_target",
        lambda *_args, **_kwargs: pytest.fail("stale shrink must not migrate"),
    )

    result = core._apply_pending_uvm_kv_shrink_if_ready()

    assert result is not None
    assert result["ok"] is False
    assert result["reason"] == "candidate_tail_no_longer_free"
    assert result["pending"] is False
    assert core._uvm_pending_kv_shrink_target_blocks is None


def test_unified_autoscale_pressure_cancels_low_streak_and_doubles(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=16, total_blocks=64)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE", "1")
    block_pool = core.scheduler.kv_cache_manager.block_pool
    block_pool.active_blocks = 4
    output = _autoscale_scheduler_output()
    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]
    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]
    assert core._uvm_kv_autoscale_low_streak == 2

    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
    policy.consume_backup_activation = lambda: {"blocks": 1, "events": 1}
    calls: list[tuple[int, str, bool]] = []

    def resize(
        target_blocks: int,
        *,
        reason: str,
        release_expert_wave: bool,
    ) -> dict[str, object]:
        calls.append((target_blocks, reason, release_expert_wave))
        return {"ok": True, "reason": "applied"}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", resize)
    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]

    assert core._uvm_kv_autoscale_low_streak == 0
    assert core._uvm_pending_kv_shrink_target_blocks is None
    assert calls == [(32, "autoscale_backup_or_preemption_growth", True)]


def test_unified_autoscale_repeatedly_halves_to_lower_bound(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=32, total_blocks=64)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "8")
    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
    core.scheduler.kv_cache_manager.block_pool.active_blocks = 4
    targets: list[int] = []

    def resize(target_blocks: int, **_kwargs: object) -> dict[str, object]:
        targets.append(target_blocks)
        policy._dynamic_pool_applied_target_blocks = target_blocks
        return {"ok": True, "reason": "applied"}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", resize)
    output = _autoscale_scheduler_output()
    for _ in range(3):
        core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]
    first_shrink = core._apply_pending_uvm_kv_shrink_if_ready()
    assert first_shrink is not None and first_shrink["ok"] is True
    for _ in range(3):
        core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]
    second_shrink = core._apply_pending_uvm_kv_shrink_if_ready()
    assert second_shrink is not None and second_shrink["ok"] is True
    for _ in range(3):
        core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]

    assert targets == [16, 8]
    assert core._uvm_pending_kv_shrink_target_blocks is None


def test_unified_autoscale_uses_previous_allowed_plan_class_for_shrink(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=2730, total_blocks=4095)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "1364")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "4095")
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS",
        "1364,2730,4095",
    )
    core.scheduler.kv_cache_manager.block_pool.active_blocks = 4
    output = _autoscale_scheduler_output()

    for _ in range(3):
        core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]

    decision = core._uvm_kv_autoscale_last_decision
    assert decision["candidate_shrink_target_blocks"] == 1364
    assert core._uvm_pending_kv_shrink_target_blocks == 1364


def test_plan_class_shrink_waits_for_global_idle_boundary(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=2730, total_blocks=4095)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "1364")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "4095")
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS",
        "1364,2730,4095",
    )
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_MAP",
        '{"1364":"cold70","2730":"cold80","4095":"cold89"}',
    )
    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
    core.scheduler.kv_cache_manager.block_pool.active_blocks = 4
    resize_calls: list[int] = []
    activated: list[str] = []

    def resize(target_blocks: int, **_kwargs: object) -> dict[str, object]:
        resize_calls.append(target_blocks)
        policy._dynamic_pool_applied_target_blocks = target_blocks
        return {"ok": True, "reason": "applied"}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", resize)
    monkeypatch.setattr(
        core,
        "activate_uvm_expert_graph_class",
        lambda class_name: activated.append(class_name)
        or {"ok": True, "reason": "class_activated"},
    )
    output = _autoscale_scheduler_output()
    for _ in range(3):
        core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]

    core.scheduler.unfinished = True
    assert core._apply_pending_uvm_kv_shrink_if_ready() is None
    assert resize_calls == []
    assert core._uvm_pending_kv_shrink_target_blocks == 1364

    core.scheduler.unfinished = False
    committed = core._apply_pending_uvm_kv_shrink_if_ready()
    assert committed is not None and committed["ok"] is True
    assert resize_calls == [1364]
    assert activated == ["cold70"]
    assert core._uvm_pending_kv_shrink_target_blocks is None


def test_plan_class_shrink_runs_with_active_request_at_scheduler_boundary(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=2730, total_blocks=4095)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_KV_SCHEDULER_BOUNDARY_RESIZE_ENABLE", "1")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "1364")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "4095")
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS",
        "1364,2730,4095",
    )
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_MAP",
        '{"1364":"cold70","2730":"cold80","4095":"cold89"}',
    )
    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
    core.scheduler.kv_cache_manager.block_pool.active_blocks = 4
    core.scheduler.unfinished = True
    core.scheduler.running.append(object())
    resize_calls: list[int] = []
    activated: list[str] = []

    def resize(target_blocks: int, **_kwargs: object) -> dict[str, object]:
        resize_calls.append(target_blocks)
        policy._dynamic_pool_applied_target_blocks = target_blocks
        return {"ok": True, "reason": "applied"}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", resize)
    monkeypatch.setattr(
        core,
        "_activate_uvm_expert_graph_class_at_quiescent_boundary",
        lambda class_name, **_kwargs: activated.append(class_name)
        or {"ok": True, "reason": "class_activated"},
    )
    output = _autoscale_scheduler_output()
    for _ in range(3):
        core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]

    committed = core._apply_pending_uvm_kv_shrink_if_ready()

    assert committed is not None and committed["ok"] is True
    assert resize_calls == [1364]
    assert activated == ["cold70"]
    assert core.scheduler.running
    assert core._uvm_pending_kv_shrink_target_blocks is None


def test_unified_autoscale_growth_stops_at_upper_bound(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=32, total_blocks=64)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "40")
    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
    policy.consume_backup_activation = lambda: {"blocks": 1, "events": 1}
    targets: list[int] = []

    def resize(target_blocks: int, **_kwargs: object) -> dict[str, object]:
        targets.append(target_blocks)
        policy._dynamic_pool_applied_target_blocks = target_blocks
        return {"ok": True, "reason": "applied"}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", resize)
    output = _autoscale_scheduler_output()
    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]
    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]

    assert targets == [40]
    assert core._uvm_kv_autoscale_last_decision["growth"]["reason"] == (
        "already_at_autoscale_upper_bound"
    )


def test_backup_only_autoscale_waits_for_pressure_threshold(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=24, total_blocks=64)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "40")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_BACKUP_ONLY", "1")
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_BACKUP_PRESSURE_THRESHOLD_BLOCKS",
        "4",
    )
    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
    policy.backup_reserve_blocks = 8
    pressures = iter((3, 1))
    policy.consume_backup_activation = lambda: {
        "blocks": next(pressures),
        "events": 1,
    }
    block_pool = core.scheduler.kv_cache_manager.block_pool
    remaining = iter((5, 4))
    block_pool.snapshot_uvm_dynamic_queues = lambda: {
        "schema": "test_uvm_kv_three_queue_snapshot_v1",
        "applied_target_blocks": 24,
        "active_blocks": 20,
        "gpu_free_blocks": 0,
        "backup_target_blocks": 8,
        "gpu_backup_blocks": next(remaining),
        "cpu_offline_blocks": 40,
        "invariant_ok": True,
    }
    core._uvm_kv_autoscale_predicted_blocks = 100
    core._uvm_kv_autoscale_prediction_error_blocks = 10.0
    calls: list[tuple[int, str, bool]] = []

    def resize(
        target_blocks: int,
        *,
        reason: str,
        release_expert_wave: bool,
    ) -> dict[str, object]:
        calls.append((target_blocks, reason, release_expert_wave))
        return {"ok": True, "reason": "applied"}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", resize)
    output = _autoscale_scheduler_output(preemptions=2)

    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]
    first = core._uvm_kv_autoscale_last_decision
    assert first["growth_trigger_policy"] == "backup_threshold_only"
    assert first["predicted_blocks"] == 0
    assert first["prediction_error_blocks"] == 0
    assert first["backup_pressure_level_blocks"] == 3
    assert first["backup_pressure_threshold_reached"] is False
    assert calls == []

    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]
    second = core._uvm_kv_autoscale_last_decision
    assert second["backup_pressure_level_blocks"] == 4
    assert second["backup_pressure_threshold_reached"] is True
    assert second["action"] == "grow"
    assert calls == [(40, "autoscale_backup_threshold_growth", False)]


def test_e10_autoscale_geometry_hits_exact_four_and_six_gib_targets(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=1364, total_blocks=5460)
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "1364")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "4095")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_GROW_FACTOR", "2")
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_RESIZE_GRANULARITY_BLOCKS", "1365"
    )
    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
    targets: list[int] = []

    def resize(target_blocks: int, **_kwargs: object) -> dict[str, object]:
        targets.append(target_blocks)
        policy._dynamic_pool_applied_target_blocks = target_blocks
        return {"ok": True, "reason": "applied"}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", resize)
    core._request_uvm_kv_autoscale_growth(
        required_blocks=1368,
        reason="e10_test",
        urgent=True,
    )
    core._request_uvm_kv_autoscale_growth(
        required_blocks=2734,
        reason="e10_test",
        urgent=True,
    )

    assert targets == [2730, 4095]


def test_e10_autoscale_frozen_targets_hit_exact_two_four_eight_ten_gib(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=1364, total_blocks=6826)
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "1364")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "6826")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_GROW_FACTOR", "2")
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_RESIZE_GRANULARITY_BLOCKS", "1365"
    )
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS",
        "1364,2730,5460,6826",
    )
    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
    targets: list[int] = []

    def resize(target_blocks: int, **_kwargs: object) -> dict[str, object]:
        targets.append(target_blocks)
        policy._dynamic_pool_applied_target_blocks = target_blocks
        return {"ok": True, "reason": "applied"}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", resize)
    for required_blocks in (1368, 2734, 5464):
        core._request_uvm_kv_autoscale_growth(
            required_blocks=required_blocks,
            reason="e10_frozen_tier_test",
            urgent=True,
        )

    assert targets == [2730, 5460, 6826]
    config = core._uvm_kv_autoscale_config(policy)
    assert config["minimum_blocks"] == 1364
    assert config["maximum_blocks"] == 6826
    assert config["allowed_target_blocks"] == [1364, 2730, 5460, 6826]


def test_online_growth_switches_frozen_plan_class_before_physical_kv(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=24, total_blocks=40)
    core.scheduler.unfinished = True
    core.scheduler.running.append(object())
    monkeypatch.setenv("VLLM_UVM_KV_SCHEDULER_BOUNDARY_RESIZE_ENABLE", "1")
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_MAP",
        '{"24": "cold29", "40": "cold30"}',
    )
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_BEFORE_GROW",
        "1",
    )
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_RESUME_PREFILL_BEFORE_PLAN_CLASS",
        "1",
    )
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW", "1")
    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
    events: list[str] = []

    def prepare(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("prepare")
        return {"ok": True}

    def finalize(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("finalize")
        return {"ok": True}

    policy.prepare_dynamic_pool_resize = prepare
    policy.finalize_dynamic_pool_resize = finalize

    def rpc(method: str, *, args: tuple[object, ...]) -> dict[str, object]:
        if method == "activate_uvm_expert_graph_class":
            assert args == ("cold30", True, True, True)
            events.append(f"activate:{args[0]}")
            return {"ok": True, "class_name": args[0]}
        assert method == "resize_uvm_kv_cache_target"
        assert args == (24, 40, [])
        events.append("resize")
        return {"ok": True, "made_progress": True}

    monkeypatch.setattr(core, "_uvm_single_worker_rpc", rpc)
    result = core._resize_uvm_kv_target(
        40,
        reason="backup_threshold",
        release_expert_wave=True,
    )

    assert result["ok"] is True
    assert result["plan_class_before_grow"] == {
        "ok": True,
        "class_name": "cold30",
    }
    assert events == ["activate:cold30", "prepare", "resize", "finalize"]


def test_synchronous_growth_class_is_not_requeued(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=24, total_blocks=40)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "40")
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_MAP",
        '{"24": "cold29", "40": "cold30"}',
    )

    def resize(
        target_blocks: int,
        *,
        reason: str,
        release_expert_wave: bool,
    ) -> dict[str, object]:
        assert target_blocks == 40
        assert reason == "backup_threshold"
        assert release_expert_wave is False
        return {
            "ok": True,
            "reason": "applied",
            "plan_class_before_grow": {
                "ok": True,
                "class_name": "cold30",
            },
        }

    monkeypatch.setattr(core, "_resize_uvm_kv_target", resize)
    result = core._request_uvm_kv_autoscale_growth(
        required_blocks=24,
        reason="backup_threshold",
        urgent=True,
    )

    assert result["ok"] is True
    assert core._uvm_kv_autoscale_pending_plan_class is None


def test_autoscale_bounds_are_aligned_to_resize_granularity(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=16, total_blocks=64)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "5")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "46")
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_RESIZE_GRANULARITY_BLOCKS",
        "16",
    )
    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy

    config = core._uvm_kv_autoscale_config(policy)

    assert config["minimum_blocks"] == 16
    assert config["maximum_blocks"] == 32
    assert config["resize_granularity_blocks"] == 16


def test_autoscale_conflicting_aligned_bounds_fail_closed(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=32, total_blocks=64)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS", "33")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "32")
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_RESIZE_GRANULARITY_BLOCKS",
        "16",
    )

    decision = core._observe_uvm_kv_autoscale_pressure(
        _autoscale_scheduler_output()  # type: ignore[arg-type]
    )

    assert decision["ok"] is False
    assert decision["action"] == "hold_invalid_autoscale_bounds"
    assert decision["config"]["minimum_blocks"] == 48
    assert decision["config"]["maximum_blocks"] == 32
    assert decision["config"]["bounds_valid"] is False


def test_proactive_high_watermark_respects_minimum_residency(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=16, total_blocks=64)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MIN_RESIDENCY_STEPS", "3")
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_HIGH_WATERMARK", "0.75")
    core.scheduler.kv_cache_manager.block_pool.active_blocks = 12
    targets: list[int] = []
    monkeypatch.setattr(
        core,
        "_resize_uvm_kv_target",
        lambda target_blocks, **_kwargs: (
            targets.append(target_blocks) or {"ok": True, "reason": "applied"}
        ),
    )
    output = _autoscale_scheduler_output()

    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]
    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]
    assert targets == []
    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]

    assert targets == [32]


def test_prediction_error_guard_uses_positive_ewma_and_decays(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=32, total_blocks=64)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_PREDICTION_ERROR_ALPHA",
        "0.5",
    )
    core._uvm_kv_autoscale_predicted_blocks = 4
    core.scheduler.kv_cache_manager.block_pool.active_blocks = 6
    output = _autoscale_scheduler_output()

    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]
    assert core._uvm_kv_autoscale_prediction_error_blocks == 1.0
    assert core._uvm_kv_autoscale_last_decision["required_blocks"] == 7

    core._maybe_handle_uvm_kv_preemption_pressure(output)  # type: ignore[arg-type]
    assert core._uvm_kv_autoscale_prediction_error_blocks == 0.5


def test_next_request_prediction_uses_single_geometric_controller(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=8, total_blocks=32)
    _enable_test_autoscale(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS", "32")
    monkeypatch.setenv("VLLM_UVM_KV_NEXT_REQUEST_LOW_TARGET_BLOCKS", "3")
    monkeypatch.setenv("VLLM_UVM_KV_NEXT_REQUEST_HIGH_TARGET_BLOCKS", "24")
    calls: list[int] = []

    def resize(target_blocks: int, **_kwargs: object) -> dict[str, object]:
        calls.append(target_blocks)
        return {"ok": True, "reason": "applied"}

    monkeypatch.setattr(core, "_resize_uvm_kv_target", resize)
    result = core.apply_uvm_kv_next_request_pressure([96], 16)

    assert result["ok"] is True
    assert result["pressure"]["predicted_kv_blocks"] == 7
    assert result["pressure"]["decision_source"] == (
        "unified_kv_autoscale_controller"
    )
    assert result["pressure"]["legacy_low_high_target_ignored"] is True
    assert result["pressure"]["new_target_blocks"] == 16
    assert calls == [16]


def test_incremental_expert_restore_wave_is_balanced_by_layer(
    monkeypatch: Any,
) -> None:
    core = _make_core()
    monkeypatch.setenv("VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE", "1")
    calls: list[tuple[str, tuple[object, ...]]] = []

    def rpc(method: str, *, args: tuple[object, ...]) -> dict[str, object]:
        calls.append((method, args))
        if method == "get_uvm_incremental_expert_restore_candidates":
            return {
                "ok": True,
                "candidates": {"layer.0": 3, "layer.1": 7},
            }
        assert method == "run_uvm_incremental_expert_restore"
        return {"ok": True, "reason": "restored", "restored_expert_count": 2}

    monkeypatch.setattr(core, "_uvm_single_worker_rpc", rpc)
    result = core._restore_uvm_incremental_expert_wave()

    assert result["ok"] is True
    assert result["restored_expert_count"] == 2
    assert calls == [
        ("get_uvm_incremental_expert_restore_candidates", ()),
        (
            "run_uvm_incremental_expert_restore",
            ({"layer.0": 3, "layer.1": 7},),
        ),
    ]


def test_autoscale_target_activates_configured_fixed_plan_class(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=16, total_blocks=64)
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_MAP",
        '{"8": "C8", "16": "C6", "32": "C4", "64": "C2"}',
    )
    activated: list[str] = []

    def activate(class_name: str) -> dict[str, object]:
        activated.append(class_name)
        return {"ok": True, "reason": "class_activated"}

    monkeypatch.setattr(core, "activate_uvm_expert_graph_class", activate)
    core._note_uvm_kv_autoscale_resize(16)

    assert core._uvm_kv_autoscale_pending_plan_class == "C6"
    result = core._apply_pending_uvm_expert_plan_class_if_ready()
    assert result == {"ok": True, "reason": "class_activated"}
    assert activated == ["C6"]
    assert core._uvm_kv_autoscale_pending_plan_class is None


def test_plan_class_activation_resumes_prefill_topology_when_configured(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=16, total_blocks=64)
    monkeypatch.setenv(
        "VLLM_UVM_KV_AUTO_SCALE_RESUME_PREFILL_BEFORE_PLAN_CLASS",
        "1",
    )
    calls: list[tuple[str, tuple[object, ...]]] = []

    def rpc(method: str, *, args: tuple[object, ...]) -> dict[str, object]:
        calls.append((method, args))
        return {"ok": True, "reason": "class_activated"}

    monkeypatch.setattr(core, "_uvm_single_worker_rpc", rpc)
    result = core.activate_uvm_expert_graph_class("cold70")

    assert result["ok"] is True
    assert calls == [
        (
            "activate_uvm_expert_graph_class",
            ("cold70", True, True, False),
        )
    ]


def test_worker_resize_commits_target_after_expert_and_physical_steps(
    monkeypatch: Any,
) -> None:
    core = _make_core()
    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
    events: list[str] = []

    def prepare(
        target_blocks: int,
        *,
        block_pool: object,
        reason: str,
    ) -> dict[str, object]:
        assert target_blocks == 12
        assert block_pool is core.scheduler.kv_cache_manager.block_pool
        assert reason == "test_pressure"
        events.append("prepare")
        return {"ok": True}

    def finalize(
        preparation: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        assert preparation["ok"] is True
        assert kwargs["worker_ok"] is True
        assert kwargs["worker_made_progress"] is True
        events.append("apply")
        return {"ok": True}

    policy.prepare_dynamic_pool_resize = prepare
    policy.finalize_dynamic_pool_resize = finalize

    def fake_collective_rpc(
        method: str,
        args: tuple[object, ...] = (),
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        if method == "get_uvm_incremental_expert_offload_candidates":
            assert args == ("cold",)
            events.append("candidates")
            return [
                {
                    "ok": True,
                    "candidates": {"model.layers.7.mlp.experts": 6},
                }
            ]
        if method == "run_uvm_incremental_expert_offload":
            assert args == (
                False,
                {"model.layers.7.mlp.experts": 6},
            )
            events.append("expert")
            return [{"ok": True}]
        assert method == "resize_uvm_kv_cache_target"
        assert args == (8, 12, [])
        events.append("physical")
        return [{"ok": True, "made_progress": True}]

    monkeypatch.setattr(core, "collective_rpc", fake_collective_rpc)

    result = core._resize_uvm_kv_target(
        12,
        reason="test_pressure",
        release_expert_wave=True,
    )

    assert result == {"ok": True, "reason": "applied"}
    assert events == [
        "candidates",
        "expert",
        "prepare",
        "physical",
        "apply",
    ]


def test_grow_prepare_failure_rolls_back_balanced_expert_wave(
    monkeypatch: Any,
) -> None:
    core = _make_core(applied_target=8, total_blocks=32)
    policy = core.scheduler.kv_cache_manager.uvm_kv_runtime_policy
    policy.prepare_dynamic_pool_resize = lambda *_args, **_kwargs: {"ok": False}
    events: list[str] = []

    def rpc(
        method: str,
        *,
        args: tuple[object, ...],
    ) -> dict[str, object]:
        if method == "get_uvm_incremental_expert_offload_candidates":
            events.append("select")
            return {"ok": True, "candidates": {"layer.7": 6}}
        if method == "run_uvm_incremental_expert_offload":
            events.append("offload")
            return {"ok": True, "released_expert_count": 1}
        assert method == "run_uvm_incremental_expert_restore"
        assert args == ({"layer.7": 6},)
        events.append("restore")
        return {"ok": True, "restored_expert_count": 1}

    monkeypatch.setattr(core, "_uvm_single_worker_rpc", rpc)
    result = core._resize_uvm_kv_target(
        16,
        reason="prepare_failure",
        release_expert_wave=True,
    )

    assert result["ok"] is False
    assert result["reason"] == "prepare_failed"
    assert result["expert_rollback"] == {
        "ok": True,
        "restored_expert_count": 1,
    }
    assert events == ["select", "offload", "restore"]


def test_autoscale_shrink_allows_active_prefix_when_tail_is_free(
    monkeypatch: Any,
) -> None:
    core, policy, pool = _make_real_policy_core()
    active_block = pool.free_block_queue.popleft()
    active_block.free_queue_kind = None
    active_block.ref_cnt = 1
    active_block.lifecycle_state = "used_free"
    assert active_block.block_id == 1
    core.scheduler.unfinished = True

    def collective_rpc(
        method: str,
        args: tuple[object, ...] = (),
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        assert method == "resize_uvm_kv_cache_target"
        assert args == (6, 3, [])
        return [{"ok": True, "made_progress": True}]

    monkeypatch.setattr(core, "collective_rpc", collective_rpc)
    result = core._resize_uvm_kv_target(
        3,
        reason="autoscale_active_prefix_shrink",
        release_expert_wave=False,
        allow_active_prefix_shrink=True,
    )

    assert result["ok"] is True
    assert policy._dynamic_pool_applied_target_blocks == 3
    assert active_block.ref_cnt == 1
    assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]


def test_autoscale_shrink_rejects_active_block_in_candidate_tail(
    monkeypatch: Any,
) -> None:
    core, policy, pool = _make_real_policy_core()
    active_tail = pool.blocks[6]
    pool.free_block_queue.remove(active_tail)
    active_tail.free_queue_kind = None
    active_tail.ref_cnt = 1
    active_tail.lifecycle_state = "used_free"
    core.scheduler.unfinished = True
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_FAIL_FAST", "0")

    result = core._resize_uvm_kv_target(
        3,
        reason="unsafe_active_tail_shrink",
        release_expert_wave=False,
        allow_active_prefix_shrink=True,
    )

    assert result["ok"] is False
    assert result["reason"] == "prepare_failed"
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert active_tail.ref_cnt == 1
    assert _queue_ids(pool.offline_block_queue) == []


def test_online_growth_preserves_active_prefix_blocks(
    monkeypatch: Any,
) -> None:
    core, policy, pool = _make_real_policy_core()
    pool.apply_uvm_dynamic_pool_target(3, old_target_blocks=6)
    policy._dynamic_pool_applied_target_blocks = 3
    policy.dynamic_pool_target_blocks = 3
    for block_id in (4, 5, 6):
        policy._mark_dynamic_pool_block_cpu(block=pool.blocks[block_id])
    active = pool.free_block_queue.popleft()
    active.free_queue_kind = None
    active.ref_cnt = 1
    active.lifecycle_state = "used_free"
    core.scheduler.unfinished = True
    core.scheduler.running.append(object())
    monkeypatch.setenv("VLLM_UVM_KV_SCHEDULER_BOUNDARY_RESIZE_ENABLE", "1")

    def collective_rpc(
        method: str,
        args: tuple[object, ...] = (),
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        assert method == "resize_uvm_kv_cache_target"
        assert args == (3, 6, [])
        assert active.ref_cnt == 1
        return [{"ok": True, "made_progress": True}]

    monkeypatch.setattr(core, "collective_rpc", collective_rpc)
    result = core._resize_uvm_kv_target(
        6,
        reason="online_active_prefix_growth",
        release_expert_wave=False,
    )

    assert result["ok"] is True
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert active.ref_cnt == 1
    assert active.free_queue_kind is None
    assert _queue_ids(pool.offline_block_queue) == []


def test_equal_applied_target_with_dirty_blocks_runs_workers_and_reconciles(
    monkeypatch: Any,
) -> None:
    core, policy, pool = _make_real_policy_core()
    _seed_partial_shrink(policy, pool)
    rpc_calls: list[tuple[object, ...]] = []

    def collective_rpc(
        method: str,
        args: tuple[object, ...] = (),
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        assert method == "resize_uvm_kv_cache_target"
        assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
        assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
        rpc_calls.append(args)
        return [
            _worker_resize_result(
                ok=True,
                made_progress=True,
            )
        ]

    monkeypatch.setattr(core, "collective_rpc", collective_rpc)

    result = core._resize_uvm_kv_target(
        6,
        reason="cross-entry-revert",
        release_expert_wave=False,
    )

    assert result["ok"] is True
    assert rpc_calls == [(6, 6, [4, 5, 6])]
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert policy._dynamic_pool_reconciliation_block_ids == set()
    assert not policy._dynamic_pool_reconciliation_needed
    assert not policy._dynamic_pool_external_reconciliation_required
    assert _queue_ids(pool.free_block_queue) == [4, 5, 6, 1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == []
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.GPU
        for block_id in (4, 5, 6)
    )


def test_worker_partial_does_not_commit_and_requires_external_retry(
    monkeypatch: Any,
) -> None:
    core, policy, pool = _make_real_policy_core()
    _seed_partial_shrink(policy, pool)
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_FAIL_FAST", "0")

    def collective_rpc(
        _method: str,
        args: tuple[object, ...] = (),
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        return [
            _worker_resize_result(
                ok=False,
                made_progress=True,
            )
        ]

    monkeypatch.setattr(core, "collective_rpc", collective_rpc)

    result = core._resize_uvm_kv_target(
        6,
        reason="worker-partial",
        release_expert_wave=False,
    )

    assert result["ok"] is False
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert policy._dynamic_pool_reconciliation_block_ids == {4, 5, 6}
    assert policy._dynamic_pool_external_reconciliation_required
    assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.UNKNOWN
        for block_id in (4, 5, 6)
    )
    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="ordinary-refresh",
    ) == {"ok": False, "reason": "external_reconciliation_required"}


def test_unexpected_worker_cardinality_keeps_entire_plan_quarantined(
    monkeypatch: Any,
) -> None:
    core, policy, pool = _make_real_policy_core()
    _seed_partial_shrink(policy, pool)
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_FAIL_FAST", "0")

    def collective_rpc(
        _method: str,
        args: tuple[object, ...] = (),
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        complete = _worker_resize_result(
            ok=True,
            made_progress=True,
        )
        incomplete = _worker_resize_result(
            ok=False,
            made_progress=True,
        )
        return [complete, incomplete]

    monkeypatch.setattr(core, "collective_rpc", collective_rpc)

    result = core._resize_uvm_kv_target(
        6,
        reason="worker-cardinality-mismatch",
        release_expert_wave=False,
    )

    assert result["ok"] is False
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert policy._dynamic_pool_reconciliation_block_ids == {4, 5, 6}
    assert policy._dynamic_pool_external_reconciliation_required
    assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]


def test_worker_rpc_exception_marks_clean_shrink_dirty_before_return(
    monkeypatch: Any,
) -> None:
    core, policy, pool = _make_real_policy_core()
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_FAIL_FAST", "0")

    def collective_rpc(*_args: object, **_kwargs: object) -> list[object]:
        assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
        assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
        raise RuntimeError("mock worker transport failure")

    monkeypatch.setattr(core, "collective_rpc", collective_rpc)

    result = core._resize_uvm_kv_target(
        3,
        reason="rpc-exception",
        release_expert_wave=False,
    )

    assert result["ok"] is False
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert policy._dynamic_pool_reconciliation_block_ids == {4, 5, 6}
    assert policy._dynamic_pool_external_reconciliation_required
    assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.UNKNOWN
        for block_id in (4, 5, 6)
    )


def test_zero_progress_worker_failure_rolls_back_new_quarantine(
    monkeypatch: Any,
) -> None:
    core, policy, pool = _make_real_policy_core()
    monkeypatch.setenv("VLLM_UVM_KV_ADAPTIVE_FAIL_FAST", "0")

    def collective_rpc(
        _method: str,
        args: tuple[object, ...] = (),
        **_kwargs: object,
    ) -> list[dict[str, object]]:
        assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
        return [
            _worker_resize_result(
                ok=False,
                made_progress=False,
            )
        ]

    monkeypatch.setattr(core, "collective_rpc", collective_rpc)

    result = core._resize_uvm_kv_target(
        3,
        reason="worker-zero-progress",
        release_expert_wave=False,
    )

    assert result["ok"] is False
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert policy._dynamic_pool_reconciliation_block_ids == set()
    assert not policy._dynamic_pool_external_reconciliation_required
    assert _queue_ids(pool.free_block_queue) == [4, 5, 6, 1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == []


def test_dynamic_expert_swap_plan_preserves_count_and_uses_heat_delta() -> None:
    plan = EngineCore._plan_uvm_dynamic_expert_swaps(
        [
            {
                "ok": True,
                "layers": {
                    "layer.7": {
                        "global_num_experts": 6,
                        "local_num_experts": 6,
                        "expert_token_counts": [0, 2, 80, 1, 5, 100],
                        "remote_experts": [0, 5],
                    }
                },
            }
        ],
        max_swaps_per_layer=1,
        min_count_delta=10,
    )

    assert plan["ok"] is True
    assert plan["swaps_by_layer"] == {
        "layer.7": [
            {
                "promote_expert": 5,
                "demote_expert": 3,
                "promote_count": 100,
                "demote_count": 1,
                "count_delta": 99,
            }
        ]
    }
    layer = plan["layers"]["layer.7"]
    assert layer["remote_count"] == 2
    assert layer["target_remote_experts"] == [0, 3]


def test_dynamic_expert_swap_plan_excludes_non_swappable_experts() -> None:
    plan = EngineCore._plan_uvm_dynamic_expert_swaps(
        [
            {
                "ok": True,
                "layers": {
                    "layer.7": {
                        "global_num_experts": 6,
                        "local_num_experts": 6,
                        "expert_token_counts": [0, 1, 2, 80, 90, 100],
                        "remote_experts": [5],
                        "swappable_experts": [3, 4, 5],
                    }
                },
            }
        ],
        max_swaps_per_layer=1,
        min_count_delta=1,
    )

    assert plan["ok"] is True
    assert plan["layers"]["layer.7"]["target_remote_experts"] == [3]
    assert plan["layers"]["layer.7"]["swappable_expert_count"] == 3
    assert plan["swaps_by_layer"]["layer.7"][0]["demote_expert"] == 3


def test_dynamic_expert_swap_plan_rejects_distributed_expert_slices() -> None:
    plan = EngineCore._plan_uvm_dynamic_expert_swaps(
        [
            {
                "ok": True,
                "layers": {
                    "layer.7": {
                        "global_num_experts": 8,
                        "local_num_experts": 4,
                        "expert_token_counts": [0] * 8,
                        "remote_experts": [0],
                    }
                },
            }
        ],
        max_swaps_per_layer=1,
        min_count_delta=1,
    )

    assert plan["ok"] is False
    assert "distributed_expert_placement_unsupported" in plan["errors"][0]


def test_dynamic_expert_heat_ewma_and_fixed_class_selection() -> None:
    core = _make_core()
    first = core._smooth_uvm_dynamic_expert_heat(
        [{"ok": True, "layers": {"layer.7": {"expert_token_counts": [0, 100]}}}],
        alpha=0.25,
    )
    second = core._smooth_uvm_dynamic_expert_heat(
        [{"ok": True, "layers": {"layer.7": {"expert_token_counts": [100, 0]}}}],
        alpha=0.25,
    )

    assert first[0]["layers"]["layer.7"]["expert_token_counts"] == [0.0, 100.0]
    assert second[0]["layers"]["layer.7"]["expert_token_counts"] == [25.0, 75.0]
    assert EngineCore._select_uvm_expert_plan_class(
        (2.0, 4.0, 6.0, 8.0), budget_gib=5.3
    ) == 4.0


def test_periodic_dynamic_expert_rebalance_obeys_interval(
    monkeypatch: Any,
) -> None:
    core = _make_core()
    monkeypatch.setenv("VLLM_UVM_DYNAMIC_EXPERT_AUTO_REBALANCE_ENABLE", "1")
    monkeypatch.setenv("VLLM_UVM_DYNAMIC_EXPERT_REBALANCE_INTERVAL_MS", "60000")
    calls = 0

    def rebalance(**_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"ok": True, "reason": "expert_swaps_committed"}

    monkeypatch.setattr(core, "rebalance_uvm_remote_experts", rebalance)

    first = core._maybe_auto_rebalance_uvm_remote_experts()
    second = core._maybe_auto_rebalance_uvm_remote_experts()

    assert first is not None and first["rebalance_generation"] == 1
    assert second is None
    assert calls == 1
