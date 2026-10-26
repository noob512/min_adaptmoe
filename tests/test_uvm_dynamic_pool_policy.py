# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from vllm.device_allocator.uvm_region import RegionState
from vllm.v1.core import uvm_kv_runtime_policy as policy_module
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.uvm_kv_runtime_policy import UvmKvRuntimePolicy

pytestmark = pytest.mark.cpu_test


def test_physical_span_snapshot_preserves_reserved_va_geometry() -> None:
    policy_module.register_uvm_kv_cache_physical_ranges(
        [
            {
                "span_id": "layer.1",
                "ptr": 0x400000,
                "size_bytes": 0x20000,
                "page_size": 0x1000,
                "max_block_id": 15,
            },
            {
                "span_id": "layer.0",
                "ptr": 0x200000,
                "size_bytes": 0x20000,
                "page_size": 0x1000,
                "max_block_id": 15,
            },
        ],
        clear_existing=True,
    )

    assert policy_module.snapshot_uvm_kv_cache_physical_spans() == [
        {
            "span_id": "layer.0",
            "ptr": 0x200000,
            "size_bytes": 0x20000,
            "page_size": 0x1000,
            "max_block_id": 15,
        },
        {
            "span_id": "layer.1",
            "ptr": 0x400000,
            "size_bytes": 0x20000,
            "page_size": 0x1000,
            "max_block_id": 15,
        },
    ]
    policy_module.register_uvm_kv_cache_physical_ranges([], clear_existing=True)


def test_physical_span_collection_can_rematerialize_null_block() -> None:
    policy_module.register_uvm_kv_cache_physical_ranges(
        [
            {
                "span_id": "layer.0",
                "ptr": 0x200000,
                "size_bytes": 0x20000,
                "page_size": 0x1000,
                "max_block_id": 15,
            }
        ],
        clear_existing=True,
    )
    try:
        assert policy_module.collect_uvm_kv_cache_physical_ranges_for_blocks(
            [0, 1, 2]
        ) == [(0x200000, 0x3000)]
    finally:
        policy_module.register_uvm_kv_cache_physical_ranges(
            [], clear_existing=True
        )


def _make_policy(
    *,
    applied_target: int,
    target: int,
    physical_migration: bool = True,
) -> UvmKvRuntimePolicy:
    policy = object.__new__(UvmKvRuntimePolicy)
    policy.dynamic_pool_enable = True
    policy.dynamic_pool_target_blocks = target
    policy.dynamic_pool_physical_migration_enable = physical_migration
    policy.dynamic_pool_physical_migration_sync = True
    policy.dynamic_pool_physical_migration_executor = "auto"
    policy.dynamic_pool_physical_migration_device = 0
    policy.total_blocks = 6
    policy.num_kv_cache_groups = 1
    policy._dynamic_pool_applied_target_blocks = applied_target
    policy._dynamic_pool_saved_lifecycle = {}
    policy._dynamic_pool_reconciliation_needed = False
    policy._dynamic_pool_reconciliation_block_ids = set()
    policy._dynamic_pool_external_reconciliation_required = False
    policy._logical_kv_blocks = {}
    return policy


def _make_pool(policy: UvmKvRuntimePolicy) -> BlockPool:
    pool = BlockPool(
        num_gpu_blocks=7,
        enable_caching=False,
        hash_block_size=16,
        uvm_kv_runtime_policy=policy,
    )
    policy.register_stage_l5_kv_blocks(pool)
    return pool


def _queue_ids(queue: Any) -> list[int]:
    return [block.block_id for block in queue.get_all_free_blocks()]


@pytest.mark.parametrize(
    ("successes", "expected"),
    [
        (0, {"ok": False, "reason": "partial", "made_progress": False}),
        (1, {"ok": False, "reason": "partial", "made_progress": True}),
        (2, {"ok": True, "reason": "complete", "made_progress": True}),
    ],
)
def test_physical_migration_has_fixed_minimal_result(
    monkeypatch: pytest.MonkeyPatch,
    successes: int,
    expected: dict[str, object],
) -> None:
    policy = _make_policy(applied_target=6, target=3)
    ranges = [(0x1000, 4096), (0x3000, 4096)]
    monkeypatch.setattr(
        policy_module,
        "collect_uvm_kv_cache_physical_ranges_for_blocks",
        lambda block_ids: ranges,
    )
    monkeypatch.setattr(
        policy_module,
        "batch_offload_ranges_to_cpu_with_executor",
        lambda migration_ranges, **kwargs: (successes, "test", {}, {}),
    )

    assert (
        policy._migrate_dynamic_pool_physical_blocks(
            block_ids=[4, 6],
            direction="shrink",
        )
        == expected
    )


def test_unchanged_target_and_active_shrink_do_not_touch_queues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _make_policy(applied_target=6, target=6)
    pool = _make_pool(policy)
    monkeypatch.setattr(
        policy,
        "_dynamic_pool_migration_outcome",
        lambda **kwargs: pytest.fail("idempotent resize attempted migration"),
    )

    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="test",
    ) == {"ok": True, "reason": "already_applied"}

    policy.dynamic_pool_target_blocks = 3
    allocated = pool.free_block_queue.popleft_n(4)
    for block in allocated:
        block.free_queue_kind = None
        block.ref_cnt = 1
    active_block = allocated[-1]
    assert active_block.block_id == 4
    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="test",
    ) == {
        "ok": False,
        "reason": "active_blocks_must_drain_before_shrink",
    }
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert _queue_ids(pool.offline_block_queue) == []


def test_partial_shrink_stays_quarantined_and_marks_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _make_policy(applied_target=6, target=3)
    pool = _make_pool(policy)
    events: list[tuple[object, ...]] = []
    apply_target = pool.apply_uvm_dynamic_pool_target

    def record_apply(target: int, *, old_target_blocks: int) -> dict[str, int]:
        events.append(("queue", target, old_target_blocks))
        return apply_target(target, old_target_blocks=old_target_blocks)

    def partial_migration(**kwargs: Any) -> dict[str, object]:
        events.append(("migration", kwargs["direction"], tuple(kwargs["block_ids"])))
        return {"ok": False, "reason": "partial", "made_progress": True}

    monkeypatch.setattr(pool, "apply_uvm_dynamic_pool_target", record_apply)
    monkeypatch.setattr(
        policy,
        "_migrate_dynamic_pool_physical_blocks",
        partial_migration,
    )

    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="test",
    ) == {"ok": False, "reason": "partial"}
    assert events == [
        ("queue", 3, 6),
        ("migration", "shrink", (4, 5, 6)),
    ]
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.UNKNOWN
        for block_id in (4, 5, 6)
    )


def test_reverted_target_after_partial_shrink_restores_before_idempotence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _make_policy(applied_target=6, target=3)
    pool = _make_pool(policy)
    events: list[tuple[object, ...]] = []
    apply_target = pool.apply_uvm_dynamic_pool_target

    def record_apply(target: int, *, old_target_blocks: int) -> dict[str, int]:
        events.append(("queue", target, old_target_blocks))
        return apply_target(target, old_target_blocks=old_target_blocks)

    def migrate(**kwargs: Any) -> dict[str, object]:
        direction = kwargs["direction"]
        events.append(("migration", direction, tuple(kwargs["block_ids"])))
        if direction == "shrink":
            return {"ok": False, "reason": "partial", "made_progress": True}
        return {"ok": True, "reason": "complete", "made_progress": True}

    monkeypatch.setattr(pool, "apply_uvm_dynamic_pool_target", record_apply)
    monkeypatch.setattr(
        policy,
        "_migrate_dynamic_pool_physical_blocks",
        migrate,
    )

    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="test",
    ) == {"ok": False, "reason": "partial"}
    assert policy._dynamic_pool_reconciliation_needed
    policy.dynamic_pool_target_blocks = 6
    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="test",
    ) == {"ok": True, "reason": "applied"}
    assert events == [
        ("queue", 3, 6),
        ("migration", "shrink", (4, 5, 6)),
        ("migration", "grow", (4, 5, 6)),
        ("queue", 6, 3),
    ]
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert not policy._dynamic_pool_reconciliation_needed
    assert _queue_ids(pool.free_block_queue) == [4, 5, 6, 1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == []
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.GPU
        for block_id in (4, 5, 6)
    )
    assert all(
        pool.blocks[block_id].lifecycle_state == "never_used" for block_id in (4, 5, 6)
    )

    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="test",
    ) == {"ok": True, "reason": "already_applied"}
    assert len(events) == 4


def test_intermediate_target_after_partial_shrink_reconciles_both_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _make_policy(applied_target=6, target=3)
    pool = _make_pool(policy)
    migrations: list[tuple[str, tuple[int, ...]]] = []

    def migrate(**kwargs: Any) -> dict[str, object]:
        migrations.append(
            (str(kwargs["direction"]), tuple(kwargs["block_ids"]))
        )
        if len(migrations) == 1:
            return {"ok": False, "reason": "partial", "made_progress": True}
        return {"ok": True, "reason": "complete", "made_progress": True}

    monkeypatch.setattr(
        policy,
        "_migrate_dynamic_pool_physical_blocks",
        migrate,
    )

    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="partial-shrink",
    ) == {"ok": False, "reason": "partial"}
    assert policy._dynamic_pool_reconciliation_block_ids == {4, 5, 6}

    policy.dynamic_pool_target_blocks = 5
    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="intermediate-target",
    ) == {"ok": True, "reason": "applied"}

    assert migrations == [
        ("shrink", (4, 5, 6)),
        ("shrink", (6,)),
        ("grow", (4, 5)),
    ]
    assert policy._dynamic_pool_applied_target_blocks == 5
    assert not policy._dynamic_pool_reconciliation_needed
    assert policy._dynamic_pool_reconciliation_block_ids == set()
    assert _queue_ids(pool.free_block_queue) == [4, 5, 1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == [6]
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.GPU
        for block_id in (4, 5)
    )
    assert policy._logical_kv_blocks[(0, 6)].state == RegionState.CPU

    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="idempotent",
    ) == {"ok": True, "reason": "already_applied"}


def test_mixed_reconciliation_partial_keeps_every_block_quarantined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _make_policy(applied_target=6, target=3)
    pool = _make_pool(policy)
    outcomes = iter(
        [
            {"ok": False, "reason": "partial", "made_progress": True},
            {"ok": True, "reason": "complete", "made_progress": True},
            {"ok": False, "reason": "partial", "made_progress": True},
        ]
    )
    monkeypatch.setattr(
        policy,
        "_migrate_dynamic_pool_physical_blocks",
        lambda **_kwargs: next(outcomes),
    )

    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="partial-shrink",
    ) == {"ok": False, "reason": "partial"}
    policy.dynamic_pool_target_blocks = 5
    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="mixed-partial",
    ) == {"ok": False, "reason": "partial"}

    assert policy._dynamic_pool_applied_target_blocks == 6
    assert policy._dynamic_pool_reconciliation_needed
    assert policy._dynamic_pool_reconciliation_block_ids == {4, 5, 6}
    assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.UNKNOWN
        for block_id in (4, 5, 6)
    )


@pytest.mark.parametrize(
    "restore_outcome",
    [
        {"ok": False, "reason": "partial", "made_progress": True},
        {
            "ok": False,
            "reason": "no_registered_ranges",
            "made_progress": False,
        },
    ],
    ids=["partial", "zero_progress_failure"],
)
def test_incomplete_partial_shrink_restore_remains_quarantined_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    restore_outcome: dict[str, object],
) -> None:
    policy = _make_policy(applied_target=6, target=3)
    pool = _make_pool(policy)
    queue_events: list[tuple[int, int]] = []
    migration_directions: list[str] = []
    apply_target = pool.apply_uvm_dynamic_pool_target

    def record_apply(target: int, *, old_target_blocks: int) -> dict[str, int]:
        queue_events.append((target, old_target_blocks))
        return apply_target(target, old_target_blocks=old_target_blocks)

    def incomplete_migration(**kwargs: Any) -> dict[str, object]:
        direction = str(kwargs["direction"])
        migration_directions.append(direction)
        if direction == "shrink":
            return {"ok": False, "reason": "partial", "made_progress": True}
        return restore_outcome

    monkeypatch.setattr(pool, "apply_uvm_dynamic_pool_target", record_apply)
    monkeypatch.setattr(
        policy,
        "_migrate_dynamic_pool_physical_blocks",
        incomplete_migration,
    )

    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="test",
    ) == {"ok": False, "reason": "partial"}
    policy.dynamic_pool_target_blocks = 6
    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="test",
    ) == {"ok": False, "reason": restore_outcome["reason"]}
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert policy._dynamic_pool_reconciliation_needed
    assert queue_events == [(3, 6)]
    assert migration_directions == ["shrink", "grow"]
    assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.UNKNOWN
        for block_id in (4, 5, 6)
    )

    monkeypatch.setattr(
        policy,
        "_migrate_dynamic_pool_physical_blocks",
        lambda **kwargs: (
            migration_directions.append(str(kwargs["direction"]))
            or {"ok": True, "reason": "complete", "made_progress": True}
        ),
    )
    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="retry",
    ) == {"ok": True, "reason": "applied"}
    assert not policy._dynamic_pool_reconciliation_needed
    assert queue_events == [(3, 6), (6, 3)]
    assert migration_directions == ["shrink", "grow", "grow"]
    assert _queue_ids(pool.offline_block_queue) == []
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.GPU
        for block_id in (4, 5, 6)
    )


def test_partial_grow_keeps_blocks_offline_and_marks_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _make_policy(applied_target=3, target=6)
    pool = _make_pool(policy)
    pool.apply_uvm_dynamic_pool_target(3, old_target_blocks=6)
    for block_id in (4, 5, 6):
        policy._mark_dynamic_pool_block_cpu(block=pool.blocks[block_id])

    monkeypatch.setattr(
        policy,
        "_migrate_dynamic_pool_physical_blocks",
        lambda **kwargs: {
            "ok": False,
            "reason": "partial",
            "made_progress": True,
        },
    )

    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="test",
    ) == {"ok": False, "reason": "partial"}
    assert policy._dynamic_pool_applied_target_blocks == 3
    assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.UNKNOWN
        for block_id in (4, 5, 6)
    )


def test_reverted_target_after_partial_grow_releases_dirty_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _make_policy(applied_target=3, target=6)
    pool = _make_pool(policy)
    pool.apply_uvm_dynamic_pool_target(3, old_target_blocks=6)
    for block_id in (4, 5, 6):
        policy._mark_dynamic_pool_block_cpu(block=pool.blocks[block_id])
    migrations: list[tuple[str, tuple[int, ...]]] = []

    def migrate(**kwargs: Any) -> dict[str, object]:
        migrations.append(
            (str(kwargs["direction"]), tuple(kwargs["block_ids"]))
        )
        if len(migrations) == 1:
            return {"ok": False, "reason": "partial", "made_progress": True}
        return {"ok": True, "reason": "complete", "made_progress": True}

    monkeypatch.setattr(
        policy,
        "_migrate_dynamic_pool_physical_blocks",
        migrate,
    )

    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="partial-grow",
    ) == {"ok": False, "reason": "partial"}
    policy.dynamic_pool_target_blocks = 3
    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="reverted-target",
    ) == {"ok": True, "reason": "applied"}

    assert migrations == [
        ("grow", (4, 5, 6)),
        ("shrink", (4, 5, 6)),
    ]
    assert policy._dynamic_pool_applied_target_blocks == 3
    assert not policy._dynamic_pool_reconciliation_needed
    assert policy._dynamic_pool_reconciliation_block_ids == set()
    assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.CPU
        for block_id in (4, 5, 6)
    )

    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="idempotent",
    ) == {"ok": True, "reason": "already_applied"}


def test_intermediate_target_after_partial_grow_reconciles_both_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _make_policy(applied_target=3, target=6)
    pool = _make_pool(policy)
    pool.apply_uvm_dynamic_pool_target(3, old_target_blocks=6)
    for block_id in (4, 5, 6):
        policy._mark_dynamic_pool_block_cpu(block=pool.blocks[block_id])
    migrations: list[tuple[str, tuple[int, ...]]] = []

    def migrate(**kwargs: Any) -> dict[str, object]:
        migrations.append(
            (str(kwargs["direction"]), tuple(kwargs["block_ids"]))
        )
        if len(migrations) == 1:
            return {"ok": False, "reason": "partial", "made_progress": True}
        return {"ok": True, "reason": "complete", "made_progress": True}

    monkeypatch.setattr(
        policy,
        "_migrate_dynamic_pool_physical_blocks",
        migrate,
    )

    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="partial-grow",
    ) == {"ok": False, "reason": "partial"}
    policy.dynamic_pool_target_blocks = 5
    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="intermediate-target",
    ) == {"ok": True, "reason": "applied"}

    assert migrations == [
        ("grow", (4, 5, 6)),
        ("shrink", (6,)),
        ("grow", (4, 5)),
    ]
    assert policy._dynamic_pool_applied_target_blocks == 5
    assert not policy._dynamic_pool_reconciliation_needed
    assert policy._dynamic_pool_reconciliation_block_ids == set()
    assert _queue_ids(pool.free_block_queue) == [4, 5, 1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == [6]
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.GPU
        for block_id in (4, 5)
    )
    assert policy._logical_kv_blocks[(0, 6)].state == RegionState.CPU


def test_zero_progress_shrink_is_the_only_failure_that_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _make_policy(applied_target=6, target=3)
    pool = _make_pool(policy)
    events: list[tuple[object, ...]] = []
    apply_target = pool.apply_uvm_dynamic_pool_target

    def record_apply(target: int, *, old_target_blocks: int) -> dict[str, int]:
        events.append(("queue", target, old_target_blocks))
        return apply_target(target, old_target_blocks=old_target_blocks)

    def failed_migration(**kwargs: Any) -> dict[str, object]:
        events.append(("migration", kwargs["direction"]))
        return {
            "ok": False,
            "reason": "no_registered_ranges",
            "made_progress": False,
        }

    monkeypatch.setattr(pool, "apply_uvm_dynamic_pool_target", record_apply)
    monkeypatch.setattr(
        policy,
        "_migrate_dynamic_pool_physical_blocks",
        failed_migration,
    )

    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="test",
    ) == {"ok": False, "reason": "no_registered_ranges"}
    assert events == [
        ("queue", 3, 6),
        ("migration", "shrink"),
        ("queue", 6, 3),
    ]
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert _queue_ids(pool.free_block_queue) == [4, 5, 6, 1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == []
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.GPU
        for block_id in (4, 5, 6)
    )


def test_full_grow_migrates_before_exposing_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _make_policy(applied_target=3, target=6)
    pool = _make_pool(policy)
    pool.apply_uvm_dynamic_pool_target(3, old_target_blocks=6)
    for block_id in (4, 5, 6):
        policy._mark_dynamic_pool_block_cpu(block=pool.blocks[block_id])

    events: list[tuple[object, ...]] = []
    apply_target = pool.apply_uvm_dynamic_pool_target

    def record_apply(target: int, *, old_target_blocks: int) -> dict[str, int]:
        events.append(("queue", target, old_target_blocks))
        return apply_target(target, old_target_blocks=old_target_blocks)

    def complete_migration(**kwargs: Any) -> dict[str, object]:
        events.append(("migration", kwargs["direction"], tuple(kwargs["block_ids"])))
        return {"ok": True, "reason": "complete", "made_progress": True}

    monkeypatch.setattr(pool, "apply_uvm_dynamic_pool_target", record_apply)
    monkeypatch.setattr(
        policy,
        "_migrate_dynamic_pool_physical_blocks",
        complete_migration,
    )

    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="test",
    ) == {"ok": True, "reason": "applied"}
    assert events == [
        ("migration", "grow", (4, 5, 6)),
        ("queue", 6, 3),
    ]
    assert policy._dynamic_pool_applied_target_blocks == 6
    assert _queue_ids(pool.free_block_queue) == [4, 5, 6, 1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == []
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.GPU
        for block_id in (4, 5, 6)
    )


@pytest.mark.parametrize(
    ("physical_migration", "external_ready", "outcome_reason"),
    [
        (True, True, "external_complete"),
        (False, False, "logical_only"),
    ],
)
def test_external_and_logical_only_paths_do_not_repeat_physical_migration(
    monkeypatch: pytest.MonkeyPatch,
    physical_migration: bool,
    external_ready: bool,
    outcome_reason: str,
) -> None:
    policy = _make_policy(
        applied_target=6,
        target=3,
        physical_migration=physical_migration,
    )
    pool = _make_pool(policy)
    monkeypatch.setattr(
        policy,
        "_migrate_dynamic_pool_physical_blocks",
        lambda **kwargs: pytest.fail("physical migration was repeated"),
    )
    assert policy._dynamic_pool_migration_outcome(
        block_ids=[4, 5, 6],
        direction="shrink",
        external_physical_ready=external_ready,
    ) == {
        "ok": True,
        "reason": outcome_reason,
        "made_progress": external_ready,
    }

    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="test",
        external_physical_ready=external_ready,
    ) == {"ok": True, "reason": "applied"}
    assert policy._dynamic_pool_applied_target_blocks == 3
    assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.CPU
        for block_id in (4, 5, 6)
    )


@pytest.mark.parametrize(
    ("physical_migration", "external_ready"),
    [(True, True), (False, False)],
    ids=["external", "logical_only"],
)
def test_dirty_partition_reconciliation_supports_nonphysical_completion(
    monkeypatch: pytest.MonkeyPatch,
    physical_migration: bool,
    external_ready: bool,
) -> None:
    policy = _make_policy(
        applied_target=6,
        target=5,
        physical_migration=physical_migration,
    )
    pool = _make_pool(policy)
    pool.apply_uvm_dynamic_pool_target(3, old_target_blocks=6)
    for block_id in (4, 5, 6):
        policy._mark_dynamic_pool_block_unknown(block=pool.blocks[block_id])
    policy._dynamic_pool_reconciliation_needed = True
    policy._dynamic_pool_reconciliation_block_ids = {4, 5, 6}
    physical_calls: list[tuple[str, tuple[int, ...]]] = []

    def run_physical_migration(**kwargs: Any) -> dict[str, object]:
        physical_calls.append(
            (str(kwargs["direction"]), tuple(kwargs["block_ids"]))
        )
        return {"ok": True, "reason": "complete", "made_progress": True}

    monkeypatch.setattr(
        policy,
        "_migrate_dynamic_pool_physical_blocks",
        run_physical_migration,
    )

    assert policy._ensure_dynamic_pool_applied(
        block_pool=pool,
        reason="nonphysical-reconciliation",
        external_physical_ready=external_ready,
    ) == {"ok": True, "reason": "applied"}

    assert policy._dynamic_pool_applied_target_blocks == 5
    assert not policy._dynamic_pool_reconciliation_needed
    assert policy._dynamic_pool_reconciliation_block_ids == set()
    assert _queue_ids(pool.free_block_queue) == [4, 5, 1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == [6]
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.GPU
        for block_id in (4, 5)
    )
    assert policy._logical_kv_blocks[(0, 6)].state == RegionState.CPU
    assert physical_calls == (
        [("grow", (4, 5))]
        if physical_migration
        else []
    )


def test_stale_external_resize_keeps_zero_progress_plan_quarantined() -> None:
    policy = _make_policy(applied_target=6, target=3)
    pool = _make_pool(policy)

    preparation = policy.prepare_dynamic_pool_resize(
        3,
        block_pool=pool,
        reason="test",
    )
    assert preparation["ok"] is True
    assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]

    # Simulate a control-plane update after prepare but before the worker
    # response. Even a reported zero-progress failure is no longer safe to
    # roll back because it belongs to a stale generation.
    policy._dynamic_pool_generation = 1
    result = policy.finalize_dynamic_pool_resize(
        preparation,
        block_pool=pool,
        worker_ok=False,
        worker_made_progress=False,
    )

    assert result["ok"] is False
    assert result["reason"] == "dynamic_resize_plan_changed"
    assert policy._dynamic_pool_reconciliation_block_ids == {4, 5, 6}
    assert policy._dynamic_pool_reconciliation_needed
    assert policy._dynamic_pool_external_reconciliation_required
    assert _queue_ids(pool.free_block_queue) == [1, 2, 3]
    assert _queue_ids(pool.offline_block_queue) == [4, 5, 6]
    assert all(
        policy._logical_kv_blocks[(0, block_id)].state == RegionState.UNKNOWN
        for block_id in (4, 5, 6)
    )
