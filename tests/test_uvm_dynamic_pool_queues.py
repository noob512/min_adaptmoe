# SPDX-License-Identifier: Apache-2.0

from typing import Any

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    make_block_hash_with_group_id,
)


class _FakeDynamicPolicy:
    def __init__(self, applied_target: int) -> None:
        self.applied_target = applied_target
        self.backup_reserve_blocks = 0
        self.backup_activations: list[int] = []
        self.refresh_reasons: list[str] = []
        self.backup_reserve_target_map: dict[int, int] = {}

    def dynamic_pool_active(self) -> bool:
        return True

    def is_block_id_allocatable(self, block_id: int) -> bool:
        return 0 < block_id <= self.applied_target

    def refresh_dynamic_pool_target(self, block_pool: Any, reason: str) -> None:
        self.refresh_reasons.append(reason)

    def register_stage_l5_kv_blocks(self, block_pool: Any) -> None:
        return

    def on_blocks_allocated(self, *, blocks: list[Any], reason: str) -> None:
        return

    def on_blocks_freed(self, *, blocks: list[Any], reason: str) -> None:
        return

    def on_backup_blocks_activated(self, count: int, *, reason: str) -> None:
        self.backup_activations.append(count)

    def get_backup_reserve_blocks(self) -> int:
        return self.backup_reserve_target_map.get(
            self.applied_target,
            self.backup_reserve_blocks,
        )


def _ids(queue: Any) -> list[int]:
    return [block.block_id for block in queue.get_all_free_blocks()]


def test_dynamic_pool_partitions_gpu_and_offline_free_queues() -> None:
    policy = _FakeDynamicPolicy(applied_target=16)
    pool = BlockPool(
        num_gpu_blocks=17,
        enable_caching=False,
        hash_block_size=16,
        uvm_kv_runtime_policy=policy,  # type: ignore[arg-type]
    )

    transition = pool.apply_uvm_dynamic_pool_target(
        8,
        old_target_blocks=16,
    )
    policy.applied_target = 8

    assert transition["moved_offline"] == 8
    assert _ids(pool.free_block_queue) == list(range(1, 9))
    assert _ids(pool.offline_block_queue) == list(range(9, 17))
    assert pool.get_num_free_blocks() == 8

    block = pool.get_new_blocks(1)[0]
    assert block.block_id == 1
    assert block.free_queue_kind is None

    pool.free_blocks([block])
    assert _ids(pool.free_block_queue) == list(range(2, 9)) + [1]
    assert _ids(pool.offline_block_queue) == list(range(9, 17))


def test_backup_queue_is_used_only_after_ordinary_gpu_queue() -> None:
    policy = _FakeDynamicPolicy(applied_target=8)
    policy.backup_reserve_blocks = 2
    pool = BlockPool(
        num_gpu_blocks=9,
        enable_caching=False,
        hash_block_size=16,
        uvm_kv_runtime_policy=policy,  # type: ignore[arg-type]
    )

    assert _ids(pool.free_block_queue) == list(range(1, 7))
    assert _ids(pool.backup_block_queue) == [7, 8]
    assert pool.get_num_free_blocks() == 8
    initial = pool.snapshot_uvm_dynamic_queues()
    assert initial["invariant_ok"] is True
    assert initial["accounted_blocks"] == 9
    assert initial["gpu_free_block_ids"] == list(range(1, 7))
    assert initial["gpu_backup_block_ids"] == [7, 8]
    assert initial["cpu_offline_block_ids"] == []

    ordinary = pool.get_new_blocks(6)
    assert [block.block_id for block in ordinary] == list(range(1, 7))
    assert policy.backup_activations == []

    reserve = pool.get_new_blocks(1)
    assert reserve[0].block_id == 7
    assert policy.backup_activations == [1]
    under_pressure = pool.snapshot_uvm_dynamic_queues()
    assert under_pressure["invariant_ok"] is True
    assert under_pressure["gpu_free_blocks"] == 0
    assert under_pressure["gpu_backup_block_ids"] == [8]
    assert under_pressure["active_block_ids"] == list(range(1, 8))

    pool.free_blocks([ordinary[0], reserve[0]])
    assert pool.backup_block_queue.num_free_blocks == 2
    assert pool.get_num_free_blocks() == 3
    refilled = pool.snapshot_uvm_dynamic_queues()
    assert refilled["invariant_ok"] is True
    assert refilled["gpu_backup_blocks"] == 2


def test_queue_snapshot_reports_contiguous_free_tail_not_minimum_free_id() -> None:
    policy = _FakeDynamicPolicy(applied_target=8)
    pool = BlockPool(
        num_gpu_blocks=9,
        enable_caching=False,
        hash_block_size=16,
        uvm_kv_runtime_policy=policy,  # type: ignore[arg-type]
    )
    low = pool.get_new_blocks(1)[0]
    high = pool.blocks[8]
    pool.free_block_queue.remove(high)
    high.free_queue_kind = None
    high.ref_cnt = 1
    high.lifecycle_state = "used_free"

    snapshot = pool.snapshot_uvm_dynamic_queues()

    assert low.block_id == 1
    assert snapshot["min_gpu_free_block_id"] == 2
    assert snapshot["max_active_block_id"] == 8
    assert snapshot["tail_free_start_block_id"] == 9
    assert snapshot["tail_free_block_count"] == 0


def test_backup_queue_target_can_follow_the_physical_kv_tier() -> None:
    policy = _FakeDynamicPolicy(applied_target=8)
    policy.backup_reserve_target_map = {8: 2, 12: 6}
    pool = BlockPool(
        num_gpu_blocks=17,
        enable_caching=False,
        hash_block_size=16,
        uvm_kv_runtime_policy=policy,  # type: ignore[arg-type]
    )

    assert pool.snapshot_uvm_dynamic_queues()["backup_target_blocks"] == 2
    pool.apply_uvm_dynamic_pool_target(8, old_target_blocks=16)
    policy.applied_target = 12
    transition = pool.apply_uvm_dynamic_pool_target(12, old_target_blocks=8)

    assert transition["backup_free_blocks"] == 6
    snapshot = pool.snapshot_uvm_dynamic_queues()
    assert snapshot["backup_target_blocks"] == 6
    assert snapshot["gpu_free_blocks"] == 6
    assert snapshot["gpu_backup_blocks"] == 6
    assert snapshot["cpu_offline_blocks"] == 4
    assert snapshot["invariant_ok"] is True


def test_dynamic_pool_grow_promotes_new_gpu_blocks_before_warm_blocks() -> None:
    policy = _FakeDynamicPolicy(applied_target=16)
    pool = BlockPool(
        num_gpu_blocks=17,
        enable_caching=False,
        hash_block_size=16,
        uvm_kv_runtime_policy=policy,  # type: ignore[arg-type]
    )
    pool.apply_uvm_dynamic_pool_target(8, old_target_blocks=16)
    policy.applied_target = 8

    block = pool.get_new_blocks(1)[0]
    pool.free_blocks([block])
    assert _ids(pool.free_block_queue)[-1] == 1

    transition = pool.apply_uvm_dynamic_pool_target(10, old_target_blocks=8)
    policy.applied_target = 10

    assert transition["moved_gpu"] == 2
    assert _ids(pool.free_block_queue)[:2] == [9, 10]
    assert pool.blocks[9].lifecycle_state == "never_used"
    assert pool.blocks[10].lifecycle_state == "never_used"
    assert _ids(pool.offline_block_queue) == list(range(11, 17))


def test_prefix_cache_does_not_return_offline_block() -> None:
    policy = _FakeDynamicPolicy(applied_target=16)
    pool = BlockPool(
        num_gpu_blocks=17,
        enable_caching=True,
        hash_block_size=16,
        uvm_kv_runtime_policy=policy,  # type: ignore[arg-type]
    )
    block_hash = BlockHash(b"cached-prefix")
    cache_key = make_block_hash_with_group_id(block_hash, 0)
    block = pool.blocks[9]
    block.block_hash = cache_key
    pool.cached_block_hash_to_block.insert(cache_key, block)

    pool.apply_uvm_dynamic_pool_target(8, old_target_blocks=16)
    policy.applied_target = 8

    assert pool.get_cached_block(block_hash, [0]) is None

    pool.apply_uvm_dynamic_pool_target(10, old_target_blocks=8)
    policy.applied_target = 10
    assert pool.get_cached_block(block_hash, [0]) == [block]
    assert policy.refresh_reasons == [
        "prefix_cache_lookup",
        "prefix_cache_lookup",
    ]


def test_active_block_freed_after_shrink_enters_offline_queue() -> None:
    policy = _FakeDynamicPolicy(applied_target=16)
    pool = BlockPool(
        num_gpu_blocks=17,
        enable_caching=False,
        hash_block_size=16,
        uvm_kv_runtime_policy=policy,  # type: ignore[arg-type]
    )

    allocated = pool.get_new_blocks(9)
    active_tail = allocated[-1]
    assert active_tail.block_id == 9

    transition = pool.apply_uvm_dynamic_pool_target(8, old_target_blocks=16)
    policy.applied_target = 8

    assert transition["deferred_active"] == 1
    assert active_tail.free_queue_kind is None
    pool.free_blocks([active_tail])
    assert active_tail.free_queue_kind == "offline"
    assert _ids(pool.offline_block_queue)[-1] == 9
