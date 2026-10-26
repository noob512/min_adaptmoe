# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
)
from vllm.v1.core.uvm_kv_runtime_policy import UvmKvRuntimePolicy
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashToBlockMap:
    """
    Cache of blocks that are used for prefix caching. It caches blocks
    from hash directly to a block or multiple blocks
    (i.e. {block_hash: KVCacheBlocks})
    - Mostly block_hash maps to a single KVCacheBlock, and KVCacheBlocks
        would simply be a KVCacheBlock.
    - Otherwise, KVCacheBlocks is a dict from {block_id: KVCacheBlock}

    A cached block is a full block with a block hash that can be used
    for prefix caching.
    The cached block may be used by running requests or in the
    free_block_queue that could potentially be evicted.

    NOTE #1: We currently don't de-duplicate the blocks in the cache,
    meaning that if a block becomes full and is cached, we don't check
    if there is already an identical block in the cache. This is because
    we want to make sure the allocated block IDs won't change so that
    block tables are append-only.
    NOTE #2: The union type is introduced in order to reduce GC costs
    from the inner dict.
    """

    def __init__(self):
        self._cache: dict[
            BlockHashWithGroupId, KVCacheBlock | dict[int, KVCacheBlock]
        ] = {}

    def get_one_block(self, key: BlockHashWithGroupId) -> KVCacheBlock | None:
        """
        Gets any block with the given block hash key.
        """
        blocks = self._cache.get(key)
        if blocks is not None:
            if isinstance(blocks, KVCacheBlock):
                return blocks
            if isinstance(blocks, dict):
                return next(iter(blocks.values()))
            self._unexpected_blocks_type(blocks)
        return None

    def insert(self, key: BlockHashWithGroupId, block: KVCacheBlock) -> None:
        """
        Inserts the KVCacheBlock to the cache
        """
        blocks = self._cache.get(key)
        if blocks is None:
            # When key is not found, attach a single block to the key
            self._cache[key] = block
        elif isinstance(blocks, KVCacheBlock):
            # If there's a block with the same key, merge the original block
            # and the new block into a dict
            self._cache[key] = {blocks.block_id: blocks, block.block_id: block}
        elif isinstance(blocks, dict):
            # If it's already a dict, simply insert the block
            blocks[block.block_id] = block
        else:
            self._unexpected_blocks_type(blocks)

    def pop(self, key: BlockHashWithGroupId, block_id: int) -> KVCacheBlock | None:
        """
        Checks if block_hash exists and pop block_id from the cache
        """
        blocks = self._cache.pop(key, None)
        if blocks is None:
            # block_hash not found in the cache
            return None
        # TODO(Jialin): If key is found, block_id should always present
        # in blocks. We currently keep the original behaviour for safety.
        #
        # Will add block_id == blocks.block_id assertion and
        # use del blocks[block_id] instead as followup.
        if isinstance(blocks, KVCacheBlock):
            if blocks.block_id == block_id:
                return blocks
            # If the single block ID doesn't match, we should put the
            # block back (it should happen rarely)
            self._cache[key] = blocks
            return None
        if isinstance(blocks, dict):
            # Try to pop block_id from the block dict, and if dict still
            # contain blocks, put back to the cache.
            block = blocks.pop(block_id, None)
            if len(blocks) > 0:
                self._cache[key] = blocks
            return block
        self._unexpected_blocks_type(blocks)
        return None

    def __len__(self) -> int:
        return len(self._cache)

    def _unexpected_blocks_type(self, blocks: Any) -> None:
        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate, free and cache the kv cache blocks. The
    free_block_queue stores the free blocks in eviction order to enable
    allocation, free, and cache eviction. The cached_block_hash_to_block
    maps between block hash and cached block to support finding cached blocks
    by their block hash.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        enable_caching: Whether to enable prefix caching.
        hash_block_size: The block size of which the block hashes are computed.
            The actual block size usually equals hash_block_size, but in cases
            where different KV cache groups have different block sizes, the
            actual block size can be a multiple of hash_block_size.
        enable_kv_cache_events: Whether to enable kv cache events.
    """

    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        hash_block_size: int,
        enable_kv_cache_events: bool = False,
        uvm_kv_runtime_policy: UvmKvRuntimePolicy | None = None,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.enable_caching = enable_caching
        self.hash_block_size = hash_block_size
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks (including eviction candidates when caching is
        # enabled).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        # GPU-resident burst reserve. Blocks in this queue consume HBM but are
        # kept out of the ordinary free queue until allocation pressure needs
        # them. The runtime policy controls its target size.
        self.backup_block_queue = FreeKVCacheBlockQueue([])
        self.offline_block_queue = FreeKVCacheBlockQueue([])

        # Cache for block lookup
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
        self.null_block.free_queue_kind = None
        for block in self.blocks[1:]:
            block.free_queue_kind = "gpu"

        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue: list[KVCacheEvent] = []
        self.uvm_kv_runtime_policy = uvm_kv_runtime_policy
        if self.uvm_kv_runtime_policy is not None:
            self.uvm_kv_runtime_policy.register_stage_l5_kv_blocks(self)
            self._refill_uvm_backup_queue()

    def _uvm_backup_target_blocks(self) -> int:
        policy = self.uvm_kv_runtime_policy
        if policy is None or not policy.dynamic_pool_active():
            return 0
        resolver = getattr(policy, "get_backup_reserve_blocks", None)
        if callable(resolver):
            return max(int(resolver()), 0)
        return max(int(getattr(policy, "backup_reserve_blocks", 0)), 0)

    def _refill_uvm_backup_queue(self) -> int:
        """Move ordinary GPU-free blocks into the logical burst reserve."""
        wanted = self._uvm_backup_target_blocks()
        missing = max(wanted - self.backup_block_queue.num_free_blocks, 0)
        count = min(missing, self.free_block_queue.num_free_blocks)
        if count == 0:
            return 0
        blocks = self.free_block_queue.get_all_free_blocks()[-count:]
        for block in blocks:
            self.free_block_queue.remove(block)
            block.free_queue_kind = "backup"
        self.backup_block_queue.append_n(blocks)
        return count

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        """Get the cached block by the block hash for each group in
        `kv_cache_group_ids`, or None if cache miss for any group.
        If there are duplicated blocks, we return the first block in the cache.

        Args:
            block_hash: The hash value of the block.
            kv_cache_group_ids: The ids of the KV cache groups.

        Returns:
            The cached blocks if exists, or None.
        """
        if (
            self.uvm_kv_runtime_policy is not None
            and self.uvm_kv_runtime_policy.dynamic_pool_active()
        ):
            self.uvm_kv_runtime_policy.refresh_dynamic_pool_target(
                self,
                "prefix_cache_lookup",
            )
        cached_blocks = []
        for group_id in kv_cache_group_ids:
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, group_id
            )
            block = self.cached_block_hash_to_block.get_one_block(
                block_hash_with_group_id
            )
            if not block:
                return None
            if (
                self.uvm_kv_runtime_policy is not None
                and self.uvm_kv_runtime_policy.dynamic_pool_active()
                and not self.uvm_kv_runtime_policy.is_block_id_allocatable(
                    block.block_id
                )
            ):
                return None
            if block.ref_cnt == 0 and block.free_queue_kind != "gpu":
                return None
            cached_blocks.append(block)
        return cached_blocks

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        kv_cache_group_id: int,
    ) -> None:
        """Cache a list of full blocks for prefix caching.
        This function takes a list of blocks that will have their block hash
        metadata to be updated and cached. Given a request, it updates the
        metadata for each block and caching it in the
        `cached_block_hash_to_block`.
        The block hashes values are computed by the Request object immediately
        when it is created and when new tokens are appended.

        Args:
            request: The request to cache the blocks.
            blocks: All blocks in the request.
            num_cached_blocks: The number of blocks that are already cached.
            num_full_blocks: The number of blocks that are full and should
                be cached after this function.
            block_size: Number of tokens in each block.
            kv_cache_group_id: The id of the KV cache group.
        """
        if num_cached_blocks >= num_full_blocks:
            return
        new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
        assert len(request.block_hashes) >= num_full_blocks
        if block_size == self.hash_block_size:
            # Common case.
            block_hashes: BlockHashList = request.block_hashes
        else:
            # block_size is a multiple of hash_block_size. This happens when
            # different KV cache groups have different block sizes.
            assert block_size % self.hash_block_size == 0
            # Recalculate block_hashes at the granularity of block_size, using
            # the original block_hashes (at the granularity of hash_block_size).
            block_hashes = BlockHashListWithBlockSize(
                request.block_hashes, self.hash_block_size, block_size
            )

        new_block_hashes = block_hashes[num_cached_blocks:]
        new_hashes: list[ExternalBlockHash] | None = (
            [] if self.enable_kv_cache_events else None
        )
        for i, blk in enumerate(new_full_blocks):
            assert blk.block_hash is None
            block_hash = new_block_hashes[i]

            # Update and added the full block to the cache.
            block_hash_with_group_id = make_block_hash_with_group_id(
                block_hash, kv_cache_group_id
            )
            blk.block_hash = block_hash_with_group_id
            self.cached_block_hash_to_block.insert(block_hash_with_group_id, blk)
            if new_hashes is not None:
                new_hashes.append(maybe_convert_block_hash(block_hash))

        if self.enable_kv_cache_events:
            if num_cached_blocks == 0:
                parent_block_hash: ExternalBlockHash | None = None
            else:
                parent_block = blocks[num_cached_blocks - 1]
                assert parent_block.block_hash is not None
                parent_block_hash = maybe_convert_block_hash(
                    get_block_hash(parent_block.block_hash)
                )

            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=new_hashes,
                    parent_block_hash=parent_block_hash,
                    token_ids=request.all_token_ids[
                        num_cached_blocks * block_size : num_full_blocks * block_size
                    ],
                    block_size=block_size,
                    lora_id=request.lora_request.adapter_id
                    if request.lora_request
                    else None,
                    medium=MEDIUM_GPU,
                )
            )

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Note that we do not check block cache in this function.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.
        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        dynamic_pool_active = (
            self.uvm_kv_runtime_policy is not None
            and self.uvm_kv_runtime_policy.dynamic_pool_active()
        )
        ret: list[KVCacheBlock] | None = None
        if dynamic_pool_active:
            ordinary = min(num_blocks, self.free_block_queue.num_free_blocks)
            ret = self.free_block_queue.popleft_n(ordinary)
            for block in ret:
                block.free_queue_kind = None
            backup_count = num_blocks - ordinary
            if backup_count:
                backup = self.backup_block_queue.popleft_n(backup_count)
                for block in backup:
                    block.free_queue_kind = None
                ret.extend(backup)
                notify = getattr(
                    self.uvm_kv_runtime_policy,
                    "on_backup_blocks_activated",
                    None,
                )
                if notify is not None:
                    notify(backup_count, reason="get_new_blocks")
        elif self.uvm_kv_runtime_policy is not None:
            ret = self.uvm_kv_runtime_policy.select_free_blocks_for_allocation(
                block_pool=self,
                num_blocks=num_blocks,
                enable_caching=self.enable_caching,
            )
            if ret is not None:
                for block in ret:
                    self.free_block_queue.remove(block)
                    block.free_queue_kind = None
        if ret is None:
            ret = self.free_block_queue.popleft_n(num_blocks)
            for block in ret:
                block.free_queue_kind = None
        if len(ret) != num_blocks:
            raise ValueError(
                f"Cannot get {num_blocks} free blocks from the dynamic pool"
            )

        # In order to only iterate the list once, we duplicated code a bit
        if self.enable_caching:
            for block in ret:
                self._maybe_evict_cached_block(block)
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                block.lifecycle_state = "used_free"
        else:
            for block in ret:
                assert block.ref_cnt == 0
                block.ref_cnt += 1
                block.lifecycle_state = "used_free"
        if self.uvm_kv_runtime_policy is not None:
            self.uvm_kv_runtime_policy.on_blocks_allocated(
                blocks=ret,
                reason="get_new_blocks",
            )
        return ret

    def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
        """
        If a block is cached in `cached_block_hash_to_block`, we reset its hash
        metadata and evict it from the cache.

        Args:
            block: The block to evict.

        Returns:
            True if the block is evicted, False otherwise.
        """
        block_hash = block.block_hash
        if block_hash is None:
            # The block doesn't have hash, eviction is not needed
            return False

        if self.cached_block_hash_to_block.pop(block_hash, block.block_id) is None:
            # block not found in cached_block_hash_to_block,
            # eviction is not needed
            return False

        block.reset_hash()

        if self.enable_kv_cache_events:
            # FIXME (Chen): Not sure whether we should return `hash_value`
            # or `(hash_value, group_id)` here. But it's fine now because
            # we disable hybrid kv cache manager when kv cache event is
            # enabled, so there is only one group.
            self.kv_event_queue.append(
                BlockRemoved(
                    block_hashes=[maybe_convert_block_hash(get_block_hash(block_hash))],
                    medium=MEDIUM_GPU,
                )
            )
        return True

    def evict_cached_free_blocks(self, max_blocks: int) -> int:
        """Evict prefix-cache metadata from free blocks in LRU order.

        This is Stage J's conservative runtime eviction executor. It only
        touches blocks that are already in the free queue (`ref_cnt == 0`), so
        active request block tables are not modified.
        """
        if max_blocks <= 0 or not self.enable_caching:
            return 0

        evicted = 0
        for block in self.free_block_queue.get_all_free_blocks():
            if evicted >= max_blocks:
                break
            if block.is_null or block.ref_cnt != 0 or block.block_hash is None:
                continue
            if self._maybe_evict_cached_block(block):
                evicted += 1
        return evicted

    def touch(self, blocks: tuple[Sequence[KVCacheBlock], ...]) -> None:
        """Touch a block increases its reference count by 1, and may remove
        the block from the free queue. This is used when a block is hit by
        another request with the same prefix.

        Args:
            blocks: A list of blocks to touch.
        """
        for blocks_per_group in blocks:
            if self.uvm_kv_runtime_policy is not None:
                self.uvm_kv_runtime_policy.on_blocks_allocated(
                    blocks=list(blocks_per_group),
                    reason="prefix_cache_touch",
                )
            for block in blocks_per_group:
                # ref_cnt=0 means this block is in the free list (i.e. eviction
                # candidate), so remove it.
                if block.ref_cnt == 0 and not block.is_null:
                    self._remove_from_current_free_queue(block)
                block.ref_cnt += 1
                block.lifecycle_state = "used_free"

    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.
        """
        # Materialize the iterable to allow multiple passes.
        blocks_list = list(ordered_blocks)
        for block in blocks_list:
            block.ref_cnt -= 1
            if block.ref_cnt == 0 and not block.is_null:
                block.lifecycle_state = "used_free"
        released = [
            block
            for block in blocks_list
            if block.ref_cnt == 0 and not block.is_null
        ]
        gpu_blocks: list[KVCacheBlock] = []
        offline_blocks: list[KVCacheBlock] = []
        dynamic_pool_active = (
            self.uvm_kv_runtime_policy is not None
            and self.uvm_kv_runtime_policy.dynamic_pool_active()
        )
        for block in released:
            if (
                dynamic_pool_active
                and not self.uvm_kv_runtime_policy.is_block_id_allocatable(
                    block.block_id
                )
            ):
                block.lifecycle_state = "cpu_offloaded"
                block.free_queue_kind = "offline"
                offline_blocks.append(block)
            else:
                block.free_queue_kind = "gpu"
                gpu_blocks.append(block)
        self.free_block_queue.append_n(gpu_blocks)
        self.offline_block_queue.append_n(offline_blocks)
        self._refill_uvm_backup_queue()

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalid prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        num_used_blocks = self.num_gpu_blocks - (
            self.free_block_queue.num_free_blocks
            + self.backup_block_queue.num_free_blocks
            + self.offline_block_queue.num_free_blocks
        )
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()

        # Remove all hashes from all blocks.
        for block in self.blocks:
            block.reset_hash()

        logger.info("Successfully reset prefix cache")

        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.
        """
        if (
            self.uvm_kv_runtime_policy is not None
            and self.uvm_kv_runtime_policy.dynamic_pool_active()
        ):
            self.uvm_kv_runtime_policy.refresh_dynamic_pool_target(
                self,
                "block_pool_free_count",
            )
        return (
            self.free_block_queue.num_free_blocks
            + self.backup_block_queue.num_free_blocks
        )

    def _remove_from_current_free_queue(self, block: KVCacheBlock) -> None:
        if block.free_queue_kind == "gpu":
            self.free_block_queue.remove(block)
        elif block.free_queue_kind == "backup":
            self.backup_block_queue.remove(block)
        elif block.free_queue_kind == "offline":
            self.offline_block_queue.remove(block)
        else:
            raise RuntimeError(
                f"Free block {block.block_id} has no queue membership"
            )
        block.free_queue_kind = None

    def apply_uvm_dynamic_pool_target(
        self,
        target_blocks: int,
        *,
        old_target_blocks: int,
    ) -> dict[str, int]:
        """Move free block metadata between GPU and CPU-offline queues."""
        target = min(max(int(target_blocks), 0), self.num_gpu_blocks - 1)
        old_target = min(
            max(int(old_target_blocks), 0), self.num_gpu_blocks - 1
        )
        moved_gpu = 0
        moved_offline = 0
        deferred_active = 0

        if target < old_target:
            for block_id in range(target + 1, old_target + 1):
                block = self.blocks[block_id]
                if block.ref_cnt != 0:
                    deferred_active += 1
                    continue
                if block.free_queue_kind in ("gpu", "backup"):
                    self._remove_from_current_free_queue(block)
                    block.free_queue_kind = "offline"
                    self.offline_block_queue.append(block)
                    moved_offline += 1
        elif target > old_target:
            promoted: list[KVCacheBlock] = []
            for block_id in range(old_target + 1, target + 1):
                block = self.blocks[block_id]
                if block.ref_cnt != 0:
                    deferred_active += 1
                    continue
                if block.free_queue_kind == "offline":
                    self.offline_block_queue.remove(block)
                    block.free_queue_kind = "gpu"
                    promoted.append(block)
                    moved_gpu += 1
            # Newly GPU-resident blocks are fresh allocation candidates.
            self.free_block_queue.prepend_n(promoted)

        self._refill_uvm_backup_queue()

        return {
            "moved_gpu": moved_gpu,
            "moved_offline": moved_offline,
            "deferred_active": deferred_active,
            "gpu_free_blocks": self.free_block_queue.num_free_blocks,
            "backup_free_blocks": self.backup_block_queue.num_free_blocks,
            "offline_free_blocks": self.offline_block_queue.num_free_blocks,
        }

    def snapshot_uvm_dynamic_queues(self) -> dict[str, Any]:
        """Return a fail-closed audit snapshot of the three KV queues.

        The scheduler owns BlockPool mutation, so callers must take this
        snapshot at a scheduler boundary.  The method deliberately reports
        both counts and block ids: an E2E qualification must prove queue
        disjointness and whole-pool conservation, not merely plausible totals.
        """
        gpu_ids = [
            int(block.block_id)
            for block in self.free_block_queue.get_all_free_blocks()
        ]
        backup_ids = [
            int(block.block_id)
            for block in self.backup_block_queue.get_all_free_blocks()
        ]
        offline_ids = [
            int(block.block_id)
            for block in self.offline_block_queue.get_all_free_blocks()
        ]
        active_ids = [
            int(block.block_id)
            for block in self.blocks
            if not block.is_null and int(block.ref_cnt) > 0
        ]
        null_ids = [int(block.block_id) for block in self.blocks if block.is_null]
        unqueued_free_ids = [
            int(block.block_id)
            for block in self.blocks
            if (
                not block.is_null
                and int(block.ref_cnt) == 0
                and block.free_queue_kind is None
            )
        ]

        partitions = {
            "gpu_free": gpu_ids,
            "gpu_backup": backup_ids,
            "cpu_offline": offline_ids,
            "active": active_ids,
            "null": null_ids,
        }
        flattened = [
            block_id for block_ids in partitions.values() for block_id in block_ids
        ]
        unique_ids = set(flattened)
        membership_matches = all(
            all(
                self.blocks[block_id].free_queue_kind == queue_kind
                for block_id in block_ids
            )
            for queue_kind, block_ids in (
                ("gpu", gpu_ids),
                ("backup", backup_ids),
                ("offline", offline_ids),
            )
        )
        applied_target = (
            int(
                getattr(
                    self.uvm_kv_runtime_policy,
                    "_dynamic_pool_applied_target_blocks",
                    self.num_gpu_blocks - 1,
                )
            )
            if self.uvm_kv_runtime_policy is not None
            else self.num_gpu_blocks - 1
        )
        # A geometric shrink can preserve the compact GPU prefix only when
        # every block above the proposed target is free.  The minimum free ID
        # cannot prove that property because active IDs become sparse after
        # requests release and reuse blocks.  The highest active ID, however,
        # defines the beginning of the largest provably free tail.
        max_active_block_id = max(active_ids, default=0)
        tail_free_start_block_id = min(
            max(max_active_block_id + 1, 1),
            applied_target + 1,
        )
        tail_free_block_count = max(
            applied_target - tail_free_start_block_id + 1,
            0,
        )
        gpu_resident_free_ids = gpu_ids + backup_ids
        all_ids = set(range(self.num_gpu_blocks))
        duplicate_memberships = len(flattened) - len(unique_ids)
        invariant_ok = (
            duplicate_memberships == 0
            and unique_ids == all_ids
            and not unqueued_free_ids
            and membership_matches
            and len(null_ids) == 1
        )
        return {
            "schema": "vllm_uvm_kv_three_queue_snapshot_v1",
            "total_blocks_including_null": int(self.num_gpu_blocks),
            "total_allocatable_blocks": int(self.num_gpu_blocks - 1),
            "applied_target_blocks": applied_target,
            "backup_target_blocks": self._uvm_backup_target_blocks(),
            "gpu_free_blocks": len(gpu_ids),
            "gpu_backup_blocks": len(backup_ids),
            "cpu_offline_blocks": len(offline_ids),
            "active_blocks": len(active_ids),
            "active_block_ids_known": True,
            "max_active_block_id": max_active_block_id,
            "tail_free_start_block_id": tail_free_start_block_id,
            "tail_free_block_count": tail_free_block_count,
            "min_gpu_free_block_id": min(gpu_resident_free_ids, default=None),
            "null_blocks": len(null_ids),
            "accounted_blocks": len(flattened),
            "duplicate_memberships": duplicate_memberships,
            "membership_matches_queue_kind": membership_matches,
            "unqueued_free_block_ids": unqueued_free_ids,
            "invariant_ok": invariant_ok,
            "gpu_free_block_ids": gpu_ids,
            "gpu_backup_block_ids": backup_ids,
            "cpu_offline_block_ids": offline_ids,
            "active_block_ids": active_ids,
            "null_block_ids": null_ids,
        }

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """

        # Subtract 1 to account for null block.
        if (
            self.uvm_kv_runtime_policy is not None
            and self.uvm_kv_runtime_policy.dynamic_pool_active()
        ):
            total_gpu_blocks = (
                self.uvm_kv_runtime_policy.get_effective_num_gpu_blocks(self)
            )
        else:
            total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)

    def take_events(self) -> list[KVCacheEvent]:
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events
