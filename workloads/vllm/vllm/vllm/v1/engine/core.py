# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import math
import os
import queue
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Generator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from inspect import isclass, signature
from logging import DEBUG
from typing import Any, TypeVar, cast

import msgspec
import zmq

from vllm.config import ParallelConfig, VllmConfig
from vllm.distributed import stateless_destroy_torch_distributed_process_group
from vllm.envs import enable_envs_cache
from vllm.logger import init_logger
from vllm.logging_utils.dump_input import dump_engine_exception
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import engine_receiver_cache_from_config
from vllm.tasks import POOLING_TASKS, SupportedTask
from vllm.transformers_utils.config import maybe_register_config_serialize_by_value
from vllm.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
)
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.utils.network_utils import make_zmq_socket
from vllm.utils.system_utils import decorate_logs, set_process_title
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import (
    EngineCoreOutputs,
    EngineCoreRequest,
    EngineCoreRequestType,
    ReconfigureDistributedRequest,
    ReconfigureRankType,
    UtilityOutput,
    UtilityResult,
)
from vllm.v1.engine.utils import (
    EngineHandshakeMetadata,
    EngineZmqAddresses,
    get_device_indices,
)
from vllm.v1.executor import Executor
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.structured_output import StructuredOutputManager
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

POLLING_TIMEOUT_S = 2.5
HANDSHAKE_TIMEOUT_MINS = 5

_R = TypeVar("_R")  # Return type for collective_rpc


class EngineCore:
    """Inner loop of vLLM's Engine."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        executor_fail_callback: Callable | None = None,
    ):
        # plugins need to be loaded at the engine/scheduler level too
        from vllm.plugins import load_general_plugins

        load_general_plugins()

        self.vllm_config = vllm_config
        if vllm_config.parallel_config.data_parallel_rank == 0:
            logger.info(
                "Initializing a V1 LLM engine (v%s) with config: %s",
                VLLM_VERSION,
                vllm_config,
            )

        self.log_stats = log_stats

        # Setup Model.
        self.model_executor = executor_class(vllm_config)
        if executor_fail_callback is not None:
            self.model_executor.register_failure_callback(executor_fail_callback)

        self.available_gpu_memory_for_kv_cache = -1

        # Setup KV Caches and update CacheConfig after profiling.
        num_gpu_blocks, num_cpu_blocks, kv_cache_config = self._initialize_kv_caches(
            vllm_config
        )

        vllm_config.cache_config.num_gpu_blocks = num_gpu_blocks
        vllm_config.cache_config.num_cpu_blocks = num_cpu_blocks
        self.collective_rpc("initialize_cache", args=(num_gpu_blocks, num_cpu_blocks))

        self.structured_output_manager = StructuredOutputManager(vllm_config)

        # Setup scheduler.
        Scheduler = vllm_config.scheduler_config.get_scheduler_cls()

        if len(kv_cache_config.kv_cache_groups) == 0:
            # Encoder models without KV cache don't support
            # chunked prefill. But do SSM models?
            logger.info("Disabling chunked prefill for model without KVCache")
            vllm_config.scheduler_config.enable_chunked_prefill = False

        scheduler_block_size = (
            vllm_config.cache_config.block_size
            * vllm_config.parallel_config.decode_context_parallel_size
            * vllm_config.parallel_config.prefill_context_parallel_size
        )

        self.scheduler: SchedulerInterface = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=self.structured_output_manager,
            include_finished_set=vllm_config.parallel_config.data_parallel_size > 1,
            log_stats=self.log_stats,
            block_size=scheduler_block_size,
        )
        kv_cache_manager = getattr(self.scheduler, "kv_cache_manager", None)
        self._validate_uvm_dynamic_pool_single_gpu(
            vllm_config,
            getattr(kv_cache_manager, "uvm_kv_runtime_policy", None),
        )
        self.use_spec_decode = vllm_config.speculative_config is not None
        if self.scheduler.connector is not None:  # type: ignore
            self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # type: ignore

        self.mm_registry = mm_registry = MULTIMODAL_REGISTRY
        self.mm_receiver_cache = engine_receiver_cache_from_config(
            vllm_config, mm_registry
        )

        # If a KV connector is initialized for scheduler, we want to collect
        # handshake metadata from all workers so the connector in the scheduler
        # will have the full context
        kv_connector = self.scheduler.get_kv_connector()
        if kv_connector is not None:
            # Collect and store KV connector xfer metadata from workers
            # (after KV cache registration)
            xfer_handshake_metadata = (
                self.model_executor.get_kv_connector_handshake_metadata()
            )

            if xfer_handshake_metadata:
                # xfer_handshake_metadata is list of dicts from workers
                # Each dict already has structure {tp_rank: metadata}
                # Merge all worker dicts into a single dict
                content: dict[int, Any] = {}
                for worker_dict in xfer_handshake_metadata:
                    if worker_dict is not None:
                        content.update(worker_dict)
                kv_connector.set_xfer_handshake_metadata(content)

        # Setup batch queue for pipeline parallelism.
        # Batch queue for scheduled batches. This enables us to asynchronously
        # schedule and execute batches, and is required by pipeline parallelism
        # to eliminate pipeline bubbles.
        self.batch_queue_size = self.model_executor.max_concurrent_batches
        self.batch_queue: (
            deque[tuple[Future[ModelRunnerOutput], SchedulerOutput]] | None
        ) = None
        if self.batch_queue_size > 1:
            logger.info("Batch queue is enabled with size %d", self.batch_queue_size)
            self.batch_queue = deque(maxlen=self.batch_queue_size)
        self._uvm_pending_kv_grow_target_blocks: int | None = None
        self._uvm_pending_kv_grow_reason = ""
        self._uvm_pending_kv_shrink_target_blocks: int | None = None
        self._uvm_pending_kv_shrink_reason = ""
        self._uvm_kv_autoscale_step = 0
        self._uvm_kv_autoscale_low_streak = 0
        self._uvm_kv_autoscale_last_resize_step = 0
        self._uvm_kv_autoscale_predicted_blocks = 0
        self._uvm_kv_autoscale_prediction_error_blocks = 0.0
        self._uvm_kv_autoscale_expert_restore_pending = False
        self._uvm_kv_autoscale_pending_plan_class: str | None = None
        self._uvm_kv_autoscale_last_decision: dict[str, Any] = {}
        # The ordinary (non-pipelined) step loop is serialized, but keeping an
        # explicit execution guard makes scheduler-boundary maintenance fail
        # closed if a future call site becomes re-entrant.
        self._uvm_model_execution_inflight = False
        self._uvm_dynamic_expert_heat_ewma: dict[str, list[float]] = {}
        self._uvm_dynamic_expert_last_rebalance_ns = 0
        self._uvm_dynamic_expert_rebalance_generation = 0

        self.is_ec_producer = (
            vllm_config.ec_transfer_config is not None
            and vllm_config.ec_transfer_config.is_ec_producer
        )
        self.is_pooling_model = vllm_config.model_config.runner_type == "pooling"

        self.request_block_hasher: Callable[[Request], list[BlockHash]] | None = None
        if vllm_config.cache_config.enable_prefix_caching or kv_connector is not None:
            caching_hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(caching_hash_fn)

            self.request_block_hasher = get_request_block_hasher(
                scheduler_block_size, caching_hash_fn
            )

        self._maybe_apply_pre_service_uvm_kv_dynamic_pool()

        self.step_fn = (
            self.step if self.batch_queue is None else self.step_with_batch_queue
        )
        self.async_scheduling = vllm_config.scheduler_config.async_scheduling

        # Mark the startup heap as static so that it's ignored by GC.
        # Reduces pause times of oldest generation collections.
        freeze_gc_heap()
        # If enable, attach GC debugger after static variable freeze.
        maybe_attach_gc_debug_callback()

    def _initialize_kv_caches(
        self, vllm_config: VllmConfig
    ) -> tuple[int, int, KVCacheConfig]:
        start = time.time()

        # Get all kv cache needed by the model
        kv_cache_specs = self.model_executor.get_kv_cache_specs()

        has_kv_cache = any(kv_cache_spec for kv_cache_spec in kv_cache_specs)
        if has_kv_cache:
            if os.environ.get("VLLM_ELASTIC_EP_SCALE_UP_LAUNCH") == "1":
                dp_group = getattr(self, "dp_group", None)
                assert dp_group is not None
                self.available_gpu_memory_for_kv_cache = (
                    ParallelConfig.sync_kv_cache_memory_size(dp_group, -1)
                )
                available_gpu_memory = [self.available_gpu_memory_for_kv_cache] * len(
                    kv_cache_specs
                )
            else:
                # Profiles the peak memory usage of the model to determine how
                # much memory can be allocated for kv cache.
                available_gpu_memory = self.model_executor.determine_available_memory()
                self.available_gpu_memory_for_kv_cache = available_gpu_memory[0]
        else:
            # Attention free models don't need memory for kv cache
            available_gpu_memory = [0] * len(kv_cache_specs)

        assert len(kv_cache_specs) == len(available_gpu_memory)

        kv_cache_configs = get_kv_cache_configs(
            vllm_config, kv_cache_specs, available_gpu_memory
        )
        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        num_gpu_blocks = scheduler_kv_cache_config.num_blocks
        num_cpu_blocks = 0

        # Initialize kv cache and warmup the execution
        self.model_executor.initialize_from_config(kv_cache_configs)

        elapsed = time.time() - start
        logger.info_once(
            "init engine (profile, create kv cache, warmup model) took %.2f seconds",
            elapsed,
            scope="local",
        )
        return num_gpu_blocks, num_cpu_blocks, scheduler_kv_cache_config

    def _maybe_apply_pre_service_uvm_kv_dynamic_pool(self) -> None:
        stage = (
            os.environ.get(
                "VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE",
                "post_warmup",
            )
            .strip()
            .lower()
            .replace("-", "_")
        )
        post_warmup_target = (
            os.environ.get("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "")
            or os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_POST_STAGE_M_TARGET_BLOCKS", "")
        ).strip()
        initial_target = os.environ.get(
            "VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS",
            "",
        ).strip()
        explicit_apply = (
            os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_APPLY", "0") == "1"
        )
        automatic_post_warmup_apply = bool(
            post_warmup_target or initial_target
        ) and stage in ("post_warmup", "after_warmup", "deferred")
        if not (explicit_apply or automatic_post_warmup_apply):
            return

        target_text = (
            os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_TARGET_BLOCKS", "")
            or os.environ.get("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "")
            or os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_POST_STAGE_M_TARGET_BLOCKS", "")
            or os.environ.get("VLLM_UVM_KV_DYNAMIC_POOL_INITIAL_BLOCKS", "")
        ).strip()
        if not target_text:
            logger.warning(
                "UVM KV pre-service dynamic pool apply requested, but no target "
                "blocks env was provided"
            )
            return

        try:
            target_blocks = int(target_text)
        except ValueError:
            logger.warning(
                "Ignoring invalid UVM KV pre-service target blocks value: %r",
                target_text,
            )
            return

        kv_cache_manager = getattr(self.scheduler, "kv_cache_manager", None)
        policy = getattr(kv_cache_manager, "uvm_kv_runtime_policy", None)
        block_pool = getattr(kv_cache_manager, "block_pool", None)
        if policy is None or block_pool is None:
            logger.warning(
                "UVM KV pre-service dynamic pool apply skipped: "
                "policy or block pool unavailable"
            )
            return

        self._apply_uvm_kv_dynamic_pool_target(
            target_blocks,
            policy=policy,
            block_pool=block_pool,
            reason="pre_service_before_ready",
            metadata={"physical_migration_complete": True},
        )

        grow_text = (
            os.environ.get(
                "VLLM_UVM_KV_DYNAMIC_POOL_PRE_SERVICE_GROW_TARGET_BLOCKS", ""
            )
            or os.environ.get(
                "VLLM_UVM_KV_DYNAMIC_POOL_POST_STAGE_M_GROW_TARGET_BLOCKS", ""
            )
        ).strip()
        if not grow_text:
            return

        try:
            grow_target_blocks = int(grow_text)
        except ValueError:
            logger.warning(
                "Ignoring invalid UVM KV pre-service grow target blocks value: %r",
                grow_text,
            )
            return

        if grow_target_blocks <= 0:
            return

        if grow_target_blocks == target_blocks:
            return

        self._resize_uvm_kv_target(
            grow_target_blocks,
            reason="pre_service_before_ready_grow",
            release_expert_wave=False,
        )

    def _apply_uvm_kv_dynamic_pool_target(
        self,
        target_blocks: int,
        *,
        policy: Any,
        block_pool: Any,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return policy.apply_unified_kv_target_blocks(
            target_blocks,
            block_pool=block_pool,
            reason=reason,
            metadata=metadata,
        )

    @staticmethod
    def _uvm_env_flag(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _uvm_env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)) or default)
        except ValueError:
            return default

    @staticmethod
    def _uvm_env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)) or default)
        except ValueError:
            return default

    def _ensure_uvm_kv_autoscale_state(self) -> None:
        """Initialize autoscale state for normal and object.__new__ test cores."""
        defaults: dict[str, Any] = {
            "_uvm_pending_kv_shrink_target_blocks": None,
            "_uvm_pending_kv_shrink_reason": "",
            "_uvm_kv_autoscale_step": 0,
            "_uvm_kv_autoscale_low_streak": 0,
            "_uvm_kv_autoscale_last_resize_step": 0,
            "_uvm_kv_autoscale_predicted_blocks": 0,
            "_uvm_kv_autoscale_prediction_error_blocks": 0.0,
            "_uvm_kv_autoscale_expert_restore_pending": False,
            "_uvm_kv_autoscale_pending_plan_class": None,
            "_uvm_kv_autoscale_last_decision": {},
            "_uvm_model_execution_inflight": False,
        }
        for name, value in defaults.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    def _uvm_scheduler_quiescent_for_resize(self) -> bool:
        """Return whether residency may change between scheduler steps.

        Running requests do not make the ordinary v1 loop unsafe: after its
        model future has completed, their read-only expert placement and free
        KV tail may change before the next ``schedule()`` call.  Pipelined
        batch queues remain conservative because they can retain worker
        futures whose ranges are not represented by ``scheduler.running``.
        """
        self._ensure_uvm_kv_autoscale_state()
        if bool(self._uvm_model_execution_inflight):
            return False
        batch_queue = getattr(self, "batch_queue", None)
        if batch_queue is None:
            return True
        # Online switching is intentionally disabled for the pipelined path.
        # It may still switch at the old globally-idle boundary.
        return not batch_queue and not self.scheduler.has_unfinished_requests()

    def _uvm_scheduler_boundary_resize_enabled(self) -> bool:
        return self._uvm_env_flag(
            "VLLM_UVM_KV_SCHEDULER_BOUNDARY_RESIZE_ENABLE",
            default=False,
        )

    @staticmethod
    def _align_uvm_kv_blocks_up(value: int, granularity: int) -> int:
        width = max(int(granularity), 1)
        return ((max(int(value), 0) + width - 1) // width) * width

    @staticmethod
    def _align_uvm_kv_blocks_down(value: int, granularity: int) -> int:
        width = max(int(granularity), 1)
        return (max(int(value), 0) // width) * width

    def _uvm_kv_autoscale_config(self, policy: Any) -> dict[str, Any]:
        total = max(int(getattr(policy, "total_blocks", 0)), 0)
        backup_resolver = getattr(policy, "get_backup_reserve_blocks", None)
        backup = max(
            int(
                backup_resolver()
                if callable(backup_resolver)
                else getattr(policy, "backup_reserve_blocks", 0)
            ),
            0,
        )
        granularity = max(
            self._uvm_env_int(
                "VLLM_UVM_KV_AUTO_SCALE_RESIZE_GRANULARITY_BLOCKS",
                1,
            ),
            1,
        )
        safety = max(
            self._uvm_env_int("VLLM_UVM_KV_AUTO_SCALE_SAFETY_BLOCKS", 0),
            0,
        )
        initial = max(int(getattr(policy, "dynamic_pool_initial_blocks", 0)), 0)
        default_minimum = initial if initial > 0 else backup + safety + 1
        minimum = self._uvm_env_int(
            "VLLM_UVM_KV_AUTO_SCALE_MIN_TARGET_BLOCKS",
            default_minimum,
        )
        # Keep at least one ordinary allocation block in addition to reserve
        # and prediction headroom. The total-block clamp remains authoritative.
        minimum = min(max(minimum, backup + safety + 1), total)
        minimum = min(minimum, total)
        maximum = self._uvm_env_int(
            "VLLM_UVM_KV_AUTO_SCALE_MAX_TARGET_BLOCKS",
            total,
        )
        maximum = min(max(maximum, 0), total)
        allowed_targets: list[int] = []
        allowed_raw = os.environ.get(
            "VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS",
            "",
        ).strip()
        if allowed_raw:
            try:
                parsed_targets = [
                    int(item.strip())
                    for item in allowed_raw.split(",")
                    if item.strip()
                ]
                if (
                    not parsed_targets
                    or parsed_targets != sorted(set(parsed_targets))
                    or any(target <= 0 or target > total for target in parsed_targets)
                ):
                    raise ValueError(
                        "targets must be unique, increasing, positive, and within the pool"
                    )
                allowed_targets = [
                    target
                    for target in parsed_targets
                    if minimum <= target <= maximum
                ]
                if not allowed_targets:
                    raise ValueError("no allowed target remains within configured bounds")
                minimum = allowed_targets[0]
                maximum = allowed_targets[-1]
            except ValueError as exc:
                logger.warning(
                    "Ignoring invalid "
                    "VLLM_UVM_KV_AUTO_SCALE_ALLOWED_TARGET_BLOCKS: %s",
                    exc,
                )
                allowed_targets = []
        if not allowed_targets:
            minimum = min(
                self._align_uvm_kv_blocks_up(minimum, granularity),
                total,
            )
            maximum = self._align_uvm_kv_blocks_down(
                maximum,
                granularity,
            )
        bounds_valid = minimum <= maximum
        high_watermark = min(
            max(
                self._uvm_env_float(
                    "VLLM_UVM_KV_AUTO_SCALE_HIGH_WATERMARK",
                    0.80,
                ),
                0.01,
            ),
            1.0,
        )
        backup_only_growth = self._uvm_env_flag(
            "VLLM_UVM_KV_AUTO_SCALE_BACKUP_ONLY",
            default=False,
        )
        backup_pressure_threshold = max(
            self._uvm_env_int(
                "VLLM_UVM_KV_AUTO_SCALE_BACKUP_PRESSURE_THRESHOLD_BLOCKS",
                1,
            ),
            1,
        )
        return {
            "minimum_blocks": minimum,
            "maximum_blocks": maximum,
            "bounds_valid": bounds_valid,
            "bounds_error": (
                "aligned_minimum_exceeds_hbm_upper_bound"
                if not bounds_valid
                else ""
            ),
            "backup_blocks": backup,
            "safety_blocks": safety,
            "low_pressure_streak": max(
                self._uvm_env_int(
                    "VLLM_UVM_KV_AUTO_SCALE_LOW_PRESSURE_STREAK",
                    3,
                ),
                1,
            ),
            "grow_factor": max(
                self._uvm_env_int("VLLM_UVM_KV_AUTO_SCALE_GROW_FACTOR", 2),
                2,
            ),
            "shrink_divisor": max(
                self._uvm_env_int(
                    "VLLM_UVM_KV_AUTO_SCALE_SHRINK_DIVISOR",
                    2,
                ),
                2,
            ),
            "high_watermark": high_watermark,
            "active_high_watermark_growth": self._uvm_env_flag(
                "VLLM_UVM_KV_AUTO_SCALE_ACTIVE_HIGH_WATERMARK_GROW",
                default=False,
            ),
            "backup_only_growth": backup_only_growth,
            "backup_pressure_threshold_blocks": backup_pressure_threshold,
            "minimum_residency_steps": max(
                self._uvm_env_int(
                    "VLLM_UVM_KV_AUTO_SCALE_MIN_RESIDENCY_STEPS",
                    3,
                ),
                0,
            ),
            "resize_granularity_blocks": granularity,
            "allowed_target_blocks": allowed_targets,
            "synchronized_tier_steps": self._uvm_env_flag(
                "VLLM_UVM_KV_AUTO_SCALE_SYNCHRONIZED_TIER_STEPS",
                default=False,
            ),
            "prediction_error_alpha": min(
                max(
                    self._uvm_env_float(
                        "VLLM_UVM_KV_AUTO_SCALE_PREDICTION_ERROR_ALPHA",
                        0.25,
                    ),
                    0.0,
                ),
                1.0,
            ),
        }

    def _uvm_kv_autoscale_queue_snapshot(
        self,
        *,
        block_pool: Any,
        current_target: int,
    ) -> dict[str, Any]:
        snapshot_fn = getattr(block_pool, "snapshot_uvm_dynamic_queues", None)
        if callable(snapshot_fn):
            snapshot = snapshot_fn()
            if isinstance(snapshot, dict):
                snapshot = dict(snapshot)
                active_ids = snapshot.get("active_block_ids")
                if isinstance(active_ids, list):
                    normalized_active_ids = [
                        int(block_id)
                        for block_id in active_ids
                        if int(block_id) > 0
                    ]
                    max_active = max(normalized_active_ids, default=0)
                    snapshot.setdefault("active_block_ids_known", True)
                    snapshot.setdefault("max_active_block_id", max_active)
                    snapshot.setdefault(
                        "tail_free_start_block_id",
                        min(max_active + 1, int(current_target) + 1),
                    )
                    snapshot.setdefault(
                        "tail_free_block_count",
                        max(int(current_target) - max_active, 0),
                    )
                else:
                    snapshot.setdefault("active_block_ids_known", False)
                return snapshot
        free_blocks = 0
        get_free = getattr(block_pool, "get_num_free_blocks", None)
        if callable(get_free):
            free_blocks = max(int(get_free()), 0)
        return {
            "schema": "vllm_uvm_kv_autoscale_fallback_snapshot_v1",
            "applied_target_blocks": int(current_target),
            "active_blocks": max(int(current_target) - free_blocks, 0),
            "gpu_free_blocks": free_blocks,
            "gpu_backup_blocks": 0,
            "cpu_offline_blocks": 0,
            "active_block_ids_known": False,
            "max_active_block_id": None,
            "tail_free_start_block_id": None,
            "tail_free_block_count": None,
            "invariant_ok": True,
        }

    def _uvm_kv_autoscale_plan_class_for_target(
        self,
        target_blocks: int,
    ) -> str | None:
        """Resolve a model-specific KV target to a pre-captured Graph class."""
        raw = os.environ.get(
            "VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_MAP",
            "",
        ).strip()
        if not raw:
            return None
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict) or not payload:
                raise ValueError("mapping must be a non-empty JSON object")
            thresholds = sorted(
                (int(limit), str(class_name).strip())
                for limit, class_name in payload.items()
            )
            if any(limit < 0 or not class_name for limit, class_name in thresholds):
                raise ValueError("limits must be non-negative and classes non-empty")
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Ignoring invalid VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_MAP: %s",
                exc,
            )
            return None
        target = max(int(target_blocks), 0)
        for limit, class_name in thresholds:
            if target <= limit:
                return class_name
        return thresholds[-1][1]

    def _apply_pending_uvm_expert_plan_class_if_ready(
        self,
    ) -> dict[str, Any] | None:
        self._ensure_uvm_kv_autoscale_state()
        class_name = self._uvm_kv_autoscale_pending_plan_class
        if class_name is None:
            return None
        if self._uvm_scheduler_boundary_resize_enabled():
            if not self._uvm_scheduler_quiescent_for_resize():
                return None
            result = self._activate_uvm_expert_graph_class_at_quiescent_boundary(
                class_name
            )
        else:
            if self.scheduler.has_unfinished_requests() or getattr(
                self,
                "batch_queue",
                None,
            ):
                return None
            result = self.activate_uvm_expert_graph_class(class_name)
        if bool(result.get("ok")):
            self._uvm_kv_autoscale_pending_plan_class = None
        else:
            logger.warning(
                "KV autoscale plan-class activation remains pending: %r",
                result,
            )
        return result

    def _note_uvm_kv_autoscale_resize(self, target_blocks: int) -> None:
        self._ensure_uvm_kv_autoscale_state()
        self._uvm_kv_autoscale_last_resize_step = int(
            self._uvm_kv_autoscale_step
        )
        self._uvm_kv_autoscale_low_streak = 0
        self._uvm_pending_kv_shrink_target_blocks = None
        self._uvm_pending_kv_shrink_reason = ""
        target_class = self._uvm_kv_autoscale_plan_class_for_target(
            int(target_blocks)
        )
        self._uvm_kv_autoscale_pending_plan_class = target_class
        logger.info(
            "UVM KV autoscale committed target=%d at step=%d",
            int(target_blocks),
            int(self._uvm_kv_autoscale_step),
        )

    def _cancel_uvm_pending_kv_shrink(self, reason: str) -> None:
        self._ensure_uvm_kv_autoscale_state()
        if self._uvm_pending_kv_shrink_target_blocks is not None:
            logger.info(
                "Cancelling pending UVM KV shrink target=%d: %s",
                int(self._uvm_pending_kv_shrink_target_blocks),
                reason,
            )
        self._uvm_pending_kv_shrink_target_blocks = None
        self._uvm_pending_kv_shrink_reason = ""

    def _request_uvm_kv_autoscale_growth(
        self,
        *,
        required_blocks: int,
        reason: str,
        urgent: bool,
    ) -> dict[str, Any]:
        components = self._uvm_dynamic_pool_components()
        if components is None:
            return {"ok": False, "reason": "dynamic_pool_unavailable"}
        policy, _ = components
        self._ensure_uvm_kv_autoscale_state()
        config = self._uvm_kv_autoscale_config(policy)
        old_target = int(policy._dynamic_pool_applied_target_blocks)
        if not bool(config["bounds_valid"]):
            self._uvm_kv_autoscale_low_streak = 0
            self._cancel_uvm_pending_kv_shrink("invalid_autoscale_bounds")
            return {
                "ok": False,
                "reason": str(config["bounds_error"]),
                "old_target_blocks": old_target,
                "new_target_blocks": old_target,
            }
        residency_steps = int(self._uvm_kv_autoscale_step) - int(
            self._uvm_kv_autoscale_last_resize_step
        )
        scheduler_unfinished = bool(self.scheduler.has_unfinished_requests())
        if (
            not urgent
            and residency_steps < int(config["minimum_residency_steps"])
        ):
            return {
                "ok": True,
                "reason": "minimum_residency_not_met",
                "old_target_blocks": old_target,
                "new_target_blocks": old_target,
                "residency_steps": residency_steps,
            }

        # By default a non-urgent signal keeps the tier that still fits the
        # active cohort. An explicit proactive policy may grow at high water
        # while requests run; it still obeys minimum residency and the safe
        # scheduler-boundary checks below.
        if (
            scheduler_unfinished
            and not urgent
            and not bool(config["active_high_watermark_growth"])
        ):
            self._uvm_kv_autoscale_low_streak = 0
            self._cancel_uvm_pending_kv_shrink(
                "active_cohort_high_watermark_pressure"
            )
            return {
                "ok": True,
                "reason": "active_cohort_fits_current_target",
                "old_target_blocks": old_target,
                "new_target_blocks": old_target,
                "pending": False,
                "residency_steps": residency_steps,
            }

        desired_target = min(
            max(
                old_target * int(config["grow_factor"]),
                int(required_blocks),
                int(config["minimum_blocks"]),
            ),
            int(config["maximum_blocks"]),
        )
        allowed_targets = list(config.get("allowed_target_blocks") or [])
        if allowed_targets:
            target = next(
                (
                    int(candidate)
                    for candidate in allowed_targets
                    if int(candidate) >= desired_target
                ),
                int(allowed_targets[-1]),
            )
        else:
            target = self._align_uvm_kv_blocks_up(
                desired_target,
                int(config["resize_granularity_blocks"]),
            )
            target = min(
                max(target, int(config["minimum_blocks"])),
                int(config["maximum_blocks"]),
            )
        self._uvm_kv_autoscale_low_streak = 0
        self._uvm_kv_autoscale_expert_restore_pending = False
        self._cancel_uvm_pending_kv_shrink("growth_pressure")
        if target <= old_target:
            return {
                "ok": True,
                "reason": "already_at_autoscale_upper_bound",
                "old_target_blocks": old_target,
                "new_target_blocks": old_target,
                "capacity_limited": int(required_blocks) > old_target,
            }

        batch_queue = getattr(self, "batch_queue", None)
        scheduler_boundary_enabled = (
            self._uvm_scheduler_boundary_resize_enabled()
        )
        online_boundary = (
            scheduler_boundary_enabled
            and self._uvm_scheduler_quiescent_for_resize()
        )
        if (
            scheduler_unfinished or (batch_queue is not None and batch_queue)
        ) and not online_boundary:
            pending_target = int(
                getattr(self, "_uvm_pending_kv_grow_target_blocks", 0) or 0
            )
            self._uvm_pending_kv_grow_target_blocks = max(pending_target, target)
            self._uvm_pending_kv_grow_reason = reason
            return {
                "ok": True,
                "reason": (
                    "growth_deferred_until_safe_scheduler_boundary"
                    if scheduler_boundary_enabled
                    else "growth_deferred_until_request_boundary"
                ),
                "old_target_blocks": old_target,
                "new_target_blocks": target,
                "pending": True,
                "scheduler_unfinished": scheduler_unfinished,
                "queued_batch_count": (
                    len(batch_queue) if batch_queue is not None else 0
                ),
            }

        if bool(config.get("synchronized_tier_steps")):
            result = self._commit_uvm_kv_synchronized_growth_path(
                old_target=old_target,
                target=target,
                reason=reason,
                config=config,
            )
        else:
            result = self._resize_uvm_kv_target(
                target,
                reason=reason,
                release_expert_wave=self._uvm_env_flag(
                    "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE",
                    default=False,
                ),
            )
            if bool(result.get("ok")):
                self._note_uvm_kv_autoscale_resize(target)
                if isinstance(result.get("plan_class_before_grow"), dict):
                    # The target class was committed synchronously before the
                    # physical KV grow.  Do not enqueue an identical activation
                    # for the later idle boundary.
                    self._uvm_kv_autoscale_pending_plan_class = None
        return {
            **result,
            "old_target_blocks": old_target,
            "new_target_blocks": target,
            "pending": False,
            "scheduler_boundary_resize_enabled": scheduler_boundary_enabled,
            "committed_with_running_requests": bool(
                scheduler_unfinished and online_boundary
            ),
        }

    def _commit_uvm_kv_synchronized_growth_path(
        self,
        *,
        old_target: int,
        target: int,
        reason: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Commit adjacent KV/Plan-Class tiers as separate transactions."""
        allowed_targets = [
            int(value) for value in (config.get("allowed_target_blocks") or [])
        ]
        if old_target not in allowed_targets or target not in allowed_targets:
            return {
                "ok": False,
                "reason": "synchronized_tier_endpoint_not_registered",
                "old_target_blocks": int(old_target),
                "new_target_blocks": int(target),
            }
        old_index = allowed_targets.index(int(old_target))
        target_index = allowed_targets.index(int(target))
        if target_index <= old_index:
            return {"ok": True, "reason": "already_at_target", "tier_steps": []}

        tier_results: list[dict[str, Any]] = []
        applied_target = int(old_target)
        for step_target in allowed_targets[old_index + 1 : target_index + 1]:
            step_result = self._resize_uvm_kv_target(
                int(step_target),
                reason=reason,
                release_expert_wave=self._uvm_env_flag(
                    "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE",
                    default=False,
                ),
            )
            tier_results.append(
                {
                    "old_target_blocks": applied_target,
                    "new_target_blocks": int(step_target),
                    "result": step_result,
                }
            )
            if not bool(step_result.get("ok")):
                return {
                    "ok": False,
                    "reason": "synchronized_tier_step_failed",
                    "requested_target_blocks": int(target),
                    "applied_target_blocks": applied_target,
                    "tier_steps": tier_results,
                }
            applied_target = int(step_target)
            self._note_uvm_kv_autoscale_resize(applied_target)
            if isinstance(step_result.get("plan_class_before_grow"), dict):
                self._uvm_kv_autoscale_pending_plan_class = None

        return {
            "ok": True,
            "reason": "applied_synchronized_tier_steps",
            "requested_target_blocks": int(target),
            "applied_target_blocks": applied_target,
            "tier_steps": tier_results,
        }

    def _observe_uvm_kv_autoscale_pressure(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[str, Any]:
        components = self._uvm_dynamic_pool_components()
        if components is None:
            return {"ok": False, "reason": "dynamic_pool_unavailable"}
        policy, block_pool = components
        self._ensure_uvm_kv_autoscale_state()
        self._uvm_kv_autoscale_step += 1
        step = int(self._uvm_kv_autoscale_step)
        config = self._uvm_kv_autoscale_config(policy)
        current_target = int(policy._dynamic_pool_applied_target_blocks)
        if not bool(config["bounds_valid"]):
            self._uvm_kv_autoscale_low_streak = 0
            self._cancel_uvm_pending_kv_shrink("invalid_autoscale_bounds")
            decision = {
                "schema": "vllm_uvm_kv_autoscale_decision_v1",
                "ok": False,
                "step": step,
                "action": "hold_invalid_autoscale_bounds",
                "reason": str(config["bounds_error"]),
                "current_target_blocks": current_target,
                "low_streak": 0,
                "config": config,
            }
            self._uvm_kv_autoscale_last_decision = decision
            return decision
        snapshot = self._uvm_kv_autoscale_queue_snapshot(
            block_pool=block_pool,
            current_target=current_target,
        )
        active_blocks = max(int(snapshot.get("active_blocks", 0)), 0)

        consume_backup = getattr(policy, "consume_backup_activation", None)
        backup_pressure = (
            consume_backup()
            if callable(consume_backup)
            else {"blocks": 0, "events": 0}
        )
        backup_blocks = max(int(backup_pressure.get("blocks", 0)), 0)
        preemption_count = len(
            getattr(scheduler_output, "preempted_req_ids", None) or ()
        )
        backup_only_growth = bool(config["backup_only_growth"])
        predicted_blocks = (
            0
            if backup_only_growth
            else max(int(self._uvm_kv_autoscale_predicted_blocks), 0)
        )
        previous_error = max(
            float(self._uvm_kv_autoscale_prediction_error_blocks),
            0.0,
        )
        alpha = float(config["prediction_error_alpha"])
        if backup_only_growth:
            self._uvm_kv_autoscale_predicted_blocks = 0
            self._uvm_kv_autoscale_prediction_error_blocks = 0.0
        elif predicted_blocks > 0:
            observed_error = max(active_blocks - predicted_blocks, 0)
            self._uvm_kv_autoscale_prediction_error_blocks = (
                alpha * observed_error + (1.0 - alpha) * previous_error
            )
            self._uvm_kv_autoscale_predicted_blocks = 0
        else:
            self._uvm_kv_autoscale_prediction_error_blocks = (
                1.0 - alpha
            ) * previous_error
        error_blocks = (
            0
            if backup_only_growth
            else math.ceil(self._uvm_kv_autoscale_prediction_error_blocks)
        )
        required_blocks = (
            max(active_blocks, predicted_blocks)
            + int(config["backup_blocks"])
            + int(config["safety_blocks"])
            + error_blocks
        )
        high_threshold = max(
            math.ceil(current_target * float(config["high_watermark"])),
            1,
        )
        backup_target_blocks = max(
            int(snapshot.get("backup_target_blocks", config["backup_blocks"])),
            0,
        )
        backup_remaining_blocks = max(
            int(snapshot.get("gpu_backup_blocks", backup_target_blocks)),
            0,
        )
        backup_pressure_level = max(
            backup_blocks,
            backup_target_blocks - backup_remaining_blocks,
        )
        backup_threshold_reached = (
            backup_blocks > 0
            and backup_pressure_level
            >= int(config["backup_pressure_threshold_blocks"])
        )
        urgent_growth = (
            backup_threshold_reached
            if backup_only_growth
            else backup_blocks > 0 or preemption_count > 0
        )
        proactive_growth = (
            False
            if backup_only_growth
            else required_blocks >= high_threshold
        )
        eligible_round = (
            int(getattr(scheduler_output, "total_num_scheduled_tokens", 0)) > 0
        )
        dirty = bool(policy.get_dynamic_pool_reconciliation_block_ids()) or bool(
            getattr(policy, "_dynamic_pool_external_reconciliation_required", False)
        )
        residency_steps = step - int(self._uvm_kv_autoscale_last_resize_step)
        residency_ready = residency_steps >= int(
            config["minimum_residency_steps"]
        )

        decision: dict[str, Any] = {
            "schema": "vllm_uvm_kv_autoscale_decision_v1",
            "ok": True,
            "step": step,
            "current_target_blocks": current_target,
            "active_blocks": active_blocks,
            "predicted_blocks": predicted_blocks,
            "prediction_error_blocks": error_blocks,
            "required_blocks": required_blocks,
            "backup_activated_blocks": backup_blocks,
            "backup_activation_events": max(
                int(backup_pressure.get("events", 0)), 0
            ),
            "backup_pressure_level_blocks": backup_pressure_level,
            "backup_pressure_threshold_blocks": int(
                config["backup_pressure_threshold_blocks"]
            ),
            "backup_pressure_threshold_reached": backup_threshold_reached,
            "growth_trigger_policy": (
                "backup_threshold_only"
                if backup_only_growth
                else "backup_preemption_or_high_watermark"
            ),
            "preemption_count": preemption_count,
            "high_threshold_blocks": high_threshold,
            "eligible_round": eligible_round,
            "residency_steps": residency_steps,
            "queue_snapshot": snapshot,
            "config": config,
        }
        if snapshot.get("invariant_ok") is False:
            self._uvm_kv_autoscale_low_streak = 0
            self._cancel_uvm_pending_kv_shrink("queue_invariant_failed")
            decision.update(
                {
                    "ok": False,
                    "action": "hold_queue_invariant_failed",
                    "low_streak": 0,
                }
            )
            self._uvm_kv_autoscale_last_decision = decision
            return decision
        if urgent_growth or proactive_growth:
            self._uvm_kv_autoscale_low_streak = 0
            self._cancel_uvm_pending_kv_shrink("high_pressure")
            growth = self._request_uvm_kv_autoscale_growth(
                required_blocks=required_blocks,
                reason=(
                    "autoscale_backup_threshold_growth"
                    if backup_only_growth
                    else (
                        "autoscale_backup_or_preemption_growth"
                        if urgent_growth
                        else "autoscale_high_watermark_growth"
                    )
                ),
                urgent=urgent_growth or required_blocks > current_target,
            )
            decision.update(
                {
                    "action": "grow",
                    "growth": growth,
                    "low_streak": 0,
                }
            )
            self._uvm_kv_autoscale_last_decision = decision
            return decision

        desired_shrink_target = max(
            (current_target + int(config["shrink_divisor"]) - 1)
            // int(config["shrink_divisor"]),
            int(config["minimum_blocks"]),
        )
        allowed_targets = list(config.get("allowed_target_blocks") or [])
        if allowed_targets:
            # Frozen Plan Classes are discrete residency tiers. Move to the
            # immediate predecessor instead of rounding the arithmetic target
            # upward: for the E13 K2/K4/K6 geometry, ceil(K4 / 2) is 1365
            # while the registered K2 tier is 1364 blocks. Upward rounding
            # would therefore select K4 again and make K2 unreachable.
            shrink_target = max(
                (
                    int(candidate)
                    for candidate in allowed_targets
                    if int(candidate) < current_target
                ),
                default=current_target,
            )
        else:
            shrink_target = self._align_uvm_kv_blocks_up(
                desired_shrink_target,
                int(config["resize_granularity_blocks"]),
            )
            shrink_target = max(
                shrink_target,
                int(config["minimum_blocks"]),
            )
        active_ids_known = bool(snapshot.get("active_block_ids_known", False))
        max_active_value = snapshot.get("max_active_block_id")
        max_active_block_id = (
            max(int(max_active_value), 0)
            if active_ids_known and max_active_value is not None
            else None
        )
        candidate_tail_all_free = (
            max_active_block_id is not None
            and max_active_block_id <= shrink_target
        )
        # Keep the proof next to the queue snapshot in diagnostics.  This is
        # deliberately not derived from min(free IDs), which says nothing
        # about an active block near the current target.
        snapshot["candidate_shrink_target_blocks"] = shrink_target
        snapshot["candidate_tail_all_free"] = candidate_tail_all_free
        low_pressure = (
            eligible_round
            and residency_ready
            and not dirty
            and getattr(self, "_uvm_pending_kv_grow_target_blocks", None) is None
            and required_blocks <= shrink_target
            and shrink_target < current_target
            and candidate_tail_all_free
        )
        if low_pressure:
            self._uvm_kv_autoscale_low_streak += 1
            if self._uvm_kv_autoscale_low_streak >= int(
                config["low_pressure_streak"]
            ):
                self._uvm_pending_kv_shrink_target_blocks = shrink_target
                self._uvm_pending_kv_shrink_reason = (
                    "autoscale_sustained_low_pressure"
                )
                action = "shrink_pending"
            else:
                action = "observe_low_pressure"
        elif eligible_round:
            self._uvm_kv_autoscale_low_streak = 0
            self._cancel_uvm_pending_kv_shrink("low_pressure_streak_broken")
            action = "hold"
        else:
            action = "ignore_empty_round"
        decision.update(
            {
                "action": action,
                "low_streak": int(self._uvm_kv_autoscale_low_streak),
                "candidate_shrink_target_blocks": shrink_target,
                "max_active_block_id": max_active_block_id,
                "tail_free_start_block_id": snapshot.get(
                    "tail_free_start_block_id"
                ),
                "tail_free_block_count": snapshot.get("tail_free_block_count"),
                "candidate_tail_all_free": candidate_tail_all_free,
                "pending_shrink_target_blocks": (
                    self._uvm_pending_kv_shrink_target_blocks
                ),
            }
        )
        self._uvm_kv_autoscale_last_decision = decision
        return decision

    def _uvm_dynamic_pool_components(self) -> tuple[Any, Any] | None:
        kv_cache_manager = getattr(self.scheduler, "kv_cache_manager", None)
        policy = getattr(kv_cache_manager, "uvm_kv_runtime_policy", None)
        block_pool = getattr(kv_cache_manager, "block_pool", None)
        if policy is None or block_pool is None or not policy.dynamic_pool_active():
            return None
        return policy, block_pool

    @staticmethod
    def _validate_uvm_dynamic_pool_single_gpu(
        vllm_config: VllmConfig,
        policy: Any | None,
    ) -> None:
        if policy is None or not policy.dynamic_pool_active():
            return
        if int(vllm_config.parallel_config.world_size) != 1:
            raise RuntimeError("Current UVM dynamic KV resize supports one GPU only")

    def _uvm_single_worker_rpc(
        self,
        method: str,
        *,
        args: tuple[Any, ...],
    ) -> Any:
        results = self.collective_rpc(method, args=args)
        if len(results) != 1:
            raise RuntimeError("UVM dynamic KV resize requires one GPU worker")
        return results[0]

    def _resize_uvm_kv_target(
        self,
        target_blocks: int,
        *,
        reason: str,
        release_expert_wave: bool,
        allow_active_prefix_shrink: bool = False,
    ) -> dict[str, Any]:
        components = self._uvm_dynamic_pool_components()
        if components is None:
            return {"ok": False, "reason": "dynamic_pool_unavailable"}
        policy, block_pool = components
        old_target = int(policy._dynamic_pool_applied_target_blocks)
        target = min(max(int(target_blocks), 0), int(policy.total_blocks))
        autoscale_config = self._uvm_kv_autoscale_config(policy)
        synchronized_targets = [
            int(value)
            for value in (autoscale_config.get("allowed_target_blocks") or [])
        ]
        if (
            bool(autoscale_config.get("synchronized_tier_steps"))
            and target != old_target
        ):
            if (
                old_target not in synchronized_targets
                or target not in synchronized_targets
                or abs(
                    synchronized_targets.index(target)
                    - synchronized_targets.index(old_target)
                )
                != 1
            ):
                return {
                    "ok": False,
                    "reason": "non_adjacent_synchronized_tier_resize",
                    "old_target_blocks": old_target,
                    "new_target_blocks": target,
                    "allowed_target_blocks": synchronized_targets,
                }
        released_experts: dict[str, int] = {}
        plan_class_before_grow: dict[str, Any] | None = None
        previous_plan_class: str | None = None

        def rollback_released_experts() -> dict[str, Any] | None:
            if not released_experts:
                return None
            try:
                rollback = self._uvm_single_worker_rpc(
                    "run_uvm_incremental_expert_restore",
                    args=(released_experts,),
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "reason": "expert_release_rollback_rpc_failed",
                    "error": repr(exc),
                }
            return rollback if isinstance(rollback, dict) else {
                "ok": False,
                "reason": "invalid_expert_release_rollback_result",
            }

        def rollback_plan_class() -> dict[str, Any] | None:
            if plan_class_before_grow is None or previous_plan_class is None:
                return None
            try:
                rollback = self._uvm_single_worker_rpc(
                    "activate_uvm_expert_graph_class",
                    args=(previous_plan_class, True),
                )
            except Exception as exc:
                return {
                    "ok": False,
                    "reason": "plan_class_rollback_rpc_failed",
                    "error": repr(exc),
                }
            return rollback if isinstance(rollback, dict) else {
                "ok": False,
                "reason": "invalid_plan_class_rollback_result",
            }

        dirty_block_ids = policy.get_dynamic_pool_reconciliation_block_ids()
        if target == old_target and not dirty_block_ids:
            return {"ok": True, "reason": "already_at_target"}
        if (
            self._uvm_scheduler_boundary_resize_enabled()
            and not self._uvm_scheduler_quiescent_for_resize()
        ):
            return {"ok": False, "reason": "scheduler_not_quiescent"}
        if (
            target < old_target
            and self.scheduler.has_unfinished_requests()
            and not allow_active_prefix_shrink
        ):
            return {"ok": False, "reason": "scheduler_busy"}
        if target < old_target and allow_active_prefix_shrink:
            batch_queue = getattr(self, "batch_queue", None)
            if batch_queue is not None and batch_queue:
                return {"ok": False, "reason": "inflight_batch_queue"}

        if (
            target > old_target
            and self._uvm_env_flag(
                "VLLM_UVM_KV_AUTO_SCALE_PLAN_CLASS_BEFORE_GROW",
                default=False,
            )
        ):
            previous_plan_class = self._uvm_kv_autoscale_plan_class_for_target(
                old_target
            )
            target_plan_class = self._uvm_kv_autoscale_plan_class_for_target(
                target
            )
            if (
                target_plan_class is not None
                and target_plan_class != previous_plan_class
            ):
                try:
                    activation = (
                        self._activate_uvm_expert_graph_class_at_quiescent_boundary(
                            target_plan_class,
                            defer_dual_refresh=self._uvm_env_flag(
                                "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW",
                                default=False,
                            ),
                        )
                    )
                except Exception as exc:
                    activation = {
                        "ok": False,
                        "reason": "plan_class_activation_rpc_failed",
                        "error": repr(exc),
                    }
                if not isinstance(activation, dict) or not bool(
                    activation.get("ok")
                ):
                    return {
                        "ok": False,
                        "reason": "plan_class_activation_failed",
                        "plan_class_before_grow": activation,
                    }
                plan_class_before_grow = activation
                # The frozen tier difference already releases the required
                # experts as one grouped transaction. Do not also execute the
                # legacy per-layer incremental wave for this resize.
                release_expert_wave = False

        if release_expert_wave and target > old_target:
            try:
                candidates = self._uvm_single_worker_rpc(
                    "get_uvm_incremental_expert_offload_candidates",
                    args=("cold",),
                )
            except Exception:
                candidates = {"ok": False, "candidates": {}}
            if not isinstance(candidates, dict) or not bool(candidates.get("ok")):
                if self._uvm_env_flag("VLLM_UVM_KV_ADAPTIVE_FAIL_FAST", default=True):
                    raise RuntimeError("Adaptive expert selection failed")
                return {"ok": False, "reason": "expert_selection_failed"}

            allow_warm = not bool(candidates.get("candidates"))
            if allow_warm:
                try:
                    candidates = self._uvm_single_worker_rpc(
                        "get_uvm_incremental_expert_offload_candidates",
                        args=("warm",),
                    )
                except Exception:
                    candidates = {"ok": False, "candidates": {}}
                if not isinstance(candidates, dict) or not bool(candidates.get("ok")):
                    if self._uvm_env_flag(
                        "VLLM_UVM_KV_ADAPTIVE_FAIL_FAST", default=True
                    ):
                        raise RuntimeError("Adaptive expert selection failed")
                    return {
                        "ok": False,
                        "reason": "expert_selection_failed",
                    }

            selected_experts = candidates.get("candidates") or {}
            if not isinstance(selected_experts, dict):
                return {
                    "ok": False,
                    "reason": "expert_selection_invalid",
                }
            if selected_experts:
                released_experts = {
                    str(layer_key): int(expert_id)
                    for layer_key, expert_id in selected_experts.items()
                }
                try:
                    released = self._uvm_single_worker_rpc(
                        "run_uvm_incremental_expert_offload",
                        args=(allow_warm, selected_experts),
                    )
                except Exception:
                    released = {"ok": False}
                if not isinstance(released, dict) or not bool(released.get("ok")):
                    rollback = rollback_released_experts()
                    if self._uvm_env_flag(
                        "VLLM_UVM_KV_ADAPTIVE_FAIL_FAST", default=True
                    ):
                        raise RuntimeError("Adaptive expert release failed")
                    return {
                        "ok": False,
                        "reason": "expert_release_failed",
                        "expert_rollback": rollback,
                    }

        if not bool(
            getattr(policy, "dynamic_pool_physical_migration_enable", True)
        ) and not bool(
            getattr(
                policy,
                "_dynamic_pool_external_reconciliation_required",
                False,
            )
        ):
            applied = self._apply_uvm_kv_dynamic_pool_target(
                target,
                policy=policy,
                block_pool=block_pool,
                reason=reason,
            )
            result = {
                "ok": bool(applied.get("ok")),
                "reason": str(applied.get("reason") or "apply_failed"),
            }
            if plan_class_before_grow is not None:
                result["plan_class_before_grow"] = plan_class_before_grow
            if not result["ok"]:
                result["expert_rollback"] = rollback_released_experts()
                result["plan_class_rollback"] = rollback_plan_class()
            return result

        preparation = policy.prepare_dynamic_pool_resize(
            target,
            block_pool=block_pool,
            reason=reason,
        )
        if not bool(preparation.get("ok")):
            return {
                "ok": False,
                "reason": "prepare_failed",
                "expert_rollback": rollback_released_experts(),
                "plan_class_rollback": rollback_plan_class(),
                "plan_class_before_grow": plan_class_before_grow,
            }

        try:
            worker = self._uvm_single_worker_rpc(
                "resize_uvm_kv_cache_target",
                args=(old_target, target, dirty_block_ids),
            )
            if not isinstance(worker, dict):
                worker = {"ok": False, "made_progress": True}
        except Exception:
            # An RPC error cannot prove that no migration was submitted.
            worker = {"ok": False, "made_progress": True}

        finalized = policy.finalize_dynamic_pool_resize(
            preparation,
            block_pool=block_pool,
            worker_ok=bool(worker.get("ok")),
            worker_made_progress=bool(worker.get("made_progress")),
        )
        if not bool(finalized.get("ok")):
            safe_to_restore_experts = (
                not bool(worker.get("made_progress"))
                and not policy.get_dynamic_pool_reconciliation_block_ids()
                and not bool(
                    getattr(
                        policy,
                        "_dynamic_pool_external_reconciliation_required",
                        False,
                    )
                )
            )
            expert_rollback = (
                rollback_released_experts()
                if safe_to_restore_experts
                else None
            )
            plan_class_rollback = (
                rollback_plan_class() if safe_to_restore_experts else None
            )
            if self._uvm_env_flag(
                "VLLM_UVM_KV_ADAPTIVE_FAIL_FAST",
                default=True,
            ):
                raise RuntimeError("Adaptive KV resize failed")
            return {
                "ok": False,
                "reason": "physical_migration_failed",
                "expert_rollback": expert_rollback,
                "expert_rollback_deferred": bool(released_experts)
                and not safe_to_restore_experts,
                "plan_class_rollback": plan_class_rollback,
                "plan_class_rollback_deferred": (
                    plan_class_before_grow is not None
                    and not safe_to_restore_experts
                ),
                "plan_class_before_grow": plan_class_before_grow,
            }
        result = {"ok": True, "reason": "applied"}
        if plan_class_before_grow is not None:
            result["plan_class_before_grow"] = plan_class_before_grow
        return result

    def _maybe_handle_uvm_kv_preemption_pressure(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        adaptive_enabled = self._uvm_env_flag(
            "VLLM_UVM_KV_ADAPTIVE_PRESSURE_ENABLE",
            default=False,
        )
        components = self._uvm_dynamic_pool_components()
        if components is None:
            return
        if self._uvm_env_flag(
            "VLLM_UVM_KV_AUTO_SCALE_ENABLE",
            default=False,
        ):
            self._observe_uvm_kv_autoscale_pressure(scheduler_output)
            return
        policy, _ = components
        standalone_backup_grow_enabled = self._uvm_env_flag(
            "VLLM_UVM_KV_BACKUP_PRESSURE_STANDALONE_GROW_ENABLE",
            default=True,
        )
        consume_backup = getattr(policy, "consume_backup_activation", None)
        backup_pressure = (
            consume_backup()
            if consume_backup is not None and standalone_backup_grow_enabled
            else {"blocks": 0, "events": 0}
        )
        backup_blocks = max(int(backup_pressure.get("blocks", 0)), 0)
        preemption_count = len(
            getattr(scheduler_output, "preempted_req_ids", None) or ()
        )
        if backup_blocks == 0 and (not adaptive_enabled or preemption_count == 0):
            return

        old_target = int(policy._dynamic_pool_applied_target_blocks)
        preemption_grow = (
            max(
                self._uvm_env_int(
                    "VLLM_UVM_KV_ADAPTIVE_GROW_BLOCKS_PER_PREEMPTION",
                    128,
                ),
                1,
            ) * preemption_count
            if adaptive_enabled
            else 0
        )
        configured_backup_grow = max(
            int(getattr(policy, "backup_grow_blocks", 0)), 0
        )
        backup_grow = (
            max(configured_backup_grow, backup_blocks)
            if backup_blocks
            else 0
        )
        new_target = min(
            old_target + max(preemption_grow, backup_grow),
            int(policy.total_blocks),
        )
        if new_target <= old_target:
            logger.warning(
                "KV preemption pressure detected at maximum target=%d",
                old_target,
            )
            return
        batch_queue = getattr(self, "batch_queue", None)
        if batch_queue is not None and batch_queue:
            pending_target = int(
                getattr(self, "_uvm_pending_kv_grow_target_blocks", 0) or 0
            )
            self._uvm_pending_kv_grow_target_blocks = max(
                pending_target,
                new_target,
            )
            self._uvm_pending_kv_grow_reason = (
                "scheduler_preemption_pressure_after_batch_queue_drain"
            )
            logger.info(
                "Deferring UVM KV grow until %d in-flight batches drain: "
                "old_target=%d pending_target=%d preemptions=%d",
                len(batch_queue),
                old_target,
                self._uvm_pending_kv_grow_target_blocks,
                preemption_count,
            )
            return
        self._resize_uvm_kv_target(
            new_target,
            reason=(
                "backup_reserve_activation"
                if backup_blocks
                else "scheduler_preemption_pressure"
            ),
            release_expert_wave=(
                backup_blocks > 0
                or self._uvm_env_flag(
                    "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE",
                    default=False,
                )
            ),
        )

    def _apply_pending_uvm_kv_growth_if_ready(self) -> dict[str, Any] | None:
        target = getattr(self, "_uvm_pending_kv_grow_target_blocks", None)
        if target is None:
            return None
        if self._uvm_scheduler_boundary_resize_enabled():
            if not self._uvm_scheduler_quiescent_for_resize():
                return None
        else:
            batch_queue = getattr(self, "batch_queue", None)
            if batch_queue is not None and batch_queue:
                return None
            # Compatibility mode retains the old global-idle rule.
            running = getattr(self.scheduler, "running", None)
            if running is not None:
                if running:
                    return None
            elif self.scheduler.has_unfinished_requests():
                return None

        reason = (
            getattr(self, "_uvm_pending_kv_grow_reason", "")
            or "scheduler_preemption_pressure_after_batch_queue_drain"
        )
        self._uvm_pending_kv_grow_target_blocks = None
        self._uvm_pending_kv_grow_reason = ""
        components = self._uvm_dynamic_pool_components()
        policy = components[0] if components is not None else None
        current_target = (
            int(policy._dynamic_pool_applied_target_blocks)
            if policy is not None
            else int(target)
        )
        config = (
            self._uvm_kv_autoscale_config(policy)
            if policy is not None
            else {}
        )
        if bool(config.get("synchronized_tier_steps")):
            result = self._commit_uvm_kv_synchronized_growth_path(
                old_target=current_target,
                target=int(target),
                reason=reason,
                config=config,
            )
        else:
            result = self._resize_uvm_kv_target(
                int(target),
                reason=reason,
                release_expert_wave=self._uvm_env_flag(
                    "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE",
                    default=False,
                ),
            )
        if not bool(result.get("ok")):
            self._uvm_pending_kv_grow_target_blocks = int(target)
            self._uvm_pending_kv_grow_reason = reason
        elif not bool(config.get("synchronized_tier_steps")) and self._uvm_env_flag(
            "VLLM_UVM_KV_AUTO_SCALE_ENABLE",
            default=False,
        ):
            self._note_uvm_kv_autoscale_resize(int(target))
        return result

    def _restore_uvm_incremental_expert_wave(self) -> dict[str, Any]:
        """Restore at most one autoscale-offloaded expert in every MoE layer."""
        if not self._uvm_env_flag(
            "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE",
            default=False,
        ) or not self._uvm_env_flag(
            "VLLM_UVM_KV_AUTO_SCALE_EXPERT_RESTORE_ENABLE",
            default=True,
        ):
            return {"ok": True, "reason": "incremental_expert_restore_disabled"}
        try:
            candidates = self._uvm_single_worker_rpc(
                "get_uvm_incremental_expert_restore_candidates",
                args=(),
            )
        except Exception as exc:
            return {
                "ok": False,
                "reason": "expert_restore_selection_failed",
                "error": repr(exc),
            }
        if not isinstance(candidates, dict) or not bool(candidates.get("ok")):
            return {
                "ok": False,
                "reason": "expert_restore_selection_failed",
                "candidates": candidates,
            }
        selected = candidates.get("candidates") or {}
        if not selected:
            return {"ok": True, "reason": "no_incremental_experts_to_restore"}
        try:
            restored = self._uvm_single_worker_rpc(
                "run_uvm_incremental_expert_restore",
                args=(selected,),
            )
        except Exception as exc:
            return {
                "ok": False,
                "reason": "expert_restore_rpc_failed",
                "error": repr(exc),
                "selected_experts": selected,
            }
        if not isinstance(restored, dict):
            return {
                "ok": False,
                "reason": "invalid_expert_restore_result",
                "selected_experts": selected,
            }
        return {
            **restored,
            "selected_experts": selected,
        }

    def _apply_pending_uvm_kv_shrink_if_ready(self) -> dict[str, Any] | None:
        """Commit a geometric shrink only at an audited scheduler boundary."""
        self._ensure_uvm_kv_autoscale_state()
        target = self._uvm_pending_kv_shrink_target_blocks
        if target is None:
            online_restore_ready = (
                self._uvm_scheduler_boundary_resize_enabled()
                and self._uvm_scheduler_quiescent_for_resize()
            )
            if (
                self._uvm_kv_autoscale_expert_restore_pending
                and (
                    online_restore_ready
                    or (
                        not self.scheduler.has_unfinished_requests()
                        and not getattr(self, "batch_queue", None)
                    )
                )
            ):
                expert_restore = self._restore_uvm_incremental_expert_wave()
                self._uvm_kv_autoscale_expert_restore_pending = not bool(
                    expert_restore.get("ok")
                )
                return {
                    "ok": bool(expert_restore.get("ok")),
                    "reason": "pending_expert_restore_retried",
                    "expert_restore": expert_restore,
                }
            return None
        if getattr(self, "_uvm_pending_kv_grow_target_blocks", None) is not None:
            self._cancel_uvm_pending_kv_shrink("pending_growth_has_priority")
            return {"ok": False, "reason": "pending_growth_has_priority"}
        online_boundary = self._uvm_scheduler_boundary_resize_enabled()
        if online_boundary:
            if not self._uvm_scheduler_quiescent_for_resize():
                return None
        else:
            batch_queue = getattr(self, "batch_queue", None)
            if batch_queue is not None and batch_queue:
                return None
        components = self._uvm_dynamic_pool_components()
        if components is None:
            return {"ok": False, "reason": "dynamic_pool_unavailable"}
        policy, block_pool = components
        old_target = int(policy._dynamic_pool_applied_target_blocks)
        if int(target) >= old_target:
            self._cancel_uvm_pending_kv_shrink("stale_shrink_target")
            return {"ok": True, "reason": "stale_shrink_target"}
        commit_snapshot = self._uvm_kv_autoscale_queue_snapshot(
            block_pool=block_pool,
            current_target=old_target,
        )
        commit_max_active = commit_snapshot.get("max_active_block_id")
        commit_tail_free = (
            commit_snapshot.get("invariant_ok") is True
            and commit_snapshot.get("active_block_ids_known") is True
            and commit_max_active is not None
            and int(commit_max_active) <= int(target)
        )
        if not commit_tail_free:
            self._uvm_kv_autoscale_low_streak = 0
            self._cancel_uvm_pending_kv_shrink("candidate_tail_no_longer_free")
            return {
                "ok": False,
                "reason": "candidate_tail_no_longer_free",
                "old_target_blocks": old_target,
                "new_target_blocks": int(target),
                "pending": False,
                "queue_snapshot": commit_snapshot,
            }
        # Compatibility mode keeps the historical global-idle Plan Class gate.
        # Online mode has already proved execution quiescence above, so it may
        # shrink KV and restore the corresponding class between token steps.
        if (
            self._uvm_kv_autoscale_plan_class_for_target(int(target)) is not None
            and self.scheduler.has_unfinished_requests()
            and not online_boundary
        ):
            return None

        result = self._resize_uvm_kv_target(
            int(target),
            reason=self._uvm_pending_kv_shrink_reason
            or "autoscale_sustained_low_pressure",
            release_expert_wave=False,
            allow_active_prefix_shrink=True,
        )
        if not bool(result.get("ok")):
            return {
                **result,
                "old_target_blocks": old_target,
                "new_target_blocks": int(target),
                "pending": True,
            }

        self._note_uvm_kv_autoscale_resize(int(target))
        restore_enabled = self._uvm_env_flag(
            "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE",
            default=False,
        ) and self._uvm_env_flag(
            "VLLM_UVM_KV_AUTO_SCALE_EXPERT_RESTORE_ENABLE",
            default=True,
        )
        online_restore_ready = (
            self._uvm_scheduler_boundary_resize_enabled()
            and self._uvm_scheduler_quiescent_for_resize()
        )
        if (
            self.scheduler.has_unfinished_requests()
            and restore_enabled
            and not online_restore_ready
        ):
            expert_restore = {
                "ok": True,
                "reason": "expert_restore_deferred_until_idle",
                "pending": True,
            }
            self._uvm_kv_autoscale_expert_restore_pending = True
        else:
            expert_restore = self._restore_uvm_incremental_expert_wave()
            self._uvm_kv_autoscale_expert_restore_pending = not bool(
                expert_restore.get("ok")
            )
        if not bool(expert_restore.get("ok")):
            # KV shrink remains correct and committed. Keeping the experts
            # remote is the conservative degraded state; a later maintenance
            # round can retry promotion without regrowing KV.
            logger.warning(
                "KV shrink committed but expert restore remains pending: %r",
                expert_restore,
            )
        plan_class = self._apply_pending_uvm_expert_plan_class_if_ready()
        return {
            **result,
            "old_target_blocks": old_target,
            "new_target_blocks": int(target),
            "pending": False,
            "expert_restore": expert_restore,
            "plan_class": plan_class,
        }

    def get_uvm_kv_autoscale_snapshot(self) -> dict[str, Any]:
        """Return controller and queue state for tests and E2E qualification."""
        self._ensure_uvm_kv_autoscale_state()
        components = self._uvm_dynamic_pool_components()
        if components is None:
            return {"ok": False, "reason": "dynamic_pool_unavailable"}
        policy, block_pool = components
        current_target = int(policy._dynamic_pool_applied_target_blocks)
        return {
            "ok": True,
            "schema": "vllm_uvm_kv_autoscale_snapshot_v1",
            "enabled": self._uvm_env_flag(
                "VLLM_UVM_KV_AUTO_SCALE_ENABLE",
                default=False,
            ),
            "scheduler_boundary_resize_enabled": (
                self._uvm_scheduler_boundary_resize_enabled()
            ),
            "scheduler_quiescent_for_resize": (
                self._uvm_scheduler_quiescent_for_resize()
            ),
            "step": int(self._uvm_kv_autoscale_step),
            "low_streak": int(self._uvm_kv_autoscale_low_streak),
            "last_resize_step": int(self._uvm_kv_autoscale_last_resize_step),
            "predicted_blocks": int(self._uvm_kv_autoscale_predicted_blocks),
            "prediction_error_blocks": float(
                self._uvm_kv_autoscale_prediction_error_blocks
            ),
            "expert_restore_pending": bool(
                self._uvm_kv_autoscale_expert_restore_pending
            ),
            "pending_plan_class": self._uvm_kv_autoscale_pending_plan_class,
            "pending_grow_target_blocks": getattr(
                self,
                "_uvm_pending_kv_grow_target_blocks",
                None,
            ),
            "pending_shrink_target_blocks": (
                self._uvm_pending_kv_shrink_target_blocks
            ),
            "current_target_blocks": current_target,
            "config": self._uvm_kv_autoscale_config(policy),
            "last_decision": dict(self._uvm_kv_autoscale_last_decision),
            "queue_snapshot": self._uvm_kv_autoscale_queue_snapshot(
                block_pool=block_pool,
                current_target=current_target,
            ),
        }

    def apply_uvm_kv_dynamic_pool_target(
        self,
        target_blocks: int,
        reason: str = "manual_utility",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kv_cache_manager = getattr(self.scheduler, "kv_cache_manager", None)
        policy = getattr(kv_cache_manager, "uvm_kv_runtime_policy", None)
        block_pool = getattr(kv_cache_manager, "block_pool", None)
        if policy is None or block_pool is None:
            return {
                "ok": False,
                "reason": "policy_or_block_pool_unavailable",
            }

        result = self._resize_uvm_kv_target(
            int(target_blocks),
            reason=reason,
            release_expert_wave=bool(
                (metadata or {}).get("release_expert_wave", False)
            ),
        )
        return {
            "ok": bool(result.get("ok")),
            "reason": str(result.get("reason") or "apply_failed"),
        }

    def apply_uvm_kv_next_request_pressure(
        self,
        prompt_token_counts: list[int] | tuple[int, ...],
        max_output_tokens_per_request: int,
        reason: str = "next_request_pressure_policy",
    ) -> dict[str, Any]:
        """Choose and apply a physical KV tier for the next request batch.

        The caller supplies request metadata, not a target block count.  This
        method owns the pressure calculation and low/high target selection so
        the request-boundary controller cannot silently substitute a frozen
        migration plan for an automatic policy decision.
        """
        components = self._uvm_dynamic_pool_components()
        if components is None:
            return {
                "ok": False,
                "reason": "dynamic_pool_unavailable",
            }
        policy, block_pool = components
        prompts = tuple(int(value) for value in prompt_token_counts)
        max_output_tokens = int(max_output_tokens_per_request)
        if not prompts or any(value < 0 for value in prompts):
            return {"ok": False, "reason": "invalid_prompt_token_counts"}
        if max_output_tokens < 0:
            return {"ok": False, "reason": "invalid_max_output_tokens"}

        block_size_tokens = int(getattr(block_pool, "hash_block_size", 0) or 0)
        if block_size_tokens <= 0:
            return {"ok": False, "reason": "invalid_kv_block_size"}
        predicted_kv_tokens = sum(
            prompt_tokens + max_output_tokens for prompt_tokens in prompts
        )
        predicted_kv_blocks = sum(
            (prompt_tokens + max_output_tokens + block_size_tokens - 1)
            // block_size_tokens
            for prompt_tokens in prompts
        )

        total_blocks = max(int(policy.total_blocks), 0)
        current_target = min(
            max(int(policy._dynamic_pool_applied_target_blocks), 0),
            total_blocks,
        )
        scheduler_unfinished = bool(self.scheduler.has_unfinished_requests())
        batch_queue = getattr(self, "batch_queue", None)
        queued_batches = len(batch_queue) if batch_queue is not None else 0
        autoscale_enabled = self._uvm_env_flag(
            "VLLM_UVM_KV_AUTO_SCALE_ENABLE",
            default=False,
        )
        # The legacy low/high selector performs a synchronous resize and still
        # requires an idle request boundary. The unified autoscaler may accept
        # a late-arrival prediction while busy; growth is queued and committed
        # at the next audited boundary before that cohort is admitted.
        if (scheduler_unfinished or queued_batches) and not autoscale_enabled:
            return {
                "ok": False,
                "reason": "request_boundary_not_idle",
            }
        low_target = min(
            max(
                self._uvm_env_int(
                    "VLLM_UVM_KV_NEXT_REQUEST_LOW_TARGET_BLOCKS",
                    current_target,
                ),
                0,
            ),
            total_blocks,
        )
        high_target = min(
            max(
                self._uvm_env_int(
                    "VLLM_UVM_KV_NEXT_REQUEST_HIGH_TARGET_BLOCKS",
                    total_blocks,
                ),
                0,
            ),
            total_blocks,
        )
        threshold_blocks = max(
            self._uvm_env_int(
                "VLLM_UVM_KV_NEXT_REQUEST_PRESSURE_THRESHOLD_BLOCKS",
                max(low_target + 1, 1),
            ),
            1,
        )
        pressure_class = (
            "high" if predicted_kv_blocks >= threshold_blocks else "low"
        )
        target_blocks = high_target if pressure_class == "high" else low_target
        pressure = {
            "schema": "vllm_uvm_next_request_physical_kv_pressure_v1",
            "decision_source": "engine_next_request_pressure_policy",
            "previous_request_finished": not scheduler_unfinished,
            "scheduler_unfinished": scheduler_unfinished,
            "queued_batch_count": queued_batches,
            "next_batch_size": len(prompts),
            "next_prompt_tokens": sum(prompts),
            "next_max_output_tokens_per_request": max_output_tokens,
            "predicted_kv_tokens": predicted_kv_tokens,
            "predicted_kv_blocks": predicted_kv_blocks,
            "kv_block_size_tokens": block_size_tokens,
            "pressure_threshold_blocks": threshold_blocks,
            "pressure_class": pressure_class,
            "low_target_blocks": low_target,
            "high_target_blocks": high_target,
            "old_target_blocks": current_target,
            "new_target_blocks": target_blocks,
            "maximum_reserved_blocks": total_blocks,
            "physical_target_changed": current_target != target_blocks,
        }
        if autoscale_enabled:
            self._ensure_uvm_kv_autoscale_state()
            config = self._uvm_kv_autoscale_config(policy)
            if not bool(config["bounds_valid"]):
                self._uvm_kv_autoscale_predicted_blocks = 0
                pressure.update(
                    {
                        "decision_source": "unified_kv_autoscale_controller",
                        "autoscale_config": config,
                        "legacy_low_high_target_ignored": True,
                    }
                )
                return {
                    "ok": False,
                    "reason": str(config["bounds_error"]),
                    "pressure": pressure,
                }
            self._uvm_kv_autoscale_predicted_blocks = predicted_kv_blocks
            error_blocks = math.ceil(
                self._uvm_kv_autoscale_prediction_error_blocks
            )
            required_blocks = (
                predicted_kv_blocks
                + int(config["backup_blocks"])
                + int(config["safety_blocks"])
                + error_blocks
            )
            high_threshold = max(
                math.ceil(
                    current_target * float(config["high_watermark"])
                ),
                1,
            )
            if required_blocks >= high_threshold:
                result = self._request_uvm_kv_autoscale_growth(
                    required_blocks=required_blocks,
                    reason="next_request_autoscale_prediction",
                    urgent=required_blocks > current_target,
                )
            else:
                result = {
                    "ok": True,
                    "reason": "prediction_recorded_without_resize",
                    "old_target_blocks": current_target,
                    "new_target_blocks": current_target,
                }
            plan_class = self._apply_pending_uvm_expert_plan_class_if_ready()
            autoscale_target = int(
                result.get("new_target_blocks", current_target)
            )
            pressure.update(
                {
                    "decision_source": "unified_kv_autoscale_controller",
                    "required_blocks": required_blocks,
                    "prediction_error_blocks": error_blocks,
                    "high_threshold_blocks": high_threshold,
                    "autoscale_config": config,
                    "new_target_blocks": autoscale_target,
                    "physical_target_changed": (
                        autoscale_target != current_target
                    ),
                    "legacy_low_high_target_ignored": True,
                }
            )
            return {
                "ok": bool(result.get("ok")),
                "reason": str(result.get("reason") or "apply_failed"),
                "pressure": pressure,
                "autoscale": result,
                "plan_class": plan_class,
            }
        result = self._resize_uvm_kv_target(
            target_blocks,
            reason=reason,
            release_expert_wave=self._uvm_env_flag(
                "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE",
                default=False,
            ),
        )
        return {
            "ok": bool(result.get("ok")),
            "reason": str(result.get("reason") or "apply_failed"),
            "pressure": pressure,
        }

    @staticmethod
    def _plan_uvm_dynamic_expert_swaps(
        snapshot_results: list[Any],
        *,
        max_swaps_per_layer: int,
        min_count_delta: int,
    ) -> dict[str, Any]:
        aggregated: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        for worker_index, worker_result in enumerate(snapshot_results):
            if not isinstance(worker_result, dict) or not bool(worker_result.get("ok")):
                errors.append(f"worker:{worker_index}:snapshot_failed")
                continue
            layers = worker_result.get("layers")
            if not isinstance(layers, dict):
                errors.append(f"worker:{worker_index}:layers_missing")
                continue
            for layer_key, layer_value in layers.items():
                if not isinstance(layer_value, dict):
                    errors.append(f"worker:{worker_index}:{layer_key}:invalid")
                    continue
                try:
                    global_count = int(layer_value["global_num_experts"])
                    local_count = int(layer_value["local_num_experts"])
                    counts = [
                        float(value) for value in layer_value["expert_token_counts"]
                    ]
                    remote = {int(value) for value in layer_value["remote_experts"]}
                    raw_swappable = layer_value.get("swappable_experts")
                    swappable = (
                        {int(value) for value in raw_swappable}
                        if isinstance(raw_swappable, list)
                        else set(range(global_count))
                    )
                except (KeyError, TypeError, ValueError):
                    errors.append(f"worker:{worker_index}:{layer_key}:malformed")
                    continue
                if local_count != global_count:
                    errors.append(
                        f"worker:{worker_index}:{layer_key}:"
                        "distributed_expert_placement_unsupported"
                    )
                    continue
                if len(counts) != global_count:
                    errors.append(
                        f"worker:{worker_index}:{layer_key}:count_width_mismatch"
                    )
                    continue
                if (
                    not remote <= swappable
                    or any(expert < 0 or expert >= global_count for expert in swappable)
                ):
                    errors.append(
                        f"worker:{worker_index}:{layer_key}:invalid_swappable_set"
                    )
                    continue
                normalized_key = str(layer_key)
                current = aggregated.get(normalized_key)
                if current is None:
                    aggregated[normalized_key] = {
                        "global_num_experts": global_count,
                        "counts": counts,
                        "remote": remote,
                        "swappable": swappable,
                    }
                    continue
                if (
                    int(current["global_num_experts"]) != global_count
                    or set(current["remote"]) != remote
                    or set(current["swappable"]) != swappable
                ):
                    errors.append(
                        f"worker:{worker_index}:{layer_key}:replica_state_mismatch"
                    )
                    continue
                current["counts"] = [
                    left + right for left, right in zip(current["counts"], counts)
                ]

        swaps_by_layer: dict[str, list[dict[str, object]]] = {}
        layer_summaries: dict[str, dict[str, object]] = {}
        limit = max(int(max_swaps_per_layer), 0)
        minimum_delta = max(int(min_count_delta), 0)
        for layer_key, layer in sorted(aggregated.items()):
            counts = list(layer["counts"])
            remote = set(layer["remote"])
            swappable = set(layer["swappable"])
            target_count = len(remote)
            ranked = sorted(
                swappable,
                key=lambda expert: (counts[expert], expert),
            )
            target_remote = set(ranked[:target_count])
            promote = sorted(
                remote - target_remote,
                key=lambda expert: (-counts[expert], expert),
            )
            demote = sorted(
                target_remote - remote,
                key=lambda expert: (counts[expert], expert),
            )
            selected: list[dict[str, object]] = []
            for promote_expert, demote_expert in zip(promote, demote):
                count_delta = counts[promote_expert] - counts[demote_expert]
                if count_delta < minimum_delta or len(selected) >= limit:
                    continue
                selected.append(
                    {
                        "promote_expert": promote_expert,
                        "demote_expert": demote_expert,
                        "promote_count": counts[promote_expert],
                        "demote_count": counts[demote_expert],
                        "count_delta": count_delta,
                    }
                )
            if selected:
                swaps_by_layer[layer_key] = selected
            layer_summaries[layer_key] = {
                "remote_count": target_count,
                "swappable_expert_count": len(swappable),
                "current_remote_experts": sorted(remote),
                "target_remote_experts": sorted(target_remote),
                "eligible_swap_count": min(len(promote), len(demote)),
                "selected_swap_count": len(selected),
                "selected_swaps": selected,
                "observed_assignments": sum(counts),
            }
        return {
            "ok": not errors,
            "errors": errors,
            "swaps_by_layer": swaps_by_layer,
            "selected_layers": len(swaps_by_layer),
            "selected_swaps": sum(len(value) for value in swaps_by_layer.values()),
            "layers": layer_summaries,
        }

    def _smooth_uvm_dynamic_expert_heat(
        self,
        snapshot_results: list[Any],
        *,
        alpha: float,
    ) -> list[Any]:
        """Apply a persistent CPU EWMA without mutating worker snapshots."""
        weight = min(max(float(alpha), 0.0), 1.0)
        state = getattr(self, "_uvm_dynamic_expert_heat_ewma", None)
        if not isinstance(state, dict):
            state = {}
            self._uvm_dynamic_expert_heat_ewma = state
        smoothed_results: list[Any] = []
        for worker_index, worker_result in enumerate(snapshot_results):
            if not isinstance(worker_result, dict):
                smoothed_results.append(worker_result)
                continue
            copied = dict(worker_result)
            layers = worker_result.get("layers")
            if not isinstance(layers, dict):
                smoothed_results.append(copied)
                continue
            copied_layers: dict[str, Any] = {}
            for layer_key, layer_value in layers.items():
                if not isinstance(layer_value, dict):
                    copied_layers[str(layer_key)] = layer_value
                    continue
                layer_copy = dict(layer_value)
                observed = layer_value.get("expert_token_counts")
                if not isinstance(observed, list):
                    copied_layers[str(layer_key)] = layer_copy
                    continue
                key = f"{worker_index}:{layer_key}"
                previous = state.get(key)
                values = [float(value) for value in observed]
                if previous is None or len(previous) != len(values):
                    smoothed = values
                else:
                    smoothed = [
                        weight * value + (1.0 - weight) * old
                        for value, old in zip(values, previous)
                    ]
                state[key] = smoothed
                layer_copy["observed_expert_token_counts"] = list(observed)
                layer_copy["expert_token_counts"] = smoothed
                layer_copy["heat_ewma_alpha"] = weight
                copied_layers[str(layer_key)] = layer_copy
            copied["layers"] = copied_layers
            smoothed_results.append(copied)
        return smoothed_results

    @staticmethod
    def _select_uvm_expert_plan_class(
        classes_gib: tuple[float, ...],
        *,
        budget_gib: float,
    ) -> float | None:
        """Select the largest fixed class that cannot exceed the HBM budget."""
        classes = sorted({float(value) for value in classes_gib if value > 0})
        eligible = [value for value in classes if value <= float(budget_gib)]
        return eligible[-1] if eligible else None

    def _maybe_auto_rebalance_uvm_remote_experts(self) -> dict[str, Any] | None:
        """Run one periodic heat-driven swap at an audited idle boundary."""
        if not self._uvm_env_flag(
            "VLLM_UVM_DYNAMIC_EXPERT_AUTO_REBALANCE_ENABLE",
            default=False,
        ):
            return None
        if self.scheduler.has_unfinished_requests() or getattr(
            self,
            "batch_queue",
            None,
        ):
            return None
        interval_ms = max(
            self._uvm_env_int("VLLM_UVM_DYNAMIC_EXPERT_REBALANCE_INTERVAL_MS", 60000),
            1,
        )
        now_ns = time.monotonic_ns()
        last_ns = int(getattr(self, "_uvm_dynamic_expert_last_rebalance_ns", 0))
        if last_ns and now_ns - last_ns < interval_ms * 1_000_000:
            return None
        self._uvm_dynamic_expert_last_rebalance_ns = now_ns
        result = self.rebalance_uvm_remote_experts(
            max_swaps_per_layer=max(
                self._uvm_env_int("VLLM_UVM_DYNAMIC_EXPERT_MAX_SWAPS_PER_LAYER", 1),
                0,
            ),
            min_count_delta=max(
                self._uvm_env_int("VLLM_UVM_DYNAMIC_EXPERT_MIN_COUNT_DELTA", 1),
                0,
            ),
            reset_heat=True,
        )
        if bool(result.get("ok")):
            self._uvm_dynamic_expert_rebalance_generation = int(
                getattr(self, "_uvm_dynamic_expert_rebalance_generation", 0)
            ) + 1
        result["rebalance_generation"] = int(
            getattr(self, "_uvm_dynamic_expert_rebalance_generation", 0)
        )
        result["interval_ms"] = interval_ms
        return result

    def reset_uvm_dynamic_expert_heat(self) -> dict[str, Any]:
        """Reset routing windows while preserving expert placement."""
        if self.scheduler.has_unfinished_requests():
            return {"ok": False, "reason": "maintenance_requires_idle_scheduler"}
        if getattr(self, "batch_queue", None):
            return {"ok": False, "reason": "maintenance_requires_drained_batches"}
        snapshots = self.collective_rpc(
            "get_uvm_dynamic_expert_heat",
            args=(True,),
        )
        return {
            "ok": bool(snapshots)
            and all(
                isinstance(result, dict) and bool(result.get("ok"))
                for result in snapshots
            ),
            "reason": "heat_reset",
            "worker_results": snapshots,
        }

    def capture_uvm_expert_graph_class(self, class_name: str) -> dict[str, Any]:
        """Capture a fixed class at an audited global idle boundary."""
        if self.scheduler.has_unfinished_requests() or getattr(
            self, "batch_queue", None
        ):
            return {"ok": False, "reason": "maintenance_requires_idle_scheduler"}
        try:
            result = self._uvm_single_worker_rpc(
                "capture_uvm_expert_graph_class",
                args=(str(class_name), True),
            )
        except Exception as exc:
            return {
                "ok": False,
                "reason": "class_capture_rpc_failed",
                "error": repr(exc),
            }
        return result if isinstance(result, dict) else {
            "ok": False,
            "reason": "invalid_worker_result",
        }

    def activate_uvm_expert_graph_class(self, class_name: str) -> dict[str, Any]:
        """Activate a captured class without triggering lazy recapture."""
        if self.scheduler.has_unfinished_requests() or getattr(
            self, "batch_queue", None
        ):
            return {"ok": False, "reason": "maintenance_requires_idle_scheduler"}
        return self._activate_uvm_expert_graph_class_at_quiescent_boundary(
            class_name
        )

    def _activate_uvm_expert_graph_class_at_quiescent_boundary(
        self,
        class_name: str,
        *,
        defer_dual_refresh: bool = False,
    ) -> dict[str, Any]:
        """Activate a class after the previous model future has completed.

        Unlike the public maintenance API, this internal path permits running
        requests.  Their next token step observes the new read-only expert
        residency, while no already-launched kernel can race the table switch.
        """
        if not self._uvm_scheduler_quiescent_for_resize():
            return {"ok": False, "reason": "scheduler_not_quiescent"}
        try:
            result = self._uvm_single_worker_rpc(
                "activate_uvm_expert_graph_class",
                args=(
                    str(class_name),
                    True,
                    self._uvm_env_flag(
                        "VLLM_UVM_KV_AUTO_SCALE_RESUME_PREFILL_BEFORE_PLAN_CLASS",
                        default=False,
                    ),
                    bool(defer_dual_refresh),
                ),
            )
        except Exception as exc:
            return {
                "ok": False,
                "reason": "class_activation_rpc_failed",
                "error": repr(exc),
            }
        return result if isinstance(result, dict) else {
            "ok": False,
            "reason": "invalid_worker_result",
        }

    def rebalance_uvm_remote_experts(
        self,
        max_swaps_per_layer: int = 1,
        min_count_delta: int = 1,
        reset_heat: bool = True,
    ) -> dict[str, Any]:
        """Recompute cold experts from recent routing and exchange placement."""
        if not self._uvm_env_flag(
            "VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE",
            default=False,
        ):
            return {"ok": False, "reason": "dynamic_expert_swap_disabled"}
        if self.scheduler.has_unfinished_requests():
            return {"ok": False, "reason": "maintenance_requires_idle_scheduler"}
        if getattr(self, "batch_queue", None):
            return {"ok": False, "reason": "maintenance_requires_drained_batches"}

        snapshots = self.collective_rpc(
            "get_uvm_dynamic_expert_heat",
            args=(bool(reset_heat),),
        )
        try:
            alpha = float(
                os.environ.get("VLLM_UVM_DYNAMIC_EXPERT_HEAT_EWMA_ALPHA", "0.25")
                or 0.25
            )
        except ValueError:
            alpha = 0.25
        smoothed_snapshots = self._smooth_uvm_dynamic_expert_heat(
            snapshots,
            alpha=alpha,
        )
        plan = self._plan_uvm_dynamic_expert_swaps(
            smoothed_snapshots,
            max_swaps_per_layer=max_swaps_per_layer,
            min_count_delta=min_count_delta,
        )
        if not bool(plan.get("ok")):
            return {
                "ok": False,
                "reason": "heat_snapshot_invalid",
                "snapshot_results": snapshots,
                "smoothed_snapshot_results": smoothed_snapshots,
                "plan": plan,
            }
        swaps_by_layer = plan.get("swaps_by_layer") or {}
        if not swaps_by_layer:
            return {
                "ok": True,
                "reason": "placement_already_matches_heat",
                "snapshot_results": snapshots,
                "smoothed_snapshot_results": smoothed_snapshots,
                "plan": plan,
                "worker_results": [],
            }
        worker_results = self.collective_rpc(
            "run_uvm_dynamic_expert_swaps",
            args=(swaps_by_layer,),
        )
        execution_ok = bool(worker_results) and all(
            isinstance(result, dict) and bool(result.get("ok"))
            for result in worker_results
        )
        return {
            "ok": execution_ok,
            "reason": (
                "expert_swaps_committed" if execution_ok else "expert_swap_failed"
            ),
            "snapshot_results": snapshots,
            "smoothed_snapshot_results": smoothed_snapshots,
            "plan": plan,
            "worker_results": worker_results,
        }

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_executor.supported_tasks

    def add_request(self, request: Request, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )

        if pooling_params := request.pooling_params:
            supported_pooling_tasks = [
                task for task in self.get_supported_tasks() if task in POOLING_TASKS
            ]

            if pooling_params.task not in supported_pooling_tasks:
                raise ValueError(
                    f"Unsupported task: {pooling_params.task!r} "
                    f"Supported tasks: {supported_pooling_tasks}"
                )

        if request.kv_transfer_params is not None and (
            not self.scheduler.get_kv_connector()
        ):
            logger.warning(
                "Got kv_transfer_params, but no KVConnector found. "
                "Disabling KVTransfer for this request."
            )

        self.scheduler.add_request(request)

    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        self.scheduler.finish_requests(request_ids, RequestStatus.FINISHED_ABORTED)

    @contextmanager
    def log_error_detail(self, scheduler_output: SchedulerOutput):
        """Execute the model and log detailed info on failure."""
        try:
            yield
        except Exception as err:
            # We do not want to catch BaseException here since we're only
            # interested in dumping info when the exception is due to an
            # error from execute_model itself.

            # NOTE: This method is exception-free
            dump_engine_exception(
                self.vllm_config, scheduler_output, self.scheduler.make_stats()
            )
            raise err

    def _log_err_callback(self, scheduler_output: SchedulerOutput):
        """Log error details of a future that's not expected to return a result."""

        def callback(f, sched_output=scheduler_output):
            with self.log_error_detail(sched_output):
                result = f.result()
                assert result is None

        return callback

    def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
        """Schedule, execute, and make output.

        Returns tuple of outputs and a flag indicating whether the model
        was executed.
        """

        prefill_step_timing_audit = self._uvm_env_flag(
            "VLLM_PREFILL_ENGINE_STEP_TIMING_AUDIT", default=False
        )
        step_started_ns = time.perf_counter_ns()

        # A late arrival may have queued a grow while the previous model step
        # was in flight. Commit it at this quiescent boundary before scheduling
        # more work; running requests need not have completed.
        self._apply_pending_uvm_kv_growth_if_ready()
        after_initial_growth_ns = time.perf_counter_ns()

        # Check for any requests remaining in the scheduler - unfinished,
        # or finished and not yet removed from the batch.
        if not self.scheduler.has_requests():
            return {}, False
        scheduler_output = self.scheduler.schedule()
        after_schedule_ns = time.perf_counter_ns()
        self._maybe_handle_uvm_kv_preemption_pressure(scheduler_output)
        after_pressure_ns = time.perf_counter_ns()
        self._ensure_uvm_kv_autoscale_state()
        after_autoscale_state_ns = time.perf_counter_ns()
        self._uvm_model_execution_inflight = True
        try:
            future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            after_execute_model_ns = time.perf_counter_ns()
            grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
            after_grammar_ns = time.perf_counter_ns()
            with self.log_error_detail(scheduler_output):
                model_output = future.result()
                after_future_result_ns = time.perf_counter_ns()
                if model_output is None:
                    model_output = self.model_executor.sample_tokens(grammar_output)
            after_model_result_ns = time.perf_counter_ns()
        finally:
            self._uvm_model_execution_inflight = False

        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        after_scheduler_update_ns = time.perf_counter_ns()
        self._apply_pending_uvm_kv_growth_if_ready()
        after_final_growth_ns = time.perf_counter_ns()
        self._apply_pending_uvm_kv_shrink_if_ready()
        after_shrink_ns = time.perf_counter_ns()
        self._apply_pending_uvm_expert_plan_class_if_ready()
        after_plan_class_ns = time.perf_counter_ns()
        self._maybe_auto_rebalance_uvm_remote_experts()
        after_rebalance_ns = time.perf_counter_ns()

        if prefill_step_timing_audit and scheduler_output.scheduled_new_reqs:
            logger.info(
                "PREFILL_ENGINE_STEP_TIMING initial_growth_us=%.3f "
                "schedule_us=%.3f pressure_us=%.3f autoscale_state_us=%.3f "
                "execute_model_us=%.3f grammar_us=%.3f future_result_us=%.3f "
                "sample_tokens_us=%.3f "
                "scheduler_update_us=%.3f final_growth_us=%.3f shrink_us=%.3f "
                "plan_class_us=%.3f rebalance_us=%.3f total_us=%.3f",
                (after_initial_growth_ns - step_started_ns) / 1e3,
                (after_schedule_ns - after_initial_growth_ns) / 1e3,
                (after_pressure_ns - after_schedule_ns) / 1e3,
                (after_autoscale_state_ns - after_pressure_ns) / 1e3,
                (after_execute_model_ns - after_autoscale_state_ns) / 1e3,
                (after_grammar_ns - after_execute_model_ns) / 1e3,
                (after_future_result_ns - after_grammar_ns) / 1e3,
                (after_model_result_ns - after_future_result_ns) / 1e3,
                (after_scheduler_update_ns - after_model_result_ns) / 1e3,
                (after_final_growth_ns - after_scheduler_update_ns) / 1e3,
                (after_shrink_ns - after_final_growth_ns) / 1e3,
                (after_plan_class_ns - after_shrink_ns) / 1e3,
                (after_rebalance_ns - after_plan_class_ns) / 1e3,
                (after_rebalance_ns - step_started_ns) / 1e3,
            )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    def post_step(self, model_executed: bool) -> None:
        # When using async scheduling we can't get draft token ids in advance,
        # so we update draft token ids in the worker process and don't
        # need to update draft token ids here.
        if not self.async_scheduling and self.use_spec_decode and model_executed:
            # Take the draft token ids.
            draft_token_ids = self.model_executor.take_draft_token_ids()
            if draft_token_ids is not None:
                self.scheduler.update_draft_token_ids(draft_token_ids)

    def step_with_batch_queue(
        self,
    ) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
        """Schedule and execute batches with the batch queue.
        Note that if nothing to output in this step, None is returned.

        The execution flow is as follows:
        1. Try to schedule a new batch if the batch queue is not full.
        If a new batch is scheduled, directly return an empty engine core
        output. In other words, fulfilling the batch queue has a higher priority
        than getting model outputs.
        2. If there is no new scheduled batch, meaning that the batch queue
        is full or no other requests can be scheduled, we block until the first
        batch in the job queue is finished.
        3. Update the scheduler from the output.
        """
        batch_queue = self.batch_queue
        assert batch_queue is not None
        self._apply_pending_uvm_kv_growth_if_ready()

        # Try to schedule a new batch if the batch queue is not full, but
        # the scheduler may return an empty batch if all requests are scheduled.
        # Note that this is not blocking.
        assert len(batch_queue) < self.batch_queue_size

        model_executed = False
        deferred_scheduler_output = None
        if (
            self.scheduler.has_requests()
            and self._uvm_pending_kv_grow_target_blocks is None
        ):
            scheduler_output = self.scheduler.schedule()
            self._maybe_handle_uvm_kv_preemption_pressure(scheduler_output)
            exec_future = self.model_executor.execute_model(
                scheduler_output, non_block=True
            )
            if not self.is_ec_producer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0

            if self.is_pooling_model or not model_executed:
                # No sampling required (no requests scheduled).
                future = cast(Future[ModelRunnerOutput], exec_future)
            else:
                exec_future.add_done_callback(self._log_err_callback(scheduler_output))

                if not scheduler_output.pending_structured_output_tokens:
                    # We aren't waiting for any tokens, get any grammar output
                    # and sample immediately.
                    grammar_output = self.scheduler.get_grammar_bitmask(
                        scheduler_output
                    )
                    future = self.model_executor.sample_tokens(
                        grammar_output, non_block=True
                    )
                else:
                    # We need to defer sampling until we have processed the model output
                    # from the prior step.
                    deferred_scheduler_output = scheduler_output

            if not deferred_scheduler_output:
                # Add this step's future to the queue.
                batch_queue.appendleft((future, scheduler_output))
                if (
                    model_executed
                    and len(batch_queue) < self.batch_queue_size
                    and not batch_queue[-1][0].done()
                ):
                    # Don't block on next worker response unless the queue is full
                    # or there are no more requests to schedule.
                    return None, True

        elif not batch_queue:
            # Queue is empty. We should not reach here since this method should
            # only be called when the scheduler contains requests or the queue
            # is non-empty.
            return None, False

        # Block until the next result is available.
        future, scheduler_output = batch_queue.pop()
        with self.log_error_detail(scheduler_output):
            model_output = future.result()

        engine_core_outputs = self.scheduler.update_from_output(
            scheduler_output, model_output
        )
        self._apply_pending_uvm_kv_shrink_if_ready()
        self._apply_pending_uvm_expert_plan_class_if_ready()
        self._maybe_auto_rebalance_uvm_remote_experts()

        # NOTE(nick): We can either handle the deferred tasks here or save
        # in a field and do it immediately once step_with_batch_queue is
        # re-called. The latter slightly favors TTFT over TPOT/throughput.
        if deferred_scheduler_output:
            # We now have the tokens needed to compute the bitmask for the
            # deferred request. Get the bitmask and call sample tokens.
            grammar_output = self.scheduler.get_grammar_bitmask(
                deferred_scheduler_output
            )
            future = self.model_executor.sample_tokens(grammar_output, non_block=True)
            batch_queue.appendleft((future, deferred_scheduler_output))

        self._apply_pending_uvm_kv_growth_if_ready()
        self._apply_pending_uvm_kv_shrink_if_ready()
        self._apply_pending_uvm_expert_plan_class_if_ready()

        return engine_core_outputs, model_executed

    def shutdown(self):
        self.structured_output_manager.clear_backend()
        if self.model_executor:
            self.model_executor.shutdown()
        if self.scheduler:
            self.scheduler.shutdown()

    def profile(self, is_start: bool = True):
        self.model_executor.profile(is_start)

    def reset_mm_cache(self):
        # NOTE: Since this is mainly for debugging, we don't attempt to
        # re-sync the internal caches (P0 sender, P1 receiver)
        if self.scheduler.has_unfinished_requests():
            logger.warning(
                "Resetting the multi-modal cache when requests are "
                "in progress may lead to desynced internal caches."
            )

        # The cache either exists in EngineCore or WorkerWrapperBase
        if self.mm_receiver_cache is not None:
            self.mm_receiver_cache.clear_cache()

        self.model_executor.reset_mm_cache()

    def reset_prefix_cache(self):
        self.scheduler.reset_prefix_cache()

    def sleep(self, level: int = 1):
        if self._uvm_env_flag(
            "VLLM_UVM_KV_ADAPTIVE_SHRINK_ON_SLEEP",
            default=False,
        ):
            components = self._uvm_dynamic_pool_components()
            if components is not None:
                policy, _ = components
                if self.scheduler.has_unfinished_requests():
                    raise RuntimeError(
                        "Cannot shrink the UVM KV pool for sleep while requests "
                        "are still active"
                    )
                default_sleep_target = (
                    int(policy.dynamic_pool_initial_blocks)
                    if int(policy.dynamic_pool_initial_blocks) > 0
                    else int(policy._dynamic_pool_applied_target_blocks)
                )
                sleep_target = max(
                    self._uvm_env_int(
                        "VLLM_UVM_KV_ADAPTIVE_SLEEP_TARGET_BLOCKS",
                        default_sleep_target,
                    ),
                    0,
                )
                if sleep_target < int(policy._dynamic_pool_applied_target_blocks):
                    self._resize_uvm_kv_target(
                        sleep_target,
                        reason="engine_sleep_shrink",
                        release_expert_wave=False,
                    )
        self.model_executor.sleep(level)

    def wake_up(self, tags: list[str] | None = None):
        self.model_executor.wake_up(tags)

    def is_sleeping(self) -> bool:
        return self.model_executor.is_sleeping

    def execute_dummy_batch(self):
        self.model_executor.execute_dummy_batch()

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self.model_executor.add_lora(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self.model_executor.remove_lora(lora_id)

    def list_loras(self) -> set[int]:
        return self.model_executor.list_loras()

    def pin_lora(self, lora_id: int) -> bool:
        return self.model_executor.pin_lora(lora_id)

    def save_sharded_state(
        self,
        path: str,
        pattern: str | None = None,
        max_size: int | None = None,
    ) -> None:
        self.model_executor.save_sharded_state(
            path=path, pattern=pattern, max_size=max_size
        )

    def collective_rpc(
        self,
        method: str | Callable[..., _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]:
        return self.model_executor.collective_rpc(method, timeout, args, kwargs)

    def preprocess_add_request(self, request: EngineCoreRequest) -> tuple[Request, int]:
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # Note on thread safety: no race condition.
        # `mm_receiver_cache` is reset at the end of LLMEngine init,
        # and will only be accessed in the input processing thread afterwards.
        if self.mm_receiver_cache is not None and request.mm_features:
            request.mm_features = self.mm_receiver_cache.get_and_update_features(
                request.mm_features
            )

        req = Request.from_engine_core_request(request, self.request_block_hasher)
        if req.use_structured_output:
            # Note on thread safety: no race condition.
            # `grammar_init` is only invoked in input processing thread. For
            # `structured_output_manager`, each request is independent and
            # grammar compilation is async. Scheduler always checks grammar
            # compilation status before scheduling request.
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave


class EngineCoreProc(EngineCore):
    """ZMQ-wrapper for running EngineCore in background process."""

    ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
        engine_index: int = 0,
    ):
        self.input_queue = queue.Queue[tuple[EngineCoreRequestType, Any]]()
        self.output_queue = queue.Queue[tuple[int, EngineCoreOutputs] | bytes]()
        executor_fail_callback = lambda: self.input_queue.put_nowait(
            (EngineCoreRequestType.EXECUTOR_FAILED, b"")
        )

        self.engine_index = engine_index
        identity = self.engine_index.to_bytes(length=2, byteorder="little")
        self.engines_running = False

        with self._perform_handshakes(
            handshake_address,
            identity,
            local_client,
            vllm_config,
            client_handshake_address,
        ) as addresses:
            self.client_count = len(addresses.outputs)

            # Set up data parallel environment.
            self.has_coordinator = addresses.coordinator_output is not None
            self.frontend_stats_publish_address = (
                addresses.frontend_stats_publish_address
            )
            logger.debug(
                "Has DP Coordinator: %s, stats publish address: %s",
                self.has_coordinator,
                self.frontend_stats_publish_address,
            )
            # Only publish request queue stats to coordinator for "internal"
            # and "hybrid" LB modes .
            self.publish_dp_lb_stats = (
                self.has_coordinator
                and not vllm_config.parallel_config.data_parallel_external_lb
            )

            self._init_data_parallel(vllm_config)

            super().__init__(
                vllm_config, executor_class, log_stats, executor_fail_callback
            )

            # Background Threads and Queues for IO. These enable us to
            # overlap ZMQ socket IO with GPU since they release the GIL,
            # and to overlap some serialization/deserialization with the
            # model forward pass.
            # Threads handle Socket <-> Queues and core_busy_loop uses Queue.
            ready_event = threading.Event()
            input_thread = threading.Thread(
                target=self.process_input_sockets,
                args=(
                    addresses.inputs,
                    addresses.coordinator_input,
                    identity,
                    ready_event,
                ),
                daemon=True,
            )
            input_thread.start()

            self.output_thread = threading.Thread(
                target=self.process_output_sockets,
                args=(
                    addresses.outputs,
                    addresses.coordinator_output,
                    self.engine_index,
                ),
                daemon=True,
            )
            self.output_thread.start()

            # Don't complete handshake until DP coordinator ready message is
            # received.
            while not ready_event.wait(timeout=10):
                if not input_thread.is_alive():
                    raise RuntimeError("Input socket thread died during startup")
                assert addresses.coordinator_input is not None
                logger.info("Waiting for READY message from DP Coordinator...")

        # Enable environment variable cache (e.g. assume no more
        # environment variable overrides after this point)
        enable_envs_cache()

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        """
        Perform startup handshakes.

        For DP=1 or offline mode, this is with the colocated front-end process.

        For DP>1 with internal load-balancing this is with the shared front-end
        process which may reside on a different node.

        For DP>1 with external or hybrid load-balancing, two handshakes are
        performed:
            - With the rank 0 front-end process which retrieves the
              DP Coordinator ZMQ addresses and DP process group address.
            - With the colocated front-end process which retrieves the
              client input/output socket addresses.
        with the exception of the rank 0 and colocated engines themselves which
        don't require the second handshake.

        Here, "front-end" process can mean the process containing the engine
        core client (which is the API server process in the case the API
        server is not scaled out), OR the launcher process running the
        run_multi_api_server() function in serve.py.
        """
        input_ctx = zmq.Context()
        is_local = local_client and client_handshake_address is None
        headless = not local_client
        handshake = self._perform_handshake(
            input_ctx,
            handshake_address,
            identity,
            is_local,
            headless,
            vllm_config,
            vllm_config.parallel_config,
        )
        if client_handshake_address is None:
            with handshake as addresses:
                yield addresses
        else:
            assert local_client
            local_handshake = self._perform_handshake(
                input_ctx, client_handshake_address, identity, True, False, vllm_config
            )
            with handshake as addresses, local_handshake as client_addresses:
                addresses.inputs = client_addresses.inputs
                addresses.outputs = client_addresses.outputs
                yield addresses

        # Update config which may have changed from the handshake
        vllm_config.__post_init__()

    @contextmanager
    def _perform_handshake(
        self,
        ctx: zmq.Context,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        headless: bool,
        vllm_config: VllmConfig,
        parallel_config_to_update: ParallelConfig | None = None,
    ) -> Generator[EngineZmqAddresses, None, None]:
        with make_zmq_socket(
            ctx,
            handshake_address,
            zmq.DEALER,
            identity=identity,
            linger=5000,
            bind=False,
        ) as handshake_socket:
            # Register engine with front-end.
            addresses = self.startup_handshake(
                handshake_socket, local_client, headless, parallel_config_to_update
            )
            yield addresses

            # Send ready message.
            num_gpu_blocks = vllm_config.cache_config.num_gpu_blocks
            # We pass back the coordinator stats update address here for the
            # external LB case for our colocated front-end to use (coordinator
            # only runs with rank 0).
            dp_stats_address = self.frontend_stats_publish_address

            # Include config hash for DP configuration validation
            ready_msg = {
                "status": "READY",
                "local": local_client,
                "headless": headless,
                "num_gpu_blocks": num_gpu_blocks,
                "dp_stats_address": dp_stats_address,
            }
            if vllm_config.parallel_config.data_parallel_size > 1:
                ready_msg["parallel_config_hash"] = (
                    vllm_config.parallel_config.compute_hash()
                )

            handshake_socket.send(msgspec.msgpack.encode(ready_msg))

    @staticmethod
    def startup_handshake(
        handshake_socket: zmq.Socket,
        local_client: bool,
        headless: bool,
        parallel_config: ParallelConfig | None = None,
    ) -> EngineZmqAddresses:
        # Send registration message.
        handshake_socket.send(
            msgspec.msgpack.encode(
                {
                    "status": "HELLO",
                    "local": local_client,
                    "headless": headless,
                }
            )
        )

        # Receive initialization message.
        logger.debug("Waiting for init message from front-end.")
        if not handshake_socket.poll(timeout=HANDSHAKE_TIMEOUT_MINS * 60_000):
            raise RuntimeError(
                "Did not receive response from front-end "
                f"process within {HANDSHAKE_TIMEOUT_MINS} "
                f"minutes"
            )
        init_bytes = handshake_socket.recv()
        init_message: EngineHandshakeMetadata = msgspec.msgpack.decode(
            init_bytes, type=EngineHandshakeMetadata
        )
        logger.debug("Received init message: %s", init_message)

        if parallel_config is not None:
            for key, value in init_message.parallel_config.items():
                setattr(parallel_config, key, value)

        return init_message.addresses

    @staticmethod
    def run_engine_core(*args, dp_rank: int = 0, local_dp_rank: int = 0, **kwargs):
        """Launch EngineCore busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        # Ensure we can serialize transformer config after spawning
        maybe_register_config_serialize_by_value()

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        engine_core: EngineCoreProc | None = None
        try:
            parallel_config: ParallelConfig = kwargs["vllm_config"].parallel_config
            if parallel_config.data_parallel_size > 1 or dp_rank > 0:
                set_process_title("EngineCore", f"DP{dp_rank}")
                decorate_logs()
                # Set data parallel rank for this engine process.
                parallel_config.data_parallel_rank = dp_rank
                parallel_config.data_parallel_rank_local = local_dp_rank
                engine_core = DPEngineCoreProc(*args, **kwargs)
            else:
                set_process_title("EngineCore")
                decorate_logs()
                engine_core = EngineCoreProc(*args, **kwargs)

            engine_core.run_busy_loop()

        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception as e:
            if engine_core is None:
                logger.exception("EngineCore failed to start.")
            else:
                logger.exception("EngineCore encountered a fatal error.")
                engine_core._send_engine_dead()
            raise e
        finally:
            if engine_core is not None:
                engine_core.shutdown()

    def _init_data_parallel(self, vllm_config: VllmConfig):
        pass

    def run_busy_loop(self):
        """Core busy loop of the EngineCore."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()
            # 2) Step the engine core and return the outputs.
            self._process_engine_step()

    def _process_input_queue(self):
        """Exits when an engine step needs to be performed."""

        waited = False
        while (
            not self.engines_running
            and not self.scheduler.has_requests()
            and not self.batch_queue
        ):
            if logger.isEnabledFor(DEBUG) and self.input_queue.empty():
                logger.debug("EngineCore waiting for work.")
                waited = True
            req = self.input_queue.get()
            self._handle_client_request(*req)

        if waited:
            logger.debug("EngineCore loop active.")

        # Handle any more client requests.
        while not self.input_queue.empty():
            req = self.input_queue.get_nowait()
            self._handle_client_request(*req)

    def _process_engine_step(self) -> bool:
        """Called only when there are unfinished local requests."""

        # Step the engine core.
        outputs, model_executed = self.step_fn()
        # Put EngineCoreOutputs into the output queue.
        for output in outputs.items() if outputs else ():
            self.output_queue.put_nowait(output)
        # Post-step hook.
        self.post_step(model_executed)

        return model_executed

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        """Dispatch request from client."""

        if request_type == EngineCoreRequestType.ADD:
            req, request_wave = request
            self.add_request(req, request_wave)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        elif request_type == EngineCoreRequestType.UTILITY:
            client_idx, call_id, method_name, args = request
            output = UtilityOutput(call_id)
            try:
                method = getattr(self, method_name)
                result = method(*self._convert_msgspec_args(method, args))
                output.result = UtilityResult(result)
            except BaseException as e:
                logger.exception("Invocation of %s method failed", method_name)
                output.failure_message = (
                    f"Call to {method_name} method failed: {str(e)}"
                )
            self.output_queue.put_nowait(
                (client_idx, EngineCoreOutputs(utility_output=output))
            )
        elif request_type == EngineCoreRequestType.EXECUTOR_FAILED:
            raise RuntimeError("Executor failed.")
        else:
            logger.error(
                "Unrecognized input request type encountered: %s", request_type
            )

    @staticmethod
    def _convert_msgspec_args(method, args):
        """If a provided arg type doesn't match corresponding target method
        arg type, try converting to msgspec object."""
        if not args:
            return args
        arg_types = signature(method).parameters.values()
        assert len(args) <= len(arg_types)
        return tuple(
            msgspec.convert(v, type=p.annotation)
            if isclass(p.annotation)
            and issubclass(p.annotation, msgspec.Struct)
            and not isinstance(v, p.annotation)
            else v
            for v, p in zip(args, arg_types)
        )

    def _send_engine_dead(self):
        """Send EngineDead status to the EngineCoreClient."""

        # Put ENGINE_CORE_DEAD in the queue.
        self.output_queue.put_nowait(EngineCoreProc.ENGINE_CORE_DEAD)

        # Wait until msg sent by the daemon before shutdown.
        self.output_thread.join(timeout=5.0)
        if self.output_thread.is_alive():
            logger.fatal(
                "vLLM shutdown signal from EngineCore failed "
                "to send. Please report this issue."
            )

    def process_input_sockets(
        self,
        input_addresses: list[str],
        coord_input_address: str | None,
        identity: bytes,
        ready_event: threading.Event,
    ):
        """Input socket IO thread."""

        # Msgpack serialization decoding.
        add_request_decoder = MsgpackDecoder(EngineCoreRequest)
        generic_decoder = MsgpackDecoder()

        with ExitStack() as stack, zmq.Context() as ctx:
            input_sockets = [
                stack.enter_context(
                    make_zmq_socket(
                        ctx, input_address, zmq.DEALER, identity=identity, bind=False
                    )
                )
                for input_address in input_addresses
            ]
            if coord_input_address is None:
                coord_socket = None
            else:
                coord_socket = stack.enter_context(
                    make_zmq_socket(
                        ctx,
                        coord_input_address,
                        zmq.XSUB,
                        identity=identity,
                        bind=False,
                    )
                )
                # Send subscription message to coordinator.
                coord_socket.send(b"\x01")

            # Register sockets with poller.
            poller = zmq.Poller()
            for input_socket in input_sockets:
                # Send initial message to each input socket - this is required
                # before the front-end ROUTER socket can send input messages
                # back to us.
                input_socket.send(b"")
                poller.register(input_socket, zmq.POLLIN)

            if coord_socket is not None:
                # Wait for ready message from coordinator.
                assert coord_socket.recv() == b"READY"
                poller.register(coord_socket, zmq.POLLIN)

            ready_event.set()
            del ready_event
            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    type_frame, *data_frames = input_socket.recv_multipart(copy=False)
                    request_type = EngineCoreRequestType(bytes(type_frame.buffer))

                    # Deserialize the request data.
                    if request_type == EngineCoreRequestType.ADD:
                        request = add_request_decoder.decode(data_frames)
                        request = self.preprocess_add_request(request)
                    else:
                        request = generic_decoder.decode(data_frames)

                    # Push to input queue for core busy loop.
                    self.input_queue.put_nowait((request_type, request))

    def process_output_sockets(
        self,
        output_paths: list[str],
        coord_output_path: str | None,
        engine_index: int,
    ):
        """Output socket IO thread."""

        # Msgpack serialization encoding.
        encoder = MsgpackEncoder()
        # Send buffers to reuse.
        reuse_buffers: list[bytearray] = []
        # Keep references to outputs and buffers until zmq is finished
        # with them (outputs may contain tensors/np arrays whose
        # backing buffers were extracted for zero-copy send).
        pending = deque[tuple[zmq.MessageTracker, Any, bytearray]]()

        # We must set linger to ensure the ENGINE_CORE_DEAD
        # message is sent prior to closing the socket.
        with ExitStack() as stack, zmq.Context() as ctx:
            sockets = [
                stack.enter_context(
                    make_zmq_socket(ctx, output_path, zmq.PUSH, linger=4000)
                )
                for output_path in output_paths
            ]
            coord_socket = (
                stack.enter_context(
                    make_zmq_socket(
                        ctx, coord_output_path, zmq.PUSH, bind=False, linger=4000
                    )
                )
                if coord_output_path is not None
                else None
            )
            max_reuse_bufs = len(sockets) + 1

            while True:
                output = self.output_queue.get()
                if output == EngineCoreProc.ENGINE_CORE_DEAD:
                    for socket in sockets:
                        socket.send(output)
                    break
                assert not isinstance(output, bytes)
                client_index, outputs = output
                outputs.engine_index = engine_index

                if client_index == -1:
                    # Don't reuse buffer for coordinator message
                    # which will be very small.
                    assert coord_socket is not None
                    coord_socket.send_multipart(encoder.encode(outputs))
                    continue

                # Reclaim buffers that zmq is finished with.
                while pending and pending[-1][0].done:
                    reuse_buffers.append(pending.pop()[2])

                buffer = reuse_buffers.pop() if reuse_buffers else bytearray()
                buffers = encoder.encode_into(outputs, buffer)
                tracker = sockets[client_index].send_multipart(
                    buffers, copy=False, track=True
                )
                if not tracker.done:
                    ref = outputs if len(buffers) > 1 else None
                    pending.appendleft((tracker, ref, buffer))
                elif len(reuse_buffers) < max_reuse_bufs:
                    # Limit the number of buffers to reuse.
                    reuse_buffers.append(buffer)


class DPEngineCoreProc(EngineCoreProc):
    """ZMQ-wrapper for running EngineCore in background process
    in a data parallel context."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        handshake_address: str,
        executor_class: type[Executor],
        log_stats: bool,
        client_handshake_address: str | None = None,
    ):
        # Counts forward-passes of the model so that we can synchronize
        # finished with DP peers every N steps.
        self.step_counter = 0
        self.current_wave = 0
        self.last_counts = (0, 0)

        # Initialize the engine.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        super().__init__(
            vllm_config,
            local_client,
            handshake_address,
            executor_class,
            log_stats,
            client_handshake_address,
            dp_rank,
        )

    def _init_data_parallel(self, vllm_config: VllmConfig):
        # Configure GPUs and stateless process group for data parallel.
        dp_rank = vllm_config.parallel_config.data_parallel_rank
        dp_size = vllm_config.parallel_config.data_parallel_size
        local_dp_rank = vllm_config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        if vllm_config.kv_transfer_config is not None:
            # modify the engine_id and append the local_dp_rank to it to ensure
            # that the kv_transfer_config is unique for each DP rank.
            vllm_config.kv_transfer_config.engine_id = (
                f"{vllm_config.kv_transfer_config.engine_id}_dp{local_dp_rank}"
            )
            logger.debug(
                "Setting kv_transfer_config.engine_id to %s",
                vllm_config.kv_transfer_config.engine_id,
            )

        self.dp_rank = dp_rank
        self.dp_group = vllm_config.parallel_config.stateless_init_dp_group()

    def shutdown(self):
        super().shutdown()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def add_request(self, request: Request, request_wave: int = 0):
        if self.has_coordinator and request_wave != self.current_wave:
            if request_wave > self.current_wave:
                self.current_wave = request_wave
            elif not self.engines_running:
                # Request received for an already-completed wave, notify
                # front-end that we need to start the next one.
                self.output_queue.put_nowait(
                    (-1, EngineCoreOutputs(start_wave=self.current_wave))
                )

        super().add_request(request, request_wave)

    def _handle_client_request(
        self, request_type: EngineCoreRequestType, request: Any
    ) -> None:
        if request_type == EngineCoreRequestType.START_DP_WAVE:
            new_wave, exclude_eng_index = request
            if exclude_eng_index != self.engine_index and (
                new_wave >= self.current_wave
            ):
                self.current_wave = new_wave
                if not self.engines_running:
                    logger.debug("EngineCore starting idle loop for wave %d.", new_wave)
                    self.engines_running = True
        else:
            super()._handle_client_request(request_type, request)

    def _maybe_publish_request_counts(self):
        if not self.publish_dp_lb_stats:
            return

        # Publish our request counts (if they've changed).
        counts = self.scheduler.get_request_counts()
        if counts != self.last_counts:
            self.last_counts = counts
            stats = SchedulerStats(
                *counts, step_counter=self.step_counter, current_wave=self.current_wave
            )
            self.output_queue.put_nowait((-1, EngineCoreOutputs(scheduler_stats=stats)))

    def run_busy_loop(self):
        """Core busy loop of the EngineCore for data parallel case."""

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # 1) Poll the input queue until there is work to do.
            self._process_input_queue()

            # 2) Step the engine core.
            executed = self._process_engine_step()
            self._maybe_publish_request_counts()

            local_unfinished_reqs = self.scheduler.has_unfinished_requests()
            if not executed:
                if not local_unfinished_reqs and not self.engines_running:
                    # All engines are idle.
                    continue

                # We are in a running state and so must execute a dummy pass
                # if the model didn't execute any ready requests.
                self.execute_dummy_batch()

            # 3) All-reduce operation to determine global unfinished reqs.
            self.engines_running = self._has_global_unfinished_reqs(
                local_unfinished_reqs
            )

            if not self.engines_running:
                if self.dp_rank == 0 or not self.has_coordinator:
                    # Notify client that we are pausing the loop.
                    logger.debug(
                        "Wave %d finished, pausing engine loop.", self.current_wave
                    )
                    # In the coordinator case, dp rank 0 sends updates to the
                    # coordinator. Otherwise (offline spmd case), each rank
                    # sends the update to its colocated front-end process.
                    client_index = -1 if self.has_coordinator else 0
                    self.output_queue.put_nowait(
                        (
                            client_index,
                            EngineCoreOutputs(wave_complete=self.current_wave),
                        )
                    )
                # Increment wave count and reset step counter.
                self.current_wave += 1
                self.step_counter = 0

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Optimization - only perform finish-sync all-reduce every 32 steps.
        self.step_counter += 1
        if self.step_counter % 32 != 0:
            return True

        return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)

    def reinitialize_distributed(
        self, reconfig_request: ReconfigureDistributedRequest
    ) -> None:
        stateless_destroy_torch_distributed_process_group(self.dp_group)
        self.shutdown()

        parallel_config = self.vllm_config.parallel_config
        old_dp_size = parallel_config.data_parallel_size
        parallel_config.data_parallel_size = reconfig_request.new_data_parallel_size
        if reconfig_request.new_data_parallel_rank != -1:
            parallel_config.data_parallel_rank = reconfig_request.new_data_parallel_rank
        # local rank specifies device visibility, it should not be changed
        assert (
            reconfig_request.new_data_parallel_rank_local
            == ReconfigureRankType.KEEP_CURRENT_RANK
        )
        parallel_config.data_parallel_master_ip = (
            reconfig_request.new_data_parallel_master_ip
        )
        parallel_config.data_parallel_master_port = (
            reconfig_request.new_data_parallel_master_port
        )
        if reconfig_request.new_data_parallel_rank != -2:
            self.dp_rank = parallel_config.data_parallel_rank
            self.dp_group = parallel_config.stateless_init_dp_group()
        reconfig_request.new_data_parallel_master_port = (
            parallel_config.data_parallel_master_port
        )

        self.model_executor.reinitialize_distributed(reconfig_request)
        if reconfig_request.new_data_parallel_size > old_dp_size:
            assert self.available_gpu_memory_for_kv_cache > 0
            # pass available_gpu_memory_for_kv_cache from existing
            # engine-cores to new engine-cores so they can directly
            # use it in _initialize_kv_caches() rather than profiling.
            ParallelConfig.sync_kv_cache_memory_size(
                self.dp_group, self.available_gpu_memory_for_kv_cache
            )
            # NOTE(yongji): newly joined workers require dummy_run even
            # CUDA graph is not used
            self.model_executor.collective_rpc("compile_or_warm_up_model")
        if (
            reconfig_request.new_data_parallel_rank
            == ReconfigureRankType.SHUTDOWN_CURRENT_RANK
        ):
            self.shutdown()
            logger.info("DPEngineCoreProc %s shutdown", self.dp_rank)
        else:
            logger.info(
                "Distributed environment reinitialized for DP rank %s", self.dp_rank
            )


class DPEngineCoreActor(DPEngineCoreProc):
    """
    Ray actor for running EngineCore in a data parallel context
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_client: bool,
        addresses: EngineZmqAddresses,
        executor_class: type[Executor],
        log_stats: bool,
        dp_rank: int = 0,
        local_dp_rank: int = 0,
    ):
        self.addresses = addresses
        vllm_config.parallel_config.data_parallel_rank = dp_rank
        vllm_config.parallel_config.data_parallel_rank_local = local_dp_rank

        # Set CUDA_VISIBLE_DEVICES as early as possible in actor life cycle
        # NOTE: in MP we set CUDA_VISIBLE_DEVICES at process creation time,
        # and this cannot be done in the same way for Ray because:
        # 1) Ray manages life cycle of all ray workers (including
        # DPEngineCoreActor)
        # 2) Ray sets CUDA_VISIBLE_DEVICES based on num_gpus configuration
        # To bypass 2, we need to also set
        # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES, but vLLM workers created
        # thereafter would have CUDA_VISIBLE_DEVICES set, which is sticky:
        # https://github.com/ray-project/ray/blob/e752fc319ddedd9779a0989b6d3613909bad75c9/python/ray/_private/worker.py#L456 # noqa: E501
        # This is problematic because when the vLLM worker (a Ray actor)
        # executes a task, it indexes into the sticky CUDA_VISIBLE_DEVICES
        # rather than directly using the GPU ID, potentially resulting in
        # index out of bounds error. See:
        # https://github.com/ray-project/ray/pull/40461/files#diff-31e8159767361e4bc259b6d9883d9c0d5e5db780fcea4a52ead4ee3ee4a59a78R1860 # noqa: E501
        # and get_accelerator_ids_for_accelerator_resource() in worker.py
        # of ray.
        self._set_visible_devices(vllm_config, local_dp_rank)

        super().__init__(vllm_config, local_client, "", executor_class, log_stats)

    def _set_visible_devices(self, vllm_config: VllmConfig, local_dp_rank: int):
        from vllm.platforms import current_platform

        if current_platform.is_xpu():
            pass
        else:
            device_control_env_var = current_platform.device_control_env_var
            self._set_cuda_visible_devices(
                vllm_config, local_dp_rank, device_control_env_var
            )

    def _set_cuda_visible_devices(
        self, vllm_config: VllmConfig, local_dp_rank: int, device_control_env_var: str
    ):
        world_size = vllm_config.parallel_config.world_size
        # Set CUDA_VISIBLE_DEVICES or equivalent.
        try:
            value = get_device_indices(
                device_control_env_var, local_dp_rank, world_size
            )
            os.environ[device_control_env_var] = value
        except IndexError as e:
            raise Exception(
                f"Error setting {device_control_env_var}: "
                f"local range: [{local_dp_rank * world_size}, "
                f"{(local_dp_rank + 1) * world_size}) "
                f'base value: "{os.getenv(device_control_env_var)}"'
            ) from e

    @contextmanager
    def _perform_handshakes(
        self,
        handshake_address: str,
        identity: bytes,
        local_client: bool,
        vllm_config: VllmConfig,
        client_handshake_address: str | None,
    ):
        """
        For Ray, we don't need to actually perform handshake.
        All addresses information is known before the actor creation.
        Therefore, we simply yield these addresses.
        """
        yield self.addresses

    def wait_for_init(self):
        """
        Wait until the engine core is initialized.

        This is just an empty method. When ray.get() on this method
        (or any other method of the actor) returns, it is guaranteed
        that actor creation (i.e., __init__) is complete.
        """
        pass

    def run(self):
        """
        Run the engine core busy loop.
        """
        try:
            self.run_busy_loop()
        except SystemExit:
            logger.debug("EngineCore exiting.")
            raise
        except Exception:
            logger.exception("EngineCore encountered a fatal error.")
            raise
        finally:
            self.shutdown()
