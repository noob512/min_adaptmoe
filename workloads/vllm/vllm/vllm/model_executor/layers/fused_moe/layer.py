# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import time
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from enum import Enum
from functools import partial
from typing import Literal, cast, get_args, overload

import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.parallel import ExpertPlacementStrategy
from vllm.device_allocator.uvm import (
    uvm_enabled_allocation_phase,
    uvm_local_memory_pool,
    uvm_slot_cuda_fence_enabled,
)
from vllm.device_allocator.uvm_pool_residency import (
    commit_uvm_regions_gpu_placement,
    prepare_uvm_regions_for_gpu_placement,
    protect_uvm_slot_direct_groups,
    register_uvm_region,
    release_uvm_regions,
    restore_uvm_expert_groups_to_slots,
    restore_uvm_registered_layer_plan,
    restore_uvm_regions,
    restore_uvm_regions_to_slots,
    validate_uvm_slot_direct_session_contract,
)
from vllm.device_allocator.uvm_region import (
    ExpertRecord,
    RegionState,
    compile_expert_envelopes,
    normalize_region_state,
)
from vllm.distributed import (
    get_dp_group,
    get_ep_group,
    get_pcp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.fused_moe import zero_experts_compute_triton
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    init_aiter_topK_meta_data,
)
from vllm.model_executor.layers.fused_moe.routing_simulator import RoutingSimulator
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    is_flashinfer_supporting_global_sf,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import (
    aux_stream,
    current_stream,
    direct_register_custom_op,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id

if current_platform.is_cuda_alike():
    from .fused_moe import eplb_map_to_physical_and_record, fused_experts
else:
    fused_experts = None  # type: ignore
    FusedMoEPermuteExpertsUnpermute = object  # type: ignore
    FusedMoEPrepareAndFinalize = object  # type: ignore

    def _eplb_map_to_physical_and_record(
        topk_ids: torch.Tensor,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> torch.Tensor:
        # CPU fallback: no EPLB so just return as is
        return topk_ids

    eplb_map_to_physical_and_record = _eplb_map_to_physical_and_record
from vllm.model_executor.layers.fused_moe.fused_moe import grouped_topk
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (  # noqa: E501
    rocm_aiter_grouped_topk,
)

if current_platform.is_tpu():
    from .moe_pallas import fused_moe as fused_moe_pallas
else:
    fused_moe_pallas = None  # type: ignore

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

logger = init_logger(__name__)

_UVM_INIT_WEIGHT_DRIVER_RELEASE_MODES = (
    "driver_release",
    "driver_release_cpu",
    "offload_release",
    "release_cpu",
)
_UVM_EXPERT_512K_ENVELOPE_BYTES = 512 * 1024
_UVM_EXPERT_2MB_ENVELOPE_BYTES = 2 * 1024 * 1024
_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST_ENV = (
    "VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST"
)
_UVM_SLOT_FORMAL_RESTORE_ALL_COLD_ENV = (
    "VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD"
)


def _env_flag_enabled(env_name: str, default: bool = False) -> bool:
    env_value = os.environ.get(env_name)
    if env_value is None:
        return default
    return env_value.lower() in ("1", "true", "yes", "on")


def _env_int(env_name: str, default: int) -> int:
    env_value = os.environ.get(env_name)
    if env_value is None:
        return default
    try:
        return int(env_value)
    except ValueError:
        logger.warning(
            "Invalid integer for %s=%s; using %s",
            env_name,
            env_value,
            default,
        )
        return default


def _validate_uvm_slot_restore_execution(vllm_config: VllmConfig) -> None:
    unsupported: list[str] = []
    if not vllm_config.model_config.enforce_eager:
        unsupported.append("CUDA graph execution")
    if vllm_config.parallel_config.enable_dbo:
        unsupported.append("dual-batch overlap")
    if vllm_config.parallel_config.world_size_across_dp != 1:
        unsupported.append("multi-GPU execution")
    if unsupported:
        raise RuntimeError(
            "UVM slot_restore currently requires single-GPU eager execution "
            "with DBO disabled; unsupported: " + ", ".join(unsupported)
        )


def _uvm_slot_formal_restore_all_cold_enabled(
    access_mode: str,
    vllm_config: VllmConfig,
) -> bool:
    enabled = _env_flag_enabled(
        _UVM_SLOT_FORMAL_RESTORE_ALL_COLD_ENV,
        default=False,
    )
    if enabled and access_mode != "slot_restore":
        raise RuntimeError(
            f"{_UVM_SLOT_FORMAL_RESTORE_ALL_COLD_ENV}=1 requires "
            "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=slot_restore"
        )
    if access_mode == "slot_restore":
        _validate_uvm_slot_restore_execution(vllm_config)
    return enabled


def _align_down(value: int, alignment: int) -> int:
    alignment = max(int(alignment), 1)
    return int(value) - (int(value) % alignment)


def _align_up(value: int, alignment: int) -> int:
    alignment = max(int(alignment), 1)
    return ((int(value) + alignment - 1) // alignment) * alignment


def _parse_plan_expert_keys(path: str, plan_key: str) -> set[tuple[int, int]]:
    if not path:
        return set()
    try:
        with open(path, encoding="utf-8") as handle:
            plan = json.load(handle)
    except OSError:
        logger.warning("Failed to read Stage I plan file: %s", path, exc_info=True)
        return set()
    except json.JSONDecodeError:
        logger.warning("Failed to parse Stage I plan file: %s", path, exc_info=True)
        return set()

    records = plan.get(plan_key)
    if not isinstance(records, list):
        return set()

    keys: set[tuple[int, int]] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            layer_id = int(record["layer_id"])
            expert_id = int(record["expert_id"])
        except (KeyError, TypeError, ValueError):
            continue
        keys.add((layer_id, expert_id))
    return keys


def _load_uvm_weight_plan(path: str) -> dict[str, object]:
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            plan = json.load(handle)
    except OSError:
        logger.warning("Failed to read UVM weight plan file: %s", path, exc_info=True)
        return {}
    except json.JSONDecodeError:
        logger.warning("Failed to parse UVM weight plan file: %s", path, exc_info=True)
        return {}
    return plan if isinstance(plan, dict) else {}


def _uvm_weight_plan_policy(path: str) -> str:
    plan = _load_uvm_weight_plan(path)
    metadata = plan.get("metadata")
    if not isinstance(metadata, dict):
        return ""
    policy = metadata.get("policy")
    return str(policy) if policy is not None else ""


class FusedMoeWeightScaleSupported(Enum):
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    global_num_experts: int,
    expert_placement_strategy: ExpertPlacementStrategy = "linear",
    num_fused_shared_experts: int = 0,
    return_expert_mask: bool = False,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    """
    Calculates how many experts should be assigned to each rank for EP and
    creates a mapping from global to local expert index. Experts are
    distributed evenly across ranks. Any remaining are assigned to the
    last rank.

    Args:
        ep_size: The size of the expert parallel group
        ep_rank: The rank of the current process in the expert parallel
            group
        global_num_experts: The total number of experts in the model.
        expert_placement_strategy: The expert placement strategy.

    Returns:
        tuple[int, Optional[torch.Tensor]]: A tuple containing:
            - local_num_experts (int): The number of experts assigned
                to the current rank.
            - expert_map (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts,) mapping from global to local index.
                Contains -1 for experts not assigned to the current rank.
                Returns None if ep_size is 1.
            - expert_mask (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts + num_fused_shared_experts + 1,)
                containing 1 for experts assigned to the current rank
                and 0 for sentinel.
                Returns None if ep_size is 1.
                Used only when AITER MOE is enabled.
    """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None, None)

    # Distribute experts as evenly as possible to each rank.
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    # Create an expert map for the local experts
    if expert_placement_strategy == "linear":
        start_idx = ep_rank * base_experts + min(ep_rank, remainder)
        expert_map[start_idx : start_idx + local_num_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    elif expert_placement_strategy == "round_robin":
        local_log_experts = torch.arange(
            ep_rank, global_num_experts, ep_size, dtype=torch.int32
        )

        expert_map[local_log_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    else:
        raise ValueError(
            "Unsupported expert placement strategy "
            f"'{expert_placement_strategy}', expected one of "
            f"{get_args(ExpertPlacementStrategy)}"
        )

    expert_mask = None
    if return_expert_mask:
        expert_mask = torch.ones(
            (global_num_experts + num_fused_shared_experts + 1,), dtype=torch.int32
        )
        expert_mask[-1] = 0
        expert_mask[:global_num_experts] = expert_map > -1
        expert_map = torch.cat(
            (
                expert_map,
                torch.tensor(
                    [local_num_experts + i for i in range(num_fused_shared_experts)],
                    dtype=torch.int32,
                ),
            ),
            dim=0,
        )

    return (local_num_experts, expert_map, expert_mask)


def determine_expert_placement_strategy(
    expert_placement_strategy: ExpertPlacementStrategy,
    moe_parallel_config: FusedMoEParallelConfig,
    num_expert_group: int | None,
    num_redundant_experts: int,
    enable_eplb: bool,
) -> ExpertPlacementStrategy:
    if expert_placement_strategy == "round_robin":
        round_robin_supported = (
            (num_expert_group is not None and num_expert_group > 1)
            and num_redundant_experts == 0
            and not enable_eplb
        )

        if not round_robin_supported:
            logger.warning(
                "Round-robin expert placement is only supported for "
                "models with multiple expert groups and no redundant "
                "experts. Falling back to linear expert placement."
            )
            return "linear"
        if (
            moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.use_deepep_ll_kernels
        ):
            logger.warning(
                "Round-robin expert placement currently only supports "
                "the DeepEP low-latency backend, but '%s' was configured. "
                "Falling back to linear expert placement.",
                moe_parallel_config.all2all_backend,
            )
            return "linear"

    return expert_placement_strategy


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    """
    Compresses the expert map by removing any -1 entries.

    Args:
        expert_map (torch.Tensor): A tensor of shape (global_num_experts,)
            mapping from global to local index. Contains -1 for experts not
            assigned to the current rank.

    Returns:
        str: A string mapping from local to global index.
            Using str to support hashing for logging once only.
    """
    global_indices = torch.where(expert_map != -1)[0]
    local_indices = expert_map[global_indices]
    return ", ".join(
        f"{local_index.item()}->{global_index.item()}"
        for local_index, global_index in zip(local_indices, global_indices)
    )


def maybe_roundup_hidden_size(
    hidden_size: int,
    act_dtype: torch.dtype,
    quant_config: QuantizationConfig | None,
    moe_parallel_config: FusedMoEParallelConfig,
    is_lora_enabled: bool,
) -> int:
    """
    Given layer hidden size and MoE configurations, round up hidden_size
    if necessary.

    Args:
        hidden_size: Layer hidden-size
        act_dtype: Data type of the layer activations.
        quant_config: Fused MoE quantization configuration.
        moe_parallel_config: Fused MoE parallelization strategy configuration.
        is_lora_enabled: True if the engine is enabled with LoRA. This
            is used in the case of mxfp4 quantization in selecting the
            MxFP4Backend.

    Return:
        Rounded up hidden_size if rounding up is required based on the configs.
        Original hidden size otherwise.
    """
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_roundup_layer_hidden_size,
    )

    hidden_size = maybe_roundup_layer_hidden_size(
        hidden_size, act_dtype, moe_parallel_config
    )

    # we are padding globally so EP buffer allocation works
    if quant_config and quant_config.get_name() == "mxfp4":
        from vllm.model_executor.layers.quantization.mxfp4 import (
            Mxfp4Backend,
            get_mxfp4_backend,
        )

        current_mxfp4_backend = get_mxfp4_backend(is_lora_enabled)
        if (
            current_mxfp4_backend == Mxfp4Backend.SM90_FI_MXFP4_BF16
            or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_CUTLASS
        ):
            hidden_size = round_up(hidden_size, 128)
        elif (
            current_platform.is_rocm()
            or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_MXFP8_TRTLLM
            or current_mxfp4_backend == Mxfp4Backend.SM100_FI_MXFP4_BF16
        ):
            hidden_size = round_up(hidden_size, 256)

    return hidden_size


@CustomOp.register("fused_moe")
class FusedMoE(CustomOp):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all_reduce on the output of the layer
        renormalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
        enable_eplb: Whether to enable expert parallelism load balancer.
    """

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int | None = None,
        topk_group: int | None = None,
        quant_config: QuantizationConfig | None = None,
        tp_size: int | None = None,
        ep_size: int | None = None,
        dp_size: int | None = None,
        pcp_size: int | None = None,
        prefix: str = "",
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        is_act_and_mul: bool = True,
        enable_eplb: bool = False,
        num_redundant_experts: int = 0,
        has_bias: bool = False,
        is_sequence_parallel=False,
        zero_expert_num: int | None = 0,
        zero_expert_type: str | None = None,
        expert_mapping: list[tuple[str, str, int, str]] | None = None,
        n_shared_experts: int | None = None,
        routing_method_type: int | None = None,
    ):
        super().__init__()

        # Allow disabling of the separate shared experts stream for
        # debug purposes.
        # TODO: Remove this after more extensive testings with TP/DP
        # and other execution modes
        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.info_once("Disabling MoE shared_experts cuda stream")
            self.shared_experts_stream = None
        else:
            # TODO(rob): enable shared expert overlap with non-cuda-alike.
            # aux_stream() returns None on non-cuda-alike platforms.
            self.shared_experts_stream = aux_stream()
            if self.shared_experts_stream is not None:
                logger.info_once("Enabled separate cuda stream for MoE shared_experts")

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        vllm_config = get_current_vllm_config()
        self.vllm_config = vllm_config

        # FIXME (varun): We should have a better way of inferring the activation
        # datatype. This works for now as the tensor datatype entering the MoE
        # operation is typically unquantized (i.e. float16/bfloat16).
        if vllm_config.model_config is not None:
            moe_in_dtype = vllm_config.model_config.dtype
        else:
            # TODO (bnell): This is a hack to get test_mixtral_moe to work
            # since model_config is not set in the pytest test.
            moe_in_dtype = params_dtype

        tp_size_ = (
            tp_size if tp_size is not None else get_tensor_model_parallel_world_size()
        )
        dp_size_ = dp_size if dp_size is not None else get_dp_group().world_size
        pcp_size_ = pcp_size if pcp_size is not None else get_pcp_group().world_size

        self.is_sequence_parallel = is_sequence_parallel
        self.sp_size = tp_size_ if is_sequence_parallel else 1

        self.moe_parallel_config: FusedMoEParallelConfig = FusedMoEParallelConfig.make(
            tp_size_=tp_size_,
            pcp_size_=pcp_size_,
            dp_size_=dp_size_,
            vllm_parallel_config=vllm_config.parallel_config,
        )

        self.global_num_experts = num_experts + num_redundant_experts
        self.logical_num_experts = num_experts
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type

        # Expert mapping used in self.load_weights
        self.expert_mapping = expert_mapping

        # Round up hidden size if needed.
        hidden_size = maybe_roundup_hidden_size(
            hidden_size,
            moe_in_dtype,
            quant_config,
            self.moe_parallel_config,
            is_lora_enabled=self.vllm_config.lora_config is not None,
        )

        # For smuggling this layer into the fused moe custom op
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

        self.enable_eplb = enable_eplb
        self.expert_load_view: torch.Tensor | None = None
        self.logical_to_physical_map: torch.Tensor | None = None
        self.logical_replica_count: torch.Tensor | None = None
        self.expert_placement_strategy: ExpertPlacementStrategy = (
            vllm_config.parallel_config.expert_placement_strategy
        )

        # ROCm aiter shared experts fusion
        self.rocm_aiter_fmoe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
        self.aiter_fmoe_shared_expert_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )

        self.num_fused_shared_experts = (
            n_shared_experts
            if n_shared_experts is not None and self.aiter_fmoe_shared_expert_enabled
            else 0
        )
        if (
            not self.aiter_fmoe_shared_expert_enabled
            and self.num_fused_shared_experts != 0
        ):
            raise ValueError(
                "n_shared_experts is only supported on ROCm aiter when "
                "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS is enabled"
            )

        # Determine expert maps
        if self.use_ep:
            if self.enable_eplb:
                assert self.global_num_experts % self.ep_size == 0, (
                    "EPLB currently only supports even distribution of "
                    "experts across ranks."
                )
            else:
                assert num_redundant_experts == 0, (
                    "Redundant experts are only supported with EPLB."
                )

            self.expert_placement_strategy = determine_expert_placement_strategy(
                expert_placement_strategy=self.expert_placement_strategy,
                moe_parallel_config=self.moe_parallel_config,
                num_expert_group=num_expert_group,
                num_redundant_experts=num_redundant_experts,
                enable_eplb=self.enable_eplb,
            )

            self.expert_map: torch.Tensor | None
            local_num_experts, expert_map, expert_mask = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=self.expert_placement_strategy,
                num_fused_shared_experts=self.num_fused_shared_experts,
                return_expert_mask=self.rocm_aiter_fmoe_enabled,
            )
            self.local_num_experts = local_num_experts
            self.register_buffer("expert_map", expert_map)
            self.register_buffer("expert_mask", expert_mask)
            self._maybe_init_expert_routing_tables()
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Expert "
                "placement strategy: %s. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.",
                self.ep_rank,
                self.ep_size,
                self.expert_placement_strategy,
                self.local_num_experts,
                self.global_num_experts,
                get_compressed_expert_map(self.expert_map),
            )
        else:
            self.local_num_experts, self.expert_map, self.expert_mask = (
                self.global_num_experts,
                None,
                None,
            )

        self.top_k = top_k

        self._init_aiter_shared_experts_topK_buffer(
            vllm_config=vllm_config, dp_size=dp_size_
        )

        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.activation = activation

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError(
                "Only softmax scoring function is supported for non-grouped topk."
            )

        # ToDo: Better logic to determine the routing method type
        if routing_method_type is not None:
            self.routing_method_type = routing_method_type
        else:
            if scoring_func == "sigmoid":
                if self.use_grouped_topk:
                    self.routing_method_type = RoutingMethodType.DeepSeekV3
                elif self.top_k == 1:
                    self.routing_method_type = RoutingMethodType.Llama4
            elif self.scoring_func == "softmax":
                self.routing_method_type = (
                    RoutingMethodType.Renormalize
                    if not self.renormalize
                    else RoutingMethodType.RenormalizeNaive
                )
            else:
                self.routing_method_type = RoutingMethodType.TopK

        self.moe_config: FusedMoEConfig = FusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=moe_in_dtype,
            max_num_tokens=envs.VLLM_MOE_DP_CHUNK_SIZE,
            has_bias=has_bias,
            is_act_and_mul=is_act_and_mul,
            is_lora_enabled=vllm_config.lora_config is not None,
        )
        self.moe_config_use_flashinfer_cutlass_kernels = (
            self.moe_config.use_flashinfer_cutlass_kernels
        )
        self.uvm_moe_trace_enabled = _env_flag_enabled(
            "VLLM_UVM_MOE_ROUTING_TRACE_ENABLE", default=False
        )
        self.uvm_moe_trace_file = os.environ.get(
            "VLLM_UVM_MOE_ROUTING_TRACE_FILE",
            "vllm_uvm_moe_routing_trace.jsonl",
        )
        self._uvm_moe_trace_step = 0
        self._uvm_expert_records: dict[tuple[int, int], ExpertRecord] = {}
        self.uvm_expert_offload_access_mode = (
            os.environ.get(
                "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE",
                "restore",
            )
            .strip()
            .lower()
        )
        if self.uvm_expert_offload_access_mode not in (
            "restore",
            "remote_read",
            "slot_restore",
        ):
            logger.warning(
                "Unknown VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=%s; using restore",
                self.uvm_expert_offload_access_mode,
            )
            self.uvm_expert_offload_access_mode = "restore"
        self.uvm_slot_formal_restore_all_cold = (
            _uvm_slot_formal_restore_all_cold_enabled(
                self.uvm_expert_offload_access_mode,
                vllm_config,
            )
        )
        self.uvm_weight_executor = os.environ.get(
            "VLLM_UVM_WEIGHT_EXECUTOR",
            "cpp_batch",
        )
        if self.uvm_weight_executor not in (
            "cpp_batch",
            "driver_batch",
            "auto",
        ):
            logger.warning(
                "Unknown VLLM_UVM_WEIGHT_EXECUTOR=%s; using cpp_batch",
                self.uvm_weight_executor,
            )
            self.uvm_weight_executor = "cpp_batch"
        legacy_2mb_envelope_enable = _env_flag_enabled(
            "VLLM_UVM_EXPERT_2MB_ENVELOPE_ENABLE",
            default=False,
        )
        legacy_2mb_envelope_bytes = _env_int(
            "VLLM_UVM_EXPERT_2MB_ENVELOPE_BYTES",
            default=_UVM_EXPERT_2MB_ENVELOPE_BYTES,
        )
        if legacy_2mb_envelope_bytes <= 0:
            legacy_2mb_envelope_bytes = _UVM_EXPERT_2MB_ENVELOPE_BYTES
        legacy_2mb_envelope_bytes = _align_up(
            legacy_2mb_envelope_bytes,
            4096,
        )
        generic_envelope_env_present = (
            "VLLM_UVM_EXPERT_ENVELOPE_ENABLE" in os.environ
            or "VLLM_UVM_EXPERT_ENVELOPE_BYTES" in os.environ
        )
        self.uvm_expert_envelope_enable = _env_flag_enabled(
            "VLLM_UVM_EXPERT_ENVELOPE_ENABLE",
            default=legacy_2mb_envelope_enable,
        )
        self.uvm_expert_envelope_bytes = _env_int(
            "VLLM_UVM_EXPERT_ENVELOPE_BYTES",
            default=(
                _UVM_EXPERT_512K_ENVELOPE_BYTES
                if generic_envelope_env_present
                else legacy_2mb_envelope_bytes
            ),
        )
        if self.uvm_expert_envelope_bytes <= 0:
            self.uvm_expert_envelope_bytes = (
                _UVM_EXPERT_512K_ENVELOPE_BYTES
                if generic_envelope_env_present
                else _UVM_EXPERT_2MB_ENVELOPE_BYTES
            )
        self.uvm_expert_envelope_bytes = _align_up(
            self.uvm_expert_envelope_bytes,
            64 * 1024,
        )
        self.uvm_init_weight_offload_enable = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE",
            default=False,
        )
        self.uvm_init_weight_offload_mode = os.environ.get(
            "VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE",
            "trace_only",
        )
        if self.uvm_init_weight_offload_mode not in (
            "trace_only",
            *_UVM_INIT_WEIGHT_DRIVER_RELEASE_MODES,
        ):
            logger.warning(
                "Unknown VLLM_UVM_INIT_WEIGHT_OFFLOAD_MODE=%s; using trace_only",
                self.uvm_init_weight_offload_mode,
            )
            self.uvm_init_weight_offload_mode = "trace_only"
        self.uvm_init_weight_offload_plan_file = os.environ.get(
            "VLLM_UVM_INIT_WEIGHT_PLAN_FILE",
            os.environ.get("VLLM_UVM_WEIGHT_OFFLOAD_PLAN_FILE", ""),
        )
        self.uvm_init_weight_offload_plan_keys = _parse_plan_expert_keys(
            self.uvm_init_weight_offload_plan_file,
            "offload_plan",
        )
        self.uvm_init_weight_hot_plan_keys = _parse_plan_expert_keys(
            self.uvm_init_weight_offload_plan_file,
            "hot_plan",
        ) | (
            _parse_plan_expert_keys(
                self.uvm_init_weight_offload_plan_file,
                "prefetch_plan",
            )
            - self.uvm_init_weight_offload_plan_keys
        )
        self.uvm_init_weight_warm_plan_keys = _parse_plan_expert_keys(
            self.uvm_init_weight_offload_plan_file,
            "warm_plan",
        )
        self.uvm_init_weight_plan_policy = _uvm_weight_plan_policy(
            self.uvm_init_weight_offload_plan_file,
        )
        self._uvm_init_weight_offload_plan_loaded = bool(
            self.uvm_init_weight_offload_plan_file
            and os.path.isfile(self.uvm_init_weight_offload_plan_file)
        )
        self.uvm_init_weight_complete_expert_plan = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_COMPLETE_EXPERT_PLAN",
            default=self.uvm_init_weight_offload_enable,
        )
        self.uvm_init_weight_warm_offload_enable = _env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE",
            default=False,
        )
        self.uvm_incremental_expert_offload_enable = _env_flag_enabled(
            "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE",
            default=False,
        )
        self.uvm_dynamic_expert_swap_enable = _env_flag_enabled(
            "VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE",
            default=False,
        )
        self._uvm_dynamic_expert_heat_counts: torch.Tensor | None = None
        self._uvm_dynamic_expert_heat_buffers: list[torch.Tensor] = []
        self._uvm_dynamic_expert_heat_active_index = 0
        self._uvm_dynamic_expert_heat_epoch = 0
        self._uvm_dynamic_expert_heat_graph_stable = False
        self._uvm_incremental_offload_in_progress = False
        self._uvm_incremental_selected_global_expert_id: int | None = None
        self._uvm_offloaded_global_expert_ids: set[int] = set()
        self._uvm_remote_read_global_expert_ids: set[int] = set()
        # Tracks only experts released by runtime KV-pressure waves. Initial
        # static-plan experts are deliberately excluded so KV shrink can undo
        # exactly one balanced wave without changing the frozen base plan.
        self._uvm_incremental_offloaded_global_expert_ids: set[int] = set()
        self._uvm_init_weight_offload_done = False
        managed_expert_weights_requested = _env_flag_enabled(
            "VLLM_UVM_EXPERT_WEIGHTS_MANAGED",
            default=False,
        )
        # Imported lazily to avoid a models.utils <-> FusedMoE import cycle.
        # set_cpu_offload_max_bytes() freezes the exact-name plan before model
        # construction, so this decision is stable for the layer lifetime.
        from vllm.model_executor.models.utils import (
            cpu_offload_uvm_original_va_static_alias_enabled,
            use_uvm_expert_weight_allocation,
        )

        self.uvm_expert_weights_managed = use_uvm_expert_weight_allocation(
            prefix,
            managed_requested=managed_expert_weights_requested,
        )
        # Fail-fast fault mode deliberately cannot populate a fresh managed
        # allocation on first GPU touch. Selected original-VA parameters are
        # therefore placed explicitly during initialization; finalize() later
        # moves them to CPU and installs the sealed static GPU mappings.
        self.uvm_expert_weights_explicit_prefetch = bool(
            self.uvm_expert_weights_managed
            and cpu_offload_uvm_original_va_static_alias_enabled()
        )
        self.quant_config = quant_config

        def _get_quant_method() -> FusedMoEMethodBase:
            """
            Helper method to ensure self.quant_method is never None and
            of the proper type.
            """
            quant_method = None
            if self.quant_config is not None:
                quant_method = self.quant_config.get_quant_method(self, prefix)
            if quant_method is None:
                quant_method = UnquantizedFusedMoEMethod(self.moe_config)
            assert isinstance(quant_method, FusedMoEMethodBase)
            return quant_method

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        self.quant_method: FusedMoEMethodBase = _get_quant_method()

        if not self.moe_config.is_act_and_mul:
            # Avoid circular import
            from vllm.model_executor.layers.quantization.modelopt import (
                ModelOptFp8MoEMethod,
            )

            if not isinstance(
                self.quant_method, (UnquantizedFusedMoEMethod, ModelOptFp8MoEMethod)
            ):
                raise NotImplementedError(
                    "is_act_and_mul=False is supported only for unquantized "
                    "and ModelOpt FP8 moe for now"
                )
            if not current_platform.is_cuda():
                raise NotImplementedError(
                    "is_act_and_mul=False is supported only for CUDA for now"
                )

        if self.enable_eplb and not self.quant_method.supports_eplb:
            # TODO: Add support for additional quantization methods.
            # The implementation for other quantization methods does not
            # contain essential differences, but the current quant API
            # design causes duplicated work when extending to new
            # quantization methods, so I'm leaving it for now.
            # If you plan to add support for more quantization methods,
            # please refer to the implementation in `Fp8MoEMethod`.
            raise NotImplementedError(
                f"EPLB is not supported {self.quant_method.__class__.__name__}. "
                "EPLB is only supported for FP8 quantization for now."
            )

        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": params_dtype,
            "weight_loader": self.weight_loader,
            "global_num_experts": self.global_num_experts,
        }
        # need full intermediate size pre-sharding for WNA16 act order
        if self.quant_method.__class__.__name__ in (
            "GPTQMarlinMoEMethod",
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            moe_quant_params["intermediate_size_full"] = intermediate_size

        self.quant_method.create_weights(layer=self, **moe_quant_params)

        # Chunked all2all staging tensor
        self.batched_hidden_states: torch.Tensor | None = None
        self.batched_router_logits: torch.Tensor | None = None

    def uvm_expert_weight_allocation_scope(self):
        if not self.uvm_expert_weights_managed:
            return nullcontext()
        if self.uvm_expert_weights_explicit_prefetch:
            return uvm_local_memory_pool(enable_prefetch=True)
        return uvm_local_memory_pool()

    # Note: maybe_init_modular_kernel should only be called by
    # prepare_communication_buffer_for_model.
    # This is called after all weight loading and post-processing, so it
    # should be safe to swap out the quant_method.
    def maybe_init_modular_kernel(self) -> None:
        self.ensure_moe_quant_config_init()
        # routing_tables only needed for round-robin expert placement with
        # DeepEP all2all backend.
        routing_tables = self._maybe_init_expert_routing_tables()
        prepare_finalize = self.quant_method.maybe_make_prepare_finalize(
            routing_tables=routing_tables
        )
        if prepare_finalize is not None:
            logger.debug(
                "%s for %s(%s)", prepare_finalize.__class__.__name__, self, id(self)
            )
            self.quant_method = FusedMoEModularMethod.make(
                self, self.quant_method, prepare_finalize, self.shared_experts
            )

    @property
    def shared_experts(self) -> torch.nn.Module | None:
        return None

    @property
    def gate(self) -> torch.nn.Module | None:
        return None

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def pcp_size(self):
        return self.moe_parallel_config.pcp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

    @property
    def pcp_rank(self):
        return self.moe_parallel_config.pcp_rank

    @property
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep

    @property
    def use_pplx_kernels(self):
        return self.moe_parallel_config.use_pplx_kernels

    @property
    def use_deepep_ht_kernels(self):
        return self.moe_parallel_config.use_deepep_ht_kernels

    @property
    def use_deepep_ll_kernels(self):
        return self.moe_parallel_config.use_deepep_ll_kernels

    @property
    def use_flashinfer_cutlass_kernels(self):
        return (
            self.moe_quant_config is not None
            and self.moe_quant_config.quant_dtype == "nvfp4"
            and self.moe_config_use_flashinfer_cutlass_kernels
        )

    @property
    def use_marlin_kernels(self):
        return getattr(self.quant_method, "use_marlin", False)

    @property
    def use_dp_chunking(self) -> bool:
        return (
            self.moe_parallel_config.use_pplx_kernels
            or self.moe_parallel_config.use_deepep_ll_kernels
            or (self.dp_size > 1 and self.use_flashinfer_cutlass_kernels)
        )

    @property
    def is_internal_router(self) -> bool:
        # By default, router/gate is called before FusedMoE forward pass
        return False

    def _maybe_init_expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        # Currently routing_tables only needed for round-robin expert placement
        # with DeepEP-ll all2all backend.
        if (
            self.expert_placement_strategy != "round_robin"
            or not self.use_deepep_ll_kernels
        ):
            return None

        if hasattr(self, "expert_global_to_physical"):
            return cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                (
                    self.expert_global_to_physical,
                    self.expert_physical_to_global,
                    self.expert_local_to_global,
                ),
            )

        if self.expert_map is None:
            return None

        routing_tables = self.ensure_round_robin_expert_routing_tables(
            global_num_experts=self.global_num_experts,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            local_num_experts=self.local_num_experts,
            device=self.expert_map.device,
        )

        global_to_physical, physical_to_global, local_global = routing_tables
        self.register_buffer("expert_global_to_physical", global_to_physical)
        self.register_buffer("expert_physical_to_global", physical_to_global)
        self.register_buffer("expert_local_to_global", local_global)

        return routing_tables

    @staticmethod
    def ensure_round_robin_expert_routing_tables(
        global_num_experts: int,
        ep_size: int,
        ep_rank: int,
        local_num_experts: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device_kwargs = {"device": device} if device is not None else {}
        global_indices = torch.arange(
            global_num_experts, dtype=torch.long, **device_kwargs
        )
        owner = torch.remainder(global_indices, ep_size)
        local_index = torch.div(global_indices, ep_size, rounding_mode="floor")
        base = global_num_experts // ep_size
        remainder = global_num_experts % ep_size
        physical_offset = owner * base
        if remainder > 0:
            remainder_tensor = torch.tensor(
                remainder, dtype=torch.long, **device_kwargs
            )
            physical_offset = physical_offset + torch.minimum(owner, remainder_tensor)

        global_to_physical = physical_offset + local_index
        physical_to_global = torch.empty_like(global_to_physical)
        physical_to_global[global_to_physical] = global_indices

        local_global = torch.arange(
            ep_rank,
            global_num_experts,
            ep_size,
            dtype=torch.long,
            **device_kwargs,
        )
        if local_global.numel() != local_num_experts:
            local_global = local_global[:local_num_experts]

        return (global_to_physical, physical_to_global, local_global)

    def update_expert_map(self):
        # ep_size and ep_rank should already be updated
        assert self.expert_map is not None
        with self.expert_map.device:
            local_num_experts, expert_map, expert_mask = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
                expert_placement_strategy=self.expert_placement_strategy,
                num_fused_shared_experts=self.num_fused_shared_experts,
                return_expert_mask=self.rocm_aiter_fmoe_enabled,
            )
            self.local_num_experts = local_num_experts
            self.register_buffer("expert_map", expert_map)
            self.register_buffer("expert_mask", expert_mask)
            self._maybe_init_expert_routing_tables()
            if self.aiter_fmoe_shared_expert_enabled:
                self._init_aiter_shared_experts_topK_buffer(
                    vllm_config=get_current_vllm_config(),
                    dp_size=get_dp_group().world_size,
                )

    def _maybe_setup_shared_experts_stream(
        self,
        hidden_states: torch.Tensor,
        has_separate_shared_experts: bool,
        use_chunked_impl: bool,
    ) -> tuple[bool, torch.Tensor | None]:
        use_shared_experts_stream = (
            has_separate_shared_experts
            and not use_chunked_impl
            and self.shared_experts_stream is not None
            and (
                hidden_states.shape[0]
                <= envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
            )
        )

        hidden_states_clone: torch.Tensor | None = None
        if use_shared_experts_stream:
            assert self.shared_experts_stream is not None

            # Clone BEFORE switching streams to avoid race condition
            # where routed_expert kernel may mutate hidden_states.
            hidden_states_clone = hidden_states.clone()

            # Record that the clone will be used by shared_experts_stream
            # to avoid gc issue from deallocation of hidden_states_clone
            # For more details: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html # noqa: E501
            # NOTE: We dont need shared_output.record_stream(current_stream())
            # because we synch the streams before using shared_output.
            hidden_states_clone.record_stream(self.shared_experts_stream)

            # Mark sync start point for the separate shared experts
            # stream here since we want to run in parallel with the
            # router/gate (next op below)
            assert self.shared_experts_stream is not None
            self.shared_experts_stream.wait_stream(current_stream())

        return use_shared_experts_stream, hidden_states_clone

    def _load_per_tensor_weight_scale(
        self,
        shard_id: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_combined_w13_weight_scale(
        self,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        param: torch.Tensor,
        tp_rank: int,
    ):
        """
        Load w13 weight scales assuming that w1 weight scales and w3 weight
        scales are stored in the same loaded_weight tensor.
        """
        shard_size = param.shape[shard_dim]
        loaded_weight = loaded_weight.narrow(
            shard_dim, shard_size * tp_rank, shard_size
        )
        param.copy_(loaded_weight)

    def _load_model_weight_or_group_weight_scale(
        self,
        shard_dim: int,
        expert_data: torch.Tensor,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full_w2: bool = False,
    ):
        """
        Load grouped weight scales for group quantization or model weights
            :param shard_dim: dimension to shard
            :param expert_data: parameter for a particular expert
            :param shard_id: either w1, w2, or w3
            :param loaded_weight: checkpoint weight to load into the param
            :param tp_rank: tensor parallel rank
            :param load_full_w2: whether or not the w2 loaded should be sharded.
        """
        if shard_id == "w2":
            # In the case where we have actorder/g_idx, we do not partition the
            # w2 scales, as indicated by `load_full` argument, for all tp cases
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                load_full=load_full_w2,
            )
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        if self.moe_config.is_act_and_mul:
            shard_size = expert_data.shape[shard_dim] // 2
        else:
            shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        expert_data.copy_(loaded_weight)

    def _load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
        # w2, down_proj: Load into only logical weight of w2.
        expert_data.copy_(loaded_weight)

    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(
        self,
        shard_id: str,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        if shard_id == "w2":
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

    def _init_aiter_shared_experts_topK_buffer(
        self, vllm_config: VllmConfig, dp_size: int
    ):
        if self.num_fused_shared_experts > 0:
            init_aiter_topK_meta_data(
                n_routed_experts=self.global_num_experts,
                n_shared_experts=self.num_fused_shared_experts,
                top_k=self.top_k,
                tp_rank=self.ep_rank if self.use_ep else self.tp_rank,
                tp_size=self.ep_size if self.use_ep else self.tp_size,
                shared_experts_score=1.0,
                max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens
                * dp_size,
                is_EP=self.use_ep,
            )
        self.local_num_experts += self.num_fused_shared_experts

    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[False],
    ) -> None: ...

    @overload
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: Literal[True],
    ) -> bool: ...

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if self.quant_config and self.quant_config.get_name() == "mxfp4":
            # (FIXME) for gpt-oss all experts are combined
            if "bias" in weight_name:
                dim1 = loaded_weight.shape[1]
                param.data[:, :dim1].copy_(loaded_weight)
            else:
                dim1 = loaded_weight.shape[1]
                dim2 = loaded_weight.shape[2]
                param.data[:, :dim1, :dim2].copy_(loaded_weight)
            return True if return_success else None

        quant_method_name = self.quant_method.__class__.__name__
        global_expert_id = expert_id
        expert_id = self._map_global_expert_id_to_local_expert_id(global_expert_id)

        allow_flashinfer = getattr(self.quant_method, "allow_flashinfer", False)
        moe_backend = getattr(self.quant_method, "flashinfer_moe_backend", None)

        use_global_sf = (
            allow_flashinfer
            and is_flashinfer_supporting_global_sf(moe_backend)
            and "input_scale" in weight_name
            and quant_method_name == "ModelOptNvFp4FusedMoE"
        )

        if expert_id == -1 and not use_global_sf:
            # Failed to load this param since it's not local to this rank
            return False if return_success else None
        # Hereafter, `expert_id` is local physical id

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        if self.quant_method.__class__.__name__ in (
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            loaded_weight = loaded_weight.t().contiguous()

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but got {shard_id}.")

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        is_gguf_weight = getattr(param, "is_gguf_weight", False)
        is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
        if is_gguf_weight_type:
            param.weight_type = loaded_weight.item()
            param.data.copy_(loaded_weight)
            return True if return_success else None

        # Case for BitsAndBytes
        use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
        if use_bitsandbytes_4bit:
            shard_dim = 0

            expert_data = param.data[expert_id]
            if shard_id == "w2":
                expert_data.copy_(loaded_weight)
            elif shard_id in ("w1", "w3"):
                # BNB inflight quantization has already sharded the weights
                full_load = True
                self._load_w13(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full=full_load,
                )
            return True if return_success else None

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        # Materialize GGUF UninitializedParameter
        if is_gguf_weight and isinstance(param, UninitializedParameter):
            final_shape = list(loaded_weight.shape)
            if shard_id in ["w1", "w3"]:
                final_shape[1] *= 2
            final_shape[shard_dim] = final_shape[shard_dim] // self.tp_size
            param.materialize(final_shape, dtype=loaded_weight.dtype)

        expert_data = param.data if full_load else param.data[expert_id]

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                "compressed" in quant_method_name.lower()
                and param.data[expert_id] != 1
                and (param.data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param,
                loaded_weight=loaded_weight,
                expert_id=global_expert_id if use_global_sf else expert_id,
            )
            return True if return_success else None

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank,
            )
            return True if return_success else None

        # TODO @dsikka: ModelOpt should follow the proper MoE loading pattern
        if "ModelOpt" in quant_method_name:
            # Determine per-tensor weight scale patterns based on variant
            # Use the dedicated method instead of brittle string matching
            uses_weight_scale_2 = self.quant_method.uses_weight_scale_2_pattern()

            # Call _load_per_tensor_weight_scale() to load per-tensor (scalar)
            # weights scales.
            # Input scales are always per-tensor.
            # Weight scales: FP4 uses "weight_scale_2" and FP8 uses
            # "weight_scale" for per-tensor scales.
            is_per_tensor = (
                "weight_scale_2" in weight_name
                if uses_weight_scale_2
                else "weight_scale" in weight_name
            ) or "input_scale" in weight_name
            if is_per_tensor:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
                return True if return_success else None

            # If the weight is w13_weight_scale and w13_weight_scales are
            # combined into single loaded_weight, call
            # _load_combined_w13_weight_scale() to load it.
            # This is checked by comparing the hidden_out dims of the
            # loaded_weight and the param.
            if "w13_weight_scale" in weight_name:
                loaded_weight_hidden_out = loaded_weight.shape[-2]
                param_hidden_out = param.data.shape[-2] * self.tp_size
                if loaded_weight_hidden_out == param_hidden_out:
                    self._load_combined_w13_weight_scale(
                        shard_dim=shard_dim,
                        loaded_weight=loaded_weight,
                        param=param,
                        tp_rank=self.tp_rank,
                    )
                    return True if return_success else None

            # For other weights, call _load_model_weight_or_group_weight_scale()
            # to load it.
            if "weight" in weight_name:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                )
            return True if return_success else None

        # Case weight scales, zero_points and offset, weight/input global scales
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                )
            elif quant_method in [
                FusedMoeWeightScaleSupported.GROUP.value,
                FusedMoeWeightScaleSupported.BLOCK.value,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full_w2=getattr(param, "load_full_w2", False),
                )
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            else:
                WEIGHT_SCALE_SUPPORTED = [e.value for e in FusedMoeWeightScaleSupported]
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}"
                )
            return True if return_success else None

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return True if return_success else None

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank,
            )
            return True if return_success else None

        return False if return_success else None

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        if (expert_mapping := self.expert_mapping) is None:
            raise ValueError(
                "`self.expert_mapping` must be provided to "
                "load weights using `self.load_weights`."
            )
        for expert_name, loaded_weight in weights:
            qual_name = f"{self.layer_name}.{expert_name}"
            for param_name, weight_name, expert_id, shard_id in expert_mapping:
                if weight_name not in qual_name:
                    continue
                weight_name = qual_name.replace(weight_name, param_name)
                param_name = weight_name.removeprefix(f"{self.layer_name}.")
                param = getattr(self, param_name)
                success = self.weight_loader(
                    param=param,
                    loaded_weight=loaded_weight,
                    weight_name=weight_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    logger.debug(
                        "Loaded %s for expert %d into %s",
                        param_name,
                        expert_id,
                        self.layer_name,
                    )
                    yield param_name

    def get_expert_weights(self) -> Iterable[torch.Tensor]:
        def _maybe_make_contiguous(
            name: str, p: torch.nn.Parameter
        ) -> torch.nn.Parameter:
            """
            In some cases, the last 2 dimensions (the non-expert dimensions)
            of the weight scale tensor are transposed. This function
            transforms the tensor (view update) so the tensor is contiguous().
            Example: A non-contiguous scale tensor,
              `x` of shape (E, 32, 16) and stride (512, 1, 32) is transformed to
              `x_` of shape (E, 16, 32) and stride (512, 32, 1).
              Note that we specifically use torch.transpose() so `x_` refers
              to the same underlying memory. The tensors `x` and `x_`, pointing
              to the same underlying memory make this transformation safe in the
              context of EPLB. i.e. It is the same memory and just the view
              is different.
            Note: This function handles the "weight_scale" tensors specifically.
            This could however be generalized to handle similar tensors.
            """
            if p.ndim != 3:
                return p
            if p.is_contiguous():
                # Already contiguous. do nothing.
                return p
            # p is non-contiguous. We only handle the case where the last 2
            # dimensions of the scales tensor is transposed. We can handle
            # other cases when they become relevant.
            is_transposed_12 = p.stride(1) == 1 and p.stride(2) != 1
            if "weight_scale" not in name or not is_transposed_12:
                # do nothing.
                return p

            # Do not update the layer paramater as the layer's MoE operations would
            # expect the parameter's tensor to the same shape / stride. Instead,
            # make a new torch.nn.Parameter that is used just in the context of
            # EPLB.
            return torch.nn.Parameter(
                torch.transpose(p.data, 1, 2), requires_grad=False
            )

        weights = list(self.named_parameters())
        weights = [(name, _maybe_make_contiguous(name, p)) for name, p in weights]

        assert all(
            weight.is_contiguous()
            for name, weight in weights
            if not name.startswith("_shared_experts.")
        )

        # Filter out the non-expert weights.
        # `e_score_correction_bias` is a bias for each logical expert,
        # with shape (num_logical_experts,), not an expert weight.
        NON_EXPERT_WEIGHTS = {
            "e_score_correction_bias",
        }

        return [
            weight.view(self.local_num_experts, -1)
            for name, weight in weights
            if name not in NON_EXPERT_WEIGHTS
            and weight.shape != torch.Size([])
            and not name.startswith("_shared_experts.")
            # exclude parameters from non-expert submodules (e.g. gate/shared)
            and not name.startswith("_gate.")
        ]

    def set_eplb_state(
        self,
        moe_layer_idx: int,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        """
        Register the EPLB state in this layer.

        This is used later in forward pass, where we get the expert mapping
        and record the load metrics in `expert_load_view`.
        """
        self.expert_load_view = expert_load_view[moe_layer_idx]
        self.logical_to_physical_map = logical_to_physical_map[moe_layer_idx]
        self.logical_replica_count = logical_replica_count[moe_layer_idx]

    def ensure_moe_quant_config_init(self):
        if self.quant_method.moe_quant_config is None:
            # Note: the moe_quant_config can't be constructed until after
            # weight loading post processing.
            self.quant_method.moe_quant_config = (
                self.quant_method.get_fused_moe_quant_config(self)
            )

    @property
    def moe_quant_config(self) -> FusedMoEQuantConfig | None:
        self.ensure_moe_quant_config_init()
        return self.quant_method.moe_quant_config

    def ensure_dp_chunking_init(self):
        if not self.use_dp_chunking or self.batched_hidden_states is not None:
            return

        states_shape: tuple[int, ...]
        logits_shape: tuple[int, ...]

        moe = self.moe_config

        if self.vllm_config.parallel_config.enable_dbo:
            states_shape = (2, moe.max_num_tokens, self.hidden_size)
            logits_shape = (2, moe.max_num_tokens, self.logical_num_experts)
        else:
            states_shape = (moe.max_num_tokens, self.hidden_size)
            logits_shape = (moe.max_num_tokens, self.logical_num_experts)

        self.batched_hidden_states = torch.zeros(
            states_shape, dtype=moe.in_dtype, device=torch.cuda.current_device()
        )

        self.batched_router_logits = torch.zeros(
            logits_shape, dtype=moe.in_dtype, device=torch.cuda.current_device()
        )

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        Returns:
                (topk_weights, topk_ids, zero_expert_result)
                (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                The weights, expert ids, and zero expert computation result.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_topk,
            fused_topk_bias,
        )

        if self.enable_eplb:
            if self.quant_method.supports_eplb:
                if self.expert_load_view is None:
                    raise ValueError(
                        "enable_eplb=True requiere expert_load_view != None"
                    )
                if self.logical_to_physical_map is None:
                    raise ValueError(
                        "enable_eplb=True requiere logical_to_physical_map != None"
                    )
                if self.logical_replica_count is None:
                    raise ValueError(
                        "enable_eplb=True requiere logical_replica_count != None"
                    )
            else:
                raise NotImplementedError(
                    f"EPLB is not supported for {self.quant_method.method_name}."
                )

        indices_type = self.quant_method.topk_indices_dtype
        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY

        # Check if we should use a routing simulation strategy
        if routing_strategy != "":
            topk_weights, topk_ids = RoutingSimulator.simulate_routing(
                hidden_states=hidden_states,
                router_logits=router_logits,
                strategy_name=routing_strategy,
                top_k=self.top_k,
                indices_type=indices_type,
            )

        # DeepSeekv2 uses grouped_top_k
        elif self.use_grouped_topk:
            assert self.topk_group is not None
            assert self.num_expert_group is not None
            if rocm_aiter_ops.is_fused_moe_enabled():
                if not rocm_aiter_ops.is_fusion_moe_shared_experts_enabled():
                    assert self.num_fused_shared_experts == 0
                grouped_topk_impl = partial(
                    rocm_aiter_grouped_topk,
                    num_fused_shared_experts=self.num_fused_shared_experts,
                )
            else:
                grouped_topk_impl = grouped_topk

            topk_weights, topk_ids = grouped_topk_impl(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias,
            )
        elif self.e_score_correction_bias is not None:
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                e_score_correction_bias=self.e_score_correction_bias.data,
                topk=self.top_k,
                renormalize=self.renormalize,
            )
            if self.routed_scaling_factor != 1.0:
                topk_weights *= self.routed_scaling_factor
        elif self.custom_routing_function is None:
            topk_weights, topk_ids, token_expert_indices = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
                indices_type=indices_type,
            )
        else:
            topk_weights, topk_ids = self.custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
            )

        self._record_uvm_dynamic_expert_heat(topk_ids)

        if self.enable_eplb:
            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
            )

        if (indices_type is not None) and topk_ids.dtype != indices_type:
            topk_ids = topk_ids.to(dtype=indices_type)

        assert topk_ids.dtype == indices_type or indices_type is None

        # Compute zero expert result if needed
        if (
            self.zero_expert_num is not None
            and self.zero_expert_num > 0
            and self.zero_expert_type is not None
            and self.global_num_experts is not None
        ):
            zero_expert_result = zero_experts_compute_triton(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=self.global_num_experts,
                zero_expert_type=self.zero_expert_type,
                hidden_states=hidden_states,
            )
        else:
            zero_expert_result = None
        self._log_uvm_moe_routing_trace(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        self._uvm_restore_active_experts(topk_ids)
        self._uvm_moe_trace_step += 1
        return topk_weights, topk_ids, zero_expert_result

    def _log_uvm_moe_routing_trace(
        self,
        *,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> None:
        num_tokens = int(hidden_states.shape[0]) if hidden_states.ndim > 0 else 1
        if not self.uvm_moe_trace_enabled:
            return
        try:
            ids = topk_ids.detach()
            valid_ids = ids[ids >= 0].to(dtype=torch.long)
            counts = torch.bincount(
                valid_ids.flatten(),
                minlength=max(int(self.global_num_experts), 0),
            ).cpu()
            weight_sums: dict[str, float] = {}
            try:
                weights = topk_weights.detach()
                if weights.shape == ids.shape:
                    valid_weights = weights[ids >= 0].to(dtype=torch.float32)
                    weighted_counts = torch.bincount(
                        valid_ids.flatten(),
                        weights=valid_weights.flatten(),
                        minlength=max(int(self.global_num_experts), 0),
                    ).cpu()
                    weight_sums = {
                        str(int(expert_id)): float(weighted_counts[expert_id].item())
                        for expert_id in torch.nonzero(weighted_counts, as_tuple=False)
                        .flatten()
                        .tolist()
                        if weighted_counts[expert_id].item() > 0
                    }
            except Exception:
                logger.debug(
                    "Failed to compute weighted UVM MoE routing trace",
                    exc_info=True,
                )
            active = torch.nonzero(counts, as_tuple=False).flatten().tolist()
            active_counts = {
                str(int(expert_id)): int(counts[expert_id].item())
                for expert_id in active
                if counts[expert_id].item() > 0
            }
            record = {
                "timestamp": time.time(),
                "pid": os.getpid(),
                "layer_name": self.layer_name,
                "step": self._uvm_moe_trace_step,
                "num_tokens": num_tokens,
                "top_k": int(self.top_k),
                "global_num_experts": int(self.global_num_experts),
                "logical_num_experts": int(self.logical_num_experts),
                "local_num_experts": int(self.local_num_experts),
                "topk_shape": list(topk_ids.shape),
                "router_logits_shape": list(router_logits.shape),
                "expert_token_counts": active_counts,
                "expert_weight_sums": weight_sums,
                "active_experts": [int(expert_id) for expert_id in active],
                "topk_weight_sum": float(topk_weights.detach().sum().cpu().item()),
                "enable_eplb": bool(self.enable_eplb),
            }
            if getattr(self, "uvm_slot_formal_restore_all_cold", False):
                record.update(
                    {
                        "slot_formal_restore_all_cold": True,
                        "slot_formal_restore_expert_count": len(
                            self._uvm_offloaded_global_expert_ids
                        ),
                    }
                )
            with open(self.uvm_moe_trace_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception:
            logger.debug("Failed to write UVM MoE routing trace", exc_info=True)

    def _record_uvm_dynamic_expert_heat(self, topk_ids: torch.Tensor) -> None:
        """Accumulate logical expert assignments on GPU until maintenance."""
        if not self.uvm_dynamic_expert_swap_enable:
            return
        try:
            ids = topk_ids.detach().flatten().to(dtype=torch.long)
            if ids.numel() == 0:
                return
            if (
                self._uvm_dynamic_expert_heat_counts is None
                or self._uvm_dynamic_expert_heat_counts.device != ids.device
            ):
                self._uvm_dynamic_expert_heat_buffers = [
                    torch.zeros(
                        int(self.global_num_experts),
                        dtype=torch.int64,
                        device=ids.device,
                    )
                    for _ in range(2)
                ]
                self._uvm_dynamic_expert_heat_active_index = 0
                self._uvm_dynamic_expert_heat_counts = (
                    self._uvm_dynamic_expert_heat_buffers[0]
                )
            if ids.is_cuda and torch.cuda.is_current_stream_capturing():
                # A replayed CUDA Graph always writes to the address captured
                # here. Maintenance must therefore zero this stable counter
                # in place instead of switching the Python-side buffer pointer.
                self._uvm_dynamic_expert_heat_graph_stable = True
            elif bool(
                torch.any((ids < 0) | (ids >= int(self.global_num_experts))).item()
            ):
                return
            self._uvm_dynamic_expert_heat_counts.scatter_add_(
                0,
                ids,
                torch.ones_like(ids, dtype=torch.int64),
            )
        except Exception:
            logger.debug("Failed to record dynamic expert heat", exc_info=True)

    def get_uvm_dynamic_expert_heat(
        self,
        *,
        reset: bool = False,
    ) -> dict[str, object]:
        """Materialize one layer's routing window at a maintenance boundary."""
        counts_tensor = self._uvm_dynamic_expert_heat_counts
        if counts_tensor is None:
            counts = [0] * int(self.global_num_experts)
        else:
            # The counter is created during model inference and is therefore an
            # inference tensor. Maintenance RPCs run outside that context.
            with torch.inference_mode():
                graph_stable = bool(
                    getattr(
                        self,
                        "_uvm_dynamic_expert_heat_graph_stable",
                        False,
                    )
                )
                if (
                    reset
                    and not graph_stable
                    and len(self._uvm_dynamic_expert_heat_buffers) == 2
                ):
                    completed_index = self._uvm_dynamic_expert_heat_active_index
                    self._uvm_dynamic_expert_heat_active_index = 1 - completed_index
                    self._uvm_dynamic_expert_heat_counts = (
                        self._uvm_dynamic_expert_heat_buffers[
                            self._uvm_dynamic_expert_heat_active_index
                        ]
                    )
                    counts_tensor = self._uvm_dynamic_expert_heat_buffers[
                        completed_index
                    ]
                    self._uvm_dynamic_expert_heat_epoch += 1
                elif reset:
                    self._uvm_dynamic_expert_heat_epoch += 1
                counts = [int(value) for value in counts_tensor.detach().cpu().tolist()]
                if reset:
                    counts_tensor.zero_()
        layer_id = self._uvm_layer_id()
        static_alias_fn = getattr(self, "_uvm_static_alias_owned_experts", None)
        static_alias_owned = (
            static_alias_fn(layer_id)
            if layer_id is not None and callable(static_alias_fn)
            else set()
        )
        expert_records = getattr(self, "_uvm_expert_records", None)
        swappable_experts = sorted(
            int(expert_id)
            for record_layer_id, expert_id in (
                expert_records
                if isinstance(expert_records, dict)
                else {
                    (int(layer_id), expert_id): None
                    for expert_id in range(int(self.global_num_experts))
                }
            )
            if layer_id is not None
            and int(record_layer_id) == int(layer_id)
            and int(expert_id) not in static_alias_owned
        )
        return {
            "ok": True,
            "layer_name": self.layer_name,
            "layer_id": layer_id,
            "global_num_experts": int(self.global_num_experts),
            "local_num_experts": int(self.local_num_experts),
            "expert_token_counts": counts,
            "remote_experts": sorted(self._uvm_remote_read_global_expert_ids),
            "swappable_experts": swappable_experts,
            "static_alias_owned_experts": sorted(static_alias_owned),
            "incremental_offloaded_experts": sorted(
                getattr(
                    self,
                    "_uvm_incremental_offloaded_global_expert_ids",
                    set(),
                )
            ),
            "reset": bool(reset),
            "heat_epoch": int(self._uvm_dynamic_expert_heat_epoch),
            "double_buffered": (
                len(self._uvm_dynamic_expert_heat_buffers) == 2
                and not bool(
                    getattr(
                        self,
                        "_uvm_dynamic_expert_heat_graph_stable",
                        False,
                    )
                )
            ),
            "counter_mode": (
                "graph_stable_synchronous"
                if bool(
                    getattr(
                        self,
                        "_uvm_dynamic_expert_heat_graph_stable",
                        False,
                    )
                )
                else "double_buffered"
            ),
        }

    def _uvm_layer_id(self) -> int | None:
        parts = self.layer_name.split(".")
        for index, part in enumerate(parts[:-1]):
            if part in ("layers", "layer"):
                try:
                    return int(parts[index + 1])
                except ValueError:
                    return None
        return None

    def _uvm_global_to_local_expert(self, global_expert_id: int) -> int:
        if self.expert_map is None:
            return global_expert_id
        if 0 <= global_expert_id < int(self.expert_map.numel()):
            return int(self.expert_map[global_expert_id].item())
        return -1

    def _uvm_active_expert_pairs(
        self,
        topk_ids: torch.Tensor,
    ) -> list[tuple[int, int]]:
        ids = topk_ids.detach()
        valid_ids = ids[ids >= 0].to(dtype=torch.long)
        if valid_ids.numel() == 0:
            return []
        unique_ids = torch.unique(valid_ids).cpu().tolist()
        expert_pairs: list[tuple[int, int]] = []
        for expert_id in unique_ids:
            global_expert_id = int(expert_id)
            local_expert_id = self._uvm_global_to_local_expert(global_expert_id)
            if 0 <= local_expert_id < int(self.local_num_experts):
                expert_pairs.append((global_expert_id, local_expert_id))
        return sorted(set(expert_pairs))

    def _uvm_all_local_expert_pairs(self) -> list[tuple[int, int]]:
        expert_pairs: set[tuple[int, int]] = set()
        mapped_local_ids: set[int] = set()
        for global_expert_id in range(max(int(self.global_num_experts), 0)):
            local_expert_id = self._uvm_global_to_local_expert(global_expert_id)
            if 0 <= local_expert_id < int(self.local_num_experts):
                expert_pairs.add((global_expert_id, local_expert_id))
                mapped_local_ids.add(local_expert_id)

        for local_expert_id in range(max(int(self.local_num_experts), 0)):
            if local_expert_id in mapped_local_ids:
                continue
            if local_expert_id < int(self.global_num_experts):
                expert_pairs.add((local_expert_id, local_expert_id))
        return sorted(expert_pairs)

    def _uvm_init_weight_expert_is_hot(
        self,
        layer_id: int | None,
        global_expert_id: int,
    ) -> bool:
        if layer_id is None:
            return False
        plan_key = (layer_id, global_expert_id)
        if plan_key in self.uvm_init_weight_offload_plan_keys:
            return False
        return plan_key in self.uvm_init_weight_hot_plan_keys

    def _uvm_init_weight_expert_class(
        self,
        *,
        layer_id: int | None,
        global_expert_id: int,
    ) -> str:
        if self._uvm_init_weight_expert_is_hot(layer_id, global_expert_id):
            return "hot"
        if (
            layer_id is not None
            and (
                layer_id,
                global_expert_id,
            )
            in self.uvm_init_weight_offload_plan_keys
        ):
            return "cold"
        return "warm"

    @staticmethod
    def _uvm_static_alias_owned_experts(layer_id: int) -> set[int]:
        from vllm.model_executor.models.utils import (
            cpu_offload_uvm_original_va_static_alias_experts_for_layer,
        )

        return set(
            cpu_offload_uvm_original_va_static_alias_experts_for_layer(
                int(layer_id)
            )
        )

    def _uvm_stage_m_offload_expert_pairs(
        self,
        layer_id: int,
        all_local_pairs: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        # Runtime KV-pressure waves are deliberately independent from the
        # initialization-time Stage-M pass.  In particular, an original-VA
        # static-alias plan may have already released its frozen base experts
        # during finalize(), while later explicit waves still need the plan
        # classification helpers below.  Do not require callers to enable the
        # post-warmup Stage-M pass merely to enumerate runtime candidates.
        if not (
            self.uvm_init_weight_offload_enable
            or self.uvm_incremental_expert_offload_enable
        ):
            return []

        cold_pairs: list[tuple[int, int]] = []
        warm_pairs: list[tuple[int, int]] = []
        static_alias_owned = self._uvm_static_alias_owned_experts(layer_id)
        layer_warm_plan_ids = {
            expert_id
            for plan_layer_id, expert_id in self.uvm_init_weight_warm_plan_keys
            if plan_layer_id == layer_id
        }
        for global_expert_id, local_expert_id in all_local_pairs:
            if global_expert_id in (
                self._uvm_offloaded_global_expert_ids
                | getattr(self, "_uvm_remote_read_global_expert_ids", set())
                | static_alias_owned
            ):
                continue
            expert_class = self._uvm_init_weight_expert_class(
                layer_id=layer_id,
                global_expert_id=global_expert_id,
            )
            if expert_class == "cold":
                cold_pairs.append((global_expert_id, local_expert_id))
                continue
            if expert_class != "warm":
                continue
            if layer_warm_plan_ids:
                if global_expert_id in layer_warm_plan_ids:
                    warm_pairs.append((global_expert_id, local_expert_id))
                continue
            if (
                self.uvm_init_weight_warm_offload_enable
                and self.uvm_init_weight_complete_expert_plan
            ):
                warm_pairs.append((global_expert_id, local_expert_id))

        # Limits consume this list from the front, so warm experts cannot be
        # released until all selected cold experts have been considered.
        return sorted(cold_pairs) + sorted(warm_pairs)

    def has_remaining_uvm_incremental_experts(self, expert_class: str) -> bool:
        return self.uvm_next_incremental_expert(expert_class) is not None

    def uvm_next_incremental_expert(self, expert_class: str) -> int | None:
        """Return this layer's next global expert candidate."""
        layer_id = self._uvm_layer_id()
        if layer_id is None:
            return None
        pairs = self._uvm_stage_m_offload_expert_pairs(
            layer_id,
            self._uvm_all_local_expert_pairs(),
        )
        for global_expert_id, _ in pairs:
            if (
                self._uvm_init_weight_expert_class(
                    layer_id=layer_id,
                    global_expert_id=global_expert_id,
                )
                != expert_class
            ):
                continue
            return int(global_expert_id)
        return None

    def run_uvm_incremental_weight_offload(
        self,
        *,
        allow_warm: bool,
        selected_global_expert_id: int | None = None,
    ) -> dict[str, object]:
        if not self.uvm_incremental_expert_offload_enable:
            return {"ok": True, "reason": "disabled", "released_experts": []}
        remote_read = self._uvm_stage_m_uses_remote_read()
        tracked_experts = (
            self._uvm_remote_read_global_expert_ids
            if remote_read
            else self._uvm_offloaded_global_expert_ids
        )
        if selected_global_expert_id is None:
            selected_global_expert_id = self.uvm_next_incremental_expert("cold")
            if selected_global_expert_id is None and allow_warm:
                selected_global_expert_id = self.uvm_next_incremental_expert("warm")
        if selected_global_expert_id is None:
            return {
                "ok": True,
                "reason": "no_candidates",
                "released_experts": [],
            }
        layer_id = self._uvm_layer_id()
        if (
            layer_id is not None
            and int(selected_global_expert_id)
            in self._uvm_static_alias_owned_experts(layer_id)
        ):
            return {
                "ok": False,
                "reason": "expert_owned_by_static_alias_plan",
                "layer_id": int(layer_id),
                "selected_global_expert_id": int(selected_global_expert_id),
                "released_experts": [],
            }
        before = set(tracked_experts)
        self._uvm_incremental_offload_in_progress = True
        self._uvm_incremental_selected_global_expert_id = int(selected_global_expert_id)
        self._uvm_init_weight_offload_done = False
        try:
            self.run_uvm_init_weight_offload()
        finally:
            self._uvm_incremental_offload_in_progress = False
            self._uvm_incremental_selected_global_expert_id = None
        released = sorted(set(tracked_experts) - before)
        incremental = getattr(
            self,
            "_uvm_incremental_offloaded_global_expert_ids",
            None,
        )
        if not isinstance(incremental, set):
            incremental = set()
            self._uvm_incremental_offloaded_global_expert_ids = incremental
        incremental.update(int(expert_id) for expert_id in released)
        return {
            "ok": True,
            "reason": "released" if released else "no_capacity_or_candidates",
            "layer_id": self._uvm_layer_id(),
            "released_experts": released,
            "released_count": len(released),
            "allow_warm": bool(allow_warm),
            "selected_global_expert_id": selected_global_expert_id,
            "access_mode": "remote_read" if remote_read else "restore",
        }

    def uvm_next_incremental_restore_expert(self) -> int | None:
        """Choose the hottest expert released by a prior KV-pressure wave."""
        incremental = set(
            getattr(
                self,
                "_uvm_incremental_offloaded_global_expert_ids",
                set(),
            )
        )
        tracked = (
            self._uvm_remote_read_global_expert_ids
            if self._uvm_stage_m_uses_remote_read()
            else self._uvm_offloaded_global_expert_ids
        )
        candidates = sorted(incremental & set(tracked))
        if not candidates:
            return None
        counts_tensor = getattr(self, "_uvm_dynamic_expert_heat_counts", None)
        if counts_tensor is None:
            # Deterministic LIFO-like restoration without requiring heat
            # collection. Heat-enabled deployments take the branch below.
            return int(candidates[-1])
        try:
            counts = counts_tensor.detach()
            return int(
                max(
                    candidates,
                    key=lambda expert_id: (
                        int(counts[int(expert_id)].item()),
                        -int(expert_id),
                    ),
                )
            )
        except Exception:
            logger.debug(
                "Failed to rank incremental expert restoration by heat",
                exc_info=True,
            )
            return int(candidates[-1])

    def run_uvm_incremental_weight_restore(
        self,
        *,
        selected_global_expert_id: int,
    ) -> dict[str, object]:
        """Undo one runtime offload while preserving the static cold plan."""
        expert_id = int(selected_global_expert_id)
        incremental = getattr(
            self,
            "_uvm_incremental_offloaded_global_expert_ids",
            None,
        )
        tracked = (
            self._uvm_remote_read_global_expert_ids
            if self._uvm_stage_m_uses_remote_read()
            else self._uvm_offloaded_global_expert_ids
        )
        if not isinstance(incremental, set) or expert_id not in incremental:
            return {
                "ok": False,
                "reason": "expert_not_incrementally_offloaded",
                "selected_global_expert_id": expert_id,
                # Rollback calls name the complete intended wave. A layer
                # whose release never committed is already in the desired
                # local state and can be treated as an idempotent success by
                # the worker-level transaction.
                "already_local": expert_id not in tracked,
            }
        layer_id = self._uvm_layer_id()
        if layer_id is None:
            return {
                "ok": False,
                "reason": "layer_id_unavailable",
                "selected_global_expert_id": expert_id,
            }
        if expert_id not in tracked:
            return {
                "ok": False,
                "reason": "incremental_expert_tracking_mismatch",
                "selected_global_expert_id": expert_id,
            }
        if (int(layer_id), expert_id) not in self._uvm_expert_records:
            return {
                "ok": False,
                "reason": "expert_record_missing",
                "selected_global_expert_id": expert_id,
            }
        device = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
        result = self._uvm_restore_expert_record(
            layer_id=int(layer_id),
            global_expert_id=expert_id,
            device=device,
        )
        if not bool(result.get("ok")):
            return {
                "ok": False,
                "reason": "expert_restore_failed",
                "selected_global_expert_id": expert_id,
                "restore_result": result,
            }
        tracked.remove(expert_id)
        incremental.remove(expert_id)
        return {
            "ok": True,
            "reason": "restored",
            "layer_id": int(layer_id),
            "restored_experts": [expert_id],
            "restored_count": 1,
            "selected_global_expert_id": expert_id,
            "restore_result": result,
        }

    def run_uvm_dynamic_remote_read_swaps(
        self,
        swaps: list[dict[str, object]],
    ) -> dict[str, object]:
        """Swap explicitly named GPU and remote-read expert records.

        Commit the complete demotion wave before restoring any promotion.
        Expert slices are smaller than a UVM VA block, so a later demotion can
        evict an already-promoted neighbour in the same parent block.  Making
        every H2D restore the final physical operation avoids freezing that
        stale CPU residency into the subsequent full-dual view.
        """
        if not self.uvm_dynamic_expert_swap_enable:
            return {"ok": False, "reason": "dynamic_expert_swap_disabled"}
        if not self._uvm_stage_m_uses_remote_read():
            return {"ok": False, "reason": "remote_read_mode_required"}
        layer_id = self._uvm_layer_id()
        if layer_id is None:
            return {"ok": False, "reason": "layer_id_unavailable"}

        device = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
        results: list[dict[str, object]] = []
        prepared: list[dict[str, object]] = []
        seen_promotions: set[int] = set()
        seen_demotions: set[int] = set()
        for swap_index, swap in enumerate(swaps):
            try:
                promote_expert = int(swap["promote_expert"])
                demote_expert = int(swap["demote_expert"])
            except (KeyError, TypeError, ValueError):
                return {
                    "ok": False,
                    "reason": "invalid_swap_record",
                    "swap_index": swap_index,
                    "results": results,
                }

            transaction_id = (
                f"layer{layer_id}:swap{swap_index}:"
                f"{promote_expert}->{demote_expert}:{time.time_ns()}"
            )
            if (
                promote_expert not in self._uvm_remote_read_global_expert_ids
                or demote_expert in self._uvm_remote_read_global_expert_ids
                or promote_expert == demote_expert
                or promote_expert in seen_promotions
                or demote_expert in seen_demotions
            ):
                return {
                    "ok": False,
                    "reason": "stale_or_invalid_swap",
                    "transaction_id": transaction_id,
                    "promote_expert": promote_expert,
                    "demote_expert": demote_expert,
                    "remote_experts": sorted(self._uvm_remote_read_global_expert_ids),
                    "results": results,
                }
            seen_promotions.add(promote_expert)
            seen_demotions.add(demote_expert)

            promote_record = self._uvm_expert_records.get(
                (int(layer_id), promote_expert)
            )
            demote_record = self._uvm_expert_records.get(
                (int(layer_id), demote_expert)
            )
            if promote_record is None or demote_record is None:
                return {
                    "ok": False,
                    "reason": "expert_record_missing",
                    "transaction_id": transaction_id,
                    "promote_expert": promote_expert,
                    "demote_expert": demote_expert,
                    "results": results,
                }

            promote_ranges: list[dict[str, int]] = []
            for region_id in promote_record.envelope_region_ids:
                parts = region_id.rsplit(":", 2)
                if len(parts) != 3:
                    continue
                try:
                    base = int(parts[-2], 16)
                    length = int(parts[-1], 16)
                except ValueError:
                    continue
                if base > 0 and length > 0:
                    promote_ranges.append({"base": base, "length": length})
            prepared.append(
                {
                    "transaction_id": transaction_id,
                    "promote_expert": promote_expert,
                    "demote_expert": demote_expert,
                    "promote_region_count": len(
                        promote_record.envelope_region_ids
                    ),
                    "demote_region_count": len(
                        demote_record.envelope_region_ids
                    ),
                    "promote_ranges": promote_ranges,
                }
            )

        for pair_result in prepared:
            demote_expert = int(pair_result["demote_expert"])
            demote_result = self._uvm_release_expert_record(
                layer_id=layer_id,
                global_expert_id=demote_expert,
                device=device,
            )
            pair_result["demote_result"] = demote_result
            if not bool(demote_result.get("ok")):
                results.append({**pair_result, "ok": False})
                return {
                    "ok": False,
                    "reason": "demote_release_failed",
                    "remote_experts": sorted(self._uvm_remote_read_global_expert_ids),
                    "results": results,
                }

            # Demotion frees capacity before promotion. A failed promotion leaves
            # both experts remote, which is conservative and recoverable.
            self._uvm_remote_read_global_expert_ids.add(demote_expert)

        for pair_result in prepared:
            promote_expert = int(pair_result["promote_expert"])
            demote_expert = int(pair_result["demote_expert"])
            promote_result = self._uvm_restore_expert_record(
                layer_id=layer_id,
                global_expert_id=promote_expert,
                device=device,
            )
            pair_result["promote_result"] = promote_result
            if not bool(promote_result.get("ok")):
                results.append({**pair_result, "ok": False})
                return {
                    "ok": False,
                    "reason": "promote_restore_failed",
                    "partial_demote_committed": True,
                    "remote_experts": sorted(self._uvm_remote_read_global_expert_ids),
                    "results": results,
                }

            self._uvm_remote_read_global_expert_ids.remove(promote_expert)
            incremental = getattr(
                self,
                "_uvm_incremental_offloaded_global_expert_ids",
                None,
            )
            if isinstance(incremental, set) and promote_expert in incremental:
                # The heat swap changes the identity occupying one runtime
                # KV-pressure offload slot. Transfer ownership so a later KV
                # shrink still restores exactly one expert for this layer.
                incremental.remove(promote_expert)
                incremental.add(demote_expert)
            results.append({**pair_result, "ok": True})

        return {
            "ok": True,
            "reason": "swapped" if results else "no_swaps",
            "layer_id": layer_id,
            "mutation_order": "all_demotions_then_all_promotions",
            "swapped_count": len(results),
            "remote_experts": sorted(self._uvm_remote_read_global_expert_ids),
            "results": results,
        }

    def _uvm_stage_m_uses_remote_read(self) -> bool:
        return (
            getattr(self, "uvm_expert_offload_access_mode", "restore")
            == "remote_read"
            and self.uvm_init_weight_offload_mode
            in _UVM_INIT_WEIGHT_DRIVER_RELEASE_MODES
        )

    def _refresh_uvm_init_weight_plan(self) -> None:
        if (
            not self._uvm_init_weight_offload_plan_loaded
            and self.uvm_init_weight_offload_plan_file
            and os.path.isfile(self.uvm_init_weight_offload_plan_file)
        ):
            self.uvm_init_weight_offload_plan_keys = _parse_plan_expert_keys(
                self.uvm_init_weight_offload_plan_file,
                "offload_plan",
            )
            self.uvm_init_weight_hot_plan_keys = _parse_plan_expert_keys(
                self.uvm_init_weight_offload_plan_file,
                "hot_plan",
            ) | (
                _parse_plan_expert_keys(
                    self.uvm_init_weight_offload_plan_file,
                    "prefetch_plan",
                )
                - self.uvm_init_weight_offload_plan_keys
            )
            self.uvm_init_weight_warm_plan_keys = _parse_plan_expert_keys(
                self.uvm_init_weight_offload_plan_file,
                "warm_plan",
            )
            self.uvm_init_weight_plan_policy = _uvm_weight_plan_policy(
                self.uvm_init_weight_offload_plan_file,
            )
            self._uvm_init_weight_offload_plan_loaded = True

    def _uvm_expert_weight_slices(
        self,
        local_expert_id: int,
    ) -> list[tuple[int, int, tuple[int, int]]]:
        """Return temporary tensor geometry used only during envelope compile."""
        slices: list[tuple[int, int, tuple[int, int]]] = []
        for attr_name in ("w13_weight", "w2_weight"):
            tensor = getattr(self, attr_name, None)
            if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                continue
            if tensor.ndim == 0 or local_expert_id >= int(tensor.shape[0]):
                continue
            total_bytes = int(tensor.numel()) * int(tensor.element_size())
            if int(tensor.shape[0]) <= 0 or total_bytes <= 0:
                continue
            slice_bytes = total_bytes // int(tensor.shape[0])
            if slice_bytes <= 0:
                continue
            tensor_base = int(tensor.data_ptr())
            tensor_end = tensor_base + total_bytes
            ptr = tensor_base + local_expert_id * slice_bytes
            slices.append((ptr, slice_bytes, (tensor_base, tensor_end)))
        return slices

    def _uvm_compile_expert_records(self, layer_id: int) -> None:
        records: dict[tuple[int, int], ExpertRecord] = {}
        envelope_owners: dict[str, tuple[int, int]] = {}
        geometry_by_region: dict[str, tuple[int, int]] = {}
        static_alias_owned = self._uvm_static_alias_owned_experts(layer_id)
        self._uvm_static_alias_owned_global_expert_ids = set(static_alias_owned)
        for global_expert_id, local_expert_id in self._uvm_all_local_expert_pairs():
            slices = self._uvm_expert_weight_slices(local_expert_id)
            compile_slices: list[tuple[int, int]] = []
            block_bytes = int(self.uvm_expert_envelope_bytes)
            if block_bytes <= 0:
                raise ValueError("uvm_expert_envelope_bytes must be positive")
            for ptr, size_bytes, (tensor_base, tensor_end) in slices:
                envelope_start = _align_down(ptr, block_bytes)
                envelope_end = _align_up(ptr + size_bytes, block_bytes)
                if envelope_start < tensor_base or envelope_end > tensor_end:
                    raise RuntimeError(
                        "Expert envelope exceeds tensor allocation: "
                        f"layer_id={layer_id}, global_expert_id={global_expert_id}, "
                        f"ptr=0x{ptr:x}, bytes={size_bytes}, "
                        f"envelope=0x{envelope_start:x}-0x{envelope_end:x}, "
                        f"tensor=0x{tensor_base:x}-0x{tensor_end:x}"
                    )
                compile_slices.append((ptr, size_bytes))
            record = compile_expert_envelopes(
                layer_id=layer_id,
                global_expert_id=global_expert_id,
                slices=compile_slices,
                envelope_bytes=block_bytes,
                existing_owners=envelope_owners,
            )
            # Static-alias finalize registered whole expert slices before the
            # runtime controller was initialized.  Their envelope geometry
            # overlaps those live driver regions, so retain logical records
            # for inventory purposes but leave physical region ownership with
            # the alias session.  Explicit runtime waves are fail-closed above
            # if they ever try to select one of these frozen base experts.
            if int(global_expert_id) not in static_alias_owned:
                for region_id in record.envelope_region_ids:
                    _, base_text, length_text = region_id.rsplit(":", 2)
                    geometry_by_region[region_id] = (
                        int(base_text, 16),
                        int(length_text, 16),
                    )
            # Registration records geometry, not physical residency.  Final
            # post-warmup placement is the first operation allowed to commit
            # survivor experts to GPU.
            record.aggregate_state = RegionState.UNKNOWN
            records[record.key] = record

        for region_id, (base, length) in geometry_by_region.items():
            try:
                register_uvm_region(region_id, base, length, state="unknown")
            except Exception as exc:
                raise RuntimeError(
                    "Failed to register explicit UVM Expert envelope: "
                    f"layer_id={layer_id}, region_id={region_id!r}, "
                    f"base=0x{base:x}, length={length}"
                ) from exc
        self._uvm_expert_records = records

    def _uvm_init_hot_expert_records(self) -> list[ExpertRecord]:
        """Return records surviving the complete initial cold-release pass."""
        if (
            getattr(self, "uvm_init_weight_offload_mode", "trace_only")
            not in _UVM_INIT_WEIGHT_DRIVER_RELEASE_MODES
        ):
            return []
        cold_expert_ids = set(
            getattr(self, "_uvm_offloaded_global_expert_ids", set())
        ) | set(getattr(self, "_uvm_remote_read_global_expert_ids", set()))
        static_alias_owned = set(
            getattr(self, "_uvm_static_alias_owned_global_expert_ids", set())
        )
        return [
            record
            for _, record in sorted(self._uvm_expert_records.items())
            if record.global_expert_id not in cold_expert_ids
            and record.global_expert_id not in static_alias_owned
        ]

    def get_uvm_init_hot_expert_region_ids(self) -> list[str]:
        """Collect every envelope owned by post-release survivor experts."""
        return list(
            dict.fromkeys(
                region_id
                for record in self._uvm_init_hot_expert_records()
                for region_id in record.envelope_region_ids
            )
        )

    def set_uvm_init_hot_expert_state(
        self,
        state: RegionState | str,
    ) -> int:
        """Update survivor aggregate state after the global placement phase."""
        target = normalize_region_state(state)
        records = self._uvm_init_hot_expert_records()
        for record in records:
            record.aggregate_state = target
        return len(records)

    def get_uvm_init_weight_placement_snapshot(self) -> dict[str, object]:
        """Return JSON-ready expert ownership for the optional runtime manifest."""
        layer_id = self._uvm_layer_id()
        remote_ids = set(self._uvm_remote_read_global_expert_ids)
        offloaded_ids = set(self._uvm_offloaded_global_expert_ids)
        incremental_ids = set(
            getattr(
                self,
                "_uvm_incremental_offloaded_global_expert_ids",
                set(),
            )
        )
        experts: list[dict[str, object]] = []
        region_owners: list[dict[str, object]] = []
        for (_, expert_id), record in sorted(self._uvm_expert_records.items()):
            placement = "hot"
            if int(expert_id) in remote_ids:
                placement = "remote"
            elif int(expert_id) in offloaded_ids:
                placement = "offloaded"
            experts.append(
                {
                    "layer_id": int(record.layer_id),
                    "expert_id": int(record.global_expert_id),
                    "placement": placement,
                    "incremental_runtime_offload": bool(
                        int(expert_id) in incremental_ids
                    ),
                    "app_state": record.aggregate_state.value,
                    "region_count": len(record.envelope_region_ids),
                }
            )
            region_owners.extend(
                {
                    "region_id": region_id,
                    "layer_id": int(record.layer_id),
                    "expert_id": int(record.global_expert_id),
                    "placement": placement,
                }
                for region_id in record.envelope_region_ids
            )
        release_audits = getattr(
            self,
            "_uvm_init_weight_release_audits",
            {},
        )
        return {
            "layer_name": self.layer_name,
            "layer_id": layer_id,
            "access_mode": self.uvm_expert_offload_access_mode,
            "plan_file": self.uvm_init_weight_offload_plan_file,
            "plan_policy": self.uvm_init_weight_plan_policy,
            "incremental_offloaded_experts": sorted(incremental_ids),
            "experts": experts,
            "region_owners": region_owners,
            "release_audits": [
                dict(release_audits[region_id])
                for region_id in sorted(release_audits)
            ],
        }

    def _uvm_record_expert_state(
        self,
        layer_id: int,
        global_expert_id: int,
        state: RegionState,
    ) -> None:
        record = self._uvm_expert_records.get((int(layer_id), int(global_expert_id)))
        if record is not None:
            record.aggregate_state = state

    def uvm_prefill_oracle_plan_expert_ids(self) -> list[int]:
        """Return this layer's still-CPU experts selected by the fixed plan."""
        layer_id = self._uvm_layer_id()
        if layer_id is None:
            return []
        tracked = (
            self._uvm_remote_read_global_expert_ids
            if self.uvm_expert_offload_access_mode == "remote_read"
            else self._uvm_offloaded_global_expert_ids
        )
        planned = {
            int(expert_id)
            for plan_layer_id, expert_id in self.uvm_init_weight_offload_plan_keys
            if int(plan_layer_id) == int(layer_id)
        }
        return sorted(planned & tracked)

    def prepare_uvm_prefill_oracle_experts(
        self,
        expert_ids: Iterable[int],
        *,
        device: int,
    ) -> dict[str, object]:
        """Submit fixed-plan expert prefetch on the current CUDA stream."""
        layer_id = self._uvm_layer_id()
        requested = sorted({int(expert_id) for expert_id in expert_ids})
        if layer_id is None:
            return {"ok": False, "reason": "layer_id_unavailable"}
        if not requested:
            return {
                "ok": True,
                "reason": "no_experts",
                "layer_id": int(layer_id),
                "expert_ids": [],
                "region_ids": [],
                "requested_bytes": 0,
            }

        records: list[ExpertRecord] = []
        missing: list[int] = []
        tracked = (
            self._uvm_remote_read_global_expert_ids
            if self.uvm_expert_offload_access_mode == "remote_read"
            else self._uvm_offloaded_global_expert_ids
        )
        for expert_id in requested:
            record = self._uvm_expert_records.get((int(layer_id), expert_id))
            if record is None or expert_id not in tracked:
                missing.append(expert_id)
            else:
                records.append(record)
        if missing:
            return {
                "ok": False,
                "reason": "expert_not_offloaded_or_unregistered",
                "layer_id": int(layer_id),
                "missing_expert_ids": missing,
            }

        region_ids = list(
            dict.fromkeys(
                region_id
                for record in records
                for region_id in record.envelope_region_ids
            )
        )
        requested_bytes = 0
        for region_id in region_ids:
            try:
                requested_bytes += int(region_id.rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                return {
                    "ok": False,
                    "reason": "invalid_region_id",
                    "layer_id": int(layer_id),
                    "region_id": region_id,
                }

        result = prepare_uvm_regions_for_gpu_placement(
            region_ids,
            device=int(device),
            executor=self.uvm_weight_executor,
        )
        if bool(result.get("ok")):
            for record in records:
                record.aggregate_state = RegionState.PREFETCHING_GPU
        return {
            **result,
            "layer_id": int(layer_id),
            "expert_ids": requested,
            "region_ids": region_ids,
            "region_count": len(region_ids),
            "requested_bytes": int(requested_bytes),
            "initial_access_mode": self.uvm_expert_offload_access_mode,
        }

    def commit_uvm_prefill_oracle_experts(
        self,
        expert_ids: Iterable[int],
        region_ids: Iterable[str],
    ) -> dict[str, object]:
        """Commit fixed-plan placement after its CUDA ready event completed."""
        layer_id = self._uvm_layer_id()
        requested_experts = sorted({int(value) for value in expert_ids})
        requested_regions = list(dict.fromkeys(str(value) for value in region_ids))
        if layer_id is None:
            return {"ok": False, "reason": "layer_id_unavailable"}
        result = commit_uvm_regions_gpu_placement(
            requested_regions,
            synchronized=True,
        )
        state = RegionState.GPU if bool(result.get("ok")) else RegionState.UNKNOWN
        tracked = (
            self._uvm_remote_read_global_expert_ids
            if self.uvm_expert_offload_access_mode == "remote_read"
            else self._uvm_offloaded_global_expert_ids
        )
        for expert_id in requested_experts:
            self._uvm_record_expert_state(layer_id, expert_id, state)
            if state == RegionState.GPU:
                tracked.discard(expert_id)
            else:
                tracked.add(expert_id)
        return {
            **result,
            "layer_id": int(layer_id),
            "expert_ids": requested_experts,
            "region_ids": requested_regions,
        }

    def _uvm_release_expert_record(
        self,
        *,
        layer_id: int,
        global_expert_id: int,
        device: int,
    ) -> dict[str, object]:
        record = self._uvm_expert_records[(int(layer_id), int(global_expert_id))]
        audit_enabled = bool(
            os.environ.get(_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST_ENV, "").strip()
        )
        release_kwargs: dict[str, object] = {
            "device": device,
            "remote_read": self._uvm_stage_m_uses_remote_read(),
            "immutable_cpu_shadow": _env_flag_enabled(
                "VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW"
            ),
        }
        if audit_enabled:
            release_kwargs["include_audit"] = True
        result = release_uvm_regions(
            record.envelope_region_ids,
            **release_kwargs,
        )
        if audit_enabled:
            raw_audits = result.get("audits")
            expected_regions = set(record.envelope_region_ids)
            if not isinstance(raw_audits, dict) or set(raw_audits) != expected_regions:
                raise RuntimeError(
                    "Release audit is incomplete: "
                    f"layer_id={layer_id}, expert_id={global_expert_id}"
                )
            audit_store = getattr(
                self,
                "_uvm_init_weight_release_audits",
                None,
            )
            if not isinstance(audit_store, dict):
                audit_store = {}
                self._uvm_init_weight_release_audits = audit_store
            for region_id in sorted(expected_regions):
                raw_audit = raw_audits[region_id]
                if not isinstance(raw_audit, dict) or raw_audit.get("ok") is not True:
                    raise RuntimeError(
                        "Release audit did not prove success: "
                        f"layer_id={layer_id}, expert_id={global_expert_id}, "
                        f"region_id={region_id!r}, audit={raw_audit!r}"
                    )
                audit_store[region_id] = {
                    **raw_audit,
                    "region_id": region_id,
                    "layer_id": int(layer_id),
                    "expert_id": int(global_expert_id),
                }
        state = RegionState.CPU if bool(result.get("ok")) else RegionState.UNKNOWN
        self._uvm_record_expert_state(layer_id, global_expert_id, state)
        return result

    def _uvm_restore_expert_record(
        self,
        *,
        layer_id: int,
        global_expert_id: int,
        device: int,
    ) -> dict[str, object]:
        record = self._uvm_expert_records[(int(layer_id), int(global_expert_id))]
        if self.uvm_expert_offload_access_mode == "slot_restore":
            validate_uvm_slot_direct_session_contract(
                access_mode=self.uvm_expert_offload_access_mode,
                device=int(device),
            )
            result = restore_uvm_regions_to_slots(
                record.envelope_region_ids,
                layer_id=layer_id,
                expert_id=global_expert_id,
                device=device,
            )
        else:
            result = restore_uvm_regions(
                record.envelope_region_ids,
                device=device,
                executor=self.uvm_weight_executor,
            )
        if bool(result.get("ok")):
            # A slot mapping is logically GPU-accessible, but its ordinary UVM
            # backing remains CPU-resident. Keep the aggregate state honest so
            # cache eviction never needs a cross-layer state repair callback.
            state = (
                RegionState.CPU
                if self.uvm_expert_offload_access_mode == "slot_restore"
                else RegionState.GPU
            )
        else:
            state = RegionState.UNKNOWN
        self._uvm_record_expert_state(layer_id, global_expert_id, state)
        return result

    def _uvm_restore_expert_records_to_slots(
        self,
        *,
        layer_id: int,
        global_expert_ids: list[int],
        device: int,
    ) -> dict[str, object]:
        """Submit one complete fused-MoE working set as a slot transaction."""
        contract = validate_uvm_slot_direct_session_contract(
            access_mode=self.uvm_expert_offload_access_mode,
            device=int(device),
        )
        if bool(contract["registered_plan"]):
            result = restore_uvm_registered_layer_plan(
                layer_id=int(layer_id),
                expert_ids=global_expert_ids,
                device=int(device),
            )
            state = (
                RegionState.CPU if bool(result.get("ok")) else RegionState.UNKNOWN
            )
            for global_expert_id in global_expert_ids:
                self._uvm_record_expert_state(layer_id, global_expert_id, state)
            return result

        groups = [
            (
                int(global_expert_id),
                self._uvm_expert_records[
                    (int(layer_id), int(global_expert_id))
                ].envelope_region_ids,
            )
            for global_expert_id in global_expert_ids
        ]
        result = restore_uvm_expert_groups_to_slots(
            groups,
            layer_id=int(layer_id),
            device=int(device),
        )
        state = RegionState.CPU if bool(result.get("ok")) else RegionState.UNKNOWN
        for global_expert_id in global_expert_ids:
            self._uvm_record_expert_state(layer_id, global_expert_id, state)
        return result

    def get_uvm_slot_init_expert_groups(
        self,
    ) -> tuple[int, list[tuple[int, list[str]]]] | None:
        """Return the immutable cold-layer geometry for bank initialization."""
        if self.uvm_expert_offload_access_mode != "slot_restore":
            return None
        layer_id = self._uvm_layer_id()
        if layer_id is None or not self._uvm_offloaded_global_expert_ids:
            return None
        groups = [
            (
                int(global_expert_id),
                list(
                    self._uvm_expert_records[
                        (int(layer_id), int(global_expert_id))
                    ].envelope_region_ids
                ),
            )
            for global_expert_id in sorted(
                self._uvm_offloaded_global_expert_ids
            )
        ]
        return int(layer_id), groups

    def _uvm_restore_active_experts(self, topk_ids: torch.Tensor) -> None:
        if (
            self.uvm_expert_offload_access_mode not in ("restore", "slot_restore")
            or not self._uvm_offloaded_global_expert_ids
        ):
            return
        layer_id = self._uvm_layer_id()
        if layer_id is None:
            return
        if getattr(self, "uvm_slot_formal_restore_all_cold", False):
            active_ids = set(self._uvm_offloaded_global_expert_ids)
        else:
            active_ids = {
                global_expert_id
                for global_expert_id, _ in self._uvm_active_expert_pairs(topk_ids)
                if global_expert_id in self._uvm_offloaded_global_expert_ids
            }
        if not active_ids:
            return
        device = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
        kernel_stream_fence = (
            self.uvm_expert_offload_access_mode == "slot_restore"
            and uvm_slot_cuda_fence_enabled()
            and _env_flag_enabled("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN")
            and _env_flag_enabled("VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT")
        )
        if (
            self.uvm_expert_offload_access_mode == "slot_restore"
            and torch.cuda.is_available()
            and not kernel_stream_fence
        ):
            # Neither mode 20 nor registered-plan execution can observe
            # ordinary CUDA-kernel lifetime. This device boundary applies to
            # every slot transition (not only an eviction fallback), and proves
            # that the old layer stopped using the bank before its PTE teardown.
            torch.cuda.synchronize(device)
        protected_groups = (
            [(int(layer_id), expert_id) for expert_id in sorted(active_ids)]
            if self.uvm_expert_offload_access_mode == "slot_restore"
            else []
        )
        with protect_uvm_slot_direct_groups(protected_groups):
            sorted_active_ids = sorted(active_ids)
            if self.uvm_expert_offload_access_mode == "slot_restore":
                result = self._uvm_restore_expert_records_to_slots(
                    layer_id=layer_id,
                    global_expert_ids=sorted_active_ids,
                    device=device,
                )
                if not bool(result.get("ok")):
                    raise RuntimeError(
                        "Failed to restore active expert regions: "
                        f"layer_id={layer_id}, expert_ids={sorted_active_ids}, "
                        f"result={result!r}"
                    )
            else:
                for global_expert_id in sorted_active_ids:
                    result = self._uvm_restore_expert_record(
                        layer_id=layer_id,
                        global_expert_id=global_expert_id,
                        device=device,
                    )
                    if not bool(result.get("ok")):
                        raise RuntimeError(
                            "Failed to restore active expert regions: "
                            f"layer_id={layer_id}, expert_id={global_expert_id}, "
                            f"result={result!r}"
                        )
                    self._uvm_offloaded_global_expert_ids.remove(global_expert_id)
        if torch.cuda.is_available() and not kernel_stream_fence:
            torch.cuda.synchronize(device)

    def run_uvm_init_weight_offload(self) -> None:
        """Register all local experts, then offload the caller-selected set."""
        incremental_transaction = bool(
            self.uvm_incremental_expert_offload_enable
            and self._uvm_incremental_offload_in_progress
            and self._uvm_incremental_selected_global_expert_id is not None
        )
        # INIT_WEIGHT_OFFLOAD_ENABLE owns only the one-shot Stage-M placement.
        # An explicit runtime wave has its own enable flag and transaction
        # marker, so it can safely operate after static-alias finalize without
        # re-running the frozen base plan.  A normal warmup/finalize call with
        # INIT_WEIGHT_OFFLOAD_ENABLE=0 remains a strict no-op.
        if not self.uvm_init_weight_offload_enable and not incremental_transaction:
            return
        if not getattr(
            self,
            "uvm_expert_envelope_enable",
            False,
        ):
            raise RuntimeError(
                "Stage-M expert offload requires envelope registration: set "
                "VLLM_UVM_EXPERT_ENVELOPE_ENABLE=1"
            )
        if self._uvm_init_weight_offload_done:
            return
        self._uvm_init_weight_offload_done = True
        self._refresh_uvm_init_weight_plan()

        try:
            layer_id = self._uvm_layer_id()
            if layer_id is None:
                return
            all_local_pairs = self._uvm_all_local_expert_pairs()
            if not getattr(self, "_uvm_expert_records", None):
                self._uvm_compile_expert_records(layer_id)
            if self.uvm_incremental_expert_offload_enable:
                selected_expert = self._uvm_incremental_selected_global_expert_id
                if (
                    self._uvm_incremental_offload_in_progress
                    and selected_expert is not None
                ):
                    expert_pairs = [
                        pair for pair in all_local_pairs if pair[0] == selected_expert
                    ]
                else:
                    expert_pairs = []
            else:
                expert_pairs = self._uvm_stage_m_offload_expert_pairs(
                    layer_id,
                    all_local_pairs,
                )

            if self.uvm_init_weight_offload_mode == "trace_only":
                return

            device = (
                int(torch.cuda.current_device())
                if torch.cuda.is_available()
                else 0
            )
            for global_expert_id, _ in expert_pairs:
                result = self._uvm_release_expert_record(
                    layer_id=layer_id,
                    global_expert_id=global_expert_id,
                    device=device,
                )
                if not bool(result.get("ok")):
                    raise RuntimeError(
                        "Failed to release expert regions: "
                        f"layer_id={layer_id}, expert_id={global_expert_id}, "
                        f"result={result!r}"
                    )
                if self._uvm_stage_m_uses_remote_read():
                    self._uvm_remote_read_global_expert_ids.add(
                        int(global_expert_id)
                    )
                else:
                    self._uvm_offloaded_global_expert_ids.add(
                        int(global_expert_id)
                    )
            return

        except Exception:
            self._uvm_init_weight_offload_done = False
            logger.warning("Failed to run Stage M init weight offload", exc_info=True)
            raise

    def must_reduce_shared_expert_outputs(self) -> bool:
        """
        The shared_experts are typically computed using the RowParallelLinear
        layer. The result of this function is typically used as
        the reduce_results argument to the module.
        When just tensor-parallel is used, it is not required to reduce
        the shared_experts results immediately. Instead we reduce at the
        once at the end of the MoE op. (Refer to DeepSeekV2MoE module)
        With EP and all2all kernels - this is no longer viable as all
        GPU ranks in DP, produce the complete set of hidden_states.
        Therefore it is required that we reduce the shared_experts output
        early.
        """
        assert self.quant_method is not None
        return (
            isinstance(self.quant_method, FusedMoEModularMethod)
            and self.quant_method.fused_experts.output_is_reduced()
        )

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        """
        Some combine kernels reduce across GPU ranks by default.
        """
        if self.must_reduce_shared_expert_outputs():
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        og_hidden_states = hidden_states.shape[-1]
        if self.hidden_size != og_hidden_states:
            hidden_states = F.pad(
                hidden_states,
                (0, self.hidden_size - og_hidden_states),
                mode="constant",
                value=0.0,
            )

        def reduce_output(states: torch.Tensor) -> torch.Tensor:
            # Slice before all_reduce to enable possible fusion
            if self.hidden_size != og_hidden_states:
                states = states[..., :og_hidden_states]

            if (
                not self.is_sequence_parallel
                and not self.use_dp_chunking
                and self.reduce_results
                and (self.tp_size > 1 or self.ep_size > 1)
            ):
                states = self.maybe_all_reduce_tensor_model_parallel(states)
            return states

        if self.shared_experts is None:
            if current_platform.is_tpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                fused_output = self.forward_impl(hidden_states, router_logits)
                assert not isinstance(fused_output, tuple)
            else:
                fused_output = torch.ops.vllm.moe_forward(
                    hidden_states, router_logits, self.layer_name
                )
            if self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(fused_output, tuple)
                fused_output, zero_expert_result = fused_output
                result = (
                    reduce_output(fused_output)
                    + zero_expert_result[..., :og_hidden_states]
                )
            else:
                result = reduce_output(fused_output)
            return result
        else:
            if current_platform.is_tpu():
                # TODO: Once the OOM issue for the TPU backend is resolved, we
                # will switch to using the moe_forward custom op.
                shared_output, fused_output = self.forward_impl(
                    hidden_states, router_logits
                )
            else:
                shared_output, fused_output = torch.ops.vllm.moe_forward_shared(
                    hidden_states, router_logits, self.layer_name
                )
            result = (
                reduce_output(shared_output),
                reduce_output(fused_output),
            )
            return result

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        with uvm_enabled_allocation_phase("enabled:moe"):
            return self.forward_native(hidden_states, router_logits)

    def forward_impl_chunked(
        self,
        full_hidden_states: torch.Tensor,
        full_router_logits: torch.Tensor,
        has_separate_shared_experts: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.batched_hidden_states is not None
        assert self.batched_router_logits is not None
        assert self.batched_hidden_states.dtype == full_hidden_states.dtype
        assert self.batched_router_logits.dtype == full_router_logits.dtype
        # Check size compatibility.
        assert self.batched_hidden_states.size(-1) == full_hidden_states.size(-1)
        assert self.batched_router_logits.size(-1) == full_router_logits.size(-1)

        full_fused_final_hidden_states = torch.empty_like(full_hidden_states)
        if self.shared_experts is not None:
            full_shared_final_hidden_states = torch.empty_like(full_hidden_states)

        def process_chunk(chunk_start, chunk_end, skip_result_store=False):
            chunk_size = chunk_end - chunk_start
            hidden_states = full_hidden_states[chunk_start:chunk_end, :]
            router_logits = full_router_logits[chunk_start:chunk_end, :]

            assert self.batched_hidden_states is not None
            assert self.batched_router_logits is not None
            # This is only true when DBO has been enabled in the config.
            # Both tensors will have an outer dimension for the ubatch id
            if self.batched_hidden_states.dim() == 3:
                assert self.batched_router_logits.dim() == 3
                batch_buffer_idx = dbo_current_ubatch_id()
                batched_hidden_states = self.batched_hidden_states[batch_buffer_idx, :]
                batched_router_logits = self.batched_router_logits[batch_buffer_idx, :]
            else:
                batched_hidden_states = self.batched_hidden_states
                batched_router_logits = self.batched_router_logits

            assert (
                batched_hidden_states.size(0)  # type: ignore
                >= chunk_size
            )
            assert (
                batched_router_logits.size(0)  # type: ignore
                >= chunk_size
            )
            staged_hidden_states = batched_hidden_states[:chunk_size, :]  # type: ignore
            staged_router_logits = batched_router_logits[:chunk_size, :]  # type: ignore
            staged_hidden_states.copy_(hidden_states, non_blocking=True)
            staged_router_logits.copy_(router_logits, non_blocking=True)

            # Matrix multiply.
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=staged_hidden_states,
                router_logits=staged_router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map
                if not self.rocm_aiter_fmoe_enabled
                else self.expert_mask,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias,
                activation=self.activation,
                enable_eplb=self.enable_eplb,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
            )
            if has_separate_shared_experts:
                assert not isinstance(final_hidden_states, tuple)
                assert self.shared_experts is not None

                shared_output = self.shared_experts(staged_hidden_states)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )

            if self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, tuple)
                assert self.shared_experts is None
                final_hidden_states, zero_expert_result = final_hidden_states
                if zero_expert_result is not None:
                    final_hidden_states += zero_expert_result

            if not skip_result_store:
                if self.shared_experts is None:
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states, non_blocking=True
                    )
                else:
                    full_shared_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[0], non_blocking=True
                    )
                    full_fused_final_hidden_states[chunk_start:chunk_end, :].copy_(
                        final_hidden_states[1], non_blocking=True
                    )

        ctx = get_forward_context()
        # flashinfer_cutlass_kernels can handle: optional DP + TP/EP
        max_tokens_across_dispatchers = ctx.dp_metadata.max_tokens_across_dp_cpu
        moe_dp_chunk_size_per_rank = self.moe_config.max_num_tokens

        # If the input to the MoE is sequence parallel then divide by sp_size
        # to find the maximum number of tokens for any individual dispatcher.
        if self.is_sequence_parallel:
            max_tokens_across_dispatchers = cdiv(
                max_tokens_across_dispatchers, self.sp_size
            )

        num_tokens = full_hidden_states.size(0)
        for chunk_idx, chunk_start_ in enumerate(
            range(0, max_tokens_across_dispatchers, moe_dp_chunk_size_per_rank)
        ):
            chunk_start = chunk_start_
            chunk_end = min(
                chunk_start + moe_dp_chunk_size_per_rank, max_tokens_across_dispatchers
            )
            # clamp start and end
            chunk_start = min(chunk_start, num_tokens - 1)
            chunk_end = min(chunk_end, num_tokens)
            with ctx.dp_metadata.chunked_sizes(
                self.sp_size, moe_dp_chunk_size_per_rank, chunk_idx
            ):
                process_chunk(
                    chunk_start, chunk_end, skip_result_store=chunk_start_ >= num_tokens
                )

        if self.shared_experts is None:
            result = full_fused_final_hidden_states
        else:
            result = (full_shared_final_hidden_states, full_fused_final_hidden_states)
        return result

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.quant_method is not None

        self.ensure_moe_quant_config_init()
        self.ensure_dp_chunking_init()

        has_separate_shared_experts = (
            not isinstance(self.quant_method, FusedMoEModularMethod)
            and self.shared_experts is not None
        )

        use_chunked_impl = self.use_dp_chunking

        use_shared_experts_stream, hidden_states_clone = (
            self._maybe_setup_shared_experts_stream(
                hidden_states, has_separate_shared_experts, use_chunked_impl
            )
        )

        # If router/gate provided, then apply it here.
        # (Note: This code runs only when "overlapped mode" is on to allow
        #        parallel execution of shared experts with the FusedMoE via
        #        separate cuda stream)
        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)

        if use_chunked_impl:
            return self.forward_impl_chunked(
                hidden_states, router_logits, has_separate_shared_experts
            )

        do_naive_dispatch_combine: bool = self.dp_size > 1 and not isinstance(
            self.quant_method, FusedMoEModularMethod
        )

        ctx = get_forward_context()
        sp_ctx = (
            ctx.dp_metadata.sp_local_sizes(self.sp_size)
            if ctx.dp_metadata
            else nullcontext()
        )

        with sp_ctx:
            if do_naive_dispatch_combine:
                hidden_states_combined, router_logits = get_ep_group().dispatch(
                    hidden_states, router_logits, self.is_sequence_parallel
                )
            # Run shared experts before matrix multiply.
            # because matrix multiply maybe modify the hidden_states.
            if has_separate_shared_experts and not use_shared_experts_stream:
                assert self.shared_experts is not None
                shared_output = self.shared_experts(hidden_states)

            # NOTE: Similar with DP, PCP also needs dispatch and combine. For
            # simplicity, AgRsAll2All was added separately for PCP here. Maybe
            # we should modify All2AllManager abstract to better support PCP.
            if self.pcp_size > 1:
                hidden_states = get_pcp_group().all_gather(
                    hidden_states,
                    dim=0,
                )
                router_logits = get_pcp_group().all_gather(
                    router_logits,
                    dim=0,
                )

            # Matrix multiply.
            final_hidden_states = self.quant_method.apply(
                layer=self,
                x=hidden_states_combined
                if do_naive_dispatch_combine
                else hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                renormalize=self.renormalize,
                use_grouped_topk=self.use_grouped_topk,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map
                if not self.rocm_aiter_fmoe_enabled
                else self.expert_mask,
                topk_group=self.topk_group,
                num_expert_group=self.num_expert_group,
                custom_routing_function=self.custom_routing_function,
                scoring_func=self.scoring_func,
                routed_scaling_factor=self.routed_scaling_factor,
                e_score_correction_bias=self.e_score_correction_bias,
                activation=self.activation,
                apply_router_weight_on_input=self.apply_router_weight_on_input,
                enable_eplb=self.enable_eplb,
                expert_load_view=self.expert_load_view,
                logical_to_physical_map=self.logical_to_physical_map,
                logical_replica_count=self.logical_replica_count,
            )
            if has_separate_shared_experts:
                assert self.shared_experts is not None

                if use_shared_experts_stream:
                    # Run shared experts in parallel on a separate stream
                    # NOTE: We start the separate stream here and mark the
                    # sync end point immediately after it is done. This is
                    # important to avoid excessive stream allocations by the cuda
                    # graph replay later.
                    with torch.cuda.stream(self.shared_experts_stream):
                        # Note that hidden_states clone() is necessary here to avoid
                        # conflict with the main stream
                        shared_output = self.shared_experts(hidden_states_clone)
                    current_stream().wait_stream(self.shared_experts_stream)

                final_hidden_states = (
                    shared_output,
                    final_hidden_states,
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, tuple)
                final_hidden_states, zero_expert_result = final_hidden_states

            def combine_output(states: torch.Tensor) -> torch.Tensor:
                if do_naive_dispatch_combine:
                    states = get_ep_group().combine(states, self.is_sequence_parallel)

                if self.pcp_size > 1:
                    states = get_pcp_group().reduce_scatter(
                        states,
                        dim=0,
                    )

                return states

            if self.shared_experts is not None:
                result = (
                    final_hidden_states[0],
                    combine_output(final_hidden_states[1]),
                )
            elif self.zero_expert_num is not None and self.zero_expert_num > 0:
                assert isinstance(final_hidden_states, torch.Tensor)
                result = (combine_output(final_hidden_states), zero_expert_result)
            else:
                result = combine_output(final_hidden_states)
            return result

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        num_redundant_experts: int = 0,
    ) -> list[tuple[str, str, int, str]]:
        num_physical_experts = num_experts + num_redundant_experts

        # In the returned mapping:
        # - `expert_id` is the physical expert id
        # - `weight_name` contains the weight name of the logical expert
        # So that we should map the expert id to logical in `weight_name`
        physical_to_logical_map = (
            EplbState.build_initial_global_physical_to_logical_map(
                num_experts, num_redundant_experts
            )
        )

        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                "experts.w13_"
                if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                else "experts.w2_",
                f"experts.{physical_to_logical_map[expert_id]}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_physical_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def extra_repr(self) -> str:
        s = (
            f"global_num_experts={self.global_num_experts}, "
            f"local_num_experts={self.local_num_experts}, "
            f"top_k={self.top_k}, "
            f"intermediate_size_per_partition={self.intermediate_size_per_partition}, "  # noqa: E501
            f"tp_size={self.tp_size},\n"
            f"ep_size={self.ep_size}, "
            f"reduce_results={self.reduce_results}, "
            f"renormalize={self.renormalize}, "
            f"use_grouped_topk={self.use_grouped_topk}"
        )

        if self.use_grouped_topk:
            s += f", num_expert_group={self.num_expert_group}, topk_group={self.topk_group}"  # noqa: E501

        s += f", scoring_func='{self.scoring_func}', activation='{self.activation}'"  # noqa: E501

        return s


def moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.shared_experts is None
    with uvm_enabled_allocation_phase("enabled:moe"):
        return self.forward_impl(hidden_states, router_logits)


def moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="moe_forward",
    op_func=moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def moe_forward_shared(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    assert self.shared_experts is not None
    with uvm_enabled_allocation_phase("enabled:moe"):
        return self.forward_impl(hidden_states, router_logits)


def moe_forward_shared_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    shared_out = torch.empty_like(hidden_states)
    fused_out = torch.empty_like(hidden_states)
    return shared_out, fused_out


direct_register_custom_op(
    op_name="moe_forward_shared",
    op_func=moe_forward_shared,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_shared_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)

# Mark the FusedMoE weight_loader as supporting MoE-specific parameters
# to avoid expensive runtime reflection in model loading code
FusedMoE.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
