# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import hashlib
import itertools
import json
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
from functools import reduce
from itertools import product
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from tqdm import tqdm

import vllm.envs as envs
from vllm.attention import Attention, AttentionType
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionMetadata,
    MultipleOf,
)
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (
    CompilationMode,
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    update_config,
)
from vllm.device_allocator.uvm import (
    batch_advise_ranges_cpu_with_executor,
    batch_advise_ranges_preferred_location,
    batch_offload_ranges_to_cpu_with_executor,
    batch_prefetch_ranges_to_device,
    batch_prefetch_ranges_to_device_with_executor,
    batch_prepare_ranges_for_native_remote_access_with_executor,
    batch_prepare_ranges_for_remote_read_with_executor,
    driver_batch_query_ranges_residency_bytes,
    probe_driver_capable_range,
    uvm_allocation_phase,
    uvm_enabled_allocation_phase,
)
from vllm.device_allocator.uvm_pool_residency import (
    close_uvm_slot_direct_policy_session,
    commit_uvm_regions_gpu_placement,
    freeze_uvm_slot_direct_session_contract,
    preallocate_uvm_expert_groups_to_slots,
    preallocate_uvm_registered_plan_second_bank,
    prepare_uvm_regions_for_gpu_placement,
    query_uvm_regions_physical_residency,
    register_uvm_slot_direct_layer_plans,
    snapshot_uvm_region_registry,
    transition_uvm_regions_to_remote_read,
)
from vllm.distributed.ec_transfer import get_ec_transfer, has_ec_transfer
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.distributed.parallel_state import (
    get_dcp_group,
    get_pp_group,
    get_tp_group,
    graph_capture,
    is_global_first_rank,
    prepare_communication_buffer_for_model,
)
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    XDRotaryEmbedding,
)
from vllm.model_executor.model_loader import TensorizerLoader, get_model_loader
from vllm.model_executor.models.interfaces import (
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsXDRoPE,
    is_mixture_of_experts,
    supports_eagle3,
    supports_mrope,
    supports_multimodal_pruning,
    supports_transcription,
    supports_xdrope,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
    is_pooling_model,
    is_text_generation_model,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    BatchedTensorInputs,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.jsontree import json_map_leaves
from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import DeviceMemoryProfiler
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import (
    get_dtype_size,
    kv_cache_dtype_str_to_dtype,
    supports_dynamo,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    create_fast_prefill_custom_backend,
    get_dcp_local_seq_lens,
    reorder_batch_to_split_decodes_and_prefills,
    split_attn_metadata,
)
from vllm.v1.core.uvm_kv_runtime_policy import (
    collect_uvm_kv_cache_physical_ranges_for_blocks,
    register_uvm_kv_cache_physical_ranges,
)
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    EncoderOnlyAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ECConnectorOutput,
    KVConnectorOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
    SamplerOutput,
    make_empty_encoder_model_runner_output,
)
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.ec_connector_model_runner_mixin import ECConnectorModelRunnerMixin
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.ubatch_utils import (
    UBatchSlice,
    UBatchSlices,
    check_ubatch_thresholds,
)
from vllm.v1.worker.utils import is_residual_scattered_for_sp

from .utils import (
    AttentionGroup,
    MultiModalBudget,
    add_kv_sharing_layers_to_kv_cache_groups,
    bind_kv_cache,
    gather_mm_placeholders,
    sanity_check_mm_encoder_outputs,
    scatter_mm_placeholders,
)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput

logger = init_logger(__name__)


def _build_pure_prefill_dummy_shape(
    *,
    num_tokens: int,
    num_reqs: int,
    max_model_len: int,
) -> tuple[list[int], int]:
    """Build a valid zero-context Prefill shape for a dummy model run.

    ``_dummy_run`` historically uses the total batch token count as every
    request's sequence length. That is useful for some memory-profile runs,
    but it is not a valid representation of a multi-request pure Prefill and
    can exceed the per-request block-table capacity. Reserve warmup callers
    use this helper to describe the actual balanced request lengths instead.
    """
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    if num_reqs <= 0:
        raise ValueError(f"num_reqs must be positive, got {num_reqs}")
    if max_model_len <= 0:
        raise ValueError(
            f"max_model_len must be positive, got {max_model_len}"
        )

    effective_num_reqs = min(int(num_tokens), int(num_reqs))
    min_tokens_per_req, remainder = divmod(num_tokens, effective_num_reqs)
    request_lengths = [min_tokens_per_req] * effective_num_reqs
    request_lengths[-1] += remainder
    max_query_len = max(request_lengths)
    if max_query_len > max_model_len:
        raise ValueError(
            "Pure-Prefill dummy shape exceeds max_model_len: "
            f"tokens={num_tokens} requests={effective_num_reqs} "
            f"max_query_len={max_query_len} max_model_len={max_model_len}"
        )
    return request_lengths, max_query_len


def _uvm_kv_tail_range_for_allocation(
    *,
    ptr: int,
    size_bytes: int,
    page_size: int,
    total_blocks: int,
    target_blocks: int,
) -> tuple[int, int, int, int] | None:
    """Return (ptr, bytes, first_block, last_block) for one KV CPU tail."""
    if ptr <= 0 or size_bytes <= 0 or page_size <= 0 or total_blocks <= 0:
        return None
    clamped_target = min(max(int(target_blocks), 0), int(total_blocks))
    allocation_blocks = int(size_bytes) // int(page_size)
    max_block_id = min(int(total_blocks), allocation_blocks - 1)
    first_cpu_block = clamped_target + 1
    if first_cpu_block > max_block_id:
        return None
    tail_ptr = int(ptr) + first_cpu_block * int(page_size)
    tail_bytes = (max_block_id - first_cpu_block + 1) * int(page_size)
    return tail_ptr, tail_bytes, first_cpu_block, max_block_id


def _uvm_kv_ranges_have_exclusive_residency(
    ranges: Sequence[tuple[int, int]],
    residency: dict[str, int | bool],
    *,
    location: str,
) -> bool:
    """Return whether every byte is resident exclusively at one location."""
    if not ranges:
        return True
    expected_bytes = sum(int(size) for _, size in ranges)
    if expected_bytes <= 0 or location not in ("cpu", "gpu"):
        return False
    local_key = f"{location}_bytes"
    remote_key = "gpu_bytes" if location == "cpu" else "cpu_bytes"
    return (
        residency.get("ok") is True
        and int(residency.get("queried_bytes", -1)) == expected_bytes
        and int(residency.get("any_bytes", -1)) == expected_bytes
        and int(residency.get(local_key, -1)) == expected_bytes
        and int(residency.get(remote_key, -1)) == 0
        and int(residency.get("unknown_bytes", -1)) == 0
    )


AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict


def _moe_token_expert_trace_enabled() -> bool:
    for env_name in (
        "VLLM_MOE_TOKEN_EXPERT_TRACE_ENABLE",
        "VLLM_QWEN3_MOE_EXPERT_TRACE_ENABLE",
    ):
        if os.environ.get(env_name, "0").lower() in ("1", "true", "yes", "on"):
            return True
    return False


# Wrapper for ModelRunnerOutput to support overlapped execution.
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        logprobs_tensors: LogprobsTensors | None,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.cuda.Stream,
        vocab_size: int,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self.async_copy_ready_event = torch.Event()

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            self._logprobs_tensors_cpu = (
                self._logprobs_tensors.to_cpu_nonblocking()
                if self._logprobs_tensors
                else None
            )
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.

        This function blocks until the copy is finished.
        """
        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        self.async_copy_ready_event.synchronize()

        # Release the device tensors once the copy has completed.
        del self._logprobs_tensors
        del self._sampled_token_ids
        if max_gen_len == 1:
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            for i in self._invalid_req_indices:
                valid_sampled_token_ids[i].clear()
            cu_num_tokens = None
        else:
            valid_sampled_token_ids, cu_num_tokens = RejectionSampler.parse_output(
                self.sampled_token_ids_cpu,
                self.vocab_size,
                self._invalid_req_indices,
                return_cu_num_tokens=self._logprobs_tensors_cpu is not None,
            )

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        if self._logprobs_tensors_cpu:
            output.logprobs = self._logprobs_tensors_cpu.tolists(cu_num_tokens)
        return output


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    # Keep per-layer attention metadata alive until sampling has been queued.
    # This is required by the opt-in Triton MLA pure-Prefill full-Graph path:
    # its GraphExec can still be consuming device buffers owned by this
    # metadata after execute_model() has returned to the engine.
    retained_attn_metadata: PerLayerAttnMetadata | None
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: ECConnectorOutput | None


class GPUModelRunner(
    LoRAModelRunnerMixin, KVConnectorModelRunnerMixin, ECConnectorModelRunnerMixin
):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        from vllm.model_executor.models.utils import (
            set_cpu_offload_max_bytes,
            validate_cpu_offload_graph_static_alias_runtime,
        )

        set_cpu_offload_max_bytes(int(self.cache_config.cpu_offload_gb * 1024**3))
        # Validate before model construction/profile execution. In particular,
        # the Python H2D hook is intentionally captured by CUDA Graph and must
        # not first be traced by Dynamo/Inductor.
        validate_cpu_offload_graph_static_alias_runtime(vllm_config)

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, self.model_config
        )

        self.is_pooling_model = model_config.runner_type == "pooling"
        self.enable_prompt_embeds = model_config.enable_prompt_embeds
        self.is_multimodal_raw_input_only_model = (
            model_config.is_multimodal_raw_input_only_model
        )
        # This will be overridden in load_model()
        self.is_multimodal_pruning_enabled = False
        self.max_model_len = model_config.max_model_len

        # Always set to false after the first forward pass
        self.calculate_kv_scales = self.cache_config.calculate_kv_scales
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size
        self.dcp_rank = 0 if self.dcp_world_size <= 1 else get_dcp_group().rank_in_group
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        self.broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend == "external_launcher"
            and len(get_pp_group().ranks) > 0
        )

        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(parallel_config)
        self.hidden_size = model_config.get_hidden_size()
        self.attention_chunk_size = model_config.attention_chunk_size
        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = model_config.uses_alibi

        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        self.uses_xdrope_dim = model_config.uses_xdrope_dim
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config
        )

        if self.model_config.is_encoder_decoder:
            # Maximum length of the encoder input, only for encoder-decoder
            # models.
            self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens
        else:
            self.max_encoder_len = 0

        # Sampler
        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)

        self.eplb_state: EplbState | None = None
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []
        self._uvm_kv_caches_by_layer: dict[str, Any] = {}
        # Initialize in initialize_kv_cache_tensors
        self.cross_layers_kv_cache: torch.Tensor | None = None
        self.cross_layers_attn_backend: type[AttentionBackend] | None = None
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []
        # self.kv_cache_config: KVCacheConfig

        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}

        self.use_aux_hidden_state_outputs = False
        # Set up speculative decoding.
        # NOTE(Jiayi): currently we put the entire draft model on
        # the last PP rank. This is not ideal if there are many
        # layers in the draft model.
        if self.speculative_config and get_pp_group().is_last_rank:
            self.drafter: (
                NgramProposer | SuffixDecodingProposer | EagleProposer | MedusaProposer
            )
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.method == "suffix":
                self.drafter = SuffixDecodingProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device, self)
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = (
                        self.drafter.eagle3_use_aux_hidden_state
                    )
            elif self.speculative_config.method == "medusa":
                self.drafter = MedusaProposer(
                    vllm_config=self.vllm_config, device=self.device
                )
            else:
                raise ValueError(
                    "Unknown speculative decoding method: "
                    f"{self.speculative_config.method}"
                )
            self.rejection_sampler = RejectionSampler(self.sampler)

        self.num_spec_tokens = 0
        if self.speculative_config:
            self.num_spec_tokens = self.speculative_config.num_speculative_tokens

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}
        self.comm_stream = torch.cuda.Stream()

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        logits_processors = model_config.logits_processors
        custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (
            tuple(logits_processors) if logits_processors is not None else ()
        )
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            # We need to use the encoder length for encoder-decoer
            # because of KV cache for cross-attention.
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
            kernel_block_sizes=[self.cache_config.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                self.pin_memory,
                self.is_pooling_model,
                custom_logitsprocs,
            ),
            # We currently don't know whether a particular custom logits processor
            # uses output token ids so we set this conservatively.
            logitsprocs_need_output_token_ids=bool(custom_logitsprocs),
            is_pooling_model=self.is_pooling_model,
            cp_kv_cache_interleave_size=self.parallel_config.cp_kv_cache_interleave_size,
        )

        self.use_async_scheduling = self.scheduler_config.async_scheduling
        # Separate cuda stream for overlapping transfer of sampled token ids from
        # GPU to CPU when async scheduling is enabled.
        self.async_output_copy_stream: torch.cuda.Stream | None = None
        # cuda event to synchronize use of reused CPU tensors between steps
        # when async scheduling is enabled.
        self.prepare_inputs_event: torch.Event | None = None
        if self.use_async_scheduling:
            self.async_output_copy_stream = torch.cuda.Stream()
            self.prepare_inputs_event = torch.Event()

        # self.cudagraph_batch_sizes sorts in ascending order.
        if (
            self.compilation_config.cudagraph_capture_sizes
            and self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
        ):
            self.cudagraph_batch_sizes = sorted(
                self.compilation_config.cudagraph_capture_sizes
            )

        # Cache the device properties.
        self._init_device_properties()

        # Persistent buffers for CUDA graphs.
        self.input_ids = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
        self.positions = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
        self._moe_token_trace_metadata: dict[str, Any] | None = None
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )
        self.seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        self.encoder_seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens = self._make_buffer(
                self.max_num_reqs, dtype=torch.int32
            )
        # Because inputs_embeds may be bfloat16 and we don't need a numpy
        # version of this tensor, avoid a RuntimeError by not creating a
        # numpy buffer.
        self.inputs_embeds = self._make_buffer(
            self.max_num_tokens, self.hidden_size, dtype=self.dtype, numpy=False
        )
        self.is_token_ids = self._make_buffer(self.max_num_tokens, dtype=torch.bool)
        self.discard_request_indices = self._make_buffer(
            self.max_num_reqs, dtype=torch.int64
        )
        self.num_discarded_requests = 0

        self.num_decode_draft_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int32
        )
        self.num_accepted_tokens = self._make_buffer(
            self.max_num_reqs, dtype=torch.int64
        )

        # Only relevant for multimodal models
        if self.supports_mm_inputs:
            self.is_mm_embed = self._make_buffer(self.max_num_tokens, dtype=torch.bool)

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = self._make_buffer(
                (3, self.max_num_tokens + 1), dtype=torch.int64
            )

        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            # Similar to mrope but use assigned dimension number for RoPE, 4 as default.
            self.xdrope_positions = self._make_buffer(
                (self.uses_xdrope_dim, self.max_num_tokens + 1), dtype=torch.int64
            )

        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: IntermediateTensors | None = None

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(
            max(self.max_num_reqs + 1, self.max_model_len, self.max_num_tokens),
            dtype=np.int64,
        )

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device
            )

        self.uniform_decode_query_len = 1 + self.num_spec_tokens

        # Cudagraph dispatcher for runtime cudagraph dispatching.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        self.mm_budget = (
            MultiModalBudget(
                self.model_config,
                self.scheduler_config,
                self.mm_registry,
            )
            if self.supports_mm_inputs
            else None
        )

        self.reorder_batch_threshold: int | None = None

        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        self.runner_only_attn_layers: set[str] = set()

        # Cached outputs.
        self._draft_token_ids: list[list[int]] | torch.Tensor | None = None
        self.transfer_event = torch.Event()
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )

        # Pre-allocated tensor for copying valid sampled token counts to CPU,
        # with dedicated stream for overlapping and event for coordination.
        self.valid_sampled_token_count_event: torch.Event | None = None
        self.valid_sampled_token_count_copy_stream: torch.cuda.Stream | None = None
        if self.use_async_scheduling and self.num_spec_tokens:
            self.valid_sampled_token_count_event = torch.Event()
            self.valid_sampled_token_count_copy_stream = torch.cuda.Stream()
        self.valid_sampled_token_count_cpu = torch.empty(
            self.max_num_reqs,
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )

        # Ephemeral state transferred between execute_model() and sample_tokens().
        self.execute_model_state: ExecuteModelState | None = None
        self.kv_connector_output: KVConnectorOutput | None = None
        self._uvm_init_weight_placement_done = False
        self._uvm_original_va_phase_prefill_seen = False
        self._uvm_original_va_phase_decode_cutover_done = False
        self._uvm_original_va_phase_poison_reason: str | None = None
        self._uvm_original_va_phase_transition_result: dict[str, Any] | None = None

    def reset_mm_cache(self) -> None:
        if self.mm_budget:
            self.mm_budget.reset_cache()

    def _get_positions(self, num_tokens: Any):
        if isinstance(num_tokens, int):
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, :num_tokens]
            if self.uses_xdrope_dim > 0:
                return self.xdrope_positions.gpu[:, :num_tokens]
            return self.positions.gpu[:num_tokens]
        else:
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, num_tokens]
            if self.uses_xdrope_dim > 0:
                return self.xdrope_positions.gpu[:, num_tokens]
            return self.positions.gpu[num_tokens]

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            with_numpy=numpy,
        )

    def _init_model_kwargs(self, num_tokens: int):
        model_kwargs = dict[str, Any]()

        if not self.is_pooling_model:
            return model_kwargs

        num_reqs = self.input_batch.num_reqs
        pooling_params = self.input_batch.get_pooling_params()

        token_type_id_requests = dict[int, Any]()
        for i, param in enumerate(pooling_params):
            if (
                param.extra_kwargs is not None
                and (token_types := param.extra_kwargs.get("compressed_token_type_ids"))
                is not None
            ):
                token_type_id_requests[i] = token_types

        if len(token_type_id_requests) == 0:
            return model_kwargs

        seq_lens = self.seq_lens.gpu[:num_reqs]
        token_type_ids = []

        for i in range(num_reqs):
            pos = token_type_id_requests.get(i, seq_lens[i])
            ids = (torch.arange(seq_lens[i]) >= pos).int()
            token_type_ids.append(ids)

        model_kwargs["token_type_ids"] = torch.concat(token_type_ids).to(
            device=self.device
        )
        return model_kwargs

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        # Attention free models have zero kv_cache_goups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if self.reorder_batch_threshold is not None:
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold,
            )

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        """Initialize attributes from torch.cuda.get_device_properties"""
        self.device_properties = torch.cuda.get_device_properties(self.device)
        self.num_sms = self.device_properties.multi_processor_count

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        torch.cuda.synchronize()

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.num_prompt_logprobs.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state

            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            if self.uses_xdrope_dim > 0:
                self._init_xdrope_positions(req_state)

            reqs_to_add.append(req_state)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs

        # Wait until valid_sampled_tokens_count is copied to cpu,
        # then use it to update actual num_computed_tokens of each request.
        valid_sampled_token_count = self._get_valid_sampled_token_count()

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]
            req_index = self.input_batch.req_id_to_index.get(req_id)

            # prev_num_draft_len is used in async scheduling mode with
            # spec decode. it indicates if need to update num_computed_tokens
            # of the request. for example:
            # fist step: num_computed_tokens = 0, spec_tokens = [],
            # prev_num_draft_len = 0.
            # second step: num_computed_tokens = 100(prompt lenth),
            # spec_tokens = [a,b], prev_num_draft_len = 0.
            # third step: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
            # prev_num_draft_len = 2.
            # num_computed_tokens in first step and second step does't contain
            # the spec tokens length, but in third step it contains the
            # spec tokens length. we only need to update num_computed_tokens
            # when prev_num_draft_len > 0.
            if req_state.prev_num_draft_len:
                if req_index is None:
                    req_state.prev_num_draft_len = 0
                else:
                    assert self.input_batch.prev_req_id_to_index is not None
                    prev_req_index = self.input_batch.prev_req_id_to_index[req_id]
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1
                    num_rejected = req_state.prev_num_draft_len - num_accepted
                    num_computed_tokens -= num_rejected
                    req_state.output_token_ids.extend([-1] * num_accepted)

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (
                    num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                )
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(new_token_ids[-num_new_tokens:])
            elif num_output_tokens < len(req_state.output_token_ids):
                # Some output tokens were discarded due to a sync-KV-load
                # failure. Align the cached state.
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    self.input_batch.num_tokens[req_index] = end_idx
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.

                if self.use_async_scheduling and num_output_tokens > 0:
                    # We must recover the output token ids for resumed requests in the
                    # async scheduling case, so that correct input_ids are obtained.
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index:end_token_index
                ] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index
                self.input_batch.num_tokens[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, []
            )
            num_spec_tokens = len(spec_token_ids)
            # For async scheduling, token_ids_cpu assigned from
            # spec_token_ids are placeholders and will be overwritten in
            # _prepare_input_ids.
            if num_spec_tokens:
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index
                ] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec tokens.
                self.input_batch.num_tokens[req_index] += num_spec_tokens

            # When speculative decoding is used with structured output,
            # the scheduler can drop draft tokens that do not
            # conform to the schema. This can result in
            # scheduler_output.scheduled_spec_decode_tokens being empty,
            # even when speculative decoding is enabled.
            self.input_batch.spec_token_ids[req_index].clear()
            self.input_batch.spec_token_ids[req_index].extend(spec_token_ids)

            # there are no draft tokens with async scheduling,
            # we clear the spec_decoding info in scheduler_output and
            # use normal sampling but rejection_sampling.
            if self.use_async_scheduling:
                req_state.prev_num_draft_len = num_spec_tokens
                if num_spec_tokens and self._draft_token_ids is None:
                    scheduler_output.total_num_scheduled_tokens -= num_spec_tokens
                    scheduler_output.num_scheduled_tokens[req_id] -= num_spec_tokens
                    scheduler_output.scheduled_spec_decode_tokens.pop(req_id, None)
        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor
    ) -> None:
        """Update the cached states after model execution.

        This is used for MTP/EAGLE for hybrid models, as in linear attention,
        only the last token's state is kept. In MTP/EAGLE, for draft tokens
        the state are kept util we decide how many tokens are accepted for
        each sequence, and a shifting is done during the next iteration
        based on the number of accepted tokens.
        """
        if not self.model_config.is_hybrid or not self.speculative_config:
            return

        # Find the number of accepted tokens for each sequence.
        num_accepted_tokens = (
            (
                torch.cat(
                    [
                        output_token_ids,
                        torch.full(
                            (output_token_ids.size(0), 1),
                            -1,
                            device=output_token_ids.device,
                        ),
                    ],
                    dim=1,
                )
                == -1
            )
            .int()
            .argmax(-1)
            .cpu()
            .numpy()
        )
        for i, num_tokens in enumerate(num_accepted_tokens):
            self.input_batch.num_accepted_tokens_cpu[i] = num_tokens

    def _init_mrope_positions(self, req_state: CachedRequestState):
        model = self.get_model()
        assert supports_mrope(model), "M-RoPE support is not implemented."
        assert req_state.prompt_token_ids is not None, (
            "M-RoPE requires prompt_token_ids to be available."
        )
        mrope_model = cast(SupportsMRoPE, model)

        req_state.mrope_positions, req_state.mrope_position_delta = (
            mrope_model.get_mrope_input_positions(
                req_state.prompt_token_ids,
                req_state.mm_features,
            )
        )

    def _init_xdrope_positions(self, req_state: CachedRequestState):
        model = self.get_model()
        xdrope_model = cast(SupportsXDRoPE, model)
        assert req_state.prompt_token_ids is not None, (
            "XD-RoPE requires prompt_token_ids to be available."
        )
        assert supports_xdrope(model), "XD-RoPE support is not implemented."

        req_state.xdrope_positions = xdrope_model.get_xdrope_input_positions(
            req_state.prompt_token_ids,
            req_state.mm_features,
        )

    def _extract_mm_kwargs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> BatchedTensorInputs:
        if not scheduler_output or not self.is_multimodal_raw_input_only_model:
            return {}

        mm_kwargs = list[MultiModalKwargsItem]()
        for req in scheduler_output.scheduled_new_reqs:
            for feature in req.mm_features:
                if feature.data is not None:
                    mm_kwargs.append(feature.data)

        # Input all modalities at once
        model = cast(SupportsMultiModal, self.model)
        mm_kwargs_combined: BatchedTensorInputs = {}
        for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
            merge_by_field_config=model.merge_by_field_config,
            multimodal_cpu_fields=model.multimodal_cpu_fields,
        ):
            mm_kwargs_combined.update(mm_kwargs_group)

        return mm_kwargs_combined

    def _dummy_mm_kwargs(self, num_seqs: int) -> BatchedTensorInputs:
        if not self.is_multimodal_raw_input_only_model:
            return {}

        mm_budget = self.mm_budget
        assert mm_budget is not None

        dummy_modality = mm_budget.get_modality_with_max_tokens()
        return self._get_mm_dummy_batch(dummy_modality, num_seqs)

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _prepare_input_ids(
        self,
        scheduler_output: "SchedulerOutput",
        total_num_scheduled_tokens: int,
        cu_num_tokens: np.ndarray,
    ) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_common_req_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        indices_match = True
        max_flattened_index = -1
        total_num_spec_tokens = 0
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                draft_len = len(scheduled_spec_tokens.get(req_id, ()))
                total_num_spec_tokens += draft_len
                flattened_index = cu_num_tokens[cur_index].item() - 1
                # example: cu_num_tokens = [2, 5, 8], draft_tokens = [1, 2, 2]
                # sample_flattened_indices = [0, 2, 5]
                # spec_flattened_indices = [1,   3, 4,    6, 7]
                sample_flattened_indices.append(flattened_index - draft_len)
                spec_flattened_indices.extend(
                    range(flattened_index - draft_len + 1, flattened_index + 1)
                )
                start = prev_index * self.num_spec_tokens
                # prev_draft_token_indices is used to find which draft_tokens_id
                # should be copied to input_ids
                # example: prev draft_tokens_id [[1,2], [3,4], [5, 6]]
                # flatten draft_tokens_id [1,2,3,4,5,6]
                # draft_len of each request [1, 2, 1]
                # then prev_draft_token_indices is [0,   2, 3,   4]
                prev_draft_token_indices.extend(range(start, start + draft_len))
                indices_match &= prev_index == flattened_index
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_commmon_tokens = len(sample_flattened_indices)
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
        if num_commmon_tokens < total_without_spec:
            # If not all requests are decodes from the last iteration,
            # We need to copy the input_ids_cpu to the GPU first.
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_commmon_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids.cpu will have all the input ids.
            return
        if indices_match and max_flattened_index == (num_commmon_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids.gpu[:num_commmon_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_commmon_tokens, 0],
                non_blocking=True,
            )
            if self.enable_prompt_embeds:
                self.is_token_ids.gpu[:num_commmon_tokens] = True
            return
        # Upload the index tensors asynchronously so the scatter can be non-blocking.
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )

        # Scatter the draft tokens after the sampled tokens are scattered.
        if self._draft_token_ids is None or not spec_flattened_indices:
            return

        assert isinstance(self._draft_token_ids, torch.Tensor)
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices, dtype=torch.int64, pin_memory=self.pin_memory
        ).to(self.device, non_blocking=True)

        # because input_ids dtype is torch.int32,
        # so convert draft_token_ids to torch.int32 here.
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)
        self._draft_token_ids = None

        self.input_ids.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )

    def _get_encoder_seq_lens(
        self,
        num_scheduled_tokens: dict[str, int],
        kv_cache_spec: KVCacheSpec,
        num_reqs: int,
    ) -> tuple[torch.Tensor | None, np.ndarray | None]:
        if not isinstance(kv_cache_spec, CrossAttentionSpec):
            return None, None

        # Build encoder_seq_lens array mapping request indices to
        # encoder lengths for inputs scheduled in this batch
        for req_id in num_scheduled_tokens:
            req_index = self.input_batch.req_id_to_index[req_id]
            req_state = self.requests[req_id]
            if req_state.mm_features is None:
                self.encoder_seq_lens.np[req_index] = 0
                continue

            # Get the total number of encoder input tokens for running encoder requests
            # whether encoding is finished or not so that cross-attention knows how
            # many encoder tokens to attend to.
            encoder_input_tokens = sum(
                feature.mm_position.length for feature in req_state.mm_features
            )
            self.encoder_seq_lens.np[req_index] = encoder_input_tokens

        self.encoder_seq_lens.copy_to_gpu(num_reqs)
        encoder_seq_lens = self.encoder_seq_lens.gpu[:num_reqs]
        encoder_seq_lens_cpu = self.encoder_seq_lens.np[:num_reqs]

        return encoder_seq_lens, encoder_seq_lens_cpu

    def _make_moe_token_trace_metadata(
        self,
        *,
        req_indices: np.ndarray,
        positions_np: np.ndarray,
        num_scheduled_tokens: np.ndarray,
        cu_num_tokens: np.ndarray,
        total_num_scheduled_tokens: int,
        num_reqs: int,
    ) -> dict[str, Any] | None:
        if not _moe_token_expert_trace_enabled():
            return None

        try:
            req_ids_by_index = list(self.input_batch.req_ids[:num_reqs])
            request_indices = [
                int(value) for value in req_indices[:total_num_scheduled_tokens]
            ]
            request_ids = [
                req_ids_by_index[req_index]
                if 0 <= req_index < len(req_ids_by_index)
                else None
                for req_index in request_indices
            ]
            input_token_ids = self.input_ids.cpu[:total_num_scheduled_tokens].tolist()
            return {
                "source": "gpu_model_runner",
                "pid": os.getpid(),
                "num_reqs": int(num_reqs),
                "num_actual_tokens": int(total_num_scheduled_tokens),
                "req_ids_by_index": req_ids_by_index,
                "request_indices": request_indices,
                "request_ids": request_ids,
                "token_positions": [
                    int(value) for value in positions_np[:total_num_scheduled_tokens]
                ],
                "token_ids": [int(value) for value in input_token_ids],
                "num_scheduled_tokens": [
                    int(value) for value in num_scheduled_tokens[:num_reqs]
                ],
                "num_computed_tokens": [
                    int(value)
                    for value in self.input_batch.num_computed_tokens_cpu[:num_reqs]
                ],
                "query_start_loc": [0]
                + [int(value) for value in cu_num_tokens[:num_reqs]],
            }
        except Exception:
            logger.debug(
                "Failed to build MoE token expert trace metadata", exc_info=True
            )
            return None

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
        max_num_scheduled_tokens: int,
    ) -> tuple[
        torch.Tensor,
        SpecDecodeMetadata | None,
        UBatchSlices | None,
        torch.Tensor | None,
    ]:
        """
        :return: tuple[
            logits_indices, spec_decode_metadata,
            ubatch_slices, num_tokens_across_dp,
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        self._moe_token_trace_metadata = None

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

        # Get positions.
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
            out=positions_np,
        )

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Calculate XD-RoPE positions.
        # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
        if self.uses_xdrope_dim > 0:
            self._calc_xdrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices)

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        self._moe_token_trace_metadata = self._make_moe_token_trace_metadata(
            req_indices=req_indices,
            positions_np=positions_np,
            num_scheduled_tokens=num_scheduled_tokens,
            cu_num_tokens=cu_num_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            num_reqs=num_reqs,
        )
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids_tensor.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens],
            )

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds:
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # Skip if trying to read beyond available embeddings
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # Copy available embeddings
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[
                        output_idx : output_idx + actual_num_sched
                    ].copy_(req_embeds[start_pos:actual_end])

                output_idx += num_sched

        self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)

        # Prepare the attention metadata.
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]

        num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
        num_tokens_padded = self._get_num_input_tokens(num_tokens_unpadded)
        uniform_decode = (
            max_num_scheduled_tokens == self.uniform_decode_query_len
        ) and (total_num_scheduled_tokens == num_reqs * max_num_scheduled_tokens)

        # Disable DP padding when running eager to avoid excessive padding when
        # running prefills. This lets us set enforce_eager on the prefiller in
        # a P/D setup and still use CUDA graphs (enabled by this padding) on the
        # decoder.
        allow_dp_padding = self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE

        ubatch_slices, num_tokens_across_dp = coordinate_batch_across_dp(
            num_tokens_unpadded=num_tokens_unpadded,
            parallel_config=self.parallel_config,
            allow_microbatching=True,
            allow_dp_padding=allow_dp_padding,
            num_tokens_padded=num_tokens_padded,
            uniform_decode=uniform_decode,
            num_scheduled_tokens_per_request=num_scheduled_tokens,
        )

        self.seq_lens.np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
        )
        # Fill unused with 0 for full cuda graph mode.
        self.seq_lens.np[num_reqs:].fill(0)
        self.seq_lens.copy_to_gpu()

        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)

        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        discard_requests_mask = self.seq_lens.np[:num_reqs] < num_tokens_np
        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[: self.num_discarded_requests] = (
            discard_request_indices
        )

        self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

        # Copy the tensors to the GPU.
        self._prepare_input_ids(
            scheduler_output,
            total_num_scheduled_tokens,
            cu_num_tokens,
        )

        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        elif self.uses_xdrope_dim > 0:
            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            self.xdrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.xdrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        else:
            # Common case (1D positions)
            self.positions.copy_to_gpu(total_num_scheduled_tokens)

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            num_draft_tokens = None
            spec_decode_metadata = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                num_decode_draft_tokens[req_idx] = (
                    len(draft_token_ids)
                    if (
                        self.input_batch.num_computed_tokens_cpu[req_idx]
                        >= self.input_batch.num_prompt_tokens[req_idx]
                    )
                    else -1
                )
            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens
            )
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1
            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()

        # Hot-Swap lora model
        if self.lora_config:
            assert (
                np.sum(num_sampled_tokens)
                <= self.vllm_config.scheduler_config.max_num_batched_tokens
            )
            self.set_active_loras(
                self.input_batch, num_scheduled_tokens, num_sampled_tokens
            )

        return (
            logits_indices,
            spec_decode_metadata,
            ubatch_slices,
            num_tokens_across_dp,
        )

    def _build_attention_metadata(
        self,
        total_num_scheduled_tokens: int,
        max_num_scheduled_tokens: int,
        num_reqs: int,
        ubatch_slices: UBatchSlices | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens: dict[str, int] | None = None,
        cascade_attn_prefix_lens: list[list[int]] | None = None,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        """
        :return: tuple[attn_metadata, spec_decode_common_attn_metadata]
        """
        logits_indices_padded = None
        num_logits_indices = None
        if logits_indices is not None:
            num_logits_indices = logits_indices.size(0)
            if self.cache_config.kv_sharing_fast_prefill:
                logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                    logits_indices
                )

        # update seq_lens of decode reqs under DCP.
        if self.dcp_world_size > 1:
            self.dcp_local_seq_lens.cpu[:num_reqs] = get_dcp_local_seq_lens(
                self.seq_lens.cpu[:num_reqs],
                self.dcp_world_size,
                self.dcp_rank,
                self.parallel_config.cp_kv_cache_interleave_size,
            )
            self.dcp_local_seq_lens.copy_to_gpu(num_reqs)

        attn_metadata: PerLayerAttnMetadata = {}
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]

        # Used in the below loop
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]
        query_start_loc_cpu = self.query_start_loc.cpu[: num_reqs + 1]
        seq_lens = self.seq_lens.gpu[:num_reqs]
        seq_lens_cpu = self.seq_lens.cpu[:num_reqs]
        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
            :num_reqs
        ]

        dcp_local_seq_lens, dcp_local_seq_lens_cpu = None, None
        if self.dcp_world_size > 1:
            dcp_local_seq_lens = self.dcp_local_seq_lens.gpu[:num_reqs]
            dcp_local_seq_lens_cpu = self.dcp_local_seq_lens.cpu[:num_reqs]

        spec_decode_common_attn_metadata = None

        if for_cudagraph_capture:
            # For some attention backends (e.g. FA) with sliding window models we need
            # to make sure the backend see a max_seq_len that is larger to the sliding
            # window size when capturing to make sure the correct kernel is selected.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = self.seq_lens.np[:num_reqs].max().item()

        if use_spec_decode:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs]
            )
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_gid, kv_cache_group in enumerate(
            self.kv_cache_config.kv_cache_groups
        ):
            encoder_seq_lens, encoder_seq_lens_cpu = self._get_encoder_seq_lens(
                num_scheduled_tokens or {},
                kv_cache_group.kv_cache_spec,
                num_reqs,
            )

            if isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                # Encoder-only layers do not have KV cache, so we need to
                # create a dummy block table and slot mapping for them.
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (total_num_scheduled_tokens,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_gid]
                blk_table_tensor = blk_table.get_device_tensor(num_reqs)
                slot_mapping = blk_table.slot_mapping.gpu[:total_num_scheduled_tokens]

                # Fill unused with -1. Needed for reshape_and_cache in full cuda
                # graph mode.
                blk_table.slot_mapping.gpu[total_num_scheduled_tokens:].fill_(-1)

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                max_seq_len=max_seq_len,
                block_table_tensor=blk_table_tensor,
                slot_mapping=slot_mapping,
                logits_indices_padded=logits_indices_padded,
                num_logits_indices=num_logits_indices,
                causal=True,
                encoder_seq_lens=encoder_seq_lens,
                encoder_seq_lens_cpu=encoder_seq_lens_cpu,
                dcp_local_seq_lens=dcp_local_seq_lens,
                dcp_local_seq_lens_cpu=dcp_local_seq_lens_cpu,
            )

            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(self.drafter, EagleProposer):
                    if self.drafter.attn_layer_names[0] in kv_cache_group.layer_names:
                        spec_decode_common_attn_metadata = common_attn_metadata
                else:
                    spec_decode_common_attn_metadata = common_attn_metadata

            for attn_gid, attn_group in enumerate(self.attn_groups[kv_cache_gid]):
                cascade_attn_prefix_len = (
                    cascade_attn_prefix_lens[kv_cache_gid][attn_gid]
                    if cascade_attn_prefix_lens
                    else 0
                )
                builder = attn_group.get_metadata_builder()

                extra_attn_metadata_args = {}
                if use_spec_decode and isinstance(builder, GDNAttentionMetadataBuilder):
                    extra_attn_metadata_args = dict(
                        num_accepted_tokens=self.num_accepted_tokens.gpu[:num_reqs],
                        num_decode_draft_tokens_cpu=self.num_decode_draft_tokens.cpu[
                            :num_reqs
                        ],
                    )

                if ubatch_slices is not None:
                    common_attn_metadata_list = split_attn_metadata(
                        ubatch_slices, common_attn_metadata
                    )
                    for ubid, common_attn_metadata in enumerate(
                        common_attn_metadata_list
                    ):
                        builder = attn_group.get_metadata_builder(ubatch_id=ubid)
                        if for_cudagraph_capture:
                            attn_metadata_i = builder.build_for_cudagraph_capture(
                                common_attn_metadata
                            )
                        else:
                            attn_metadata_i = builder.build(
                                common_prefix_len=cascade_attn_prefix_len,
                                common_attn_metadata=common_attn_metadata,
                            )
                        for layer_name in kv_cache_group.layer_names:
                            assert type(attn_metadata) is list
                            attn_metadata[ubid][layer_name] = attn_metadata_i
                else:
                    assert isinstance(attn_metadata, dict)
                    if for_cudagraph_capture:
                        attn_metadata_i = builder.build_for_cudagraph_capture(
                            common_attn_metadata
                        )
                    else:
                        attn_metadata_i = builder.build(
                            common_prefix_len=cascade_attn_prefix_len,
                            common_attn_metadata=common_attn_metadata,
                            **extra_attn_metadata_args,
                        )
                    for layer_name in attn_group.layer_names:
                        attn_metadata[layer_name] = attn_metadata_i

        return attn_metadata, spec_decode_common_attn_metadata

    def _compute_cascade_attn_prefix_lens(
        self,
        num_scheduled_tokens: np.ndarray,
        num_common_prefix_blocks: list[int],
    ) -> list[list[int]] | None:
        """
        :return: Optional[cascade_attn_prefix_lens]
            cascade_attn_prefix_lens is 2D: ``[kv_cache_group_id][attn_group_idx]``,
            None if we should not use cascade attention
        """

        use_cascade_attn = False
        num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups)
        cascade_attn_prefix_lens: list[list[int]] = [
            [] for _ in range(num_kv_cache_groups)
        ]

        for kv_cache_gid in range(num_kv_cache_groups):
            for attn_group in self.attn_groups[kv_cache_gid]:
                if isinstance(attn_group.kv_cache_spec, EncoderOnlyAttentionSpec):
                    cascade_attn_prefix_len = 0
                else:
                    # 0 if cascade attention should not be used
                    cascade_attn_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        num_common_prefix_blocks[kv_cache_gid],
                        attn_group.kv_cache_spec,
                        attn_group.get_metadata_builder(),
                    )
                cascade_attn_prefix_lens[kv_cache_gid].append(cascade_attn_prefix_len)
                use_cascade_attn |= cascade_attn_prefix_len > 0

        return cascade_attn_prefix_lens if use_cascade_attn else None

    def _compute_cascade_attn_prefix_len(
        self,
        num_scheduled_tokens: np.ndarray,
        num_common_prefix_blocks: int,
        kv_cache_spec: KVCacheSpec,
        attn_metadata_builder: AttentionMetadataBuilder,
    ) -> int:
        """Compute the length of the common prefix for cascade attention.

        NOTE(woosuk): The common prefix length returned by this function
        represents the length used specifically for cascade attention, not the
        actual number of tokens shared between requests. When cascade attention
        is disabled (use_cascade=False), this function returns 0 even if
        requests share common tokens. Additionally, the common prefix length is
        truncated to a multiple of the block size and may be further truncated
        due to implementation details explained below.

        Args:
            num_scheduled_tokens: Number of tokens scheduled per request.
            num_common_prefix_blocks: Number of shared KV cache blocks.

        Returns:
            int: Length of common prefix in tokens.
        """

        common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size
        if common_prefix_len == 0:
            # Common case.
            return 0

        # NOTE(woosuk): Cascade attention uses two attention kernels: one
        # for the common prefix and the other for the rest. For the first
        # kernel, we concatenate all the query tokens (possibly from
        # different requests) and treat them as if they are from the same
        # request. Then, we use bi-directional attention to process the
        # common prefix in the KV cache. Importantly, this means that the
        # first kernel does not do any masking.

        # Consider the following example:
        # Request 1's input query: [D, E, X]
        # Request 1's kv cache: [A, B, C, D, E, X]
        # Request 1's num_computed_tokens: 3 (i.e., [A, B, C])
        # Request 2's input query: [E, Y]
        # Request 2's kv cache: [A, B, C, D, E, Y]
        # Request 2's num_computed_tokens: 4 (i.e., [A, B, C, D])

        # If we use [A, B, C, D, E] as the common prefix, then the
        # first kernel will compute the bi-directional attention between
        # input query [D, E, X, E, Y] and common prefix [A, B, C, D, E].
        # However, this is wrong because D in Request 1 should not attend to
        # E in the common prefix (i.e., we need masking).
        # To avoid this, [A, B, C, D] should be the common prefix.
        # That is, the common prefix should be capped by the minimum
        # num_computed_tokens among the requests, and plus one to include
        # the first token of the query.

        # In practice, we use [A, B, C] as the common prefix, instead of
        # [A, B, C, D] (i.e., the common prefix is capped by the minimum
        # num_computed_tokens, without plus one).
        # This is because of an implementation detail: We want to always
        # use two kernels for cascade attention. Let's imagine:
        # Request 3's input query: [D]
        # Request 3's kv cache: [A, B, C, D]
        # Request 3's num_computed_tokens: 3 (i.e., [A, B, C])
        # If we use [A, B, C, D] as the common prefix for Request 1-3,
        # then Request 3 will be processed only by the first kernel,
        # and the second kernel will get an empty input. While this is not
        # a fundamental problem, our current implementation does not support
        # this case.
        num_reqs = len(num_scheduled_tokens)
        common_prefix_len = min(
            common_prefix_len, self.input_batch.num_computed_tokens_cpu[:num_reqs].min()
        )
        # common_prefix_len should be a multiple of the block size.
        common_prefix_len = (
            common_prefix_len // kv_cache_spec.block_size * kv_cache_spec.block_size
        )
        use_sliding_window = isinstance(kv_cache_spec, SlidingWindowSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.sliding_window is not None
        )
        use_local_attention = isinstance(kv_cache_spec, ChunkedLocalAttentionSpec) or (
            isinstance(kv_cache_spec, FullAttentionSpec)
            and kv_cache_spec.attention_chunk_size is not None
        )
        assert isinstance(kv_cache_spec, AttentionSpec)
        use_cascade = attn_metadata_builder.use_cascade_attention(
            common_prefix_len=common_prefix_len,
            query_lens=num_scheduled_tokens,
            num_query_heads=self.num_query_heads,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            use_alibi=self.use_alibi,
            use_sliding_window=use_sliding_window,
            use_local_attention=use_local_attention,
            num_sms=self.num_sms,
            dcp_world_size=self.dcp_world_size,
        )
        return common_prefix_len if use_cascade else 0

    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions.cpu[:, dst_start:dst_end] = req.mrope_positions[
                    :, src_start:src_end
                ]
                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                assert req.mrope_position_delta is not None
                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.mrope_positions.np,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                mrope_pos_ptr += completion_part_len

    def _calc_xdrope_positions(self, scheduler_output: "SchedulerOutput"):
        xdrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.xdrope_positions is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds
            )

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's xdrope_positions are pre-computed
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.xdrope_positions.cpu[:, dst_start:dst_end] = req.xdrope_positions[
                    :, src_start:src_end
                ]
                xdrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's xdrope_positions on-the-fly
                dst_start = xdrope_pos_ptr
                dst_end = xdrope_pos_ptr + completion_part_len

                XDRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.xdrope_positions.np,
                    out_offset=dst_start,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                xdrope_pos_ptr += completion_part_len

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int32
        )
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
        )
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32
        )
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True
        )
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).to(
            self.device, non_blocking=True
        )
        logits_indices = torch.from_numpy(logits_indices).to(
            self.device, non_blocking=True
        )
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True
        )
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True
        )

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    def _prepare_kv_sharing_fast_prefill(
        self,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        assert self.kv_sharing_fast_prefill_logits_indices is not None
        num_logits = logits_indices.shape[0]
        assert num_logits > 0
        self.kv_sharing_fast_prefill_logits_indices[:num_logits].copy_(logits_indices)
        # There might have leftover indices in logits_indices[num_logits:]
        # from previous iterations, whose values may be greater than the
        # batch size in the current iteration. To ensure indices are always
        # valid, we fill the padded indices with the last index.
        self.kv_sharing_fast_prefill_logits_indices[num_logits:].fill_(
            logits_indices[-1].item()
        )
        if (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and num_logits <= self.cudagraph_batch_sizes[-1]
        ):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_logits_padded = self.vllm_config.pad_for_cudagraph(num_logits)
        else:
            num_logits_padded = num_logits
        logits_indices_padded = self.kv_sharing_fast_prefill_logits_indices[
            :num_logits_padded
        ]
        return logits_indices_padded

    def _batch_mm_kwargs_from_scheduler(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[list[MultiModalKwargsItem], list[tuple[str, PlaceholderRange]]]:
        """Batch multimodal kwargs from scheduled encoder inputs.

        Args:
            scheduler_output: The scheduler output containing scheduled encoder
                inputs.

        Returns:
            A tuple of (mm_kwargs, req_ids_pos) where:
            - mm_kwargs: List of multimodal kwargs items to be batched
            - mm_hashes_pos: List of (mm_hash, position_info) tuples
        """
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return [], []
        # Batch the multi-modal inputs.
        mm_kwargs = list[MultiModalKwargsItem]()
        # list of tuple (mm_hash, position_info)
        mm_hashes_pos = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue
                mm_hash = mm_feature.identifier
                mm_kwargs.append(mm_feature.data)
                mm_hashes_pos.append((mm_hash, mm_feature.mm_position))

        return mm_kwargs, mm_hashes_pos

    def _execute_mm_encoder(
        self, scheduler_output: "SchedulerOutput"
    ) -> list[torch.Tensor]:
        # Batch the multi-modal inputs using the helper method.
        mm_kwargs, mm_hashes_pos = self._batch_mm_kwargs_from_scheduler(
            scheduler_output
        )

        if not mm_kwargs:
            return []

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        model = cast(SupportsMultiModal, self.model)
        encoder_outputs: list[torch.Tensor] = []
        for modality, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
            mm_kwargs,
            device=self.device,
            pin_memory=self.pin_memory,
            merge_by_field_config=model.merge_by_field_config,
            multimodal_cpu_fields=model.multimodal_cpu_fields,
        ):
            curr_group_outputs: list[torch.Tensor] = []

            # EVS-related change.
            # (ekhvedchenia): Temporary hack to limit peak memory usage when
            # processing multimodal data. This solves the issue with scheduler
            # putting too many video samples into a single batch. Scheduler
            # uses pruned vision tokens count to compare it versus compute
            # budget which is incorrect (Either input media size or non-pruned
            # output vision tokens count should be considered)
            # TODO(ywang96): Fix memory profiling to take EVS into account and
            # remove this hack.
            if (
                self.is_multimodal_pruning_enabled
                and modality == "video"
                and num_items > 1
            ):
                for video_mm_kwargs_item in filter(
                    lambda item: item.modality == "video", mm_kwargs
                ):
                    _, _, micro_batch_mm_inputs = next(
                        group_mm_kwargs_by_modality(
                            [video_mm_kwargs_item],
                            device=self.device,
                            pin_memory=self.pin_memory,
                            merge_by_field_config=model.merge_by_field_config,
                            multimodal_cpu_fields=model.multimodal_cpu_fields,
                        )
                    )

                    micro_batch_outputs = model.embed_multimodal(
                        **micro_batch_mm_inputs
                    )

                    curr_group_outputs.extend(micro_batch_outputs)
            else:
                # Run the encoder.
                # `curr_group_outputs` is either of the following:
                # 1. A tensor of shape (num_items, feature_size, hidden_size)
                # in case feature_size is fixed across all multimodal items.
                # 2. A list or tuple (length: num_items) of tensors,
                # each of shape (feature_size, hidden_size) in case the feature
                # size is dynamic depending on the input multimodal items.
                curr_group_outputs = model.embed_multimodal(**mm_kwargs_group)  # type: ignore[assignment]

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )
            encoder_outputs.extend(curr_group_outputs)

        # Cache the encoder outputs by mm_hash
        for (mm_hash, pos_info), output in zip(mm_hashes_pos, encoder_outputs):
            self.encoder_cache[mm_hash] = scatter_mm_placeholders(
                output,
                is_embed=pos_info.is_embed,
            )
            logger.debug("Finish execute for mm hash %s", mm_hash)
            self.maybe_save_ec_to_connector(self.encoder_cache, mm_hash)

        return encoder_outputs

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
        shift_computed_tokens: int = 0,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        mm_embeds = list[torch.Tensor]()
        is_mm_embed = self.is_mm_embed.cpu
        is_mm_embed[:total_num_scheduled_tokens] = False

        req_start_idx = 0
        should_sync_mrope_positions = False
        should_sync_xdrope_positions = False

        for req_id in self.input_batch.req_ids:
            mm_embeds_req: list[torch.Tensor] = []

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = req_state.num_computed_tokens + shift_computed_tokens

            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens,
                )
                assert start_idx < end_idx

                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]

                req_start_pos = req_start_idx + start_pos - num_computed_tokens
                is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = (
                    True if is_embed is None else is_embed
                )

                mm_embeds_item = gather_mm_placeholders(
                    encoder_output[start_idx:end_idx],
                    is_embed=is_embed,
                )
                mm_embeds_req.append(mm_embeds_item)

            if self.is_multimodal_pruning_enabled and self.uses_mrope:
                assert req_state.mrope_positions is not None
                should_sync_mrope_positions = True
                mm_embeds_req, new_mrope_positions, new_delta = (
                    self.model.recompute_mrope_positions(
                        input_ids=req_state.prompt_token_ids,
                        multimodal_embeddings=mm_embeds_req,
                        mrope_positions=req_state.mrope_positions,
                        num_computed_tokens=req_state.num_computed_tokens,
                    )
                )
                req_state.mrope_positions.copy_(new_mrope_positions)
                req_state.mrope_position_delta = new_delta

            mm_embeds.extend(mm_embeds_req)
            req_start_idx += num_scheduled_tokens

        is_mm_embed = self.is_mm_embed.copy_to_gpu(total_num_scheduled_tokens)

        if should_sync_mrope_positions:
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        if should_sync_xdrope_positions:
            self._calc_xdrope_positions(scheduler_output)
            self.xdrope_positions.copy_to_gpu(total_num_scheduled_tokens)

        return mm_embeds, is_mm_embed

    def get_model(self) -> nn.Module:
        # get raw model out of the cudagraph wrapper.
        if isinstance(self.model, (CUDAGraphWrapper, UBatchWrapper)):
            return self.model.unwrap()
        return self.model

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        supported_tasks = list(model.pooler.get_supported_tasks())

        if self.scheduler_config.enable_chunked_prefill:
            if "token_embed" in supported_tasks:
                supported_tasks.remove("token_embed")
            if "token_classify" in supported_tasks:
                supported_tasks.remove("token_classify")

            logger.debug_once(
                "Chunked prefill is not supported with "
                "token_embed and token_classify tasks "
                "which using ALL pooling. "
                "Please turn off chunked prefill by "
                "`--no-enable-chunked-prefill` before using it."
            )

        if "score" in supported_tasks:
            num_labels = getattr(self.model_config.hf_config, "num_labels", 0)
            if num_labels != 1:
                supported_tasks.remove("score")
                logger.debug_once("Score API is only enabled for num_labels == 1.")

        return supported_tasks

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def sync_and_slice_intermediate_tensors(
        self,
        num_tokens: int,
        intermediate_tensors: IntermediateTensors | None,
        sync_self: bool,
    ) -> IntermediateTensors:
        assert self.intermediate_tensors is not None

        tp = self.vllm_config.parallel_config.tensor_parallel_size
        is_rs = is_residual_scattered_for_sp(self.vllm_config, num_tokens)

        # When sequence parallelism is enabled, the "residual" tensor is sharded
        # across tensor parallel ranks, so each rank only needs its own slice.
        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                is_scattered = k == "residual" and is_rs
                copy_len = num_tokens // tp if is_scattered else num_tokens
                self.intermediate_tensors[k][:copy_len].copy_(
                    v[:copy_len], non_blocking=True
                )

        return IntermediateTensors(
            {
                k: v[: num_tokens // tp]
                if k == "residual" and is_rs
                else v[:num_tokens]
                for k, v in self.intermediate_tensors.items()
            }
        )

    def eplb_step(self, is_dummy: bool = False, is_profile: bool = False) -> None:
        """
        Step for the EPLB (Expert Parallelism Load Balancing) state.
        """
        if not self.parallel_config.enable_eplb:
            return

        assert self.eplb_state is not None
        model = self.get_model()
        assert is_mixture_of_experts(model)
        self.eplb_state.step(
            is_dummy,
            is_profile,
            log_stats=self.parallel_config.eplb_config.log_balancedness,
        )

    # This is where the second ubatch is adjusted to account for the padding.
    # Should be called after attention metadata creation. This just pads
    # the second ubatch slice out to the total number of tokens
    # (num_tokens + padding)
    @staticmethod
    def pad_out_ubatch_slice(ubatch_slices: UBatchSlices, num_total_tokens: int):
        padded_second_ubatch_slice = slice(
            ubatch_slices[1].token_slice.start, num_total_tokens
        )
        ubatch_slices[1] = UBatchSlice(
            padded_second_ubatch_slice, padded_second_ubatch_slice
        )

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
    ) -> ModelRunnerOutput:
        assert self.input_batch.num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )

        hidden_states = hidden_states[:num_scheduled_tokens]
        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np.tolist(), device=hidden_states.device
        )
        seq_lens_cpu = self.seq_lens.cpu[: self.input_batch.num_reqs]

        model = cast(VllmModelForPooling, self.model)
        raw_pooler_output: PoolerOutput = model.pooler(
            hidden_states=hidden_states,
            pooling_metadata=pooling_metadata,
        )
        raw_pooler_output = json_map_leaves(
            lambda x: x.to("cpu", non_blocking=True),
            raw_pooler_output,
        )
        self._sync_device()

        pooler_output: list[torch.Tensor | None] = []
        for raw_output, seq_len, prompt_len in zip(
            raw_pooler_output, seq_lens_cpu, pooling_metadata.prompt_lens
        ):
            output = raw_output if seq_len == prompt_len else None
            pooler_output.append(output)

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
        )

    def _get_num_input_tokens(self, num_scheduled_tokens: int) -> int:
        if (
            self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
            and hasattr(self, "cudagraph_batch_sizes")
            and self.cudagraph_batch_sizes
            and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]
        ):
            # Use CUDA graphs.
            # Add padding to the batch size.
            return self.vllm_config.pad_for_cudagraph(num_scheduled_tokens)

        # Eager mode.
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if (
            self.compilation_config.pass_config.enable_sequence_parallelism
            and tp_size > 1
        ):
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,  # Padded
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
        dict[str, Any],
        ECConnectorOutput | None,
    ]:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        is_first_rank = get_pp_group().is_first_rank

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        ec_connector_output = None

        if (
            self.supports_mm_inputs
            and is_first_rank
            and not self.model_config.is_encoder_decoder
        ):
            # Run the multimodal encoder if any.
            with self.maybe_get_ec_connector_output(
                scheduler_output,
                encoder_cache=self.encoder_cache,
            ) as ec_connector_output:
                self._execute_mm_encoder(scheduler_output)
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds_scheduled = self.model.embed_input_ids(
                self.input_ids.gpu[:num_scheduled_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)

            input_ids = None
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = {
                **self._init_model_kwargs(num_scheduled_tokens),
                **self._extract_mm_kwargs(scheduler_output),
            }
        elif self.enable_prompt_embeds and is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the CUDA graph all the time. The v0
            # engine avoids this by "double compiling" the CUDA graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the CUDA graph will be more performant (like in the else case
            # below).
            token_ids_idx = (
                self.is_token_ids.gpu[:num_scheduled_tokens]
                .nonzero(as_tuple=False)
                .squeeze(1)
            )
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs(num_input_tokens)
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs(num_input_tokens)

        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )

        if (
            self.model_config.is_encoder_decoder
            and scheduler_output.scheduled_encoder_inputs
        ):
            # Run the encoder, just like we do with other multimodal inputs.
            # For an encoder-decoder model, our processing here is a bit
            # simpler, because the outputs are just passed to the decoder.
            # We are not doing any prompt replacement. We also will only
            # ever have a single encoder input.
            encoder_outputs = self._execute_mm_encoder(scheduler_output)
            model_kwargs.update({"encoder_outputs": encoder_outputs})

        return (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        )

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        with uvm_enabled_allocation_phase("enabled:sampler"):
            if spec_decode_metadata is None:
                # Update output token ids with tokens sampled in last step
                # if async scheduling and required by current sampling params.
                self.input_batch.update_async_output_token_ids()
                return self.sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )

            sampler_output = self.rejection_sampler(
                spec_decode_metadata,
                None,  # draft_probs
                logits,
                sampling_metadata,
            )
        self._update_states_after_model_execute(sampler_output.sampled_token_ids)
        return sampler_output

    def _uvm_enabled_step_phase(self, scheduler_output: "SchedulerOutput") -> str:
        """Return a finer UVM phase for the current enabled serving step."""
        if scheduler_output.total_num_scheduled_tokens <= 0:
            return "enabled:idle_step"

        new_request_ids = {
            request.req_id for request in scheduler_output.scheduled_new_reqs
        }
        token_counts = scheduler_output.num_scheduled_tokens
        has_prefill_like_work = bool(new_request_ids) or any(
            count > self.uniform_decode_query_len for count in token_counts.values()
        )
        has_decode_like_work = any(
            req_id not in new_request_ids
            and 0 < count <= self.uniform_decode_query_len
            for req_id, count in token_counts.items()
        )

        if has_prefill_like_work and has_decode_like_work:
            return "enabled:mixed_step"
        if has_prefill_like_work:
            return "enabled:prefill_step"
        return "enabled:decode_step"

    def _uvm_prefill_p3_decode_remote_enabled(self) -> bool:
        return self._env_flag_enabled(
            "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ", default=False
        )

    def _poison_uvm_prefill_p3_decode_remote(self, reason: str) -> None:
        self._uvm_prefill_p3_decode_remote_state = "poisoned"
        self._uvm_prefill_p3_decode_remote_poison_reason = str(reason)
        model = self.get_model()
        for module in model.modules():
            if getattr(module, "uvm_expert_offload_access_mode", None) in (
                "slot_restore",
                "remote_read",
            ):
                # No subsequent fused-MoE forward may use either a stale slot
                # mapping or a partially established remote mapping.
                module.uvm_expert_offload_access_mode = "poisoned"

    def _switch_uvm_prefill_p3_to_decode_remote_read(self) -> dict[str, Any]:
        """Close P3 and establish remote READ PTEs before the first decode.

        The feature is deliberately scoped to one eager, single-GPU request.
        The device synchronization is the proof that the last prefill kernel
        stopped touching slot backing; close and remote-map then run before
        any decode kernel can be submitted.
        """
        started_ns = time.perf_counter_ns()
        if getattr(self, "_uvm_prefill_p3_decode_remote_state", None) == "poisoned":
            raise RuntimeError(
                "P3-to-remote transition is poisoned: "
                + str(
                    getattr(
                        self,
                        "_uvm_prefill_p3_decode_remote_poison_reason",
                        "unknown",
                    )
                )
            )

        modules: list[Any] = []
        all_region_ids: list[str] = []
        module_experts: list[tuple[Any, set[int]]] = []
        for module in self.get_model().modules():
            if getattr(module, "uvm_expert_offload_access_mode", None) != "slot_restore":
                continue
            expert_ids = set(
                int(value)
                for value in getattr(module, "_uvm_offloaded_global_expert_ids", set())
            )
            if not expert_ids:
                continue
            layer_id_fn = getattr(module, "_uvm_layer_id", None)
            layer_id = layer_id_fn() if callable(layer_id_fn) else None
            records = getattr(module, "_uvm_expert_records", None)
            if layer_id is None or not isinstance(records, dict):
                self._poison_uvm_prefill_p3_decode_remote(
                    "cold expert metadata is incomplete"
                )
                raise RuntimeError("P3-to-remote cold expert metadata is incomplete")
            region_ids: list[str] = []
            for expert_id in sorted(expert_ids):
                record = records.get((int(layer_id), int(expert_id)))
                lanes = getattr(record, "envelope_region_ids", None)
                if not lanes:
                    self._poison_uvm_prefill_p3_decode_remote(
                        f"missing region lanes for layer={layer_id}, expert={expert_id}"
                    )
                    raise RuntimeError("P3-to-remote region lanes are incomplete")
                region_ids.extend(str(value) for value in lanes)
            modules.append(module)
            module_experts.append((module, expert_ids))
            all_region_ids.extend(region_ids)

        all_region_ids = list(dict.fromkeys(all_region_ids))
        if not modules or not all_region_ids:
            self._poison_uvm_prefill_p3_decode_remote("no active P3 cold regions")
            raise RuntimeError("P3-to-remote transition has no cold regions")

        device = int(torch.cuda.current_device())
        try:
            torch.cuda.synchronize(device)
            synchronized_ns = time.perf_counter_ns()
            close_result = close_uvm_slot_direct_policy_session(device=device)
            closed_ns = time.perf_counter_ns()
            if not bool(close_result.get("ok")):
                raise RuntimeError(f"P3 session close failed: {close_result!r}")
            driver_close = close_result.get("result")
            abort_audit = (
                driver_close.get("abort")
                if isinstance(driver_close, dict)
                else None
            )
            close_contract_ok = (
                isinstance(abort_audit, dict)
                and abort_audit.get("ok") is True
                and abort_audit.get("contract_ok") is True
                and abort_audit.get("session_cleared") is True
                and int(abort_audit.get("remaining_slot_count", -1)) == 0
                and int(abort_audit.get("remaining_bytes", -1)) == 0
            )
            if not close_contract_ok:
                raise RuntimeError(
                    "P3 close did not prove every slot unmapped: "
                    f"{close_result!r}"
                )
            remote_result = transition_uvm_regions_to_remote_read(
                all_region_ids,
                device=device,
            )
            remote_ns = time.perf_counter_ns()
            if not bool(remote_result.get("ok")):
                raise RuntimeError(
                    f"remote-read mapping failed: {remote_result!r}"
                )

            # All fallible driver work and all metadata validation completed.
            # No model forward runs concurrently with these plain Python state
            # updates, so this is the logical commit point for every layer.
            for module, expert_ids in module_experts:
                module._uvm_remote_read_global_expert_ids = set(expert_ids)
                module._uvm_offloaded_global_expert_ids.clear()
                module.uvm_expert_offload_access_mode = "remote_read"
            self._uvm_slot_init_bank_done = False
            self._uvm_slot_init_bank_result = None
            self._uvm_slot_registered_plan_result = None
            self._uvm_slot_session_contract = None
            self._uvm_prefill_p3_decode_remote_state = "decode_remote_read"
            remote_batch_result = (
                remote_result.get("batch_result")
                if isinstance(remote_result.get("batch_result"), dict)
                else {}
            )
            remote_mapping_ioctl_calls = int(
                remote_batch_result.get("mapping_ioctl_calls", 0)
            )
            remote_total_ioctl_calls = int(
                remote_batch_result.get("ioctl_calls", 0)
            )
            result = {
                "schema": "vllm_prefill_p3_decode_remote_transition_v1",
                "ok": True,
                "reason": "ok",
                "module_count": len(modules),
                "expert_count": sum(len(values) for _, values in module_experts),
                "region_count": len(all_region_ids),
                "sync_ns": synchronized_ns - started_ns,
                "close_ns": closed_ns - synchronized_ns,
                "remote_map_ns": remote_ns - closed_ns,
                "wall_ns": remote_ns - started_ns,
                "close_contract_verified": close_contract_ok,
                "remote_batch": bool(remote_result.get("batch", False)),
                "remote_batch_ioctl_calls": remote_mapping_ioctl_calls,
                "remote_batch_total_ioctl_calls": remote_total_ioctl_calls,
                "remote_batch_fd_probe_ioctl_calls": (
                    remote_total_ioctl_calls - remote_mapping_ioctl_calls
                ),
                "remote_batch_fd_candidate_attempts": int(
                    remote_batch_result.get("fd_candidate_attempts", 0)
                ),
                "remote_mapped_bytes": int(
                    remote_batch_result.get("remote_mapped_bytes", 0)
                ),
            }
            result_path = os.environ.get(
                "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_RESULT_FILE", ""
            ).strip()
            if result_path:
                absolute = os.path.abspath(result_path)
                os.makedirs(os.path.dirname(absolute), exist_ok=True)
                temporary = f"{absolute}.tmp.{os.getpid()}"
                try:
                    with open(temporary, "w", encoding="utf-8") as handle:
                        json.dump(result, handle, indent=2, sort_keys=True)
                        handle.write("\n")
                    os.replace(temporary, absolute)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
            self._uvm_prefill_p3_decode_remote_result = result
            logger.info("P3 prefill to remote-read decode transition: %s", result)
            return result
        except Exception as exc:
            self._poison_uvm_prefill_p3_decode_remote(repr(exc))
            raise RuntimeError(
                "P3 prefill to remote-read decode transition failed; "
                "the worker is poisoned and must be torn down"
            ) from exc

    def _maybe_switch_uvm_prefill_p3_to_decode_remote_read(
        self,
        scheduler_output: "SchedulerOutput",
        step_phase: str,
    ) -> None:
        if not self._uvm_prefill_p3_decode_remote_enabled():
            return
        state = getattr(
            self, "_uvm_prefill_p3_decode_remote_state", "prefill_p3"
        )
        if state == "poisoned":
            raise RuntimeError(
                "P3-to-remote worker is poisoned: "
                + str(
                    getattr(
                        self,
                        "_uvm_prefill_p3_decode_remote_poison_reason",
                        "unknown",
                    )
                )
            )
        if step_phase == "enabled:mixed_step":
            self._poison_uvm_prefill_p3_decode_remote(
                "mixed prefill/decode batches are unsupported"
            )
            raise RuntimeError(
                "P3 prefill/remote-read decode requires an unmixed single request"
            )
        if step_phase == "enabled:prefill_step":
            if state != "prefill_p3":
                self._poison_uvm_prefill_p3_decode_remote(
                    "a prefill step arrived after decode cutover"
                )
                raise RuntimeError("P3 prefill arrived after remote-read cutover")
            self._uvm_prefill_p3_seen = True
            return
        if step_phase != "enabled:decode_step":
            return
        if state == "decode_remote_read":
            return
        if not getattr(self, "_uvm_prefill_p3_seen", False):
            self._poison_uvm_prefill_p3_decode_remote(
                "decode arrived before a P3 prefill step"
            )
            raise RuntimeError("P3-to-remote decode arrived before prefill")
        if len(scheduler_output.num_scheduled_tokens) != 1:
            self._poison_uvm_prefill_p3_decode_remote(
                "decode cutover requires exactly one request"
            )
            raise RuntimeError("P3-to-remote decode cutover requires one request")
        self._switch_uvm_prefill_p3_to_decode_remote_read()

    def _poison_uvm_original_va_phase(self, reason: str) -> None:
        from vllm.model_executor.models.utils import (
            poison_cpu_offload_uvm_original_va_phase,
        )

        self._uvm_original_va_phase_poison_reason = str(reason)
        poison_cpu_offload_uvm_original_va_phase(str(reason))

    def _write_uvm_original_va_phase_result(self, result: dict[str, Any]) -> None:
        result_path = os.environ.get(
            "VLLM_UVM_ORIGINAL_VA_PHASE_RESULT_FILE", ""
        ).strip()
        if not result_path:
            return
        absolute = os.path.abspath(result_path)
        os.makedirs(os.path.dirname(absolute), exist_ok=True)
        temporary = f"{absolute}.tmp.{os.getpid()}"
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, absolute)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _destroy_uvm_original_va_alias_prefill_graphs(self) -> dict[str, Any]:
        """Reset only non-uniform alias Graphs before cmd99 frees backing."""
        from vllm.model_executor.models.utils import (
            certify_cpu_offload_uvm_original_va_alias_graphs_destroyed,
            get_cpu_offload_graph_static_alias_diagnostics,
        )

        model = self.model
        if not isinstance(model, CUDAGraphWrapper):
            raise RuntimeError(
                "Selective alias Graph teardown requires a CUDAGraphWrapper"
            )

        def record(descriptor: BatchDescriptor) -> dict[str, Any]:
            return {
                "num_tokens": int(descriptor.num_tokens),
                "uniform_decode": bool(descriptor.uniform_decode),
                "has_lora": bool(descriptor.has_lora),
            }

        entries = model.concrete_cudagraph_entries
        alias_descriptors = sorted(
            (
                descriptor
                for descriptor, entry in entries.items()
                if not descriptor.uniform_decode and entry.cudagraph is not None
            ),
            key=tuple,
        )
        remote_descriptors = sorted(
            (
                descriptor
                for descriptor, entry in entries.items()
                if descriptor.uniform_decode and entry.cudagraph is not None
            ),
            key=tuple,
        )
        diagnostics = get_cpu_offload_graph_static_alias_diagnostics()
        partition = diagnostics.get("dual_graph_partition_certificate")
        if not isinstance(partition, dict):
            raise RuntimeError("Dual Graph partition certificate is absent")
        expected_alias = tuple(tuple(item) for item in partition.get("alias_entries", ()))
        expected_remote = tuple(
            tuple(item) for item in partition.get("remote_entries", ())
        )
        actual_alias = tuple(
            sorted(
                (
                    descriptor.num_tokens,
                    descriptor.uniform_decode,
                    descriptor.has_lora,
                )
                for descriptor in alias_descriptors
            )
        )
        actual_remote = tuple(
            sorted(
                (
                    descriptor.num_tokens,
                    descriptor.uniform_decode,
                    descriptor.has_lora,
                )
                for descriptor in remote_descriptors
            )
        )
        if actual_alias != expected_alias or actual_remote != expected_remote:
            raise RuntimeError(
                "Live CUDA Graph entries differ from the dual-topology "
                "capture certificate"
            )

        torch.cuda.synchronize(self.device)
        destroyed_records: list[dict[str, Any]] = []
        for descriptor in alias_descriptors:
            entry = entries[descriptor]
            graph = entry.cudagraph
            if graph is None:
                raise RuntimeError("Alias Graph disappeared during teardown")
            graph.reset()
            entry.cudagraph = None
            entry.output = None
            entry.input_addresses = None
            destroyed_records.append(record(descriptor))
            del entries[descriptor]
        gc.collect()
        torch.cuda.synchronize(self.device)

        remaining_records = [record(value) for value in remote_descriptors]
        result = certify_cpu_offload_uvm_original_va_alias_graphs_destroyed(
            destroyed_alias_entries=destroyed_records,
            remaining_remote_entries=remaining_records,
        )
        result["live_entry_count_after"] = len(entries)
        return result

    def _collect_uvm_runtime_remote_expert_ranges(
        self,
    ) -> tuple[tuple[int, int], ...]:
        """Return live incremental remote-expert ranges outside the alias plan."""
        if not self._env_flag_enabled(
            "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE",
            default=False,
        ):
            return ()

        ranges: set[tuple[int, int]] = set()
        for module in self.get_model().modules():
            snapshot_fn = getattr(
                module,
                "get_uvm_init_weight_placement_snapshot",
                None,
            )
            if not callable(snapshot_fn):
                continue
            snapshot = snapshot_fn()
            owners = (
                snapshot.get("region_owners")
                if isinstance(snapshot, dict)
                else None
            )
            if not isinstance(owners, list):
                continue
            for owner in owners:
                if not isinstance(owner, dict) or owner.get("placement") != "remote":
                    continue
                region_id = owner.get("region_id")
                if not isinstance(region_id, str):
                    raise RuntimeError("Remote expert region has no stable identity")
                try:
                    _prefix, base_text, length_text = region_id.rsplit(":", 2)
                    base = int(base_text, 16)
                    length = int(length_text, 16)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Malformed remote expert region identity: {region_id!r}"
                    ) from exc
                if base <= 0 or length <= 0:
                    raise RuntimeError(
                        f"Invalid remote expert region geometry: {region_id!r}"
                    )
                ranges.add((base, length))
        return tuple(sorted(ranges))

    def _maybe_switch_uvm_original_va_prefill_to_decode_remote(
        self,
        scheduler_output: "SchedulerOutput",
        step_phase: str,
    ) -> None:
        from vllm.model_executor.models.utils import (
            activate_cpu_offload_uvm_original_va_prefill_topology,
            cpu_offload_uvm_original_va_park_resume_enabled,
            cpu_offload_uvm_original_va_phase_mode,
            cpu_offload_uvm_original_va_resume_token_threshold,
            cpu_offload_uvm_original_va_token_threshold,
            get_cpu_offload_graph_static_alias_diagnostics,
            transition_cpu_offload_uvm_original_va_to_remote,
        )

        mode = cpu_offload_uvm_original_va_phase_mode()
        park_resume = cpu_offload_uvm_original_va_park_resume_enabled()
        if mode == "off":
            return
        if self._uvm_original_va_phase_poison_reason is not None:
            raise RuntimeError(
                "Original-VA phase worker is poisoned: "
                + self._uvm_original_va_phase_poison_reason
            )

        diagnostics = get_cpu_offload_graph_static_alias_diagnostics()
        if mode == "remote_only":
            if not (
                diagnostics.get("lifecycle_state") == "remote"
                and diagnostics.get("remote_map_active") is True
                and int(diagnostics.get("session_id", -1)) == 0
                and int(diagnostics.get("active_driver_backing_bytes", -1)) == 0
            ):
                self._poison_uvm_original_va_phase(
                    "remote_only serving step observed a non-remote topology"
                )
                raise RuntimeError(
                    "remote_only serving requires audited remote original-VA PTEs"
                )
            return

        if mode == "token_threshold":
            if not park_resume:
                self._poison_uvm_original_va_phase(
                    "token-threshold mode requires PARK/RESUME"
                )
                raise RuntimeError(
                    "Original-VA token-threshold mode requires PARK/RESUME"
                )
            total_num_scheduled_tokens = int(
                scheduler_output.total_num_scheduled_tokens
            )
            if total_num_scheduled_tokens <= 0:
                self._poison_uvm_original_va_phase(
                    "token-threshold mode observed a non-positive token count"
                )
                raise RuntimeError(
                    "Original-VA token-threshold mode requires a positive "
                    "total_num_scheduled_tokens"
                )
            lifecycle_state = diagnostics.get("lifecycle_state")
            park_token_threshold = (
                cpu_offload_uvm_original_va_token_threshold()
            )
            resume_token_threshold = (
                cpu_offload_uvm_original_va_resume_token_threshold()
            )
            if lifecycle_state == "sealed":
                target_remote = (
                    total_num_scheduled_tokens <= park_token_threshold
                )
                selection_reason = (
                    "park_threshold"
                    if target_remote
                    else "copy_hold"
                )
            elif lifecycle_state == "remote_parked":
                target_remote = (
                    total_num_scheduled_tokens < resume_token_threshold
                )
                selection_reason = (
                    "remote_hold"
                    if target_remote
                    else "resume_threshold"
                )
            else:
                target_remote = False
                selection_reason = "invalid_state"
            transition_result: dict[str, Any] | None = None
            try:
                if target_remote:
                    if lifecycle_state == "sealed":
                        transition_kwargs: dict[str, Any] = {
                            "reason": "token_threshold_remote",
                            "preserve_remote_graphs": False,
                        }
                        runtime_remote_ranges = (
                            self._collect_uvm_runtime_remote_expert_ranges()
                        )
                        if runtime_remote_ranges:
                            transition_kwargs["runtime_remote_ranges"] = (
                                runtime_remote_ranges
                            )
                        transition_result = (
                            transition_cpu_offload_uvm_original_va_to_remote(
                                **transition_kwargs,
                            )
                        )
                    elif lifecycle_state != "remote_parked":
                        raise RuntimeError(
                            "remote selection requires sealed or "
                            f"remote_parked state, got {lifecycle_state!r}"
                        )
                else:
                    if lifecycle_state == "remote_parked":
                        transition_result = (
                            activate_cpu_offload_uvm_original_va_prefill_topology()
                        )
                        if transition_result.get("ok") is not True:
                            raise RuntimeError(
                                "PARK/RESUME did not restore the sealed topology"
                            )
                    elif lifecycle_state != "sealed":
                        raise RuntimeError(
                            "copy selection requires remote_parked or "
                            f"sealed state, got {lifecycle_state!r}"
                        )
            except Exception as exc:
                reason = (
                    "token-threshold topology switch failed: "
                    f"tokens={total_num_scheduled_tokens}, "
                    f"park_threshold={park_token_threshold}, "
                    f"resume_threshold={resume_token_threshold}, target="
                    f"{'remote' if target_remote else 'copy'}, error={exc!r}"
                )
                self._poison_uvm_original_va_phase(reason)
                raise RuntimeError(reason) from exc

            after = get_cpu_offload_graph_static_alias_diagnostics()
            if target_remote:
                valid = (
                    after.get("lifecycle_state") == "remote_parked"
                    and after.get("remote_map_active") is True
                    and int(after.get("session_id", 0)) > 0
                    and int(after.get("active_driver_backing_bytes", 0)) > 0
                )
            else:
                valid = (
                    after.get("lifecycle_state") == "sealed"
                    and after.get("sealed") is True
                    and after.get("remote_map_active") is False
                    and int(after.get("session_id", 0)) > 0
                    and int(after.get("active_driver_backing_bytes", 0)) > 0
                )
            if not valid:
                reason = (
                    "token-threshold post-switch topology contract failed: "
                    f"tokens={total_num_scheduled_tokens}, "
                    f"park_threshold={park_token_threshold}, "
                    f"resume_threshold={resume_token_threshold}, target="
                    f"{'remote' if target_remote else 'copy'}"
                )
                self._poison_uvm_original_va_phase(reason)
                raise RuntimeError(reason)
            self._uvm_original_va_phase_transition_result = {
                "schema": "vllm_uvm_original_va_token_threshold_v2",
                "ok": True,
                "phase_mode": mode,
                "step_phase": step_phase,
                "total_num_scheduled_tokens": total_num_scheduled_tokens,
                "token_threshold": park_token_threshold,
                "park_token_threshold": park_token_threshold,
                "resume_token_threshold": resume_token_threshold,
                "selection_reason": selection_reason,
                "selected_access": "remote" if target_remote else "copy",
                "topology_changed": transition_result is not None,
                "transition": transition_result,
                "diagnostics_after": after,
            }
            return

        if mode not in (
            "prefill_remote_decode",
            "prefill_graph_remote_decode",
        ):
            self._poison_uvm_original_va_phase(f"unknown phase mode {mode!r}")
            raise RuntimeError(f"Unsupported original-VA phase mode {mode!r}")
        if step_phase == "enabled:mixed_step":
            self._poison_uvm_original_va_phase(
                "mixed prefill/decode batches are unsupported"
            )
            raise RuntimeError(
                "Original-VA alias-prefill/remote-decode requires an unmixed request"
            )
        if step_phase == "enabled:prefill_step":
            if self._uvm_original_va_phase_decode_cutover_done:
                if not park_resume:
                    self._poison_uvm_original_va_phase(
                        "a prefill step arrived after remote decode cutover"
                    )
                    raise RuntimeError(
                        "Original-VA prefill arrived after one-way remote cutover"
                    )
                resumed = activate_cpu_offload_uvm_original_va_prefill_topology()
                if resumed.get("ok") is not True:
                    self._poison_uvm_original_va_phase(
                        "parked aliases failed to resume for a new prefill"
                    )
                    raise RuntimeError(
                        "Original-VA parked aliases failed to resume"
                    )
                self._uvm_original_va_phase_decode_cutover_done = False
                diagnostics = get_cpu_offload_graph_static_alias_diagnostics()
            if not (
                diagnostics.get("lifecycle_state") == "sealed"
                and diagnostics.get("sealed") is True
                and int(diagnostics.get("session_id", 0)) > 0
                and int(diagnostics.get("active_driver_backing_bytes", 0)) > 0
            ):
                self._poison_uvm_original_va_phase(
                    "prefill step observed a non-sealed alias topology"
                )
                raise RuntimeError(
                    "Original-VA prefill requires sealed static-alias banks"
                )
            self._uvm_original_va_phase_prefill_seen = True
            return
        if step_phase != "enabled:decode_step":
            return
        if self._uvm_original_va_phase_decode_cutover_done:
            expected_remote_state = "remote_parked" if park_resume else "remote"
            if not (
                diagnostics.get("lifecycle_state") == expected_remote_state
                and diagnostics.get("remote_map_active") is True
                and (
                    int(diagnostics.get("session_id", 0)) > 0
                    if park_resume
                    else int(diagnostics.get("session_id", -1)) == 0
                )
            ):
                self._poison_uvm_original_va_phase(
                    "decode topology drifted after remote cutover"
                )
                raise RuntimeError("Original-VA remote decode topology drifted")
            return
        if not self._uvm_original_va_phase_prefill_seen:
            self._poison_uvm_original_va_phase(
                "decode arrived before an alias-prefill step"
            )
            raise RuntimeError("Original-VA remote decode arrived before prefill")
        decode_counts = tuple(scheduler_output.num_scheduled_tokens.values())
        if not decode_counts or any(
            count != self.uniform_decode_query_len for count in decode_counts
        ):
            self._poison_uvm_original_va_phase(
                "first decode cutover requires a strict uniform decode step"
            )
            raise RuntimeError(
                "Original-VA remote decode cutover requires strict uniform decode"
            )

        alias_graph_teardown: dict[str, Any] | None = None
        if mode == "prefill_graph_remote_decode" and not park_resume:
            try:
                alias_graph_teardown = (
                    self._destroy_uvm_original_va_alias_prefill_graphs()
                )
            except Exception as exc:
                self._poison_uvm_original_va_phase(
                    "selective alias-prefill Graph teardown failed"
                )
                raise RuntimeError(
                    "Failed to retire alias-prefill CUDA Graphs before "
                    "original-VA remote cutover"
                ) from exc
            diagnostics = get_cpu_offload_graph_static_alias_diagnostics()

        preserve_remote_graphs = bool(diagnostics.get("remote_graph_captured"))
        transition_kwargs: dict[str, Any] = {
            "reason": "first_decode_cutover",
            "preserve_remote_graphs": preserve_remote_graphs,
        }
        runtime_remote_ranges = self._collect_uvm_runtime_remote_expert_ranges()
        if runtime_remote_ranges:
            transition_kwargs["runtime_remote_ranges"] = runtime_remote_ranges
        transition = transition_cpu_offload_uvm_original_va_to_remote(
            **transition_kwargs,
        )
        after = get_cpu_offload_graph_static_alias_diagnostics()
        expected_remote_state = "remote_parked" if park_resume else "remote"
        if not (
            transition.get("ok") is True
            and after.get("lifecycle_state") == expected_remote_state
            and after.get("remote_map_active") is True
            and (
                int(after.get("session_id", 0)) > 0
                if park_resume
                else int(after.get("session_id", -1)) == 0
            )
            and (
                int(after.get("active_driver_backing_bytes", 0)) > 0
                if park_resume
                else int(after.get("active_driver_backing_bytes", -1)) == 0
            )
        ):
            self._poison_uvm_original_va_phase(
                "post-cutover remote metadata contract failed"
            )
            raise RuntimeError(
                "Original-VA remote decode cutover metadata contract failed"
            )
        result = {
            "schema": "vllm_uvm_original_va_phase_transition_v1",
            "ok": True,
            "phase_mode": mode,
            "park_resume": park_resume,
            "remote_graph_preserved": preserve_remote_graphs,
            "alias_graph_teardown": alias_graph_teardown,
            "transition": transition,
            "diagnostics_after": after,
        }
        self._uvm_original_va_phase_transition_result = result
        self._uvm_original_va_phase_decode_cutover_done = True
        self._write_uvm_original_va_phase_result(result)
        logger.info("Original-VA prefill-to-remote decode transition: %s", result)

    def _validate_uvm_original_va_graph_replay_for_step(
        self,
        *,
        original_va_phase_mode: str,
        enabled_step_phase: str,
        cudagraph_runtime_mode: CUDAGraphMode,
    ) -> None:
        """Validate the live PTE topology against the selected Graph class."""
        if original_va_phase_mode == "off":
            return
        if (
            original_va_phase_mode == "prefill_graph_remote_decode"
            and enabled_step_phase
            in ("enabled:prefill_step", "enabled:decode_step")
            and cudagraph_runtime_mode == CUDAGraphMode.NONE
        ):
            raise RuntimeError(
                "FULL_SEPARATE phase serving cannot fall back to eager execution: "
                f"step_phase={enabled_step_phase}"
            )
        if cudagraph_runtime_mode == CUDAGraphMode.NONE:
            return

        from vllm.model_executor.models.utils import (
            validate_cpu_offload_uvm_original_va_alias_graph_replay,
            validate_cpu_offload_uvm_original_va_remote_graph_replay,
        )

        if (
            original_va_phase_mode == "prefill_graph_remote_decode"
            and enabled_step_phase == "enabled:prefill_step"
        ):
            validate_cpu_offload_uvm_original_va_alias_graph_replay()
        else:
            validate_cpu_offload_uvm_original_va_remote_graph_replay()

    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> tuple[
        dict[str, int],
        LogprobsLists | None,
        list[list[int]],
        dict[str, LogprobsTensors | None],
        list[str],
        dict[str, int],
        list[int],
    ]:
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        discard_sampled_tokens_req_indices = self.discard_request_indices.np[
            : self.num_discarded_requests
        ]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices = []
        cu_num_tokens: list[int] | None = None
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
                # Mask out the sampled tokens that should not be sampled.
                for i in discard_sampled_tokens_req_indices:
                    valid_sampled_token_ids[int(i)].clear()
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids, cu_num_tokens = RejectionSampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                    discard_sampled_tokens_req_indices,
                    return_cu_num_tokens=logprobs_tensors is not None,
                )
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)

            # Cache the sampled tokens on the GPU and avoid CPU sync.
            # These will be copied into input_ids in the next step
            # when preparing inputs.
            # With spec decoding, this is done in propose_draft_token_ids().
            if self.input_batch.prev_sampled_token_ids is None:
                assert sampled_token_ids.shape[-1] == 1
                self.input_batch.prev_sampled_token_ids = sampled_token_ids
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        logprobs_lists = (
            logprobs_tensors.tolists(cu_num_tokens)
            if not self.use_async_scheduling and logprobs_tensors is not None
            else None
        )

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    @contextmanager
    def synchronize_input_prep(self):
        if self.prepare_inputs_event is None:
            yield
            return

        # Ensure prior step has finished with reused CPU tensors.
        # This is required in the async scheduling case because
        # the CPU->GPU transfer happens async.
        self.prepare_inputs_event.synchronize()
        try:
            yield
        finally:
            self.prepare_inputs_event.record()

    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ) -> Any:
        """Helper method to call the model forward pass.

        This method can be overridden by subclasses for model execution.
        Motivation: We can inspect only this method versus
        the whole execute_model, which has additional logic.

        Args:
            input_ids: Input token IDs
            positions: Token positions
            intermediate_tensors: Tensors from previous pipeline stages
            inputs_embeds: Input embeddings (alternative to input_ids)
            **model_kwargs: Additional model arguments

        Returns:
            Model output tensor
        """
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        prefill_pregraph_timing_audit = self._env_flag_enabled(
            "VLLM_PREFILL_PREGRAPH_TIMING_AUDIT", default=False
        )
        pregraph_execute_started_ns = time.perf_counter_ns()

        # self._draft_token_ids is None when `input_fits_in_drafter=False`
        # and there is no draft tokens scheduled. so it need to update the
        # spec_decoding info in scheduler_output with async_scheduling.
        # use deepcopy to avoid the modification has influence on the
        # scheduler_output in engine core process.
        # TODO(Ronald1995): deepcopy is expensive when there is a large
        # number of requests, optimize it later.
        if (
            self.use_async_scheduling
            and self.num_spec_tokens
            and self._draft_token_ids is None
        ):
            scheduler_output = deepcopy(scheduler_output)

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with record_function_or_nullcontext("gpu_model_runner: preprocess"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                self._update_states(scheduler_output)
                after_update_states_ns = time.perf_counter_ns()

                if has_ec_transfer() and get_ec_transfer().is_producer:
                    with self.maybe_get_ec_connector_output(
                        scheduler_output,
                        encoder_cache=self.encoder_cache,
                    ) as ec_connector_output:
                        self._execute_mm_encoder(scheduler_output)
                        return make_empty_encoder_model_runner_output(scheduler_output)

                if not num_scheduled_tokens:
                    if (
                        self.parallel_config.distributed_executor_backend
                        == "external_launcher"
                        and self.parallel_config.data_parallel_size > 1
                    ):
                        # this is a corner case when both external launcher
                        # and DP are enabled, num_scheduled_tokens could be
                        # 0, and has_unfinished_requests in the outer loop
                        # returns True. before returning early here we call
                        # dummy run to ensure coordinate_batch_across_dp
                        # is called into to avoid out of sync issues.
                        self._dummy_run(self._get_num_input_tokens(1))
                    if not has_kv_transfer_group():
                        # Return empty ModelRunnerOutput if no work to do.
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(
                        scheduler_output, self.vllm_config
                    )
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs"
                    )

                num_reqs = self.input_batch.num_reqs
                req_ids = self.input_batch.req_ids
                tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
                num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
                max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())

                before_prepare_inputs_ns = time.perf_counter_ns()
                (
                    logits_indices,
                    spec_decode_metadata,
                    ubatch_slices,
                    num_tokens_across_dp,
                ) = self._prepare_inputs(
                    scheduler_output, num_scheduled_tokens_np, max_num_scheduled_tokens
                )
                after_prepare_inputs_ns = time.perf_counter_ns()

                cascade_attn_prefix_lens = None
                # Disable cascade attention when using microbatching (DBO)
                if self.cascade_attn_enabled and ubatch_slices is None:
                    # Pre-compute cascade attention prefix lengths
                    # NOTE: Must be AFTER _prepare_inputs uses self.input_batch state
                    cascade_attn_prefix_lens = self._compute_cascade_attn_prefix_lens(
                        num_scheduled_tokens_np,
                        scheduler_output.num_common_prefix_blocks,
                    )

                # TODO(lucas): move cudagraph dispatching here:
                #   https://github.com/vllm-project/vllm/issues/23789

                total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
                use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
                before_attention_metadata_ns = time.perf_counter_ns()
                attn_metadata, spec_decode_common_attn_metadata = (
                    self._build_attention_metadata(
                        total_num_scheduled_tokens=total_num_scheduled_tokens,
                        max_num_scheduled_tokens=max_num_scheduled_tokens,
                        num_reqs=num_reqs,
                        ubatch_slices=ubatch_slices,
                        logits_indices=logits_indices,
                        use_spec_decode=use_spec_decode,
                        num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
                        cascade_attn_prefix_lens=cascade_attn_prefix_lens,
                    )
                )
                after_attention_metadata_ns = time.perf_counter_ns()

                dp_rank = self.parallel_config.data_parallel_rank
                if ubatch_slices:
                    assert num_tokens_across_dp is not None
                    num_input_tokens = int(num_tokens_across_dp[dp_rank].item())
                    self.pad_out_ubatch_slice(ubatch_slices, num_input_tokens)
                elif num_tokens_across_dp is not None:
                    num_input_tokens = int(num_tokens_across_dp[dp_rank].item())
                else:
                    num_input_tokens = self._get_num_input_tokens(
                        scheduler_output.total_num_scheduled_tokens
                    )

                before_preprocess_ns = time.perf_counter_ns()
                (
                    input_ids,
                    inputs_embeds,
                    positions,
                    intermediate_tensors,
                    model_kwargs,
                    ec_connector_output,
                ) = self._preprocess(
                    scheduler_output, num_input_tokens, intermediate_tensors
                )
                after_preprocess_ns = time.perf_counter_ns()

            enabled_step_phase = self._uvm_enabled_step_phase(scheduler_output)
            uniform_decode = (
                max_num_scheduled_tokens == self.uniform_decode_query_len
            ) and (num_scheduled_tokens == num_reqs * max_num_scheduled_tokens)
            batch_desc = BatchDescriptor(
                num_tokens=num_input_tokens,
                uniform_decode=uniform_decode,
                has_lora=len(self.input_batch.lora_id_to_lora_request) > 0,
            )
            cudagraph_runtime_mode, batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(
                    batch_desc,
                    use_cascade_attn=cascade_attn_prefix_lens is not None,
                )
            )
            from vllm.model_executor.models.utils import (
                cpu_offload_uvm_original_va_phase_mode,
            )

            original_va_phase_mode = cpu_offload_uvm_original_va_phase_mode()
            if (
                original_va_phase_mode != "off"
                and original_va_phase_mode
                not in ("prefill_graph_remote_decode", "remote_only")
                and enabled_step_phase != "enabled:decode_step"
            ):
                # FULL_DECODE_ONLY can classify a one-token new request as a
                # uniform decode shape. A remote Graph must never replay while
                # alias-prefill PTEs are active, so phase wins over shape. In
                # remote_only mode the audited remote topology is immutable for
                # the whole serving lifetime, therefore Prefill FULL replay is
                # safe and is validated again immediately before model forward.
                cudagraph_runtime_mode = CUDAGraphMode.NONE

        # Set cudagraph mode to none if calc_kv_scales is true.
        # KV scales calculation involves dynamic operations that are incompatible
        # with CUDA graph capture.
        if self.calculate_kv_scales:
            cudagraph_runtime_mode = CUDAGraphMode.NONE
            # Mark KV scales as calculated after the first forward pass
            self.calculate_kv_scales = False

        after_dispatch_ns = time.perf_counter_ns()

        if self._env_flag_enabled(
            "VLLM_CUDAGRAPH_DISPATCH_AUDIT", default=False
        ):
            dispatched_tokens = (
                int(batch_descriptor.num_tokens)
                if batch_descriptor is not None
                else None
            )
            audit_key = (
                enabled_step_phase,
                int(batch_desc.num_tokens),
                bool(batch_desc.uniform_decode),
                cudagraph_runtime_mode.name,
                dispatched_tokens,
            )
            seen = getattr(self, "_cudagraph_dispatch_audit_seen", None)
            if seen is None:
                seen = self._cudagraph_dispatch_audit_seen = set()
            if audit_key not in seen:
                seen.add(audit_key)
                logger.info(
                    "CUDA_GRAPH_RUNTIME_AUDIT phase=%s requested_tokens=%d "
                    "uniform_decode=%s runtime_mode=%s dispatched_tokens=%s",
                    enabled_step_phase,
                    int(batch_desc.num_tokens),
                    bool(batch_desc.uniform_decode),
                    cudagraph_runtime_mode.name,
                    dispatched_tokens,
                )

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        prefill_control_timing_audit = self._env_flag_enabled(
            "VLLM_UVM_PREFILL_CONTROL_TIMING_AUDIT", default=False
        ) and enabled_step_phase == "enabled:prefill_step"
        prefill_gpu_event_timing_audit = self._env_flag_enabled(
            "VLLM_PREFILL_GPU_EVENT_TIMING_AUDIT", default=False
        ) and enabled_step_phase == "enabled:prefill_step"
        if prefill_gpu_event_timing_audit:
            gpu_timing_events = getattr(self, "_prefill_gpu_timing_events", None)
            if gpu_timing_events is None:
                gpu_timing_events = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                self._prefill_gpu_timing_events = gpu_timing_events
        control_started_ns = time.perf_counter_ns()
        self._maybe_switch_uvm_original_va_prefill_to_decode_remote(
            scheduler_output,
            enabled_step_phase,
        )
        after_original_va_switch_ns = time.perf_counter_ns()
        self._maybe_switch_uvm_prefill_p3_to_decode_remote_read(
            scheduler_output,
            enabled_step_phase,
        )
        after_p3_switch_ns = time.perf_counter_ns()
        self._validate_uvm_original_va_graph_replay_for_step(
            original_va_phase_mode=original_va_phase_mode,
            enabled_step_phase=enabled_step_phase,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
        )
        after_replay_validation_ns = time.perf_counter_ns()
        if prefill_gpu_event_timing_audit:
            gpu_timing_events[0].record()
        with (
            uvm_allocation_phase(enabled_step_phase),
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                ubatch_slices=ubatch_slices,
                moe_token_trace_metadata=self._moe_token_trace_metadata,
            ),
            record_function_or_nullcontext("gpu_model_runner: forward"),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
        ):
            with uvm_enabled_allocation_phase("enabled:model_forward"):
                model_output = self._model_forward(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )
        if prefill_gpu_event_timing_audit:
            gpu_timing_events[1].record()
        after_model_submit_ns = time.perf_counter_ns()
        if prefill_control_timing_audit:
            logger.info(
                "UVM_PREFILL_CONTROL_TIMING original_va_switch_us=%.3f "
                "p3_switch_us=%.3f replay_validation_us=%.3f "
                "context_and_model_submit_us=%.3f total_us=%.3f",
                (after_original_va_switch_ns - control_started_ns) / 1e3,
                (after_p3_switch_ns - after_original_va_switch_ns) / 1e3,
                (after_replay_validation_ns - after_p3_switch_ns) / 1e3,
                (after_model_submit_ns - after_replay_validation_ns) / 1e3,
                (after_model_submit_ns - control_started_ns) / 1e3,
            )
        if (
            prefill_pregraph_timing_audit
            and enabled_step_phase == "enabled:prefill_step"
        ):
            logger.info(
                "PREFILL_PREGRAPH_TIMING update_states_us=%.3f "
                "pre_prepare_us=%.3f prepare_inputs_us=%.3f "
                "pre_attention_us=%.3f attention_metadata_us=%.3f "
                "pre_preprocess_us=%.3f preprocess_us=%.3f "
                "dispatch_us=%.3f phase_switch_validate_us=%.3f "
                "context_and_model_submit_us=%.3f total_to_submit_us=%.3f",
                (after_update_states_ns - pregraph_execute_started_ns) / 1e3,
                (before_prepare_inputs_ns - after_update_states_ns) / 1e3,
                (after_prepare_inputs_ns - before_prepare_inputs_ns) / 1e3,
                (before_attention_metadata_ns - after_prepare_inputs_ns) / 1e3,
                (after_attention_metadata_ns - before_attention_metadata_ns) / 1e3,
                (before_preprocess_ns - after_attention_metadata_ns) / 1e3,
                (after_preprocess_ns - before_preprocess_ns) / 1e3,
                (after_dispatch_ns - after_preprocess_ns) / 1e3,
                (after_replay_validation_ns - control_started_ns) / 1e3,
                (after_model_submit_ns - after_replay_validation_ns) / 1e3,
                (after_model_submit_ns - pregraph_execute_started_ns) / 1e3,
            )

        with (
            uvm_allocation_phase(enabled_step_phase),
            record_function_or_nullcontext("gpu_model_runner: postprocess"),
        ):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    self.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # Return the pooling output.
                    output = self._pool(
                        hidden_states, num_scheduled_tokens, num_scheduled_tokens_np
                    )
                    output.kv_connector_output = kv_connector_output
                    return output

                sample_hidden_states = hidden_states[logits_indices]
                with uvm_enabled_allocation_phase("enabled:compute_logits"):
                    logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.
                assert not self.is_pooling_model

                sample_hidden_states = hidden_states[logits_indices]
                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual": not is_residual_scattered_for_sp(
                            self.vllm_config, num_input_tokens
                        )
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors,
                    )
                    logits = None
                else:
                    with uvm_enabled_allocation_phase("enabled:compute_logits"):
                        logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data: dict[str, Any] = {}
                if logits is not None:
                    with uvm_enabled_allocation_phase("enabled:logits"):
                        model_output_broadcast_data["logits"] = logits.contiguous()

                broadcasted = get_pp_group().broadcast_tensor_dict(
                    model_output_broadcast_data, src=len(get_pp_group().ranks) - 1
                )
                assert broadcasted is not None
                logits = broadcasted["logits"]

        if prefill_gpu_event_timing_audit:
            gpu_timing_events[2].record()

        retain_prefill_graph_metadata = (
            self._env_flag_enabled(
                "VLLM_TRITON_MLA_EXPERIMENTAL_FULL_CUDAGRAPH",
                default=False,
            )
            and enabled_step_phase == "enabled:prefill_step"
            and cudagraph_runtime_mode != CUDAGraphMode.NONE
        )
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            attn_metadata if retain_prefill_graph_metadata else None,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
        )
        self.kv_connector_output = kv_connector_output
        return None

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        sample_timing_audit = self._env_flag_enabled(
            "VLLM_PREFILL_SAMPLE_TIMING_AUDIT", default=False
        )
        sample_started_ns = time.perf_counter_ns()
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            if not kv_connector_output:
                return None  # type: ignore[return-value]

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            _retained_attn_metadata,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # `_retained_attn_metadata` deliberately remains a local reference for
        # the rest of sample_tokens(). Sampling is submitted after model
        # forward on the same launch stream, so by the time this function has
        # finished using its outputs the Prefill Graph can no longer consume
        # those metadata buffers. No host-side stream synchronize is needed.

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        step_phase = self._uvm_enabled_step_phase(scheduler_output)
        before_sample_ns = time.perf_counter_ns()
        with (
            uvm_allocation_phase(step_phase),
            record_function_or_nullcontext("gpu_model_runner: sample"),
        ):
            sampler_output = self._sample(logits, spec_decode_metadata)
        if sample_timing_audit and step_phase == "enabled:prefill_step":
            gpu_timing_events = getattr(self, "_prefill_gpu_timing_events", None)
            if gpu_timing_events is not None:
                gpu_timing_events[3].record()
        after_sample_ns = time.perf_counter_ns()

        self.input_batch.prev_sampled_token_ids = None

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("gpu_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                )

        spec_config = self.speculative_config
        use_padded_batch_for_eagle = (
            spec_config is not None
            and spec_config.use_eagle()
            and not spec_config.disable_padded_drafter_batch
        )
        effective_drafter_max_model_len = self.max_model_len
        if effective_drafter_max_model_len is None:
            effective_drafter_max_model_len = self.model_config.max_model_len
        if (
            spec_config is not None
            and spec_config.draft_model_config is not None
            and spec_config.draft_model_config.max_model_len is not None
        ):
            effective_drafter_max_model_len = (
                spec_config.draft_model_config.max_model_len
            )
        input_fits_in_drafter = spec_decode_common_attn_metadata and (
            spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
            <= effective_drafter_max_model_len
        )
        if use_padded_batch_for_eagle:
            assert self.speculative_config is not None
            assert isinstance(self.drafter, EagleProposer)
            sampled_token_ids = sampler_output.sampled_token_ids
            if input_fits_in_drafter:
                # EAGLE speculative decoding can use the GPU sampled tokens
                # as inputs, and does not need to wait for bookkeeping to finish.
                propose_draft_token_ids(sampled_token_ids)
            elif self.valid_sampled_token_count_event is not None:
                assert spec_decode_common_attn_metadata is not None
                next_token_ids, valid_sampled_tokens_count = (
                    self.drafter.prepare_next_token_ids_padded(
                        spec_decode_common_attn_metadata,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_indices.gpu,
                        self.num_discarded_requests,
                    )
                )
                self._copy_valid_sampled_token_count(
                    next_token_ids, valid_sampled_tokens_count
                )

        before_bookkeeping_ns = time.perf_counter_ns()
        with record_function_or_nullcontext("gpu_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
                spec_decode_metadata,
            )
        after_bookkeeping_ns = time.perf_counter_ns()
        graph_gpu_ms = None
        logits_gpu_ms = None
        sampler_gpu_ms = None
        model_to_sampler_gpu_ms = None
        if sample_timing_audit and step_phase == "enabled:prefill_step":
            gpu_timing_events = getattr(self, "_prefill_gpu_timing_events", None)
            if gpu_timing_events is not None:
                graph_gpu_ms = gpu_timing_events[0].elapsed_time(
                    gpu_timing_events[1]
                )
                logits_gpu_ms = gpu_timing_events[1].elapsed_time(
                    gpu_timing_events[2]
                )
                sampler_gpu_ms = gpu_timing_events[2].elapsed_time(
                    gpu_timing_events[3]
                )
                model_to_sampler_gpu_ms = gpu_timing_events[0].elapsed_time(
                    gpu_timing_events[3]
                )

        if (
            self.speculative_config
            and not use_padded_batch_for_eagle
            and input_fits_in_drafter
        ):
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

        with record_function_or_nullcontext("gpu_model_runner: eplb"):
            self.eplb_step()
        after_eplb_ns = time.perf_counter_ns()
        with record_function_or_nullcontext("gpu_model_runner: ModelRunnerOutput"):
            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                pooler_output=[],
                kv_connector_output=kv_connector_output,
                ec_connector_output=ec_connector_output
                if self.supports_mm_inputs
                else None,
                num_nans_in_logits=num_nans_in_logits,
            )
        after_output_ns = time.perf_counter_ns()

        if sample_timing_audit and step_phase == "enabled:prefill_step":
            logger.info(
                "PREFILL_SAMPLE_TIMING setup_us=%.3f sample_submit_us=%.3f "
                "pre_bookkeeping_us=%.3f bookkeeping_sync_us=%.3f "
                "eplb_us=%.3f output_us=%.3f total_us=%.3f graph_gpu_ms=%s",
                (before_sample_ns - sample_started_ns) / 1e3,
                (after_sample_ns - before_sample_ns) / 1e3,
                (before_bookkeeping_ns - after_sample_ns) / 1e3,
                (after_bookkeeping_ns - before_bookkeeping_ns) / 1e3,
                (after_eplb_ns - after_bookkeeping_ns) / 1e3,
                (after_output_ns - after_eplb_ns) / 1e3,
                (after_output_ns - sample_started_ns) / 1e3,
                "none" if graph_gpu_ms is None else f"{graph_gpu_ms:.6f}",
            )
            logger.info(
                "PREFILL_GPU_STAGE_TIMING graph_gpu_ms=%s logits_gpu_ms=%s "
                "sampler_gpu_ms=%s model_to_sampler_gpu_ms=%s",
                "none" if graph_gpu_ms is None else f"{graph_gpu_ms:.6f}",
                "none" if logits_gpu_ms is None else f"{logits_gpu_ms:.6f}",
                "none" if sampler_gpu_ms is None else f"{sampler_gpu_ms:.6f}",
                (
                    "none"
                    if model_to_sampler_gpu_ms is None
                    else f"{model_to_sampler_gpu_ms:.6f}"
                ),
            )

        if not self.use_async_scheduling:
            return output
        with record_function_or_nullcontext(
            "gpu_model_runner: AsyncGPUModelRunnerOutput"
        ):
            async_output = AsyncGPUModelRunnerOutput(
                model_runner_output=output,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs_tensors=sampler_output.logprobs_tensors,
                invalid_req_indices=invalid_req_indices,
                async_output_copy_stream=self.async_output_copy_stream,
                vocab_size=self.input_batch.vocab_size,
            )
        with record_function_or_nullcontext(
            "gpu_model_runner: set_async_sampled_token_ids"
        ):
            # Save ref of sampled_token_ids CPU tensor if the batch contains
            # any requests with sampling params that require output ids.
            self.input_batch.set_async_sampled_token_ids(
                async_output.sampled_token_ids_cpu,
                async_output.async_copy_ready_event,
            )

        return async_output

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if self._draft_token_ids is None:
            return None
        req_ids = self.input_batch.req_ids
        if isinstance(self._draft_token_ids, torch.Tensor):
            draft_token_ids = self._draft_token_ids.tolist()
        else:
            draft_token_ids = self._draft_token_ids
        self._draft_token_ids = None
        return DraftTokenIds(req_ids, draft_token_ids)

    def _copy_valid_sampled_token_count(
        self, next_token_ids: torch.Tensor, valid_sampled_tokens_count: torch.Tensor
    ) -> None:
        if self.valid_sampled_token_count_event is None:
            return

        default_stream = torch.cuda.current_stream()
        # Initialize a new stream to overlap the copy operation with
        # prepare_input of draft model.
        with torch.cuda.stream(self.valid_sampled_token_count_copy_stream):
            self.valid_sampled_token_count_copy_stream.wait_stream(default_stream)  # type: ignore
            counts = valid_sampled_tokens_count
            counts_cpu = self.valid_sampled_token_count_cpu
            counts_cpu[: counts.shape[0]].copy_(counts, non_blocking=True)
            self.valid_sampled_token_count_event.record()

        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

    def _get_valid_sampled_token_count(self) -> list[int]:
        # Wait until valid_sampled_tokens_count is copied to cpu,
        prev_sampled_token_ids = self.input_batch.prev_sampled_token_ids
        if (
            self.valid_sampled_token_count_event is None
            or prev_sampled_token_ids is None
        ):
            return []

        counts_cpu = self.valid_sampled_token_count_cpu
        self.valid_sampled_token_count_event.synchronize()
        return counts_cpu[: prev_sampled_token_ids.shape[0]].tolist()

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> list[list[int]] | torch.Tensor:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        spec_config = self.speculative_config
        assert spec_config is not None
        if spec_config.method == "ngram":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, NgramProposer)
            draft_token_ids = self.drafter.propose(
                sampled_token_ids,
                self.input_batch.req_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
                self.input_batch.spec_decode_unsupported_reqs,
            )
        elif spec_config.method == "suffix":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, SuffixDecodingProposer)
            draft_token_ids = self.drafter.propose(self.input_batch, sampled_token_ids)
        elif spec_config.method == "medusa":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, MedusaProposer)

            if sample_hidden_states.shape[0] == len(sampled_token_ids):
                # The input to the target model does not include draft tokens.
                hidden_states = sample_hidden_states
            else:
                indices = []
                offset = 0
                assert spec_decode_metadata is not None, (
                    "No spec decode metadata for medusa"
                )
                for num_draft, tokens in zip(
                    spec_decode_metadata.num_draft_tokens, sampled_token_ids
                ):
                    indices.append(offset + len(tokens) - 1)
                    offset += num_draft + 1
                indices = torch.tensor(indices, device=self.device)
                hidden_states = sample_hidden_states[indices]

            draft_token_ids = self.drafter.propose(
                target_hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
            )
        elif spec_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)

            if spec_config.disable_padded_drafter_batch:
                # When padded-batch is disabled, the sampled_token_ids should be
                # the cpu-side list[list[int]] of valid sampled tokens for each
                # request, with invalid requests having empty lists.
                assert isinstance(sampled_token_ids, list), (
                    "sampled_token_ids should be a python list when"
                    "padded-batch is disabled."
                )
                next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    scheduler_output.num_scheduled_tokens,
                )
            else:
                # When using padded-batch, the sampled_token_ids should be
                # the gpu tensor of sampled tokens for each request, of shape
                # (num_reqs, num_spec_tokens + 1) with rejected tokens having
                # value -1.
                assert isinstance(sampled_token_ids, torch.Tensor), (
                    "sampled_token_ids should be a torch.Tensor when"
                    "padded-batch is enabled."
                )
                next_token_ids, valid_sampled_tokens_count = (
                    self.drafter.prepare_next_token_ids_padded(
                        common_attn_metadata,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_indices.gpu,
                        self.num_discarded_requests,
                    )
                )
                self._copy_valid_sampled_token_count(
                    next_token_ids, valid_sampled_tokens_count
                )

            if spec_decode_metadata is None:
                token_indices_to_sample = None
                # input_ids can be None for multimodal models.
                target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                target_positions = self._get_positions(num_scheduled_tokens)
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1
                    )
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
            else:
                if spec_config.disable_padded_drafter_batch:
                    token_indices_to_sample = None
                    common_attn_metadata, token_indices = self.drafter.prepare_inputs(
                        common_attn_metadata,
                        sampled_token_ids,
                        spec_decode_metadata.num_draft_tokens,
                    )
                else:
                    common_attn_metadata, token_indices, token_indices_to_sample = (
                        self.drafter.prepare_inputs_padded(
                            common_attn_metadata,
                            spec_decode_metadata,
                            valid_sampled_tokens_count,
                        )
                    )

                target_token_ids = self.input_ids.gpu[token_indices]
                target_positions = self._get_positions(token_indices)
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[token_indices] for h in aux_hidden_states], dim=-1
                    )
                else:
                    target_hidden_states = hidden_states[token_indices]

            if self.supports_mm_inputs:
                mm_embed_inputs = self._gather_mm_embeddings(
                    scheduler_output,
                    shift_computed_tokens=1,
                )
            else:
                mm_embed_inputs = None

            draft_token_ids = self.drafter.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                last_token_indices=token_indices_to_sample,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
                mm_embed_inputs=mm_embed_inputs,
            )

        return draft_token_ids

    def update_config(self, overrides: dict[str, Any]) -> None:
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, (
                f"Config `{config_name}` not supported. "
                f"Allowed configs: {allowed_config_names}"
            )
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    @staticmethod
    def _env_flag_enabled(env_name: str, default: bool = False) -> bool:
        env_value = os.environ.get(env_name)
        if env_value is None:
            return default
        return env_value.lower() in ("1", "true", "yes", "on")

    def _uvm_fixed_weight_device_direct_enabled(self) -> bool:
        return self._env_flag_enabled(
            "VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_ENABLE", default=False
        )

    @staticmethod
    def _uvm_weight_semantic_tags(name: str) -> dict[str, Any]:
        parts = name.split(".")
        lower_parts = [part.lower() for part in parts]
        layer_id: int | None = None
        expert_id: int | None = None
        is_moe_expert = False
        is_main_weight = False
        is_moe_expert_main_weight = False
        semantic_type = "other"

        for idx, part in enumerate(parts):
            if part == "layers" and idx + 1 < len(parts):
                try:
                    layer_id = int(parts[idx + 1])
                except ValueError:
                    pass
            if part == "experts" and idx + 1 < len(parts):
                try:
                    expert_id = int(parts[idx + 1])
                    is_moe_expert = True
                except ValueError:
                    pass

        lowered = name.lower()
        leaf = parts[-1].lower() if parts else lowered
        if "experts" in lower_parts or "fused_experts" in lowered:
            is_moe_expert = True

        if is_moe_expert:
            semantic_type = "moe_expert"
            if leaf in ("w13_weight", "w13"):
                is_main_weight = True
                is_moe_expert_main_weight = True
            elif leaf in ("w2_weight", "w2"):
                is_main_weight = True
                is_moe_expert_main_weight = True
            elif leaf in ("w3_weight", "w3"):
                is_main_weight = True
                is_moe_expert_main_weight = True
            elif leaf in ("w1_weight", "w1"):
                is_main_weight = True
                is_moe_expert_main_weight = True
            elif any(token in leaf for token in ("scale", "scales", "amax")):
                semantic_type = "moe_expert_aux"
        elif "router" in lowered or "gate" in lowered:
            semantic_type = "moe_router"
        elif "lm_head" in lowered:
            semantic_type = "lm_head"
        elif "embed" in lowered:
            semantic_type = "embedding"
        elif "self_attn" in lowered or ".attn." in lowered:
            semantic_type = "attention"
        elif "shared_expert" in lowered:
            semantic_type = "mlp_dense"
        elif "mlp" in lowered:
            semantic_type = "mlp_dense"
        elif "norm" in lowered:
            semantic_type = "norm"
        elif any(token in lowered for token in ("scale", "scales", "amax")):
            semantic_type = "quant_aux"
        elif any(token in lowered for token in ("rotary", "cos_cached", "sin_cached")):
            semantic_type = "positional"
        if not is_main_weight:
            is_main_weight = leaf in (
                "weight",
                "qweight",
                "wq",
                "qweight_g_idx",
            ) or leaf.endswith("_weight")

        fixed_weight = not is_moe_expert
        return {
            "layer_id": layer_id,
            "expert_id": expert_id,
            "is_moe_expert": is_moe_expert,
            "is_main_weight": is_main_weight,
            "is_moe_expert_main_weight": is_moe_expert_main_weight,
            "semantic_type": semantic_type,
            "weight_class": semantic_type,
            "fixed_weight": fixed_weight,
            "protected": fixed_weight,
            "evictable": not fixed_weight,
            "placement_policy": "fixed_gpu" if fixed_weight else "expert_policy",
        }

    def _materialize_fixed_weights_device_direct(self) -> None:
        if not self._uvm_fixed_weight_device_direct_enabled():
            return
        if not self._env_flag_enabled("VLLM_USE_UVM", default=False):
            logger.warning(
                "Ignoring VLLM_UVM_FIXED_WEIGHT_DEVICE_DIRECT_ENABLE because "
                "VLLM_USE_UVM is not enabled."
            )
            return

        model = self.get_model()
        try:
            parameter_iter = model.named_parameters(
                recurse=True, remove_duplicate=False
            )
        except TypeError:
            parameter_iter = model.named_parameters(recurse=True)

        processed_params: set[int] = set()

        with torch.no_grad():
            for name, param in parameter_iter:
                if param is None:
                    continue
                param_id = id(param)
                if param_id in processed_params:
                    continue
                processed_params.add(param_id)

                tensor = param.data
                semantic = self._uvm_weight_semantic_tags(name)
                if (
                    not bool(semantic.get("fixed_weight"))
                    or bool(semantic.get("is_moe_expert"))
                    or not tensor.is_cuda
                    or int(tensor.numel()) <= 0
                ):
                    continue

                try:
                    with uvm_allocation_phase("fixed_weight_device_direct"):
                        native_tensor = torch.empty_strided(
                            tuple(tensor.size()),
                            tuple(tensor.stride()),
                            dtype=tensor.dtype,
                            device=tensor.device,
                        )
                    native_tensor.copy_(tensor, non_blocking=False)
                    param.data = native_tensor
                except Exception as exc:
                    logger.warning(
                        "Failed to materialize fixed weight %s with "
                        "device-direct allocation: %s",
                        name,
                        exc,
                    )

        try:
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            logger.debug(
                "Failed to synchronize/empty cache after fixed weight "
                "device-direct materialization",
                exc_info=True,
            )

    def _register_uvm_kv_cache_physical_ranges(
        self,
        kv_cache_config: KVCacheConfig,
        kv_caches: dict[str, Any],
    ) -> None:
        records: list[dict[str, Any]] = []
        total_blocks = max(int(kv_cache_config.num_blocks) - 1, 0)

        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            page_size = int(group.kv_cache_spec.page_size_bytes)
            if page_size <= 0:
                continue
            for layer_name in group.layer_names:
                cache_value = kv_caches.get(layer_name)
                tensors = (
                    cache_value if isinstance(cache_value, list) else [cache_value]
                )
                for tensor_index, tensor in enumerate(tensors):
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    if not tensor.is_cuda:
                        continue
                    if not tensor.is_contiguous():
                        continue
                    ptr = int(tensor.data_ptr())
                    size_bytes = int(tensor.numel() * tensor.element_size())
                    if ptr <= 0 or size_bytes <= 0:
                        continue
                    allocation_blocks = size_bytes // page_size
                    max_block_id = min(total_blocks, allocation_blocks - 1)
                    if max_block_id <= 0:
                        continue
                    records.append(
                        {
                            "span_id": (
                                f"kv_phys:{group_id}:{layer_name}:{tensor_index}"
                            ),
                            "ptr": ptr,
                            "size_bytes": size_bytes,
                            "page_size": page_size,
                            "max_block_id": max_block_id,
                        }
                    )

        register_uvm_kv_cache_physical_ranges(
            records,
            clear_existing=True,
        )

    def _collect_uvm_post_warmup_kv_tail_ranges(
        self,
        target_blocks: int,
    ) -> tuple[list[tuple[int, int]], dict[str, Any]]:
        """Collect contiguous free KV tails beyond the resident block target."""
        kv_cache_config = getattr(self, "kv_cache_config", None)
        kv_caches = getattr(self, "_uvm_kv_caches_by_layer", {})
        if kv_cache_config is None or not kv_caches:
            return [], {
                "ok": False,
                "reason": "kv_cache_not_initialized",
                "target_blocks": int(target_blocks),
            }

        total_blocks = max(int(kv_cache_config.num_blocks) - 1, 0)
        clamped_target = min(max(int(target_blocks), 0), total_blocks)
        ranges: list[tuple[int, int]] = []
        seen_ranges: set[tuple[int, int]] = set()

        for group in kv_cache_config.kv_cache_groups:
            page_size = int(group.kv_cache_spec.page_size_bytes)
            if page_size <= 0:
                continue
            for layer_name in group.layer_names:
                cache_value = kv_caches.get(layer_name)
                tensors = (
                    cache_value if isinstance(cache_value, list) else [cache_value]
                )
                for tensor in tensors:
                    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda:
                        continue
                    if not tensor.is_contiguous():
                        continue
                    ptr = int(tensor.data_ptr())
                    size_bytes = int(tensor.numel() * tensor.element_size())
                    tail_range = _uvm_kv_tail_range_for_allocation(
                        ptr=ptr,
                        size_bytes=size_bytes,
                        page_size=page_size,
                        total_blocks=total_blocks,
                        target_blocks=clamped_target,
                    )
                    if tail_range is None:
                        continue
                    tail_ptr, tail_bytes, _, _ = tail_range
                    range_key = (tail_ptr, tail_bytes)
                    if tail_bytes <= 0 or range_key in seen_ranges:
                        continue
                    seen_ranges.add(range_key)
                    ranges.append(range_key)

        collection: dict[str, Any] = {
            "ok": True,
            "reason": "ranges_collected" if ranges else "target_covers_all_ranges",
            "target_blocks": clamped_target,
        }
        return ranges, collection

    def rebalance_uvm_kv_cache_after_warmup(
        self,
        target_blocks: int,
    ) -> dict[str, Any]:
        """Converge fixed-VA KV pages to the requested initial residency.

        The default path preserves the historical tail-only rebalance.  Lazy
        reserve additionally prefetches the active prefix because the maximum
        KV allocation was created with ``torch.empty`` and no allocator-level
        prefetch; in that mode reserving a tensor does not imply GPU residency.
        """
        ranges, collection = self._collect_uvm_post_warmup_kv_tail_ranges(target_blocks)
        record: dict[str, Any] = {
            "ok": bool(collection.get("ok")),
            "reason": str(collection.get("reason") or "unknown"),
            "target_blocks": int(collection.get("target_blocks") or target_blocks),
            "offload_complete": False,
        }
        if not collection.get("ok"):
            return record
        executor = (
            os.environ.get(
                "VLLM_UVM_POST_WARMUP_KV_OFFLOAD_EXECUTOR",
                "auto",
            )
            .strip()
            .lower()
        )
        if executor not in ("cpp_batch", "driver_batch", "auto"):
            logger.warning(
                "Ignoring invalid VLLM_UVM_POST_WARMUP_KV_OFFLOAD_EXECUTOR=%r; "
                "using auto",
                executor,
            )
            executor = "auto"

        lazy_reserve = os.environ.get(
            "VLLM_UVM_KV_CACHE_LAZY_RESERVE",
            "0",
        ).strip().lower() in ("1", "true", "yes", "on")
        complete = True
        made_progress = False
        device = int(torch.cuda.current_device()) if lazy_reserve else 0
        head_ranges: list[tuple[int, int]] = []

        cpu_materialization_ranges = ranges
        if lazy_reserve:
            # ``torch.empty`` reserves the maximum VA without committing every
            # page.  A direct high target therefore cannot assume that a GPU
            # prefetch alone creates backing for the untouched pages.  First
            # materialize the complete fixed-VA pool on CPU, then migrate the
            # requested active prefix to GPU.  Dynamic low->high already takes
            # this path implicitly because the low tail is CPU resident.
            kv_cache_config = getattr(self, "kv_cache_config", None)
            total_blocks = (
                max(int(kv_cache_config.num_blocks) - 1, 0)
                if kv_cache_config is not None
                else 0
            )
            cpu_materialization_ranges = (
                collect_uvm_kv_cache_physical_ranges_for_blocks(
                    list(range(0, total_blocks + 1))
                )
            )
            if total_blocks > 0 and not cpu_materialization_ranges:
                complete = False

        if complete and cpu_materialization_ranges:
            successes, _, _, _ = (
                batch_offload_ranges_to_cpu_with_executor(
                    cpu_materialization_ranges,
                    sync=True,
                    executor=executor,
                )
            )
            complete = int(successes) == len(cpu_materialization_ranges)
            made_progress = True

        if complete and lazy_reserve:
            clamped_target = int(record["target_blocks"])
            head_ranges = collect_uvm_kv_cache_physical_ranges_for_blocks(
                list(range(0, clamped_target + 1))
            )
            if clamped_target > 0 and not head_ranges:
                complete = False
            elif head_ranges:
                advised = batch_advise_ranges_preferred_location(
                    head_ranges,
                    device,
                )
                if int(advised) != len(head_ranges):
                    complete = False
                else:
                    successes, _, _ = (
                        batch_prefetch_ranges_to_device_with_executor(
                            head_ranges,
                            device,
                            executor=executor,
                        )
                    )
                    complete = int(successes) == len(head_ranges)
                made_progress = True

        if lazy_reserve and made_progress:
            try:
                torch.cuda.synchronize(device)
            except Exception:
                complete = False

        if complete:
            head_complete = True
            if lazy_reserve and head_ranges:
                head_residency = driver_batch_query_ranges_residency_bytes(
                    head_ranges,
                    device=device,
                    wait_for_trackers=False,
                )
                head_complete = _uvm_kv_ranges_have_exclusive_residency(
                    head_ranges,
                    head_residency,
                    location="gpu",
                )

            tail_complete = True
            if ranges:
                tail_residency = driver_batch_query_ranges_residency_bytes(
                    ranges,
                    device=device,
                    wait_for_trackers=False,
                )
                tail_complete = _uvm_kv_ranges_have_exclusive_residency(
                    ranges,
                    tail_residency,
                    location="cpu",
                )

            complete = head_complete and tail_complete
            if not complete:
                retry_accepted = True
                if not tail_complete and ranges:
                    successes, _, _, _ = (
                        batch_offload_ranges_to_cpu_with_executor(
                            ranges,
                            sync=True,
                            executor=executor,
                        )
                    )
                    retry_accepted = int(successes) == len(ranges)

                if not head_complete and lazy_reserve and head_ranges:
                    advised = batch_advise_ranges_preferred_location(
                        head_ranges,
                        device,
                    )
                    successes, _, _ = (
                        batch_prefetch_ranges_to_device_with_executor(
                            head_ranges,
                            device,
                            executor=executor,
                        )
                    )
                    retry_accepted = (
                        retry_accepted
                        and int(advised) == len(head_ranges)
                        and int(successes) == len(head_ranges)
                    )

                if retry_accepted and lazy_reserve and not head_complete:
                    try:
                        torch.cuda.synchronize(device)
                    except Exception:
                        retry_accepted = False

                if retry_accepted:
                    if lazy_reserve and head_ranges:
                        head_residency = (
                            driver_batch_query_ranges_residency_bytes(
                                head_ranges,
                                device=device,
                                wait_for_trackers=False,
                            )
                        )
                        head_complete = (
                            _uvm_kv_ranges_have_exclusive_residency(
                                head_ranges,
                                head_residency,
                                location="gpu",
                            )
                        )
                    if ranges:
                        tail_residency = (
                            driver_batch_query_ranges_residency_bytes(
                                ranges,
                                device=device,
                                wait_for_trackers=False,
                            )
                        )
                        tail_complete = (
                            _uvm_kv_ranges_have_exclusive_residency(
                                ranges,
                                tail_residency,
                                location="cpu",
                            )
                        )
                    complete = head_complete and tail_complete

        record["offload_complete"] = complete
        return record

    def resize_uvm_kv_cache_target(
        self,
        old_target_blocks: int,
        new_target_blocks: int,
        reconciliation_block_ids: list[int] | tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        """Physically converge the only worker's complete KV resize plan."""
        kv_cache_config = getattr(self, "kv_cache_config", None)
        if kv_cache_config is None:
            return {"ok": False, "made_progress": False}
        total_blocks = max(int(kv_cache_config.num_blocks) - 1, 0)
        old_target = min(max(int(old_target_blocks), 0), total_blocks)
        new_target = min(max(int(new_target_blocks), 0), total_blocks)
        first_block = min(old_target, new_target) + 1
        last_block = max(old_target, new_target)
        target_delta_ids = set(range(first_block, last_block + 1))
        requested_block_ids = sorted(
            target_delta_ids
            | {
                int(block_id)
                for block_id in (reconciliation_block_ids or ())
                if 0 < int(block_id) <= total_blocks
            }
        )
        requested_cpu_block_ids = [
            block_id
            for block_id in requested_block_ids
            if block_id > new_target
        ]
        requested_gpu_block_ids = [
            block_id
            for block_id in requested_block_ids
            if block_id <= new_target
        ]
        if not requested_block_ids:
            return {"ok": True, "made_progress": False}

        executor = os.environ.get(
            "VLLM_UVM_KV_DYNAMIC_POOL_PHYSICAL_MIGRATION_EXECUTOR",
            "auto",
        ).strip().lower()
        device = int(torch.cuda.current_device())
        made_progress = False
        complete = True

        require_full_dual_view = self._env_flag_enabled(
            "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW",
            default=False,
        )
        prepare_dual_before_grow = bool(
            require_full_dual_view
            and requested_gpu_block_ids
            and self._env_flag_enabled(
                "VLLM_UVM_KV_PREPARE_DUAL_BEFORE_GROW", default=False
            )
        )
        dual_view_prepare: dict[str, Any] | None = None
        if prepare_dual_before_grow:
            # The enclosing Plan Class transaction has released newly cold
            # experts and installed their aliases, but may have deferred the
            # detached children. Reserve those page tables before KV consumes
            # the released HBM. This runs at the same quiescent boundary and
            # does not evict any managed page to allocate a page table.
            try:
                from vllm.model_executor.models.utils import (
                    refresh_cpu_offload_uvm_original_va_full_dual_view,
                )

                dual_view_prepare = (
                    refresh_cpu_offload_uvm_original_va_full_dual_view()
                )
                if dual_view_prepare.get("ok") is not True:
                    return {
                        "ok": False,
                        "made_progress": False,
                        "dual_view_prepare": dual_view_prepare,
                    }
            except Exception as exc:
                logger.exception("Full dual-view preparation before KV grow failed")
                return {
                    "ok": False,
                    "made_progress": False,
                    "dual_view_prepare": {
                        "ok": False,
                        "reason": "dual_view_prepare_failed",
                        "error": repr(exc),
                    },
                }

        for direction, block_ids in (
            ("shrink", requested_cpu_block_ids),
            ("grow", requested_gpu_block_ids),
        ):
            if not block_ids:
                continue
            ranges = collect_uvm_kv_cache_physical_ranges_for_blocks(block_ids)
            if not ranges:
                complete = False
                break

            try:
                made_progress = True
                if direction == "grow":
                    advised = batch_advise_ranges_preferred_location(
                        ranges,
                        device,
                    )
                    if int(advised) != len(ranges):
                        complete = False
                        break
                    successes, _, _ = (
                        batch_prefetch_ranges_to_device_with_executor(
                            ranges,
                            device,
                            executor=executor,
                        )
                    )
                else:
                    successes, _, _, _ = (
                        batch_offload_ranges_to_cpu_with_executor(
                            ranges,
                            sync=False,
                            executor=executor,
                        )
                    )
                complete = int(successes) == len(ranges)
                if not complete:
                    logger.warning(
                        "Incomplete physical KV %s: successes=%d ranges=%d",
                        direction,
                        int(successes),
                        len(ranges),
                    )
            except Exception:
                logger.exception(
                    "Physical KV %s failed for %d ranges",
                    direction,
                    len(ranges),
                )
                complete = False
            if not complete:
                break

        if made_progress:
            try:
                torch.cuda.synchronize(device)
            except Exception:
                logger.exception("Physical KV resize synchronization failed")
                complete = False
        reconcile_hot_after_grow = bool(
            require_full_dual_view
            and requested_gpu_block_ids
            and self._env_flag_enabled(
                "VLLM_UVM_KV_RECONCILE_HOT_AFTER_GROW", default=False
            )
        )
        hot_residency_reconcile: dict[str, Any] | None = None
        kv_residency_after_hot_reconcile: dict[str, Any] | None = None
        if reconcile_hot_after_grow and complete and made_progress:
            try:
                from vllm.model_executor.models.utils import (
                    reconcile_cpu_offload_uvm_original_va_sealed_hot_residency,
                )

                # KV prefetch can evict ordinary hot expert pages even when
                # the cold cutover released the requested logical byte count.
                # The manual-migration serving contract forbids relying on a
                # later model fault to restore them. Repair at this quiescent
                # boundary, then verify the repair did not evict active KV.
                hot_residency_reconcile = (
                    reconcile_cpu_offload_uvm_original_va_sealed_hot_residency(
                        preserve_allocator_cache=True
                    )
                )
                complete = hot_residency_reconcile.get("ok") is True
                if complete:
                    # Audit the same GPU prefix as initialization, including
                    # the reserved null block. MLA blocks can be 18 KiB, so
                    # starting at block 1 would split a 4 KiB physical page.
                    # The paired target boundary keeps the full prefix aligned.
                    active_kv_ranges = (
                        collect_uvm_kv_cache_physical_ranges_for_blocks(
                            list(range(0, new_target + 1))
                        )
                    )
                    kv_residency_after_hot_reconcile = (
                        driver_batch_query_ranges_residency_bytes(
                            active_kv_ranges,
                            device=device,
                            wait_for_trackers=True,
                        )
                    )
                    complete = bool(active_kv_ranges) and (
                        _uvm_kv_ranges_have_exclusive_residency(
                            active_kv_ranges,
                            kv_residency_after_hot_reconcile,
                            location="gpu",
                        )
                    )
                    logger.info(
                        "Post-growth active KV residency audit: target=%d "
                        "ok=%s residency=%r",
                        new_target,
                        complete,
                        kv_residency_after_hot_reconcile,
                    )
            except Exception as exc:
                logger.exception("Hot expert reconciliation after KV grow failed")
                hot_residency_reconcile = {
                    "ok": False,
                    "reason": "hot_residency_reconcile_failed",
                    "error": repr(exc),
                }
                complete = False
        dual_view_refresh: dict[str, Any] | None = None
        if require_full_dual_view and complete and made_progress:
            try:
                from vllm.model_executor.models.utils import (
                    refresh_cpu_offload_uvm_original_va_full_dual_view,
                )

                if prepare_dual_before_grow or reconcile_hot_after_grow:
                    # KV migration may relocate ordinary hot pages. A nested
                    # refresh would skip already selected blocks; rebuild their
                    # ordinary leaves using the current physical addresses.
                    # Prepared children are reused without allocating again.
                    dual_view_refresh = (
                        refresh_cpu_offload_uvm_original_va_full_dual_view(
                            nested_refresh=False
                        )
                    )
                else:
                    dual_view_refresh = (
                        refresh_cpu_offload_uvm_original_va_full_dual_view()
                    )
                if dual_view_refresh.get("ok") is not True:
                    complete = False
            except Exception as exc:
                logger.exception(
                    "Full dual-view refresh after physical KV resize failed"
                )
                dual_view_refresh = {
                    "ok": False,
                    "reason": "dual_view_refresh_failed",
                    "error": repr(exc),
                }
                complete = False
        result: dict[str, Any] = {
            "ok": complete,
            "made_progress": made_progress,
        }
        if require_full_dual_view:
            result["dual_view_refresh"] = dual_view_refresh
        if prepare_dual_before_grow:
            result["dual_view_prepare"] = dual_view_prepare
        if reconcile_hot_after_grow:
            result["hot_residency_reconcile"] = hot_residency_reconcile
            result["kv_residency_after_hot_reconcile"] = (
                kv_residency_after_hot_reconcile
            )
        return result

    def load_model(self, eep_scale_up: bool = False) -> None:
        """
        Args:
            eep_scale_up: 布尔值，指示此次模型加载是否是为了弹性专家并行 (Elastic Expert Parallelism, EEP) 扩容。
        """
        # 记录全局日志，提示开始加载特定的模型
        logger.info_once(
            "Starting to load model %s...",
            self.model_config.model,
            scope="global",
        )

        # 如果是 EEP 扩容操作，则从 EplbState 获取当前的全局专家负载状态、旧的全局专家索引以及 rank 映射。
        # 否则，初始化为 None。这对于在运行时动态调整专家模型的分布非常重要。
        global_expert_loads, old_global_expert_indices_per_model, rank_mapping = (
            EplbState.get_eep_state(self.parallel_config)
            if eep_scale_up
            else (None, None, None)
        )

        # 检查是否启用了专家并行负载均衡 (Expert Parallelism Load Balancing, EPLB)
        if self.parallel_config.enable_eplb:
            self.eplb_state = EplbState(self.parallel_config, self.device)
            eplb_models = 0  # 追踪应用了 EPLB 的模型数量（主模型 + 潜在的草稿模型）

        # 使用设备内存分析器作为上下文管理器，以精准统计加载模型所消耗的显存
        with DeviceMemoryProfiler() as m:
            time_before_load = time.perf_counter()  # 记录加载开始时间

            # 获取当前配置对应的模型加载器
            model_loader = get_model_loader(self.load_config)

            with uvm_allocation_phase("load_model"):
                # 实际执行主模型的权重加载
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config, model_config=self.model_config
                )

                # 如果配置了 LoRA (Low-Rank Adaptation)，则将 LoRA 权重加载/合并到基础模型中
                if self.lora_config:
                    self.model = self.load_lora_model(
                        self.model, self.vllm_config, self.device
                    )

                # 检查系统是否配置了用于投机解码 (Speculative Decoding) 的草稿模型 (drafter)
                if hasattr(self, "drafter"):
                    logger.info_once("Loading drafter model...")
                    self.drafter.load_model(self.model)  # 加载草稿模型

                    # 如果草稿模型是混合专家 (MoE) 架构，并且系统启用了 EPLB
                    if (
                        hasattr(self.drafter, "model")
                        and is_mixture_of_experts(self.drafter.model)
                        and self.parallel_config.enable_eplb
                    ):
                        # 获取投机解码的配置，验证草稿模型的配置有效性
                        spec_config = self.vllm_config.speculative_config
                        assert spec_config is not None
                        assert spec_config.draft_model_config is not None
                        logger.info_once(
                            "EPLB is enabled for drafter model %s.",
                            spec_config.draft_model_config.model,
                        )

                        # 为草稿模型提取对应的专家负载和旧索引信息
                        global_expert_load = (
                            global_expert_loads[eplb_models]
                            if global_expert_loads
                            else None
                        )
                        old_global_expert_indices = (
                            old_global_expert_indices_per_model[eplb_models]
                            if old_global_expert_indices_per_model
                            else None
                        )

                        # 确保 EPLB 状态已初始化，并将草稿模型及其配置注册到 EPLB 状态管理中
                        if self.eplb_state is None:
                            self.eplb_state = EplbState(
                                self.parallel_config, self.device
                            )
                        self.eplb_state.add_model(
                            self.drafter.model,
                            spec_config.draft_model_config,
                            global_expert_load,
                            old_global_expert_indices,
                            rank_mapping,
                        )
                        eplb_models += 1  # 增加注册的 EPLB 模型计数

            # 处理 EAGLE3 接口所需的辅助隐藏状态输出
            # EAGLE 是一种高效的投机解码/自回归生成加速技术，通常依赖特定的隐藏层状态
            if self.use_aux_hidden_state_outputs:
                if not supports_eagle3(self.get_model()):
                    raise RuntimeError(
                        "Model does not support EAGLE3 interface but "
                        "aux_hidden_state_outputs was requested"
                    )

                # 尝试从投机解码配置中获取需要提取隐藏状态的辅助层索引
                # 否则使用模型默认定义的层
                aux_layers = self._get_eagle3_aux_layers_from_config()
                if aux_layers:
                    logger.info(
                        "Using auxiliary layers from speculative config: %s",
                        aux_layers,
                    )
                else:
                    aux_layers = self.model.get_eagle3_aux_hidden_state_layers()

                # 将获取到的辅助层配置应用到主模型
                self.model.set_aux_hidden_state_layers(aux_layers)

            time_after_load = time.perf_counter()  # 记录加载结束时间

        # 记录模型加载消耗的总显存和总耗时
        self.model_memory_usage = m.consumed_memory
        logger.info_once(
            "Model loading took %.4f GiB memory and %.6f seconds",
            self.model_memory_usage / GiB_bytes,
            time_after_load - time_before_load,
            scope="local",
        )

        # 为分布式训练/推理准备通信缓冲区（例如为张量并行分配 All-Reduce 缓冲区）
        prepare_communication_buffer_for_model(self.model)
        # 如果存在草稿模型，同样为其准备通信缓冲区
        if (drafter := getattr(self, "drafter", None)) and (
            drafter_model := getattr(drafter, "model", None)
        ):
            prepare_communication_buffer_for_model(drafter_model)

        # 配置多模态剪枝 (Multimodal Pruning) 支持
        mm_config = self.model_config.multimodal_config
        self.is_multimodal_pruning_enabled = (
            supports_multimodal_pruning(self.get_model())
            and mm_config is not None
            and mm_config.is_multimodal_pruning_enabled()
        )

        # 针对主模型：如果是 MoE 架构且启用了 EPLB，则为其配置负载均衡
        if is_mixture_of_experts(self.model) and self.parallel_config.enable_eplb:
            logger.info_once("EPLB is enabled for model %s.", self.model_config.model)
            global_expert_load = (
                global_expert_loads[eplb_models] if global_expert_loads else None
            )
            old_global_expert_indices = (
                old_global_expert_indices_per_model[eplb_models]
                if old_global_expert_indices_per_model
                else None
            )
            assert self.eplb_state is not None
            # 将主模型注册到 EPLB 管理器中
            self.eplb_state.add_model(
                self.model,
                self.model_config,
                global_expert_load,
                old_global_expert_indices,
                rank_mapping,
            )
            # 如果 EPLB 支持异步模式，则开启异步通信/更新循环
            if self.eplb_state.is_async:
                self.eplb_state.start_async_loop(rank_mapping=rank_mapping)

        self._materialize_fixed_weights_device_direct()

        # The UVM original-VA static-alias backend must freeze the final
        # checkpoint/post-processing storage before any profile run, compile
        # wrapper, or CUDA Graph capture can observe the model.  The helper is
        # a no-op for the default VMM backend and when static aliasing is off.
        from vllm.model_executor.models.utils import (
            finalize_cpu_offload_graph_static_alias,
            initialize_cpu_offload_uvm_original_va_phase_topology,
        )

        finalize_cpu_offload_graph_static_alias()
        initialize_cpu_offload_uvm_original_va_phase_topology()

        # 处理模型图编译加速
        # 1. 尝试使用原生的 PyTorch 图编译 (torch.compile)
        if (
            self.vllm_config.compilation_config.mode
            == CompilationMode.STOCK_TORCH_COMPILE
            and supports_dynamo()
        ):
            backend = self.vllm_config.compilation_config.init_backend(self.vllm_config)
            compilation_counter.stock_torch_compile_count += 1
            # 开启 fullgraph 进行最大化优化编译
            self.model.compile(fullgraph=True, backend=backend)
            return

        # 注意：对于其他编译模式，CUDA Graph 的行为由 vllm 的 Wrapper 和 Dispatcher 控制

        # 2. 根据编译配置，使用 CUDAGraphWrapper 或 UBatchWrapper 包装模型以提升推理性能
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None

        # 如果启用了完整的 CUDA Graph 并且未启用 DBO (Dynamic Batch Optimization/动态批次优化)
        if cudagraph_mode.has_full_cudagraphs() and not self.parallel_config.enable_dbo:
            self.model = CUDAGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )
        # 如果启用了 DBO，则使用 UBatchWrapper（用于处理 micro-batches）包装模型
        elif self.parallel_config.enable_dbo:
            if cudagraph_mode.has_full_cudagraphs():
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.FULL, self.device
                )
            else:
                self.model = UBatchWrapper(
                    self.model, self.vllm_config, CUDAGraphMode.NONE, self.device
                )

    def _get_eagle3_aux_layers_from_config(self) -> tuple[int, ...] | None:
        """Extract Eagle3 auxiliary layer indices from speculative config.

        These indices specify which hidden states from the base model should
        be used as auxiliary inputs for the Eagle3 drafter model during
        speculative decoding.

        Returns:
            Tuple of layer indices if found in draft model config,
            None otherwise.
        """
        if not (self.speculative_config and self.speculative_config.draft_model_config):
            return None

        hf_config = self.speculative_config.draft_model_config.hf_config
        if not hasattr(hf_config, "eagle_aux_hidden_state_layer_ids"):
            return None

        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)

        return None

    def reload_weights(self) -> None:
        assert getattr(self, "model", None) is not None, (
            "Cannot reload weights before model is loaded."
        )
        from vllm.model_executor.models.utils import (
            get_cpu_offload_graph_static_alias_diagnostics,
        )

        alias_state = get_cpu_offload_graph_static_alias_diagnostics()
        if (
            alias_state.get("backend") == "uvm_original_va"
            and int(alias_state.get("session_id", 0)) != 0
        ):
            raise RuntimeError(
                "Cannot reload weights while a sealed UVM original-VA "
                "static-alias session or captured CUDA Graph is live; destroy "
                "the Graph topology first"
            )
        model_loader = get_model_loader(self.load_config)
        logger.info("Reloading weights inplace...")
        model_loader.load_weights(self.get_model(), model_config=self.model_config)

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        TensorizerLoader.save_model(
            self.get_model(),
            tensorizer_config=tensorizer_config,
            model_config=self.model_config,
        )

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                # This can happen if the request was preempted in prefill stage.
                continue

            # Get metadata for this request.
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True
            )

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            with uvm_enabled_allocation_phase("enabled:compute_logits"):
                logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt logprobs.
            with uvm_enabled_allocation_phase("enabled:logits"):
                logprobs = self.sampler.compute_logprobs(logits)
                token_ids, logprobs, ranks = self.sampler.gather_logprobs(
                    logprobs, num_prompt_logprobs, tgt_token_ids
                )

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_nans_in_logits(
        self,
        logits: torch.Tensor | None,
    ) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])
                    if num_nans_for_index is not None and req_index < logits.shape[0]
                    else 0
                )
            return num_nans_in_logits
        except IndexError:
            return {}

    @contextmanager
    def maybe_randomize_inputs(self, input_ids: torch.Tensor):
        """
        Randomize input_ids if VLLM_RANDOMIZE_DP_DUMMY_INPUTS is set.
        This is to help balance expert-selection
         - during profile_run
         - during DP rank dummy run
        """
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1
        if not randomize_inputs:
            yield
        else:
            import functools

            @functools.cache
            def rand_input_ids() -> torch.Tensor:
                return torch.randint_like(
                    self.input_ids.gpu,
                    low=0,
                    high=self.model_config.get_vocab_size(),
                    dtype=input_ids.dtype,
                )

            logger.debug_once("Randomizing dummy data for DP Rank")
            input_ids.copy_(rand_input_ids()[: input_ids.size(0)], non_blocking=True)
            yield
            input_ids.fill_(0)

    def _get_mm_dummy_batch(
        self,
        modality: str,
        max_items_per_batch: int,
    ) -> BatchedTensorInputs:
        """Dummy data for profiling and precompiling multimodal models."""
        assert self.mm_budget is not None

        dummy_decoder_data = self.mm_registry.get_decoder_dummy_data(
            model_config=self.model_config,
            seq_len=self.max_model_len,
            mm_counts={modality: 1},
            cache=self.mm_budget.cache,
        )
        dummy_mm_data = dummy_decoder_data.multi_modal_data

        # Result in the maximum GPU consumption of the model
        dummy_mm_item = dummy_mm_data[modality][0]
        dummy_mm_items = [dummy_mm_item] * max_items_per_batch

        model = cast(SupportsMultiModal, self.model)
        return next(
            mm_kwargs_group
            for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
                dummy_mm_items,
                device=self.device,
                pin_memory=self.pin_memory,
                merge_by_field_config=model.merge_by_field_config,
                multimodal_cpu_fields=model.multimodal_cpu_fields,
            )
        )

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        activate_lora: bool = False,
        is_graph_capturing: bool = False,
        pure_prefill_num_reqs: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
            activate_lora: If False, dummy_run is performed without LoRAs.
            pure_prefill_num_reqs: If set, construct a balanced, zero-context
                Prefill batch with at most this many requests. Unlike the
                legacy profiling shape, sequence lengths equal the scheduled
                query lengths and must fit ``max_model_len``.
        """
        assert (
            cudagraph_runtime_mode is None
            or cudagraph_runtime_mode.valid_runtime_modes()
        )

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if pure_prefill_num_reqs is not None:
            if uniform_decode or create_mixed_batch:
                raise ValueError(
                    "pure_prefill_num_reqs cannot be combined with "
                    "uniform_decode or create_mixed_batch"
                )
            requested_num_reqs = min(
                int(pure_prefill_num_reqs), int(max_num_reqs)
            )
            num_scheduled_tokens_list, max_query_len = (
                _build_pure_prefill_dummy_shape(
                    num_tokens=num_tokens,
                    num_reqs=requested_num_reqs,
                    max_model_len=self.max_model_len,
                )
            )
            num_reqs = len(num_scheduled_tokens_list)
        elif create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        total_num_scheduled_tokens = int(num_scheduled_tokens.sum())
        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        # Disable DP padding when running eager
        allow_dp_padding = self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE

        # We currently only microbatch if the number of tokens is
        # over a certain threshold.
        ubatch_slices, num_tokens_across_dp = coordinate_batch_across_dp(
            num_tokens_unpadded=total_num_scheduled_tokens,
            parallel_config=self.vllm_config.parallel_config,
            allow_microbatching=allow_microbatching,
            allow_dp_padding=allow_dp_padding,
            num_tokens_padded=total_num_scheduled_tokens,
            uniform_decode=uniform_decode,
            num_scheduled_tokens_per_request=num_scheduled_tokens,
        )
        num_tokens_after_padding = num_tokens
        if num_tokens_across_dp is not None:
            dp_rank = self.parallel_config.data_parallel_rank
            num_tokens_after_padding = int(num_tokens_across_dp[dp_rank])

        # filter out the valid batch descriptor
        _cg_mode, batch_descriptor = (
            self.cudagraph_dispatcher.dispatch(
                BatchDescriptor(
                    num_tokens=num_tokens_after_padding,
                    uniform_decode=uniform_decode,
                    has_lora=activate_lora and self.lora_config is not None,
                )
            )
            if not is_profile
            else (CUDAGraphMode.NONE, None)
        )
        if cudagraph_runtime_mode is not None:
            # we allow forcing NONE when the dispatcher disagrees to support
            # warm ups for cudagraph capture
            assert (
                cudagraph_runtime_mode == CUDAGraphMode.NONE
                or cudagraph_runtime_mode == _cg_mode
            ), (
                f"Cudagraph runtime mode mismatch at dummy_run. "
                f"Expected {_cg_mode}, but got {cudagraph_runtime_mode}."
            )
        else:
            cudagraph_runtime_mode = _cg_mode

        attn_metadata: PerLayerAttnMetadata | None = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            if create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            elif pure_prefill_num_reqs is not None:
                seq_lens = num_scheduled_tokens_list
            else:
                seq_lens = max_query_len  # type: ignore[assignment]
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
            self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()

            attn_metadata, _ = self._build_attention_metadata(
                total_num_scheduled_tokens=num_tokens,
                max_num_scheduled_tokens=max_query_len,
                num_reqs=num_reqs,
                ubatch_slices=ubatch_slices,
                for_cudagraph_capture=True,
            )

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            activate_lora,
            remove_lora,
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_after_padding <= self.max_num_tokens
            model_kwargs = self._init_model_kwargs(num_tokens_after_padding)
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_after_padding]
                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_after_padding]
                model_kwargs = self._init_model_kwargs(num_tokens_after_padding)
            else:
                input_ids = self.input_ids.gpu[:num_tokens_after_padding]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_after_padding]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_after_padding]
            else:
                positions = self.positions.gpu[:num_tokens_after_padding]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device,
                        )
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens_after_padding, None, False
                )

            if ubatch_slices is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_after_padding = ubatch_slices[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_after_padding

            with (
                self.maybe_randomize_inputs(input_ids),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_after_padding,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    ubatch_slices=ubatch_slices,
                ),
            ):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                use_cudagraphs = (
                    cudagraph_runtime_mode.has_mode(CUDAGraphMode.PIECEWISE)
                    and not self.speculative_config.enforce_eager
                )

                # Note(gnovack) - We need to disable cudagraphs for one of the two
                # lora cases when cudagraph_specialize_lora is enabled. This is a
                # short term mitigation for issue mentioned in
                # https://github.com/vllm-project/vllm/issues/28334
                if self.compilation_config.cudagraph_specialize_lora and activate_lora:
                    use_cudagraphs = False

                self.drafter.dummy_run(
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                )

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        logit_indices_device = torch.from_numpy(logit_indices).to(
            self.device, non_blocking=True
        )
        return hidden_states, hidden_states[logit_indices_device]

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # The dummy hidden states may contain special values,
        # like `inf` or `nan`.
        # To avoid breaking the sampler, overwrite the existing warmup buffer
        # in-place instead of allocating another full-size rand_like tensor.
        dummy_request_count = hidden_states.shape[0]

        try:
            with uvm_enabled_allocation_phase("enabled:dummy_sampler"):
                hidden_states.uniform_(-1e-3, 1e-3)

            with uvm_enabled_allocation_phase("enabled:compute_logits"):
                logits = self.model.compute_logits(hidden_states)
            num_reqs = logits.size(0)

            dummy_tensors = lambda v: torch.full(
                (num_reqs,), v, device=self.device
            )

            dummy_metadata = SamplingMetadata(
                temperature=dummy_tensors(0.5),
                all_greedy=False,
                all_random=False,
                top_p=dummy_tensors(0.9),
                top_k=dummy_tensors(logits.size(1) - 1),
                generators={},
                max_num_logprobs=None,
                no_penalties=True,
                prompt_token_ids=None,
                frequency_penalties=dummy_tensors(0.1),
                presence_penalties=dummy_tensors(0.1),
                repetition_penalties=dummy_tensors(0.1),
                output_token_ids=[[] for _ in range(num_reqs)],
                spec_token_ids=[[] for _ in range(num_reqs)],
                allowed_token_ids_mask=None,
                bad_words_token_ids={},
                logitsprocs=LogitsProcessors(),
            )
            with uvm_enabled_allocation_phase("enabled:sampler"):
                sampler_output = self.sampler(
                    logits=logits, sampling_metadata=dummy_metadata
                )
        except Exception as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{dummy_request_count} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e
        if self.speculative_config:
            draft_token_ids = [[0] for _ in range(num_reqs)]
            dummy_spec_decode_metadata = SpecDecodeMetadata.make_dummy(
                draft_token_ids, self.device
            )

            num_tokens = sum(len(ids) for ids in draft_token_ids)
            # draft_probs = torch.randn(
            #     num_tokens, logits.shape[-1], device=self.device,
            #     dtype=logits.dtype)
            draft_probs = None
            logits = torch.randn(
                num_tokens + num_reqs,
                logits.shape[-1],
                device=self.device,
                dtype=logits.dtype,
            )
            self.rejection_sampler(
                dummy_spec_decode_metadata,
                draft_probs,
                logits,
                dummy_metadata,
            )
        return sampler_output

    def _dummy_pooler_run_task(
        self,
        hidden_states: torch.Tensor,
        task: PoolingTask,
    ) -> PoolerOutput:
        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs

        req_num_tokens = num_tokens // num_reqs

        dummy_prompt_lens = torch.tensor(
            num_scheduled_tokens_list,
            device="cpu",
        )
        dummy_token_ids = torch.zeros(
            (num_reqs, req_num_tokens), dtype=torch.int32, device=self.device
        )

        model = cast(VllmModelForPooling, self.get_model())
        dummy_pooling_params = PoolingParams(task=task)
        dummy_pooling_params.verify(task=task, model_config=self.model_config)
        to_update = model.pooler.get_pooling_updates(task)
        to_update.apply(dummy_pooling_params)

        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            pooling_params=[dummy_pooling_params] * num_reqs,
        )

        dummy_metadata.build_pooling_cursor(
            num_scheduled_tokens_list, device=hidden_states.device
        )

        try:
            return model.pooler(
                hidden_states=hidden_states, pooling_metadata=dummy_metadata
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler "
                    f"({task=}) with {num_reqs} dummy requests. Please try "
                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine."
                ) from e
            else:
                raise e

    @torch.inference_mode()
    def _dummy_pooler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> PoolerOutput:
        # Find the task that has the largest output for subsequent steps
        supported_pooling_tasks = self.get_supported_pooling_tasks()

        if not supported_pooling_tasks:
            if self.scheduler_config.enable_chunked_prefill:
                raise RuntimeError(
                    f"Model {self.model_config.model} does not support "
                    "any pooling tasks with chunked prefill enabled. "
                    "Please add --no-enable-chunked-prefill to your "
                    "config or CLI args. See "
                    "https://docs.vllm.ai/en/latest/models/pooling_models.html "
                    "to learn more."
                )
            else:
                raise RuntimeError(
                    f"Model {self.model_config.model} does not support "
                    "any pooling tasks. See "
                    "https://docs.vllm.ai/en/latest/models/pooling_models.html "
                    "to learn more."
                )

        output_size = dict[PoolingTask, float]()
        for task in supported_pooling_tasks:
            # Run a full batch with each task to ensure none of them OOMs
            output = self._dummy_pooler_run_task(hidden_states, task)
            output_size[task] = sum(o.nbytes for o in output)
            del output  # Allow GC

        max_task = max(output_size.items(), key=lambda x: x[1])[0]
        return self._dummy_pooler_run_task(hidden_states, max_task)

    def profile_run(self) -> None:
        with uvm_allocation_phase("enabled:warmup"):
            # Profile with multimodal encoder & encoder cache.
            if self.supports_mm_inputs:
                mm_config = self.model_config.multimodal_config
                if mm_config is not None and mm_config.skip_mm_profiling:
                    logger.info(
                        "Skipping memory profiling for multimodal encoder and "
                        "encoder cache."
                    )
                else:
                    mm_budget = self.mm_budget
                    assert mm_budget is not None

                    if (encoder_budget := mm_budget.get_encoder_budget()) > 0:
                        # NOTE: Currently model is profiled with a single non-text
                        # modality with the max possible input tokens even when
                        # it supports multiple.
                        dummy_modality = mm_budget.get_modality_with_max_tokens()
                        max_mm_items_per_batch = mm_budget.max_items_per_batch_by_modality[
                            dummy_modality
                        ]

                        logger.info(
                            "Encoder cache will be initialized with a budget of "
                            "%s tokens, and profiled with %s %s items of the "
                            "maximum feature size.",
                            encoder_budget,
                            max_mm_items_per_batch,
                            dummy_modality,
                        )

                        # Create dummy batch of multimodal inputs.
                        batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                            dummy_modality,
                            max_mm_items_per_batch,
                        )

                        # Run multimodal encoder.
                        dummy_encoder_outputs = self.model.embed_multimodal(
                            **batched_dummy_mm_inputs
                        )

                        sanity_check_mm_encoder_outputs(
                            dummy_encoder_outputs,
                            expected_num_items=max_mm_items_per_batch,
                        )

                        # NOTE: This happens when encoder cache needs to store
                        # the embeddings that encoder outputs are scattered onto.
                        # In this case we create dummy embeddings of size
                        # (max_tokens_for_modality, hidden_size) and scatter
                        # encoder output into it.
                        encoder_output_shape = dummy_encoder_outputs[0].shape
                        max_mm_tokens_per_item = mm_budget.max_tokens_by_modality[
                            dummy_modality
                        ]
                        if encoder_output_shape[0] < max_mm_tokens_per_item:
                            encoder_hidden_size = encoder_output_shape[-1]
                            expanded_outputs = []
                            for output in dummy_encoder_outputs:
                                expanded = output.new_zeros(
                                    (max_mm_tokens_per_item, encoder_hidden_size)
                                )
                                num_tokens = output.shape[0]
                                expanded[:num_tokens].copy_(output)
                                expanded_outputs.append(expanded)

                            dummy_encoder_outputs = expanded_outputs

                        # Cache the dummy encoder outputs.
                        self.encoder_cache["tmp"] = dict(enumerate(dummy_encoder_outputs))

            # Add `is_profile` here to pre-allocate communication buffers
            with uvm_enabled_allocation_phase("enabled:dummy_run"):
                hidden_states, last_hidden_states = self._dummy_run(
                    self.max_num_tokens, is_profile=True
                )
            if get_pp_group().is_last_rank:
                if self.is_pooling_model:
                    output = self._dummy_pooler_run(hidden_states)
                else:
                    output = self._dummy_sampler_run(last_hidden_states)
            else:
                output = None
            self._sync_device()
            del hidden_states, output
            self.encoder_cache.clear()
            gc.collect()

    def _write_uvm_init_weight_placement_manifest(
        self,
        expert_modules: list[Any],
    ) -> None:
        """Persist an auditable post-warmup expert placement snapshot."""
        output_path = os.environ.get(
            "VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST", ""
        ).strip()
        if not output_path:
            return

        snapshots: list[dict[str, Any]] = []
        for module in expert_modules:
            snapshot_fn = getattr(
                module, "get_uvm_init_weight_placement_snapshot", None
            )
            if callable(snapshot_fn):
                value = snapshot_fn()
                if not isinstance(value, dict):
                    raise RuntimeError("Expert placement snapshot is not an object")
                snapshots.append(value)

        plan_files = {
            str(value.get("plan_file", ""))
            for value in snapshots
            if str(value.get("plan_file", ""))
        }
        failures: list[str] = []
        if not snapshots:
            failures.append("no_expert_snapshots")
        layer_ids = [value.get("layer_id") for value in snapshots]
        if len(layer_ids) != len(set(layer_ids)) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in layer_ids
        ):
            failures.append("layer_ids_missing_or_duplicate")
        if len(plan_files) != 1:
            failures.append("plan_file_not_unique")

        experts: list[dict[str, Any]] = []
        owners: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        for snapshot in snapshots:
            experts.extend(
                value
                for value in snapshot.get("experts", [])
                if isinstance(value, dict)
            )
            owners.extend(
                value
                for value in snapshot.get("region_owners", [])
                if isinstance(value, dict)
            )
            audits.extend(
                value
                for value in snapshot.get("release_audits", [])
                if isinstance(value, dict)
            )

        expert_keys = [
            (int(value["layer_id"]), int(value["expert_id"])) for value in experts
        ]
        if len(expert_keys) != len(set(expert_keys)):
            failures.append("duplicate_expert_coordinates")
        region_ids = [str(value.get("region_id", "")) for value in owners]
        if not region_ids or any(not value for value in region_ids):
            failures.append("region_owners_empty_or_invalid")
        if len(region_ids) != len(set(region_ids)):
            failures.append("duplicate_region_ownership")

        registry = snapshot_uvm_region_registry(region_ids)
        registry_records = registry.get("regions", [])
        registry_by_id = {
            str(value["region_id"]): value
            for value in registry_records
            if isinstance(value, dict) and value.get("region_id")
        }
        if registry.get("ok") is not True or set(registry_by_id) != set(region_ids):
            failures.append("application_registry_snapshot_incomplete")

        remote_experts = sorted(
            (int(value["layer_id"]), int(value["expert_id"]))
            for value in experts
            if value.get("placement") == "remote"
        )
        offloaded_experts = sorted(
            (int(value["layer_id"]), int(value["expert_id"]))
            for value in experts
            if value.get("placement") == "offloaded"
        )
        hot_experts = sorted(
            (int(value["layer_id"]), int(value["expert_id"]))
            for value in experts
            if value.get("placement") == "hot"
        )
        owner_by_region = {str(value["region_id"]): value for value in owners}
        remote_region_ids = sorted(
            region_id
            for region_id, owner in owner_by_region.items()
            if owner.get("placement") == "remote"
        )
        hot_region_ids = sorted(
            region_id
            for region_id, owner in owner_by_region.items()
            if owner.get("placement") == "hot"
        )
        remote_region_bytes = sum(
            int(registry_by_id[region_id]["length"])
            for region_id in remote_region_ids
            if region_id in registry_by_id
        )
        hot_region_bytes = sum(
            int(registry_by_id[region_id]["length"])
            for region_id in hot_region_ids
            if region_id in registry_by_id
        )

        audit_by_region = {
            str(value.get("region_id", "")): value
            for value in audits
            if value.get("region_id")
        }
        if set(audit_by_region) != set(remote_region_ids) | {
            region_id
            for region_id, owner in owner_by_region.items()
            if owner.get("placement") == "offloaded"
        }:
            failures.append("release_audit_region_set_mismatch")
        for region_id in remote_region_ids:
            audit = audit_by_region.get(region_id, {})
            registered = registry_by_id.get(region_id, {})
            length = int(registered.get("length", 0))
            if not (
                audit.get("ok") is True
                and audit.get("remote_read") is True
                and int(audit.get("registered_length", 0)) == length
                and int(audit.get("length", 0)) == length
                and int(audit.get("cpu_resident_bytes", 0)) >= length
                and int(audit.get("remote_mapped_bytes", 0)) >= length
                and int(audit.get("remote_map_failure_blocks", -1)) == 0
                and audit.get("app_state_after") == "cpu"
            ):
                failures.append(f"remote_release_audit_failed:{region_id}")
                break
        if any(
            registry_by_id.get(region_id, {}).get("state") != "cpu"
            for region_id in remote_region_ids
        ):
            failures.append("remote_registry_state_not_cpu")
        if any(
            registry_by_id.get(region_id, {}).get("state") != "gpu"
            for region_id in hot_region_ids
        ):
            failures.append("hot_registry_state_not_gpu")

        torch.cuda.synchronize()
        physical = query_uvm_regions_physical_residency(
            remote_region_ids,
            device=int(torch.cuda.current_device()),
            wait_for_trackers=False,
        )
        if not remote_region_ids:
            physical_ok = physical.get("ok") is True
        else:
            physical_ok = (
                physical.get("ok") is True
                and int(physical.get("queried_bytes", -1)) == remote_region_bytes
                and int(physical.get("cpu_bytes", -1)) == remote_region_bytes
                and int(physical.get("gpu_bytes", -1)) == 0
                and int(physical.get("unknown_bytes", -1)) == 0
            )
        if not physical_ok:
            failures.append("remote_physical_residency_query_failed")

        plan_path = next(iter(plan_files), "")
        plan_sha256: str | None = None
        expected_cold: list[tuple[int, int]] = []
        expected_hot: list[tuple[int, int]] = []
        if plan_path:
            try:
                with open(plan_path, "rb") as handle:
                    plan_bytes = handle.read()
                plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
                plan = json.loads(plan_bytes)
                expected_cold = sorted(
                    (int(value["layer_id"]), int(value["expert_id"]))
                    for value in plan.get("offload_plan", [])
                )
                expected_hot = sorted(
                    (int(value["layer_id"]), int(value["expert_id"]))
                    for value in plan.get("hot_plan", plan.get("prefetch_plan", []))
                )
            except Exception as exc:
                failures.append(f"plan_read_failed:{type(exc).__name__}")
        if remote_experts != expected_cold:
            failures.append("runtime_remote_coordinates_do_not_match_plan")
        if hot_experts != expected_hot:
            failures.append("runtime_hot_coordinates_do_not_match_plan")
        if offloaded_experts:
            failures.append("unexpected_restore_offloaded_experts")

        manifest = {
            "schema_version": 1,
            "status": "ok" if not failures else "error",
            "complete": not failures,
            "failures": failures,
            "created_at_epoch": time.time(),
            "plan_file": plan_path,
            "plan_sha256": plan_sha256,
            "layer_count": len(snapshots),
            "expert_count": len(experts),
            "remote_expert_count": len(remote_experts),
            "hot_expert_count": len(hot_experts),
            "offloaded_expert_count": len(offloaded_experts),
            "remote_experts": [list(value) for value in remote_experts],
            "hot_experts": [list(value) for value in hot_experts],
            "remote_region_count": len(remote_region_ids),
            "hot_region_count": len(hot_region_ids),
            "remote_region_bytes": remote_region_bytes,
            "hot_region_bytes": hot_region_bytes,
            "release_time_remote_mapping_verified": not any(
                value.startswith("remote_release_audit_failed")
                or value == "release_audit_region_set_mismatch"
                for value in failures
            ),
            "snapshot_time_remote_physical_residency": physical,
            "snapshot_time_remote_mapping_query_available": False,
            "registry": registry,
            "release_audits": audits,
            "module_snapshots": snapshots,
        }
        parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent, exist_ok=True)
        temporary = f"{output_path}.tmp.{os.getpid()}"
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, output_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        if failures:
            raise RuntimeError(
                "UVM placement manifest failed integrity checks: "
                + ", ".join(failures[:8])
            )

    def run_uvm_init_time_weight_placement(self) -> None:
        """Finalize cold and Hot Expert placement after model warmup."""
        self._run_uvm_matched_cpu_offload_manifest()
        if not self._env_flag_enabled(
            "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE",
            default=False,
        ):
            return
        if self._uvm_init_weight_placement_done:
            return

        expert_modules: list[Any] = []
        hot_modules: list[Any] = []
        hot_region_ids: list[str] = []
        try:
            # Complete all Cold releases before placing the remaining Hot Experts.
            with uvm_enabled_allocation_phase("enabled:init_weight_offload"):
                for module in self.get_model().modules():
                    init_offload = getattr(
                        module, "run_uvm_init_weight_offload", None
                    )
                    if not callable(init_offload):
                        continue
                    init_offload()
                    expert_modules.append(module)

                bank_candidates: list[
                    tuple[int, list[tuple[int, list[str]]]]
                ] = []
                for module in expert_modules:
                    collect_groups = getattr(
                        module,
                        "get_uvm_slot_init_expert_groups",
                        None,
                    )
                    if not callable(collect_groups):
                        continue
                    candidate = collect_groups()
                    if candidate is not None:
                        bank_candidates.append(candidate)
                bank_candidates.sort(key=lambda item: item[0])
                self._uvm_slot_init_bank_candidates = bank_candidates

                for module in expert_modules:
                    region_ids = module.get_uvm_init_hot_expert_region_ids()
                    if region_ids:
                        hot_modules.append(module)
                        hot_region_ids.extend(region_ids)

            if hot_region_ids:
                device = int(torch.cuda.current_device())
                result = prepare_uvm_regions_for_gpu_placement(
                    hot_region_ids,
                    device=device,
                    executor=hot_modules[0].uvm_weight_executor,
                )
                if not result["ok"]:
                    raise RuntimeError(
                        f"Hot Expert prefetch failed: result={result!r}"
                    )

                torch.cuda.synchronize(device)
                result = commit_uvm_regions_gpu_placement(
                    hot_region_ids,
                    synchronized=True,
                )
                if not result["ok"]:
                    raise RuntimeError("Hot Expert placement commit failed")

                for module in hot_modules:
                    module.set_uvm_init_hot_expert_state("gpu")

            self._uvm_init_weight_placement_done = True
            GPUModelRunner._write_uvm_init_weight_placement_manifest(
                self, expert_modules
            )
        except Exception:
            if hot_region_ids:
                commit_uvm_regions_gpu_placement(
                    hot_region_ids,
                    synchronized=False,
                )
            for module in hot_modules:
                module.set_uvm_init_hot_expert_state("unknown")
            self._uvm_init_weight_placement_done = False
            raise

    @staticmethod
    def _validate_uvm_slot_64k_layout(
        ranges: Sequence[tuple[int, int]],
        *,
        layer_id: int,
    ) -> None:
        """Validate sparse 64 KiB plan geometry before mode 8 runs.

        Missing portions of a 2 MiB block are legal and remain ordinary hot
        leaves. The registered plan still explicitly selects the 64 KiB path;
        this validation never converts sparse geometry to a 4 KiB plan.
        """
        page_bytes = 64 * 1024
        va_block_bytes = 2 * 1024 * 1024
        max_entry_bytes = 4 * 1024 * 1024
        uint64_max = (1 << 64) - 1
        block_slices: dict[int, list[tuple[int, int]]] = {}

        for index, (base, length) in enumerate(ranges):
            if base <= 0 or length <= 0:
                raise RuntimeError(
                    f"Registered UVM slot layer {layer_id} range {index} has "
                    "a non-positive base or length"
                )
            if length > max_entry_bytes:
                raise RuntimeError(
                    f"Registered UVM slot layer {layer_id} range {index} "
                    "exceeds the 4 MiB slot-direct limit"
                )
            if base > uint64_max - length:
                raise RuntimeError(
                    f"Registered UVM slot layer {layer_id} range {index} "
                    "virtual-address end overflows uint64"
                )
            if base % page_bytes or length % page_bytes:
                raise RuntimeError(
                    f"Registered UVM slot layer {layer_id} range {index} is "
                    "not 64 KiB aligned"
                )

            cursor = base
            end = base + length
            while cursor < end:
                block_base = cursor - cursor % va_block_bytes
                slice_end = min(end, block_base + va_block_bytes)
                block_slices.setdefault(block_base, []).append(
                    (cursor, slice_end)
                )
                cursor = slice_end

        for block_base, slices in block_slices.items():
            previous_end: int | None = None
            for start, end in sorted(slices):
                if previous_end is not None and start < previous_end:
                    raise RuntimeError(
                        "Registered UVM slot 64 KiB ranges overlap; "
                        f"layer {layer_id} block 0x{block_base:x} contains "
                        "overlapping slices"
                    )
                previous_end = end

    @staticmethod
    def _validate_uvm_slot_init_plan_candidates(
        candidates: list[tuple[int, list[tuple[int, list[str]]]]],
        *,
        registered_plan: bool,
    ) -> None:
        """Reject an invalid immutable plan set before the first mode-8 call."""
        if not candidates:
            raise RuntimeError("Slot init-bank has no layer candidates")

        layer_ids = [layer_id for layer_id, _groups in candidates]
        if any(
            isinstance(layer_id, bool)
            or not isinstance(layer_id, int)
            or layer_id < 0
            or layer_id > (1 << 32) - 1
            for layer_id in layer_ids
        ):
            raise RuntimeError("Slot init-bank layer_id is outside uint32")
        if len(layer_ids) != len(set(layer_ids)):
            raise RuntimeError("Slot init-bank layer_id values must be unique")

        if registered_plan:
            transition_count = len(candidates)
            total_plan_count = 1 + transition_count
            if transition_count > 63 or total_plan_count > 64:
                raise RuntimeError(
                    "Registered UVM slot plans exceed the kernel limit: "
                    f"transition={transition_count}, total={total_plan_count}, "
                    "maximum transition=63, total=64"
                )
            initial_plan_id = (min(layer_ids) << 2) | 1
            plan_ids = [initial_plan_id] + [
                (layer_id << 2) | 2 for layer_id in layer_ids
            ]
            if len(plan_ids) != len(set(plan_ids)):
                raise RuntimeError("Registered UVM slot plan_id values must be unique")

        reference_expert_shapes: tuple[tuple[int, ...], ...] | None = None
        reference_lane_lengths: tuple[int, ...] | None = None
        reference_lane_count: int | None = None
        for layer_id, groups in candidates:
            if not groups:
                raise RuntimeError(f"Slot layer {layer_id} has no expert groups")
            expert_ids = [expert_id for expert_id, _regions in groups]
            if any(
                isinstance(expert_id, bool)
                or not isinstance(expert_id, int)
                or expert_id < 0
                or expert_id > (1 << 32) - 1
                for expert_id in expert_ids
            ):
                raise RuntimeError(f"Slot layer {layer_id} has an invalid expert_id")
            if len(expert_ids) != len(set(expert_ids)):
                raise RuntimeError(f"Slot layer {layer_id} repeats an expert_id")
            if any(not regions for _expert_id, regions in groups):
                raise RuntimeError(f"Slot layer {layer_id} has an empty expert lane set")

            region_ids = [
                str(region_id)
                for _expert_id, regions in groups
                for region_id in regions
            ]
            if not region_ids or any(not value for value in region_ids):
                raise RuntimeError(f"Slot layer {layer_id} has an empty lane")
            if len(region_ids) != len(set(region_ids)):
                raise RuntimeError(f"Slot layer {layer_id} repeats a lane")
            if registered_plan and len(region_ids) > 1152:
                raise RuntimeError(
                    f"Slot layer {layer_id} exceeds the 1152-region model contract"
                )
            snapshot = snapshot_uvm_region_registry(region_ids)
            if snapshot.get("ok") is not True:
                raise RuntimeError(
                    f"Slot layer {layer_id} registry geometry is incomplete"
                )
            geometry: dict[str, tuple[int, int]] = {}
            for record in snapshot.get("regions", []):
                if not isinstance(record, dict):
                    continue
                region_id = str(record.get("region_id", ""))
                base = record.get("base")
                length = record.get("length")
                if (
                    not region_id
                    or not isinstance(base, int)
                    or isinstance(base, bool)
                    or not isinstance(length, int)
                    or isinstance(length, bool)
                    or region_id in geometry
                ):
                    raise RuntimeError(
                        f"Slot layer {layer_id} registry geometry is invalid"
                    )
                geometry[region_id] = (base, length)
            if set(geometry) != set(region_ids):
                raise RuntimeError(
                    f"Slot layer {layer_id} registry geometry is incomplete"
                )
            if registered_plan:
                GPUModelRunner._validate_uvm_slot_64k_layout(
                    list(geometry.values()),
                    layer_id=layer_id,
                )
            lengths = {
                region_id: length
                for region_id, (_base, length) in geometry.items()
            }
            shapes = {
                int(expert_id): tuple(
                    sorted(lengths[str(region_id)] for region_id in regions)
                )
                for expert_id, regions in groups
            }
            expert_shapes = tuple(sorted(shapes.values()))
            lane_lengths = tuple(sorted(lengths.values()))
            if reference_expert_shapes is None:
                reference_expert_shapes = expert_shapes
                reference_lane_lengths = lane_lengths
                reference_lane_count = len(region_ids)
            elif (
                expert_shapes != reference_expert_shapes
                or lane_lengths != reference_lane_lengths
                or len(region_ids) != reference_lane_count
            ):
                raise RuntimeError(
                    "Registered UVM slot layers must have identical lane-size "
                    f"geometry; mismatch at layer {layer_id}"
                )

    def finalize_uvm_slot_bank_after_warmup(self) -> None:
        """Reserve the fixed bank only after all model warmup has finished.

        Weight placement may run at ``pre_kv`` for other policies. Reserving
        576 MiB there would perturb KV allocation, compilation, autotuning, or
        CUDA graph capture. GPUWorker calls this finalizer immediately before
        the compile/warmup RPC returns, which is still before serving starts.
        """
        init_bank = self._env_flag_enabled(
            "VLLM_UVM_SLOT_DIRECT_INIT_BANK", default=False
        )
        registered_plan = self._env_flag_enabled(
            "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", default=False
        )
        dual_bank_event = self._env_flag_enabled(
            "VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT", default=False
        )
        cuda_fence = self._env_flag_enabled(
            "VLLM_UVM_SLOT_DIRECT_CUDA_FENCE", default=False
        )
        access_mode = os.environ.get(
            "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "restore"
        ).strip().lower()
        prefill_p3_decode_remote = self._uvm_prefill_p3_decode_remote_enabled()
        if prefill_p3_decode_remote:
            missing: list[str] = []
            if access_mode != "slot_restore":
                missing.append("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=slot_restore")
            for name, enabled in (
                ("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", registered_plan),
                ("VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT", dual_bank_event),
                ("VLLM_UVM_SLOT_DIRECT_CUDA_FENCE", cuda_fence),
            ):
                if not enabled:
                    missing.append(name)
            if missing:
                raise RuntimeError(
                    "P3-prefill/remote-read-decode requires the complete P3 "
                    "fence configuration; missing: " + ", ".join(missing)
                )
            unsupported: list[str] = []
            if bool(getattr(self.scheduler_config, "enable_chunked_prefill", False)):
                unsupported.append("chunked prefill")
            if int(getattr(self.scheduler_config, "max_num_seqs", 1)) != 1:
                unsupported.append("max_num_seqs != 1")
            if bool(getattr(self, "use_async_scheduling", False)):
                unsupported.append("async scheduling")
            if self.speculative_config is not None:
                unsupported.append("speculative decoding")
            if unsupported:
                raise RuntimeError(
                    "P3-prefill/remote-read-decode currently requires one "
                    "synchronous, unchunked request; unsupported: "
                    + ", ".join(unsupported)
                )
        if init_bank and access_mode != "slot_restore":
            raise RuntimeError(
                "UVM slot init-bank requires "
                "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=slot_restore so the "
                "frozen shutdown contract owns its cleanup"
            )
        if registered_plan:
            required_flags = (
                "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE",
                "VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD",
                "VLLM_UVM_SLOT_DIRECT_INIT_BANK",
                "VLLM_UVM_SLOT_DIRECT_REUSE_BANK",
                "VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES",
                "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION",
                "VLLM_UVM_SLOT_DIRECT_SHARED_PUSH",
                "VLLM_UVM_SLOT_DIRECT_64K_PTE",
            )
            disabled = [
                name
                for name in required_flags
                if not self._env_flag_enabled(name, default=False)
            ]
            if access_mode != "slot_restore":
                disabled.append(
                    "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=slot_restore"
                )
            if disabled:
                raise RuntimeError(
                    "Registered UVM slot plans require the complete fixed-bank "
                    "configuration; missing: " + ", ".join(disabled)
                )
        if dual_bank_event and not registered_plan:
            raise RuntimeError(
                "UVM slot dual-bank event mode requires registered plans"
            )
        if cuda_fence and (not registered_plan or not dual_bank_event):
            raise RuntimeError(
                "UVM slot CUDA fence mode requires registered plans and "
                "dual-bank event mode"
            )

        bank_candidates = getattr(self, "_uvm_slot_init_bank_candidates", None)
        if init_bank:
            if not bank_candidates:
                raise RuntimeError(
                    "Slot init-bank was requested but no cold slot layer was "
                    "registered before warmup completion"
                )
            self._validate_uvm_slot_init_plan_candidates(
                bank_candidates,
                registered_plan=registered_plan,
            )
            if dual_bank_event:
                layer_ids = [int(layer_id) for layer_id, _groups in bank_candidates]
                expected = list(
                    range(min(layer_ids), min(layer_ids) + len(layer_ids))
                )
                if len(layer_ids) < 2 or len(layer_ids) % 2 != 0:
                    raise RuntimeError(
                        "Dual-bank registered UVM slot plans require a non-empty "
                        "even number of layers"
                    )
                if layer_ids != expected:
                    raise RuntimeError(
                        "Dual-bank registered UVM slot layer ids must be contiguous"
                    )

        device = int(torch.cuda.current_device())
        contract = freeze_uvm_slot_direct_session_contract(
            registered_plan=registered_plan,
            access_mode=access_mode,
            device=device,
        )
        if init_bank and not bool(contract.get("slot_session_enabled")):
            raise RuntimeError(
                "Slot init-bank contract does not enable scoped session cleanup"
            )
        self._uvm_slot_session_contract = contract
        if not init_bank:
            return
        if getattr(self, "_uvm_slot_init_bank_done", False):
            return
        assert bank_candidates is not None
        bank_layer_id, bank_groups = bank_candidates[0]
        try:
            bank_result = preallocate_uvm_expert_groups_to_slots(
                bank_groups,
                layer_id=bank_layer_id,
                device=device,
            )
            if not bool(bank_result.get("ok")):
                raise RuntimeError(
                    "Slot init-bank preallocation failed: "
                    f"layer_id={bank_layer_id}, result={bank_result!r}"
                )
            bank_results = [bank_result]
            if dual_bank_event:
                second_layer_id, second_bank_groups = bank_candidates[1]
                second_bank_result = preallocate_uvm_registered_plan_second_bank(
                    second_bank_groups,
                    layer_id=second_layer_id,
                    device=device,
                )
                if not bool(second_bank_result.get("ok")):
                    raise RuntimeError(
                        "Slot second-bank preallocation failed: "
                        f"layer_id={second_layer_id}, result={second_bank_result!r}"
                    )
                bank_results.append(second_bank_result)
            self._uvm_slot_init_bank_result = (
                {
                    "ok": True,
                    "dual_bank_event": True,
                    "bank_results": bank_results,
                }
                if dual_bank_event
                else bank_result
            )
            if registered_plan:
                plan_result = register_uvm_slot_direct_layer_plans(
                    bank_candidates,
                    device=device,
                )
                if not bool(plan_result.get("ok")):
                    raise RuntimeError(
                        "Slot layer-plan registration failed: "
                        f"result={plan_result!r}"
                    )
                self._uvm_slot_registered_plan_result = plan_result
            self._uvm_slot_init_bank_done = True
        except Exception as exc:
            cleanup_result: dict[str, Any] | None = None
            cleanup_exc: Exception | None = None
            try:
                cleanup_result = close_uvm_slot_direct_policy_session(
                    device=device
                )
            except Exception as rollback_exc:
                cleanup_exc = rollback_exc
            finally:
                # Runner state must never advertise a usable bank after any
                # initialization or rollback failure, including an exception
                # raised directly by the cleanup wrapper.
                self._uvm_slot_init_bank_done = False
                self._uvm_slot_init_bank_result = None
                self._uvm_slot_registered_plan_result = None
                self._uvm_slot_session_contract = None
            if cleanup_exc is not None:
                raise RuntimeError(
                    f"{exc}; slot init rollback raised: {cleanup_exc!r}"
                ) from exc
            assert cleanup_result is not None
            if not bool(cleanup_result.get("ok")):
                raise RuntimeError(
                    f"{exc}; slot init rollback also failed: {cleanup_result!r}"
                ) from exc
            raise

    @staticmethod
    def _uvm_incremental_expert_layer_key(module: Any) -> str:
        layer_name = str(getattr(module, "layer_name", "") or "")
        if layer_name:
            return layer_name
        layer_id_fn = getattr(module, "_uvm_layer_id", None)
        layer_id = layer_id_fn() if callable(layer_id_fn) else None
        return f"layer:{layer_id}"

    def get_uvm_incremental_expert_offload_candidates(
        self,
        expert_class: str,
    ) -> dict[str, Any]:
        """Return one local candidate for every MoE layer, or no wave.

        A KV resize must never reclaim experts from only the layers that
        happen to have a candidate.  If one layer is exhausted, discard the
        whole class-specific wave so the caller can try another class (or
        leave the HBM as safety headroom).
        """
        normalized_class = str(expert_class).strip().lower()
        if normalized_class not in ("cold", "warm"):
            return {"ok": False, "candidates": {}}
        candidates: dict[str, int] = {}
        eligible_layers: set[str] = set()
        for module in self.get_model().modules():
            next_candidate = getattr(
                module,
                "uvm_next_incremental_expert",
                None,
            )
            if not callable(next_candidate):
                continue
            layer_key = self._uvm_incremental_expert_layer_key(module)
            if layer_key in eligible_layers:
                return {"ok": False, "candidates": {}}
            eligible_layers.add(layer_key)
            global_expert_id = next_candidate(normalized_class)
            if global_expert_id is None:
                continue
            candidates[layer_key] = int(global_expert_id)
        if candidates and set(candidates) != eligible_layers:
            # Partial waves violate the equal-per-layer residency invariant.
            # Returning no candidates lets EngineCore try the all-layer warm
            # class, while still preventing any mutation in this class.
            candidates = {}
        return {"ok": True, "candidates": candidates}

    def _begin_uvm_full_dual_residency_mutation(
        self,
        *,
        reason: str,
    ) -> dict[str, Any] | None:
        """Expose ordinary UVM leaves before changing expert residency.

        A full-dual session freezes complete 2 MiB children, including the
        ordinary (non-alias) leaves which share a parent with cold experts.
        Residency mutations must therefore use the same discard/PARK boundary
        as a synchronous Plan Class switch.  The matching finish helper
        rebuilds both children before either CUDA Graph can replay.
        """
        if not self._env_flag_enabled(
            "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW",
            default=False,
        ):
            return None
        from vllm.device_allocator import uvm_static_alias
        from vllm.model_executor.models import utils as offload_utils

        diagnostics = (
            offload_utils.get_cpu_offload_graph_static_alias_diagnostics()
        )
        state_before = str(diagnostics.get("lifecycle_state") or "")
        session_id = int(diagnostics.get("session_id") or 0)
        if state_before not in ("sealed", "remote_parked") or session_id <= 0:
            raise RuntimeError(
                "Full-dual residency mutation requires a sealed or parked "
                f"session: state={state_before!r}, session_id={session_id}"
            )
        torch.cuda.synchronize(self.device)
        discarded = dict(
            uvm_static_alias.driver_static_alias_seal(
                session_id,
                device=int(self.device.index),
                discard_dual=True,
            )
        )
        if discarded.get("ok") is not True:
            raise RuntimeError(
                f"Failed to discard full-dual children before {reason}: "
                f"{discarded!r}"
            )
        parked = None
        if state_before == "sealed":
            parked = dict(
                uvm_static_alias.driver_static_alias_park(
                    session_id,
                    device=int(self.device.index),
                    synchronize=False,
                    defer_cpu_restore=False,
                )
            )
            if parked.get("ok") is not True:
                raise RuntimeError(
                    f"Failed to PARK aliases before {reason}: {parked!r}"
                )
            offload_utils._set_uvm_original_alias_parked_state(parked)
        torch.cuda.synchronize(self.device)
        return {
            "reason": str(reason),
            "state_before": state_before,
            "session_id": session_id,
            "dual_discard": discarded,
            "park": parked,
        }

    def _finish_uvm_full_dual_residency_mutation(
        self,
        transaction: dict[str, Any] | None,
        *,
        canonicalize_runtime_hot_ranges: tuple[tuple[int, int], ...] | None = None,
    ) -> dict[str, Any] | None:
        if transaction is None:
            return None
        from vllm.device_allocator import uvm_static_alias
        from vllm.model_executor.models import utils as offload_utils

        session_id = int(transaction["session_id"])
        torch.cuda.synchronize(self.device)
        # DISCARD followed by PARK leaves the session's read-only consumer
        # aliases absent from the live host page table. A fresh dual recipe
        # cannot be built from that parked topology because there are no slot
        # leaves to freeze. First restore the legacy live aliases, then freeze
        # the rebuilt GPU/CPU children. If the caller entered in the parked
        # state, switch back to the CPU child only after the recipe exists.
        resumed = dict(
            uvm_static_alias.driver_static_alias_resume(
                session_id,
                device=int(self.device.index),
                synchronize=False,
            )
        )
        if resumed.get("ok") is not True:
            raise RuntimeError(
                "Failed to RESUME aliases before rebuilding full-dual "
                f"children after {transaction['reason']}: {resumed!r}"
            )
        offload_utils._set_uvm_original_alias_sealed_state(resumed)
        torch.cuda.synchronize(self.device)
        hot_reconciliation = None
        if canonicalize_runtime_hot_ranges is not None:
            # A dynamic exchange changes ordinary hot leaves while the
            # complete dual children are discarded.  Residency helpers can
            # report a successful H2D duplicate even when the live mapping is
            # still the immutable CPU shadow.  Canonicalize the complete
            # non-remote hot complement before freezing the new GPU child.
            # This is deliberately inside the discard/refresh transaction:
            # doing it after refresh would repair ordinary leaves but leave
            # the retained Prefill child stale.
            reconcile_hot = (
                offload_utils.
                reconcile_cpu_offload_uvm_original_va_sealed_hot_residency
            )
            hot_reconciliation = reconcile_hot(
                preserve_allocator_cache=True,
                runtime_remote_ranges=(
                    self._collect_uvm_runtime_remote_expert_ranges()
                ),
                mapping_refresh_ranges=canonicalize_runtime_hot_ranges,
            )
            if hot_reconciliation.get("ok") is not True:
                raise RuntimeError(
                    "Failed to canonicalize dynamic hot experts before "
                    f"full-dual refresh: {hot_reconciliation!r}"
                )
        refreshed = dict(
            uvm_static_alias.driver_static_alias_seal(
                session_id,
                device=int(self.device.index),
                refresh_dual=True,
            )
        )
        if refreshed.get("ok") is not True:
            raise RuntimeError(
                "Failed to rebuild full-dual children after "
                f"{transaction['reason']}: {refreshed!r}"
            )
        parked = None
        if transaction["state_before"] == "remote_parked":
            parked = dict(
                uvm_static_alias.driver_static_alias_park(
                    session_id,
                    device=int(self.device.index),
                    synchronize=False,
                    defer_cpu_restore=False,
                )
            )
            if parked.get("ok") is not True:
                raise RuntimeError(
                    "Failed to PARK aliases after rebuilding full-dual "
                    f"children after {transaction['reason']}: {parked!r}"
                )
            offload_utils._set_uvm_original_alias_parked_state(parked)
        torch.cuda.synchronize(self.device)
        return {
            "resume_before_dual_refresh": resumed,
            "hot_reconciliation_before_dual_refresh": hot_reconciliation,
            "dual_refresh": refreshed,
            "resume": resumed if transaction["state_before"] == "sealed" else None,
            "park_after_dual_refresh": parked,
            "state_after": transaction["state_before"],
        }

    def run_uvm_incremental_expert_offload(
        self,
        allow_warm: bool | None = None,
        selected_experts_by_layer: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Release the EngineCore-selected expert in each local MoE layer."""
        if not self._env_flag_enabled(
            "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE",
            default=False,
        ):
            return {"ok": True}

        modules = [
            module
            for module in self.get_model().modules()
            if callable(
                getattr(module, "run_uvm_incremental_weight_offload", None)
            )
        ]
        module_layers = {
            self._uvm_incremental_expert_layer_key(module) for module in modules
        }
        if len(module_layers) != len(modules):
            return {"ok": False, "reason": "duplicate_expert_layer_key"}
        if (
            selected_experts_by_layer is not None
            and set(selected_experts_by_layer) != module_layers
        ):
            return {
                "ok": False,
                "reason": "unbalanced_expert_layer_selection",
                "selected_layer_count": len(selected_experts_by_layer),
                "expected_layer_count": len(module_layers),
            }
        full_dual_transaction = self._begin_uvm_full_dual_residency_mutation(
            reason="incremental_expert_offload"
        )
        has_cold = any(
            bool(module.has_remaining_uvm_incremental_experts("cold"))
            for module in modules
        )
        effective_allow_warm = (
            not has_cold if allow_warm is None else bool(allow_warm)
        )
        released_all = True
        visited_layers: set[str] = set()
        release_results: list[dict[str, Any]] = []
        with uvm_enabled_allocation_phase("enabled:incremental_weight_offload"):
            for module in modules:
                selected_global_expert_id: int | None = None
                if selected_experts_by_layer is not None:
                    layer_key = self._uvm_incremental_expert_layer_key(module)
                    selected_value = selected_experts_by_layer.get(layer_key)
                    if selected_value is None:
                        continue
                    selected_global_expert_id = int(selected_value)
                    visited_layers.add(layer_key)
                result = module.run_uvm_incremental_weight_offload(
                    allow_warm=effective_allow_warm,
                    selected_global_expert_id=selected_global_expert_id,
                )
                layer_key = self._uvm_incremental_expert_layer_key(module)
                release_results.append(
                    {
                        "layer_key": layer_key,
                        "selected_global_expert_id": selected_global_expert_id,
                        **dict(result),
                    }
                )
                released = {
                    int(expert_id)
                    for expert_id in result.get("released_experts", [])
                }
                released_all = released_all and bool(result.get("ok"))
                if selected_global_expert_id is not None:
                    released_all = (
                        released_all
                        and selected_global_expert_id in released
                    )
        if selected_experts_by_layer is not None:
            released_all = released_all and visited_layers == set(
                selected_experts_by_layer
            )
        full_dual_completion = self._finish_uvm_full_dual_residency_mutation(
            full_dual_transaction
        )
        response = {
            "ok": released_all,
            "allow_warm": effective_allow_warm,
            "selected_layer_count": (
                len(selected_experts_by_layer)
                if selected_experts_by_layer is not None
                else len(release_results)
            ),
            "visited_layer_count": len(visited_layers),
            "released_expert_count": sum(
                int(
                    result.get("released_count")
                    or len(result.get("released_experts") or [])
                )
                for result in release_results
            ),
            "results": release_results,
        }
        if full_dual_transaction is not None:
            response["full_dual_transaction"] = full_dual_transaction
            response["full_dual_completion"] = full_dual_completion
        return response

    def get_uvm_incremental_expert_restore_candidates(self) -> dict[str, Any]:
        """Return one autoscale-offloaded expert per layer, or no wave."""
        candidates: dict[str, int] = {}
        eligible_layers: set[str] = set()
        for module in self.get_model().modules():
            next_candidate = getattr(
                module,
                "uvm_next_incremental_restore_expert",
                None,
            )
            if not callable(next_candidate):
                continue
            layer_key = self._uvm_incremental_expert_layer_key(module)
            if layer_key in eligible_layers:
                return {"ok": False, "candidates": {}}
            eligible_layers.add(layer_key)
            global_expert_id = next_candidate()
            if global_expert_id is None:
                continue
            candidates[layer_key] = int(global_expert_id)
        if candidates and set(candidates) != eligible_layers:
            # A partial restore would make per-layer expert capacity diverge.
            # Treat it as inconsistent state and keep the pending retry set.
            return {"ok": False, "candidates": {}}
        return {"ok": True, "candidates": candidates}

    def run_uvm_incremental_expert_restore(
        self,
        selected_experts_by_layer: dict[str, int],
    ) -> dict[str, Any]:
        """Restore exactly the selected balanced runtime-offload wave."""
        selected = {
            str(layer_key): int(expert_id)
            for layer_key, expert_id in selected_experts_by_layer.items()
        }
        modules = [
            module
            for module in self.get_model().modules()
            if callable(
                getattr(module, "run_uvm_incremental_weight_restore", None)
            )
        ]
        module_layers = {
            self._uvm_incremental_expert_layer_key(module) for module in modules
        }
        if len(module_layers) != len(modules):
            return {"ok": False, "reason": "duplicate_expert_layer_key"}
        if set(selected) != module_layers:
            return {
                "ok": False,
                "reason": "unbalanced_expert_layer_selection",
                "selected_layer_count": len(selected),
                "expected_layer_count": len(module_layers),
            }
        visited_layers: set[str] = set()
        restore_results: list[dict[str, Any]] = []
        restored_modules: dict[str, tuple[Any, int]] = {}
        rollback_results: list[dict[str, Any]] = []
        restored_all = True
        with uvm_enabled_allocation_phase("enabled:incremental_weight_restore"):
            for module in modules:
                restore = getattr(
                    module,
                    "run_uvm_incremental_weight_restore",
                    None,
                )
                if not callable(restore):
                    continue
                layer_key = self._uvm_incremental_expert_layer_key(module)
                selected_expert = selected.get(layer_key)
                if selected_expert is None:
                    continue
                visited_layers.add(layer_key)
                try:
                    result = restore(
                        selected_global_expert_id=int(selected_expert),
                    )
                except Exception as exc:
                    result = {
                        "ok": False,
                        "reason": "expert_restore_raised",
                        "error": repr(exc),
                        "restored_experts": [],
                    }
                restore_results.append(
                    {
                        "layer_key": layer_key,
                        "selected_global_expert_id": int(selected_expert),
                        **dict(result),
                    }
                )
                restored = {
                    int(expert_id)
                    for expert_id in result.get("restored_experts", [])
                }
                if int(selected_expert) in restored:
                    restored_modules[layer_key] = (
                        module,
                        int(selected_expert),
                    )
                satisfied = (
                    bool(result.get("ok"))
                    and int(selected_expert) in restored
                ) or bool(result.get("already_local"))
                restored_all = (
                    restored_all
                    and satisfied
                )
            restored_all = restored_all and visited_layers == set(selected)
            rollback_ok = True
            if not restored_all:
                # Restoration happens after KV shrink and is allowed to fail,
                # but a partial success must not leave different expert counts
                # across layers. Re-offload every expert that was restored in
                # this attempt so a later maintenance round can retry the same
                # balanced wave.
                for layer_key, (module, expert_id) in restored_modules.items():
                    release = getattr(
                        module,
                        "run_uvm_incremental_weight_offload",
                        None,
                    )
                    if not callable(release):
                        rollback = {
                            "ok": False,
                            "reason": "expert_restore_rollback_unavailable",
                            "released_experts": [],
                        }
                    else:
                        try:
                            rollback = release(
                                allow_warm=True,
                                selected_global_expert_id=expert_id,
                            )
                        except Exception as exc:
                            rollback = {
                                "ok": False,
                                "reason": "expert_restore_rollback_raised",
                                "error": repr(exc),
                                "released_experts": [],
                            }
                    released = {
                        int(value)
                        for value in rollback.get("released_experts", [])
                    }
                    item_ok = bool(rollback.get("ok")) and expert_id in released
                    rollback_ok = rollback_ok and item_ok
                    rollback_results.append(
                        {
                            "layer_key": layer_key,
                            "selected_global_expert_id": expert_id,
                            **dict(rollback),
                            "ok": item_ok,
                        }
                    )
        response = {
            "ok": restored_all,
            "reason": "restored" if restored_all else "restore_incomplete",
            "selected_layer_count": len(selected),
            "visited_layer_count": len(visited_layers),
            "restored_expert_count": sum(
                int(
                    result.get("restored_count")
                    or len(result.get("restored_experts") or [])
                )
                for result in restore_results
            ),
            "results": restore_results,
        }
        if not restored_all:
            response.update(
                {
                    "rollback_ok": rollback_ok,
                    "rollback_results": rollback_results,
                    "net_restored_expert_count": (
                        0 if rollback_ok else len(restored_modules)
                    ),
                }
            )
        return response

    def get_uvm_dynamic_expert_heat(
        self,
        reset: bool = False,
    ) -> dict[str, Any]:
        """Collect per-layer routing heat and current remote expert sets."""
        if not self._env_flag_enabled(
            "VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE",
            default=False,
        ):
            return {"ok": False, "reason": "dynamic_expert_swap_disabled"}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        layers: dict[str, dict[str, Any]] = {}
        for module in self.get_model().modules():
            snapshot = getattr(module, "get_uvm_dynamic_expert_heat", None)
            if not callable(snapshot):
                continue
            layer_key = self._uvm_incremental_expert_layer_key(module)
            value = snapshot(reset=bool(reset))
            if isinstance(value, dict):
                layers[layer_key] = value
        return {
            "ok": True,
            "reason": "heat_collected",
            "reset": bool(reset),
            "layer_count": len(layers),
            "layers": layers,
        }

    def run_uvm_dynamic_expert_swaps(
        self,
        swaps_by_layer: dict[str, list[dict[str, object]]],
    ) -> dict[str, Any]:
        """Execute EngineCore-selected remote/GPU expert exchanges."""
        if not self._env_flag_enabled(
            "VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE",
            default=False,
        ):
            return {"ok": False, "reason": "dynamic_expert_swap_disabled"}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        full_dual_transaction = self._begin_uvm_full_dual_residency_mutation(
            reason="dynamic_expert_swap"
        )
        results: list[dict[str, object]] = []
        visited_layers: set[str] = set()
        with uvm_enabled_allocation_phase("enabled:dynamic_expert_swap"):
            for module in self.get_model().modules():
                run_swaps = getattr(
                    module,
                    "run_uvm_dynamic_remote_read_swaps",
                    None,
                )
                if not callable(run_swaps):
                    continue
                layer_key = self._uvm_incremental_expert_layer_key(module)
                layer_swaps = swaps_by_layer.get(layer_key)
                if not layer_swaps:
                    continue
                visited_layers.add(layer_key)
                value = run_swaps(layer_swaps)
                results.append({**value, "layer_key": layer_key})
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        missing_layers = sorted(set(swaps_by_layer) - visited_layers)
        execution_ok = (
            not missing_layers
            and all(bool(result.get("ok")) for result in results)
        )
        promoted_ranges = tuple(
            (int(item["base"]), int(item["length"]))
            for result in results
            for pair in result.get("results", [])
            for item in pair.get("promote_ranges", [])
        )
        full_dual_completion = self._finish_uvm_full_dual_residency_mutation(
            full_dual_transaction,
            canonicalize_runtime_hot_ranges=promoted_ranges,
        )
        response = {
            "ok": execution_ok,
            "reason": "swaps_executed" if not missing_layers else "layers_missing",
            "selected_layers": len(swaps_by_layer),
            "visited_layers": len(visited_layers),
            "missing_layers": missing_layers,
            "swapped_count": sum(
                int(result.get("swapped_count") or 0) for result in results
            ),
            "results": results,
        }
        if full_dual_transaction is not None:
            response["full_dual_transaction"] = full_dual_transaction
            response["full_dual_completion"] = full_dual_completion
        return response

    def _load_uvm_matched_cpu_offload_manifest(self) -> list[dict[str, object]]:
        manifest_file = os.environ.get(
            "VLLM_UVM_MATCHED_CPU_OFFLOAD_MANIFEST_FILE", ""
        )
        if not manifest_file:
            return []
        records: list[dict[str, object]] = []
        seen_names: set[str] = set()
        try:
            with open(manifest_file, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("event") != "cpu_offload_parameter":
                        continue
                    name = record.get("full_parameter_name")
                    if (
                        isinstance(name, str)
                        and name
                        and name not in seen_names
                    ):
                        records.append(record)
                        seen_names.add(name)
        except OSError as exc:
            logger.warning(
                "Failed to read matched CPU offload manifest %s: %s",
                manifest_file,
                exc,
            )
        return records

    def _uvm_matched_named_tensor_index(
        self,
    ) -> dict[str, tuple[torch.Tensor, str]]:
        model = self.get_model()
        index: dict[str, tuple[torch.Tensor, str]] = {}
        for name, tensor in model.named_parameters(remove_duplicate=False):
            index.setdefault(name, (tensor, "parameter"))
        for name, tensor in model.named_buffers(remove_duplicate=False):
            index.setdefault(name, (tensor, "buffer"))
        return index


    def _run_uvm_matched_cpu_offload_manifest(self) -> int:
        if os.environ.get("VLLM_UVM_MATCHED_CPU_OFFLOAD_ENABLE", "0") != "1":
            return 0
        if getattr(self, "_uvm_matched_cpu_offload_done", False):
            return 0
        self._uvm_matched_cpu_offload_done = True

        manifest_records = self._load_uvm_matched_cpu_offload_manifest()
        mode = os.environ.get(
            "VLLM_UVM_MATCHED_CPU_OFFLOAD_MODE", "offload_cpu"
        )
        if mode not in {
            "trace_only",
            "advise_cpu",
            "prefetch_cpu",
            "offload_cpu",
            "remote_read",
            "remote_read_bg",
            "remote_read_accessed_by",
            "native_remote_access",
            "prefetch_gpu",
        }:
            logger.warning(
                "Unknown VLLM_UVM_MATCHED_CPU_OFFLOAD_MODE=%s; using trace_only",
                mode,
            )
            mode = "trace_only"
        executor = os.environ.get("VLLM_UVM_MATCHED_CPU_OFFLOAD_EXECUTOR", "auto")
        if executor not in {"cpp_batch", "driver_batch", "auto"}:
            logger.warning(
                "Unknown VLLM_UVM_MATCHED_CPU_OFFLOAD_EXECUTOR=%s; using auto",
                executor,
            )
            executor = "auto"
        sync = os.environ.get("VLLM_UVM_MATCHED_CPU_OFFLOAD_SYNC", "1") == "1"
        max_bytes = int(os.environ.get(
            "VLLM_UVM_MATCHED_CPU_OFFLOAD_MAX_BYTES", "0"
        ) or "0")
        if not manifest_records:
            return 0

        ranges: list[tuple[int, int]] = []
        items: list[dict[str, object]] = []
        attempted_bytes = 0
        tensor_index = self._uvm_matched_named_tensor_index()
        matched_names: set[str] = set()
        for order, record in enumerate(manifest_records, start=1):
            name = record.get("full_parameter_name")
            if not isinstance(name, str) or not name:
                continue
            tensor_entry = tensor_index.get(name)
            if tensor_entry is None:
                continue
            tensor, tensor_kind = tensor_entry
            bytes_ = int(tensor.numel() * tensor.element_size())
            if max_bytes > 0 and attempted_bytes + bytes_ > max_bytes:
                continue
            ptr = int(tensor.data_ptr())
            driver_capable, driver_reason, driver_probe_result = (
                probe_driver_capable_range(ptr, bytes_)
            )
            ranges.append((ptr, bytes_))
            items.append(
                {
                    "event": "matched_cpu_offload_parameter",
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "mode": mode,
                    "parameter_name": name,
                    "parameter_order": order,
                    "tensor_kind": tensor_kind,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "ptr": f"0x{ptr:x}",
                    "bytes": bytes_,
                    "manifest_bytes": record.get("bytes"),
                    "manifest_shape": record.get("shape"),
                    "manifest_dtype": record.get("dtype"),
                    "driver_capable": driver_capable,
                    "driver_capability_reason": driver_reason,
                    "driver_probe_result": driver_probe_result,
                }
            )
            attempted_bytes += bytes_
            matched_names.add(name)

        if not ranges:
            return 0

        if mode == "trace_only":
            successes = len(ranges)
            executor_used = "trace_only"
        elif mode == "advise_cpu":
            successes, executor_used, _ = (
                batch_advise_ranges_cpu_with_executor(
                    ranges,
                    executor=executor,
                )
            )
        elif mode in {"prefetch_cpu", "offload_cpu"}:
            successes, executor_used, _, _ = (
                batch_offload_ranges_to_cpu_with_executor(
                    ranges,
                    sync=sync,
                    executor=executor,
                )
            )
        elif mode == "native_remote_access":
            (
                successes,
                executor_used,
                _,
                _,
                _,
            ) = batch_prepare_ranges_for_native_remote_access_with_executor(
                ranges,
                sync=sync,
                executor=executor,
                device=torch.cuda.current_device(),
            )
        elif mode in {"remote_read", "remote_read_bg", "remote_read_accessed_by"}:
            (
                successes,
                executor_used,
                _,
                _,
                _,
                _,
            ) = batch_prepare_ranges_for_remote_read_with_executor(
                ranges,
                sync=sync,
                executor=executor,
                device=torch.cuda.current_device(),
                accessed_by=mode == "remote_read_accessed_by",
            )
        else:
            successes = batch_prefetch_ranges_to_device(ranges, device=0)
            executor_used = "cpp_batch"

        if sync and successes > 0:
            try:
                torch.cuda.synchronize()
            except Exception:
                logger.debug(
                    "Failed to synchronize after matched UVM CPU offload",
                    exc_info=True,
                )
        logger.info(
            "Matched UVM CPU offload mode=%s manifest_names=%d matched=%d "
            "missing=%d successes=%d attempted_bytes=%d executor=%s",
            mode,
            len(manifest_records),
            len(items),
            len(manifest_records) - len(matched_names),
            successes,
            attempted_bytes,
            executor_used,
        )
        return successes

    def shutdown_cpu_offload_graph_static_alias(self) -> dict[str, Any]:
        """Reset captured Graphs before releasing static-alias backing.

        Neither cmd99 nor CUDA VMM may unmap an alias while a Graph executable
        still owns its VA. Keep this ordering in the model runner, where the
        concrete Graph entries are owned, instead of relying on interpreter-
        finalization order.
        """
        from vllm.model_executor.models.utils import (
            destroy_cpu_offload_graph_static_alias,
            get_cpu_offload_graph_static_alias_diagnostics,
        )

        diagnostics = get_cpu_offload_graph_static_alias_diagnostics()
        if not diagnostics.get("enabled"):
            return destroy_cpu_offload_graph_static_alias(
                graphs_destroyed=True,
                synchronize=False,
            )

        torch.cuda.synchronize(self.device)
        destroyed_graph_count = 0
        model = getattr(self, "model", None)
        if isinstance(model, CUDAGraphWrapper):
            reset_all = getattr(model, "reset_all_cudagraph_classes", None)
            if reset_all is not None:
                destroyed_graph_count = int(reset_all())
            else:
                entries = model.concrete_cudagraph_entries
                for entry in tuple(entries.values()):
                    if entry.cudagraph is not None:
                        entry.cudagraph.reset()
                        destroyed_graph_count += 1
                    entry.cudagraph = None
                    entry.output = None
                    entry.input_addresses = None
                entries.clear()
        gc.collect()
        torch.cuda.synchronize(self.device)

        result = destroy_cpu_offload_graph_static_alias(
            graphs_destroyed=True,
            synchronize=False,
        )
        result["destroyed_graph_count"] = destroyed_graph_count
        if not bool(result.get("ok")):
            raise RuntimeError(
                "Failed to tear down static aliases after "
                f"destroying {destroyed_graph_count} CUDA Graphs: {result!r}"
            )
        return result

    def rebuild_cpu_offload_uvm_expert_plan_at_idle_boundary(
        self,
        expert_plan_file: str,
        *,
        reason: str,
        scheduler_idle: bool,
        expected_selected_bytes: int | None = None,
        pressure: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Rebuild upload/remote plans between two completed requests.

        The scheduler owns the idle proof; this method owns concrete CUDA
        Graph teardown and recapture.  Keeping those responsibilities explicit
        prevents a plan change from racing an active request or leaving a
        Graph executable bound to an old upload arena.
        """
        from vllm.model_executor.models.utils import (
            get_cpu_offload_graph_static_alias_diagnostics,
            reallocate_cpu_offload_uvm_original_va_expert_parameters,
            rebuild_cpu_offload_uvm_original_va_expert_plan,
        )

        if scheduler_idle is not True:
            raise RuntimeError(
                "Dynamic expert-plan rebuild requires an audited idle "
                "scheduler boundary"
            )
        model = getattr(self, "model", None)
        if not isinstance(model, CUDAGraphWrapper):
            raise RuntimeError(
                "Dynamic expert-plan rebuild requires a CUDAGraphWrapper"
            )

        started_ns = time.perf_counter_ns()
        torch.cuda.synchronize(self.device)
        graph_reset_started_ns = time.perf_counter_ns()
        entries = model.concrete_cudagraph_entries
        destroyed_graph_count = 0
        destroyed_descriptors: list[dict[str, Any]] = []
        for descriptor, entry in tuple(entries.items()):
            graph = entry.cudagraph
            if graph is not None:
                graph.reset()
                destroyed_graph_count += 1
            destroyed_descriptors.append(
                {
                    "num_tokens": int(descriptor.num_tokens),
                    "uniform_decode": bool(descriptor.uniform_decode),
                    "has_lora": bool(descriptor.has_lora),
                }
            )
            entry.cudagraph = None
            entry.output = None
            entry.input_addresses = None
        entries.clear()
        gc.collect()
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

        # torch.cuda.CUDAGraph.reset() drops the last live user of the shared
        # graph pool. Reusing that now-retired pool id for another capture can
        # trip CUDACachingAllocator's ``use_count > 0`` invariant. A dynamic
        # plan also owns different upload arenas, so start its captures in a
        # fresh pool after every old graph and output reference is gone.
        model.graph_pool = torch.cuda.graph_pool_handle()
        graphs_destroyed_ns = time.perf_counter_ns()
        if destroyed_graph_count <= 0:
            raise RuntimeError(
                "Dynamic expert-plan rebuild found no captured CUDA Graphs"
            )

        nonstable_va_baseline = os.environ.get(
            "VLLM_UVM_E5_NONSTABLE_VA_RECAPTURE", "0"
        ).strip() == "1"
        rebuild_fn = (
            reallocate_cpu_offload_uvm_original_va_expert_parameters
            if nonstable_va_baseline
            else rebuild_cpu_offload_uvm_original_va_expert_plan
        )
        rebuild = rebuild_fn(
            expert_plan_file,
            reason=reason,
            graphs_destroyed=True,
            expected_selected_bytes=expected_selected_bytes,
            pressure=pressure,
        )
        plan_rebuilt_ns = time.perf_counter_ns()
        graph_memory_bytes = self.capture_model()
        graphs_captured_ns = time.perf_counter_ns()

        diagnostics = get_cpu_offload_graph_static_alias_diagnostics()
        if not (
            diagnostics.get("lifecycle_state") == "sealed"
            and diagnostics.get("remote_graph_captured") is True
            and diagnostics.get("alias_graph_captured") is True
            and isinstance(
                diagnostics.get("dual_graph_partition_certificate"), dict
            )
            and int(diagnostics.get("plan_generation", -1))
            == int(rebuild.get("generation", -2))
        ):
            raise RuntimeError(
                "Dynamic expert-plan recapture did not restore the certified "
                f"dual topology: {diagnostics!r}"
            )
        self._uvm_original_va_phase_decode_cutover_done = False
        self._uvm_original_va_phase_prefill_seen = False
        result = {
            "schema": "vllm_uvm_request_boundary_plan_rebuild_v1",
            "ok": True,
            "reason": reason,
            "scheduler_idle": True,
            "pressure": dict(pressure or {}),
            "destroyed_graph_count": destroyed_graph_count,
            "destroyed_graph_descriptors": destroyed_descriptors,
            "graph_pool_renewed": True,
            "parameter_va_mode": (
                "nonstable_reallocate_recapture"
                if nonstable_va_baseline
                else "stable_rebuild_recapture"
            ),
            "graph_memory_bytes": int(graph_memory_bytes),
            "graph_reset_ns": graphs_destroyed_ns - graph_reset_started_ns,
            "plan_rebuild_ns": plan_rebuilt_ns - graphs_destroyed_ns,
            "graph_capture_ns": graphs_captured_ns - plan_rebuilt_ns,
            "wall_ns": graphs_captured_ns - started_ns,
            "rebuild": rebuild,
            "diagnostics_after": diagnostics,
        }
        logger.info(
            "Rebuilt UVM expert plan generation=%d selected_bytes=%d "
            "graphs=%d wall_ms=%.3f",
            int(rebuild.get("generation", -1)),
            int(rebuild.get("selected_bytes", 0)),
            destroyed_graph_count,
            float(result["wall_ns"]) / 1e6,
        )
        return result

    def _reconcile_uvm_plan_residency_before_graph_class_switch(
        self,
    ) -> dict[str, Any]:
        """Restore the active Plan Class residency after Graph capture.

        A pure-Prefill capture may leave some logically-hot expert slices on
        the CPU.  A subsequent Plan Class delta assumes the active class is
        physically exact; without this repair, an already-remote slice is
        incorrectly counted as a failed GPU-to-CPU release.  Reconcile at the
        certified startup boundary before switching to the next class.
        """
        from vllm.model_executor.models.utils import (
            get_cpu_offload_graph_static_alias_diagnostics,
            reconcile_cpu_offload_uvm_original_va_remote_graph_residency,
            reconcile_cpu_offload_uvm_original_va_sealed_hot_residency,
        )

        diagnostics = get_cpu_offload_graph_static_alias_diagnostics()
        if not bool(diagnostics.get("enabled")):
            return {"ok": True, "reason": "static_alias_disabled"}
        state = str(diagnostics.get("lifecycle_state") or "")
        if state == "sealed":
            return reconcile_cpu_offload_uvm_original_va_sealed_hot_residency(
                preserve_allocator_cache=True
            )
        if state in {"remote", "remote_parked"}:
            return reconcile_cpu_offload_uvm_original_va_remote_graph_residency(
                preserve_allocator_cache=True
            )
        raise RuntimeError(
            "Graph Class switch residency reconciliation encountered an "
            f"unsupported lifecycle state: {state!r}"
        )

    def capture_uvm_expert_graph_class_at_idle_boundary(
        self,
        class_name: str,
        *,
        scheduler_idle: bool,
    ) -> dict[str, Any]:
        """Capture one additional class while retaining existing GraphExecs."""
        if scheduler_idle is not True:
            return {"ok": False, "reason": "scheduler_not_idle"}
        model = getattr(self, "model", None)
        if not isinstance(model, CUDAGraphWrapper):
            return {"ok": False, "reason": "cudagraph_wrapper_unavailable"}
        from vllm.model_executor.models.utils import (
            begin_cpu_offload_uvm_plan_class_graph_capture,
            cpu_offload_uvm_plan_class_registry,
            register_cpu_offload_uvm_plan_class_graph_certificates,
            switch_cpu_offload_uvm_original_va_plan_class,
        )

        torch.cuda.synchronize(self.device)
        before = model.cudagraph_class_diagnostics()
        normalized = model._normalize_cudagraph_class(class_name)
        existing = before["classes"].get(normalized)
        if existing and bool(existing.get("fully_captured")):
            return {
                "ok": True,
                "reason": "class_already_captured",
                "class_name": normalized,
                "graph_memory_bytes": 0,
                "capture_ns": 0,
                "before": before,
                "after": before,
            }

        registry = cpu_offload_uvm_plan_class_registry()
        plan_switch = None
        residency_reconcile = None
        if bool(registry.get("enabled")):
            if normalized not in registry["classes"]:
                return {"ok": False, "reason": "plan_class_not_registered"}
            if registry.get("active_class") != normalized:
                residency_reconcile = (
                    self._reconcile_uvm_plan_residency_before_graph_class_switch()
                )
                if residency_reconcile.get("ok") is not True:
                    return {
                        "ok": False,
                        "reason": "active_class_residency_reconcile_failed",
                        "residency_reconcile": residency_reconcile,
                    }
                plan_switch = switch_cpu_offload_uvm_original_va_plan_class(
                    normalized,
                    reason="graph_class_capture",
                    graphs_retained=True,
                )
                if plan_switch.get("ok") is not True:
                    return {
                        "ok": False,
                        "reason": "plan_class_switch_failed",
                        "plan_switch": plan_switch,
                        "residency_reconcile": residency_reconcile,
                    }

        # Equal-width plan classes differ only in which expert occupies each
        # stable slot. Their tensor shapes, Graph-visible VAs, and dispatch
        # descriptors are identical, so retaining another GraphExec table
        # would waste HBM without changing the captured program.
        target_width = registry.get("classes", {}).get(normalized, {}).get(
            "experts_per_layer"
        )
        equal_shape_source = next(
            (
                source_class
                for source_class, source_state in before["classes"].items()
                if source_class != normalized
                and bool(source_state.get("fully_captured"))
                and registry.get("classes", {}).get(source_class, {}).get(
                    "experts_per_layer"
                )
                == target_width
            ),
            None,
        )
        if equal_shape_source is not None:
            aliased = model.register_cudagraph_class_alias(
                normalized, equal_shape_source
            )
            if aliased.get("ok") is not True:
                return {
                    **aliased,
                    "class_name": normalized,
                    "graph_memory_bytes": 0,
                    "capture_ns": 0,
                    "certificate": None,
                    "plan_switch": plan_switch,
                    "residency_reconcile": residency_reconcile,
                    "before": before,
                    "after": model.cudagraph_class_diagnostics(),
                }
            certificate = (
                register_cpu_offload_uvm_plan_class_graph_certificates(
                    normalized
                )
            )
            return {
                **aliased,
                "class_name": normalized,
                "graph_memory_bytes": 0,
                "capture_ns": 0,
                "certificate": certificate,
                "plan_switch": plan_switch,
                "residency_reconcile": residency_reconcile,
                "before": before,
                "after": model.cudagraph_class_diagnostics(),
            }

        # The normal startup capture initially lives under "default". Adopt
        # it as the configured initial class without duplicating GraphExecs.
        active_class = before.get("active_class")
        active_state = before["classes"].get(active_class, {})
        if (
            bool(registry.get("enabled"))
            and registry.get("active_class") == normalized
            and bool(active_state.get("fully_captured"))
            and normalized not in before["classes"]
        ):
            adopted = model.register_active_cudagraph_class(normalized)
            if adopted.get("ok") is not True:
                return {
                    **adopted,
                    "class_name": normalized,
                    "graph_memory_bytes": 0,
                    "capture_ns": 0,
                    "certificate": None,
                    "plan_switch": plan_switch,
                    "before": before,
                    "after": model.cudagraph_class_diagnostics(),
                }
            certificate = (
                register_cpu_offload_uvm_plan_class_graph_certificates(
                    normalized
                )
            )
            after = model.cudagraph_class_diagnostics()
            return {
                **adopted,
                "class_name": normalized,
                "graph_memory_bytes": 0,
                "capture_ns": 0,
                "certificate": certificate,
                "plan_switch": plan_switch,
                "before": before,
                "after": after,
            }

        model.begin_cudagraph_class_capture(normalized)
        if bool(registry.get("enabled")):
            begin_cpu_offload_uvm_plan_class_graph_capture(normalized)
        started_ns = time.perf_counter_ns()
        graph_memory_bytes = self.capture_model()
        torch.cuda.synchronize(self.device)
        elapsed_ns = time.perf_counter_ns() - started_ns
        after = model.cudagraph_class_diagnostics()
        class_state = after["classes"].get(normalized, {})
        ok = bool(class_state.get("fully_captured"))
        certificate = None
        if ok and bool(registry.get("enabled")):
            certificate = register_cpu_offload_uvm_plan_class_graph_certificates(
                normalized
            )
        return {
            "ok": ok,
            "reason": "class_captured" if ok else "class_capture_incomplete",
            "class_name": normalized,
            "graph_memory_bytes": int(graph_memory_bytes),
            "capture_ns": elapsed_ns,
            "certificate": certificate,
            "plan_switch": plan_switch,
            "residency_reconcile": residency_reconcile,
            "before": before,
            "after": after,
        }

    def activate_uvm_expert_graph_class_at_idle_boundary(
        self,
        class_name: str,
        *,
        scheduler_idle: bool,
        resume_prefill_before_switch: bool = False,
        defer_dual_refresh: bool = False,
    ) -> dict[str, Any]:
        """Switch tables at an EngineCore-certified execution boundary.

        ``scheduler_idle`` is the historical RPC field name.  A true value may
        now represent a scheduler-quiescent point between token steps even
        while requests remain logically running.
        """
        if scheduler_idle is not True:
            return {"ok": False, "reason": "scheduler_not_idle"}
        # Future completion is the EngineCore proof; this synchronization also
        # closes work submitted on auxiliary streams before residency changes.
        torch.cuda.synchronize(self.device)
        # A Plan Class switch may run between Decode token steps.  Append-only
        # class growth cannot safely defer the dual-view refresh while the
        # alias session is parked: the new remote child is not complete yet.
        # Temporarily resume the sealed Prefill topology, finish both children,
        # then park again before the next Decode step.
        preserve_remote_decode_topology = bool(
            getattr(self, "_uvm_original_va_phase_decode_cutover_done", False)
        )
        effective_resume_prefill = bool(resume_prefill_before_switch)
        effective_defer_dual_refresh = bool(
            defer_dual_refresh and not preserve_remote_decode_topology
        )
        eager_plan_classes = os.environ.get(
            "VLLM_CPU_OFFLOAD_PLAN_CLASS_EAGER_ENABLE", "0"
        ).strip().lower() in ("1", "true", "yes", "on")
        if eager_plan_classes:
            from vllm.model_executor.models.utils import (
                activate_cpu_offload_uvm_original_va_prefill_topology,
                cpu_offload_uvm_plan_class_registry,
                get_cpu_offload_graph_static_alias_diagnostics,
                switch_cpu_offload_uvm_original_va_plan_class,
                transition_cpu_offload_uvm_original_va_to_remote,
            )

            mode = getattr(self.compilation_config.cudagraph_mode, "name", "")
            if mode != "NONE" or not self.model_config.enforce_eager:
                return {
                    "ok": False,
                    "reason": "eager_plan_class_requires_eager_none_mode",
                    "cudagraph_mode": mode,
                }
            registry = cpu_offload_uvm_plan_class_registry()
            normalized = str(class_name).strip().lower()
            if not bool(registry.get("enabled")):
                return {"ok": False, "reason": "plan_class_registry_disabled"}
            if normalized not in registry.get("classes", {}):
                return {
                    "ok": False,
                    "reason": "plan_class_not_registered",
                    "class_name": normalized,
                }
            started_ns = time.perf_counter_ns()
            prefill_topology_resume = None
            if effective_resume_prefill:
                prefill_topology_resume = (
                    activate_cpu_offload_uvm_original_va_prefill_topology()
                )
                if prefill_topology_resume.get("ok") is not True:
                    return {
                        "ok": False,
                        "reason": "prefill_topology_resume_failed",
                        "class_name": normalized,
                        "prefill_topology_resume_before_plan_switch": (
                            prefill_topology_resume
                        ),
                    }
            plan_switch = switch_cpu_offload_uvm_original_va_plan_class(
                normalized,
                reason="eager_plan_class_activation",
                graphs_retained=True,
                defer_dual_refresh=effective_defer_dual_refresh,
            )
            remote_decode_repark = None
            if (
                plan_switch.get("ok") is True
                and preserve_remote_decode_topology
            ):
                diagnostics = get_cpu_offload_graph_static_alias_diagnostics()
                transition = transition_cpu_offload_uvm_original_va_to_remote(
                    reason="online_plan_class_decode_repark",
                    preserve_remote_graphs=bool(
                        diagnostics.get("remote_graph_captured")
                    ),
                    runtime_remote_ranges=(
                        self._collect_uvm_runtime_remote_expert_ranges()
                    ),
                )
                after = get_cpu_offload_graph_static_alias_diagnostics()
                expected_state = (
                    "remote_parked"
                    if bool(after.get("park_resume_enabled"))
                    else "remote"
                )
                remote_decode_repark = {
                    "ok": (
                        transition.get("ok") is True
                        and after.get("lifecycle_state") == expected_state
                        and after.get("remote_map_active") is True
                    ),
                    "transition": transition,
                    "diagnostics_after": after,
                }
                if remote_decode_repark["ok"] is not True:
                    raise RuntimeError(
                        "Online Plan Class switch failed to restore the "
                        "remote Decode topology"
                    )
            torch.cuda.synchronize(self.device)
            return {
                "ok": plan_switch.get("ok") is True,
                "reason": (
                    "eager_plan_class_activated"
                    if plan_switch.get("ok") is True
                    else "eager_plan_class_switch_failed"
                ),
                "class_name": normalized,
                "execution_mode": "eager",
                "plan_switch": plan_switch,
                "prefill_topology_resume_before_plan_switch": (
                    prefill_topology_resume
                ),
                "preserved_remote_decode_topology": (
                    preserve_remote_decode_topology
                ),
                "dual_view_refresh_deferred": effective_defer_dual_refresh,
                "remote_decode_repark": remote_decode_repark,
                "switch_ns": time.perf_counter_ns() - started_ns,
            }
        model = getattr(self, "model", None)
        if not isinstance(model, CUDAGraphWrapper):
            return {"ok": False, "reason": "cudagraph_wrapper_unavailable"}
        from vllm.model_executor.models.utils import (
            activate_cpu_offload_uvm_original_va_prefill_topology,
            activate_cpu_offload_uvm_plan_class_graph_certificates,
            cpu_offload_uvm_plan_class_registry,
            switch_cpu_offload_uvm_original_va_plan_class,
        )

        torch.cuda.synchronize(self.device)
        started_ns = time.perf_counter_ns()
        normalized = model._normalize_cudagraph_class(class_name)
        before = model.cudagraph_class_diagnostics()
        class_state = before["classes"].get(normalized)
        if not class_state or not bool(class_state.get("fully_captured")):
            return {
                "ok": False,
                "reason": "class_not_fully_captured",
                "class_name": normalized,
                "diagnostics": before,
            }
        registry = cpu_offload_uvm_plan_class_registry()
        plan_switch = None
        registry_enabled = bool(registry.get("enabled"))
        previous_class = str(registry.get("active_class"))
        previous_graph_class = str(before.get("active_class"))
        prefill_topology_resume = None
        if registry_enabled and previous_graph_class != previous_class:
            return {
                "ok": False,
                "reason": "active_graph_plan_class_mismatch",
                "class_name": normalized,
                "active_graph_class": previous_graph_class,
                "active_plan_class": previous_class,
                "diagnostics": before,
            }
        if registry_enabled and previous_class != normalized:
            if effective_resume_prefill:
                prefill_topology_resume = (
                    activate_cpu_offload_uvm_original_va_prefill_topology()
                )
                if prefill_topology_resume.get("ok") is not True:
                    return {
                        "ok": False,
                        "reason": "prefill_topology_resume_failed",
                        "class_name": normalized,
                        "prefill_topology_resume_before_plan_switch": (
                            prefill_topology_resume
                        ),
                        "diagnostics": before,
                    }
            plan_switch = switch_cpu_offload_uvm_original_va_plan_class(
                normalized,
                reason="graph_class_activation",
                graphs_retained=True,
                defer_dual_refresh=defer_dual_refresh,
            )
            if plan_switch.get("ok") is not True:
                return {
                    "ok": False,
                    "reason": "plan_class_switch_failed",
                    "class_name": normalized,
                    "plan_switch": plan_switch,
                    "prefill_topology_resume_before_plan_switch": (
                        prefill_topology_resume
                    ),
                    "diagnostics": model.cudagraph_class_diagnostics(),
                }
        certificate = None
        if registry_enabled:
            certificate = activate_cpu_offload_uvm_plan_class_graph_certificates(
                normalized
            )
            if certificate.get("ok") is not True:
                rollback_plan = None
                rollback_certificate = None
                if previous_class != normalized:
                    rollback_plan = (
                        switch_cpu_offload_uvm_original_va_plan_class(
                            previous_class,
                            reason="certificate_activation_rollback",
                            graphs_retained=True,
                        )
                    )
                    if rollback_plan.get("ok") is True:
                        rollback_certificate = (
                            activate_cpu_offload_uvm_plan_class_graph_certificates(
                                previous_class
                            )
                        )
                    if not (
                        rollback_plan.get("ok") is True
                        and rollback_certificate is not None
                        and rollback_certificate.get("ok") is True
                    ):
                        raise RuntimeError(
                            "Plan class certificate activation failed and the "
                            "previous class could not be restored: "
                            f"plan={rollback_plan!r}, "
                            f"certificate={rollback_certificate!r}"
                        )
                return {
                    "ok": False,
                    "reason": "class_certificate_activation_failed",
                    "class_name": normalized,
                    "plan_switch": plan_switch,
                    "certificate": certificate,
                    "prefill_topology_resume_before_plan_switch": (
                        prefill_topology_resume
                    ),
                    "rollback_plan": rollback_plan,
                    "rollback_certificate": rollback_certificate,
                    "diagnostics": model.cudagraph_class_diagnostics(),
                }
        result = model.activate_cudagraph_class(normalized)
        if result.get("ok") is not True and registry_enabled:
            rollback_plan = None
            rollback_certificate = None
            if previous_class != normalized:
                rollback_plan = switch_cpu_offload_uvm_original_va_plan_class(
                    previous_class,
                    reason="graph_table_activation_rollback",
                    graphs_retained=True,
                )
                if rollback_plan.get("ok") is True:
                    rollback_certificate = (
                        activate_cpu_offload_uvm_plan_class_graph_certificates(
                            previous_class
                        )
                    )
                if not (
                    rollback_plan.get("ok") is True
                    and rollback_certificate is not None
                    and rollback_certificate.get("ok") is True
                ):
                    raise RuntimeError(
                        "CUDA Graph class activation failed and the previous "
                        "plan class could not be restored: "
                        f"plan={rollback_plan!r}, "
                        f"certificate={rollback_certificate!r}"
                    )
            return {
                **result,
                "plan_switch": plan_switch,
                "certificate": certificate,
                "prefill_topology_resume_before_plan_switch": (
                    prefill_topology_resume
                ),
                "rollback_plan": rollback_plan,
                "rollback_certificate": rollback_certificate,
                "switch_ns": time.perf_counter_ns() - started_ns,
                "diagnostics": model.cudagraph_class_diagnostics(),
            }
        torch.cuda.synchronize(self.device)
        return {
            **result,
            "plan_switch": plan_switch,
            "certificate": certificate,
            "prefill_topology_resume_before_plan_switch": (
                prefill_topology_resume
            ),
            "switch_ns": time.perf_counter_ns() - started_ns,
            "diagnostics": model.cudagraph_class_diagnostics(),
        }

    def capture_configured_uvm_expert_graph_classes_at_idle_boundary(
        self,
        *,
        scheduler_idle: bool,
    ) -> dict[str, Any]:
        """Capture every configured fixed class once, then restore the initial class."""
        if scheduler_idle is not True:
            return {"ok": False, "reason": "scheduler_not_idle"}
        from vllm.model_executor.models.utils import (
            cpu_offload_uvm_plan_class_registry,
        )

        registry = cpu_offload_uvm_plan_class_registry()
        if not bool(registry.get("enabled")):
            return {
                "ok": True,
                "reason": "plan_class_registry_disabled",
                "graph_memory_bytes": 0,
                "class_results": [],
            }
        class_records = registry["classes"]
        ordered_classes = sorted(
            class_records,
            key=lambda name: (
                class_records[name].get("experts_per_layer", 0),
                name,
            ),
        )
        active = str(registry["active_class"])
        class_results: list[dict[str, Any]] = []
        # Adopt the initial startup capture before selecting empty tables for
        # the other classes.
        class_results.append(
            self.capture_uvm_expert_graph_class_at_idle_boundary(
                active, scheduler_idle=True
            )
        )
        if class_results[-1].get("ok") is not True:
            return {
                "ok": False,
                "reason": "initial_class_registration_failed",
                "class_results": class_results,
            }
        for class_name in ordered_classes:
            if class_name == active:
                continue
            result = self.capture_uvm_expert_graph_class_at_idle_boundary(
                class_name, scheduler_idle=True
            )
            class_results.append(result)
            if result.get("ok") is not True:
                return {
                    "ok": False,
                    "reason": "class_capture_failed",
                    "failed_class": class_name,
                    "class_results": class_results,
                }
        initial = str(registry["initial_class"])
        restore_residency_reconcile = None
        if str(registry.get("active_class")) != initial:
            restore_residency_reconcile = (
                self._reconcile_uvm_plan_residency_before_graph_class_switch()
            )
            if restore_residency_reconcile.get("ok") is not True:
                return {
                    "ok": False,
                    "reason": "restore_residency_reconcile_failed",
                    "initial_class": initial,
                    "class_results": class_results,
                    "restore_residency_reconcile": restore_residency_reconcile,
                    "diagnostics": self.model.cudagraph_class_diagnostics(),
                }
        restore = self.activate_uvm_expert_graph_class_at_idle_boundary(
            initial, scheduler_idle=True
        )
        graph_memory_bytes = sum(
            int(result.get("graph_memory_bytes", 0)) for result in class_results
        )
        return {
            "ok": restore.get("ok") is True,
            "reason": (
                "all_classes_captured"
                if restore.get("ok") is True
                else "initial_class_restore_failed"
            ),
            "initial_class": initial,
            "class_results": class_results,
            "restore_residency_reconcile": restore_residency_reconcile,
            "restore": restore,
            "graph_memory_bytes": graph_memory_bytes,
            "diagnostics": self.model.cudagraph_class_diagnostics(),
        }

    def capture_model(
        self,
        before_pure_prefill_capture: Callable[[], None] | None = None,
    ) -> int:
        from vllm.model_executor.models.utils import (
            activate_cpu_offload_uvm_original_va_prefill_topology,
            certify_cpu_offload_uvm_original_va_dual_graph_partition,
            cpu_offload_graph_static_alias_capture_context,
            cpu_offload_uvm_original_va_phase_mode,
            prepare_cpu_offload_uvm_original_va_profile_topology,
            prepare_cpu_offload_uvm_original_va_remote_graph_capture,
            refresh_cpu_offload_uvm_original_va_full_dual_view,
            reconcile_cpu_offload_uvm_original_va_remote_graph_residency,
            reconcile_cpu_offload_uvm_original_va_sealed_hot_residency,
            validate_cpu_offload_graph_static_alias_runtime,
        )

        validate_cpu_offload_graph_static_alias_runtime(self.vllm_config)
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        original_va_phase_mode = cpu_offload_uvm_original_va_phase_mode()
        dual_topology_graphs = (
            original_va_phase_mode == "prefill_graph_remote_decode"
        )
        capture_prefill_before_hot_restore = (
            dual_topology_graphs
            and os.environ.get(
                "VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE", "0"
            ).strip()
            == "1"
        )

        # The ordinary path repairs hot placement before Graph capture.  The
        # opt-in large-Prefill path instead keeps every expert on its audited
        # remote READ mapping until the final Prefill Graph (and therefore its
        # private allocator pool) is permanently live.  Only then are hot
        # pages restored into the remaining HBM.
        if capture_prefill_before_hot_restore:
            prepare_cpu_offload_uvm_original_va_profile_topology(
                transition_reason="prefill_graph_capture_all_remote",
                preserve_remote_graphs=False,
            )
        else:
            # Phase-switch experiments capture decode kernels only after
            # original Parameter VAs have audited CPU-backed remote READ PTEs.
            # This keeps every upload memcpy node out of the Decode Graph.
            prepare_cpu_offload_uvm_original_va_remote_graph_capture()

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()

        @contextmanager
        def freeze_gc():
            # Optimize garbage collection during CUDA graph capture.
            # Clean up, then freeze all remaining objects from being included
            # in future collections.
            gc.collect()
            should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
            if should_freeze:
                gc.freeze()
            try:
                yield
            finally:
                if should_freeze:
                    gc.unfreeze()
                    gc.collect()

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        set_cudagraph_capturing_enabled(True)
        with freeze_gc(), graph_capture(device=self.device):
            start_free_gpu_memory = torch.cuda.mem_get_info()[0]
            cudagraph_mode = self.compilation_config.cudagraph_mode
            assert cudagraph_mode is not None

            if self.lora_config:
                if self.compilation_config.cudagraph_specialize_lora:
                    lora_cases = [True, False]
                else:
                    lora_cases = [True]
            else:
                lora_cases = [False]

            def capture_mixed_graphs() -> None:
                if cudagraph_mode.mixed_mode() == CUDAGraphMode.NONE:
                    return
                mixed_batch_sizes = self.cudagraph_batch_sizes
                if cudagraph_mode == CUDAGraphMode.FULL_SEPARATE:
                    # FULL_SEPARATE assigns small uniform shapes to the
                    # dedicated Decode graph set.  Capture mixed/Prefill only
                    # for the non-uniform dispatcher keys, including on the
                    # ordinary native path without dual-topology aliases.
                    # Otherwise a size such as 1 is captured twice and the
                    # mixed dummy run disagrees with the dispatcher.
                    mixed_batch_sizes = sorted(
                        {
                            descriptor.num_tokens
                            for descriptor in self.cudagraph_dispatcher.cudagraph_keys[
                                CUDAGraphMode.FULL
                            ]
                            if not descriptor.uniform_decode
                        }
                    )
                    if not mixed_batch_sizes:
                        raise RuntimeError(
                            "FULL_SEPARATE Graph capture has no dedicated "
                            "Prefill size above the uniform-Decode range"
                        )
                compilation_cases = list(
                    product(reversed(mixed_batch_sizes), lora_cases)
                )
                self._capture_cudagraphs(
                    compilation_cases,
                    cudagraph_runtime_mode=cudagraph_mode.mixed_mode(),
                    uniform_decode=False,
                )

            def capture_pure_prefill_graphs() -> None:
                """Capture sealed-topology Graphs with only Prefill requests."""
                if cudagraph_mode.mixed_mode() == CUDAGraphMode.NONE:
                    return
                prefill_batch_sizes = sorted(
                    {
                        descriptor.num_tokens
                        for descriptor in self.cudagraph_dispatcher.cudagraph_keys[
                            CUDAGraphMode.FULL
                        ]
                        if not descriptor.uniform_decode
                    }
                )
                if not prefill_batch_sizes:
                    raise RuntimeError(
                        "Dual-topology Graph capture has no dedicated pure-Prefill "
                        "size above the uniform-decode range"
                    )
                compilation_cases = list(
                    product(reversed(prefill_batch_sizes), lora_cases)
                )
                configured_num_reqs = os.environ.get(
                    "VLLM_CUDAGRAPH_PURE_PREFILL_NUM_REQS"
                )
                pure_prefill_num_reqs = self.scheduler_config.max_num_seqs
                if configured_num_reqs is not None:
                    pure_prefill_num_reqs = int(configured_num_reqs)
                    if not (
                        1
                        <= pure_prefill_num_reqs
                        <= self.scheduler_config.max_num_seqs
                    ):
                        raise ValueError(
                            "VLLM_CUDAGRAPH_PURE_PREFILL_NUM_REQS must be in "
                            f"[1, {self.scheduler_config.max_num_seqs}], got "
                            f"{pure_prefill_num_reqs}"
                        )
                    logger.info(
                        "Capturing pure-Prefill CUDA Graphs with %d requests "
                        "per configured shape",
                        pure_prefill_num_reqs,
                    )
                self._capture_cudagraphs(
                    compilation_cases,
                    cudagraph_runtime_mode=cudagraph_mode.mixed_mode(),
                    uniform_decode=False,
                    pure_prefill_num_reqs=pure_prefill_num_reqs,
                )

            def capture_uniform_decode_graphs() -> None:
                if not (
                    cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
                    and cudagraph_mode.separate_routine()
                ):
                    return
                max_num_tokens = (
                    self.scheduler_config.max_num_seqs * self.uniform_decode_query_len
                )
                decode_cudagraph_batch_sizes = [
                    x
                    for x in self.cudagraph_batch_sizes
                    if max_num_tokens >= x >= self.uniform_decode_query_len
                ]
                compilation_cases_decode = list(
                    product(reversed(decode_cudagraph_batch_sizes), lora_cases)
                )
                self._capture_cudagraphs(
                    compilation_cases=compilation_cases_decode,
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    uniform_decode=True,
                )

            if dual_topology_graphs:
                if cudagraph_mode != CUDAGraphMode.FULL_SEPARATE:
                    raise RuntimeError(
                        "prefill_graph_remote_decode requires "
                        "CUDAGraphMode.FULL_SEPARATE"
                    )

                # Remote graphs are captured first while all original
                # Parameter VAs have audited CPU-backed READ PTEs. Their
                # BatchDescriptor keys are uniform-decode=True.
                with cpu_offload_graph_static_alias_capture_context(
                    expected_topology="remote"
                ):
                    capture_uniform_decode_graphs()

                preseed_prefill_pool = (
                    os.environ.get(
                        "VLLM_UVM_PREFILL_GRAPH_POOL_PRESEED", "0"
                    ).strip()
                    == "1"
                )
                if capture_prefill_before_hot_restore and preseed_prefill_pool:
                    raise RuntimeError(
                        "Capture-before-hot-restore and disposable Prefill "
                        "pool preseed are mutually exclusive"
                    )
                if capture_prefill_before_hot_restore:
                    if before_pure_prefill_capture is not None:
                        torch.cuda.synchronize(self.device)
                        before_pure_prefill_capture()

                    # RESUME only the frozen cold alias banks.  Hot ranges
                    # intentionally remain on their audited CPU remote READ
                    # mappings while the *final*, retained Prefill Graph is
                    # captured.  Ending the profile flag enables the normal
                    # per-layer H2D copy path, so this is not an allocator-only
                    # or bypass graph.
                    torch.cuda.synchronize(self.device)
                    activate_cpu_offload_uvm_original_va_prefill_topology(
                        finish_all_remote_profile=True
                    )
                    with cpu_offload_graph_static_alias_capture_context(
                        expected_topology="sealed"
                    ):
                        capture_pure_prefill_graphs()

                    # The final GraphExecs now permanently own the exact
                    # allocator pool needed by serving.  Restore hot experts
                    # only into HBM left after those non-evictable allocations
                    # are established; never empty the allocator cache here.
                    torch.cuda.synchronize(self.device)
                    reconcile_cpu_offload_uvm_original_va_sealed_hot_residency(
                        preserve_allocator_cache=True
                    )
                    # The all-remote capture path rebuilt its retained 4K
                    # children while every hot expert was CPU-resident. Hot
                    # restoration changes the ordinary leaves again; refresh
                    # the sealed GPU child before serving or switching to the
                    # pre-captured remote Decode child.
                    refresh_cpu_offload_uvm_original_va_full_dual_view(
                        nested_refresh=False
                    )
                if preseed_prefill_pool:
                    if before_pure_prefill_capture is not None:
                        # Release a diagnostic sentinel before pool seeding,
                        # while every hot expert is still remote.  The pool,
                        # not the sentinel extent, owns the resulting native
                        # reserve from this point onward.
                        torch.cuda.synchronize(self.device)
                        before_pure_prefill_capture()

                    model = self.model
                    if not isinstance(model, CUDAGraphWrapper):
                        raise RuntimeError(
                            "Prefill Graph pool preseed requires a "
                            "CUDAGraphWrapper"
                        )
                    prefill_sizes = sorted(
                        {
                            descriptor.num_tokens
                            for descriptor in self.cudagraph_dispatcher.cudagraph_keys[
                                CUDAGraphMode.FULL
                            ]
                            if not descriptor.uniform_decode
                        }
                    )
                    if not prefill_sizes:
                        raise RuntimeError(
                            "Prefill Graph pool preseed found no pure-Prefill shape"
                        )
                    seed_tokens = max(prefill_sizes)
                    seed_class = "prefill_pool_seed"
                    model.begin_cudagraph_class_capture(seed_class, replace=True)
                    reserved_before_seed = int(
                        torch.cuda.memory_reserved(self.device)
                    )
                    free_before_seed = int(
                        torch.cuda.mem_get_info(self.device)[0]
                    )
                    bypass_name = (
                        "VLLM_UVM_PREFILL_ALLOCATOR_REUSE_DIAGNOSTIC_"
                        "BYPASS_MOE_KERNEL"
                    )
                    bypass_tokens_name = (
                        "VLLM_UVM_PREFILL_ALLOCATOR_REUSE_DIAGNOSTIC_TOKENS"
                    )
                    previous_bypass = os.environ.get(bypass_name)
                    previous_bypass_tokens = os.environ.get(bypass_tokens_name)
                    try:
                        os.environ[bypass_name] = "1"
                        os.environ[bypass_tokens_name] = str(seed_tokens)
                        with cpu_offload_graph_static_alias_capture_context(
                            expected_topology="remote"
                        ):
                            self._capture_cudagraphs(
                                [
                                    (seed_tokens, activate_lora)
                                    for activate_lora in lora_cases
                                ],
                                cudagraph_runtime_mode=cudagraph_mode.mixed_mode(),
                                uniform_decode=False,
                                pure_prefill_num_reqs=(
                                    self.scheduler_config.max_num_seqs
                                ),
                            )
                    finally:
                        if previous_bypass is None:
                            os.environ.pop(bypass_name, None)
                        else:
                            os.environ[bypass_name] = previous_bypass
                        if previous_bypass_tokens is None:
                            os.environ.pop(bypass_tokens_name, None)
                        else:
                            os.environ[bypass_tokens_name] = previous_bypass_tokens

                    torch.cuda.synchronize(self.device)
                    reserved_after_seed = int(
                        torch.cuda.memory_reserved(self.device)
                    )
                    free_after_seed = int(torch.cuda.mem_get_info(self.device)[0])
                    discard = model.discard_cudagraph_class(
                        seed_class,
                        activate_class="default",
                    )
                    if discard.get("ok") is not True:
                        raise RuntimeError(
                            f"Failed to discard Prefill pool seed Graph: {discard!r}"
                        )
                    gc.collect()
                    torch.cuda.synchronize(self.device)
                    reserved_after_discard = int(
                        torch.cuda.memory_reserved(self.device)
                    )
                    free_after_discard = int(
                        torch.cuda.mem_get_info(self.device)[0]
                    )
                    from vllm.model_executor.models.utils import (
                        _append_uvm_original_alias_lifecycle,
                    )

                    _append_uvm_original_alias_lifecycle(
                        "cpu_offload_uvm_prefill_graph_pool_preseeded",
                        {
                            "operation": "prefill_graph_pool_preseed",
                            "ok": True,
                            "seed_tokens": int(seed_tokens),
                            "seed_num_requests": int(
                                self.scheduler_config.max_num_seqs
                            ),
                            "reserved_before_seed": reserved_before_seed,
                            "reserved_after_seed": reserved_after_seed,
                            "reserved_delta": (
                                reserved_after_seed - reserved_before_seed
                            ),
                            "reserved_after_discard": reserved_after_discard,
                            "free_before_seed": free_before_seed,
                            "free_after_seed": free_after_seed,
                            "free_after_discard": free_after_discard,
                            "discard": discard,
                        },
                    )

                if not capture_prefill_before_hot_restore:
                    # Decode Graph capture owns persistent/native allocations.
                    # Repair any hot placement it disturbed while the remote
                    # topology is active, retaining allocator cache.
                    torch.cuda.synchronize(self.device)
                    reconcile_cpu_offload_uvm_original_va_remote_graph_residency(
                        preserve_allocator_cache=True
                    )
                    refresh_cpu_offload_uvm_original_va_full_dual_view()

                    # Restore the same original VAs to sealed dual-bank
                    # aliases, then capture independent Prefill entries.
                    torch.cuda.synchronize(self.device)
                    activate_cpu_offload_uvm_original_va_prefill_topology()
                    if (
                        before_pure_prefill_capture is not None
                        and not preseed_prefill_pool
                    ):
                        torch.cuda.synchronize(self.device)
                        before_pure_prefill_capture()
                    with cpu_offload_graph_static_alias_capture_context(
                        expected_topology="sealed"
                    ):
                        capture_pure_prefill_graphs()

                model = self.model
                if not isinstance(model, CUDAGraphWrapper):
                    raise RuntimeError(
                        "Dual-topology capture requires a CUDAGraphWrapper"
                    )

                def entry_records(uniform_decode: bool) -> list[dict[str, Any]]:
                    records: list[dict[str, Any]] = []
                    for descriptor, entry in sorted(
                        model.concrete_cudagraph_entries.items(),
                        key=lambda item: tuple(item[0]),
                    ):
                        if descriptor.uniform_decode != uniform_decode:
                            continue
                        if entry.cudagraph is None:
                            raise RuntimeError(
                                "A captured dual-topology Graph entry is empty"
                            )
                        records.append(
                            {
                                "num_tokens": int(descriptor.num_tokens),
                                "uniform_decode": bool(
                                    descriptor.uniform_decode
                                ),
                                "has_lora": bool(descriptor.has_lora),
                            }
                        )
                    return records

                certify_cpu_offload_uvm_original_va_dual_graph_partition(
                    remote_entries=entry_records(True),
                    alias_entries=entry_records(False),
                )
            else:
                # Legacy paths retain their single capture topology and
                # ordering, except that FULL_SEPARATE must use a true
                # pure-Prefill dummy batch.  Its whole purpose is to keep the
                # Prefill and uniform-Decode graph classes distinct; using a
                # mixed dummy batch here produces incompatible Triton MLA
                # metadata when the real serving step contains only Prefill.
                with cpu_offload_graph_static_alias_capture_context():
                    if cudagraph_mode == CUDAGraphMode.FULL_SEPARATE:
                        capture_pure_prefill_graphs()
                    else:
                        capture_mixed_graphs()
                    capture_uniform_decode_graphs()

            torch.cuda.synchronize()
            end_free_gpu_memory = torch.cuda.mem_get_info()[0]

        # Disable cudagraph capturing globally, so any unexpected cudagraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may do lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info_once(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
            scope="local",
        )
        return cuda_graph_size

    def _capture_cudagraphs(
        self,
        compilation_cases: list[tuple[int, bool]],
        cudagraph_runtime_mode: CUDAGraphMode,
        uniform_decode: bool,
        pure_prefill_num_reqs: int | None = None,
    ):
        assert (
            cudagraph_runtime_mode != CUDAGraphMode.NONE
            and cudagraph_runtime_mode.valid_runtime_modes()
        ), f"Invalid cudagraph runtime mode: {cudagraph_runtime_mode}"

        if pure_prefill_num_reqs is not None and uniform_decode:
            raise ValueError(
                "pure_prefill_num_reqs cannot be used for uniform Decode Graphs"
            )

        allocator_reuse_diagnostic = (
            os.environ.get(
                "VLLM_UVM_PREFILL_ALLOCATOR_REUSE_DIAGNOSTIC", "0"
            ).strip()
            == "1"
        )
        diagnostic_tokens = int(
            os.environ.get(
                "VLLM_UVM_PREFILL_ALLOCATOR_REUSE_DIAGNOSTIC_TOKENS", "16384"
            )
        )
        diagnostic_repeats = int(
            os.environ.get(
                "VLLM_UVM_PREFILL_ALLOCATOR_REUSE_DIAGNOSTIC_REPEATS", "2"
            )
        )
        if allocator_reuse_diagnostic and diagnostic_repeats < 2:
            raise ValueError(
                "Prefill allocator reuse diagnostic requires at least two repeats"
            )

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():
            compilation_cases = tqdm(
                compilation_cases,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing CUDA graphs ({}, {})".format(
                    (
                        "decode"
                        if uniform_decode
                        else (
                            "pure prefill"
                            if pure_prefill_num_reqs is not None
                            else "mixed prefill-decode"
                        )
                    ),
                    cudagraph_runtime_mode.name,
                ),
            )

        # We skip EPLB here since we don't want to record dummy metrics
        for num_tokens, activate_lora in compilation_cases:
            # We currently only capture ubatched graphs when its a FULL
            # cudagraph, a uniform decode batch, and the number of tokens
            # is above the threshold. Otherwise we just capture a non-ubatched
            # version of the graph
            allow_microbatching = (
                self.parallel_config.enable_dbo
                and cudagraph_runtime_mode == CUDAGraphMode.FULL
                and uniform_decode
                and check_ubatch_thresholds(
                    config=self.vllm_config.parallel_config,
                    num_tokens=num_tokens,
                    uniform_decode=uniform_decode,
                )
            )

            if (
                allocator_reuse_diagnostic
                and pure_prefill_num_reqs is not None
                and num_tokens == diagnostic_tokens
            ):
                from vllm.model_executor.models.utils import (
                    _append_uvm_original_alias_lifecycle,
                )

                device = int(self.device.index or 0)
                torch.cuda.synchronize(self.device)
                torch.cuda.memory._record_memory_history(
                    enabled="all",
                    context="alloc",
                    stacks="python",
                    max_entries=200000,
                    device=device,
                    clear_history=True,
                )

                def allocator_state() -> tuple[dict[str, int], dict[str, Any]]:
                    torch.cuda.synchronize(self.device)
                    free_bytes, total_bytes = (
                        int(value)
                        for value in torch.cuda.mem_get_info(self.device)
                    )
                    stats = torch.cuda.memory_stats(device)
                    state = {
                        "free_bytes": free_bytes,
                        "total_bytes": total_bytes,
                        "allocated_bytes": int(
                            torch.cuda.memory_allocated(device)
                        ),
                        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                        "active_bytes": int(
                            stats.get("active_bytes.all.current", 0)
                        ),
                        "inactive_split_bytes": int(
                            stats.get("inactive_split_bytes.all.current", 0)
                        ),
                        "segment_count": int(stats.get("segment.all.current", 0)),
                        "allocation_count": int(
                            stats.get("allocation.all.current", 0)
                        ),
                    }
                    return state, torch.cuda.memory._snapshot(device)

                before, snapshot = allocator_state()
                device_traces = snapshot.get("device_traces", [])
                trace_count = (
                    len(device_traces[device])
                    if device < len(device_traces)
                    else 0
                )
                repeat_records: list[dict[str, Any]] = []
                diagnostic_started_ns = time.perf_counter_ns()
                try:
                    for repeat_index in range(diagnostic_repeats):
                        run_started_ns = time.perf_counter_ns()
                        with uvm_allocation_phase(
                            "prefill_allocator_reuse_diagnostic:"
                            f"tokens={num_tokens}:repeat={repeat_index + 1}"
                        ):
                            outputs = self._dummy_run(
                                num_tokens,
                                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                force_attention=True,
                                uniform_decode=False,
                                allow_microbatching=False,
                                skip_eplb=True,
                                remove_lora=False,
                                activate_lora=activate_lora,
                                pure_prefill_num_reqs=pure_prefill_num_reqs,
                            )
                        torch.cuda.synchronize(self.device)
                        del outputs
                        gc.collect()
                        torch.cuda.synchronize(self.device)

                        after, snapshot = allocator_state()
                        device_traces = snapshot.get("device_traces", [])
                        trace_entries = (
                            device_traces[device]
                            if device < len(device_traces)
                            else []
                        )
                        if trace_count > len(trace_entries):
                            trace_count = 0
                        new_entries = trace_entries[trace_count:]
                        segment_allocs = [
                            {
                                "address": int(entry.get("addr", 0)),
                                "bytes": int(entry.get("size", 0)),
                                "stream": int(entry.get("stream", 0)),
                            }
                            for entry in new_entries
                            if entry.get("action") == "segment_alloc"
                        ]
                        record = {
                            "repeat": repeat_index + 1,
                            "wall_ns": time.perf_counter_ns() - run_started_ns,
                            "before": before,
                            "after": after,
                            "delta": {
                                key: int(after[key]) - int(before[key])
                                for key in before
                                if key != "total_bytes"
                            },
                            "segment_alloc_event_count": len(segment_allocs),
                            "segment_alloc_event_bytes": sum(
                                int(event["bytes"]) for event in segment_allocs
                            ),
                            "segment_alloc_events": segment_allocs,
                        }
                        repeat_records.append(record)
                        _append_uvm_original_alias_lifecycle(
                            "cpu_offload_uvm_prefill_allocator_reuse_diagnostic_repeat",
                            {
                                "operation": "prefill_allocator_reuse_diagnostic",
                                "num_tokens": int(num_tokens),
                                "num_requests": int(pure_prefill_num_reqs),
                                **record,
                            },
                        )
                        before = after
                        trace_count = len(trace_entries)
                except Exception as exc:
                    _append_uvm_original_alias_lifecycle(
                        "cpu_offload_uvm_prefill_allocator_reuse_diagnostic",
                        {
                            "operation": "prefill_allocator_reuse_diagnostic",
                            "ok": False,
                            "num_tokens": int(num_tokens),
                            "num_requests": int(pure_prefill_num_reqs),
                            "repeat_records": repeat_records,
                            "error": repr(exc),
                            "wall_ns": time.perf_counter_ns()
                            - diagnostic_started_ns,
                        },
                    )
                    raise

                first_segment_bytes = int(
                    repeat_records[0]["segment_alloc_event_bytes"]
                )
                later_segment_bytes = sum(
                    int(record["segment_alloc_event_bytes"])
                    for record in repeat_records[1:]
                )
                _append_uvm_original_alias_lifecycle(
                    "cpu_offload_uvm_prefill_allocator_reuse_diagnostic",
                    {
                        "operation": "prefill_allocator_reuse_diagnostic",
                        "ok": True,
                        "num_tokens": int(num_tokens),
                        "num_requests": int(pure_prefill_num_reqs),
                        "repeat_records": repeat_records,
                        "first_segment_alloc_bytes": first_segment_bytes,
                        "later_segment_alloc_bytes": later_segment_bytes,
                        "reuse_proved": bool(
                            first_segment_bytes > 0 and later_segment_bytes == 0
                        ),
                        "wall_ns": time.perf_counter_ns()
                        - diagnostic_started_ns,
                    },
                )
                raise RuntimeError(
                    "Prefill allocator reuse diagnostic completed; stopping "
                    "before hot-expert restoration and formal Graph capture"
                )

            for _ in range(self.compilation_config.cudagraph_num_of_warmups):
                # Use CUDAGraphRuntimeStyle.NONE (default) for warmup.
                # But be careful, warm up with `NONE`is orthogonal to
                # if we want to warm up attention or not. This is
                # different from the case where `FULL` implies capture
                # attention while `PIECEWISE` implies no attention.
                force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL
                with uvm_allocation_phase("enabled:dummy_run"):
                    self._dummy_run(
                        num_tokens,
                        cudagraph_runtime_mode=CUDAGraphMode.NONE,
                        force_attention=force_attention,
                        uniform_decode=uniform_decode,
                        allow_microbatching=allow_microbatching,
                        skip_eplb=True,
                        remove_lora=False,
                        activate_lora=activate_lora,
                        pure_prefill_num_reqs=pure_prefill_num_reqs,
                    )
            with uvm_allocation_phase("enabled:dummy_run"):
                self._dummy_run(
                    num_tokens,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    uniform_decode=uniform_decode,
                    allow_microbatching=allow_microbatching,
                    skip_eplb=True,
                    remove_lora=False,
                    activate_lora=activate_lora,
                    is_graph_capturing=True,
                    pure_prefill_num_reqs=pure_prefill_num_reqs,
                )
        self.maybe_remove_all_loras(self.lora_config)

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]], set[type[AttentionBackend]]]:
            layer_type = cast(type[Any], AttentionLayerBase)
            layers = get_layers_from_vllm_config(
                self.vllm_config, layer_type, kv_cache_group_spec.layer_names
            )
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()

                if layer_name in self.kv_sharing_fast_prefill_eligible_layers:
                    attn_backend = create_fast_prefill_custom_backend(
                        "FastPrefill",
                        attn_backend,  # type: ignore[arg-type]
                    )

                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                key = (full_cls_name, layer_kv_cache_spec)
                attn_backends[key] = AttentionGroupKey(
                    attn_backend, layer_kv_cache_spec
                )
                attn_backend_layers[key].append(layer_name)
            return (
                {attn_backends[k]: v for k, v in attn_backend_layers.items()},
                set(group_key.attn_backend for group_key in attn_backends.values()),
            )

        def create_attn_groups(
            attn_backends_map: dict[AttentionGroupKey, list[str]],
            kv_cache_group_id: int,
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for (attn_backend, kv_cache_spec), layer_names in attn_backends_map.items():
                attn_group = AttentionGroup(
                    attn_backend,
                    layer_names,
                    kv_cache_spec,
                    kv_cache_group_id,
                )

                attn_groups.append(attn_group)
            return attn_groups

        attention_backend_maps = []
        attention_backend_list = []
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            attention_backend_maps.append(attn_backends[0])
            attention_backend_list.append(attn_backends[1])

        # Resolve cudagraph_mode before actually initialize metadata_builders
        self._check_and_update_cudagraph_mode(
            attention_backend_list, kv_cache_config.kv_cache_groups
        )

        for i, attn_backend_map in enumerate(attention_backend_maps):
            self.attn_groups.append(create_attn_groups(attn_backend_map, i))

    def initialize_metadata_builders(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Create the metadata builders for all KV cache groups and attn groups.
        """
        for kv_cache_group_id in range(len(kv_cache_config.kv_cache_groups)):
            for attn_group in self.attn_groups[kv_cache_group_id]:
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_sizes[kv_cache_group_id]
                    if kv_cache_group_id < len(kernel_block_sizes)
                    else None,
                    num_metadata_builders=1
                    if not self.parallel_config.enable_dbo
                    else 2,
                )
        # Calculate reorder batch threshold (if needed)
        # Note (tdoublep): do this *after* constructing builders,
        # because some of them change the threshold at init time.
        self.calculate_reorder_batch_threshold()

    def _check_and_update_cudagraph_mode(
        self,
        attention_backends: list[set[type[AttentionBackend]]],
        kv_cache_groups: list[KVCacheGroupSpec],
    ) -> None:
        """
        Resolve the cudagraph_mode when there are multiple attention
        groups with potential conflicting CUDA graph support.
        Then initialize the cudagraph_dispatcher based on the resolved
        cudagraph_mode.
        """
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_backend_name = None

        for attn_backend_set, kv_cache_group in zip(
            attention_backends, kv_cache_groups
        ):
            for attn_backend in attn_backend_set:
                builder_cls = attn_backend.get_builder_cls()

                cg_support = builder_cls.get_cudagraph_support(
                    self.vllm_config, kv_cache_group.kv_cache_spec
                )
                if cg_support.value < min_cg_support.value:
                    min_cg_support = cg_support
                    min_cg_backend_name = attn_backend.__name__
        # Flexible resolve the cudagraph mode
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        # check cudagraph for mixed batch is supported
        if (
            cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL
            and min_cg_support != AttentionCGSupport.ALWAYS
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_backend_name} backend (support: "
                f"{min_cg_support})"
            )
            if min_cg_support == AttentionCGSupport.NEVER:
                # if not supported any full cudagraphs, just raise it.
                msg += (
                    "; please try cudagraph_mode=PIECEWISE, and "
                    "make sure compilation mode is VLLM_COMPILE"
                )
                raise ValueError(msg)

            # attempt to resolve the full cudagraph related mode
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=FULL_AND_PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_AND_PIECEWISE
                )
            else:
                msg += "; setting cudagraph_mode=FULL_DECODE_ONLY"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.FULL_DECODE_ONLY
                )
            logger.warning(msg)

        # check that if we are doing decode full-cudagraphs it is supported
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                f"with {min_cg_backend_name} backend (support: "
                f"{min_cg_support})"
            )
            if self.compilation_config.mode == CompilationMode.VLLM_COMPILE and (
                self.compilation_config.splitting_ops_contain_attention()
                or self.compilation_config.use_inductor_graph_partition
            ):
                msg += (
                    "; setting cudagraph_mode=PIECEWISE because "
                    "attention is compiled piecewise"
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.PIECEWISE
                )
            else:
                msg += (
                    "; setting cudagraph_mode=NONE because "
                    "attention is not compiled piecewise"
                )
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.NONE
                )
            logger.warning(msg)

        # check that if we are doing spec-decode + decode full-cudagraphs it is
        # supported
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and self.uniform_decode_query_len > 1
            and min_cg_support.value < AttentionCGSupport.UNIFORM_BATCH.value
        ):
            msg = (
                f"CUDAGraphMode.{cudagraph_mode.name} is not supported"
                f" with spec-decode for attention backend "
                f"{min_cg_backend_name} (support: {min_cg_support})"
            )
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.PIECEWISE
                )
            else:
                msg += "; setting cudagraph_mode=NONE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = (
                    CUDAGraphMode.NONE
                )
            logger.warning(msg)

        # double check that we can support full cudagraph if they are requested
        # even after automatic downgrades
        if (
            cudagraph_mode.has_full_cudagraphs()
            and min_cg_support == AttentionCGSupport.NEVER
        ):
            raise ValueError(
                f"CUDAGraphMode.{cudagraph_mode.name} is not "
                f"supported with {min_cg_backend_name} backend ("
                f"support:{min_cg_support}) "
                "; please try cudagraph_mode=PIECEWISE, "
                "and make sure compilation mode is VLLM_COMPILE"
            )

        # if we have dedicated decode cudagraphs, and spec-decode is enabled,
        # we need to adjust the cudagraph sizes to be a multiple of the uniform
        # decode query length to avoid: https://github.com/vllm-project/vllm/issues/28207
        # temp-fix: https://github.com/vllm-project/vllm/issues/28207#issuecomment-3504004536
        # Will be removed in the near future when we have seperate cudagraph capture
        # sizes for decode and mixed prefill-decode.
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            and cudagraph_mode.separate_routine()
            and self.uniform_decode_query_len > 1
        ):
            self.compilation_config.adjust_cudagraph_sizes_for_spec_decode(
                self.uniform_decode_query_len, self.parallel_config.tensor_parallel_size
            )
            capture_sizes = self.compilation_config.cudagraph_capture_sizes
            self.cudagraph_batch_sizes = (
                capture_sizes if capture_sizes is not None else []
            )

        # Trigger cudagraph dispatching keys initialization after
        # resolved cudagraph mode.
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            cudagraph_mode, self.uniform_decode_query_len
        )

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Choose the minimum reorder batch threshold from all attention groups.
        Backends should be able to support lower threshold then what they request
        just may have a performance penalty due to that backend treating decodes
        as prefills.
        """
        min_none_high = lambda a, b: a if b is None else b if a is None else min(a, b)

        reorder_batch_thresholds: list[int | None] = [
            group.get_metadata_builder().reorder_batch_threshold
            for group in self._attn_group_iterator()
        ]
        # If there are no attention groups (attention-free model) or no backend
        # reports a threshold, leave reordering disabled.
        if len(reorder_batch_thresholds) == 0:
            self.reorder_batch_threshold = None
            return
        self.reorder_batch_threshold = reduce(min_none_high, reorder_batch_thresholds)  # type: ignore[assignment]

    @staticmethod
    def select_common_block_size(
        kv_manager_block_size: int, attn_groups: list[AttentionGroup]
    ) -> int:
        """
        Select a block size that is supported by all backends and is a factor of
        kv_manager_block_size.

        If kv_manager_block_size is supported by all backends, return it directly.
        Otherwise, return the max supported size.

        Args:
            kv_manager_block_size: Block size of KV cache
            attn_groups: List of attention groups

        Returns:
            The selected block size

        Raises:
            ValueError: If no valid block size found
        """

        def block_size_is_supported(
            backends: list[type[AttentionBackend]], block_size: int
        ) -> bool:
            """
            Check if the block size is supported by all backends.
            """
            for backend in backends:
                is_supported = False
                for supported_size in backend.get_supported_kernel_block_sizes():
                    if isinstance(supported_size, int):
                        if block_size == supported_size:
                            is_supported = True
                    elif isinstance(supported_size, MultipleOf):
                        if block_size % supported_size.base == 0:
                            is_supported = True
                    else:
                        raise ValueError(f"Unknown supported size: {supported_size}")
                if not is_supported:
                    return False
            return True

        backends = [group.backend for group in attn_groups]

        # Case 1: if the block_size of kv cache manager is supported by all backends,
        # return it directly
        if block_size_is_supported(backends, kv_manager_block_size):
            return kv_manager_block_size

        # Case 2: otherwise, the block_size must be an `int`-format supported size of
        # at least one backend. Iterate over all `int`-format supported sizes in
        # descending order and return the first one that is supported by all backends.
        # Simple proof:
        # If the supported size b is in MultipleOf(x_i) format for all attention
        # backends i, and b a factor of kv_manager_block_size, then
        # kv_manager_block_size also satisfies MultipleOf(x_i) for all i. We will
        # return kv_manager_block_size in case 1.
        all_int_supported_sizes = set(
            supported_size
            for backend in backends
            for supported_size in backend.get_supported_kernel_block_sizes()
            if isinstance(supported_size, int)
        )

        for supported_size in sorted(all_int_supported_sizes, reverse=True):
            if kv_manager_block_size % supported_size != 0:
                continue
            if block_size_is_supported(backends, supported_size):
                return supported_size
        raise ValueError(f"No common block size for {kv_manager_block_size}. ")

    def may_reinitialize_input_batch(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
            if not isinstance(kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec)
        ]

        if block_sizes != [self.cache_config.block_size] or kernel_block_sizes != [
            self.cache_config.block_size
        ]:
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details."
            )
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max(self.max_model_len, self.max_encoder_len),
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                kernel_block_sizes=kernel_block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,
                is_pooling_model=self.is_pooling_model,
                num_speculative_tokens=self.num_spec_tokens,
            )

    def _allocate_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        lazy_reserve = os.environ.get(
            "VLLM_UVM_KV_CACHE_LAZY_RESERVE",
            "0",
        ).strip().lower() in ("1", "true", "yes", "on")
        page_size_by_layer: dict[str, int] = {}
        if lazy_reserve:
            for group in kv_cache_config.kv_cache_groups:
                page_size = int(group.kv_cache_spec.page_size_bytes)
                for layer_name in group.layer_names:
                    page_size_by_layer[layer_name] = page_size

        lazy_null_blocks: list[tuple[torch.Tensor, int]] = []
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            if lazy_reserve:
                tensor = torch.empty(
                    kv_cache_tensor.size, dtype=torch.int8, device=self.device
                )
                null_block_bytes = max(
                    (
                        page_size_by_layer.get(layer_name, 0)
                        for layer_name in kv_cache_tensor.shared_by
                    ),
                    default=0,
                )
                if null_block_bytes <= 0:
                    raise RuntimeError(
                        "Cannot lazily reserve KV cache without a valid null "
                        f"block size for {kv_cache_tensor.shared_by!r}"
                    )
                lazy_null_blocks.append(
                    (tensor, min(int(tensor.numel()), null_block_bytes))
                )
            else:
                tensor = torch.zeros(
                    kv_cache_tensor.size, dtype=torch.int8, device=self.device
                )
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        if lazy_null_blocks:
            device = int(torch.cuda.current_device())
            ranges = [
                (int(tensor.data_ptr()), int(null_block_bytes))
                for tensor, null_block_bytes in lazy_null_blocks
            ]
            advised = batch_advise_ranges_preferred_location(ranges, device)
            prefetched = batch_prefetch_ranges_to_device(ranges, device)
            if int(advised) != len(ranges) or int(prefetched) != len(ranges):
                raise RuntimeError(
                    "Failed to materialize lazy KV null blocks before first "
                    f"GPU write: advised={advised}/{len(ranges)} "
                    f"prefetched={prefetched}/{len(ranges)}"
                )
            torch.cuda.synchronize(device)
            for tensor, null_block_bytes in lazy_null_blocks:
                tensor[:null_block_bytes].zero_()
            torch.cuda.synchronize(device)

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), (
            "Some layers are not correctly initialized"
        )
        return kv_cache_raw_tensors

    def _attn_group_iterator(self) -> Iterator[AttentionGroup]:
        return itertools.chain.from_iterable(self.attn_groups)

    def _kv_cache_spec_attn_group_iterator(self) -> Iterator[AttentionGroup]:
        if not self.kv_cache_config.kv_cache_groups:
            return
        for attn_groups in self.attn_groups:
            yield from attn_groups

    def _prepare_kernel_block_sizes(self, kv_cache_config: KVCacheConfig) -> list[int]:
        """
        Generate kernel_block_sizes that matches each block_size.

        For attention backends that support virtual block splitting,
        use the supported block sizes from the backend.
        For other backends (like Mamba), use the same block size (no splitting).

        Args:
            kv_cache_config: The KV cache configuration.

        Returns:
            list[int]: List of kernel block sizes for each cache group.
        """
        kernel_block_sizes = []
        for kv_cache_gid, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                # All layers in the UniformTypeKVCacheSpecs have the same type,
                # Pick an arbitrary one to dispatch.
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            elif isinstance(kv_cache_spec, AttentionSpec):
                # This is an attention backend that supports virtual
                # block splitting. Get the supported block sizes from
                # all backends in the group.
                attn_groups = self.attn_groups[kv_cache_gid]
                kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
                selected_kernel_size = self.select_common_block_size(
                    kv_manager_block_size, attn_groups
                )
                kernel_block_sizes.append(selected_kernel_size)
            elif isinstance(kv_cache_spec, MambaSpec):
                # This is likely Mamba or other non-attention cache,
                # no splitting.
                kernel_block_sizes.append(kv_cache_spec.block_size)
            else:
                raise NotImplementedError(
                    f"unknown kv cache spec {kv_cache_group.kv_cache_spec}"
                )
        return kernel_block_sizes

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            if group.kv_cache_group_id == len(kernel_block_sizes):
                # There may be a last group for layers without kv cache.
                continue
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    num_blocks_per_kv_block = (
                        kv_cache_spec.block_size // kernel_block_size
                    )
                    kernel_num_blocks = num_blocks * num_blocks_per_kv_block

                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        kernel_num_blocks,
                        kernel_block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        cache_dtype_str=self.cache_config.cache_dtype,
                    )
                    dtype = kv_cache_spec.dtype
                    try:
                        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                    # The allocation respects the backend-defined stride order
                    # to ensure the semantic remains consistent for each
                    # backend. We first obtain the generic kv cache shape and
                    # then permute it according to the stride order which could
                    # result in a non-contiguous tensor.
                    kv_cache_shape = tuple(
                        kv_cache_shape[i] for i in kv_cache_stride_order
                    )
                    # Maintain original KV shape view.
                    inv_order = [
                        kv_cache_stride_order.index(i)
                        for i in range(len(kv_cache_stride_order))
                    ]
                    kv_caches[layer_name] = (
                        kv_cache_raw_tensors[layer_name]
                        .view(dtype)
                        .view(kv_cache_shape)
                        .permute(*inv_order)
                    )
                elif isinstance(kv_cache_spec, MambaSpec):
                    has_mamba = True
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    state_tensors = []
                    storage_offset_bytes = 0
                    for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        num_element_per_page = (
                            kv_cache_spec.page_size_bytes // dtype_size
                        )
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        target_stride = (num_element_per_page, *stride[1:])
                        assert storage_offset_bytes % dtype_size == 0
                        tensor = torch.as_strided(
                            raw_tensor.view(dtype),
                            size=target_shape,
                            stride=target_stride,
                            storage_offset=storage_offset_bytes // dtype_size,
                        )
                        state_tensors.append(tensor)
                        storage_offset_bytes += stride[0] * dtype_size

                    kv_caches[layer_name] = state_tensors
                else:
                    raise NotImplementedError

        if has_attn and has_mamba:
            self._update_hybrid_attention_mamba_layout(kv_caches)

        return kv_caches

    def _update_hybrid_attention_mamba_layout(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """
        Update the layout of attention layers from (2, num_blocks, ...) to
        (num_blocks, 2, ...).

        Args:
            kv_caches: The KV cache buffer of each layer.
        """

        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                kv_cache = kv_caches[layer_name]
                if isinstance(kv_cache_spec, AttentionSpec) and kv_cache.shape[0] == 2:
                    assert kv_cache.shape[1] != 2, (
                        "Fail to determine whether the layout is "
                        "(2, num_blocks, ...) or (num_blocks, 2, ...) for "
                        f"a tensor of shape {kv_cache.shape}"
                    )
                    hidden_size = kv_cache.shape[2:].numel()
                    kv_cache.as_strided_(
                        size=kv_cache.shape,
                        stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
                    )

    def initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
            kernel_block_sizes: The kernel block sizes for each KV cache group.

        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """

        # Try creating KV caches optimized for kv-connector transfers
        cache_dtype = self.cache_config.cache_dtype
        if self.use_uniform_kv_cache(self.attn_groups, cache_dtype):
            kv_caches, cross_layers_kv_cache, attn_backend = (
                self.allocate_uniform_kv_caches(
                    kv_cache_config,
                    self.attn_groups,
                    cache_dtype,
                    self.device,
                    kernel_block_sizes,
                )
            )
            self.cross_layers_kv_cache = cross_layers_kv_cache
            self.cross_layers_attn_backend = attn_backend
        else:
            # Fallback to the general case
            # Initialize the memory buffer for KV cache
            kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)

            # Change the memory buffer to the desired shape
            kv_caches = self._reshape_kv_cache_tensors(
                kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes
            )

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        num_attn_module = (
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
        )
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )
        self._uvm_kv_caches_by_layer = kv_caches
        self._register_uvm_kv_cache_physical_ranges(
            kv_cache_config,
            kv_caches,
        )
        return kv_caches

    def maybe_add_kv_sharing_layers_to_kv_cache_groups(
        self, kv_cache_config: KVCacheConfig
    ) -> None:
        """
        Add layers that re-use KV cache to KV cache group of its target layer.
        Mapping of KV cache tensors happens in `initialize_kv_cache_tensors()`
        """
        if not self.shared_kv_cache_layers:
            # No cross-layer KV sharing, return
            return

        add_kv_sharing_layers_to_kv_cache_groups(
            self.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            self.runner_only_attn_layers,
        )

        if self.cache_config.kv_sharing_fast_prefill:
            # In You Only Cache Once (https://arxiv.org/abs/2405.05254) or other
            # similar KV sharing setups, only the layers that generate KV caches
            # are involved in the prefill phase, enabling prefill to early exit.
            attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
            for layer_name in reversed(attn_layers):
                if layer_name in self.shared_kv_cache_layers:
                    self.kv_sharing_fast_prefill_eligible_layers.add(layer_name)
                else:
                    break

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        with uvm_allocation_phase("initialize_kv_cache"):
            kv_cache_config = deepcopy(kv_cache_config)
            self.kv_cache_config = kv_cache_config
            self.may_add_encoder_only_layers_to_kv_cache_config()
            self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
            self.initialize_attn_backend(kv_cache_config)
            # The kernel block size for all KV cache groups. For example, if
            # kv_cache_manager uses block_size 256 for a given group, but the attention
            # backends for that group only supports block_size 64, we will return
            # kernel_block_size 64 and split the 256-token-block to 4 blocks with 64
            # tokens each.
            kernel_block_sizes = self._prepare_kernel_block_sizes(kv_cache_config)

            # create metadata builders
            self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

            # Reinitialize need to after initialize_attn_backend
            self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
            kv_tensor_pool = nullcontext()
            if (
                self._env_flag_enabled("VLLM_UVM_KV_CACHE_MANAGED")
                and self._env_flag_enabled("VLLM_UVM_KV_CACHE_LAZY_RESERVE")
                and not self.vllm_config.model_config.enable_sleep_mode
            ):
                from vllm.device_allocator.uvm import uvm_local_memory_pool

                # Only KV tensors are lazy. Metadata builders above allocate
                # scratch buffers used by chunked prefills; leaving those
                # unbacked fails on their first write in manual-migration mode.
                kv_tensor_pool = uvm_local_memory_pool(enable_prefetch=False)
            with kv_tensor_pool:
                kv_caches = self.initialize_kv_cache_tensors(
                    kv_cache_config, kernel_block_sizes
                )

            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                # validate all draft model layers belong to the same kv cache
                # group
                self.drafter.validate_same_kv_cache_group(kv_cache_config)

            if has_kv_transfer_group():
                kv_transfer_group = get_kv_transfer_group()
                if self.cross_layers_kv_cache is not None:
                    assert self.cross_layers_attn_backend is not None
                    kv_transfer_group.register_cross_layers_kv_cache(
                        self.cross_layers_kv_cache, self.cross_layers_attn_backend
                    )
                else:
                    kv_transfer_group.register_kv_caches(kv_caches)
                kv_transfer_group.set_host_xfer_buffer_ops(copy_kv_blocks)

            if self.dcp_world_size > 1:
                layer_type = cast(type[Any], AttentionLayerBase)
                layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
                for layer in layers.values():
                    layer_impl = getattr(layer, "impl", None)
                    if layer_impl is None:
                        continue
                    assert layer_impl.need_to_return_lse_for_decode, (
                        "DCP requires attention impls to return"
                        " the softmax lse for decode, but the impl "
                        f"{layer_impl.__class__.__name__} "
                        "does not return the softmax lse for decode."
                    )

    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:
        """
        Add encoder-only layers to the KV cache config.
        """
        block_size = self.vllm_config.cache_config.block_size
        encoder_only_attn_specs: dict[AttentionSpec, list[str]] = defaultdict(list)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if attn_module.attn_type == AttentionType.ENCODER_ONLY:
                attn_spec: AttentionSpec = EncoderOnlyAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                )
                encoder_only_attn_specs[attn_spec].append(layer_name)
                self.runner_only_attn_layers.add(layer_name)
        if len(encoder_only_attn_specs) > 0:
            assert len(encoder_only_attn_specs) == 1, (
                "Only support one encoder-only attention spec now"
            )
            spec, layer_names = encoder_only_attn_specs.popitem()
            self.kv_cache_config.kv_cache_groups.append(
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec)
            )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        if has_ec_transfer() and get_ec_transfer().is_producer:
            return {}
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention) and (
                kv_tgt_layer := attn_module.kv_sharing_target_layer_name
            ):
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                kv_cache_spec[layer_name] = spec

        return kv_cache_spec

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # This is a short term mitigation for issue mentioned in
        # https://github.com/vllm-project/vllm/issues/22754.
        # `tolist` would trigger a cuda wise stream sync, which
        # would block other copy ops from other cuda streams.
        # A cuda event sync would avoid such a situation. Since
        # this is in the critical path of every single model
        # forward loop, this has caused perf issue for a disagg
        # setup.
        pinned = self.sampled_token_ids_pinned_cpu[: sampled_token_ids.shape[0]]
        pinned.copy_(sampled_token_ids, non_blocking=True)
        self.transfer_event.record()
        self.transfer_event.synchronize()
        return pinned.tolist()
