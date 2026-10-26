# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""

import json
import os
import threading
import time
import typing
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from itertools import islice
from typing import Any

import torch
from torch import nn

from vllm.attention import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors

from .interfaces import MixtureOfExperts, SupportsEagle3, SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

_QWEN3_MOE_TRACE_LOCK = threading.Lock()
_QWEN3_MOE_TRACE_STEP = 0


def _qwen3_moe_trace_enabled() -> bool:
    for env_name in (
        "VLLM_QWEN3_MOE_EXPERT_TRACE_ENABLE",
        "VLLM_MOE_TOKEN_EXPERT_TRACE_ENABLE",
    ):
        if os.environ.get(env_name, "0").lower() in ("1", "true", "yes", "on"):
            return True
    return False


def _qwen3_moe_trace_file() -> str:
    return os.environ.get(
        "VLLM_QWEN3_MOE_EXPERT_TRACE_FILE",
        os.environ.get("VLLM_MOE_TOKEN_EXPERT_TRACE_FILE", ""),
    )


def _qwen3_moe_trace_next_step() -> int:
    global _QWEN3_MOE_TRACE_STEP
    with _QWEN3_MOE_TRACE_LOCK:
        _QWEN3_MOE_TRACE_STEP += 1
        return _QWEN3_MOE_TRACE_STEP


def _qwen3_moe_trace_active_experts(
    *,
    step: int,
    layer_name: str,
    layer_id: int | None,
    num_tokens: int,
    top_k: int,
    num_experts: int,
    topk_ids: torch.Tensor,
) -> None:
    trace_file = _qwen3_moe_trace_file()
    if not trace_file:
        return

    flat_ids = topk_ids.detach().reshape(-1).to("cpu").tolist()
    valid_ids = [int(expert_id) for expert_id in flat_ids if int(expert_id) >= 0]
    counts = Counter(valid_ids)
    active_experts = sorted(counts)
    with _QWEN3_MOE_TRACE_LOCK:
        record = {
            "event": "qwen3_moe_expert_trace",
            "timestamp": time.time(),
            "pid": os.getpid(),
            "step": step,
            "layer_name": layer_name,
            "layer_id": layer_id,
            "num_tokens": int(num_tokens),
            "top_k": int(top_k),
            "num_experts": int(num_experts),
            "expert_assignments": int(len(valid_ids)),
            "active_expert_count": int(len(active_experts)),
            "active_expert_fraction": (
                float(len(active_experts) / num_experts) if num_experts else None
            ),
            "active_experts": active_experts,
            "expert_token_counts": {
                str(expert_id): int(counts[expert_id])
                for expert_id in active_experts
            },
        }
        os.makedirs(os.path.dirname(trace_file) or ".", exist_ok=True)
        with open(trace_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _qwen3_moe_slice_trace_values(
    values: list[Any],
    *,
    start: int,
    count: int,
    default: Any,
) -> list[Any]:
    sliced = values[start : start + count]
    if len(sliced) < count:
        sliced = sliced + [default] * (count - len(sliced))
    return sliced[:count]


def _qwen3_moe_current_token_metadata(
    *,
    num_tokens: int,
    is_sequence_parallel: bool,
) -> dict[str, Any]:
    if not is_forward_context_available():
        return {"metadata_available": False}

    metadata = get_forward_context().moe_token_trace_metadata
    if not isinstance(metadata, dict):
        return {"metadata_available": False}

    request_ids = list(metadata.get("request_ids") or [])
    request_indices = list(metadata.get("request_indices") or [])
    token_positions = list(metadata.get("token_positions") or [])
    token_ids = list(metadata.get("token_ids") or [])
    slice_start = 0

    if is_sequence_parallel:
        original_count = int(metadata.get("num_actual_tokens") or len(request_ids))
        tp_size = max(int(get_tensor_model_parallel_world_size()), 1)
        tp_rank = int(get_tensor_model_parallel_rank())
        padded_count = ((original_count + tp_size - 1) // tp_size) * tp_size
        chunk = padded_count // tp_size if tp_size else original_count
        slice_start = tp_rank * chunk

    return {
        "metadata_available": True,
        "metadata_source": metadata.get("source"),
        "num_reqs": metadata.get("num_reqs"),
        "num_actual_tokens": metadata.get("num_actual_tokens"),
        "token_slice_offset": int(metadata.get("token_slice_offset") or 0)
        + slice_start,
        "request_ids": _qwen3_moe_slice_trace_values(
            request_ids,
            start=slice_start,
            count=num_tokens,
            default=None,
        ),
        "request_indices": _qwen3_moe_slice_trace_values(
            request_indices,
            start=slice_start,
            count=num_tokens,
            default=-1,
        ),
        "token_positions": _qwen3_moe_slice_trace_values(
            token_positions,
            start=slice_start,
            count=num_tokens,
            default=-1,
        ),
        "token_ids": _qwen3_moe_slice_trace_values(
            token_ids,
            start=slice_start,
            count=num_tokens,
            default=-1,
        ),
    }


def _qwen3_moe_trace_token_experts(
    *,
    step: int,
    layer_name: str,
    layer_id: int | None,
    top_k: int,
    num_experts: int,
    topk_ids: torch.Tensor,
    is_sequence_parallel: bool,
) -> None:
    trace_file = _qwen3_moe_trace_file()
    if not trace_file:
        return

    try:
        num_tokens = int(topk_ids.shape[0]) if topk_ids.ndim > 0 else 1
        ids = topk_ids.detach().reshape(num_tokens, -1).to("cpu")
        ids_list = [
            [int(expert_id) for expert_id in row if int(expert_id) >= 0]
            for row in ids.tolist()
        ]
        token_metadata = _qwen3_moe_current_token_metadata(
            num_tokens=num_tokens,
            is_sequence_parallel=is_sequence_parallel,
        )
        request_ids = token_metadata.get("request_ids") or [None] * num_tokens
        token_positions = token_metadata.get("token_positions") or [-1] * num_tokens
        mapped_tokens = sum(
            1
            for request_id, token_position in zip(request_ids, token_positions)
            if request_id is not None and int(token_position) >= 0
        )
        record = {
            "event": "qwen3_moe_token_expert_trace",
            "timestamp": time.time(),
            "pid": os.getpid(),
            "step": step,
            "layer_name": layer_name,
            "layer_id": layer_id,
            "num_tokens": num_tokens,
            "mapped_tokens": int(mapped_tokens),
            "top_k": int(top_k),
            "num_experts": int(num_experts),
            "topk_ids": ids_list,
            "is_sequence_parallel": bool(is_sequence_parallel),
            **token_metadata,
        }
        with _QWEN3_MOE_TRACE_LOCK:
            os.makedirs(os.path.dirname(trace_file) or ".", exist_ok=True)
            with open(trace_file, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception:
        logger.debug("Failed to write Qwen3 MoE token expert trace", exc_info=True)


class Qwen3MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=True,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            routing_method_type=RoutingMethodType.Renormalize,
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )
        self.layer_name = prefix
        try:
            self.layer_id: int | None = extract_layer_index(prefix)
        except Exception:
            self.layer_id = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() <= 2, (
            "Qwen3MoeSparseMoeBlock only supports 1D or 2D inputs"
        )
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        if _qwen3_moe_trace_enabled():
            _, topk_ids, _ = self.experts.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            trace_step = _qwen3_moe_trace_next_step()
            _qwen3_moe_trace_active_experts(
                step=trace_step,
                layer_name=self.layer_name,
                layer_id=self.layer_id,
                num_tokens=hidden_states.shape[0],
                top_k=self.experts.top_k,
                num_experts=self.n_routed_experts,
                topk_ids=topk_ids,
            )
            _qwen3_moe_trace_token_experts(
                step=trace_step,
                layer_name=self.layer_name,
                layer_id=self.layer_id,
                top_k=self.experts.top_k,
                num_experts=self.n_routed_experts,
                topk_ids=topk_ids,
                is_sequence_parallel=self.is_sequence_parallel,
            )
        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        # return to 1d if input is 1d
        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class Qwen3MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int = 8192,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (layer_idx not in mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(
                vllm_config=vllm_config, prefix=f"{prefix}.mlp"
            )
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        pre_mlp_callback: Callable[[], None] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if pre_mlp_callback is not None:
            pre_mlp_callback()
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class _Qwen3MoePrefillOraclePrefetch:
    """Experimental fixed-plan L+1 prefetch used only by the prefill probe."""

    def __init__(self) -> None:
        self.mode = os.environ.get(
            "VLLM_UVM_PREFILL_ORACLE_PREFETCH_MODE", "off"
        ).strip().lower()
        if self.mode not in ("off", "sync", "async"):
            logger.warning(
                "Unknown VLLM_UVM_PREFILL_ORACLE_PREFETCH_MODE=%s; disabling",
                self.mode,
            )
            self.mode = "off"
        try:
            self.min_tokens = max(
                int(os.environ.get("VLLM_UVM_PREFILL_ORACLE_MIN_TOKENS", "2")),
                1,
            )
        except ValueError:
            self.min_tokens = 2
        self.fail_fast = os.environ.get(
            "VLLM_UVM_PREFILL_ORACLE_FAIL_FAST", "1"
        ).lower() in ("1", "true", "yes", "on")
        self.trace_file = os.environ.get(
            "VLLM_UVM_PREFILL_ORACLE_TRACE_FILE", ""
        )
        self.prefetch_stream: torch.cuda.Stream | None = None
        self.device: int | None = None
        self.executor: ThreadPoolExecutor | None = None
        self.modules: dict[int, FusedMoE] = {}
        self.configured_experts: dict[int, list[int]] = {}
        self.tickets: dict[int, dict[str, Any]] = {}
        self.completed_layers: set[int] = set()
        self.layer_records: list[dict[str, Any]] = []
        self.active = False
        self.completed_once = False
        self.generation = 0
        self.num_tokens = 0
        self.started_ns = 0

    def _shutdown_executor(self) -> None:
        executor, self.executor = self.executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    def _reject(
        self,
        message: str,
        *,
        cause: BaseException | None = None,
    ) -> bool:
        try:
            self.abort(message)
        except BaseException as drain_error:
            raise RuntimeError(
                f"{message}; failed to drain prefill oracle pipeline"
            ) from drain_error
        if self.fail_fast:
            raise RuntimeError(message) from cause
        logger.error(message)
        return False

    @staticmethod
    def _expert_module(layer: nn.Module) -> FusedMoE | None:
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        return experts if isinstance(experts, FusedMoE) else None

    def begin(
        self,
        layer_items: list[tuple[int, nn.Module]],
        *,
        num_tokens: int,
    ) -> bool:
        self.active = False
        if (
            self.mode == "off"
            or self.completed_once
            or int(num_tokens) < self.min_tokens
            or not torch.cuda.is_available()
        ):
            return False

        self.modules = {}
        self.configured_experts = {}
        for layer_id, layer in layer_items:
            module = self._expert_module(layer)
            if module is None:
                continue
            configured = sorted(
                int(expert_id)
                for plan_layer_id, expert_id in module.uvm_init_weight_offload_plan_keys
                if int(plan_layer_id) == int(layer_id)
            )
            if configured:
                self.modules[int(layer_id)] = module
                self.configured_experts[int(layer_id)] = configured

        if not self.configured_experts:
            return self._reject("Prefill oracle plan contains no local experts")

        # Model warmup runs before Stage-M post-warmup offload.  Ignore those
        # forwards and activate only after every configured layer was released.
        if not all(
            bool(getattr(module, "_uvm_init_weight_offload_done", False))
            for module in self.modules.values()
        ):
            return False

        if any(
            module.uvm_weight_executor != "cpp_batch"
            for module in self.modules.values()
        ):
            return self._reject("Prefill oracle prefetch requires cpp_batch")

        remaining = {
            layer_id: module.uvm_prefill_oracle_plan_expert_ids()
            for layer_id, module in self.modules.items()
        }
        mismatched = {
            layer_id: {
                "configured": self.configured_experts[layer_id],
                "remaining": remaining[layer_id],
            }
            for layer_id in self.modules
            if remaining[layer_id] != self.configured_experts[layer_id]
        }
        if mismatched:
            return self._reject(
                f"Prefill oracle plan/offloaded state mismatch: {mismatched!r}"
            )

        device = int(torch.cuda.current_device())
        self.device = device
        if self.prefetch_stream is None:
            self.prefetch_stream = torch.cuda.Stream(device=device)
        if self.mode == "async":
            self.executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="qwen-prefill-oracle",
            )
        self.tickets = {}
        self.completed_layers = set()
        self.layer_records = []
        self.generation += 1
        self.num_tokens = int(num_tokens)
        self.started_ns = time.perf_counter_ns()
        self.active = True

        first_layer_id = int(layer_items[0][0])
        entry_event = torch.cuda.Event(enable_timing=True)
        entry_event.record(torch.cuda.current_stream(device))
        submitted = self._submit(
            target_layer_id=first_layer_id,
            source_layer_id=None,
            trigger_event=entry_event,
            use_worker=False,
        )
        if submitted and first_layer_id in self.modules:
            # The first layer has no preceding MoE compute to overlap.  Make it
            # an identical, fully synchronous cost in both comparison modes.
            self._complete_ticket(first_layer_id)
        return bool(self.active)

    def _run_prepare(self, ticket: dict[str, Any]) -> dict[str, Any]:
        """Run one prepare operation, optionally on the executor thread."""
        stream = typing.cast(torch.cuda.Stream, ticket["prefetch_stream"])
        trigger = typing.cast(torch.cuda.Event, ticket["trigger_event"])
        prefetch_start = typing.cast(torch.cuda.Event, ticket["prefetch_start"])
        ready = typing.cast(torch.cuda.Event, ticket["ready"])
        module = typing.cast(FusedMoE, ticket["module"])
        expert_ids = typing.cast(list[int], ticket["configured_expert_ids"])
        device = int(ticket["device"])
        result: dict[str, object] = {
            "ok": False,
            "reason": "submission_not_started",
        }
        submission_error: BaseException | None = None
        ready_error: BaseException | None = None
        prefetch_start_recorded = False
        ready_recorded = False
        worker_start_ns = time.perf_counter_ns()

        try:
            with torch.cuda.device(device), torch.cuda.stream(stream):
                try:
                    stream.wait_event(trigger)
                    prefetch_start.record(stream)
                    prefetch_start_recorded = True
                    result = module.prepare_uvm_prefill_oracle_experts(
                        expert_ids,
                        device=device,
                    )
                except BaseException as exc:
                    submission_error = exc
                    result = {
                        "ok": False,
                        "reason": "submission_exception",
                    }
                finally:
                    try:
                        ready.record(stream)
                        ready_recorded = True
                    except BaseException as exc:
                        ready_error = exc
        except BaseException as exc:
            if submission_error is None:
                submission_error = exc

        # If entering the CUDA context failed before the inner finally, make a
        # best effort to create a drain point without hiding the first error.
        if not ready_recorded:
            try:
                with torch.cuda.device(device), torch.cuda.stream(stream):
                    ready.record(stream)
                    ready_recorded = True
            except BaseException as exc:
                if ready_error is None:
                    ready_error = exc

        return {
            "result": result,
            "submission_error": submission_error,
            "ready_error": ready_error,
            "prefetch_start_recorded": prefetch_start_recorded,
            "ready_recorded": ready_recorded,
            "prepare_host_ms": (time.perf_counter_ns() - worker_start_ns) / 1e6,
            "worker_thread_id": threading.get_ident(),
        }

    def _submit(
        self,
        *,
        target_layer_id: int,
        source_layer_id: int | None,
        trigger_event: torch.cuda.Event,
        use_worker: bool,
    ) -> bool:
        if not self.active or target_layer_id not in self.modules:
            return True
        if (
            target_layer_id in self.tickets
            or target_layer_id in self.completed_layers
        ):
            return self._reject(
                f"Duplicate prefill oracle ticket for layer {target_layer_id}"
            )
        assert self.prefetch_stream is not None
        assert self.device is not None
        module = self.modules[target_layer_id]
        expert_ids = self.configured_experts[target_layer_id]
        prefetch_start = torch.cuda.Event(enable_timing=True)
        ready = torch.cuda.Event(enable_timing=True)
        ticket: dict[str, Any] = {
            "target_layer_id": int(target_layer_id),
            "source_layer_id": source_layer_id,
            "module": module,
            "configured_expert_ids": list(expert_ids),
            "device": int(self.device),
            "dispatch_kind": "async_worker" if use_worker else "foreground",
            "trigger_event": trigger_event,
            "prefetch_stream": self.prefetch_stream,
            "prefetch_start": prefetch_start,
            "ready": ready,
        }
        self.tickets[target_layer_id] = ticket
        submit_start_ns = time.perf_counter_ns()
        if use_worker:
            assert self.executor is not None
            try:
                ticket["future"] = self.executor.submit(
                    self._run_prepare,
                    ticket,
                )
            except BaseException as exc:
                self.tickets.pop(target_layer_id, None)
                return self._reject(
                    "Failed to dispatch prefill oracle worker: "
                    f"target_layer={target_layer_id}",
                    cause=exc,
                )
        else:
            ticket["outcome"] = self._run_prepare(ticket)
        ticket["loop_submit_us"] = (
            time.perf_counter_ns() - submit_start_ns
        ) / 1e3
        return True

    @staticmethod
    def _apply_result_to_ticket(
        ticket: dict[str, Any],
        result: dict[str, object],
    ) -> None:
        ticket["expert_ids"] = list(result.get("expert_ids") or [])
        ticket["region_ids"] = list(result.get("region_ids") or [])
        ticket["requested_bytes"] = int(result.get("requested_bytes") or 0)
        ticket["region_count"] = int(result.get("region_count") or 0)
        ticket["initial_access_mode"] = result.get("initial_access_mode")

    def _resolve_and_drain_ticket(
        self,
        ticket: dict[str, Any],
    ) -> tuple[dict[str, Any], float, float]:
        future_wait_start_ns = time.perf_counter_ns()
        outcome = ticket.get("outcome")
        if outcome is None:
            future = typing.cast(Future[dict[str, Any]], ticket["future"])
            outcome = future.result()
            ticket["outcome"] = outcome
        future_wait_ms = (time.perf_counter_ns() - future_wait_start_ns) / 1e6

        ready_wait_start_ns = time.perf_counter_ns()
        if bool(outcome.get("ready_recorded")):
            ready = typing.cast(torch.cuda.Event, ticket["ready"])
            ready.synchronize()
        else:
            # No event means no safe fail-open point.  Synchronize the entire
            # dedicated stream before inspecting or falling back.
            stream = typing.cast(torch.cuda.Stream, ticket["prefetch_stream"])
            device = int(ticket["device"])
            with torch.cuda.device(device):
                stream.synchronize()
        ready_wait_ms = (time.perf_counter_ns() - ready_wait_start_ns) / 1e6
        return typing.cast(dict[str, Any], outcome), future_wait_ms, ready_wait_ms

    def _complete_ticket(self, layer_id: int) -> None:
        if layer_id in self.completed_layers:
            return
        ticket = self.tickets.get(int(layer_id))
        if ticket is None:
            self._reject(f"Missing prefill oracle ticket for layer {layer_id}")
            return

        wait_start_ns = time.perf_counter_ns()
        try:
            outcome, future_wait_ms, ready_wait_ms = (
                self._resolve_and_drain_ticket(ticket)
            )
        except BaseException as exc:
            self._reject(
                f"Failed to drain prefill oracle ticket for layer {layer_id}",
                cause=exc,
            )
            return
        exposed_wait_ms = (time.perf_counter_ns() - wait_start_ns) / 1e6
        result = typing.cast(dict[str, object], outcome["result"])
        submission_error = typing.cast(
            BaseException | None,
            outcome.get("submission_error"),
        )
        ready_error = typing.cast(
            BaseException | None,
            outcome.get("ready_error"),
        )
        if submission_error is not None or ready_error is not None:
            self._reject(
                "Prefill oracle worker raised: "
                f"target_layer={layer_id} result={result!r}",
                cause=submission_error or ready_error,
            )
            return
        if not bool(result.get("ok")):
            self._reject(
                "Failed to submit prefill oracle prefetch: "
                f"target_layer={layer_id} result={result!r}"
            )
            return

        self._apply_result_to_ticket(ticket, result)
        prefetch_start = typing.cast(torch.cuda.Event, ticket["prefetch_start"])
        ready = typing.cast(torch.cuda.Event, ticket["ready"])
        if not bool(outcome.get("prefetch_start_recorded")):
            self._reject(
                f"Missing prefetch start event for layer {layer_id}"
            )
            return
        prefetch_ms = float(prefetch_start.elapsed_time(ready))
        module = typing.cast(FusedMoE, ticket["module"])
        commit = module.commit_uvm_prefill_oracle_experts(
            typing.cast(list[int], ticket["expert_ids"]),
            typing.cast(list[str], ticket["region_ids"]),
        )
        if not bool(commit.get("ok")):
            # commit() already degraded the complete batch to UNKNOWN, so an
            # abort must not try to commit the same ticket a second time.
            self.tickets.pop(int(layer_id), None)
            self._reject(
                "Failed to commit prefill oracle prefetch: "
                f"layer={layer_id} result={commit!r}"
            )
            return
        self.completed_layers.add(int(layer_id))
        self.tickets.pop(int(layer_id), None)
        self.layer_records.append(
            {
                "target_layer_id": int(layer_id),
                "source_layer_id": ticket["source_layer_id"],
                "expert_ids": ticket["expert_ids"],
                "expert_count": len(typing.cast(list[int], ticket["expert_ids"])),
                "region_count": ticket["region_count"],
                "requested_bytes": ticket["requested_bytes"],
                "initial_access_mode": ticket["initial_access_mode"],
                "dispatch_kind": ticket["dispatch_kind"],
                "loop_submit_us": ticket["loop_submit_us"],
                "prepare_host_ms": outcome["prepare_host_ms"],
                "worker_thread_id": outcome["worker_thread_id"],
                "future_wait_ms": future_wait_ms,
                "ready_wait_ms": ready_wait_ms,
                "prefetch_cuda_ms": prefetch_ms,
                "exposed_wait_ms": exposed_wait_ms,
            }
        )

    def wait_for_layer(self, layer_id: int) -> None:
        if not self.active or layer_id not in self.modules:
            return
        self._complete_ticket(int(layer_id))

    def submit_next_layer(
        self,
        *,
        source_layer_id: int,
        target_layer_id: int,
        trigger_event: torch.cuda.Event,
    ) -> None:
        if not self.active:
            return
        if target_layer_id not in self.modules:
            return
        submitted = self._submit(
            target_layer_id=int(target_layer_id),
            source_layer_id=int(source_layer_id),
            trigger_event=trigger_event,
            use_worker=self.mode == "async",
        )
        if submitted and self.mode == "sync":
            # A true no-overlap control: do not let the target attention hide
            # any part of migration, even if CUDA submission becomes cheap.
            self._complete_ticket(int(target_layer_id))

    def abort(self, reason: str = "") -> None:
        """Drain in-flight work and leave region tracking in a safe state."""
        self.active = False
        errors: list[BaseException] = []
        for layer_id, ticket in list(self.tickets.items()):
            try:
                outcome, _, _ = self._resolve_and_drain_ticket(ticket)
                result = typing.cast(dict[str, object], outcome["result"])
                if bool(result.get("ok")):
                    self._apply_result_to_ticket(ticket, result)
                    module = typing.cast(FusedMoE, ticket["module"])
                    commit = module.commit_uvm_prefill_oracle_experts(
                        typing.cast(list[int], ticket["expert_ids"]),
                        typing.cast(list[str], ticket["region_ids"]),
                    )
                    if not bool(commit.get("ok")):
                        raise RuntimeError(
                            "abort commit failed: "
                            f"layer={layer_id} result={commit!r}"
                        )
            except BaseException as exc:
                errors.append(exc)
            finally:
                self.tickets.pop(layer_id, None)
        self._shutdown_executor()
        if errors:
            detail = f" ({reason})" if reason else ""
            raise RuntimeError(
                f"Could not safely drain prefill oracle pipeline{detail}"
            ) from errors[0]

    def finish(self) -> None:
        if not self.active:
            return
        expected_layers = sorted(self.modules)
        completed_layers = sorted(
            int(record["target_layer_id"]) for record in self.layer_records
        )
        if completed_layers != expected_layers:
            self._reject(
                "Incomplete prefill oracle pipeline: "
                f"expected={expected_layers} completed={completed_layers}"
            )
            return
        record = {
            "event": "qwen3_moe_prefill_oracle_prefetch",
            "timestamp": time.time(),
            "pid": os.getpid(),
            "generation": self.generation,
            "mode": self.mode,
            "num_tokens": self.num_tokens,
            "layer_count": len(self.layer_records),
            "expert_count": sum(
                int(item["expert_count"]) for item in self.layer_records
            ),
            "requested_bytes": sum(
                int(item["requested_bytes"]) for item in self.layer_records
            ),
            "prefetch_cuda_sum_ms": sum(
                float(item["prefetch_cuda_ms"]) for item in self.layer_records
            ),
            "exposed_wait_sum_ms": sum(
                float(item["exposed_wait_ms"]) for item in self.layer_records
            ),
            "prepare_host_sum_ms": sum(
                float(item["prepare_host_ms"]) for item in self.layer_records
            ),
            "loop_submit_sum_ms": sum(
                float(item["loop_submit_us"]) for item in self.layer_records
            )
            / 1e3,
            "host_pipeline_ms": (time.perf_counter_ns() - self.started_ns) / 1e6,
            "layers": self.layer_records,
        }
        if self.trace_file:
            with _QWEN3_MOE_TRACE_LOCK:
                os.makedirs(os.path.dirname(self.trace_file) or ".", exist_ok=True)
                with open(self.trace_file, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.completed_once = True
        self.active = False
        self._shutdown_executor()


@support_torch_compile
class Qwen3MoeModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Qwen3MoeDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self._uvm_prefill_oracle = _Qwen3MoePrefillOraclePrefetch()
        if self._uvm_prefill_oracle.mode != "off":
            if not vllm_config.model_config.enforce_eager:
                raise RuntimeError("Prefill oracle prefetch requires eager mode")
            if parallel_config.enable_dbo:
                raise RuntimeError("Prefill oracle prefetch does not support DBO")
            if (
                int(parallel_config.tensor_parallel_size) != 1
                or parallel_config.enable_expert_parallel
            ):
                raise RuntimeError(
                    "Prefill oracle prefetch currently requires TP=1 and EP disabled"
                )
        # Track layers for auxiliary hidden state outputs (EAGLE3)
        self.aux_hidden_state_layers: tuple[int, ...] = ()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        layer_items = list(
            enumerate(
                islice(self.layers, self.start_layer, self.end_layer),
                start=self.start_layer,
            )
        )
        oracle_active = self._uvm_prefill_oracle.begin(
            layer_items,
            num_tokens=int(hidden_states.shape[0]),
        )
        try:
            for item_index, (layer_idx, layer) in enumerate(layer_items):
                # Collect auxiliary hidden states if specified
                if layer_idx in self.aux_hidden_state_layers:
                    aux_hidden_state = (
                        hidden_states + residual
                        if residual is not None
                        else hidden_states
                    )
                    aux_hidden_states.append(aux_hidden_state)
                source_start_event = None
                if (
                    oracle_active
                    and item_index + 1 < len(layer_items)
                    and self._uvm_prefill_oracle.mode == "async"
                ):
                    source_start_event = torch.cuda.Event(enable_timing=True)
                    source_start_event.record(torch.cuda.current_stream())
                pre_mlp_callback = None
                if oracle_active:
                    pre_mlp_callback = partial(
                        self._uvm_prefill_oracle.wait_for_layer,
                        layer_idx,
                    )
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    residual,
                    pre_mlp_callback=pre_mlp_callback,
                )
                if oracle_active and item_index + 1 < len(layer_items):
                    if self._uvm_prefill_oracle.mode == "async":
                        assert source_start_event is not None
                        trigger_event = source_start_event
                    else:
                        trigger_event = torch.cuda.Event(enable_timing=True)
                        trigger_event.record(torch.cuda.current_stream())
                    self._uvm_prefill_oracle.submit_next_layer(
                        source_layer_id=layer_idx,
                        target_layer_id=layer_items[item_index + 1][0],
                        trigger_event=trigger_event,
                    )

            if oracle_active:
                self._uvm_prefill_oracle.finish()
        except BaseException:
            if oracle_active:
                try:
                    self._uvm_prefill_oracle.abort("model forward failed")
                except BaseException:
                    logger.exception("Failed to abort prefill oracle pipeline")
            raise

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        # Return auxiliary hidden states if collected
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if (
                        name_mapped.endswith(ignore_suffixes)
                        and name_mapped not in params_dict
                    ):
                        continue

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale"
                        )
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv-scale is not loaded.",  # noqa: E501
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3MoeForCausalLM(
    nn.Module, SupportsPP, SupportsLoRA, SupportsEagle3, MixtureOfExperts
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ]
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config
        # Only perform the following mapping when Qwen3MoeMLP exists
        if getattr(config, "mlp_only_layers", []):
            self.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]
        self.model = Qwen3MoeModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.expert_weights = []

        self.moe_layers = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, Qwen3MoeDecoderLayer)
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            raise RuntimeError("No Qwen3MoE layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
