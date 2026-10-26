# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import hashlib
import itertools
import json
import os
import threading
import time
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, overload

import torch
import torch.nn as nn
from torch.func import functional_call
from transformers import PretrainedConfig
from typing_extensions import deprecated

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)
from vllm.model_executor.model_loader.online_quantization import (
    support_quantized_model_reload_from_hp_weights,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import supports_any_eagle
from vllm.multimodal import NestedTensors
from vllm.sequence import IntermediateTensors
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import (
    is_pin_memory_available,
    is_uva_available,
)
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    get_cuda_view_from_cpu_tensor,
)

logger = init_logger(__name__)

WeightsMapping = Mapping[str, str | None]
"""If a key maps to a value of `None`, the corresponding weight is ignored."""


@dataclass
class WeightsMapper:
    """Maps the name of each weight if they match the following patterns."""

    orig_to_new_substr: WeightsMapping = field(default_factory=dict)
    orig_to_new_prefix: WeightsMapping = field(default_factory=dict)
    orig_to_new_suffix: WeightsMapping = field(default_factory=dict)

    def __or__(self, other: "WeightsMapper") -> "WeightsMapper":
        """Combine two `WeightsMapper`s by merging their mappings."""
        return WeightsMapper(
            orig_to_new_substr={**self.orig_to_new_substr, **other.orig_to_new_substr},
            orig_to_new_prefix={**self.orig_to_new_prefix, **other.orig_to_new_prefix},
            orig_to_new_suffix={**self.orig_to_new_suffix, **other.orig_to_new_suffix},
        )

    def _map_name(self, key: str) -> str | None:
        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None

                key = key.replace(substr, new_key, 1)

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None

                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None

                key = new_key.join(key.rsplit(suffix, 1))

        return key

    def apply(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        return (
            (out_name, data)
            for name, data in weights
            if (out_name := self._map_name(name)) is not None
        )

    def apply_list(self, values: list[str]) -> list[str]:
        return [
            out_name
            for name in values
            if (out_name := self._map_name(name)) is not None
        ]

    def apply_dict(self, values: dict[str, Any]) -> dict[str, Any]:
        return {
            out_name: value
            for name, value in values.items()
            if (out_name := self._map_name(name)) is not None
        }


class AutoWeightsLoader:
    """
    Helper class to load weights into a [`torch.nn.Module`][]. It is able
    to automatically detect child modules and parameters while iterating over
    the weights only once.

    The weight loading logic for individual modules can be overridden
    by defining a `load_weights` method.

    Similarly, the weight loading logic for individual parameters can be
    overridden by defining a `weight_loader` method.

    Detailed weight loading information can be viewed by setting the
    environment variable `VLLM_LOGGING_LEVEL=DEBUG`.
    """

    # Models trained using early version ColossalAI or quantized by
    # GPTQModel may include these tensors in checkpoint. Skip them.
    ROTARY_EMBEDS_UNUSED_WEIGHTS = [
        "rotary_pos_emb.inv_freq",
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
    ]

    def __init__(
        self,
        module: nn.Module,
        *,
        skip_prefixes: list[str] | None = None,
        skip_substrs: list[str] | None = None,
        ignore_unexpected_prefixes: list[str] | None = None,
        ignore_unexpected_suffixes: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.module = module
        self.skip_prefixes = skip_prefixes or []
        self.skip_substrs = skip_substrs or []
        self.ignore_unexpected_prefixes = ignore_unexpected_prefixes or []
        self.ignore_unexpected_suffixes = ignore_unexpected_suffixes or []
        # update default skip_substrs
        self.skip_substrs += self.ROTARY_EMBEDS_UNUSED_WEIGHTS

    def _groupby_prefix(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[tuple[str, Iterable[tuple[str, torch.Tensor]]]]:
        weights_by_parts = (
            (weight_name.split(".", 1), weight_data)
            for weight_name, weight_data in weights
        )

        for prefix, group in itertools.groupby(weights_by_parts, key=lambda x: x[0][0]):
            yield (
                prefix,
                # Because maxsplit=1 in weight_name.split(...),
                # the length of `parts` must either be 1 or 2
                (
                    ("" if len(parts) == 1 else parts[1], weights_data)
                    for parts, weights_data in group
                ),
            )

    def _get_qualname(self, prefix: str, rest: str) -> str:
        if prefix == "":
            return rest
        if rest == "":
            return prefix

        return ".".join((prefix, rest))

    def _can_skip(self, qualname: str) -> bool:
        return any(qualname.startswith(p) for p in self.skip_prefixes) or any(
            substr in qualname for substr in self.skip_substrs
        )

    def _can_ignore_unexpected(self, qualname: str) -> bool:
        iup = (qualname.startswith(p) for p in self.ignore_unexpected_prefixes)
        ius = (qualname.endswith(s) for s in self.ignore_unexpected_suffixes)
        return any(iup) or any(ius)

    def _load_param(
        self,
        base_prefix: str,
        param: nn.Parameter,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        for weight_name, weight_data in weights:
            weight_qualname = self._get_qualname(base_prefix, weight_name)

            if self._can_skip(weight_qualname):
                logger.debug("Skipping weight %s", weight_qualname)

                continue

            if weight_name != "":
                if self._can_ignore_unexpected(weight_qualname):
                    logger.debug("Ignoring weight %s", weight_qualname)

                    continue

                raise ValueError(
                    f"Attempted to load nested weight '{weight_qualname}' "
                    f"into a single parameter '{base_prefix}'"
                )

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight_data)

            logger.debug("Loaded weight %s with shape %s", weight_qualname, param.shape)

            yield weight_qualname

    def _add_loadable_non_param_tensors(
        self, module: nn.Module, child_params: dict[str, torch.Tensor]
    ):
        """
        Add tensor names that are not in the model params that may be in the
        safetensors, e.g., batch normalization stats.
        """
        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.LazyBatchNorm1d,
                nn.LazyBatchNorm2d,
                nn.LazyBatchNorm3d,
                nn.SyncBatchNorm,
            ),
        ):
            module_state_dict = module.state_dict()
            for stat_name in ("running_mean", "running_var", "num_batches_tracked"):
                child_params[stat_name] = module_state_dict[stat_name]

    def _load_module(
        self,
        base_prefix: str,
        module: nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        if isinstance(module, PPMissingLayer):
            return

        # Avoid infinite recursion since this function is typically
        # called inside load_weights of the module itself
        if module != self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                loaded_params = module_load_weights(weights)
                if loaded_params is None:
                    logger.warning(
                        "Unable to collect loaded parameters for module %s", module
                    )
                else:
                    yield from map(
                        lambda x: self._get_qualname(base_prefix, x),
                        loaded_params,
                    )

        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))

        # Add missing tensors the weight loader needs to be able to load
        # that aren't registered as params, e.g., batchnorm statistics.
        self._add_loadable_non_param_tensors(module, child_params)

        for child_prefix, child_weights in self._groupby_prefix(weights):
            prefix = self._get_qualname(base_prefix, child_prefix)

            if child_prefix in child_modules:
                if self._can_skip(prefix + "."):
                    logger.debug("Skipping module %s", prefix)

                    continue

                yield from self._load_module(
                    prefix, child_modules[child_prefix], child_weights
                )
            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)

                    continue

                yield from self._load_param(
                    prefix, child_params[child_prefix], child_weights
                )
            else:
                can_skip_module = self._can_skip(prefix + ".")
                can_skip_param = self._can_skip(prefix)
                if can_skip_module or can_skip_param:
                    logger.debug("Skipping missing %s", prefix)

                    continue

                can_ignore_module = self._can_ignore_unexpected(prefix + ".")
                can_ignore_param = self._can_ignore_unexpected(prefix)
                if can_ignore_module or can_ignore_param:
                    logger.debug("Ignoring missing %s", prefix)

                    continue

                msg = (
                    f"There is no module or parameter named '{prefix}' "
                    f"in {type(self.module).__name__}"
                )
                raise ValueError(msg)

    @support_quantized_model_reload_from_hp_weights
    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        mapper: WeightsMapper | None = None,
    ) -> set[str]:
        if mapper is not None:
            weights = mapper.apply(weights)
        # filter out weights with first-prefix/substr to skip in name
        weights = (
            (name, weight) for name, weight in weights if not self._can_skip(name)
        )

        autoloaded_weights = set(self._load_module("", self.module, weights))
        return autoloaded_weights


def init_vllm_registered_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    hf_config: PretrainedConfig | None = None,
    architectures: list[str] | None = None,
) -> nn.Module:
    """
    Helper function to initialize an inner model registered to vLLM,
    based on the arguments passed to the outer vLLM model.
    """
    from vllm.model_executor.model_loader.utils import initialize_model

    if hf_config is None and architectures is not None:
        # So that the architectures field is overridden
        hf_config = vllm_config.model_config.hf_config

    if hf_config is not None:
        vllm_config = vllm_config.with_hf_config(hf_config, architectures=architectures)

    return initialize_model(vllm_config=vllm_config, prefix=prefix)


@overload
def flatten_bn(x: torch.Tensor) -> torch.Tensor: ...


@overload
def flatten_bn(x: list[torch.Tensor]) -> list[torch.Tensor]: ...


@overload
def flatten_bn(
    x: list[torch.Tensor] | torch.Tensor,
    *,
    concat: Literal[True],
) -> torch.Tensor: ...


@overload
def flatten_bn(
    x: list[torch.Tensor] | torch.Tensor,
    *,
    concat: bool = False,
) -> list[torch.Tensor] | torch.Tensor: ...


def flatten_bn(
    x: list[torch.Tensor] | torch.Tensor,
    *,
    concat: bool = False,
) -> list[torch.Tensor] | torch.Tensor:
    """
    Flatten the `B` and `N` dimensions of batched multimodal inputs.

    The input tensor should have shape `(B, N, ...)`.
    """
    if isinstance(x, torch.Tensor):
        return x.flatten(0, 1)

    if concat:
        return torch.cat(x)

    return [x_n for x_b in x for x_n in x_b]


def _flatten_embeddings(embeddings: NestedTensors) -> torch.Tensor:
    """
    Recursively flattens and concatenates NestedTensors on all but the last
    dimension.
    """

    if isinstance(embeddings, torch.Tensor):
        # Flatten all but the last dimension.
        return embeddings.flatten(0, -2)

    return torch.cat(tuple(_flatten_embeddings(t) for t in embeddings))


def _embedding_count_expression(embeddings: NestedTensors) -> str:
    """
    Constructs a debugging representation of the number of embeddings in the
    NestedTensors.
    """

    if isinstance(embeddings, torch.Tensor):
        return " x ".join([str(dim) for dim in embeddings.shape[:-1]])

    return " + ".join(_embedding_count_expression(inner) for inner in embeddings)


def split_list_into_ranges(lst: torch.Tensor, interval: int) -> list[list[int]]:
    ranges: list[list[int]] = [[] for _ in range((max(lst) // interval) + 1)]
    for num in lst:
        index = num // interval
        ranges[index].append(num)
    return ranges


def _merge_multimodal_embeddings(
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    is_multimodal: torch.Tensor,
) -> torch.Tensor:
    """
    Merge `multimodal_embeddings` into `inputs_embeds` by overwriting the
    positions in `inputs_embeds` corresponding to placeholder tokens in
    `input_ids`.

    Note:
        This updates `inputs_embeds` in place.
    """
    if len(multimodal_embeddings) == 0:
        return inputs_embeds

    mm_embeds_flat = _flatten_embeddings(multimodal_embeddings)
    input_dtype = inputs_embeds.dtype

    try:
        # For debugging
        # inputs_embeds[is_multimodal] = mm_embeds_flat.to(dtype=input_dtype)

        # NOTE: This can avoid D2H sync (#22105), but fails to
        # raise an error if is_multimodal.sum() < len(mm_embeds_flat)
        inputs_embeds.masked_scatter_(
            is_multimodal.unsqueeze(-1), mm_embeds_flat.to(dtype=input_dtype)
        )
    except RuntimeError as e:
        num_actual_tokens = len(mm_embeds_flat)
        num_expected_tokens = is_multimodal.sum().item()

        if num_actual_tokens != num_expected_tokens:
            expr = _embedding_count_expression(multimodal_embeddings)

            raise ValueError(
                f"Attempted to assign {expr} = {num_actual_tokens} "
                f"multimodal tokens to {num_expected_tokens} placeholders"
            ) from e

        raise ValueError("Error during masked scatter operation") from e

    return inputs_embeds


@deprecated(
    "`merge_multimodal_embeddings` has been replaced with "
    "`SupportsMultiModal.embed_input_ids` and will be "
    "removed in v0.12."
)
def merge_multimodal_embeddings(
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    multimodal_embeddings: NestedTensors,
    placeholder_token_id: int | list[int],
) -> torch.Tensor:
    """
    Merge `multimodal_embeddings` into `inputs_embeds` by overwriting the
    positions in `inputs_embeds` corresponding to placeholder tokens in
    `input_ids`.

    `placeholder_token_id` can be a list of token ids (e.g, token ids
    of img_start, img_break, and img_end tokens) when needed: This means
    the order of these tokens in the `input_ids` MUST MATCH the order of
    their embeddings in `multimodal_embeddings` since we need to
    slice-merge instead of individually scattering.

    For example, if input_ids is "TTTTTSIIIBIIIBIIIETTT", where
    - T is text token
    - S is image start token
    - I is image embedding token
    - B is image break token
    - E is image end token.

    Then the image embeddings (that correspond to I's) from vision encoder
    must be padded with embeddings of S, B, and E in the same order of
    input_ids for a correct embedding merge.

    Note:
        This updates `inputs_embeds` in place.
    """
    if isinstance(placeholder_token_id, list):
        is_multimodal = isin_list(input_ids, placeholder_token_id)
    else:
        is_multimodal = input_ids == placeholder_token_id

    return _merge_multimodal_embeddings(
        inputs_embeds,
        multimodal_embeddings=multimodal_embeddings,
        is_multimodal=is_multimodal,
    )


def isin_list(
    elements: torch.Tensor,
    test_elements_list: list[int],
) -> torch.Tensor:
    test_elements = torch.tensor(
        test_elements_list,
        pin_memory=is_pin_memory_available(),
    ).to(device=elements.device, non_blocking=True)

    return torch.isin(elements, test_elements)


class LayerFn(Protocol):
    def __call__(self, prefix: str) -> torch.nn.Module: ...


class PPMissingLayer(torch.nn.Identity):
    """
    A placeholder layer for missing layers in a pipeline parallel model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        """Return the first arg from args or the first value from kwargs."""
        return args[0] if args else next(iter(kwargs.values()))


_CPU_OFFLOAD_BYTES = 0
_CPU_OFFLOAD_MAX_BYTES = 0
_CPU_OFFLOAD_ORDER = 0
_CPU_OFFLOAD_EXACT_NAMES_FILE = ""
_CPU_OFFLOAD_EXACT_NAMES: frozenset[str] | None = None
_CPU_OFFLOAD_EXACT_PENDING: set[str] = set()
_CPU_OFFLOAD_EXACT_SEEN: set[str] = set()
_CPU_OFFLOAD_STATIC_BANK = False
_CPU_OFFLOAD_STATIC_BANKS: dict[tuple[Any, ...], torch.Tensor] = {}
_CPU_OFFLOAD_STATIC_BANK_BYTES = 0
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS = False
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN: dict[str, "_GraphStaticAliasNamePlan"] = {}
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE = ""
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER: dict[int, tuple[int, ...]] = {}
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE = ""
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASS_FILES: dict[str, str] = {}
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS: dict[
    str, dict[int, tuple[int, ...]]
] = {}
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASS_CAPACITY_GIB: dict[str, float] = {}
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ACTIVE_PLAN_CLASS: str | None = None
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_INITIAL_PLAN_CLASS: str | None = None
_CPU_OFFLOAD_UVM_STATIC_ALIAS_ENVELOPE_BYTES = 0
_CPU_OFFLOAD_UVM_PLAN_CLASS_SWITCH_HISTORY: list[dict[str, Any]] = []
_CPU_OFFLOAD_UVM_PLAN_CLASS_SOURCE_PREFILL_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_PLAN_CLASS_INSTRUCTIONS: dict[
    tuple[str, str, tuple[tuple[int, tuple[int, ...]], ...]], dict[str, Any]
] = {}
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS: dict[
    tuple[str, str, int], "_GraphStaticAliasArena"
] = {}
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES = 0
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS = 0
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE = False
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND = "vmm"
_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_LOCK = threading.RLock()
_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS: dict[
    tuple[str, str, int], "_UvmOriginalAliasArena"
] = {}
_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS: dict[str, "_UvmOriginalAliasBinding"] = {}
_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID = 0
_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "disabled"
_CPU_OFFLOAD_UVM_STATIC_ALIAS_PREALLOCATE_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_STATIC_ALIAS_MAP_RESULTS: list[dict[str, Any]] = []
_CPU_OFFLOAD_UVM_STATIC_ALIAS_SEAL_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_STATIC_ALIAS_ORIGINAL_RELEASE_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_STATIC_ALIAS_COLD_RESIDENCY_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_RESIDENCY_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_CUDA_READ_BASELINE: dict[
    tuple[int, tuple[tuple[str, tuple[int, ...]], ...]],
    tuple[str, dict[str, tuple[int, int]]],
] = {}
_CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED = False
_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE = "off"
_CPU_OFFLOAD_UVM_ORIGINAL_VA_TOKEN_THRESHOLD = 64
_CPU_OFFLOAD_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD = 65
_CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED = False
_CPU_OFFLOAD_UVM_STATIC_ALIAS_PARK_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_STATIC_ALIAS_RESUME_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY: str | None = None
_CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT = 0
_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED = False
_CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED = False
_CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED = False
_CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT = 0
_CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT = 0
_CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES = 0
_CPU_OFFLOAD_UVM_CAPTURE_VA_SNAPSHOT: tuple[int, ...] = ()
_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_GRAPH_CERTIFICATES_BY_CLASS: dict[str, dict[str, Any]] = {}
_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
_CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT: dict[str, Any] | None = None
_CPU_OFFLOAD_UVM_REMOTE_MAP_HISTORY: list[dict[str, Any]] = []
_CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE = False
_CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY: list[dict[str, Any]] = []
_CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT: tuple[int, ...] = ()
_CPU_OFFLOAD_UVM_PLAN_GENERATION = 0
_CPU_OFFLOAD_UVM_PLAN_REBUILD_HISTORY: list[dict[str, Any]] = []
_CPU_OFFLOAD_UVM_PREFILL_LAYER_TRACE_STATE: dict[int, dict[str, Any]] = {}
_CPU_OFFLOAD_UVM_STATIC_ALIAS_LOCK = threading.RLock()
_CPU_OFFLOAD_OWNER_THREAD_ID: int | None = None
_CPU_OFFLOAD_OWNER_DEVICE_INDEX: int | None = None
_CPU_OFFLOAD_OWNER_STREAM_ID: int | None = None


@dataclass(frozen=True)
class _GraphStaticAliasNamePlan:
    layer_container: str
    layer_index: int
    local_name: str
    bank_id: int
    consumer_index: int
    consumer_count: int


@dataclass
class _GraphStaticAliasArena:
    allocation: Any
    upload: torch.Tensor
    signature: tuple[Any, ...]
    requested_bytes: int
    aligned_bytes: int
    backing_bytes: int
    bindings: list["_GraphStaticAliasBinding"] = field(default_factory=list)


@dataclass(frozen=True)
class _GraphStaticAliasBinding:
    arena: _GraphStaticAliasArena
    consumer: torch.Tensor
    plan: _GraphStaticAliasNamePlan


@dataclass
class _UvmOriginalAliasArena:
    upload_storage: torch.Tensor
    upload: torch.Tensor
    signature: tuple[Any, ...]
    requested_bytes: int
    aligned_bytes: int
    backing_layer_id: int
    pte_alignment: int
    pte_flags: int
    region_bytes: int
    packed_slice_bytes: int = 0
    envelope_expert_count: int = 0
    consumers: dict[int, "_UvmOriginalAliasBinding"] = field(default_factory=dict)


@dataclass
class _UvmOriginalAliasBinding:
    full_parameter_name: str
    arena: _UvmOriginalAliasArena
    consumer: torch.Tensor
    parameter: Any
    plan: _GraphStaticAliasNamePlan
    requested_bytes: int
    parameter_bytes: int = 0
    expert_ids: tuple[int, ...] | None = None
    expert_stride_bytes: int = 0
    consumer_ranges: tuple[tuple[int, int], ...] = ()
    source: torch.Tensor | None = None
    source_slot_expert_ids: list[int | None] = field(default_factory=list)
    source_refill_required: bool = True


def _parse_graph_static_alias_name(
    full_parameter_name: str,
) -> tuple[str, int, str]:
    parts = full_parameter_name.split(".")
    candidates = [
        index
        for index in range(len(parts) - 2)
        if parts[index] == "layers" and parts[index + 1].isdigit()
    ]
    if not candidates:
        raise ValueError(
            "Graph static-alias parameter names must contain "
            f"'.layers.<integer>.': {full_parameter_name!r}"
        )
    layer_marker = candidates[-1]
    local_parts = parts[layer_marker + 2 :]
    if not local_parts:
        raise ValueError(
            "Graph static-alias parameter name has no layer-local suffix: "
            f"{full_parameter_name!r}"
        )
    return (
        ".".join(parts[: layer_marker + 1]),
        int(parts[layer_marker + 1]),
        ".".join(local_parts),
    )


def _build_graph_static_alias_plan(
    exact_names: frozenset[str],
) -> dict[str, _GraphStaticAliasNamePlan]:
    grouped: dict[tuple[str, str, int], list[tuple[int, str]]] = {}
    parsed: dict[str, tuple[str, int, str]] = {}
    for full_name in exact_names:
        layer_container, layer_index, local_name = _parse_graph_static_alias_name(
            full_name
        )
        parsed[full_name] = (layer_container, layer_index, local_name)
        bank_id = layer_index & 1
        grouped.setdefault((layer_container, local_name, bank_id), []).append(
            (layer_index, full_name)
        )

    result: dict[str, _GraphStaticAliasNamePlan] = {}
    for (layer_container, local_name, bank_id), members in grouped.items():
        members.sort()
        layer_indices = [layer_index for layer_index, _ in members]
        if len(layer_indices) != len(set(layer_indices)):
            raise ValueError(
                "Graph static-alias exact-name plan contains more than one "
                f"{local_name!r} parameter for the same layer and bank"
            )
        consumer_count = len(members)
        for consumer_index, (layer_index, full_name) in enumerate(members):
            result[full_name] = _GraphStaticAliasNamePlan(
                layer_container=layer_container,
                layer_index=layer_index,
                local_name=local_name,
                bank_id=bank_id,
                consumer_index=consumer_index,
                consumer_count=consumer_count,
            )
    assert result.keys() == parsed.keys()
    return result


def _cpu_offload_manifest_file() -> str:
    return os.environ.get("VLLM_CPU_OFFLOAD_MANIFEST_FILE", "")


def _append_cpu_offload_manifest(record: dict[str, Any]) -> None:
    manifest_file = _cpu_offload_manifest_file()
    if not manifest_file:
        return
    try:
        os.makedirs(os.path.dirname(manifest_file) or ".", exist_ok=True)
        with open(manifest_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError as exc:
        logger.warning(
            "Failed to append vLLM CPU offload manifest %s: %s",
            manifest_file,
            exc,
        )


def _load_cpu_offload_exact_names(exact_names_file: str) -> frozenset[str]:
    try:
        with open(exact_names_file, encoding="utf-8") as handle:
            contents = handle.read()
    except OSError as exc:
        raise ValueError(
            "Unable to read VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE "
            f"{exact_names_file!r}: {exc}"
        ) from exc

    stripped = contents.strip()
    if not stripped:
        raise ValueError(
            "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE must contain at least one "
            "full parameter name"
        )

    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE starts with '[' but is "
                f"not a valid JSON string list: {exc}"
            ) from exc
        if not isinstance(parsed, list) or not all(
            isinstance(name, str) for name in parsed
        ):
            raise ValueError(
                "JSON VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE must be a list of strings"
            )
        names = parsed
    else:
        names = [line.strip() for line in contents.splitlines() if line.strip()]

    if not names or any(not name.strip() for name in names):
        raise ValueError(
            "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE contains an empty parameter name"
        )
    if any(name != name.strip() for name in names):
        raise ValueError(
            "JSON parameter names in VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE must not "
            "have leading or trailing whitespace"
        )
    if len(names) != len(set(names)):
        raise ValueError(
            "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE contains duplicate parameter names"
        )
    return frozenset(names)


def _load_graph_static_alias_expert_plan(
    path: str,
    name_plan: Mapping[str, _GraphStaticAliasNamePlan],
) -> dict[int, tuple[int, ...]]:
    """Load one frozen, equal-width expert slice plan.

    The exact-name file still selects complete ``w13_weight``/``w2_weight``
    Parameters so those allocations are born in managed storage.  This
    optional plan narrows the static aliases and H2D traffic to expert slices
    inside those Parameters.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Unable to load "
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE "
            f"{path!r}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Static-alias expert plan must be a JSON object")

    selected: dict[int, list[int]] = {}
    selected_by_layer = payload.get("selected_by_layer")
    if selected_by_layer is not None:
        if not isinstance(selected_by_layer, dict):
            raise ValueError("selected_by_layer must be a JSON object")
        for raw_layer, raw_experts in selected_by_layer.items():
            try:
                layer_id = int(raw_layer)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid selected_by_layer key {raw_layer!r}"
                ) from exc
            if isinstance(raw_experts, (str, bytes)) or not isinstance(
                raw_experts, list
            ):
                raise ValueError(
                    f"selected_by_layer[{raw_layer!r}] must be a list"
                )
            selected[layer_id] = list(raw_experts)
    else:
        records = payload.get("offload_plan")
        if not isinstance(records, list):
            raise ValueError(
                "Static-alias expert plan requires selected_by_layer or "
                "offload_plan"
            )
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"offload_plan[{index}] is not an object")
            layer_id = record.get("layer_id")
            expert_id = record.get("expert_id")
            if (
                not isinstance(layer_id, int)
                or isinstance(layer_id, bool)
                or not isinstance(expert_id, int)
                or isinstance(expert_id, bool)
            ):
                raise ValueError(
                    f"offload_plan[{index}] requires integer layer_id/expert_id"
                )
            selected.setdefault(int(layer_id), []).append(int(expert_id))

    expected_layers = {entry.layer_index for entry in name_plan.values()}
    if set(selected) != expected_layers:
        missing = sorted(expected_layers - set(selected))
        unexpected = sorted(set(selected) - expected_layers)
        raise ValueError(
            "Static-alias expert plan layers differ from the exact-name plan: "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    normalized: dict[int, tuple[int, ...]] = {}
    widths: set[int] = set()
    for layer_id, raw_experts in sorted(selected.items()):
        if any(
            not isinstance(expert_id, int)
            or isinstance(expert_id, bool)
            or expert_id < 0
            for expert_id in raw_experts
        ):
            raise ValueError(
                f"Layer {layer_id} contains an invalid expert id"
            )
        experts = tuple(sorted(int(value) for value in raw_experts))
        if not experts or len(experts) != len(set(experts)):
            raise ValueError(
                f"Layer {layer_id} expert ids must be non-empty and unique"
            )
        normalized[layer_id] = experts
        widths.add(len(experts))
    if len(widths) != 1:
        raise ValueError(
            "Every layer in a static-alias expert plan must select the same "
            f"number of experts, got widths={sorted(widths)!r}"
        )
    return normalized


def _normalize_uvm_plan_class_name(value: str) -> str:
    normalized = str(value).strip().lower()
    if not normalized or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
        for character in normalized
    ):
        raise ValueError(f"Invalid UVM expert plan class name: {value!r}")
    return normalized


def _uvm_plan_class_phase_contract_allowed(
    original_va_phase_mode: str,
    eager_plan_classes: bool,
) -> bool:
    if original_va_phase_mode == "prefill_graph_remote_decode":
        return True
    return eager_plan_classes and original_va_phase_mode in (
        "prefill_remote_decode",
        "token_threshold",
    )


def _load_graph_static_alias_plan_classes(
    path: str,
    name_plan: Mapping[str, _GraphStaticAliasNamePlan],
) -> tuple[
    dict[str, str],
    dict[str, dict[int, tuple[int, ...]]],
    dict[str, float],
    str,
]:
    """Load fixed plan classes whose maximum width defines one stable VA envelope."""
    manifest_path = os.path.realpath(str(path).strip())
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to load static-alias plan-class manifest {path!r}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("classes"), dict):
        raise ValueError("Static-alias plan-class manifest requires a classes object")
    allow_equal_expert_widths = payload.get("allow_equal_expert_widths", False)
    if type(allow_equal_expert_widths) is not bool:
        raise ValueError("allow_equal_expert_widths must be a boolean")
    raw_classes = payload["classes"]
    if not raw_classes:
        raise ValueError("Stable plan classes require at least one capacity class")

    base_dir = os.path.dirname(manifest_path)
    class_files: dict[str, str] = {}
    plans: dict[str, dict[int, tuple[int, ...]]] = {}
    capacities: dict[str, float] = {}
    for raw_name, raw_record in raw_classes.items():
        class_name = _normalize_uvm_plan_class_name(str(raw_name))
        if class_name in class_files:
            raise ValueError(f"Duplicate normalized plan class {class_name!r}")
        if isinstance(raw_record, str):
            raw_file = raw_record
            capacity_gib = None
        elif isinstance(raw_record, dict):
            raw_file = raw_record.get("plan_file")
            capacity_gib = raw_record.get("capacity_gib")
        else:
            raise ValueError(f"Plan class {raw_name!r} must be a path or object")
        if not isinstance(raw_file, str) or not raw_file.strip():
            raise ValueError(f"Plan class {raw_name!r} has no plan_file")
        class_path = raw_file.strip()
        if not os.path.isabs(class_path):
            class_path = os.path.join(base_dir, class_path)
        class_path = os.path.realpath(class_path)
        class_files[class_name] = class_path
        plans[class_name] = _load_graph_static_alias_expert_plan(
            class_path, name_plan
        )
        if capacity_gib is not None:
            if (
                not isinstance(capacity_gib, (int, float))
                or isinstance(capacity_gib, bool)
                or float(capacity_gib) <= 0
            ):
                raise ValueError(
                    f"Plan class {raw_name!r} capacity_gib must be positive"
                )
            capacities[class_name] = float(capacity_gib)

    widths = {
        class_name: len(next(iter(plan.values())))
        for class_name, plan in plans.items()
    }
    if (
        not allow_equal_expert_widths
        and len(set(widths.values())) != len(widths)
    ):
        raise ValueError(
            "Fixed plan classes must have distinct expert widths: "
            f"{widths!r}"
        )
    initial = _normalize_uvm_plan_class_name(
        str(payload.get("initial_class") or max(widths, key=widths.get))
    )
    if initial not in plans:
        raise ValueError(f"initial_class {initial!r} is not registered")
    return class_files, plans, capacities, initial


def cpu_offload_uvm_plan_class_registry() -> dict[str, Any]:
    """Return immutable fixed-class metadata used by worker startup/control."""
    return {
        "enabled": bool(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS),
        "manifest_file": _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE or None,
        "initial_class": _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_INITIAL_PLAN_CLASS,
        "active_class": _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ACTIVE_PLAN_CLASS,
        "classes": {
            class_name: {
                "plan_file": _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASS_FILES[
                    class_name
                ],
                "capacity_gib": (
                    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASS_CAPACITY_GIB.get(
                        class_name
                    )
                ),
                "experts_per_layer": len(next(iter(plan.values()))),
                "selected_expert_count": sum(len(values) for values in plan.values()),
            }
            for class_name, plan in sorted(
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS.items()
            )
        },
    }


def _configured_graph_static_alias_backend() -> str:
    raw = (
        os.environ.get("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND", "vmm")
        .strip()
        .lower()
        .replace("-", "_")
    )
    aliases = {
        "": "vmm",
        "vmm": "vmm",
        "cuda_vmm": "vmm",
        "uvm": "uvm_original_va",
        "original_va": "uvm_original_va",
        "uvm_original_va": "uvm_original_va",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND must be 'vmm' "
            f"or 'uvm_original_va', got {raw!r}"
        ) from exc


def _configured_uvm_original_va_phase_mode() -> str:
    raw = (
        os.environ.get("VLLM_UVM_ORIGINAL_VA_PHASE_MODE", "off")
        .strip()
        .lower()
        .replace("-", "_")
    )
    aliases = {
        "": "off",
        "off": "off",
        "disabled": "off",
        "prefill_remote_decode": "prefill_remote_decode",
        "alias_prefill_remote_decode": "prefill_remote_decode",
        "hybrid": "prefill_remote_decode",
        "token_threshold": "token_threshold",
        "token_adaptive": "token_threshold",
        "prefill_graph_remote_decode": "prefill_graph_remote_decode",
        "alias_graph_prefill_remote_graph_decode": (
            "prefill_graph_remote_decode"
        ),
        "dual_topology_graph": "prefill_graph_remote_decode",
        "remote_only": "remote_only",
        "remote": "remote_only",
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        raise ValueError(
            "VLLM_UVM_ORIGINAL_VA_PHASE_MODE must be 'off', "
            "'prefill_remote_decode', 'token_threshold', "
            "'prefill_graph_remote_decode', or "
            f"'remote_only', got {raw!r}"
        ) from exc


def _configured_uvm_original_va_token_threshold() -> int:
    raw = os.environ.get("VLLM_UVM_ORIGINAL_VA_TOKEN_THRESHOLD", "64").strip()
    try:
        threshold = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_UVM_ORIGINAL_VA_TOKEN_THRESHOLD must be a positive int, "
            f"got {raw!r}"
        ) from exc
    if threshold <= 0:
        raise ValueError(
            "VLLM_UVM_ORIGINAL_VA_TOKEN_THRESHOLD must be a positive int, "
            f"got {raw!r}"
        )
    return threshold


def _configured_uvm_original_va_resume_token_threshold(
    park_threshold: int,
) -> int:
    raw = os.environ.get(
        "VLLM_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD",
        str(int(park_threshold) + 1),
    ).strip()
    try:
        threshold = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD must be an int "
            "greater than VLLM_UVM_ORIGINAL_VA_TOKEN_THRESHOLD, "
            f"got {raw!r}"
        ) from exc
    if threshold <= int(park_threshold):
        raise ValueError(
            "VLLM_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD must be greater "
            "than VLLM_UVM_ORIGINAL_VA_TOKEN_THRESHOLD, got "
            f"park={park_threshold}, resume={threshold}"
        )
    return threshold


def _configure_cpu_offload_exact_names(max_bytes: int) -> None:
    global _CPU_OFFLOAD_EXACT_NAMES_FILE
    global _CPU_OFFLOAD_EXACT_NAMES, _CPU_OFFLOAD_EXACT_PENDING
    global _CPU_OFFLOAD_EXACT_SEEN, _CPU_OFFLOAD_STATIC_BANK
    global _CPU_OFFLOAD_STATIC_BANKS, _CPU_OFFLOAD_STATIC_BANK_BYTES
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASS_FILES
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASS_CAPACITY_GIB
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ACTIVE_PLAN_CLASS
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_INITIAL_PLAN_CLASS
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_ENVELOPE_BYTES
    global _CPU_OFFLOAD_UVM_PLAN_CLASS_SWITCH_HISTORY
    global _CPU_OFFLOAD_UVM_PLAN_CLASS_SOURCE_PREFILL_RESULT
    global _CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT
    global _CPU_OFFLOAD_UVM_PLAN_CLASS_INSTRUCTIONS
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_PREALLOCATE_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_MAP_RESULTS
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SEAL_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_ORIGINAL_RELEASE_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_COLD_RESIDENCY_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_RESIDENCY_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_CUDA_READ_BASELINE
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED
    global _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE
    global _CPU_OFFLOAD_UVM_ORIGINAL_VA_TOKEN_THRESHOLD
    global _CPU_OFFLOAD_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD
    global _CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_PARK_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_RESUME_RESULT
    global _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY
    global _CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT
    global _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT
    global _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES
    global _CPU_OFFLOAD_UVM_CAPTURE_VA_SNAPSHOT
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE
    global _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE
    global _CPU_OFFLOAD_UVM_GRAPH_CERTIFICATES_BY_CLASS
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_HISTORY
    global _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE
    global _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY
    global _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
    global _CPU_OFFLOAD_UVM_PLAN_GENERATION
    global _CPU_OFFLOAD_UVM_PLAN_REBUILD_HISTORY
    global _CPU_OFFLOAD_OWNER_THREAD_ID, _CPU_OFFLOAD_OWNER_DEVICE_INDEX
    global _CPU_OFFLOAD_OWNER_STREAM_ID

    if (
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID
        or _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED
        or (
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
            in (
                "finalizing",
                "sealed",
                "remote_parked",
                "poisoned",
                "cleanup_pending",
            )
        )
    ):
        raise RuntimeError(
            "Cannot reconfigure CPU offload while a UVM original-VA "
            "static-alias session or cleanup is active; destroy its Graphs "
            "and session first"
        )

    exact_names_file = os.environ.get("VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE", "").strip()
    static_bank = _strict_env_flag("VLLM_CPU_OFFLOAD_STATIC_BANK", False)
    graph_static_alias = _strict_env_flag("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", False)
    graph_static_alias_eager = _strict_env_flag(
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER", False
    )
    graph_static_alias_backend = _configured_graph_static_alias_backend()
    original_va_phase_mode = _configured_uvm_original_va_phase_mode()
    original_va_token_threshold = _configured_uvm_original_va_token_threshold()
    original_va_resume_token_threshold = (
        _configured_uvm_original_va_resume_token_threshold(
            original_va_token_threshold
        )
    )
    park_resume_enabled = _strict_env_flag(
        "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME", False
    )
    graph_static_alias_expert_plan_file = os.environ.get(
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE", ""
    ).strip()
    graph_static_alias_plan_classes_file = os.environ.get(
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE", ""
    ).strip()
    eager_plan_classes = _strict_env_flag(
        "VLLM_CPU_OFFLOAD_PLAN_CLASS_EAGER_ENABLE", False
    )
    remote_only_direct_forward = _strict_env_flag(
        "VLLM_UVM_REMOTE_ONLY_DIRECT_FORWARD", False
    )

    _CPU_OFFLOAD_EXACT_NAMES_FILE = ""
    _CPU_OFFLOAD_EXACT_NAMES = None
    _CPU_OFFLOAD_EXACT_PENDING = set()
    _CPU_OFFLOAD_EXACT_SEEN = set()
    _CPU_OFFLOAD_STATIC_BANK = False
    _CPU_OFFLOAD_STATIC_BANKS = {}
    _CPU_OFFLOAD_STATIC_BANK_BYTES = 0
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS = False
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN = {}
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE = ""
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER = {}
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE = ""
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASS_FILES = {}
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS = {}
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASS_CAPACITY_GIB = {}
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ACTIVE_PLAN_CLASS = None
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_INITIAL_PLAN_CLASS = None
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_ENVELOPE_BYTES = 0
    _CPU_OFFLOAD_UVM_PLAN_CLASS_SWITCH_HISTORY = []
    _CPU_OFFLOAD_UVM_PLAN_CLASS_SOURCE_PREFILL_RESULT = None
    _CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT = None
    _CPU_OFFLOAD_UVM_PLAN_CLASS_INSTRUCTIONS = {}
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS = {}
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES = 0
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS = 0
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE = False
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND = "vmm"
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS = {}
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS = {}
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID = 0
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "disabled"
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_PREALLOCATE_RESULT = None
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_MAP_RESULTS = []
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_SEAL_RESULT = None
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_ORIGINAL_RELEASE_RESULT = None
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_COLD_RESIDENCY_RESULT = None
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_RESIDENCY_RESULT = None
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_CUDA_READ_BASELINE = {}
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT = None
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT = None
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED = False
    _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE = "off"
    _CPU_OFFLOAD_UVM_ORIGINAL_VA_TOKEN_THRESHOLD = 64
    _CPU_OFFLOAD_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD = 65
    _CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED = False
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_PARK_RESULT = None
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_RESUME_RESULT = None
    _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY = None
    _CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT = 0
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED = False
    _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED = False
    _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED = False
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT = 0
    _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT = 0
    _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES = 0
    _CPU_OFFLOAD_UVM_CAPTURE_VA_SNAPSHOT = ()
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE = None
    _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE = None
    _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE = None
    _CPU_OFFLOAD_UVM_GRAPH_CERTIFICATES_BY_CLASS = {}
    _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
    _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT = None
    _CPU_OFFLOAD_UVM_REMOTE_MAP_HISTORY = []
    _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE = False
    _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY = []
    _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT = ()
    _CPU_OFFLOAD_UVM_PLAN_GENERATION = 0
    _CPU_OFFLOAD_UVM_PLAN_REBUILD_HISTORY = []
    _CPU_OFFLOAD_OWNER_THREAD_ID = None
    _CPU_OFFLOAD_OWNER_DEVICE_INDEX = None
    _CPU_OFFLOAD_OWNER_STREAM_ID = None

    if (static_bank or graph_static_alias) and not exact_names_file:
        raise ValueError(
            "Each of VLLM_CPU_OFFLOAD_STATIC_BANK=1 and "
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 requires "
            "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE"
        )
    if static_bank and graph_static_alias:
        raise ValueError(
            "VLLM_CPU_OFFLOAD_STATIC_BANK=1 and "
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 are mutually exclusive"
        )
    if graph_static_alias_eager and not graph_static_alias:
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER=1 requires "
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1"
        )
    if graph_static_alias_expert_plan_file and not (
        graph_static_alias and graph_static_alias_backend == "uvm_original_va"
    ):
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE requires "
            "the uvm_original_va graph static-alias backend"
        )
    if graph_static_alias_plan_classes_file and not (
        graph_static_alias
        and graph_static_alias_backend == "uvm_original_va"
        and _uvm_plan_class_phase_contract_allowed(
            original_va_phase_mode,
            eager_plan_classes,
        )
        and park_resume_enabled
    ):
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE requires "
            "uvm_original_va and PARK/RESUME, plus either "
            "prefill_graph_remote_decode or the explicit "
            "VLLM_CPU_OFFLOAD_PLAN_CLASS_EAGER_ENABLE=1 "
            "prefill_remote_decode/token_threshold contract"
        )
    if graph_static_alias_eager and original_va_phase_mode != "off":
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER=1 is only valid with "
            "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=off"
        )
    if remote_only_direct_forward and not (
        graph_static_alias
        and graph_static_alias_backend == "uvm_original_va"
        and original_va_phase_mode == "remote_only"
    ):
        raise ValueError(
            "VLLM_UVM_REMOTE_ONLY_DIRECT_FORWARD=1 requires the "
            "uvm_original_va graph static-alias backend and "
            "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=remote_only"
        )
    if not graph_static_alias and graph_static_alias_backend != "vmm":
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND=uvm_original_va "
            "requires VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1"
        )
    if original_va_phase_mode != "off" and not (
        graph_static_alias and graph_static_alias_backend == "uvm_original_va"
    ):
        raise ValueError(
            "VLLM_UVM_ORIGINAL_VA_PHASE_MODE requires the "
            "uvm_original_va graph static-alias backend"
        )
    if original_va_phase_mode != "off" and _strict_env_flag(
        "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ", False
    ):
        raise ValueError(
            "VLLM_UVM_ORIGINAL_VA_PHASE_MODE and the legacy "
            "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ path are mutually exclusive"
        )
    if original_va_phase_mode == "token_threshold" and not park_resume_enabled:
        raise ValueError(
            "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=token_threshold requires "
            "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1"
        )
    if park_resume_enabled and original_va_phase_mode not in (
        "prefill_remote_decode",
        "token_threshold",
        "prefill_graph_remote_decode",
    ):
        raise ValueError(
            "VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME=1 requires an "
            "alias-prefill/remote-decode or token-threshold phase mode"
        )
    if not exact_names_file:
        return
    exact_backend = _normalized_cpu_offload_backend()
    if exact_backend not in ("copy", "uva"):
        raise ValueError(
            "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE is supported only with "
            "VLLM_CPU_OFFLOAD_BACKEND=copy or uva"
        )
    if exact_backend == "uva" and (static_bank or graph_static_alias):
        raise ValueError(
            "Exact-name UVA offload does not support static-bank or "
            "graph-static-alias modes"
        )
    if max_bytes <= 0:
        raise ValueError(
            "Exact-name CPU offload requires a positive cpu_offload_gb budget"
        )

    exact_names = _load_cpu_offload_exact_names(exact_names_file)
    _CPU_OFFLOAD_EXACT_NAMES_FILE = exact_names_file
    _CPU_OFFLOAD_EXACT_NAMES = exact_names
    _CPU_OFFLOAD_EXACT_PENDING = set(exact_names)
    _CPU_OFFLOAD_STATIC_BANK = static_bank
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS = graph_static_alias
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND = graph_static_alias_backend
    _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE = original_va_phase_mode
    _CPU_OFFLOAD_UVM_ORIGINAL_VA_TOKEN_THRESHOLD = original_va_token_threshold
    _CPU_OFFLOAD_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD = (
        original_va_resume_token_threshold
    )
    _CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED = park_resume_enabled
    if graph_static_alias:
        if _strict_env_flag("VLLM_CPU_OFFLOAD_COPY_SYNC", False):
            raise ValueError(
                "VLLM_CPU_OFFLOAD_COPY_SYNC=1 is capture-unsafe with "
                "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1"
            )
        if graph_static_alias_backend == "uvm_original_va" and not _strict_env_flag(
            "VLLM_UVM_EXPERT_WEIGHTS_MANAGED", False
        ):
            raise ValueError(
                "The uvm_original_va graph static-alias backend requires "
                "VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1 so expert parameters are "
                "born in managed storage and retain their allocation VAs"
            )
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN = _build_graph_static_alias_plan(
            exact_names
        )
        if graph_static_alias_backend == "uvm_original_va":
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "collecting"
            if graph_static_alias_plan_classes_file:
                (
                    class_files,
                    class_plans,
                    class_capacities,
                    initial_class,
                ) = _load_graph_static_alias_plan_classes(
                    graph_static_alias_plan_classes_file,
                    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN,
                )
                if graph_static_alias_expert_plan_file:
                    requested_initial_path = os.path.realpath(
                        graph_static_alias_expert_plan_file
                    )
                    matching_classes = [
                        class_name
                        for class_name, class_path in class_files.items()
                        if class_path == requested_initial_path
                    ]
                    if len(matching_classes) != 1:
                        raise ValueError(
                            "The configured expert plan must identify exactly "
                            "one registered plan class"
                        )
                    initial_class = matching_classes[0]
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASSES_FILE = (
                    os.path.realpath(graph_static_alias_plan_classes_file)
                )
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASS_FILES = class_files
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS = class_plans
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASS_CAPACITY_GIB = (
                    class_capacities
                )
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_INITIAL_PLAN_CLASS = initial_class
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ACTIVE_PLAN_CLASS = initial_class
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE = class_files[
                    initial_class
                ]
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER = dict(
                    class_plans[initial_class]
                )
            elif graph_static_alias_expert_plan_file:
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE = (
                    graph_static_alias_expert_plan_file
                )
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER = (
                    _load_graph_static_alias_expert_plan(
                        graph_static_alias_expert_plan_file,
                        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN,
                    )
                )


def set_cpu_offload_max_bytes(max_bytes: int) -> None:
    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES, _CPU_OFFLOAD_ORDER
    _CPU_OFFLOAD_BYTES = 0
    _CPU_OFFLOAD_MAX_BYTES = max_bytes
    _CPU_OFFLOAD_ORDER = 0
    _configure_cpu_offload_exact_names(max_bytes)
    manifest_file = _cpu_offload_manifest_file()
    if manifest_file and os.environ.get("VLLM_CPU_OFFLOAD_MANIFEST_RESET", "1") == "1":
        try:
            os.makedirs(os.path.dirname(manifest_file) or ".", exist_ok=True)
            with open(manifest_file, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": "cpu_offload_config",
                            "timestamp": time.time(),
                            "pid": os.getpid(),
                            "max_offload_bytes": int(max_bytes),
                            "copy_exact_names": bool(
                                _CPU_OFFLOAD_EXACT_NAMES is not None
                            ),
                            "copy_exact_name_count": len(
                                _CPU_OFFLOAD_EXACT_NAMES or ()
                            ),
                            "copy_exact_names_file": (
                                _CPU_OFFLOAD_EXACT_NAMES_FILE or None
                            ),
                            "copy_static_bank": bool(_CPU_OFFLOAD_STATIC_BANK),
                            "graph_static_alias": bool(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS),
                            "graph_static_alias_backend": (
                                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND
                                if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS
                                else None
                            ),
                            "graph_static_alias_expert_plan_file": (
                                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE
                                or None
                            ),
                            "graph_static_alias_selected_expert_count": sum(
                                len(expert_ids)
                                for expert_ids in (
                                    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER.values()
                                )
                            ),
                            "uvm_original_va_phase_mode": (
                                _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE
                                if cpu_offload_uvm_original_va_static_alias_enabled()
                                else "off"
                            ),
                            "uvm_original_va_token_threshold": (
                                _CPU_OFFLOAD_UVM_ORIGINAL_VA_TOKEN_THRESHOLD
                                if cpu_offload_uvm_original_va_static_alias_enabled()
                                else None
                            ),
                            "uvm_original_va_resume_token_threshold": (
                                _CPU_OFFLOAD_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD
                                if cpu_offload_uvm_original_va_static_alias_enabled()
                                else None
                            ),
                            "uvm_token_threshold_fast_transition": (
                                _strict_env_flag(
                                    "VLLM_UVM_TOKEN_THRESHOLD_FAST_TRANSITION",
                                    False,
                                )
                                if cpu_offload_uvm_original_va_static_alias_enabled()
                                else False
                            ),
                            "graph_static_alias_bank_count": (
                                2 if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS else 0
                            ),
                            "copy_bank_bytes": 0,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        except OSError as exc:
            logger.warning(
                "Failed to initialize vLLM CPU offload manifest %s: %s",
                manifest_file,
                exc,
            )


def _cpu_offload_backend() -> str:
    return os.environ.get("VLLM_CPU_OFFLOAD_BACKEND", "uva").strip().lower()


def _normalized_cpu_offload_backend() -> str:
    backend = _cpu_offload_backend()
    if backend in ("", "native", "native_uva", "pinned_uva"):
        return "uva"
    if backend in ("copy_on_forward", "streaming_copy"):
        return "copy"
    return backend


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _strict_env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be 0 or 1, got {raw!r}")


def _copy_exact_names_enabled() -> bool:
    return _CPU_OFFLOAD_EXACT_NAMES is not None


def cpu_offload_graph_static_alias_enabled() -> bool:
    """Return whether either experimental graph static-alias path is selected."""
    return _CPU_OFFLOAD_GRAPH_STATIC_ALIAS


def cpu_offload_uvm_original_va_static_alias_enabled() -> bool:
    return bool(
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS
        and _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND == "uvm_original_va"
    )


def cpu_offload_uvm_original_va_phase_mode() -> str:
    """Return the frozen original-VA serving topology policy."""
    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return "off"
    return _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE


def cpu_offload_uvm_original_va_token_threshold() -> int:
    """Return the frozen per-step token threshold for topology selection."""
    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return 64
    return _CPU_OFFLOAD_UVM_ORIGINAL_VA_TOKEN_THRESHOLD


def cpu_offload_uvm_original_va_resume_token_threshold() -> int:
    """Return the token count at which a parked topology resumes copy."""
    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return 65
    return _CPU_OFFLOAD_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD


def cpu_offload_uvm_original_va_profile_all_remote_active() -> bool:
    """Return whether startup profiling temporarily uses all-remote experts."""
    return bool(
        cpu_offload_uvm_original_va_static_alias_enabled()
        and _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE
    )


def cpu_offload_uvm_original_va_park_resume_enabled() -> bool:
    """Return whether sealed aliases are parked instead of destroyed."""
    return bool(
        cpu_offload_uvm_original_va_static_alias_enabled()
        and _CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED
    )


def cpu_offload_uvm_original_va_static_alias_experts_for_layer(
    layer_id: int,
) -> frozenset[int]:
    """Return experts whose original VA is owned by the active alias plan.

    Runtime expert-residency records share the same managed Parameter
    allocations.  The driver forbids registering a second region that
    overlaps an already registered static-alias release range, so callers use
    this immutable ownership view to leave those experts under the alias
    session's registry instead of constructing duplicate envelope regions.
    """
    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return frozenset()
    return frozenset(
        int(expert_id)
        for expert_id in _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER.get(
            int(layer_id),
            (),
        )
    )


def use_uvm_expert_weight_allocation(
    prefix: str,
    *,
    managed_requested: bool,
) -> bool:
    """Select managed allocation only for complete original-VA expert pairs.

    The legacy UVM paths intentionally allocate every FusedMoE main weight
    from the local managed pool.  The original-VA static-alias path is more
    restrictive: only parameters present in its frozen exact-name plan may be
    managed, because unaliased managed expert weights cannot fault safely
    while automatic UVM servicing is disabled during CUDA Graph execution.
    """
    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return bool(managed_requested)

    if not managed_requested:
        raise RuntimeError(
            "VLLM_UVM_EXPERT_WEIGHTS_MANAGED changed after the UVM "
            "original-VA static-alias configuration was frozen"
        )
    if not prefix:
        raise ValueError(
            "UVM original-VA expert allocation requires a non-empty "
            "FusedMoE module prefix"
        )
    if _CPU_OFFLOAD_EXACT_NAMES is None:
        raise RuntimeError(
            "UVM original-VA expert allocation was queried before the "
            "exact-name configuration was initialized"
        )

    expected = {
        f"{prefix}.w13_weight",
        f"{prefix}.w2_weight",
    }
    descendant_prefix = f"{prefix}."
    selected = {
        name
        for name in _CPU_OFFLOAD_EXACT_NAMES
        if name.startswith(descendant_prefix)
    }
    if not selected:
        return False
    if selected != expected:
        missing = sorted(expected - selected)
        unexpected = sorted(selected - expected)
        raise ValueError(
            "UVM original-VA FusedMoE selection must contain exactly one "
            "complete w13_weight/w2_weight pair per selected layer; "
            f"prefix={prefix!r}, missing={missing!r}, unexpected={unexpected!r}"
        )
    return True


def get_cpu_offload_graph_static_alias_diagnostics() -> dict[str, Any]:
    """Return read-only process-local counters and immutable alias VAs."""
    arenas: list[dict[str, Any]] = []
    ordered_uvm_bindings = sorted(
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values(),
        key=lambda binding: binding.full_parameter_name,
    )
    original_vas = tuple(
        int(binding.consumer.data_ptr()) for binding in ordered_uvm_bindings
    )
    source_vas = tuple(
        int(binding.source.data_ptr()) if binding.source is not None else None
        for binding in ordered_uvm_bindings
    )
    for (layer_container, local_name, bank_id), arena in sorted(
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS.items()
    ):
        allocation = arena.allocation
        arenas.append(
            {
                "layer_container": layer_container,
                "local_name": local_name,
                "bank_id": int(bank_id),
                "token": int(allocation.token),
                "requested_bytes": int(arena.requested_bytes),
                "aligned_bytes": int(arena.aligned_bytes),
                "backing_bytes": int(arena.backing_bytes),
                "upload_va": int(allocation.upload_va),
                "consumer_vas": tuple(int(value) for value in allocation.consumer_vas),
            }
        )
    for (layer_container, local_name, bank_id), arena in sorted(
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.items()
    ):
        ordered_consumers = [binding for _, binding in sorted(arena.consumers.items())]
        arenas.append(
            {
                "layer_container": layer_container,
                "local_name": local_name,
                "bank_id": int(bank_id),
                "backing_layer_id": int(arena.backing_layer_id),
                "requested_bytes": int(arena.requested_bytes),
                "aligned_bytes": int(arena.aligned_bytes),
                "backing_bytes": int(arena.aligned_bytes),
                "upload_va": int(arena.upload.data_ptr()),
                "consumer_vas": tuple(
                    int(binding.consumer.data_ptr()) for binding in ordered_consumers
                ),
                "consumer_ranges": tuple(
                    _uvm_original_alias_binding_consumer_ranges(binding)
                    for binding in ordered_consumers
                ),
                "expert_ids": tuple(
                    binding.expert_ids for binding in ordered_consumers
                ),
                "expert_stride_bytes": tuple(
                    int(binding.expert_stride_bytes)
                    for binding in ordered_consumers
                ),
                "selected_bytes": tuple(
                    int(binding.requested_bytes) for binding in ordered_consumers
                ),
                "parameter_bytes": tuple(
                    int(binding.parameter_bytes) for binding in ordered_consumers
                ),
                "original_vas": tuple(
                    int(binding.consumer.data_ptr()) for binding in ordered_consumers
                ),
                "source_vas": tuple(
                    int(binding.source.data_ptr())
                    if binding.source is not None
                    else None
                    for binding in ordered_consumers
                ),
                "source_ready_count": sum(
                    binding.source is not None for binding in arena.consumers.values()
                ),
                "pte_alignment": int(arena.pte_alignment),
                "pte_mode": "64K" if arena.pte_flags else "4K",
                "region_bytes": int(arena.region_bytes),
                "envelope_expert_count": int(arena.envelope_expert_count),
                "envelope_bytes": int(arena.aligned_bytes),
                "active_backing_bytes": int(arena.requested_bytes),
            }
        )
    return {
        "enabled": bool(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS),
        "backend": (
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND
            if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS
            else None
        ),
        "lifecycle_state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
        "sealed": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "sealed",
        "park_resume_enabled": bool(_CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED),
        "parked": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "remote_parked",
        "session_id": int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
        "original_vas": original_vas,
        "source_vas": source_vas,
        "expert_plan_file": (
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE or None
        ),
        "plan_class_registry": cpu_offload_uvm_plan_class_registry(),
        "active_plan_class": _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ACTIVE_PLAN_CLASS,
        "initial_plan_class": _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_INITIAL_PLAN_CLASS,
        "stable_envelope_bytes": int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ENVELOPE_BYTES),
        "expert_plan": tuple(
            (int(layer_id), tuple(expert_ids))
            for layer_id, expert_ids in sorted(
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER.items()
            )
        ),
        "selected_expert_count": sum(
            len(expert_ids)
            for expert_ids in _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER.values()
        ),
        "selected_parameter_count": len(ordered_uvm_bindings),
        "selected_parameter_bytes_per_forward": sum(
            int(binding.requested_bytes) for binding in ordered_uvm_bindings
        ),
        "full_parameter_bytes": sum(
            int(binding.parameter_bytes) for binding in ordered_uvm_bindings
        ),
        "consumer_range_count": sum(
            len(_uvm_original_alias_binding_consumer_ranges(binding))
            for binding in ordered_uvm_bindings
        ),
        "hot_range_count": sum(
            len(_uvm_original_alias_binding_hot_ranges(binding))
            for binding in ordered_uvm_bindings
        ),
        "capture_phase": bool(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE),
        "forward_call_count": int(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS),
        "arena_count": len(arenas),
        "backing_bytes": int(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES),
        "owner_thread_id": _CPU_OFFLOAD_OWNER_THREAD_ID,
        "owner_device_index": _CPU_OFFLOAD_OWNER_DEVICE_INDEX,
        "last_python_stream_id": _CPU_OFFLOAD_OWNER_STREAM_ID,
        "arenas": arenas,
        "preallocate_result": _CPU_OFFLOAD_UVM_STATIC_ALIAS_PREALLOCATE_RESULT,
        "map_results": tuple(_CPU_OFFLOAD_UVM_STATIC_ALIAS_MAP_RESULTS),
        "seal_result": _CPU_OFFLOAD_UVM_STATIC_ALIAS_SEAL_RESULT,
        "original_release_result": (
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_ORIGINAL_RELEASE_RESULT
        ),
        "cold_residency_result": (
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_COLD_RESIDENCY_RESULT
        ),
        "hot_residency_result": (
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_RESIDENCY_RESULT
        ),
        "destroy_result": _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT,
        "park_result": _CPU_OFFLOAD_UVM_STATIC_ALIAS_PARK_RESULT,
        "resume_result": _CPU_OFFLOAD_UVM_STATIC_ALIAS_RESUME_RESULT,
        "slot_close_result": (_CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT),
        "cleanup_pending": (_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "cleanup_pending"),
        "cleanup_unresolved": bool(_CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED),
        "phase_mode": _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE,
        "token_threshold": int(_CPU_OFFLOAD_UVM_ORIGINAL_VA_TOKEN_THRESHOLD),
        "resume_token_threshold": int(
            _CPU_OFFLOAD_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD
        ),
        "capture_topology": _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY,
        "capture_success_count": int(_CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT),
        "remote_graph_captured": bool(_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED),
        "remote_graph_certificate": _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE,
        "alias_graph_captured": bool(_CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED),
        "alias_graphs_destroyed": bool(
            _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED
        ),
        "alias_graph_certificate": _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE,
        "dual_graph_partition_certificate": (
            _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE
        ),
        "remote_graph_remaining_count": int(
            _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT
        ),
        "capture_forward_count": int(_CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT),
        "capture_actual_h2d_bytes": int(
            _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES
        ),
        "remote_map_active": bool(_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE),
        "remote_map_result": _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT,
        "remote_map_history": tuple(_CPU_OFFLOAD_UVM_REMOTE_MAP_HISTORY),
        "profile_all_remote_active": bool(
            _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE
        ),
        "phase_transition_history": tuple(
            _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY
        ),
        "remote_transition_history": tuple(
            _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY
        ),
        "original_va_snapshot": _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT,
        "plan_generation": int(_CPU_OFFLOAD_UVM_PLAN_GENERATION),
        "plan_rebuild_history": tuple(_CPU_OFFLOAD_UVM_PLAN_REBUILD_HISTORY),
        "plan_class_switch_history": tuple(
            _CPU_OFFLOAD_UVM_PLAN_CLASS_SWITCH_HISTORY
        ),
        "plan_class_source_prefill": (
            _CPU_OFFLOAD_UVM_PLAN_CLASS_SOURCE_PREFILL_RESULT
        ),
        "plan_class_immutable_cpu_shadow": (
            _CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT
        ),
        "plan_class_instruction_registry": tuple(
            {
                "instruction_id": str(record["instruction_id"]),
                "old_class": str(key[0]),
                "new_class": str(key[1]),
                "cached_entry_count": int(
                    len(record["target_map_entries"])
                ),
                "added_cold_range_count": int(
                    len(record["residency_ranges"][0])
                ),
                "removed_cold_range_count": int(
                    len(record["residency_ranges"][1])
                ),
            }
            for key, record in sorted(
                _CPU_OFFLOAD_UVM_PLAN_CLASS_INSTRUCTIONS.items()
            )
        ),
        "graph_certificate_classes": tuple(
            sorted(_CPU_OFFLOAD_UVM_GRAPH_CERTIFICATES_BY_CLASS)
        ),
        "active_driver_backing_bytes": int(
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
            if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
            in ("sealed", "remote_parked")
            else 0
        ),
    }


def validate_cpu_offload_graph_static_alias_runtime(
    vllm_config: VllmConfig,
) -> None:
    """Fail closed on execution modes that can reuse one bank concurrently."""
    if not _CPU_OFFLOAD_GRAPH_STATIC_ALIAS:
        return

    parallel_config = vllm_config.parallel_config
    unsupported: list[str] = []
    if parallel_config.tensor_parallel_size != 1:
        unsupported.append(
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}"
        )
    if parallel_config.pipeline_parallel_size != 1:
        unsupported.append(
            f"pipeline_parallel_size={parallel_config.pipeline_parallel_size}"
        )
    if parallel_config.data_parallel_size != 1:
        unsupported.append(f"data_parallel_size={parallel_config.data_parallel_size}")
    if parallel_config.enable_dbo:
        unsupported.append("dual-batch overlap")

    cudagraph_mode = vllm_config.compilation_config.cudagraph_mode
    mode_name = getattr(cudagraph_mode, "name", str(cudagraph_mode))
    allowed_cudagraph_modes = {"FULL", "FULL_DECODE_ONLY", "FULL_SEPARATE"}
    if _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE != "off":
        allowed_cudagraph_modes.add("NONE")
    if _strict_env_flag("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER", False):
        allowed_cudagraph_modes.add("NONE")
    if mode_name not in allowed_cudagraph_modes:
        unsupported.append(f"cudagraph_mode={mode_name}")
    if (
        _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE
        == "prefill_graph_remote_decode"
        and mode_name != "FULL_SEPARATE"
    ):
        unsupported.append(
            "prefill_graph_remote_decode requires cudagraph_mode=FULL_SEPARATE"
        )
    if (
        _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE == "token_threshold"
        and mode_name != "NONE"
    ):
        unsupported.append("token_threshold requires cudagraph_mode=NONE")
    if (
        _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE == "token_threshold"
        and not _CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED
    ):
        unsupported.append("token_threshold requires PARK/RESUME")
    if (
        mode_name == "FULL_SEPARATE"
        and _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE
        != "prefill_graph_remote_decode"
    ):
        unsupported.append(
            "FULL_SEPARATE is reserved for prefill_graph_remote_decode"
        )
    compilation_mode = vllm_config.compilation_config.mode
    compilation_mode_name = getattr(compilation_mode, "name", str(compilation_mode))
    if compilation_mode_name != "NONE":
        unsupported.append(f"compilation_mode={compilation_mode_name}")
    if unsupported:
        raise RuntimeError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS=1 currently requires "
            "single-GPU, non-overlapping FULL/FULL_DECODE_ONLY/FULL_SEPARATE "
            "execution "
            "(or NONE for an explicit original-VA phase/eager-alias experiment) "
            "with CompilationMode.NONE; "
            "unsupported: " + ", ".join(unsupported)
        )


def set_cpu_offload_graph_static_alias_capture_phase(
    active: bool,
    *,
    expected_topology: str | None = None,
) -> None:
    """Mark vLLM's warmup/capture phase for audit records."""
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE
    if not _CPU_OFFLOAD_GRAPH_STATIC_ALIAS:
        return
    if active and cpu_offload_uvm_original_va_static_alias_enabled():
        actual_topology = (
            "remote"
            if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "remote_parked"
            else _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
        )
        if expected_topology is not None:
            required_state = expected_topology
        elif (
            _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE
            == "prefill_graph_remote_decode"
        ):
            required_state = actual_topology
            if required_state not in ("remote", "sealed"):
                raise RuntimeError(
                    "Dual-topology CUDA Graph capture requires either an "
                    "audited remote or sealed alias state; current state is "
                    f"{required_state!r}"
                )
        else:
            required_state = (
                "remote"
                if _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE
                in ("prefill_remote_decode", "remote_only")
                else "sealed"
            )
        if actual_topology != required_state:
            required_description = (
                "completely mapped and sealed"
                if required_state == "sealed"
                else "audited remote"
            )
            raise RuntimeError(
                "The UVM original-VA CUDA Graph capture topology must be "
                f"{required_description}; current state is "
                f"{_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE!r}"
            )
        if required_state == "remote" and not _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE:
            raise RuntimeError(
                "Remote CUDA Graph capture requires an audited active remote "
                "READ mapping for every original parameter VA"
            )
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE = bool(active)
    stream_id = int(torch.cuda.current_stream().cuda_stream)
    _append_cpu_offload_manifest(
        {
            "event": "cpu_offload_graph_static_alias_capture_phase",
            "timestamp": time.time(),
            "pid": os.getpid(),
            "active": bool(active),
            "stream_id": stream_id,
            "forward_call_count": int(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS),
            "backing_bytes": int(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES),
            "arena_count": (
                len(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS)
                + len(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS)
            ),
            "backend": _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND,
            "lifecycle_state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
            "sealed": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "sealed",
            "session_id": int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
        }
    )


@contextmanager
def cpu_offload_graph_static_alias_capture_context(
    *,
    expected_topology: str | None = None,
):
    global _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY
    global _CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED
    global _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT
    global _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES
    global _CPU_OFFLOAD_UVM_CAPTURE_VA_SNAPSHOT
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE

    if not _CPU_OFFLOAD_GRAPH_STATIC_ALIAS:
        yield
        return
    topology = (
        (
            "remote"
            if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "remote_parked"
            else _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
        )
        if cpu_offload_uvm_original_va_static_alias_enabled()
        else "static_alias"
    )
    if expected_topology is not None and topology != expected_topology:
        raise RuntimeError(
            "CUDA Graph capture topology differs from the requested phase: "
            f"expected={expected_topology!r}, actual={topology!r}"
        )
    _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT = 0
    _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES = 0
    _CPU_OFFLOAD_UVM_CAPTURE_VA_SNAPSHOT = (
        _uvm_original_parameter_va_snapshot()
        if topology in ("remote", "sealed")
        else ()
    )
    succeeded = False
    set_cpu_offload_graph_static_alias_capture_phase(
        True,
        expected_topology=expected_topology,
    )
    try:
        yield
        if topology == "remote":
            parameter_vas_after = _uvm_original_parameter_va_snapshot()
            if not _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE:
                raise RuntimeError(
                    "Remote mapping became inactive during CUDA Graph capture"
                )
            if _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT <= 0:
                raise RuntimeError(
                    "Remote CUDA Graph capture observed no selected expert "
                    "forward while the stream was capturing"
                )
            if _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES != 0:
                raise RuntimeError(
                    "Remote CUDA Graph capture submitted an H2D copy"
                )
            if parameter_vas_after != _CPU_OFFLOAD_UVM_CAPTURE_VA_SNAPSHOT:
                raise RuntimeError(
                    "Original Parameter VAs changed during remote Graph capture"
                )
            _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE = {
                "schema": "uvm_original_va_remote_graph_certificate_v1",
                "topology": "remote",
                "phase_mode": _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE,
                "parameter_vas": parameter_vas_after,
                "capture_forward_count": int(
                    _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT
                ),
                "actual_h2d_bytes": int(
                    _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES
                ),
                "remote_map_requested_bytes": int(
                    (_CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT or {}).get(
                        "requested_bytes", 0
                    )
                ),
            }
        elif topology == "sealed":
            parameter_vas_after = _uvm_original_parameter_va_snapshot()
            if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE != "sealed":
                raise RuntimeError(
                    "Alias topology changed during CUDA Graph capture"
                )
            if _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT <= 0:
                raise RuntimeError(
                    "Alias CUDA Graph capture observed no selected expert "
                    "forward while the stream was capturing"
                )
            if _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES <= 0:
                raise RuntimeError(
                    "Alias CUDA Graph capture did not record an H2D upload"
                )
            if parameter_vas_after != _CPU_OFFLOAD_UVM_CAPTURE_VA_SNAPSHOT:
                raise RuntimeError(
                    "Original Parameter VAs changed during alias Graph capture"
                )
            _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE = {
                "schema": "uvm_original_va_alias_graph_certificate_v1",
                "topology": "sealed",
                "phase_mode": _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE,
                "parameter_vas": parameter_vas_after,
                "capture_forward_count": int(
                    _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT
                ),
                "actual_h2d_bytes": int(
                    _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES
                ),
                "session_id": int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
                "backing_bytes": int(
                    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
                ),
            }
        succeeded = True
    finally:
        set_cpu_offload_graph_static_alias_capture_phase(False)
        if succeeded:
            _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY = topology
            _CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT += 1
            if topology == "remote":
                _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED = True
            elif topology == "sealed" and (
                cpu_offload_uvm_original_va_static_alias_enabled()
            ):
                _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED = True
                _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED = False
            _append_cpu_offload_manifest(
                {
                    "event": "cpu_offload_graph_static_alias_capture_succeeded",
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "topology": topology,
                    "capture_success_count": int(
                        _CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT
                    ),
                    "remote_graph_captured": bool(
                        _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED
                    ),
                    "alias_graph_captured": bool(
                        _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED
                    ),
                }
            )


def _validate_copy_exact_context(
    device: torch.device,
    *,
    check_stream: bool,
) -> None:
    global _CPU_OFFLOAD_OWNER_THREAD_ID, _CPU_OFFLOAD_OWNER_DEVICE_INDEX
    global _CPU_OFFLOAD_OWNER_STREAM_ID

    if getattr(device, "type", None) != "cuda":
        raise RuntimeError(
            "Exact-name CPU offload requires parameters on CUDA device 0"
        )
    device_count = int(torch.cuda.device_count())
    if device_count != 1:
        raise RuntimeError(
            "Exact-name CPU offload is fail-closed for a single GPU; "
            f"torch.cuda.device_count() returned {device_count}"
        )
    device_index = getattr(device, "index", None)
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    if int(device_index) != 0 or int(torch.cuda.current_device()) != 0:
        raise RuntimeError("Exact-name CPU offload supports only current CUDA device 0")

    thread_id = threading.get_ident()
    if _CPU_OFFLOAD_OWNER_THREAD_ID is None:
        _CPU_OFFLOAD_OWNER_THREAD_ID = thread_id
        _CPU_OFFLOAD_OWNER_DEVICE_INDEX = int(device_index)
    elif (
        thread_id != _CPU_OFFLOAD_OWNER_THREAD_ID
        or int(device_index) != _CPU_OFFLOAD_OWNER_DEVICE_INDEX
    ):
        raise RuntimeError(
            "Exact-name CPU offload must be constructed and executed on one "
            "thread and CUDA device"
        )

    if not check_stream:
        return
    stream = torch.cuda.current_stream(device)
    cuda_stream = getattr(stream, "cuda_stream", None)
    if cuda_stream is None:
        raise RuntimeError(
            "Exact-name CPU offload cannot identify the current CUDA stream"
        )
    stream_id = int(cuda_stream)
    if _CPU_OFFLOAD_OWNER_STREAM_ID is None:
        _CPU_OFFLOAD_OWNER_STREAM_ID = stream_id
    elif _CPU_OFFLOAD_GRAPH_STATIC_ALIAS:
        # vLLM warms up on its graph-capture stream, while torch.cuda.graph
        # temporarily uses an internal capture stream. Both transitions are
        # serialized by vLLM. Concurrent execution is rejected separately by
        # validate_cpu_offload_graph_static_alias_runtime().
        _CPU_OFFLOAD_OWNER_STREAM_ID = stream_id
    elif stream_id != _CPU_OFFLOAD_OWNER_STREAM_ID:
        raise RuntimeError(
            "Exact-name CPU offload static-bank/copy execution must remain on "
            "one CUDA stream"
        )


def _create_graph_static_alias_binding(
    *,
    full_parameter_name: str,
    tensor: torch.Tensor,
    device: torch.device,
    requested_bytes: int,
) -> _GraphStaticAliasBinding:
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES

    plan = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN.get(full_parameter_name)
    if plan is None:
        raise RuntimeError(
            "Graph static-alias parameter is missing from the precomputed "
            f"exact-name plan: {full_parameter_name!r}"
        )
    is_contiguous = getattr(tensor, "is_contiguous", None)
    if is_contiguous is not None and not bool(is_contiguous()):
        raise RuntimeError(
            "Graph static-alias currently requires contiguous parameters: "
            f"{full_parameter_name!r}"
        )

    device_index = device.index
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    shape = tuple(int(value) for value in tensor.size())
    strides = tuple(int(value) for value in tensor.stride())
    signature = (
        shape,
        strides,
        tensor.dtype,
        int(device_index),
        int(requested_bytes),
        int(plan.consumer_count),
    )
    arena_key = (plan.layer_container, plan.local_name, plan.bank_id)
    arena = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS.get(arena_key)
    if arena is None:
        from vllm.device_allocator.cumem import StaticAliasAllocation

        allocation = StaticAliasAllocation.create(
            device=int(device_index),
            size=int(requested_bytes),
            consumer_count=int(plan.consumer_count),
        )
        upload = allocation.upload_tensor(
            shape,
            tensor.dtype,
            strides=strides,
        )
        arena = _GraphStaticAliasArena(
            allocation=allocation,
            upload=upload,
            signature=signature,
            requested_bytes=int(allocation.requested_size),
            aligned_bytes=int(allocation.aligned_size),
            backing_bytes=int(allocation.backing_bytes),
        )
        if int(upload.data_ptr()) != int(allocation.upload_va):
            raise RuntimeError(
                "Graph static-alias upload tensor VA disagrees with the "
                f"allocator for {full_parameter_name!r}"
            )
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS[arena_key] = arena
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES += arena.backing_bytes
    elif arena.signature != signature:
        raise RuntimeError(
            "Graph static-alias layers sharing one bank/local key have "
            f"different tensor signatures for {full_parameter_name!r}: "
            f"expected {arena.signature!r}, got {signature!r}"
        )

    allocation = arena.allocation
    consumer = allocation.consumer_tensor(
        int(plan.consumer_index),
        shape,
        tensor.dtype,
        strides=strides,
    )
    expected_consumer_va = int(allocation.consumer_vas[plan.consumer_index])
    if int(consumer.data_ptr()) != expected_consumer_va:
        raise RuntimeError(
            "Graph static-alias consumer tensor VA disagrees with the "
            f"allocator for {full_parameter_name!r}"
        )
    if int(consumer.data_ptr()) == int(arena.upload.data_ptr()):
        raise RuntimeError(
            "Graph static-alias upload and consumer mappings unexpectedly "
            f"share one VA for {full_parameter_name!r}"
        )
    binding = _GraphStaticAliasBinding(
        arena=arena,
        consumer=consumer,
        plan=plan,
    )
    arena.bindings.append(binding)
    return binding


_UVM_ORIGINAL_ALIAS_PAGE_BYTES = 4 * 1024
_UVM_ORIGINAL_ALIAS_64K_PTE_BYTES = 64 * 1024
_UVM_ORIGINAL_ALIAS_64K_ENVELOPE_BYTES = 2 * 1024 * 1024
_UVM_ORIGINAL_ALIAS_DEFAULT_REGION_BYTES = 512 * 1024
_UVM_ORIGINAL_ALIAS_MAX_REGION_BYTES = 4 * 1024 * 1024
_UVM_ORIGINAL_ALIAS_BACKING_LAYER_BASE = 0x70000000


def _uvm_original_alias_region_bytes() -> int:
    raw = os.environ.get(
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES",
        str(_UVM_ORIGINAL_ALIAS_DEFAULT_REGION_BYTES),
    )
    try:
        value = int(raw, 0)
    except ValueError as exc:
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES must be an "
            f"integer, got {raw!r}"
        ) from exc
    if (
        value <= 0
        or value > _UVM_ORIGINAL_ALIAS_MAX_REGION_BYTES
        or value % _UVM_ORIGINAL_ALIAS_PAGE_BYTES
    ):
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES must be a "
            "positive 4 KiB multiple no larger than 4 MiB"
        )
    return value


def _uvm_original_alias_requested_pte_mode() -> str:
    raw = (
        os.environ.get(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE",
            os.environ.get("UVM_STATIC_ALIAS_PTE_MODE", "auto"),
        )
        .strip()
        .lower()
    )
    if raw not in ("auto", "4k", "64k"):
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE must be auto, 4k, "
            f"or 64k, got {raw!r}"
        )
    return raw


def _uvm_original_alias_backing_layer_id(arena_index: int) -> int:
    raw = os.environ.get(
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_LAYER_ID_BASE",
        str(_UVM_ORIGINAL_ALIAS_BACKING_LAYER_BASE),
    )
    try:
        base = int(raw, 0)
    except ValueError as exc:
        raise ValueError(
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_LAYER_ID_BASE must be an "
            f"integer, got {raw!r}"
        ) from exc
    result = base + int(arena_index)
    if result < 0 or result > (1 << 32) - 1:
        raise ValueError("UVM original-VA backing layer id exceeds uint32")
    return result


def _allocate_uvm_original_alias_upload(
    *,
    tensor: torch.Tensor,
    device: torch.device,
    requested_bytes: int,
    upload_size: tuple[int, ...] | None = None,
    upload_stride: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate one 2 MiB-aligned managed upload VA without replacing params."""
    from vllm.device_allocator.uvm import uvm_local_memory_pool

    if requested_bytes <= 0 or requested_bytes % _UVM_ORIGINAL_ALIAS_PAGE_BYTES:
        raise RuntimeError(
            "UVM original-VA static alias requires each selected parameter "
            "size to be a positive 4 KiB multiple"
        )
    allocation_bytes = int(requested_bytes) + _UVM_ORIGINAL_ALIAS_64K_ENVELOPE_BYTES
    with torch.cuda.device(device), uvm_local_memory_pool(enable_prefetch=False):
        storage = torch.empty(
            allocation_bytes,
            dtype=torch.uint8,
            device=device,
        )
    storage_va = int(storage.data_ptr())
    alignment = _UVM_ORIGINAL_ALIAS_64K_ENVELOPE_BYTES
    upload_va = (storage_va + alignment - 1) & ~(alignment - 1)
    upload_offset = upload_va - storage_va
    byte_view = storage[upload_offset : upload_offset + requested_bytes]
    if int(byte_view.data_ptr()) != upload_va or byte_view.numel() != requested_bytes:
        raise RuntimeError("Unable to construct a complete aligned managed upload VA")
    typed_storage = byte_view.view(tensor.dtype)
    size = upload_size or tuple(int(value) for value in tensor.size())
    stride = upload_stride or tuple(int(value) for value in tensor.stride())
    upload = torch.as_strided(
        typed_storage,
        size=size,
        stride=stride,
    )
    if int(upload.data_ptr()) != upload_va:
        raise RuntimeError("Managed upload tensor changed the aligned upload VA")
    return storage, upload


def _uvm_original_alias_parameter_geometry(
    *,
    full_parameter_name: str,
    tensor: torch.Tensor,
    parameter_bytes: int,
    experts_by_layer: Mapping[int, tuple[int, ...]] | None = None,
) -> tuple[
    tuple[int, ...] | None,
    int,
    int,
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, int], ...],
]:
    """Return the packed-source geometry for one selected Parameter.

    A missing expert plan retains the original whole-Parameter behavior.  An
    expert plan packs the selected leading-dimension slices while retaining
    their discontiguous consumer VAs in the original managed allocation.
    """
    plan = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN.get(full_parameter_name)
    if plan is None:
        raise RuntimeError(
            "UVM original-VA parameter is missing from the precomputed plan: "
            f"{full_parameter_name!r}"
        )
    shape = tuple(int(value) for value in tensor.size())
    strides = tuple(int(value) for value in tensor.stride())
    consumer_va = int(tensor.data_ptr())
    selected_experts = (
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER
        if experts_by_layer is None
        else experts_by_layer
    )
    expert_ids = selected_experts.get(plan.layer_index)
    if expert_ids is None:
        return (
            None,
            0,
            int(parameter_bytes),
            shape,
            strides,
            ((consumer_va, int(parameter_bytes)),),
        )

    if not shape or shape[0] <= 0:
        raise RuntimeError(
            "Expert-slice static alias requires an expert-leading tensor: "
            f"{full_parameter_name!r} has shape={shape!r}"
        )
    if expert_ids[-1] >= shape[0]:
        raise RuntimeError(
            f"Expert id {expert_ids[-1]} exceeds the leading dimension "
            f"of {full_parameter_name!r}: {shape[0]}"
        )
    if int(tensor.numel()) % shape[0]:
        raise RuntimeError(
            "Expert-slice tensor has a non-integral leading-dimension payload: "
            f"{full_parameter_name!r}"
        )
    expert_stride_bytes = int(strides[0]) * int(tensor.element_size())
    expert_payload_bytes = (
        int(tensor.numel()) // shape[0] * int(tensor.element_size())
    )
    if expert_stride_bytes != expert_payload_bytes:
        raise RuntimeError(
            "Expert-slice static alias requires densely packed expert slices "
            f"for {full_parameter_name!r}: stride_bytes="
            f"{expert_stride_bytes}, payload_bytes={expert_payload_bytes}"
        )
    if expert_stride_bytes % _UVM_ORIGINAL_ALIAS_PAGE_BYTES:
        raise RuntimeError(
            f"Expert stride for {full_parameter_name!r} is not 4 KiB aligned"
        )
    if int(parameter_bytes) != shape[0] * expert_stride_bytes:
        raise RuntimeError(
            "Expert-slice Parameter byte size does not match its dense expert "
            f"geometry for {full_parameter_name!r}"
        )

    selected_bytes = len(expert_ids) * expert_stride_bytes
    upload_shape = (len(expert_ids), *shape[1:])
    consumer_ranges = tuple(
        (consumer_va + expert_id * expert_stride_bytes, expert_stride_bytes)
        for expert_id in expert_ids
    )
    return (
        expert_ids,
        expert_stride_bytes,
        selected_bytes,
        upload_shape,
        strides,
        consumer_ranges,
    )


def _uvm_original_alias_envelope_expert_count(
    full_parameter_name: str,
    active_expert_ids: tuple[int, ...] | None,
) -> int:
    if not _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS:
        return len(active_expert_ids) if active_expert_ids is not None else 0
    plan = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN[full_parameter_name]
    counts = [
        len(experts_by_layer[plan.layer_index])
        for experts_by_layer in (
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS.values()
        )
    ]
    envelope_count = max(counts)
    if active_expert_ids is None or len(active_expert_ids) > envelope_count:
        raise RuntimeError(
            f"Invalid stable envelope for {full_parameter_name!r}: "
            f"active={active_expert_ids!r}, envelope_count={envelope_count}"
        )
    return envelope_count


def _uvm_original_alias_binding_upload_view(
    binding: _UvmOriginalAliasBinding,
) -> torch.Tensor:
    if binding.expert_ids is None:
        return binding.arena.upload
    return binding.arena.upload.narrow(0, 0, len(binding.expert_ids))


def _uvm_original_alias_binding_source_view(
    binding: _UvmOriginalAliasBinding,
) -> torch.Tensor:
    if binding.source is None:
        raise RuntimeError(
            f"Pinned source is absent for {binding.full_parameter_name!r}"
        )
    if binding.expert_ids is None:
        return binding.source
    return binding.source.narrow(0, 0, len(binding.expert_ids))


def _refill_uvm_original_alias_binding_source(
    binding: _UvmOriginalAliasBinding,
) -> None:
    source = binding.source
    if source is None:
        raise RuntimeError(
            f"Pinned source is absent for {binding.full_parameter_name!r}"
        )
    current = binding.parameter.data
    if binding.expert_ids is None:
        source.copy_(current, non_blocking=True)
        binding.source_slot_expert_ids = []
        return
    slot_experts = list(getattr(binding, "source_slot_expert_ids", ()))
    source_slots = int(source.size(0))
    if len(slot_experts) != source_slots:
        slot_experts = [None] * source_slots
    for packed_index, expert_id in enumerate(binding.expert_ids):
        source[packed_index].copy_(current[expert_id], non_blocking=True)
        slot_experts[packed_index] = int(expert_id)
    binding.source_slot_expert_ids = slot_experts


def _prefill_uvm_original_alias_plan_class_sources() -> dict[str, Any]:
    """Populate inactive packed slots for a nested class ladder at startup.

    The source allocation already has the maximum registered class width.  A
    nested ladder can therefore keep every future cold expert in its eventual
    packed slot without allocating another host tensor.  Runtime class
    switches can then reuse these immutable copies instead of reading the same
    weights back from the GPU.
    """
    if not _strict_env_flag(
        "VLLM_CPU_OFFLOAD_PLAN_CLASS_PREFILL_SOURCES", False
    ):
        return {"enabled": False, "reason": "disabled"}
    classes = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS
    initial = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_INITIAL_PLAN_CLASS
    if not classes or initial not in classes:
        return {"enabled": False, "reason": "plan_classes_unavailable"}

    widths = {
        class_name: len(next(iter(plan.values())))
        for class_name, plan in classes.items()
    }
    ordered_classes = sorted(classes, key=lambda name: (widths[name], name))
    if ordered_classes[0] != initial:
        raise RuntimeError(
            "Plan-class source prefill currently requires the initial class "
            "to be the smallest nested tier"
        )

    initial_plan = classes[initial]
    canonical_by_layer: dict[int, tuple[int, ...]] = {}
    for layer_index, initial_ids in sorted(initial_plan.items()):
        canonical = [int(value) for value in initial_ids]
        previous = set(canonical)
        for class_name in ordered_classes[1:]:
            try:
                raw_ids = tuple(
                    int(value) for value in classes[class_name][layer_index]
                )
            except KeyError as exc:
                raise RuntimeError(
                    f"Plan class {class_name!r} omits layer {layer_index}"
                ) from exc
            current = set(raw_ids)
            if not previous < current:
                raise RuntimeError(
                    "Plan-class source prefill requires strictly nested cold "
                    f"tiers at layer {layer_index}: previous={sorted(previous)!r}, "
                    f"class={class_name!r}, current={sorted(current)!r}"
                )
            canonical.extend(value for value in raw_ids if value not in previous)
            previous = current
        canonical_by_layer[int(layer_index)] = tuple(canonical)

    copied_slots = 0
    copied_bytes = 0
    reused_slots = 0
    binding_count = 0
    for full_name, binding in sorted(
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
    ):
        if binding.expert_ids is None:
            continue
        if binding.source is None:
            raise RuntimeError(
                f"Pinned source is absent for plan-class prefill {full_name!r}"
            )
        target_ids = canonical_by_layer[binding.plan.layer_index]
        source_slots = int(binding.source.size(0))
        if len(target_ids) > source_slots:
            raise RuntimeError(
                "Nested plan-class union exceeds the stable source envelope: "
                f"name={full_name!r}, union={len(target_ids)}, "
                f"source_slots={source_slots}"
            )
        slot_experts = list(
            getattr(binding, "source_slot_expert_ids", ())
        )
        if len(slot_experts) != source_slots:
            slot_experts = [None] * source_slots
        current = binding.parameter.data
        binding_count += 1
        for slot_index, expert_id in enumerate(target_ids):
            if slot_experts[slot_index] == expert_id:
                reused_slots += 1
                continue
            binding.source[slot_index].copy_(
                current[int(expert_id)], non_blocking=True
            )
            slot_experts[slot_index] = int(expert_id)
            copied_slots += 1
            copied_bytes += int(binding.expert_stride_bytes)
        binding.source_slot_expert_ids = slot_experts

    return {
        "enabled": True,
        "reason": "nested_union_prefilled",
        "initial_class": str(initial),
        "ordered_classes": tuple(ordered_classes),
        "binding_count": int(binding_count),
        "copied_slot_count": int(copied_slots),
        "copied_bytes": int(copied_bytes),
        "reused_slot_count": int(reused_slots),
        "largest_experts_per_layer": int(max(widths.values())),
    }


def _uvm_original_alias_plan_class_shadow_records(
) -> tuple[tuple[str, int, int, int], ...]:
    """Return immutable ``(name, expert, VA, bytes)`` shadow geometry."""
    classes = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS
    initial = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_INITIAL_PLAN_CLASS
    if len(classes) < 2 or initial not in classes:
        raise RuntimeError("Immutable CPU shadows require multiple plan classes")

    initial_plan = classes[initial]
    future_by_layer: dict[int, tuple[int, ...]] = {}
    for layer_index, initial_ids in sorted(initial_plan.items()):
        initial_set = {int(value) for value in initial_ids}
        union = {
            int(value)
            for plan in classes.values()
            for value in plan[int(layer_index)]
        }
        future_by_layer[int(layer_index)] = tuple(sorted(union - initial_set))

    records: list[tuple[str, int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for full_name, binding in sorted(
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
    ):
        if binding.expert_ids is None or binding.expert_stride_bytes <= 0:
            raise RuntimeError(
                "Immutable CPU shadows require expert-sliced Parameter bindings"
            )
        stride = int(binding.expert_stride_bytes)
        parameter_base = int(binding.consumer.data_ptr())
        for expert_id in future_by_layer[binding.plan.layer_index]:
            geometry = (parameter_base + int(expert_id) * stride, stride)
            if (
                geometry[0] % (64 * 1024)
                or geometry[1] % (64 * 1024)
                or geometry in seen
            ):
                raise RuntimeError(
                    "Immutable CPU shadow geometry must be unique and 64 KiB "
                    f"aligned: name={full_name!r}, expert={expert_id}, "
                    f"range=0x{geometry[0]:x}+{geometry[1]}"
                )
            seen.add(geometry)
            records.append(
                (full_name, int(expert_id), geometry[0], geometry[1])
            )
    return tuple(records)


def _uvm_original_alias_expected_shadow_cpu_bytes(
    ranges: Iterable[tuple[int, int]],
    *,
    additional_cpu_ranges: Iterable[tuple[int, int]] = (),
) -> int:
    """Count bytes expected on CPU from shadows and explicit CPU ranges."""
    certificate = _CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT
    shadow_ranges: tuple[tuple[int, int], ...] = ()
    if isinstance(certificate, dict) and certificate.get("enabled") is True:
        if certificate.get("reason") == "read_duplicate_shadows_ready":
            shadow_records = _uvm_original_alias_plan_class_shadow_records()
            shadow_ranges = tuple(
                (base, length) for _, _, base, length in shadow_records
            )
            shadow_bytes = sum(length for _, length in shadow_ranges)
            if (
                int(certificate.get("range_count", -1)) != len(shadow_ranges)
                or int(certificate.get("requested_bytes", -1)) != shadow_bytes
            ):
                raise RuntimeError(
                    "Immutable CPU shadow certificate no longer matches "
                    "Parameter geometry: "
                    f"certificate={certificate!r}, "
                    f"ranges={len(shadow_ranges)}, bytes={shadow_bytes}"
                )
        elif certificate.get("reason") == (
            "full_parameter_read_duplicate_shadows_ready"
        ):
            shadow_ranges = tuple(
                (
                    int(binding.consumer.data_ptr()),
                    int(binding.consumer.numel() * binding.consumer.element_size()),
                )
                for _, binding in sorted(
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                )
            )
            shadow_bytes = sum(length for _, length in shadow_ranges)
            if (
                int(certificate.get("range_count", -1)) != len(shadow_ranges)
                or int(certificate.get("requested_bytes", -1)) != shadow_bytes
            ):
                raise RuntimeError(
                    "Full-Parameter immutable CPU shadow certificate no "
                    "longer matches Parameter geometry: "
                    f"certificate={certificate!r}, "
                    f"ranges={len(shadow_ranges)}, bytes={shadow_bytes}"
                )
        elif certificate.get("reason") != "no_future_cold_ranges":
            raise RuntimeError(
                f"Unknown immutable CPU shadow certificate: {certificate!r}"
            )

    coverage = [
        (int(base), int(length))
        for base, length in (*shadow_ranges, *tuple(additional_cpu_ranges))
        if int(length) > 0
    ]
    coverage.sort()
    merged_coverage: list[tuple[int, int]] = []
    for base, length in coverage:
        end = base + length
        if not merged_coverage:
            merged_coverage.append((base, length))
            continue
        previous_base, previous_length = merged_coverage[-1]
        previous_end = previous_base + previous_length
        if base > previous_end:
            merged_coverage.append((base, length))
        else:
            merged_coverage[-1] = (
                previous_base,
                max(previous_end, end) - previous_base,
            )

    overlap_bytes = 0
    for raw_base, raw_length in ranges:
        base = int(raw_base)
        end = base + int(raw_length)
        for covered_base, covered_length in merged_coverage:
            covered_end = covered_base + covered_length
            overlap_bytes += max(
                0,
                min(end, covered_end) - max(base, covered_base),
            )
    return overlap_bytes


def _uvm_original_alias_subtract_ranges(
    ranges: Iterable[tuple[int, int]],
    excluded_ranges: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Return ``ranges`` minus a normalized union of excluded byte ranges."""
    excluded = sorted(
        (int(base), int(length))
        for base, length in excluded_ranges
        if int(length) > 0
    )
    merged_excluded: list[tuple[int, int]] = []
    for base, length in excluded:
        end = base + length
        if base <= 0 or end <= base:
            raise ValueError("Excluded UVM range has invalid geometry")
        if not merged_excluded:
            merged_excluded.append((base, length))
            continue
        previous_base, previous_length = merged_excluded[-1]
        previous_end = previous_base + previous_length
        if base > previous_end:
            merged_excluded.append((base, length))
            continue
        merged_excluded[-1] = (
            previous_base,
            max(previous_end, end) - previous_base,
        )

    result: list[tuple[int, int]] = []
    for raw_base, raw_length in ranges:
        base = int(raw_base)
        length = int(raw_length)
        end = base + length
        if base <= 0 or length <= 0 or end <= base:
            raise ValueError("Source UVM range has invalid geometry")
        cursor = base
        for excluded_base, excluded_length in merged_excluded:
            excluded_end = excluded_base + excluded_length
            if excluded_end <= cursor:
                continue
            if excluded_base >= end:
                break
            if excluded_base > cursor:
                result.append((cursor, excluded_base - cursor))
            cursor = max(cursor, min(end, excluded_end))
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end - cursor))
    return tuple(result)


def _uvm_original_alias_intersect_ranges(
    ranges: Iterable[tuple[int, int]],
    coverage_ranges: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    """Clip source ranges to the normalized union of valid coverage."""
    coverage = sorted(
        (int(base), int(length))
        for base, length in coverage_ranges
        if int(length) > 0
    )
    merged_coverage: list[tuple[int, int]] = []
    for base, length in coverage:
        end = base + length
        if base <= 0 or end <= base:
            raise ValueError("Coverage UVM range has invalid geometry")
        if not merged_coverage:
            merged_coverage.append((base, length))
            continue
        previous_base, previous_length = merged_coverage[-1]
        previous_end = previous_base + previous_length
        if base > previous_end:
            merged_coverage.append((base, length))
        else:
            merged_coverage[-1] = (
                previous_base,
                max(previous_end, end) - previous_base,
            )

    intersections: list[tuple[int, int]] = []
    for raw_base, raw_length in ranges:
        base = int(raw_base)
        length = int(raw_length)
        end = base + length
        if base <= 0 or length <= 0 or end <= base:
            raise ValueError("Source UVM range has invalid geometry")
        for covered_base, covered_length in merged_coverage:
            covered_end = covered_base + covered_length
            if covered_end <= base:
                continue
            if covered_base >= end:
                break
            overlap_base = max(base, covered_base)
            overlap_end = min(end, covered_end)
            if overlap_end > overlap_base:
                intersections.append(
                    (overlap_base, overlap_end - overlap_base)
                )

    intersections.sort()
    merged: list[tuple[int, int]] = []
    for base, length in intersections:
        end = base + length
        if not merged:
            merged.append((base, length))
            continue
        previous_base, previous_length = merged[-1]
        previous_end = previous_base + previous_length
        if base > previous_end:
            merged.append((base, length))
        else:
            merged[-1] = (
                previous_base,
                max(previous_end, end) - previous_base,
            )
    return tuple(merged)


def _prepare_uvm_original_alias_plan_class_immutable_shadows(
    *,
    device: int,
) -> dict[str, Any]:
    """Create CPU read-duplicates for experts used by larger cold tiers.

    Pinned plan-class sources have a different VA and therefore cannot be used
    directly by the existing UVM page tables. At initialization we additionally
    materialize the same immutable contents in the original Parameter VA's CPU
    backing. ReadMostly residency keeps the current GPU copy live, so a later
    scheduler-quiesced cutover can discard it without a GPU-to-CPU transfer.
    """
    if not _strict_env_flag(
        "VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW", False
    ):
        return {"enabled": False, "reason": "disabled"}
    full_parameter_shadow = _strict_env_flag(
        "VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW", False
    )
    source_prefill_ready = bool(
        isinstance(_CPU_OFFLOAD_UVM_PLAN_CLASS_SOURCE_PREFILL_RESULT, dict)
        and _CPU_OFFLOAD_UVM_PLAN_CLASS_SOURCE_PREFILL_RESULT.get("enabled")
        is True
    )
    # A multi-tier shadow needs the plan-class source envelope because future
    # cold experts are not present in the initial packed source. A complete
    # Parameter shadow is different: it duplicates the current managed
    # Parameter VA's own CPU backing and is equally valid for a fixed, single
    # cold tier. This is the E3 contract, where every arm is an independent
    # cold process and no plan-class change occurs in-process.
    if not source_prefill_ready and not full_parameter_shadow:
        raise RuntimeError(
            "Immutable CPU shadows require successful plan-class source prefill"
        )

    initial = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_INITIAL_PLAN_CLASS
    if full_parameter_shadow:
        full_records = tuple(
            (
                full_name,
                int(binding.consumer.data_ptr()),
                int(binding.consumer.numel() * binding.consumer.element_size()),
            )
            for full_name, binding in sorted(
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
            )
        )
        ranges = [(base, length) for _, base, length in full_records]
        range_owners: list[tuple[str, int | str]] = [
            (full_name, "full_parameter")
            for full_name, _, _ in full_records
        ]
        for full_name, base, length in full_records:
            if base % (2 * 1024 * 1024) or length % (2 * 1024 * 1024):
                raise RuntimeError(
                    "Full-Parameter dual-view shadow requires 2 MiB aligned "
                    f"managed Parameters: name={full_name!r}, "
                    f"range=0x{base:x}+{length}"
                )
    else:
        shadow_records = _uvm_original_alias_plan_class_shadow_records()
        ranges = [(base, length) for _, _, base, length in shadow_records]
        range_owners = [
            (full_name, expert_id)
            for full_name, expert_id, _, _ in shadow_records
        ]

    if not ranges:
        return {"enabled": True, "reason": "no_future_cold_ranges"}

    from vllm.device_allocator import uvm

    requested_bytes = sum(length for _, length in ranges)
    readmostly_count = int(
        uvm.batch_advise_ranges_read_mostly(ranges, device=device)
    )
    prefetched_count = int(uvm.batch_prefetch_ranges_to_cpu(ranges))
    if readmostly_count != len(ranges) or prefetched_count != len(ranges):
        raise RuntimeError(
            "Failed to materialize every immutable CPU shadow: "
            f"readmostly={readmostly_count}, prefetched={prefetched_count}, "
            f"expected={len(ranges)}"
        )
    torch.cuda.synchronize(device)

    residency = dict(
        uvm.driver_batch_query_ranges_residency_bytes(
            ranges, device=device, wait_for_trackers=False
        )
    )
    if not (
        residency.get("ok") is True
        and int(residency.get("queried_bytes", -1)) == requested_bytes
        and int(residency.get("cpu_bytes", -1)) == requested_bytes
        and int(residency.get("gpu_bytes", -1)) == requested_bytes
        and int(residency.get("any_bytes", -1)) == requested_bytes
        and int(residency.get("unknown_bytes", -1)) == 0
    ):
        raise RuntimeError(
            "Immutable CPU shadows are not simultaneously CPU/GPU resident: "
            f"{residency!r}"
        )

    geometry_hash = hashlib.sha256()
    for (base, length), (full_name, expert_id) in zip(
        ranges, range_owners, strict=True
    ):
        geometry_hash.update(
            f"{full_name}\t{expert_id}\t{base:x}\t{length:x}\n".encode("utf-8")
        )
    return {
        "enabled": True,
        "reason": (
            "full_parameter_read_duplicate_shadows_ready"
            if full_parameter_shadow
            else "read_duplicate_shadows_ready"
        ),
        "full_parameter_shadow": bool(full_parameter_shadow),
        "initial_class": str(initial) if initial is not None else None,
        "fixed_plan_full_parameter_shadow": bool(
            full_parameter_shadow and not source_prefill_ready
        ),
        "range_count": len(ranges),
        "requested_bytes": int(requested_bytes),
        "readmostly_count": int(readmostly_count),
        "prefetched_count": int(prefetched_count),
        "residency": residency,
        "geometry_sha256": geometry_hash.hexdigest(),
    }


def _uvm_original_alias_binding_consumer_ranges(
    binding: _UvmOriginalAliasBinding,
) -> tuple[tuple[int, int], ...]:
    ranges = tuple(getattr(binding, "consumer_ranges", ()) or ())
    if ranges:
        return ranges
    return (
        (
            int(binding.consumer.data_ptr()),
            int(binding.requested_bytes),
        ),
    )


def _release_uvm_original_alias_consumer_gpu_chunks(
    *,
    device: int,
    known_cpu_ranges: Iterable[tuple[int, int]] = (),
) -> dict[str, Any]:
    """Release the original GPU chunks before alias MAP takes ownership.

    Moving managed pages to CPU is not sufficient here: UVM may retain the
    empty GPU chunks and continue charging their allocation commitment.  The
    explicit release ioctl splits those chunks at 64 KiB granularity and
    returns them to the GPU allocator while the original VA and CPU copy stay
    valid.  Pinned sources must already be complete before this helper runs.
    """
    bindings = [
        binding
        for _, binding in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items())
    ]
    sliced = [binding for binding in bindings if binding.expert_ids is not None]
    if not sliced:
        return {
            "ok": True,
            "required": False,
            "reason": "whole_parameter_topology",
            "region_count": 0,
            "requested_bytes": 0,
            "released_bytes": 0,
        }
    if len(sliced) != len(bindings):
        raise RuntimeError(
            "UVM original-VA topology mixes whole parameters and expert slices"
        )

    known_cpu_geometry = {
        (int(base), int(length)) for base, length in known_cpu_ranges
    }
    specs: list[tuple[str, int, int]] = []
    seen_ranges: set[tuple[int, int]] = set()
    for binding in sliced:
        expert_ids = tuple(int(value) for value in binding.expert_ids or ())
        ranges = _uvm_original_alias_binding_consumer_ranges(binding)
        if len(expert_ids) != len(ranges):
            raise RuntimeError(
                "UVM original-VA expert/range cardinality changed before release: "
                f"{binding.full_parameter_name!r}"
            )
        for expert_id, (base, length) in zip(expert_ids, ranges):
            base = int(base)
            length = int(length)
            if (
                base <= 0
                or length <= 0
                or base % (64 * 1024)
                or length % (64 * 1024)
            ):
                raise RuntimeError(
                    "UVM original-VA cold release range is not 64 KiB aligned: "
                    f"{binding.full_parameter_name!r}, expert={expert_id}, "
                    f"range=0x{base:x}+{length}"
                )
            geometry = (base, length)
            if geometry in seen_ranges:
                raise RuntimeError(
                    "UVM original-VA cold release range has multiple owners: "
                    f"range=0x{base:x}+{length}"
                )
            seen_ranges.add(geometry)
            region_id = (
                "static_alias_release:"
                f"{binding.full_parameter_name}:{binding.plan.layer_index}:"
                f"{expert_id}:{base:x}:{length:x}"
            )
            specs.append((region_id, base, length))

    # A whole-Parameter ReadMostly shadow deliberately leaves every expert
    # page resident on both CPU and GPU.  The ordinary residency-pool release
    # path models a migration from GPU to CPU and rejects that duplicate state.
    # Use the immutable-shadow cutover here as well as for later plan deltas:
    # it validates the existing CPU copy, discards the GPU copy without a
    # device-to-host transfer, and installs the read-only CPU mapping needed by
    # the original Parameter VA.  Ranges inherited from a rebuilt cold plan are
    # already CPU-only and therefore need validation rather than another
    # cutover.
    shadow_certificate = _CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT
    full_parameter_shadow = bool(
        isinstance(shadow_certificate, dict)
        and shadow_certificate.get("enabled") is True
        and shadow_certificate.get("full_parameter_shadow") is True
    )
    if full_parameter_shadow:
        from vllm.device_allocator import uvm

        spec_geometry = {(base, length) for _, base, length in specs}
        if not known_cpu_geometry <= spec_geometry:
            raise RuntimeError(
                "UVM original-VA known-CPU geometry is absent from the new "
                "cold plan"
            )
        cutover_ranges = tuple(
            (base, length)
            for _, base, length in specs
            if (base, length) not in known_cpu_geometry
        )
        cutover_bytes = sum(length for _, length in cutover_ranges)
        shadow_cutover = dict(
            uvm.driver_cutover_ranges_to_immutable_cpu_shadow(
                cutover_ranges,
                device=device,
            )
        )
        if not (
            shadow_cutover.get("ok") is True
            and int(shadow_cutover.get("range_count", -1))
            == len(cutover_ranges)
            and int(shadow_cutover.get("success_count", -1))
            == len(cutover_ranges)
            and int(shadow_cutover.get("failure_count", -1)) == 0
            and int(shadow_cutover.get("requested_bytes", -1))
            == cutover_bytes
            and int(shadow_cutover.get("released_gpu_bytes", -1))
            == cutover_bytes
            and int(
                shadow_cutover.get("retained_cpu_shadow_bytes", -1)
            )
            == cutover_bytes
            and int(shadow_cutover.get("remote_mapped_bytes", -1))
            == cutover_bytes
        ):
            raise RuntimeError(
                "Initial immutable CPU shadow cutover did not release every "
                f"new cold expert: {shadow_cutover!r}"
            )

        known_cpu_residency: dict[str, Any] | None = None
        if known_cpu_geometry:
            known_cpu_bytes = sum(length for _, length in known_cpu_geometry)
            known_cpu_residency = dict(
                uvm.driver_batch_query_ranges_residency_bytes(
                    tuple(sorted(known_cpu_geometry)),
                    device=device,
                    wait_for_trackers=False,
                )
            )
            if not (
                known_cpu_residency.get("ok") is True
                and int(known_cpu_residency.get("queried_bytes", -1))
                == known_cpu_bytes
                and int(known_cpu_residency.get("cpu_bytes", -1))
                == known_cpu_bytes
                and int(known_cpu_residency.get("gpu_bytes", -1)) == 0
                and int(known_cpu_residency.get("unknown_bytes", -1)) == 0
            ):
                raise RuntimeError(
                    "Retained immutable CPU shadow ranges are not CPU-only: "
                    f"{known_cpu_residency!r}"
                )

        geometry_hash = hashlib.sha256()
        for region_id, base, length in specs:
            geometry_hash.update(
                f"{region_id}\t{base:x}\t{length:x}\n".encode("ascii")
            )
        return {
            "ok": True,
            "required": True,
            "reason": "immutable_cpu_shadow_cutover",
            "region_count": len(specs),
            "requested_bytes": sum(length for _, _, length in specs),
            "attempted_bytes": int(cutover_bytes),
            "released_bytes": int(cutover_bytes),
            "made_progress_count": len(cutover_ranges),
            "known_cpu_range_count": len(known_cpu_geometry),
            "known_cpu_residency": known_cpu_residency,
            "shadow_cutover": shadow_cutover,
            "temporary_regions_unregistered": True,
            "geometry_sha256": geometry_hash.hexdigest(),
        }

    from vllm.device_allocator.uvm_pool_residency import (
        reconcile_uvm_regions_cpu_state_from_physical_residency,
        register_uvm_region,
        release_uvm_regions,
        unregister_uvm_regions,
    )

    registered_region_ids: list[str] = []
    release: dict[str, Any] = {}
    known_cpu_state_reconciliation: dict[str, Any] | None = None
    released_bytes = 0
    attempted_bytes = 0
    made_progress_count = 0
    try:
        for region_id, base, length in specs:
            register_uvm_region(region_id, base, length, state="unknown")
            registered_region_ids.append(region_id)

        region_ids = [region_id for region_id, _, _ in specs]
        known_cpu_region_ids = [
            region_id
            for region_id, base, length in specs
            if (base, length) in known_cpu_geometry
        ]
        if len(known_cpu_region_ids) != len(known_cpu_geometry):
            raise RuntimeError(
                "UVM original-VA known-CPU geometry is absent from the new "
                "cold plan"
            )
        if known_cpu_region_ids:
            known_cpu_state_reconciliation = dict(
                reconcile_uvm_regions_cpu_state_from_physical_residency(
                    known_cpu_region_ids,
                    device=int(device),
                    wait_for_trackers=False,
                )
            )
            if not (
                known_cpu_state_reconciliation.get("ok") is True
                and known_cpu_state_reconciliation.get("physical_cpu_only")
                is True
            ):
                raise RuntimeError(
                    "UVM original-VA shared cold ranges did not prove CPU "
                    "residency before plan rebuild release: "
                    f"{known_cpu_state_reconciliation!r}"
                )
        cpu_state_reconciliation = (
            reconcile_uvm_regions_cpu_state_from_physical_residency(
                region_ids,
                device=int(device),
                wait_for_trackers=False,
            )
        )
        if (
            cpu_state_reconciliation.get("physical_cpu_only") is True
            and cpu_state_reconciliation.get("ok") is not True
        ):
            raise RuntimeError(
                "UVM original-VA cold release proved existing CPU residency "
                "but could not restore the explicit-region state: "
                f"{cpu_state_reconciliation!r}"
            )
        release = release_uvm_regions(
            region_ids,
            device=int(device),
            remote_read=False,
            include_audit=True,
        )
        statuses = release.get("statuses")
        audits = release.get("audits")
        if not isinstance(statuses, dict) or not isinstance(audits, dict):
            raise RuntimeError(
                "UVM original-VA cold release returned incomplete audit tables"
            )

        for region_id, base, length in specs:
            audit = audits.get(region_id)
            valid = (
                release.get("ok") is True
                and statuses.get(region_id) is True
                and isinstance(audit, dict)
                and audit.get("ok") is True
                and audit.get("remote_read") is False
                and int(audit.get("registered_base", -1)) == base
                and int(audit.get("registered_length", -1)) == length
                and int(audit.get("base", -1)) == base
                and int(audit.get("length", -1)) == length
                and audit.get("app_state_after") == "cpu"
            )
            if not valid:
                raise RuntimeError(
                    "UVM original-VA cold release did not prove completion: "
                    f"region_id={region_id!r}, audit={audit!r}"
                )
            released_bytes += int(audit.get("released_bytes", 0))
            attempted_bytes += int(audit.get("attempted_bytes", 0))
            made_progress_count += int(bool(audit.get("made_progress")))
    finally:
        if registered_region_ids:
            unregister = unregister_uvm_regions(tuple(registered_region_ids))
            if unregister.get("ok") is not True:
                raise RuntimeError(
                    "UVM original-VA cold release could not unregister its "
                    f"temporary residency regions: {unregister!r}"
                )

    geometry_hash = hashlib.sha256()
    for region_id, base, length in specs:
        geometry_hash.update(
            f"{region_id}\t{base:x}\t{length:x}\n".encode("ascii")
        )
    return {
        "ok": True,
        "required": True,
        "reason": "released",
        "region_count": len(specs),
        "requested_bytes": sum(length for _, _, length in specs),
        "attempted_bytes": int(attempted_bytes),
        "released_bytes": int(released_bytes),
        "made_progress_count": int(made_progress_count),
        "known_cpu_range_count": len(known_cpu_geometry),
        "known_cpu_state_reconciliation": known_cpu_state_reconciliation,
        "cpu_state_reconciliation": cpu_state_reconciliation,
        "temporary_regions_unregistered": True,
        "geometry_sha256": geometry_hash.hexdigest(),
    }


def _uvm_original_alias_binding_hot_ranges(
    binding: _UvmOriginalAliasBinding,
) -> tuple[tuple[int, int], ...]:
    """Return the complement of packed cold expert ranges in one Parameter."""
    expert_ids = getattr(binding, "expert_ids", None)
    parameter_bytes = int(getattr(binding, "parameter_bytes", 0) or 0)
    if not expert_ids or parameter_bytes <= 0:
        return ()

    parameter_base = int(binding.consumer.data_ptr())
    parameter_end = parameter_base + parameter_bytes
    cold_ranges = sorted(_uvm_original_alias_binding_consumer_ranges(binding))
    result: list[tuple[int, int]] = []
    cursor = parameter_base
    for cold_base, cold_bytes in cold_ranges:
        cold_end = int(cold_base) + int(cold_bytes)
        if cold_base < cursor or cold_end > parameter_end:
            raise RuntimeError(
                "Expert-slice consumer ranges overlap or escape their Parameter: "
                f"{binding.full_parameter_name!r}"
            )
        if cold_base > cursor:
            result.append((cursor, int(cold_base) - cursor))
        cursor = cold_end
    if cursor < parameter_end:
        result.append((cursor, parameter_end - cursor))
    return tuple(result)


def audit_cpu_offload_uvm_original_va_post_workspace_residency(
    *,
    num_tokens: int,
    weight_tensors: Mapping[str, torch.Tensor],
    workspace_tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Query hot physical residency immediately before a fused-MoE launch.

    This is an opt-in diagnostic for distinguishing managed-page eviction
    from a GPU page-table/channel visibility failure.  Unlike the CUDA-read
    gate, the driver residency query never dereferences an expert VA and
    therefore cannot fault a CPU-resident hot page back into GPU memory.
    """
    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return {"performed": False, "reason": "disabled"}
    if _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE:
        return {"performed": False, "reason": "profile_all_remote"}
    if (
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE
        and not _strict_env_flag(
            "VLLM_UVM_FUSED_MOE_POST_WORKSPACE_AUDIT_CAPTURE_WARMUP", False
        )
    ):
        return {"performed": False, "reason": "capture_phase"}
    if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE != "sealed":
        return {
            "performed": False,
            "reason": "topology_not_sealed",
            "state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
        }
    if torch.cuda.is_current_stream_capturing():
        return {"performed": False, "reason": "stream_capturing"}
    if not weight_tensors:
        raise ValueError("Post-workspace residency audit requires weight tensors")

    first_weight = next(iter(weight_tensors.values()))
    if not first_weight.is_cuda:
        raise ValueError("Post-workspace residency audit requires CUDA weights")
    device = int(
        first_weight.device.index
        if first_weight.device.index is not None
        else (_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0)
    )
    if any(
        not tensor.is_cuda or int(tensor.device.index or 0) != device
        for tensor in (*weight_tensors.values(), *workspace_tensors.values())
    ):
        raise ValueError(
            "Post-workspace residency audit tensors must share one CUDA device"
        )

    def tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
        return {
            "data_ptr": int(tensor.data_ptr()),
            "bytes": int(tensor.numel() * tensor.element_size()),
            "storage_bytes": int(tensor.untyped_storage().nbytes()),
            "shape": tuple(int(value) for value in tensor.shape),
            "dtype": str(tensor.dtype),
        }

    binding_records: list[dict[str, Any]] = []
    matched_binding_names: set[str] = set()
    for tensor_name, tensor in weight_tensors.items():
        tensor_base = int(tensor.data_ptr())
        tensor_bytes = int(tensor.numel() * tensor.element_size())
        tensor_end = tensor_base + tensor_bytes
        matches: list[dict[str, Any]] = []
        for binding_name, binding in sorted(
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
        ):
            parameter_base = int(binding.consumer.data_ptr())
            parameter_bytes = int(binding.parameter_bytes)
            parameter_end = parameter_base + parameter_bytes
            if parameter_base <= tensor_base and tensor_end <= parameter_end:
                matched_binding_names.add(binding_name)
                matches.append(
                    {
                        "full_parameter_name": binding_name,
                        "parameter_base": parameter_base,
                        "parameter_bytes": parameter_bytes,
                        "tensor_offset_bytes": tensor_base - parameter_base,
                        "expert_stride_bytes": int(binding.expert_stride_bytes),
                        "cold_expert_ids": tuple(binding.expert_ids or ()),
                    }
                )
        binding_records.append(
            {
                "tensor_name": str(tensor_name),
                **tensor_record(tensor),
                "binding_matches": matches,
            }
        )

    hot_ranges = [
        hot_range
        for _, binding in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items())
        for hot_range in _uvm_original_alias_binding_hot_ranges(binding)
    ]
    current_hot_ranges = [
        hot_range
        for binding_name, binding in sorted(
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
        )
        if binding_name in matched_binding_names
        for hot_range in _uvm_original_alias_binding_hot_ranges(binding)
    ]
    hot_bytes = sum(int(length) for _, length in hot_ranges)
    current_hot_bytes = sum(int(length) for _, length in current_hot_ranges)
    workspace_records = {
        str(name): tensor_record(tensor)
        for name, tensor in workspace_tensors.items()
    }
    workspace_bytes = sum(
        int(tensor.numel() * tensor.element_size())
        for tensor in workspace_tensors.values()
    )

    started_ns = time.perf_counter_ns()
    begin = {
        "operation": "post_workspace_hot_residency_audit",
        "audit_status": "begin",
        "performed": True,
        "num_tokens": int(num_tokens),
        "device": device,
        "hot_range_count": len(hot_ranges),
        "hot_bytes": int(hot_bytes),
        "current_hot_range_count": len(current_hot_ranges),
        "current_hot_bytes": int(current_hot_bytes),
        "workspace_bytes": int(workspace_bytes),
        "workspaces": workspace_records,
        "weights": binding_records,
    }
    _append_uvm_original_alias_lifecycle(
        "cpu_offload_uvm_post_workspace_residency_audit_begin", begin
    )

    try:
        from vllm.device_allocator import uvm

        torch.cuda.synchronize(device)
        free_bytes, total_bytes = (
            int(value) for value in torch.cuda.mem_get_info(device)
        )
        hot_physical = dict(
            uvm.driver_batch_query_ranges_residency_bytes(
                hot_ranges,
                device=device,
                wait_for_trackers=False,
            )
        )
        current_hot_physical = dict(
            uvm.driver_batch_query_ranges_residency_bytes(
                current_hot_ranges,
                device=device,
                wait_for_trackers=False,
            )
        )
        hot_cuda_read_gate: dict[str, Any] | None = None
        cold_cuda_read_gate: dict[str, Any] | None = None
        if _strict_env_flag(
            "VLLM_UVM_FUSED_MOE_PRE_KERNEL_ALIAS_READ_GATE", False
        ):
            hot_cuda_read_gate = _uvm_original_alias_hot_cuda_read_gate(
                stage="runtime_post_workspace_pre_kernel",
                device=device,
            )
            cold_cuda_read_gate = (
                _uvm_original_alias_current_cold_cuda_read_gate(
                    stage="runtime_post_workspace_pre_kernel",
                    device=device,
                    binding_names=tuple(sorted(matched_binding_names)),
                )
            )
        memory = {
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        expected_hot_cpu_bytes = (
            _uvm_original_alias_expected_shadow_cpu_bytes(hot_ranges)
        )
        expected_current_hot_cpu_bytes = (
            _uvm_original_alias_expected_shadow_cpu_bytes(current_hot_ranges)
        )
        hot_gpu_only = bool(
            hot_physical.get("ok") is True
            and int(hot_physical.get("queried_bytes", -1)) == hot_bytes
            and int(hot_physical.get("gpu_bytes", -1)) == hot_bytes
            and int(hot_physical.get("cpu_bytes", -1))
            == expected_hot_cpu_bytes
            and int(hot_physical.get("unknown_bytes", -1)) == 0
        )
        current_hot_gpu_only = bool(
            current_hot_physical.get("ok") is True
            and int(current_hot_physical.get("queried_bytes", -1))
            == current_hot_bytes
            and int(current_hot_physical.get("gpu_bytes", -1))
            == current_hot_bytes
            and int(current_hot_physical.get("cpu_bytes", -1))
            == expected_current_hot_cpu_bytes
            and int(current_hot_physical.get("unknown_bytes", -1)) == 0
        )
        result = {
            **begin,
            "audit_status": "ok",
            "ok": True,
            "wall_ns": time.perf_counter_ns() - started_ns,
            "memory": memory,
            "hot_physical_residency": hot_physical,
            "current_hot_physical_residency": current_hot_physical,
            "expected_hot_cpu_shadow_bytes": int(expected_hot_cpu_bytes),
            "expected_current_hot_cpu_shadow_bytes": int(
                expected_current_hot_cpu_bytes
            ),
            "hot_gpu_only": hot_gpu_only,
            "current_hot_gpu_only": current_hot_gpu_only,
            "hot_cuda_read_gate": hot_cuda_read_gate,
            "cold_cuda_read_gate": cold_cuda_read_gate,
        }
    except Exception as exc:
        result = {
            **begin,
            "audit_status": "error",
            "ok": False,
            "wall_ns": time.perf_counter_ns() - started_ns,
            "error": repr(exc),
        }
    _append_uvm_original_alias_lifecycle(
        "cpu_offload_uvm_post_workspace_residency_audit", result
    )
    return result


def trace_cpu_offload_uvm_original_va_prefill_memory(
    *,
    stage: str,
    num_tokens: int,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Record an allocation/residency boundary without touching expert VAs.

    This diagnostic complements the post-workspace audit.  It is deliberately
    callable before the first MoE workspace exists so a large-Prefill run can
    distinguish attention allocations, caching-allocator segment growth, and
    the individual fused-MoE scratch allocations.  Residency is queried from
    driver masks; no hot page is dereferenced or migrated by this helper.
    """
    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return {"performed": False, "reason": "disabled"}
    if _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE:
        return {"performed": False, "reason": "profile_all_remote"}
    if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE:
        return {"performed": False, "reason": "capture_phase"}
    if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE != "sealed":
        return {
            "performed": False,
            "reason": "topology_not_sealed",
            "state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
        }
    if torch.cuda.is_current_stream_capturing():
        return {"performed": False, "reason": "stream_capturing"}

    cuda_tensors = {
        str(name): tensor
        for name, tensor in tensors.items()
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda
    }
    if not cuda_tensors:
        return {"performed": False, "reason": "no_cuda_tensors"}
    first_tensor = next(iter(cuda_tensors.values()))
    device = int(first_tensor.device.index or 0)
    if any(int(tensor.device.index or 0) != device for tensor in cuda_tensors.values()):
        raise ValueError("Prefill memory trace tensors must share one CUDA device")

    from vllm.device_allocator import uvm

    def tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
        ptr = int(tensor.data_ptr())
        logical_bytes = int(tensor.numel() * tensor.element_size())
        storage_bytes = int(tensor.untyped_storage().nbytes())
        managed, managed_reason, managed_code = uvm.probe_driver_capable_range(
            ptr, max(logical_bytes, 1)
        )
        return {
            "data_ptr": ptr,
            "bytes": logical_bytes,
            "storage_bytes": storage_bytes,
            "shape": tuple(int(value) for value in tensor.shape),
            "dtype": str(tensor.dtype),
            "managed": bool(managed),
            "pointer_reason": str(managed_reason),
            "pointer_code": int(managed_code),
        }

    tensor_records = {
        name: tensor_record(tensor) for name, tensor in cuda_tensors.items()
    }
    hot_ranges = [
        hot_range
        for _, binding in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items())
        for hot_range in _uvm_original_alias_binding_hot_ranges(binding)
    ]
    hot_bytes = sum(int(length) for _, length in hot_ranges)
    started_ns = time.perf_counter_ns()
    begin = {
        "operation": "prefill_memory_trace",
        "trace_status": "begin",
        "performed": True,
        "trace_stage": str(stage),
        "num_tokens": int(num_tokens),
        "device": device,
        "hot_range_count": len(hot_ranges),
        "hot_bytes": int(hot_bytes),
        "tensors": tensor_records,
    }
    _append_uvm_original_alias_lifecycle(
        "cpu_offload_uvm_prefill_memory_trace_begin", begin
    )

    try:
        torch.cuda.synchronize(device)
        free_bytes, total_bytes = (
            int(value) for value in torch.cuda.mem_get_info(device)
        )
        hot_physical = dict(
            uvm.driver_batch_query_ranges_residency_bytes(
                hot_ranges,
                device=device,
                wait_for_trackers=False,
            )
        )
        stats = torch.cuda.memory_stats(device)
        snapshot = torch.cuda.memory_snapshot()
        raw_segments = [
            segment
            for segment in snapshot
            if int(segment.get("device", device)) == device
        ]
        segments = [
            {
                "address": int(segment.get("address", 0)),
                "total_size": int(segment.get("total_size", 0)),
                "allocated_size": int(segment.get("allocated_size", 0)),
                "active_size": int(segment.get("active_size", 0)),
                "requested_size": int(segment.get("requested_size", 0)),
                "segment_type": str(segment.get("segment_type", "unknown")),
                "stream": int(segment.get("stream", 0)),
            }
            for segment in raw_segments
        ]
        segments.sort(key=lambda value: (value["address"], value["total_size"]))
        for tensor in tensor_records.values():
            ptr = int(tensor["data_ptr"])
            owner = next(
                (
                    segment
                    for segment in segments
                    if int(segment["address"])
                    <= ptr
                    < int(segment["address"]) + int(segment["total_size"])
                ),
                None,
            )
            tensor["allocator_segment"] = owner

        memory = {
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "active_bytes": int(stats.get("active_bytes.all.current", 0)),
            "requested_bytes": int(stats.get("requested_bytes.all.current", 0)),
            "inactive_split_bytes": int(
                stats.get("inactive_split_bytes.all.current", 0)
            ),
            "segment_bytes": int(stats.get("reserved_bytes.all.current", 0)),
            "segment_count": int(stats.get("segment.all.current", 0)),
            "allocation_count": int(stats.get("allocation.all.current", 0)),
            "active_count": int(stats.get("active.all.current", 0)),
            "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
            "num_ooms": int(stats.get("num_ooms", 0)),
        }
        result = {
            **begin,
            "trace_status": "ok",
            "ok": True,
            "wall_ns": time.perf_counter_ns() - started_ns,
            "memory": memory,
            "hot_physical_residency": hot_physical,
            "allocator_segments": segments,
        }
    except Exception as exc:
        result = {
            **begin,
            "trace_status": "error",
            "ok": False,
            "wall_ns": time.perf_counter_ns() - started_ns,
            "error": repr(exc),
        }
    _append_uvm_original_alias_lifecycle(
        "cpu_offload_uvm_prefill_memory_trace", result
    )
    return result


def trace_cpu_offload_uvm_original_va_prefill_layer_boundary(
    *,
    stage: str,
    num_tokens: int,
    tensors: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Trace allocator segment growth at a cheap per-layer boundary.

    The full Prefill memory trace above intentionally records every allocator
    segment and every hot range.  Repeating that payload at every decoder
    layer makes the diagnostic itself too invasive.  This companion trace
    records allocator history once, emits only newly created native CUDA
    segments (including their allocation stacks), and queries hot residency
    only when the allocator reserve or segment set changes.
    """
    if not _strict_env_flag("VLLM_UVM_PREFILL_LAYER_MEMORY_TRACE", False):
        return {"performed": False, "reason": "disabled"}
    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return {"performed": False, "reason": "offload_disabled"}
    if _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE:
        return {"performed": False, "reason": "profile_all_remote"}
    if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE:
        return {"performed": False, "reason": "capture_phase"}
    if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE != "sealed":
        return {
            "performed": False,
            "reason": "topology_not_sealed",
            "state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
        }
    if torch.cuda.is_current_stream_capturing():
        return {"performed": False, "reason": "stream_capturing"}

    cuda_tensors = [
        tensor
        for tensor in tensors.values()
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda
    ]
    if not cuda_tensors:
        return {"performed": False, "reason": "no_cuda_tensors"}
    device = int(cuda_tensors[0].device.index or 0)
    if any(int(tensor.device.index or 0) != device for tensor in cuda_tensors):
        raise ValueError("Prefill layer trace tensors must share one CUDA device")

    state = _CPU_OFFLOAD_UVM_PREFILL_LAYER_TRACE_STATE.setdefault(device, {})
    started_ns = time.perf_counter_ns()
    try:
        if not state.get("history_enabled", False):
            max_entries = int(
                os.environ.get(
                    "VLLM_UVM_PREFILL_ALLOCATOR_HISTORY_MAX_ENTRIES", "100000"
                )
            )
            torch.cuda.memory._record_memory_history(
                enabled="all",
                context="alloc",
                stacks="python",
                max_entries=max_entries,
                device=device,
                clear_history=True,
            )
            state["history_enabled"] = True
            state["history_max_entries"] = max_entries

        torch.cuda.synchronize(device)
        free_bytes, total_bytes = (
            int(value) for value in torch.cuda.mem_get_info(device)
        )
        stats = torch.cuda.memory_stats(device)
        snapshot = torch.cuda.memory._snapshot(device)
        raw_segments = [
            segment
            for segment in snapshot.get("segments", [])
            if int(segment.get("device", device)) == device
        ]
        current_addresses = {
            int(segment.get("address", 0)) for segment in raw_segments
        }
        previous_addresses = set(state.get("segment_addresses", ()))
        new_addresses = current_addresses - previous_addresses

        device_traces = snapshot.get("device_traces", [])
        trace_entries = (
            device_traces[device] if device < len(device_traces) else []
        )
        previous_trace_count = int(state.get("trace_count", 0))
        if previous_trace_count > len(trace_entries):
            previous_trace_count = 0
        new_trace_entries = trace_entries[previous_trace_count:]

        def compact_frames(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
            frames = list(entry.get("frames", []))
            # PyTorch orders allocator frames from the allocation site out to
            # the process entry point. Keep the operator-side context while
            # bounding manifest size.
            return [
                {
                    "filename": str(frame.get("filename", "")),
                    "line": int(frame.get("line", 0)),
                    "name": str(frame.get("name", "")),
                }
                for frame in frames[:32]
            ]

        segment_alloc_events = [
            {
                "action": "segment_alloc",
                "address": int(entry.get("addr", 0)),
                "bytes": int(entry.get("size", 0)),
                "stream": int(entry.get("stream", 0)),
                "frames": compact_frames(entry),
            }
            for entry in new_trace_entries
            if entry.get("action") == "segment_alloc"
        ]
        new_segments = []
        for segment in raw_segments:
            address = int(segment.get("address", 0))
            if address not in new_addresses:
                continue
            blocks = list(segment.get("blocks", []))
            new_segments.append(
                {
                    "address": address,
                    "total_size": int(segment.get("total_size", 0)),
                    "allocated_size": int(segment.get("allocated_size", 0)),
                    "active_size": int(segment.get("active_size", 0)),
                    "segment_type": str(segment.get("segment_type", "unknown")),
                    "stream": int(segment.get("stream", 0)),
                    "blocks": [
                        {
                            "address": int(block.get("address", 0)),
                            "size": int(block.get("size", 0)),
                            "requested_size": int(block.get("requested_size", 0)),
                            "state": str(block.get("state", "unknown")),
                            "frames": compact_frames(block),
                        }
                        for block in blocks
                    ],
                }
            )
        new_segments.sort(key=lambda value: value["address"])

        memory = {
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "active_bytes": int(stats.get("active_bytes.all.current", 0)),
            "inactive_split_bytes": int(
                stats.get("inactive_split_bytes.all.current", 0)
            ),
            "segment_count": int(stats.get("segment.all.current", 0)),
            "allocation_count": int(stats.get("allocation.all.current", 0)),
        }
        previous_memory = dict(state.get("memory", {}))
        delta = {
            key: int(value) - int(previous_memory.get(key, value))
            for key, value in memory.items()
            if key != "total_bytes"
        }
        allocator_changed = bool(
            not previous_memory
            or new_addresses
            or delta.get("reserved_bytes", 0)
            or delta.get("segment_count", 0)
        )

        hot_physical: dict[str, Any] | None = None
        if allocator_changed:
            from vllm.device_allocator import uvm

            hot_ranges = [
                hot_range
                for _, binding in sorted(
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                )
                for hot_range in _uvm_original_alias_binding_hot_ranges(binding)
            ]
            hot_physical = dict(
                uvm.driver_batch_query_ranges_residency_bytes(
                    hot_ranges,
                    device=device,
                    wait_for_trackers=False,
                )
            )

        result = {
            "operation": "prefill_layer_memory_trace",
            "performed": True,
            "ok": True,
            "trace_stage": str(stage),
            "num_tokens": int(num_tokens),
            "device": device,
            "wall_ns": time.perf_counter_ns() - started_ns,
            "memory": memory,
            "delta": delta,
            "new_segments": new_segments,
            "segment_alloc_events": segment_alloc_events,
            "history_entry_count": len(trace_entries),
            "hot_physical_residency": hot_physical,
        }
        state["memory"] = memory
        state["segment_addresses"] = tuple(sorted(current_addresses))
        state["trace_count"] = len(trace_entries)
    except Exception as exc:
        result = {
            "operation": "prefill_layer_memory_trace",
            "performed": True,
            "ok": False,
            "trace_stage": str(stage),
            "num_tokens": int(num_tokens),
            "device": device,
            "wall_ns": time.perf_counter_ns() - started_ns,
            "error": repr(exc),
        }
    _append_uvm_original_alias_lifecycle(
        "cpu_offload_uvm_prefill_layer_memory_trace", result
    )
    return result


def _uvm_original_alias_hot_cuda_read_gate(
    *,
    stage: str,
    device: int,
) -> dict[str, Any] | None:
    """Prove that sampled hot VA pages are GPU-readable and unchanged.

    Residency queries only inspect physical ownership bits. They cannot catch
    a missing GPU PDE after selected cold subchunks are released or replaced by
    static aliases. This diagnostic/startup gate issues real CUDA reads from
    one byte in every hot intersection of the configured VA granularity. The
    first invocation at each granularity records an immutable byte checksum
    per Parameter; later stages at that granularity must reproduce it exactly.

    A begin record is flushed before the CUDA work, so a fatal mapping error
    still identifies the first failing transition in the manifest.
    """
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_CUDA_READ_BASELINE

    if not _strict_env_flag(
        "VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE", False
    ):
        return None

    default_block_bytes = 2 * 1024 * 1024
    granularity_env = "VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE_GRANULARITY_BYTES"
    if str(stage).startswith("runtime_"):
        granularity_env = (
            "VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE_RUNTIME_GRANULARITY_BYTES"
        )
    raw_block_bytes = os.environ.get(
        granularity_env,
        os.environ.get(
            "VLLM_CPU_OFFLOAD_UVM_HOT_CUDA_READ_GATE_GRANULARITY_BYTES",
            str(default_block_bytes),
        ),
    )
    try:
        block_bytes = int(raw_block_bytes)
    except ValueError as exc:
        raise RuntimeError(
            f"{granularity_env} must be an integer, got {raw_block_bytes!r}"
        ) from exc
    if block_bytes < 4096 or block_bytes % 4096:
        raise RuntimeError(
            f"{granularity_env} must be a positive 4 KiB multiple, "
            f"got {block_bytes}"
        )
    sample_bias_bytes = 4 * 1024
    binding_specs: list[tuple[str, torch.Tensor, tuple[int, ...]]] = []
    hot_residency_ranges: list[tuple[int, int]] = []
    total_hot_bytes = 0
    total_samples = 0
    for name, binding in sorted(
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
    ):
        parameter = binding.consumer
        if not parameter.is_cuda or not parameter.is_contiguous():
            raise RuntimeError(
                "Hot CUDA-read gate requires contiguous CUDA Parameters: "
                f"{name!r}"
            )
        parameter_base = int(parameter.data_ptr())
        parameter_bytes = int(binding.parameter_bytes)
        indices: list[int] = []
        for hot_base, hot_bytes in _uvm_original_alias_binding_hot_ranges(
            binding
        ):
            hot_base = int(hot_base)
            hot_bytes = int(hot_bytes)
            hot_end = hot_base + hot_bytes
            if (
                hot_bytes <= 0
                or hot_base < parameter_base
                or hot_end > parameter_base + parameter_bytes
            ):
                raise RuntimeError(
                    f"Invalid hot CUDA-read range for {name!r}: "
                    f"0x{hot_base:x}+{hot_bytes}"
                )
            total_hot_bytes += hot_bytes
            hot_residency_ranges.append((hot_base, hot_bytes))
            block_base = hot_base - (hot_base % block_bytes)
            while block_base < hot_end:
                intersection_start = max(hot_base, block_base)
                intersection_end = min(hot_end, block_base + block_bytes)
                sample_address = min(
                    intersection_start + sample_bias_bytes,
                    intersection_end - 1,
                )
                indices.append(sample_address - parameter_base)
                block_base += block_bytes
        if indices:
            unique_indices = tuple(sorted(set(indices)))
            binding_specs.append((name, parameter, unique_indices))
            total_samples += len(unique_indices)

    hot_physical_residency: dict[str, Any] | None = None
    if str(stage).startswith("runtime_") and hot_residency_ranges:
        from vllm.device_allocator import uvm

        hot_physical_residency = dict(
            uvm.driver_batch_query_ranges_residency_bytes(
                hot_residency_ranges,
                device=device,
                wait_for_trackers=False,
            )
        )

    begin = {
        "operation": "hot_cuda_read_gate",
        "gate_stage": str(stage),
        "gate_status": "begin",
        "parameter_count": len(binding_specs),
        "sample_count": total_samples,
        "hot_bytes": total_hot_bytes,
        "sample_granularity_bytes": block_bytes,
        "sample_bias_bytes": sample_bias_bytes,
        "hot_physical_residency": hot_physical_residency,
    }
    _append_uvm_original_alias_lifecycle(
        "cpu_offload_uvm_hot_cuda_read_gate_begin", begin
    )

    observed: dict[str, tuple[int, int]] = {}
    with torch.cuda.device(device):
        for name, parameter, indices in binding_specs:
            byte_view = parameter.view(torch.uint8).reshape(-1)
            index = torch.tensor(
                indices,
                dtype=torch.int64,
                device=parameter.device,
            )
            sampled = torch.index_select(byte_view, 0, index)
            checksum = int(sampled.to(torch.int64).sum().item())
            observed[name] = (len(indices), checksum)
        torch.cuda.synchronize(device)

    # Plan Class changes intentionally change the hot expert set. Comparing
    # the resulting sample dictionary to the previous tier therefore reports
    # a false data-corruption event because both the indices and counts differ.
    # Keep one baseline per exact hot geometry: a K2 -> K6 -> K2 round trip then
    # still compares the returned K2 bytes against its original baseline.
    baseline_key = (
        block_bytes,
        tuple((name, indices) for name, _parameter, indices in binding_specs),
    )
    baseline = _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_CUDA_READ_BASELINE.get(
        baseline_key
    )
    if baseline is None:
        baseline_stage = str(stage)
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_CUDA_READ_BASELINE[baseline_key] = (
            baseline_stage,
            dict(observed),
        )
    else:
        baseline_stage, baseline_observed = baseline
        if observed != baseline_observed:
            changed = sorted(
                name
                for name in set(observed)
                | set(baseline_observed)
                if observed.get(name) != baseline_observed.get(name)
            )
            failure = {
                **begin,
                "gate_status": "mismatch",
                "baseline_stage": baseline_stage,
                "changed_parameter_count": len(changed),
                "changed_parameters": changed,
                "changed_observations": {
                    name: {
                        "baseline": baseline_observed.get(name),
                        "observed": observed.get(name),
                    }
                    for name in changed
                },
            }
            _append_uvm_original_alias_lifecycle(
                "cpu_offload_uvm_hot_cuda_read_gate_mismatch", failure
            )
            raise RuntimeError(
                "Hot CUDA-read gate detected changed expert bytes after "
                f"{stage}: {changed[:8]!r}"
            )

    checksum_sum = sum(value[1] for value in observed.values())
    result = {
        **begin,
        "gate_status": "ok",
        "baseline_stage": baseline_stage,
        "checksum_sum": int(checksum_sum),
    }
    _append_uvm_original_alias_lifecycle(
        "cpu_offload_uvm_hot_cuda_read_gate_ok", result
    )
    return result


def _uvm_original_alias_current_cold_cuda_read_gate(
    *,
    stage: str,
    device: int,
    binding_names: tuple[str, ...],
    verify_source_bytes: bool = True,
) -> dict[str, Any] | None:
    """Validate sampled cold consumer aliases against their pinned sources.

    With ``verify_source_bytes=True`` this diagnostic runs after the module
    wrapper has enqueued a pinned-source to upload-VA copy and proves both the
    copy and the sealed alias mapping.  With it disabled, the helper is a
    read-only post-RESUME PTE fence: values in arenas shared by several layers
    are intentionally not compared, but every sampled consumer mapping must be
    GPU-readable before CUDA Graph replay.  A begin record is flushed first so
    an invalid alias remains attributable even if CUDA is poisoned by the read.
    """
    if not _strict_env_flag(
        "VLLM_UVM_FUSED_MOE_PRE_KERNEL_ALIAS_READ_GATE", False
    ):
        return None
    if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE != "sealed":
        raise RuntimeError(
            "Cold CUDA-read gate requires a sealed alias topology, got "
            f"{_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE!r}"
        )
    raw_block_bytes = os.environ.get(
        "VLLM_UVM_FUSED_MOE_PRE_KERNEL_COLD_GATE_GRANULARITY_BYTES",
        str(512 * 1024),
    )
    try:
        block_bytes = int(raw_block_bytes)
    except ValueError as exc:
        raise RuntimeError(
            "VLLM_UVM_FUSED_MOE_PRE_KERNEL_COLD_GATE_GRANULARITY_BYTES "
            f"must be an integer, got {raw_block_bytes!r}"
        ) from exc
    if block_bytes < 4096 or block_bytes % 4096:
        raise RuntimeError(
            "Cold CUDA-read gate granularity must be a positive 4 KiB "
            f"multiple, got {block_bytes}"
        )

    sample_bias_bytes = 4 * 1024
    binding_specs: list[
        tuple[str, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]
    ] = []
    residency_ranges: list[tuple[int, int]] = []
    total_cold_bytes = 0
    total_samples = 0
    for name in binding_names:
        binding = _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.get(name)
        if binding is None:
            raise RuntimeError(f"Cold CUDA-read gate lost binding {name!r}")
        expert_ids = tuple(int(value) for value in (binding.expert_ids or ()))
        source = binding.source
        if not expert_ids or source is None:
            continue
        parameter = binding.consumer
        if not parameter.is_cuda or not parameter.is_contiguous():
            raise RuntimeError(
                "Cold CUDA-read gate requires a contiguous CUDA Parameter: "
                f"{name!r}"
            )
        if source.device.type != "cpu" or not source.is_contiguous():
            raise RuntimeError(
                "Cold CUDA-read gate requires a contiguous CPU source: "
                f"{name!r}"
            )
        parameter_base = int(parameter.data_ptr())
        parameter_bytes = int(binding.parameter_bytes)
        source_bytes = int(source.numel() * source.element_size())
        expert_stride = int(binding.expert_stride_bytes)
        consumer_ranges = _uvm_original_alias_binding_consumer_ranges(binding)
        if len(consumer_ranges) != len(expert_ids):
            raise RuntimeError(
                "Cold CUDA-read gate found expert/range cardinality drift for "
                f"{name!r}"
            )

        consumer_indices: list[int] = []
        source_indices: list[int] = []
        for packed_index, (consumer_base, length) in enumerate(consumer_ranges):
            consumer_base = int(consumer_base)
            length = int(length)
            if length != expert_stride:
                raise RuntimeError(
                    "Cold CUDA-read gate found an unexpected expert stride for "
                    f"{name!r}: {length} != {expert_stride}"
                )
            residency_ranges.append((consumer_base, length))
            total_cold_bytes += length
            offset = 0
            while offset < length:
                intersection_end = min(length, offset + block_bytes)
                sample_offset = min(
                    offset + sample_bias_bytes,
                    intersection_end - 1,
                )
                consumer_index = consumer_base - parameter_base + sample_offset
                source_index = packed_index * expert_stride + sample_offset
                if not 0 <= consumer_index < parameter_bytes:
                    raise RuntimeError(
                        f"Cold consumer sample escapes {name!r}: {consumer_index}"
                    )
                if not 0 <= source_index < source_bytes:
                    raise RuntimeError(
                        f"Cold source sample escapes {name!r}: {source_index}"
                    )
                consumer_indices.append(consumer_index)
                source_indices.append(source_index)
                offset += block_bytes
        binding_specs.append(
            (
                name,
                parameter,
                source,
                tuple(consumer_indices),
                tuple(source_indices),
            )
        )
        total_samples += len(consumer_indices)

    cold_physical_residency: dict[str, Any] | None = None
    if residency_ranges:
        try:
            from vllm.device_allocator import uvm

            cold_physical_residency = dict(
                uvm.driver_batch_query_ranges_residency_bytes(
                    residency_ranges,
                    device=device,
                    wait_for_trackers=False,
                )
            )
        except Exception as exc:
            cold_physical_residency = {
                "ok": False,
                "error": repr(exc),
            }

    begin = {
        "operation": "current_cold_cuda_read_gate",
        "gate_stage": str(stage),
        "gate_status": "begin",
        "binding_count": len(binding_specs),
        "binding_names": tuple(name for name, *_ in binding_specs),
        "sample_count": int(total_samples),
        "cold_bytes": int(total_cold_bytes),
        "sample_granularity_bytes": int(block_bytes),
        "sample_bias_bytes": int(sample_bias_bytes),
        "verify_source_bytes": bool(verify_source_bytes),
        "cold_physical_residency": cold_physical_residency,
    }
    _append_uvm_original_alias_lifecycle(
        "cpu_offload_uvm_current_cold_cuda_read_gate_begin", begin
    )

    observed: dict[str, tuple[int, int]] = {}
    changed: list[str] = []
    with torch.cuda.device(device):
        torch.cuda.current_stream(device).synchronize()
        for name, parameter, source, consumer_indices, source_indices in binding_specs:
            parameter_bytes_view = parameter.view(torch.uint8).reshape(-1)
            gpu_index = torch.tensor(
                consumer_indices,
                dtype=torch.int64,
                device=parameter.device,
            )
            actual = torch.index_select(parameter_bytes_view, 0, gpu_index).cpu()
            if verify_source_bytes:
                source_bytes_view = source.view(torch.uint8).reshape(-1)
                cpu_index = torch.tensor(source_indices, dtype=torch.int64)
                expected = torch.index_select(source_bytes_view, 0, cpu_index)
                if not torch.equal(actual, expected):
                    changed.append(name)
            observed[name] = (
                len(consumer_indices),
                int(actual.to(torch.int64).sum().item()),
            )
        torch.cuda.synchronize(device)
    if changed:
        raise RuntimeError(
            "Cold CUDA-read gate detected bytes differing from pinned sources: "
            f"{changed!r}"
        )

    result = {
        **begin,
        "gate_status": "ok",
        "checksum_sum": int(sum(value[1] for value in observed.values())),
    }
    _append_uvm_original_alias_lifecycle(
        "cpu_offload_uvm_current_cold_cuda_read_gate_ok", result
    )
    return result


def _uvm_original_alias_arena_regions(
    arena: _UvmOriginalAliasArena,
) -> Iterable[tuple[int, int, int]]:
    """Yield (region_id, packed_offset, bytes), never crossing an expert."""
    offset = 0
    region_id = 0
    while offset < arena.requested_bytes:
        length = min(arena.region_bytes, arena.requested_bytes - offset)
        if arena.packed_slice_bytes:
            slice_remaining = arena.packed_slice_bytes - (
                offset % arena.packed_slice_bytes
            )
            length = min(length, slice_remaining)
        yield region_id, offset, length
        offset += length
        region_id += 1


def _create_uvm_original_alias_binding(
    *,
    full_parameter_name: str,
    parameter: Any,
    tensor: torch.Tensor,
    device: torch.device,
    requested_bytes: int,
) -> _UvmOriginalAliasBinding:
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_ENVELOPE_BYTES

    plan = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN.get(full_parameter_name)
    if plan is None:
        raise RuntimeError(
            "UVM original-VA parameter is missing from the precomputed plan: "
            f"{full_parameter_name!r}"
        )
    if not tensor.is_contiguous():
        raise RuntimeError(
            "UVM original-VA static alias currently requires contiguous "
            f"parameters: {full_parameter_name!r}"
        )
    if plan.consumer_count + 1 > 64:
        raise RuntimeError(
            "UVM original-VA bank exceeds the driver's 64-alias-per-slot "
            f"limit for {full_parameter_name!r}"
        )
    if requested_bytes <= 0 or requested_bytes % _UVM_ORIGINAL_ALIAS_PAGE_BYTES:
        raise RuntimeError(
            "UVM original-VA static alias cannot safely round a parameter "
            "outside its allocation; its exact byte size must be 4 KiB "
            f"aligned: {full_parameter_name!r} has {requested_bytes} bytes"
        )
    consumer_va = int(tensor.data_ptr())
    if consumer_va <= 0 or consumer_va % _UVM_ORIGINAL_ALIAS_PAGE_BYTES:
        raise RuntimeError(
            "UVM original-VA parameter allocation is not 4 KiB aligned: "
            f"{full_parameter_name!r} VA=0x{consumer_va:x}"
        )

    device_index = device.index
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    shape = tuple(int(value) for value in tensor.size())
    strides = tuple(int(value) for value in tensor.stride())
    (
        expert_ids,
        expert_stride_bytes,
        selected_bytes,
        upload_shape,
        upload_strides,
        consumer_ranges,
    ) = _uvm_original_alias_parameter_geometry(
        full_parameter_name=full_parameter_name,
        tensor=tensor,
        parameter_bytes=int(requested_bytes),
    )
    envelope_expert_count = _uvm_original_alias_envelope_expert_count(
        full_parameter_name,
        expert_ids,
    )
    envelope_bytes = int(selected_bytes)
    envelope_shape = upload_shape
    if expert_ids is not None:
        envelope_bytes = envelope_expert_count * int(expert_stride_bytes)
        envelope_shape = (envelope_expert_count, *shape[1:])
    signature = (
        shape,
        strides,
        tensor.dtype,
        int(device_index),
        int(envelope_bytes),
        envelope_shape,
        int(plan.consumer_count),
    )
    arena_key = (plan.layer_container, plan.local_name, plan.bank_id)
    arena = _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.get(arena_key)
    if arena is None:
        upload_storage, upload = _allocate_uvm_original_alias_upload(
            tensor=tensor,
            device=device,
            requested_bytes=int(envelope_bytes),
            upload_size=envelope_shape,
            upload_stride=upload_strides,
        )
        arena = _UvmOriginalAliasArena(
            upload_storage=upload_storage,
            upload=upload,
            signature=signature,
            requested_bytes=int(selected_bytes),
            aligned_bytes=int(envelope_bytes),
            backing_layer_id=_uvm_original_alias_backing_layer_id(
                len(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS)
            ),
            pte_alignment=0,
            pte_flags=0,
            region_bytes=_uvm_original_alias_region_bytes(),
            packed_slice_bytes=int(expert_stride_bytes),
            envelope_expert_count=int(envelope_expert_count),
        )
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS[arena_key] = arena
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES += int(selected_bytes)
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_ENVELOPE_BYTES += int(envelope_bytes)
    elif arena.signature != signature:
        raise RuntimeError(
            "UVM original-VA layers sharing a bank/local key have different "
            f"tensor signatures for {full_parameter_name!r}: expected "
            f"{arena.signature!r}, got {signature!r}"
        )
    elif arena.requested_bytes != int(selected_bytes):
        raise RuntimeError(
            "Layers sharing one stable upload arena selected different active "
            f"widths for {full_parameter_name!r}: arena={arena.requested_bytes}, "
            f"binding={selected_bytes}"
        )
    if plan.consumer_index in arena.consumers:
        raise RuntimeError(
            "UVM original-VA plan assigned the same consumer index twice for "
            f"{full_parameter_name!r}"
        )
    binding = _UvmOriginalAliasBinding(
        full_parameter_name=full_parameter_name,
        arena=arena,
        consumer=tensor,
        parameter=parameter,
        plan=plan,
        requested_bytes=int(selected_bytes),
        parameter_bytes=int(requested_bytes),
        expert_ids=expert_ids,
        expert_stride_bytes=int(expert_stride_bytes),
        consumer_ranges=consumer_ranges,
    )
    arena.consumers[plan.consumer_index] = binding
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS[full_parameter_name] = binding
    return binding


def _uvm_original_alias_select_pte_mode() -> tuple[int, int, str]:
    from vllm.device_allocator.uvm_static_alias import (
        UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K,
    )

    target_ranges = [
        (int(arena.upload.data_ptr()), int(arena.requested_bytes))
        for arena in _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.values()
    ]
    target_ranges.extend(
        consumer_range
        for binding in _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values()
        for consumer_range in binding.consumer_ranges
    )
    if not target_ranges:
        raise RuntimeError("UVM original-VA static-alias topology is empty")
    page_aligned = all(
        base % _UVM_ORIGINAL_ALIAS_PAGE_BYTES == 0
        and length % _UVM_ORIGINAL_ALIAS_PAGE_BYTES == 0
        for base, length in target_ranges
    )
    if not page_aligned:
        raise RuntimeError(
            "UVM original-VA topology contains a target outside a complete "
            "4 KiB PTE envelope"
        )
    envelope = _UVM_ORIGINAL_ALIAS_64K_ENVELOPE_BYTES
    supports_64k = all(
        base % envelope == 0 and length % envelope == 0
        for base, length in target_ranges
    )
    requested = _uvm_original_alias_requested_pte_mode()
    if requested == "64k" and not supports_64k:
        raise RuntimeError(
            "64K PTE mode was requested, but at least one original parameter "
            "does not cover a complete 2 MiB-aligned VA-block envelope"
        )
    use_64k = supports_64k and requested != "4k"
    return (
        _UVM_ORIGINAL_ALIAS_64K_PTE_BYTES
        if use_64k
        else _UVM_ORIGINAL_ALIAS_PAGE_BYTES,
        UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_PTE_64K if use_64k else 0,
        "64K" if use_64k else "4K",
    )


def _uvm_original_alias_map_entries(
    pte_flags: int,
) -> list[tuple[int, int, int, int, int, int, int]]:
    from vllm.device_allocator.uvm_static_alias import (
        UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW,
    )

    entries: list[tuple[int, int, int, int, int, int, int]] = []
    for _, arena in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.items()):
        if len(arena.consumers) != int(arena.signature[-1]):
            raise RuntimeError(
                "UVM original-VA arena is missing one or more planned consumer "
                f"VAs for backing layer {arena.backing_layer_id}"
            )
        for region_id, offset, length in _uvm_original_alias_arena_regions(arena):
            if length % arena.pte_alignment:
                raise RuntimeError(
                    "UVM original-VA final region does not match the selected "
                    "PTE granularity"
                )
            identity = (arena.backing_layer_id, region_id, region_id)
            entries.append(
                (
                    *identity,
                    UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW | pte_flags,
                    int(arena.upload.data_ptr()) + offset,
                    length,
                    0,
                )
            )
            for _, binding in sorted(arena.consumers.items()):
                consumer_va = int(binding.consumer.data_ptr()) + offset
                if binding.expert_ids is not None:
                    if binding.expert_stride_bytes <= 0:
                        raise RuntimeError(
                            "Expert-slice alias has no positive expert stride"
                        )
                    packed_expert = offset // binding.expert_stride_bytes
                    expert_offset = offset % binding.expert_stride_bytes
                    if (
                        packed_expert >= len(binding.expert_ids)
                        or expert_offset + length > binding.expert_stride_bytes
                    ):
                        raise RuntimeError(
                            "A packed alias region crosses an expert boundary for "
                            f"{binding.full_parameter_name!r}"
                        )
                    consumer_va = (
                        int(binding.consumer.data_ptr())
                        + binding.expert_ids[packed_expert]
                        * binding.expert_stride_bytes
                        + expert_offset
                    )
                entries.append(
                    (
                        *identity,
                        pte_flags,
                        consumer_va,
                        length,
                        0,
                    )
                )
    return entries


def _uvm_original_alias_preallocate_entries() -> list[
    tuple[int, int, int, int, int, int, int]
]:
    from vllm.device_allocator.uvm import UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT

    entries: list[tuple[int, int, int, int, int, int, int]] = []
    for _, arena in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.items()):
        for region_id, offset, length in _uvm_original_alias_arena_regions(arena):
            entries.append(
                (
                    int(arena.upload.data_ptr()) + offset,
                    length,
                    int(arena.backing_layer_id),
                    region_id,
                    region_id,
                    0,
                    UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT,
                )
            )
    return entries


def _append_uvm_original_alias_lifecycle(event: str, result: dict[str, Any]) -> None:
    _append_cpu_offload_manifest(
        {
            "event": event,
            "timestamp": time.time(),
            "pid": os.getpid(),
            "backend": "uvm_original_va",
            "state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
            "session_id": int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
            **result,
        }
    )


def _uvm_original_alias_destroy_cleared(result: Mapping[str, Any]) -> bool:
    """Trust cmd99's authoritative empty-session tuple, even on mismatch."""
    return bool(result.get("ok")) or bool(
        result.get("cache_cleared")
        and result.get("session_cleared")
        and int(result.get("remaining_alias_count") or 0) == 0
        and int(result.get("remaining_alias_bytes") or 0) == 0
        and int(result.get("remaining_backing_bytes") or 0) == 0
    )


def _uvm_original_alias_next_session_id(
    result: Mapping[str, Any], requested_session_id: int
) -> int:
    """Validate one MAP response before trusting its returned session id."""
    if not bool(result.get("ok")):
        raise RuntimeError(f"MAP did not succeed: {dict(result)!r}")
    raw_session_id = result.get("session_id")
    if (
        not isinstance(raw_session_id, int)
        or isinstance(raw_session_id, bool)
        or raw_session_id <= 0
    ):
        raise RuntimeError(
            "A successful UVM original-VA MAP returned an invalid session id"
        )
    returned_session_id = int(raw_session_id)
    if requested_session_id not in (0, returned_session_id):
        raise RuntimeError(
            "UVM original-VA MAP changed session id while appending: "
            f"requested={requested_session_id}, returned={returned_session_id}"
        )
    return returned_session_id


def _uvm_original_alias_close_slot_session(
    uvm: Any, *, device: int
) -> tuple[bool, dict[str, Any] | None]:
    """Close mode-8 backing state while retaining a structured retry result."""
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT

    if device not in uvm._uvm_slot_direct_sessions:
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT = None
        return True, None
    try:
        raw_result = uvm.close_uvm_slot_direct_session(device=device)
        result = dict(raw_result)
    except Exception as exc:
        result = {
            "ok": False,
            "reason": "slot_session_close_exception",
            "closed": False,
            "error": repr(exc),
        }
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT = result
    return bool(result.get("ok")), result


def finalize_cpu_offload_graph_static_alias(
    *,
    known_cpu_ranges: Iterable[tuple[int, int]] = (),
) -> dict[str, Any]:
    """Build and seal original-parameter-VA aliases after weights are loaded.

    This must run before memory profiling or any CUDA Graph capture. It is a
    no-op for disabled and CUDA-VMM configurations so the model-runner hook can
    be unconditional.
    """
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_PREALLOCATE_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_MAP_RESULTS
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SEAL_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_ORIGINAL_RELEASE_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_COLD_RESIDENCY_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_RESIDENCY_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
    global _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
    global _CPU_OFFLOAD_UVM_PLAN_CLASS_SOURCE_PREFILL_RESULT
    global _CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT

    if not _CPU_OFFLOAD_GRAPH_STATIC_ALIAS:
        return {
            "ok": True,
            "operation": "finalize",
            "reason": "disabled",
            "backend": None,
        }
    if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND != "uvm_original_va":
        return {
            "ok": True,
            "operation": "finalize",
            "reason": "vmm_backend_noop",
            "backend": _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND,
        }

    with _CPU_OFFLOAD_UVM_STATIC_ALIAS_LOCK:
        if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "sealed":
            return {
                "ok": True,
                "operation": "finalize",
                "reason": "already_sealed",
                "backend": "uvm_original_va",
                "state": "sealed",
                "sealed": True,
                "session_id": int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
            }
        if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE not in ("collecting", "remote"):
            raise RuntimeError(
                "Cannot finalize UVM original-VA aliases from lifecycle state "
                f"{_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE!r}"
            )
        _require_all_copy_exact_names_seen()
        if set(_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS) != set(
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN
        ):
            missing = sorted(
                set(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN)
                - set(_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS)
            )
            raise RuntimeError(
                "UVM original-VA bindings do not cover the exact-name plan; "
                f"missing={missing[:8]!r}"
            )

        from vllm.device_allocator import uvm, uvm_static_alias

        device = int(_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0)
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "finalizing"
        _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT = None
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT = None
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_ORIGINAL_RELEASE_RESULT = None
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_COLD_RESIDENCY_RESULT = None
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_RESIDENCY_RESULT = None
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED = False
        session_id = 0
        map_results: list[dict[str, Any]] = []
        map_side_effects = False
        map_session_unresolved = False
        profile_hot_shadow_cutover: dict[str, Any] | None = None
        try:
            pte_alignment, pte_flags, pte_mode = _uvm_original_alias_select_pte_mode()
            for arena in _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.values():
                arena.pte_alignment = pte_alignment
                arena.pte_flags = pte_flags
                if arena.region_bytes % pte_alignment:
                    raise RuntimeError(
                        "Configured UVM static-alias region size is not a "
                        f"multiple of the selected {pte_mode} PTE granularity"
                    )
                managed, reason, code = uvm.probe_driver_capable_range(
                    int(arena.upload.data_ptr()), int(arena.aligned_bytes)
                )
                if not managed:
                    raise RuntimeError(
                        "UVM original-VA upload arena is not a driver-capable "
                        f"managed allocation: reason={reason}, code={code}"
                    )

            source_created_count = 0
            source_reused_count = 0
            for binding in _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values():
                current = binding.parameter.data
                if int(current.data_ptr()) != int(binding.consumer.data_ptr()):
                    raise RuntimeError(
                        "A selected expert parameter changed allocation VA "
                        "after construction: "
                        f"{binding.full_parameter_name!r}"
                    )
                for range_base, range_bytes in (
                    _uvm_original_alias_binding_consumer_ranges(binding)
                ):
                    managed, reason, code = uvm.probe_driver_capable_range(
                        int(range_base), int(range_bytes)
                    )
                    if not managed:
                        raise RuntimeError(
                            "Selected expert range is not a driver-capable "
                            "managed allocation despite "
                            "VLLM_UVM_EXPERT_WEIGHTS_MANAGED=1: "
                            f"{binding.full_parameter_name!r}, "
                            f"range=0x{range_base:x}+{range_bytes}, "
                            f"reason={reason}, code={code}"
                        )
                source = binding.source
                if source is None:
                    source = torch.empty_strided(
                        size=binding.arena.upload.size(),
                        stride=binding.arena.upload.stride(),
                        dtype=current.dtype,
                        layout=current.layout,
                        device="cpu",
                        pin_memory=True,
                    )
                    if not source.is_pinned():
                        raise RuntimeError(
                            "UVM original-VA source allocation is not pinned for "
                            f"{binding.full_parameter_name!r}"
                        )
                    binding.source = source
                    source_created_count += 1
                else:
                    if (
                        not source.is_pinned()
                        or tuple(source.size())
                        != tuple(binding.arena.upload.size())
                        or tuple(source.stride())
                        != tuple(binding.arena.upload.stride())
                        or source.dtype != current.dtype
                    ):
                        raise RuntimeError(
                            "A retained UVM original-VA pinned source changed "
                            f"geometry for {binding.full_parameter_name!r}"
                        )
                    source_reused_count += 1
                if binding.source_refill_required:
                    _refill_uvm_original_alias_binding_source(binding)
                else:
                    # The non-stable-VA E5 baseline populated the new managed
                    # Parameter directly from this pinned source.  Reading the
                    # CPU-resident managed VA back through a CUDA D2H copy is
                    # redundant and violates fail-fast remote-access policy.
                    # This certificate is one-shot; later generations return
                    # to the normal source refresh contract.
                    binding.source_refill_required = True
            _CPU_OFFLOAD_UVM_PLAN_CLASS_SOURCE_PREFILL_RESULT = (
                _prefill_uvm_original_alias_plan_class_sources()
            )
            torch.cuda.synchronize(device)
            _CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT = (
                _prepare_uvm_original_alias_plan_class_immutable_shadows(
                    device=device
                )
            )

            _uvm_original_alias_hot_cuda_read_gate(
                stage="before_original_release",
                device=device,
            )

            original_release = (
                _release_uvm_original_alias_consumer_gpu_chunks(
                    device=device,
                    known_cpu_ranges=known_cpu_ranges,
                )
            )
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_ORIGINAL_RELEASE_RESULT = (
                original_release
            )
            _uvm_original_alias_hot_cuda_read_gate(
                stage="after_original_release",
                device=device,
            )

            shadow_certificate = _CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT
            full_parameter_shadow = bool(
                isinstance(shadow_certificate, dict)
                and shadow_certificate.get("enabled") is True
                and shadow_certificate.get("full_parameter_shadow") is True
            )
            profile_hot_ranges = [
                hot_range
                for binding in _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values()
                for hot_range in _uvm_original_alias_binding_hot_ranges(binding)
            ]
            profile_hot_bytes = sum(
                length for _, length in profile_hot_ranges
            )
            if (
                full_parameter_shadow
                and _strict_env_flag(
                    "VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE", False
                )
                and _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE
                == "prefill_graph_remote_decode"
                and profile_hot_ranges
            ):
                # D needs every expert remotely readable for its initial
                # maximum-token profile.  Perform the hot shadow cutover
                # before SEAL constructs the current-tier Prefill/Decode
                # children. Cutting hot pages over after SEAL would leave a
                # detached child referring to released GPU chunks.
                profile_hot_shadow_cutover = dict(
                    uvm.driver_cutover_ranges_to_immutable_cpu_shadow(
                        profile_hot_ranges,
                        device=device,
                    )
                )
                if not (
                    profile_hot_shadow_cutover.get("ok") is True
                    and int(
                        profile_hot_shadow_cutover.get("range_count", -1)
                    )
                    == len(profile_hot_ranges)
                    and int(
                        profile_hot_shadow_cutover.get("success_count", -1)
                    )
                    == len(profile_hot_ranges)
                    and int(
                        profile_hot_shadow_cutover.get("failure_count", -1)
                    )
                    == 0
                    and int(
                        profile_hot_shadow_cutover.get(
                            "requested_bytes", -1
                        )
                    )
                    == profile_hot_bytes
                    and int(
                        profile_hot_shadow_cutover.get(
                            "released_gpu_bytes", -1
                        )
                    )
                    == profile_hot_bytes
                    and int(
                        profile_hot_shadow_cutover.get(
                            "retained_cpu_shadow_bytes", -1
                        )
                    )
                    == profile_hot_bytes
                    and int(
                        profile_hot_shadow_cutover.get(
                            "remote_mapped_bytes", -1
                        )
                    )
                    == profile_hot_bytes
                ):
                    raise RuntimeError(
                        "Pre-SEAL all-remote hot shadow cutover was "
                        f"incomplete: {profile_hot_shadow_cutover!r}"
                    )
                _uvm_original_alias_hot_cuda_read_gate(
                    stage="after_profile_hot_shadow_cutover",
                    device=device,
                )

            current_original_vas = tuple(
                int(binding.consumer.data_ptr())
                for _, binding in sorted(
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                )
            )
            if (
                _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
                and current_original_vas != _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
            ):
                raise RuntimeError(
                    "An original expert Parameter VA changed between topology "
                    "transitions"
                )
            _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT = current_original_vas

            cold_residency_ranges = [
                consumer_range
                for binding in _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values()
                for consumer_range in (
                    _uvm_original_alias_binding_consumer_ranges(binding)
                )
            ]
            # The immutable-shadow cutover above has already proved that every
            # cold consumer page is CPU-resident and GPU-readable. Reapplying
            # PreferredLocation/AccessedBy to thousands of individual expert
            # slices would split the whole-Parameter managed range at each
            # expert boundary. Those artificial VA-range edges prevent one
            # complete 2 MiB child table from covering a mixed cold/hot block.
            # Only upload arenas still need native remote preparation here.
            residency_ranges = [] if full_parameter_shadow else list(
                cold_residency_ranges
            )
            residency_ranges.extend(
                (int(arena.upload.data_ptr()), int(arena.aligned_bytes))
                for arena in _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.values()
            )
            prepared = uvm.batch_prepare_ranges_for_native_remote_access_with_executor(
                residency_ranges,
                sync=True,
                device=device,
            )
            if int(prepared[0]) != len(residency_ranges):
                raise RuntimeError(
                    "Failed to make every original/upload managed range CPU "
                    f"resident before MAP: result={prepared!r}"
                )
            _uvm_original_alias_hot_cuda_read_gate(
                stage="after_native_remote_prepare",
                device=device,
            )

            cold_residency_bytes = sum(
                length for _, length in cold_residency_ranges
            )
            hot_residency_ranges = [
                hot_range
                for binding in _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values()
                for hot_range in _uvm_original_alias_binding_hot_ranges(binding)
            ]
            hot_residency_bytes = sum(
                length for _, length in hot_residency_ranges
            )
            expert_slice_topology = any(
                binding.expert_ids is not None
                for binding in _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values()
            )
            cold_residency: dict[str, Any] | None = None
            hot_residency: dict[str, Any] | None = None
            if expert_slice_topology:
                cold_residency = dict(
                    uvm.driver_batch_query_ranges_residency_bytes(
                        cold_residency_ranges,
                        device=device,
                        wait_for_trackers=False,
                    )
                )
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_COLD_RESIDENCY_RESULT = (
                    cold_residency
                )
                if not (
                    cold_residency.get("ok") is True
                    and int(cold_residency.get("queried_bytes", -1))
                    == cold_residency_bytes
                    and int(cold_residency.get("cpu_bytes", -1))
                    == cold_residency_bytes
                    and int(cold_residency.get("gpu_bytes", -1)) == 0
                    and int(cold_residency.get("any_bytes", -1))
                    == cold_residency_bytes
                    and int(cold_residency.get("unknown_bytes", -1)) == 0
                ):
                    raise RuntimeError(
                        "Cold expert residency was not exclusively CPU-backed "
                        f"before static-alias MAP: {cold_residency!r}"
                    )

                hot_residency = dict(
                    uvm.driver_batch_query_ranges_residency_bytes(
                        hot_residency_ranges,
                        device=device,
                        wait_for_trackers=False,
                    )
                )
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_HOT_RESIDENCY_RESULT = hot_residency
                expected_hot_cpu_bytes = (
                    _uvm_original_alias_expected_shadow_cpu_bytes(
                        hot_residency_ranges
                    )
                )
                expected_hot_gpu_bytes = (
                    0
                    if profile_hot_shadow_cutover is not None
                    else hot_residency_bytes
                )
                if hot_residency_ranges and not (
                    hot_residency.get("ok") is True
                    and int(hot_residency.get("queried_bytes", -1))
                    == hot_residency_bytes
                    and int(hot_residency.get("gpu_bytes", -1))
                    == expected_hot_gpu_bytes
                    and int(hot_residency.get("cpu_bytes", -1))
                    == expected_hot_cpu_bytes
                    and int(hot_residency.get("any_bytes", -1))
                    == hot_residency_bytes
                    and int(hot_residency.get("unknown_bytes", -1)) == 0
                ):
                    raise RuntimeError(
                        "Hot expert residency changed while preparing cold slices: "
                        f"{hot_residency!r}"
                    )

            preallocate_entries = _uvm_original_alias_preallocate_entries()
            preallocate = uvm.driver_slot_preallocate_ranges(
                preallocate_entries, device=device
            )
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_PREALLOCATE_RESULT = preallocate
            if not bool(preallocate.get("ok")):
                raise RuntimeError(
                    f"UVM original-VA backing preallocation failed: {preallocate!r}"
                )
            _uvm_original_alias_hot_cuda_read_gate(
                stage="after_slot_preallocate",
                device=device,
            )

            map_entries = _uvm_original_alias_map_entries(pte_flags)
            maximum = uvm_static_alias.UVM_MOE_STATIC_ALIAS_MAX_ENTRIES
            for offset in range(0, len(map_entries), maximum):
                mapped = uvm_static_alias.driver_static_alias_map(
                    map_entries[offset : offset + maximum],
                    session_id=session_id,
                    device=device,
                )
                map_results.append(mapped)
                raw_returned_session_id = mapped.get("session_id")
                returned_session_id = (
                    int(raw_returned_session_id)
                    if isinstance(raw_returned_session_id, int)
                    and not isinstance(raw_returned_session_id, bool)
                    and raw_returned_session_id > 0
                    else 0
                )
                map_side_effects = (
                    map_side_effects
                    or any(
                        bool(mapped.get(field))
                        for field in (
                            "ok",
                            "made_progress",
                            "poisoned",
                            "side_effect_indeterminate",
                        )
                    )
                    or returned_session_id > 0
                )
                if (
                    session_id == 0
                    and not bool(mapped.get("ok"))
                    and (
                        bool(mapped.get("made_progress"))
                        or bool(mapped.get("poisoned"))
                        or bool(mapped.get("side_effect_indeterminate"))
                        or returned_session_id > 0
                    )
                ):
                    # A failed first MAP cannot make its returned id trusted.
                    # Keep the backing fd alive and require process teardown.
                    map_session_unresolved = True
                if session_id > 0 and returned_session_id not in (0, session_id):
                    # Do not issue cmd99 against an unexpected id. The known
                    # session remains cleanable, but the contradictory response
                    # means another possible side effect cannot be disproved.
                    map_session_unresolved = True
                try:
                    next_session_id = _uvm_original_alias_next_session_id(
                        mapped, session_id
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        "UVM original-VA MAP failed at batch "
                        f"{len(map_results) - 1}: {mapped!r}"
                    ) from exc
                session_id = next_session_id
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID = session_id
                _uvm_original_alias_hot_cuda_read_gate(
                    stage=f"after_map_batch_{len(map_results)}",
                    device=device,
                )
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_MAP_RESULTS = map_results

            expected_alias_count = len(map_entries)
            expected_alias_bytes = sum(entry[5] for entry in map_entries)
            seal = uvm_static_alias.driver_static_alias_seal(
                session_id,
                expected_alias_count=expected_alias_count,
                expected_alias_bytes=expected_alias_bytes,
                device=device,
            )
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_SEAL_RESULT = seal
            if not bool(seal.get("ok")):
                raise RuntimeError(f"UVM original-VA SEAL failed: {seal!r}")
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "sealed"
            _uvm_original_alias_hot_cuda_read_gate(
                stage="after_seal",
                device=device,
            )
            result = {
                "ok": True,
                "operation": "finalize",
                "reason": "ok",
                "backend": "uvm_original_va",
                "state": "sealed",
                "sealed": True,
                "session_id": session_id,
                "pte_mode": pte_mode,
                "region_bytes": int(
                    next(
                        iter(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.values())
                    ).region_bytes
                ),
                "backing_bytes": int(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES),
                "preallocate_count": len(preallocate_entries),
                "alias_count": expected_alias_count,
                "alias_bytes": expected_alias_bytes,
                "map_batch_count": len(map_results),
                "source_created_count": int(source_created_count),
                "source_reused_count": int(source_reused_count),
                "plan_class_source_prefill": (
                    _CPU_OFFLOAD_UVM_PLAN_CLASS_SOURCE_PREFILL_RESULT
                ),
                "plan_class_immutable_cpu_shadow": (
                    _CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT
                ),
                "profile_hot_shadow_cutover": profile_hot_shadow_cutover,
                "original_release": original_release,
                "cold_residency": cold_residency,
                "hot_residency": hot_residency,
                "cold_residency_range_count": len(cold_residency_ranges),
                "cold_residency_bytes": int(cold_residency_bytes),
                "hot_residency_range_count": len(hot_residency_ranges),
                "hot_residency_bytes": int(hot_residency_bytes),
                "native_remote_prepare_skipped_cold_ranges": bool(
                    full_parameter_shadow
                ),
                "upload_vas": tuple(
                    int(arena.upload.data_ptr())
                    for _, arena in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.items())
                ),
                "original_parameter_vas": tuple(
                    int(binding.consumer.data_ptr())
                    for _, binding in sorted(
                        _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                    )
                ),
                "pinned_source_vas": tuple(
                    int(binding.source.data_ptr())
                    for _, binding in sorted(
                        _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                    )
                    if binding.source is not None
                ),
            }
            _append_uvm_original_alias_lifecycle(
                "cpu_offload_uvm_static_alias_finalized", result
            )
            return result
        except Exception:
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_MAP_RESULTS = map_results
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "cleanup_pending"
            active_session_id = int(
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID or session_id
            )
            aliases_cleared = not map_side_effects
            cleanup_unresolved = bool(map_session_unresolved)
            if active_session_id:
                try:
                    cleanup = uvm_static_alias.driver_static_alias_destroy(
                        active_session_id,
                        device=device,
                        release_backing=True,
                        synchronize=True,
                    )
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT = cleanup
                    aliases_cleared = _uvm_original_alias_destroy_cleared(cleanup)
                    if aliases_cleared:
                        _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID = 0
                except Exception as cleanup_exc:
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT = {
                        "ok": False,
                        "operation": "destroy",
                        "reason": "destroy_exception",
                        "session_id": active_session_id,
                        "error": repr(cleanup_exc),
                    }
                    logger.exception(
                        "Failed to clean up a partially finalized UVM "
                        "original-VA static-alias session"
                    )
            elif map_side_effects:
                # There was no response from which a session id could safely
                # be adopted. Closing mode-8 backing could leave a live alias
                # pointing at released memory, so retain every resource until
                # CUDA/UVM process teardown.
                cleanup_unresolved = True

            _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED = cleanup_unresolved
            if aliases_cleared and not cleanup_unresolved:
                slot_closed, _ = _uvm_original_alias_close_slot_session(
                    uvm, device=device
                )
                if slot_closed:
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "remote"
                else:
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "cleanup_pending"
            else:
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "cleanup_pending"
            raise


def _normalize_uvm_dual_graph_entries(
    entries: Iterable[Mapping[str, Any]],
) -> tuple[tuple[int, bool, bool], ...]:
    normalized: list[tuple[int, bool, bool]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("CUDA Graph entry certificates must be mappings")
        num_tokens = entry.get("num_tokens")
        uniform_decode = entry.get("uniform_decode")
        has_lora = entry.get("has_lora", False)
        if not isinstance(num_tokens, int) or isinstance(num_tokens, bool):
            raise TypeError("CUDA Graph entry num_tokens must be an int")
        if num_tokens <= 0:
            raise ValueError("CUDA Graph entry num_tokens must be positive")
        if not isinstance(uniform_decode, bool) or not isinstance(has_lora, bool):
            raise TypeError(
                "CUDA Graph entry uniform_decode/has_lora fields must be bool"
            )
        normalized.append((num_tokens, uniform_decode, has_lora))
    if len(normalized) != len(set(normalized)):
        raise ValueError("CUDA Graph entry certificate contains duplicates")
    return tuple(sorted(normalized))


def certify_cpu_offload_uvm_original_va_dual_graph_partition(
    *,
    remote_entries: Iterable[Mapping[str, Any]],
    alias_entries: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind independently captured Graph entries to their PTE topologies."""
    global _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE

    if _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE != (
        "prefill_graph_remote_decode"
    ):
        raise RuntimeError(
            "A dual Graph partition certificate requires "
            "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=prefill_graph_remote_decode"
        )
    remote = _normalize_uvm_dual_graph_entries(remote_entries)
    alias = _normalize_uvm_dual_graph_entries(alias_entries)
    if not remote or not alias:
        raise RuntimeError("Both remote and alias Graph entry sets are required")
    if any(not uniform for _, uniform, _ in remote):
        raise RuntimeError("Every remote Graph entry must be uniform-decode")
    if any(uniform for _, uniform, _ in alias):
        raise RuntimeError("Every alias Graph entry must be non-uniform")
    if not _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED or not isinstance(
        _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE, dict
    ):
        raise RuntimeError("Remote Graph capture certificate is absent")
    if not _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED or not isinstance(
        _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE, dict
    ):
        raise RuntimeError("Alias Graph capture certificate is absent")
    current_vas = _uvm_original_parameter_va_snapshot()
    if tuple(
        _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE.get("parameter_vas", ())
    ) != current_vas or tuple(
        _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE.get("parameter_vas", ())
    ) != current_vas:
        raise RuntimeError("Dual Graph capture Parameter VA fingerprints differ")
    certificate = {
        "schema": "uvm_original_va_dual_graph_partition_v1",
        "phase_mode": _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE,
        "parameter_vas": current_vas,
        "remote_entries": remote,
        "alias_entries": alias,
        "remote_entry_count": len(remote),
        "alias_entry_count": len(alias),
    }
    _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE = certificate
    _append_cpu_offload_manifest(
        {
            "event": "cpu_offload_uvm_original_va_dual_graph_partition",
            "timestamp": time.time(),
            "pid": os.getpid(),
            **certificate,
        }
    )
    return dict(certificate)


def begin_cpu_offload_uvm_plan_class_graph_capture(class_name: str) -> None:
    """Open an independent certificate epoch while older GraphExecs stay live."""
    global _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY
    global _CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE
    global _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE

    normalized = _normalize_uvm_plan_class_name(class_name)
    if normalized not in _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS:
        raise RuntimeError(f"Plan class {normalized!r} is not registered")
    if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE:
        raise RuntimeError("Cannot begin a plan-class capture during CUDA capture")
    _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY = None
    _CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT = 0
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED = False
    _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED = False
    _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED = False
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT = 0
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE = None
    _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE = None
    _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE = None


def register_cpu_offload_uvm_plan_class_graph_certificates(
    class_name: str,
) -> dict[str, Any]:
    normalized = _normalize_uvm_plan_class_name(class_name)
    if not (
        _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED
        and _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED
        and isinstance(_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE, dict)
        and isinstance(_CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE, dict)
        and isinstance(_CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE, dict)
    ):
        raise RuntimeError(
            f"Plan class {normalized!r} has no complete dual-Graph certificate"
        )
    bundle = {
        "capture_topology": _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY,
        "capture_success_count": int(_CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT),
        "remote_graph_captured": True,
        "alias_graph_captured": True,
        "alias_graphs_destroyed": False,
        "remote_graph_remaining_count": int(
            _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT
        ),
        "remote": dict(_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE),
        "alias": dict(_CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE),
        "partition": dict(_CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE),
        "parameter_vas": _uvm_original_parameter_va_snapshot(),
        "upload_vas": tuple(
            int(arena.upload.data_ptr())
            for _, arena in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.items())
        ),
        "source_vas": tuple(
            int(binding.source.data_ptr())
            for _, binding in sorted(
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
            )
            if binding.source is not None
        ),
    }
    _CPU_OFFLOAD_UVM_GRAPH_CERTIFICATES_BY_CLASS[normalized] = bundle
    return dict(bundle)


def activate_cpu_offload_uvm_plan_class_graph_certificates(
    class_name: str,
) -> dict[str, Any]:
    global _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY
    global _CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE
    global _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE

    normalized = _normalize_uvm_plan_class_name(class_name)
    bundle = _CPU_OFFLOAD_UVM_GRAPH_CERTIFICATES_BY_CLASS.get(normalized)
    if bundle is None:
        return {"ok": False, "reason": "class_certificate_not_registered"}
    current_parameter_vas = _uvm_original_parameter_va_snapshot()
    current_upload_vas = tuple(
        int(arena.upload.data_ptr())
        for _, arena in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.items())
    )
    current_source_vas = tuple(
        int(binding.source.data_ptr())
        for _, binding in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items())
        if binding.source is not None
    )
    if (
        tuple(bundle["parameter_vas"]) != current_parameter_vas
        or tuple(bundle["upload_vas"]) != current_upload_vas
        or tuple(bundle["source_vas"]) != current_source_vas
    ):
        return {"ok": False, "reason": "class_certificate_va_mismatch"}
    _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY = bundle["capture_topology"]
    _CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT = int(bundle["capture_success_count"])
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED = True
    _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED = True
    _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED = False
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT = int(
        bundle["remote_graph_remaining_count"]
    )
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE = dict(bundle["remote"])
    _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE = dict(bundle["alias"])
    _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE = dict(bundle["partition"])
    return {"ok": True, "reason": "class_certificate_activated"}


def certify_cpu_offload_uvm_original_va_alias_graphs_destroyed(
    *,
    destroyed_alias_entries: Iterable[Mapping[str, Any]],
    remaining_remote_entries: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Certify selective alias-Graph reset before cmd99 frees its backing."""
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT

    certificate = _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE
    if not isinstance(certificate, dict):
        raise RuntimeError("Dual Graph partition certificate is absent")
    if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE != "sealed":
        raise RuntimeError(
            "Alias Graphs may only be retired while alias PTEs are sealed"
        )
    destroyed = _normalize_uvm_dual_graph_entries(destroyed_alias_entries)
    remaining = _normalize_uvm_dual_graph_entries(remaining_remote_entries)
    expected_alias = tuple(certificate.get("alias_entries", ()))
    expected_remote = tuple(certificate.get("remote_entries", ()))
    if destroyed != expected_alias:
        raise RuntimeError(
            "Destroyed alias Graph entries differ from the capture certificate"
        )
    if remaining != expected_remote:
        raise RuntimeError(
            "Remaining remote Graph entries differ from the capture certificate"
        )
    if not remaining:
        raise RuntimeError("No remote decode Graph remains after alias teardown")
    current_vas = _uvm_original_parameter_va_snapshot()
    if tuple(certificate.get("parameter_vas", ())) != current_vas:
        raise RuntimeError("Parameter VA changed before alias Graph teardown")
    _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED = True
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT = len(remaining)
    result = {
        "schema": "uvm_original_va_alias_graph_teardown_v1",
        "event": "cpu_offload_uvm_original_va_alias_graphs_destroyed",
        "timestamp": time.time(),
        "pid": os.getpid(),
        "ok": True,
        "destroyed_alias_entries": destroyed,
        "destroyed_alias_graph_count": len(destroyed),
        "remaining_remote_entries": remaining,
        "remaining_remote_graph_count": len(remaining),
        "parameter_vas": current_vas,
    }
    _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(result)
    _append_cpu_offload_manifest(result)
    return dict(result)


def _uvm_original_alias_can_preserve_remote_graphs(
    *, synchronize: bool
) -> tuple[bool, str]:
    if _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE not in (
        "prefill_remote_decode",
        "prefill_graph_remote_decode",
    ):
        return False, "phase_mode_is_not_hybrid"
    if not _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED:
        return False, "no_successful_remote_graph_capture"
    if (
        _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE
        == "prefill_graph_remote_decode"
    ):
        if not _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED:
            return False, "alias_graph_capture_certificate_missing"
        if not _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED:
            return False, "alias_prefill_graphs_are_still_live"
        if _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT <= 0:
            return False, "no_certified_remote_graph_remains"
    certificate = _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE
    if not isinstance(certificate, dict):
        return False, "remote_graph_certificate_missing"
    if certificate.get("topology") != "remote":
        return False, "remote_graph_certificate_topology_mismatch"
    if int(certificate.get("capture_forward_count", 0)) <= 0:
        return False, "remote_graph_certificate_has_no_forwards"
    if int(certificate.get("actual_h2d_bytes", -1)) != 0:
        return False, "remote_graph_certificate_contains_h2d"
    try:
        current_vas = _uvm_original_parameter_va_snapshot()
    except Exception:
        return False, "current_parameter_va_snapshot_failed"
    if tuple(certificate.get("parameter_vas", ())) != current_vas:
        return False, "remote_graph_parameter_va_fingerprint_changed"
    if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE:
        return False, "cuda_graph_capture_is_active"
    if not synchronize:
        return False, "preserved_graph_transition_requires_synchronize"
    return True, "ok"


def _destroy_vmm_graph_static_alias(
    *,
    graphs_destroyed: bool,
    synchronize: bool,
    remote_graphs_preserved: bool,
) -> dict[str, Any]:
    """Release every CUDA VMM alias arena after its Graph owners are gone.

    DLPack tensors keep native live references to each allocation. Break the
    upload/consumer references owned by this integration before asking the
    allocator to synchronize and unmap. A failed arena remains registered so
    teardown can be retried; successfully closed arenas are never presented to
    the native allocator twice.
    """
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES

    if (
        not isinstance(graphs_destroyed, bool)
        or not isinstance(synchronize, bool)
        or not isinstance(remote_graphs_preserved, bool)
        or remote_graphs_preserved
    ):
        return {
            "ok": False,
            "operation": "destroy",
            "reason": "validation_failed",
            "error": (
                "VMM teardown requires unambiguous bool arguments and cannot "
                "preserve CUDA Graphs"
            ),
            "backend": "vmm",
        }
    if not graphs_destroyed:
        return {
            "ok": False,
            "operation": "destroy",
            "reason": "cuda_graphs_still_live",
            "error": "Destroy every CUDA Graph which captured VMM aliases first",
            "backend": "vmm",
        }
    if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE:
        return {
            "ok": False,
            "operation": "destroy",
            "reason": "cuda_graph_capture_active",
            "error": "VMM aliases cannot be destroyed during Graph capture",
            "backend": "vmm",
        }

    with _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_LOCK:
        arenas = list(sorted(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS.items()))
        configured_backing_bytes = int(
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
        )
        if not arenas:
            if configured_backing_bytes != 0:
                return {
                    "ok": False,
                    "operation": "destroy",
                    "reason": "vmm_state_inconsistent",
                    "error": "VMM backing bytes remain without a live arena",
                    "backend": "vmm",
                    "remaining_backing_bytes": configured_backing_bytes,
                }
            return {
                "ok": True,
                "operation": "destroy",
                "reason": "no_active_allocations",
                "backend": "vmm",
                "arena_count_before": 0,
                "destroyed_arena_count": 0,
                "remaining_arena_count": 0,
                "released_backing_bytes": 0,
                "remaining_backing_bytes": 0,
            }

        computed_backing_bytes = sum(
            int(arena.backing_bytes) for _, arena in arenas
        )
        if computed_backing_bytes != configured_backing_bytes:
            return {
                "ok": False,
                "operation": "destroy",
                "reason": "vmm_state_inconsistent",
                "error": (
                    "Registered VMM arena bytes disagree with the global "
                    "backing-byte counter"
                ),
                "backend": "vmm",
                "arena_count_before": len(arenas),
                "registered_backing_bytes": computed_backing_bytes,
                "configured_backing_bytes": configured_backing_bytes,
            }

        # Drop every tensor reference owned by the integration before the
        # native allocator checks its DLPack live-reference count. Bindings
        # remain reachable from wrapped module closures, so clear their tensor
        # field explicitly instead of merely removing the global arena map.
        for _, arena in arenas:
            object.__setattr__(arena, "upload", None)
            for binding in tuple(arena.bindings):
                object.__setattr__(binding, "consumer", None)
        gc.collect()

        close_results: list[dict[str, Any]] = []
        closed_keys: list[tuple[str, str, int]] = []
        released_backing_bytes = 0
        for arena_key, arena in arenas:
            allocation = arena.allocation
            token = getattr(allocation, "token", None)
            try:
                if synchronize:
                    allocation.close(synchronize=True)
                    barrier = "close_synchronize"
                else:
                    # The caller may already have synchronized CUDA, but the
                    # native allocator intentionally requires its own teardown
                    # barrier bit. Set it after releasing our tensor views.
                    allocation.synchronize()
                    allocation.close(synchronize=False)
                    barrier = "explicit_synchronize"
                if getattr(allocation, "closed", None) is not True:
                    raise RuntimeError(
                        "VMM allocation close returned without a closed state"
                    )
            except Exception as exc:
                close_results.append(
                    {
                        "arena_key": arena_key,
                        "token": token,
                        "ok": False,
                        "error": repr(exc),
                    }
                )
                continue

            closed_keys.append(arena_key)
            released_backing_bytes += int(arena.backing_bytes)
            close_results.append(
                {
                    "arena_key": arena_key,
                    "token": token,
                    "ok": True,
                    "barrier": barrier,
                    "backing_bytes": int(arena.backing_bytes),
                }
            )

        for arena_key in closed_keys:
            arena = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS.pop(arena_key)
            arena.bindings.clear()
        remaining_backing_bytes = sum(
            int(arena.backing_bytes)
            for arena in _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS.values()
        )
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES = remaining_backing_bytes
        ok = not _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS
        return {
            "ok": ok,
            "operation": "destroy",
            "reason": "ok" if ok else "vmm_close_failed",
            "backend": "vmm",
            "arena_count_before": len(arenas),
            "destroyed_arena_count": len(closed_keys),
            "remaining_arena_count": len(
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS
            ),
            "released_backing_bytes": released_backing_bytes,
            "remaining_backing_bytes": remaining_backing_bytes,
            "close_results": tuple(close_results),
        }


def destroy_cpu_offload_graph_static_alias(
    *,
    graphs_destroyed: bool,
    synchronize: bool = True,
    remote_graphs_preserved: bool = False,
    defer_cpu_restore: bool = False,
) -> dict[str, Any]:
    """Destroy static aliases after Graph ownership is proven safe.

    Original-VA aliases may retain only an audited remote Graph. CUDA VMM
    aliases always require all captured Graphs to be destroyed first.
    """
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE

    if not _CPU_OFFLOAD_GRAPH_STATIC_ALIAS:
        return {
            "ok": True,
            "operation": "destroy",
            "reason": "disabled",
            "backend": None,
        }
    if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND == "vmm":
        return _destroy_vmm_graph_static_alias(
            graphs_destroyed=graphs_destroyed,
            synchronize=synchronize,
            remote_graphs_preserved=remote_graphs_preserved,
        )
    if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND != "uvm_original_va":
        return {
            "ok": False,
            "operation": "destroy",
            "reason": "unsupported_backend",
            "backend": _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND,
        }
    if (
        not isinstance(graphs_destroyed, bool)
        or not isinstance(synchronize, bool)
        or not isinstance(remote_graphs_preserved, bool)
        or not isinstance(defer_cpu_restore, bool)
        or (graphs_destroyed and remote_graphs_preserved)
    ):
        return {
            "ok": False,
            "operation": "destroy",
            "reason": "validation_failed",
            "error": (
                "graphs_destroyed, synchronize, remote_graphs_preserved, and "
                "defer_cpu_restore must be unambiguous bool values"
            ),
            "backend": "uvm_original_va",
        }

    with _CPU_OFFLOAD_UVM_STATIC_ALIAS_LOCK:
        session_id = int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID)
        cleanup_pending = _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "cleanup_pending"
        if session_id == 0 and not cleanup_pending:
            return {
                "ok": True,
                "operation": "destroy",
                "reason": "no_active_session",
                "backend": "uvm_original_va",
                "state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
                "session_id": 0,
            }
        preserved_graph_contract = False
        if not graphs_destroyed:
            preserved_graph_contract, contract_reason = (
                _uvm_original_alias_can_preserve_remote_graphs(
                    synchronize=synchronize
                )
            )
            if not remote_graphs_preserved or not preserved_graph_contract:
                return {
                    "ok": False,
                    "operation": "destroy",
                    "reason": "cuda_graphs_still_live",
                    "error": (
                        "Destroy alias-topology Graphs before cmd99; only a "
                        "successfully captured remote-only Graph may remain live"
                    ),
                    "preserved_graph_contract_reason": contract_reason,
                    "backend": "uvm_original_va",
                    "state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
                    "session_id": session_id,
                }

        from vllm.device_allocator import uvm, uvm_static_alias

        device = int(_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0)
        if session_id == 0:
            if _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED:
                return {
                    "ok": False,
                    "operation": "destroy",
                    "reason": "cleanup_unresolved_requires_process_teardown",
                    "error": (
                        "A MAP may have changed kernel state without returning "
                        "a trustworthy session id; retaining slot backing and "
                        "the UVM fd until process teardown"
                    ),
                    "backend": "uvm_original_va",
                    "state": "cleanup_pending",
                    "session_id": 0,
                    "destroy": _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT,
                    "slot_close": _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT,
                }
            slot_closed, slot_close = _uvm_original_alias_close_slot_session(
                uvm, device=device
            )
            if not slot_closed:
                return {
                    "ok": False,
                    "operation": "destroy",
                    "reason": "slot_session_close_failed",
                    "backend": "uvm_original_va",
                    "state": "cleanup_pending",
                    "session_id": 0,
                    "destroy": _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT,
                    "slot_close": slot_close,
                }
            destroyed = _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT
            destroy_contract_ok = destroyed is None or bool(destroyed.get("ok"))
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "remote"
            result = {
                "ok": destroy_contract_ok,
                "operation": "destroy",
                "reason": (
                    "slot_session_close_retried"
                    if destroy_contract_ok
                    else "cmd99_contract_mismatch_session_cleared"
                ),
                "backend": "uvm_original_va",
                "state": "remote",
                "session_id": 0,
                "destroy": destroyed,
                "slot_close": slot_close,
            }
            _append_uvm_original_alias_lifecycle(
                "cpu_offload_uvm_static_alias_destroyed", result
            )
            return result

        try:
            destroyed = dict(
                uvm_static_alias.driver_static_alias_destroy(
                    session_id,
                    device=device,
                    release_backing=True,
                    defer_cpu_restore=defer_cpu_restore,
                    synchronize=synchronize,
                )
            )
        except Exception as exc:
            destroyed = {
                "ok": False,
                "operation": "destroy",
                "reason": "destroy_exception",
                "session_id": session_id,
                "error": repr(exc),
            }
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT = destroyed
        if not _uvm_original_alias_destroy_cleared(destroyed):
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "cleanup_pending"
            return {
                "ok": False,
                "operation": "destroy",
                "reason": "cmd99_failed",
                "backend": "uvm_original_va",
                "state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
                "session_id": session_id,
                "destroy": destroyed,
            }
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID = 0
        _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
        if _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED:
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "cleanup_pending"
            return {
                "ok": False,
                "operation": "destroy",
                "reason": "cleanup_unresolved_requires_process_teardown",
                "backend": "uvm_original_va",
                "state": "cleanup_pending",
                "session_id": 0,
                "destroy": destroyed,
                "slot_close": _CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT,
            }
        slot_closed, slot_close = _uvm_original_alias_close_slot_session(
            uvm, device=device
        )
        if not slot_closed:
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "cleanup_pending"
            result = {
                "ok": False,
                "operation": "destroy",
                "reason": "slot_session_close_failed",
                "backend": "uvm_original_va",
                "state": "cleanup_pending",
                "session_id": 0,
                "destroy": destroyed,
                "slot_close": slot_close,
            }
            _append_uvm_original_alias_lifecycle(
                "cpu_offload_uvm_static_alias_destroyed", result
            )
            return result
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "remote"
        result = {
            "ok": bool(destroyed.get("ok")),
            "operation": "destroy",
            "reason": (
                "ok"
                if bool(destroyed.get("ok"))
                else "cmd99_contract_mismatch_session_cleared"
            ),
            "backend": "uvm_original_va",
            "state": "remote",
            "session_id": 0,
            "destroy": destroyed,
            "slot_close": slot_close,
            "remote_graphs_preserved": bool(preserved_graph_contract),
        }
        _append_uvm_original_alias_lifecycle(
            "cpu_offload_uvm_static_alias_destroyed", result
        )
        return result


def _uvm_original_parameter_va_snapshot() -> tuple[int, ...]:
    vas: list[int] = []
    for _, binding in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()):
        current = binding.parameter.data
        current_va = int(current.data_ptr())
        expected_va = int(binding.consumer.data_ptr())
        if current_va != expected_va:
            raise RuntimeError(
                "A selected expert Parameter changed allocation VA: "
                f"{binding.full_parameter_name!r}, current=0x{current_va:x}, "
                f"expected=0x{expected_va:x}"
            )
        vas.append(current_va)
    return tuple(vas)


def release_cpu_offload_uvm_original_va_shutdown_references() -> dict[str, Any]:
    """Release process-global tensor references after audited shutdown.

    ``destroy_cpu_offload_graph_static_alias`` tears down driver mappings and
    backing, but serving keeps binding objects alive because module forward
    closures may still need them.  At final process shutdown those globals
    must be released while CUDA and UVM are still initialized; otherwise
    managed Parameters, upload arenas, and pinned sources are left to the
    interpreter's unspecified module-destruction order.
    """
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_ENVELOPE_BYTES
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
    global _CPU_OFFLOAD_UVM_PLAN_CLASS_INSTRUCTIONS
    global _CPU_OFFLOAD_UVM_PLAN_CLASS_SOURCE_PREFILL_RESULT
    global _CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT
    global _CPU_OFFLOAD_OWNER_THREAD_ID
    global _CPU_OFFLOAD_OWNER_DEVICE_INDEX
    global _CPU_OFFLOAD_OWNER_STREAM_ID

    with _CPU_OFFLOAD_UVM_STATIC_ALIAS_LOCK:
        if _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID != 0:
            raise RuntimeError(
                "Cannot release original-VA references while a driver "
                "session is active"
            )
        if _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED:
            raise RuntimeError(
                "Cannot release original-VA references while driver cleanup "
                "is unresolved"
            )
        if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE not in (
            "disabled",
            "remote",
            "shutdown",
        ):
            raise RuntimeError(
                "Cannot release original-VA references from lifecycle state "
                f"{_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE!r}"
            )

        binding_count = len(_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS)
        arena_count = len(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS)
        source_count = 0
        for binding in _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values():
            if binding.source is not None:
                source_count += 1
                binding.source = None

        _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS = {}
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS = {}
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS = {}
        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES = 0
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_ENVELOPE_BYTES = 0
        _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT = ()
        _CPU_OFFLOAD_UVM_PLAN_CLASS_INSTRUCTIONS = {}
        _CPU_OFFLOAD_UVM_PLAN_CLASS_SOURCE_PREFILL_RESULT = None
        _CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT = None
        _CPU_OFFLOAD_OWNER_THREAD_ID = None
        _CPU_OFFLOAD_OWNER_DEVICE_INDEX = None
        _CPU_OFFLOAD_OWNER_STREAM_ID = None
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "shutdown"
        return {
            "ok": True,
            "operation": "release_shutdown_references",
            "binding_count": binding_count,
            "arena_count": arena_count,
            "source_count": source_count,
            "state": "shutdown",
        }


def _reset_uvm_original_va_graph_certificates_for_plan_rebuild() -> None:
    """Forget certificates only after their concrete Graphs were reset."""
    global _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY
    global _CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT
    global _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT
    global _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES
    global _CPU_OFFLOAD_UVM_CAPTURE_VA_SNAPSHOT
    global _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE
    global _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE
    global _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE

    if _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE:
        raise RuntimeError("Cannot reset CUDA Graph certificates during capture")
    _CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY = None
    _CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT = 0
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED = False
    _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED = False
    _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED = True
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT = 0
    _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT = 0
    _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES = 0
    _CPU_OFFLOAD_UVM_CAPTURE_VA_SNAPSHOT = ()
    _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE = None
    _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE = None
    _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE = None


def _uvm_original_alias_plan_geometry(
    experts_by_layer: Mapping[int, tuple[int, ...]],
) -> dict[
    str,
    tuple[
        tuple[int, ...] | None,
        int,
        int,
        tuple[int, ...],
        tuple[int, ...],
        tuple[tuple[int, int], ...],
    ],
]:
    result = {}
    for full_name, binding in sorted(
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
    ):
        result[full_name] = _uvm_original_alias_parameter_geometry(
            full_parameter_name=full_name,
            tensor=binding.consumer,
            parameter_bytes=int(binding.parameter_bytes),
            experts_by_layer=experts_by_layer,
        )
    return result


def _uvm_original_alias_plan_signature(
    experts_by_layer: Mapping[int, tuple[int, ...]],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    return tuple(
        (int(layer_index), tuple(int(value) for value in expert_ids))
        for layer_index, expert_ids in sorted(experts_by_layer.items())
    )


def _uvm_original_alias_stabilize_plan_slots(
    old_geometry: Mapping[str, tuple[Any, ...]],
    target_geometry: Mapping[str, tuple[Any, ...]],
) -> tuple[dict[str, tuple[Any, ...]], dict[str, Any]]:
    """Keep retained experts in the same packed slot whenever possible.

    Plan files describe expert sets, not a semantically significant packed
    order.  Preserving the old slot index avoids refilling immutable expert
    data and lets the backing-resize transaction retain the prefix slots shared
    by two capacity classes.
    """
    stabilized: dict[str, tuple[Any, ...]] = {}
    retained_slot_count = 0
    relocated_slot_count = 0
    changed_slot_count = 0
    for full_name, raw_target in sorted(target_geometry.items()):
        try:
            raw_old = old_geometry[full_name]
            binding = _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS[full_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Stable slot geometry is missing {full_name!r}"
            ) from exc
        old_ids = raw_old[0]
        target_ids = raw_target[0]
        if old_ids is None or target_ids is None:
            if old_ids != target_ids or raw_old[2] != raw_target[2]:
                raise RuntimeError(
                    f"Non-expert alias geometry changed for {full_name!r}"
                )
            stabilized[full_name] = tuple(raw_target)
            continue

        target_set = set(target_ids)
        assigned: list[int | None] = [None] * len(target_ids)
        used: set[int] = set()
        for slot_index in range(min(len(old_ids), len(target_ids))):
            expert_id = int(old_ids[slot_index])
            if expert_id in target_set and expert_id not in used:
                assigned[slot_index] = expert_id
                used.add(expert_id)
                retained_slot_count += 1
        remaining = [int(value) for value in target_ids if int(value) not in used]
        remaining_iter = iter(remaining)
        for slot_index, expert_id in enumerate(assigned):
            if expert_id is None:
                assigned[slot_index] = next(remaining_iter)
        stable_ids = tuple(int(value) for value in assigned)
        if set(stable_ids) != target_set or len(stable_ids) != len(target_ids):
            raise RuntimeError(
                f"Stable slot assignment changed the target set for {full_name!r}"
            )
        for slot_index, expert_id in enumerate(stable_ids):
            old_expert = old_ids[slot_index] if slot_index < len(old_ids) else None
            if old_expert != expert_id:
                changed_slot_count += 1
                if expert_id in set(old_ids):
                    relocated_slot_count += 1
        stride = int(raw_target[1])
        consumer_base = int(binding.consumer.data_ptr())
        consumer_ranges = tuple(
            (consumer_base + expert_id * stride, stride)
            for expert_id in stable_ids
        )
        stabilized[full_name] = (
            stable_ids,
            stride,
            int(raw_target[2]),
            raw_target[3],
            raw_target[4],
            consumer_ranges,
        )
    return stabilized, {
        "retained_slot_count": int(retained_slot_count),
        "relocated_slot_count": int(relocated_slot_count),
        "changed_slot_count": int(changed_slot_count),
    }


def _refill_uvm_original_alias_plan_delta(
    old_geometry: Mapping[str, tuple[Any, ...]],
    target_geometry: Mapping[str, tuple[Any, ...]],
) -> dict[str, Any]:
    """Refresh only packed source slots whose logical expert changed."""
    requested_slot_count = 0
    requested_bytes = 0
    changed_slot_count = 0
    changed_bytes = 0
    reused_slot_count = 0
    reused_bytes = 0
    changed_bindings: list[str] = []
    for full_name, binding in sorted(
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
    ):
        old_ids = old_geometry[full_name][0]
        target_ids = target_geometry[full_name][0]
        if old_ids is None or target_ids is None:
            continue
        if binding.source is None:
            raise RuntimeError(
                f"Pinned source is absent for {binding.full_parameter_name!r}"
            )
        binding_changed = False
        current = binding.parameter.data
        stride = int(target_geometry[full_name][1])
        slot_experts = list(
            getattr(binding, "source_slot_expert_ids", ())
        )
        source_slots = int(binding.source.size(0))
        if len(slot_experts) != source_slots:
            slot_experts = [None] * source_slots
        for slot_index, expert_id in enumerate(target_ids):
            old_expert = old_ids[slot_index] if slot_index < len(old_ids) else None
            if old_expert == expert_id:
                continue
            requested_slot_count += 1
            requested_bytes += stride
            if slot_experts[slot_index] == int(expert_id):
                reused_slot_count += 1
                reused_bytes += stride
                continue
            binding.source[slot_index].copy_(
                current[int(expert_id)], non_blocking=True
            )
            slot_experts[slot_index] = int(expert_id)
            changed_slot_count += 1
            changed_bytes += stride
            binding_changed = True
        binding.source_slot_expert_ids = slot_experts
        if binding_changed:
            changed_bindings.append(full_name)
    return {
        "mode": "delta",
        "requested_slot_count": int(requested_slot_count),
        "requested_bytes": int(requested_bytes),
        "changed_binding_count": len(changed_bindings),
        "changed_slot_count": int(changed_slot_count),
        "changed_bytes": int(changed_bytes),
        "reused_prefilled_slot_count": int(reused_slot_count),
        "reused_prefilled_bytes": int(reused_bytes),
        "changed_bindings": tuple(changed_bindings),
    }


def _uvm_original_alias_geometry_expert_ranges(
    geometry: Mapping[str, tuple[Any, ...]],
) -> dict[str, dict[int, tuple[int, int]]]:
    result: dict[str, dict[int, tuple[int, int]]] = {}
    for full_name, values in geometry.items():
        expert_ids = values[0]
        if expert_ids is None:
            continue
        binding = _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS[full_name]
        stride = int(values[1])
        base = int(binding.consumer.data_ptr())
        result[full_name] = {
            int(expert_id): (base + int(expert_id) * stride, stride)
            for expert_id in expert_ids
        }
    return result


def _uvm_original_alias_plan_residency_delta_ranges(
    old_geometry: Mapping[str, tuple[Any, ...]],
    target_geometry: Mapping[str, tuple[Any, ...]],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Return immutable added-cold and removed-cold range lists."""
    old_ranges = _uvm_original_alias_geometry_expert_ranges(old_geometry)
    target_ranges = _uvm_original_alias_geometry_expert_ranges(target_geometry)
    added_cold: list[tuple[int, int]] = []
    removed_cold: list[tuple[int, int]] = []
    for full_name in sorted(set(old_ranges) | set(target_ranges)):
        old = old_ranges.get(full_name, {})
        target = target_ranges.get(full_name, {})
        added_cold.extend(target[expert] for expert in sorted(set(target) - set(old)))
        removed_cold.extend(old[expert] for expert in sorted(set(old) - set(target)))
    return tuple(added_cold), tuple(removed_cold)


def _prepare_uvm_original_alias_plan_residency_delta(
    old_geometry: Mapping[str, tuple[Any, ...]],
    target_geometry: Mapping[str, tuple[Any, ...]],
    *,
    device: int,
    prepared_ranges: tuple[
        tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]
    ]
    | None = None,
    advise_removed: bool = True,
) -> dict[str, Any]:
    """Move only newly cold experts to CPU and newly hot experts to GPU."""
    from vllm.device_allocator import uvm

    immutable_shadow_enabled = _strict_env_flag(
        "VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW", False
    ) and not _strict_env_flag(
        "VLLM_CPU_OFFLOAD_PLAN_CLASS_FORCE_SYNC_MIGRATION", False
    )
    force_sync_migration = _strict_env_flag(
        "VLLM_CPU_OFFLOAD_PLAN_CLASS_FORCE_SYNC_MIGRATION", False
    )
    if immutable_shadow_enabled and not (
        isinstance(_CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT, dict)
        and _CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT.get("enabled") is True
    ):
        raise RuntimeError(
            "Immutable CPU shadow cutover lacks an initialization certificate"
        )

    if prepared_ranges is None:
        added_cold, removed_cold = (
            _uvm_original_alias_plan_residency_delta_ranges(
                old_geometry, target_geometry
            )
        )
    else:
        added_cold, removed_cold = prepared_ranges

    added_bytes = sum(length for _, length in added_cold)
    removed_bytes = sum(length for _, length in removed_cold)
    shadow_cutover: dict[str, Any] | None = None
    read_duplicate_promotion: dict[str, Any] | None = None
    if added_cold:
        if immutable_shadow_enabled:
            if _strict_env_flag(
                "VLLM_CPU_OFFLOAD_PLAN_CLASS_REPROMOTE_BEFORE_CUTOVER",
                False,
            ):
                # Graph replay can revoke UVM's read-duplicate certificate
                # while leaving both physical copies present.  The native
                # ReadMostly + CPU-prefetch sequence is the same operation
                # used to build the startup immutable shadows: it reasserts
                # read_duplicated_pages and installs a GPU READ PTE without
                # changing the Parameter VA.  Merely invoking Phase-F action
                # 18 is insufficient for already dual-resident pages because
                # that fast path treats them as an idempotent no-copy case.
                readmostly_count = int(
                    uvm.batch_advise_ranges_read_mostly(
                        added_cold, device=device
                    )
                )
                prefetch_count = int(
                    uvm.batch_prefetch_ranges_to_cpu(added_cold)
                )
                torch.cuda.synchronize(device)
                promotion_residency = dict(
                    uvm.driver_batch_query_ranges_residency_bytes(
                        added_cold,
                        device=device,
                        wait_for_trackers=False,
                    )
                )
                read_duplicate_promotion = {
                    "ok": bool(
                        readmostly_count == len(added_cold)
                        and prefetch_count == len(added_cold)
                        and promotion_residency.get("ok") is True
                        and int(
                            promotion_residency.get("queried_bytes", -1)
                        )
                        == added_bytes
                        and int(promotion_residency.get("gpu_bytes", -1))
                        == added_bytes
                        and int(promotion_residency.get("cpu_bytes", -1))
                        == added_bytes
                        and int(promotion_residency.get("any_bytes", -1))
                        == added_bytes
                        and int(promotion_residency.get("unknown_bytes", -1))
                        == 0
                    ),
                    "operation": "native_readmostly_cpu_prefetch",
                    "range_count": len(added_cold),
                    "success_count": prefetch_count,
                    "failure_count": len(added_cold) - prefetch_count,
                    "requested_bytes": int(added_bytes),
                    "readmostly_count": readmostly_count,
                    "prefetch_count": prefetch_count,
                    "residency": promotion_residency,
                }
                if not (
                    read_duplicate_promotion.get("ok") is True
                    and int(
                        read_duplicate_promotion.get("range_count", -1)
                    )
                    == len(added_cold)
                    and int(
                        read_duplicate_promotion.get("success_count", -1)
                    )
                    == len(added_cold)
                    and int(
                        read_duplicate_promotion.get("failure_count", -1)
                    )
                    == 0
                    and int(
                        read_duplicate_promotion.get("requested_bytes", -1)
                    )
                    == added_bytes
                ):
                    raise RuntimeError(
                        "Failed to re-promote every added cold expert to an "
                        "immutable CPU/GPU read duplicate before cutover: "
                        f"{read_duplicate_promotion!r}"
                    )
            shadow_cutover = dict(
                uvm.driver_cutover_ranges_to_immutable_cpu_shadow(
                    added_cold, device=device
                )
            )
            if not (
                shadow_cutover.get("ok") is True
                and int(shadow_cutover.get("range_count", -1))
                == len(added_cold)
                and int(shadow_cutover.get("requested_bytes", -1))
                == added_bytes
                and int(shadow_cutover.get("released_gpu_bytes", -1))
                == added_bytes
                and int(
                    shadow_cutover.get("retained_cpu_shadow_bytes", -1)
                )
                == added_bytes
                and int(shadow_cutover.get("remote_mapped_bytes", -1))
                == added_bytes
            ):
                raise RuntimeError(
                    "Immutable CPU shadow cutover did not release every added "
                    f"cold expert: {shadow_cutover!r}"
                )
            prepared: Any = shadow_cutover
        elif force_sync_migration:
            prepared = dict(
                uvm.driver_cutover_ranges_to_immutable_cpu_shadow(
                    added_cold,
                    device=device,
                    force_synchronous_migration=True,
                )
            )
            if not (
                prepared.get("ok") is True
                and int(prepared.get("range_count", -1)) == len(added_cold)
                and int(prepared.get("success_count", -1)) == len(added_cold)
                and int(prepared.get("failure_count", -1)) == 0
                and int(prepared.get("requested_bytes", -1)) == added_bytes
                and int(prepared.get("released_gpu_bytes", -1)) == added_bytes
                and int(prepared.get("retained_cpu_shadow_bytes", -1))
                == added_bytes
                and int(prepared.get("remote_mapped_bytes", -1)) == added_bytes
            ):
                raise RuntimeError(
                    f"Failed to move every added cold expert to CPU: {prepared!r}"
                )
        else:
            prepared = (
                uvm.batch_prepare_ranges_for_native_remote_access_with_executor(
                    added_cold,
                    sync=True,
                    device=device,
                )
            )
            if int(prepared[0]) != len(added_cold):
                raise RuntimeError(
                    f"Failed to move every added cold expert to CPU: {prepared!r}"
                )
    else:
        prepared = (0, "none", 0)

    if removed_cold:
        advised = (
            int(uvm.batch_advise_ranges_preferred_location(removed_cold, device))
            if advise_removed and not force_sync_migration
            else 0
        )
        prefetched, executor, driver_result = (
            uvm.batch_prefetch_ranges_to_device_with_executor(
                removed_cold,
                device,
                executor="cpp_batch",
            )
        )
        if (
            (
                advise_removed
                and not force_sync_migration
                and advised != len(removed_cold)
            )
            or int(prefetched) != len(removed_cold)
        ):
            raise RuntimeError(
                "Failed to restore every removed cold expert to GPU: "
                f"advised={advised}, prefetched={prefetched}, "
                f"expected={len(removed_cold)}, executor={executor!r}, "
                f"driver_result={driver_result}"
            )
    else:
        advised = 0
        prefetched = 0
        executor = "none"
        driver_result = 0

    if added_cold or removed_cold:
        torch.cuda.synchronize(device)
    added_residency: dict[str, Any] | None = None
    removed_residency: dict[str, Any] | None = None
    if added_cold:
        added_residency = dict(
            uvm.driver_batch_query_ranges_residency_bytes(
                added_cold, device=device, wait_for_trackers=False
            )
        )
        if not (
            added_residency.get("ok") is True
            and int(added_residency.get("queried_bytes", -1)) == added_bytes
            and int(added_residency.get("cpu_bytes", -1)) == added_bytes
            and int(added_residency.get("gpu_bytes", -1)) == 0
            and int(added_residency.get("unknown_bytes", -1)) == 0
        ):
            raise RuntimeError(
                f"Added cold expert residency is not CPU-only: {added_residency!r}"
            )
    if removed_cold:
        removed_residency = dict(
            uvm.driver_batch_query_ranges_residency_bytes(
                removed_cold, device=device, wait_for_trackers=False
            )
        )
        retained_shadow_expected = bool(
            immutable_shadow_enabled
            or (
                force_sync_migration
                and isinstance(_CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT, dict)
                and _CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT.get("enabled")
                is True
            )
        )
        expected_removed_cpu_bytes = (
            removed_bytes if retained_shadow_expected else 0
        )
        if not (
            removed_residency.get("ok") is True
            and int(removed_residency.get("queried_bytes", -1)) == removed_bytes
            and int(removed_residency.get("gpu_bytes", -1)) == removed_bytes
            and int(removed_residency.get("cpu_bytes", -1))
            == expected_removed_cpu_bytes
            and int(removed_residency.get("unknown_bytes", -1)) == 0
        ):
            raise RuntimeError(
                "Removed cold expert residency is not GPU-only: "
                f"{removed_residency!r}"
            )
    return {
        "mode": (
            "immutable_cpu_shadow_delta"
            if immutable_shadow_enabled
            else "delta"
        ),
        "immutable_cpu_shadow_enabled": bool(immutable_shadow_enabled),
        "read_duplicate_promotion": read_duplicate_promotion,
        "shadow_cutover": shadow_cutover,
        "added_cold_range_count": len(added_cold),
        "added_cold_bytes": int(added_bytes),
        "removed_cold_range_count": len(removed_cold),
        "removed_cold_bytes": int(removed_bytes),
        "prepare": prepared,
        "advice_requested": bool(
            advise_removed
            and removed_cold
            and not force_sync_migration
        ),
        "advised_count": advised,
        "prefetched_count": int(prefetched),
        "prefetch_executor": str(executor),
        "prefetch_driver_result": int(driver_result),
        "added_cold_residency": added_residency,
        "removed_cold_residency": removed_residency,
    }


def _merge_uvm_original_alias_plan_residency_phases(
    before_detach: Mapping[str, Any] | None,
    after_detach: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Combine split add-cold/remove-cold residency transactions."""
    if before_detach is None:
        return dict(after_detach) if after_detach is not None else None
    if after_detach is None:
        return dict(before_detach)
    return {
        "mode": "split_deferred_removed",
        "immutable_cpu_shadow_enabled": bool(
            before_detach.get("immutable_cpu_shadow_enabled")
            or after_detach.get("immutable_cpu_shadow_enabled")
        ),
        "read_duplicate_promotion": before_detach.get(
            "read_duplicate_promotion"
        ),
        "shadow_cutover": before_detach.get("shadow_cutover"),
        "added_cold_range_count": int(
            before_detach.get("added_cold_range_count", 0)
        ),
        "added_cold_bytes": int(before_detach.get("added_cold_bytes", 0)),
        "removed_cold_range_count": int(
            after_detach.get("removed_cold_range_count", 0)
        ),
        "removed_cold_bytes": int(after_detach.get("removed_cold_bytes", 0)),
        "prepare": before_detach.get("prepare"),
        "advice_requested": bool(after_detach.get("advice_requested")),
        "advised_count": int(after_detach.get("advised_count", 0)),
        "prefetched_count": int(after_detach.get("prefetched_count", 0)),
        "prefetch_executor": str(
            after_detach.get("prefetch_executor", "none")
        ),
        "prefetch_driver_result": int(
            after_detach.get("prefetch_driver_result", 0)
        ),
        "added_cold_residency": before_detach.get("added_cold_residency"),
        "removed_cold_residency": after_detach.get(
            "removed_cold_residency"
        ),
        "before_detach": dict(before_detach),
        "after_detach": dict(after_detach),
    }


def _apply_uvm_original_alias_plan_geometry(
    geometry: Mapping[
        str,
        tuple[
            tuple[int, ...] | None,
            int,
            int,
            tuple[int, ...],
            tuple[int, ...],
            tuple[tuple[int, int], ...],
        ],
    ],
    *,
    refill_sources: bool,
) -> int:
    global _CPU_OFFLOAD_BYTES
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES

    arena_active_bytes: dict[int, int] = {}
    selected_parameter_bytes = 0
    for full_name, binding in sorted(
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
    ):
        try:
            (
                expert_ids,
                expert_stride_bytes,
                selected_bytes,
                _upload_shape,
                _upload_strides,
                consumer_ranges,
            ) = geometry[full_name]
        except KeyError as exc:
            raise RuntimeError(
                f"Plan geometry is missing {full_name!r}"
            ) from exc
        if selected_bytes <= 0 or selected_bytes > binding.arena.aligned_bytes:
            raise RuntimeError(
                f"Plan geometry escapes stable envelope for {full_name!r}: "
                f"selected={selected_bytes}, envelope={binding.arena.aligned_bytes}"
            )
        if expert_ids is not None and len(expert_ids) > int(
            binding.arena.envelope_expert_count
        ):
            raise RuntimeError(
                f"Plan expert count escapes stable envelope for {full_name!r}"
            )
        arena_key = id(binding.arena)
        previous_bytes = arena_active_bytes.setdefault(
            arena_key, int(selected_bytes)
        )
        if previous_bytes != int(selected_bytes):
            raise RuntimeError(
                "Consumers sharing one upload arena require one active width"
            )
        binding.requested_bytes = int(selected_bytes)
        binding.expert_ids = expert_ids
        binding.expert_stride_bytes = int(expert_stride_bytes)
        binding.consumer_ranges = tuple(consumer_ranges)
        selected_parameter_bytes += int(selected_bytes)

    for arena in _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.values():
        try:
            arena.requested_bytes = arena_active_bytes[id(arena)]
        except KeyError as exc:
            raise RuntimeError("Stable envelope arena has no active consumer") from exc
    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES = sum(
        int(arena.requested_bytes)
        for arena in _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.values()
    )
    _CPU_OFFLOAD_BYTES = int(selected_parameter_bytes)
    if refill_sources:
        for binding in _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values():
            _refill_uvm_original_alias_binding_source(binding)
    return int(selected_parameter_bytes)


def _prepare_uvm_original_alias_plan_residency(*, device: int) -> dict[str, Any]:
    from vllm.device_allocator import uvm

    cold_ranges = [
        consumer_range
        for binding in _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values()
        for consumer_range in _uvm_original_alias_binding_consumer_ranges(binding)
    ]
    hot_ranges = [
        hot_range
        for binding in _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values()
        for hot_range in _uvm_original_alias_binding_hot_ranges(binding)
    ]
    cold_bytes = sum(length for _, length in cold_ranges)
    hot_bytes = sum(length for _, length in hot_ranges)
    prepared = uvm.batch_prepare_ranges_for_native_remote_access_with_executor(
        cold_ranges,
        sync=True,
        device=device,
    )
    if int(prepared[0]) != len(cold_ranges):
        raise RuntimeError(
            f"Failed to make new cold experts CPU resident: {prepared!r}"
        )
    advised = int(uvm.batch_advise_ranges_preferred_location(hot_ranges, device))
    prefetched, executor, driver_result = (
        uvm.batch_prefetch_ranges_to_device_with_executor(
            hot_ranges,
            device,
            executor="cpp_batch",
        )
    )
    if advised != len(hot_ranges) or int(prefetched) != len(hot_ranges):
        raise RuntimeError(
            "Failed to restore every hot expert while changing plan class: "
            f"advised={advised}, prefetched={prefetched}, "
            f"expected={len(hot_ranges)}, executor={executor!r}, "
            f"driver_result={driver_result}"
        )
    torch.cuda.synchronize(device)
    cold_residency = dict(
        uvm.driver_batch_query_ranges_residency_bytes(
            cold_ranges, device=device, wait_for_trackers=False
        )
    )
    hot_residency = dict(
        uvm.driver_batch_query_ranges_residency_bytes(
            hot_ranges, device=device, wait_for_trackers=False
        )
    )
    expected_hot_cpu_bytes = (
        _uvm_original_alias_expected_shadow_cpu_bytes(hot_ranges)
    )
    if cold_ranges and not (
        cold_residency.get("ok") is True
        and int(cold_residency.get("queried_bytes", -1)) == cold_bytes
        and int(cold_residency.get("cpu_bytes", -1)) == cold_bytes
        and int(cold_residency.get("gpu_bytes", -1)) == 0
        and int(cold_residency.get("unknown_bytes", -1)) == 0
    ):
        raise RuntimeError(
            f"New cold expert residency is not CPU-only: {cold_residency!r}"
        )
    if hot_ranges and not (
        hot_residency.get("ok") is True
        and int(hot_residency.get("queried_bytes", -1)) == hot_bytes
        and int(hot_residency.get("gpu_bytes", -1)) == hot_bytes
        and int(hot_residency.get("cpu_bytes", -1))
        == expected_hot_cpu_bytes
        and int(hot_residency.get("unknown_bytes", -1)) == 0
    ):
        raise RuntimeError(
            f"New hot expert residency is not GPU-only: {hot_residency!r}"
        )
    return {
        "cold_range_count": len(cold_ranges),
        "cold_bytes": int(cold_bytes),
        "hot_range_count": len(hot_ranges),
        "hot_bytes": int(hot_bytes),
        "prepare": prepared,
        "advised_count": advised,
        "prefetched_count": int(prefetched),
        "prefetch_executor": str(executor),
        "prefetch_driver_result": int(driver_result),
        "cold_residency": cold_residency,
        "hot_residency": hot_residency,
        "expected_hot_cpu_shadow_bytes": int(expected_hot_cpu_bytes),
    }


def _set_uvm_original_alias_parked_state(result: Mapping[str, Any]) -> None:
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_PARK_RESULT

    _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "remote_parked"
    _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = True
    _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT = dict(result)
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_PARK_RESULT = dict(result)


def _set_uvm_original_alias_sealed_state(result: Mapping[str, Any]) -> None:
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_RESUME_RESULT

    _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "sealed"
    _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
    _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT = None
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_RESUME_RESULT = dict(result)


def _uvm_original_alias_delta_batches(
    entries: tuple[tuple[int, int, int, int, int, int, int], ...],
    *,
    maximum_entries: int,
    keep_backing_identities_whole: bool,
) -> tuple[tuple[tuple[int, int, int, int, int, int, int], ...], ...]:
    """Split a plan-class delta without exceeding one static-alias ioctl."""
    if maximum_entries <= 0:
        raise ValueError("Static-alias delta batch size must be positive")
    if not entries:
        return ()
    if not keep_backing_identities_whole:
        return tuple(
            tuple(entries[offset : offset + maximum_entries])
            for offset in range(0, len(entries), maximum_entries)
        )

    grouped: dict[
        tuple[int, int, int],
        list[tuple[int, int, int, int, int, int, int]],
    ] = {}
    for entry in entries:
        grouped.setdefault((entry[0], entry[1], entry[2]), []).append(entry)

    batches: list[list[tuple[int, int, int, int, int, int, int]]] = []
    current: list[tuple[int, int, int, int, int, int, int]] = []
    for identity_entries in grouped.values():
        if len(identity_entries) > maximum_entries:
            raise ValueError(
                "One backing identity exceeds the static-alias ioctl limit: "
                f"identity={identity_entries[0][:3]!r}, "
                f"entries={len(identity_entries)}, limit={maximum_entries}"
            )
        if current and len(current) + len(identity_entries) > maximum_entries:
            batches.append(current)
            current = []
        current.extend(identity_entries)
    if current:
        batches.append(current)
    return tuple(tuple(batch) for batch in batches)


def _uvm_original_alias_quiesced_delta_batched(
    *,
    session_id: int,
    entries: tuple[tuple[int, int, int, int, int, int, int], ...],
    operation: str,
    device: int,
    defer_dual_refresh: bool,
) -> dict[str, Any]:
    """Apply an oversized quiesced delta using bounded static-alias ioctls."""
    from vllm.device_allocator import uvm_static_alias

    if operation not in ("append", "remove"):
        raise ValueError(f"Unsupported quiesced delta operation: {operation!r}")
    batches = _uvm_original_alias_delta_batches(
        entries,
        maximum_entries=uvm_static_alias.UVM_MOE_STATIC_ALIAS_MAX_ENTRIES,
        keep_backing_identities_whole=operation == "remove",
    )
    if not batches:
        raise ValueError("A quiesced delta requires at least one entry")

    batch_results: list[dict[str, Any]] = []
    completed_entries = 0
    for batch_index, batch in enumerate(batches):
        if operation == "append":
            # Intermediate appends invalidate the old recipe. The last append
            # rebuilds it once unless an enclosing KV migration owns the final
            # full refresh after changing the physical layout.
            flags = uvm_static_alias.UVM_MOE_STATIC_ALIAS_MAP_FLAG_QUIESCED_APPEND
            if defer_dual_refresh or batch_index + 1 < len(batches):
                flags |= (
                    uvm_static_alias.UVM_MOE_STATIC_ALIAS_MAP_FLAG_DEFER_DUAL_REFRESH
                )
            one_result = dict(
                uvm_static_alias.driver_static_alias_map(
                    batch,
                    session_id=session_id,
                    device=device,
                    flags=flags,
                )
            )
        else:
            one_result = dict(
                uvm_static_alias.driver_static_alias_quiesced_remove(
                    session_id,
                    batch,
                    device=device,
                    synchronize=False,
                )
            )
        one_result["batch_index"] = batch_index
        one_result["batch_entry_count"] = len(batch)
        batch_results.append(one_result)
        if one_result.get("ok") is not True:
            made_progress = completed_entries > 0 or bool(
                one_result.get("made_progress")
            )
            return {
                **one_result,
                "ok": False,
                "reason": f"quiesced_{operation}_batch_failed",
                "operation": operation,
                "requested_count": len(entries),
                "completed_count": completed_entries,
                "batch_count": len(batches),
                "completed_batch_count": batch_index,
                "batch_results": batch_results,
                "made_progress": made_progress,
                # Earlier successful ioctls make a later failure a partial
                # whole-delta commit. Force the caller onto its full rebuild
                # rollback instead of assuming the old topology is intact.
                "side_effect_indeterminate": bool(
                    made_progress
                    or one_result.get("side_effect_indeterminate")
                ),
                "ioctl_calls": sum(
                    int(result.get("ioctl_calls", 0))
                    for result in batch_results
                ),
            }
        completed_entries += len(batch)

    final_result = batch_results[-1]
    return {
        **final_result,
        "ok": True,
        "reason": "ok",
        "operation": operation,
        "requested_count": len(entries),
        "completed_count": completed_entries,
        "batch_count": len(batches),
        "completed_batch_count": len(batches),
        "batch_results": batch_results,
        "made_progress": completed_entries > 0,
        "side_effect_indeterminate": False,
        "ioctl_calls": sum(
            int(result.get("ioctl_calls", 0)) for result in batch_results
        ),
        "dual_view_refresh_deferred": bool(
            operation == "append" and defer_dual_refresh
        ),
    }


def _rollback_uvm_original_alias_plan_class_switch(
    *,
    old_geometry: Mapping[str, tuple[Any, ...]],
    target_geometry: Mapping[str, tuple[Any, ...]],
    old_consumer_entries: tuple[tuple[int, int, int, int, int, int, int], ...],
    pre_state: str,
    same_capacity: bool,
    rebind_committed: bool,
    fast_rollback_safe: bool,
    old_session_destroyed: bool,
    old_backing_bytes: int,
    parameter_vas: tuple[int, ...],
    upload_vas: tuple[int, ...],
    source_vas: tuple[int, ...],
    old_remote_map_active: bool,
    old_remote_map_result: Mapping[str, Any] | None,
    device: int,
    old_map_entries: tuple[
        tuple[int, int, int, int, int, int, int], ...
    ] = (),
    rollback_preallocate_entries: tuple[
        tuple[int, int, int, int, int, int, int], ...
    ] = (),
    orphan_preallocate_entries: tuple[
        tuple[int, int, int, int, int, int, int], ...
    ] = (),
) -> dict[str, Any]:
    """Restore the complete pre-switch topology or raise and poison upstream.

    A retained-backing rollback is preferred because it only needs another
    parked REBIND.  If that path is unavailable or any transition poisons the
    session, destroy the current session and materialize the old class again.
    The stable upload/source/Parameter allocations remain alive in both cases.
    """
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT

    from vllm.device_allocator import uvm, uvm_static_alias

    started_ns = time.perf_counter_ns()
    force_sync_dual_transaction = (
        _strict_env_flag(
            "VLLM_CPU_OFFLOAD_PLAN_CLASS_FORCE_SYNC_MIGRATION", False
        )
        and _strict_env_flag(
            "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW", False
        )
    )
    # Multi-class initialization prefills every immutable pinned source. A
    # failed synchronous cutover can leave a managed Parameter range only
    # partially GPU-readable, so reading it back during rollback would turn a
    # recoverable residency failure into a CUDA illegal-address. Reuse the
    # certified sources in this mode.
    _apply_uvm_original_alias_plan_geometry(
        old_geometry,
        refill_sources=not force_sync_dual_transaction,
    )
    torch.cuda.synchronize(device)

    retained_error: str | None = None
    rollback_park: dict[str, Any] | None = None
    rollback_rebind: dict[str, Any] | None = None
    rollback_resume: dict[str, Any] | None = None
    rollback_residency: dict[str, Any] | None = None
    rollback_destroy: dict[str, Any] | None = None
    rollback_close: dict[str, Any] | None = None
    rollback_finalize: dict[str, Any] | None = None
    rollback_preallocate: dict[str, Any] | None = None
    orphan_release: dict[str, Any] | None = None
    rollback_dual_discard: dict[str, Any] | None = None
    rollback_dual_refresh: dict[str, Any] | None = None
    if not old_session_destroyed and fast_rollback_safe:
        try:
            session_id = int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID)
            if session_id <= 0:
                raise RuntimeError("Retained rollback lost its alias session")
            if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "sealed":
                rollback_park = dict(
                    uvm_static_alias.driver_static_alias_park(
                        session_id,
                        device=device,
                        synchronize=False,
                    )
                )
                if rollback_park.get("ok") is not True:
                    raise RuntimeError(
                        f"Rollback PARK failed: {rollback_park!r}"
                    )
                _set_uvm_original_alias_parked_state(rollback_park)
            elif _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE != "remote_parked":
                raise RuntimeError(
                    "Retained rollback requires a sealed or parked session, "
                    f"got {_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE!r}"
                )

            if rebind_committed:
                if old_map_entries:
                    if rollback_preallocate_entries:
                        rollback_preallocate = dict(
                            uvm.driver_slot_preallocate_ranges(
                                rollback_preallocate_entries,
                                device=device,
                            )
                        )
                        if rollback_preallocate.get("ok") is not True:
                            raise RuntimeError(
                                "Rollback could not preallocate removed old "
                                f"backing: {rollback_preallocate!r}"
                            )
                    rollback_rebind = dict(
                        uvm_static_alias.driver_static_alias_rebind(
                            session_id,
                            old_map_entries,
                            device=device,
                            synchronize=False,
                            resize=True,
                            expected_backing_bytes=old_backing_bytes,
                        )
                    )
                elif same_capacity:
                    rollback_rebind = dict(
                        uvm_static_alias.driver_static_alias_rebind(
                            session_id,
                            old_consumer_entries,
                            device=device,
                            synchronize=False,
                        )
                    )
                else:
                    raise RuntimeError(
                        "Rollback lacks the complete old topology for resize"
                    )
                if rollback_rebind.get("ok") is not True:
                    raise RuntimeError(
                        f"Rollback REBIND failed: {rollback_rebind!r}"
                    )
            elif orphan_preallocate_entries:
                orphan_release = dict(
                    uvm.driver_slot_release_ranges(
                        tuple(
                            (entry[2], entry[3], entry[4])
                            for entry in orphan_preallocate_entries
                        ),
                        device=device,
                    )
                )
                if orphan_release.get("ok") is not True:
                    raise RuntimeError(
                        "Rollback could not release uncommitted added backing: "
                        f"{orphan_release!r}"
                    )

            # Detached dual-view children freeze ordinary Parameter leaves.
            # Restore ordinary UVM page tables before reversing residency, then
            # rebuild both children from the restored physical layout before a
            # possible RESUME publishes the GPU child again.
            if force_sync_dual_transaction:
                rollback_dual_discard = dict(
                    uvm_static_alias.driver_static_alias_seal(
                        session_id,
                        device=device,
                        discard_dual=True,
                    )
                )
                if rollback_dual_discard.get("ok") is not True:
                    raise RuntimeError(
                        "Rollback could not discard the current full dual view: "
                        f"{rollback_dual_discard!r}"
                    )

            # Reverse exactly the residency difference applied by the forward
            # switch. A full-plan pass both touches retained experts and, on
            # an early preallocation failure, can leave the newly cold pages
            # in the target state.
            rollback_residency = _prepare_uvm_original_alias_plan_residency_delta(
                target_geometry,
                old_geometry,
                device=device,
            )
            if force_sync_dual_transaction:
                rollback_dual_refresh = dict(
                    uvm_static_alias.driver_static_alias_seal(
                        session_id,
                        device=device,
                        refresh_dual=True,
                    )
                )
                if rollback_dual_refresh.get("ok") is not True:
                    raise RuntimeError(
                        "Rollback could not rebuild the restored full dual view: "
                        f"{rollback_dual_refresh!r}"
                    )
            if pre_state == "sealed":
                rollback_resume = dict(
                    uvm_static_alias.driver_static_alias_resume(
                        session_id,
                        device=device,
                        synchronize=False,
                    )
                )
                if rollback_resume.get("ok") is not True:
                    raise RuntimeError(
                        f"Rollback RESUME failed: {rollback_resume!r}"
                    )
                _set_uvm_original_alias_sealed_state(rollback_resume)
            else:
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "remote_parked"
                _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = old_remote_map_active
                _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT = (
                    dict(old_remote_map_result)
                    if old_remote_map_result is not None
                    else None
                )
        except Exception as exc:
            retained_error = repr(exc)
        else:
            mode = "retained_backing_rebind"
            result = {
                "ok": True,
                "mode": mode,
                "park": rollback_park,
                "rebind": rollback_rebind,
                "preallocate": rollback_preallocate,
                "orphan_release": orphan_release,
                "resume": rollback_resume,
                "residency": rollback_residency,
                "dual_discard": rollback_dual_discard,
                "dual_refresh": rollback_dual_refresh,
                "wall_ns": time.perf_counter_ns() - started_ns,
            }
            if int(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES) != int(
                old_backing_bytes
            ):
                raise RuntimeError(
                    "Retained rollback restored an unexpected backing size"
                )
            if pre_state != _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE:
                raise RuntimeError(
                    "Retained rollback did not restore the lifecycle state"
                )
            if _uvm_original_parameter_va_snapshot() != parameter_vas:
                raise RuntimeError("Rollback changed Parameter VAs")
            if tuple(
                int(arena.upload.data_ptr())
                for _, arena in sorted(
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.items()
                )
            ) != upload_vas:
                raise RuntimeError("Rollback changed upload VAs")
            if tuple(
                int(binding.source.data_ptr())
                for _, binding in sorted(
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                )
                if binding.source is not None
            ) != source_vas:
                raise RuntimeError("Rollback changed pinned source VAs")
            return result

    # Either the old backing was already released or the in-place recovery
    # could not be trusted.  A full alias/backing rebuild is still safe because
    # Graph-visible allocations are the stable envelope, not driver backing.
    if _CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED:
        raise RuntimeError(
            "Cannot rebuild the old class while alias cleanup is unresolved"
        )
    active_session_id = int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID)
    if active_session_id > 0:
        rollback_destroy = dict(
            uvm_static_alias.driver_static_alias_destroy(
                active_session_id,
                device=device,
                release_backing=True,
                synchronize=False,
            )
        )
        if not _uvm_original_alias_destroy_cleared(rollback_destroy):
            raise RuntimeError(
                "Rollback could not authoritatively destroy the current alias "
                f"session: {rollback_destroy!r}"
            )
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID = 0

    closed, raw_close = _uvm_original_alias_close_slot_session(
        uvm, device=device
    )
    rollback_close = dict(raw_close or {"ok": closed})
    if not closed:
        raise RuntimeError(
            f"Rollback could not close the current backing: {rollback_close!r}"
        )

    _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "remote"
    _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
    _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT = None
    rollback_finalize = finalize_cpu_offload_graph_static_alias()
    if rollback_finalize.get("ok") is not True:
        raise RuntimeError(
            f"Rollback could not materialize the old class: {rollback_finalize!r}"
        )
    if pre_state == "remote_parked":
        rollback_park = dict(
            uvm_static_alias.driver_static_alias_park(
                int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
                device=device,
                synchronize=False,
            )
        )
        if rollback_park.get("ok") is not True:
            raise RuntimeError(
                f"Rollback PARK after rematerialization failed: {rollback_park!r}"
            )
        _set_uvm_original_alias_parked_state(rollback_park)

    if int(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES) != int(
        old_backing_bytes
    ):
        raise RuntimeError("Rollback rebuilt an unexpected backing size")
    if pre_state != _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE:
        raise RuntimeError("Rollback rebuild did not restore the lifecycle state")
    if _uvm_original_parameter_va_snapshot() != parameter_vas:
        raise RuntimeError("Rollback rebuild changed Parameter VAs")
    if tuple(
        int(arena.upload.data_ptr())
        for _, arena in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.items())
    ) != upload_vas:
        raise RuntimeError("Rollback rebuild changed upload VAs")
    if tuple(
        int(binding.source.data_ptr())
        for _, binding in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items())
        if binding.source is not None
    ) != source_vas:
        raise RuntimeError("Rollback rebuild changed pinned source VAs")
    return {
        "ok": True,
        "mode": "backing_rematerialized",
        "retained_error": retained_error,
        "destroy": rollback_destroy,
        "slot_close": rollback_close,
        "finalize": rollback_finalize,
        "park": rollback_park,
        "wall_ns": time.perf_counter_ns() - started_ns,
    }


def switch_cpu_offload_uvm_original_va_plan_class(
    class_name: str,
    *,
    reason: str,
    graphs_retained: bool,
    defer_dual_refresh: bool = False,
    expected_selected_bytes: int | None = None,
    pressure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Switch one fixed class while keeping all Graph-visible VAs alive."""
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ACTIVE_PLAN_CLASS
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
    global _CPU_OFFLOAD_UVM_PLAN_GENERATION
    global _CPU_OFFLOAD_UVM_PLAN_CLASS_INSTRUCTIONS

    target_class = _normalize_uvm_plan_class_name(class_name)
    if target_class not in _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS:
        return {
            "ok": False,
            "reason": "plan_class_not_registered",
            "class_name": target_class,
        }
    if graphs_retained is not True:
        raise RuntimeError(
            "Stable class switching requires an explicit retained-Graph contract"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Plan-class switching requires a non-empty reason")
    pressure_record = dict(pressure or {})
    json.dumps(pressure_record, sort_keys=True)

    with _CPU_OFFLOAD_UVM_STATIC_ALIAS_LOCK:
        old_class = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ACTIVE_PLAN_CLASS
        if old_class == target_class:
            return {
                "ok": True,
                "reason": "plan_class_already_active",
                "old_class": old_class,
                "new_class": target_class,
                "generation": int(_CPU_OFFLOAD_UVM_PLAN_GENERATION),
            }
        if old_class not in _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS:
            raise RuntimeError("The active stable plan class is not registered")
        pre_state = _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
        if pre_state not in ("sealed", "remote_parked"):
            raise RuntimeError(
                "Stable class switching requires sealed/parked state, got "
                f"{pre_state!r}"
            )
        if pre_state == "remote_parked" and not (
            _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
        ):
            raise RuntimeError(
                "A parked plan class requires an active remote-read topology"
            )
        if not _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS:
            raise RuntimeError("Stable class switching found no bindings")

        target_plan = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS[target_class]
        old_plan = dict(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER)
        old_plan_file = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE
        instruction_enabled = _strict_env_flag(
            "VLLM_CPU_OFFLOAD_PLAN_CLASS_PRECOMPILE", False
        )
        instruction_key = (
            str(old_class),
            str(target_class),
            _uvm_original_alias_plan_signature(old_plan),
        )
        instruction = (
            _CPU_OFFLOAD_UVM_PLAN_CLASS_INSTRUCTIONS.get(instruction_key)
            if instruction_enabled
            else None
        )
        instruction_cache_hit = instruction is not None
        instruction_id = f"{old_class}->{target_class}"
        if instruction is not None:
            old_geometry = instruction["old_geometry"]
            target_geometry = instruction["target_geometry"]
            slot_assignment = instruction["slot_assignment"]
            active_target_plan = instruction["active_target_plan"]
            old_selected_bytes = int(instruction["old_selected_bytes"])
            new_selected_bytes = int(instruction["new_selected_bytes"])
            target_backing_bytes = int(instruction["target_backing_bytes"])
        else:
            old_geometry = _uvm_original_alias_plan_geometry(old_plan)
            raw_target_geometry = _uvm_original_alias_plan_geometry(target_plan)
            target_geometry, slot_assignment = (
                _uvm_original_alias_stabilize_plan_slots(
                    old_geometry,
                    raw_target_geometry,
                )
            )
            active_target_plan: dict[int, tuple[int, ...]] = {}
            for full_name, values in target_geometry.items():
                expert_ids = values[0]
                if expert_ids is None:
                    continue
                layer_index = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN[
                    full_name
                ].layer_index
                previous = active_target_plan.setdefault(
                    layer_index, tuple(int(value) for value in expert_ids)
                )
                if previous != tuple(expert_ids):
                    raise RuntimeError(
                        "Stable slot assignment differs within one expert layer"
                    )
            old_selected_bytes = sum(value[2] for value in old_geometry.values())
            new_selected_bytes = sum(value[2] for value in target_geometry.values())
            target_backing_bytes = 0
            for arena in _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.values():
                arena_widths = {
                    int(target_geometry[binding.full_parameter_name][2])
                    for binding in _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.values()
                    if binding.arena is arena
                }
                if len(arena_widths) != 1:
                    raise RuntimeError(
                        "Target class has inconsistent widths in one stable arena"
                    )
                target_backing_bytes += arena_widths.pop()
        if (
            expected_selected_bytes is not None
            and new_selected_bytes != int(expected_selected_bytes)
        ):
            raise RuntimeError(
                "Plan class selected an unexpected byte count: "
                f"actual={new_selected_bytes}, expected={expected_selected_bytes}"
            )
        if new_selected_bytes > _CPU_OFFLOAD_MAX_BYTES:
            raise RuntimeError(
                "Plan class exceeds configured CPU-offload budget: "
                f"selected={new_selected_bytes}, budget={_CPU_OFFLOAD_MAX_BYTES}"
            )

        device = int(_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0)
        parameter_vas = _uvm_original_parameter_va_snapshot()
        upload_vas = tuple(
            int(arena.upload.data_ptr())
            for _, arena in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.items())
        )
        source_vas = tuple(
            int(binding.source.data_ptr())
            for _, binding in sorted(
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
            )
            if binding.source is not None
        )
        if len(source_vas) != len(_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS):
            raise RuntimeError("Stable class switching requires every pinned source")
        old_remote_map_active = bool(_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE)
        old_remote_map_result = (
            dict(_CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT)
            if _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT is not None
            else None
        )
        old_backing_bytes = int(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES)
        same_capacity = target_backing_bytes == old_backing_bytes
        started_ns = time.perf_counter_ns()
        park_result: dict[str, Any] | None = None
        rebind_result: dict[str, Any] | None = None
        destroy_result: dict[str, Any] | None = None
        close_result: dict[str, Any] | None = None
        finalize_result: dict[str, Any] | None = None
        residency_result: dict[str, Any] | None = None
        residency_before_detach: dict[str, Any] | None = None
        source_refill_result: dict[str, Any] | None = None
        preallocate_result: dict[str, Any] | None = None
        resume_result: dict[str, Any] | None = None
        quiesced_delta_result: dict[str, Any] | None = None
        dual_discard_result: dict[str, Any] | None = None
        post_restore_dual_refresh_result: dict[str, Any] | None = None
        rollback_result: dict[str, Any] | None = None
        old_map_entries: tuple[
            tuple[int, int, int, int, int, int, int], ...
        ] = ()
        target_map_entries: tuple[
            tuple[int, int, int, int, int, int, int], ...
        ] = ()
        added_preallocate_entries: tuple[
            tuple[int, int, int, int, int, int, int], ...
        ] = ()
        rollback_preallocate_entries: tuple[
            tuple[int, int, int, int, int, int, int], ...
        ] = ()
        old_consumer_entries: tuple[
            tuple[int, int, int, int, int, int, int], ...
        ] = ()
        rebind_committed = False
        old_session_destroyed = False
        fast_rollback_safe = True
        quiesced_delta_used = False
        delta_add_entries: tuple[
            tuple[int, int, int, int, int, int, int], ...
        ] = ()
        delta_remove_entries: tuple[
            tuple[int, int, int, int, int, int, int], ...
        ] = ()
        try:
            from vllm.device_allocator import uvm, uvm_static_alias

            pte_flags = next(
                iter(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.values())
            ).pte_flags
            if instruction is not None:
                if int(instruction["pte_flags"]) != int(pte_flags):
                    raise RuntimeError(
                        "Precompiled plan-class instruction PTE mode changed"
                    )
                old_map_entries = instruction["old_map_entries"]
            else:
                old_map_entries = tuple(
                    _uvm_original_alias_map_entries(pte_flags)
                )
            old_consumer_entries = tuple(
                entry
                for entry in old_map_entries
                if not (
                    entry[3]
                    & uvm_static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW
                )
            )
            torch.cuda.synchronize(device)
            force_sync_dual_transaction = (
                _strict_env_flag(
                    "VLLM_CPU_OFFLOAD_PLAN_CLASS_FORCE_SYNC_MIGRATION", False
                )
                and _strict_env_flag(
                    "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW", False
                )
            )
            if force_sync_dual_transaction:
                dual_discard_result = dict(
                    uvm_static_alias.driver_static_alias_seal(
                        int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
                        device=device,
                        discard_dual=True,
                    )
                )
                if dual_discard_result.get("ok") is not True:
                    raise RuntimeError(
                        "Failed to discard the full dual view before synchronous "
                        f"Plan Class migration: {dual_discard_result!r}"
                    )
                # DISCARD_DUAL restores the original UVM hierarchy, but the
                # current cold expert slices are still live slot aliases. Some
                # newly-cold slices share a 2 MiB VA block with those aliases;
                # a residency transaction over such a mixed block must first
                # remove every consumer slot mapping. Keep the session parked
                # through target residency and REBIND. RESUME then installs the
                # new slot recipe before REFRESH_DUAL freezes its two views.
                if pre_state == "sealed":
                    park_result = dict(
                        uvm_static_alias.driver_static_alias_park(
                            int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
                            device=device,
                            synchronize=False,
                            defer_cpu_restore=False,
                        )
                    )
                    if park_result.get("side_effect_indeterminate") or (
                        park_result.get("poisoned")
                        and park_result.get("ok") is not True
                    ):
                        fast_rollback_safe = False
                    if park_result.get("ok") is not True:
                        raise RuntimeError(
                            "PARK before synchronous Plan Class migration failed: "
                            f"{park_result!r}"
                        )
                    _set_uvm_original_alias_parked_state(park_result)
            _apply_uvm_original_alias_plan_geometry(
                target_geometry, refill_sources=False
            )
            source_refill_result = _refill_uvm_original_alias_plan_delta(
                old_geometry,
                target_geometry,
            )
            if int(source_refill_result["changed_slot_count"]) > 0:
                torch.cuda.synchronize(device)
            if instruction is not None:
                target_map_entries = instruction["target_map_entries"]
                old_preallocate_entries = instruction[
                    "old_preallocate_entries"
                ]
                target_preallocate_entries = instruction[
                    "target_preallocate_entries"
                ]
                residency_ranges = instruction["residency_ranges"]
            else:
                target_map_entries = tuple(
                    _uvm_original_alias_map_entries(pte_flags)
                )
                old_preallocate_entries = tuple(
                    (
                        entry[4],
                        entry[5],
                        entry[0],
                        entry[1],
                        entry[2],
                        entry[6],
                        uvm.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT,
                    )
                    for entry in old_map_entries
                    if entry[3]
                    & uvm_static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW
                )
                target_preallocate_entries = tuple(
                    _uvm_original_alias_preallocate_entries()
                )
                residency_ranges = (
                    _uvm_original_alias_plan_residency_delta_ranges(
                        old_geometry, target_geometry
                    )
                )
            old_backing_keys = {
                (entry[2], entry[3], entry[4])
                for entry in old_preallocate_entries
            }
            target_backing_keys = {
                (entry[2], entry[3], entry[4])
                for entry in target_preallocate_entries
            }
            added_preallocate_entries = tuple(
                entry
                for entry in target_preallocate_entries
                if (entry[2], entry[3], entry[4]) not in old_backing_keys
            )
            rollback_preallocate_entries = tuple(
                entry
                for entry in old_preallocate_entries
                if (entry[2], entry[3], entry[4]) not in target_backing_keys
            )
            if instruction is not None and "delta_add_entries" in instruction:
                delta_add_entries = instruction["delta_add_entries"]
                delta_remove_entries = instruction["delta_remove_entries"]
            else:
                old_entry_set = set(old_map_entries)
                target_entry_set = set(target_map_entries)
                delta_add_entries = tuple(
                    entry
                    for entry in target_map_entries
                    if entry not in old_entry_set
                )
                delta_remove_entries = tuple(
                    entry
                    for entry in old_map_entries
                    if entry not in target_entry_set
                )

            quiesced_delta_enabled = _strict_env_flag(
                "VLLM_CPU_OFFLOAD_PLAN_CLASS_QUIESCED_DELTA", False
            )
            quiesced_delta_shape_supported = True
            if delta_remove_entries:
                try:
                    _uvm_original_alias_delta_batches(
                        delta_remove_entries,
                        maximum_entries=(
                            uvm_static_alias.UVM_MOE_STATIC_ALIAS_MAX_ENTRIES
                        ),
                        keep_backing_identities_whole=True,
                    )
                except ValueError:
                    quiesced_delta_shape_supported = False
            quiesced_delta_eligible = (
                quiesced_delta_enabled
                and not force_sync_dual_transaction
                # An append must install new consumer aliases and is valid
                # only while the alias-prefill topology is sealed.  A remove
                # can also shrink a parked session: its consumer PTEs are
                # already absent, but its retained registry/backing must be
                # updated before a future RESUME.
                and (
                    pre_state == "sealed"
                    or (
                        pre_state == "remote_parked"
                        and bool(delta_remove_entries)
                        and not bool(delta_add_entries)
                    )
                )
                and bool(delta_add_entries) != bool(delta_remove_entries)
                and quiesced_delta_shape_supported
            )
            # Newly cold experts must become CPU-resident before allocating
            # their slot backing, both to bound peak HBM and to preserve the
            # upload source. Newly hot experts are the inverse: their managed
            # Parameter ranges must be detached from the old consumer aliases
            # before they can be restored to GPU. Split mixed transitions into
            # a pre-REBIND add-cold phase and post-REBIND remove-cold phase.
            added_cold_ranges, removed_cold_ranges = residency_ranges
            deferred_removed_residency = bool(removed_cold_ranges)
            if added_cold_ranges:
                residency_before_detach = (
                    _prepare_uvm_original_alias_plan_residency_delta(
                        old_geometry,
                        target_geometry,
                        device=device,
                        prepared_ranges=(added_cold_ranges, ()),
                    )
                )

            if added_preallocate_entries:
                preallocate_result = dict(
                    uvm.driver_slot_preallocate_ranges(
                        added_preallocate_entries,
                        device=device,
                    )
                )
                if preallocate_result.get("ok") is not True:
                    raise RuntimeError(
                        "Failed to preallocate added class backing: "
                        f"{preallocate_result!r}"
                    )
            else:
                preallocate_result = {
                    "ok": True,
                    "reason": "no_added_backing",
                    "requested_count": 0,
                    "requested_bytes": 0,
                }

            if quiesced_delta_eligible:
                if delta_add_entries:
                    quiesced_delta_result = dict(
                        _uvm_original_alias_quiesced_delta_batched(
                            session_id=int(
                                _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID
                            ),
                            entries=delta_add_entries,
                            operation="append",
                            device=device,
                            defer_dual_refresh=(
                                defer_dual_refresh
                                or force_sync_dual_transaction
                            ),
                        )
                    )
                    quiesced_delta_result["mode"] = "quiesced_append"
                else:
                    quiesced_delta_result = dict(
                        _uvm_original_alias_quiesced_delta_batched(
                            session_id=int(
                                _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID
                            ),
                            entries=delta_remove_entries,
                            operation="remove",
                            device=device,
                            defer_dual_refresh=False,
                        )
                    )
                    quiesced_delta_result["mode"] = "quiesced_remove"
                if quiesced_delta_result.get("side_effect_indeterminate") or (
                    quiesced_delta_result.get("poisoned")
                    and quiesced_delta_result.get("ok") is not True
                ):
                    fast_rollback_safe = False
                if quiesced_delta_result.get("ok") is not True:
                    raise RuntimeError(
                        "Static-alias quiesced delta failed: "
                        f"{quiesced_delta_result!r}"
                    )
                quiesced_delta_used = True
                rebind_result = quiesced_delta_result
                rebind_committed = True
            else:
                if (
                    pre_state == "sealed"
                    and _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "sealed"
                ):
                    park_result = dict(
                        uvm_static_alias.driver_static_alias_park(
                            int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
                            device=device,
                            synchronize=False,
                            defer_cpu_restore=True,
                        )
                    )
                    if park_result.get("side_effect_indeterminate") or (
                        park_result.get("poisoned")
                        and park_result.get("ok") is not True
                    ):
                        fast_rollback_safe = False
                    if park_result.get("ok") is not True:
                        raise RuntimeError(
                            "PARK before class switch failed: "
                            f"{park_result!r}"
                        )
                    _set_uvm_original_alias_parked_state(park_result)

                rebind_result = dict(
                    uvm_static_alias.driver_static_alias_rebind(
                        int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
                        target_map_entries,
                        device=device,
                        synchronize=False,
                        resize=True,
                        expected_backing_bytes=target_backing_bytes,
                    )
                )
                if rebind_result.get("side_effect_indeterminate") or (
                    rebind_result.get("poisoned")
                    and rebind_result.get("ok") is not True
                ):
                    fast_rollback_safe = False
                if rebind_result.get("ok") is not True:
                    raise RuntimeError(
                        "Static-alias resizing REBIND failed: "
                        f"{rebind_result!r}"
                    )
                rebind_committed = True
                if pre_state == "sealed" and not force_sync_dual_transaction:
                    resume_result = dict(
                        uvm_static_alias.driver_static_alias_resume(
                            int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
                            device=device,
                            synchronize=False,
                        )
                    )
                    if resume_result.get("side_effect_indeterminate") or (
                        resume_result.get("poisoned")
                        and resume_result.get("ok") is not True
                    ):
                        fast_rollback_safe = False
                    if resume_result.get("ok") is not True:
                        raise RuntimeError(
                            "RESUME after resizing REBIND failed: "
                            f"{resume_result!r}"
                        )
                    _set_uvm_original_alias_sealed_state(resume_result)

            if deferred_removed_residency:
                residency_after_detach = (
                    _prepare_uvm_original_alias_plan_residency_delta(
                        old_geometry,
                        target_geometry,
                        device=device,
                        prepared_ranges=((), removed_cold_ranges),
                        advise_removed=False,
                    )
                )
                residency_result = (
                    _merge_uvm_original_alias_plan_residency_phases(
                        residency_before_detach,
                        residency_after_detach,
                    )
                )
            else:
                residency_result = residency_before_detach

            # REFRESH_DUAL derives the immutable selection from the live slot
            # PTE mask. A force-synchronous transaction discarded the previous
            # children and kept the session parked while residency changed, so
            # its mask is intentionally empty until RESUME. Publish the new
            # REBIND topology first; the scheduler quiescence certificate makes
            # this intermediate GPU view safe and no model kernel is launched.
            if pre_state == "sealed" and force_sync_dual_transaction:
                resume_result = dict(
                    uvm_static_alias.driver_static_alias_resume(
                        int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
                        device=device,
                        synchronize=False,
                    )
                )
                if resume_result.get("side_effect_indeterminate") or (
                    resume_result.get("poisoned")
                    and resume_result.get("ok") is not True
                ):
                    fast_rollback_safe = False
                if resume_result.get("ok") is not True:
                    raise RuntimeError(
                        "RESUME after synchronous Plan Class migration failed: "
                        f"{resume_result!r}"
                    )
                _set_uvm_original_alias_sealed_state(resume_result)

            # A synchronous Plan Class transaction deliberately destroyed the
            # frozen children before changing expert residency. Rebuild them
            # now unless the enclosing KV grow transaction requested one final
            # refresh after its own physical migration. The legacy hot-restore
            # path still requires an immediate full rebuild as before.
            immediate_dual_refresh = (
                deferred_removed_residency or force_sync_dual_transaction
            )
            if immediate_dual_refresh:
                post_restore_dual_refresh_result = (
                    refresh_cpu_offload_uvm_original_va_full_dual_view(
                        nested_refresh=False
                    )
                )
                if post_restore_dual_refresh_result.get("ok") is not True:
                    raise RuntimeError(
                        "Post-switch full dual-view refresh failed: "
                        f"{post_restore_dual_refresh_result!r}"
                    )

            if _uvm_original_parameter_va_snapshot() != parameter_vas:
                raise RuntimeError("Parameter VAs changed during stable class switch")
            if tuple(
                int(arena.upload.data_ptr())
                for _, arena in sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.items())
            ) != upload_vas:
                raise RuntimeError("Upload VAs changed during stable class switch")
            if tuple(
                int(binding.source.data_ptr())
                for _, binding in sorted(
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                )
                if binding.source is not None
            ) != source_vas:
                raise RuntimeError("Pinned source VAs changed during class switch")
            if int(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES) != int(
                target_backing_bytes
            ):
                raise RuntimeError(
                    "Plan-class switch materialized an unexpected backing size"
                )
            if pre_state != _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE:
                raise RuntimeError(
                    "Plan-class switch changed the serving lifecycle state"
                )

            instruction_registered = False
            if instruction_enabled and instruction is None:
                _CPU_OFFLOAD_UVM_PLAN_CLASS_INSTRUCTIONS[instruction_key] = {
                    "instruction_id": instruction_id,
                    "old_geometry": old_geometry,
                    "target_geometry": target_geometry,
                    "slot_assignment": slot_assignment,
                    "active_target_plan": dict(active_target_plan),
                    "old_selected_bytes": int(old_selected_bytes),
                    "new_selected_bytes": int(new_selected_bytes),
                    "target_backing_bytes": int(target_backing_bytes),
                    "pte_flags": int(pte_flags),
                    "old_map_entries": old_map_entries,
                    "target_map_entries": target_map_entries,
                    "old_preallocate_entries": old_preallocate_entries,
                    "target_preallocate_entries": target_preallocate_entries,
                    "residency_ranges": residency_ranges,
                    "delta_add_entries": delta_add_entries,
                    "delta_remove_entries": delta_remove_entries,
                }
                instruction_registered = True

            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER = dict(
                active_target_plan
            )
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE = (
                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASS_FILES[target_class]
            )
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ACTIVE_PLAN_CLASS = target_class
            _CPU_OFFLOAD_UVM_PLAN_GENERATION += 1
            result = {
                "schema": "vllm_uvm_incremental_plan_class_switch_v2",
                "ok": True,
                "reason": reason,
                "old_class": old_class,
                "new_class": target_class,
                "generation": int(_CPU_OFFLOAD_UVM_PLAN_GENERATION),
                "pre_state": pre_state,
                "post_state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
                "same_capacity_rebind": bool(same_capacity),
                "incremental_topology": True,
                "session_reused": True,
                "graphs_retained": True,
                "instruction_id": instruction_id,
                "instruction_cache_enabled": bool(instruction_enabled),
                "instruction_cache_hit": bool(instruction_cache_hit),
                "instruction_registered": bool(instruction_registered),
                "instruction_cached_entry_count": int(len(target_map_entries)),
                "quiesced_delta_enabled": bool(quiesced_delta_enabled),
                "quiesced_delta_eligible": bool(quiesced_delta_eligible),
                "quiesced_delta_used": bool(quiesced_delta_used),
                "quiesced_delta_add_count": int(len(delta_add_entries)),
                "quiesced_delta_remove_count": int(len(delta_remove_entries)),
                "dual_view_refresh_deferred": bool(
                    defer_dual_refresh
                    and (
                        quiesced_delta_used and bool(delta_add_entries)
                    )
                ),
                "enclosing_migration_dual_refresh_requested": bool(
                    defer_dual_refresh
                ),
                "pre_migration_dual_discard": dual_discard_result,
                "quiesced_delta": quiesced_delta_result,
                "instruction_registry_size": int(
                    len(_CPU_OFFLOAD_UVM_PLAN_CLASS_INSTRUCTIONS)
                ),
                "old_selected_bytes": int(old_selected_bytes),
                "new_selected_bytes": int(new_selected_bytes),
                "old_backing_bytes": old_backing_bytes,
                "new_backing_bytes": int(
                    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
                ),
                "stable_envelope_bytes": int(
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_ENVELOPE_BYTES
                ),
                "parameter_vas": parameter_vas,
                "upload_vas": upload_vas,
                "source_vas": source_vas,
                "pressure": pressure_record,
                "slot_assignment": slot_assignment,
                "park": park_result,
                "source_refill": source_refill_result,
                "source_refill_saved_bytes": max(
                    0,
                    int(new_selected_bytes)
                    - int((source_refill_result or {}).get("changed_bytes", 0)),
                ),
                "preallocate": preallocate_result,
                "rebind": rebind_result,
                "resume": resume_result,
                "retained_backing_count": int(
                    (rebind_result or {}).get("retained_backing_count", 0)
                ),
                "added_backing_count": int(
                    (rebind_result or {}).get("added_backing_count", 0)
                ),
                "removed_backing_count": int(
                    (rebind_result or {}).get("removed_backing_count", 0)
                ),
                "destroy": destroy_result,
                "slot_close": close_result,
                "finalize": finalize_result,
                "residency": residency_result,
                "post_restore_dual_refresh": post_restore_dual_refresh_result,
                "wall_ns": time.perf_counter_ns() - started_ns,
            }
            _CPU_OFFLOAD_UVM_PLAN_CLASS_SWITCH_HISTORY.append(result)
            _append_cpu_offload_manifest(
                {
                    "event": "cpu_offload_uvm_stable_plan_class_switched",
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    **result,
                }
            )
            return result
        except Exception as exc:
            rollback_error: Exception | None = None
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER = old_plan
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE = old_plan_file
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ACTIVE_PLAN_CLASS = old_class
            try:
                rollback_result = (
                    _rollback_uvm_original_alias_plan_class_switch(
                        old_geometry=old_geometry,
                        target_geometry=target_geometry,
                        old_consumer_entries=old_consumer_entries,
                        pre_state=pre_state,
                        same_capacity=same_capacity,
                        rebind_committed=rebind_committed,
                        fast_rollback_safe=fast_rollback_safe,
                        old_session_destroyed=old_session_destroyed,
                        old_backing_bytes=old_backing_bytes,
                        parameter_vas=parameter_vas,
                        upload_vas=upload_vas,
                        source_vas=source_vas,
                        old_remote_map_active=old_remote_map_active,
                        old_remote_map_result=old_remote_map_result,
                        device=device,
                        old_map_entries=old_map_entries,
                        rollback_preallocate_entries=(
                            rollback_preallocate_entries
                        ),
                        orphan_preallocate_entries=added_preallocate_entries,
                    )
                )
            except Exception as rollback_exc:
                rollback_error = rollback_exc
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "poisoned"
                _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
                rollback_result = {
                    "ok": False,
                    "reason": "driver_topology_rollback_failed",
                    "error": repr(rollback_exc),
                }
            failed = {
                "schema": "vllm_uvm_incremental_plan_class_switch_v2",
                "ok": False,
                "reason": reason,
                "old_class": old_class,
                "new_class": target_class,
                "pre_state": pre_state,
                "post_state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
                "same_capacity_rebind": bool(same_capacity),
                "incremental_topology": True,
                "graphs_retained": True,
                "pressure": pressure_record,
                "slot_assignment": slot_assignment,
                "park": park_result,
                "source_refill": source_refill_result,
                "preallocate": preallocate_result,
                "rebind": rebind_result,
                "resume": resume_result,
                "destroy": destroy_result,
                "slot_close": close_result,
                "finalize": finalize_result,
                "residency": residency_result,
                "pre_migration_dual_discard": dual_discard_result,
                "post_restore_dual_refresh": post_restore_dual_refresh_result,
                "rollback": rollback_result,
                "error": repr(exc),
                "wall_ns": time.perf_counter_ns() - started_ns,
            }
            _CPU_OFFLOAD_UVM_PLAN_CLASS_SWITCH_HISTORY.append(failed)
            _append_cpu_offload_manifest(
                {
                    "event": "cpu_offload_uvm_stable_plan_class_switch_failed",
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    **failed,
                }
            )
            if rollback_error is not None:
                raise RuntimeError(
                    "Stable plan-class switch and driver topology rollback "
                    "both failed; the process is poisoned"
                ) from rollback_error
            return failed


def rebuild_cpu_offload_uvm_original_va_expert_plan(
    expert_plan_file: str,
    *,
    reason: str,
    graphs_destroyed: bool,
    expected_selected_bytes: int | None = None,
    pressure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild one original-VA expert plan at an idle request boundary.

    The complete expert Parameters stay in their original managed allocations.
    Long-lived module-forward closures retain their binding objects; this
    function replaces the arena/source/range fields of those objects in place.
    Consequently Parameter objects and VAs do not change, while both the
    prefill upload plan and decode remote-read ranges can change.

    Concrete CUDA Graphs must be reset by their model-runner owner first.  The
    caller must recapture them after this function returns a sealed topology.
    """
    global _CPU_OFFLOAD_BYTES
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT
    global _CPU_OFFLOAD_UVM_PLAN_GENERATION

    if not cpu_offload_uvm_original_va_static_alias_enabled():
        raise RuntimeError("Dynamic expert-plan rebuild requires original-VA aliases")
    if _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE != "prefill_graph_remote_decode":
        raise RuntimeError(
            "Dynamic expert-plan rebuild currently requires "
            "prefill_graph_remote_decode"
        )
    if graphs_destroyed is not True:
        raise RuntimeError(
            "Dynamic expert-plan rebuild requires every concrete CUDA Graph "
            "to be reset first"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Dynamic expert-plan rebuild requires a non-empty reason")
    plan_path = os.path.realpath(str(expert_plan_file).strip())
    if not plan_path or not os.path.isfile(plan_path):
        raise ValueError(f"Dynamic expert plan is absent: {expert_plan_file!r}")
    if expected_selected_bytes is not None and expected_selected_bytes <= 0:
        raise ValueError("expected_selected_bytes must be positive when provided")
    pressure_record = dict(pressure or {})
    try:
        json.dumps(pressure_record, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("pressure metadata must be JSON serializable") from exc

    with _CPU_OFFLOAD_UVM_STATIC_ALIAS_LOCK:
        pre_state = _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
        if pre_state not in ("sealed", "remote", "remote_parked"):
            raise RuntimeError(
                "Dynamic expert-plan rebuild requires a stable serving "
                f"topology, got {pre_state!r}"
            )
        if not _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS:
            raise RuntimeError("Dynamic expert-plan rebuild found no bindings")

        new_experts = _load_graph_static_alias_expert_plan(
            plan_path,
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN,
        )
        old_experts = dict(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER)
        old_plan_path = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE
        old_bindings = dict(_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS)
        old_arenas = dict(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS)
        old_cold_ranges = {
            (int(base), int(length))
            for binding in old_bindings.values()
            for base, length in _uvm_original_alias_binding_consumer_ranges(
                binding
            )
        }
        parameter_vas_before = _uvm_original_parameter_va_snapshot()
        if (
            _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
            and parameter_vas_before != _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
        ):
            raise RuntimeError("Parameter VAs changed before dynamic plan rebuild")

        started_ns = time.perf_counter_ns()
        synchronized_ns = started_ns
        destroyed_ns = started_ns
        bindings_rebuilt_ns = started_ns
        hot_restored_ns = started_ns
        pre_destroy_resume: dict[str, Any] | None = None
        destroy_result: dict[str, Any] | None = None
        hot_restore: dict[str, Any] | None = None
        finalize_result: dict[str, Any] | None = None
        attempted_generation = _CPU_OFFLOAD_UVM_PLAN_GENERATION + 1
        try:
            device = int(_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0)
            torch.cuda.synchronize(device)
            synchronized_ns = time.perf_counter_ns()

            # A request ending in remote decode leaves the consumer ranges
            # parked on their original Parameter VAs.  Destroying that session
            # releases its slot backing, but does not first restore packed
            # Parameter ranges that are about to become hot in the next plan.
            # Resume while the old Graphs are already gone and the old session
            # is still intact, then destroy only the sealed topology.
            if pre_state == "remote_parked":
                from vllm.device_allocator import uvm_static_alias

                session_id = int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID)
                if session_id <= 0:
                    raise RuntimeError(
                        "Dynamic expert-plan rebuild found a parked topology "
                        "without an active alias session"
                    )
                pre_destroy_resume = dict(
                    uvm_static_alias.driver_static_alias_resume(
                        session_id,
                        device=device,
                        synchronize=False,
                    )
                )
                if pre_destroy_resume.get("ok") is not True:
                    raise RuntimeError(
                        "Failed to resume the parked alias session before "
                        f"dynamic rebuild destroy: {pre_destroy_resume!r}"
                    )
                _set_uvm_original_alias_sealed_state(pre_destroy_resume)
                torch.cuda.synchronize(device)

            destroy_result = destroy_cpu_offload_graph_static_alias(
                graphs_destroyed=True,
                synchronize=False,
            )
            if destroy_result.get("ok") is not True:
                raise RuntimeError(
                    "Failed to release the previous alias session before "
                    f"dynamic rebuild: {destroy_result!r}"
                )
            destroyed_ns = time.perf_counter_ns()
            _reset_uvm_original_va_graph_certificates_for_plan_rebuild()

            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE = plan_path
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER = new_experts
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS = {}
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS = {}
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES = 0
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "collecting"
            _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
            _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT = None

            for full_name, retained in sorted(old_bindings.items()):
                current = retained.parameter.data
                if int(current.data_ptr()) != int(retained.consumer.data_ptr()):
                    raise RuntimeError(
                        "A Parameter changed VA while rebuilding its expert plan: "
                        f"{full_name!r}"
                    )
                replacement = _create_uvm_original_alias_binding(
                    full_parameter_name=full_name,
                    parameter=retained.parameter,
                    tensor=retained.consumer,
                    device=current.device,
                    requested_bytes=int(retained.parameter_bytes),
                )
                retained.arena = replacement.arena
                retained.consumer = replacement.consumer
                retained.plan = replacement.plan
                retained.requested_bytes = replacement.requested_bytes
                retained.parameter_bytes = replacement.parameter_bytes
                retained.expert_ids = replacement.expert_ids
                retained.expert_stride_bytes = replacement.expert_stride_bytes
                retained.consumer_ranges = replacement.consumer_ranges
                retained.source = None
                replacement.arena.consumers[replacement.plan.consumer_index] = (
                    retained
                )
                retained.parameter._vllm_uvm_original_alias_binding = retained

            _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS = old_bindings
            selected_bytes = sum(
                int(binding.requested_bytes) for binding in old_bindings.values()
            )
            if (
                expected_selected_bytes is not None
                and selected_bytes != int(expected_selected_bytes)
            ):
                raise RuntimeError(
                    "Dynamic expert plan selected an unexpected byte count: "
                    f"actual={selected_bytes}, expected={expected_selected_bytes}"
                )
            if selected_bytes > _CPU_OFFLOAD_MAX_BYTES:
                raise RuntimeError(
                    "Dynamic expert plan exceeds the configured offload budget: "
                    f"selected={selected_bytes}, budget={_CPU_OFFLOAD_MAX_BYTES}"
                )
            _CPU_OFFLOAD_BYTES = selected_bytes
            bindings_rebuilt_ns = time.perf_counter_ns()

            # Experts removed from the cold set may still carry the previous
            # request's CPU remote mapping.  Restore the complete complement to
            # GPU before finalize audits the new hot/cold partition.
            from vllm.device_allocator import uvm

            hot_ranges = [
                hot_range
                for _, binding in sorted(old_bindings.items())
                for hot_range in _uvm_original_alias_binding_hot_ranges(binding)
            ]
            hot_bytes = sum(length for _, length in hot_ranges)
            torch.cuda.empty_cache()
            advised = int(
                uvm.batch_advise_ranges_preferred_location(hot_ranges, device)
            )
            prefetched, executor, driver_result = (
                uvm.batch_prefetch_ranges_to_device_with_executor(
                    hot_ranges,
                    device,
                    executor="cpp_batch",
                )
            )
            if advised != len(hot_ranges) or int(prefetched) != len(hot_ranges):
                raise RuntimeError(
                    "Failed to restore every new hot expert range before "
                    "dynamic finalize: "
                    f"advised={advised}, prefetched={prefetched}, "
                    f"expected={len(hot_ranges)}, executor={executor!r}, "
                    f"driver_result={driver_result}"
                )
            torch.cuda.synchronize(device)
            hot_residency = dict(
                uvm.driver_batch_query_ranges_residency_bytes(
                    hot_ranges,
                    device=device,
                    wait_for_trackers=False,
                )
            )
            expected_hot_cpu_bytes = (
                _uvm_original_alias_expected_shadow_cpu_bytes(hot_ranges)
            )
            if hot_ranges and not (
                hot_residency.get("ok") is True
                and int(hot_residency.get("queried_bytes", -1)) == hot_bytes
                and int(hot_residency.get("gpu_bytes", -1)) == hot_bytes
                and int(hot_residency.get("cpu_bytes", -1))
                == expected_hot_cpu_bytes
                and int(hot_residency.get("unknown_bytes", -1)) == 0
            ):
                raise RuntimeError(
                    "New hot expert residency is incomplete before dynamic "
                    f"finalize: {hot_residency!r}"
                )
            hot_restore = {
                "range_count": len(hot_ranges),
                "bytes": int(hot_bytes),
                "advised_count": advised,
                "prefetched_count": int(prefetched),
                "executor": str(executor),
                "driver_result": int(driver_result),
                "residency": hot_residency,
                "expected_cpu_shadow_bytes": int(expected_hot_cpu_bytes),
            }
            hot_restored_ns = time.perf_counter_ns()

            new_cold_ranges = {
                (int(base), int(length))
                for binding in old_bindings.values()
                for base, length in _uvm_original_alias_binding_consumer_ranges(
                    binding
                )
            }
            shared_cpu_ranges = old_cold_ranges & new_cold_ranges
            finalize_result = finalize_cpu_offload_graph_static_alias(
                known_cpu_ranges=shared_cpu_ranges,
            )
            if finalize_result.get("ok") is not True:
                raise RuntimeError(
                    f"Dynamic expert-plan finalize failed: {finalize_result!r}"
                )
            parameter_vas_after = _uvm_original_parameter_va_snapshot()
            if parameter_vas_after != parameter_vas_before:
                raise RuntimeError("Parameter VAs changed during dynamic plan rebuild")
            _CPU_OFFLOAD_UVM_PLAN_GENERATION = attempted_generation
            finished_ns = time.perf_counter_ns()
            with open(plan_path, "rb") as plan_handle:
                plan_sha256 = hashlib.sha256(plan_handle.read()).hexdigest()
            result = {
                "event": "cpu_offload_uvm_original_va_plan_rebuilt",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "ok": True,
                "reason": reason,
                "generation": attempted_generation,
                "pre_state": pre_state,
                "post_state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
                "old_plan_file": old_plan_path or None,
                "new_plan_file": plan_path,
                "new_plan_sha256": plan_sha256,
                "old_selected_expert_count": sum(
                    len(values) for values in old_experts.values()
                ),
                "new_selected_expert_count": sum(
                    len(values) for values in new_experts.values()
                ),
                "selected_bytes": int(selected_bytes),
                "backing_bytes": int(
                    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
                ),
                "parameter_vas_before": parameter_vas_before,
                "parameter_vas_after": parameter_vas_after,
                "pressure": pressure_record,
                "sync_ns": synchronized_ns - started_ns,
                "alias_destroy_ns": destroyed_ns - synchronized_ns,
                "binding_rebuild_ns": bindings_rebuilt_ns - destroyed_ns,
                "hot_restore_ns": hot_restored_ns - bindings_rebuilt_ns,
                "finalize_ns": finished_ns - hot_restored_ns,
                "wall_ns": finished_ns - started_ns,
                "pre_destroy_resume": pre_destroy_resume,
                "destroy": destroy_result,
                "hot_restore": hot_restore,
                "finalize": finalize_result,
            }
            _CPU_OFFLOAD_UVM_PLAN_REBUILD_HISTORY.append(result)
            _append_cpu_offload_manifest(result)
            old_arenas.clear()
            gc.collect()
            return result
        except Exception as exc:
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "poisoned"
            _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
            failed = {
                "event": "cpu_offload_uvm_original_va_plan_rebuilt",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "ok": False,
                "reason": reason,
                "generation": attempted_generation,
                "pre_state": pre_state,
                "post_state": "poisoned",
                "old_plan_file": old_plan_path or None,
                "new_plan_file": plan_path,
                "pressure": pressure_record,
                "wall_ns": time.perf_counter_ns() - started_ns,
                "pre_destroy_resume": pre_destroy_resume,
                "destroy": destroy_result,
                "hot_restore": hot_restore,
                "finalize": finalize_result,
                "error": repr(exc),
            }
            _CPU_OFFLOAD_UVM_PLAN_REBUILD_HISTORY.append(failed)
            _append_cpu_offload_manifest(failed)
            raise RuntimeError(
                "Dynamic original-VA expert-plan rebuild failed; the process "
                "is poisoned and must be torn down"
            ) from exc


def reallocate_cpu_offload_uvm_original_va_expert_parameters(
    expert_plan_file: str,
    *,
    reason: str,
    graphs_destroyed: bool,
    expected_selected_bytes: int | None = None,
    pressure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Rebuild a full-parameter plan at new managed Parameter VAs.

    This is the deliberately expensive non-stable-VA E5 baseline.  It is
    narrower than the production stable-VA rebuild: every selected Parameter
    must be selected in full, every old Graph must already be destroyed, and
    the old sealed alias session is destroyed before new bindings are built.
    Pinned sources retained by the sealed session provide the authoritative
    bytes, so changing the Parameter allocation cannot read an already
    detached consumer mapping.

    Each replacement allocation is made while its corresponding old
    Parameter allocation is still live, which guarantees a different VA for
    that Parameter.  It is then migrated to CPU before the next replacement;
    this avoids temporarily adding the complete selected weight set to HBM.
    The caller must recapture every CUDA Graph after this function returns.
    """
    global _CPU_OFFLOAD_BYTES
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_ENVELOPE_BYTES
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT
    global _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
    global _CPU_OFFLOAD_UVM_PLAN_GENERATION

    if not cpu_offload_uvm_original_va_static_alias_enabled():
        raise RuntimeError("Non-stable-VA rebuild requires original-VA aliases")
    if _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE != "prefill_graph_remote_decode":
        raise RuntimeError(
            "Non-stable-VA rebuild currently requires "
            "prefill_graph_remote_decode"
        )
    if graphs_destroyed is not True:
        raise RuntimeError(
            "Non-stable-VA rebuild requires every concrete CUDA Graph to be "
            "reset first"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Non-stable-VA rebuild requires a non-empty reason")
    plan_path = os.path.realpath(str(expert_plan_file).strip())
    if not plan_path or not os.path.isfile(plan_path):
        raise ValueError(f"Non-stable-VA plan is absent: {expert_plan_file!r}")
    pressure_record = dict(pressure or {})
    json.dumps(pressure_record, sort_keys=True)

    with _CPU_OFFLOAD_UVM_STATIC_ALIAS_LOCK:
        pre_state = _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
        if pre_state not in ("sealed", "remote", "remote_parked"):
            raise RuntimeError(
                "Non-stable-VA rebuild requires a stable serving topology, "
                f"got {pre_state!r}"
            )
        if not _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS:
            raise RuntimeError("Non-stable-VA rebuild found no bindings")

        new_experts = _load_graph_static_alias_expert_plan(
            plan_path,
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN,
        )
        retained_bindings = dict(_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS)
        old_plan_path = _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE
        old_vas_by_name = {
            name: int(binding.parameter.data.data_ptr())
            for name, binding in sorted(retained_bindings.items())
        }
        old_vas = tuple(old_vas_by_name.values())
        if (
            _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
            and old_vas != _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
        ):
            raise RuntimeError(
                "Parameter VAs changed before the non-stable-VA rebuild"
            )

        # The E5 non-stable baseline must have an authoritative full tensor in
        # pinned host memory.  A partial cold-expert source cannot reconstruct
        # the hot complement after the old Parameter allocation is released.
        for name, binding in sorted(retained_bindings.items()):
            source = binding.source
            parameter = binding.parameter.data
            if (
                int(binding.requested_bytes) != int(binding.parameter_bytes)
                or source is None
                or not source.is_pinned()
                or int(source.numel() * source.element_size())
                != int(binding.parameter_bytes)
                or tuple(source.size()) != tuple(parameter.size())
                or tuple(source.stride()) != tuple(parameter.stride())
                or source.dtype != parameter.dtype
            ):
                raise RuntimeError(
                    "Non-stable-VA rebuild requires a full pinned source for "
                    f"every selected Parameter: {name!r}"
                )

        started_ns = time.perf_counter_ns()
        attempted_generation = _CPU_OFFLOAD_UVM_PLAN_GENERATION + 1
        destroy_result: dict[str, Any] | None = None
        finalize_result: dict[str, Any] | None = None
        migration_records: list[dict[str, Any]] = []
        try:
            device = int(_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0)
            torch.cuda.synchronize(device)
            synchronized_ns = time.perf_counter_ns()

            destroy_result = destroy_cpu_offload_graph_static_alias(
                graphs_destroyed=True,
                synchronize=False,
            )
            if destroy_result.get("ok") is not True:
                raise RuntimeError(
                    "Failed to destroy the old alias session before "
                    f"Parameter reallocation: {destroy_result!r}"
                )
            destroyed_ns = time.perf_counter_ns()
            _reset_uvm_original_va_graph_certificates_for_plan_rebuild()

            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE = plan_path
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER = new_experts
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS = {}
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS = {}
            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES = 0
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_ENVELOPE_BYTES = 0
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "collecting"
            _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
            _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT = None
            # finalize() normally rejects a changed original-VA fingerprint.
            # Clearing the retired generation here is the explicit semantic
            # distinction of this baseline, not a relaxation of replay checks.
            _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT = ()

            from vllm.device_allocator import uvm
            from vllm.device_allocator.uvm import uvm_local_memory_pool

            for name, retained in sorted(retained_bindings.items()):
                old_tensor = retained.parameter.data
                old_va = int(old_tensor.data_ptr())
                source = retained.source
                assert source is not None
                with torch.cuda.device(device), uvm_local_memory_pool(
                    enable_prefetch=False
                ):
                    replacement_tensor = torch.empty_strided(
                        size=old_tensor.size(),
                        stride=old_tensor.stride(),
                        dtype=old_tensor.dtype,
                        layout=old_tensor.layout,
                        device=old_tensor.device,
                    )
                new_va = int(replacement_tensor.data_ptr())
                if new_va == old_va:
                    raise RuntimeError(
                        "Managed allocator reused a live Parameter VA during "
                        f"the non-stable baseline: {name!r}"
                    )
                requested_bytes = int(retained.parameter_bytes)
                advised = int(
                    uvm.batch_advise_ranges_preferred_location(
                        ((new_va, requested_bytes),), -1
                    )
                )
                prefetched = int(
                    uvm.batch_prefetch_ranges_to_device(
                        ((new_va, requested_bytes),), -1
                    )
                )
                torch.cuda.synchronize(device)
                if advised != 1 or prefetched != 1:
                    raise RuntimeError(
                        "Could not migrate a replacement Parameter to CPU "
                        f"before rebuilding aliases: {name!r}, "
                        f"advised={advised}, prefetched={prefetched}"
                    )
                # The non-stable baseline has intentionally destroyed its
                # retained GPU backing and may not have room for even one
                # complete replacement Parameter on GPU.  Both allocations
                # are CPU-accessible (pinned source and cudaMallocManaged
                # destination), so populate on CPU instead of introducing an
                # extra H2D residency spike before finalize recreates backing.
                import ctypes

                copy_started_ns = time.perf_counter_ns()
                ctypes.memmove(
                    ctypes.c_void_p(new_va),
                    ctypes.c_void_p(int(source.data_ptr())),
                    requested_bytes,
                )
                copy_ns = time.perf_counter_ns() - copy_started_ns
                retained.parameter.data = replacement_tensor
                replacement = _create_uvm_original_alias_binding(
                    full_parameter_name=name,
                    parameter=retained.parameter,
                    tensor=replacement_tensor,
                    device=replacement_tensor.device,
                    requested_bytes=requested_bytes,
                )
                retained.arena = replacement.arena
                retained.consumer = replacement.consumer
                retained.plan = replacement.plan
                retained.requested_bytes = replacement.requested_bytes
                retained.parameter_bytes = replacement.parameter_bytes
                retained.expert_ids = replacement.expert_ids
                retained.expert_stride_bytes = replacement.expert_stride_bytes
                retained.consumer_ranges = replacement.consumer_ranges
                replacement.arena.consumers[replacement.plan.consumer_index] = (
                    retained
                )
                retained.parameter._vllm_uvm_original_alias_binding = retained
                retained.source_refill_required = False
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS[name] = retained
                migration_records.append(
                    {
                        "full_parameter_name": name,
                        "bytes": requested_bytes,
                        "old_va": old_va,
                        "new_va": new_va,
                        "va_changed": True,
                        "cpu_advise_count": advised,
                        "cpu_prefetch_count": prefetched,
                        "cpu_memcpy_ns": copy_ns,
                    }
                )
                del old_tensor

            torch.cuda.empty_cache()
            reallocated_ns = time.perf_counter_ns()
            new_vas_by_name = {
                name: int(binding.parameter.data.data_ptr())
                for name, binding in sorted(retained_bindings.items())
            }
            if any(
                new_vas_by_name[name] == old_vas_by_name[name]
                for name in old_vas_by_name
            ):
                raise RuntimeError("At least one Parameter VA did not change")

            selected_bytes = sum(
                int(binding.requested_bytes)
                for binding in retained_bindings.values()
            )
            if (
                expected_selected_bytes is not None
                and selected_bytes != int(expected_selected_bytes)
            ):
                raise RuntimeError(
                    "Non-stable-VA plan selected an unexpected byte count: "
                    f"actual={selected_bytes}, expected={expected_selected_bytes}"
                )
            _CPU_OFFLOAD_BYTES = selected_bytes
            finalize_result = finalize_cpu_offload_graph_static_alias()
            if finalize_result.get("ok") is not True:
                raise RuntimeError(
                    "Non-stable-VA alias finalize failed: "
                    f"{finalize_result!r}"
                )
            new_vas = _uvm_original_parameter_va_snapshot()
            if new_vas != tuple(new_vas_by_name.values()):
                raise RuntimeError(
                    "Parameter VA fingerprint changed during new alias finalize"
                )
            _CPU_OFFLOAD_UVM_PLAN_GENERATION = attempted_generation
            finished_ns = time.perf_counter_ns()
            result = {
                "event": "cpu_offload_uvm_original_va_parameters_reallocated",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "ok": True,
                "reason": reason,
                "generation": attempted_generation,
                "pre_state": pre_state,
                "post_state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
                "old_plan_file": old_plan_path or None,
                "new_plan_file": plan_path,
                "selected_bytes": int(selected_bytes),
                "parameter_count": len(retained_bindings),
                "parameter_vas_before": old_vas,
                "parameter_vas_after": new_vas,
                "all_parameter_vas_changed": True,
                "reallocations": migration_records,
                "pressure": pressure_record,
                "sync_ns": synchronized_ns - started_ns,
                "alias_destroy_ns": destroyed_ns - synchronized_ns,
                "parameter_reallocate_ns": reallocated_ns - destroyed_ns,
                "finalize_ns": finished_ns - reallocated_ns,
                "wall_ns": finished_ns - started_ns,
                "destroy": destroy_result,
                "finalize": finalize_result,
            }
            _CPU_OFFLOAD_UVM_PLAN_REBUILD_HISTORY.append(result)
            _append_cpu_offload_manifest(result)
            return result
        except Exception as exc:
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "poisoned"
            _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
            failed = {
                "event": "cpu_offload_uvm_original_va_parameters_reallocated",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "ok": False,
                "reason": reason,
                "generation": attempted_generation,
                "pre_state": pre_state,
                "post_state": "poisoned",
                "old_plan_file": old_plan_path or None,
                "new_plan_file": plan_path,
                "pressure": pressure_record,
                "wall_ns": time.perf_counter_ns() - started_ns,
                "destroy": destroy_result,
                "finalize": finalize_result,
                "reallocations": migration_records,
                "error": repr(exc),
            }
            _CPU_OFFLOAD_UVM_PLAN_REBUILD_HISTORY.append(failed)
            _append_cpu_offload_manifest(failed)
            raise RuntimeError(
                "Non-stable-VA Parameter rebuild failed; the process is "
                "poisoned and must be torn down"
            ) from exc


def poison_cpu_offload_uvm_original_va_phase(reason: str) -> None:
    """Prevent any later selected-expert forward after a phase violation."""
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE

    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return
    _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "poisoned"
    _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
    record = {
        "event": "cpu_offload_uvm_original_va_phase_poisoned",
        "timestamp": time.time(),
        "pid": os.getpid(),
        "ok": False,
        "reason": str(reason),
        "post_state": "poisoned",
    }
    _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(record)
    _append_cpu_offload_manifest(record)


def _uvm_original_alias_fast_runtime_transition_enabled() -> bool:
    return bool(
        _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE == "token_threshold"
        and _strict_env_flag(
            "VLLM_UVM_TOKEN_THRESHOLD_FAST_TRANSITION", False
        )
    )


def _uvm_original_alias_runtime_stream_barrier(
    *,
    stage: str,
    device: int,
) -> dict[str, Any]:
    """Wait only for the transition-owning CUDA stream to reach a safe point.

    The PARK/RESUME ioctl is synchronous and completes its PTE publication and
    TLB invalidations before returning.  Recording an event on the model
    stream therefore gives the CPU an exact pre-mutation dependency; kernels
    submitted on the same stream after the ioctl naturally observe the new
    mapping.  Unlike ``torch.cuda.synchronize``, this does not drain unrelated
    streams, and unlike the diagnostic CUDA-read gates it does not walk or
    sample every hot Parameter range.
    """
    started_ns = time.perf_counter_ns()
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        event = torch.cuda.Event(enable_timing=False)
        event.record(stream)
        event.synchronize()
        stream_id = int(stream.cuda_stream)
    finished_ns = time.perf_counter_ns()
    result = {
        "event": "cpu_offload_uvm_runtime_stream_barrier",
        "timestamp": time.time(),
        "pid": os.getpid(),
        "ok": True,
        "stage": str(stage),
        "device": int(device),
        "stream_id": stream_id,
        "wait_ns": finished_ns - started_ns,
        "scope": "current_stream",
    }
    _append_cpu_offload_manifest(result)
    return result


def transition_cpu_offload_uvm_original_va_to_remote(
    *,
    reason: str,
    preserve_remote_graphs: bool,
    runtime_remote_ranges: Iterable[tuple[int, int]] = (),
) -> dict[str, Any]:
    """Replace sealed alias PTEs with audited CPU-backed remote READ PTEs."""
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT
    global _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_PARK_RESULT

    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return {
            "ok": True,
            "reason": "disabled",
            "pre_state": "disabled",
            "post_state": "disabled",
        }
    if _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE == "off":
        raise RuntimeError(
            "An original-VA phase transition was requested while "
            "VLLM_UVM_ORIGINAL_VA_PHASE_MODE=off"
        )
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Original-VA remote transition requires a reason")
    if not isinstance(preserve_remote_graphs, bool):
        raise TypeError("preserve_remote_graphs must be bool")

    normalized_runtime_remote_ranges: list[tuple[int, int]] = []
    for raw_base, raw_length in runtime_remote_ranges:
        base = int(raw_base)
        length = int(raw_length)
        if base <= 0 or length <= 0:
            raise ValueError(
                "Runtime remote expert ranges require positive base and length"
            )
        end = base + length
        if end <= base:
            raise ValueError("Runtime remote expert range overflowed")
        normalized_runtime_remote_ranges.append((base, length))
    normalized_runtime_remote_ranges.sort()
    merged_runtime_remote_ranges: list[tuple[int, int]] = []
    for base, length in normalized_runtime_remote_ranges:
        end = base + length
        if not merged_runtime_remote_ranges:
            merged_runtime_remote_ranges.append((base, length))
            continue
        previous_base, previous_length = merged_runtime_remote_ranges[-1]
        previous_end = previous_base + previous_length
        if base > previous_end:
            merged_runtime_remote_ranges.append((base, length))
            continue
        merged_runtime_remote_ranges[-1] = (
            previous_base,
            max(previous_end, end) - previous_base,
        )

    def runtime_remote_overlap_bytes(
        ranges: Iterable[tuple[int, int]],
    ) -> int:
        overlap_bytes = 0
        for raw_base, raw_length in ranges:
            base = int(raw_base)
            end = base + int(raw_length)
            for remote_base, remote_length in merged_runtime_remote_ranges:
                remote_end = remote_base + remote_length
                if remote_end <= base:
                    continue
                if remote_base >= end:
                    break
                overlap_bytes += max(
                    0,
                    min(end, remote_end) - max(base, remote_base),
                )
        return overlap_bytes

    with _CPU_OFFLOAD_UVM_STATIC_ALIAS_LOCK:
        pre_state = _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
        if pre_state in ("remote", "remote_parked") and (
            _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
        ):
            return {
                "ok": True,
                "reason": "already_remote",
                "pre_state": pre_state,
                "post_state": pre_state,
                "remote_map": _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT,
            }
        if pre_state not in ("sealed", "remote", "remote_parked"):
            raise RuntimeError(
                "Cannot establish original-VA remote topology from state "
                f"{pre_state!r}"
            )

        started_ns = time.perf_counter_ns()
        parameter_vas_before = _uvm_original_parameter_va_snapshot()
        if (
            _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
            and parameter_vas_before != _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT
        ):
            raise RuntimeError("Original Parameter VAs changed before remote cutover")
        _CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT = parameter_vas_before
        destroy_result: dict[str, Any] | None = None
        park_result: dict[str, Any] | None = None
        remote_map: dict[str, Any] | None = None
        physical: dict[str, Any] | None = None
        stream_barrier: dict[str, Any] | None = None
        fast_runtime_transition = (
            _uvm_original_alias_fast_runtime_transition_enabled()
        )
        synchronized_ns = started_ns
        destroyed_ns = started_ns
        mapped_ns = started_ns
        try:
            device = int(_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0)
            if fast_runtime_transition:
                stream_barrier = _uvm_original_alias_runtime_stream_barrier(
                    stage="before_park",
                    device=device,
                )
            else:
                torch.cuda.synchronize(device)
            synchronized_ns = time.perf_counter_ns()
            if pre_state == "sealed":
                if _CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED:
                    from vllm.device_allocator import uvm_static_alias

                    if not fast_runtime_transition:
                        _uvm_original_alias_hot_cuda_read_gate(
                            stage="runtime_remote_before_park",
                            device=device,
                        )
                    park_result = dict(
                        uvm_static_alias.driver_static_alias_park(
                            int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
                            device=device,
                            synchronize=False,
                        )
                    )
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_PARK_RESULT = park_result
                    if not bool(park_result.get("ok")):
                        raise RuntimeError(
                            "Failed to park original-VA consumer aliases before "
                            f"remote map: {park_result!r}"
                        )
                    if not fast_runtime_transition:
                        _uvm_original_alias_hot_cuda_read_gate(
                            stage="runtime_remote_after_park",
                            device=device,
                        )
                else:
                    destroy_result = destroy_cpu_offload_graph_static_alias(
                        graphs_destroyed=not preserve_remote_graphs,
                        synchronize=True,
                        remote_graphs_preserved=preserve_remote_graphs,
                        defer_cpu_restore=True,
                    )
                    if not bool(destroy_result.get("ok")):
                        raise RuntimeError(
                            "Failed to detach original-VA aliases before remote map: "
                            f"{destroy_result!r}"
                        )
            destroyed_ns = time.perf_counter_ns()

            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "remote_mapping"
            _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
            from vllm.device_allocator import uvm

            ranges = [
                consumer_range
                for _, binding in sorted(
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                )
                for consumer_range in (
                    _uvm_original_alias_binding_consumer_ranges(binding)
                )
            ]
            requested_bytes = sum(length for _, length in ranges)
            full_dual_view = bool(
                park_result is not None
                and park_result.get("dual_view_full_coverage") is True
                and int(park_result.get("dual_view_block_count", 0)) > 0
            )
            if _strict_env_flag(
                "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW", False
            ) and not full_dual_view:
                raise RuntimeError(
                    "The serving configuration requires complete dual-view "
                    f"coverage, but PARK returned: {park_result!r}"
                )
            if full_dual_view:
                # PARK has atomically published the prevalidated CPU child for
                # every consumer-touched 2 MiB parent. Reinstalling the same
                # CPU PTEs range by range would only repeat tens of thousands
                # of VA-block walks and TLB invalidations. Preserve the public
                # remote-map contract with a certificate derived from the
                # successful kernel transition.
                remote_map = {
                    "ok": True,
                    "reason": "prebuilt_full_dual_view",
                    "requested_count": len(ranges),
                    "success_count": len(ranges),
                    "failure_count": 0,
                    "requested_bytes": int(requested_bytes),
                    "cpu_resident_bytes": int(requested_bytes),
                    "remote_mapped_bytes": int(requested_bytes),
                    "covered_bytes": int(requested_bytes),
                    "unknown_bytes": 0,
                    "ioctl_calls": 0,
                    "dual_view_block_count": int(
                        park_result["dual_view_block_count"]
                    ),
                }
            else:
                remote_map = dict(
                    uvm.driver_prepare_ranges_remote_read_audited(
                        ranges,
                        device=device,
                    )
                )
            mapped_ns = time.perf_counter_ns()
            remote_contract_ok = (
                remote_map.get("ok") is True
                and int(remote_map.get("requested_count", -1)) == len(ranges)
                and int(remote_map.get("success_count", -1)) == len(ranges)
                and int(remote_map.get("failure_count", -1)) == 0
                and int(remote_map.get("requested_bytes", -1)) == requested_bytes
                and int(remote_map.get("cpu_resident_bytes", -1))
                >= requested_bytes
                and int(remote_map.get("remote_mapped_bytes", -1))
                >= requested_bytes
                and int(remote_map.get("covered_bytes", -1)) >= requested_bytes
                and int(remote_map.get("unknown_bytes", -1)) == 0
            )
            if not remote_contract_ok:
                raise RuntimeError(
                    "Audited original-VA remote READ mapping was incomplete: "
                    f"{remote_map!r}"
                )
            if not fast_runtime_transition:
                _uvm_original_alias_hot_cuda_read_gate(
                    stage="runtime_remote_after_remote_map",
                    device=device,
                )

            # The audited remote-map ioctl above already validates every
            # changed range and returns its CPU-resident, remotely-mapped and
            # covered byte counts.  Re-querying all cold and hot ranges here
            # is useful for qualification, but needlessly repeats a complete
            # VA-block walk on the serving critical path.  Production runs
            # with fail-fast managed faults and disabled migration heuristics
            # can opt out while retaining the mutation-time contract above.
            post_map_physical_audit_requested = _strict_env_flag(
                "VLLM_UVM_RUNTIME_POST_MAP_RESIDENCY_AUDIT", True
            )
            post_map_physical_audit_enabled = bool(
                post_map_physical_audit_requested
                and not fast_runtime_transition
            )
            if post_map_physical_audit_enabled:
                torch.cuda.synchronize(device)
                physical = dict(
                    uvm.driver_batch_query_ranges_residency_bytes(
                        ranges,
                        device=device,
                        wait_for_trackers=False,
                    )
                )
                physical_contract_ok = (
                    physical.get("ok") is True
                    and int(physical.get("queried_bytes", -1))
                    == requested_bytes
                    and int(physical.get("cpu_bytes", -1)) == requested_bytes
                    and int(physical.get("gpu_bytes", -1)) == 0
                    and int(physical.get("any_bytes", -1)) == requested_bytes
                    and int(physical.get("unknown_bytes", -1)) == 0
                )
                if not physical_contract_ok:
                    raise RuntimeError(
                        "Original-VA remote physical residency audit failed: "
                        f"{physical!r}"
                    )

            hot_physical_audit_requested = _strict_env_flag(
                "VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT", True
            )
            hot_physical_audit_enabled = (
                post_map_physical_audit_enabled
                and hot_physical_audit_requested
            )
            hot_ranges: list[tuple[int, int]] = []
            hot_requested_bytes = 0
            hot_runtime_remote_bytes = 0
            hot_expected_gpu_bytes = 0
            hot_expected_cpu_bytes = 0
            hot_physical: dict[str, Any] | None = None
            if hot_physical_audit_enabled:
                hot_ranges = [
                    hot_range
                    for _, binding in sorted(
                        _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                    )
                    for hot_range in _uvm_original_alias_binding_hot_ranges(
                        binding
                    )
                ]
                hot_requested_bytes = sum(
                    length for _, length in hot_ranges
                )
                hot_runtime_remote_bytes = runtime_remote_overlap_bytes(
                    hot_ranges
                )
                hot_expected_gpu_bytes = (
                    hot_requested_bytes - hot_runtime_remote_bytes
                )
                hot_expected_cpu_bytes = (
                    _uvm_original_alias_expected_shadow_cpu_bytes(
                        hot_ranges,
                        additional_cpu_ranges=merged_runtime_remote_ranges,
                    )
                )
                hot_physical = dict(
                    uvm.driver_batch_query_ranges_residency_bytes(
                        hot_ranges,
                        device=device,
                        wait_for_trackers=False,
                    )
                )
                if hot_ranges and not (
                    hot_physical.get("ok") is True
                    and int(hot_physical.get("queried_bytes", -1))
                    == hot_requested_bytes
                    and int(hot_physical.get("gpu_bytes", -1))
                    == hot_expected_gpu_bytes
                    and int(hot_physical.get("cpu_bytes", -1))
                    == hot_expected_cpu_bytes
                    and int(hot_physical.get("any_bytes", -1))
                    == hot_requested_bytes
                    and int(hot_physical.get("unknown_bytes", -1)) == 0
                ):
                    raise RuntimeError(
                        "Hot experts lost exclusive GPU residency during remote "
                        f"cutover: {hot_physical!r}"
                    )

            parameter_vas_after = _uvm_original_parameter_va_snapshot()
            if parameter_vas_after != parameter_vas_before:
                raise RuntimeError(
                    "Original Parameter VAs changed during remote cutover"
                )
            finished_ns = time.perf_counter_ns()
            _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT = remote_map
            _CPU_OFFLOAD_UVM_REMOTE_MAP_HISTORY.append(dict(remote_map))
            _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = True
            post_state = (
                "remote_parked"
                if _CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED
                else "remote"
            )
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = post_state
            result = {
                "event": "cpu_offload_uvm_original_va_remote_transition",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "ok": True,
                "reason": reason,
                "pre_state": pre_state,
                "post_state": post_state,
                "sync_ns": synchronized_ns - started_ns,
                "destroy_ns": destroyed_ns - synchronized_ns,
                "alias_detach_ns": destroyed_ns - synchronized_ns,
                "alias_detach_operation": (
                    "park" if park_result is not None else "destroy"
                ),
                "remote_map_ns": mapped_ns - destroyed_ns,
                "remote_map_operation": (
                    "prebuilt_full_dual_view"
                    if full_dual_view
                    else "audited_range_install"
                ),
                "physical_audit_ns": finished_ns - mapped_ns,
                "wall_ns": finished_ns - started_ns,
                "parameter_vas_before": parameter_vas_before,
                "parameter_vas_after": parameter_vas_after,
                "preserve_remote_graphs": bool(preserve_remote_graphs),
                "destroy": destroy_result,
                "park": park_result,
                "remote_map": remote_map,
                "physical_residency": physical,
                "fast_runtime_transition": fast_runtime_transition,
                "stream_barrier": stream_barrier,
                "post_map_physical_audit_requested": (
                    post_map_physical_audit_requested
                ),
                "post_map_physical_audit_enabled": (
                    post_map_physical_audit_enabled
                ),
                "hot_physical_residency": hot_physical,
                "hot_physical_audit_requested": hot_physical_audit_requested,
                "hot_physical_audit_enabled": hot_physical_audit_enabled,
                "hot_physical_audit_diagnostics_skipped": (
                    not hot_physical_audit_enabled
                ),
                "remote_range_count": len(ranges),
                "hot_range_count": len(hot_ranges),
                "hot_requested_bytes": int(hot_requested_bytes),
                "runtime_remote_range_count": len(
                    merged_runtime_remote_ranges
                ),
                "runtime_remote_bytes": sum(
                    length for _, length in merged_runtime_remote_ranges
                ),
                "hot_runtime_remote_bytes": int(hot_runtime_remote_bytes),
                "hot_expected_cpu_bytes": int(hot_expected_cpu_bytes),
                "hot_expected_gpu_bytes": int(hot_expected_gpu_bytes),
            }
            _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(result)
            _append_cpu_offload_manifest(result)
            return result
        except Exception as exc:
            finished_ns = time.perf_counter_ns()
            _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "poisoned"
            try:
                failed_parameter_vas_after = (
                    _uvm_original_parameter_va_snapshot()
                )
            except Exception:
                failed_parameter_vas_after = ()
            failed = {
                "event": "cpu_offload_uvm_original_va_remote_transition",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "ok": False,
                "reason": reason,
                "error": repr(exc),
                "pre_state": pre_state,
                "post_state": "poisoned",
                "sync_ns": synchronized_ns - started_ns,
                "destroy_ns": destroyed_ns - synchronized_ns,
                "alias_detach_ns": destroyed_ns - synchronized_ns,
                "alias_detach_operation": (
                    "park" if park_result is not None else "destroy"
                ),
                "remote_map_ns": mapped_ns - destroyed_ns,
                "wall_ns": finished_ns - started_ns,
                "parameter_vas_before": parameter_vas_before,
                "parameter_vas_after": failed_parameter_vas_after,
                "preserve_remote_graphs": bool(preserve_remote_graphs),
                "destroy": destroy_result,
                "park": park_result,
                "remote_map": remote_map,
                "physical_residency": physical,
            }
            _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(failed)
            _append_cpu_offload_manifest(failed)
            raise RuntimeError(
                "Original-VA remote topology transition failed; the process is "
                "poisoned and must be torn down"
            ) from exc


def initialize_cpu_offload_uvm_original_va_phase_topology() -> dict[str, Any]:
    """Put a remote-only process into its serving topology after model load."""
    mode = cpu_offload_uvm_original_va_phase_mode()
    if mode != "remote_only":
        return {"ok": True, "reason": "not_remote_only", "phase_mode": mode}
    return transition_cpu_offload_uvm_original_va_to_remote(
        reason="remote_only_model_load",
        preserve_remote_graphs=False,
    )


def prepare_cpu_offload_uvm_original_va_profile_topology(
    *,
    transition_reason: str = "startup_profile_all_remote",
    preserve_remote_graphs: bool = False,
) -> dict[str, Any]:
    """Temporarily make every expert CPU-remote for startup profiling.

    The explicit-KV vLLM path still executes a maximum-token profile forward
    before allocating the KV cache.  Its native activation/workspace
    allocations can evict an otherwise-hot managed expert.  With the custom
    driver's fail-fast migration mode, the following MoE read is then rejected
    instead of silently migrating the page.

    When explicitly enabled, install audited CPU-backed READ mappings for the
    hot complement as well as the frozen cold set.  The ordinary post-KV
    residency reconciliation restores hot experts to the GPU before serving
    warmup or Graph capture.  Parameter VAs never change.
    """
    global _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE

    if not isinstance(transition_reason, str) or not transition_reason.strip():
        raise ValueError("All-remote topology transition requires a reason")
    if not isinstance(preserve_remote_graphs, bool):
        raise TypeError("preserve_remote_graphs must be bool")
    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return {"ok": True, "reason": "disabled"}
    if not _strict_env_flag(
        "VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE", False
    ):
        return {"ok": True, "reason": "not_requested"}
    mode = cpu_offload_uvm_original_va_phase_mode()
    if mode == "off":
        return {"ok": True, "reason": "phase_mode_off"}
    if _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE:
        return {
            "ok": True,
            "reason": "already_active",
            "state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
        }

    started_ns = time.perf_counter_ns()
    pre_state = _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    parameter_vas_before = _uvm_original_parameter_va_snapshot()
    hot_prepare: tuple[Any, ...] | None = None
    hot_shadow_cutover: dict[str, Any] | None = None
    hot_preparation_residency: dict[str, Any] | None = None
    hot_remote_map: dict[str, Any] | None = None
    hot_physical: dict[str, Any] | None = None
    hot_dual_refresh: dict[str, Any] | None = None
    transition: dict[str, Any] | None = None
    hot_range_count = 0
    hot_bytes = 0
    force_synchronous_shadow_refresh = False
    hot_shadow_readmostly_refresh_count = 0
    try:
        if pre_state not in ("sealed", "remote", "remote_parked"):
            raise RuntimeError(
                "All-remote profile topology requires a sealed or remote "
                f"original-VA state, got {pre_state!r}"
            )

        from vllm.device_allocator import uvm

        device = int(_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0)
        hot_ranges = [
            hot_range
            for _, binding in sorted(
                _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
            )
            for hot_range in _uvm_original_alias_binding_hot_ranges(binding)
        ]
        hot_range_count = len(hot_ranges)
        hot_bytes = sum(length for _, length in hot_ranges)
        if not hot_ranges:
            return {"ok": True, "reason": "no_hot_ranges"}

        shadow_certificate = _CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT
        full_parameter_shadow = bool(
            isinstance(shadow_certificate, dict)
            and shadow_certificate.get("enabled") is True
            and shadow_certificate.get("full_parameter_shadow") is True
        )
        force_synchronous_shadow_refresh = _strict_env_flag(
            "VLLM_CPU_OFFLOAD_PLAN_CLASS_FORCE_SYNC_MIGRATION", False
        )
        if full_parameter_shadow:
            # PARK the current-tier pair while its Prefill child still
            # describes the live hot GPU pages.  Modifying those hot pages
            # first would make the prebuilt child stale, and both PARK and a
            # dual-view refresh must reject that inconsistent state.
            if pre_state == "sealed":
                transition = transition_cpu_offload_uvm_original_va_to_remote(
                    reason=transition_reason,
                    preserve_remote_graphs=preserve_remote_graphs,
                    # PARK publishes the full-Parameter CPU child before the
                    # explicit shadow cutover below.  Its hot leaves are
                    # therefore intentionally CPU-remote during this
                    # temporary all-remote profile, not missing GPU pages.
                    runtime_remote_ranges=hot_ranges,
                )
                if transition.get("ok") is not True:
                    raise RuntimeError(
                        "Failed to establish the cold remote topology before "
                        f"the startup shadow cutover: {transition!r}"
                    )
            elif not _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE:
                raise RuntimeError(
                    "All-remote shadow cutover found an unaudited remote state"
                )

            hot_preparation_residency = dict(
                uvm.driver_batch_query_ranges_residency_bytes(
                    hot_ranges,
                    device=device,
                    wait_for_trackers=False,
                )
            )
            hot_already_cpu_only = bool(
                hot_preparation_residency.get("ok") is True
                and int(
                    hot_preparation_residency.get("queried_bytes", -1)
                )
                == hot_bytes
                and int(hot_preparation_residency.get("cpu_bytes", -1))
                == hot_bytes
                and int(hot_preparation_residency.get("gpu_bytes", -1)) == 0
                and int(hot_preparation_residency.get("any_bytes", -1))
                == hot_bytes
                and int(hot_preparation_residency.get("unknown_bytes", -1))
                == 0
            )
            hot_cpu_gpu_duplicate = bool(
                hot_preparation_residency.get("ok") is True
                and int(
                    hot_preparation_residency.get("queried_bytes", -1)
                )
                == hot_bytes
                and int(hot_preparation_residency.get("cpu_bytes", -1))
                == hot_bytes
                and int(hot_preparation_residency.get("gpu_bytes", -1))
                == hot_bytes
                and int(hot_preparation_residency.get("any_bytes", -1))
                == hot_bytes
                and int(hot_preparation_residency.get("unknown_bytes", -1))
                == 0
            )
            if not (hot_already_cpu_only or hot_cpu_gpu_duplicate):
                raise RuntimeError(
                    "Full-Parameter shadow profile found partial hot expert "
                    f"residency: {hot_preparation_residency!r}"
                )

            # Every original Parameter page already has an immutable CPU read
            # duplicate.  Do not ask cudaMemAdvise to copy the hot complement
            # to its own CPU backing again: besides being redundant, a large
            # fragmented hot set can exceed the native migration path after
            # the cold aliases have split the VA policy tree.  The cutover
            # action validates the existing duplicate, drops the GPU copy and
            # installs the CPU READ mapping in one audited transaction.
            if hot_already_cpu_only:
                hot_shadow_cutover = {
                    "ok": True,
                    "reason": "already_cpu_shadow",
                    "range_count": len(hot_ranges),
                    "success_count": len(hot_ranges),
                    "failure_count": 0,
                    "requested_bytes": int(hot_bytes),
                    "released_gpu_bytes": 0,
                    "retained_cpu_shadow_bytes": int(hot_bytes),
                    "remote_mapped_bytes": int(hot_bytes),
                    "ioctl_calls": 0,
                }
            else:
                # A previously replayed Prefill Graph can revoke or stale the
                # historical ReadMostly duplicate while the residency bitmap
                # still reports both CPU and GPU copies.  Multi-class Graph
                # pre-capture therefore refreshes the CPU bytes from the
                # authoritative current GPU weights as part of the synchronous
                # cutover.  Fixed single-class runs keep the cheaper immutable
                # duplicate path.
                hot_shadow_cutover = dict(
                    uvm.driver_cutover_ranges_to_immutable_cpu_shadow(
                        hot_ranges,
                        device=device,
                        force_synchronous_migration=(
                            force_synchronous_shadow_refresh
                        ),
                    )
                )
            expected_released_gpu_bytes = (
                0 if hot_already_cpu_only else hot_bytes
            )
            if not (
                hot_shadow_cutover.get("ok") is True
                and int(hot_shadow_cutover.get("range_count", -1))
                == len(hot_ranges)
                and int(hot_shadow_cutover.get("success_count", -1))
                == len(hot_ranges)
                and int(hot_shadow_cutover.get("failure_count", -1)) == 0
                and int(hot_shadow_cutover.get("requested_bytes", -1))
                == hot_bytes
                and int(hot_shadow_cutover.get("released_gpu_bytes", -1))
                == expected_released_gpu_bytes
                and int(
                    hot_shadow_cutover.get("retained_cpu_shadow_bytes", -1)
                )
                == hot_bytes
                and int(hot_shadow_cutover.get("remote_mapped_bytes", -1))
                == hot_bytes
            ):
                raise RuntimeError(
                    "Failed to cut every hot expert range over to its "
                    f"immutable CPU shadow: {hot_shadow_cutover!r}"
                )
            if force_synchronous_shadow_refresh and not hot_already_cpu_only:
                # The synchronous refresh deliberately breaks the stale
                # read-duplication state before copying GPU -> CPU. Re-arm the
                # immutable ReadMostly policy while the refreshed CPU copy is
                # the sole resident copy, so the later hot-GPU restore keeps
                # this new CPU shadow for the next class transition.
                hot_shadow_readmostly_refresh_count = int(
                    uvm.batch_advise_ranges_read_mostly(
                        hot_ranges, device=device
                    )
                )
                if hot_shadow_readmostly_refresh_count != len(hot_ranges):
                    raise RuntimeError(
                        "Failed to restore ReadMostly policy after synchronous "
                        "hot shadow refresh: "
                        f"actual={hot_shadow_readmostly_refresh_count}, "
                        f"expected={len(hot_ranges)}"
                    )
            hot_prepare = (
                len(hot_ranges),
                "immutable_cpu_shadow_cutover",
                len(hot_ranges),
                len(hot_ranges),
                len(hot_ranges),
            )
            hot_remote_map = {
                "ok": True,
                "reason": "immutable_cpu_shadow_cutover",
                "requested_count": len(hot_ranges),
                "success_count": len(hot_ranges),
                "failure_count": 0,
                "requested_bytes": int(hot_bytes),
                "cpu_resident_bytes": int(hot_bytes),
                "remote_mapped_bytes": int(hot_bytes),
                "covered_bytes": int(hot_bytes),
                "unknown_bytes": 0,
                "ioctl_calls": int(
                    hot_shadow_cutover.get("ioctl_calls", 1)
                ),
            }

            # PARK published the retained CPU child before the hot complement
            # was cut over to its immutable CPU shadow. The cold slot leaves
            # remain valid, but the ordinary leaves in that detached child
            # still name the old GPU physical pages. Re-materialize both
            # complete 4K children from the new live residency before any
            # remote Graph is captured; otherwise later allocator activity can
            # reuse those released pages and make the retained child read
            # unrelated data.
            if _strict_env_flag(
                "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW", False
            ):
                hot_dual_refresh = (
                    refresh_cpu_offload_uvm_original_va_full_dual_view(
                        nested_refresh=False
                    )
                )
                if hot_dual_refresh.get("ok") is not True:
                    raise RuntimeError(
                        "Failed to refresh the parked dual view after hot "
                        f"CPU cutover: {hot_dual_refresh!r}"
                    )
        else:
            # The audited driver primitive verifies and installs remote READ
            # PTEs; it intentionally does not migrate GPU-resident pages to
            # CPU. Move the hot complement first using native UVM APIs, then
            # require the driver audit to prove complete CPU residency and
            # GPU READ coverage.
            hot_prepare = tuple(
                uvm.batch_prepare_ranges_for_native_remote_access_with_executor(
                    hot_ranges,
                    sync=True,
                    device=device,
                )
            )
            if not hot_prepare or int(hot_prepare[0]) != len(hot_ranges):
                raise RuntimeError(
                    "Failed to place every hot expert range on CPU for startup "
                    f"profiling: result={hot_prepare!r}, "
                    f"expected={len(hot_ranges)}"
                )

            torch.cuda.synchronize(device)
            hot_remote_map = dict(
                uvm.driver_prepare_ranges_remote_read_audited(
                    hot_ranges,
                    device=device,
                )
            )
        hot_remote_contract_ok = (
            hot_remote_map.get("ok") is True
            and int(hot_remote_map.get("requested_count", -1))
            == len(hot_ranges)
            and int(hot_remote_map.get("success_count", -1))
            == len(hot_ranges)
            and int(hot_remote_map.get("failure_count", -1)) == 0
            and int(hot_remote_map.get("requested_bytes", -1)) == hot_bytes
            and int(hot_remote_map.get("cpu_resident_bytes", -1))
            >= hot_bytes
            and int(hot_remote_map.get("remote_mapped_bytes", -1))
            >= hot_bytes
            and int(hot_remote_map.get("covered_bytes", -1)) >= hot_bytes
            and int(hot_remote_map.get("unknown_bytes", -1)) == 0
        )
        if not hot_remote_contract_ok:
            raise RuntimeError(
                "Hot expert all-remote profile mapping was incomplete: "
                f"{hot_remote_map!r}"
            )

        # A sealed alias owns the frozen cold ranges.  Detach or park those
        # aliases only after the hot complement has a complete remote map, and
        # tell the transition audit that those hot bytes are intentionally CPU
        # resident for this temporary profile topology.
        if pre_state == "sealed":
            if transition is None:
                transition = transition_cpu_offload_uvm_original_va_to_remote(
                    reason=transition_reason,
                    preserve_remote_graphs=preserve_remote_graphs,
                    runtime_remote_ranges=hot_ranges,
                )
                if transition.get("ok") is not True:
                    raise RuntimeError(
                        "Failed to establish cold remote topology for startup "
                        f"profiling: {transition!r}"
                    )
        elif not _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE:
            raise RuntimeError(
                "All-remote profile topology found an unaudited remote state"
            )

        torch.cuda.synchronize(device)
        hot_physical = dict(
            uvm.driver_batch_query_ranges_residency_bytes(
                hot_ranges,
                device=device,
                wait_for_trackers=False,
            )
        )
        if not (
            hot_physical.get("ok") is True
            and int(hot_physical.get("queried_bytes", -1)) == hot_bytes
            and int(hot_physical.get("cpu_bytes", -1)) == hot_bytes
            and int(hot_physical.get("gpu_bytes", -1)) == 0
            and int(hot_physical.get("any_bytes", -1)) == hot_bytes
            and int(hot_physical.get("unknown_bytes", -1)) == 0
        ):
            raise RuntimeError(
                "Hot experts were not exclusively CPU resident in the "
                f"all-remote profile topology: {hot_physical!r}"
            )

        parameter_vas_after = _uvm_original_parameter_va_snapshot()
        if parameter_vas_after != parameter_vas_before:
            raise RuntimeError(
                "Original Parameter VAs changed while preparing startup "
                "all-remote profiling"
            )
        _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE = True
        result = {
            "event": "cpu_offload_uvm_original_va_profile_all_remote_prepared",
            "timestamp": time.time(),
            "pid": os.getpid(),
            "ok": True,
            "reason": "prepared",
            "transition_reason": transition_reason,
            "preserve_remote_graphs": preserve_remote_graphs,
            "pre_state": pre_state,
            "post_state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
            "wall_ns": time.perf_counter_ns() - started_ns,
            "parameter_vas_before": parameter_vas_before,
            "parameter_vas_after": parameter_vas_after,
            "hot_range_count": len(hot_ranges),
            "hot_bytes": int(hot_bytes),
            "hot_prepare": hot_prepare,
            "hot_prepare_operation": (
                (
                    "sync_cpu_remote_cutover"
                    if force_synchronous_shadow_refresh
                    and hot_shadow_cutover is not None
                    and hot_shadow_cutover.get("reason") != "already_cpu_shadow"
                    else "immutable_cpu_shadow_cutover"
                )
                if hot_shadow_cutover is not None
                else "native_uvm_migration"
            ),
            "hot_shadow_cutover": hot_shadow_cutover,
            "hot_shadow_readmostly_refresh_count": int(
                hot_shadow_readmostly_refresh_count
            ),
            "hot_preparation_residency": hot_preparation_residency,
            "hot_remote_map": hot_remote_map,
            "hot_dual_refresh": hot_dual_refresh,
            "hot_physical_residency": hot_physical,
            "cold_transition": transition,
        }
        _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(result)
        _append_cpu_offload_manifest(result)
        return result
    except Exception as exc:
        _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE = False
        _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "poisoned"
        failed = {
            "event": "cpu_offload_uvm_original_va_profile_all_remote_prepared",
            "timestamp": time.time(),
            "pid": os.getpid(),
            "ok": False,
            "reason": "failed",
            "transition_reason": transition_reason,
            "preserve_remote_graphs": preserve_remote_graphs,
            "error": repr(exc),
            "pre_state": pre_state,
            "post_state": "poisoned",
            "wall_ns": time.perf_counter_ns() - started_ns,
            "parameter_vas_before": parameter_vas_before,
            "hot_range_count": int(hot_range_count),
            "hot_bytes": int(hot_bytes),
            "hot_prepare": hot_prepare,
            "hot_prepare_operation": (
                (
                    "sync_cpu_remote_cutover"
                    if force_synchronous_shadow_refresh
                    and hot_shadow_cutover is not None
                    and hot_shadow_cutover.get("reason") != "already_cpu_shadow"
                    else "immutable_cpu_shadow_cutover"
                )
                if hot_shadow_cutover is not None
                else "native_uvm_migration"
            ),
            "hot_shadow_cutover": hot_shadow_cutover,
            "hot_shadow_readmostly_refresh_count": int(
                hot_shadow_readmostly_refresh_count
            ),
            "hot_preparation_residency": hot_preparation_residency,
            "hot_remote_map": hot_remote_map,
            "hot_dual_refresh": hot_dual_refresh,
            "hot_physical_residency": hot_physical,
            "cold_transition": transition,
        }
        _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(failed)
        _append_cpu_offload_manifest(failed)
        raise RuntimeError(
            "Original-VA all-remote startup profile preparation failed; the "
            "process is poisoned and must be torn down"
        ) from exc


def reconcile_cpu_offload_uvm_original_va_sealed_hot_residency(
    *,
    preserve_allocator_cache: bool = True,
    runtime_remote_ranges: Iterable[tuple[int, int]] = (),
    mapping_refresh_ranges: Iterable[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Restore the hot complement after KV allocation in sealed-alias mode.

    A fixed alias/copy phase keeps the cold expert aliases sealed rather than
    transitioning them to the audited remote topology used by Graph Decode.
    Consequently the remote-graph reconciliation below is not applicable,
    even though a later KV/native allocation can still evict managed hot
    pages.  This helper repairs only the disjoint hot ranges, leaves every
    cold alias/PTE untouched, preserves intentional allocator reserve by
    default, and fails before warmup unless residency and free headroom are
    exact.
    """
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE

    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return {"ok": True, "reason": "disabled"}

    raw_min_free = os.environ.get(
        "VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES",
        str(256 * 1024 * 1024),
    ).strip()
    try:
        min_free_bytes = int(raw_min_free)
    except ValueError as exc:
        raise ValueError(
            "VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES must be an integer"
        ) from exc
    if min_free_bytes < 0:
        raise ValueError(
            "VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES must be non-negative"
        )

    with _CPU_OFFLOAD_UVM_STATIC_ALIAS_LOCK:
        pre_state = _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
        if pre_state != "sealed" or _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE:
            raise RuntimeError(
                "Sealed hot-residency reconciliation requires the active "
                f"alias topology; state={pre_state!r}, "
                f"remote_map_active={_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE!r}"
            )

        started_ns = time.perf_counter_ns()
        parameter_vas_before = _uvm_original_parameter_va_snapshot()
        hot_physical_before: dict[str, Any] | None = None
        hot_physical_after: dict[str, Any] | None = None
        hot_advise_count = 0
        hot_advise_skipped_for_full_dual_view = False
        hot_prefetch_count = 0
        hot_prefetch_executor = "none"
        hot_prefetch_driver_result = 0
        hot_prefetch_skipped_already_resident = False
        hot_mapping_refresh_required = False
        hot_mapping_cutover: dict[str, Any] | None = None
        hot_range_count = 0
        hot_bytes = 0
        hot_gpu_range_count = 0
        hot_expected_gpu_bytes = 0
        hot_runtime_remote_bytes = 0
        normalized_runtime_remote_ranges = tuple(
            (int(base), int(length))
            for base, length in runtime_remote_ranges
        )
        normalized_mapping_refresh_ranges = (
            None
            if mapping_refresh_ranges is None
            else tuple(
                (int(base), int(length))
                for base, length in mapping_refresh_ranges
                if int(base) > 0 and int(length) > 0
            )
        )
        mapping_refresh_target_count = 0
        mapping_refresh_target_bytes = 0
        mapping_refresh_prepare_prefetch_count = 0
        mapping_refresh_prepare_prefetch_executor = "none"
        mapping_refresh_prepare_prefetch_driver_result = 0
        free_before = -1
        free_after = -1
        total_bytes = -1
        allocator_reserved_before = -1
        allocator_reserved_after = -1
        try:
            from vllm.device_allocator import uvm

            device = int(_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0)
            torch.cuda.synchronize(device)
            free_before, total_bytes = (
                int(value) for value in torch.cuda.mem_get_info(device)
            )
            allocator_reserved_before = int(torch.cuda.memory_reserved(device))
            if not preserve_allocator_cache:
                torch.cuda.empty_cache()

            hot_ranges = [
                hot_range
                for _, binding in sorted(
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                )
                for hot_range in _uvm_original_alias_binding_hot_ranges(binding)
            ]
            hot_range_count = len(hot_ranges)
            hot_bytes = sum(length for _, length in hot_ranges)
            hot_gpu_ranges = _uvm_original_alias_subtract_ranges(
                hot_ranges,
                normalized_runtime_remote_ranges,
            )
            hot_gpu_range_count = len(hot_gpu_ranges)
            hot_expected_gpu_bytes = sum(
                length for _, length in hot_gpu_ranges
            )
            hot_runtime_remote_bytes = hot_bytes - hot_expected_gpu_bytes
            # Runtime expert release can leave a CPU READ PTE active for an
            # adjacent hot fragment while ReadMostly still retains an exact
            # GPU duplicate.  Residency alone cannot certify which leaf a
            # subsequent full-dual refresh will freeze.  Re-prefetch the GPU
            # complement once after any runtime-remote mutation so the live
            # mapping, not only the physical duplicate, is canonical.
            if normalized_mapping_refresh_ranges is None:
                mapping_refresh_gpu_ranges = (
                    hot_gpu_ranges
                    if normalized_runtime_remote_ranges
                    else []
                )
            else:
                mapping_refresh_gpu_ranges = (
                    _uvm_original_alias_intersect_ranges(
                        normalized_mapping_refresh_ranges,
                        hot_gpu_ranges,
                    )
                )
            mapping_refresh_target_count = len(mapping_refresh_gpu_ranges)
            mapping_refresh_target_bytes = sum(
                length for _, length in mapping_refresh_gpu_ranges
            )
            hot_mapping_refresh_required = bool(mapping_refresh_gpu_ranges)
            expected_hot_cpu_bytes = (
                _uvm_original_alias_expected_shadow_cpu_bytes(
                    hot_ranges,
                    additional_cpu_ranges=normalized_runtime_remote_ranges,
                )
            )
            hot_physical_before = dict(
                uvm.driver_batch_query_ranges_residency_bytes(
                    hot_ranges,
                    device=device,
                    wait_for_trackers=False,
                )
            )

            hot_already_resident = bool(
                hot_ranges
                and hot_physical_before.get("ok") is True
                and int(hot_physical_before.get("queried_bytes", -1))
                == hot_bytes
                and int(hot_physical_before.get("gpu_bytes", -1))
                == hot_expected_gpu_bytes
                and int(hot_physical_before.get("cpu_bytes", -1))
                == expected_hot_cpu_bytes
                and int(hot_physical_before.get("any_bytes", -1))
                == hot_bytes
                and int(hot_physical_before.get("unknown_bytes", -1)) == 0
            )

            shadow_certificate = _CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT
            full_parameter_shadow = bool(
                isinstance(shadow_certificate, dict)
                and shadow_certificate.get("enabled") is True
                and shadow_certificate.get("full_parameter_shadow") is True
            )
            if (
                hot_mapping_refresh_required
                and (
                    hot_already_resident
                    or normalized_mapping_refresh_ranges is not None
                )
                and full_parameter_shadow
                and mapping_refresh_gpu_ranges
            ):
                if normalized_mapping_refresh_ranges is not None:
                    # Some promoted envelope leaves can still be physically
                    # absent even though the expert-level restore returned
                    # success.  The immutable-shadow cutover requires an
                    # extant GPU copy for every byte, so materialize the small
                    # explicit promotion set first.  The following
                    # cutover+prefetch pair then makes the live GPU PTE, not a
                    # retained ReadMostly duplicate, canonical.
                    (
                        mapping_refresh_prepare_prefetch_count,
                        mapping_refresh_prepare_prefetch_executor,
                        mapping_refresh_prepare_prefetch_driver_result,
                    ) = uvm.batch_prefetch_ranges_to_device_with_executor(
                        mapping_refresh_gpu_ranges,
                        device,
                        executor="cpp_batch",
                    )
                    if int(mapping_refresh_prepare_prefetch_count) != len(
                        mapping_refresh_gpu_ranges
                    ):
                        raise RuntimeError(
                            "Failed to materialize every promoted hot range "
                            "before immutable-shadow cutover: "
                            f"prefetched={mapping_refresh_prepare_prefetch_count}, "
                            f"expected={len(mapping_refresh_gpu_ranges)}, "
                            "executor="
                            f"{mapping_refresh_prepare_prefetch_executor!r}, "
                            "driver_result="
                            f"{mapping_refresh_prepare_prefetch_driver_result}"
                        )
                    torch.cuda.synchronize(device)
                # A retained ReadMostly GPU duplicate does not prove the live
                # PTE maps that duplicate. Drop it against the immutable CPU
                # shadow first; the mandatory prefetch below then rebuilds an
                # unambiguous GPU mapping for every non-remote hot fragment.
                hot_mapping_cutover = dict(
                    uvm.driver_cutover_ranges_to_immutable_cpu_shadow(
                        mapping_refresh_gpu_ranges,
                        device=device,
                        force_synchronous_migration=False,
                    )
                )
                if not (
                    hot_mapping_cutover.get("ok") is True
                    and int(hot_mapping_cutover.get("range_count", -1))
                    == len(mapping_refresh_gpu_ranges)
                    and int(hot_mapping_cutover.get("success_count", -1))
                    == len(mapping_refresh_gpu_ranges)
                    and int(hot_mapping_cutover.get("failure_count", -1)) == 0
                    and int(hot_mapping_cutover.get("requested_bytes", -1))
                    == mapping_refresh_target_bytes
                    and (
                        0
                        <= int(
                            hot_mapping_cutover.get("released_gpu_bytes", -1)
                        )
                        <= mapping_refresh_target_bytes
                    )
                ):
                    raise RuntimeError(
                        "Failed to canonicalize runtime-mutated hot mappings "
                        f"against immutable CPU shadows: {hot_mapping_cutover!r}"
                    )

            if hot_already_resident and not hot_mapping_refresh_required:
                # cudaMemPrefetchAsync is not a harmless no-op for a large
                # managed range which is already resident on the target GPU:
                # it still walks every VA block and can tear down/rebuild GPU
                # mappings.  Besides wasting several seconds at startup, that
                # unnecessary rewrite used to perturb the legacy E2 alias
                # topology before Graph capture.  The exact residency query
                # above is the certificate that no repair is needed.
                hot_prefetch_skipped_already_resident = True
            elif hot_gpu_ranges:
                prefetch_ranges = (
                    mapping_refresh_gpu_ranges
                    if normalized_mapping_refresh_ranges is not None
                    else hot_gpu_ranges
                )
                if _strict_env_flag(
                    "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW", False
                ):
                    # Immutable full-Parameter shadows split the managed VA
                    # policy tree around thousands of cold slices. A later
                    # cudaMemAdvise over the complementary hot fragments is
                    # neither required (migration heuristics are disabled) nor
                    # reliable across those artificial boundaries. The
                    # explicit driver prefetch and exact residency audit below
                    # are the serving contract.
                    hot_advise_skipped_for_full_dual_view = True
                else:
                    hot_advise_count = int(
                        uvm.batch_advise_ranges_preferred_location(
                            prefetch_ranges,
                            device,
                        )
                    )
                    if hot_advise_count != len(prefetch_ranges):
                        raise RuntimeError(
                            "Failed to set GPU preferred location for every "
                            "sealed hot expert range: "
                            f"advised={hot_advise_count}, "
                            f"expected={len(prefetch_ranges)}"
                        )
                (
                    hot_prefetch_count,
                    hot_prefetch_executor,
                    hot_prefetch_driver_result,
                ) = uvm.batch_prefetch_ranges_to_device_with_executor(
                    prefetch_ranges,
                    device,
                    executor="cpp_batch",
                )
                if int(hot_prefetch_count) != len(prefetch_ranges):
                    raise RuntimeError(
                        "Failed to restore every sealed hot expert range to "
                        "GPU: "
                        f"prefetched={hot_prefetch_count}, "
                        f"expected={len(prefetch_ranges)}, "
                        f"executor={hot_prefetch_executor!r}, "
                        f"driver_result={hot_prefetch_driver_result}"
                    )

            torch.cuda.synchronize(device)
            hot_physical_after = dict(
                uvm.driver_batch_query_ranges_residency_bytes(
                    hot_ranges,
                    device=device,
                    wait_for_trackers=False,
                )
            )
            if hot_ranges and not (
                hot_physical_after.get("ok") is True
                and int(hot_physical_after.get("queried_bytes", -1)) == hot_bytes
                and int(hot_physical_after.get("gpu_bytes", -1))
                == hot_expected_gpu_bytes
                and int(hot_physical_after.get("cpu_bytes", -1))
                == expected_hot_cpu_bytes
                and int(hot_physical_after.get("any_bytes", -1)) == hot_bytes
                and int(hot_physical_after.get("unknown_bytes", -1)) == 0
            ):
                raise RuntimeError(
                    "Hot experts were not exclusively GPU resident after "
                    f"sealed reconciliation: {hot_physical_after!r}"
                )

            free_after = int(torch.cuda.mem_get_info(device)[0])
            allocator_reserved_after = int(torch.cuda.memory_reserved(device))
            if (
                preserve_allocator_cache
                and allocator_reserved_after < allocator_reserved_before
            ):
                raise RuntimeError(
                    "Allocator reserve shrank during sealed hot-expert "
                    "reconciliation: "
                    f"before={allocator_reserved_before}, "
                    f"after={allocator_reserved_after}"
                )
            if free_after < min_free_bytes:
                raise RuntimeError(
                    "Insufficient GPU headroom after sealed hot-expert "
                    "reconciliation: "
                    f"free_bytes={free_after}, required_bytes={min_free_bytes}"
                )

            parameter_vas_after = _uvm_original_parameter_va_snapshot()
            if parameter_vas_after != parameter_vas_before:
                raise RuntimeError(
                    "Original Parameter VAs changed during sealed hot-expert "
                    "reconciliation"
                )

            result = {
                "event": (
                    "cpu_offload_uvm_original_va_sealed_hot_residency_reconciled"
                ),
                "timestamp": time.time(),
                "pid": os.getpid(),
                "ok": True,
                "pre_state": pre_state,
                "post_state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
                "wall_ns": time.perf_counter_ns() - started_ns,
                "parameter_vas_before": parameter_vas_before,
                "parameter_vas_after": parameter_vas_after,
                "hot_range_count": int(hot_range_count),
                "hot_bytes": int(hot_bytes),
                "hot_gpu_range_count": int(hot_gpu_range_count),
                "hot_expected_gpu_bytes": int(hot_expected_gpu_bytes),
                "runtime_remote_range_count": len(
                    normalized_runtime_remote_ranges
                ),
                "hot_runtime_remote_bytes": int(hot_runtime_remote_bytes),
                "expected_hot_cpu_shadow_bytes": int(
                    expected_hot_cpu_bytes
                ),
                "hot_advise_count": int(hot_advise_count),
                "hot_advise_skipped_for_full_dual_view": bool(
                    hot_advise_skipped_for_full_dual_view
                ),
                "hot_prefetch_count": int(hot_prefetch_count),
                "hot_prefetch_executor": str(hot_prefetch_executor),
                "hot_prefetch_driver_result": int(
                    hot_prefetch_driver_result
                ),
                "hot_prefetch_skipped_already_resident": bool(
                    hot_prefetch_skipped_already_resident
                ),
                "hot_mapping_refresh_required": bool(
                    hot_mapping_refresh_required
                ),
                "mapping_refresh_scope": (
                    "hot_gpu_complement"
                    if normalized_mapping_refresh_ranges is None
                    else "explicit_runtime_promotions"
                ),
                "mapping_refresh_target_count": int(
                    mapping_refresh_target_count
                ),
                "mapping_refresh_target_bytes": int(
                    mapping_refresh_target_bytes
                ),
                "mapping_refresh_prepare_prefetch_count": int(
                    mapping_refresh_prepare_prefetch_count
                ),
                "mapping_refresh_prepare_prefetch_executor": str(
                    mapping_refresh_prepare_prefetch_executor
                ),
                "mapping_refresh_prepare_prefetch_driver_result": int(
                    mapping_refresh_prepare_prefetch_driver_result
                ),
                "hot_mapping_cutover": hot_mapping_cutover,
                "hot_physical_residency_before": hot_physical_before,
                "hot_physical_residency_after": hot_physical_after,
                "free_bytes_before": int(free_before),
                "free_bytes_after": int(free_after),
                "total_bytes": int(total_bytes),
                "minimum_free_bytes": int(min_free_bytes),
                "allocator_cache_preserved": bool(preserve_allocator_cache),
                "allocator_reserved_before": int(allocator_reserved_before),
                "allocator_reserved_after": int(allocator_reserved_after),
            }
            _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(result)
            _append_cpu_offload_manifest(result)
            return result
        except Exception as exc:
            _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "poisoned"
            failed = {
                "event": (
                    "cpu_offload_uvm_original_va_sealed_hot_residency_reconciled"
                ),
                "timestamp": time.time(),
                "pid": os.getpid(),
                "ok": False,
                "error": repr(exc),
                "pre_state": pre_state,
                "post_state": "poisoned",
                "wall_ns": time.perf_counter_ns() - started_ns,
                "parameter_vas_before": parameter_vas_before,
                "hot_range_count": int(hot_range_count),
                "hot_bytes": int(hot_bytes),
                "hot_gpu_range_count": int(hot_gpu_range_count),
                "hot_expected_gpu_bytes": int(hot_expected_gpu_bytes),
                "runtime_remote_range_count": len(
                    normalized_runtime_remote_ranges
                ),
                "hot_runtime_remote_bytes": int(hot_runtime_remote_bytes),
                "hot_advise_count": int(hot_advise_count),
                "hot_advise_skipped_for_full_dual_view": bool(
                    hot_advise_skipped_for_full_dual_view
                ),
                "hot_prefetch_count": int(hot_prefetch_count),
                "hot_prefetch_executor": str(hot_prefetch_executor),
                "hot_prefetch_driver_result": int(
                    hot_prefetch_driver_result
                ),
                "hot_mapping_refresh_required": bool(
                    hot_mapping_refresh_required
                ),
                "hot_mapping_cutover": hot_mapping_cutover,
                "hot_physical_residency_before": hot_physical_before,
                "hot_physical_residency_after": hot_physical_after,
                "free_bytes_before": int(free_before),
                "free_bytes_after": int(free_after),
                "total_bytes": int(total_bytes),
                "minimum_free_bytes": int(min_free_bytes),
                "allocator_cache_preserved": bool(preserve_allocator_cache),
                "allocator_reserved_before": int(allocator_reserved_before),
                "allocator_reserved_after": int(allocator_reserved_after),
            }
            _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(failed)
            _append_cpu_offload_manifest(failed)
            raise RuntimeError(
                "Original-VA sealed hot-expert reconciliation failed; the "
                "process is poisoned and must be torn down"
            ) from exc


def refresh_cpu_offload_uvm_original_va_full_dual_view(
    *,
    nested_refresh: bool | None = None,
) -> dict[str, Any]:
    """Refresh both cold-tier children after ordinary UVM residency moves.

    The cold slot leaves and Parameter VAs are immutable. KV allocation,
    CUDA-Graph capture, or allocator warmup can nevertheless relocate a hot
    page shared by the same 2 MiB parent. Rebuild the ordinary leaves in both
    detached children at a GPU-quiescent boundary so later PARK/RESUME only
    swaps parent entries and never publishes a stale hot physical address.
    """
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE

    if not _strict_env_flag(
        "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW", False
    ):
        return {"ok": True, "reason": "full_dual_view_not_required"}

    with _CPU_OFFLOAD_UVM_STATIC_ALIAS_LOCK:
        from vllm.device_allocator import uvm_static_alias

        state_before = _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
        session_id = int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID or 0)
        if state_before not in ("sealed", "remote_parked") or session_id <= 0:
            raise RuntimeError(
                "Full dual-view refresh requires a sealed or parked session: "
                f"state={state_before!r}, session_id={session_id}"
            )

        device = int(_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0)
        parameter_vas_before = _uvm_original_parameter_va_snapshot()
        torch.cuda.synchronize(device)
        started_ns = time.perf_counter_ns()
        use_nested_refresh = (
            _strict_env_flag(
                "VLLM_CPU_OFFLOAD_PLAN_CLASS_INCREMENTAL_DUAL_REFRESH",
                False,
            )
            if nested_refresh is None
            else bool(nested_refresh)
        )
        refreshed = dict(
            uvm_static_alias.driver_static_alias_seal(
                session_id,
                device=device,
                refresh_dual=True,
                nested_refresh=use_nested_refresh,
            )
        )
        torch.cuda.synchronize(device)
        parameter_vas_after = _uvm_original_parameter_va_snapshot()
        result = {
            "event": "cpu_offload_uvm_full_dual_view_refreshed",
            "timestamp": time.time(),
            "pid": os.getpid(),
            "ok": refreshed.get("ok") is True,
            "state_before": state_before,
            "state_after": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
            "session_id": session_id,
            "nested_refresh": use_nested_refresh,
            "wall_ns": time.perf_counter_ns() - started_ns,
            "parameter_vas_before": parameter_vas_before,
            "parameter_vas_after": parameter_vas_after,
            "refresh": refreshed,
        }
        if parameter_vas_after != parameter_vas_before:
            result["ok"] = False
            result["error"] = "original Parameter VAs changed during refresh"
        _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(result)
        _append_cpu_offload_manifest(result)
        if result["ok"] is not True:
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "poisoned"
            raise RuntimeError(
                "Full dual-view post-residency refresh failed; the process is "
                f"poisoned and must be torn down: {result!r}"
            )
        return result


def reconcile_cpu_offload_uvm_original_va_remote_graph_residency(
    *,
    preserve_allocator_cache: bool = False,
) -> dict[str, Any]:
    """Repair and audit expert residency immediately before Graph capture.

    ``remote_only`` establishes its PTE topology directly after model load.
    Memory profiling and KV-cache allocation happen later and may evict an
    otherwise-hot managed expert.  A CUDA Graph cannot service that first
    access as a normal UVM fault, so re-establish the complete cold/hot
    placement after those allocations and fail before capture unless it is
    exact and leaves a configured amount of free device memory.

    ``preserve_allocator_cache`` is used only after the explicit all-remote
    Prefill reserve warmup.  In that path the inactive native CUDA segments are
    intentional serving capacity: releasing them before hot-expert restoration
    lets hot managed pages occupy the same physical ranges, so the first real
    Prefill has to evict those pages to recreate its native segments.
    """
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    global _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE

    if not cpu_offload_uvm_original_va_static_alias_enabled():
        return {"ok": True, "reason": "disabled"}

    raw_min_free = os.environ.get(
        "VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES",
        str(256 * 1024 * 1024),
    ).strip()
    try:
        min_free_bytes = int(raw_min_free)
    except ValueError as exc:
        raise ValueError(
            "VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES must be an integer"
        ) from exc
    if min_free_bytes < 0:
        raise ValueError(
            "VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES must be non-negative"
        )

    with _CPU_OFFLOAD_UVM_STATIC_ALIAS_LOCK:
        pre_state = _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
        if pre_state not in ("remote", "remote_parked") or not (
            _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
        ):
            raise RuntimeError(
                "Graph residency reconciliation requires an audited active "
                f"remote topology; state={pre_state!r}, "
                f"remote_map_active={_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE!r}"
            )

        started_ns = time.perf_counter_ns()
        profile_all_remote_before = bool(
            _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE
        )
        parameter_vas_before = _uvm_original_parameter_va_snapshot()
        remote_map: dict[str, Any] | None = None
        cold_physical: dict[str, Any] | None = None
        hot_physical: dict[str, Any] | None = None
        hot_advise_count = 0
        hot_advise_skipped_for_full_dual_view = False
        hot_prefetch_count = 0
        hot_prefetch_executor = "none"
        hot_prefetch_driver_result = 0
        cold_range_count = 0
        hot_range_count = 0
        cold_bytes = 0
        hot_bytes = 0
        free_bytes = -1
        total_bytes = -1
        allocator_reserved_before = -1
        allocator_reserved_after_cleanup = -1
        allocator_reserved_after = -1
        post_reconcile_tlb_flush: dict[str, Any] | None = None
        parameter_mapping_rebuild: dict[str, Any] | None = None
        try:
            from vllm.device_allocator import uvm

            device = int(_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0)
            torch.cuda.synchronize(device)
            allocator_reserved_before = int(torch.cuda.memory_reserved(device))
            if not preserve_allocator_cache:
                # Profile/autotune tensors can remain in PyTorch's cache even
                # after their Python owners are gone. Return those blocks
                # before asking UVM to restore every hot expert. The final
                # post-reserve reconciliation deliberately skips this call.
                torch.cuda.empty_cache()
            allocator_reserved_after_cleanup = int(
                torch.cuda.memory_reserved(device)
            )

            cold_ranges = [
                consumer_range
                for _, binding in sorted(
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                )
                for consumer_range in (
                    _uvm_original_alias_binding_consumer_ranges(binding)
                )
            ]
            hot_ranges = [
                hot_range
                for _, binding in sorted(
                    _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                )
                for hot_range in _uvm_original_alias_binding_hot_ranges(binding)
            ]
            cold_range_count = len(cold_ranges)
            hot_range_count = len(hot_ranges)
            cold_bytes = sum(length for _, length in cold_ranges)
            hot_bytes = sum(length for _, length in hot_ranges)

            # Reinstall the audited CPU remote-read mapping.  This is
            # deliberately repeated after KV allocation so the capture gate
            # proves the current topology rather than trusting model-load
            # metadata.
            remote_map = dict(
                uvm.driver_prepare_ranges_remote_read_audited(
                    cold_ranges,
                    device=device,
                )
            )
            remote_contract_ok = (
                remote_map.get("ok") is True
                and int(remote_map.get("requested_count", -1))
                == len(cold_ranges)
                and int(remote_map.get("success_count", -1))
                == len(cold_ranges)
                and int(remote_map.get("failure_count", -1)) == 0
                and int(remote_map.get("requested_bytes", -1)) == cold_bytes
                and int(remote_map.get("cpu_resident_bytes", -1)) >= cold_bytes
                and int(remote_map.get("remote_mapped_bytes", -1)) >= cold_bytes
                and int(remote_map.get("covered_bytes", -1)) >= cold_bytes
                and int(remote_map.get("unknown_bytes", -1)) == 0
            )
            if not remote_contract_ok:
                raise RuntimeError(
                    "Cold expert remote-read reconciliation was incomplete: "
                    f"{remote_map!r}"
                )

            if hot_ranges:
                if _strict_env_flag(
                    "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW", False
                ):
                    hot_advise_skipped_for_full_dual_view = True
                else:
                    hot_advise_count = int(
                        uvm.batch_advise_ranges_preferred_location(
                            hot_ranges,
                            device,
                        )
                    )
                    if hot_advise_count != len(hot_ranges):
                        raise RuntimeError(
                            "Failed to set GPU preferred location for every "
                            "hot expert range: "
                            f"advised={hot_advise_count}, "
                            f"expected={len(hot_ranges)}"
                        )
                (
                    hot_prefetch_count,
                    hot_prefetch_executor,
                    hot_prefetch_driver_result,
                ) = uvm.batch_prefetch_ranges_to_device_with_executor(
                    hot_ranges,
                    device,
                    executor="cpp_batch",
                )
                if int(hot_prefetch_count) != len(hot_ranges):
                    raise RuntimeError(
                        "Failed to restore every hot expert range to GPU: "
                        f"prefetched={hot_prefetch_count}, "
                        f"expected={len(hot_ranges)}, "
                        f"executor={hot_prefetch_executor!r}, "
                        f"driver_result={hot_prefetch_driver_result}"
                    )

            torch.cuda.synchronize(device)
            full_dual_view_required = _strict_env_flag(
                "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW", False
            )
            rebuild_parameter_mappings = _strict_env_flag(
                "VLLM_UVM_REBUILD_PARAMETER_GPU_READ_MAPPINGS", False
            )
            if rebuild_parameter_mappings and not full_dual_view_required:
                # KV allocation can split or evict a 2 MiB parent shared by a
                # cold and a hot expert while leaving UVM's residency bits
                # unchanged. A hot-only prefetch then legitimately does no
                # work, even though the hardware parent entry is stale. Repair
                # complete Parameter allocations so both sides of every shared
                # parent are rebuilt in one transaction. Action 23 neither
                # migrates nor copies physical data.
                parameter_ranges = [
                    (
                        int(binding.consumer.data_ptr()),
                        int(binding.parameter_bytes),
                    )
                    for _, binding in sorted(
                        _CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS.items()
                    )
                ]
                parameter_mapping_rebuild = dict(
                    uvm.driver_rebuild_ranges_gpu_read_mappings(
                        parameter_ranges,
                        device=device,
                    )
                )
                if not (
                    parameter_mapping_rebuild.get("ok") is True
                    and int(
                        parameter_mapping_rebuild.get(
                            "requested_count", -1
                        )
                    )
                    == len(parameter_ranges)
                    and int(
                        parameter_mapping_rebuild.get("success_count", -1)
                    )
                    == len(parameter_ranges)
                    and int(
                        parameter_mapping_rebuild.get("failure_count", -1)
                    )
                    == 0
                ):
                    raise RuntimeError(
                        "Complete Parameter GPU READ mapping rebuild failed: "
                        f"{parameter_mapping_rebuild!r}"
                    )

            if _strict_env_flag(
                "VLLM_UVM_POST_RECONCILE_FULL_TLB_INVALIDATE", False
            ):
                if not cold_ranges:
                    raise RuntimeError(
                        "Post-reconcile full TLB invalidate lacks a managed "
                        "anchor range"
                    )
                # Action 21 is idempotent for an already-audited cold range.
                # With the corresponding opt-in UVM module parameter it also
                # provides a synchronous, all-GPC TLB completion boundary
                # after the later hot-expert GPU prefetch has finished.
                post_reconcile_tlb_flush = dict(
                    uvm.driver_prepare_ranges_remote_read_audited(
                        cold_ranges[:1],
                        device=device,
                    )
                )
                if not (
                    post_reconcile_tlb_flush.get("ok") is True
                    and int(
                        post_reconcile_tlb_flush.get("success_count", -1)
                    )
                    == 1
                    and int(
                        post_reconcile_tlb_flush.get("failure_count", -1)
                    )
                    == 0
                ):
                    raise RuntimeError(
                        "Post-reconcile full GPU TLB invalidate failed: "
                        f"{post_reconcile_tlb_flush!r}"
                    )
            cold_physical = dict(
                uvm.driver_batch_query_ranges_residency_bytes(
                    cold_ranges,
                    device=device,
                    wait_for_trackers=False,
                )
            )
            if not (
                cold_physical.get("ok") is True
                and int(cold_physical.get("queried_bytes", -1)) == cold_bytes
                and int(cold_physical.get("cpu_bytes", -1)) == cold_bytes
                and int(cold_physical.get("gpu_bytes", -1)) == 0
                and int(cold_physical.get("any_bytes", -1)) == cold_bytes
                and int(cold_physical.get("unknown_bytes", -1)) == 0
            ):
                raise RuntimeError(
                    "Cold experts lost exclusive CPU residency at the Graph "
                    f"capture gate: {cold_physical!r}"
                )

            hot_physical = dict(
                uvm.driver_batch_query_ranges_residency_bytes(
                    hot_ranges,
                    device=device,
                    wait_for_trackers=False,
                )
            )
            expected_hot_cpu_bytes = (
                _uvm_original_alias_expected_shadow_cpu_bytes(hot_ranges)
            )
            if hot_ranges and not (
                hot_physical.get("ok") is True
                and int(hot_physical.get("queried_bytes", -1)) == hot_bytes
                and int(hot_physical.get("gpu_bytes", -1)) == hot_bytes
                and int(hot_physical.get("cpu_bytes", -1))
                == expected_hot_cpu_bytes
                and int(hot_physical.get("any_bytes", -1)) == hot_bytes
                and int(hot_physical.get("unknown_bytes", -1)) == 0
            ):
                raise RuntimeError(
                    "Hot experts were not exclusively GPU resident at the "
                    f"Graph capture gate: {hot_physical!r}"
                )

            free_bytes, total_bytes = (
                int(value) for value in torch.cuda.mem_get_info(device)
            )
            allocator_reserved_after = int(torch.cuda.memory_reserved(device))
            if (
                preserve_allocator_cache
                and allocator_reserved_after < allocator_reserved_before
            ):
                raise RuntimeError(
                    "Prefill allocator reserve shrank during hot-expert "
                    "restoration: "
                    f"before={allocator_reserved_before}, "
                    f"after={allocator_reserved_after}"
                )
            if free_bytes < min_free_bytes:
                raise RuntimeError(
                    "Insufficient GPU headroom after expert residency "
                    "reconciliation: "
                    f"free_bytes={free_bytes}, required_bytes={min_free_bytes}"
                )

            parameter_vas_after = _uvm_original_parameter_va_snapshot()
            if parameter_vas_after != parameter_vas_before:
                raise RuntimeError(
                    "Original Parameter VAs changed during Graph residency "
                    "reconciliation"
                )

            _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT = remote_map
            _CPU_OFFLOAD_UVM_REMOTE_MAP_HISTORY.append(dict(remote_map))
            _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE = False
            finished_ns = time.perf_counter_ns()
            result = {
                "event": "cpu_offload_uvm_original_va_graph_residency_reconciled",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "ok": True,
                "pre_state": pre_state,
                "post_state": pre_state,
                "wall_ns": finished_ns - started_ns,
                "parameter_vas_before": parameter_vas_before,
                "parameter_vas_after": parameter_vas_after,
                "cold_range_count": len(cold_ranges),
                "cold_bytes": int(cold_bytes),
                "hot_range_count": len(hot_ranges),
                "hot_bytes": int(hot_bytes),
                "expected_hot_cpu_shadow_bytes": int(
                    expected_hot_cpu_bytes
                ),
                "hot_advise_count": int(hot_advise_count),
                "hot_advise_skipped_for_full_dual_view": bool(
                    hot_advise_skipped_for_full_dual_view
                ),
                "hot_prefetch_count": int(hot_prefetch_count),
                "hot_prefetch_executor": hot_prefetch_executor,
                "hot_prefetch_driver_result": int(hot_prefetch_driver_result),
                "remote_map": remote_map,
                "parameter_mapping_rebuild": parameter_mapping_rebuild,
                "post_reconcile_tlb_flush": post_reconcile_tlb_flush,
                "cold_physical_residency": cold_physical,
                "hot_physical_residency": hot_physical,
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
                "minimum_free_bytes": int(min_free_bytes),
                "allocator_cache_preserved": bool(preserve_allocator_cache),
                "allocator_reserved_before": int(allocator_reserved_before),
                "allocator_reserved_after_cleanup": int(
                    allocator_reserved_after_cleanup
                ),
                "allocator_reserved_after": int(allocator_reserved_after),
                "profile_all_remote_before": profile_all_remote_before,
                "profile_all_remote_after": False,
            }
            _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(result)
            _append_cpu_offload_manifest(result)
            return result
        except Exception as exc:
            _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
            _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "poisoned"
            failed = {
                "event": "cpu_offload_uvm_original_va_graph_residency_reconciled",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "ok": False,
                "error": repr(exc),
                "pre_state": pre_state,
                "post_state": "poisoned",
                "wall_ns": time.perf_counter_ns() - started_ns,
                "parameter_vas_before": parameter_vas_before,
                "remote_map": remote_map,
                "parameter_mapping_rebuild": parameter_mapping_rebuild,
                "post_reconcile_tlb_flush": post_reconcile_tlb_flush,
                "cold_physical_residency": cold_physical,
                "hot_physical_residency": hot_physical,
                "cold_range_count": int(cold_range_count),
                "cold_bytes": int(cold_bytes),
                "hot_range_count": int(hot_range_count),
                "hot_bytes": int(hot_bytes),
                "hot_advise_count": int(hot_advise_count),
                "hot_advise_skipped_for_full_dual_view": bool(
                    hot_advise_skipped_for_full_dual_view
                ),
                "hot_prefetch_count": int(hot_prefetch_count),
                "hot_prefetch_executor": hot_prefetch_executor,
                "hot_prefetch_driver_result": int(hot_prefetch_driver_result),
                "free_bytes": int(free_bytes),
                "total_bytes": int(total_bytes),
                "minimum_free_bytes": int(min_free_bytes),
                "allocator_cache_preserved": bool(preserve_allocator_cache),
                "allocator_reserved_before": int(allocator_reserved_before),
                "allocator_reserved_after_cleanup": int(
                    allocator_reserved_after_cleanup
                ),
                "allocator_reserved_after": int(allocator_reserved_after),
                "profile_all_remote_before": profile_all_remote_before,
                "profile_all_remote_after": bool(
                    _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE
                ),
            }
            _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(failed)
            _append_cpu_offload_manifest(failed)
            raise RuntimeError(
                "Original-VA Graph residency reconciliation failed; the "
                "process is poisoned and must be torn down"
            ) from exc


def prepare_cpu_offload_uvm_original_va_remote_graph_capture() -> dict[str, Any]:
    """Establish the no-H2D remote topology immediately before capture."""
    mode = cpu_offload_uvm_original_va_phase_mode()
    if mode == "off":
        return {"ok": True, "reason": "phase_mode_off", "phase_mode": mode}
    if _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED:
        raise RuntimeError(
            "A certified original-VA remote Graph already exists; recapture and "
            "lazy capture are unsupported for the phase-switch experiment"
        )
    transition = transition_cpu_offload_uvm_original_va_to_remote(
        reason="remote_graph_capture_prepare",
        preserve_remote_graphs=False,
    )
    reconciliation = (
        reconcile_cpu_offload_uvm_original_va_remote_graph_residency()
    )
    return {
        **transition,
        "graph_residency_reconciliation": reconciliation,
    }


def activate_cpu_offload_uvm_original_va_prefill_topology(
    *,
    finish_all_remote_profile: bool = False,
) -> dict[str, Any]:
    """Restore sealed alias banks after all initialization Graph replays end.

    ``finish_all_remote_profile`` is used by the capture-before-hot-restore
    startup path.  It ends the temporary profile bypass after RESUME so the
    real Prefill H2D nodes are recorded, while deliberately leaving the hot
    complement on its audited CPU remote-read mappings until the final Graph
    and all of its allocator segments are live.
    """
    global _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
    global _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_RESUME_RESULT
    global _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE

    mode = cpu_offload_uvm_original_va_phase_mode()
    if mode not in (
        "prefill_remote_decode",
        "token_threshold",
        "prefill_graph_remote_decode",
    ):
        return {"ok": True, "reason": "no_alias_prefill", "phase_mode": mode}
    state = _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
    if state == "sealed":
        return {"ok": True, "reason": "already_sealed", "state": state}
    expected_remote_state = (
        "remote_parked" if _CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED else "remote"
    )
    if state != expected_remote_state or not _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE:
        raise RuntimeError(
            "Cannot activate alias-prefill topology from lifecycle state "
            f"{state!r}"
        )
    started_ns = time.perf_counter_ns()
    before = _uvm_original_parameter_va_snapshot()
    snapshot_finished_ns = time.perf_counter_ns()
    finalized: dict[str, Any] | None = None
    resumed: dict[str, Any] | None = None
    runtime_cold_read_gate: dict[str, Any] | None = None
    stream_barrier: dict[str, Any] | None = None
    fast_runtime_transition = (
        _uvm_original_alias_fast_runtime_transition_enabled()
    )
    barrier_finished_ns = snapshot_finished_ns
    resume_finished_ns = snapshot_finished_ns
    if _CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED:
        from vllm.device_allocator import uvm_static_alias

        device = int(_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0)
        if fast_runtime_transition:
            stream_barrier = _uvm_original_alias_runtime_stream_barrier(
                stage="before_resume",
                device=device,
            )
        else:
            torch.cuda.synchronize(device)
            _uvm_original_alias_hot_cuda_read_gate(
                stage="runtime_prefill_before_resume",
                device=device,
            )
        barrier_finished_ns = time.perf_counter_ns()
        resumed = dict(
            uvm_static_alias.driver_static_alias_resume(
                int(_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID),
                device=device,
                synchronize=False,
            )
        )
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_RESUME_RESULT = resumed
        if not bool(resumed.get("ok")):
            poison_cpu_offload_uvm_original_va_phase(
                f"static-alias resume failed: {resumed!r}"
            )
            raise RuntimeError(
                "Failed to resume original-VA aliases after remote decode"
            )
        # The ioctl contract has now restored every consumer mapping.  Publish
        # that state before optional CUDA read gates inspect the consumer VAs;
        # leaving the userspace state as ``remote_parked`` here would make a
        # successful driver RESUME indistinguishable from a parked topology.
        _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = False
        _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "sealed"
        resume_finished_ns = time.perf_counter_ns()
        if not fast_runtime_transition:
            _uvm_original_alias_hot_cuda_read_gate(
                stage="runtime_prefill_after_resume",
                device=device,
            )
        if (
            not fast_runtime_transition
            and not finish_all_remote_profile
            and _strict_env_flag(
                "VLLM_UVM_RUNTIME_POST_RESUME_COLD_READ_GATE", False
            )
        ):
            # The four upload/backing arenas are shared by many layers, so
            # their current bytes cannot equal every layer's source at once.
            # Read each sampled consumer PTE without mutating the backing.  It
            # is both a fail-closed mapping check and a completion fence before
            # the retained Prefill Graph touches those VAs.
            runtime_cold_read_gate = (
                _uvm_original_alias_current_cold_cuda_read_gate(
                    stage="runtime_prefill_after_resume",
                    device=device,
                    binding_names=tuple(
                        sorted(_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS)
                    ),
                    verify_source_bytes=False,
                )
            )
    else:
        finalized = finalize_cpu_offload_graph_static_alias()
    after = _uvm_original_parameter_va_snapshot()
    if before != after:
        poison_cpu_offload_uvm_original_va_phase(
            "Parameter VA changed while restoring alias-prefill topology"
        )
        raise RuntimeError("Original Parameter VA changed while restoring aliases")
    profile_all_remote_before = bool(_CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE)
    if finish_all_remote_profile:
        if not profile_all_remote_before:
            raise RuntimeError(
                "Prefill capture requested an all-remote profile handoff, but "
                "the all-remote profile topology is not active"
            )
        _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE = False
    result = {
        "event": "cpu_offload_uvm_original_va_alias_prefill_activated",
        "timestamp": time.time(),
        "pid": os.getpid(),
        "ok": bool(
            (resumed or finalized or {"ok": False}).get("ok")
        ),
        "reason": "remote_capture_complete",
        "pre_state": state,
        "post_state": _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE,
        "wall_ns": time.perf_counter_ns() - started_ns,
        "snapshot_ns": snapshot_finished_ns - started_ns,
        "stream_dependency_ns": (
            barrier_finished_ns - snapshot_finished_ns
        ),
        "resume_ns": resume_finished_ns - barrier_finished_ns,
        "fast_runtime_transition": fast_runtime_transition,
        "stream_barrier": stream_barrier,
        "parameter_vas_before": before,
        "parameter_vas_after": after,
        "finalize": finalized,
        "resume": resumed,
        "runtime_cold_read_gate": runtime_cold_read_gate,
        "finish_all_remote_profile": bool(finish_all_remote_profile),
        "profile_all_remote_before": profile_all_remote_before,
        "profile_all_remote_after": bool(
            _CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE
        ),
    }
    _CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY.append(result)
    _append_cpu_offload_manifest(result)
    return result


def validate_cpu_offload_uvm_original_va_remote_graph_replay() -> None:
    """Fail before launch unless the certified remote Graph topology is active."""
    mode = cpu_offload_uvm_original_va_phase_mode()
    if mode == "off":
        return
    certificate = _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE
    remote_map = _CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT
    failures: list[str] = []
    expected_state = (
        "remote_parked" if _CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED else "remote"
    )
    if expected_state != _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE:
        failures.append(f"state={_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE}")
    if _CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED:
        if _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID <= 0:
            failures.append("parked_alias_session_missing")
    elif _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID != 0:
        failures.append(f"session={_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID}")
    if not _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE:
        failures.append("remote_map_inactive")
    if not _CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED:
        failures.append("remote_graph_not_captured")
    if not isinstance(certificate, dict):
        failures.append("remote_graph_certificate_missing")
    else:
        if certificate.get("topology") != "remote":
            failures.append("remote_graph_certificate_topology_mismatch")
        if int(certificate.get("actual_h2d_bytes", -1)) != 0:
            failures.append("capture_h2d_nonzero")
        if tuple(certificate.get("parameter_vas", ())) != (
            _uvm_original_parameter_va_snapshot()
        ):
            failures.append("parameter_va_fingerprint_changed")
    if mode == "prefill_graph_remote_decode":
        if not isinstance(
            _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE, dict
        ):
            failures.append("dual_graph_partition_certificate_missing")
        if _CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED:
            if _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED:
                failures.append("retained_alias_prefill_graphs_destroyed")
            partition = _CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE
            if not isinstance(partition, dict) or int(
                partition.get("remote_entry_count", 0)
            ) <= 0:
                failures.append("retained_remote_decode_graph_missing")
        elif not _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED:
            failures.append("alias_prefill_graphs_still_live")
        if (
            not _CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED
            and _CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT <= 0
        ):
            failures.append("remote_decode_graph_not_preserved")
    if not isinstance(remote_map, dict) or remote_map.get("ok") is not True:
        failures.append("remote_map_audit_missing")
    if failures:
        poison_cpu_offload_uvm_original_va_phase(
            "remote Graph replay contract failed: " + ", ".join(failures)
        )
        raise RuntimeError(
            "Refusing original-VA remote CUDA Graph replay: "
            + ", ".join(failures)
        )


def validate_cpu_offload_uvm_original_va_alias_graph_replay() -> None:
    """Fail before launch unless a certified sealed alias Graph is active."""
    mode = cpu_offload_uvm_original_va_phase_mode()
    if mode != "prefill_graph_remote_decode":
        raise RuntimeError(
            "Alias-prefill Graph replay requires "
            "prefill_graph_remote_decode phase mode"
        )
    certificate = _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE
    failures: list[str] = []
    if _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE != "sealed":
        failures.append(f"state={_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE}")
    if _CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID <= 0:
        failures.append("alias_session_missing")
    if _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE:
        failures.append("remote_map_unexpectedly_active")
    if not _CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED:
        failures.append("alias_graph_not_captured")
    if _CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED:
        failures.append("alias_graphs_already_destroyed")
    if not isinstance(certificate, dict):
        failures.append("alias_graph_certificate_missing")
    else:
        if certificate.get("topology") != "sealed":
            failures.append("alias_graph_certificate_topology_mismatch")
        if int(certificate.get("actual_h2d_bytes", 0)) <= 0:
            failures.append("alias_graph_certificate_has_no_h2d")
        if tuple(certificate.get("parameter_vas", ())) != (
            _uvm_original_parameter_va_snapshot()
        ):
            failures.append("parameter_va_fingerprint_changed")
    if not isinstance(_CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE, dict):
        failures.append("dual_graph_partition_certificate_missing")
    if failures:
        poison_cpu_offload_uvm_original_va_phase(
            "alias Graph replay contract failed: " + ", ".join(failures)
        )
        raise RuntimeError(
            "Refusing original-VA alias CUDA Graph replay: "
            + ", ".join(failures)
        )


def _require_all_copy_exact_names_seen() -> None:
    if not _copy_exact_names_enabled() or not _CPU_OFFLOAD_EXACT_PENDING:
        return
    missing = sorted(_CPU_OFFLOAD_EXACT_PENDING)
    preview = ", ".join(repr(name) for name in missing[:8])
    if len(missing) > 8:
        preview += f", ... ({len(missing)} total)"
    raise RuntimeError(
        f"Exact-name CPU offload did not find every requested parameter: {preview}"
    )


def _is_uvm_cpu_offload_backend(backend: str) -> bool:
    return backend in (
        "isolated_uvm_migration",
        "isolated_uvm_remote",
        "isolated_uvm_readmostly_remote",
        "isolated_uvm_prepare_remote",
        "isolated_uvm_prepare_remote_noreadmostly",
        "isolated_uvm_prepare_remote_noreadmostly_noaccessedby",
        "isolated_uvm_hybrid_remote",
        "isolated_uvm_hybrid_remote_noreadmostly",
        "isolated_uvm_hybrid_remote_noreadmostly_noaccessedby",
        "uvm_remote",
        "uvm_pure_remote",
    )


def _hybrid_start_mode() -> str:
    return (
        os.environ.get("VLLM_UVM_HYBRID_START_MODE", "gate_after_first").strip().lower()
    )


def maybe_offload_to_cpu(
    module: torch.nn.Module,
    module_prefix: str | None = None,
) -> torch.nn.Module:
    if (params := next(module.parameters(), None)) is None:
        return module

    device = params.device

    if device == torch.device("cpu"):
        return module

    global _CPU_OFFLOAD_MAX_BYTES, _CPU_OFFLOAD_BYTES, _CPU_OFFLOAD_ORDER
    global _CPU_OFFLOAD_EXACT_PENDING, _CPU_OFFLOAD_EXACT_SEEN
    global _CPU_OFFLOAD_STATIC_BANK_BYTES
    global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS
    exact_names_enabled = _copy_exact_names_enabled()
    if not exact_names_enabled and _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
        return module

    pin_memory = is_pin_memory_available()
    uva_available = is_uva_available()

    offload_backend = _normalized_cpu_offload_backend()
    copy_on_forward = offload_backend == "copy"
    uva_offloading = offload_backend == "uva"
    direct_cuda_parameter = not copy_on_forward
    if uva_offloading:
        assert uva_available, "V1 CPU offloading requires uva support"
    if (
        offload_backend != "uva"
        and not copy_on_forward
        and not _is_uvm_cpu_offload_backend(offload_backend)
    ):
        raise ValueError(
            "Unknown VLLM_CPU_OFFLOAD_BACKEND="
            f"{offload_backend!r}; use 'uva', 'copy', "
            "'isolated_uvm_migration', "
            "'isolated_uvm_remote', 'isolated_uvm_readmostly_remote', "
            "'isolated_uvm_prepare_remote', "
            "'isolated_uvm_prepare_remote_noreadmostly', "
            "'isolated_uvm_prepare_remote_noreadmostly_noaccessedby', "
            "'isolated_uvm_hybrid_remote', or "
            "'isolated_uvm_hybrid_remote_noreadmostly', or "
            "'isolated_uvm_hybrid_remote_noreadmostly_noaccessedby'"
        )

    # offload parameters to CPU
    # use pin_memory if possible, which helps cudagraph capture speed
    offloaded_parameters = False
    exact_local_parameter_names: set[str] = set()
    static_bank_keys: dict[str, tuple[Any, ...]] = {}
    graph_static_alias_bindings: dict[
        str, _GraphStaticAliasBinding | _UvmOriginalAliasBinding
    ] = {}
    static_signature_lanes: dict[tuple[Any, ...], int] = {}
    hybrid_remote_ranges: list[tuple[int, int]] = []
    hybrid_remote_device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    for param_index, (param_name, p) in enumerate(module.named_parameters()):
        if not exact_names_enabled and _CPU_OFFLOAD_BYTES >= _CPU_OFFLOAD_MAX_BYTES:
            # we use per-parameter offloading
            # one module might have some parameters offloaded and some not
            break
        bytes_before = p.data.numel() * p.data.element_size()
        full_parameter_name = (
            f"{module_prefix}.{param_name}"
            if module_prefix and param_name
            else (module_prefix or param_name)
        )
        uvm_original_alias = bool(
            exact_names_enabled
            and _CPU_OFFLOAD_GRAPH_STATIC_ALIAS
            and _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND == "uvm_original_va"
        )
        accounted_bytes = int(bytes_before)

        if exact_names_enabled:
            assert _CPU_OFFLOAD_EXACT_NAMES is not None
            if full_parameter_name not in _CPU_OFFLOAD_EXACT_NAMES:
                continue
            if full_parameter_name in _CPU_OFFLOAD_EXACT_SEEN:
                raise RuntimeError(
                    "Exact-name CPU offload encountered the same full parameter "
                    f"name more than once: {full_parameter_name!r}"
                )
            if not pin_memory:
                raise RuntimeError(
                    "Exact-name CPU offload requires pinned host memory, but "
                    "is_pin_memory_available() returned False"
                )
            if uvm_original_alias:
                accounted_bytes = int(
                    _uvm_original_alias_parameter_geometry(
                        full_parameter_name=full_parameter_name,
                        tensor=p.data,
                        parameter_bytes=int(bytes_before),
                    )[2]
                )
            if _CPU_OFFLOAD_BYTES + accounted_bytes > _CPU_OFFLOAD_MAX_BYTES:
                raise RuntimeError(
                    "Exact-name CPU offload selection exceeds cpu_offload_gb: "
                    f"adding {full_parameter_name!r} ({accounted_bytes} "
                    "selected bytes) "
                    f"would exceed {_CPU_OFFLOAD_MAX_BYTES} bytes"
                )
            _validate_copy_exact_context(device, check_stream=False)

        manifest_extra: dict[str, Any] = {"offload_backend": offload_backend}

        if uvm_original_alias:
            binding = _create_uvm_original_alias_binding(
                full_parameter_name=full_parameter_name,
                parameter=p,
                tensor=p.data,
                device=device,
                requested_bytes=int(bytes_before),
            )
            graph_static_alias_bindings[param_name] = binding
            exact_local_parameter_names.add(param_name)
            direct_cuda_parameter = False
            p._vllm_uvm_original_alias_binding = binding
            manifest_extra.update(
                {
                    "cpu_data_pinned": None,
                    "cpu_source_deferred_until_weights_loaded": True,
                    "copy_on_forward": True,
                    "copy_exact_names": True,
                    "copy_exact_names_file": _CPU_OFFLOAD_EXACT_NAMES_FILE,
                    "copy_static_bank": False,
                    "graph_static_alias": True,
                    "graph_static_alias_backend": "uvm_original_va",
                    "graph_static_alias_layer_index": binding.plan.layer_index,
                    "graph_static_alias_bank_id": binding.plan.bank_id,
                    "graph_static_alias_consumer_index": (binding.plan.consumer_index),
                    "graph_static_alias_consumer_count": (binding.plan.consumer_count),
                    "graph_static_alias_upload_va": int(
                        binding.arena.upload.data_ptr()
                    ),
                    "graph_static_alias_source_ptr": None,
                    "graph_static_alias_consumer_va": int(binding.consumer.data_ptr()),
                    "graph_static_alias_requested_bytes": int(binding.requested_bytes),
                    "graph_static_alias_parameter_bytes": int(
                        binding.parameter_bytes
                    ),
                    "graph_static_alias_expert_ids": (
                        list(binding.expert_ids)
                        if binding.expert_ids is not None
                        else None
                    ),
                    "graph_static_alias_expert_stride_bytes": int(
                        binding.expert_stride_bytes
                    ),
                    "graph_static_alias_consumer_ranges": tuple(
                        binding.consumer_ranges
                    ),
                    "graph_static_alias_aligned_bytes": int(
                        binding.arena.aligned_bytes
                    ),
                    "graph_static_alias_backing_bytes": int(
                        getattr(
                            binding.arena,
                            "requested_bytes",
                            binding.requested_bytes,
                        )
                    ),
                    "graph_static_alias_granularity": None,
                    "graph_static_alias_upload_access": "read_write",
                    "graph_static_alias_consumer_access": "read_only",
                    "graph_static_alias_total_backing_bytes": int(
                        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
                    ),
                    "uvm_original_parameter_va": int(binding.consumer.data_ptr()),
                    "uvm_original_parameter_va_retained": True,
                }
            )
        elif _is_uvm_cpu_offload_backend(offload_backend):
            from vllm.device_allocator.uvm import (
                batch_advise_ranges_accessed_by,
                batch_advise_ranges_preferred_location,
                batch_advise_ranges_read_mostly,
                batch_prefetch_ranges_to_device,
                driver_prepare_ranges_remote_read,
                driver_prepare_ranges_remote_read_bg_deferred,
                ensure_uvm_runtime_library,
                probe_driver_capable_range,
                round_ranges_to_pages,
                uvm_local_memory_pool,
            )

            ensure_uvm_runtime_library()
            with torch.cuda.device(device), uvm_local_memory_pool():
                uvm_data = torch.empty_strided(
                    size=p.data.size(),
                    stride=p.data.stride(),
                    dtype=p.data.dtype,
                    layout=p.data.layout,
                    device=device,
                )
            uvm_data.copy_(p.data)

            ptr = int(uvm_data.data_ptr())
            ranges = [(ptr, int(bytes_before))]
            phase_f_ranges = ranges
            if offload_backend in (
                "isolated_uvm_prepare_remote",
                "isolated_uvm_prepare_remote_noreadmostly",
                "isolated_uvm_prepare_remote_noreadmostly_noaccessedby",
                "isolated_uvm_hybrid_remote",
                "isolated_uvm_hybrid_remote_noreadmostly",
                "isolated_uvm_hybrid_remote_noreadmostly_noaccessedby",
            ) and _env_flag("VLLM_UVM_HYBRID_ROUND_PAGES", True):
                phase_f_ranges = round_ranges_to_pages(ranges)
            device_index = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
            prefetch_cpu_result = batch_prefetch_ranges_to_device(ranges, -1)
            if _env_flag("VLLM_CPU_OFFLOAD_UVM_SYNC", True):
                # Phase-F prepare only acts on CPU-resident, non-GPU-resident
                # pages. Make the CPU prefetch visible before advising and
                # preparing remote-read/bg-promotion state.
                torch.cuda.synchronize()
            remote_like_backend = offload_backend in (
                "isolated_uvm_remote",
                "isolated_uvm_readmostly_remote",
                "isolated_uvm_prepare_remote",
                "isolated_uvm_prepare_remote_noreadmostly",
                "isolated_uvm_prepare_remote_noreadmostly_noaccessedby",
                "isolated_uvm_hybrid_remote",
                "isolated_uvm_hybrid_remote_noreadmostly",
                "isolated_uvm_hybrid_remote_noreadmostly_noaccessedby",
                "uvm_remote",
                "uvm_pure_remote",
            )
            preferred_cpu_result = 0
            if remote_like_backend:
                preferred_cpu_result = batch_advise_ranges_preferred_location(
                    ranges, -1
                )
            readmostly_result = 0
            readmostly_rounded_result = 0
            readmostly_backend = offload_backend in (
                "isolated_uvm_readmostly_remote",
                "isolated_uvm_prepare_remote",
                "isolated_uvm_hybrid_remote",
            )
            if readmostly_backend:
                readmostly_result = batch_advise_ranges_read_mostly(
                    ranges,
                    device=int(device_index),
                )
                if phase_f_ranges != ranges and _env_flag(
                    "VLLM_UVM_HYBRID_ROUND_READMOSTLY", True
                ):
                    readmostly_rounded_result = batch_advise_ranges_read_mostly(
                        phase_f_ranges,
                        device=int(device_index),
                    )
            accessed_by_result = 0
            skip_accessed_by_backend = offload_backend in (
                "isolated_uvm_prepare_remote_noreadmostly_noaccessedby",
                "isolated_uvm_hybrid_remote_noreadmostly_noaccessedby",
            )
            if remote_like_backend and not skip_accessed_by_backend:
                accessed_by_result = batch_advise_ranges_accessed_by(
                    ranges, int(device_index)
                )
            remote_read_bg_result = (0, 0, 0, 0)
            if offload_backend in (
                "isolated_uvm_prepare_remote",
                "isolated_uvm_prepare_remote_noreadmostly",
                "isolated_uvm_prepare_remote_noreadmostly_noaccessedby",
            ):
                remote_read_bg_result = driver_prepare_ranges_remote_read(
                    phase_f_ranges,
                    device=int(device_index),
                )
            elif offload_backend in (
                "isolated_uvm_hybrid_remote",
                "isolated_uvm_hybrid_remote_noreadmostly",
                "isolated_uvm_hybrid_remote_noreadmostly_noaccessedby",
            ):
                remote_read_bg_result = driver_prepare_ranges_remote_read_bg_deferred(
                    phase_f_ranges,
                    device=int(device_index),
                )
                hybrid_remote_ranges.extend(phase_f_ranges)
            if _env_flag("VLLM_CPU_OFFLOAD_UVM_SYNC", True):
                torch.cuda.synchronize()
            managed, managed_reason, managed_code = probe_driver_capable_range(
                ptr, int(bytes_before)
            )

            p._vllm_offloaded_uvm_data = uvm_data
            p.data = uvm_data
            manifest_extra.update(
                {
                    "uvm_managed": bool(managed),
                    "uvm_managed_reason": managed_reason,
                    "uvm_managed_probe_code": int(managed_code),
                    "uvm_data_ptr": ptr,
                    "uvm_prefetch_cpu_result": int(prefetch_cpu_result),
                    "uvm_preferred_cpu_result": int(preferred_cpu_result),
                    "uvm_readmostly_result": int(readmostly_result),
                    "uvm_readmostly_rounded_result": int(readmostly_rounded_result),
                    "uvm_accessed_by_result": int(accessed_by_result),
                    "uvm_accessed_by_device": int(device_index),
                    "uvm_phase_f_range_count": len(phase_f_ranges),
                    "uvm_phase_f_bytes": int(sum(size for _, size in phase_f_ranges)),
                    "uvm_phase_f_rounded": bool(phase_f_ranges != ranges),
                    "uvm_phase_f_first_ptr": int(
                        phase_f_ranges[0][0] if phase_f_ranges else 0
                    ),
                    "uvm_phase_f_first_bytes": int(
                        phase_f_ranges[0][1] if phase_f_ranges else 0
                    ),
                    "uvm_remote_read_bg_success": int(remote_read_bg_result[0]),
                    "uvm_remote_read_bg_failure": int(remote_read_bg_result[1]),
                    "uvm_remote_read_bg_rm_status": int(remote_read_bg_result[2]),
                    "uvm_remote_read_bg_ioctl_ret": int(remote_read_bg_result[3]),
                    "uvm_bg_start_deferred": bool(
                        offload_backend
                        in (
                            "isolated_uvm_hybrid_remote",
                            "isolated_uvm_hybrid_remote_noreadmostly",
                            "isolated_uvm_hybrid_remote_noreadmostly_noaccessedby",
                        )
                    ),
                    "uvm_sync": _env_flag("VLLM_CPU_OFFLOAD_UVM_SYNC", True),
                    "cpu_data_pinned": False,
                }
            )
        else:
            # `torch.empty_like` does not support `pin_memory` argument
            cpu_data = torch.empty_strided(
                size=p.data.size(),
                stride=p.data.stride(),
                dtype=p.data.dtype,
                layout=p.data.layout,
                device="cpu",
                pin_memory=pin_memory,
            )
            cpu_data.copy_(p.data)
            cpu_data_pinned = bool(
                cpu_data.is_pinned() if hasattr(cpu_data, "is_pinned") else False
            )
            if exact_names_enabled and not cpu_data_pinned:
                raise RuntimeError(
                    "Exact-name CPU offload allocated pageable memory for "
                    f"{full_parameter_name!r}; pinned memory is required"
                )

            bank_key: tuple[Any, ...] | None = None
            bank_lane: int | None = None
            bank_tensor_bytes = 0
            graph_static_alias_binding: _GraphStaticAliasBinding | None = None
            if exact_names_enabled and _CPU_OFFLOAD_GRAPH_STATIC_ALIAS:
                graph_static_alias_binding = _create_graph_static_alias_binding(
                    full_parameter_name=full_parameter_name,
                    tensor=p.data,
                    device=device,
                    requested_bytes=int(bytes_before),
                )
                graph_static_alias_bindings[param_name] = graph_static_alias_binding
            elif exact_names_enabled and _CPU_OFFLOAD_STATIC_BANK:
                size = tuple(int(value) for value in p.data.size())
                stride = tuple(int(value) for value in p.data.stride())
                signature = (
                    size,
                    stride,
                    p.data.dtype,
                    getattr(device, "type", None),
                    int(_CPU_OFFLOAD_OWNER_DEVICE_INDEX or 0),
                )
                bank_lane = static_signature_lanes.get(signature, 0)
                static_signature_lanes[signature] = bank_lane + 1
                bank_key = (*signature, bank_lane)
                bank = _CPU_OFFLOAD_STATIC_BANKS.get(bank_key)
                if bank is None:
                    bank = torch.empty_strided(
                        size=p.data.size(),
                        stride=p.data.stride(),
                        dtype=p.data.dtype,
                        layout=p.data.layout,
                        device=device,
                    )
                    _CPU_OFFLOAD_STATIC_BANKS[bank_key] = bank
                    bank_tensor_bytes = int(bank.numel() * bank.element_size())
                    _CPU_OFFLOAD_STATIC_BANK_BYTES += bank_tensor_bytes
                else:
                    bank_tensor_bytes = int(bank.numel() * bank.element_size())
                static_bank_keys[param_name] = bank_key

            if not uva_offloading:
                p.data = cpu_data
                direct_cuda_parameter = False
            else:
                # keep the cpu data alive
                p._vllm_offloaded_cpu_data = cpu_data
                p.data = get_cuda_view_from_cpu_tensor(cpu_data)
            manifest_extra.update(
                {
                    "cpu_data_pinned": cpu_data_pinned,
                    "copy_on_forward": bool(copy_on_forward),
                }
            )
            if exact_names_enabled:
                exact_local_parameter_names.add(param_name)
                alias_allocation = (
                    graph_static_alias_binding.arena.allocation
                    if graph_static_alias_binding is not None
                    else None
                )
                alias_plan = (
                    graph_static_alias_binding.plan
                    if graph_static_alias_binding is not None
                    else None
                )
                manifest_extra.update(
                    {
                        "copy_exact_names": True,
                        "copy_exact_names_file": _CPU_OFFLOAD_EXACT_NAMES_FILE,
                        "copy_static_bank": bool(_CPU_OFFLOAD_STATIC_BANK),
                        "graph_static_alias": bool(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS),
                        "graph_static_alias_layer_index": (
                            alias_plan.layer_index if alias_plan is not None else None
                        ),
                        "graph_static_alias_bank_id": (
                            alias_plan.bank_id if alias_plan is not None else None
                        ),
                        "graph_static_alias_consumer_index": (
                            alias_plan.consumer_index
                            if alias_plan is not None
                            else None
                        ),
                        "graph_static_alias_consumer_count": (
                            alias_plan.consumer_count
                            if alias_plan is not None
                            else None
                        ),
                        "graph_static_alias_token": (
                            int(alias_allocation.token)
                            if alias_allocation is not None
                            else None
                        ),
                        "graph_static_alias_upload_va": (
                            int(alias_allocation.upload_va)
                            if alias_allocation is not None
                            else None
                        ),
                        "graph_static_alias_source_ptr": (
                            int(cpu_data.data_ptr())
                            if graph_static_alias_binding is not None
                            else None
                        ),
                        "graph_static_alias_consumer_va": (
                            int(graph_static_alias_binding.consumer.data_ptr())
                            if graph_static_alias_binding is not None
                            else None
                        ),
                        "graph_static_alias_requested_bytes": (
                            int(alias_allocation.requested_size)
                            if alias_allocation is not None
                            else None
                        ),
                        "graph_static_alias_aligned_bytes": (
                            int(alias_allocation.aligned_size)
                            if alias_allocation is not None
                            else None
                        ),
                        "graph_static_alias_backing_bytes": (
                            int(alias_allocation.backing_bytes)
                            if alias_allocation is not None
                            else None
                        ),
                        "graph_static_alias_granularity": (
                            int(alias_allocation.granularity)
                            if alias_allocation is not None
                            else None
                        ),
                        "graph_static_alias_upload_access": (
                            str(alias_allocation.upload_access)
                            if alias_allocation is not None
                            else None
                        ),
                        "graph_static_alias_consumer_access": (
                            str(alias_allocation.consumer_access)
                            if alias_allocation is not None
                            else None
                        ),
                        "graph_static_alias_total_backing_bytes": int(
                            _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
                        ),
                        "copy_bank_bytes": int(_CPU_OFFLOAD_STATIC_BANK_BYTES),
                        "copy_bank_tensor_bytes": int(bank_tensor_bytes),
                        "copy_bank_lane": bank_lane,
                        "copy_bank_key": (
                            {
                                "shape": list(bank_key[0]),
                                "stride": list(bank_key[1]),
                                "dtype": str(bank_key[2]),
                                "device_type": bank_key[3],
                                "device_index": bank_key[4],
                                "lane": bank_key[5],
                            }
                            if bank_key is not None
                            else None
                        ),
                    }
                )

        if exact_names_enabled:
            _CPU_OFFLOAD_EXACT_SEEN.add(full_parameter_name)
            _CPU_OFFLOAD_EXACT_PENDING.remove(full_parameter_name)

        _CPU_OFFLOAD_BYTES += accounted_bytes
        offloaded_parameters = True
        _CPU_OFFLOAD_ORDER += 1
        record = {
            "event": "cpu_offload_parameter",
            "timestamp": time.time(),
            "pid": os.getpid(),
            "module_prefix": module_prefix,
            "module_class": module.__class__.__name__,
            "parameter_name": param_name,
            "full_parameter_name": full_parameter_name,
            "param_index_in_module": param_index,
            "offload_order": _CPU_OFFLOAD_ORDER,
            "shape": list(p.data.size()),
            "stride": list(p.data.stride()),
            "dtype": str(p.data.dtype),
            "bytes": int(accounted_bytes),
            "parameter_bytes": int(bytes_before),
            "cumulative_offload_bytes": int(_CPU_OFFLOAD_BYTES),
            "max_offload_bytes": int(_CPU_OFFLOAD_MAX_BYTES),
            "pin_memory": bool(pin_memory),
            "uva_available": bool(uva_available),
            "uva_offloading": bool(offload_backend == "uva" and uva_offloading),
            "uvm_offloading": bool(
                _is_uvm_cpu_offload_backend(offload_backend) or uvm_original_alias
            ),
            "device_before": str(device),
            "device_after": str(p.data.device),
        }
        record.update(manifest_extra)
        _append_cpu_offload_manifest(record)

    uvm_remote_only_direct_forward = bool(
        offloaded_parameters
        and not direct_cuda_parameter
        and exact_names_enabled
        and _CPU_OFFLOAD_GRAPH_STATIC_ALIAS
        and _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND == "uvm_original_va"
        and _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE == "remote_only"
        and _strict_env_flag("VLLM_UVM_REMOTE_ONLY_DIRECT_FORWARD", False)
    )
    if uvm_remote_only_direct_forward:
        state_items = list(module.state_dict().items())
        state_keys = {key for key, _ in state_items}
        missing_local = exact_local_parameter_names - state_keys
        if missing_local:
            raise RuntimeError(
                "UVM remote-only direct forward found missing selected "
                f"parameters: {sorted(missing_local)!r}"
            )
        for key, value in state_items:
            binding = graph_static_alias_bindings.get(key)
            if binding is None:
                if value.device.type == "cpu":
                    raise RuntimeError(
                        "UVM remote-only direct forward found an unregistered "
                        f"CPU state tensor: {key!r}"
                    )
                continue
            if not isinstance(binding, _UvmOriginalAliasBinding):
                raise RuntimeError(
                    "UVM remote-only direct forward received a non-UVM "
                    f"binding for {key!r}"
                )
            if value.device.type != "cuda" or int(value.data_ptr()) != int(
                binding.consumer.data_ptr()
            ):
                raise RuntimeError(
                    "UVM remote-only direct forward selected Parameter changed "
                    f"device or VA during initialization: {binding.full_parameter_name!r}"
                )
        _append_cpu_offload_manifest(
            {
                "event": "cpu_offload_uvm_original_va_direct_forward",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "module_prefix": module_prefix,
                "module_class": module.__class__.__name__,
                "phase_mode": _CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE,
                "module_forward_wrapped": False,
                "selected_parameter_count": len(graph_static_alias_bindings),
                "selected_parameter_bytes": sum(
                    int(binding.requested_bytes)
                    for binding in graph_static_alias_bindings.values()
                    if isinstance(binding, _UvmOriginalAliasBinding)
                ),
            }
        )

    if (
        offloaded_parameters
        and not direct_cuda_parameter
        and not uvm_remote_only_direct_forward
    ):
        original_forward = module.forward
        copy_forward_index = 0
        copy_trace = _env_flag("VLLM_CPU_OFFLOAD_COPY_TRACE", False)
        copy_sync = _env_flag("VLLM_CPU_OFFLOAD_COPY_SYNC", False)
        copy_static_bank = bool(exact_names_enabled and _CPU_OFFLOAD_STATIC_BANK)
        graph_static_alias = bool(
            exact_names_enabled and _CPU_OFFLOAD_GRAPH_STATIC_ALIAS
        )
        uvm_original_graph_alias = bool(
            graph_static_alias
            and _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND == "uvm_original_va"
        )
        uvm_original_bindings: tuple[_UvmOriginalAliasBinding, ...] = ()
        if uvm_original_graph_alias:
            state_items = list(module.state_dict().items())
            state_keys = {key for key, _ in state_items}
            missing_local = exact_local_parameter_names - state_keys
            if missing_local:
                raise RuntimeError(
                    "Exact-name original-VA parameters disappeared from "
                    f"module.state_dict(): {sorted(missing_local)!r}"
                )
            if set(graph_static_alias_bindings) != exact_local_parameter_names:
                raise RuntimeError(
                    "UVM original-VA cached bindings do not cover the selected "
                    "module parameters"
                )
            for key, value in state_items:
                binding = graph_static_alias_bindings.get(key)
                if binding is None:
                    if value.device.type == "cpu":
                        raise RuntimeError(
                            "Original-VA fast forward found an unregistered CPU "
                            f"state tensor: {key!r}"
                        )
                    continue
                if not isinstance(binding, _UvmOriginalAliasBinding):
                    raise RuntimeError(
                        "Original-VA fast forward received a non-UVM binding for "
                        f"{key!r}"
                    )
                if value.device.type != "cuda" or int(value.data_ptr()) != int(
                    binding.consumer.data_ptr()
                ):
                    raise RuntimeError(
                        "Original-VA selected parameter changed device or VA "
                        f"during initialization: {key!r}"
                    )
            uvm_original_bindings = tuple(
                binding
                for _, binding in sorted(graph_static_alias_bindings.items())
                if isinstance(binding, _UvmOriginalAliasBinding)
            )
            if len(uvm_original_bindings) != len(exact_local_parameter_names):
                raise RuntimeError(
                    "Original-VA fast forward could not freeze every selected "
                    "module binding"
                )

        def forward(*args, **kwargs):
            global _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS
            global _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT
            global _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES
            nonlocal copy_forward_index
            module.forward = original_forward
            copy_forward_index += 1
            graph_forward_call_index: int | None = None
            graph_capture_active: bool | None = None
            graph_capture_phase = False
            stream_id: int | None = None
            copy_bytes = 0
            copy_tensor_count = 0
            graph_static_alias_actual_h2d_bytes = 0
            graph_static_alias_actual_h2d_tensor_count = 0
            copy_cuda_ms: float | None = None
            copy_submit_us: float | None = None
            copy_ok = False
            try:
                if exact_names_enabled:
                    _require_all_copy_exact_names_seen()
                    _validate_copy_exact_context(
                        device,
                        check_stream=(copy_static_bank or graph_static_alias),
                    )
                if graph_static_alias:
                    _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS += 1
                    graph_forward_call_index = int(
                        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS
                    )
                    graph_capture_active = bool(
                        torch.cuda.is_current_stream_capturing()
                    )
                    graph_capture_phase = bool(
                        _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE
                    )
                    stream_id = int(torch.cuda.current_stream(device).cuda_stream)

                if uvm_original_graph_alias:
                    lifecycle_state = _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
                    if lifecycle_state == "sealed":
                        if _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE:
                            raise RuntimeError(
                                "UVM original-VA sealed prefill retained an active "
                                "remote map"
                            )
                    elif lifecycle_state not in ("remote", "remote_parked") or not (
                        _CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE
                    ):
                        raise RuntimeError(
                            "UVM original-VA direct forward requires a sealed "
                            "prefill topology or an audited active remote-decode "
                            f"topology; current state is {lifecycle_state!r}, "
                            "remote_map_active="
                            f"{_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE!r}"
                        )

                    # Validate every cached binding before submitting any copy so
                    # a stale Parameter cannot leave a partially updated bank.
                    for binding in uvm_original_bindings:
                        current = binding.parameter.data
                        if current.device.type != "cuda" or int(
                            current.data_ptr()
                        ) != int(binding.consumer.data_ptr()):
                            raise RuntimeError(
                                "UVM original-VA selected Parameter changed "
                                "device or VA after initialization: "
                                f"{binding.full_parameter_name!r}"
                            )
                        if lifecycle_state == "sealed" and binding.source is None:
                            raise RuntimeError(
                                "UVM original-VA alias was sealed without a "
                                "pinned host source for "
                                f"{binding.full_parameter_name!r}"
                            )
                    # A request-boundary plan rebuild mutates these long-lived
                    # binding objects in place.  Do not freeze the selected
                    # byte count in the module closure created at model load.
                    copy_bytes = sum(
                        int(binding.requested_bytes)
                        for binding in uvm_original_bindings
                    )
                    copy_tensor_count = len(uvm_original_bindings)
                else:
                    state_items = list(module.state_dict().items())
                    if exact_names_enabled:
                        state_keys = {key for key, _ in state_items}
                        missing_local = exact_local_parameter_names - state_keys
                        if missing_local:
                            raise RuntimeError(
                                "Exact-name CPU offload parameters disappeared "
                                "from module.state_dict(): "
                                f"{sorted(missing_local)!r}"
                            )
                    for key, value in state_items:
                        if (
                            exact_names_enabled
                            and key in exact_local_parameter_names
                            and value.device.type != "cpu"
                        ):
                            raise RuntimeError(
                                "Exact-name static-alias parameter is on the "
                                f"wrong device: {key!r} is on {value.device}"
                            )
                        if (
                            copy_static_bank or graph_static_alias
                        ) and key in exact_local_parameter_names:
                            copy_bytes += value.numel() * value.element_size()
                            copy_tensor_count += 1
                        elif (
                            copy_static_bank or graph_static_alias
                        ) and value.device.type == "cpu":
                            raise RuntimeError(
                                "Static bank/alias forward found an unregistered "
                                f"CPU state tensor: {key!r}"
                            )
                        elif (
                            not copy_static_bank
                            and not graph_static_alias
                            and value.device.type == "cpu"
                        ):
                            copy_bytes += value.numel() * value.element_size()
                            copy_tensor_count += 1

                start_event = None
                end_event = None
                if copy_sync:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record(torch.cuda.current_stream(device))

                copy_submit_start_ns = time.perf_counter_ns()
                if uvm_original_graph_alias:
                    if lifecycle_state == "sealed":
                        for binding in uvm_original_bindings:
                            assert binding.source is not None
                            _uvm_original_alias_binding_upload_view(binding).copy_(
                                _uvm_original_alias_binding_source_view(binding),
                                non_blocking=True,
                            )
                            graph_static_alias_actual_h2d_bytes += int(
                                binding.requested_bytes
                            )
                            graph_static_alias_actual_h2d_tensor_count += 1
                elif graph_static_alias:
                    device_state = {}
                    for key, value in state_items:
                        binding = graph_static_alias_bindings.get(key)
                        if binding is None:
                            device_state[key] = value
                            continue
                        binding.arena.upload.copy_(value, non_blocking=True)
                        graph_static_alias_actual_h2d_bytes += int(
                            value.numel() * value.element_size()
                        )
                        graph_static_alias_actual_h2d_tensor_count += 1
                        device_state[key] = binding.consumer
                elif copy_static_bank:
                    device_state = {}
                    for key, value in state_items:
                        bank_key = static_bank_keys.get(key)
                        if bank_key is None:
                            device_state[key] = value
                            continue
                        bank = _CPU_OFFLOAD_STATIC_BANKS.get(bank_key)
                        if bank is None:
                            raise RuntimeError(
                                "Static CPU-offload bank was reset or released "
                                f"before forward for {key!r}"
                            )
                        bank.copy_(value, non_blocking=True)
                        device_state[key] = bank
                else:
                    device_state = {
                        # GPU values are a no-op. CPU values allocate a
                        # temporary CUDA tensor and enqueue H2D on the current
                        # stream. Keep this native path unchanged when no exact
                        # name file is configured.
                        key: value.to(device, non_blocking=True)
                        for key, value in state_items
                    }
                copy_submit_us = (time.perf_counter_ns() - copy_submit_start_ns) / 1e3

                if copy_sync:
                    assert start_event is not None and end_event is not None
                    end_event.record(torch.cuda.current_stream(device))
                    end_event.synchronize()
                    copy_cuda_ms = float(start_event.elapsed_time(end_event))

                if uvm_original_graph_alias:
                    output = original_forward(*args, **kwargs)
                else:
                    output = functional_call(
                        module,
                        device_state,
                        args=args,
                        kwargs=kwargs,
                        # Some architectures expose the same Parameter under
                        # multiple state-dict names (for example DeepSeek-V2 MLA).
                        # device_state already preserves each alias explicitly;
                        # asking functional_call to re-tie them rejects the map.
                        tie_weights=False,
                    )
                copy_ok = True
                return output
            finally:
                module.forward = forward
                if (
                    uvm_original_graph_alias
                    and graph_capture_phase
                    and graph_capture_active
                ):
                    _CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT += 1
                    _CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES += int(
                        graph_static_alias_actual_h2d_bytes
                    )
                if copy_trace:
                    _append_cpu_offload_manifest(
                        {
                            "event": "cpu_offload_copy_forward",
                            "timestamp": time.time(),
                            "pid": os.getpid(),
                            "module_prefix": module_prefix,
                            "module_class": module.__class__.__name__,
                            "forward_index": copy_forward_index,
                            "copy_bytes": int(copy_bytes),
                            "copy_tensor_count": int(copy_tensor_count),
                            "copy_submit_us": copy_submit_us,
                            "copy_cuda_ms": copy_cuda_ms,
                            "copy_sync": bool(copy_sync),
                            "copy_exact_names": bool(exact_names_enabled),
                            "copy_exact_name_count": len(exact_local_parameter_names),
                            "copy_exact_names_file": (
                                _CPU_OFFLOAD_EXACT_NAMES_FILE
                                if exact_names_enabled
                                else None
                            ),
                            "copy_static_bank": bool(copy_static_bank),
                            "graph_static_alias": bool(graph_static_alias),
                            "graph_static_alias_backend": (
                                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND
                                if graph_static_alias
                                else None
                            ),
                            "uvm_original_va_direct_forward": bool(
                                uvm_original_graph_alias
                            ),
                            "graph_static_alias_lifecycle_state": (
                                _CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE
                                if uvm_original_graph_alias
                                else None
                            ),
                            "graph_static_alias_actual_h2d_bytes": int(
                                graph_static_alias_actual_h2d_bytes
                            ),
                            "graph_static_alias_actual_h2d_tensor_count": int(
                                graph_static_alias_actual_h2d_tensor_count
                            ),
                            "graph_static_alias_copy_submitted": bool(
                                graph_static_alias_actual_h2d_tensor_count
                            ),
                            "graph_static_alias_forward_call_index": (
                                graph_forward_call_index
                            ),
                            "graph_static_alias_capture_active": (graph_capture_active),
                            "graph_static_alias_capture_phase": bool(
                                graph_capture_phase
                            ),
                            "graph_static_alias_stream_id": stream_id,
                            "graph_static_alias_backing_bytes": int(
                                _CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
                                if graph_static_alias
                                else 0
                            ),
                            "graph_static_alias_arena_count": int(
                                len(_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS)
                                + len(_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS)
                                if graph_static_alias
                                else 0
                            ),
                            "copy_bank_bytes": int(
                                _CPU_OFFLOAD_STATIC_BANK_BYTES
                                if copy_static_bank
                                else 0
                            ),
                            "copy_bank_tensor_count": int(
                                len(_CPU_OFFLOAD_STATIC_BANKS)
                                if copy_static_bank
                                else 0
                            ),
                            "ok": bool(copy_ok),
                        }
                    )

        module.forward = forward

    if offloaded_parameters and offload_backend in (
        "isolated_uvm_hybrid_remote",
        "isolated_uvm_hybrid_remote_noreadmostly",
        "isolated_uvm_hybrid_remote_noreadmostly_noaccessedby",
    ):
        from vllm.device_allocator.uvm import (
            driver_pause_ranges_bg_promotion,
            driver_resume_ranges_bg_promotion,
            driver_start_ranges_bg_promotion,
            round_ranges_to_pages,
        )

        original_forward = module.forward
        bg_started = False
        hybrid_remote_gate_ranges = round_ranges_to_pages(hybrid_remote_ranges)
        hybrid_start_mode = _hybrid_start_mode()
        hybrid_gate_enabled = _env_flag("VLLM_UVM_HYBRID_GATE_ENABLE", True)

        if hybrid_start_mode in ("eager", "immediate", "after_prepare"):
            start_result = driver_start_ranges_bg_promotion(
                hybrid_remote_gate_ranges,
                device=int(hybrid_remote_device_index),
            )
            bg_started = int(start_result[0]) > 0
            _append_cpu_offload_manifest(
                {
                    "event": "cpu_offload_hybrid_remote_eager_start",
                    "timestamp": time.time(),
                    "pid": os.getpid(),
                    "module_prefix": module_prefix,
                    "module_class": module.__class__.__name__,
                    "range_count": len(hybrid_remote_gate_ranges),
                    "bytes": int(sum(size for _, size in hybrid_remote_gate_ranges)),
                    "start_success": int(start_result[0]),
                    "start_failure": int(start_result[1]),
                    "start_rm_status": int(start_result[2]),
                    "start_ioctl_ret": int(start_result[3]),
                    "bg_started": bool(bg_started),
                    "start_mode": hybrid_start_mode,
                }
            )

        if not hybrid_gate_enabled:
            return module

        def hybrid_forward(*args, **kwargs):
            nonlocal bg_started
            pause_result = driver_pause_ranges_bg_promotion(
                hybrid_remote_gate_ranges,
                device=int(hybrid_remote_device_index),
            )
            ok = False
            try:
                output = original_forward(*args, **kwargs)
                ok = True
                return output
            finally:
                resume_result = driver_resume_ranges_bg_promotion(
                    hybrid_remote_gate_ranges,
                    device=int(hybrid_remote_device_index),
                )
                start_result = (0, 0, 0, 0)
                if ok and not bg_started:
                    start_result = driver_start_ranges_bg_promotion(
                        hybrid_remote_gate_ranges,
                        device=int(hybrid_remote_device_index),
                    )
                    bg_started = int(start_result[0]) > 0
                _append_cpu_offload_manifest(
                    {
                        "event": "cpu_offload_hybrid_remote_gate",
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                        "module_prefix": module_prefix,
                        "module_class": module.__class__.__name__,
                        "range_count": len(hybrid_remote_gate_ranges),
                        "bytes": int(
                            sum(size for _, size in hybrid_remote_gate_ranges)
                        ),
                        "pause_success": int(pause_result[0]),
                        "pause_failure": int(pause_result[1]),
                        "pause_rm_status": int(pause_result[2]),
                        "pause_ioctl_ret": int(pause_result[3]),
                        "resume_success": int(resume_result[0]),
                        "resume_failure": int(resume_result[1]),
                        "resume_rm_status": int(resume_result[2]),
                        "resume_ioctl_ret": int(resume_result[3]),
                        "start_success": int(start_result[0]),
                        "start_failure": int(start_result[1]),
                        "start_rm_status": int(start_result[2]),
                        "start_ioctl_ret": int(start_result[3]),
                        "bg_started": bool(bg_started),
                        "start_mode": hybrid_start_mode,
                        "gate_enabled": bool(hybrid_gate_enabled),
                    }
                )

        module.forward = hybrid_forward

    return module


def make_layers(
    num_hidden_layers: int,
    layer_fn: LayerFn,
    prefix: str,
) -> tuple[int, int, torch.nn.ModuleList]:
    """Make a list of layers with the given layer function, taking
    pipeline parallelism into account.
    """
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.distributed.utils import get_pp_indices

    start_layer, end_layer = get_pp_indices(
        num_hidden_layers, get_pp_group().rank_in_group, get_pp_group().world_size
    )
    modules = torch.nn.ModuleList(
        [PPMissingLayer() for _ in range(start_layer)]
        + [
            maybe_offload_to_cpu(
                layer_fn(prefix=f"{prefix}.{idx}"),
                module_prefix=f"{prefix}.{idx}",
            )
            for idx in range(start_layer, end_layer)
        ]
        + [PPMissingLayer() for _ in range(end_layer, num_hidden_layers)]
    )
    _require_all_copy_exact_names_seen()
    return start_layer, end_layer, modules


# NOTE: don't use lru_cache here because it can prevent garbage collection
_model_to_pp_missing_layer_names: dict[int, list[str]] = {}


def get_pp_missing_layer_names(model: torch.nn.Module) -> list[str]:
    """Get the names of the missing layers in a pipeline parallel model."""
    model_id = id(model)
    if model_id in _model_to_pp_missing_layer_names:
        return _model_to_pp_missing_layer_names[model_id]

    missing_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, PPMissingLayer):
            # NOTE: the trailing dot is used to match the prefix of the layer.
            # without the dot, we could match a layer that is not missing,
            # e.g., 'encoder.layer.1' would match 'encoder.layer.11'
            missing_layer_names.append(name + ".")
    _model_to_pp_missing_layer_names[model_id] = missing_layer_names

    return missing_layer_names


def is_pp_missing_parameter(name: str, model: torch.nn.Module) -> bool:
    """Check if a parameter is missing in a pipeline parallel model."""
    if isinstance(model, PPMissingLayer):
        return True

    return any(
        name.startswith(missing_layer_name)
        for missing_layer_name in get_pp_missing_layer_names(model)
    )


def make_empty_intermediate_tensors_factory(keys: list[str], hidden_size: int):
    def make_empty_intermediate_tensors(
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                key: torch.zeros((batch_size, hidden_size), dtype=dtype, device=device)
                for key in keys
            }
        )

    return make_empty_intermediate_tensors


def maybe_prefix(prefix: str, name: str) -> str:
    """Add a prefix to a name if the prefix is non-empty.

    Args:
        prefix: The prefix to add. If empty, no prefix will be added.
        name: The name to potentially prefix.

    Returns:
        The string "prefix.name" if prefix was non-empty, otherwise just "name".
    """
    return name if not prefix else f"{prefix}.{name}"


def get_draft_quant_config(
    vllm_config: VllmConfig,
) -> QuantizationConfig | None:
    """Get quantization config for Draft models.

    Draft models should use their own quantization config instead of the verifier/target
    model's config. This helper retrieves the draft model's quantization config.

    Args:
        vllm_config: The vLLM configuration object.

    Returns:
        The draft model's config if available, None otherwise.
    """
    draft_model_config = vllm_config.speculative_config.draft_model_config
    draft_load_config = vllm_config.load_config

    return (
        VllmConfig.get_quantization_config(draft_model_config, draft_load_config)
        if draft_model_config
        else None
    )


def extract_layer_index(layer_name: str, num_attn_module: int = 1) -> int:
    """
    Extract the layer index from the module name.
    Examples:
    - "encoder.layers.0" -> 0
    - "encoder.layers.1.self_attn" -> 1
    - "2.self_attn" -> 2
    - "model.encoder.layers.0.sub.1" -> ValueError if num_attn_module == 1
    """
    subnames = layer_name.split(".")
    int_vals: list[int] = []
    for subname in subnames:
        try:
            int_vals.append(int(subname))
        except ValueError:
            continue
    if num_attn_module == 1 or "attn" not in layer_name:
        assert len(int_vals) == 1, (
            f"layer name {layer_name} should only contain one integer"
        )

        return int_vals[0]
    else:
        assert len(int_vals) <= 2, (
            f"layer name {layer_name} should contain most two integers"
        )
        layer_index = (
            int_vals[0] * num_attn_module + int_vals[1]
            if len(int_vals) == 2
            else int_vals[0]
        )
        return layer_index


def cast_overflow_tensors(
    tensors: torch.Tensor,
    offset: float = 1000,
) -> torch.Tensor:
    if tensors.isinf().any() or tensors.isnan().any():
        clamp_value = torch.finfo(tensors.dtype).max - offset
        tensors = torch.clamp(tensors, min=-clamp_value, max=clamp_value)
    return tensors


def fast_topk(
    values: torch.Tensor, topk: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized topk implementation that uses torch.max for k=1 case.

    This function provides better performance for the common case of k=1
    by using torch.max instead of the more general torch.topk.

    Args:
        values: Input tensor to find top-k values from
        topk: Number of top values to return (k). Must be > 0.
        dim: Dimension along which to compute topk

    Returns:
        Tuple of (values, indices) where values are the top-k values
        and indices are their corresponding indices in the input tensor
    """
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        return torch.max(values, dim=dim, keepdim=True)
    else:
        # Use topk for efficiency with larger k values
        return torch.topk(values, topk, dim=dim)


# Chunk x along the num_tokens axis for sequence parallelism
# NOTE: This is wrapped in a torch custom op to work around the following issue:
# The output tensor can have a sequence length 0 at small input sequence lengths
# even though we explicitly pad to avoid this.
def sequence_parallel_chunk(x: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.sequence_parallel_chunk_impl(x)


def sequence_parallel_chunk_impl(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()

    # all_gather needs the sequence length to be divisible by tp_size
    seq_len = x.size(0)
    remainder = seq_len % tp_size
    if remainder != 0:
        pad_len = tp_size - remainder
        y = nn.functional.pad(x, (0, 0, 0, pad_len))
    else:
        y = x

    chunk = y.shape[0] // tp_size
    start = tp_rank * chunk
    return torch.narrow(y, 0, start, chunk)


def sequence_parallel_chunk_impl_fake(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    seq_len = cdiv(x.size(0), tp_size)
    shape = list(x.shape)
    shape[0] = seq_len
    out = torch.empty(shape, dtype=x.dtype, device=x.device)
    return out


direct_register_custom_op(
    op_name="sequence_parallel_chunk_impl",
    op_func=sequence_parallel_chunk_impl,
    fake_impl=sequence_parallel_chunk_impl_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


def process_eagle_weight(
    model: nn.Module,
    name: str,
) -> None:
    """
    Update EAGLE model flags based on loaded weight name.
    This should be called during weight loading to detect if a model
    has its own lm_head or embed_tokens weight.
    Args:
        model: The model instance (must support EAGLE)
        name: The name of the weight to process
    """
    if not supports_any_eagle(model):
        return

    # To prevent overriding with target model's layers
    if "lm_head" in name:
        model.has_own_lm_head = True
    if "embed_tokens" in name:
        model.has_own_embed_tokens = True
