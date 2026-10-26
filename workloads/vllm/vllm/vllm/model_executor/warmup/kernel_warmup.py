# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Warmup kernels used during model execution.
This is useful specifically for JIT'ed kernels as we don't want JIT'ing to
happen during model execution.
"""

import os
from typing import TYPE_CHECKING

import torch

import vllm.envs as envs
from vllm.device_allocator.uvm import uvm_allocation_phase
from vllm.logger import init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import deep_gemm_warmup
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def kernel_warmup(worker: "Worker"):
    # Deep GEMM warmup
    do_deep_gemm_warmup = (
        envs.VLLM_USE_DEEP_GEMM
        and is_deep_gemm_supported()
        and envs.VLLM_DEEP_GEMM_WARMUP != "skip"
    )
    if do_deep_gemm_warmup:
        model = worker.get_model()
        max_tokens = worker.scheduler_config.max_num_batched_tokens
        with uvm_allocation_phase("kernel_warmup:deep_gemm"):
            deep_gemm_warmup(model, max_tokens)

    # FlashInfer autotune for Hopper (SM 9.0) and Blackwell (SM 10.0) GPUs
    if has_flashinfer() and current_platform.has_device_capability(90):
        with uvm_allocation_phase("kernel_warmup:flashinfer_autotune"):
            flashinfer_autotune(worker.model_runner)

    # FlashInfer attention warmup
    # Only warmup if the model has FlashInfer attention groups
    # and is not a pooling model
    def _is_flashinfer_backend(backend):
        try:
            return backend.get_name() == "FLASHINFER"
        except NotImplementedError:
            return False

    # NOTE: we add check for empty attn_groups to avoid errors when
    # deploying models such as E instances and encoder-only models.
    # As for those models, worker.model_runner.attn_groups is empty.
    # This change is made during EPD feature development.
    if (
        not worker.model_runner.is_pooling_model
        and worker.model_runner.attn_groups
        and all(
            _is_flashinfer_backend(group.backend)
            for groups in worker.model_runner.attn_groups
            for group in groups
        )
    ):
        logger.info("Warming up FlashInfer attention.")
        # Warmup with mixed batch containing both prefill and decode tokens
        # This is to warm up both prefill and decode attention kernels
        with uvm_allocation_phase("kernel_warmup:flashinfer_attention"):
            worker.model_runner._dummy_run(
                num_tokens=16,
                skip_eplb=True,
                is_profile=True,
                force_attention=True,
                create_mixed_batch=True,
            )


def flashinfer_autotune(runner: "GPUModelRunner") -> None:
    """
    Autotune FlashInfer operations.
    FlashInfer have many implementations for the same operation,
    autotuning runs benchmarks for each implementation and stores
    the results. The results are cached transparently and
    future calls to FlashInfer will use the best implementation.
    Without autotuning, FlashInfer will rely on heuristics, which may
    be significantly slower.
    """
    from vllm.utils.flashinfer import autotune

    with torch.inference_mode(), autotune():
        # We skip EPLB here since we don't want to record dummy metrics
        # When autotuning with number of tokens m, flashinfer will autotune
        # operations for all number of tokens up to m.
        # So we only need to run with the max number of tokens.
        max_tokens = int(runner.scheduler_config.max_num_batched_tokens)
        raw_limit = os.environ.get(
            "VLLM_FLASHINFER_AUTOTUNE_MAX_TOKENS", "0"
        ).strip()
        try:
            configured_limit = int(raw_limit or "0")
        except ValueError as exc:
            raise ValueError(
                "VLLM_FLASHINFER_AUTOTUNE_MAX_TOKENS must be an integer, "
                f"got {raw_limit!r}"
            ) from exc
        if configured_limit > 0:
            max_tokens = min(max_tokens, configured_limit)
            logger.info(
                "Clamped FlashInfer autotune dummy run to %d tokens "
                "(scheduler maximum %d)",
                max_tokens,
                runner.scheduler_config.max_num_batched_tokens,
            )
        runner._dummy_run(
            max_tokens,
            skip_eplb=True,
            is_profile=True,
        )
