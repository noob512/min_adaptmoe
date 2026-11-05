# SPDX-License-Identifier: Apache-2.0

import inspect
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.device_allocator import uvm as uvm_module
from vllm.device_allocator import uvm_pool_residency as residency_module
from vllm.device_allocator.uvm_region import PhysicalKVSpan
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.models import utils as offload_utils
from vllm.v1.core.uvm_kv_runtime_policy import UvmKvRuntimePolicy
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor
from vllm.v1.worker import gpu_model_runner as gpu_model_runner_module
from vllm.v1.worker import gpu_worker as gpu_worker_module
from vllm.v1.worker.gpu_model_runner import (
    GPUModelRunner,
    _build_pure_prefill_dummy_shape,
    _uvm_kv_tail_range_for_allocation,
)
from vllm.v1.worker.gpu_worker import Worker


@pytest.fixture(autouse=True)
def _isolated_slot_session_contract(monkeypatch: Any):
    monkeypatch.setattr(residency_module, "_SLOT_DIRECT_SESSION_CONTRACT", None)


def _mock_complete_slot_registry(monkeypatch: Any) -> None:
    va_block_bytes = 2 * 1024 * 1024
    base = 0x40000000
    monkeypatch.setattr(
        gpu_model_runner_module,
        "snapshot_uvm_region_registry",
        lambda region_ids: {
            "ok": True,
            "regions": [
                {
                    "region_id": str(region_id),
                    "base": base + index * va_block_bytes,
                    "length": va_block_bytes,
                }
                for index, region_id in enumerate(region_ids)
            ],
        },
    )


def _enable_registered_slot_plan(monkeypatch: Any) -> None:
    for name in (
        "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE",
        "VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD",
        "VLLM_UVM_SLOT_DIRECT_INIT_BANK",
        "VLLM_UVM_SLOT_DIRECT_REUSE_BANK",
        "VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES",
        "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION",
        "VLLM_UVM_SLOT_DIRECT_SHARED_PUSH",
        "VLLM_UVM_SLOT_DIRECT_64K_PTE",
        "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "slot_restore")


def _exclusive_residency_result(
    ranges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    location: str,
) -> dict[str, int | bool]:
    expected_bytes = sum(size for _, size in ranges)
    return {
        "ok": True,
        "status": len(ranges),
        "queried_bytes": expected_bytes,
        "gpu_bytes": expected_bytes if location == "gpu" else 0,
        "cpu_bytes": expected_bytes if location == "cpu" else 0,
        "any_bytes": expected_bytes,
        "unknown_bytes": 0,
    }


def test_uvm_kv_tail_range_respects_block_boundaries() -> None:
    page_size = 4096
    ptr = 0x100000

    assert _uvm_kv_tail_range_for_allocation(
        ptr=ptr,
        size_bytes=11 * page_size,
        page_size=page_size,
        total_blocks=10,
        target_blocks=3,
    ) == (ptr + 4 * page_size, 7 * page_size, 4, 10)

    assert (
        _uvm_kv_tail_range_for_allocation(
            ptr=ptr,
            size_bytes=11 * page_size,
            page_size=page_size,
            total_blocks=10,
            target_blocks=10,
        )
        is None
    )


def test_pure_prefill_dummy_shape_uses_actual_request_lengths() -> None:
    request_lengths, max_query_len = _build_pure_prefill_dummy_shape(
        num_tokens=16384,
        num_reqs=16,
        max_model_len=8320,
    )

    assert request_lengths == [1024] * 16
    assert sum(request_lengths) == 16384
    assert max_query_len == 1024


def test_pure_prefill_dummy_shape_rejects_oversized_request() -> None:
    with pytest.raises(ValueError, match="exceeds max_model_len"):
        _build_pure_prefill_dummy_shape(
            num_tokens=16384,
            num_reqs=1,
            max_model_len=8320,
        )


def test_dual_graph_capture_repairs_hot_residency_before_prefill_topology() -> None:
    source = inspect.getsource(GPUModelRunner.capture_model)
    ordinary_path = source.index("if not capture_prefill_before_hot_restore:")
    repair_call = "reconcile_cpu_offload_uvm_original_va_remote_graph_residency("
    activate_call = "activate_cpu_offload_uvm_original_va_prefill_topology()"

    assert source.count(repair_call) == 1
    assert source.index(repair_call, ordinary_path) < source.index(
        activate_call, ordinary_path
    )


def test_capture_before_hot_restore_keeps_final_prefill_graph_pool() -> None:
    source = inspect.getsource(GPUModelRunner.capture_model)
    handoff = source.index("finish_all_remote_profile=True")
    capture = source.index("capture_pure_prefill_graphs()", handoff)
    restore = source.index(
        "reconcile_cpu_offload_uvm_original_va_sealed_hot_residency(", capture
    )

    assert handoff < capture < restore
    assert "preserve_allocator_cache=True" in source[restore : restore + 200]


def test_prefill_reserve_warmup_requests_valid_pure_prefill_shape(
    monkeypatch: Any,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeModelRunner:
        max_model_len = 8320

        def _dummy_run(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append(kwargs)
            return torch.empty(0), torch.empty(0)

    monkeypatch.setenv(
        "VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS", "16384"
    )
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_phase_mode",
        lambda: "prefill_graph_remote_decode",
    )
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_profile_all_remote_active",
        lambda: True,
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "uvm_allocation_phase",
        lambda _phase: nullcontext(),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (4096, 8192))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 1024)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 512)

    worker = object.__new__(Worker)
    worker.device = torch.device("cpu")
    worker.scheduler_config = SimpleNamespace(
        max_num_batched_tokens=16384,
        max_num_seqs=16,
    )
    worker.model_runner = FakeModelRunner()

    worker._warm_up_uvm_prefill_allocator_reserve()

    assert calls == [
        {
            "num_tokens": 16384,
            "cudagraph_runtime_mode": gpu_worker_module.CUDAGraphMode.NONE,
            "force_attention": True,
            "allow_microbatching": False,
            "skip_eplb": True,
            "remove_lora": False,
            "pure_prefill_num_reqs": 16,
        }
    ]


def test_post_warmup_kv_rebalance_returns_control_fields_only(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._collect_uvm_post_warmup_kv_tail_ranges = (  # type: ignore[method-assign]
        lambda _target: (
            [(0x100000, 0x2000)],
            {"ok": True, "reason": "ranges_collected", "target_blocks": 4},
        )
    )
    monkeypatch.setenv("VLLM_UVM_POST_WARMUP_KV_OFFLOAD_EXECUTOR", "cpp_batch")
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_offload_ranges_to_cpu_with_executor",
        lambda ranges, *, sync, executor: (
            len(ranges),
            executor,
            {"diagnostic": "unused"},
            {"diagnostic": "unused"},
        ),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "driver_batch_query_ranges_residency_bytes",
        lambda ranges, *, device, wait_for_trackers: _exclusive_residency_result(
            ranges, "cpu"
        ),
    )

    result = runner.rebalance_uvm_kv_cache_after_warmup(4)

    assert result == {
        "ok": True,
        "reason": "ranges_collected",
        "target_blocks": 4,
        "offload_complete": True,
    }


@pytest.mark.parametrize("inner_raises", [False, True])
def test_nested_uvm_pool_only_changes_prefetch(
    monkeypatch: Any, inner_raises: bool,
) -> None:
    events: list[object] = []
    selected = False

    @contextmanager
    def select_pool(pool):
        nonlocal selected
        assert not selected, "PyTorch rejects nested recording of one pool"
        selected = True
        events.append("enter")
        try:
            yield
        finally:
            selected = False
            events.append("exit")

    monkeypatch.setattr(uvm_module, "_uvm_local_prefetch_stack", [])
    monkeypatch.setattr(uvm_module, "_uvm_local_allocator", object())
    monkeypatch.setattr(uvm_module, "_uvm_local_mem_pool", object())
    monkeypatch.setattr(uvm_module, "_uvm_lib", SimpleNamespace(
        uvm_set_prefetch=lambda flag: events.append(flag)))
    monkeypatch.setattr(uvm_module, "ensure_uvm_runtime_library", lambda: None)
    monkeypatch.setattr(uvm_module, "find_uvm_allocator_library", lambda: "test.so")
    monkeypatch.setattr(uvm_module.torch.cuda.memory, "use_mem_pool", select_pool)
    with uvm_module.uvm_local_memory_pool(enable_prefetch=True):
        try:
            with uvm_module.uvm_local_memory_pool(enable_prefetch=False):
                assert selected
                assert not uvm_module._uvm_local_prefetch_enabled()
                if inner_raises:
                    raise ValueError("test")
        except ValueError:
            assert inner_raises
        assert uvm_module._uvm_local_prefetch_enabled()
    assert events == [1, "enter", 0, 1, "exit", 0]
    assert not uvm_module._uvm_local_prefetch_stack


@pytest.mark.parametrize(
    "managed,lazy,sleep,expected",
    [(True, True, False, [True, False]),
     (True, False, False, [True, True]),
     (False, True, False, [None, None]),
     (True, True, True, [None, None])],
)
def test_lazy_kv_pool_keeps_attention_workspace_backed(
    monkeypatch: Any, managed: bool, lazy: bool, sleep: bool, expected: list,
) -> None:
    prefetch_stack: list[bool] = []
    observed: list[bool | None] = []

    @contextmanager
    def pool(*, enable_prefetch: bool):
        prefetch_stack.append(enable_prefetch)
        try:
            yield
        finally:
            prefetch_stack.pop()

    monkeypatch.setenv("VLLM_UVM_KV_CACHE_MANAGED", str(int(managed)))
    monkeypatch.setenv("VLLM_UVM_KV_CACHE_LAZY_RESERVE", str(int(lazy)))
    monkeypatch.setattr(uvm_module, "uvm_local_memory_pool", pool)
    monkeypatch.setattr(gpu_model_runner_module, "uvm_allocation_phase",
                        lambda *_: nullcontext())
    monkeypatch.setattr(gpu_model_runner_module, "has_kv_transfer_group", lambda: False)
    runner = object.__new__(GPUModelRunner)
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(enable_sleep_mode=sleep))
    runner.speculative_config = None
    runner.dcp_world_size = 1
    runner._prepare_kernel_block_sizes = lambda *_: []
    for name in ("may_add_encoder_only_layers_to_kv_cache_config",
                 "maybe_add_kv_sharing_layers_to_kv_cache_groups",
                 "initialize_attn_backend", "may_reinitialize_input_batch"):
        setattr(runner, name, lambda *_: None)

    def allocation(*_args):
        observed.append(prefetch_stack[-1] if prefetch_stack else None)
        return {}

    runner.initialize_metadata_builders = allocation
    runner.initialize_kv_cache_tensors = allocation
    worker = object.__new__(Worker)
    worker.vllm_config = runner.vllm_config
    with worker._maybe_get_kv_cache_uvm_memory_pool_context():
        runner.initialize_kv_cache(SimpleNamespace())
    assert observed == expected
    assert not prefetch_stack


def test_lazy_kv_reserve_touches_only_null_block(monkeypatch: Any) -> None:
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.runner_only_attn_layers = set()
    config = SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(size=64, shared_by=["layer.0"])],
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["layer.0"],
                kv_cache_spec=SimpleNamespace(page_size_bytes=16),
            )
        ],
    )
    real_empty = torch.empty

    def filled_empty(size: int, *, dtype: torch.dtype, device: torch.device):
        return real_empty(size, dtype=dtype, device=device).fill_(7)

    monkeypatch.setenv("VLLM_UVM_KV_CACHE_LAZY_RESERVE", "1")
    monkeypatch.setattr(gpu_model_runner_module.torch, "empty", filled_empty)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_advise_ranges_preferred_location",
        lambda ranges, _device: len(ranges),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_prefetch_ranges_to_device",
        lambda ranges, _device: len(ranges),
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "current_device",
        lambda: 0,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda _device: None,
    )

    tensors = runner._allocate_kv_cache_tensors(config)

    tensor = tensors["layer.0"]
    assert torch.count_nonzero(tensor[:16]).item() == 0
    assert torch.all(tensor[16:] == 7)


def test_lazy_post_warmup_kv_rebalance_converges_head_and_tail(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner.kv_cache_config = SimpleNamespace(num_blocks=11)
    runner._collect_uvm_post_warmup_kv_tail_ranges = (  # type: ignore[method-assign]
        lambda _target: (
            [(0x200000, 0x6000)],
            {"ok": True, "reason": "ranges_collected", "target_blocks": 4},
        )
    )
    events: list[tuple[str, object]] = []
    monkeypatch.setenv("VLLM_UVM_KV_CACHE_LAZY_RESERVE", "1")
    monkeypatch.setenv("VLLM_UVM_POST_WARMUP_KV_OFFLOAD_EXECUTOR", "driver_batch")
    monkeypatch.setattr(
        gpu_model_runner_module,
        "collect_uvm_kv_cache_physical_ranges_for_blocks",
        lambda block_ids: events.append(("collect", tuple(block_ids)))
        or [(0x100000, 0x4000)],
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_offload_ranges_to_cpu_with_executor",
        lambda ranges, *, sync, executor: events.append(
            ("offload", (tuple(ranges), sync, executor))
        )
        or (len(ranges), executor, 0, 0),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_advise_ranges_preferred_location",
        lambda ranges, device: events.append(("advise", (tuple(ranges), device)))
        or len(ranges),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda ranges, device, *, executor: events.append(
            ("prefetch", (tuple(ranges), device, executor))
        )
        or (len(ranges), executor, 0),
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "current_device",
        lambda: 0,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda device: events.append(("sync", device)),
    )

    def query_residency(
        ranges: list[tuple[int, int]],
        *,
        device: int,
        wait_for_trackers: bool,
    ) -> dict[str, int | bool]:
        range_tuple = tuple(ranges)
        events.append(("query", (range_tuple, device, wait_for_trackers)))
        location = "gpu" if range_tuple == ((0x100000, 0x4000),) else "cpu"
        return _exclusive_residency_result(ranges, location)

    monkeypatch.setattr(
        gpu_model_runner_module,
        "driver_batch_query_ranges_residency_bytes",
        query_residency,
    )

    result = runner.rebalance_uvm_kv_cache_after_warmup(4)

    assert result["offload_complete"] is True
    assert events == [
        ("collect", tuple(range(0, 11))),
        ("offload", (((0x100000, 0x4000),), True, "driver_batch")),
        ("collect", (0, 1, 2, 3, 4)),
        ("advise", (((0x100000, 0x4000),), 0)),
        ("prefetch", (((0x100000, 0x4000),), 0, "driver_batch")),
        ("sync", 0),
        ("query", (((0x100000, 0x4000),), 0, False)),
        ("query", (((0x200000, 0x6000),), 0, False)),
    ]


def test_lazy_post_warmup_kv_rebalance_rejects_partial_head_after_retry(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner.kv_cache_config = SimpleNamespace(num_blocks=11)
    tail_ranges = [(0x105000, 0x6000)]
    head_ranges = [(0x100000, 0x5000)]
    all_ranges = [(0x100000, 0xB000)]
    runner._collect_uvm_post_warmup_kv_tail_ranges = (  # type: ignore[method-assign]
        lambda _target: (
            tail_ranges,
            {"ok": True, "reason": "ranges_collected", "target_blocks": 4},
        )
    )
    events: list[str] = []
    monkeypatch.setenv("VLLM_UVM_KV_CACHE_LAZY_RESERVE", "1")
    monkeypatch.setenv("VLLM_UVM_POST_WARMUP_KV_OFFLOAD_EXECUTOR", "driver_batch")
    monkeypatch.setattr(
        gpu_model_runner_module,
        "collect_uvm_kv_cache_physical_ranges_for_blocks",
        lambda block_ids: all_ranges if len(block_ids) == 11 else head_ranges,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_offload_ranges_to_cpu_with_executor",
        lambda ranges, *, sync, executor: events.append("offload")
        or (len(ranges), executor, 0, 0),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_advise_ranges_preferred_location",
        lambda ranges, device: events.append("advise") or len(ranges),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda ranges, device, *, executor: events.append("prefetch")
        or (len(ranges), executor, 0),
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "current_device",
        lambda: 0,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda device: events.append("sync"),
    )

    def query_residency(
        ranges: list[tuple[int, int]],
        *,
        device: int,
        wait_for_trackers: bool,
    ) -> dict[str, int | bool]:
        if ranges == tail_ranges:
            return _exclusive_residency_result(ranges, "cpu")
        expected_bytes = sum(size for _, size in ranges)
        cpu_bytes = 0x2000
        return {
            "ok": True,
            "status": len(ranges),
            "queried_bytes": expected_bytes,
            "gpu_bytes": expected_bytes - cpu_bytes,
            "cpu_bytes": cpu_bytes,
            "any_bytes": expected_bytes,
            "unknown_bytes": 0,
        }

    monkeypatch.setattr(
        gpu_model_runner_module,
        "driver_batch_query_ranges_residency_bytes",
        query_residency,
    )

    result = runner.rebalance_uvm_kv_cache_after_warmup(4)

    assert result["offload_complete"] is False
    assert events.count("offload") == 1
    assert events.count("advise") == 2
    assert events.count("prefetch") == 2
    assert events.count("sync") == 2


def test_pre_kv_stage_orders_expert_placement_before_kv(
    monkeypatch: Any,
) -> None:
    events: list[str] = []

    class FakeModelRunner:
        def run_uvm_init_time_weight_placement(self) -> None:
            events.append("cold_placement")

        def initialize_kv_cache(self, kv_cache_config: object) -> None:
            events.append("kv_allocation")

    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE", "pre_kv")
    monkeypatch.setattr(
        gpu_worker_module,
        "ensure_kv_transfer_initialized",
        lambda config, kv_config: None,
    )

    worker = object.__new__(Worker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=False)
    )
    worker.model_runner = FakeModelRunner()
    worker._kv_cache_uvm_allocations_enabled = lambda: False
    worker._maybe_get_kv_cache_uvm_memory_pool_context = nullcontext

    worker.initialize_from_config(object())

    assert events == ["cold_placement", "kv_allocation"]


@pytest.mark.parametrize(
    ("cudagraph_mode", "enforce_eager", "release_after_kv"),
    [
        (gpu_worker_module.CUDAGraphMode.FULL_DECODE_ONLY, False, True),
        (gpu_worker_module.CUDAGraphMode.NONE, True, True),
        (gpu_worker_module.CUDAGraphMode.FULL_SEPARATE, False, False),
    ],
)
def test_pre_kv_sentinel_lifetime_matches_graph_mode(
    monkeypatch: Any,
    cudagraph_mode: Any,
    enforce_eager: bool,
    release_after_kv: bool,
) -> None:
    events: list[str] = []
    sentinel = object()

    class FakeModelRunner:
        def initialize_kv_cache(self, kv_cache_config: object) -> None:
            events.append("kv_allocation")

        def run_uvm_init_time_weight_placement(self) -> None:
            events.append("cold_placement")

    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE", "post_warmup")
    monkeypatch.setattr(
        gpu_worker_module,
        "ensure_kv_transfer_initialized",
        lambda config, kv_config: None,
    )
    monkeypatch.setattr(
        Worker,
        "_allocate_uvm_native_hole_sentinel",
        lambda self, **kwargs: events.append("sentinel_allocation") or sentinel,
    )
    monkeypatch.setattr(
        Worker,
        "_release_uvm_native_hole_sentinel",
        lambda self, value, *, stage: events.append(
            f"sentinel_release:{stage}"
        ),
    )

    worker = object.__new__(Worker)
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            enable_sleep_mode=False,
            enforce_eager=enforce_eager,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
    )
    worker.model_runner = FakeModelRunner()
    worker._kv_cache_uvm_allocations_enabled = lambda: False
    worker._maybe_get_kv_cache_uvm_memory_pool_context = nullcontext
    worker._uvm_pre_kv_native_hole_sentinel = None

    worker.initialize_from_config(object())

    expected = ["sentinel_allocation", "kv_allocation"]
    if release_after_kv:
        expected.append(
            "sentinel_release:after_kv_before_kernel_warmup"
        )
        assert worker._uvm_pre_kv_native_hole_sentinel is None
    else:
        assert worker._uvm_pre_kv_native_hole_sentinel is sentinel
    assert events == expected


def test_post_warmup_rebalance_orders_kv_before_experts(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    control_file = tmp_path / "kv_dynamic_pool.control"

    class FakeModelRunner:
        def rebalance_uvm_kv_cache_after_warmup(
            self,
            target_blocks: int,
        ) -> dict[str, Any]:
            assert control_file.read_text(encoding="utf-8") == "target_blocks=4\n"
            events.append(f"kv:{target_blocks}")
            return {"offload_complete": True}

        def run_uvm_init_time_weight_placement(self) -> None:
            events.append("experts")

    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE", "post_warmup")
    monkeypatch.setenv("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "4")
    monkeypatch.setenv("VLLM_UVM_KV_DYNAMIC_POOL_CONTROL_FILE", str(control_file))
    monkeypatch.setenv("VLLM_USE_UVM", "1")

    worker = object.__new__(Worker)
    worker.model_runner = FakeModelRunner()
    worker._kv_cache_uvm_allocations_enabled = lambda: True

    worker._run_uvm_post_warmup_rebalance()

    assert events == ["kv:4", "experts"]

    worker._run_uvm_post_warmup_rebalance()
    assert events == ["kv:4", "experts"]


def test_serving_kv_resize_reconvergence_reapplies_target(
    monkeypatch: Any,
) -> None:
    calls: list[int] = []

    class FakeModelRunner:
        def rebalance_uvm_kv_cache_after_warmup(
            self,
            target_blocks: int,
        ) -> dict[str, Any]:
            calls.append(target_blocks)
            return {"offload_complete": True}

    monkeypatch.setenv("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "1023")
    worker = object.__new__(Worker)
    worker.model_runner = FakeModelRunner()
    worker._kv_cache_uvm_allocations_enabled = lambda: True

    worker._reconverge_uvm_kv_cache_for_serving()
    worker._reconverge_uvm_kv_cache_for_serving()

    assert calls == [1023, 1023]


def test_serving_kv_resize_reconvergence_fails_closed(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "1023")
    monkeypatch.setenv("VLLM_UVM_POST_WARMUP_REBALANCE_FAIL_FAST", "1")
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace(
        rebalance_uvm_kv_cache_after_warmup=lambda _target: {
            "offload_complete": False,
            "reason": "incomplete",
        }
    )
    worker._kv_cache_uvm_allocations_enabled = lambda: True

    with pytest.raises(RuntimeError, match="physical reconvergence failed"):
        worker._reconverge_uvm_kv_cache_for_serving()


def test_post_warmup_target_automatically_enables_dynamic_pool(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE", "post_warmup")
    monkeypatch.setenv("VLLM_UVM_KV_POST_WARMUP_TARGET_BLOCKS", "4")
    monkeypatch.delenv("VLLM_UVM_KV_RUNTIME_ENABLE", raising=False)
    monkeypatch.delenv("VLLM_UVM_KV_DYNAMIC_POOL_ENABLE", raising=False)

    policy = UvmKvRuntimePolicy(
        KVCacheConfig(
            num_blocks=11,
            kv_cache_tensors=[KVCacheTensor(size=11 * 4096, shared_by=["layer.0"])],
            kv_cache_groups=[],
        )
    )

    assert policy.enabled is True
    assert policy.dynamic_pool_active() is True
    assert policy.total_blocks == 10


def test_registered_slot_plan_configuration_fails_before_bank_allocation(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", "1")
    monkeypatch.delenv("VLLM_UVM_SLOT_DIRECT_INIT_BANK", raising=False)
    allocated = False

    def preallocate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal allocated
        allocated = True
        return {"ok": True}

    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        preallocate,
    )

    with pytest.raises(RuntimeError, match="complete fixed-bank configuration"):
        runner.finalize_uvm_slot_bank_after_warmup()
    assert allocated is False


def test_prefill_p3_decode_remote_requires_complete_fence_configuration(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    monkeypatch.setenv("VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ", "1")
    monkeypatch.setenv("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "slot_restore")
    monkeypatch.delenv("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", raising=False)
    monkeypatch.delenv("VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT", raising=False)
    monkeypatch.delenv("VLLM_UVM_SLOT_DIRECT_CUDA_FENCE", raising=False)

    with pytest.raises(RuntimeError, match="complete P3 fence configuration"):
        runner.finalize_uvm_slot_bank_after_warmup()


def test_uvm_step_phase_treats_one_token_new_request_as_prefill() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.uniform_decode_query_len = 1
    new_request = SimpleNamespace(req_id="new")
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1,
        scheduled_new_reqs=[new_request],
        num_scheduled_tokens={"new": 1},
    )

    assert runner._uvm_enabled_step_phase(scheduler_output) == (
        "enabled:prefill_step"
    )


def test_original_va_alias_prefill_switches_before_first_decode(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_original_va_phase_prefill_seen = False
    runner._uvm_original_va_phase_decode_cutover_done = False
    runner._uvm_original_va_phase_poison_reason = None
    runner._uvm_original_va_phase_transition_result = None
    runner.uniform_decode_query_len = 1

    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE",
        "prefill_remote_decode",
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "sealed")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", 7)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES", 4096
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED", True)

    runner._maybe_switch_uvm_original_va_prefill_to_decode_remote(
        SimpleNamespace(num_scheduled_tokens={"request": 8}),
        "enabled:prefill_step",
    )
    assert runner._uvm_original_va_phase_prefill_seen is True

    calls: list[dict[str, Any]] = []

    def transition(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "remote"
        offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID = 0
        offload_utils._CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE = True
        return {"ok": True, "pre_state": "sealed", "post_state": "remote"}

    monkeypatch.setattr(
        offload_utils,
        "transition_cpu_offload_uvm_original_va_to_remote",
        transition,
    )
    runner._maybe_switch_uvm_original_va_prefill_to_decode_remote(
        SimpleNamespace(num_scheduled_tokens={"request": 1}),
        "enabled:decode_step",
    )

    assert calls == [
        {
            "reason": "first_decode_cutover",
            "preserve_remote_graphs": True,
        }
    ]
    assert runner._uvm_original_va_phase_decode_cutover_done is True
    assert runner._uvm_original_va_phase_transition_result is not None


def test_original_va_collects_incremental_remote_expert_ranges(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE", "1")
    module = SimpleNamespace(
        get_uvm_init_weight_placement_snapshot=lambda: {
            "region_owners": [
                {
                    "region_id": "weights:0:4:envelope:400000:80000",
                    "placement": "remote",
                },
                {
                    "region_id": "weights:0:5:envelope:480000:80000",
                    "placement": "hot",
                },
                {
                    "region_id": "weights:0:4:envelope:400000:80000",
                    "placement": "remote",
                },
            ]
        }
    )
    runner = object.__new__(GPUModelRunner)
    runner.get_model = lambda: SimpleNamespace(  # type: ignore[method-assign]
        modules=lambda: iter([module])
    )

    assert runner._collect_uvm_runtime_remote_expert_ranges() == (
        (0x400000, 0x80000),
    )


def _make_original_va_phase_runner_for_boundary_tests() -> Any:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_original_va_phase_prefill_seen = False
    runner._uvm_original_va_phase_decode_cutover_done = False
    runner._uvm_original_va_phase_poison_reason = None
    runner._uvm_original_va_phase_transition_result = None
    runner.uniform_decode_query_len = 1
    return runner


def _enable_original_va_hybrid_boundary_state(monkeypatch: Any) -> None:
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE",
        "prefill_remote_decode",
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "sealed")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", 7)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES", 4096
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED", False)


def test_original_va_hybrid_rejects_mixed_prefill_decode_batch(
    monkeypatch: Any,
) -> None:
    runner = _make_original_va_phase_runner_for_boundary_tests()
    _enable_original_va_hybrid_boundary_state(monkeypatch)

    with pytest.raises(RuntimeError, match="requires an unmixed request"):
        runner._maybe_switch_uvm_original_va_prefill_to_decode_remote(
            SimpleNamespace(num_scheduled_tokens={"request": 2}),
            "enabled:mixed_step",
        )

    assert runner._uvm_original_va_phase_poison_reason == (
        "mixed prefill/decode batches are unsupported"
    )


def _enable_original_va_token_threshold_state(
    monkeypatch: Any,
    *,
    lifecycle_state: str,
    park_resume: bool = True,
) -> None:
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE",
        "token_threshold",
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_ORIGINAL_VA_TOKEN_THRESHOLD", 64
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD",
        65,
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED", park_resume
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", lifecycle_state
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", 7)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES", 4096
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE",
        lifecycle_state == "remote_parked",
    )


def test_original_va_token_threshold_mixed_64_tokens_parks(
    monkeypatch: Any,
) -> None:
    runner = _make_original_va_phase_runner_for_boundary_tests()
    runner._collect_uvm_runtime_remote_expert_ranges = lambda: ()
    _enable_original_va_token_threshold_state(
        monkeypatch, lifecycle_state="sealed"
    )
    calls: list[dict[str, Any]] = []

    def park(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        monkeypatch.setattr(
            offload_utils,
            "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE",
            "remote_parked",
        )
        monkeypatch.setattr(
            offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", True
        )
        return {"ok": True, "post_state": "remote_parked"}

    monkeypatch.setattr(
        offload_utils,
        "transition_cpu_offload_uvm_original_va_to_remote",
        park,
    )
    runner._maybe_switch_uvm_original_va_prefill_to_decode_remote(
        SimpleNamespace(
            total_num_scheduled_tokens=64,
            num_scheduled_tokens={"decode": 1, "prefill": 63},
        ),
        "enabled:mixed_step",
    )

    assert calls == [
        {
            "reason": "token_threshold_remote",
            "preserve_remote_graphs": False,
        }
    ]
    assert runner._uvm_original_va_phase_poison_reason is None
    assert runner._uvm_original_va_phase_transition_result["selected_access"] == (
        "remote"
    )


def test_original_va_token_threshold_mixed_65_tokens_resumes(
    monkeypatch: Any,
) -> None:
    runner = _make_original_va_phase_runner_for_boundary_tests()
    _enable_original_va_token_threshold_state(
        monkeypatch, lifecycle_state="remote_parked"
    )
    calls: list[str] = []

    def resume() -> dict[str, Any]:
        calls.append("resume")
        monkeypatch.setattr(
            offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "sealed"
        )
        monkeypatch.setattr(
            offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", False
        )
        return {"ok": True, "post_state": "sealed"}

    monkeypatch.setattr(
        offload_utils,
        "activate_cpu_offload_uvm_original_va_prefill_topology",
        resume,
    )
    runner._maybe_switch_uvm_original_va_prefill_to_decode_remote(
        SimpleNamespace(
            total_num_scheduled_tokens=65,
            num_scheduled_tokens={"decode": 1, "prefill": 64},
        ),
        "enabled:mixed_step",
    )

    assert calls == ["resume"]
    assert runner._uvm_original_va_phase_poison_reason is None
    assert runner._uvm_original_va_phase_transition_result["selected_access"] == (
        "copy"
    )


def test_original_va_token_threshold_hysteresis_holds_current_topology(
    monkeypatch: Any,
) -> None:
    runner = _make_original_va_phase_runner_for_boundary_tests()
    _enable_original_va_token_threshold_state(
        monkeypatch, lifecycle_state="sealed"
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD",
        128,
    )
    monkeypatch.setattr(
        offload_utils,
        "transition_cpu_offload_uvm_original_va_to_remote",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("copy topology switched inside hysteresis band")
        ),
    )
    runner._maybe_switch_uvm_original_va_prefill_to_decode_remote(
        SimpleNamespace(total_num_scheduled_tokens=80),
        "enabled:mixed_step",
    )
    result = runner._uvm_original_va_phase_transition_result
    assert result["selected_access"] == "copy"
    assert result["selection_reason"] == "copy_hold"
    assert result["topology_changed"] is False

    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE",
        "remote_parked",
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", True
    )
    monkeypatch.setattr(
        offload_utils,
        "activate_cpu_offload_uvm_original_va_prefill_topology",
        lambda: (_ for _ in ()).throw(
            AssertionError("remote topology switched inside hysteresis band")
        ),
    )
    runner._maybe_switch_uvm_original_va_prefill_to_decode_remote(
        SimpleNamespace(total_num_scheduled_tokens=80),
        "enabled:mixed_step",
    )
    result = runner._uvm_original_va_phase_transition_result
    assert result["selected_access"] == "remote"
    assert result["selection_reason"] == "remote_hold"
    assert result["topology_changed"] is False


def test_original_va_token_threshold_does_not_repeat_target_switch(
    monkeypatch: Any,
) -> None:
    runner = _make_original_va_phase_runner_for_boundary_tests()
    _enable_original_va_token_threshold_state(
        monkeypatch, lifecycle_state="remote_parked"
    )
    monkeypatch.setattr(
        offload_utils,
        "transition_cpu_offload_uvm_original_va_to_remote",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("remote topology was parked twice")
        ),
    )
    runner._maybe_switch_uvm_original_va_prefill_to_decode_remote(
        SimpleNamespace(total_num_scheduled_tokens=64),
        "enabled:decode_step",
    )
    assert runner._uvm_original_va_phase_transition_result[
        "topology_changed"
    ] is False

    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "sealed"
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", False
    )
    monkeypatch.setattr(
        offload_utils,
        "activate_cpu_offload_uvm_original_va_prefill_topology",
        lambda: (_ for _ in ()).throw(
            AssertionError("sealed topology was resumed twice")
        ),
    )
    runner._maybe_switch_uvm_original_va_prefill_to_decode_remote(
        SimpleNamespace(total_num_scheduled_tokens=65),
        "enabled:prefill_step",
    )
    assert runner._uvm_original_va_phase_transition_result[
        "topology_changed"
    ] is False


def test_original_va_token_threshold_requires_park_resume(
    monkeypatch: Any,
) -> None:
    runner = _make_original_va_phase_runner_for_boundary_tests()
    _enable_original_va_token_threshold_state(
        monkeypatch,
        lifecycle_state="sealed",
        park_resume=False,
    )

    with pytest.raises(RuntimeError, match="requires PARK/RESUME"):
        runner._maybe_switch_uvm_original_va_prefill_to_decode_remote(
            SimpleNamespace(total_num_scheduled_tokens=64),
            "enabled:mixed_step",
        )
    assert runner._uvm_original_va_phase_poison_reason == (
        "token-threshold mode requires PARK/RESUME"
    )


def _enable_original_va_graph_park_resume_state(
    monkeypatch: Any, *, lifecycle_state: str
) -> None:
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE",
        "prefill_graph_remote_decode",
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED", True)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", lifecycle_state
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", 7)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES", 4096
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE",
        lifecycle_state == "remote_parked",
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED", True)


def test_original_va_graph_park_resume_rejects_mixed_batch(
    monkeypatch: Any,
) -> None:
    runner = _make_original_va_phase_runner_for_boundary_tests()
    _enable_original_va_graph_park_resume_state(
        monkeypatch, lifecycle_state="sealed"
    )

    with pytest.raises(RuntimeError, match="requires an unmixed request"):
        runner._maybe_switch_uvm_original_va_prefill_to_decode_remote(
            SimpleNamespace(num_scheduled_tokens={"decode": 1, "prefill": 8}),
            "enabled:mixed_step",
        )

    assert runner._uvm_original_va_phase_prefill_seen is False
    assert runner._uvm_original_va_phase_decode_cutover_done is False
    assert runner._uvm_original_va_phase_poison_reason == (
        "mixed prefill/decode batches are unsupported"
    )


def test_original_va_graph_without_park_resume_rejects_mixed_batch(
    monkeypatch: Any,
) -> None:
    runner = _make_original_va_phase_runner_for_boundary_tests()
    _enable_original_va_graph_park_resume_state(
        monkeypatch, lifecycle_state="sealed"
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED", False)

    with pytest.raises(RuntimeError, match="requires an unmixed request"):
        runner._maybe_switch_uvm_original_va_prefill_to_decode_remote(
            SimpleNamespace(num_scheduled_tokens={"decode": 1, "prefill": 8}),
            "enabled:mixed_step",
        )

    assert runner._uvm_original_va_phase_poison_reason == (
        "mixed prefill/decode batches are unsupported"
    )


def test_original_va_graph_mixed_batch_never_resumes_parked_aliases(
    monkeypatch: Any,
) -> None:
    runner = _make_original_va_phase_runner_for_boundary_tests()
    runner._uvm_original_va_phase_prefill_seen = True
    runner._uvm_original_va_phase_decode_cutover_done = True
    _enable_original_va_graph_park_resume_state(
        monkeypatch, lifecycle_state="remote_parked"
    )
    calls: list[str] = []

    def resume() -> dict[str, Any]:
        calls.append("resume")
        monkeypatch.setattr(
            offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "sealed"
        )
        monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", False)
        return {"ok": True}

    monkeypatch.setattr(
        offload_utils,
        "activate_cpu_offload_uvm_original_va_prefill_topology",
        resume,
    )

    with pytest.raises(RuntimeError, match="requires an unmixed request"):
        runner._maybe_switch_uvm_original_va_prefill_to_decode_remote(
            SimpleNamespace(num_scheduled_tokens={"decode": 1, "prefill": 8}),
            "enabled:mixed_step",
        )

    assert calls == []
    assert runner._uvm_original_va_phase_prefill_seen is True
    assert runner._uvm_original_va_phase_decode_cutover_done is True
    assert runner._uvm_original_va_phase_poison_reason == (
        "mixed prefill/decode batches are unsupported"
    )


def test_original_va_graph_phase_uses_separate_replay_certificates(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    calls: list[str] = []
    monkeypatch.setattr(
        offload_utils,
        "validate_cpu_offload_uvm_original_va_alias_graph_replay",
        lambda: calls.append("alias"),
    )
    monkeypatch.setattr(
        offload_utils,
        "validate_cpu_offload_uvm_original_va_remote_graph_replay",
        lambda: calls.append("remote"),
    )

    runner._validate_uvm_original_va_graph_replay_for_step(
        original_va_phase_mode="prefill_graph_remote_decode",
        enabled_step_phase="enabled:prefill_step",
        cudagraph_runtime_mode=gpu_model_runner_module.CUDAGraphMode.FULL,
    )
    runner._validate_uvm_original_va_graph_replay_for_step(
        original_va_phase_mode="prefill_graph_remote_decode",
        enabled_step_phase="enabled:decode_step",
        cudagraph_runtime_mode=gpu_model_runner_module.CUDAGraphMode.FULL,
    )

    assert calls == ["alias", "remote"]


def test_original_va_graph_phase_rejects_eager_fallback() -> None:
    runner = object.__new__(GPUModelRunner)

    with pytest.raises(RuntimeError, match="cannot fall back to eager"):
        runner._validate_uvm_original_va_graph_replay_for_step(
            original_va_phase_mode="prefill_graph_remote_decode",
            enabled_step_phase="enabled:prefill_step",
            cudagraph_runtime_mode=gpu_model_runner_module.CUDAGraphMode.NONE,
        )


def test_original_va_hybrid_rejects_prefill_after_remote_cutover(
    monkeypatch: Any,
) -> None:
    runner = _make_original_va_phase_runner_for_boundary_tests()
    _enable_original_va_hybrid_boundary_state(monkeypatch)

    runner._maybe_switch_uvm_original_va_prefill_to_decode_remote(
        SimpleNamespace(num_scheduled_tokens={"request": 8}),
        "enabled:prefill_step",
    )

    def transition(**_kwargs: Any) -> dict[str, Any]:
        monkeypatch.setattr(
            offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "remote"
        )
        monkeypatch.setattr(
            offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", 0
        )
        monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", True)
        return {"ok": True, "pre_state": "sealed", "post_state": "remote"}

    monkeypatch.setattr(
        offload_utils,
        "transition_cpu_offload_uvm_original_va_to_remote",
        transition,
    )
    runner._maybe_switch_uvm_original_va_prefill_to_decode_remote(
        SimpleNamespace(num_scheduled_tokens={"request": 1}),
        "enabled:decode_step",
    )

    with pytest.raises(RuntimeError, match="one-way remote cutover"):
        runner._maybe_switch_uvm_original_va_prefill_to_decode_remote(
            SimpleNamespace(num_scheduled_tokens={"request": 8}),
            "enabled:prefill_step",
        )

    assert runner._uvm_original_va_phase_poison_reason == (
        "a prefill step arrived after remote decode cutover"
    )


def _complete_p3_close_result() -> dict[str, Any]:
    return {
        "ok": True,
        "result": {
            "abort": {
                "ok": True,
                "contract_ok": True,
                "session_cleared": True,
                "remaining_slot_count": 0,
                "remaining_bytes": 0,
            }
        },
    }


def test_prefill_p3_switches_to_remote_read_before_first_decode(
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    module = SimpleNamespace(
        uvm_expert_offload_access_mode="slot_restore",
        _uvm_offloaded_global_expert_ids={7, 9},
        _uvm_remote_read_global_expert_ids=set(),
        _uvm_layer_id=lambda: 3,
        _uvm_expert_records={
            (3, 7): SimpleNamespace(envelope_region_ids=["r7a", "r7b"]),
            (3, 9): SimpleNamespace(envelope_region_ids=["r9a"]),
        },
    )
    model = SimpleNamespace(modules=lambda: iter([module]))
    runner = object.__new__(GPUModelRunner)
    runner.get_model = lambda: model  # type: ignore[method-assign]
    runner._uvm_prefill_p3_seen = True
    runner._uvm_prefill_p3_decode_remote_state = "prefill_p3"
    runner._uvm_slot_init_bank_done = True
    runner._uvm_slot_init_bank_result = {"ok": True}
    runner._uvm_slot_registered_plan_result = {"ok": True}
    runner._uvm_slot_session_contract = {"slot_session_enabled": True}
    monkeypatch.setenv("VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ", "1")
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda _device: events.append("sync"),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "close_uvm_slot_direct_policy_session",
        lambda **_kwargs: events.append("close") or _complete_p3_close_result(),
    )

    def remote_map(region_ids: list[str], **_kwargs: Any) -> dict[str, Any]:
        events.append("remote:" + ",".join(region_ids))
        return {"ok": True}

    monkeypatch.setattr(
        gpu_model_runner_module,
        "transition_uvm_regions_to_remote_read",
        remote_map,
    )

    runner._maybe_switch_uvm_prefill_p3_to_decode_remote_read(
        SimpleNamespace(num_scheduled_tokens={"request": 1}),
        "enabled:decode_step",
    )

    assert events == ["sync", "close", "remote:r7a,r7b,r9a"]
    assert module.uvm_expert_offload_access_mode == "remote_read"
    assert module._uvm_offloaded_global_expert_ids == set()
    assert module._uvm_remote_read_global_expert_ids == {7, 9}
    assert runner._uvm_prefill_p3_decode_remote_state == "decode_remote_read"
    assert runner._uvm_slot_init_bank_done is False

    runner._maybe_switch_uvm_prefill_p3_to_decode_remote_read(
        SimpleNamespace(num_scheduled_tokens={"request": 1}),
        "enabled:decode_step",
    )
    assert events == ["sync", "close", "remote:r7a,r7b,r9a"]


def test_prefill_p3_remote_read_failure_poison_is_sticky(
    monkeypatch: Any,
) -> None:
    module = SimpleNamespace(
        uvm_expert_offload_access_mode="slot_restore",
        _uvm_offloaded_global_expert_ids={2},
        _uvm_remote_read_global_expert_ids=set(),
        _uvm_layer_id=lambda: 1,
        _uvm_expert_records={
            (1, 2): SimpleNamespace(envelope_region_ids=["r2"]),
        },
    )
    model = SimpleNamespace(modules=lambda: iter([module]))
    runner = object.__new__(GPUModelRunner)
    runner.get_model = lambda: model  # type: ignore[method-assign]
    runner._uvm_prefill_p3_seen = True
    runner._uvm_prefill_p3_decode_remote_state = "prefill_p3"
    monkeypatch.setenv("VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ", "1")
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "synchronize", lambda _device: None
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "close_uvm_slot_direct_policy_session",
        lambda **_kwargs: _complete_p3_close_result(),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "transition_uvm_regions_to_remote_read",
        lambda *_args, **_kwargs: {"ok": False, "reason": "injected"},
    )

    with pytest.raises(RuntimeError, match="worker is poisoned"):
        runner._maybe_switch_uvm_prefill_p3_to_decode_remote_read(
            SimpleNamespace(num_scheduled_tokens={"request": 1}),
            "enabled:decode_step",
        )

    assert runner._uvm_prefill_p3_decode_remote_state == "poisoned"
    assert module.uvm_expert_offload_access_mode == "poisoned"
    with pytest.raises(RuntimeError, match="worker is poisoned"):
        runner._maybe_switch_uvm_prefill_p3_to_decode_remote_read(
            SimpleNamespace(num_scheduled_tokens={"request": 1}),
            "enabled:decode_step",
        )


def test_prefill_p3_remote_rejects_mixed_step_before_driver_work(
    monkeypatch: Any,
) -> None:
    module = SimpleNamespace(uvm_expert_offload_access_mode="slot_restore")
    model = SimpleNamespace(modules=lambda: iter([module]))
    runner = object.__new__(GPUModelRunner)
    runner.get_model = lambda: model  # type: ignore[method-assign]
    monkeypatch.setenv("VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ", "1")
    monkeypatch.setattr(
        gpu_model_runner_module,
        "close_uvm_slot_direct_policy_session",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("mixed step must fail before close")
        ),
    )

    with pytest.raises(RuntimeError, match="unmixed single request"):
        runner._maybe_switch_uvm_prefill_p3_to_decode_remote_read(
            SimpleNamespace(num_scheduled_tokens={"prefill": 4, "decode": 1}),
            "enabled:mixed_step",
        )

    assert runner._uvm_prefill_p3_decode_remote_state == "poisoned"


def test_remote_transition_requires_complete_mapping_audit(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("VLLM_UVM_PREFILL_P3_DECODE_REMOTE_BATCH", "0")
    monkeypatch.setattr(
        residency_module,
        "release_uvm_regions",
        lambda region_ids, **_kwargs: {
            "ok": True,
            "statuses": {region_id: True for region_id in region_ids},
            "audits": {
                "r0": {
                    "ok": True,
                    "remote_read": True,
                    "remote_map_failure_blocks": 0,
                    "cpu_resident_bytes": 512 * 1024,
                    "remote_mapped_bytes": 512 * 1024,
                    "registered_length": 512 * 1024,
                },
                "r1": {
                    "ok": True,
                    "remote_read": True,
                    "remote_map_failure_blocks": 1,
                    "cpu_resident_bytes": 512 * 1024,
                    "remote_mapped_bytes": 0,
                    "registered_length": 512 * 1024,
                },
            },
        },
    )

    result = residency_module.transition_uvm_regions_to_remote_read(["r0", "r1"])

    assert result["ok"] is False
    assert result["reason"] == "remote_read_audit_failed"


def test_remote_transition_uses_one_audited_batch_plan(
    monkeypatch: Any,
) -> None:
    registry = SimpleNamespace(
        regions={
            "r0": residency_module.KernelRegion(
                "r0", 0x40000000, 512 * 1024, residency_module.RegionState.CPU
            ),
            "r1": residency_module.KernelRegion(
                "r1", 0x50000000, 512 * 1024, residency_module.RegionState.CPU
            ),
        }
    )
    monkeypatch.setattr(residency_module, "_EXPLICIT_REGION_REGISTRY", registry)
    captured: list[list[tuple[int, int, int]]] = []

    def prepare(
        regions: list[tuple[int, int, int]], **_kwargs: Any
    ) -> dict[str, Any]:
        captured.append(regions)
        return {
            "ok": True,
            "ioctl_calls": 1,
            "remote_mapped_bytes": 1024 * 1024,
        }

    monkeypatch.setattr(
        residency_module,
        "driver_prepare_regions_remote_read_audited",
        prepare,
    )

    result = residency_module.transition_uvm_regions_to_remote_read(["r0", "r1"])

    assert result["ok"] is True
    assert result["batch"] is True
    assert captured == [
        [
            (
                residency_module._stable_driver_region_id("r0"),
                0x40000000,
                512 * 1024,
            ),
            (
                residency_module._stable_driver_region_id("r1"),
                0x50000000,
                512 * 1024,
            ),
        ]
    ]


def test_audited_remote_batch_checks_every_range_result(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        uvm_module,
        "_phase_f_gpu_uuid",
        lambda _device: uvm_module._PhaseFUuid(),
    )
    monkeypatch.setattr(uvm_module, "_iter_existing_uvm_fd_sources", lambda: [23])
    monkeypatch.setattr(
        uvm_module,
        "_initialize_duplicated_uvm_fd",
        lambda source_fd: 123 if source_fd == 23 else None,
    )
    monkeypatch.setattr(
        uvm_module,
        "_open_initialized_uvm_fd",
        lambda: (_ for _ in ()).throw(AssertionError("fallback fd must not open")),
    )
    monkeypatch.setattr(uvm_module.os, "close", lambda _fd: None)
    calls = 0
    actions: list[int] = []

    def ioctl(_fd: int, _request: int, params: Any) -> int:
        nonlocal calls
        calls += 1
        actions.append(int(params.action))
        params.success_count = params.count
        params.failure_count = 0
        params.rmStatus = 0
        for index in range(int(params.count)):
            length = int(params.ranges[index].length)
            item = params.results[index]
            item.status = 0
            item.queried_bytes = length
            item.resident_cpu_bytes = length
            item.resident_gpu_bytes = length
            item.resident_any_bytes = length
            item.resident_unknown_bytes = 0
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module.driver_prepare_ranges_remote_read_audited(
        [(0x100000 + i * 0x10000, 0x10000) for i in range(65)]
    )

    assert result["ok"] is True
    assert result["ioctl_calls"] == 2
    assert result["success_count"] == 65
    assert result["remote_mapped_bytes"] == 65 * 0x10000
    assert result["fd_candidate_attempts"] == 1
    assert result["fd_candidate_ioctl_calls"] == 2
    assert result["selected_fd_source"] == 23
    assert result["selected_fd_kind"] == "existing"
    assert calls == 2
    assert actions == [
        uvm_module.UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_AUDITED,
        uvm_module.UVM_PHASE_F_ACTION_PREPARE_REMOTE_READ_AUDITED,
    ]


def test_audited_remote_batch_retries_zero_progress_invalid_device_fd(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        uvm_module,
        "_phase_f_gpu_uuid",
        lambda _device: uvm_module._PhaseFUuid(),
    )
    monkeypatch.setattr(
        uvm_module,
        "_iter_existing_uvm_fd_sources",
        lambda: [10, 20],
    )
    monkeypatch.setattr(
        uvm_module,
        "_initialize_duplicated_uvm_fd",
        lambda source_fd: source_fd + 100,
    )
    monkeypatch.setattr(
        uvm_module,
        "_open_initialized_uvm_fd",
        lambda: (_ for _ in ()).throw(AssertionError("fallback fd must not open")),
    )
    closed: list[int] = []
    monkeypatch.setattr(uvm_module.os, "close", lambda fd: closed.append(int(fd)))
    calls: list[int] = []

    def ioctl(fd: int, request: int, params: Any) -> int:
        calls.append(fd)
        assert request == uvm_module._UVM_PHASE_F_BATCH_EXECUTOR_IOCTL
        if fd == 110:
            params.success_count = 0
            params.failure_count = params.count
            params.rmStatus = uvm_module._NV_ERR_INVALID_DEVICE
            for index in range(int(params.count)):
                length = int(params.ranges[index].length)
                item = params.results[index]
                item.status = uvm_module._NV_ERR_INVALID_DEVICE
                item.queried_bytes = length
                item.resident_unknown_bytes = length
            return 0
        assert fd == 120
        params.success_count = params.count
        params.failure_count = 0
        params.rmStatus = 0
        for index in range(int(params.count)):
            length = int(params.ranges[index].length)
            item = params.results[index]
            item.status = 0
            item.queried_bytes = length
            item.resident_cpu_bytes = length
            item.resident_gpu_bytes = length
            item.resident_any_bytes = length
            item.resident_unknown_bytes = 0
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module.driver_prepare_ranges_remote_read_audited(
        [(0x100000, 0x10000)]
    )

    assert result["ok"] is True
    assert result["fd_candidate_attempts"] == 2
    assert result["fd_candidate_ioctl_calls"] == 2
    assert result["selected_fd_source"] == 20
    assert result["selected_fd_kind"] == "existing"
    assert result["made_progress"] is True
    assert result["fd_mismatch_proven"] is False
    assert calls == [110, 120]
    assert closed == [110, 120]


def test_audited_remote_batch_partial_progress_never_switches_fd(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        uvm_module,
        "_phase_f_gpu_uuid",
        lambda _device: uvm_module._PhaseFUuid(),
    )
    monkeypatch.setattr(
        uvm_module,
        "_iter_existing_uvm_fd_sources",
        lambda: [10, 20],
    )
    monkeypatch.setattr(
        uvm_module,
        "_initialize_duplicated_uvm_fd",
        lambda source_fd: source_fd + 100,
    )
    monkeypatch.setattr(
        uvm_module,
        "_open_initialized_uvm_fd",
        lambda: (_ for _ in ()).throw(AssertionError("fallback fd must not open")),
    )
    calls: list[int] = []
    monkeypatch.setattr(uvm_module.os, "close", lambda _fd: None)

    def ioctl(fd: int, _request: int, params: Any) -> int:
        calls.append(fd)
        assert fd == 110
        params.success_count = 1
        params.failure_count = params.count - 1
        params.rmStatus = uvm_module._NV_ERR_INVALID_DEVICE
        length = int(params.ranges[0].length)
        item = params.results[0]
        item.status = 0
        item.queried_bytes = length
        item.resident_cpu_bytes = length
        item.resident_gpu_bytes = length
        item.resident_any_bytes = length
        item.resident_unknown_bytes = 0
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module.driver_prepare_ranges_remote_read_audited(
        [(0x100000, 0x10000), (0x110000, 0x10000)]
    )

    assert result["ok"] is False
    assert result["reason"] == "batch_ioctl_failed"
    assert result["made_progress"] is True
    assert result["fd_candidate_attempts"] == 1
    assert result["selected_fd_source"] == 10
    assert calls == [110]


def test_audited_remote_batch_indeterminate_zero_progress_never_switches_fd(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        uvm_module,
        "_phase_f_gpu_uuid",
        lambda _device: uvm_module._PhaseFUuid(),
    )
    monkeypatch.setattr(
        uvm_module,
        "_iter_existing_uvm_fd_sources",
        lambda: [10, 20],
    )
    monkeypatch.setattr(
        uvm_module,
        "_initialize_duplicated_uvm_fd",
        lambda source_fd: source_fd + 100,
    )
    monkeypatch.setattr(
        uvm_module,
        "_open_initialized_uvm_fd",
        lambda: (_ for _ in ()).throw(AssertionError("fallback fd must not open")),
    )
    calls: list[int] = []
    monkeypatch.setattr(uvm_module.os, "close", lambda _fd: None)

    def ioctl(fd: int, _request: int, params: Any) -> int:
        calls.append(fd)
        assert fd == 110
        params.success_count = 0
        params.failure_count = params.count
        params.rmStatus = uvm_module._NV_ERR_INVALID_DEVICE
        # A missing per-range preflight signature makes side effects
        # indeterminate even when all aggregate byte counters are zero.
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)
    result = uvm_module.driver_prepare_ranges_remote_read_audited(
        [(0x100000, 0x10000)]
    )

    assert result["ok"] is False
    assert result["made_progress"] is False
    assert result["fd_mismatch_proven"] is False
    assert result["fd_candidate_attempts"] == 1
    assert calls == [110]


def _complete_registered_remote_batch(
    params: Any,
    lengths: dict[int, int],
    *,
    newly_mapped: bool = True,
) -> None:
    params.success_count = params.count
    params.failure_count = 0
    params.rmStatus = 0
    for index in range(int(params.count)):
        length = lengths[int(params.region_ids[index])]
        item = params.results[index]
        item.status = 0
        item.queried_bytes = length
        item.resident_cpu_bytes = length
        item.resident_gpu_bytes = length if newly_mapped else 0
        item.resident_any_bytes = length
        item.resident_unknown_bytes = 0


def test_registered_remote_batch_uses_owner_fd_and_exact_new_mapping(
    monkeypatch: Any,
) -> None:
    regions = [
        (index + 1, 0x40000000 + index * 0x10000, 0x10000)
        for index in range(65)
    ]
    lengths = {region_id: length for region_id, _base, length in regions}
    monkeypatch.setattr(
        uvm_module,
        "_phase_f_gpu_uuid",
        lambda _device: uvm_module._PhaseFUuid(),
    )
    monkeypatch.setattr(uvm_module, "_iter_existing_uvm_fd_sources", lambda: [10, 20])
    monkeypatch.setattr(uvm_module, "_initialize_duplicated_uvm_fd", lambda fd: fd + 100)
    monkeypatch.setattr(uvm_module.os, "close", lambda _fd: None)
    calls: list[int] = []

    def ioctl(fd: int, request: int, params: Any) -> int:
        calls.append(fd)
        assert request == uvm_module._UVM_PHASE_F_BATCH_REGIONS_IOCTL
        if fd == 110:
            params.success_count = 0
            params.failure_count = params.count
            params.rmStatus = 0
            return 0
        _complete_registered_remote_batch(params, lengths)
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_prepare_regions_remote_read_audited(regions)

    assert result["ok"] is True
    assert result["fd_candidate_attempts"] == 2
    assert result["ioctl_calls"] == 3
    assert result["mapping_ioctl_calls"] == 2
    assert result["remote_mapped_bytes"] == 65 * 0x10000
    assert calls == [110, 120, 120]


def test_registered_remote_batch_rejects_existing_mapping_without_fd_retry(
    monkeypatch: Any,
) -> None:
    regions = [(1, 0x40000000, 0x80000)]
    monkeypatch.setattr(
        uvm_module,
        "_phase_f_gpu_uuid",
        lambda _device: uvm_module._PhaseFUuid(),
    )
    monkeypatch.setattr(uvm_module, "_iter_existing_uvm_fd_sources", lambda: [10, 20])
    monkeypatch.setattr(uvm_module, "_initialize_duplicated_uvm_fd", lambda fd: fd + 100)
    monkeypatch.setattr(uvm_module.os, "close", lambda _fd: None)
    calls: list[int] = []

    def ioctl(fd: int, _request: int, params: Any) -> int:
        calls.append(fd)
        _complete_registered_remote_batch(params, {1: 0x80000}, newly_mapped=False)
        return 0

    monkeypatch.setattr(uvm_module, "_libc_ioctl", ioctl)

    result = uvm_module.driver_prepare_regions_remote_read_audited(regions)

    assert result["ok"] is False
    assert result["reason"] == "range_audit_failed"
    assert result["made_progress"] is True
    assert calls == [110]


@pytest.mark.parametrize(
    "regions",
    [
        [(True, 0x40000000, 0x80000)],
        [(1, 0x40000001, 0x80000)],
        [(1, 0x40000000, 0x80000), (2, 0x40040000, 0x80000)],
        [(1, 0x40000000, 0x80000), (1, 0x50000000, 0x80000)],
    ],
)
def test_registered_remote_batch_rejects_invalid_input_before_ioctl(
    monkeypatch: Any,
    regions: list[tuple[Any, int, int]],
) -> None:
    monkeypatch.setattr(
        uvm_module,
        "_libc_ioctl",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected ioctl")),
    )

    result = uvm_module.driver_prepare_regions_remote_read_audited(regions)

    assert result["ok"] is False
    assert result["ioctl_calls"] == 0


def test_phase_f_ctypes_batch_abi_matches_kernel_header() -> None:
    assert uvm_module._phase_f_batch_abi_matches_kernel() is True


def test_init_bank_requires_cleanup_owning_slot_access_mode(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [(0, [(0, ["region-0"])])]
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_INIT_BANK", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", "0")
    monkeypatch.setenv("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "restore")
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid access mode must fail before mode8")
        ),
    )

    with pytest.raises(RuntimeError, match="slot init-bank requires"):
        runner.finalize_uvm_slot_bank_after_warmup()
    assert residency_module.get_uvm_slot_direct_session_contract() is None


@pytest.mark.parametrize(
    ("candidates", "message"),
    [
        (
            [(0, [(0, ["r0"])]), (0, [(0, ["r1"])])],
            "layer_id values must be unique",
        ),
        (
            [(layer_id, [(0, [f"r{layer_id}"])]) for layer_id in range(64)],
            "exceed the kernel limit",
        ),
    ],
)
def test_registered_slot_plan_preflight_rejects_before_mode8(
    monkeypatch: Any,
    candidates: list[tuple[int, list[tuple[int, list[str]]]]],
    message: str,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = candidates
    _enable_registered_slot_plan(monkeypatch)
    allocated = False

    def preallocate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal allocated
        allocated = True
        return {"ok": True}

    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        preallocate,
    )

    with pytest.raises(RuntimeError, match=message):
        runner.finalize_uvm_slot_bank_after_warmup()
    assert allocated is False


def test_registered_slot_plan_preflight_allows_different_expert_ids(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [
        (0, [(0, ["l0-e0"]), (1, ["l0-e1"])]),
        (1, [(2, ["l1-e2"]), (3, ["l1-e3"])]),
    ]
    _enable_registered_slot_plan(monkeypatch)
    _mock_complete_slot_registry(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    calls: list[str] = []
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: calls.append("mode8") or {"ok": True},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "register_uvm_slot_direct_layer_plans",
        lambda *_args, **_kwargs: calls.append("register") or {"ok": True},
    )

    runner.finalize_uvm_slot_bank_after_warmup()

    assert calls == ["mode8", "register"]


@pytest.mark.parametrize(
    ("records", "message"),
    [
        (
            [
                {
                    "region_id": "r0",
                    "base": 0x40001000,
                    "length": 2 * 1024 * 1024,
                }
            ],
            "not 64 KiB aligned",
        ),
        (
            [
                {
                    "region_id": "r0",
                    "base": 0x40000000,
                    "length": 2 * 1024 * 1024 - 4096,
                }
            ],
            "not 64 KiB aligned",
        ),
        (
            [
                {
                    "region_id": "r0",
                    "base": 0x40000000,
                    "length": 1024 * 1024,
                },
                {
                    "region_id": "r1",
                    "base": 0x40080000,
                    "length": 1024 * 1024,
                },
            ],
            "64 KiB ranges overlap",
        ),
    ],
)
def test_registered_slot_plan_rejects_invalid_64k_layout_before_mode8(
    monkeypatch: Any,
    records: list[dict[str, Any]],
    message: str,
) -> None:
    runner = object.__new__(GPUModelRunner)
    region_ids = [str(record["region_id"]) for record in records]
    runner._uvm_slot_init_bank_candidates = [(0, [(0, region_ids)])]
    _enable_registered_slot_plan(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "snapshot_uvm_region_registry",
        lambda requested: {
            "ok": list(requested) == region_ids,
            "regions": records,
        },
    )
    mode8_calls = 0

    def preallocate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal mode8_calls
        mode8_calls += 1
        return {"ok": True}

    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        preallocate,
    )

    with pytest.raises(RuntimeError, match=message):
        runner.finalize_uvm_slot_bank_after_warmup()
    assert mode8_calls == 0
    assert residency_module.get_uvm_slot_direct_session_contract() is None


def test_registered_slot_plan_allows_sparse_512k_envelopes(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    region_ids = ["cold-envelope-0", "cold-envelope-1"]
    records = [
        {
            "region_id": region_ids[0],
            "base": 0x40000000,
            "length": 512 * 1024,
        },
        {
            "region_id": region_ids[1],
            "base": 0x40100000,
            "length": 512 * 1024,
        },
    ]
    runner._uvm_slot_init_bank_candidates = [(0, [(7, region_ids)])]
    _enable_registered_slot_plan(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "snapshot_uvm_region_registry",
        lambda requested: {
            "ok": list(requested) == region_ids,
            "regions": records,
        },
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    calls: list[str] = []
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: calls.append("mode8") or {"ok": True},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "register_uvm_slot_direct_layer_plans",
        lambda *_args, **_kwargs: calls.append("register") or {"ok": True},
    )

    runner.finalize_uvm_slot_bank_after_warmup()

    assert calls == ["mode8", "register"]


def test_slot_finalizer_freezes_contract_and_rejects_later_env_drift(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [(0, [(0, ["region-0"])])]
    _enable_registered_slot_plan(monkeypatch)
    _mock_complete_slot_registry(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    calls: list[str] = []
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: calls.append("mode8") or {"ok": True},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "register_uvm_slot_direct_layer_plans",
        lambda *_args, **_kwargs: calls.append("register") or {"ok": True},
    )

    runner.finalize_uvm_slot_bank_after_warmup()
    assert runner._uvm_slot_session_contract["registered_plan"] is True
    assert runner._uvm_slot_session_contract["slot_session_enabled"] is True
    assert calls == ["mode8", "register"]

    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN", "0")
    with pytest.raises(RuntimeError, match="contract changed"):
        runner.finalize_uvm_slot_bank_after_warmup()
    assert calls == ["mode8", "register"]


def test_slot_finalizer_allocates_two_banks_before_dual_plan_registration(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [
        (layer_id, [(0, [f"region-{layer_id}"])]) for layer_id in range(4)
    ]
    _enable_registered_slot_plan(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CUDA_FENCE", "1")
    _mock_complete_slot_registry(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda _groups, *, layer_id, device: (
            calls.append(("bank0", int(layer_id))) or {"ok": True}
        ),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_registered_plan_second_bank",
        lambda _groups, *, layer_id, device: (
            calls.append(("bank1", int(layer_id))) or {"ok": True}
        ),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "register_uvm_slot_direct_layer_plans",
        lambda specs, *, device: (
            calls.append(("register", len(specs))) or {"ok": True}
        ),
    )

    runner.finalize_uvm_slot_bank_after_warmup()

    assert calls == [("bank0", 0), ("bank1", 1), ("register", 4)]
    assert runner._uvm_slot_init_bank_result["dual_bank_event"] is True
    assert len(runner._uvm_slot_init_bank_result["bank_results"]) == 2
    assert dict(runner._uvm_slot_session_contract["environment"])[
        "VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT"
    ] == "1"
    assert dict(runner._uvm_slot_session_contract["environment"])[
        "VLLM_UVM_SLOT_DIRECT_CUDA_FENCE"
    ] == "1"


def test_slot_finalizer_rejects_cuda_fence_without_dual_bank_before_allocation(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [(0, [(0, ["region-0"])])]
    _enable_registered_slot_plan(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_CUDA_FENCE", "1")
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT", "0")
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("incomplete CUDA fence bundle must not allocate")
        ),
    )

    with pytest.raises(RuntimeError, match="requires registered plans and dual-bank"):
        runner.finalize_uvm_slot_bank_after_warmup()

    assert residency_module.get_uvm_slot_direct_session_contract() is None


def test_slot_finalizer_rejects_odd_dual_bank_layer_count_before_allocation(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [
        (layer_id, [(0, [f"region-{layer_id}"])]) for layer_id in range(3)
    ]
    _enable_registered_slot_plan(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT", "1")
    _mock_complete_slot_registry(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe dual-bank topology must not allocate")
        ),
    )

    with pytest.raises(RuntimeError, match="even number of layers"):
        runner.finalize_uvm_slot_bank_after_warmup()


def test_slot_finalizer_aborts_partial_dual_bank_initialization(
    monkeypatch: Any,
) -> None:
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [
        (layer_id, [(0, [f"region-{layer_id}"])]) for layer_id in range(4)
    ]
    _enable_registered_slot_plan(monkeypatch)
    monkeypatch.setenv("VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT", "1")
    _mock_complete_slot_registry(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_registered_plan_second_bank",
        lambda *_args, **_kwargs: {"ok": False, "reason": "injected"},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "register_uvm_slot_direct_layer_plans",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("plans must not register after bank-1 failure")
        ),
    )
    cleanup_calls: list[int] = []
    monkeypatch.setattr(
        gpu_model_runner_module,
        "close_uvm_slot_direct_policy_session",
        lambda *, device: cleanup_calls.append(int(device)) or {"ok": True},
    )

    with pytest.raises(RuntimeError, match="second-bank preallocation failed"):
        runner.finalize_uvm_slot_bank_after_warmup()

    assert cleanup_calls == [0]
    assert runner._uvm_slot_init_bank_done is False
    assert runner._uvm_slot_init_bank_result is None
    assert runner._uvm_slot_registered_plan_result is None
    assert runner._uvm_slot_session_contract is None


@pytest.mark.parametrize("failure_stage", ["preallocate", "register"])
def test_slot_init_failure_closes_partial_bank_and_resets_runner_state(
    monkeypatch: Any,
    failure_stage: str,
) -> None:
    _mock_complete_slot_registry(monkeypatch)
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [(0, [(0, ["region-0"])])]
    for name in (
        "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE",
        "VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD",
        "VLLM_UVM_SLOT_DIRECT_INIT_BANK",
        "VLLM_UVM_SLOT_DIRECT_REUSE_BANK",
        "VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES",
        "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION",
        "VLLM_UVM_SLOT_DIRECT_SHARED_PUSH",
        "VLLM_UVM_SLOT_DIRECT_64K_PTE",
        "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "slot_restore")
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: {
            "ok": failure_stage != "preallocate",
            "reason": "injected",
        },
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "register_uvm_slot_direct_layer_plans",
        lambda *_args, **_kwargs: {
            "ok": failure_stage != "register",
            "reason": "injected",
        },
    )
    cleanup_calls: list[int] = []
    monkeypatch.setattr(
        gpu_model_runner_module,
        "close_uvm_slot_direct_policy_session",
        lambda *, device: cleanup_calls.append(int(device)) or {"ok": True},
    )

    with pytest.raises(RuntimeError, match="injected"):
        runner.finalize_uvm_slot_bank_after_warmup()

    assert cleanup_calls == [0]
    assert runner._uvm_slot_init_bank_done is False
    assert runner._uvm_slot_init_bank_result is None
    assert runner._uvm_slot_registered_plan_result is None


def test_slot_init_cleanup_exception_still_resets_runner_state(
    monkeypatch: Any,
) -> None:
    _mock_complete_slot_registry(monkeypatch)
    runner = object.__new__(GPUModelRunner)
    runner._uvm_slot_init_bank_candidates = [(0, [(0, ["region-0"])])]
    runner._uvm_slot_init_bank_done = True
    runner._uvm_slot_init_bank_result = {"stale": True}
    runner._uvm_slot_registered_plan_result = {"stale": True}
    for name in (
        "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE",
        "VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD",
        "VLLM_UVM_SLOT_DIRECT_INIT_BANK",
        "VLLM_UVM_SLOT_DIRECT_REUSE_BANK",
        "VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES",
        "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION",
        "VLLM_UVM_SLOT_DIRECT_SHARED_PUSH",
        "VLLM_UVM_SLOT_DIRECT_64K_PTE",
        "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "slot_restore")
    # Reach the rollback path instead of the idempotent early return.
    runner._uvm_slot_init_bank_done = False
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "preallocate_uvm_expert_groups_to_slots",
        lambda *_args, **_kwargs: {"ok": False, "reason": "injected"},
    )

    def cleanup(*, device: int) -> dict[str, Any]:
        assert device == 0
        raise OSError("cleanup exploded")

    monkeypatch.setattr(
        gpu_model_runner_module,
        "close_uvm_slot_direct_policy_session",
        cleanup,
    )

    with pytest.raises(RuntimeError, match="slot init rollback raised"):
        runner.finalize_uvm_slot_bank_after_warmup()

    assert runner._uvm_slot_init_bank_done is False
    assert runner._uvm_slot_init_bank_result is None
    assert runner._uvm_slot_registered_plan_result is None


def test_shutdown_closes_frozen_slot_session_even_after_environment_drift(
    monkeypatch: Any,
) -> None:
    contract = {
        "slot_session_enabled": True,
        "device": 3,
        "access_mode": "slot_restore",
    }
    events: list[tuple[str, int]] = []
    monkeypatch.setattr(
        gpu_worker_module,
        "get_uvm_slot_direct_session_contract",
        lambda: contract,
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "validate_uvm_slot_direct_session_contract",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("environment changed")),
    )
    monkeypatch.setattr(
        gpu_worker_module.torch.cuda,
        "synchronize",
        lambda device: events.append(("sync", int(device))),
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "close_uvm_slot_direct_policy_session",
        lambda *, device: events.append(("close", int(device))) or {"ok": True},
    )
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace(
        ensure_kv_transfer_shutdown=lambda: events.append(("kv", -1))
    )
    worker.profiler = SimpleNamespace(
        shutdown=lambda: events.append(("profiler", -1))
    )

    with pytest.raises(RuntimeError, match="contract drifted"):
        worker.shutdown()
    assert events == [
        ("sync", 3),
        ("close", 3),
        ("kv", -1),
        ("profiler", -1),
    ]


def test_shutdown_sync_failure_retains_frozen_slot_session(
    monkeypatch: Any,
) -> None:
    contract = {
        "slot_session_enabled": True,
        "device": 2,
        "access_mode": "slot_restore",
    }
    sync_failure = OSError("sync exploded")
    events: list[str] = []
    close_calls = 0
    monkeypatch.setattr(
        gpu_worker_module,
        "get_uvm_slot_direct_session_contract",
        lambda: contract,
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "validate_uvm_slot_direct_session_contract",
        lambda **_kwargs: contract,
    )

    def synchronize(device: int) -> None:
        assert device == 2
        events.append("sync")
        raise sync_failure

    monkeypatch.setattr(gpu_worker_module.torch.cuda, "synchronize", synchronize)

    def close(**_kwargs: Any) -> dict[str, Any]:
        nonlocal close_calls
        close_calls += 1
        return {"ok": True}

    monkeypatch.setattr(
        gpu_worker_module,
        "close_uvm_slot_direct_policy_session",
        close,
    )
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace(
        ensure_kv_transfer_shutdown=lambda: events.append("kv")
    )
    worker.profiler = SimpleNamespace(shutdown=lambda: events.append("profiler"))

    with pytest.raises(RuntimeError, match="synchronize failed") as raised:
        worker.shutdown()

    assert events == ["sync", "kv", "profiler"]
    assert close_calls == 0
    slot_error = raised.value.__cause__
    assert isinstance(slot_error, RuntimeError)
    assert slot_error.__cause__ is sync_failure


def test_shutdown_combines_contract_and_close_failures_without_losing_causes(
    monkeypatch: Any,
) -> None:
    contract = {
        "slot_session_enabled": True,
        "device": 1,
        "access_mode": "slot_restore",
    }
    contract_failure = OSError("contract drifted")
    close_failure = RuntimeError("close exploded")
    events: list[str] = []
    monkeypatch.setattr(
        gpu_worker_module,
        "get_uvm_slot_direct_session_contract",
        lambda: contract,
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "validate_uvm_slot_direct_session_contract",
        lambda **_kwargs: (_ for _ in ()).throw(contract_failure),
    )

    def synchronize(_device: int) -> None:
        events.append("sync")

    def close(*, device: int) -> dict[str, Any]:
        events.append(f"close:{device}")
        raise close_failure

    monkeypatch.setattr(gpu_worker_module.torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(
        gpu_worker_module,
        "close_uvm_slot_direct_policy_session",
        close,
    )
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace(
        ensure_kv_transfer_shutdown=lambda: events.append("kv")
    )
    worker.profiler = None

    with pytest.raises(RuntimeError) as raised:
        worker.shutdown()

    assert "close failed" in str(raised.value)
    assert "contract drifted" in str(raised.value)
    assert events == ["sync", "close:1", "kv"]
    slot_error = raised.value.__cause__
    assert isinstance(slot_error, RuntimeError)
    assert slot_error.__cause__ is close_failure


def test_shutdown_does_not_open_cleanup_from_mutable_environment(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "slot_restore")
    monkeypatch.setattr(
        gpu_worker_module,
        "get_uvm_slot_direct_session_contract",
        lambda: None,
    )
    monkeypatch.setattr(
        gpu_worker_module,
        "close_uvm_slot_direct_policy_session",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("mutable environment must not trigger slot cleanup")
        ),
    )
    events: list[str] = []
    worker = object.__new__(Worker)
    worker.model_runner = SimpleNamespace(
        ensure_kv_transfer_shutdown=lambda: events.append("runner")
    )
    worker.profiler = None

    worker.shutdown()
    assert events == ["runner"]


def test_post_warmup_remote_read_runs_explicit_expert_placement(
    monkeypatch: Any,
) -> None:
    events: list[str] = []

    class FakeModelRunner:
        def run_uvm_init_time_weight_placement(self) -> None:
            events.append("experts")

    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_STAGE", "post_warmup")
    monkeypatch.setenv("VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE", "1")
    monkeypatch.setenv("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE", "remote_read")

    worker = object.__new__(Worker)
    worker.model_runner = FakeModelRunner()

    worker._run_uvm_post_warmup_rebalance()

    assert events == ["experts"]


def test_remote_only_reconciles_after_kv_before_warmup(monkeypatch: Any) -> None:
    from vllm.model_executor.models import utils as offload_utils

    events: list[str] = []
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_phase_mode",
        lambda: "remote_only",
    )
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_profile_all_remote_active",
        lambda: False,
    )
    monkeypatch.setattr(
        offload_utils,
        "reconcile_cpu_offload_uvm_original_va_remote_graph_residency",
        lambda: events.append("reconcile")
        or {
            "ok": True,
            "cold_bytes": 6,
            "hot_bytes": 22,
            "free_bytes": 512,
        },
    )

    worker = object.__new__(Worker)
    worker._reconcile_uvm_original_va_remote_before_warmup()

    assert events == ["reconcile"]


def test_prefill_remote_decode_reconciles_hot_experts_before_warmup(
    monkeypatch: Any,
) -> None:
    from vllm.model_executor.models import utils as offload_utils

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_phase_mode",
        lambda: "prefill_remote_decode",
    )
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_profile_all_remote_active",
        lambda: False,
    )
    monkeypatch.setattr(
        offload_utils,
        "reconcile_cpu_offload_uvm_original_va_sealed_hot_residency",
        lambda **kwargs: calls.append(kwargs)
        or {
            "ok": True,
            "hot_physical_residency_before": {"cpu_bytes": 6},
            "hot_physical_residency_after": {"gpu_bytes": 22},
        },
    )
    monkeypatch.setattr(
        offload_utils,
        "refresh_cpu_offload_uvm_original_va_full_dual_view",
        _unexpected_call,
    )

    worker = object.__new__(Worker)
    worker._reconcile_uvm_original_va_remote_before_warmup()

    assert calls == [{"preserve_allocator_cache": True}]


def test_phase_off_opt_in_reconciles_sealed_hot_experts_after_kv(
    monkeypatch: Any,
) -> None:
    from vllm.model_executor.models import utils as offload_utils

    calls: list[dict[str, Any]] = []
    monkeypatch.setenv("VLLM_UVM_POST_KV_HOT_RECONCILE", "1")
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_phase_mode",
        lambda: "off",
    )
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_profile_all_remote_active",
        lambda: False,
    )
    monkeypatch.setattr(
        offload_utils,
        "reconcile_cpu_offload_uvm_original_va_sealed_hot_residency",
        lambda **kwargs: calls.append(kwargs)
        or {
            "ok": True,
            "hot_bytes": 22,
            "hot_physical_residency_before": {"cpu_bytes": 6},
            "hot_physical_residency_after": {"gpu_bytes": 22},
            "free_bytes_before": 512,
            "free_bytes_after": 384,
            "allocator_reserved_after": 128,
        },
    )
    monkeypatch.setattr(
        offload_utils,
        "reconcile_cpu_offload_uvm_original_va_remote_graph_residency",
        _unexpected_call,
    )

    worker = object.__new__(Worker)
    worker._reconcile_uvm_original_va_remote_before_warmup()

    assert calls == [{"preserve_allocator_cache": True}]


def test_profile_all_remote_defers_phase_mode_reconciliation_until_capture(
    monkeypatch: Any,
) -> None:
    from vllm.model_executor.models import utils as offload_utils

    events: list[str] = []
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_phase_mode",
        lambda: "prefill_graph_remote_decode",
    )
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_profile_all_remote_active",
        lambda: True,
    )
    monkeypatch.setattr(
        offload_utils,
        "reconcile_cpu_offload_uvm_original_va_remote_graph_residency",
        lambda: events.append("reconcile")
        or {
            "ok": True,
            "cold_bytes": 6,
            "hot_bytes": 22,
            "free_bytes": 512,
        },
    )

    worker = object.__new__(Worker)
    worker._reconcile_uvm_original_va_remote_before_warmup()

    assert events == []


def test_explicit_kv_profiles_with_all_remote_topology_first(
    monkeypatch: Any,
) -> None:
    from vllm.model_executor.models import utils as offload_utils

    events: list[str] = []
    monkeypatch.setattr(
        offload_utils,
        "prepare_cpu_offload_uvm_original_va_profile_topology",
        lambda: events.append("prepare")
        or {
            "ok": True,
            "reason": "prepared",
            "hot_bytes": 22,
            "hot_range_count": 3,
        },
    )
    worker = object.__new__(Worker)
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=4096)
    worker.init_snapshot = SimpleNamespace(free_memory=8192)
    worker.model_runner = SimpleNamespace(
        profile_run=lambda: events.append("profile")
    )

    assert worker.determine_available_memory() == 4096
    assert events == ["prepare", "profile"]


def test_post_capture_sampler_warmup_preserves_dual_graphs(
    monkeypatch: Any,
) -> None:
    from vllm.model_executor.models import utils as offload_utils

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_phase_mode",
        lambda: "prefill_graph_remote_decode",
    )
    monkeypatch.setattr(
        offload_utils,
        "prepare_cpu_offload_uvm_original_va_profile_topology",
        lambda **kwargs: calls.append(kwargs)
        or {
            "ok": True,
            "reason": "prepared",
            "hot_bytes": 22,
            "hot_range_count": 3,
        },
    )

    worker = object.__new__(Worker)
    worker._prepare_uvm_original_va_post_capture_warmup()

    assert calls == [
        {
            "transition_reason": "post_capture_sampler_all_remote",
            "preserve_remote_graphs": True,
        }
    ]


def test_fixed_tier_full_dual_view_skips_redundant_post_capture_remote(
    monkeypatch: Any,
) -> None:
    from vllm.model_executor.models import utils as offload_utils

    monkeypatch.setenv("VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW", "1")
    monkeypatch.setenv(
        "VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE", "1"
    )
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_phase_mode",
        lambda: "prefill_graph_remote_decode",
    )
    monkeypatch.setattr(
        offload_utils,
        "prepare_cpu_offload_uvm_original_va_profile_topology",
        _unexpected_call,
    )

    worker = object.__new__(Worker)
    worker._prepare_uvm_original_va_post_capture_warmup()


def test_fixed_tier_retained_prefill_graph_skips_allocator_reserve(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW", "1")
    monkeypatch.setenv(
        "VLLM_UVM_PREFILL_GRAPH_CAPTURE_BEFORE_HOT_RESTORE", "1"
    )
    monkeypatch.setenv(
        "VLLM_UVM_PREFILL_ALLOCATOR_RESERVE_WARMUP_TOKENS", "16384"
    )

    worker = object.__new__(Worker)
    worker._warm_up_uvm_prefill_allocator_reserve()


def test_post_capture_sampler_warmup_restores_serving_residency(
    monkeypatch: Any,
) -> None:
    from vllm.model_executor.models import utils as offload_utils

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_phase_mode",
        lambda: "prefill_graph_remote_decode",
    )
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_profile_all_remote_active",
        lambda: True,
    )
    monkeypatch.setattr(
        offload_utils,
        "reconcile_cpu_offload_uvm_original_va_remote_graph_residency",
        lambda **kwargs: calls.append(kwargs)
        or {
            "ok": True,
            "cold_bytes": 6,
            "hot_bytes": 22,
            "free_bytes": 512,
        },
    )

    worker = object.__new__(Worker)
    worker._restore_uvm_original_va_after_post_capture_warmup()

    assert calls == [{"preserve_allocator_cache": True}]


def test_eager_phase_sampler_warmup_restores_serving_residency(
    monkeypatch: Any,
) -> None:
    from vllm.model_executor.models import utils as offload_utils

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_phase_mode",
        lambda: "prefill_remote_decode",
    )
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_profile_all_remote_active",
        lambda: True,
    )
    monkeypatch.setattr(
        offload_utils,
        "reconcile_cpu_offload_uvm_original_va_remote_graph_residency",
        lambda **kwargs: calls.append(kwargs)
        or {
            "ok": True,
            "cold_bytes": 6,
            "hot_bytes": 22,
            "free_bytes": 512,
        },
    )

    worker = object.__new__(Worker)
    worker._restore_uvm_original_va_after_post_capture_warmup()

    assert calls == [{"preserve_allocator_cache": True}]


def test_remote_only_post_warmup_restores_serving_residency(
    monkeypatch: Any,
) -> None:
    from vllm.model_executor.models import utils as offload_utils

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_phase_mode",
        lambda: "remote_only",
    )
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_original_va_profile_all_remote_active",
        lambda: False,
    )
    monkeypatch.setattr(
        offload_utils,
        "reconcile_cpu_offload_uvm_original_va_remote_graph_residency",
        lambda **kwargs: calls.append(kwargs)
        or {
            "ok": True,
            "cold_bytes": 6,
            "hot_bytes": 22,
            "free_bytes": 512,
            "allocator_reserved_before": 64,
            "allocator_reserved_after": 64,
        },
    )

    worker = object.__new__(Worker)
    worker._restore_uvm_original_va_after_post_capture_warmup()

    assert calls == [{"preserve_allocator_cache": True}]


class _FakeExpertModule:
    uvm_weight_executor = "cpp_batch"

    def __init__(
        self,
        name: str,
        region_ids: list[str],
        events: list[str],
        *,
        release_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.region_ids = region_ids
        self.events = events
        self.release_error = release_error

    def run_uvm_init_weight_offload(self) -> None:
        self.events.append(f"release:{self.name}")
        if self.release_error is not None:
            raise self.release_error

    def get_uvm_init_hot_expert_region_ids(self) -> list[str]:
        self.events.append(f"collect:{self.name}")
        return list(self.region_ids)

    def set_uvm_init_hot_expert_state(self, state: str) -> None:
        self.events.append(f"state:{self.name}:{state}")


def _make_init_weight_placement_runner(
    modules: list[object],
    *,
    matched_offload: Any | None = None,
) -> Any:
    return SimpleNamespace(
        _uvm_init_weight_placement_done=False,
        _run_uvm_matched_cpu_offload_manifest=(
            matched_offload if matched_offload is not None else lambda: None
        ),
        _env_flag_enabled=lambda _name, default=False: True,
        get_model=lambda: SimpleNamespace(modules=lambda: modules),
    )


def _patch_init_weight_placement_phase(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        gpu_model_runner_module,
        "uvm_enabled_allocation_phase",
        lambda _reason: nullcontext(),
    )


def _unexpected_call(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("unexpected call")


def test_init_weight_placement_releases_all_modules_before_one_hot_batch(
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    modules = [
        _FakeExpertModule("a", ["hot-a"], events),
        _FakeExpertModule("b", ["hot-b"], events),
    ]
    runner = _make_init_weight_placement_runner(modules)

    def prepare(
        region_ids: list[str],
        *,
        device: int,
        executor: str,
    ) -> dict[str, object]:
        assert events == [
            "release:a",
            "release:b",
            "collect:a",
            "collect:b",
        ]
        events.append(f"prepare:{','.join(region_ids)}:{device}:{executor}")
        return {"ok": True}

    def commit(
        region_ids: list[str],
        *,
        synchronized: bool,
    ) -> dict[str, object]:
        events.append(f"commit:{','.join(region_ids)}:{synchronized}")
        return {"ok": True}

    _patch_init_weight_placement_phase(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "prepare_uvm_regions_for_gpu_placement",
        prepare,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "commit_uvm_regions_gpu_placement",
        commit,
    )
    monkeypatch.setattr(gpu_model_runner_module.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda device: events.append(f"sync:{device}"),
    )

    GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert events == [
        "release:a",
        "release:b",
        "collect:a",
        "collect:b",
        "prepare:hot-a,hot-b:2:cpp_batch",
        "sync:2",
        "commit:hot-a,hot-b:True",
        "state:a:gpu",
        "state:b:gpu",
    ]
    assert runner._uvm_init_weight_placement_done is True


def test_init_weight_placement_is_idempotent(monkeypatch: Any) -> None:
    events: list[str] = []
    module = _FakeExpertModule("a", ["hot"], events)
    runner = _make_init_weight_placement_runner(
        [module],
        matched_offload=lambda: events.append("matched"),
    )

    _patch_init_weight_placement_phase(monkeypatch)
    monkeypatch.setattr(gpu_model_runner_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda _device: events.append("sync"),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "prepare_uvm_regions_for_gpu_placement",
        lambda *_args, **_kwargs: events.append("prepare") or {"ok": True},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "commit_uvm_regions_gpu_placement",
        lambda *_args, **_kwargs: events.append("commit") or {"ok": True},
    )

    GPUModelRunner.run_uvm_init_time_weight_placement(runner)
    GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert events == [
        "matched",
        "release:a",
        "collect:a",
        "prepare",
        "sync",
        "commit",
        "state:a:gpu",
        "matched",
    ]


def test_init_weight_placement_without_hot_experts_does_not_sync(
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    runner = _make_init_weight_placement_runner([_FakeExpertModule("cold", [], events)])

    _patch_init_weight_placement_phase(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", _unexpected_call
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "synchronize", _unexpected_call
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "prepare_uvm_regions_for_gpu_placement",
        _unexpected_call,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "commit_uvm_regions_gpu_placement",
        _unexpected_call,
    )

    GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert events == ["release:cold", "collect:cold"]
    assert runner._uvm_init_weight_placement_done is True


def test_init_weight_placement_manifest_proves_release_and_residency(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "offload_plan": [{"layer_id": 0, "expert_id": 1}],
                "hot_plan": [{"layer_id": 0, "expert_id": 2}],
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "placement.json"
    monkeypatch.setenv(
        "VLLM_UVM_INIT_WEIGHT_PLACEMENT_MANIFEST", str(output_path)
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "synchronize", lambda: None
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "current_device", lambda: 0
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "snapshot_uvm_region_registry",
        lambda region_ids: {
            "ok": True,
            "requested_region_count": len(region_ids),
            "missing_region_ids": [],
            "regions": [
                {
                    "region_id": "remote",
                    "base": 0x100000,
                    "length": 0x80000,
                    "state": "cpu",
                },
                {
                    "region_id": "hot",
                    "base": 0x180000,
                    "length": 0x80000,
                    "state": "gpu",
                },
            ],
        },
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "query_uvm_regions_physical_residency",
        lambda *_args, **_kwargs: {
            "ok": True,
            "queried_bytes": 0x80000,
            "cpu_bytes": 0x80000,
            "gpu_bytes": 0,
            "unknown_bytes": 0,
        },
    )
    audit = {
        "region_id": "remote",
        "ok": True,
        "remote_read": True,
        "registered_length": 0x80000,
        "length": 0x80000,
        "cpu_resident_bytes": 0x80000,
        "remote_mapped_bytes": 0x80000,
        "remote_map_failure_blocks": 0,
        "app_state_after": "cpu",
    }
    snapshot = {
        "layer_id": 0,
        "plan_file": str(plan_path),
        "experts": [
            {"layer_id": 0, "expert_id": 1, "placement": "remote"},
            {"layer_id": 0, "expert_id": 2, "placement": "hot"},
        ],
        "region_owners": [
            {
                "region_id": "remote",
                "layer_id": 0,
                "expert_id": 1,
                "placement": "remote",
            },
            {
                "region_id": "hot",
                "layer_id": 0,
                "expert_id": 2,
                "placement": "hot",
            },
        ],
        "release_audits": [audit],
    }
    module = SimpleNamespace(
        get_uvm_init_weight_placement_snapshot=lambda: snapshot
    )

    GPUModelRunner._write_uvm_init_weight_placement_manifest(
        SimpleNamespace(), [module]
    )

    manifest = json.loads(output_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["complete"] is True
    assert manifest["remote_experts"] == [[0, 1]]
    assert manifest["remote_region_bytes"] == 0x80000
    assert manifest["release_time_remote_mapping_verified"] is True


def test_init_weight_placement_prepare_failure_rolls_back_without_sync(
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    runner = _make_init_weight_placement_runner(
        [_FakeExpertModule("a", ["hot"], events)]
    )

    _patch_init_weight_placement_phase(monkeypatch)
    monkeypatch.setattr(gpu_model_runner_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda, "synchronize", _unexpected_call
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "prepare_uvm_regions_for_gpu_placement",
        lambda *_args, **_kwargs: events.append("prepare") or {"ok": False},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "commit_uvm_regions_gpu_placement",
        lambda *_args, synchronized: (
            events.append(f"commit:{synchronized}") or {"ok": False}
        ),
    )

    with pytest.raises(RuntimeError, match="Hot Expert prefetch failed"):
        GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert events == [
        "release:a",
        "collect:a",
        "prepare",
        "commit:False",
        "state:a:unknown",
    ]
    assert runner._uvm_init_weight_placement_done is False


def test_init_weight_placement_sync_failure_marks_hot_unknown(
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    runner = _make_init_weight_placement_runner(
        [_FakeExpertModule("a", ["hot"], events)]
    )

    _patch_init_weight_placement_phase(monkeypatch)
    monkeypatch.setattr(gpu_model_runner_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "prepare_uvm_regions_for_gpu_placement",
        lambda *_args, **_kwargs: events.append("prepare") or {"ok": True},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "commit_uvm_regions_gpu_placement",
        lambda *_args, synchronized: (
            events.append(f"commit:{synchronized}") or {"ok": False}
        ),
    )

    def fail_sync(_device: int) -> None:
        events.append("sync")
        raise RuntimeError("mock sync failure")

    monkeypatch.setattr(gpu_model_runner_module.torch.cuda, "synchronize", fail_sync)

    with pytest.raises(RuntimeError, match="mock sync failure"):
        GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert events == [
        "release:a",
        "collect:a",
        "prepare",
        "sync",
        "commit:False",
        "state:a:unknown",
    ]
    assert runner._uvm_init_weight_placement_done is False


def test_init_weight_placement_commit_failure_marks_hot_unknown(
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    runner = _make_init_weight_placement_runner(
        [_FakeExpertModule("a", ["hot"], events)]
    )

    _patch_init_weight_placement_phase(monkeypatch)
    monkeypatch.setattr(gpu_model_runner_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda _device: events.append("sync"),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "prepare_uvm_regions_for_gpu_placement",
        lambda *_args, **_kwargs: events.append("prepare") or {"ok": True},
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "commit_uvm_regions_gpu_placement",
        lambda *_args, synchronized: (
            events.append(f"commit:{synchronized}") or {"ok": False}
        ),
    )

    with pytest.raises(RuntimeError, match="Hot Expert placement commit failed"):
        GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert events == [
        "release:a",
        "collect:a",
        "prepare",
        "sync",
        "commit:True",
        "commit:False",
        "state:a:unknown",
    ]
    assert runner._uvm_init_weight_placement_done is False


def test_init_weight_placement_cold_failure_skips_hot_prefetch(
    monkeypatch: Any,
) -> None:
    events: list[str] = []
    modules = [
        _FakeExpertModule("a", ["hot-a"], events),
        _FakeExpertModule(
            "b",
            ["hot-b"],
            events,
            release_error=RuntimeError("Cold Expert release failed"),
        ),
    ]
    runner = _make_init_weight_placement_runner(modules)

    _patch_init_weight_placement_phase(monkeypatch)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "prepare_uvm_regions_for_gpu_placement",
        _unexpected_call,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "commit_uvm_regions_gpu_placement",
        _unexpected_call,
    )

    with pytest.raises(RuntimeError, match="Cold Expert release failed"):
        GPUModelRunner.run_uvm_init_time_weight_placement(runner)

    assert events == ["release:a", "release:b"]
    assert runner._uvm_init_weight_placement_done is False


def test_incremental_expert_worker_protocol_is_minimal(
    monkeypatch: Any,
) -> None:
    class ExpertModule:
        layer_name = "layer.7"
        uvm_incremental_expert_offload_enable = True

        @staticmethod
        def uvm_next_incremental_expert(expert_class: str) -> int | None:
            return 6 if expert_class == "cold" else None

        @staticmethod
        def has_remaining_uvm_incremental_experts(expert_class: str) -> bool:
            return expert_class == "cold"

        @staticmethod
        def run_uvm_incremental_weight_offload(
            *, allow_warm: bool, selected_global_expert_id: int | None
        ) -> dict[str, object]:
            assert allow_warm is False
            return {
                "ok": True,
                "released_experts": [selected_global_expert_id],
            }

        @staticmethod
        def uvm_next_incremental_restore_expert() -> int:
            return 6

        @staticmethod
        def run_uvm_incremental_weight_restore(
            *, selected_global_expert_id: int
        ) -> dict[str, object]:
            return {
                "ok": True,
                "restored_experts": [selected_global_expert_id],
                "restored_count": 1,
            }

    module = ExpertModule()
    runner = object.__new__(GPUModelRunner)
    runner.get_model = lambda: SimpleNamespace(modules=lambda: [module])  # type: ignore[method-assign]
    runner._env_flag_enabled = (  # type: ignore[method-assign]
        lambda name, **_kwargs: name
        == "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE"
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "uvm_enabled_allocation_phase",
        lambda _reason: nullcontext(),
    )

    candidates = GPUModelRunner.get_uvm_incremental_expert_offload_candidates(
        runner, "cold"
    )
    assert candidates == {"ok": True, "candidates": {"layer.7": 6}}
    assert GPUModelRunner.run_uvm_incremental_expert_offload(
        runner,
        False,
        {"layer.7": 6},
    ) == {
        "ok": True,
        "allow_warm": False,
        "selected_layer_count": 1,
        "visited_layer_count": 1,
        "released_expert_count": 1,
        "results": [
            {
                "layer_key": "layer.7",
                "selected_global_expert_id": 6,
                "ok": True,
                "released_experts": [6],
            }
        ],
    }
    assert GPUModelRunner.get_uvm_incremental_expert_restore_candidates(
        runner
    ) == {"ok": True, "candidates": {"layer.7": 6}}
    assert GPUModelRunner.run_uvm_incremental_expert_restore(
        runner,
        {"layer.7": 6},
    ) == {
        "ok": True,
        "reason": "restored",
        "selected_layer_count": 1,
        "visited_layer_count": 1,
        "restored_expert_count": 1,
        "results": [
            {
                "layer_key": "layer.7",
                "selected_global_expert_id": 6,
                "ok": True,
                "restored_experts": [6],
                "restored_count": 1,
            }
        ],
    }


def test_explicit_incremental_wave_does_not_enable_init_stage_m() -> None:
    released: list[tuple[int, int]] = []
    layer = SimpleNamespace(
        uvm_init_weight_offload_enable=False,
        uvm_init_weight_offload_mode="driver_release_cpu",
        uvm_incremental_expert_offload_enable=True,
        uvm_expert_envelope_enable=True,
        _uvm_incremental_offload_in_progress=False,
        _uvm_incremental_selected_global_expert_id=None,
        _uvm_init_weight_offload_done=False,
        _uvm_expert_records={(3, 7): object()},
        _uvm_offloaded_global_expert_ids=set(),
        _uvm_remote_read_global_expert_ids=set(),
        _uvm_incremental_offloaded_global_expert_ids=set(),
    )
    layer._uvm_layer_id = lambda: 3
    layer._uvm_static_alias_owned_experts = lambda _layer_id: set()
    layer._uvm_all_local_expert_pairs = lambda: [(7, 0)]
    layer._refresh_uvm_init_weight_plan = lambda: None
    layer._uvm_compile_expert_records = lambda _layer_id: (_ for _ in ()).throw(
        AssertionError("precompiled expert records must be reused")
    )
    layer._uvm_stage_m_uses_remote_read = lambda: True
    layer._uvm_release_expert_record = lambda **kwargs: (
        released.append((int(kwargs["layer_id"]), int(kwargs["global_expert_id"])))
        or {"ok": True}
    )
    layer.run_uvm_init_weight_offload = MethodType(
        FusedMoE.run_uvm_init_weight_offload,
        layer,
    )

    # Merely enabling runtime waves must not turn the ordinary warmup Stage-M
    # invocation back on after static-alias finalize released the base plan.
    FusedMoE.run_uvm_init_weight_offload(layer)
    assert released == []

    result = FusedMoE.run_uvm_incremental_weight_offload(
        layer,
        allow_warm=False,
        selected_global_expert_id=7,
    )

    assert result["ok"] is True
    assert result["released_experts"] == [7]
    assert released == [(3, 7)]
    assert layer._uvm_remote_read_global_expert_ids == {7}
    assert layer._uvm_incremental_offloaded_global_expert_ids == {7}


def test_runtime_expert_registry_skips_static_alias_owned_ranges(
    monkeypatch: Any,
) -> None:
    registrations: list[tuple[str, int, int]] = []
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.layer.register_uvm_region",
        lambda region_id, base, length, **_kwargs: registrations.append(
            (str(region_id), int(base), int(length))
        ),
    )
    layer = SimpleNamespace(
        uvm_expert_envelope_bytes=0x80000,
        _uvm_static_alias_owned_experts=lambda _layer_id: {1},
        _uvm_all_local_expert_pairs=lambda: [(0, 0), (1, 1)],
        _uvm_expert_weight_slices=lambda local_id: [
            (
                0x100000 + int(local_id) * 0x80000,
                0x80000,
                (0x100000, 0x200000),
            )
        ],
    )

    FusedMoE._uvm_compile_expert_records(layer, 3)

    assert set(layer._uvm_expert_records) == {(3, 0), (3, 1)}
    assert layer._uvm_static_alias_owned_global_expert_ids == {1}
    assert registrations == [
        ("weights:3:0:envelope:100000:80000", 0x100000, 0x80000)
    ]


def test_incremental_wave_rejects_static_alias_owned_expert() -> None:
    layer = SimpleNamespace(
        uvm_incremental_expert_offload_enable=True,
        _uvm_remote_read_global_expert_ids=set(),
        _uvm_offloaded_global_expert_ids=set(),
        _uvm_stage_m_uses_remote_read=lambda: True,
        _uvm_layer_id=lambda: 3,
        _uvm_static_alias_owned_experts=lambda _layer_id: {7},
    )

    result = FusedMoE.run_uvm_incremental_weight_offload(
        layer,
        allow_warm=False,
        selected_global_expert_id=7,
    )

    assert result == {
        "ok": False,
        "reason": "expert_owned_by_static_alias_plan",
        "layer_id": 3,
        "selected_global_expert_id": 7,
        "released_experts": [],
    }


def test_incremental_expert_wave_rejects_partial_layer_sets() -> None:
    class ExpertModule:
        def __init__(self, layer: int, candidate: int | None) -> None:
            self.layer_name = f"layer.{layer}"
            self._candidate = candidate

        def uvm_next_incremental_expert(self, _expert_class: str) -> int | None:
            return self._candidate

        def uvm_next_incremental_restore_expert(self) -> int | None:
            return self._candidate

        @staticmethod
        def run_uvm_incremental_weight_offload(**_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("an unbalanced wave must not execute")

        @staticmethod
        def run_uvm_incremental_weight_restore(**_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("an unbalanced wave must not execute")

        @staticmethod
        def has_remaining_uvm_incremental_experts(_expert_class: str) -> bool:
            return False

    modules = [ExpertModule(0, 4), ExpertModule(1, None)]
    runner = object.__new__(GPUModelRunner)
    runner.get_model = lambda: SimpleNamespace(  # type: ignore[method-assign]
        modules=lambda: modules
    )
    runner._env_flag_enabled = (  # type: ignore[method-assign]
        lambda *_args, **_kwargs: True
    )

    assert GPUModelRunner.get_uvm_incremental_expert_offload_candidates(
        runner,
        "cold",
    ) == {"ok": True, "candidates": {}}
    assert GPUModelRunner.get_uvm_incremental_expert_restore_candidates(
        runner
    ) == {"ok": False, "candidates": {}}

    offload = GPUModelRunner.run_uvm_incremental_expert_offload(
        runner,
        False,
        {"layer.0": 4},
    )
    assert offload["ok"] is False
    assert offload["reason"] == "unbalanced_expert_layer_selection"

    restore = GPUModelRunner.run_uvm_incremental_expert_restore(
        runner,
        {"layer.0": 4},
    )
    assert restore["ok"] is False
    assert restore["reason"] == "unbalanced_expert_layer_selection"


def test_incremental_restore_rolls_back_partial_layer_success(
    monkeypatch: Any,
) -> None:
    events: list[str] = []

    class ExpertModule:
        def __init__(self, layer: int, restore_ok: bool) -> None:
            self.layer_name = f"layer.{layer}"
            self._restore_ok = restore_ok

        def run_uvm_incremental_weight_restore(
            self,
            *,
            selected_global_expert_id: int,
        ) -> dict[str, Any]:
            events.append(f"restore:{self.layer_name}")
            return {
                "ok": self._restore_ok,
                "restored_experts": (
                    [selected_global_expert_id] if self._restore_ok else []
                ),
            }

        def run_uvm_incremental_weight_offload(
            self,
            *,
            allow_warm: bool,
            selected_global_expert_id: int,
        ) -> dict[str, Any]:
            assert allow_warm is True
            events.append(f"rollback:{self.layer_name}")
            return {
                "ok": True,
                "released_experts": [selected_global_expert_id],
            }

    modules = [ExpertModule(0, True), ExpertModule(1, False)]
    runner = object.__new__(GPUModelRunner)
    runner.get_model = lambda: SimpleNamespace(  # type: ignore[method-assign]
        modules=lambda: modules
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "uvm_enabled_allocation_phase",
        lambda _reason: nullcontext(),
    )

    result = GPUModelRunner.run_uvm_incremental_expert_restore(
        runner,
        {"layer.0": 4, "layer.1": 7},
    )

    assert result["ok"] is False
    assert result["rollback_ok"] is True
    assert result["net_restored_expert_count"] == 0
    assert events == ["restore:layer.0", "restore:layer.1", "rollback:layer.0"]


def test_incremental_restore_accepts_idempotently_local_rollback_layer(
    monkeypatch: Any,
) -> None:
    class ExpertModule:
        def __init__(self, layer: int, already_local: bool) -> None:
            self.layer_name = f"layer.{layer}"
            self._already_local = already_local

        def run_uvm_incremental_weight_restore(
            self,
            *,
            selected_global_expert_id: int,
        ) -> dict[str, Any]:
            if self._already_local:
                return {
                    "ok": False,
                    "reason": "expert_not_incrementally_offloaded",
                    "already_local": True,
                    "restored_experts": [],
                }
            return {
                "ok": True,
                "restored_experts": [selected_global_expert_id],
            }

        @staticmethod
        def run_uvm_incremental_weight_offload(**_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("a satisfied rollback must not be reversed")

    modules = [ExpertModule(0, False), ExpertModule(1, True)]
    runner = object.__new__(GPUModelRunner)
    runner.get_model = lambda: SimpleNamespace(  # type: ignore[method-assign]
        modules=lambda: modules
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "uvm_enabled_allocation_phase",
        lambda _reason: nullcontext(),
    )

    result = GPUModelRunner.run_uvm_incremental_expert_restore(
        runner,
        {"layer.0": 4, "layer.1": 7},
    )

    assert result["ok"] is True
    assert result["reason"] == "restored"
    assert "rollback_results" not in result


def test_incremental_restore_only_undoes_runtime_wave_and_prefers_heat() -> None:
    layer = SimpleNamespace(
        _uvm_incremental_offloaded_global_expert_ids={2, 5},
        _uvm_remote_read_global_expert_ids={1, 2, 5},
        _uvm_offloaded_global_expert_ids=set(),
        _uvm_dynamic_expert_heat_counts=torch.tensor([0, 0, 7, 0, 0, 19]),
        _uvm_expert_records={(7, 1): object(), (7, 2): object(), (7, 5): object()},
    )
    layer._uvm_stage_m_uses_remote_read = lambda: True
    layer._uvm_layer_id = lambda: 7
    layer._uvm_restore_expert_record = lambda **_kwargs: {"ok": True}

    assert FusedMoE.uvm_next_incremental_restore_expert(layer) == 5
    result = FusedMoE.run_uvm_incremental_weight_restore(
        layer,
        selected_global_expert_id=5,
    )

    assert result["ok"] is True
    assert result["restored_experts"] == [5]
    assert layer._uvm_incremental_offloaded_global_expert_ids == {2}
    assert layer._uvm_remote_read_global_expert_ids == {1, 2}

    already_local = FusedMoE.run_uvm_incremental_weight_restore(
        layer,
        selected_global_expert_id=7,
    )
    assert already_local["ok"] is False
    assert already_local["already_local"] is True


def _make_worker_resize_runner() -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.kv_cache_config = SimpleNamespace(num_blocks=7)
    runner._env_flag_enabled = lambda *_args, **_kwargs: False
    return runner


def test_worker_exact_kv_resize_runs_shrink_before_grow(
    monkeypatch: Any,
) -> None:
    runner = _make_worker_resize_runner()
    events: list[tuple[str, tuple[int, ...]] | tuple[str, int]] = []

    def collect(block_ids: list[int]) -> list[tuple[int, int]]:
        ids = tuple(block_ids)
        events.append(("collect", ids))
        return [
            (0x100000 + block_id * 0x1000, 0x1000)
            for block_id in ids
        ]

    def offload(
        ranges: list[tuple[int, int]],
        **_kwargs: object,
    ) -> tuple[int, str, int, int]:
        events.append(("offload", len(ranges)))
        return len(ranges), "test", 0, 0

    def prefetch(
        ranges: list[tuple[int, int]],
        _device: int,
        **_kwargs: object,
    ) -> tuple[int, str, int]:
        events.append(("prefetch", len(ranges)))
        return len(ranges), "test", 0

    def advise(
        ranges: list[tuple[int, int]],
        _device: int,
    ) -> int:
        events.append(("advise_gpu", len(ranges)))
        return len(ranges)

    monkeypatch.setattr(
        gpu_model_runner_module,
        "collect_uvm_kv_cache_physical_ranges_for_blocks",
        collect,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_offload_ranges_to_cpu_with_executor",
        offload,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_prefetch_ranges_to_device_with_executor",
        prefetch,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_advise_ranges_preferred_location",
        advise,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "current_device",
        lambda: 0,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda device: events.append(("sync", int(device))),
    )

    result = runner.resize_uvm_kv_cache_target(
        3,
        5,
        [4, 5, 6],
    )

    assert result == {"ok": True, "made_progress": True}
    assert events == [
        ("collect", (6,)),
        ("offload", 1),
        ("collect", (4, 5)),
        ("advise_gpu", 2),
        ("prefetch", 2),
        ("sync", 0),
    ]


@pytest.mark.parametrize("prepare_before_grow", [False, True])
def test_worker_kv_resize_refreshes_full_dual_view_after_sync(
    monkeypatch: Any,
    prepare_before_grow: bool,
) -> None:
    runner = _make_worker_resize_runner()
    runner._env_flag_enabled = lambda name, default=False: (  # type: ignore[method-assign]
        name == "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW"
        or (prepare_before_grow and name == "VLLM_UVM_KV_PREPARE_DUAL_BEFORE_GROW")
    )
    events: list[tuple[str, int] | str] = []
    monkeypatch.setattr(
        gpu_model_runner_module,
        "collect_uvm_kv_cache_physical_ranges_for_blocks",
        lambda block_ids: [(0x100000, 0x1000)] if block_ids else [],
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_advise_ranges_preferred_location",
        lambda ranges, _device: len(ranges),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda ranges, *_args, **_kwargs: (
            events.append("prefetch") or len(ranges), "test", 0
        ),
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "current_device",
        lambda: 0,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda device: events.append(("sync", int(device))),
    )
    refresh_result = {
        "ok": True,
        "state_before": "sealed",
        "refresh": {"refresh_dual": True},
    }
    monkeypatch.setattr(
        offload_utils,
        "refresh_cpu_offload_uvm_original_va_full_dual_view",
        lambda **kwargs: events.append(
            "refresh_full" if kwargs == {"nested_refresh": False} else "refresh"
        ) or refresh_result,
    )

    result = runner.resize_uvm_kv_cache_target(3, 5, [])

    expected = {
        "ok": True,
        "made_progress": True,
        "dual_view_refresh": refresh_result,
    }
    if prepare_before_grow:
        expected["dual_view_prepare"] = refresh_result
    assert result == expected
    assert events == (
        ["refresh", "prefetch", ("sync", 0), "refresh_full"]
        if prepare_before_grow
        else ["prefetch", ("sync", 0), "refresh"]
    )


@pytest.mark.parametrize("failure_stage", [None, "hot", "kv"])
@pytest.mark.parametrize("block_bytes", [4096, 18432])
def test_worker_kv_resize_reconciles_hot_and_audits_all_active_kv(
    monkeypatch: Any,
    failure_stage: str | None,
    block_bytes: int,
) -> None:
    runner = _make_worker_resize_runner()
    runner._env_flag_enabled = lambda name, default=False: name in {  # type: ignore[method-assign]
        "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW",
        "VLLM_UVM_KV_PREPARE_DUAL_BEFORE_GROW",
        "VLLM_UVM_KV_RECONCILE_HOT_AFTER_GROW",
    }
    events: list[object] = []

    def collect(block_ids: list[int]) -> list[tuple[int, int]]:
        events.append(("collect", tuple(block_ids)))
        return PhysicalKVSpan(
            span_id="layer.0", ptr=0x100000, size_bytes=8 * block_bytes,
            page_size=block_bytes, max_block_id=7,
        ).ranges_for_blocks(block_ids)

    def reconcile(**kwargs: object) -> dict[str, object]:
        assert kwargs == {"preserve_allocator_cache": True}
        events.append("reconcile_hot")
        return {"ok": failure_stage != "hot"}

    def audit(ranges: list[tuple[int, int]], **kwargs: object) -> dict:
        assert kwargs == {"device": 0, "wait_for_trackers": True}
        assert ranges == [(0x100000, 6 * block_bytes)]
        assert all(ptr % 4096 == size % 4096 == 0 for ptr, size in ranges)
        events.append("audit_kv")
        return _exclusive_residency_result(
            ranges, "cpu" if failure_stage == "kv" else "gpu"
        )

    monkeypatch.setattr(gpu_model_runner_module,
                        "collect_uvm_kv_cache_physical_ranges_for_blocks", collect)
    monkeypatch.setattr(gpu_model_runner_module,
                        "batch_advise_ranges_preferred_location",
                        lambda ranges, _device: len(ranges))
    monkeypatch.setattr(gpu_model_runner_module,
                        "batch_prefetch_ranges_to_device_with_executor",
                        lambda ranges, *_args, **_kwargs: (
                            events.append("prefetch") or len(ranges), "test", 0
                        ))
    monkeypatch.setattr(gpu_model_runner_module.torch.cuda,
                        "current_device", lambda: 0)
    monkeypatch.setattr(gpu_model_runner_module.torch.cuda, "synchronize",
                        lambda _device: events.append("sync"))
    monkeypatch.setattr(offload_utils,
                        "reconcile_cpu_offload_uvm_original_va_sealed_hot_residency",
                        reconcile)
    monkeypatch.setattr(gpu_model_runner_module,
                        "driver_batch_query_ranges_residency_bytes", audit)
    monkeypatch.setattr(offload_utils,
                        "refresh_cpu_offload_uvm_original_va_full_dual_view",
                        lambda **kwargs: events.append(("refresh", kwargs)) or {"ok": True})

    result = runner.resize_uvm_kv_cache_target(3, 5, [])

    assert result["ok"] is (failure_stage is None)
    expected: list[object] = [
        ("refresh", {}), ("collect", (4, 5)), "prefetch", "sync", "reconcile_hot"
    ]
    if failure_stage != "hot":
        expected.extend([("collect", (0, 1, 2, 3, 4, 5)), "audit_kv"])
    if failure_stage is None:
        expected.append(("refresh", {"nested_refresh": False}))
    else:
        assert result["dual_view_refresh"] is None
    assert events == expected


@pytest.mark.parametrize("failure_stage", ["legacy_post", "prepare", "prepared_post"])
def test_worker_kv_resize_refresh_failure_fails_closed(
    monkeypatch: Any,
    failure_stage: str,
) -> None:
    runner = _make_worker_resize_runner()
    runner._env_flag_enabled = lambda name, default=False: (  # type: ignore[method-assign]
        name == "VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW"
        or (failure_stage != "legacy_post" and name == "VLLM_UVM_KV_PREPARE_DUAL_BEFORE_GROW")
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "collect_uvm_kv_cache_physical_ranges_for_blocks",
        lambda block_ids: [(0x100000, 0x1000)] if block_ids else [],
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_advise_ranges_preferred_location",
        lambda ranges, _device: len(ranges),
    )
    migrations: list[int] = []
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda ranges, *_args, **_kwargs: (
            migrations.append(len(ranges)) or len(ranges), "test", 0
        ),
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "current_device",
        lambda: 0,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda _device: None,
    )

    refresh_calls = 0

    def fail_refresh(**_kwargs: object) -> dict[str, object]:
        nonlocal refresh_calls
        refresh_calls += 1
        if failure_stage == "prepared_post" and refresh_calls == 1:
            return {"ok": True}
        raise RuntimeError("mock dual-view refresh failure")

    monkeypatch.setattr(
        offload_utils,
        "refresh_cpu_offload_uvm_original_va_full_dual_view",
        fail_refresh,
    )

    result = runner.resize_uvm_kv_cache_target(3, 5, [])

    assert result["ok"] is False
    if failure_stage == "prepare":
        assert result["made_progress"] is False
        assert migrations == []
        assert result["dual_view_prepare"]["reason"] == "dual_view_prepare_failed"
    else:
        assert result["made_progress"] is True
        assert migrations == [1]
        assert result["dual_view_refresh"]["reason"] == "dual_view_refresh_failed"


def test_worker_partial_kv_resize_only_reports_complete_direction(
    monkeypatch: Any,
) -> None:
    runner = _make_worker_resize_runner()
    monkeypatch.setattr(
        gpu_model_runner_module,
        "collect_uvm_kv_cache_physical_ranges_for_blocks",
        lambda block_ids: [
            (0x100000 + block_id * 0x1000, 0x1000)
            for block_id in block_ids
        ],
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_offload_ranges_to_cpu_with_executor",
        lambda ranges, **_kwargs: (len(ranges), "test", 0, 0),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda *_args, **_kwargs: (1, "test", 0),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_advise_ranges_preferred_location",
        lambda ranges, _device: len(ranges),
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "current_device",
        lambda: 0,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        lambda _device: None,
    )

    result = runner.resize_uvm_kv_cache_target(
        3,
        5,
        [4, 5, 6],
    )

    assert result == {"ok": False, "made_progress": True}


def test_worker_kv_resize_sync_failure_never_reports_completion(
    monkeypatch: Any,
) -> None:
    runner = _make_worker_resize_runner()
    monkeypatch.setattr(
        gpu_model_runner_module,
        "collect_uvm_kv_cache_physical_ranges_for_blocks",
        lambda block_ids: [
            (0x100000 + block_id * 0x1000, 0x1000)
            for block_id in block_ids
        ],
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda ranges, *_args, **_kwargs: (len(ranges), "test", 0),
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_advise_ranges_preferred_location",
        lambda ranges, _device: len(ranges),
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "current_device",
        lambda: 0,
    )

    def fail_sync(_device: int) -> None:
        raise RuntimeError("mock KV synchronization failure")

    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "synchronize",
        fail_sync,
    )

    result = runner.resize_uvm_kv_cache_target(
        3,
        5,
        [],
    )

    assert result == {"ok": False, "made_progress": True}


def test_worker_kv_grow_requires_complete_gpu_preferred_advice(
    monkeypatch: Any,
) -> None:
    runner = _make_worker_resize_runner()
    prefetched = False
    monkeypatch.setattr(
        gpu_model_runner_module,
        "collect_uvm_kv_cache_physical_ranges_for_blocks",
        lambda block_ids: [
            (0x100000 + block_id * 0x1000, 0x1000)
            for block_id in block_ids
        ],
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_advise_ranges_preferred_location",
        lambda _ranges, _device: 0,
    )

    def unexpected_prefetch(*_args: object, **_kwargs: object):
        nonlocal prefetched
        prefetched = True
        return 0, "test", 0

    monkeypatch.setattr(
        gpu_model_runner_module,
        "batch_prefetch_ranges_to_device_with_executor",
        unexpected_prefetch,
    )
    monkeypatch.setattr(
        gpu_model_runner_module.torch.cuda,
        "current_device",
        lambda: 0,
    )

    result = runner.resize_uvm_kv_cache_target(3, 5, [])

    assert result == {"ok": False, "made_progress": True}
    assert prefetched is False
