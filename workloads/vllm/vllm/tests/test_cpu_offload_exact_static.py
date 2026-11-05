# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.models import utils as offload_utils


@dataclass
class _FakeStream:
    cuda_stream: int


class _FakeTensor:
    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (2, 2),
        stride: tuple[int, ...] = (2, 1),
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cuda:0",
        pinned: bool = False,
        pointer: int | None = None,
    ) -> None:
        self._shape = shape
        self._stride = stride
        self.dtype = dtype
        self.layout = torch.strided
        self.device = torch.device(device)
        self._pinned = pinned
        self._pointer = pointer if pointer is not None else id(self)
        self.copy_calls: list[tuple[_FakeTensor, bool]] = []
        self.to_calls: list[tuple[torch.device, bool, _FakeTensor]] = []

    def size(self) -> tuple[int, ...]:
        return self._shape

    def stride(self) -> tuple[int, ...]:
        return self._stride

    def numel(self) -> int:
        result = 1
        for value in self._shape:
            result *= value
        return result

    def element_size(self) -> int:
        return torch.empty((), dtype=self.dtype).element_size()

    def data_ptr(self) -> int:
        return self._pointer

    def is_pinned(self) -> bool:
        return self._pinned

    def copy_(self, source: _FakeTensor, *, non_blocking: bool = False) -> _FakeTensor:
        self.copy_calls.append((source, non_blocking))
        return self

    def to(
        self, device: str | torch.device, *, non_blocking: bool = False
    ) -> _FakeTensor:
        target = _FakeTensor(
            shape=self._shape,
            stride=self._stride,
            dtype=self.dtype,
            device=device,
        )
        self.to_calls.append((torch.device(device), non_blocking, target))
        return target


@dataclass
class _FakeParameter:
    data: _FakeTensor

    @property
    def device(self) -> torch.device:
        return self.data.device


class _FakeModule:
    def __init__(
        self,
        parameters: dict[str, _FakeTensor],
        *,
        extra_state: dict[str, _FakeTensor] | None = None,
    ) -> None:
        self._parameters = {
            name: _FakeParameter(tensor) for name, tensor in parameters.items()
        }
        self._extra_state = extra_state or {}

    def parameters(self) -> Iterator[_FakeParameter]:
        return iter(self._parameters.values())

    def named_parameters(self) -> Iterator[tuple[str, _FakeParameter]]:
        return iter(self._parameters.items())

    def state_dict(self) -> dict[str, _FakeTensor]:
        return {
            **{name: parameter.data for name, parameter in self._parameters.items()},
            **self._extra_state,
        }

    def forward(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return args


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch):
    allocations: list[_FakeTensor] = []
    functional_states: list[dict[str, _FakeTensor]] = []
    stream = _FakeStream(17)
    allocation_pinned = True

    def empty_strided(*, size, stride, dtype, layout, device, **kwargs):
        tensor = _FakeTensor(
            shape=tuple(size),
            stride=tuple(stride),
            dtype=dtype,
            device=device,
            pinned=(allocation_pinned if torch.device(device).type == "cpu" else False),
        )
        allocations.append(tensor)
        return tensor

    def functional_call(module, state, *, args, kwargs, tie_weights=True):
        functional_states.append(state)
        return state

    monkeypatch.setattr(offload_utils, "is_pin_memory_available", lambda: True)
    monkeypatch.setattr(offload_utils, "is_uva_available", lambda: False)
    monkeypatch.setattr(offload_utils.torch, "empty_strided", empty_strided)
    monkeypatch.setattr(offload_utils.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(offload_utils.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        offload_utils.torch.cuda, "current_stream", lambda device=None: stream
    )
    monkeypatch.setattr(offload_utils, "functional_call", functional_call)
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_BACKEND", "copy")
    monkeypatch.delenv("VLLM_CPU_OFFLOAD_COPY_SYNC", raising=False)
    monkeypatch.delenv("VLLM_CPU_OFFLOAD_COPY_TRACE", raising=False)

    class Controls:
        @property
        def allocations(self) -> list[_FakeTensor]:
            return allocations

        @property
        def functional_states(self) -> list[dict[str, _FakeTensor]]:
            return functional_states

        def set_stream(self, value: int) -> None:
            stream.cuda_stream = value

        def make_pageable(self) -> None:
            nonlocal allocation_pinned
            allocation_pinned = False

    return Controls()


def _write_exact_names(tmp_path, names: list[str], *, json_format: bool = True):
    exact_file = tmp_path / "exact_names.txt"
    if json_format:
        exact_file.write_text(json.dumps(names), encoding="utf-8")
    else:
        exact_file.write_text("\n".join(names) + "\n", encoding="utf-8")
    return exact_file


def _configure_exact(
    monkeypatch: pytest.MonkeyPatch,
    exact_file,
    *,
    static_bank: bool,
    budget: int = 1 << 20,
) -> None:
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE", str(exact_file))
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_STATIC_BANK", "1" if static_bank else "0")
    offload_utils.set_cpu_offload_max_bytes(budget)


@pytest.mark.parametrize("json_format", [True, False])
def test_exact_name_file_formats(tmp_path, json_format):
    names = ["model.layers.0.a", "model.layers.1.b"]
    exact_file = _write_exact_names(tmp_path, names, json_format=json_format)
    assert offload_utils._load_cpu_offload_exact_names(str(exact_file)) == frozenset(
        names
    )


@pytest.mark.parametrize(
    "contents",
    ["[]", '["a", "a"]', '["a", 1]', "[not-json", "\n\n"],
)
def test_exact_name_file_rejects_invalid_input(tmp_path, contents):
    exact_file = tmp_path / "invalid.txt"
    exact_file.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        offload_utils._load_cpu_offload_exact_names(str(exact_file))


def test_exact_gate_and_native_copy_forward(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.cold"])
    _configure_exact(monkeypatch, exact_file, static_bank=False)
    hot = _FakeTensor()
    cold = _FakeTensor()
    module = _FakeModule({"hot": hot, "cold": cold})

    offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")

    assert module._parameters["hot"].data is hot
    cold_cpu = module._parameters["cold"].data
    assert cold_cpu.device.type == "cpu"
    module.forward()
    assert cold_cpu.to_calls[0][:2] == (torch.device("cuda:0"), True)
    assert fake_cuda.functional_states[-1]["cold"].device.type == "cuda"
    assert offload_utils._CPU_OFFLOAD_BYTES == cold.numel() * cold.element_size()


def test_exact_name_uva_selects_only_requested_parameter(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.cold"])
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_BACKEND", "uva")
    monkeypatch.setattr(offload_utils, "is_uva_available", lambda: True)

    def cuda_view(cpu_tensor: _FakeTensor) -> _FakeTensor:
        return _FakeTensor(
            shape=cpu_tensor.size(),
            stride=cpu_tensor.stride(),
            dtype=cpu_tensor.dtype,
            device="cuda:0",
            pointer=cpu_tensor.data_ptr(),
        )

    monkeypatch.setattr(
        offload_utils, "get_cuda_view_from_cpu_tensor", cuda_view
    )
    _configure_exact(monkeypatch, exact_file, static_bank=False)
    hot = _FakeTensor()
    cold = _FakeTensor()
    module = _FakeModule({"hot": hot, "cold": cold})

    original_forward = module.forward
    offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")

    assert module._parameters["hot"].data is hot
    mapped = module._parameters["cold"].data
    assert mapped.device == torch.device("cuda:0")
    assert mapped.data_ptr() == module._parameters["cold"]._vllm_offloaded_cpu_data.data_ptr()
    assert module.forward == original_forward
    assert offload_utils._CPU_OFFLOAD_BYTES == cold.numel() * cold.element_size()


def test_exact_name_uva_rejects_alias_modes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact_file = _write_exact_names(tmp_path, ["layers.0.weight"])
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_BACKEND", "uva")
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE", str(exact_file))
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_STATIC_BANK", "1")
    with pytest.raises(ValueError, match="does not support"):
        offload_utils.set_cpu_offload_max_bytes(1 << 20)


def test_static_bank_reuses_signature_and_lane_across_layers(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.weight", "layers.1.weight"])
    _configure_exact(monkeypatch, exact_file, static_bank=True)
    source0 = _FakeTensor()
    source1 = _FakeTensor()
    module0 = _FakeModule({"weight": source0})
    module1 = _FakeModule({"weight": source1})

    offload_utils.maybe_offload_to_cpu(module0, module_prefix="layers.0")
    offload_utils.maybe_offload_to_cpu(module1, module_prefix="layers.1")

    gpu_allocations = [
        tensor for tensor in fake_cuda.allocations if tensor.device.type == "cuda"
    ]
    assert len(gpu_allocations) == 1
    bank = gpu_allocations[0]
    cpu0 = module0._parameters["weight"].data
    cpu1 = module1._parameters["weight"].data
    module0.forward()
    module1.forward()
    assert bank.copy_calls[-2:] == [(cpu0, True), (cpu1, True)]
    assert fake_cuda.functional_states[-2]["weight"] is bank
    assert fake_cuda.functional_states[-1]["weight"] is bank
    assert offload_utils._CPU_OFFLOAD_STATIC_BANK_BYTES == 16


def test_static_bank_uses_distinct_lanes_within_module(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.left", "layers.0.right"])
    _configure_exact(monkeypatch, exact_file, static_bank=True)
    module = _FakeModule({"left": _FakeTensor(), "right": _FakeTensor()})

    offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")
    module.forward()

    state = fake_cuda.functional_states[-1]
    assert state["left"] is not state["right"]
    assert len(offload_utils._CPU_OFFLOAD_STATIC_BANKS) == 2


def test_static_bank_rejects_unregistered_cpu_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.weight"])
    _configure_exact(monkeypatch, exact_file, static_bank=True)
    module = _FakeModule(
        {"weight": _FakeTensor()},
        extra_state={"unexpected": _FakeTensor(device="cpu", pinned=True)},
    )
    offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")

    with pytest.raises(RuntimeError, match="unregistered CPU"):
        module.forward()


def test_exact_mode_is_fail_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.weight"])
    _configure_exact(monkeypatch, exact_file, static_bank=False)
    monkeypatch.setattr(offload_utils, "is_pin_memory_available", lambda: False)
    module = _FakeModule({"weight": _FakeTensor()})
    with pytest.raises(RuntimeError, match="requires pinned"):
        offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")

    monkeypatch.setattr(offload_utils, "is_pin_memory_available", lambda: True)
    offload_utils.set_cpu_offload_max_bytes(1 << 20)
    monkeypatch.setattr(offload_utils.torch.cuda, "device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="single GPU"):
        offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")


def test_pageable_allocation_and_missing_name_fail_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.weight"])
    _configure_exact(monkeypatch, exact_file, static_bank=False)
    fake_cuda.make_pageable()
    module = _FakeModule({"weight": _FakeTensor()})
    with pytest.raises(RuntimeError, match="pageable"):
        offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")

    exact_file = _write_exact_names(tmp_path, ["layers.0.missing"])
    _configure_exact(monkeypatch, exact_file, static_bank=False)
    with pytest.raises(RuntimeError, match="did not find"):
        offload_utils._require_all_copy_exact_names_seen()


def test_thread_and_stream_changes_fail_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.weight"])
    _configure_exact(monkeypatch, exact_file, static_bank=True)
    thread_id = 100
    monkeypatch.setattr(offload_utils.threading, "get_ident", lambda: thread_id)
    module = _FakeModule({"weight": _FakeTensor()})
    offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")

    thread_id = 101
    with pytest.raises(RuntimeError, match="one thread"):
        module.forward()
    thread_id = 100
    module.forward()
    fake_cuda.set_stream(99)
    with pytest.raises(RuntimeError, match="one CUDA stream"):
        module.forward()


def test_copy_backend_without_exact_file_keeps_legacy_behavior(
    monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    monkeypatch.delenv("VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE", raising=False)
    monkeypatch.delenv("VLLM_CPU_OFFLOAD_STATIC_BANK", raising=False)
    monkeypatch.setattr(offload_utils, "is_pin_memory_available", lambda: False)
    offload_utils.set_cpu_offload_max_bytes(1 << 20)
    module = _FakeModule({"weight": _FakeTensor()})

    offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")
    cpu_weight = module._parameters["weight"].data
    assert cpu_weight.device.type == "cpu"
    module.forward()
    assert cpu_weight.to_calls[0][:2] == (torch.device("cuda:0"), True)


def test_manifest_records_exact_static_bank_fields(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_cuda
):
    exact_file = _write_exact_names(tmp_path, ["layers.0.weight"])
    manifest_file = tmp_path / "manifest.jsonl"
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_MANIFEST_FILE", str(manifest_file))
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_COPY_TRACE", "1")
    _configure_exact(monkeypatch, exact_file, static_bank=True)
    module = _FakeModule({"weight": _FakeTensor()})
    offload_utils.maybe_offload_to_cpu(module, module_prefix="layers.0")
    module.forward()

    records = [json.loads(line) for line in manifest_file.read_text().splitlines()]
    parameter = next(
        record for record in records if record["event"] == "cpu_offload_parameter"
    )
    forward = next(
        record for record in records if record["event"] == "cpu_offload_copy_forward"
    )
    for record in (parameter, forward):
        assert record["copy_exact_names"] is True
        assert record["copy_static_bank"] is True
        assert record["copy_bank_bytes"] == 16


def test_static_bank_requires_exact_file(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_BACKEND", "copy")
    monkeypatch.delenv("VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE", raising=False)
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_STATIC_BANK", "1")
    with pytest.raises(ValueError, match="requires"):
        offload_utils.set_cpu_offload_max_bytes(1 << 20)


def test_graph_static_alias_plan_uses_layer_parity_and_unique_consumers():
    names = frozenset(
        f"model.layers.{layer}.mlp.experts.{weight}"
        for layer in range(4)
        for weight in ("w13_weight", "w2_weight")
    )
    plan = offload_utils._build_graph_static_alias_plan(names)

    for weight in ("w13_weight", "w2_weight"):
        members = [
            plan[f"model.layers.{layer}.mlp.experts.{weight}"] for layer in range(4)
        ]
        assert [member.bank_id for member in members] == [0, 1, 0, 1]
        assert [member.consumer_index for member in members] == [0, 0, 1, 1]
        assert [member.consumer_count for member in members] == [2, 2, 2, 2]


class _FakeVmmAllocation:
    def __init__(self, token: int, *, fail_close: bool = False) -> None:
        self.token = token
        self.fail_close = fail_close
        self.closed = False
        self.calls: list[object] = []

    def synchronize(self) -> None:
        self.calls.append("synchronize")

    def close(self, *, synchronize: bool = False) -> None:
        self.calls.append(("close", synchronize))
        if self.fail_close:
            raise RuntimeError("fault-injected close failure")
        self.closed = True


def _fake_vmm_arena(
    allocation: _FakeVmmAllocation,
    *,
    layer_index: int,
    backing_bytes: int,
):
    plan = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=layer_index,
        local_name="weight",
        bank_id=layer_index & 1,
        consumer_index=0,
        consumer_count=1,
    )
    arena = offload_utils._GraphStaticAliasArena(
        allocation=allocation,
        upload=object(),
        signature=((backing_bytes,), (1,), torch.uint8, 0, backing_bytes, 1),
        requested_bytes=backing_bytes,
        aligned_bytes=backing_bytes,
        backing_bytes=backing_bytes,
    )
    binding = offload_utils._GraphStaticAliasBinding(
        arena=arena,
        consumer=object(),
        plan=plan,
    )
    arena.bindings.append(binding)
    return arena, binding


def _set_vmm_destroy_state(
    monkeypatch: pytest.MonkeyPatch,
    arenas: dict[tuple[str, str, int], Any],
) -> None:
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND", "vmm"
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS", arenas
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES",
        sum(arena.backing_bytes for arena in arenas.values()),
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE", False
    )


def test_vmm_destroy_requires_graph_teardown_without_releasing_references(
    monkeypatch: pytest.MonkeyPatch,
):
    allocation = _FakeVmmAllocation(11)
    arena, binding = _fake_vmm_arena(
        allocation, layer_index=0, backing_bytes=4096
    )
    arenas = {("model.layers", "weight", 0): arena}
    _set_vmm_destroy_state(monkeypatch, arenas)

    result = offload_utils.destroy_cpu_offload_graph_static_alias(
        graphs_destroyed=False
    )

    assert result["ok"] is False
    assert result["reason"] == "cuda_graphs_still_live"
    assert allocation.calls == []
    assert arena.upload is not None
    assert binding.consumer is not None
    assert offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS == arenas


def test_vmm_destroy_releases_views_synchronizes_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
):
    allocation = _FakeVmmAllocation(12)
    arena, binding = _fake_vmm_arena(
        allocation, layer_index=0, backing_bytes=8192
    )
    _set_vmm_destroy_state(
        monkeypatch, {("model.layers", "weight", 0): arena}
    )

    result = offload_utils.destroy_cpu_offload_graph_static_alias(
        graphs_destroyed=True,
        synchronize=False,
    )

    assert result["ok"] is True
    assert result["reason"] == "ok"
    assert result["destroyed_arena_count"] == 1
    assert result["released_backing_bytes"] == 8192
    assert allocation.calls == ["synchronize", ("close", False)]
    assert arena.upload is None
    assert binding.consumer is None
    assert arena.bindings == []
    assert offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS == {}
    assert offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES == 0

    again = offload_utils.destroy_cpu_offload_graph_static_alias(
        graphs_destroyed=True
    )
    assert again["ok"] is True
    assert again["reason"] == "no_active_allocations"
    assert allocation.calls == ["synchronize", ("close", False)]


def test_vmm_destroy_retains_only_failed_arenas_for_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    first_allocation = _FakeVmmAllocation(21)
    second_allocation = _FakeVmmAllocation(22, fail_close=True)
    first_arena, first_binding = _fake_vmm_arena(
        first_allocation, layer_index=0, backing_bytes=4096
    )
    second_arena, second_binding = _fake_vmm_arena(
        second_allocation, layer_index=1, backing_bytes=8192
    )
    first_key = ("model.layers", "weight", 0)
    second_key = ("model.layers", "weight", 1)
    _set_vmm_destroy_state(
        monkeypatch,
        {first_key: first_arena, second_key: second_arena},
    )

    failed = offload_utils.destroy_cpu_offload_graph_static_alias(
        graphs_destroyed=True
    )

    assert failed["ok"] is False
    assert failed["reason"] == "vmm_close_failed"
    assert failed["destroyed_arena_count"] == 1
    assert failed["remaining_arena_count"] == 1
    assert offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS == {
        second_key: second_arena
    }
    assert offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES == 8192
    assert first_binding.consumer is None
    assert second_binding.consumer is None

    second_allocation.fail_close = False
    retried = offload_utils.destroy_cpu_offload_graph_static_alias(
        graphs_destroyed=True
    )
    assert retried["ok"] is True
    assert retried["destroyed_arena_count"] == 1
    assert offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS == {}
    assert offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES == 0


@pytest.mark.parametrize("backend", ["vmm", "uvm_original_va"])
def test_runner_shutdown_resets_graphs_before_static_alias_destroy(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
):
    from vllm.v1.worker import gpu_model_runner as runner_module

    events: list[object] = []

    class FakeGraph:
        def reset(self) -> None:
            events.append("graph_reset")

    graph = FakeGraph()
    entry = SimpleNamespace(
        cudagraph=graph,
        output=object(),
        input_addresses=object(),
    )

    class FakeCudagraphWrapper:
        def __init__(self) -> None:
            self.concrete_cudagraph_entries = {"entry": entry}

    monkeypatch.setattr(runner_module, "CUDAGraphWrapper", FakeCudagraphWrapper)
    monkeypatch.setattr(
        offload_utils,
        "get_cpu_offload_graph_static_alias_diagnostics",
        lambda: {"enabled": True, "backend": backend},
    )

    def destroy(**kwargs: Any) -> dict[str, Any]:
        events.append(("destroy", kwargs))
        return {"ok": True, "backend": backend}

    monkeypatch.setattr(
        offload_utils, "destroy_cpu_offload_graph_static_alias", destroy
    )
    monkeypatch.setattr(
        runner_module.torch.cuda,
        "synchronize",
        lambda device=None: events.append(("synchronize", device)),
    )
    monkeypatch.setattr(
        runner_module.gc, "collect", lambda: events.append("gc_collect")
    )

    runner = object.__new__(runner_module.GPUModelRunner)
    runner.device = torch.device("cuda:0")
    runner.model = FakeCudagraphWrapper()
    result = runner.shutdown_cpu_offload_graph_static_alias()

    assert events == [
        ("synchronize", torch.device("cuda:0")),
        "graph_reset",
        "gc_collect",
        ("synchronize", torch.device("cuda:0")),
        (
            "destroy",
            {"graphs_destroyed": True, "synchronize": False},
        ),
    ]
    assert result["ok"] is True
    assert result["destroyed_graph_count"] == 1
    assert entry.cudagraph is None
    assert entry.output is None
    assert entry.input_addresses is None
    assert runner.model.concrete_cudagraph_entries == {}


def test_runner_dynamic_plan_rebuild_requires_idle_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.v1.worker import gpu_model_runner as runner_module

    runner = object.__new__(runner_module.GPUModelRunner)
    runner.model = object()
    with pytest.raises(RuntimeError, match="idle scheduler boundary"):
        runner.rebuild_cpu_offload_uvm_expert_plan_at_idle_boundary(
            "/tmp/plan.json",
            reason="test",
            scheduler_idle=False,
        )


def test_original_va_shutdown_reference_release_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    binding = SimpleNamespace(source=object())
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", 7
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "remote"
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED", False
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {"weight": binding},
    )

    with pytest.raises(RuntimeError, match="session is active"):
        offload_utils.release_cpu_offload_uvm_original_va_shutdown_references()
    assert binding.source is not None


def test_original_va_shutdown_reference_release_clears_tensor_owners(
    monkeypatch: pytest.MonkeyPatch,
):
    binding = SimpleNamespace(source=object())
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", 0
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "remote"
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED", False
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {"weight": binding},
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS", {"arena": object()}
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS", {"arena": object()}
    )

    result = (
        offload_utils.release_cpu_offload_uvm_original_va_shutdown_references()
    )

    assert result == {
        "ok": True,
        "operation": "release_shutdown_references",
        "binding_count": 1,
        "arena_count": 1,
        "source_count": 1,
        "state": "shutdown",
    }
    assert binding.source is None
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS == {}
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS == {}
    assert offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ARENAS == {}


def test_runner_dynamic_plan_rebuild_resets_and_recaptures_graphs(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.v1.worker import gpu_model_runner as runner_module

    events: list[object] = []

    class FakeGraph:
        def __init__(self, name: str) -> None:
            self.name = name

        def reset(self) -> None:
            events.append(("reset", self.name))

    @dataclass(frozen=True)
    class FakeDescriptor:
        num_tokens: int
        uniform_decode: bool
        has_lora: bool = False

    descriptors = (
        FakeDescriptor(num_tokens=1, uniform_decode=True),
        FakeDescriptor(num_tokens=314, uniform_decode=False),
    )
    entries = {
        descriptor: SimpleNamespace(
            cudagraph=FakeGraph(str(descriptor.num_tokens)),
            output=object(),
            input_addresses=object(),
        )
        for descriptor in descriptors
    }

    class FakeCudagraphWrapper:
        def __init__(self) -> None:
            self.concrete_cudagraph_entries = entries
            self.graph_pool = "old-pool"

    monkeypatch.setattr(runner_module, "CUDAGraphWrapper", FakeCudagraphWrapper)
    monkeypatch.setattr(
        runner_module.torch.cuda,
        "synchronize",
        lambda device=None: events.append(("sync", device)),
    )
    monkeypatch.setattr(
        runner_module.gc, "collect", lambda: events.append("collect")
    )
    monkeypatch.setattr(
        runner_module.torch.cuda,
        "empty_cache",
        lambda: events.append("empty_cache"),
    )
    monkeypatch.setattr(
        runner_module.torch.cuda,
        "graph_pool_handle",
        lambda: events.append("renew_graph_pool") or "new-pool",
    )

    def rebuild(*args: Any, **kwargs: Any) -> dict[str, Any]:
        events.append(("rebuild", args, kwargs))
        return {"ok": True, "generation": 1, "selected_bytes": 8192}

    monkeypatch.setattr(
        offload_utils,
        "rebuild_cpu_offload_uvm_original_va_expert_plan",
        rebuild,
    )
    monkeypatch.setattr(
        offload_utils,
        "get_cpu_offload_graph_static_alias_diagnostics",
        lambda: {
            "lifecycle_state": "sealed",
            "remote_graph_captured": True,
            "alias_graph_captured": True,
            "dual_graph_partition_certificate": {},
            "plan_generation": 1,
        },
    )

    runner = object.__new__(runner_module.GPUModelRunner)
    runner.device = torch.device("cuda:0")
    runner.model = FakeCudagraphWrapper()
    runner._uvm_original_va_phase_decode_cutover_done = True
    runner._uvm_original_va_phase_prefill_seen = True

    def capture_model() -> int:
        events.append("capture")
        return 1234

    runner.capture_model = capture_model
    result = runner.rebuild_cpu_offload_uvm_expert_plan_at_idle_boundary(
        "/tmp/next.json",
        reason="kv_pressure_high",
        scheduler_idle=True,
        expected_selected_bytes=8192,
        pressure={"predicted_kv_bytes": 4096},
    )

    assert [item for item in events if isinstance(item, tuple) and item[0] == "reset"] == [
        ("reset", "1"),
        ("reset", "314"),
    ]
    assert events.index("capture") > next(
        index
        for index, item in enumerate(events)
        if isinstance(item, tuple) and item[0] == "rebuild"
    )
    assert result["ok"] is True
    assert result["destroyed_graph_count"] == 2
    assert result["graph_pool_renewed"] is True
    assert result["graph_memory_bytes"] == 1234
    assert runner.model.concrete_cudagraph_entries == {}
    assert runner.model.graph_pool == "new-pool"
    assert runner._uvm_original_va_phase_decode_cutover_done is False
    assert runner._uvm_original_va_phase_prefill_seen is False


def test_dynamic_expert_plan_rebuild_mutates_bindings_in_place(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    events: list[str] = []
    name = "model.layers.0.weight"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"selected_by_layer": {"0": [1, 2]}}),
        encoding="utf-8",
    )
    name_plan = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=0,
        local_name="weight",
        bank_id=0,
        consumer_index=0,
        consumer_count=1,
    )
    consumer = _FakeTensor(
        shape=(4, 1024),
        stride=(1024, 1),
        dtype=torch.uint8,
        pointer=0x10000,
    )
    parameter = _FakeParameter(consumer)
    old_arena = SimpleNamespace(upload=object(), consumers={})
    retained = offload_utils._UvmOriginalAliasBinding(
        full_parameter_name=name,
        arena=old_arena,
        consumer=consumer,
        parameter=parameter,
        plan=name_plan,
        requested_bytes=4096,
        parameter_bytes=16384,
        expert_ids=(1,),
        expert_stride_bytes=4096,
        consumer_ranges=((0x11000, 4096),),
        source=_FakeTensor(device="cpu", pinned=True),
    )
    old_arena.consumers[0] = retained

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
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN", {name: name_plan}
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER",
        {0: (0,)},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE",
        "old.json",
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {name: retained},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS",
        {("model.layers", "weight", 0): old_arena},
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "remote_parked"
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", 41)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", True)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT", (0x10000,)
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_MAX_BYTES", 1 << 20)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_PLAN_GENERATION", 0)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_PLAN_REBUILD_HISTORY", [])
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE", False
    )
    monkeypatch.setattr(
        offload_utils.torch.cuda, "synchronize", lambda device=None: None
    )
    monkeypatch.setattr(offload_utils.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(offload_utils.gc, "collect", lambda: 0)
    monkeypatch.setattr(
        offload_utils,
        "destroy_cpu_offload_graph_static_alias",
        lambda **kwargs: events.append("destroy") or {"ok": True},
    )

    from vllm.device_allocator import uvm_static_alias

    monkeypatch.setattr(
        uvm_static_alias,
        "driver_static_alias_resume",
        lambda *args, **kwargs: events.append("resume")
        or {"ok": True, "operation": "resume", "session_id": 41},
    )

    def create_binding(**kwargs: Any):
        arena = SimpleNamespace(
            upload=object(),
            aligned_bytes=8192,
            consumers={},
        )
        offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS[
            ("model.layers", "weight", 0)
        ] = arena
        offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES = 8192
        replacement = offload_utils._UvmOriginalAliasBinding(
            full_parameter_name=name,
            arena=arena,
            consumer=consumer,
            parameter=parameter,
            plan=name_plan,
            requested_bytes=8192,
            parameter_bytes=16384,
            expert_ids=(1, 2),
            expert_stride_bytes=4096,
            consumer_ranges=((0x11000, 4096), (0x12000, 4096)),
        )
        arena.consumers[0] = replacement
        offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS[name] = replacement
        return replacement

    monkeypatch.setattr(
        offload_utils, "_create_uvm_original_alias_binding", create_binding
    )

    from vllm.device_allocator import uvm

    monkeypatch.setattr(
        uvm, "batch_advise_ranges_preferred_location", lambda ranges, device: len(ranges)
    )
    monkeypatch.setattr(
        uvm,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda ranges, device, executor: (len(ranges), executor, 0),
    )

    def query(ranges, **kwargs):
        total = sum(length for _, length in ranges)
        return {
            "ok": True,
            "queried_bytes": total,
            "gpu_bytes": total,
            "cpu_bytes": 0,
            "any_bytes": total,
            "unknown_bytes": 0,
        }

    monkeypatch.setattr(uvm, "driver_batch_query_ranges_residency_bytes", query)

    finalized_known_cpu_ranges: set[tuple[int, int]] = set()

    def finalize(*, known_cpu_ranges=()) -> dict[str, Any]:
        finalized_known_cpu_ranges.update(known_cpu_ranges)
        offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "sealed"
        return {"ok": True, "state": "sealed"}

    monkeypatch.setattr(
        offload_utils, "finalize_cpu_offload_graph_static_alias", finalize
    )

    result = offload_utils.rebuild_cpu_offload_uvm_original_va_expert_plan(
        str(plan_path),
        reason="test_pressure",
        graphs_destroyed=True,
        expected_selected_bytes=8192,
        pressure={"predicted_kv_bytes": 123},
    )

    assert result["ok"] is True
    assert result["generation"] == 1
    assert result["pre_destroy_resume"]["ok"] is True
    assert events.index("resume") < events.index("destroy")
    assert finalized_known_cpu_ranges == {(0x11000, 4096)}
    assert result["parameter_vas_before"] == result["parameter_vas_after"]
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS[name] is retained
    assert retained.expert_ids == (1, 2)
    assert retained.requested_bytes == 8192
    assert retained.source is None
    assert parameter._vllm_uvm_original_alias_binding is retained


@pytest.mark.parametrize("mode_name", ["FULL", "FULL_DECODE_ONLY"])
def test_graph_static_alias_runtime_accepts_full_modes(monkeypatch, mode_name):
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_dbo=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name=mode_name),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    offload_utils.validate_cpu_offload_graph_static_alias_runtime(config)


def test_graph_static_alias_runtime_allows_explicit_eager(monkeypatch):
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER", "1")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE", "off"
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_dbo=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="NONE"),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    offload_utils.validate_cpu_offload_graph_static_alias_runtime(config)


def test_graph_static_alias_runtime_rejects_implicit_eager(monkeypatch):
    monkeypatch.delenv("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER", raising=False)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE", "off"
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_dbo=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="NONE"),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    with pytest.raises(RuntimeError, match="requires single-GPU"):
        offload_utils.validate_cpu_offload_graph_static_alias_runtime(config)


@pytest.mark.parametrize(
    ("mode_name", "parallel_overrides"),
    [
        ("PIECEWISE", {}),
        ("FULL_AND_PIECEWISE", {}),
        ("FULL", {"enable_dbo": True}),
        ("FULL", {"tensor_parallel_size": 2}),
    ],
)
def test_graph_static_alias_runtime_rejects_overlap(
    monkeypatch, mode_name, parallel_overrides
):
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    parallel = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_dbo": False,
        **parallel_overrides,
    }
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(**parallel),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name=mode_name),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    with pytest.raises(RuntimeError, match="requires single-GPU"):
        offload_utils.validate_cpu_offload_graph_static_alias_runtime(config)


def test_graph_static_alias_runtime_rejects_dynamo_compile(monkeypatch):
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_dbo=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY"),
            mode=SimpleNamespace(name="VLLM_COMPILE"),
        ),
    )
    with pytest.raises(RuntimeError, match="compilation_mode=VLLM_COMPILE"):
        offload_utils.validate_cpu_offload_graph_static_alias_runtime(config)


def test_uvm_original_backend_requires_managed_expert_allocation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    exact_file = _write_exact_names(tmp_path, ["model.layers.0.weight"])
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_BACKEND", "copy")
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE", str(exact_file))
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", "1")
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND", "uvm_original_va")
    monkeypatch.delenv("VLLM_UVM_EXPERT_WEIGHTS_MANAGED", raising=False)

    with pytest.raises(ValueError, match="EXPERT_WEIGHTS_MANAGED"):
        offload_utils.set_cpu_offload_max_bytes(1 << 20)

    monkeypatch.setenv("VLLM_UVM_EXPERT_WEIGHTS_MANAGED", "1")
    offload_utils.set_cpu_offload_max_bytes(1 << 20)
    assert offload_utils.cpu_offload_uvm_original_va_static_alias_enabled()
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "collecting"
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", "0")
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND", "vmm")
    monkeypatch.delenv("VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE")
    offload_utils.set_cpu_offload_max_bytes(0)


def _set_uvm_original_exact_names(
    monkeypatch: pytest.MonkeyPatch,
    names: frozenset[str],
) -> None:
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_EXACT_NAMES", names)


@pytest.mark.parametrize("json_format", [True, False])
def test_uvm_original_managed_allocation_uses_only_selected_expert_layers(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    json_format: bool,
):
    prefix = "model.layers.0.mlp.experts"
    names = [f"{prefix}.w13_weight", f"{prefix}.w2_weight"]
    exact_file = _write_exact_names(tmp_path, names, json_format=json_format)
    _set_uvm_original_exact_names(
        monkeypatch,
        offload_utils._load_cpu_offload_exact_names(str(exact_file)),
    )

    assert offload_utils.use_uvm_expert_weight_allocation(
        prefix,
        managed_requested=True,
    )
    assert not offload_utils.use_uvm_expert_weight_allocation(
        "model.layers.1.mlp.experts",
        managed_requested=True,
    )
    # The delimiter in the descendant match prevents layers.1/layers.10
    # prefix collisions.
    assert not offload_utils.use_uvm_expert_weight_allocation(
        "model.layers.10.mlp.experts",
        managed_requested=True,
    )


@pytest.mark.parametrize("local_name", ["w13_weight", "w2_weight"])
def test_uvm_original_managed_allocation_rejects_half_selected_expert(
    monkeypatch: pytest.MonkeyPatch,
    local_name: str,
):
    prefix = "model.layers.0.mlp.experts"
    _set_uvm_original_exact_names(
        monkeypatch,
        frozenset({f"{prefix}.{local_name}"}),
    )

    with pytest.raises(ValueError, match="complete w13_weight/w2_weight pair"):
        offload_utils.use_uvm_expert_weight_allocation(
            prefix,
            managed_requested=True,
        )


def test_uvm_original_managed_allocation_rejects_unsupported_descendant(
    monkeypatch: pytest.MonkeyPatch,
):
    prefix = "model.layers.0.mlp.experts"
    _set_uvm_original_exact_names(
        monkeypatch,
        frozenset(
            {
                f"{prefix}.w13_weight",
                f"{prefix}.w2_weight",
                f"{prefix}.w13_weight_scale",
            }
        ),
    )

    with pytest.raises(ValueError, match="unexpected"):
        offload_utils.use_uvm_expert_weight_allocation(
            prefix,
            managed_requested=True,
        )


def test_uvm_original_managed_allocation_rejects_configuration_drift(
    monkeypatch: pytest.MonkeyPatch,
):
    prefix = "model.layers.0.mlp.experts"
    _set_uvm_original_exact_names(
        monkeypatch,
        frozenset({f"{prefix}.w13_weight", f"{prefix}.w2_weight"}),
    )

    with pytest.raises(RuntimeError, match="configuration was frozen"):
        offload_utils.use_uvm_expert_weight_allocation(
            prefix,
            managed_requested=False,
        )


def test_legacy_managed_expert_allocation_remains_global(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", False)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "vmm",
    )

    assert offload_utils.use_uvm_expert_weight_allocation(
        "model.layers.22.mlp.experts",
        managed_requested=True,
    )
    assert not offload_utils.use_uvm_expert_weight_allocation(
        "model.layers.22.mlp.experts",
        managed_requested=False,
    )


def test_uvm_original_capture_requires_sealed_topology(monkeypatch):
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "collecting"
    )

    with pytest.raises(RuntimeError, match="mapped and sealed"):
        offload_utils.set_cpu_offload_graph_static_alias_capture_phase(True)


def test_uvm_original_multi_region_alias_geometry(monkeypatch):
    class PointerTensor:
        def __init__(self, pointer: int):
            self._pointer = pointer

        def data_ptr(self) -> int:
            return self._pointer

    plan0 = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=0,
        local_name="weight",
        bank_id=0,
        consumer_index=0,
        consumer_count=2,
    )
    plan2 = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=2,
        local_name="weight",
        bank_id=0,
        consumer_index=1,
        consumer_count=2,
    )
    arena = offload_utils._UvmOriginalAliasArena(
        upload_storage=PointerTensor(0x200000),
        upload=PointerTensor(0x200000),
        signature=((1 << 20,), (1,), torch.uint8, 0, 1 << 20, 2),
        requested_bytes=1 << 20,
        aligned_bytes=1 << 20,
        backing_layer_id=0x70000001,
        pte_alignment=4096,
        pte_flags=0,
        region_bytes=512 * 1024,
    )
    binding0 = offload_utils._UvmOriginalAliasBinding(
        full_parameter_name="model.layers.0.weight",
        arena=arena,
        consumer=PointerTensor(0x400000),
        parameter=None,
        plan=plan0,
        requested_bytes=1 << 20,
    )
    binding2 = offload_utils._UvmOriginalAliasBinding(
        full_parameter_name="model.layers.2.weight",
        arena=arena,
        consumer=PointerTensor(0x600000),
        parameter=None,
        plan=plan2,
        requested_bytes=1 << 20,
    )
    arena.consumers = {0: binding0, 1: binding2}
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS",
        {("model.layers", "weight", 0): arena},
    )

    aliases = offload_utils._uvm_original_alias_map_entries(0)
    slots = offload_utils._uvm_original_alias_preallocate_entries()
    assert len(aliases) == 6
    assert len(slots) == 2
    assert [entry[4] for entry in aliases[:3]] == [
        0x200000,
        0x400000,
        0x600000,
    ]
    assert [entry[:3] for entry in aliases[:3]] == [
        (0x70000001, 0, 0),
    ] * 3
    assert [entry[:3] for entry in aliases[3:]] == [
        (0x70000001, 1, 1),
    ] * 3
    assert slots[0][0:6] == (
        0x200000,
        512 * 1024,
        0x70000001,
        0,
        0,
        0,
    )


def test_uvm_original_expert_slice_aliases_map_packed_offsets_to_original_vas(
    monkeypatch: pytest.MonkeyPatch,
):
    class PointerTensor:
        def __init__(self, pointer: int):
            self._pointer = pointer

        def data_ptr(self) -> int:
            return self._pointer

    plan0 = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=0,
        local_name="weight",
        bank_id=0,
        consumer_index=0,
        consumer_count=2,
    )
    plan2 = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=2,
        local_name="weight",
        bank_id=0,
        consumer_index=1,
        consumer_count=2,
    )
    arena = offload_utils._UvmOriginalAliasArena(
        upload_storage=PointerTensor(0x200000),
        upload=PointerTensor(0x200000),
        signature=((2, 8192), (8192, 1), torch.uint8, 0, 16384, (2, 8192), 2),
        requested_bytes=16384,
        aligned_bytes=16384,
        backing_layer_id=0x70000001,
        pte_alignment=4096,
        pte_flags=0,
        region_bytes=4096,
        packed_slice_bytes=8192,
    )
    binding0 = offload_utils._UvmOriginalAliasBinding(
        full_parameter_name="model.layers.0.weight",
        arena=arena,
        consumer=PointerTensor(0x400000),
        parameter=None,
        plan=plan0,
        requested_bytes=16384,
        parameter_bytes=32768,
        expert_ids=(1, 3),
        expert_stride_bytes=8192,
        consumer_ranges=((0x402000, 8192), (0x406000, 8192)),
    )
    binding2 = offload_utils._UvmOriginalAliasBinding(
        full_parameter_name="model.layers.2.weight",
        arena=arena,
        consumer=PointerTensor(0x600000),
        parameter=None,
        plan=plan2,
        requested_bytes=16384,
        parameter_bytes=32768,
        expert_ids=(0, 2),
        expert_stride_bytes=8192,
        consumer_ranges=((0x600000, 8192), (0x604000, 8192)),
    )
    arena.consumers = {0: binding0, 1: binding2}
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS",
        {("model.layers", "weight", 0): arena},
    )

    aliases = offload_utils._uvm_original_alias_map_entries(0)
    slots = offload_utils._uvm_original_alias_preallocate_entries()

    assert len(aliases) == 12
    assert len(slots) == 4
    assert [entry[4] for entry in aliases[0:3]] == [
        0x200000,
        0x402000,
        0x600000,
    ]
    assert [entry[4] for entry in aliases[3:6]] == [
        0x201000,
        0x403000,
        0x601000,
    ]
    assert [entry[4] for entry in aliases[6:9]] == [
        0x202000,
        0x406000,
        0x604000,
    ]
    assert [entry[4] for entry in aliases[9:12]] == [
        0x203000,
        0x407000,
        0x605000,
    ]
    assert offload_utils._uvm_original_alias_binding_hot_ranges(binding0) == (
        (0x400000, 8192),
        (0x404000, 8192),
    )


def test_uvm_original_expert_plan_requires_equal_complete_layer_selection(
    tmp_path,
):
    names = frozenset(
        {
            "model.layers.0.experts.w13_weight",
            "model.layers.0.experts.w2_weight",
            "model.layers.1.experts.w13_weight",
            "model.layers.1.experts.w2_weight",
        }
    )
    plan = offload_utils._build_graph_static_alias_plan(names)
    plan_file = tmp_path / "cold.json"
    plan_file.write_text(
        json.dumps({"selected_by_layer": {"0": [7, 1], "1": [4, 2]}}),
        encoding="utf-8",
    )
    assert offload_utils._load_graph_static_alias_expert_plan(
        str(plan_file), plan
    ) == {0: (1, 7), 1: (2, 4)}

    plan_file.write_text(
        json.dumps({"selected_by_layer": {"0": [7], "1": [4, 2]}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="same number of experts"):
        offload_utils._load_graph_static_alias_expert_plan(str(plan_file), plan)


def test_uvm_plan_classes_require_opt_in_for_equal_expert_widths(tmp_path):
    names = frozenset({"model.layers.0.experts.w13_weight"})
    name_plan = offload_utils._build_graph_static_alias_plan(names)
    for class_name, experts in (("base", [0, 1]), ("difference", [0, 2])):
        (tmp_path / f"{class_name}.json").write_text(
            json.dumps({"selected_by_layer": {"0": experts}}),
            encoding="utf-8",
        )
    manifest = {
        "schema": "vllm_uvm_static_alias_plan_classes_v1",
        "initial_class": "base",
        "classes": {
            class_name: {
                "capacity_gib": 1,
                "plan_file": f"{class_name}.json",
            }
            for class_name in ("base", "difference")
        },
    }
    manifest_file = tmp_path / "classes.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct expert widths"):
        offload_utils._load_graph_static_alias_plan_classes(
            str(manifest_file), name_plan
        )

    manifest["allow_equal_expert_widths"] = True
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    _, plans, capacities, initial = (
        offload_utils._load_graph_static_alias_plan_classes(
            str(manifest_file), name_plan
        )
    )
    assert initial == "base"
    assert plans == {"base": {0: (0, 1)}, "difference": {0: (0, 2)}}
    assert capacities == {"base": 1.0, "difference": 1.0}


def test_uvm_plan_classes_accept_one_fixed_tier(tmp_path):
    names = frozenset({"model.layers.0.experts.w13_weight"})
    name_plan = offload_utils._build_graph_static_alias_plan(names)
    (tmp_path / "cold70.json").write_text(
        json.dumps({"selected_by_layer": {"0": [1, 3]}}),
        encoding="utf-8",
    )
    manifest_file = tmp_path / "classes.json"
    manifest_file.write_text(
        json.dumps(
            {
                "schema": "vllm_uvm_static_alias_plan_classes_v1",
                "initial_class": "cold70",
                "classes": {
                    "cold70": {
                        "capacity_gib": 14.765625,
                        "plan_file": "cold70.json",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    files, plans, capacities, initial = (
        offload_utils._load_graph_static_alias_plan_classes(
            str(manifest_file), name_plan
        )
    )
    assert set(files) == {"cold70"}
    assert plans == {"cold70": {0: (1, 3)}}
    assert capacities == {"cold70": 14.765625}
    assert initial == "cold70"


@pytest.mark.parametrize(
    ("phase_mode", "eager_plan_classes", "expected"),
    [
        ("prefill_graph_remote_decode", False, True),
        ("prefill_remote_decode", True, True),
        ("token_threshold", True, True),
        ("token_threshold", False, False),
        ("remote_only", True, False),
    ],
)
def test_uvm_plan_class_phase_contract(
    phase_mode: str,
    eager_plan_classes: bool,
    expected: bool,
) -> None:
    assert (
        offload_utils._uvm_plan_class_phase_contract_allowed(
            phase_mode,
            eager_plan_classes,
        )
        is expected
    )


def test_uvm_original_expert_slices_release_gpu_chunks_before_map(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.device_allocator import uvm_pool_residency

    plan = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=3,
        local_name="experts.w13_weight",
        bank_id=1,
        consumer_index=0,
        consumer_count=1,
    )
    binding = offload_utils._UvmOriginalAliasBinding(
        full_parameter_name="model.layers.3.experts.w13_weight",
        arena=SimpleNamespace(),
        consumer=SimpleNamespace(),
        parameter=SimpleNamespace(),
        plan=plan,
        requested_bytes=2 * 64 * 1024,
        expert_ids=(2, 7),
        expert_stride_bytes=64 * 1024,
        consumer_ranges=(
            (0x400000, 64 * 1024),
            (0x420000, 64 * 1024),
        ),
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {binding.full_parameter_name: binding},
    )

    registered: list[tuple[str, int, int, str]] = []
    active_region_ids: set[str] = set()

    def register(region_id, base, length, *, state):
        assert region_id not in active_region_ids
        active_region_ids.add(region_id)
        registered.append((region_id, base, length, state))

    def release(region_ids, *, device, remote_read, include_audit):
        assert device == 0
        assert remote_read is False
        assert include_audit is True
        specs = {region_id: (base, length) for region_id, base, length, _ in registered}
        return {
            "ok": True,
            "statuses": {region_id: True for region_id in region_ids},
            "audits": {
                region_id: {
                    "ok": True,
                    "remote_read": False,
                    "registered_base": specs[region_id][0],
                    "registered_length": specs[region_id][1],
                    "base": specs[region_id][0],
                    "length": specs[region_id][1],
                    "app_state_after": "cpu",
                    "attempted_bytes": specs[region_id][1],
                    "released_bytes": specs[region_id][1],
                    "made_progress": True,
                }
                for region_id in region_ids
            },
        }

    monkeypatch.setattr(uvm_pool_residency, "register_uvm_region", register)
    reconcile_calls: list[tuple[str, ...]] = []

    def reconcile(region_ids, **kwargs):
        requested = tuple(region_ids)
        reconcile_calls.append(requested)
        return {
            "ok": True,
            "physical_cpu_only": len(requested) == 1,
            "reconciled": len(requested) == 1,
        }

    monkeypatch.setattr(
        uvm_pool_residency,
        "reconcile_uvm_regions_cpu_state_from_physical_residency",
        reconcile,
    )
    monkeypatch.setattr(uvm_pool_residency, "release_uvm_regions", release)
    unregistered: list[str] = []

    def unregister(region_ids):
        unregistered.extend(region_ids)
        for region_id in region_ids:
            active_region_ids.remove(region_id)
        return {"ok": True, "statuses": {region_id: True for region_id in region_ids}}

    monkeypatch.setattr(uvm_pool_residency, "unregister_uvm_regions", unregister)

    result = offload_utils._release_uvm_original_alias_consumer_gpu_chunks(
        device=0
    )

    assert result["ok"] is True
    assert result["required"] is True
    assert result["region_count"] == 2
    assert result["requested_bytes"] == 2 * 64 * 1024
    assert result["released_bytes"] == 2 * 64 * 1024
    assert result["made_progress_count"] == 2
    assert result["temporary_regions_unregistered"] is True
    assert len(result["geometry_sha256"]) == 64
    assert unregistered == [entry[0] for entry in registered]
    assert [entry[1:3] for entry in registered] == [
        (0x400000, 64 * 1024),
        (0x420000, 64 * 1024),
    ]
    assert active_region_ids == set()

    second = offload_utils._release_uvm_original_alias_consumer_gpu_chunks(
        device=0,
        known_cpu_ranges={(0x400000, 64 * 1024)},
    )
    assert second["ok"] is True
    assert second["known_cpu_range_count"] == 1
    assert second["known_cpu_state_reconciliation"]["physical_cpu_only"] is True
    assert [len(call) for call in reconcile_calls] == [2, 1, 2]
    assert second["temporary_regions_unregistered"] is True
    assert active_region_ids == set()
    assert unregistered == [entry[0] for entry in registered]


def test_uvm_original_initial_release_uses_full_parameter_shadow_cutover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import uvm

    plan = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=3,
        local_name="experts.w13_weight",
        bank_id=1,
        consumer_index=0,
        consumer_count=1,
    )
    binding = offload_utils._UvmOriginalAliasBinding(
        full_parameter_name="model.layers.3.experts.w13_weight",
        arena=SimpleNamespace(),
        consumer=SimpleNamespace(),
        parameter=SimpleNamespace(),
        plan=plan,
        requested_bytes=2 * 64 * 1024,
        expert_ids=(2, 7),
        expert_stride_bytes=64 * 1024,
        consumer_ranges=(
            (0x400000, 64 * 1024),
            (0x420000, 64 * 1024),
        ),
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {binding.full_parameter_name: binding},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT",
        {"enabled": True, "full_parameter_shadow": True},
    )

    cutover_calls: list[tuple[tuple[tuple[int, int], ...], int]] = []

    def cutover(ranges, *, device: int):
        normalized = tuple(ranges)
        cutover_calls.append((normalized, device))
        requested_bytes = sum(length for _, length in normalized)
        return {
            "ok": True,
            "range_count": len(normalized),
            "success_count": len(normalized),
            "failure_count": 0,
            "requested_bytes": requested_bytes,
            "released_gpu_bytes": requested_bytes,
            "retained_cpu_shadow_bytes": requested_bytes,
            "remote_mapped_bytes": requested_bytes,
        }

    monkeypatch.setattr(
        uvm,
        "driver_cutover_ranges_to_immutable_cpu_shadow",
        cutover,
    )
    monkeypatch.setattr(
        uvm,
        "driver_batch_query_ranges_residency_bytes",
        lambda ranges, **_kwargs: {
            "ok": True,
            "queried_bytes": sum(length for _, length in ranges),
            "cpu_bytes": sum(length for _, length in ranges),
            "gpu_bytes": 0,
            "any_bytes": sum(length for _, length in ranges),
            "unknown_bytes": 0,
        },
    )

    result = offload_utils._release_uvm_original_alias_consumer_gpu_chunks(
        device=0,
        known_cpu_ranges=((0x400000, 64 * 1024),),
    )

    assert result["ok"] is True
    assert result["reason"] == "immutable_cpu_shadow_cutover"
    assert result["region_count"] == 2
    assert result["requested_bytes"] == 2 * 64 * 1024
    assert result["attempted_bytes"] == 64 * 1024
    assert result["released_bytes"] == 64 * 1024
    assert result["made_progress_count"] == 1
    assert result["known_cpu_range_count"] == 1
    assert result["temporary_regions_unregistered"] is True
    assert len(result["geometry_sha256"]) == 64
    assert cutover_calls == [(((0x420000, 64 * 1024),), 0)]


def test_uvm_original_expert_slice_release_audit_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.device_allocator import uvm_pool_residency

    plan = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=0,
        local_name="experts.w2_weight",
        bank_id=0,
        consumer_index=0,
        consumer_count=1,
    )
    binding = offload_utils._UvmOriginalAliasBinding(
        full_parameter_name="model.layers.0.experts.w2_weight",
        arena=SimpleNamespace(),
        consumer=SimpleNamespace(),
        parameter=SimpleNamespace(),
        plan=plan,
        requested_bytes=64 * 1024,
        expert_ids=(4,),
        expert_stride_bytes=64 * 1024,
        consumer_ranges=((0x800000, 64 * 1024),),
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {binding.full_parameter_name: binding},
    )
    monkeypatch.setattr(
        uvm_pool_residency, "register_uvm_region", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        uvm_pool_residency,
        "reconcile_uvm_regions_cpu_state_from_physical_residency",
        lambda *args, **kwargs: {
            "ok": True,
            "physical_cpu_only": False,
            "reconciled": False,
        },
    )
    monkeypatch.setattr(
        uvm_pool_residency,
        "release_uvm_regions",
        lambda region_ids, **kwargs: {
            "ok": True,
            "statuses": {region_ids[0]: True},
            "audits": {
                region_ids[0]: {
                    "ok": True,
                    "remote_read": True,
                    "registered_base": 0x800000,
                    "registered_length": 64 * 1024,
                    "base": 0x800000,
                    "length": 64 * 1024,
                    "app_state_after": "cpu",
                }
            },
        },
    )
    monkeypatch.setattr(
        uvm_pool_residency,
        "unregister_uvm_regions",
        lambda region_ids: {
            "ok": True,
            "statuses": {region_id: True for region_id in region_ids},
        },
    )

    with pytest.raises(RuntimeError, match="did not prove completion"):
        offload_utils._release_uvm_original_alias_consumer_gpu_chunks(device=0)


def test_uvm_original_destroy_clear_contract():
    assert offload_utils._uvm_original_alias_destroy_cleared({"ok": True})
    assert offload_utils._uvm_original_alias_destroy_cleared(
        {
            "ok": False,
            "cache_cleared": True,
            "session_cleared": True,
            "remaining_alias_count": 0,
            "remaining_alias_bytes": 0,
            "remaining_backing_bytes": 0,
        }
    )
    assert not offload_utils._uvm_original_alias_destroy_cleared(
        {
            "ok": False,
            "session_cleared": True,
            "remaining_alias_count": 1,
            "remaining_alias_bytes": 4096,
            "remaining_backing_bytes": 0,
        }
    )


def test_uvm_original_64k_mode_keeps_512k_region_alignment(
    monkeypatch: pytest.MonkeyPatch,
):
    class PointerTensor:
        def __init__(self, pointer: int):
            self._pointer = pointer

        def data_ptr(self) -> int:
            return self._pointer

    plan = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=0,
        local_name="weight",
        bank_id=0,
        consumer_index=0,
        consumer_count=1,
    )
    arena = offload_utils._UvmOriginalAliasArena(
        upload_storage=PointerTensor(0x200000),
        upload=PointerTensor(0x200000),
        signature=((2 << 20,), (1,), torch.uint8, 0, 2 << 20, 1),
        requested_bytes=2 << 20,
        aligned_bytes=2 << 20,
        backing_layer_id=0x70000000,
        pte_alignment=0,
        pte_flags=0,
        region_bytes=512 * 1024,
    )
    binding = offload_utils._UvmOriginalAliasBinding(
        full_parameter_name="model.layers.0.weight",
        arena=arena,
        consumer=PointerTensor(0x400000),
        parameter=None,
        plan=plan,
        requested_bytes=2 << 20,
    )
    arena.consumers = {0: binding}
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS",
        {("model.layers", "weight", 0): arena},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {binding.full_parameter_name: binding},
    )
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE", "auto")

    alignment, flags, mode = offload_utils._uvm_original_alias_select_pte_mode()
    assert alignment == 64 * 1024
    assert flags != 0
    assert mode == "64K"
    assert arena.region_bytes % alignment == 0


def _set_uvm_destroy_state(
    monkeypatch: pytest.MonkeyPatch,
    *,
    session_id: int,
    state: str,
    cleanup_unresolved: bool = False,
) -> None:
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", session_id
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", state)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED",
        cleanup_unresolved,
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_DESTROY_RESULT", None
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SLOT_CLOSE_RESULT", None
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_OWNER_DEVICE_INDEX", 0)


def test_uvm_original_map_session_id_is_trusted_only_after_success():
    assert (
        offload_utils._uvm_original_alias_next_session_id(
            {"ok": True, "session_id": 17}, 0
        )
        == 17
    )
    assert (
        offload_utils._uvm_original_alias_next_session_id(
            {"ok": True, "session_id": 17}, 17
        )
        == 17
    )
    with pytest.raises(RuntimeError, match="did not succeed"):
        offload_utils._uvm_original_alias_next_session_id(
            {"ok": False, "session_id": 99}, 0
        )
    with pytest.raises(RuntimeError, match="changed session id"):
        offload_utils._uvm_original_alias_next_session_id(
            {"ok": True, "session_id": 99}, 17
        )


def test_uvm_original_failed_first_map_keeps_untrusted_session_unresolved(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.device_allocator import uvm, uvm_static_alias

    class ManagedTensor:
        def __init__(self, pointer: int, *, device: str = "cuda:0"):
            self._pointer = pointer
            self.device = torch.device(device)
            self.dtype = torch.uint8
            self.layout = torch.strided

        def data_ptr(self):
            return self._pointer

        def size(self):
            return (4096,)

        def stride(self):
            return (1,)

        def is_pinned(self):
            return self.device.type == "cpu"

        def copy_(self, source, *, non_blocking=False):
            return self

    name = "model.layers.0.weight"
    plan = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=0,
        local_name="weight",
        bank_id=0,
        consumer_index=0,
        consumer_count=1,
    )
    consumer = ManagedTensor(0x400000)
    upload = ManagedTensor(0x200000)
    arena = offload_utils._UvmOriginalAliasArena(
        upload_storage=upload,
        upload=upload,
        signature=((4096,), (1,), torch.uint8, 0, 4096, 1),
        requested_bytes=4096,
        aligned_bytes=4096,
        backing_layer_id=0x70000000,
        pte_alignment=4096,
        pte_flags=0,
        region_bytes=4096,
    )
    binding = offload_utils._UvmOriginalAliasBinding(
        full_parameter_name=name,
        arena=arena,
        consumer=consumer,
        parameter=SimpleNamespace(data=consumer),
        plan=plan,
        requested_bytes=4096,
    )
    arena.consumers = {0: binding}

    _set_uvm_destroy_state(monkeypatch, session_id=0, state="collecting")
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN", {name: plan}
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS", {name: binding}
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS",
        {("model.layers", "weight", 0): arena},
    )
    monkeypatch.setattr(
        offload_utils, "_require_all_copy_exact_names_seen", lambda: None
    )
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_alias_select_pte_mode",
        lambda: (4096, 0, "4K"),
    )
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_alias_preallocate_entries",
        lambda: [(0x200000, 4096, 0x70000000, 0, 0, 0, 4096)],
    )
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_alias_map_entries",
        lambda flags: [(0x70000000, 0, 0, flags, 0x400000, 4096, 0)],
    )
    monkeypatch.setattr(
        offload_utils.torch,
        "empty_strided",
        lambda **kwargs: ManagedTensor(0x800000, device="cpu"),
    )
    monkeypatch.setattr(offload_utils.torch.cuda, "synchronize", lambda device=0: None)
    monkeypatch.setattr(
        uvm, "probe_driver_capable_range", lambda *args: (True, "ok", 0)
    )
    monkeypatch.setattr(
        uvm,
        "batch_prepare_ranges_for_native_remote_access_with_executor",
        lambda ranges, **kwargs: (len(ranges), 0),
    )
    monkeypatch.setattr(
        uvm,
        "driver_slot_preallocate_ranges",
        lambda entries, **kwargs: {"ok": True},
    )
    monkeypatch.setattr(uvm, "_uvm_slot_direct_sessions", {0: (123, b"uuid")})
    close_calls = []
    monkeypatch.setattr(
        uvm,
        "close_uvm_slot_direct_session",
        lambda **kwargs: close_calls.append(kwargs) or {"ok": True},
    )
    monkeypatch.setattr(
        uvm_static_alias,
        "driver_static_alias_map",
        lambda *args, **kwargs: {
            "ok": False,
            "reason": "fault_injected",
            "session_id": 99,
            "made_progress": False,
            "poisoned": False,
            "side_effect_indeterminate": False,
        },
    )

    with pytest.raises(RuntimeError, match="MAP failed"):
        offload_utils.finalize_cpu_offload_graph_static_alias()

    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID == 0
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "cleanup_pending"
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED is True
    assert close_calls == []
    diagnostics = offload_utils.get_cpu_offload_graph_static_alias_diagnostics()
    assert diagnostics["cleanup_pending"] is True
    assert diagnostics["cleanup_unresolved"] is True


def test_uvm_original_destroy_retries_failed_slot_close(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.device_allocator import uvm, uvm_static_alias

    _set_uvm_destroy_state(monkeypatch, session_id=7, state="sealed")
    monkeypatch.setattr(uvm, "_uvm_slot_direct_sessions", {0: (123, b"uuid")})
    monkeypatch.setattr(
        uvm_static_alias,
        "driver_static_alias_destroy",
        lambda *args, **kwargs: {"ok": True, "session_cleared": True},
    )
    close_results = iter(
        [
            {"ok": False, "reason": "fault_injected", "closed": False},
            {"ok": True, "reason": "closed", "closed": True},
        ]
    )
    monkeypatch.setattr(
        uvm, "close_uvm_slot_direct_session", lambda **kwargs: next(close_results)
    )

    first = offload_utils.destroy_cpu_offload_graph_static_alias(graphs_destroyed=True)
    assert first["ok"] is False
    assert first["reason"] == "slot_session_close_failed"
    assert first["state"] == "cleanup_pending"
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID == 0
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "cleanup_pending"

    second = offload_utils.destroy_cpu_offload_graph_static_alias(graphs_destroyed=True)
    assert second["ok"] is True
    assert second["reason"] == "slot_session_close_retried"
    assert second["state"] == "remote"
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "remote"


def test_uvm_original_unresolved_cleanup_never_closes_backing(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.device_allocator import uvm

    _set_uvm_destroy_state(
        monkeypatch,
        session_id=0,
        state="cleanup_pending",
        cleanup_unresolved=True,
    )
    monkeypatch.setattr(uvm, "_uvm_slot_direct_sessions", {0: (123, b"uuid")})
    close_calls = []
    monkeypatch.setattr(
        uvm,
        "close_uvm_slot_direct_session",
        lambda **kwargs: close_calls.append(kwargs) or {"ok": True},
    )

    result = offload_utils.destroy_cpu_offload_graph_static_alias(graphs_destroyed=True)
    assert result["ok"] is False
    assert result["reason"] == "cleanup_unresolved_requires_process_teardown"
    assert close_calls == []
    with pytest.raises(RuntimeError, match="cleanup is active"):
        offload_utils._configure_cpu_offload_exact_names(0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("off", "off"),
        ("hybrid", "prefill_remote_decode"),
        ("alias-prefill-remote-decode", "prefill_remote_decode"),
        ("token-threshold", "token_threshold"),
        ("token_adaptive", "token_threshold"),
        ("dual-topology-graph", "prefill_graph_remote_decode"),
        ("remote", "remote_only"),
    ],
)
def test_uvm_original_phase_mode_parser(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    expected: str,
):
    monkeypatch.setenv("VLLM_UVM_ORIGINAL_VA_PHASE_MODE", raw)
    assert offload_utils._configured_uvm_original_va_phase_mode() == expected


def test_uvm_original_token_threshold_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_UVM_ORIGINAL_VA_TOKEN_THRESHOLD", raising=False)
    assert offload_utils._configured_uvm_original_va_token_threshold() == 64
    monkeypatch.setenv("VLLM_UVM_ORIGINAL_VA_TOKEN_THRESHOLD", "128")
    assert offload_utils._configured_uvm_original_va_token_threshold() == 128


def test_uvm_original_resume_token_threshold_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "VLLM_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD", raising=False
    )
    assert (
        offload_utils._configured_uvm_original_va_resume_token_threshold(64)
        == 65
    )
    monkeypatch.setenv(
        "VLLM_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD", "128"
    )
    assert (
        offload_utils._configured_uvm_original_va_resume_token_threshold(64)
        == 128
    )


@pytest.mark.parametrize("raw", ["64", "63", "not-an-int"])
def test_uvm_original_resume_token_threshold_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    monkeypatch.setenv(
        "VLLM_UVM_ORIGINAL_VA_RESUME_TOKEN_THRESHOLD", raw
    )
    with pytest.raises(ValueError, match="must be .*greater"):
        offload_utils._configured_uvm_original_va_resume_token_threshold(64)


@pytest.mark.parametrize("raw", ["0", "-1", "not-an-int"])
def test_uvm_original_token_threshold_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    monkeypatch.setenv("VLLM_UVM_ORIGINAL_VA_TOKEN_THRESHOLD", raw)
    with pytest.raises(ValueError, match="must be a positive int"):
        offload_utils._configured_uvm_original_va_token_threshold()


def test_uvm_original_phase_runtime_allows_explicit_eager(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE",
        "prefill_remote_decode",
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_dbo=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="NONE"),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    offload_utils.validate_cpu_offload_graph_static_alias_runtime(config)


def test_uvm_token_threshold_runtime_requires_eager_and_park_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE",
        "token_threshold",
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED", True
    )
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_dbo=False,
    )
    eager = SimpleNamespace(
        parallel_config=parallel,
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="NONE"),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    offload_utils.validate_cpu_offload_graph_static_alias_runtime(eager)

    graph = SimpleNamespace(
        parallel_config=parallel,
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL"),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    with pytest.raises(RuntimeError, match="requires cudagraph_mode=NONE"):
        offload_utils.validate_cpu_offload_graph_static_alias_runtime(graph)

    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED", False
    )
    with pytest.raises(RuntimeError, match="requires PARK/RESUME"):
        offload_utils.validate_cpu_offload_graph_static_alias_runtime(eager)


def test_uvm_original_fast_forward_copies_prefill_then_reads_remote(
    monkeypatch: pytest.MonkeyPatch,
    fake_cuda,
):
    name = "model.layers.0.weight"
    plan = offload_utils._GraphStaticAliasNamePlan(
        layer_container="model.layers",
        layer_index=0,
        local_name="weight",
        bank_id=0,
        consumer_index=0,
        consumer_count=1,
    )
    consumer = _FakeTensor(pointer=0x400000)
    upload = _FakeTensor(pointer=0x200000)
    source = _FakeTensor(device="cpu", pinned=True, pointer=0x600000)
    direct_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class DirectModule(_FakeModule):
        def __init__(self) -> None:
            super().__init__({"weight": consumer})
            self.state_dict_calls = 0

        def state_dict(self) -> dict[str, _FakeTensor]:
            self.state_dict_calls += 1
            return super().state_dict()

        def forward(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
            direct_calls.append((args, kwargs))
            return ("direct", *args, kwargs)

    module = DirectModule()
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_BACKEND", "copy")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_BYTES", 0)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_MAX_BYTES", 1 << 20)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_ORDER", 0)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_EXACT_NAMES_FILE", "test")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_EXACT_NAMES", frozenset({name}))
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_EXACT_PENDING", {name})
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_EXACT_SEEN", set())
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_STATIC_BANK", False)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN", {name: plan}
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "collecting"
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", False)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_OWNER_THREAD_ID", None)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_OWNER_DEVICE_INDEX", None)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_OWNER_STREAM_ID", None)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS", 0
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT", 0)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES", 0
    )

    def create_binding(*, parameter, tensor, requested_bytes, **kwargs):
        return offload_utils._UvmOriginalAliasBinding(
            full_parameter_name=name,
            arena=SimpleNamespace(upload=upload, aligned_bytes=requested_bytes),
            consumer=tensor,
            parameter=parameter,
            plan=plan,
            requested_bytes=requested_bytes,
            parameter_bytes=requested_bytes,
            consumer_ranges=((tensor.data_ptr(), requested_bytes),),
            source=source,
        )

    monkeypatch.setattr(
        offload_utils, "_create_uvm_original_alias_binding", create_binding
    )
    offload_utils.maybe_offload_to_cpu(module, module_prefix="model.layers.0")
    assert module.state_dict_calls == 1

    def forbidden(*args: Any, **kwargs: Any):
        raise AssertionError("generic state_dict/functional_call path was used")

    monkeypatch.setattr(module, "state_dict", forbidden)
    monkeypatch.setattr(offload_utils, "functional_call", forbidden)
    monkeypatch.setattr(
        offload_utils.torch.cuda, "is_current_stream_capturing", lambda: True
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_CAPTURE_PHASE", True
    )

    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "sealed")
    assert module.forward("prefill", request_id=1) == (
        "direct",
        "prefill",
        {"request_id": 1},
    )
    assert upload.copy_calls == [(source, True)]
    assert module._parameters["weight"].data is consumer

    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "remote")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", True)
    assert module.forward("decode", request_id=1) == (
        "direct",
        "decode",
        {"request_id": 1},
    )
    assert upload.copy_calls == [(source, True)]
    assert direct_calls == [
        (("prefill",), {"request_id": 1}),
        (("decode",), {"request_id": 1}),
    ]
    assert offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_FORWARD_CALLS == 2
    assert offload_utils._CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT == 2
    assert offload_utils._CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES == 16

    module._parameters["weight"].data = _FakeTensor(pointer=0x800000)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "sealed")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", False)
    with pytest.raises(RuntimeError, match="changed device or VA"):
        module.forward("stale")
    assert upload.copy_calls == [(source, True)]
    assert len(direct_calls) == 2


def test_uvm_dual_topology_runtime_requires_full_separate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE",
        "prefill_graph_remote_decode",
    )
    parallel = SimpleNamespace(
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_dbo=False,
    )
    valid = SimpleNamespace(
        parallel_config=parallel,
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL_SEPARATE"),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    offload_utils.validate_cpu_offload_graph_static_alias_runtime(valid)

    invalid = SimpleNamespace(
        parallel_config=parallel,
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL"),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    with pytest.raises(RuntimeError, match="requires cudagraph_mode=FULL_SEPARATE"):
        offload_utils.validate_cpu_offload_graph_static_alias_runtime(invalid)


def test_uvm_remote_only_runtime_rejects_full_separate(
    monkeypatch: pytest.MonkeyPatch,
):
    """Fail closed: the always-remote large-Prefill Graph path can fault."""
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE",
        "remote_only",
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            enable_dbo=False,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(name="FULL_SEPARATE"),
            mode=SimpleNamespace(name="NONE"),
        ),
    )
    with pytest.raises(RuntimeError, match="FULL_SEPARATE is reserved"):
        offload_utils.validate_cpu_offload_graph_static_alias_runtime(config)


def test_full_separate_dispatcher_partitions_prefill_and_decode_keys():
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import BatchDescriptor
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    config = SimpleNamespace(
        lora_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_SEPARATE,
            cudagraph_capture_sizes=[1, 4, 314, 1217],
            cudagraph_specialize_lora=False,
            is_attention_compiled_piecewise=lambda: False,
        ),
    )
    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(CUDAGraphMode.FULL_SEPARATE, 1)
    assert dispatcher.cudagraph_keys[CUDAGraphMode.FULL] == {
        BatchDescriptor(1, uniform_decode=True),
        BatchDescriptor(4, uniform_decode=True),
        BatchDescriptor(314, uniform_decode=False),
        BatchDescriptor(1217, uniform_decode=False),
    }


def test_uvm_original_remote_capture_issues_zero_h2d_certificate(
    monkeypatch: pytest.MonkeyPatch,
):
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
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "remote")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT",
        {"ok": True, "requested_bytes": 4096},
    )
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_parameter_va_snapshot",
        lambda: (0x400000,),
    )
    monkeypatch.setattr(
        offload_utils.torch.cuda,
        "current_stream",
        lambda device=None: SimpleNamespace(cuda_stream=17),
    )

    with offload_utils.cpu_offload_graph_static_alias_capture_context():
        offload_utils._CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT = 3
        offload_utils._CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES = 0

    certificate = offload_utils._CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE
    assert certificate is not None
    assert certificate["topology"] == "remote"
    assert certificate["parameter_vas"] == (0x400000,)
    assert certificate["capture_forward_count"] == 3
    assert certificate["actual_h2d_bytes"] == 0
    assert offload_utils._CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED is True
    assert offload_utils._CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY == "remote"


def test_uvm_original_dual_graph_certificates_survive_alias_capture(
    monkeypatch: pytest.MonkeyPatch,
):
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
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_parameter_va_snapshot",
        lambda: (0x400000, 0x800000),
    )
    monkeypatch.setattr(
        offload_utils.torch.cuda,
        "current_stream",
        lambda device=None: SimpleNamespace(cuda_stream=17),
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT", 0)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED", False)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED", False)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_ALIAS_GRAPHS_DESTROYED", False)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE", None)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE", None)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE",
        None,
    )

    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "remote")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT",
        {"ok": True, "requested_bytes": 8192},
    )
    with offload_utils.cpu_offload_graph_static_alias_capture_context(
        expected_topology="remote"
    ):
        offload_utils._CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT = 2
        offload_utils._CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES = 0

    remote_certificate = dict(
        offload_utils._CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE or {}
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "sealed")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", 7)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", False)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES",
        4096,
    )
    with offload_utils.cpu_offload_graph_static_alias_capture_context(
        expected_topology="sealed"
    ):
        offload_utils._CPU_OFFLOAD_UVM_CAPTURE_FORWARD_COUNT = 4
        offload_utils._CPU_OFFLOAD_UVM_CAPTURE_ACTUAL_H2D_BYTES = 16384

    assert offload_utils._CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE == (
        remote_certificate
    )
    assert offload_utils._CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED is True
    assert offload_utils._CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED is True
    assert offload_utils._CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE == {
        "schema": "uvm_original_va_alias_graph_certificate_v1",
        "topology": "sealed",
        "phase_mode": "prefill_graph_remote_decode",
        "parameter_vas": (0x400000, 0x800000),
        "capture_forward_count": 4,
        "actual_h2d_bytes": 16384,
        "session_id": 7,
        "backing_bytes": 4096,
    }

    partition = offload_utils.certify_cpu_offload_uvm_original_va_dual_graph_partition(
        remote_entries=[
            {"num_tokens": 1, "uniform_decode": True, "has_lora": False},
            {"num_tokens": 4, "uniform_decode": True, "has_lora": False},
        ],
        alias_entries=[
            {"num_tokens": 314, "uniform_decode": False, "has_lora": False},
            {"num_tokens": 1217, "uniform_decode": False, "has_lora": False},
        ],
    )
    assert partition["remote_entry_count"] == 2
    assert partition["alias_entry_count"] == 2

    teardown = (
        offload_utils.certify_cpu_offload_uvm_original_va_alias_graphs_destroyed(
            destroyed_alias_entries=[
                {
                    "num_tokens": 314,
                    "uniform_decode": False,
                    "has_lora": False,
                },
                {
                    "num_tokens": 1217,
                    "uniform_decode": False,
                    "has_lora": False,
                },
            ],
            remaining_remote_entries=[
                {"num_tokens": 1, "uniform_decode": True, "has_lora": False},
                {"num_tokens": 4, "uniform_decode": True, "has_lora": False},
            ],
        )
    )
    assert teardown["destroyed_alias_graph_count"] == 2
    assert teardown["remaining_remote_graph_count"] == 2
    ok, reason = offload_utils._uvm_original_alias_can_preserve_remote_graphs(
        synchronize=True
    )
    assert ok, reason


def test_uvm_original_preserved_graph_requires_zero_h2d_certificate(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE",
        "prefill_remote_decode",
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED", True)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY", "remote")
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_parameter_va_snapshot",
        lambda: (0x400000,),
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE",
        {
            "topology": "remote",
            "capture_forward_count": 1,
            "actual_h2d_bytes": 4096,
            "parameter_vas": (0x400000,),
        },
    )
    ok, reason = offload_utils._uvm_original_alias_can_preserve_remote_graphs(
        synchronize=True
    )
    assert not ok
    assert reason == "remote_graph_certificate_contains_h2d"

    offload_utils._CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE[
        "actual_h2d_bytes"
    ] = 0
    ok, reason = offload_utils._uvm_original_alias_can_preserve_remote_graphs(
        synchronize=True
    )
    assert ok
    assert reason == "ok"


@pytest.mark.parametrize(
    ("post_map_audit", "runtime_hot_audit"),
    [(True, True), (True, False), (False, True)],
)
def test_uvm_original_transition_to_remote_audits_metadata(
    monkeypatch: pytest.MonkeyPatch,
    post_map_audit: bool,
    runtime_hot_audit: bool,
):
    from vllm.device_allocator import uvm

    class PointerTensor:
        def __init__(self, pointer: int):
            self._pointer = pointer

        def data_ptr(self) -> int:
            return self._pointer

    pointer = PointerTensor(0x400000)
    binding = SimpleNamespace(
        full_parameter_name="model.layers.0.weight",
        parameter=SimpleNamespace(data=pointer),
        consumer=pointer,
        requested_bytes=4096,
        parameter_bytes=12288,
        expert_ids=(0,),
        consumer_ranges=((0x400000, 4096),),
    )
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
    if not post_map_audit:
        monkeypatch.setenv(
            "VLLM_UVM_RUNTIME_POST_MAP_RESIDENCY_AUDIT", "0"
        )
    if not runtime_hot_audit:
        monkeypatch.setenv("VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT", "0")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", 7)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {binding.full_parameter_name: binding},
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT", (0x400000,)
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_HISTORY", [])
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY", []
    )
    monkeypatch.setattr(offload_utils.torch.cuda, "synchronize", lambda device=0: None)

    destroy_calls: list[dict[str, Any]] = []

    def destroy(**kwargs):
        destroy_calls.append(kwargs)
        offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID = 0
        offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "remote"
        return {"ok": True, "state": "remote", "session_id": 0}

    monkeypatch.setattr(
        offload_utils, "destroy_cpu_offload_graph_static_alias", destroy
    )
    remote = {
        "ok": True,
        "reason": "ok",
        "requested_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "requested_bytes": 4096,
        "cpu_resident_bytes": 4096,
        "remote_mapped_bytes": 4096,
        "covered_bytes": 4096,
        "unknown_bytes": 0,
    }
    monkeypatch.setattr(
        uvm,
        "driver_prepare_ranges_remote_read_audited",
        lambda ranges, **kwargs: remote,
    )
    query_calls: list[tuple[tuple[int, int], ...]] = []

    def query_residency(ranges, **kwargs):
        normalized = tuple((int(base), int(length)) for base, length in ranges)
        query_calls.append(normalized)
        if normalized == ((0x400000, 4096),):
            return {
                "ok": True,
                "queried_bytes": 4096,
                "cpu_bytes": 4096,
                "gpu_bytes": 0,
                "any_bytes": 4096,
                "unknown_bytes": 0,
            }
        assert normalized == ((0x401000, 8192),)
        return {
            "ok": True,
            "queried_bytes": 8192,
            "cpu_bytes": 4096,
            "gpu_bytes": 4096,
            "any_bytes": 8192,
            "unknown_bytes": 0,
        }

    monkeypatch.setattr(
        uvm,
        "driver_batch_query_ranges_residency_bytes",
        query_residency,
    )
    if not (post_map_audit and runtime_hot_audit):
        monkeypatch.setattr(
            offload_utils,
            "_uvm_original_alias_expected_shadow_cpu_bytes",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("disabled hot audit computed shadow coverage")
            ),
        )

    result = offload_utils.transition_cpu_offload_uvm_original_va_to_remote(
        reason="test_cutover",
        preserve_remote_graphs=False,
        runtime_remote_ranges=((0x401000, 4096),),
    )
    assert result["ok"] is True
    assert result["pre_state"] == "sealed"
    assert result["post_state"] == "remote"
    assert result["parameter_vas_before"] == (0x400000,)
    assert result["parameter_vas_after"] == (0x400000,)
    assert result["remote_map"] == remote
    assert result["runtime_remote_range_count"] == 1
    assert result["runtime_remote_bytes"] == 4096
    assert result["post_map_physical_audit_enabled"] is post_map_audit
    assert result["hot_physical_audit_requested"] is runtime_hot_audit
    hot_audit_enabled = post_map_audit and runtime_hot_audit
    assert result["hot_physical_audit_enabled"] is hot_audit_enabled
    assert result["hot_physical_audit_diagnostics_skipped"] is (
        not hot_audit_enabled
    )
    assert result["hot_runtime_remote_bytes"] == (
        4096 if hot_audit_enabled else 0
    )
    assert result["hot_expected_gpu_bytes"] == (
        4096 if hot_audit_enabled else 0
    )
    if hot_audit_enabled:
        assert result["physical_residency"]["cpu_bytes"] == 4096
        assert result["hot_physical_residency"]["gpu_bytes"] == 4096
        assert query_calls == [
            ((0x400000, 4096),),
            ((0x401000, 8192),),
        ]
    elif post_map_audit:
        assert result["physical_residency"]["cpu_bytes"] == 4096
        assert result["hot_physical_residency"] is None
        assert query_calls == [((0x400000, 4096),)]
    else:
        assert result["physical_residency"] is None
        assert result["hot_physical_residency"] is None
        assert query_calls == []
    assert offload_utils._CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE is True
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "remote"
    assert destroy_calls == [
        {
            "graphs_destroyed": True,
            "synchronize": True,
            "remote_graphs_preserved": False,
            "defer_cpu_restore": True,
        }
    ]


def test_uvm_original_full_dual_view_skips_remote_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import uvm, uvm_static_alias

    pointer = _FakeTensor(shape=(1024,), pointer=0x400000)
    binding = SimpleNamespace(
        full_parameter_name="model.layers.0.weight",
        parameter=SimpleNamespace(data=pointer),
        consumer=pointer,
        requested_bytes=4096,
        parameter_bytes=4096,
        expert_ids=(0,),
        consumer_ranges=((0x400000, 4096),),
    )
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
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_PARK_RESUME_ENABLED", True
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "sealed"
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", 7
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {binding.full_parameter_name: binding},
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT", (0x400000,)
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_HISTORY", [])
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY", []
    )
    monkeypatch.setattr(offload_utils.torch.cuda, "synchronize", lambda _=0: None)
    monkeypatch.setattr(
        offload_utils, "_uvm_original_alias_hot_cuda_read_gate", lambda **_: None
    )
    monkeypatch.setattr(
        uvm_static_alias,
        "driver_static_alias_park",
        lambda *_args, **_kwargs: {
            "ok": True,
            "dual_view_full_coverage": True,
            "dual_view_block_count": 1,
        },
    )
    monkeypatch.setattr(
        uvm,
        "driver_prepare_ranges_remote_read_audited",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("full dual view reinstalled CPU leaf mappings")
        ),
    )
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_REQUIRE_FULL_DUAL_VIEW", "1")
    monkeypatch.setenv("VLLM_UVM_RUNTIME_POST_MAP_RESIDENCY_AUDIT", "0")
    monkeypatch.setenv("VLLM_UVM_RUNTIME_HOT_RESIDENCY_AUDIT", "0")

    result = offload_utils.transition_cpu_offload_uvm_original_va_to_remote(
        reason="test_full_dual_cutover",
        preserve_remote_graphs=True,
    )

    assert result["ok"] is True
    assert result["post_state"] == "remote_parked"
    assert result["remote_map_operation"] == "prebuilt_full_dual_view"
    assert result["remote_map"]["reason"] == "prebuilt_full_dual_view"
    assert result["remote_map"]["ioctl_calls"] == 0
    assert result["remote_map"]["covered_bytes"] == 4096


def test_uvm_original_profile_all_remote_then_restores_hot_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import uvm

    pointer = _FakeTensor(shape=(2, 1), pointer=0x400000)
    binding = SimpleNamespace(
        full_parameter_name="model.layers.0.weight",
        parameter=SimpleNamespace(data=pointer),
        consumer=pointer,
        requested_bytes=4096,
        parameter_bytes=8192,
        expert_ids=(1,),
        consumer_ranges=((0x401000, 4096),),
    )
    monkeypatch.setenv("VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE", "1")
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
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "sealed")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", False)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE", False
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {binding.full_parameter_name: binding},
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT", (0x400000,)
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_HISTORY", [])
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY", []
    )
    monkeypatch.setattr(offload_utils.torch.cuda, "synchronize", lambda device=0: None)
    empty_cache_calls: list[str] = []
    monkeypatch.setattr(
        offload_utils.torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.append("empty_cache"),
    )
    monkeypatch.setattr(
        offload_utils.torch.cuda,
        "memory_reserved",
        lambda device=0: 1234,
    )
    monkeypatch.setattr(
        offload_utils.torch.cuda,
        "mem_get_info",
        lambda device=0: (512 * 1024 * 1024, 32 * 1024 * 1024 * 1024),
    )

    def destroy(**_kwargs):
        offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID = 0
        offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "remote"
        return {"ok": True, "state": "remote", "session_id": 0}

    monkeypatch.setattr(
        offload_utils, "destroy_cpu_offload_graph_static_alias", destroy
    )
    remote_calls: list[tuple[tuple[int, int], ...]] = []

    def prepare_cpu(ranges, **_kwargs):
        normalized = tuple(ranges)
        assert normalized == ((0x400000, 4096),)
        return len(normalized), "cpp_batch_native_cuda", len(normalized), len(
            normalized
        ), len(normalized)

    def prepare_remote(ranges, **_kwargs):
        normalized = tuple(ranges)
        remote_calls.append(normalized)
        total = sum(length for _, length in normalized)
        return {
            "ok": True,
            "requested_count": len(normalized),
            "success_count": len(normalized),
            "failure_count": 0,
            "requested_bytes": total,
            "cpu_resident_bytes": total,
            "remote_mapped_bytes": total,
            "covered_bytes": total,
            "unknown_bytes": 0,
        }

    hot_on_gpu = False

    def advise_hot(ranges, _device):
        return len(tuple(ranges))

    def prefetch_hot(ranges, _device, *, executor):
        nonlocal hot_on_gpu
        normalized = tuple(ranges)
        hot_on_gpu = True
        return len(normalized), executor, 0

    def query(ranges, **_kwargs):
        normalized = tuple(ranges)
        total = sum(length for _, length in normalized)
        is_hot = normalized == ((0x400000, 4096),)
        gpu_bytes = total if is_hot and hot_on_gpu else 0
        return {
            "ok": True,
            "queried_bytes": total,
            "cpu_bytes": total - gpu_bytes,
            "gpu_bytes": gpu_bytes,
            "any_bytes": total,
            "unknown_bytes": 0,
        }

    monkeypatch.setattr(
        uvm,
        "batch_prepare_ranges_for_native_remote_access_with_executor",
        prepare_cpu,
    )
    monkeypatch.setattr(
        uvm, "driver_prepare_ranges_remote_read_audited", prepare_remote
    )
    monkeypatch.setattr(
        uvm, "batch_advise_ranges_preferred_location", advise_hot
    )
    monkeypatch.setattr(
        uvm, "batch_prefetch_ranges_to_device_with_executor", prefetch_hot
    )
    monkeypatch.setattr(uvm, "driver_batch_query_ranges_residency_bytes", query)

    prepared = offload_utils.prepare_cpu_offload_uvm_original_va_profile_topology()
    assert prepared["ok"] is True
    assert prepared["reason"] == "prepared"
    assert prepared["hot_bytes"] == 4096
    assert offload_utils.cpu_offload_uvm_original_va_profile_all_remote_active()
    assert remote_calls == [((0x400000, 4096),), ((0x401000, 4096),)]
    empty_cache_calls.clear()

    restored = (
        offload_utils.reconcile_cpu_offload_uvm_original_va_remote_graph_residency(
            preserve_allocator_cache=True
        )
    )
    assert restored["ok"] is True
    assert restored["allocator_cache_preserved"] is True
    assert restored["allocator_reserved_before"] == 1234
    assert restored["allocator_reserved_after_cleanup"] == 1234
    assert restored["allocator_reserved_after"] == 1234
    assert empty_cache_calls == []
    assert restored["profile_all_remote_before"] is True
    assert restored["profile_all_remote_after"] is False
    assert restored["hot_physical_residency"]["gpu_bytes"] == 4096
    assert not offload_utils.cpu_offload_uvm_original_va_profile_all_remote_active()


def test_uvm_original_profile_uses_full_parameter_shadow_cutover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import uvm

    pointer = _FakeTensor(shape=(2, 1), pointer=0x400000)
    binding = SimpleNamespace(
        full_parameter_name="model.layers.0.weight",
        parameter=SimpleNamespace(data=pointer),
        consumer=pointer,
        requested_bytes=4096,
        parameter_bytes=8192,
        expert_ids=(1,),
        consumer_ranges=((0x401000, 4096),),
    )
    monkeypatch.setenv("VLLM_UVM_PROFILE_ALL_EXPERTS_REMOTE", "1")
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
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "sealed")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", False)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_PROFILE_ALL_REMOTE_ACTIVE", False
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {binding.full_parameter_name: binding},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT",
        {"enabled": True, "full_parameter_shadow": True},
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY", []
    )
    monkeypatch.setattr(offload_utils.torch.cuda, "synchronize", lambda device=0: None)

    cutover_calls: list[tuple[tuple[int, int], ...]] = []
    hot_cutover_done = False

    def shadow_cutover(ranges, **_kwargs):
        nonlocal hot_cutover_done
        normalized = tuple(ranges)
        cutover_calls.append(normalized)
        hot_cutover_done = True
        total = sum(length for _, length in normalized)
        return {
            "ok": True,
            "range_count": len(normalized),
            "success_count": len(normalized),
            "failure_count": 0,
            "requested_bytes": total,
            "released_gpu_bytes": total,
            "retained_cpu_shadow_bytes": total,
            "remote_mapped_bytes": total,
            "ioctl_calls": 1,
        }

    monkeypatch.setattr(
        uvm, "driver_cutover_ranges_to_immutable_cpu_shadow", shadow_cutover
    )
    monkeypatch.setattr(
        uvm,
        "batch_prepare_ranges_for_native_remote_access_with_executor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("full Parameter shadow used native migration")
        ),
    )
    monkeypatch.setattr(
        uvm,
        "driver_prepare_ranges_remote_read_audited",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("shadow cutover reinstalled the hot CPU mapping")
        ),
    )
    transition_calls: list[dict[str, object]] = []

    def transition(**kwargs):
        transition_calls.append(kwargs)
        return {"ok": True, "post_state": "remote_parked"}

    monkeypatch.setattr(
        offload_utils,
        "transition_cpu_offload_uvm_original_va_to_remote",
        transition,
    )

    def query(ranges, **_kwargs):
        normalized = tuple(ranges)
        total = sum(length for _, length in normalized)
        return {
            "ok": True,
            "queried_bytes": total,
            "cpu_bytes": total,
            "gpu_bytes": 0 if hot_cutover_done else total,
            "any_bytes": total,
            "unknown_bytes": 0,
        }

    monkeypatch.setattr(uvm, "driver_batch_query_ranges_residency_bytes", query)

    prepared = offload_utils.prepare_cpu_offload_uvm_original_va_profile_topology()

    assert prepared["ok"] is True
    assert prepared["hot_prepare_operation"] == "immutable_cpu_shadow_cutover"
    assert prepared["hot_prepare"][1] == "immutable_cpu_shadow_cutover"
    assert prepared["hot_shadow_cutover"]["released_gpu_bytes"] == 4096
    assert prepared["hot_remote_map"]["reason"] == "immutable_cpu_shadow_cutover"
    assert cutover_calls == [((0x400000, 4096),)]
    assert transition_calls[0]["runtime_remote_ranges"] == [
        (0x400000, 4096)
    ]


def test_sealed_hot_reconcile_skips_prefetch_when_already_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import uvm

    pointer = _FakeTensor(shape=(2, 1), pointer=0x400000)
    binding = SimpleNamespace(
        full_parameter_name="model.layers.0.weight",
        parameter=SimpleNamespace(data=pointer),
        consumer=pointer,
        requested_bytes=4096,
        parameter_bytes=8192,
        expert_ids=(1,),
        consumer_ranges=((0x401000, 4096),),
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "sealed"
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", False)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {binding.full_parameter_name: binding},
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT", (0x400000,)
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT", None
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY", []
    )
    monkeypatch.setattr(offload_utils.torch.cuda, "synchronize", lambda _=0: None)
    monkeypatch.setattr(
        offload_utils.torch.cuda,
        "mem_get_info",
        lambda _=0: (512 * 1024 * 1024, 32 * 1024 * 1024 * 1024),
    )
    monkeypatch.setattr(
        offload_utils.torch.cuda, "memory_reserved", lambda _=0: 1234
    )
    monkeypatch.setattr(
        uvm,
        "driver_batch_query_ranges_residency_bytes",
        lambda ranges, **_kwargs: {
            "ok": True,
            "queried_bytes": sum(length for _, length in ranges),
            "cpu_bytes": 0,
            "gpu_bytes": sum(length for _, length in ranges),
            "any_bytes": sum(length for _, length in ranges),
            "unknown_bytes": 0,
        },
    )
    monkeypatch.setattr(
        uvm,
        "batch_advise_ranges_preferred_location",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("already-resident hot ranges were re-advised")
        ),
    )
    monkeypatch.setattr(
        uvm,
        "batch_prefetch_ranges_to_device_with_executor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("already-resident hot ranges were re-prefetched")
        ),
    )

    result = (
        offload_utils.reconcile_cpu_offload_uvm_original_va_sealed_hot_residency()
    )

    assert result["ok"] is True
    assert result["hot_prefetch_skipped_already_resident"] is True
    assert result["hot_prefetch_count"] == 0
    assert result["hot_physical_residency_before"]["gpu_bytes"] == 4096
    assert result["hot_physical_residency_after"]["gpu_bytes"] == 4096


def test_sealed_hot_reconcile_excludes_runtime_remote_ranges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import uvm

    pointer = _FakeTensor(shape=(4, 1), pointer=0x400000)
    binding = SimpleNamespace(
        full_parameter_name="model.layers.0.weight",
        parameter=SimpleNamespace(data=pointer),
        consumer=pointer,
        requested_bytes=4096,
        parameter_bytes=16384,
        expert_ids=(3,),
        consumer_ranges=((0x403000, 4096),),
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "sealed"
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", False)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {binding.full_parameter_name: binding},
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT", (0x400000,)
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY", []
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT",
        {"enabled": True, "full_parameter_shadow": True},
    )
    monkeypatch.setattr(offload_utils.torch.cuda, "synchronize", lambda _=0: None)
    monkeypatch.setattr(
        offload_utils.torch.cuda,
        "mem_get_info",
        lambda _=0: (512 * 1024 * 1024, 32 * 1024 * 1024 * 1024),
    )
    monkeypatch.setattr(
        offload_utils.torch.cuda, "memory_reserved", lambda _=0: 1234
    )
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_alias_expected_shadow_cpu_bytes",
        lambda *_args, **_kwargs: 4096,
    )
    query_count = 0

    def query(ranges, **_kwargs):
        nonlocal query_count
        query_count += 1
        total = sum(length for _, length in ranges)
        return {
            "ok": True,
            "queried_bytes": total,
            "cpu_bytes": 4096,
            # Physical residency is already exact.  A runtime-remote update
            # still requires one mapping refresh before dual-view freeze.
            "gpu_bytes": 8192,
            "any_bytes": total,
            "unknown_bytes": 0,
        }

    monkeypatch.setattr(
        uvm, "driver_batch_query_ranges_residency_bytes", query
    )
    advised: list[tuple[tuple[int, int], ...]] = []
    prefetched: list[tuple[tuple[int, int], ...]] = []
    cutovers: list[tuple[tuple[int, int], ...]] = []

    def advise(ranges, _device):
        normalized = tuple(ranges)
        advised.append(normalized)
        return len(normalized)

    def prefetch(ranges, *_args, **_kwargs):
        normalized = tuple(ranges)
        prefetched.append(normalized)
        return len(normalized), "cpp_batch", 0

    def cutover(ranges, **_kwargs):
        normalized = tuple(ranges)
        cutovers.append(normalized)
        total = sum(length for _, length in normalized)
        return {
            "ok": True,
            "range_count": len(normalized),
            "success_count": len(normalized),
            "failure_count": 0,
            "requested_bytes": total,
            "released_gpu_bytes": total,
        }

    monkeypatch.setattr(
        uvm, "batch_advise_ranges_preferred_location", advise
    )
    monkeypatch.setattr(
        uvm, "batch_prefetch_ranges_to_device_with_executor", prefetch
    )
    monkeypatch.setattr(
        uvm, "driver_cutover_ranges_to_immutable_cpu_shadow", cutover
    )

    result = offload_utils.reconcile_cpu_offload_uvm_original_va_sealed_hot_residency(
        runtime_remote_ranges=((0x401000, 4096),),
    )

    expected_gpu_ranges = ((0x400000, 4096), (0x402000, 4096))
    assert result["ok"] is True
    assert result["hot_expected_gpu_bytes"] == 8192
    assert result["hot_runtime_remote_bytes"] == 4096
    assert result["hot_mapping_refresh_required"] is True
    assert cutovers == [expected_gpu_ranges]
    assert advised == [expected_gpu_ranges]
    assert prefetched == [expected_gpu_ranges]
    assert result["hot_physical_residency_after"]["gpu_bytes"] == 8192


def test_uvm_original_graph_capture_repairs_hot_residency_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.device_allocator import uvm

    pointer = _FakeTensor(shape=(2, 1), pointer=0x400000)
    binding = SimpleNamespace(
        full_parameter_name="model.layers.0.weight",
        parameter=SimpleNamespace(data=pointer),
        consumer=pointer,
        requested_bytes=4096,
        parameter_bytes=8192,
        expert_ids=(1,),
        consumer_ranges=((0x401000, 4096),),
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS", True)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND",
        "uvm_original_va",
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_ORIGINAL_VA_PHASE_MODE", "remote_only"
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "remote")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", True)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED", False)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {binding.full_parameter_name: binding},
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_ORIGINAL_VA_SNAPSHOT", (0x400000,)
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_HISTORY", [])
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_PHASE_TRANSITION_HISTORY", []
    )
    monkeypatch.setattr(offload_utils.torch.cuda, "synchronize", lambda device=0: None)

    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        offload_utils.torch.cuda,
        "empty_cache",
        lambda: events.append(("empty_cache", None)),
    )
    monkeypatch.setattr(
        offload_utils.torch.cuda,
        "mem_get_info",
        lambda device=0: (512 * 1024 * 1024, 32 * 1024 * 1024 * 1024),
    )

    remote = {
        "ok": True,
        "requested_count": 1,
        "success_count": 1,
        "failure_count": 0,
        "requested_bytes": 4096,
        "cpu_resident_bytes": 4096,
        "remote_mapped_bytes": 4096,
        "covered_bytes": 4096,
        "unknown_bytes": 0,
    }

    def prepare_remote(ranges, **_kwargs):
        events.append(("remote", tuple(ranges)))
        return remote

    def advise_hot(ranges, device):
        events.append(("advise", (tuple(ranges), device)))
        return len(ranges)

    def prefetch_hot(ranges, device, *, executor):
        events.append(("prefetch", (tuple(ranges), device, executor)))
        return len(ranges), executor, 0

    def query(ranges, **_kwargs):
        ranges = tuple(ranges)
        total = sum(length for _, length in ranges)
        if ranges == ((0x401000, 4096),):
            return {
                "ok": True,
                "queried_bytes": total,
                "cpu_bytes": total,
                "gpu_bytes": 0,
                "any_bytes": total,
                "unknown_bytes": 0,
            }
        assert ranges == ((0x400000, 4096),)
        return {
            "ok": True,
            "queried_bytes": total,
            "cpu_bytes": 0,
            "gpu_bytes": total,
            "any_bytes": total,
            "unknown_bytes": 0,
        }

    monkeypatch.setattr(
        uvm, "driver_prepare_ranges_remote_read_audited", prepare_remote
    )
    monkeypatch.setattr(
        uvm, "batch_advise_ranges_preferred_location", advise_hot
    )
    monkeypatch.setattr(
        uvm, "batch_prefetch_ranges_to_device_with_executor", prefetch_hot
    )
    monkeypatch.setattr(uvm, "driver_batch_query_ranges_residency_bytes", query)

    result = offload_utils.prepare_cpu_offload_uvm_original_va_remote_graph_capture()
    repaired = result["graph_residency_reconciliation"]
    assert repaired["ok"] is True
    assert repaired["cold_bytes"] == 4096
    assert repaired["hot_bytes"] == 4096
    assert repaired["hot_prefetch_count"] == 1
    assert events[:4] == [
        ("empty_cache", None),
        ("remote", ((0x401000, 4096),)),
        ("advise", (((0x400000, 4096),), 0)),
        ("prefetch", (((0x400000, 4096),), 0, "cpp_batch")),
    ]

    monkeypatch.setenv("VLLM_UVM_ORIGINAL_VA_GRAPH_MIN_FREE_BYTES", "1024")
    monkeypatch.setattr(
        offload_utils.torch.cuda, "mem_get_info", lambda device=0: (512, 4096)
    )
    with pytest.raises(RuntimeError, match="residency reconciliation failed"):
        offload_utils.reconcile_cpu_offload_uvm_original_va_remote_graph_residency()
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "poisoned"
    assert offload_utils._CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE is False


def test_uvm_plan_classes_keep_envelope_vas_and_resize_active_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert_bytes = 4096
    upload = torch.empty((8, expert_bytes), dtype=torch.uint8)
    source = torch.empty((8, expert_bytes), dtype=torch.uint8)
    upload_va = int(upload.data_ptr())
    source_va = int(source.data_ptr())
    arena = SimpleNamespace(
        upload=upload,
        requested_bytes=8 * expert_bytes,
        aligned_bytes=8 * expert_bytes,
        envelope_expert_count=8,
    )
    binding = SimpleNamespace(
        full_parameter_name="model.layers.0.experts.weight",
        arena=arena,
        requested_bytes=8 * expert_bytes,
        expert_ids=tuple(range(8)),
        expert_stride_bytes=expert_bytes,
        consumer_ranges=(),
        source=source,
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS",
        {("model.layers", "experts.weight", 0): arena},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {binding.full_parameter_name: binding},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES",
        8 * expert_bytes,
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_BYTES", 8 * expert_bytes)

    for width in (2, 4, 6, 8, 2):
        expert_ids = tuple(range(width))
        geometry = {
            binding.full_parameter_name: (
                expert_ids,
                expert_bytes,
                width * expert_bytes,
                (width, expert_bytes),
                (expert_bytes, 1),
                tuple(
                    (0x400000 + index * expert_bytes, expert_bytes)
                    for index in expert_ids
                ),
            )
        }
        selected = offload_utils._apply_uvm_original_alias_plan_geometry(
            geometry,
            refill_sources=False,
        )

        assert selected == width * expert_bytes
        assert arena.requested_bytes == width * expert_bytes
        assert arena.aligned_bytes == 8 * expert_bytes
        assert width * expert_bytes == (
            offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES
        )
        assert offload_utils._uvm_original_alias_binding_upload_view(
            binding
        ).shape == (width, expert_bytes)
        assert offload_utils._uvm_original_alias_binding_source_view(
            binding
        ).shape == (width, expert_bytes)
        assert int(arena.upload.data_ptr()) == upload_va
        assert int(binding.source.data_ptr()) == source_va


def test_uvm_plan_class_delta_preserves_slots_and_touches_only_changed_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import uvm

    expert_bytes = 16
    parameter = torch.stack(
        [torch.full((expert_bytes,), 10 + expert, dtype=torch.uint8) for expert in range(8)]
    )
    source = torch.full((8, expert_bytes), 0xEE, dtype=torch.uint8)
    old_ids = (0, 1, 2, 3)
    for slot, expert in enumerate(old_ids):
        source[slot].copy_(parameter[expert])
    full_name = "model.layers.0.experts.weight"
    binding = SimpleNamespace(
        full_parameter_name=full_name,
        parameter=SimpleNamespace(data=parameter),
        consumer=parameter,
        source=source,
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {full_name: binding},
    )
    base = int(parameter.data_ptr())

    def geometry(expert_ids: tuple[int, ...]):
        return {
            full_name: (
                expert_ids,
                expert_bytes,
                len(expert_ids) * expert_bytes,
                (len(expert_ids), expert_bytes),
                (expert_bytes, 1),
                tuple(
                    (base + expert * expert_bytes, expert_bytes)
                    for expert in expert_ids
                ),
            )
        }

    old_geometry = geometry(old_ids)
    target_geometry, assignment = (
        offload_utils._uvm_original_alias_stabilize_plan_slots(
            old_geometry,
            geometry((1, 2, 4, 5, 6, 7)),
        )
    )
    stable_ids = target_geometry[full_name][0]
    assert stable_ids == (4, 1, 2, 5, 6, 7)
    assert assignment["retained_slot_count"] == 2
    assert assignment["changed_slot_count"] == 4

    refill = offload_utils._refill_uvm_original_alias_plan_delta(
        old_geometry,
        target_geometry,
    )
    assert refill["changed_slot_count"] == 4
    assert refill["changed_bytes"] == 4 * expert_bytes
    for slot, expert in enumerate(stable_ids):
        assert torch.equal(source[slot], parameter[expert])

    calls: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    added_ranges = tuple(
        (base + expert * expert_bytes, expert_bytes) for expert in (4, 5, 6, 7)
    )
    removed_ranges = tuple(
        (base + expert * expert_bytes, expert_bytes) for expert in (0, 3)
    )

    def prepare(ranges, **_kwargs):
        calls.append(("prepare", tuple(ranges)))
        return len(ranges), "mock", 0

    def advise(ranges, _device):
        calls.append(("advise", tuple(ranges)))
        return len(ranges)

    def prefetch(ranges, _device, *, executor):
        calls.append(("prefetch", tuple(ranges)))
        return len(ranges), executor, 0

    def query(ranges, **_kwargs):
        normalized = tuple(ranges)
        total = sum(length for _, length in normalized)
        cpu = total if normalized == added_ranges else 0
        gpu = total if normalized == removed_ranges else 0
        return {
            "ok": True,
            "queried_bytes": total,
            "cpu_bytes": cpu,
            "gpu_bytes": gpu,
            "unknown_bytes": 0,
        }

    monkeypatch.setattr(
        uvm,
        "batch_prepare_ranges_for_native_remote_access_with_executor",
        prepare,
    )
    monkeypatch.setattr(uvm, "batch_advise_ranges_preferred_location", advise)
    monkeypatch.setattr(
        uvm, "batch_prefetch_ranges_to_device_with_executor", prefetch
    )
    monkeypatch.setattr(uvm, "driver_batch_query_ranges_residency_bytes", query)
    monkeypatch.setattr(offload_utils.torch.cuda, "synchronize", lambda _device=0: None)

    residency = offload_utils._prepare_uvm_original_alias_plan_residency_delta(
        old_geometry,
        target_geometry,
        device=0,
    )
    assert residency["added_cold_range_count"] == 4
    assert residency["removed_cold_range_count"] == 2
    assert calls == [
        ("prepare", added_ranges),
        ("advise", removed_ranges),
        ("prefetch", removed_ranges),
    ]


def test_uvm_plan_class_shadow_cutover_repromotes_with_native_readmostly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import uvm

    added = ((0x401000, 4096),)
    geometry = {
        "model.layers.0.experts.weight": (
            (0,),
            4096,
            4096,
            (1, 4096),
            (4096, 1),
            ((0x400000, 4096),),
        )
    }
    monkeypatch.setenv(
        "VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW", "1"
    )
    monkeypatch.setenv(
        "VLLM_CPU_OFFLOAD_PLAN_CLASS_REPROMOTE_BEFORE_CUTOVER", "1"
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_PLAN_CLASS_SHADOW_RESULT",
        {"enabled": True, "full_parameter_shadow": True},
    )
    monkeypatch.setattr(offload_utils.torch.cuda, "synchronize", lambda _device=0: None)
    calls: list[tuple[str, tuple[tuple[int, int], ...]]] = []
    cutover_done = False

    def advise(ranges, *, device):
        normalized = tuple(ranges)
        assert device == 0
        calls.append(("readmostly", normalized))
        return len(normalized)

    def prefetch(ranges):
        normalized = tuple(ranges)
        calls.append(("cpu_prefetch", normalized))
        return len(normalized)

    def query(ranges, **_kwargs):
        normalized = tuple(ranges)
        calls.append(("query", normalized))
        total = sum(length for _, length in normalized)
        return {
            "ok": True,
            "queried_bytes": total,
            "cpu_bytes": total,
            "gpu_bytes": 0 if cutover_done else total,
            "any_bytes": total,
            "unknown_bytes": 0,
        }

    def cutover(ranges, **_kwargs):
        nonlocal cutover_done
        normalized = tuple(ranges)
        calls.append(("cutover", normalized))
        cutover_done = True
        total = sum(length for _, length in normalized)
        return {
            "ok": True,
            "range_count": len(normalized),
            "success_count": len(normalized),
            "failure_count": 0,
            "requested_bytes": total,
            "released_gpu_bytes": total,
            "retained_cpu_shadow_bytes": total,
            "remote_mapped_bytes": total,
        }

    monkeypatch.setattr(uvm, "batch_advise_ranges_read_mostly", advise)
    monkeypatch.setattr(uvm, "batch_prefetch_ranges_to_cpu", prefetch)
    monkeypatch.setattr(uvm, "driver_batch_query_ranges_residency_bytes", query)
    monkeypatch.setattr(
        uvm, "driver_cutover_ranges_to_immutable_cpu_shadow", cutover
    )

    result = offload_utils._prepare_uvm_original_alias_plan_residency_delta(
        geometry,
        geometry,
        device=0,
        prepared_ranges=(added, ()),
    )

    assert result["read_duplicate_promotion"]["ok"] is True
    assert result["read_duplicate_promotion"]["operation"] == (
        "native_readmostly_cpu_prefetch"
    )
    assert result["shadow_cutover"]["ok"] is True
    assert calls == [
        ("readmostly", added),
        ("cpu_prefetch", added),
        ("query", added),
        ("cutover", added),
        ("query", added),
    ]


def test_uvm_quiesced_append_batches_oversized_plan_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import uvm_static_alias

    maximum = uvm_static_alias.UVM_MOE_STATIC_ALIAS_MAX_ENTRIES
    entries = tuple(
        (index, index, index, 0, 0x400000 + index * 4096, 4096, 0)
        for index in range(6750)
    )
    calls: list[tuple[int, int]] = []

    def append(batch, *, session_id: int, device: int, flags: int):
        assert session_id == 7
        assert device == 0
        calls.append((len(batch), flags))
        return {"ok": True, "ioctl_calls": 1}

    monkeypatch.setattr(uvm_static_alias, "driver_static_alias_map", append)

    result = offload_utils._uvm_original_alias_quiesced_delta_batched(
        session_id=7,
        entries=entries,
        operation="append",
        device=0,
        defer_dual_refresh=True,
    )

    expected_flags = (
        uvm_static_alias.UVM_MOE_STATIC_ALIAS_MAP_FLAG_QUIESCED_APPEND
        | uvm_static_alias.UVM_MOE_STATIC_ALIAS_MAP_FLAG_DEFER_DUAL_REFRESH
    )
    assert calls == [(maximum, expected_flags), (6750 - maximum, expected_flags)]
    assert result["ok"] is True
    assert result["batch_count"] == 2
    assert result["completed_count"] == 6750
    assert result["dual_view_refresh_deferred"] is True


def test_uvm_quiesced_remove_batches_keep_backing_identities_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import uvm_static_alias

    rw = uvm_static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW
    entries = tuple(
        entry
        for identity in range(3)
        for entry in (
            (0, identity, identity, rw, 0x400000 + identity * 0x2000, 4096, 0),
            (0, identity, identity, 0, 0x500000 + identity * 0x2000, 4096, 0),
        )
    )
    calls: list[tuple[tuple[int, int, int], ...]] = []

    def remove(session_id: int, batch, *, device: int, synchronize: bool):
        assert session_id == 7
        assert device == 0
        assert synchronize is False
        calls.append(tuple(entry[:3] for entry in batch))
        return {"ok": True, "ioctl_calls": 1}

    monkeypatch.setattr(uvm_static_alias, "UVM_MOE_STATIC_ALIAS_MAX_ENTRIES", 4)
    monkeypatch.setattr(
        uvm_static_alias, "driver_static_alias_quiesced_remove", remove
    )

    result = offload_utils._uvm_original_alias_quiesced_delta_batched(
        session_id=7,
        entries=entries,
        operation="remove",
        device=0,
        defer_dual_refresh=False,
    )

    assert [len(batch) for batch in calls] == [4, 2]
    assert all(len(set(batch)) * 2 == len(batch) for batch in calls)
    assert result["ok"] is True
    assert result["batch_count"] == 2
    assert result["completed_count"] == 6


@pytest.mark.parametrize(
    ("lifecycle_state", "remote_map_active", "maximum_delta_entries"),
    [("sealed", False, 1152), ("remote_parked", True, 1152), ("sealed", False, 1)],
)
def test_uvm_remove_detaches_alias_before_gpu_residency(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_state: str,
    remote_map_active: bool,
    maximum_delta_entries: int,
) -> None:
    from vllm.device_allocator import uvm, uvm_static_alias

    expert_bytes = 4096
    full_name = "model.layers.0.experts.weight"
    consumer = _FakeTensor(pointer=0x400000)
    upload = _FakeTensor(pointer=0x500000)
    source = _FakeTensor(device="cpu", pinned=True, pointer=0x600000)
    arena = SimpleNamespace(
        upload=upload,
        requested_bytes=2 * expert_bytes,
        aligned_bytes=2 * expert_bytes,
        envelope_expert_count=2,
        pte_flags=0,
    )
    binding = SimpleNamespace(
        full_parameter_name=full_name,
        parameter=SimpleNamespace(data=consumer),
        consumer=consumer,
        source=source,
        arena=arena,
        requested_bytes=2 * expert_bytes,
        expert_ids=(0, 1),
        expert_stride_bytes=expert_bytes,
        consumer_ranges=((0x400000, expert_bytes), (0x401000, expert_bytes)),
    )
    old_plan = {0: (0, 1)}
    target_plan = {0: (0,)}
    old_geometry = {
        full_name: (
            (0, 1),
            expert_bytes,
            2 * expert_bytes,
            (2, expert_bytes),
            (expert_bytes, 1),
            ((0x400000, expert_bytes), (0x401000, expert_bytes)),
        )
    }
    target_geometry = {
        full_name: (
            (0,),
            expert_bytes,
            expert_bytes,
            (1, expert_bytes),
            (expert_bytes, 1),
            ((0x400000, expert_bytes),),
        )
    }
    rw = uvm_static_alias.UVM_MOE_STATIC_ALIAS_ENTRY_FLAG_RW
    retained = (
        (0, 0, 0, rw, 0x500000, expert_bytes, 0),
        (0, 0, 0, 0, 0x400000, expert_bytes, 0),
    )
    removed = (
        (0, 1, 1, rw, 0x501000, expert_bytes, 0),
        (0, 1, 1, 0, 0x401000, expert_bytes, 0),
    )
    old_entries = retained + removed
    target_entries = retained
    target_preallocate = (
        (0x500000, expert_bytes, 0, 0, 0, 0, uvm.UVM_MOE_SLOT_REMAP_FLAG_NO_EVICT),
    )

    monkeypatch.delenv("VLLM_CPU_OFFLOAD_PLAN_CLASS_PRECOMPILE", raising=False)
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_PLAN_CLASS_QUIESCED_DELTA", "1")
    monkeypatch.setattr(
        uvm_static_alias,
        "UVM_MOE_STATIC_ALIAS_MAX_ENTRIES",
        maximum_delta_entries,
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS",
        {"cold30": old_plan, "cold29": target_plan},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN_CLASS_FILES",
        {"cold30": "cold30.json", "cold29": "cold29.json"},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_ACTIVE_PLAN_CLASS",
        "cold30",
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_LAYER",
        old_plan,
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERT_PLAN_FILE",
        "cold30.json",
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PLAN",
        {full_name: SimpleNamespace(layer_index=0)},
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS", {full_name: binding}
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS",
        {("model.layers", "experts.weight", 0): arena},
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_MAX_BYTES", 1 << 30)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_OWNER_DEVICE_INDEX", 0)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", 7)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", lifecycle_state
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", remote_map_active
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT",
        {"ok": True} if remote_map_active else None,
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES",
        2 * expert_bytes,
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_ENVELOPE_BYTES", 2 * expert_bytes
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_PLAN_GENERATION", 0)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_PLAN_CLASS_INSTRUCTIONS", {})
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_PLAN_CLASS_SWITCH_HISTORY", [])

    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_alias_plan_geometry",
        lambda plan: old_geometry if plan == old_plan else target_geometry,
    )
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_alias_stabilize_plan_slots",
        lambda old, target: (target, {"changed_slot_count": 1}),
    )
    map_entry_calls = 0

    def map_entries(_pte_flags: int):
        nonlocal map_entry_calls
        map_entry_calls += 1
        return old_entries if map_entry_calls == 1 else target_entries

    monkeypatch.setattr(offload_utils, "_uvm_original_alias_map_entries", map_entries)
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_alias_preallocate_entries",
        lambda: target_preallocate,
    )

    def apply_geometry(geometry, *, refill_sources: bool):
        assert geometry is target_geometry
        assert refill_sources is False
        offload_utils._CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES = expert_bytes
        return expert_bytes

    monkeypatch.setattr(
        offload_utils, "_apply_uvm_original_alias_plan_geometry", apply_geometry
    )
    monkeypatch.setattr(
        offload_utils,
        "_refill_uvm_original_alias_plan_delta",
        lambda old, target: {"changed_slot_count": 0, "changed_bytes": 0},
    )
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_alias_plan_residency_delta_ranges",
        lambda old, target: ((), ((0x401000, expert_bytes),)),
    )
    monkeypatch.setattr(
        offload_utils, "_uvm_original_parameter_va_snapshot", lambda: (0x400000,)
    )
    monkeypatch.setattr(offload_utils.torch.cuda, "synchronize", lambda device=0: None)
    monkeypatch.setattr(offload_utils, "_append_cpu_offload_manifest", lambda _item: None)

    events: list[str] = []

    def remove(session_id: int, entries, **_kwargs):
        assert session_id == 7
        assert tuple(entries) == removed
        events.append("remove")
        return {"ok": True, "ioctl_calls": 1, "side_effect_indeterminate": False}

    def residency(
        old, target, *, device: int, prepared_ranges, advise_removed: bool
    ):
        expected_prefix = (
            ["remove"]
            if maximum_delta_entries >= len(removed)
            else ["park", "rebind", "resume"]
        )
        assert events == expected_prefix
        assert advise_removed is False
        events.append("residency")
        return {"ok": True, "removed_cold_range_count": 1}

    monkeypatch.setattr(
        uvm_static_alias, "driver_static_alias_quiesced_remove", remove
    )
    monkeypatch.setattr(
        offload_utils, "_prepare_uvm_original_alias_plan_residency_delta", residency
    )
    monkeypatch.setattr(
        offload_utils,
        "refresh_cpu_offload_uvm_original_va_full_dual_view",
        lambda *, nested_refresh=None: events.append(
            f"refresh:{nested_refresh}"
        )
        or {"ok": True, "nested_refresh": nested_refresh},
    )
    if maximum_delta_entries >= len(removed):
        monkeypatch.setattr(
            uvm_static_alias,
            "driver_static_alias_park",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("PARK must not run")
            ),
        )
        monkeypatch.setattr(
            uvm_static_alias,
            "driver_static_alias_resume",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("RESUME must not run")
            ),
        )
    else:
        monkeypatch.setattr(
            uvm_static_alias,
            "driver_static_alias_quiesced_remove",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("oversized delta must use full REBIND")
            ),
        )
        monkeypatch.setattr(
            uvm_static_alias,
            "driver_static_alias_park",
            lambda *_args, **_kwargs: events.append("park") or {"ok": True},
        )
        monkeypatch.setattr(
            uvm_static_alias,
            "driver_static_alias_rebind",
            lambda *_args, **_kwargs: events.append("rebind")
            or {"ok": True, "side_effect_indeterminate": False},
        )
        monkeypatch.setattr(
            uvm_static_alias,
            "driver_static_alias_resume",
            lambda *_args, **_kwargs: events.append("resume") or {"ok": True},
        )

    result = offload_utils.switch_cpu_offload_uvm_original_va_plan_class(
        "cold29", reason="unit_test", graphs_retained=True
    )

    assert result["ok"] is True
    assert result["quiesced_delta_used"] is (
        maximum_delta_entries >= len(removed)
    )
    assert result["quiesced_delta_remove_count"] == 2
    expected_events = (
        ["remove", "residency", "refresh:False"]
        if maximum_delta_entries >= len(removed)
        else ["park", "rebind", "resume", "residency", "refresh:False"]
    )
    assert events == expected_events
    assert result["post_restore_dual_refresh"] == {
        "ok": True,
        "nested_refresh": False,
    }


def test_uvm_mixed_plan_residency_merges_pre_and_post_detach_phases() -> None:
    before = {
        "mode": "immutable_cpu_shadow_delta",
        "immutable_cpu_shadow_enabled": True,
        "shadow_cutover": {"ok": True},
        "added_cold_range_count": 2,
        "added_cold_bytes": 8192,
        "removed_cold_range_count": 0,
        "removed_cold_bytes": 0,
        "prepare": {"ok": True},
        "advice_requested": False,
        "advised_count": 0,
        "prefetched_count": 0,
        "prefetch_executor": "none",
        "prefetch_driver_result": 0,
        "added_cold_residency": {"gpu_bytes": 0, "cpu_bytes": 8192},
        "removed_cold_residency": None,
    }
    after = {
        "mode": "immutable_cpu_shadow_delta",
        "immutable_cpu_shadow_enabled": True,
        "shadow_cutover": None,
        "added_cold_range_count": 0,
        "added_cold_bytes": 0,
        "removed_cold_range_count": 3,
        "removed_cold_bytes": 12288,
        "prepare": (0, "none", 0),
        "advice_requested": False,
        "advised_count": 0,
        "prefetched_count": 3,
        "prefetch_executor": "cpp_batch",
        "prefetch_driver_result": 0,
        "added_cold_residency": None,
        "removed_cold_residency": {"gpu_bytes": 12288, "cpu_bytes": 12288},
    }

    result = offload_utils._merge_uvm_original_alias_plan_residency_phases(
        before, after
    )

    assert result is not None
    assert result["mode"] == "split_deferred_removed"
    assert result["added_cold_range_count"] == 2
    assert result["removed_cold_range_count"] == 3
    assert result["prefetched_count"] == 3
    assert result["added_cold_residency"] == before["added_cold_residency"]
    assert result["removed_cold_residency"] == after["removed_cold_residency"]


def test_uvm_nested_plan_class_prefill_eliminates_runtime_source_refill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert_bytes = 16
    parameter = torch.stack(
        [
            torch.full((expert_bytes,), 20 + expert, dtype=torch.uint8)
            for expert in range(6)
        ]
    )
    source = torch.full((4, expert_bytes), 0xEE, dtype=torch.uint8)
    source[0].copy_(parameter[0])
    source[1].copy_(parameter[1])
    full_name = "model.layers.0.experts.weight"
    binding = SimpleNamespace(
        full_parameter_name=full_name,
        parameter=SimpleNamespace(data=parameter),
        consumer=parameter,
        source=source,
        source_slot_expert_ids=[0, 1, None, None],
        expert_ids=(0, 1),
        expert_stride_bytes=expert_bytes,
        plan=SimpleNamespace(layer_index=0),
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {full_name: binding},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS",
        {
            "c2": {0: (0, 1)},
            "c4": {0: (0, 1, 2, 3)},
        },
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_INITIAL_PLAN_CLASS",
        "c2",
    )
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_PLAN_CLASS_PREFILL_SOURCES", "1")

    prefill = offload_utils._prefill_uvm_original_alias_plan_class_sources()

    assert prefill["enabled"] is True
    assert prefill["copied_slot_count"] == 2
    assert prefill["copied_bytes"] == 2 * expert_bytes
    assert binding.source_slot_expert_ids == [0, 1, 2, 3]
    assert torch.equal(source[2], parameter[2])
    assert torch.equal(source[3], parameter[3])

    base = int(parameter.data_ptr())

    def geometry(expert_ids: tuple[int, ...]):
        return {
            full_name: (
                expert_ids,
                expert_bytes,
                len(expert_ids) * expert_bytes,
                (len(expert_ids), expert_bytes),
                (expert_bytes, 1),
                tuple(
                    (base + expert * expert_bytes, expert_bytes)
                    for expert in expert_ids
                ),
            )
        }

    old_geometry = geometry((0, 1))
    target_geometry, _ = offload_utils._uvm_original_alias_stabilize_plan_slots(
        old_geometry,
        geometry((0, 1, 2, 3)),
    )
    refill = offload_utils._refill_uvm_original_alias_plan_delta(
        old_geometry,
        target_geometry,
    )
    assert refill["requested_slot_count"] == 2
    assert refill["changed_slot_count"] == 0
    assert refill["changed_bytes"] == 0
    assert refill["reused_prefilled_slot_count"] == 2
    assert refill["reused_prefilled_bytes"] == 2 * expert_bytes


def test_uvm_single_plan_class_certifies_existing_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expert_bytes = 16
    parameter = torch.stack(
        [torch.full((expert_bytes,), expert, dtype=torch.uint8) for expert in range(3)]
    )
    source = parameter[:2].clone()
    binding = SimpleNamespace(
        full_parameter_name="model.layers.0.experts.weight",
        parameter=SimpleNamespace(data=parameter),
        consumer=parameter,
        source=source,
        source_slot_expert_ids=[0, 1],
        expert_ids=(0, 1),
        expert_stride_bytes=expert_bytes,
        plan=SimpleNamespace(layer_index=0),
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {binding.full_parameter_name: binding},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS",
        {"cold70": {0: (0, 1)}},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_INITIAL_PLAN_CLASS",
        "cold70",
    )
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_PLAN_CLASS_PREFILL_SOURCES", "1")

    result = offload_utils._prefill_uvm_original_alias_plan_class_sources()

    assert result["enabled"] is True
    assert result["ordered_classes"] == ("cold70",)
    assert result["copied_slot_count"] == 0
    assert result["reused_slot_count"] == 2


def test_uvm_plan_class_initialization_materializes_future_cpu_shadows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import uvm

    expert_bytes = 64 * 1024
    parameter_base = 0x400000
    full_name = "model.layers.0.experts.weight"
    binding = SimpleNamespace(
        full_parameter_name=full_name,
        consumer=_FakeTensor(pointer=parameter_base),
        source=_FakeTensor(device="cpu", pinned=True, pointer=0x800000),
        expert_ids=(0, 1),
        expert_stride_bytes=expert_bytes,
        plan=SimpleNamespace(layer_index=0),
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {full_name: binding},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EXPERTS_BY_CLASS",
        {
            "cold2": {0: (0, 1)},
            "cold3": {0: (0, 1, 3)},
        },
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_INITIAL_PLAN_CLASS",
        "cold2",
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_PLAN_CLASS_SOURCE_PREFILL_RESULT",
        {"enabled": True},
    )
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW", "1")

    expected_ranges = ((parameter_base + 3 * expert_bytes, expert_bytes),)
    calls: list[tuple[str, tuple[tuple[int, int], ...]]] = []

    def advise(ranges, *, device: int):
        assert device == 0
        calls.append(("read_mostly", tuple(ranges)))
        return len(ranges)

    def prefetch_cpu(ranges):
        calls.append(("prefetch_cpu", tuple(ranges)))
        return len(ranges)

    def query(ranges, **_kwargs):
        normalized = tuple(ranges)
        calls.append(("query", normalized))
        total = sum(length for _, length in normalized)
        return {
            "ok": True,
            "queried_bytes": total,
            "cpu_bytes": total,
            "gpu_bytes": total,
            "any_bytes": total,
            "unknown_bytes": 0,
        }

    monkeypatch.setattr(uvm, "batch_advise_ranges_read_mostly", advise)
    monkeypatch.setattr(uvm, "batch_prefetch_ranges_to_cpu", prefetch_cpu)
    monkeypatch.setattr(uvm, "driver_batch_query_ranges_residency_bytes", query)
    monkeypatch.setattr(offload_utils.torch.cuda, "synchronize", lambda _device=0: None)

    result = offload_utils._prepare_uvm_original_alias_plan_class_immutable_shadows(
        device=0
    )

    assert result["enabled"] is True
    assert result["reason"] == "read_duplicate_shadows_ready"
    assert result["initial_class"] == "cold2"
    assert result["range_count"] == 1
    assert result["requested_bytes"] == expert_bytes
    assert result["readmostly_count"] == 1
    assert result["prefetched_count"] == 1
    assert len(result["geometry_sha256"]) == 64
    assert calls == [
        ("read_mostly", expected_ranges),
        ("prefetch_cpu", expected_ranges),
        ("query", expected_ranges),
    ]


def test_uvm_plan_class_full_parameter_shadow_preserves_managed_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import uvm

    parameter_bytes = 2 * 1024 * 1024
    parameter_base = 0x400000
    full_name = "model.layers.0.experts.weight"
    consumer = _FakeTensor(
        shape=(parameter_bytes,),
        stride=(1,),
        dtype=torch.uint8,
        pointer=parameter_base,
    )
    binding = SimpleNamespace(consumer=consumer)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {full_name: binding},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_INITIAL_PLAN_CLASS",
        "cold2",
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_PLAN_CLASS_SOURCE_PREFILL_RESULT",
        {"enabled": True},
    )
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW", "1")
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW", "1")

    expected_ranges = ((parameter_base, parameter_bytes),)
    calls: list[tuple[str, tuple[tuple[int, int], ...]]] = []

    def advise(ranges, *, device: int):
        assert device == 0
        calls.append(("read_mostly", tuple(ranges)))
        return len(ranges)

    def prefetch_cpu(ranges):
        calls.append(("prefetch_cpu", tuple(ranges)))
        return len(ranges)

    def query(ranges, **_kwargs):
        normalized = tuple(ranges)
        calls.append(("query", normalized))
        total = sum(length for _, length in normalized)
        return {
            "ok": True,
            "queried_bytes": total,
            "cpu_bytes": total,
            "gpu_bytes": total,
            "any_bytes": total,
            "unknown_bytes": 0,
        }

    monkeypatch.setattr(uvm, "batch_advise_ranges_read_mostly", advise)
    monkeypatch.setattr(uvm, "batch_prefetch_ranges_to_cpu", prefetch_cpu)
    monkeypatch.setattr(uvm, "driver_batch_query_ranges_residency_bytes", query)
    monkeypatch.setattr(offload_utils.torch.cuda, "synchronize", lambda _device=0: None)

    result = offload_utils._prepare_uvm_original_alias_plan_class_immutable_shadows(
        device=0
    )

    assert result["enabled"] is True
    assert result["reason"] == "full_parameter_read_duplicate_shadows_ready"
    assert result["full_parameter_shadow"] is True
    assert result["initial_class"] == "cold2"
    assert result["range_count"] == 1
    assert result["requested_bytes"] == parameter_bytes
    assert result["readmostly_count"] == 1
    assert result["prefetched_count"] == 1
    assert len(result["geometry_sha256"]) == 64
    assert calls == [
        ("read_mostly", expected_ranges),
        ("prefetch_cpu", expected_ranges),
        ("query", expected_ranges),
    ]


def test_uvm_fixed_plan_full_parameter_shadow_needs_no_class_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import uvm

    parameter_bytes = 2 * 1024 * 1024
    parameter_base = 0x600000
    consumer = _FakeTensor(
        shape=(parameter_bytes,),
        stride=(1,),
        dtype=torch.uint8,
        pointer=parameter_base,
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {"model.layers.0.experts.weight": SimpleNamespace(consumer=consumer)},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_INITIAL_PLAN_CLASS",
        None,
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_PLAN_CLASS_SOURCE_PREFILL_RESULT",
        {"enabled": False, "reason": "plan_classes_unavailable"},
    )
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_PLAN_CLASS_IMMUTABLE_CPU_SHADOW", "1")
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_DUAL_VIEW_FULL_PARAMETER_SHADOW", "1")

    expected_ranges = ((parameter_base, parameter_bytes),)
    monkeypatch.setattr(
        uvm,
        "batch_advise_ranges_read_mostly",
        lambda ranges, **_kwargs: len(tuple(ranges)),
    )
    monkeypatch.setattr(
        uvm,
        "batch_prefetch_ranges_to_cpu",
        lambda ranges: len(tuple(ranges)),
    )
    monkeypatch.setattr(
        uvm,
        "driver_batch_query_ranges_residency_bytes",
        lambda ranges, **_kwargs: {
            "ok": True,
            "queried_bytes": parameter_bytes,
            "cpu_bytes": parameter_bytes,
            "gpu_bytes": parameter_bytes,
            "any_bytes": parameter_bytes,
            "unknown_bytes": 0,
        },
    )
    monkeypatch.setattr(
        offload_utils.torch.cuda, "synchronize", lambda _device=0: None
    )

    result = offload_utils._prepare_uvm_original_alias_plan_class_immutable_shadows(
        device=0
    )

    assert result["enabled"] is True
    assert result["reason"] == "full_parameter_read_duplicate_shadows_ready"
    assert result["full_parameter_shadow"] is True
    assert result["fixed_plan_full_parameter_shadow"] is True
    assert result["initial_class"] is None
    assert result["range_count"] == 1
    assert result["requested_bytes"] == parameter_bytes
    assert expected_ranges == ((parameter_base, parameter_bytes),)


def test_uvm_plan_class_certificate_activation_restores_class_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    consumer = _FakeTensor(pointer=0x400000)
    upload = _FakeTensor(pointer=0x500000)
    source = _FakeTensor(device="cpu", pinned=True, pointer=0x600000)
    binding = SimpleNamespace(
        parameter=SimpleNamespace(data=consumer),
        consumer=consumer,
        source=source,
    )
    arena = SimpleNamespace(upload=upload)
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {"model.layers.0.weight": binding},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS",
        {("model.layers", "weight", 0): arena},
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_GRAPH_CERTIFICATES_BY_CLASS", {}
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CAPTURED", True)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_ALIAS_GRAPH_CAPTURED", True)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY", "c2")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT", 2)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_GRAPH_REMAINING_COUNT", 1
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE",
        {"class_name": "c2", "parameter_vas": (0x400000,)},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE",
        {"class_name": "c2", "parameter_vas": (0x400000,)},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE",
        {"class_name": "c2", "parameter_vas": (0x400000,)},
    )
    offload_utils.register_cpu_offload_uvm_plan_class_graph_certificates("c2")

    offload_utils._CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY = "c8"
    offload_utils._CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT = 8
    offload_utils._CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE = {
        "class_name": "c8",
        "parameter_vas": (0x400000,),
    }
    offload_utils._CPU_OFFLOAD_UVM_ALIAS_GRAPH_CERTIFICATE = {
        "class_name": "c8",
        "parameter_vas": (0x400000,),
    }
    offload_utils._CPU_OFFLOAD_UVM_DUAL_GRAPH_PARTITION_CERTIFICATE = {
        "class_name": "c8",
        "parameter_vas": (0x400000,),
    }
    offload_utils.register_cpu_offload_uvm_plan_class_graph_certificates("c8")

    result = offload_utils.activate_cpu_offload_uvm_plan_class_graph_certificates(
        "c2"
    )

    assert result == {"ok": True, "reason": "class_certificate_activated"}
    assert offload_utils._CPU_OFFLOAD_UVM_CAPTURE_TOPOLOGY == "c2"
    assert offload_utils._CPU_OFFLOAD_UVM_CAPTURE_SUCCESS_COUNT == 2
    assert offload_utils._CPU_OFFLOAD_UVM_REMOTE_GRAPH_CERTIFICATE == {
        "class_name": "c2",
        "parameter_vas": (0x400000,),
    }


def _install_plan_class_rollback_state(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_FakeTensor, _FakeTensor]:
    consumer = _FakeTensor(pointer=0x400000)
    upload = _FakeTensor(pointer=0x500000)
    source = _FakeTensor(device="cpu", pinned=True, pointer=0x600000)
    arena = SimpleNamespace(upload=upload)
    binding = SimpleNamespace(
        parameter=SimpleNamespace(data=consumer),
        consumer=consumer,
        source=source,
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS",
        {("model.layers", "weight", 0): arena},
    )
    monkeypatch.setattr(
        offload_utils,
        "_CPU_OFFLOAD_UVM_STATIC_ALIAS_BINDINGS",
        {"model.layers.0.weight": binding},
    )
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKING_BYTES", 4096
    )
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID", 7)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE", "sealed")
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_ACTIVE", False)
    monkeypatch.setattr(offload_utils, "_CPU_OFFLOAD_UVM_REMOTE_MAP_RESULT", None)
    monkeypatch.setattr(
        offload_utils, "_CPU_OFFLOAD_UVM_STATIC_ALIAS_CLEANUP_UNRESOLVED", False
    )
    monkeypatch.setattr(
        offload_utils,
        "_apply_uvm_original_alias_plan_geometry",
        lambda geometry, *, refill_sources: 4096,
    )
    monkeypatch.setattr(
        offload_utils,
        "_prepare_uvm_original_alias_plan_residency_delta",
        lambda old, target, *, device: {
            "ok": True,
            "device": device,
            "old": old,
            "target": target,
        },
    )
    monkeypatch.setattr(offload_utils.torch.cuda, "synchronize", lambda device=0: None)
    return upload, source


def test_uvm_plan_class_rollback_rebinds_committed_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import uvm_static_alias

    upload, source = _install_plan_class_rollback_state(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(
        uvm_static_alias,
        "driver_static_alias_park",
        lambda *args, **kwargs: calls.append("park") or {"ok": True},
    )
    monkeypatch.setattr(
        uvm_static_alias,
        "driver_static_alias_rebind",
        lambda *args, **kwargs: calls.append("rebind") or {"ok": True},
    )
    monkeypatch.setattr(
        uvm_static_alias,
        "driver_static_alias_resume",
        lambda *args, **kwargs: calls.append("resume") or {"ok": True},
    )
    old_entries = ((0, 0, 0, 0, 0x400000, 4096, 0),)
    old_geometry = {"old": (1,)}
    target_geometry = {"target": (2,)}

    result = offload_utils._rollback_uvm_original_alias_plan_class_switch(
        old_geometry=old_geometry,
        target_geometry=target_geometry,
        old_consumer_entries=old_entries,
        pre_state="sealed",
        same_capacity=True,
        rebind_committed=True,
        fast_rollback_safe=True,
        old_session_destroyed=False,
        old_backing_bytes=4096,
        parameter_vas=(0x400000,),
        upload_vas=(int(upload.data_ptr()),),
        source_vas=(int(source.data_ptr()),),
        old_remote_map_active=False,
        old_remote_map_result=None,
        device=0,
    )

    assert result["ok"] is True
    assert result["mode"] == "retained_backing_rebind"
    assert calls == ["park", "rebind", "resume"]
    assert result["residency"]["old"] == target_geometry
    assert result["residency"]["target"] == old_geometry
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE == "sealed"


def test_uvm_plan_class_rollback_rematerializes_released_backing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.device_allocator import uvm_static_alias

    upload, source = _install_plan_class_rollback_state(monkeypatch)
    calls: list[tuple[str, int]] = []

    def destroy(session_id: int, **_kwargs: Any) -> dict[str, Any]:
        calls.append(("destroy", session_id))
        return {"ok": True}

    def finalize() -> dict[str, Any]:
        calls.append(("finalize", 0))
        offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID = 11
        offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_STATE = "sealed"
        return {"ok": True, "session_id": 11}

    monkeypatch.setattr(
        uvm_static_alias, "driver_static_alias_destroy", destroy
    )
    monkeypatch.setattr(
        offload_utils,
        "_uvm_original_alias_close_slot_session",
        lambda uvm, *, device: (True, {"ok": True}),
    )
    monkeypatch.setattr(
        offload_utils, "finalize_cpu_offload_graph_static_alias", finalize
    )

    result = offload_utils._rollback_uvm_original_alias_plan_class_switch(
        old_geometry={},
        target_geometry={},
        old_consumer_entries=(),
        pre_state="sealed",
        same_capacity=False,
        rebind_committed=False,
        fast_rollback_safe=True,
        old_session_destroyed=True,
        old_backing_bytes=4096,
        parameter_vas=(0x400000,),
        upload_vas=(int(upload.data_ptr()),),
        source_vas=(int(source.data_ptr()),),
        old_remote_map_active=False,
        old_remote_map_result=None,
        device=0,
    )

    assert result["ok"] is True
    assert result["mode"] == "backing_rematerialized"
    assert calls == [("destroy", 7), ("finalize", 0)]
    assert offload_utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_SESSION_ID == 11
