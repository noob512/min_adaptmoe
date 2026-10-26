from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.model_executor.models import utils as offload_utils
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _runner(*, eager: bool = True, mode: str = "NONE") -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.device = 0
    runner.model_config = SimpleNamespace(enforce_eager=eager)
    runner.compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(name=mode)
    )
    runner.model = object()
    return runner


def test_eager_plan_class_switches_without_graph_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    events: list[str] = []
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_PLAN_CLASS_EAGER_ENABLE", "1")
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_plan_class_registry",
        lambda: {
            "enabled": True,
            "active_class": "cold70",
            "classes": {"cold70": {}, "cold80": {}},
        },
    )
    monkeypatch.setattr(
        offload_utils,
        "activate_cpu_offload_uvm_original_va_prefill_topology",
        lambda: events.append("resume") or {"ok": True},
    )

    def switch(class_name: str, **kwargs):
        events.append(f"switch:{class_name}")
        assert kwargs["graphs_retained"] is True
        assert kwargs["defer_dual_refresh"] is True
        return {"ok": True, "old_class": "cold70", "new_class": class_name}

    monkeypatch.setattr(
        offload_utils,
        "switch_cpu_offload_uvm_original_va_plan_class",
        switch,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.torch.cuda.synchronize",
        lambda device=0: None,
    )

    result = runner.activate_uvm_expert_graph_class_at_idle_boundary(
        "cold80",
        scheduler_idle=True,
        resume_prefill_before_switch=True,
        defer_dual_refresh=True,
    )

    assert result["ok"] is True
    assert result["reason"] == "eager_plan_class_activated"
    assert result["execution_mode"] == "eager"
    assert events == ["resume", "switch:cold80"]


def test_eager_plan_class_reparks_decode_topology_after_sealed_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    runner._uvm_original_va_phase_decode_cutover_done = True
    events: list[str] = []
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_PLAN_CLASS_EAGER_ENABLE", "1")
    monkeypatch.setattr(
        offload_utils,
        "cpu_offload_uvm_plan_class_registry",
        lambda: {
            "enabled": True,
            "active_class": "cold70",
            "classes": {"cold70": {}, "cold80": {}},
        },
    )
    monkeypatch.setattr(
        offload_utils,
        "activate_cpu_offload_uvm_original_va_prefill_topology",
        lambda: events.append("resume") or {"ok": True},
    )
    monkeypatch.setattr(
        offload_utils,
        "get_cpu_offload_graph_static_alias_diagnostics",
        lambda: {
            "lifecycle_state": "remote_parked",
            "remote_map_active": True,
            "park_resume_enabled": True,
            "remote_graph_captured": False,
        },
    )
    monkeypatch.setattr(
        offload_utils,
        "transition_cpu_offload_uvm_original_va_to_remote",
        lambda **kwargs: events.append("repark") or {"ok": True},
    )

    def switch(class_name: str, **kwargs):
        events.append(f"switch:{class_name}")
        assert kwargs["defer_dual_refresh"] is False
        return {"ok": True, "old_class": "cold70", "new_class": class_name}

    monkeypatch.setattr(
        offload_utils,
        "switch_cpu_offload_uvm_original_va_plan_class",
        switch,
    )
    monkeypatch.setattr(
        "vllm.v1.worker.gpu_model_runner.torch.cuda.synchronize",
        lambda device=0: None,
    )

    result = runner.activate_uvm_expert_graph_class_at_idle_boundary(
        "cold80",
        scheduler_idle=True,
        resume_prefill_before_switch=True,
    )

    assert result["ok"] is True
    assert result["prefill_topology_resume_before_plan_switch"] == {"ok": True}
    assert result["preserved_remote_decode_topology"] is True
    assert result["dual_view_refresh_deferred"] is False
    assert result["remote_decode_repark"]["ok"] is True
    assert events == ["resume", "switch:cold80", "repark"]


def test_eager_plan_class_gate_rejects_graph_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(eager=False, mode="FULL_DECODE_ONLY")
    monkeypatch.setenv("VLLM_CPU_OFFLOAD_PLAN_CLASS_EAGER_ENABLE", "1")

    result = runner.activate_uvm_expert_graph_class_at_idle_boundary(
        "cold80", scheduler_idle=True
    )

    assert result["ok"] is False
    assert result["reason"] == "eager_plan_class_requires_eager_none_mode"
