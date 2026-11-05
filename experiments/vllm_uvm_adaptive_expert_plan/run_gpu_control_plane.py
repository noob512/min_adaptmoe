# SPDX-License-Identifier: Apache-2.0
"""GPU microexperiment for adaptive expert heat collection and planning."""

from __future__ import annotations

import json
import statistics
import time
from types import SimpleNamespace

import torch

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.v1.engine.core import EngineCore


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda")
    num_experts = 128
    assignments = 8192
    iterations = 500
    topk_ids = torch.randint(
        0, num_experts, (assignments, 8), device=device, dtype=torch.int64
    )
    layer = SimpleNamespace(
        uvm_dynamic_expert_swap_enable=True,
        global_num_experts=num_experts,
        local_num_experts=num_experts,
        layer_name="model.layers.0.mlp.experts",
        _uvm_dynamic_expert_heat_counts=None,
        _uvm_dynamic_expert_heat_buffers=[],
        _uvm_dynamic_expert_heat_active_index=0,
        _uvm_dynamic_expert_heat_epoch=0,
        _uvm_remote_read_global_expert_ids=set(range(32)),
        _uvm_layer_id=lambda: 0,
    )

    for _ in range(20):
        FusedMoE._record_uvm_dynamic_expert_heat(layer, topk_ids)
    FusedMoE.get_uvm_dynamic_expert_heat(layer, reset=True)
    torch.cuda.synchronize()

    samples_us: list[float] = []
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        FusedMoE._record_uvm_dynamic_expert_heat(layer, topk_ids)
        end.record()
        end.synchronize()
        samples_us.append(float(begin.elapsed_time(end)) * 1000.0)

    first = FusedMoE.get_uvm_dynamic_expert_heat(layer, reset=True)
    FusedMoE._record_uvm_dynamic_expert_heat(layer, topk_ids)
    second = FusedMoE.get_uvm_dynamic_expert_heat(layer, reset=True)

    core = object.__new__(EngineCore)
    core._uvm_dynamic_expert_heat_ewma = {}
    first_smoothed = core._smooth_uvm_dynamic_expert_heat(
        [{"ok": True, "layers": {"layer.0": first}}], alpha=0.25
    )
    second_smoothed = core._smooth_uvm_dynamic_expert_heat(
        [{"ok": True, "layers": {"layer.0": second}}], alpha=0.25
    )
    selected_class = EngineCore._select_uvm_expert_plan_class(
        (2.0, 4.0, 6.0, 8.0), budget_gib=5.3
    )
    expected_first = assignments * 8 * iterations
    result = {
        "schema": "vllm_uvm_adaptive_expert_control_plane_v1",
        "timestamp_ns": time.time_ns(),
        "gpu": torch.cuda.get_device_name(),
        "num_experts": num_experts,
        "assignments_per_iteration": assignments * 8,
        "iterations": iterations,
        "heat_update_us_median": statistics.median(samples_us),
        "heat_update_us_p95": sorted(samples_us)[int(0.95 * len(samples_us))],
        "first_epoch": first["heat_epoch"],
        "second_epoch": second["heat_epoch"],
        "double_buffered": first["double_buffered"] and second["double_buffered"],
        "first_observed_assignments": sum(first["expert_token_counts"]),
        "first_expected_assignments": expected_first,
        "second_observed_assignments": sum(second["expert_token_counts"]),
        "second_expected_assignments": assignments * 8,
        "ewma_first_sum": sum(
            first_smoothed[0]["layers"]["layer.0"]["expert_token_counts"]
        ),
        "ewma_second_sum": sum(
            second_smoothed[0]["layers"]["layer.0"]["expert_token_counts"]
        ),
        "selected_class_for_5_3_gib_budget": selected_class,
        "checks": {
            "first_count_exact": sum(first["expert_token_counts"])
            == expected_first,
            "second_count_exact": sum(second["expert_token_counts"])
            == assignments * 8,
            "epochs_advance": second["heat_epoch"] == first["heat_epoch"] + 1,
            "class_rounds_down_safely": selected_class == 4.0,
        },
    }
    result["ok"] = all(result["checks"].values()) and result["double_buffered"]
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
