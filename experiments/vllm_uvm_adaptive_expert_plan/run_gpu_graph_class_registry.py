# SPDX-License-Identifier: Apache-2.0
"""Exercise two retained CUDA Graph classes over one stable GPU slot."""

from __future__ import annotations

import json
import statistics
import time

import torch

from vllm.compilation.cuda_graph import CUDAGraphEntry, CUDAGraphWrapper
from vllm.forward_context import BatchDescriptor


def capture(slot: torch.Tensor, output: torch.Tensor, scale: float) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        torch.mul(slot, scale, out=output)
    return graph


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    wrapper = object.__new__(CUDAGraphWrapper)
    wrapper.concrete_cudagraph_entries = {}
    wrapper._cudagraph_class_entries = {
        "default": wrapper.concrete_cudagraph_entries
    }
    wrapper._active_cudagraph_class = "default"

    slot = torch.zeros(4096, device="cuda")
    slot_va = int(slot.data_ptr())
    outputs: dict[str, torch.Tensor] = {}
    graph_ids: dict[str, int] = {}
    for class_name, scale in (("c2", 2.0), ("c4", 4.0)):
        wrapper.begin_cudagraph_class_capture(class_name)
        output = torch.empty_like(slot)
        graph = capture(slot, output, scale)
        descriptor = BatchDescriptor(4096)
        wrapper.concrete_cudagraph_entries[descriptor] = CUDAGraphEntry(
            descriptor,
            cudagraph=graph,
            output=output,
            input_addresses=[slot_va],
        )
        outputs[class_name] = output
        graph_ids[class_name] = id(graph)

    switch_samples_us: list[float] = []
    checks: list[bool] = []
    for iteration in range(1000):
        class_name = "c2" if iteration % 2 == 0 else "c4"
        value = float(iteration + 1)
        slot.fill_(value)
        started_ns = time.perf_counter_ns()
        activated = wrapper.activate_cudagraph_class(class_name)
        switch_samples_us.append((time.perf_counter_ns() - started_ns) / 1000.0)
        entry = next(iter(wrapper.concrete_cudagraph_entries.values()))
        assert entry.cudagraph is not None
        entry.cudagraph.replay()
        torch.cuda.synchronize()
        scale = 2.0 if class_name == "c2" else 4.0
        checks.append(bool(torch.all(outputs[class_name] == value * scale).item()))
        checks.append(bool(activated.get("ok")))

    diagnostics = wrapper.cudagraph_class_diagnostics()
    result = {
        "schema": "vllm_uvm_gpu_graph_class_registry_v1",
        "gpu": torch.cuda.get_device_name(),
        "slot_va": slot_va,
        "slot_va_stable": int(slot.data_ptr()) == slot_va,
        "class_count": diagnostics["class_count"],
        "classes": diagnostics["classes"],
        "graph_ids_stable": graph_ids == {
            name: id(next(iter(wrapper._cudagraph_class_entries[name].values())).cudagraph)
            for name in graph_ids
        },
        "switches": 1000,
        "switch_us_median": statistics.median(switch_samples_us),
        "switch_us_p95": sorted(switch_samples_us)[950],
        "replay_checks": len(checks),
        "ok": all(checks),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
