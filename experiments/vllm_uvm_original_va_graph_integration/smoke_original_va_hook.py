#!/usr/bin/env python3
"""Small end-to-end smoke for the vLLM original-VA offload hook."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--layer-mib", type=int, default=8)
    args = parser.parse_args()
    require(args.replays > 0 and args.layer_mib > 0, "invalid smoke size")

    names = [f"model.layers.{index}.weight" for index in range(4)]
    names_path = args.output.with_suffix(args.output.suffix + ".names.txt")
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.jsonl")
    names_path.parent.mkdir(parents=True, exist_ok=True)
    names_path.write_text("\n".join(names) + "\n", encoding="utf-8")

    os.environ.update(
        {
            "VLLM_CPU_OFFLOAD_BACKEND": "copy",
            "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE": str(names_path),
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS": "1",
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND": "uvm_original_va",
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_REGION_BYTES": "524288",
            "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE": "auto",
            "VLLM_UVM_EXPERT_WEIGHTS_MANAGED": "1",
            "VLLM_CPU_OFFLOAD_COPY_SYNC": "0",
            "VLLM_CPU_OFFLOAD_COPY_TRACE": "1",
            "VLLM_CPU_OFFLOAD_MANIFEST_FILE": str(manifest_path),
            "VLLM_CPU_OFFLOAD_MANIFEST_RESET": "1",
        }
    )

    import torch
    from torch import nn

    from vllm.device_allocator import uvm
    from vllm.device_allocator.uvm_static_alias import (
        get_static_alias_diagnostics,
    )
    from vllm.model_executor.models import utils

    class EchoWeight(nn.Module):
        def __init__(self, weight: torch.Tensor) -> None:
            super().__init__()
            self.weight = nn.Parameter(weight, requires_grad=False)

        def forward(self) -> torch.Tensor:
            return self.weight

    torch.cuda.set_device(0)
    layer_bytes = args.layer_mib * 1024 * 1024
    utils.set_cpu_offload_max_bytes(layer_bytes * 4 + 4096)
    modules: list[nn.Module] = []
    outputs: list[torch.Tensor] = []
    graph: torch.cuda.CUDAGraph | None = None
    capture_stream: torch.cuda.Stream | None = None
    result: dict[str, object] = {
        "schema": "vllm_uvm_original_va_hook_smoke_v1",
        "status": "running",
        "layer_bytes": layer_bytes,
        "replays": args.replays,
    }

    try:
        for index, full_name in enumerate(names):
            with uvm.uvm_local_memory_pool(enable_prefetch=False):
                weight = torch.empty(layer_bytes, dtype=torch.uint8, device="cuda")
            module = EchoWeight(weight)
            module = utils.maybe_offload_to_cpu(
                module,
                module_prefix=full_name.rsplit(".", 1)[0],
            )
            module.weight.fill_(17 + index * 41)
            modules.append(module)
        torch.cuda.synchronize()

        original_vas = [int(module.weight.data_ptr()) for module in modules]
        finalized = utils.finalize_cpu_offload_graph_static_alias()
        require(bool(finalized.get("ok")), f"finalize failed: {finalized!r}")
        active = utils.get_cpu_offload_graph_static_alias_diagnostics()
        require(active.get("backend") == "uvm_original_va", "wrong backend")
        require(active.get("sealed") is True, "topology was not sealed")
        require(int(active.get("arena_count", 0)) == 2, "expected two banks")
        require(
            int(active.get("backing_bytes", 0)) == layer_bytes * 2,
            "unexpected two-bank backing",
        )
        require(
            sum(int(item.get("source_ready_count", 0)) for item in active["arenas"])
            == 4,
            "not every pinned source is ready",
        )

        outputs = [
            torch.empty(layer_bytes, dtype=torch.uint8, device="cuda")
            for _ in modules
        ]

        def pipeline() -> None:
            for output, module in zip(outputs, modules, strict=True):
                output.copy_(module())

        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            pipeline()
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            pipeline()

        replay_ms: list[float] = []
        for _ in range(args.replays):
            for arena in utils._CPU_OFFLOAD_UVM_STATIC_ALIAS_ARENAS.values():
                arena.upload.zero_()
            torch.cuda.synchronize()
            started = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            replay_ms.append((time.perf_counter() - started) * 1000.0)
            for index, output in enumerate(outputs):
                require(
                    bool(torch.all(output == 17 + index * 41).item()),
                    f"layer {index} mismatch after Graph replay",
                )

        require(
            original_vas == [int(module.weight.data_ptr()) for module in modules],
            "one or more original parameter VAs changed",
        )
        graph.reset()
        graph = None
        capture_stream.synchronize()
        destroyed = utils.destroy_cpu_offload_graph_static_alias(
            graphs_destroyed=True,
            synchronize=True,
        )
        require(bool(destroyed.get("ok")), f"destroy failed: {destroyed!r}")
        remote = utils.get_cpu_offload_graph_static_alias_diagnostics()
        require(remote.get("lifecycle_state") == "remote", "not in remote state")
        require(int(remote.get("session_id", -1)) == 0, "session survived cmd99")
        require(
            int(get_static_alias_diagnostics().get("session_count", -1)) == 0,
            "userspace cmd97 session survived cmd99",
        )
        for index, module in enumerate(modules):
            require(int(module.weight.data_ptr()) == original_vas[index], "VA changed")
            restored = module.weight.detach().cpu()
            require(
                bool(torch.all(restored == 17 + index * 41).item()),
                f"layer {index} CPU backing was not restored",
            )

        result.update(
            {
                "status": "ok",
                "original_vas_stable": True,
                "original_vas": original_vas,
                "finalize": finalized,
                "active_diagnostics": active,
                "replay_ms": replay_ms,
                "destroy": destroyed,
                "remote_diagnostics": remote,
            }
        )
    except Exception as exc:
        result.update({"status": "exception", "error": repr(exc)})
        raise
    finally:
        if graph is not None:
            try:
                torch.cuda.synchronize()
                graph.reset()
                graph = None
            except Exception as exc:
                result["fallback_graph_reset_error"] = repr(exc)
        try:
            diagnostics = utils.get_cpu_offload_graph_static_alias_diagnostics()
            if int(diagnostics.get("session_id", 0)) > 0 and graph is None:
                result["fallback_destroy"] = (
                    utils.destroy_cpu_offload_graph_static_alias(
                        graphs_destroyed=True,
                        synchronize=True,
                    )
                )
        except Exception as exc:
            result["fallback_destroy_error"] = repr(exc)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
