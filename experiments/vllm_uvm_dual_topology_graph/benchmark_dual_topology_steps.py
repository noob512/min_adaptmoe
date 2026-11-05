#!/usr/bin/env python3
"""Run the real-vLLM original-VA dual-topology CUDA Graph contract.

This is a narrow wrapper around the phase-step benchmark.  It fixes arm D:
remote uniform-decode Graph capture first, sealed alias non-uniform prefill
Graph capture second, one serving prefill replay, selective alias-Graph reset,
then audited remote decode Graph replay in the same process.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = (
    HERE.parent
    / "vllm_uvm_prefill_alias_decode_remote_graph"
    / "benchmark_phase_switch_steps.py"
)


def load_base():
    spec = importlib.util.spec_from_file_location("_dual_topology_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark base: {BASE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if os.environ.get("VLLM_ATTENTION_BACKEND") != "TRITON_ATTN":
        raise SystemExit(
            "dual-topology runner requires VLLM_ATTENTION_BACKEND=TRITON_ATTN"
        )
    if os.environ.get("VLLM_CPU_OFFLOAD_UVM_ORIGINAL_VA_PARK_RESUME") != "1":
        raise SystemExit("dual-topology runner requires reversible PARK/RESUME")
    if "--arm" in sys.argv:
        index = sys.argv.index("--arm")
        if index + 1 >= len(sys.argv) or sys.argv[index + 1] != "D":
            raise SystemExit("dual-topology runner only accepts --arm D")
    else:
        sys.argv.extend(("--arm", "D"))
    if "--cudagraph-capture-sizes" not in sys.argv:
        sys.argv.extend(("--cudagraph-capture-sizes", "1,4,314,1217"))
    if (
        "--park-resume-roundtrip" not in sys.argv
        and "--dynamic-plan-sequence" not in sys.argv
        and "--dynamic-kv-sequence" not in sys.argv
    ):
        sys.argv.append("--park-resume-roundtrip")
    return int(load_base().main())


if __name__ == "__main__":
    raise SystemExit(main())
