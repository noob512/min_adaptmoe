#!/usr/bin/env python3
"""Run the shared phase benchmark without imposing an alias/remote contract.

This is a diagnostic runner used to isolate FULL prefill CUDA Graph behavior
after the original-VA weights have already been switched to audited CPU-backed
remote READ PTEs.  The outer module-cycle harness remains responsible for
process isolation and restoring the stock UVM module.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_RUNNER = (
    HERE.parent
    / "vllm_p3_static_alias_graph_integration"
    / "benchmark_full_graph_steps.py"
)
DUAL_BANK_BYTES = 1_207_959_552
LOGICAL_BYTES = 2_415_919_104


def main() -> int:
    spec = importlib.util.spec_from_file_location("_remote_full_graph_base", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return int(module.run())


if __name__ == "__main__":
    raise SystemExit(main())
