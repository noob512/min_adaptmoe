#!/usr/bin/env python3
"""External E13 case-runner adapter that disables controller hysteresis."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


HERE = Path(__file__).resolve().parent
TARGET = (
    HERE.parent
    / "eurosys_moe_offload_evaluation"
    / "e13"
    / "e13_kv_adaptive_runner.py"
)

os.environ["VLLM_UVM_KV_AUTO_SCALE_LOW_PRESSURE_STREAK"] = "1"
os.environ["VLLM_UVM_KV_AUTO_SCALE_MIN_RESIDENCY_STEPS"] = "0"
os.environ["E13_EXTERNAL_NO_HYSTERESIS_ADAPTER"] = "1"
runpy.run_path(str(TARGET), run_name="__main__")
