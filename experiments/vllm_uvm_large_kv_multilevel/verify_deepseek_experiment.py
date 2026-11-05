#!/usr/bin/env python3
"""Verify the DeepSeek-V2-Lite 2/4/8/10 GiB large-KV qualification."""

from __future__ import annotations

import importlib.util
from pathlib import Path


BASE = Path(__file__).with_name("verify_experiment.py")
SPEC = importlib.util.spec_from_file_location("large_kv_base_verifier", BASE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load base verifier: {BASE}")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)

verifier.BLOCK_BYTES = 497_664
verifier.MAX_BYTES = 10_737_598_464
verifier.TARGETS = {
    "2g": 4_314,
    "4g": 8_629,
    "8g": 17_260,
    "10g": 21_575,
}
verifier.FULL_PARAMETER_BYTES = 28_789_702_656
verifier.SELECTED_PARAMETER_BYTES = 14_394_851_328
verifier.HOT_PARAMETER_BYTES = (
    verifier.FULL_PARAMETER_BYTES - verifier.SELECTED_PARAMETER_BYTES
)
verifier.TOTAL_MODEL_BYTES = 31_412_968_448
verifier.RESIDENT_MODEL_BYTES = (
    verifier.TOTAL_MODEL_BYTES - verifier.SELECTED_PARAMETER_BYTES
)


if __name__ == "__main__":
    raise SystemExit(verifier.main())
