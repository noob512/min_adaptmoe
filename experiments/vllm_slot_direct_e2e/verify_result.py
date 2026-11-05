#!/usr/bin/env python3
"""Verify one vLLM slot-direct torch E2E JSON artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    value = json.loads(args.result.read_text(encoding="utf-8"))
    assert value["status"] == "ok", value.get("error")
    assert value["schema"] == "vllm_slot_direct_torch_e2e_v1"
    assert value["geometry"]["va_stable"] is True
    assert value["geometry"]["all_ranges_partial_va_blocks"] is True
    assert value["fd_contract"]["adopted_existing_cuda_uvm_fd"] is True
    assert value["fd_contract"]["new_unrelated_uvm_va_space_fallback"] is False
    assert value["mode10"]["zero_timed_backing_allocations"] is True
    assert value["mode10"]["zero_timed_backing_evictions"] is True
    assert value["mode10"]["failure"] == 0
    assert value["mode10"]["bad_entries"] == 0
    assert value["correctness"]["checksum_equal"] is True
    assert value["correctness"]["gemm_exact_equal"] is True
    assert (
        value["correctness"]["baseline_output_sha256"]
        == value["correctness"]["slot_output_sha256"]
    )
    profile = value["mode10"]["profile_delta"]
    assert profile["slot_direct_vidmem_allocations"] == 0
    assert profile["slot_direct_vidmem_evictions"] == 0
    assert profile["slot_direct_fallbacks"] == 0
    assert profile["slot_direct_errors"] == 0
    for key in ("post_map_ordinary_residency", "post_gemm_ordinary_residency"):
        residency = value["correctness"][key]
        assert residency["gpu_bytes"] == 0
        assert residency["cpu_bytes"] == residency["any_bytes"]
        assert residency["unknown_bytes"] == 0
    print(
        "VERIFIED "
        f"regions={value['config']['regions']} "
        f"mode10_ms={value['mode10']['wall_ms']:.6f} "
        f"gib_s={value['mode10']['effective_gib_s']:.3f} "
        f"slot_vs_baseline={value['performance']['slot_vs_baseline_speed']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
