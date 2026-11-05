#!/usr/bin/env python3
"""Build one uniform-per-layer DeepSeek-V2-Lite cold-expert plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PLAN_TOOLS = REPO_ROOT / "experiments" / "hybrid_offload_2_4_6_8_compare"
sys.path.insert(0, str(PLAN_TOOLS))

from prepare_cold_plans import (  # noqa: E402
    EXPERT_BYTES,
    EXPERT_COUNT,
    LAYER_IDS,
    load_counts,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experts-per-layer", type=int, default=32)
    parser.add_argument("--serving-start-step", type=int, default=3)
    args = parser.parse_args()

    if not 1 <= args.experts_per_layer < EXPERT_COUNT:
        raise ValueError("experts-per-layer must be in [1, 63]")
    counts, coverage = load_counts(
        args.trace,
        serving_start_step=args.serving_start_step,
        prefill_step_count=1,
    )
    ranked = {
        layer_id: sorted(
            range(EXPERT_COUNT),
            key=lambda expert_id: (counts[layer_id][expert_id], expert_id),
        )
        for layer_id in LAYER_IDS
    }
    selected = {
        str(layer_id): sorted(ranked[layer_id][: args.experts_per_layer])
        for layer_id in LAYER_IDS
    }
    selected_count = len(LAYER_IDS) * args.experts_per_layer
    selected_bytes = selected_count * EXPERT_BYTES
    selected_slots = sum(
        counts[layer_id][expert_id]
        for layer_id in LAYER_IDS
        for expert_id in selected[str(layer_id)]
    )
    payload = {
        "schema": "vllm_uvm_cold_expert_plan_v1",
        "model": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "tier": f"uniform_{args.experts_per_layer}_per_layer",
        "selection_rule": (
            "minimum frozen calibration prefill+decode route slots independently "
            "within every layer; ties by expert_id"
        ),
        "selected_by_layer": selected,
        "experts_per_layer": args.experts_per_layer,
        "selected_expert_count": selected_count,
        "selected_main_bytes": selected_bytes,
        "selected_main_gib": selected_bytes / 2**30,
        "calibration_selected_route_slots": selected_slots,
        "calibration_total_route_slots": sum(
            sum(layer_counts.values()) for layer_counts in counts.values()
        ),
        "calibration_trace": str(args.trace.resolve()),
        "calibration_trace_sha256": sha256_file(args.trace),
        "calibration_coverage": coverage,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
