#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def layer_id_from_name(layer_name: Any, num_layers: int) -> int | None:
    if not isinstance(layer_name, str):
        return None
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    if match is None:
        return None
    layer_id = int(match.group(1))
    return layer_id if 0 <= layer_id < num_layers else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build explicit hot/cold expert tiers from routing traces."
    )
    parser.add_argument("--routing-trace", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--first-moe-layer", type=int, default=0)
    parser.add_argument("--cold-experts-per-layer", type=int, default=5)
    parser.add_argument("--expert-est-bytes", type=int, default=4718592)
    args = parser.parse_args()

    token_counts: dict[tuple[int, int], int] = defaultdict(int)
    trace_records = 0
    fallback_active_records = 0
    for trace_path in args.routing_trace:
        with trace_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                layer_id = layer_id_from_name(record.get("layer_name"), args.num_layers)
                if layer_id is None:
                    continue
                trace_records += 1
                counts = record.get("expert_token_counts")
                if isinstance(counts, dict):
                    for expert_text, count in counts.items():
                        try:
                            expert_id = int(expert_text)
                            count_int = int(count)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= expert_id < args.num_experts and count_int > 0:
                            token_counts[(layer_id, expert_id)] += count_int
                    continue
                active_experts = record.get("active_experts")
                if isinstance(active_experts, list):
                    fallback_active_records += 1
                    for expert in active_experts:
                        try:
                            expert_id = int(expert)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= expert_id < args.num_experts:
                            token_counts[(layer_id, expert_id)] += 1

    cold_count = min(max(args.cold_experts_per_layer, 0), args.num_experts)
    cold_plan: list[dict[str, Any]] = []
    hot_plan: list[dict[str, Any]] = []
    selected_counts: list[int] = []
    selected_by_layer: dict[str, list[int]] = {}
    first_moe_layer = min(max(args.first_moe_layer, 0), args.num_layers)
    for layer_id in range(first_moe_layer, args.num_layers):
        ranked = sorted(
            (
                token_counts[(layer_id, expert_id)],
                expert_id,
            )
            for expert_id in range(args.num_experts)
        )
        cold_ids = {expert_id for _, expert_id in ranked[:cold_count]}
        selected_by_layer[str(layer_id)] = sorted(cold_ids)
        for count, expert_id in ranked:
            record = {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "pilot_token_count": count,
            }
            if expert_id in cold_ids:
                cold_plan.append({**record, "reason": "per_layer_cold_tier"})
                selected_counts.append(count)
            else:
                hot_plan.append({**record, "reason": "hot_tier_complement"})

    observed_token_count = sum(token_counts.values())
    selected_token_count = sum(selected_counts)
    metadata = {
        "policy": "routing_heatmap_v1",
        "selection_policy": "routing_frequency_hot_cold_two_tier_v1",
        "routing_traces": [str(path) for path in args.routing_trace],
        "trace_records": trace_records,
        "fallback_active_records": fallback_active_records,
        "num_layers": args.num_layers,
        "num_experts": args.num_experts,
        "cold_experts_per_layer": cold_count,
        "cold_records": len(cold_plan),
        "hot_records": len(hot_plan),
        "cold_zero_hit_records": sum(count == 0 for count in selected_counts),
        "cold_active_records": sum(count > 0 for count in selected_counts),
        "cold_token_count": selected_token_count,
        "observed_token_count": observed_token_count,
        "cold_token_fraction": (
            selected_token_count / observed_token_count
            if observed_token_count
            else 0.0
        ),
        "expert_est_bytes": args.expert_est_bytes,
        "estimated_cold_bytes": len(cold_plan) * args.expert_est_bytes,
        "selected_by_layer": selected_by_layer,
    }
    plan = {
        "metadata": metadata,
        "offload_plan": cold_plan,
        "hot_plan": hot_plan,
        "warm_plan": [],
        "prefetch_plan": hot_plan,
        "offload_plan_records": len(cold_plan),
        "hot_plan_records": len(hot_plan),
        "warm_plan_records": 0,
        "prefetch_plan_records": len(hot_plan),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
