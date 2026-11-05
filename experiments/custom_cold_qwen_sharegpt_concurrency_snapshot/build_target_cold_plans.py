#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build balanced, nested cold-expert plans for target GiB budgets."
    )
    parser.add_argument("--routing-trace", required=True, type=Path)
    parser.add_argument("--pilot-progress", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--targets-gib", default="0,2,6,10,14,18")
    parser.add_argument("--pilot-phase", default="calibration_pilot")
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--expert-bytes", type=int, default=4_718_592)
    args = parser.parse_args()

    progress = read_jsonl(args.pilot_progress)
    starts = [
        float(item["timestamp"])
        for item in progress
        if item.get("action") == "phase_begin"
        and item.get("phase") == args.pilot_phase
    ]
    ends = [
        float(item["timestamp"])
        for item in progress
        if item.get("action") == "phase_end"
        and item.get("phase") == args.pilot_phase
    ]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise SystemExit(
            f"invalid pilot phase window: starts={starts!r}, ends={ends!r}"
        )
    window_start, window_end = starts[0], ends[0]

    all_trace_records = read_jsonl(args.routing_trace)
    trace_records = [
        item
        for item in all_trace_records
        if window_start <= float(item.get("timestamp", -1)) <= window_end
    ]
    if not trace_records:
        raise SystemExit("no routing records inside the pilot phase window")

    token_counts: dict[tuple[int, int], int] = defaultdict(int)
    layer_record_counts: dict[int, int] = defaultdict(int)
    for record in trace_records:
        layer_id = layer_id_from_name(record.get("layer_name"), args.num_layers)
        if layer_id is None:
            continue
        layer_record_counts[layer_id] += 1
        counts = record.get("expert_token_counts")
        if isinstance(counts, dict):
            items = counts.items()
        elif isinstance(counts, list):
            items = enumerate(counts)
        else:
            items = []
        for expert_text, count in items:
            try:
                expert_id = int(expert_text)
                count_int = int(count)
            except (TypeError, ValueError):
                continue
            if 0 <= expert_id < args.num_experts and count_int > 0:
                token_counts[(layer_id, expert_id)] += count_int

    missing_layers = sorted(set(range(args.num_layers)) - set(layer_record_counts))
    if missing_layers:
        raise SystemExit(f"routing trace is missing layers: {missing_layers}")

    rankings: dict[int, list[tuple[int, int]]] = {}
    for layer_id in range(args.num_layers):
        rankings[layer_id] = sorted(
            (token_counts[(layer_id, expert_id)], expert_id)
            for expert_id in range(args.num_experts)
        )

    targets = sorted({int(value) for value in args.targets_gib.split(",") if value})
    if not targets or targets[0] != 0 or any(value < 0 for value in targets):
        raise SystemExit(f"targets must be non-negative and include zero: {targets}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered_trace_path = args.output_dir / "calibration_trace_filtered.jsonl"
    filtered_trace_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in trace_records),
        encoding="utf-8",
    )

    summary_plans = []
    previous_keys: set[tuple[int, int]] = set()
    for target_gib in targets:
        target_bytes = target_gib * (1 << 30)
        target_experts = target_bytes // args.expert_bytes
        if target_experts > args.num_layers * args.num_experts:
            raise SystemExit(f"target {target_gib} GiB exceeds all expert weights")

        base, remainder = divmod(target_experts, args.num_layers)
        selected_by_layer: dict[int, int] = {
            layer_id: base for layer_id in range(args.num_layers)
        }
        if remainder:
            next_candidates = sorted(
                (
                    rankings[layer_id][base][0],
                    layer_id,
                    rankings[layer_id][base][1],
                )
                for layer_id in range(args.num_layers)
            )
            for _, layer_id, _ in next_candidates[:remainder]:
                selected_by_layer[layer_id] += 1

        cold_plan: list[dict[str, Any]] = []
        hot_plan: list[dict[str, Any]] = []
        cold_keys: set[tuple[int, int]] = set()
        for layer_id in range(args.num_layers):
            cold_ids = {
                expert_id
                for _, expert_id in rankings[layer_id][
                    : selected_by_layer[layer_id]
                ]
            }
            for count, expert_id in rankings[layer_id]:
                record = {
                    "layer_id": layer_id,
                    "expert_id": expert_id,
                    "pilot_token_count": count,
                }
                if expert_id in cold_ids:
                    cold_plan.append({**record, "reason": "balanced_cold_frequency"})
                    cold_keys.add((layer_id, expert_id))
                else:
                    hot_plan.append({**record, "reason": "hot_complement"})

        if len(cold_plan) != target_experts:
            raise SystemExit(
                f"target {target_gib}: expected {target_experts} experts, "
                f"selected {len(cold_plan)}"
            )
        if not previous_keys <= cold_keys:
            raise SystemExit(f"target {target_gib}: cold plans are not nested")
        previous_keys = cold_keys

        actual_bytes = len(cold_plan) * args.expert_bytes
        cold_token_count = sum(item["pilot_token_count"] for item in cold_plan)
        observed_token_count = sum(token_counts.values())
        per_layer_counts = list(selected_by_layer.values())
        metadata = {
            "policy": "heldout_sharegpt_balanced_cold_frequency_v1",
            "selection_policy": "per_layer_floor_plus_globally_cold_remainder",
            "routing_trace": str(args.routing_trace),
            "routing_trace_sha256": hashlib.sha256(
                args.routing_trace.read_bytes()
            ).hexdigest(),
            "filtered_trace": str(filtered_trace_path),
            "pilot_phase": args.pilot_phase,
            "pilot_window_start": window_start,
            "pilot_window_end": window_end,
            "all_trace_records": len(all_trace_records),
            "filtered_trace_records": len(trace_records),
            "layer_record_counts": {
                str(key): value for key, value in sorted(layer_record_counts.items())
            },
            "num_layers": args.num_layers,
            "num_experts": args.num_experts,
            "expert_bytes": args.expert_bytes,
            "target_gib": target_gib,
            "target_bytes": target_bytes,
            "target_expert_records_floor": target_experts,
            "actual_offload_bytes": actual_bytes,
            "actual_offload_gib": actual_bytes / (1 << 30),
            "offload_plan_records": len(cold_plan),
            "hot_plan_records": len(hot_plan),
            "min_cold_experts_per_layer": min(per_layer_counts),
            "max_cold_experts_per_layer": max(per_layer_counts),
            "cold_zero_hit_records": sum(
                item["pilot_token_count"] == 0 for item in cold_plan
            ),
            "cold_token_count": cold_token_count,
            "observed_token_count": observed_token_count,
            "cold_token_fraction": (
                cold_token_count / observed_token_count
                if observed_token_count
                else 0.0
            ),
            "nested_with_previous_target": True,
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
        plan_path = args.output_dir / f"plan_g{target_gib}.json"
        plan_path.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        metadata["plan_path"] = str(plan_path)
        metadata["plan_sha256"] = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        summary_plans.append(metadata)

    summary = {
        "routing_trace": str(args.routing_trace),
        "pilot_progress": str(args.pilot_progress),
        "plans": summary_plans,
        "all_plans_nested": True,
    }
    (args.output_dir / "plan_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
