#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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
        description="Build a per-layer top-K prefill oracle expert plan."
    )
    parser.add_argument("--routing-trace", required=True, type=Path)
    parser.add_argument("--pilot-progress", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pilot-phase", default="prefill_oracle_pilot")
    parser.add_argument("--num-layers", type=int, default=48)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--experts-per-layer", type=int, default=9)
    parser.add_argument("--expert-bytes", type=int, default=4_718_592)
    parser.add_argument("--expected-num-tokens", type=int, default=1217)
    args = parser.parse_args()
    if not 0 < args.experts_per_layer <= args.num_experts:
        parser.error("--experts-per-layer must be in [1, num-experts]")

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
        raise SystemExit(f"invalid pilot phase window: starts={starts} ends={ends}")

    all_records = read_jsonl(args.routing_trace)
    trace_records = [
        item
        for item in all_records
        if starts[0] <= float(item.get("timestamp", -1)) <= ends[0]
    ]
    if not trace_records:
        raise SystemExit("no routing records inside the pilot phase window")
    if len(trace_records) != args.num_layers:
        raise SystemExit(
            f"expected exactly {args.num_layers} pilot records, got "
            f"{len(trace_records)}"
        )

    token_counts: dict[tuple[int, int], int] = defaultdict(int)
    layer_record_counts: dict[int, int] = defaultdict(int)
    for record in trace_records:
        layer_id = layer_id_from_name(record.get("layer_name"), args.num_layers)
        if layer_id is None:
            continue
        layer_record_counts[layer_id] += 1
        num_tokens = int(record.get("num_tokens", -1))
        top_k = int(record.get("top_k", -1))
        if num_tokens != args.expected_num_tokens or top_k <= 0:
            raise SystemExit(
                f"unexpected pilot shape at layer {layer_id}: "
                f"num_tokens={num_tokens} top_k={top_k}"
            )
        counts = record.get("expert_token_counts")
        if not isinstance(counts, dict):
            raise SystemExit(f"missing expert_token_counts at layer {layer_id}")
        record_assignments = 0
        for expert_text, count in counts.items():
            try:
                expert_id = int(expert_text)
                count_int = int(count)
            except (TypeError, ValueError):
                continue
            if 0 <= expert_id < args.num_experts and count_int > 0:
                token_counts[(layer_id, expert_id)] += count_int
                record_assignments += count_int
        expected_assignments = num_tokens * top_k
        if record_assignments != expected_assignments:
            raise SystemExit(
                f"assignment mismatch at layer {layer_id}: "
                f"actual={record_assignments} expected={expected_assignments}"
            )

    missing_layers = sorted(set(range(args.num_layers)) - set(layer_record_counts))
    if missing_layers:
        raise SystemExit(f"routing trace is missing layers: {missing_layers}")
    duplicate_layers = {
        layer_id: count
        for layer_id, count in layer_record_counts.items()
        if count != 1
    }
    if duplicate_layers:
        raise SystemExit(f"pilot must contain one record per layer: {duplicate_layers}")

    selected_plan: list[dict[str, Any]] = []
    coverage_by_layer: dict[str, float] = {}
    assignments_by_layer: dict[str, int] = {}
    selected_assignments_by_layer: dict[str, int] = {}
    for layer_id in range(args.num_layers):
        ranking = sorted(
            (
                (-token_counts[(layer_id, expert_id)], expert_id)
                for expert_id in range(args.num_experts)
            )
        )
        selected_ids = {expert_id for _, expert_id in ranking[: args.experts_per_layer]}
        layer_total = sum(
            token_counts[(layer_id, expert_id)]
            for expert_id in range(args.num_experts)
        )
        selected_total = sum(
            token_counts[(layer_id, expert_id)] for expert_id in selected_ids
        )
        if any(token_counts[(layer_id, expert_id)] <= 0 for expert_id in selected_ids):
            raise SystemExit(f"selected zero-hit expert at layer {layer_id}")
        assignments_by_layer[str(layer_id)] = layer_total
        selected_assignments_by_layer[str(layer_id)] = selected_total
        coverage_by_layer[str(layer_id)] = (
            selected_total / layer_total if layer_total else 0.0
        )
        for _, expert_id in ranking:
            item = {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "pilot_token_count": token_counts[(layer_id, expert_id)],
            }
            if expert_id in selected_ids:
                selected_plan.append({**item, "reason": "prefill_oracle_topk"})

    observed_assignments = sum(assignments_by_layer.values())
    selected_assignments = sum(selected_assignments_by_layer.values())
    expected_records = args.num_layers * args.experts_per_layer
    if len(selected_plan) != expected_records:
        raise SystemExit(
            f"expected {expected_records} selected records, got {len(selected_plan)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered_trace = args.output_dir / "pilot_trace_filtered.jsonl"
    filtered_trace.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in trace_records),
        encoding="utf-8",
    )
    metadata = {
        "policy": "fixed_sharegpt_prefill_oracle_topk_v1",
        "routing_trace": str(args.routing_trace),
        "routing_trace_sha256": hashlib.sha256(
            args.routing_trace.read_bytes()
        ).hexdigest(),
        "pilot_phase": args.pilot_phase,
        "num_layers": args.num_layers,
        "num_experts": args.num_experts,
        "experts_per_layer": args.experts_per_layer,
        "expert_payload_bytes": args.expert_bytes,
        "selected_records": len(selected_plan),
        "selected_payload_bytes": len(selected_plan) * args.expert_bytes,
        "observed_assignments": observed_assignments,
        "selected_assignments": selected_assignments,
        "assignment_coverage": (
            selected_assignments / observed_assignments
            if observed_assignments
            else 0.0
        ),
        "assignment_coverage_min_layer": min(coverage_by_layer.values()),
        "assignment_coverage_max_layer": max(coverage_by_layer.values()),
        "assignment_coverage_by_layer": coverage_by_layer,
        "assignments_by_layer": assignments_by_layer,
        "selected_assignments_by_layer": selected_assignments_by_layer,
    }
    plan = {
        "metadata": metadata,
        "offload_plan": selected_plan,
        "prefetch_plan": selected_plan,
        "hot_plan": [],
        "warm_plan": [],
    }
    empty_plan = {
        "metadata": {"policy": "empty_prefill_oracle_control"},
        "offload_plan": [],
        "prefetch_plan": [],
        "hot_plan": [],
        "warm_plan": [],
    }
    plan_path = args.output_dir / "oracle_plan.json"
    empty_path = args.output_dir / "empty_plan.json"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    empty_path.write_text(
        json.dumps(empty_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        **metadata,
        "oracle_plan": str(plan_path),
        "oracle_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "empty_plan": str(empty_path),
        "empty_plan_sha256": hashlib.sha256(empty_path.read_bytes()).hexdigest(),
    }
    (args.output_dir / "plan_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
