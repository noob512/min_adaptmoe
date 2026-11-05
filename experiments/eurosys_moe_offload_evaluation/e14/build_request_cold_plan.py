#!/usr/bin/env python3
"""Build E14 cold-29 from one measured B32 request-routing window."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


NUM_LAYERS = 48
NUM_EXPERTS = 128
EXPERTS_PER_LAYER = 29
TOP_K = 8
PROMPT_TOKENS_PER_REQUEST = 8
BATCH_SIZE = 32
PREFILL_TOKENS = BATCH_SIZE * PROMPT_TOKENS_PER_REQUEST
WARMUP_DECODE_STEPS = 5
MEASURED_DECODE_STEPS = 16
DECODE_STEPS = WARMUP_DECODE_STEPS + MEASURED_DECODE_STEPS
TOKENS_PER_LAYER = PREFILL_TOKENS + BATCH_SIZE * DECODE_STEPS
EXPERT_MAIN_BYTES = 4_718_592
EXPERT_SCALE_BYTES = 1_152


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: expected a JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise SystemExit(f"{path}:{line_number}: expected a JSON object")
        records.append(value)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--routing-trace", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = load_json(args.result)
    if result.get("status") != "ok":
        raise SystemExit("routing profile result did not finish successfully")
    config = result.get("config") or {}
    expected_config = {
        "batch_sizes": [BATCH_SIZE],
        "prompt_token_limit": PROMPT_TOKENS_PER_REQUEST,
        "conditioning_repeats": 0,
        "measurement_repeats": 1,
        "warmup_decode_steps": WARMUP_DECODE_STEPS,
        "measured_decode_steps": MEASURED_DECODE_STEPS,
        "execution_mode": "eager",
        "enforce_eager": True,
        "requested_cudagraph_mode": "NONE",
        "realized_cudagraph_mode": "NONE",
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"routing profile {key} differs: {config.get(key)!r} != {expected!r}"
            )

    cycles = [
        item
        for item in result.get("cycles") or []
        if item.get("role") == "measurement"
        and int(item.get("batch_size", -1)) == BATCH_SIZE
    ]
    if len(cycles) != 1:
        raise SystemExit(f"expected one measured B32 cycle, got {len(cycles)}")
    cycle = cycles[0]
    start = float(cycle["cycle_started_epoch"])
    end = float(cycle["cycle_ended_epoch"])
    if not start < end:
        raise SystemExit("invalid measured-cycle timestamps")

    all_trace_records = read_jsonl(args.routing_trace)
    records = [
        item
        for item in all_trace_records
        if item.get("event") == "qwen3_moe_expert_trace"
        and start <= float(item.get("timestamp", -1)) <= end
    ]
    expected_records = NUM_LAYERS * (1 + DECODE_STEPS)
    if len(records) != expected_records:
        raise SystemExit(
            f"measured trace record count differs: {len(records)} != {expected_records}"
        )

    combined: dict[tuple[int, int], int] = defaultdict(int)
    prefill: dict[tuple[int, int], int] = defaultdict(int)
    decode: dict[tuple[int, int], int] = defaultdict(int)
    records_per_layer: dict[int, int] = defaultdict(int)
    tokens_per_layer: dict[int, int] = defaultdict(int)
    for record in records:
        layer_id = int(record.get("layer_id", -1))
        num_tokens = int(record.get("num_tokens", -1))
        top_k = int(record.get("top_k", -1))
        num_experts = int(record.get("num_experts", -1))
        if not 0 <= layer_id < NUM_LAYERS:
            raise SystemExit(f"invalid layer id in routing trace: {layer_id}")
        if top_k != TOP_K or num_experts != NUM_EXPERTS:
            raise SystemExit(
                f"layer {layer_id}: top-k/expert geometry differs: "
                f"{top_k}/{num_experts}"
            )
        if num_tokens not in (PREFILL_TOKENS, BATCH_SIZE):
            raise SystemExit(
                f"layer {layer_id}: unexpected forward token count {num_tokens}"
            )
        counts = record.get("expert_token_counts")
        if not isinstance(counts, dict):
            raise SystemExit(f"layer {layer_id}: expert counts are absent")
        assignments = 0
        target = prefill if num_tokens == PREFILL_TOKENS else decode
        for expert_text, count_value in counts.items():
            expert_id = int(expert_text)
            count = int(count_value)
            if not 0 <= expert_id < NUM_EXPERTS or count <= 0:
                raise SystemExit(
                    f"layer {layer_id}: invalid expert/count {expert_id}/{count}"
                )
            combined[(layer_id, expert_id)] += count
            target[(layer_id, expert_id)] += count
            assignments += count
        if assignments != num_tokens * TOP_K:
            raise SystemExit(
                f"layer {layer_id}: assignments {assignments} != "
                f"{num_tokens * TOP_K}"
            )
        records_per_layer[layer_id] += 1
        tokens_per_layer[layer_id] += num_tokens

    if set(records_per_layer) != set(range(NUM_LAYERS)):
        raise SystemExit("measured trace does not cover all 48 layers")
    if any(value != 1 + DECODE_STEPS for value in records_per_layer.values()):
        raise SystemExit("one layer has an unexpected forward-record count")
    if any(value != TOKENS_PER_LAYER for value in tokens_per_layer.values()):
        raise SystemExit("one layer has an unexpected routed-token count")

    selected_by_layer: dict[str, list[int]] = {}
    selected: set[tuple[int, int]] = set()
    for layer_id in range(NUM_LAYERS):
        ranking = sorted(
            (combined[(layer_id, expert_id)], expert_id)
            for expert_id in range(NUM_EXPERTS)
        )
        selected_ids = sorted(
            expert_id for _, expert_id in ranking[:EXPERTS_PER_LAYER]
        )
        selected_by_layer[str(layer_id)] = selected_ids
        selected.update((layer_id, expert_id) for expert_id in selected_ids)

    all_coordinates = {
        (layer_id, expert_id)
        for layer_id in range(NUM_LAYERS)
        for expert_id in range(NUM_EXPERTS)
    }
    hot = all_coordinates - selected
    combined_slots = sum(combined.values())
    selected_combined_slots = sum(combined[item] for item in selected)
    prefill_slots = sum(prefill.values())
    selected_prefill_slots = sum(prefill[item] for item in selected)
    decode_slots = sum(decode.values())
    selected_decode_slots = sum(decode[item] for item in selected)
    zero_count_coordinates = sum(combined[item] == 0 for item in selected)
    canonical = json.dumps(
        selected_by_layer, sort_keys=True, separators=(",", ":")
    ).encode()

    fixture = load_json(args.fixture)
    requests = fixture.get("requests") or []
    if len(requests) < BATCH_SIZE:
        raise SystemExit("fixture has fewer than 32 requests")
    prompt_ids = [
        list(item["prompt_token_ids"][:PROMPT_TOKENS_PER_REQUEST])
        for item in requests[:BATCH_SIZE]
    ]
    prompt_ids_sha256 = hashlib.sha256(
        json.dumps(prompt_ids, separators=(",", ":")).encode()
    ).hexdigest()

    selected_records = len(selected)
    metadata = {
        "selection_scope": "same_e14_evaluation_requests",
        "selection_leakage_notice": (
            "The same frozen request set is used for cold-expert selection and "
            "performance evaluation; this is an in-distribution oracle placement."
        ),
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "experts_per_layer": EXPERTS_PER_LAYER,
        "top_k": TOP_K,
        "profile_batch_size": BATCH_SIZE,
        "prompt_tokens_per_request": PROMPT_TOKENS_PER_REQUEST,
        "prefill_tokens_per_layer": PREFILL_TOKENS,
        "decode_steps": DECODE_STEPS,
        "decode_tokens_per_layer": BATCH_SIZE * DECODE_STEPS,
        "routed_tokens_per_layer": TOKENS_PER_LAYER,
        "profile_trace_records": len(records),
        "combined_route_slots": combined_slots,
        "selected_combined_route_slots": selected_combined_slots,
        "selected_combined_route_fraction": (
            selected_combined_slots / combined_slots
        ),
        "prefill_route_slots": prefill_slots,
        "selected_prefill_route_slots": selected_prefill_slots,
        "selected_prefill_route_fraction": selected_prefill_slots / prefill_slots,
        "decode_route_slots": decode_slots,
        "selected_decode_route_slots": selected_decode_slots,
        "selected_decode_route_fraction": selected_decode_slots / decode_slots,
        "selected_zero_count_coordinates": zero_count_coordinates,
        "offload_plan_records": selected_records,
        "hot_plan_records": len(hot),
        "selected_main_bytes": selected_records * EXPERT_MAIN_BYTES,
        "selected_main_gib": selected_records * EXPERT_MAIN_BYTES / (1 << 30),
        "selected_scale_bytes": selected_records * EXPERT_SCALE_BYTES,
        "selected_bytes_including_scales": selected_records
        * (EXPERT_MAIN_BYTES + EXPERT_SCALE_BYTES),
        "selected_by_layer_canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "fixture": str(args.fixture.resolve()),
        "fixture_sha256": sha256_file(args.fixture),
        "profile_prompt_token_ids_sha256": prompt_ids_sha256,
        "profile_result": str(args.result.resolve()),
        "profile_result_sha256": sha256_file(args.result),
        "routing_trace": str(args.routing_trace.resolve()),
        "routing_trace_sha256": sha256_file(args.routing_trace),
        "measured_cycle_started_epoch": start,
        "measured_cycle_ended_epoch": end,
        "profile_output_token_ids_sha256": cycle.get("output_token_ids_sha256"),
    }
    plan = {
        "schema": "qwen_e14_request_specific_cold_expert_plan_v1",
        "tier": "request_specific_6g",
        "source": str(args.routing_trace.resolve()),
        "source_sha256": sha256_file(args.routing_trace),
        "selection_rule": (
            "29 minimum combined Prefill+Decode route counts per layer in one "
            "measured B32 cycle over the same 32 frozen E14 requests; ties by "
            "expert_id; output ids sorted"
        ),
        "selected_by_layer": selected_by_layer,
        "offload_plan": [
            {"layer_id": layer_id, "expert_id": expert_id}
            for layer_id, expert_id in sorted(selected)
        ],
        "hot_plan": [
            {"layer_id": layer_id, "expert_id": expert_id}
            for layer_id, expert_id in sorted(hot)
        ],
        "offload_plan_records": selected_records,
        "hot_plan_records": len(hot),
        "metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
