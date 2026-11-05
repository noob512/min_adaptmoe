#!/usr/bin/env python3
"""Freeze nested per-layer Qwen3 cold-expert states for the large-KV run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


NUM_LAYERS = 48
NUM_EXPERTS = 128
TIER_WIDTHS = {
    "2g": 70,
    "4g": 80,
    "8g": 99,
    "10g": 108,
}
BASE_TIER = "2g"
EXPERT_MAIN_BYTES = 4_718_592
EXPERT_SCALE_BYTES = 1_152
EXPECTED_SOURCE_SHA256 = (
    "912732708a46291b73e70c93291758747f285b364437ce77f58d53501d072f81"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    import torch

    source = args.source.resolve()
    source_sha256 = sha256_file(source)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise SystemExit(f"calibration source hash differs: {source_sha256}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    logical_count = payload.get("logical_count")
    if not isinstance(logical_count, torch.Tensor):
        raise SystemExit("calibration source lacks logical_count")
    if tuple(logical_count.shape) != (11, NUM_LAYERS, NUM_EXPERTS):
        raise SystemExit(f"unexpected logical_count shape: {logical_count.shape}")

    decode_counts = logical_count[4:11].to(torch.int64).sum(dim=0)
    total_decode_slots = int(decode_counts.sum().item())
    expected_slots = 7 * 32 * NUM_LAYERS * 8
    if total_decode_slots != expected_slots:
        raise SystemExit(
            f"decode calibration slot total differs: {total_decode_slots}"
        )

    ranked_by_layer: dict[int, list[int]] = {}
    for layer in range(NUM_LAYERS):
        ranked = sorted(
            (int(decode_counts[layer, expert]), expert)
            for expert in range(NUM_EXPERTS)
        )
        ranked_by_layer[layer] = [expert for _, expert in ranked]

    all_coordinates = {
        (layer, expert)
        for layer in range(NUM_LAYERS)
        for expert in range(NUM_EXPERTS)
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tier_states: dict[str, dict[str, Any]] = {}
    tier_sets: dict[str, set[tuple[int, int]]] = {}
    for tier, width in TIER_WIDTHS.items():
        selected_by_layer = {
            str(layer): sorted(ranked_by_layer[layer][:width])
            for layer in range(NUM_LAYERS)
        }
        selected = {
            (layer, expert)
            for layer in range(NUM_LAYERS)
            for expert in selected_by_layer[str(layer)]
        }
        tier_sets[tier] = selected
        selected_slots = sum(
            int(decode_counts[layer, expert]) for layer, expert in selected
        )
        canonical = json.dumps(
            selected_by_layer,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        main_bytes = len(selected) * EXPERT_MAIN_BYTES
        tier_states[tier] = {
            "experts_per_layer": width,
            "incremental_waves": width - TIER_WIDTHS[BASE_TIER],
            "selected_main_bytes": main_bytes,
            "selected_main_gib": main_bytes / (1 << 30),
            "selected_decode_route_slots": selected_slots,
            "selected_decode_route_fraction": selected_slots / total_decode_slots,
            "selected_by_layer": selected_by_layer,
            "selected_by_layer_canonical_sha256": hashlib.sha256(
                canonical
            ).hexdigest(),
        }

    base_selected = tier_sets[BASE_TIER]
    base_hot = all_coordinates - base_selected
    base_state = tier_states[BASE_TIER]
    base_metadata = {
        "target_gib": 2,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "experts_per_layer": TIER_WIDTHS[BASE_TIER],
        "offload_plan_records": len(base_selected),
        "hot_plan_records": len(base_hot),
        "selected_main_bytes": base_state["selected_main_bytes"],
        "selected_scale_bytes": len(base_selected) * EXPERT_SCALE_BYTES,
        "selected_bytes_including_scales": (
            int(base_state["selected_main_bytes"])
            + len(base_selected) * EXPERT_SCALE_BYTES
        ),
        "selected_main_gib": base_state["selected_main_gib"],
        "decode_calibration_passes": list(range(4, 11)),
        "decode_calibration_tokens": 224,
        "decode_calibration_route_slots": total_decode_slots,
        "selected_decode_route_slots": base_state["selected_decode_route_slots"],
        "selected_decode_route_fraction": base_state[
            "selected_decode_route_fraction"
        ],
        "selected_by_layer_canonical_sha256": base_state[
            "selected_by_layer_canonical_sha256"
        ],
    }
    base_plan = {
        "schema": "qwen_decode_cold_expert_plan_v1",
        "tier": "large_kv_base_2g",
        "source": str(source),
        "source_sha256": source_sha256,
        "selection_rule": (
            f"{TIER_WIDTHS[BASE_TIER]} minimum summed calibration-decode route "
            "counts per layer; ties by expert_id; output ids sorted"
        ),
        "selected_by_layer": base_state["selected_by_layer"],
        "offload_plan": [
            {"layer_id": layer, "expert_id": expert}
            for layer, expert in sorted(base_selected)
        ],
        "hot_plan": [
            {"layer_id": layer, "expert_id": expert}
            for layer, expert in sorted(base_hot)
        ],
        "offload_plan_records": len(base_selected),
        "hot_plan_records": len(base_hot),
        "metadata": base_metadata,
    }
    base_path = output_dir / "cold_decode_plan_base70.json"
    write_json(base_path, base_plan)

    max_tier = max(TIER_WIDTHS, key=TIER_WIDTHS.__getitem__)
    incremental_selected = tier_sets[max_tier] - base_selected
    incremental_hot = all_coordinates - incremental_selected
    incremental_plan = {
        "schema": "qwen_decode_incremental_cold_expert_plan_v1",
        "tier": "large_kv_incremental_38",
        "source": str(source),
        "source_sha256": source_sha256,
        "selection_rule": (
            "experts ranked 71 through 108 by summed calibration-decode route "
            "count in every layer"
        ),
        "selected_by_layer": {
            str(layer): sorted(
                expert
                for selected_layer, expert in incremental_selected
                if selected_layer == layer
            )
            for layer in range(NUM_LAYERS)
        },
        "offload_plan": [
            {"layer_id": layer, "expert_id": expert}
            for layer, expert in sorted(incremental_selected)
        ],
        "hot_plan": [
            {"layer_id": layer, "expert_id": expert}
            for layer, expert in sorted(incremental_hot)
        ],
        "offload_plan_records": len(incremental_selected),
        "hot_plan_records": len(incremental_hot),
        "metadata": {
            "num_layers": NUM_LAYERS,
            "num_experts": NUM_EXPERTS,
            "experts_per_layer": TIER_WIDTHS[max_tier] - TIER_WIDTHS[BASE_TIER],
            "offload_plan_records": len(incremental_selected),
            "selected_main_bytes": len(incremental_selected) * EXPERT_MAIN_BYTES,
        },
    }
    incremental_path = output_dir / "incremental_wave_plan_38.json"
    write_json(incremental_path, incremental_plan)

    waves = []
    for wave_index in range(TIER_WIDTHS[max_tier] - TIER_WIDTHS[BASE_TIER]):
        layer_experts = {
            f"model.layers.{layer}.mlp.experts": ranked_by_layer[layer][
                TIER_WIDTHS[BASE_TIER] + wave_index
            ]
            for layer in range(NUM_LAYERS)
        }
        waves.append(
            {
                "wave": wave_index + 1,
                "layer_experts": layer_experts,
            }
        )
    wave_path = output_dir / "incremental_expert_waves.json"
    wave_payload = {
        "schema": "qwen_large_kv_incremental_expert_waves_v1",
        "source_sha256": source_sha256,
        "base_experts_per_layer": TIER_WIDTHS[BASE_TIER],
        "maximum_experts_per_layer": TIER_WIDTHS[max_tier],
        "tier_states": tier_states,
        "waves": waves,
    }
    write_json(wave_path, wave_payload)

    manifest = {
        "schema": "qwen_large_kv_multilevel_inputs_v1",
        "source": str(source),
        "source_sha256": source_sha256,
        "base_plan": str(base_path),
        "base_plan_sha256": sha256_file(base_path),
        "incremental_plan": str(incremental_path),
        "incremental_plan_sha256": sha256_file(incremental_path),
        "wave_plan": str(wave_path),
        "wave_plan_sha256": sha256_file(wave_path),
        "tiers": tier_states,
    }
    manifest_path = output_dir / "large_kv_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
