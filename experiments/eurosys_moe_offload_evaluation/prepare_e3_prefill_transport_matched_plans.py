#!/usr/bin/env python3
"""Generate frozen expert-slice plans matched to native copy byte totals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


NUM_LAYERS = 48
NUM_EXPERTS = 128
EXPERT_MAIN_BYTES = 4_718_592
EXPERT_SCALE_BYTES = 1_152
EXPECTED_SOURCE_SHA256 = (
    "912732708a46291b73e70c93291758747f285b364437ce77f58d53501d072f81"
)
TARGETS = {
    "low": {"width": 11, "native_moe_layers": 4, "native_attention_layers": 4},
    "mid": {"width": 38, "native_moe_layers": 14, "native_attention_layers": 8},
    "high": {"width": 65, "native_moe_layers": 24, "native_attention_layers": 12},
}
MOE_LAYER_BYTES = 603_979_776
ATTENTION_LAYER_BYTES = 18_874_368


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    logical_count = torch.load(
        source, map_location="cpu", weights_only=False
    ).get("logical_count")
    if not isinstance(logical_count, torch.Tensor):
        raise SystemExit("calibration source lacks logical_count")
    if tuple(logical_count.shape) != (11, NUM_LAYERS, NUM_EXPERTS):
        raise SystemExit(f"unexpected logical_count shape: {logical_count.shape}")
    decode_counts = logical_count[4:11].to(torch.int64).sum(dim=0)
    total_slots = int(decode_counts.sum().item())

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    universe = {
        (layer, expert)
        for layer in range(NUM_LAYERS)
        for expert in range(NUM_EXPERTS)
    }
    manifest: dict[str, Any] = {
        "schema": "eurosys_e3_prefill_transport_matched_plans_v1",
        "source": str(source),
        "source_sha256": source_sha256,
        "targets": {},
    }
    for tier, target in TARGETS.items():
        width = int(target["width"])
        slice_count = width * NUM_LAYERS
        ranked_by_layer = {
            layer: sorted(
                (int(decode_counts[layer, expert]), expert)
                for expert in range(NUM_EXPERTS)
            )
            for layer in range(NUM_LAYERS)
        }
        cold = {
            (layer, expert)
            for layer in range(NUM_LAYERS)
            for _, expert in ranked_by_layer[layer][:width]
        }
        if len(cold) != slice_count:
            raise SystemExit(f"{tier} slice count differs: {len(cold)}")

        hot = universe - cold
        selected_by_layer = {
            str(layer): sorted(
                expert for selected_layer, expert in cold
                if selected_layer == layer
            )
            for layer in range(NUM_LAYERS)
        }
        selected_slots = sum(
            int(decode_counts[layer, expert]) for layer, expert in cold
        )
        main_bytes = slice_count * EXPERT_MAIN_BYTES
        scale_bytes = slice_count * EXPERT_SCALE_BYTES
        native_names = [
            name
            for layer in range(int(target["native_moe_layers"]))
            for name in (
                f"model.layers.{layer}.mlp.experts.w13_weight",
                f"model.layers.{layer}.mlp.experts.w2_weight",
            )
        ] + [
            name
            for layer in range(int(target["native_attention_layers"]))
            for name in (
                f"model.layers.{layer}.self_attn.qkv_proj.weight",
                f"model.layers.{layer}.self_attn.o_proj.weight",
            )
        ]
        native_bytes = (
            int(target["native_moe_layers"]) * MOE_LAYER_BYTES
            + int(target["native_attention_layers"]) * ATTENTION_LAYER_BYTES
        )
        if native_bytes != main_bytes:
            raise SystemExit(
                f"{tier} native bytes {native_bytes} != slice bytes {main_bytes}"
            )
        canonical = json.dumps(
            selected_by_layer, sort_keys=True, separators=(",", ":")
        ).encode()
        widths = [len(value) for value in selected_by_layer.values()]
        metadata = {
            "slice_count": slice_count,
            "experts_per_layer": width,
            "minimum_experts_per_layer": min(widths),
            "maximum_experts_per_layer": max(widths),
            "selected_main_bytes": main_bytes,
            "selected_scale_bytes": scale_bytes,
            "selected_bytes_including_scales": main_bytes + scale_bytes,
            "native_copy_bytes": native_bytes,
            "native_parameter_count": len(native_names),
            "absolute_byte_difference": abs(main_bytes - native_bytes),
            "relative_byte_difference": abs(main_bytes - native_bytes) / native_bytes,
            "selected_decode_route_slots": selected_slots,
            "selected_decode_route_fraction": selected_slots / total_slots,
            "selected_by_layer_canonical_sha256": hashlib.sha256(
                canonical
            ).hexdigest(),
        }
        plan = {
            "schema": "qwen_decode_cold_expert_plan_v1",
            "tier": f"e3_prefill_transport_{tier}",
            "source": str(source),
            "source_sha256": source_sha256,
            "selection_rule": (
                f"{width} minimum summed calibration-decode route counts per "
                "layer; ties by expert_id"
            ),
            "selected_by_layer": selected_by_layer,
            "offload_plan": [
                {"layer_id": layer, "expert_id": expert}
                for layer, expert in sorted(cold)
            ],
            "hot_plan": [
                {"layer_id": layer, "expert_id": expert}
                for layer, expert in sorted(hot)
            ],
            "offload_plan_records": len(cold),
            "hot_plan_records": len(hot),
            "metadata": metadata,
        }
        path = output_dir / f"{tier}.json"
        write_json(path, plan)
        native_names_path = output_dir / f"native_names_{tier}.txt"
        native_names_path.write_text("\n".join(native_names) + "\n", encoding="utf-8")
        manifest["targets"][tier] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "native_names_path": str(native_names_path),
            "native_names_sha256": sha256_file(native_names_path),
            **metadata,
        }

    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
