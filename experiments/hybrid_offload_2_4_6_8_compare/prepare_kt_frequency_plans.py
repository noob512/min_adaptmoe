#!/usr/bin/env python3
"""Convert frozen cold-expert plans into exact KT frequency placements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


TIERS = ("2g", "4g", "6g", "8g")
NUM_LAYERS = 27
MOE_LAYERS = tuple(range(1, 27))
NUM_EXPERTS = 64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "schema": "deepseek_v2_lite_kt_frequency_placements_v1",
        "selection": "exact complement of frozen per-layer cold expert plan",
        "logical_count_shape": [1, NUM_LAYERS, NUM_EXPERTS],
        "tiers": {},
    }
    tiers: dict[str, object] = manifest["tiers"]  # type: ignore[assignment]

    for tier in TIERS:
        plan_path = args.input_dir / f"cold_plan_{tier}.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        raw = plan.get("selected_by_layer")
        if not isinstance(raw, dict) or set(map(int, raw)) != set(MOE_LAYERS):
            raise ValueError(f"{plan_path}: invalid layer coverage")

        scores = torch.zeros((1, NUM_LAYERS, NUM_EXPERTS), dtype=torch.int64)
        cold_by_layer: dict[int, tuple[int, ...]] = {}
        for layer in MOE_LAYERS:
            values = raw[str(layer)]
            if not isinstance(values, list) or not values:
                raise ValueError(f"{plan_path}: invalid layer {layer}")
            cold = tuple(sorted(int(value) for value in values))
            if len(cold) != len(set(cold)) or cold[0] < 0 or cold[-1] >= NUM_EXPERTS:
                raise ValueError(f"{plan_path}: invalid experts at layer {layer}")
            cold_by_layer[layer] = cold
            scores[0, layer, :] = 1
            scores[0, layer, list(cold)] = 0

        cold_counts = {len(values) for values in cold_by_layer.values()}
        if len(cold_counts) != 1:
            raise ValueError(f"{plan_path}: cold count is not uniform")
        cold_per_layer = cold_counts.pop()
        gpu_per_layer = NUM_EXPERTS - cold_per_layer
        total_gpu = gpu_per_layer * len(MOE_LAYERS)

        flat = scores.sum(dim=0).flatten()
        selected = torch.zeros_like(flat, dtype=torch.bool)
        selected[torch.topk(flat, k=total_gpu).indices] = True
        mask = selected.view(NUM_LAYERS, NUM_EXPERTS)
        for layer in MOE_LAYERS:
            actual_cold = tuple(torch.where(~mask[layer])[0].tolist())
            if actual_cold != cold_by_layer[layer]:
                raise RuntimeError(f"top-k placement differs at layer {layer}")

        output = args.output_dir / f"kt_frequency_{tier}.pt"
        torch.save({"logical_count": scores}, output)
        tiers[tier] = {
            "source_plan": str(plan_path.resolve()),
            "source_plan_sha256": sha256(plan_path),
            "frequency_path": str(output.resolve()),
            "frequency_sha256": sha256(output),
            "cold_experts_per_layer": cold_per_layer,
            "gpu_experts_per_layer": gpu_per_layer,
            "total_cpu_experts": cold_per_layer * len(MOE_LAYERS),
            "total_gpu_experts": total_gpu,
            "selected_main_bytes": int(plan["selected_main_bytes"]),
            "mask_uint8_sha256": hashlib.sha256(
                mask.to(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }

    output_manifest = args.output_dir / "kt_frequency_manifest.json"
    output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
