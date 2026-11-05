#!/usr/bin/env python3
"""Freeze balanced decode-cold Qwen3 expert plans for selected GiB tiers."""

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
TIER_WIDTHS = {"2g": 10, "4g": 19, "6g": 29, "7g": 33, "8g": 38}
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
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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

    # Passes 0..3 are chunked prefill. Passes 4..10 are seven one-token
    # decode forwards for 32 independent calibration requests.
    decode_counts = logical_count[4:11].to(torch.int64).sum(dim=0)
    total_decode_slots = int(decode_counts.sum().item())
    expected_slots = 7 * 32 * NUM_LAYERS * 8
    if total_decode_slots != expected_slots:
        raise SystemExit(
            f"decode calibration slot total differs: {total_decode_slots}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    exact_names = output_dir / "exact_parameter_names.txt"
    exact_names.write_text(
        "".join(
            f"model.layers.{layer}.mlp.experts.{name}\n"
            for layer in range(NUM_LAYERS)
            for name in ("w13_weight", "w2_weight")
        ),
        encoding="utf-8",
    )

    manifest: dict[str, Any] = {
        "schema": "qwen_decode_six_mechanism_inputs_v1",
        "source": str(source),
        "source_sha256": source_sha256,
        "logical_count_shape": list(logical_count.shape),
        "decode_calibration_passes": list(range(4, 11)),
        "decode_calibration_requests": 32,
        "decode_calibration_steps": 7,
        "decode_calibration_tokens": 224,
        "decode_calibration_route_slots": total_decode_slots,
        "exact_parameter_names": str(exact_names),
        "exact_parameter_count": 96,
        "full_expert_main_bytes": 28_991_029_248,
        "tiers": {},
    }

    all_coordinates = {
        (layer, expert)
        for layer in range(NUM_LAYERS)
        for expert in range(NUM_EXPERTS)
    }
    for tier, width in TIER_WIDTHS.items():
        selected_by_layer: dict[str, list[int]] = {}
        cold: set[tuple[int, int]] = set()
        selected_slots = 0
        zero_count = 0
        for layer in range(NUM_LAYERS):
            ranked = sorted(
                (int(decode_counts[layer, expert]), expert)
                for expert in range(NUM_EXPERTS)
            )
            experts = sorted(expert for _, expert in ranked[:width])
            selected_by_layer[str(layer)] = experts
            cold.update((layer, expert) for expert in experts)
            selected_slots += sum(
                int(decode_counts[layer, expert]) for expert in experts
            )
            zero_count += sum(
                int(decode_counts[layer, expert]) == 0 for expert in experts
            )
        hot = all_coordinates - cold
        canonical = json.dumps(
            selected_by_layer, sort_keys=True, separators=(",", ":")
        ).encode()
        main_bytes = len(cold) * EXPERT_MAIN_BYTES
        scale_bytes = len(cold) * EXPERT_SCALE_BYTES
        plan = {
            "schema": "qwen_decode_cold_expert_plan_v1",
            "tier": tier,
            "source": str(source),
            "source_sha256": source_sha256,
            "selection_rule": (
                f"{width} minimum summed calibration-decode route counts per "
                "layer; ties by expert_id; output ids sorted"
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
            "metadata": {
                "target_gib": int(tier.removesuffix("g")),
                "num_layers": NUM_LAYERS,
                "num_experts": NUM_EXPERTS,
                "experts_per_layer": width,
                "offload_plan_records": len(cold),
                "hot_plan_records": len(hot),
                "selected_main_bytes": main_bytes,
                "selected_scale_bytes": scale_bytes,
                "selected_bytes_including_scales": main_bytes + scale_bytes,
                "selected_main_gib": main_bytes / (1 << 30),
                "decode_calibration_passes": list(range(4, 11)),
                "decode_calibration_tokens": 224,
                "decode_calibration_route_slots": total_decode_slots,
                "selected_decode_route_slots": selected_slots,
                "selected_decode_route_fraction": (
                    selected_slots / total_decode_slots
                ),
                "selected_zero_count_coordinates": zero_count,
                "selected_by_layer_canonical_sha256": hashlib.sha256(
                    canonical
                ).hexdigest(),
            },
        }
        plan_path = output_dir / f"cold_decode_plan_{tier}.json"
        write_json(plan_path, plan)
        manifest["tiers"][tier] = {
            "plan": str(plan_path),
            "plan_sha256": sha256_file(plan_path),
            **plan["metadata"],
        }

    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
