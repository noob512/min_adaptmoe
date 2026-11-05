#!/usr/bin/env python3
"""Freeze the formal E3 Qwen cold-expert plan from a fresh calibration trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


NUM_LAYERS = 48
NUM_EXPERTS = 128
TOP_K = 8
EXPERT_MAIN_BYTES = 4_718_592
EXPERT_SCALE_BYTES = 1_152


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calibration-fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-batch", type=int, default=32)
    parser.add_argument("--expected-decode-tokens", type=int, default=224)
    parser.add_argument("--experts-per-layer", type=int, default=10)
    args = parser.parse_args()

    import torch

    source = args.source.resolve()
    fixture = args.calibration_fixture.resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    logical_count = payload.get("logical_count")
    if not isinstance(logical_count, torch.Tensor):
        raise SystemExit("calibration source lacks logical_count")
    if logical_count.ndim != 3 or tuple(logical_count.shape[1:]) != (
        NUM_LAYERS,
        NUM_EXPERTS,
    ):
        raise SystemExit(f"unexpected logical_count shape: {tuple(logical_count.shape)}")
    if logical_count.dtype == torch.bool or logical_count.min().item() < 0:
        raise SystemExit("logical_count is not a non-negative count tensor")

    slots_per_token = NUM_LAYERS * TOP_K
    expected_decode_slots = args.calibration_batch * slots_per_token
    pass_slots = logical_count.to(torch.int64).sum(dim=(1, 2))
    decode_passes = [
        index
        for index, slots in enumerate(pass_slots.tolist())
        if 0 < int(slots) <= expected_decode_slots
        and int(slots) % slots_per_token == 0
    ]
    decode_tokens = sum(int(pass_slots[index]) for index in decode_passes) // slots_per_token
    if len(decode_passes) < 2 or decode_tokens != args.expected_decode_tokens:
        raise SystemExit(
            "could not identify the expected strict decode forwards; "
            f"identified_decode_tokens={decode_tokens}, "
            f"pass route slots={pass_slots.tolist()}"
        )
    decode_counts = logical_count[decode_passes].to(torch.int64).sum(dim=0)
    total_decode_slots = int(decode_counts.sum().item())
    if total_decode_slots != args.expected_decode_tokens * slots_per_token:
        raise SystemExit("decode route-slot count differs")

    width = args.experts_per_layer
    if not 1 <= width < NUM_EXPERTS:
        raise SystemExit("experts-per-layer must be in [1, 127]")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    names = output_dir / "exact_parameter_names.txt"
    names.write_text(
        "".join(
            f"model.layers.{layer}.mlp.experts.{name}\n"
            for layer in range(NUM_LAYERS)
            for name in ("w13_weight", "w2_weight")
        ),
        encoding="utf-8",
    )

    selected_by_layer: dict[str, list[int]] = {}
    cold: set[tuple[int, int]] = set()
    selected_slots = 0
    for layer in range(NUM_LAYERS):
        ranked = sorted(
            (int(decode_counts[layer, expert]), expert)
            for expert in range(NUM_EXPERTS)
        )
        experts = sorted(expert for _, expert in ranked[:width])
        selected_by_layer[str(layer)] = experts
        cold.update((layer, expert) for expert in experts)
        selected_slots += sum(int(decode_counts[layer, expert]) for expert in experts)
    all_coordinates = {
        (layer, expert)
        for layer in range(NUM_LAYERS)
        for expert in range(NUM_EXPERTS)
    }
    hot = all_coordinates - cold
    canonical = json.dumps(
        selected_by_layer, sort_keys=True, separators=(",", ":")
    ).encode()
    main_bytes = len(cold) * EXPERT_MAIN_BYTES
    scale_bytes = len(cold) * EXPERT_SCALE_BYTES
    plan = {
        "schema": "eurosys_qwen_e3_cold_expert_plan_v2",
        "protocol_id": "eurosys-pa-v2-20260824",
        "track": "A_exact_coordinate",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "calibration_fixture": str(fixture),
        "calibration_fixture_sha256": sha256_file(fixture),
        "selection_rule": (
            f"{width} minimum fresh calibration-decode route counts per layer; "
            "ties by expert_id; output ids sorted"
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
            "num_layers": NUM_LAYERS,
            "num_experts": NUM_EXPERTS,
            "experts_per_layer": width,
            "offload_plan_records": len(cold),
            "hot_plan_records": len(hot),
            "selected_main_bytes": main_bytes,
            "selected_scale_bytes": scale_bytes,
            "selected_bytes_including_scales": main_bytes + scale_bytes,
            "selected_main_gib": main_bytes / (1 << 30),
            "calibration_batch": args.calibration_batch,
            "logical_count_shape": list(logical_count.shape),
            "pass_route_slots": [int(value) for value in pass_slots.tolist()],
            "decode_calibration_passes": decode_passes,
            "decode_calibration_tokens": decode_tokens,
            "decode_calibration_route_slots": total_decode_slots,
            "selected_decode_route_slots": selected_slots,
            "selected_decode_route_fraction": selected_slots / total_decode_slots,
            "selected_zero_count_coordinates": sum(
                int(decode_counts[layer, expert]) == 0 for layer, expert in cold
            ),
            "selected_by_layer_canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        },
    }
    plan_path = output_dir / "cold_decode_plan_2g.json"
    write_json(plan_path, plan)
    manifest = {
        "schema": "eurosys_qwen_e3_calibration_assets_v2",
        "protocol_id": "eurosys-pa-v2-20260824",
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "exact_parameter_names": str(names),
        "exact_parameter_names_sha256": sha256_file(names),
        "exact_parameter_count": 96,
        "routing_source": str(source),
        "routing_source_sha256": sha256_file(source),
        "calibration_fixture": str(fixture),
        "calibration_fixture_sha256": sha256_file(fixture),
        "metadata": plan["metadata"],
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
