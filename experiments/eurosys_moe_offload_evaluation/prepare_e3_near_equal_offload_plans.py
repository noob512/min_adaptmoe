#!/usr/bin/env python3
"""Generate frozen Qwen expert plans for requested per-layer widths."""

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
DEFAULT_WIDTHS = (11, 39, 90)
EXPECTED_SOURCE_SHA256 = (
    "912732708a46291b73e70c93291758747f285b364437ce77f58d53501d072f81"
)


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
    parser.add_argument(
        "--widths",
        nargs="+",
        type=int,
        default=DEFAULT_WIDTHS,
        help="cold experts per layer for each nested output plan",
    )
    args = parser.parse_args()

    widths = tuple(args.widths)
    if (
        not widths
        or len(widths) != len(set(widths))
        or tuple(sorted(widths)) != widths
        or any(width < 1 or width >= NUM_EXPERTS for width in widths)
    ):
        raise SystemExit("widths must be unique, increasing, and in [1, 127]")

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
    if total_slots != 7 * 32 * NUM_LAYERS * 8:
        raise SystemExit(f"decode calibration slot total differs: {total_slots}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    universe = {
        (layer, expert)
        for layer in range(NUM_LAYERS)
        for expert in range(NUM_EXPERTS)
    }
    manifest: dict[str, Any] = {
        "schema": "eurosys_e3_near_equal_offload_plans_v1",
        "source": str(source),
        "source_sha256": source_sha256,
        "widths": {},
    }
    for width in widths:
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
            selected_slots += sum(
                int(decode_counts[layer, expert]) for expert in experts
            )
        hot = universe - cold
        main_bytes = len(cold) * EXPERT_MAIN_BYTES
        scale_bytes = len(cold) * EXPERT_SCALE_BYTES
        canonical = json.dumps(
            selected_by_layer, sort_keys=True, separators=(",", ":")
        ).encode()
        plan = {
            "schema": "qwen_decode_cold_expert_plan_v1",
            "tier": f"e3_near_equal_cold{width}",
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
                "num_layers": NUM_LAYERS,
                "num_experts": NUM_EXPERTS,
                "experts_per_layer": width,
                "selected_main_bytes": main_bytes,
                "selected_scale_bytes": scale_bytes,
                "selected_bytes_including_scales": main_bytes + scale_bytes,
                "selected_main_gib": main_bytes / 2**30,
                "selected_decode_route_slots": selected_slots,
                "selected_decode_route_fraction": selected_slots / total_slots,
                "selected_by_layer_canonical_sha256": hashlib.sha256(
                    canonical
                ).hexdigest(),
            },
        }
        path = output_dir / f"cold{width}.json"
        write_json(path, plan)
        manifest["widths"][f"cold{width}"] = {
            "path": str(path),
            "sha256": sha256_file(path),
            **plan["metadata"],
        }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
