#!/usr/bin/env python3
"""Freeze the two nested cold-frequency classes used by E10 qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--superset", required=True, type=Path)
    parser.add_argument("--wave", required=True, type=Path)
    parser.add_argument("--tier-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    args = parser.parse_args()

    base = load(args.base.resolve())
    superset = load(args.superset.resolve())
    wave = load(args.wave.resolve())
    base_selected = base.get("selected_by_layer")
    superset_selected = superset.get("selected_by_layer")
    wave_records = wave.get("offload_plan")
    if not isinstance(base_selected, dict) or not isinstance(
        superset_selected, dict
    ):
        raise ValueError("base/superset selected_by_layer is absent")
    if not isinstance(wave_records, list):
        raise ValueError("incremental wave is absent")
    layer_ids = sorted(int(value) for value in base_selected)
    if layer_ids != list(range(48)) or set(base_selected) != set(superset_selected):
        raise ValueError("expected exactly Qwen layers 0..47")
    wave_by_layer: dict[int, int] = {}
    for record in wave_records:
        layer_id = int(record["layer_id"])
        expert_id = int(record["expert_id"])
        if layer_id in wave_by_layer:
            raise ValueError(f"wave repeats layer {layer_id}")
        wave_by_layer[layer_id] = expert_id
    if set(wave_by_layer) != set(layer_ids):
        raise ValueError("wave must select exactly one expert in every layer")

    selected: dict[str, list[int]] = {}
    for layer_id in layer_ids:
        key = str(layer_id)
        base_ids = {int(value) for value in base_selected[key]}
        superset_ids = {int(value) for value in superset_selected[key]}
        expert_id = wave_by_layer[layer_id]
        if len(base_ids) != 29 or len(superset_ids) != 33:
            raise ValueError(f"layer {layer_id}: unexpected frozen tier width")
        if not base_ids < superset_ids or expert_id in base_ids:
            raise ValueError(f"layer {layer_id}: wave is not a nested addition")
        if expert_id not in superset_ids:
            raise ValueError(f"layer {layer_id}: wave expert is outside superset")
        selected[key] = sorted(base_ids | {expert_id})

    base_bytes = int(base["metadata"]["selected_main_bytes"])
    superset_bytes = int(superset["metadata"]["selected_main_bytes"])
    expert_bytes = (superset_bytes - base_bytes) // (4 * len(layer_ids))
    if expert_bytes * 4 * len(layer_ids) != superset_bytes - base_bytes:
        raise ValueError("frozen plan byte difference is not four experts per layer")
    selected_bytes = base_bytes + expert_bytes * len(layer_ids)
    tier = {
        "schema": "vllm_uvm_cold_expert_plan_v1",
        "model": "Qwen/Qwen3-30B-A3B-FP8",
        "tier": "cold30",
        "selection_rule": (
            "frozen cold29 calibration-frequency tier plus the first nested "
            "cold expert from the frozen cold33 superset; ties by expert_id"
        ),
        "selected_by_layer": selected,
        "selected_expert_count": len(layer_ids) * 30,
        "selected_main_bytes": selected_bytes,
        "selected_main_gib": selected_bytes / 2**30,
        "source": {
            "base": str(args.base.resolve()),
            "base_sha256": sha256(args.base.resolve()),
            "superset": str(args.superset.resolve()),
            "superset_sha256": sha256(args.superset.resolve()),
            "wave": str(args.wave.resolve()),
            "wave_sha256": sha256(args.wave.resolve()),
        },
    }
    write(args.tier_output, tier)

    relative_base = Path(
        "../../vllm_uvm_physical_kv_pool_resize/inputs/cold_decode_plan_6g.json"
    )
    manifest = {
        "schema": "vllm_uvm_static_alias_plan_classes_v1",
        "initial_class": "cold29",
        "classes": {
            "cold29": {
                "capacity_gib": base_bytes / 2**30,
                "plan_file": str(relative_base),
            },
            "cold30": {
                "capacity_gib": selected_bytes / 2**30,
                "plan_file": args.tier_output.name,
            },
        },
        "metadata": {
            "policy": "cold-frequency",
            "nested": True,
            "difference_experts_per_layer": 1,
            "difference_selected_main_bytes": selected_bytes - base_bytes,
            "tier_output_sha256": sha256(args.tier_output),
        },
    }
    write(args.manifest_output, manifest)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
