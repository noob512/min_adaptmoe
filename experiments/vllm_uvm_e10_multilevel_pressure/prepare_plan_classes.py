#!/usr/bin/env python3
"""Freeze nested E10 plan classes from the calibrated cold ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


NUM_LAYERS = 48
NUM_EXPERTS = 128
EXPERT_MAIN_BYTES = 4_718_592
DEFAULT_WIDTHS = {"cold70": 70, "cold80": 80, "cold89": 89}
DEFAULT_TARGETS = {"cold70": 1364, "cold80": 2730, "cold89": 4095}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_mapping(raw: str, label: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in raw.split(","):
        name, separator, value = item.strip().partition(":")
        if not separator or not name:
            raise argparse.ArgumentTypeError(f"invalid {label} item: {item!r}")
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid {label} value: {item!r}"
            ) from exc
        if number <= 0 or name in result:
            raise argparse.ArgumentTypeError(f"invalid {label} item: {item!r}")
        result[name] = number
    if not result:
        raise argparse.ArgumentTypeError(f"{label} cannot be empty")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", required=True, type=Path)
    parser.add_argument("--wave-plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--widths",
        default=",".join(f"{key}:{value}" for key, value in DEFAULT_WIDTHS.items()),
        help="ordered class:experts-per-layer mapping",
    )
    parser.add_argument(
        "--targets",
        default=",".join(f"{key}:{value}" for key, value in DEFAULT_TARGETS.items()),
        help="class:KV-target-block mapping",
    )
    args = parser.parse_args()

    widths = parse_mapping(args.widths, "widths")
    targets = parse_mapping(args.targets, "targets")
    if set(widths) != set(targets):
        raise SystemExit("width and target class names differ")
    ordered_widths = list(widths.items())
    if [width for _, width in ordered_widths] != sorted(set(widths.values())):
        raise SystemExit("class widths must be unique and increasing")
    if [targets[name] for name, _ in ordered_widths] != sorted(set(targets.values())):
        raise SystemExit("KV targets must be unique and increasing")
    base_class, base_width = ordered_widths[0]

    base_path = args.base_plan.resolve()
    wave_path = args.wave_plan.resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    wave_payload = json.loads(wave_path.read_text(encoding="utf-8"))
    waves = wave_payload["waves"]
    if int(wave_payload["base_experts_per_layer"]) != base_width:
        raise SystemExit(f"wave plan does not start at {base_class}")
    if len(waves) < ordered_widths[-1][1] - base_width:
        raise SystemExit(f"wave plan is too short for {ordered_widths[-1][0]}")

    base_by_layer = {
        int(layer): {int(expert) for expert in experts}
        for layer, experts in base["selected_by_layer"].items()
    }
    if set(base_by_layer) != set(range(NUM_LAYERS)):
        raise SystemExit("base plan layer inventory differs")
    if {len(experts) for experts in base_by_layer.values()} != {base_width}:
        raise SystemExit(f"base plan is not uniform {base_class}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    class_records: dict[str, dict[str, Any]] = {}
    input_records: dict[str, dict[str, Any]] = {}
    universe = {
        (layer, expert)
        for layer in range(NUM_LAYERS)
        for expert in range(NUM_EXPERTS)
    }
    for class_name, width in ordered_widths:
        selected_by_layer = {
            layer: set(experts) for layer, experts in base_by_layer.items()
        }
        for wave in waves[: width - base_width]:
            layer_experts = wave["layer_experts"]
            for layer in range(NUM_LAYERS):
                key = f"model.layers.{layer}.mlp.experts"
                selected_by_layer[layer].add(int(layer_experts[key]))
        if {len(experts) for experts in selected_by_layer.values()} != {width}:
            raise SystemExit(f"{class_name} is not a nested uniform tier")
        selected = {
            (layer, expert)
            for layer, experts in selected_by_layer.items()
            for expert in experts
        }
        hot = universe - selected
        main_bytes = len(selected) * EXPERT_MAIN_BYTES
        plan = {
            "schema": "qwen_decode_cold_expert_plan_v1",
            "tier": f"e10_multilevel_{class_name}",
            "source": str(wave_path),
            "source_sha256": sha256_file(wave_path),
            "selection_rule": (
                f"{width} minimum summed calibration-decode route counts "
                "per layer; ties by expert_id; nested cold70 ranking"
            ),
            "selected_by_layer": {
                str(layer): sorted(experts)
                for layer, experts in sorted(selected_by_layer.items())
            },
            "offload_plan": [
                {"layer_id": layer, "expert_id": expert}
                for layer, expert in sorted(selected)
            ],
            "hot_plan": [
                {"layer_id": layer, "expert_id": expert}
                for layer, expert in sorted(hot)
            ],
            "offload_plan_records": len(selected),
            "hot_plan_records": len(hot),
            "metadata": {
                "num_layers": NUM_LAYERS,
                "num_experts": NUM_EXPERTS,
                "experts_per_layer": width,
                "selected_main_bytes": main_bytes,
                "selected_main_gib": main_bytes / 2**30,
                f"incremental_waves_from_{base_class}": width - base_width,
                "policy": "cold-frequency",
            },
        }
        plan_path = output_dir / f"{class_name}.json"
        write_json(plan_path, plan)
        class_records[class_name] = {
            "capacity_gib": main_bytes / 2**30,
            "plan_file": plan_path.name,
        }
        input_records[class_name] = {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
            "experts_per_layer": width,
            "selected_main_bytes": main_bytes,
        }

    manifest = {
        "schema": "vllm_uvm_static_alias_plan_classes_v1",
        "initial_class": base_class,
        "classes": class_records,
        "metadata": {
            "policy": "cold-frequency",
            "nested": True,
            "kv_target_blocks": targets,
            "expert_step_bytes": {
                f"{left_name}_to_{right_name}": (
                    right_width - left_width
                ) * NUM_LAYERS * EXPERT_MAIN_BYTES
                for (left_name, left_width), (right_name, right_width)
                in zip(ordered_widths, ordered_widths[1:])
            },
            "inputs": input_records,
            "base_plan": str(base_path),
            "base_plan_sha256": sha256_file(base_path),
            "wave_plan": str(wave_path),
            "wave_plan_sha256": sha256_file(wave_path),
        },
    }
    manifest_path = output_dir / "plan_classes.json"
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
