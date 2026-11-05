#!/usr/bin/env python3
"""Generate deterministic equal-capacity E6 plan-difference classes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PERCENTAGES = (5, 10, 25, 50, 100)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evenly_spaced_indices(total: int, count: int) -> tuple[int, ...]:
    require(0 < count <= total, "invalid change count")
    values = tuple(math.floor((index + 0.5) * total / count) for index in range(count))
    require(len(set(values)) == count, "difference index selection is not unique")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-plan", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expert-count", type=int, default=64)
    args = parser.parse_args()

    base_path = args.base_plan.resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    selected = base.get("selected_by_layer")
    require(isinstance(selected, dict) and selected, "base plan has no selected_by_layer")
    layers = sorted(selected, key=lambda value: int(value))
    counts = {len(selected[layer]) for layer in layers}
    require(len(counts) == 1 and next(iter(counts)) > 0, "base plan is not equal per layer")
    per_layer = next(iter(counts))
    require(
        all(
            len(set(int(value) for value in selected[layer])) == per_layer
            and all(0 <= int(value) < args.expert_count for value in selected[layer])
            for layer in layers
        ),
        "base plan contains invalid expert IDs",
    )

    flat_coordinates = [
        (layer, offset)
        for layer in layers
        for offset in range(per_layer)
    ]
    total = len(flat_coordinates)
    require(int(base.get("selected_expert_count", -1)) == total, "selected count mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    base_copy = args.output_dir / "base_c8.json"
    write_json(base_copy, base)
    classes: dict[str, dict[str, Any]] = {
        "base": {
            "capacity_gib": 8,
            "plan_file": base_copy.name,
            "difference_percent": 0,
            "changed_expert_slots": 0,
        }
    }
    generated: list[dict[str, Any]] = []

    for percentage in PERCENTAGES:
        change_count = max(1, round(total * percentage / 100))
        changed_indices = set(evenly_spaced_indices(total, change_count))
        variant = copy.deepcopy(base)
        variant_selected = {
            layer: [int(value) for value in selected[layer]] for layer in layers
        }
        replacement_pool = {
            layer: [
                value
                for value in range(args.expert_count)
                if value not in {int(item) for item in selected[layer]}
            ]
            for layer in layers
        }
        changed_by_layer: dict[str, int] = {layer: 0 for layer in layers}

        for flat_index in sorted(changed_indices):
            layer, offset = flat_coordinates[flat_index]
            candidates = replacement_pool[layer]
            require(bool(candidates), f"layer {layer} has no replacement expert")
            choice_index = (flat_index + percentage) % len(candidates)
            choice = candidates.pop(choice_index)
            variant_selected[layer][offset] = choice
            changed_by_layer[layer] += 1

        for layer in layers:
            variant_selected[layer] = sorted(variant_selected[layer])
            require(
                len(set(variant_selected[layer])) == per_layer,
                f"layer {layer} replacement introduced a duplicate",
            )
        observed = sum(
            len(set(int(value) for value in selected[layer]) ^ set(variant_selected[layer])) // 2
            for layer in layers
        )
        require(observed == change_count, "observed plan difference mismatch")
        actual_percent = 100.0 * observed / total
        variant["schema"] = "eurosys_e6_controlled_difference_plan_v1"
        variant["selected_by_layer"] = variant_selected
        variant["e6_difference"] = {
            "requested_percent": percentage,
            "actual_percent": actual_percent,
            "changed_expert_slots": observed,
            "total_expert_slots": total,
            "experts_per_layer": per_layer,
            "changed_by_layer": changed_by_layer,
            "base_plan": str(base_path),
            "base_plan_sha256": digest(base_path),
            "selection": "deterministic_evenly_spaced_coordinates",
        }
        output = args.output_dir / f"difference_{percentage:03d}.json"
        write_json(output, variant)
        name = f"d{percentage:03d}"
        classes[name] = {
            "capacity_gib": 8,
            "plan_file": output.name,
            "difference_percent": percentage,
            "actual_difference_percent": actual_percent,
            "changed_expert_slots": observed,
        }
        generated.append(
            {
                "class_name": name,
                "path": str(output.resolve()),
                "sha256": digest(output),
                **classes[name],
            }
        )

    manifest = {
        "schema": "vllm_uvm_static_alias_plan_classes_v1",
        "experiment_schema": "eurosys_e6_difference_classes_v1",
        "allow_equal_expert_widths": True,
        "initial_class": "base",
        "base_plan": str(base_path),
        "base_plan_sha256": digest(base_path),
        "experts_per_layer": per_layer,
        "total_expert_slots": total,
        "selected_main_bytes": int(base["selected_main_bytes"]),
        "classes": classes,
    }
    manifest_path = args.output_dir / "plan_classes.json"
    write_json(manifest_path, manifest)
    summary = {
        "schema": "eurosys_e6_difference_plan_generation_v1",
        "status": "ok",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": digest(manifest_path),
        "base_copy_sha256": digest(base_copy),
        "generated": generated,
    }
    write_json(args.output_dir / "generation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
