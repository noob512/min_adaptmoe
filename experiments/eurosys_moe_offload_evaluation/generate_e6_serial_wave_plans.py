#!/usr/bin/env python3
"""Expand E6 equal-capacity differences into deterministic per-layer waves."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    args = parser.parse_args()

    root = args.input_dir.resolve()
    manifest = load(root / "plan_classes.json")
    classes = manifest.get("classes")
    require(isinstance(classes, dict), "plan class manifest is incomplete")
    base_record = classes.get("base")
    require(isinstance(base_record, dict), "base class is absent")
    base = load(root / str(base_record["plan_file"]))
    base_selected = base.get("selected_by_layer")
    require(isinstance(base_selected, dict), "base selected_by_layer is absent")

    serial_manifest = copy.deepcopy(manifest)
    serial_manifest["experiment_schema"] = "eurosys_e6_serial_wave_classes_v1"
    serial_classes = serial_manifest["classes"]
    paths: dict[str, list[str]] = {}
    summary: dict[str, Any] = {}

    for target_name in ("d005", "d010", "d025", "d050", "d100"):
        target_record = classes.get(target_name)
        require(isinstance(target_record, dict), f"missing target {target_name}")
        target = load(root / str(target_record["plan_file"]))
        target_selected = target.get("selected_by_layer")
        require(isinstance(target_selected, dict), f"{target_name}: selected plan absent")
        require(set(target_selected) == set(base_selected), f"{target_name}: layers")

        changes: dict[str, list[tuple[int, int]]] = {}
        for layer in sorted(base_selected, key=int):
            old_values = {int(value) for value in base_selected[layer]}
            new_values = {int(value) for value in target_selected[layer]}
            removed = sorted(old_values - new_values)
            added = sorted(new_values - old_values)
            require(len(removed) == len(added), f"{target_name}/{layer}: capacity")
            changes[layer] = list(zip(removed, added, strict=True))
        wave_count = max(len(value) for value in changes.values())
        require(wave_count > 0, f"{target_name}: no difference")

        outbound: list[str] = []
        intermediate_names: list[str] = []
        for wave_index in range(1, wave_count + 1):
            if wave_index == wave_count:
                class_name = target_name
            else:
                class_name = f"{target_name}_w{wave_index:02d}"
                wave_plan = copy.deepcopy(base)
                selected: dict[str, list[int]] = {}
                changed_slots = 0
                for layer in sorted(base_selected, key=int):
                    values = {int(value) for value in base_selected[layer]}
                    for removed, added in changes[layer][:wave_index]:
                        values.remove(removed)
                        values.add(added)
                        changed_slots += 1
                    selected[layer] = sorted(values)
                wave_plan["schema"] = "eurosys_e6_serial_wave_plan_v1"
                wave_plan["selected_by_layer"] = selected
                wave_plan["e6_serial_wave"] = {
                    "target_class": target_name,
                    "wave_index": wave_index,
                    "wave_count": wave_count,
                    "changed_expert_slots": changed_slots,
                    "total_expert_slots": int(manifest["total_expert_slots"]),
                }
                filename = f"{class_name}.json"
                write_new(root / filename, wave_plan)
                serial_classes[class_name] = {
                    "capacity_gib": 8,
                    "plan_file": filename,
                    "serial_target": target_name,
                    "serial_wave_index": wave_index,
                    "serial_wave_count": wave_count,
                    "changed_expert_slots": changed_slots,
                }
                intermediate_names.append(class_name)
            outbound.append(class_name)

        reverse_intermediates = list(reversed(intermediate_names))
        paths[f"base->{target_name}"] = outbound
        paths[f"{target_name}->base"] = [*reverse_intermediates, "base"]
        summary[target_name] = {
            "wave_count": wave_count,
            "outbound_path": outbound,
            "return_path": paths[f"{target_name}->base"],
        }

    manifest_path = root / "serial_plan_classes.json"
    paths_path = root / "serial_transition_paths.json"
    write_new(manifest_path, serial_manifest)
    write_new(paths_path, paths)
    payload = {
        "schema": "eurosys_e6_serial_wave_generation_v1",
        "status": "ok",
        "manifest": str(manifest_path),
        "transition_paths": str(paths_path),
        "targets": summary,
    }
    write_new(root / "serial_generation_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
