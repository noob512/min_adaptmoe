#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--expected-repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.run_root.resolve()
    repeats = sorted(path for path in root.glob("repeat_[0-9][0-9]") if path.is_dir())
    if len(repeats) != args.expected_repeats:
        raise ValueError(
            f"expected {args.expected_repeats} repeat directories, found {len(repeats)}"
        )

    samples: list[dict[str, Any]] = []
    hashes: dict[str, set[str]] = {"matched_b4": set(), "matched_b1": set()}
    metrics: dict[str, dict[str, list[float]]] = {label: {} for label in hashes}
    for repeat_index, repeat in enumerate(repeats, 1):
        child = load_json(repeat / "verification.json")
        result = load_json(repeat / "result.json")
        if child.get("status") != "ok" or child.get("faults_detected") is not False:
            raise ValueError(f"child correctness/fault gate failed: {repeat}")
        if child.get("resources_released") is not True:
            raise ValueError(f"child cleanup gate failed: {repeat}")
        generations = child.get("generations")
        rebuilds = result.get("dynamic_plan_rebuilds")
        if not isinstance(generations, list) or len(generations) != 2:
            raise ValueError(f"expected two verified generations: {repeat}")
        if not isinstance(rebuilds, list) or len(rebuilds) != 2:
            raise ValueError(f"expected two rebuild records: {repeat}")

        for verified, record in zip(generations, rebuilds):
            label = str(verified.get("label"))
            if label not in metrics or record.get("label") != label:
                raise ValueError(f"generation order differs: {repeat}/{label}")
            outer = record.get("rebuild")
            if not isinstance(outer, dict) or outer.get("ok") is not True:
                raise ValueError(f"outer recapture failed: {repeat}/{label}")
            if outer.get("parameter_va_mode") != "nonstable_reallocate_recapture":
                raise ValueError(f"wrong VA mode: {repeat}/{label}")
            inner = outer.get("rebuild")
            if not isinstance(inner, dict) or inner.get("ok") is not True:
                raise ValueError(f"Parameter reallocation failed: {repeat}/{label}")
            before = inner.get("parameter_vas_before")
            after = inner.get("parameter_vas_after")
            reallocations = inner.get("reallocations")
            if not (
                isinstance(before, list)
                and isinstance(after, list)
                and len(before) == len(after) == 8
                and all(a != b for a, b in zip(before, after))
                and inner.get("all_parameter_vas_changed") is True
                and isinstance(reallocations, list)
                and len(reallocations) == 8
                and all(item.get("va_changed") is True for item in reallocations)
            ):
                raise ValueError(f"non-stable VA gate failed: {repeat}/{label}")
            if int(inner.get("selected_bytes", -1)) != 2_415_919_104:
                raise ValueError(f"placement byte mismatch: {repeat}/{label}")
            destroy = inner.get("destroy", {}).get("destroy", {})
            if not (
                isinstance(destroy, dict)
                and destroy.get("ok") is True
                and int(destroy.get("remaining_alias_count", -1)) == 0
                and int(destroy.get("remaining_backing_bytes", -1)) == 0
            ):
                raise ValueError(f"old alias cleanup failed: {repeat}/{label}")
            if int(outer.get("destroyed_graph_count", 0)) <= 0:
                raise ValueError(f"old Graphs were not destroyed: {repeat}/{label}")
            if float(outer.get("graph_capture_ns", 0)) <= 0:
                raise ValueError(f"Graph recapture missing: {repeat}/{label}")

            output_hash = str(verified.get("output_sha256", ""))
            if len(output_hash) != 64:
                raise ValueError(f"output hash missing: {repeat}/{label}")
            hashes[label].add(output_hash)
            values = {
                "rebuild_total_ms": float(verified["rebuild_elapsed_ms"]),
                "parameter_reallocate_ms": float(inner["parameter_reallocate_ns"]) / 1e6,
                "graph_capture_ms": float(outer["graph_capture_ns"]) / 1e6,
                "finalize_ms": float(inner["finalize_ns"]) / 1e6,
                "request_elapsed_ms": float(verified["request_elapsed_ms"]),
            }
            for metric, value in values.items():
                metrics[label].setdefault(metric, []).append(value)
            samples.append(
                {
                    "repeat": repeat_index,
                    "label": label,
                    "output_sha256": output_hash,
                    **values,
                }
            )

    for label, values in hashes.items():
        if len(values) != 1:
            raise ValueError(f"output hashes differ across repeats for {label}: {values}")
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(json.dumps(sample, sort_keys=True).encode("utf-8") + b"\n")
    payload = {
        "schema": "e5_nonstable_va_recapture_formal_verification_v1",
        "status": "ok",
        "run_root": str(root),
        "repeat_count": len(repeats),
        "generation_count": len(samples),
        "all_parameter_vas_changed": True,
        "all_graphs_recaptured": True,
        "all_fault_scans_clean": True,
        "all_resources_released": True,
        "samples_sha256": digest.hexdigest(),
        "samples": samples,
        "aggregate": {
            label: {metric: summary(values) for metric, values in values_by_metric.items()}
            for label, values_by_metric in metrics.items()
        },
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
