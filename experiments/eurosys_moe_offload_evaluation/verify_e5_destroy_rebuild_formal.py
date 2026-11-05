#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def metric_summary(values: list[float], *, seed: int) -> dict[str, Any]:
    if not values:
        raise ValueError("metric has no samples")
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values))) for _ in range(10_000)
    )
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "median_bootstrap_95_ci": [
            percentile(medians, 0.025),
            percentile(medians, 0.975),
        ],
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": seed,
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


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
    hashes_by_label: dict[str, set[str]] = {"matched_b4": set(), "matched_b1": set()}
    metrics_by_label: dict[str, dict[str, list[float]]] = {
        label: {} for label in hashes_by_label
    }
    for repeat_index, repeat in enumerate(repeats, start=1):
        child_verification = load_json(repeat / "verification.json")
        result = load_json(repeat / "result.json")
        if child_verification.get("status") != "ok":
            raise ValueError(f"child verification failed: {repeat}")
        if child_verification.get("faults_detected") is not False:
            raise ValueError(f"fault detected: {repeat}")
        if child_verification.get("resources_released") is not True:
            raise ValueError(f"resources not released: {repeat}")

        generations = child_verification.get("generations")
        rebuilds = result.get("dynamic_plan_rebuilds")
        if not isinstance(generations, list) or len(generations) != 2:
            raise ValueError(f"expected two verified generations: {repeat}")
        if not isinstance(rebuilds, list) or len(rebuilds) != 2:
            raise ValueError(f"expected two rebuild records: {repeat}")

        for verified, record in zip(generations, rebuilds):
            label = str(verified.get("label"))
            if label not in metrics_by_label or record.get("label") != label:
                raise ValueError(f"unexpected generation label/order: {repeat}")
            if int(verified.get("selected_bytes", -1)) != 2_415_919_104:
                raise ValueError(f"selected-byte mismatch: {repeat}/{label}")
            rebuild = record.get("rebuild")
            if not isinstance(rebuild, dict) or rebuild.get("ok") is not True:
                raise ValueError(f"outer rebuild failed: {repeat}/{label}")
            inner = rebuild.get("rebuild")
            if not isinstance(inner, dict) or inner.get("ok") is not True:
                raise ValueError(f"plan rebuild failed: {repeat}/{label}")
            before = inner.get("parameter_vas_before")
            after = inner.get("parameter_vas_after")
            if not isinstance(before, list) or before != after or len(before) != 8:
                raise ValueError(f"stable VA gate failed: {repeat}/{label}")
            destroy = inner.get("destroy", {}).get("destroy", {})
            if not (
                isinstance(destroy, dict)
                and destroy.get("ok") is True
                and destroy.get("release_backing") is True
                and int(destroy.get("remaining_alias_count", -1)) == 0
                and int(destroy.get("remaining_alias_bytes", -1)) == 0
                and int(destroy.get("remaining_backing_bytes", -1)) == 0
            ):
                raise ValueError(f"DESTROY cleanup gate failed: {repeat}/{label}")
            reconciliation = (
                inner.get("finalize", {})
                .get("original_release", {})
                .get("cpu_state_reconciliation", {})
            )
            if not (
                isinstance(reconciliation, dict)
                and reconciliation.get("ok") is True
                and reconciliation.get("physical_cpu_only") is True
                and reconciliation.get("reconciled") is True
                and int(reconciliation.get("expected_bytes", -1)) == 2_415_919_104
            ):
                raise ValueError(f"CPU residency reconciliation failed: {repeat}/{label}")

            output_hash = str(verified.get("output_sha256", ""))
            if len(output_hash) != 64:
                raise ValueError(f"missing output hash: {repeat}/{label}")
            hashes_by_label[label].add(output_hash)
            values = {
                "rebuild_total_ms": float(verified["rebuild_elapsed_ms"]),
                "plan_rebuild_ms": float(verified["plan_rebuild_ms"]),
                "graph_reset_ms": float(verified["graph_reset_ms"]),
                "graph_capture_ms": float(verified["graph_capture_ms"]),
                "request_elapsed_ms": float(verified["request_elapsed_ms"]),
                "alias_destroy_ms": float(inner["alias_destroy_ns"]) / 1e6,
                "binding_rebuild_ms": float(inner["binding_rebuild_ns"]) / 1e6,
                "hot_restore_ms": float(inner["hot_restore_ns"]) / 1e6,
                "finalize_ms": float(inner["finalize_ns"]) / 1e6,
                "destroy_ioctl_ms": float(destroy["ioctl_wall_ns"]) / 1e6,
            }
            for metric, value in values.items():
                metrics_by_label[label].setdefault(metric, []).append(value)
            samples.append(
                {
                    "repeat": repeat_index,
                    "label": label,
                    "batch_size": int(verified["batch_size"]),
                    "output_sha256": output_hash,
                    **values,
                }
            )

    for label, hashes in hashes_by_label.items():
        if len(hashes) != 1:
            raise ValueError(f"output hashes differ across repeats for {label}: {hashes}")

    aggregate: dict[str, Any] = {}
    for label_index, (label, metrics) in enumerate(metrics_by_label.items()):
        aggregate[label] = {
            metric: metric_summary(values, seed=202_608_280 + label_index * 100 + index)
            for index, (metric, values) in enumerate(sorted(metrics.items()))
        }
        aggregate[label]["output_sha256"] = next(iter(hashes_by_label[label]))

    digest = hashlib.sha256()
    for sample in samples:
        digest.update(json.dumps(sample, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    payload = {
        "schema": "e5_destroy_rebuild_formal_verification_v1",
        "status": "ok",
        "run_root": str(root),
        "repeat_count": len(repeats),
        "generation_count": len(samples),
        "all_fault_scans_clean": True,
        "all_resources_released": True,
        "stable_va_preserved": True,
        "cpu_residency_reconciliation_verified": True,
        "samples_sha256": digest.hexdigest(),
        "samples": samples,
        "aggregate": aggregate,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
