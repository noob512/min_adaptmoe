#!/usr/bin/env python3
"""Fail-closed paired comparison for E5 sequential and grouped PTE formals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: JSON root is not an object")
    return value


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    require(bool(values), "percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def input_hashes(root: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for line in (root / "input_sha256.txt").read_text(encoding="utf-8").splitlines():
        digest, path = line.split(None, 1)
        result.setdefault(Path(path.strip()).name, set()).add(digest)
    return result


def performance_samples(verification: dict[str, Any]) -> dict[int, dict[int, dict[str, Any]]]:
    output: dict[int, dict[int, dict[str, Any]]] = {1: {}, 4: {}}
    for item in verification.get("samples", []):
        if item.get("audit_trace") is True:
            continue
        match = re.search(r"_s(\d+)_b(\d+)_", str(item.get("case", "")))
        require(match is not None, f"cannot parse performance case {item.get('case')!r}")
        sample = int(match.group(1))
        batch = int(match.group(2))
        require(batch in output, f"unexpected batch {batch}")
        require(sample not in output[batch], f"duplicate B{batch} sample {sample}")
        output[batch][sample] = item
    for batch, members in output.items():
        require(set(members) == set(range(1, 11)), f"B{batch}: samples are not 1..10")
    return output


def paired_speedup(
    sequential: list[float],
    grouped: list[float],
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, Any]:
    require(len(sequential) == len(grouped) == 10, "paired sample count must be ten")
    require(all(value > 0 and math.isfinite(value) for value in sequential + grouped), "bad timing")
    point = statistics.median(sequential) / statistics.median(grouped)
    rng = random.Random(seed)
    ratios: list[float] = []
    for _ in range(resamples):
        indices = [rng.randrange(len(sequential)) for _ in sequential]
        ratios.append(
            statistics.median(sequential[index] for index in indices)
            / statistics.median(grouped[index] for index in indices)
        )
    return {
        "definition": "median(sequential_ms) / median(grouped_ms)",
        "point_estimate": point,
        "paired_bootstrap_95_ci": [percentile(ratios, 0.025), percentile(ratios, 0.975)],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequential_root", type=Path)
    parser.add_argument("grouped_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    roots = {
        "sequential": args.sequential_root.resolve(),
        "grouped": args.grouped_root.resolve(),
    }
    verifications = {name: load(root / "verification.json") for name, root in roots.items()}
    for name, verification in verifications.items():
        require(verification.get("status") == "ok", f"{name}: verifier status")
        require(verification.get("case_count") == 22, f"{name}: case count")
        config = verification.get("run_config", {})
        require(config.get("mode") == "formal", f"{name}: not formal")
        require(config.get("performance_repeats") == 10, f"{name}: repeat count")
        require(config.get("transition_mode") == name, f"{name}: transition mode")

    sequential_config = verifications["sequential"]["run_config"]
    grouped_config = verifications["grouped"]["run_config"]
    ignored_config = {"transition_mode", "schema"}
    for key in sorted(set(sequential_config) | set(grouped_config)):
        if key in ignored_config:
            continue
        require(sequential_config.get(key) == grouped_config.get(key), f"run config differs: {key}")

    critical_inputs = {
        "benchmark_dual_topology_steps.py",
        "run_experiment.sh",
        "prepare_inputs.py",
        "benchmark_phase_steps.py",
        "benchmark_full_graph_steps.py",
        "utils.py",
        "gpu_model_runner.py",
        "gpu_worker.py",
        "uvm.py",
        "uvm_static_alias.py",
        "uvm_allocator.so",
        "nvidia-uvm.ko",
        "evaluation_fixture_token_ids.json",
    }
    hashes = {name: input_hashes(root) for name, root in roots.items()}
    for basename in sorted(critical_inputs):
        left = hashes["sequential"].get(basename)
        right = hashes["grouped"].get(basename)
        require(left is not None and right is not None, f"missing input hash: {basename}")
        require(left == right, f"execution input differs: {basename}")

    samples = {name: performance_samples(value) for name, value in verifications.items()}
    metric_names = (
        "resume_wall_ms",
        "resume_ioctl_ms",
        "repark_wall_ms",
        "repark_ioctl_ms",
        "cycle_e2e_ms",
    )
    comparison: dict[str, Any] = {}
    for batch in (1, 4):
        metrics: dict[str, Any] = {}
        for metric_index, metric in enumerate(metric_names):
            sequential_values = [
                float(samples["sequential"][batch][sample][metric]) for sample in range(1, 11)
            ]
            grouped_values = [
                float(samples["grouped"][batch][sample][metric]) for sample in range(1, 11)
            ]
            metrics[metric] = {
                "sequential": {
                    "median_ms": statistics.median(sequential_values),
                    "p95_ms": percentile(sequential_values, 0.95),
                    "p99_ms": percentile(sequential_values, 0.99),
                },
                "grouped": {
                    "median_ms": statistics.median(grouped_values),
                    "p95_ms": percentile(grouped_values, 0.95),
                    "p99_ms": percentile(grouped_values, 0.99),
                },
                "sequential_over_grouped_speedup": paired_speedup(
                    sequential_values,
                    grouped_values,
                    seed=20260828 + batch * 100 + metric_index,
                ),
            }
        comparison[str(batch)] = metrics

    output = {
        "schema": "eurosys_e5_pte_mode_paired_comparison_v1",
        "status": "ok",
        "sequential_root": str(roots["sequential"]),
        "grouped_root": str(roots["grouped"]),
        "common_custom_module_sha256": sequential_config["custom_module_sha256"],
        "critical_execution_inputs_equal": True,
        "process_pairs_per_batch": 10,
        "comparison": comparison,
        "verifier": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    require(not args.output.exists(), f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
