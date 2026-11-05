#!/usr/bin/env python3
"""Verify and aggregate the E9 KTransformers one-factor formal matrix."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate(values: list[float], *, seed: int) -> dict[str, Any]:
    require(values, "empty metric")
    rng = random.Random(seed)
    count = len(values)
    medians = [
        statistics.median(values[rng.randrange(count)] for _ in range(count))
        for _ in range(10_000)
    ]
    return {
        "count": count,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "median_bootstrap_95_ci": [percentile(medians, 0.025), percentile(medians, 0.975)],
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": seed,
    }


def sequence_agreement(
    reference: list[list[int]], candidate: list[list[int]]
) -> dict[str, Any]:
    """Describe autoregressive agreement without treating FP drift as failure.

    E9 deliberately changes which FP8 experts execute on the GPU and the batch
    formation used by SGLang.  Those changes can move a greedy argmax at a
    near-tie even though all requests execute the same model.  The formal gate
    therefore requires exact output for arms that do not change the numerical
    path, and repeat stability for every arm.  This helper preserves the full
    cross-arm disagreement as a reported diagnostic.
    """

    require(len(reference) == len(candidate), "request count mismatch")
    exact_requests = 0
    equal_tokens = 0
    token_count = 0
    common_prefix_tokens = 0
    per_request: list[dict[str, Any]] = []
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        require(len(expected) == len(actual), f"request {index}: token count mismatch")
        exact = expected == actual
        exact_requests += int(exact)
        equal = sum(left == right for left, right in zip(expected, actual))
        prefix = next(
            (offset for offset, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]),
            len(expected),
        )
        equal_tokens += equal
        token_count += len(expected)
        common_prefix_tokens += prefix
        per_request.append(
            {
                "request_index": index,
                "exact": exact,
                "equal_token_positions": equal,
                "token_count": len(expected),
                "common_prefix_tokens": prefix,
            }
        )
    request_count = len(reference)
    return {
        "exact_requests": exact_requests,
        "request_count": request_count,
        "exact_request_fraction": exact_requests / request_count,
        "equal_token_positions": equal_tokens,
        "token_count": token_count,
        "equal_token_position_fraction": equal_tokens / token_count,
        "mean_common_prefix_tokens": common_prefix_tokens / request_count,
        "per_request": per_request,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    config = load(root / "run_config.json")
    blocks = int(config["process_blocks"])
    minimum = 1 if config["mode"] == "qualification" else 5
    require(blocks >= minimum, "insufficient process blocks")
    expected_arms = set(config["arms"])

    rows = [
        line.split("\t")
        for line in (root / "execution_status.tsv").read_text(encoding="utf-8").splitlines()[1:]
        if line
    ]
    require(len(rows) == blocks * len(expected_arms), "case count mismatch")
    require(all(row[-1] == "0" for row in rows), "one or more E9 cases failed")
    seen: set[tuple[int, str]] = set()
    metrics: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    hashes: dict[str, set[str]] = defaultdict(set)
    arm_hashes: dict[tuple[str, str], set[str]] = defaultdict(set)
    block_hashes: dict[tuple[int, str, str], str] = {}
    sequences: dict[tuple[int, str, str], list[list[int]]] = {}
    samples: list[dict[str, Any]] = []
    for block_raw, _position, arm, case_label, _rc in rows:
        block = int(block_raw)
        require((block, arm) not in seen, f"duplicate block/arm: {block}/{arm}")
        seen.add((block, arm))
        case_root = root / "cases" / case_label
        benchmark = load(case_root / "benchmark.json")
        runtime = load(case_root / "runtime_verification.json")
        metadata = load(case_root / "case_metadata.json")
        require(benchmark.get("status") == "ok", f"{case_label}: benchmark status")
        require(runtime.get("status") == "ok", f"{case_label}: runtime verifier")
        require((case_root / "kernel_log_returncode.txt").read_text().strip() == "0", f"{case_label}: kernel log")
        kernel = (case_root / "kernel.since.log").read_text(encoding="utf-8", errors="replace").lower()
        require("nvrm: xid" not in kernel and "fault_pde" not in kernel, f"{case_label}: kernel fault")
        expected = config["arms"][arm]
        require(int(metadata["cpu_threads"]) == int(expected["threads"]), f"{case_label}: threads")
        require(bool(metadata["cuda_graph_enabled"]) is bool(expected["graph"]), f"{case_label}: graph")
        require(metadata["placement_strategy"] == expected["placement"], f"{case_label}: placement")
        require(bool(metadata["overlap_enabled"]) is bool(expected["overlap"]), f"{case_label}: overlap")
        require(bool(metadata["mixed_chunk_enabled"]) is bool(expected["mixed"]), f"{case_label}: mixed")
        phases = benchmark.get("phases")
        require(isinstance(phases, list) and len(phases) == 2, f"{case_label}: phases")
        for phase in phases:
            phase_name = str(phase["phase"])
            require(phase_name in {"c1", "c8"}, f"{case_label}: phase name")
            require(phase.get("all_requests_produced_fixed_output") is True, f"{case_label}/{phase_name}: output")
            output_hash = str(phase["output_fingerprints_sha256"])
            hashes[phase_name].add(output_hash)
            arm_hashes[(arm, phase_name)].add(output_hash)
            block_hashes[(block, arm, phase_name)] = output_hash
            phase_sequences: list[list[int]] = []
            for request in phase.get("requests", []):
                ids = request.get("output_token_ids")
                require(
                    isinstance(ids, list)
                    and len(ids) == int(phase["expected_output_tokens"]) // int(phase["request_count"])
                    and all(isinstance(token, int) and not isinstance(token, bool) for token in ids),
                    f"{case_label}/{phase_name}: complete output token IDs",
                )
                phase_sequences.append(ids)
            require(
                len(phase_sequences) == int(phase["request_count"]),
                f"{case_label}/{phase_name}: request sequence count",
            )
            sequences[(block, arm, phase_name)] = phase_sequences
            values = {
                "throughput_tps": float(phase["output_tokens_per_second"]),
                "mean_ttft_ms": float(phase["mean_ttft_ms"]),
                "p99_ttft_ms": float(phase["p99_ttft_ms"]),
                "mean_tpot_ms": float(phase["mean_tpot_ms"]),
                "p99_tpot_ms": float(phase["p99_tpot_ms"]),
            }
            for metric, value in values.items():
                require(math.isfinite(value) and value >= 0, f"{case_label}/{phase_name}/{metric}")
                metrics[(arm, phase_name, metric)].append(value)
            samples.append({"block": block, "arm": arm, "case": case_label, "phase": phase_name, "output_sha256": output_hash, **values})
    require(len(seen) == blocks * len(expected_arms), "incomplete block/arm coverage")

    # Repeated cold processes of the same arm must be bitwise deterministic.
    # The controls below do not change expert placement or batch formation, so
    # they must also match baseline exactly.  Uniform placement and no-mixed do
    # change the floating-point execution/reduction path and are reported, but
    # are not incorrectly required to preserve an autoregressive token suffix.
    repeat_stability = {
        arm: {
            phase: len(arm_hashes[(arm, phase)]) == 1
            for phase in ("c1", "c8")
        }
        for arm in sorted(expected_arms)
    }
    require(
        all(value for phases in repeat_stability.values() for value in phases.values()),
        "one or more E9 arms are not repeat-stable across cold process blocks",
    )
    exact_control_arms = ("baseline", "threads7", "threads28", "eager", "no_overlap")
    control_exact: dict[str, dict[str, bool]] = {}
    for arm in exact_control_arms:
        control_exact[arm] = {}
        for phase in ("c1", "c8"):
            matches = all(
                block_hashes[(block, arm, phase)]
                == block_hashes[(block, "baseline", phase)]
                for block in range(1, blocks + 1)
            )
            control_exact[arm][phase] = matches
            require(matches, f"{arm}/{phase}: exact control output drift")

    cross_arm_agreement: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in sorted(expected_arms):
        cross_arm_agreement[arm] = {}
        for phase in ("c1", "c8"):
            by_block = [
                sequence_agreement(
                    sequences[(block, "baseline", phase)],
                    sequences[(block, arm, phase)],
                )
                for block in range(1, blocks + 1)
            ]
            cross_arm_agreement[arm][phase] = {
                "blocks": by_block,
                "exact_requests": sum(item["exact_requests"] for item in by_block),
                "request_count": sum(item["request_count"] for item in by_block),
                "equal_token_positions": sum(item["equal_token_positions"] for item in by_block),
                "token_count": sum(item["token_count"] for item in by_block),
            }
            cell = cross_arm_agreement[arm][phase]
            cell["exact_request_fraction"] = cell["exact_requests"] / cell["request_count"]
            cell["equal_token_position_fraction"] = (
                cell["equal_token_positions"] / cell["token_count"]
            )

    summary: dict[str, Any] = {}
    seed = 20260828
    for arm in sorted(expected_arms):
        summary[arm] = {}
        for phase in ("c1", "c8"):
            summary[arm][phase] = {}
            for metric in ("throughput_tps", "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms"):
                seed += 1
                values = metrics[(arm, phase, metric)]
                require(len(values) == blocks, f"{arm}/{phase}/{metric}: sample count")
                summary[arm][phase][metric] = aggregate(values, seed=seed)

    payload = {
        "schema": "eurosys_e9_ktransformers_formal_verification_v1",
        "status": "ok",
        "mode": config["mode"],
        "process_blocks": blocks,
        "case_count": len(rows),
        "arms": config["arms"],
        "numa_scope": config["numa_scope"],
        "correctness_gate": {
            "status": "ok",
            "same_arm_repeat_stability_required": True,
            "same_arm_repeat_stability": repeat_stability,
            "exact_control_arms": list(exact_control_arms),
            "exact_controls_match_baseline": control_exact,
            "numerical_path_arms": ["uniform", "no_mixed"],
            "numerical_path_policy": (
                "reported token agreement; exact suffix equality is not a gate because "
                "placement/batch formation changes floating-point reduction paths"
            ),
        },
        "exact_outputs_equal_across_all_arms": all(len(values) == 1 for values in hashes.values()),
        "same_arm_output_sha256": {
            arm: {
                phase: next(iter(arm_hashes[(arm, phase)]))
                for phase in ("c1", "c8")
            }
            for arm in sorted(expected_arms)
        },
        "cross_arm_token_agreement_vs_baseline": cross_arm_agreement,
        "samples": samples,
        "aggregate": summary,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
