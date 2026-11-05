#!/usr/bin/env python3
"""Summarize the four KT qualification cases into JSON and Markdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_CASES = (
    ("exact", 12),
    ("exact", 14),
    ("vendor-best", 12),
    ("vendor-best", 14),
)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    stats_manifest_path = args.run_root / "calibration" / "frequency_stats_manifest.json"
    stats = load_json(stats_manifest_path)
    cases: list[dict[str, Any]] = []
    failures: list[str] = []
    fixture_hashes: set[str] = set()

    for mode, threads in EXPECTED_CASES:
        name = f"{mode}_t{threads}"
        root = args.run_root / "cases" / name
        benchmark_path = root / "benchmark.json"
        verification_path = root / "runtime_verification.json"
        metadata_path = root / "case_metadata.json"
        missing = [
            str(path)
            for path in (benchmark_path, verification_path, metadata_path)
            if not path.is_file()
        ]
        if missing:
            failures.append(f"{name}: missing {missing}")
            continue
        benchmark = load_json(benchmark_path)
        verification = load_json(verification_path)
        metadata = load_json(metadata_path)
        if benchmark.get("status") != "ok":
            failures.append(f"{name}: benchmark status={benchmark.get('status')!r}")
        if verification.get("status") != "ok":
            failures.append(f"{name}: runtime verification failed")
        fixture_hashes.add(benchmark["fixture"]["sha256"])
        phases: dict[str, Any] = {}
        for phase in benchmark.get("phases", []):
            phase_name = phase["phase"]
            if not phase.get("all_requests_produced_fixed_output"):
                failures.append(f"{name}/{phase_name}: incomplete fixed output")
            phases[phase_name] = {
                "concurrency": phase["concurrency"],
                "output_tokens_per_second": phase["output_tokens_per_second"],
                "mean_ttft_ms": phase["mean_ttft_ms"],
                "p99_ttft_ms": phase["p99_ttft_ms"],
                "mean_tpot_ms": phase["mean_tpot_ms"],
                "p99_tpot_ms": phase["p99_tpot_ms"],
                "output_fingerprints_sha256": phase[
                    "output_fingerprints_sha256"
                ],
            }
        if set(phases) != {"c1", "c8"}:
            failures.append(f"{name}: phases={sorted(phases)}, expected c1/c8")
        cases.append(
            {
                "name": name,
                "mode": mode,
                "threads": threads,
                "metadata": metadata,
                "phases": phases,
                "runtime_verification": verification,
            }
        )

    if len(fixture_hashes) > 1:
        failures.append("qualification cases used different evaluation fixtures")

    output_hashes: dict[str, dict[str, str]] = {}
    for case in cases:
        output_hashes[case["name"]] = {
            phase: values["output_fingerprints_sha256"]
            for phase, values in case["phases"].items()
        }
    exact_hash_equal_across_threads = all(
        output_hashes.get("exact_t12", {}).get(phase)
        == output_hashes.get("exact_t14", {}).get(phase)
        for phase in ("c1", "c8")
    )
    if len(cases) == len(EXPECTED_CASES) and not exact_hash_equal_across_threads:
        failures.append("exact profile output hashes differ between 12 and 14 threads")

    snapshot = {
        "schema_version": 1,
        "experiment": "ktransformers_documentation_qualification",
        "status": "ok" if not failures else "error",
        "run_root": str(args.run_root.resolve()),
        "frequency_stats_manifest": stats,
        "evaluation_fixture_sha256": next(iter(fixture_hashes), None),
        "exact_output_hashes_equal_across_threads": exact_hash_equal_across_threads,
        "cases": cases,
        "failures": failures,
    }
    atomic_text(
        args.output_json,
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    rows = []
    for case in cases:
        for phase in ("c1", "c8"):
            values = case["phases"].get(phase)
            if values is None:
                continue
            rows.append(
                "| {mode} | {threads} | {phase} | {throughput:.3f} | "
                "{ttft:.3f} | {tpot:.3f} |".format(
                    mode=case["mode"],
                    threads=case["threads"],
                    phase=phase,
                    throughput=values["output_tokens_per_second"],
                    ttft=values["mean_ttft_ms"],
                    tpot=values["mean_tpot_ms"],
                )
            )
    markdown = "\n".join(
        [
            "# KTransformers qualification result",
            "",
            f"Status: `{snapshot['status']}`",
            "",
            "The `exact` profile preserves all routed expert contributions. The "
            "`vendor-best` profile enables expert deferral and dynamic/layerwise "
            "optimizations and therefore requires a separate quality check.",
            "",
            "| Profile | CPU threads | Phase | Output tok/s | Mean TTFT ms | Mean TPOT ms |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Integrity",
            "",
            f"- Real recorder mask SHA-256: `{stats['placement']['gpu_mask_uint8_sha256']}`",
            f"- Calibration/evaluation disjoint: `{stats['fixtures_disjoint_by_source_index']}`",
            f"- Exact hashes equal across thread counts: `{exact_hash_equal_across_threads}`",
            f"- Failures: `{failures}`",
            "",
        ]
    )
    atomic_text(args.output_md, markdown)
    print(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
