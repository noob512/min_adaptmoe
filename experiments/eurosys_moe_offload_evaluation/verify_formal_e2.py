#!/usr/bin/env python3
"""Verify E2 completeness, correctness, and equal-HBM eligibility."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SYSTEMS = ("ours", "ktransformers", "moe_infinity")
TIERS = ("0g", "2g", "4g", "6g", "8g")
WORKLOADS = ("short_dialogue", "ordinary_chat", "rag_summary", "long_generation")
CONCURRENCIES = (1, 4, 8, 16)
REPEATS = (1, 2, 3, 4, 5)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite verifier output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_nvml_peak_mib(path: Path) -> int | None:
    if not path.is_file():
        return None
    values: list[int] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 5:
                continue
            raw = row[4].strip()
            try:
                values.append(int(raw))
            except ValueError:
                continue
    return max(values) if values else None


def candidate_attempts(repeat_root: Path) -> list[Path]:
    if not repeat_root.is_dir():
        return []
    return sorted(
        (path for path in repeat_root.glob("attempt_*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
        reverse=True,
    )


def request_identity(request: dict[str, Any]) -> tuple[Any, Any]:
    return request.get("fixture_index"), request.get("prompt_token_ids_sha256")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("e2_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol-id", required=True)
    args = parser.parse_args()

    failures: list[str] = []
    records: list[dict[str, Any]] = []
    output_by_identity: dict[
        tuple[str, str, int, Any, Any], set[str]
    ] = defaultdict(set)
    peak_by_block: dict[tuple[str, str, int, str], int] = {}

    for system in SYSTEMS:
        for tier in TIERS:
            for repeat in REPEATS:
                repeat_root = (
                    args.e2_root
                    / system
                    / "deepseek_v2_lite_bf16"
                    / f"budget_{tier}"
                    / f"repeat_{repeat:02d}"
                )
                attempts = candidate_attempts(repeat_root)
                if not attempts:
                    failures.append(
                        f"missing process block: {system}/{tier}/repeat_{repeat:02d}"
                    )
                    continue
                for workload in WORKLOADS:
                    for concurrency in CONCURRENCIES:
                        cell = f"{workload}_c{concurrency}"
                        selected: Path | None = None
                        for attempt in attempts:
                            verification_path = attempt / cell / "verification.json"
                            if not verification_path.is_file():
                                continue
                            verification = load_json(verification_path)
                            if (
                                verification.get("status") == "ok"
                                and verification.get("protocol_id") == args.protocol_id
                            ):
                                selected = attempt
                                break
                        if selected is None:
                            failures.append(
                                f"no verified cell: {system}/{tier}/repeat_{repeat:02d}/{cell}"
                            )
                            continue
                        client_path = selected / cell / "client.json"
                        if not client_path.is_file():
                            failures.append(f"missing client: {client_path}")
                            continue
                        client = load_json(client_path)
                        phases = client.get("phases")
                        if not isinstance(phases, list) or len(phases) != 1:
                            failures.append(f"invalid phase count: {client_path}")
                            continue
                        phase = phases[0]
                        requests = phase.get("requests")
                        if not isinstance(requests, list) or len(requests) != concurrency:
                            failures.append(f"invalid request count: {client_path}")
                            continue
                        request_hashes: list[str] = []
                        for request in requests:
                            if not isinstance(request, dict):
                                failures.append(f"malformed request: {client_path}")
                                continue
                            output_hash = request.get("output_token_ids_sha256")
                            if not isinstance(output_hash, str):
                                failures.append(f"missing output hash: {client_path}")
                                continue
                            request_hashes.append(output_hash)
                            fixture_index, prompt_hash = request_identity(request)
                            output_by_identity[
                                (tier, workload, concurrency, fixture_index, prompt_hash)
                            ].add(output_hash)
                        block_key = (system, tier, repeat, selected.name)
                        peak = peak_by_block.get(block_key)
                        if peak is None:
                            peak = parse_nvml_peak_mib(selected / "nvml.csv")
                            if peak is not None:
                                peak_by_block[block_key] = peak
                        records.append(
                            {
                                "system": system,
                                "tier": tier,
                                "repeat": repeat,
                                "attempt": selected.name,
                                "workload": workload,
                                "concurrency": concurrency,
                                "client": str(client_path.resolve()),
                                "peak_hbm_mib": peak,
                                "request_output_hashes": request_hashes,
                                "mean_ttft_ms": phase.get("ttft_ms", {}).get("mean"),
                                "mean_tpot_ms": phase.get("itl_ms", {}).get("mean"),
                                "mean_e2e_ms": phase.get("e2e_ms", {}).get("mean"),
                                "output_tokens_per_second": phase.get(
                                    "output_tokens_per_second"
                                ),
                                "goodput_requests_per_second": phase.get(
                                    "goodput_requests_per_second"
                                ),
                            }
                        )

    numerical_mismatches = [
        {
            "tier": key[0],
            "workload": key[1],
            "concurrency": key[2],
            "fixture_index": key[3],
            "prompt_token_ids_sha256": key[4],
            "output_hashes": sorted(hashes),
        }
        for key, hashes in output_by_identity.items()
        if len(hashes) != 1
    ]
    if numerical_mismatches:
        failures.append(
            f"{len(numerical_mismatches)} cross-framework/repeat output identities differ"
        )

    budget_alignment: dict[str, Any] = {}
    for tier in TIERS:
        medians: dict[str, float | None] = {}
        for system in SYSTEMS:
            peaks = [
                value
                for (record_system, record_tier, _repeat, _attempt), value in peak_by_block.items()
                if record_system == system and record_tier == tier
            ]
            medians[system] = statistics.median(peaks) if peaks else None
        numeric = [value for value in medians.values() if value is not None]
        target = max(numeric) if numeric else None
        tolerance = max(0.03 * target, 256.0) if target is not None else None
        aligned = (
            len(numeric) == len(SYSTEMS)
            and target is not None
            and tolerance is not None
            and all(abs(value - target) <= tolerance for value in numeric)
        )
        budget_alignment[tier] = {
            "median_peak_hbm_mib": medians,
            "comparison_target_mib": target,
            "tolerance_mib": tolerance,
            "eligible_for_equal_hbm_speedup": aligned,
        }

    expected_cells = len(SYSTEMS) * len(TIERS) * len(REPEATS) * len(WORKLOADS) * len(CONCURRENCIES)
    result = {
        "schema_version": 1,
        "status": "ok" if not failures else "error",
        "experiment": "E2_main_equal_HBM_matrix",
        "scope": "formal_performance",
        "protocol_id": args.protocol_id,
        "checks": {
            "expected_verified_cells": expected_cells,
            "observed_verified_cells": len(records),
            "cross_framework_exact_token_match": not numerical_mismatches,
            "budget_mismatch_disposition": "exclude_from_speedup_but_retain_artifact",
        },
        "budget_alignment": budget_alignment,
        "numerical_mismatches": numerical_mismatches,
        "records": records,
        "failures": failures,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
