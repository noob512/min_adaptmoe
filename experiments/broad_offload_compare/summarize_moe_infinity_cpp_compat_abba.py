#!/usr/bin/env python3
"""Validate the MoE-Infinity all-chunk vs minimal-compatibility A/B/B/A."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any


EXPECTED_ORDER = ("all-chunk", "minimal", "minimal", "all-chunk")
EXPECTED_COMMIT = "ba5651897a80d9c9b7a1500cef2c68adaa63db0f"
EXPECTED_PATCH_SHA256 = (
    "6dab7b086ccc3e53cec83113799683cf8f5825cda7f50d1ba3a9263d3e4e955f"
)
EXPECTED_FIXTURE_SHA256 = (
    "14ce7cb4960109fc3744ecd10fbcbb7a8c8fca41a82ee89b7de5a118fd4eaffe"
)
EXPECTED_TOKEN_SHA256 = (
    "ffc4ffab5ed1b727350f682f89b64eab322019f0ce8d81013ae05c700a55a69a"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "values": values,
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--all-chunk-sha256", required=True)
    parser.add_argument("--minimal-sha256", required=True)
    args = parser.parse_args()

    failures: list[str] = []
    for name, value in (
        ("all-chunk", args.all_chunk_sha256),
        ("minimal", args.minimal_sha256),
    ):
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            parser.error(f"{name} SHA-256 must be 64 lowercase hex characters")
    if args.all_chunk_sha256 == args.minimal_sha256:
        parser.error("the two extension hashes must differ")

    plan_path = args.run_root / "plan.json"
    if not plan_path.is_file():
        failures.append("missing plan.json")
        plan: dict[str, Any] = {}
    else:
        plan = load_json(plan_path)
    schedule = plan.get("schedule", [])
    observed_order = tuple(
        item.get("arm") for item in schedule if isinstance(item, dict)
    )
    if observed_order != EXPECTED_ORDER:
        failures.append(f"schedule is {observed_order!r}, expected {EXPECTED_ORDER!r}")
    if plan.get("moe_infinity_commit") != EXPECTED_COMMIT:
        failures.append("plan commit mismatch")
    if plan.get("moe_infinity_patch_sha256") != EXPECTED_PATCH_SHA256:
        failures.append("plan compatibility patch mismatch")
    if plan.get("fixture_sha256") != EXPECTED_FIXTURE_SHA256:
        failures.append("plan calibration fixture mismatch")

    arm_cases: dict[str, list[dict[str, Any]]] = {
        "all-chunk": [],
        "minimal": [],
    }
    output_hashes: set[str] = set()
    for expected_slot, expected_arm in enumerate(EXPECTED_ORDER, 1):
        label = expected_arm.replace("-", "_")
        case_root = args.run_root / "cases" / f"slot{expected_slot}_{label}"
        result_path = case_root / "result.json"
        returncode_path = case_root / "returncode.txt"
        active_sha_path = case_root / "active_store.sha256"
        for path in (
            result_path,
            returncode_path,
            active_sha_path,
            case_root / "stdout.log",
            case_root / "stderr.log",
            case_root / "nvml.csv",
            case_root / "nvml.err",
        ):
            if not path.is_file():
                failures.append(f"slot {expected_slot}: missing {path.name}")
        if not result_path.is_file():
            continue
        result = load_json(result_path)
        if returncode_path.is_file() and returncode_path.read_text().strip() != "0":
            failures.append(f"slot {expected_slot}: nonzero return code")
        if result.get("status") != "ok":
            failures.append(f"slot {expected_slot}: result status={result.get('status')!r}")
        if result.get("qualification", {}).get("status") != "ok":
            failures.append(f"slot {expected_slot}: qualification gate failed")

        protocol = result.get("protocol", {})
        expected_protocol = {
            "batch_size": 1,
            "prompt_tokens": 512,
            "output_tokens": 32,
            "warmup_requests": 2,
            "measured_repetitions": 3,
            "expert_compute_threads": 1,
            "greedy": True,
            "use_native_engine": False,
        }
        for key, expected in expected_protocol.items():
            if protocol.get(key) != expected:
                failures.append(
                    f"slot {expected_slot}: protocol {key}={protocol.get(key)!r}, "
                    f"expected {expected!r}"
                )
        prompt = result.get("prompt", {})
        if prompt.get("source_file_sha256") != EXPECTED_FIXTURE_SHA256:
            failures.append(f"slot {expected_slot}: fixture file hash mismatch")
        if prompt.get("prompt_token_ids_sha256") != EXPECTED_TOKEN_SHA256:
            failures.append(f"slot {expected_slot}: token-ID hash mismatch")

        expected_store = (
            args.all_chunk_sha256
            if expected_arm == "all-chunk"
            else args.minimal_sha256
        )
        runtime_environment = result.get("environment", {})
        if runtime_environment.get("store_extension_sha256") != expected_store:
            failures.append(f"slot {expected_slot}: runtime extension hash mismatch")
        if runtime_environment.get("moe_infinity_commit") != EXPECTED_COMMIT:
            failures.append(f"slot {expected_slot}: runtime commit mismatch")
        if (
            runtime_environment.get("moe_infinity_diff_sha256")
            != EXPECTED_PATCH_SHA256
        ):
            failures.append(f"slot {expected_slot}: runtime patch mismatch")
        if active_sha_path.is_file():
            fields = active_sha_path.read_text().split()
            if not fields or fields[0] != expected_store:
                failures.append(f"slot {expected_slot}: active-store sidecar mismatch")

        case_output_hashes = result.get("summary", {}).get("output_hashes", [])
        if not isinstance(case_output_hashes, list) or len(case_output_hashes) != 1:
            failures.append(f"slot {expected_slot}: output hash is not stable")
            case_output_hash = None
        else:
            case_output_hash = str(case_output_hashes[0])
            output_hashes.add(case_output_hash)
        trials = result.get("trials", [])
        if not isinstance(trials, list) or len(trials) != 3:
            failures.append(f"slot {expected_slot}: expected three measured trials")
            continue
        tpot_values = [trial.get("decode_tpot_seconds") for trial in trials]
        ttft_values = [trial.get("ttft_seconds") for trial in trials]
        if not all(finite_positive(value) for value in tpot_values + ttft_values):
            failures.append(f"slot {expected_slot}: invalid timing value")
            continue
        arm_cases[expected_arm].append(
            {
                "slot": expected_slot,
                "output_hash": case_output_hash,
                "median_decode_tpot_seconds": statistics.median(tpot_values),
                "median_ttft_seconds": statistics.median(ttft_values),
                "measured_decode_tpot_seconds": tpot_values,
                "measured_ttft_seconds": ttft_values,
            }
        )

    if len(output_hashes) != 1:
        failures.append(f"cross-arm output hashes differ: {sorted(output_hashes)}")
    for arm in EXPECTED_ORDER[:2]:
        if len(arm_cases[arm]) != 2:
            failures.append(f"{arm}: expected two valid cold processes")

    metrics: dict[str, Any] = {}
    if not failures:
        for arm, cases in arm_cases.items():
            metrics[arm] = {
                "cold_process_decode_tpot_seconds": summary(
                    [case["median_decode_tpot_seconds"] for case in cases]
                ),
                "cold_process_ttft_seconds": summary(
                    [case["median_ttft_seconds"] for case in cases]
                ),
            }
        old_tpot = metrics["all-chunk"]["cold_process_decode_tpot_seconds"]["mean"]
        new_tpot = metrics["minimal"]["cold_process_decode_tpot_seconds"]["mean"]
        metrics["minimal_tpot_speedup_over_all_chunk"] = old_tpot / new_tpot
        old_ttft = metrics["all-chunk"]["cold_process_ttft_seconds"]["mean"]
        new_ttft = metrics["minimal"]["cold_process_ttft_seconds"]["mean"]
        metrics["minimal_ttft_speedup_over_all_chunk"] = old_ttft / new_ttft

    snapshot = {
        "schema_version": 1,
        "experiment": "moe_infinity_cpp_compatibility_abba",
        "status": "ok" if not failures else "failed",
        "publication_eligible": False,
        "interpretation": (
            "calibration-only implementation control; two cold processes per arm, "
            "not a cross-system result and not a paper EAM reproduction"
        ),
        "schedule": list(EXPECTED_ORDER),
        "all_chunk_store_sha256": args.all_chunk_sha256,
        "minimal_store_sha256": args.minimal_sha256,
        "output_hashes": sorted(output_hashes),
        "cases": arm_cases,
        "metrics": metrics,
        "failures": failures,
    }
    atomic_json(args.run_root / "cpp_compat_abba_snapshot.json", snapshot)

    lines = [
        "# MoE-Infinity C++ compatibility A/B/B/A",
        "",
        f"Status: **{snapshot['status']}**.",
        "",
        "This is a calibration-only implementation control with two independent "
        "cold processes per arm; it is not publication-eligible.",
        "",
    ]
    if not failures:
        lines.extend(
            [
                "| arm | cold processes | mean process-median TPOT (ms) | mean process-median TTFT (ms) |",
                "|---|---:|---:|---:|",
            ]
        )
        for arm in ("all-chunk", "minimal"):
            arm_metric = metrics[arm]
            lines.append(
                f"| {arm} | 2 | "
                f"{1000 * arm_metric['cold_process_decode_tpot_seconds']['mean']:.3f} | "
                f"{1000 * arm_metric['cold_process_ttft_seconds']['mean']:.3f} |"
            )
        lines.extend(
            [
                "",
                "- minimal TPOT speedup over all-chunk: "
                f"`{metrics['minimal_tpot_speedup_over_all_chunk']:.6f}x`;",
                "- minimal TTFT speedup over all-chunk: "
                f"`{metrics['minimal_ttft_speedup_over_all_chunk']:.6f}x`;",
                f"- cross-arm output hash: `{next(iter(output_hashes))}`.",
            ]
        )
    else:
        lines.extend(["Failures:", ""] + [f"- {item}" for item in failures])
    atomic_write(args.run_root / "CPP_COMPAT_ABBA_RESULTS.md", "\n".join(lines) + "\n")
    if not failures:
        atomic_json(
            args.run_root / "COMPLETED.json",
            {
                "schema_version": 1,
                "status": "complete",
                "snapshot_sha256": sha256_file(
                    args.run_root / "cpp_compat_abba_snapshot.json"
                ),
                "report_sha256": sha256_file(
                    args.run_root / "CPP_COMPAT_ABBA_RESULTS.md"
                ),
            },
        )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
