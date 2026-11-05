#!/usr/bin/env python3
"""Verify the E13 RAG-8192 B1/B5/B6 fixed-tier/adaptive matrix."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any


ARMS = ("fixed_k2", "fixed_k4", "fixed_k6", "roundtrip")
PRESSURES = (
    ("low", 1, "k2"),
    ("medium", 5, "k4"),
    ("high", 6, "k6"),
)
TARGETS = {"k2": 1364, "k4": 2730, "k6": 4095}
PLAN_CLASSES = {"k2": "cold70", "k4": "cold80", "k6": "cold89"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("empty distribution")
    return float(statistics.median(values))


def segment_label(arm: str, pressure: str, batch: int, target_tier: str) -> str:
    if arm == "roundtrip":
        return f"adaptive_{target_tier}_rag_b{batch}"
    return f"{arm}_rag_b{batch}"


def summarize(segment: dict[str, Any], pressure: str) -> dict[str, Any]:
    role = f"measured_pressure_{pressure}"
    measurements = [
        item for item in segment.get("measurements", []) if item.get("role") == role
    ]
    if len(measurements) != 2:
        raise ValueError(f"{segment.get('label')}: expected two {role} samples")

    def timing(field: str) -> float:
        return median(
            [float(item["request_timings"][field]["median"]) for item in measurements]
        )

    return {
        "samples": len(measurements),
        "output_tok_s": median(
            [float(item["output_token_throughput_per_second"]) for item in measurements]
        ),
        "request_s": median(
            [float(item["request_throughput_per_second"]) for item in measurements]
        ),
        "ttft_ms": timing("request_ttft_ms"),
        "tpot_ms": timing("request_tpot_ms"),
        "request_e2e_ms": timing("request_latency_ms"),
        "prefill_ms": median([float(item["prefill_ms"]) for item in measurements]),
        "batch_wall_ms": median([float(item["wall_ms"]) for item in measurements]),
        "steady_decode_ms": median(
            [float(item["steady_decode_last16_ms"]["median"]) for item in measurements]
        ),
        "max_running": max(
            int(item["max_scheduler_running_requests"]) for item in measurements
        ),
        "max_waiting": max(
            int(item["max_scheduler_waiting_requests"]) for item in measurements
        ),
        "output_hashes": sorted(
            {str(item["output_token_ids_sha256"]) for item in measurements}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    checks: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    for arm in ARMS:
        case_dir = root / arm
        result_path = case_dir / "result.json"
        returncode_path = case_dir / "returncode.txt"
        present = result_path.is_file() and returncode_path.is_file()
        check(f"{arm}_artifacts", present, str(case_dir))
        if not present:
            continue
        result = load_json(result_path)
        results[arm] = result
        returncode = int(returncode_path.read_text(encoding="utf-8").strip())
        check(f"{arm}_returncode", returncode == 0, returncode)
        check(f"{arm}_status", result.get("status") == "ok", result.get("status"))
        configuration = result.get("configuration") or {}
        check(
            f"{arm}_protocol",
            result.get("schema") == "eurosys_e13_k2_k6_roundtrip_v1"
            and result.get("arm") == arm
            and configuration.get("measurement_scope") == "rag_pressure_matrix"
            and configuration.get("full_separate_graph") is False
            and configuration.get("switch_transaction_excluded") is True
            and configuration.get("post_switch_warmup_excluded") is True,
            {"arm": result.get("arm"), "configuration": configuration},
        )
        expected_labels = tuple(
            segment_label(arm, pressure, batch, tier)
            for pressure, batch, tier in PRESSURES
        )
        segments = result.get("segments") or []
        observed_labels = tuple(str(item.get("label")) for item in segments)
        check(f"{arm}_segments", observed_labels == expected_labels, observed_labels)
        for segment, (pressure, batch, target_tier) in zip(segments, PRESSURES):
            fixed_tier = arm.removeprefix("fixed_") if arm != "roundtrip" else target_tier
            before = segment.get("audit_before") or {}
            after = segment.get("audit_after") or {}
            measurements = segment.get("measurements") or []
            warmups = segment.get("warmups_excluded") or []
            check(
                f"{arm}_{pressure}_tier",
                segment.get("tier") == fixed_tier
                and int(before.get("kv", {}).get("current_target_blocks", -1))
                == TARGETS[fixed_tier]
                and int(after.get("kv", {}).get("current_target_blocks", -1))
                == TARGETS[fixed_tier]
                and before.get("plan", {}).get("active_plan_class")
                == PLAN_CLASSES[fixed_tier]
                and after.get("plan", {}).get("active_plan_class")
                == PLAN_CLASSES[fixed_tier],
                {"tier": fixed_tier, "before": before, "after": after},
            )
            check(
                f"{arm}_{pressure}_measurements",
                len(warmups) == 1
                and len(measurements) == 2
                and all(
                    item.get("workload") == "rag"
                    and int(item.get("batch_size", -1)) == batch
                    and int(item.get("prompt_tokens_per_request", -1)) == 8192
                    and int(item.get("output_tokens_per_request", -1)) == 16
                    and item.get("role") == f"measured_pressure_{pressure}"
                    and int(item.get("prefill_step_count", 0)) > 0
                    and int(item.get("steady_decode_last16_ms", {}).get("count", 0))
                    > 0
                    for item in measurements
                ),
                {"warmups": len(warmups), "measurements": len(measurements)},
            )
        gpu_after_path = case_dir / "gpu.after.json"
        gpu_after = load_json(gpu_after_path) if gpu_after_path.is_file() else {}
        check(f"{arm}_gpu_clean", gpu_after.get("gpu_clean") is True, gpu_after)
        kernel_path = case_dir / "kernel.since.log"
        kernel_text = (
            kernel_path.read_text(encoding="utf-8", errors="replace").lower()
            if kernel_path.is_file()
            else ""
        )
        bad_kernel = any(
            marker in kernel_text
            for marker in ("xid", "mmu fault", "pte fault", "pde fault", "illegal address")
        )
        check(
            f"{arm}_kernel_clean",
            not bad_kernel,
            "fault marker absent" if not bad_kernel else "fault marker present",
        )

    matrix: dict[str, dict[str, Any]] = {}
    if len(results) == len(ARMS):
        for pressure, batch, target_tier in PRESSURES:
            matrix[pressure] = {}
            cross_arm_hashes: set[str] = set()
            for arm in ARMS:
                label = segment_label(arm, pressure, batch, target_tier)
                segment = next(
                    item for item in results[arm]["segments"] if item["label"] == label
                )
                row = summarize(segment, pressure)
                matrix[pressure][arm] = row
                cross_arm_hashes.update(row["output_hashes"])
            check(
                f"{pressure}_output_hash_equal",
                len(cross_arm_hashes) == 1,
                sorted(cross_arm_hashes),
            )

    transition_path: list[tuple[int, int]] = []
    if "roundtrip" in results:
        transition_path = [
            (int(item["old_target_blocks"]), int(item["new_target_blocks"]))
            for item in results["roundtrip"].get("transitions") or []
            if int(item["old_target_blocks"]) != int(item["new_target_blocks"])
        ]
        check(
            "adaptive_adjacent_transition_path",
            transition_path == [(1364, 2730), (2730, 4095)],
            transition_path,
        )

    trends: dict[str, Any] = {}
    if matrix:
        low = matrix["low"]
        middle = matrix["medium"]
        high = matrix["high"]
        trends = {
            "low_fixed_order_k2_gt_k4_gt_k6": (
                low["fixed_k2"]["output_tok_s"]
                > low["fixed_k4"]["output_tok_s"]
                > low["fixed_k6"]["output_tok_s"]
            ),
            "medium_fixed_k4_best": middle["fixed_k4"]["output_tok_s"]
            > max(
                middle["fixed_k2"]["output_tok_s"],
                middle["fixed_k6"]["output_tok_s"],
            ),
            "high_fixed_k6_best": high["fixed_k6"]["output_tok_s"]
            > max(
                high["fixed_k2"]["output_tok_s"],
                high["fixed_k4"]["output_tok_s"],
            ),
            "adaptive_over_target_fixed": {
                pressure: matrix[pressure]["roundtrip"]["output_tok_s"]
                / matrix[pressure][f"fixed_{tier}"]["output_tok_s"]
                for pressure, _, tier in PRESSURES
            },
        }

    execution_ok = len(results) == len(ARMS) and all(item["ok"] for item in checks)
    summary = {
        "schema": "eurosys_e13_rag_pressure_matrix_verification_v1",
        "status": "ok" if execution_ok else "error",
        "protocol": {
            "workload": "RAG-8192 -> 16",
            "batches": [batch for _, batch, _ in PRESSURES],
            "execution": "eager Prefill + remote Decode",
            "cold_processes_per_arm": 1,
            "warmups_excluded_per_cell": 1,
            "measured_samples_per_cell": 2,
            "switch_transactions_excluded": True,
        },
        "matrix": matrix,
        "transition_path": transition_path,
        "trends": trends,
        "checks": checks,
    }
    output = args.output or (root / "verification.json")
    atomic_text(output, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    markdown = [
        "# E13 RAG-8192 pressure matrix",
        "",
        "One target-tier warmup and all adaptive switch transactions are excluded.",
        "",
        "| Pressure | B | Arm | Output tok/s | TTFT ms | TPOT ms | Request E2E ms | Prefill ms | Batch wall ms | Steady Decode ms | Max running |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    display_arms = (
        ("fixed_k2", "Fixed-K2"),
        ("fixed_k4", "Fixed-K4"),
        ("fixed_k6", "Fixed-K6"),
        ("roundtrip", "Adaptive"),
    )
    for pressure, batch, _ in PRESSURES:
        for arm, display in display_arms:
            row = matrix.get(pressure, {}).get(arm)
            if not row:
                continue
            markdown.append(
                f"| {pressure} | {batch} | {display} | {row['output_tok_s']:.4f} | "
                f"{row['ttft_ms']:.2f} | {row['tpot_ms']:.2f} | "
                f"{row['request_e2e_ms']:.2f} | {row['prefill_ms']:.2f} | "
                f"{row['batch_wall_ms']:.2f} | {row['steady_decode_ms']:.2f} | "
                f"{row['max_running']}/{batch} |"
            )
    markdown.extend(
        [
            "",
            f"Verifier status: `{summary['status']}`.",
            f"Observed trends: `{json.dumps(trends, ensure_ascii=False)}`.",
            "This is one cold process per arm with two in-process measurements, not an independent-repeat confidence interval.",
            "",
        ]
    )
    atomic_text(root / "summary.md", "\n".join(markdown))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if execution_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
