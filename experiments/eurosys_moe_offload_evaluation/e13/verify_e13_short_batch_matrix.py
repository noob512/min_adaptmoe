#!/usr/bin/env python3
"""Verify E13 Fixed-K2/K6 Short B1/B16/B64 in-process batch sweeps."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any, Callable


ARMS = ("fixed_k2", "fixed_k6")
BATCHES = (1, 16, 64)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("empty metric distribution")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    checks: list[dict[str, Any]] = []
    cases: dict[str, dict[str, Any]] = {}

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
        cases[arm] = result
        returncode = int(returncode_path.read_text(encoding="utf-8").strip())
        check(f"{arm}_returncode", returncode == 0, returncode)
        check(f"{arm}_status", result.get("status") == "ok", result.get("status"))
        configuration = result.get("configuration") or {}
        check(
            f"{arm}_scope",
            result.get("arm") == arm
            and configuration.get("measurement_scope") == "short_matrix"
            and int(configuration.get("effective_max_num_seqs", -1)) == 64
            and tuple(configuration.get("short_matrix_batches") or ()) == BATCHES,
            {"arm": result.get("arm"), "configuration": configuration},
        )
        segments = result.get("segments") or []
        expected_labels = [f"{arm}_short_b{batch}" for batch in BATCHES]
        check(
            f"{arm}_segment_order",
            [item.get("label") for item in segments] == expected_labels,
            [item.get("label") for item in segments],
        )
        for batch, segment in zip(BATCHES, segments):
            measurements = segment.get("measurements") or []
            check(
                f"{arm}_b{batch}_measurement_contract",
                len(measurements) == 3
                and all(
                    item.get("role") == f"measured_short_b{batch}"
                    and int(item.get("batch_size", -1)) == batch
                    and int(item.get("prompt_tokens_per_request", -1)) == 128
                    and int(item.get("output_tokens_per_request", -1)) == 32
                    and int(item.get("prefill_step_count", -1)) == 1
                    and int(item.get("decode_step_ms", {}).get("count", -1)) == 31
                    and int(item.get("steady_decode_last16_ms", {}).get("count", -1)) == 16
                    and int(item.get("schedule_step_count", -1)) == 32
                    and int(item.get("max_scheduler_running_requests", -1)) == batch
                    and int(item.get("max_scheduler_waiting_requests", -1)) == 0
                    for item in measurements
                ),
                measurements,
            )
            before = segment.get("audit_before") or {}
            after = segment.get("audit_after") or {}
            check(
                f"{arm}_b{batch}_tier_stable",
                before.get("kv", {}).get("current_target_blocks")
                == after.get("kv", {}).get("current_target_blocks")
                and before.get("plan", {}).get("active_plan_class")
                == after.get("plan", {}).get("active_plan_class"),
                {"before": before, "after": after},
            )
        gpu_path = case_dir / "gpu.after.json"
        gpu = load_json(gpu_path) if gpu_path.is_file() else {}
        check(f"{arm}_gpu_clean", gpu.get("gpu_clean") is True, gpu)
        kernel_path = case_dir / "kernel.since.log"
        kernel_text = (
            kernel_path.read_text(encoding="utf-8", errors="replace").lower()
            if kernel_path.is_file()
            else ""
        )
        bad_kernel = any(
            marker in kernel_text
            for marker in ("xid", "mmu fault", "gpu fault", "illegal address")
        )
        check(
            f"{arm}_kernel_clean",
            not bad_kernel,
            "fault marker absent" if not bad_kernel else "fault marker present",
        )

    before_path = root / "module.before.json"
    after_path = root / "module.after.json"
    if before_path.is_file() and after_path.is_file():
        before = load_json(before_path)
        after = load_json(after_path)
        check(
            "stock_uvm_restored",
            before.get("module_sha256") == after.get("module_sha256")
            and after.get("gpu_clean") is True,
            {"before": before, "after": after},
        )
    else:
        check("stock_uvm_restored", False, "module snapshots absent")

    getters: dict[str, Callable[[dict[str, Any]], float]] = {
        "request_per_second": lambda item: float(item["request_throughput_per_second"]),
        "output_token_per_second": lambda item: float(item["output_token_throughput_per_second"]),
        "batch_wall_ms": lambda item: float(item["wall_ms"]),
        "prefill_ms": lambda item: float(item["prefill_ms"]),
        "decode_step_ms": lambda item: float(item["steady_decode_last16_ms"]["median"]),
        "ttft_ms": lambda item: float(item["request_timings"]["request_ttft_ms"]["median"]),
        "tpot_ms": lambda item: float(item["request_timings"]["request_tpot_ms"]["median"]),
        "e2e_ms": lambda item: float(item["request_timings"]["request_latency_ms"]["median"]),
    }
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    hashes: dict[str, dict[str, list[str]]] = {}
    for arm, result in cases.items():
        summaries[arm] = {}
        hashes[arm] = {}
        segments_by_batch = {
            int(segment["measurements"][0]["batch_size"]): segment
            for segment in result.get("segments") or []
            if segment.get("measurements")
        }
        for batch in BATCHES:
            segment = segments_by_batch.get(batch)
            if segment is None:
                continue
            measurements = segment["measurements"]
            summaries[arm][str(batch)] = {
                name: distribution([getter(item) for item in measurements])
                for name, getter in getters.items()
            }
            hashes[arm][str(batch)] = [
                str(item["output_token_ids_sha256"]) for item in measurements
            ]

    comparisons: list[dict[str, Any]] = []
    for batch in BATCHES:
        key = str(batch)
        if not all(key in summaries.get(arm, {}) for arm in ARMS):
            continue
        k2 = summaries["fixed_k2"][key]
        k6 = summaries["fixed_k6"][key]
        row: dict[str, Any] = {"batch_size": batch}
        for metric in getters:
            left = float(k2[metric]["median"])
            right = float(k6[metric]["median"])
            row[f"fixed_k2_{metric}"] = left
            row[f"fixed_k6_{metric}"] = right
            row[f"k2_over_k6_{metric}"] = left / right
        hash_equal = (
            hashes["fixed_k2"][key] == hashes["fixed_k6"][key]
            and len(set(hashes["fixed_k2"][key])) == 1
        )
        row["output_hash_equal"] = hash_equal
        check(f"b{batch}_cross_arm_output_hash_equal", hash_equal, hashes)
        comparisons.append(row)

    execution_ok = len(cases) == len(ARMS) and all(item["ok"] for item in checks)
    summary = {
        "schema": "eurosys_e13_short_batch_matrix_verification_v1",
        "status": "ok" if execution_ok else "error",
        "single_round": True,
        "arm_order": list(ARMS),
        "batch_order": list(BATCHES),
        "independent_cold_processes": 2,
        "samples_per_arm_batch": 3,
        "summaries": summaries,
        "comparisons": comparisons,
        "checks": checks,
    }
    output = args.output or (root / "verification.json")
    atomic_text(output, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    markdown = [
        "# E13 Fixed-K2/K6 Short B1/B16/B64 single-round result",
        "",
        "Each arm is one cold process; each cell has one excluded warmup and three in-process samples.",
        "",
        "| B | Arm | req/s | TTFT ms | TPOT ms | E2E ms | Prefill ms | Decode step ms |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for batch in BATCHES:
        for arm in ARMS:
            value = summaries.get(arm, {}).get(str(batch))
            if value is None:
                continue
            markdown.append(
                f"| {batch} | {arm} | {value['request_per_second']['median']:.4f} | "
                f"{value['ttft_ms']['median']:.2f} | {value['tpot_ms']['median']:.2f} | "
                f"{value['e2e_ms']['median']:.2f} | {value['prefill_ms']['median']:.2f} | "
                f"{value['decode_step_ms']['median']:.2f} |"
            )
    markdown.extend(
        [
            "",
            "This is a single-round qualification; in-process samples are not independent cold-process repeats.",
            "",
        ]
    )
    atomic_text(root / "summary.md", "\n".join(markdown))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if execution_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
