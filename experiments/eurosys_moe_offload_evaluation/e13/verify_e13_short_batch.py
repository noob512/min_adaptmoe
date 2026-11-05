#!/usr/bin/env python3
"""Verify a single-round E13 Fixed-K2/K6 Short-batch study."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Callable


ARMS = ("fixed_k2", "fixed_k6")


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
    parser.add_argument("--batch-size", type=int, choices=(6, 64, 128), default=6)
    args = parser.parse_args()
    root = args.run_root.resolve()
    batch_size = int(args.batch_size)
    scope = f"short_b{batch_size}"
    expected_prefill_steps = math.ceil(batch_size * 128 / 8192)
    expected_schedule_steps = expected_prefill_steps + 31
    expected_max_waiting = max(0, batch_size - 64)
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
        check(
            f"{arm}_scope",
            result.get("arm") == arm
            and result.get("configuration", {}).get("measurement_scope")
            == scope
            and int(
                result.get("configuration", {}).get(
                    "effective_max_num_seqs", -1
                )
            )
            >= batch_size,
            {
                "arm": result.get("arm"),
                "scope": result.get("configuration", {}).get("measurement_scope"),
            },
        )
        segments = result.get("segments") or []
        expected_label = f"{arm}_short_b{batch_size}"
        check(
            f"{arm}_segment",
            len(segments) == 1 and segments[0].get("label") == expected_label,
            [item.get("label") for item in segments],
        )
        if segments:
            segment = segments[0]
            measurements = segment.get("measurements") or []
            check(
                f"{arm}_measurement_contract",
                len(measurements) == 3
                and all(
                    item.get("workload") == "short"
                    and item.get("role") == f"measured_short_b{batch_size}"
                    and int(item.get("batch_size", -1)) == batch_size
                    and int(item.get("prompt_tokens_per_request", -1)) == 128
                    and int(item.get("output_tokens_per_request", -1)) == 32
                    and int(item.get("prefill_step_count", -1))
                    == expected_prefill_steps
                    and int(item.get("decode_step_ms", {}).get("count", -1))
                    == 31
                    and int(
                        item.get("steady_decode_last16_ms", {}).get("count", -1)
                    )
                    == 16
                    and int(item.get("schedule_step_count", -1))
                    == expected_schedule_steps
                    and int(item.get("max_scheduler_running_requests", -1))
                    == batch_size
                    and int(item.get("max_scheduler_waiting_requests", -1))
                    == expected_max_waiting
                    for item in measurements
                ),
                measurements,
            )
            before = segment.get("audit_before") or {}
            after = segment.get("audit_after") or {}
            check(
                f"{arm}_tier_stable",
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
        check(f"{arm}_kernel_clean", not bad_kernel, "fault marker absent" if not bad_kernel else "fault marker present")

    module_before_path = root / "module.before.json"
    module_after_path = root / "module.after.json"
    if module_before_path.is_file() and module_after_path.is_file():
        module_before = load_json(module_before_path)
        module_after = load_json(module_after_path)
        check(
            "stock_uvm_restored",
            module_before.get("module_sha256") == module_after.get("module_sha256")
            and module_after.get("gpu_clean") is True,
            {"before": module_before, "after": module_after},
        )
    else:
        check("stock_uvm_restored", False, "module snapshots absent")

    summaries: dict[str, dict[str, Any]] = {}
    for arm, result in cases.items():
        segments = result.get("segments") or []
        if not segments:
            continue
        measurements = segments[0].get("measurements") or []
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
        summaries[arm] = {
            name: distribution([getter(item) for item in measurements])
            for name, getter in getters.items()
        }

    comparison: dict[str, Any] = {}
    if all(arm in summaries for arm in ARMS):
        k2 = summaries["fixed_k2"]
        k6 = summaries["fixed_k6"]
        comparison = {
            "k2_over_k6_request_throughput": (
                float(k2["request_per_second"]["median"])
                / float(k6["request_per_second"]["median"])
            ),
            "k2_over_k6_output_token_throughput": (
                float(k2["output_token_per_second"]["median"])
                / float(k6["output_token_per_second"]["median"])
            ),
            "k2_over_k6_prefill_latency": (
                float(k2["prefill_ms"]["median"])
                / float(k6["prefill_ms"]["median"])
            ),
            "k2_over_k6_decode_latency": (
                float(k2["decode_step_ms"]["median"])
                / float(k6["decode_step_ms"]["median"])
            ),
            "k2_over_k6_ttft": (
                float(k2["ttft_ms"]["median"])
                / float(k6["ttft_ms"]["median"])
            ),
            "k2_over_k6_tpot": (
                float(k2["tpot_ms"]["median"])
                / float(k6["tpot_ms"]["median"])
            ),
            "k2_over_k6_e2e": (
                float(k2["e2e_ms"]["median"])
                / float(k6["e2e_ms"]["median"])
            ),
        }
        k2_hashes = [
            item["output_token_ids_sha256"]
            for item in cases["fixed_k2"]["segments"][0]["measurements"]
        ]
        k6_hashes = [
            item["output_token_ids_sha256"]
            for item in cases["fixed_k6"]["segments"][0]["measurements"]
        ]
        hashes_equal = k2_hashes == k6_hashes and len(set(k2_hashes)) == 1
        comparison["output_hash_equal"] = hashes_equal
        check("cross_arm_output_hash_equal", hashes_equal, {"k2": k2_hashes, "k6": k6_hashes})

    execution_ok = len(cases) == len(ARMS) and all(item["ok"] for item in checks)
    summary = {
        "schema": "eurosys_e13_short_batch_verification_v1",
        "status": "ok" if execution_ok else "error",
        "single_round": True,
        "independent_cold_processes": 2,
        "workload": f"Short-128/B{batch_size} -> 32",
        "summaries": summaries,
        "comparison": comparison,
        "checks": checks,
    }
    output = args.output or (root / "verification.json")
    atomic_text(output, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    markdown = [
        f"# E13 Fixed-K2 versus Fixed-K6 Short-B{batch_size} single-round result",
        "",
        "One excluded warmup and three in-process measured samples per cold process.",
        "",
        "| Arm | req/s | tok/s | TTFT ms | TPOT ms | E2E ms | Prefill ms | Decode step ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ARMS:
        if arm not in summaries:
            continue
        value = summaries[arm]
        markdown.append(
            f"| {arm} | {value['request_per_second']['median']:.4f} | "
            f"{value['output_token_per_second']['median']:.3f} | "
            f"{value['ttft_ms']['median']:.2f} | {value['tpot_ms']['median']:.2f} | "
            f"{value['e2e_ms']['median']:.2f} | {value['prefill_ms']['median']:.2f} | "
            f"{value['decode_step_ms']['median']:.2f} |"
        )
    markdown.extend(
        [
            "",
            "This is a single-round qualification; the three samples per arm are not independent cold-process repeats.",
            "",
        ]
    )
    atomic_text(root / "summary.md", "\n".join(markdown))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if execution_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
