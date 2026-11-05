#!/usr/bin/env python3
"""Verify and summarize the same-process E13 Short/B128 K2/K6 ABBA diagnosis."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any


ABBA_ORDER = (
    ("a1_k2_short_b128", "k2", 1364, "cold70"),
    ("b1_k6_short_b128", "k6", 4095, "cold89"),
    ("b2_k6_short_b128", "k6", 4095, "cold89"),
    ("a2_k2_short_b128", "k2", 1364, "cold70"),
)
SHRINK_ISOLATION_ORDER = (
    ("a0_k2_initial_short_b128", "k2", 1364, "cold70"),
    ("b0_k4_grown_short_b128", "k4", 2730, "cold80"),
    ("a1_k2_after_k4_short_b128", "k2", 1364, "cold70"),
    ("c0_k6_grown_short_b128", "k6", 4095, "cold89"),
    ("b1_k4_after_k6_short_b128", "k4", 2730, "cold80"),
    ("a2_k2_after_k6_short_b128", "k2", 1364, "cold70"),
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def dist(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("empty distribution")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def ratio(left: float, right: float) -> float | None:
    return left / right if right else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    case_dir = root / "roundtrip"
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    result_path = case_dir / "result.json"
    returncode_path = case_dir / "returncode.txt"
    check("artifacts_present", result_path.is_file() and returncode_path.is_file(), str(case_dir))
    result = load_json(result_path) if result_path.is_file() else {}
    returncode = (
        int(returncode_path.read_text(encoding="utf-8").strip())
        if returncode_path.is_file()
        else -1
    )
    check("returncode", returncode == 0, returncode)
    check("runner_status", result.get("status") == "ok", result.get("status"))
    config = result.get("configuration") or {}
    scope = config.get("measurement_scope")
    order = (
        ABBA_ORDER
        if scope == "short_b128_abba"
        else SHRINK_ISOLATION_ORDER
        if scope == "short_b128_shrink_isolation"
        else ()
    )
    check(
        "scope",
        result.get("arm") == "roundtrip"
        and bool(order)
        and int(config.get("effective_max_num_seqs", -1)) == 128,
        {"arm": result.get("arm"), "configuration": config},
    )

    segments = result.get("segments") or []
    check(
        "segment_order",
        [item.get("label") for item in segments] == [item[0] for item in order],
        [item.get("label") for item in segments],
    )
    summaries: dict[str, Any] = {}
    all_hashes: list[str] = []
    for index, (label, tier, target, plan_class) in enumerate(order):
        if index >= len(segments):
            continue
        segment = segments[index]
        warmups = segment.get("warmups_excluded") or []
        measured = segment.get("measurements") or []
        contract = len(warmups) == 1 and len(measured) == 3
        for item in warmups + measured:
            contract = contract and (
                int(item.get("batch_size", -1)) == 128
                and int(item.get("prompt_tokens_per_request", -1)) == 128
                and int(item.get("output_tokens_per_request", -1)) == 32
                and int(item.get("prefill_step_count", -1)) == 2
                and int(item.get("decode_step_ms", {}).get("count", -1)) == 31
                and len(item.get("decode_step_values_ms") or []) == 31
                and int(item.get("schedule_step_count", -1)) == 33
                and int(item.get("max_scheduler_running_requests", -1)) == 128
                and int(item.get("max_scheduler_waiting_requests", -1)) == 64
            )
        check(f"{label}_measurement_contract", contract, {"warmups": warmups, "measured": measured})
        before = segment.get("audit_before") or {}
        after = segment.get("audit_after") or {}
        stable = (
            segment.get("tier") == tier
            and int(before.get("kv", {}).get("current_target_blocks", -1)) == target
            and int(after.get("kv", {}).get("current_target_blocks", -1)) == target
            and before.get("plan", {}).get("active_plan_class") == plan_class
            and after.get("plan", {}).get("active_plan_class") == plan_class
        )
        check(f"{label}_tier_stable", stable, {"before": before, "after": after})
        if not warmups or not measured:
            continue
        decode = [float(item["steady_decode_last16_ms"]["median"]) for item in measured]
        prefill = [float(item["prefill_ms"]) for item in measured]
        throughput = [float(item["request_throughput_per_second"]) for item in measured]
        e2e = [float(item["request_timings"]["request_latency_ms"]["median"]) for item in measured]
        hashes = [str(item["output_token_ids_sha256"]) for item in measured]
        all_hashes.extend(hashes)
        summaries[label] = {
            "tier": tier,
            "warmup_decode_median_ms": float(warmups[0]["steady_decode_last16_ms"]["median"]),
            "decode_batch_medians_ms": decode,
            "decode_distribution_ms": dist(decode),
            "first_to_last_decode_ratio": ratio(decode[-1], decode[0]),
            "prefill_ms": dist(prefill),
            "request_per_second": dist(throughput),
            "e2e_ms": dist(e2e),
            "output_hashes": hashes,
        }

    hash_equal = bool(all_hashes) and len(set(all_hashes)) == 1
    check("all_outputs_equal", hash_equal, all_hashes)
    transitions = result.get("transitions") or []
    adjacent = all(
        abs(int(item.get("new_target_blocks", 0)) - int(item.get("old_target_blocks", 0)))
        in {0, 1365, 1366}
        for item in transitions
    )
    check("transitions_adjacent", bool(transitions) and adjacent, transitions)

    gpu_path = case_dir / "gpu.after.json"
    gpu = load_json(gpu_path) if gpu_path.is_file() else {}
    check("gpu_clean", gpu.get("gpu_clean") is True, gpu)
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
    check("kernel_clean", not bad_kernel, "fault marker absent" if not bad_kernel else "fault marker present")
    before_path = root / "module.before.json"
    after_path = root / "module.after.json"
    before_module = load_json(before_path) if before_path.is_file() else {}
    after_module = load_json(after_path) if after_path.is_file() else {}
    check(
        "stock_uvm_restored",
        bool(before_module)
        and before_module.get("module_sha256") == after_module.get("module_sha256")
        and after_module.get("gpu_clean") is True,
        {"before": before_module, "after": after_module},
    )

    findings: dict[str, Any] = {}
    if scope == "short_b128_abba" and all(
        label in summaries for label, *_ in ABBA_ORDER
    ):
        a1 = summaries["a1_k2_short_b128"]["decode_batch_medians_ms"]
        b1 = summaries["b1_k6_short_b128"]["decode_batch_medians_ms"]
        b2 = summaries["b2_k6_short_b128"]["decode_batch_medians_ms"]
        a2 = summaries["a2_k2_short_b128"]["decode_batch_medians_ms"]
        findings = {
            "a1_k2_first_to_last_decode_ratio": ratio(a1[-1], a1[0]),
            "b1_k6_first_to_last_decode_ratio": ratio(b1[-1], b1[0]),
            "a2_k2_over_b2_k6_decode_median": ratio(statistics.median(a2), statistics.median(b2)),
            "a2_k2_over_a1_k2_terminal_decode": ratio(statistics.median(a2), a1[-1]),
            "k2_slow_state_disappears_without_tier_change": a1[-1] <= 0.9 * a1[0],
            "returned_k2_matches_prior_terminal_within_10pct": abs(statistics.median(a2) / a1[-1] - 1.0) <= 0.1,
            "stable_k2_k6_within_10pct": abs(statistics.median(a2) / statistics.median(b2) - 1.0) <= 0.1,
        }
    elif scope == "short_b128_shrink_isolation" and all(
        label in summaries for label, *_ in SHRINK_ISOLATION_ORDER
    ):
        k2_initial = statistics.median(
            summaries["a0_k2_initial_short_b128"]["decode_batch_medians_ms"]
        )
        k2_after_k4 = statistics.median(
            summaries["a1_k2_after_k4_short_b128"]["decode_batch_medians_ms"]
        )
        k4_grown = statistics.median(
            summaries["b0_k4_grown_short_b128"]["decode_batch_medians_ms"]
        )
        k4_after_k6 = statistics.median(
            summaries["b1_k4_after_k6_short_b128"]["decode_batch_medians_ms"]
        )
        k2_after_k6 = statistics.median(
            summaries["a2_k2_after_k6_short_b128"]["decode_batch_medians_ms"]
        )
        findings = {
            "k2_initial_decode_median_ms": k2_initial,
            "k2_after_k4_decode_median_ms": k2_after_k4,
            "k2_after_k6_decode_median_ms": k2_after_k6,
            "k4_grown_decode_median_ms": k4_grown,
            "k4_after_k6_decode_median_ms": k4_after_k6,
            "k2_after_k4_over_initial": ratio(k2_after_k4, k2_initial),
            "k2_after_k6_over_initial": ratio(k2_after_k6, k2_initial),
            "k4_after_k6_over_grown": ratio(k4_after_k6, k4_grown),
            "first_adjacent_shrink_changes_k2_over_10pct": k2_after_k4 > 1.1 * k2_initial,
            "k6_to_k4_shrink_changes_k4_over_10pct": k4_after_k6 > 1.1 * k4_grown,
        }

    status = "ok" if all(item["ok"] for item in checks) else "error"
    output = {
        "schema": "eurosys_e13_short_b128_abba_verification_v1",
        "status": status,
        "single_cold_process": True,
        "measurement_scope": scope,
        "order": [item[0] for item in order],
        "summaries": summaries,
        "findings": findings,
        "checks": checks,
    }
    output_path = args.output or (root / "verification.json")
    atomic_text(output_path, json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    markdown = [
        "# E13 Short/B128 same-process K2/K6 Decode diagnosis",
        "",
        "| Segment | Tier | warmup Decode ms | measured Decode medians ms | req/s median | Prefill ms median | E2E ms median |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for label, tier, *_ in order:
        value = summaries.get(label)
        if value is None:
            continue
        decode_text = "/".join(f"{item:.2f}" for item in value["decode_batch_medians_ms"])
        markdown.append(
            f"| {label} | {tier} | {value['warmup_decode_median_ms']:.2f} | {decode_text} | "
            f"{value['request_per_second']['median']:.4f} | {value['prefill_ms']['median']:.2f} | "
            f"{value['e2e_ms']['median']:.2f} |"
        )
    markdown.extend(["", f"Verifier status: `{status}`.", ""])
    atomic_text(root / "summary.md", "\n".join(markdown))
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
