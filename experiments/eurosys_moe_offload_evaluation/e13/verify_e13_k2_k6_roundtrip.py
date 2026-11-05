#!/usr/bin/env python3
"""Verify the single-round E13 K2/K6 fixed and adaptive round-trip study."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any, Callable


ARMS = ("fixed_k2", "fixed_k6", "roundtrip")
EXPECTED_SEGMENTS = {
    "fixed_k2": ("fixed_k2",),
    "fixed_k6": ("fixed_k6",),
    "roundtrip": ("adaptive_k2_initial", "adaptive_k6", "adaptive_k2_return"),
}
EQUIVALENCE_LOWER = 0.90
EQUIVALENCE_UPPER = 1.10


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def median_measurement(
    segment: dict[str, Any],
    predicate: Callable[[dict[str, Any]], bool],
    getter: Callable[[dict[str, Any]], float],
) -> tuple[float, int]:
    values = [float(getter(item)) for item in segment["measurements"] if predicate(item)]
    if not values:
        raise ValueError(f"no matching measurements in {segment.get('label')}")
    return statistics.median(values), len(values)


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
        check(
            f"{arm}_identity",
            result.get("schema") == "eurosys_e13_k2_k6_roundtrip_v1"
            and result.get("arm") == arm,
            {"schema": result.get("schema"), "arm": result.get("arm")},
        )
        configuration = result.get("configuration") or {}
        check(
            f"{arm}_timing_contract",
            configuration.get("switch_transaction_excluded") is True
            and configuration.get("post_switch_warmup_excluded") is True
            and configuration.get("single_cold_process_per_arm") is True,
            configuration,
        )
        segments = result.get("segments") or []
        labels = tuple(item.get("label") for item in segments)
        check(f"{arm}_segment_coverage", labels == EXPECTED_SEGMENTS[arm], labels)
        for segment in segments:
            before = segment.get("audit_before") or {}
            after = segment.get("audit_after") or {}
            check(
                f"{arm}_{segment.get('label')}_tier_stable",
                before.get("kv", {}).get("current_target_blocks")
                == after.get("kv", {}).get("current_target_blocks")
                and before.get("plan", {}).get("active_plan_class")
                == after.get("plan", {}).get("active_plan_class"),
                {"before": before, "after": after},
            )
            measurements = segment.get("measurements") or []
            check(
                f"{arm}_{segment.get('label')}_functional_measurements",
                bool(measurements)
                and all(
                    int(item.get("prefill_step_count", 0)) > 0
                    and int(item.get("steady_decode_last16_ms", {}).get("count", 0)) > 0
                    and int(item.get("max_scheduler_running_requests", 0)) > 0
                    for item in measurements
                ),
                len(measurements),
            )
        gpu_after_path = case_dir / "gpu.after.json"
        gpu_after = load_json(gpu_after_path) if gpu_after_path.is_file() else {}
        check(f"{arm}_gpu_clean", gpu_after.get("gpu_clean") is True, gpu_after)
        kernel_path = case_dir / "kernel.since.log"
        kernel_text = kernel_path.read_text(encoding="utf-8", errors="replace") if kernel_path.is_file() else ""
        bad_kernel = any(
            marker in kernel_text.lower()
            for marker in ("xid", "mmu fault", "page fault", "illegal address")
        )
        check(f"{arm}_kernel_clean", not bad_kernel, "fault marker absent" if not bad_kernel else "fault marker present")

    segment_by_label: dict[str, dict[str, Any]] = {}
    for result in cases.values():
        for segment in result.get("segments") or []:
            segment_by_label[str(segment["label"])] = segment

    def metric(
        label: str,
        *,
        role: str,
        batch: int | None,
        field: str,
    ) -> tuple[float, int]:
        segment = segment_by_label[label]

        def predicate(item: dict[str, Any]) -> bool:
            return item.get("role") == role and (
                batch is None or int(item.get("batch_size", -1)) == batch
            )

        if field == "prefill_ms":
            getter = lambda item: float(item["prefill_ms"])
        elif field == "decode_ms":
            getter = lambda item: float(item["steady_decode_last16_ms"]["median"])
        elif field == "wall_ms":
            getter = lambda item: float(item["wall_ms"])
        elif field == "throughput":
            getter = lambda item: float(item["request_throughput_per_second"])
        elif field == "max_running":
            getter = lambda item: float(item["max_scheduler_running_requests"])
        elif field == "max_waiting":
            getter = lambda item: float(item["max_scheduler_waiting_requests"])
        else:
            raise ValueError(field)
        return median_measurement(segment, predicate, getter)

    fixed_comparison: dict[str, Any] = {}
    equivalence_rows: list[dict[str, Any]] = []
    output_hash_checks: list[dict[str, Any]] = []
    if all(label in segment_by_label for label in ("fixed_k2", "fixed_k6")):
        low_k2, low_count = metric("fixed_k2", role="measured_low", batch=1, field="throughput")
        low_k6, _ = metric("fixed_k6", role="measured_low", batch=1, field="throughput")
        high_k2, high_count = metric("fixed_k2", role="measured_high", batch=6, field="throughput")
        high_k6, _ = metric("fixed_k6", role="measured_high", batch=6, field="throughput")
        k2_running, _ = metric("fixed_k2", role="measured_high", batch=6, field="max_running")
        k6_running, _ = metric("fixed_k6", role="measured_high", batch=6, field="max_running")
        k2_waiting, _ = metric("fixed_k2", role="measured_high", batch=6, field="max_waiting")
        k6_waiting, _ = metric("fixed_k6", role="measured_high", batch=6, field="max_waiting")
        fixed_comparison = {
            "low_pressure": {
                "workload": "Short-128/B1 -> 32",
                "samples_per_arm": low_count,
                "fixed_k2_request_per_second": low_k2,
                "fixed_k6_request_per_second": low_k6,
                "k2_over_k6_throughput": low_k2 / low_k6,
                "hypothesis_k2_faster": low_k2 > low_k6,
            },
            "high_pressure": {
                "workload": "RAG-8192/B6 -> 16",
                "samples_per_arm": high_count,
                "fixed_k2_request_per_second": high_k2,
                "fixed_k6_request_per_second": high_k6,
                "k6_over_k2_throughput": high_k6 / high_k2,
                "fixed_k2_max_running": int(k2_running),
                "fixed_k6_max_running": int(k6_running),
                "fixed_k2_max_waiting": int(k2_waiting),
                "fixed_k6_max_waiting": int(k6_waiting),
                "k2_cannot_reside_full_batch": k2_running < 6 and k2_waiting > 0,
                # waiting_after is sampled inside the schedule wrapper before
                # the engine applies that SchedulerOutput, so its maximum can
                # include the just-injected cohort.  max_running==B is the
                # audited evidence that the full batch became resident.
                "k6_resides_full_batch": k6_running == 6,
                "hypothesis_k6_faster_or_k2_capacity_limited": (
                    high_k6 > high_k2 or (k2_running < 6 and k2_waiting > 0)
                ),
            },
        }

    comparisons = (
        ("adaptive_k2_initial", "fixed_k2", "K2 initial"),
        ("adaptive_k6", "fixed_k6", "K6 after grow"),
        ("adaptive_k2_return", "fixed_k2", "K2 after return"),
    )
    for adaptive_label, fixed_label, display in comparisons:
        if adaptive_label not in segment_by_label or fixed_label not in segment_by_label:
            continue
        for batch in (1, 8):
            for field in ("prefill_ms", "decode_ms", "wall_ms"):
                adaptive_value, samples = metric(
                    adaptive_label, role="measured_micro", batch=batch, field=field
                )
                fixed_value, _ = metric(
                    fixed_label, role="measured_micro", batch=batch, field=field
                )
                ratio = adaptive_value / fixed_value
                equivalence_rows.append(
                    {
                        "state": display,
                        "adaptive_segment": adaptive_label,
                        "fixed_segment": fixed_label,
                        "batch_size": batch,
                        "metric": field,
                        "samples_per_side": samples,
                        "fixed": fixed_value,
                        "adaptive": adaptive_value,
                        "adaptive_over_fixed": ratio,
                        "equivalent_within_10pct": EQUIVALENCE_LOWER <= ratio <= EQUIVALENCE_UPPER,
                    }
                )
        fixed_hashes = {
            (item["role"], int(item["batch_size"]), int(item["repeat_index"])): item["output_token_ids_sha256"]
            for item in segment_by_label[fixed_label]["measurements"]
            if item["role"] == "measured_micro"
        }
        adaptive_hashes = {
            (item["role"], int(item["batch_size"]), int(item["repeat_index"])): item["output_token_ids_sha256"]
            for item in segment_by_label[adaptive_label]["measurements"]
            if item["role"] == "measured_micro"
        }
        hashes_equal = fixed_hashes == adaptive_hashes and bool(fixed_hashes)
        output_hash_checks.append({"state": display, "equal": hashes_equal})
        check(f"{adaptive_label}_micro_output_hash_equal", hashes_equal, output_hash_checks[-1])

    transition_path: list[tuple[int, int]] = []
    if "roundtrip" in cases:
        transition_path = [
            (int(item["old_target_blocks"]), int(item["new_target_blocks"]))
            for item in cases["roundtrip"].get("transitions") or []
            if int(item["old_target_blocks"]) != int(item["new_target_blocks"])
        ]
        expected_path = [(1364, 2730), (2730, 4095), (4095, 2730), (2730, 1364)]
        check("roundtrip_adjacent_2gb_transition_path", transition_path == expected_path, transition_path)

    execution_ok = len(cases) == len(ARMS) and all(item["ok"] for item in checks)
    all_equivalent = len(equivalence_rows) == 18 and all(
        item["equivalent_within_10pct"] for item in equivalence_rows
    )
    hypotheses = {
        "low_pressure_k2_faster": bool(
            fixed_comparison.get("low_pressure", {}).get("hypothesis_k2_faster")
        ),
        "high_pressure_k6_faster_or_k2_capacity_limited": bool(
            fixed_comparison.get("high_pressure", {}).get(
                "hypothesis_k6_faster_or_k2_capacity_limited"
            )
        ),
        "adaptive_steady_state_all_metrics_equivalent_within_10pct": all_equivalent,
    }
    summary = {
        "schema": "eurosys_e13_k2_k6_roundtrip_verification_v1",
        "status": "ok" if execution_ok else "error",
        "single_round": True,
        "independent_cold_processes": 3,
        "equivalence_band": {
            "metric": "adaptive latency / fixed latency",
            "lower": EQUIVALENCE_LOWER,
            "upper": EQUIVALENCE_UPPER,
        },
        "fixed_comparison": fixed_comparison,
        "equivalence_rows": equivalence_rows,
        "all_equivalence_cells_within_10pct": all_equivalent,
        "output_hash_checks": output_hash_checks,
        "transition_path": transition_path,
        "hypotheses": hypotheses,
        "checks": checks,
    }
    output = args.output or (root / "verification.json")
    atomic_text(output, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    markdown = [
        "# E13 K2/K6 fixed and adaptive round-trip single-round result",
        "",
        "Switch transactions and target-tier warmups are excluded from all timed measurements.",
        "",
    ]
    if fixed_comparison:
        low = fixed_comparison["low_pressure"]
        high = fixed_comparison["high_pressure"]
        markdown.extend(
            [
                "| Pressure | Fixed-K2 req/s | Fixed-K6 req/s | Ratio | K2/K6 max running | K2/K6 max waiting |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                f"| Low | {low['fixed_k2_request_per_second']:.4f} | {low['fixed_k6_request_per_second']:.4f} | K2/K6 {low['k2_over_k6_throughput']:.3f}× | 1/1 | 0/0 |",
                f"| High | {high['fixed_k2_request_per_second']:.4f} | {high['fixed_k6_request_per_second']:.4f} | K6/K2 {high['k6_over_k2_throughput']:.3f}× | {high['fixed_k2_max_running']}/{high['fixed_k6_max_running']} | {high['fixed_k2_max_waiting']}/{high['fixed_k6_max_waiting']} |",
                "",
            ]
        )
    markdown.extend(
        [
            "| State | B | Metric | Fixed | Adaptive | A/F | Within ±10% |",
            "| --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in equivalence_rows:
        markdown.append(
            f"| {row['state']} | {row['batch_size']} | {row['metric']} | "
            f"{row['fixed']:.3f} | {row['adaptive']:.3f} | "
            f"{row['adaptive_over_fixed']:.3f}× | {row['equivalent_within_10pct']} |"
        )
    markdown.extend(
        [
            "",
            f"All 18 steady-state cells within ±10%: `{all_equivalent}`.",
            "This is a one-round qualification with repeated in-process samples, not a confidence-interval equivalence proof.",
            "",
        ]
    )
    atomic_text(root / "summary.md", "\n".join(markdown))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if execution_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
