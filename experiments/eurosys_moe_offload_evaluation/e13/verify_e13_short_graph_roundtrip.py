#!/usr/bin/env python3
"""Verify E13 FULL_SEPARATE Fixed/Adaptive Short B64/B128 round-trip."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any, Callable


ARMS = ("fixed_k2", "fixed_k6", "roundtrip")
BATCHES = (64, 128)
CAPTURE_SIZES = (64, 128, 8192)
GRAPH_CLASSES = ("cold70", "cold80", "cold89")
EQUIVALENCE_LOWER = 0.90
EQUIVALENCE_UPPER = 1.10

EXPECTED_LABELS = {
    "fixed_k2": [f"fixed_k2_short_b{batch}" for batch in BATCHES],
    "fixed_k6": [f"fixed_k6_short_b{batch}" for batch in BATCHES],
    "roundtrip": [
        *[f"adaptive_k2_initial_short_b{batch}" for batch in BATCHES],
        *[f"adaptive_k6_after_grow_short_b{batch}" for batch in BATCHES],
        *[f"adaptive_k2_after_return_short_b{batch}" for batch in BATCHES],
    ],
}


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
    output = args.output or root / "verification.json"
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
        config = result.get("configuration") or {}
        check(
            f"{arm}_scope",
            result.get("arm") == arm
            and config.get("measurement_scope") == "short_graph_roundtrip"
            and config.get("full_separate_graph") is True
            and tuple(config.get("graph_capture_sizes") or ()) == CAPTURE_SIZES
            and int(config.get("effective_max_num_seqs", -1)) == 128,
            config,
        )
        initialization = result.get("initialization") or {}
        compilation = initialization.get("realized_compilation") or {}
        graph_calls = initialization.get("graph_wrapper_calls_after_load") or {}
        graph_classes = initialization.get("graph_class_diagnostics") or {}
        class_states = graph_classes.get("classes") or {}
        expected_graph_classes = (
            GRAPH_CLASSES
            if arm == "roundtrip"
            else (("cold70",) if arm == "fixed_k2" else ("cold89",))
        )
        check(
            f"{arm}_full_separate_realized",
            compilation.get("mode") == "NONE"
            and compilation.get("cudagraph_mode") == "FULL_SEPARATE"
            and tuple(compilation.get("cudagraph_capture_sizes") or ())
            == CAPTURE_SIZES
            and int(graph_calls.get("full_capture", 0)) >= len(CAPTURE_SIZES),
            {"compilation": compilation, "graph_calls": graph_calls},
        )
        check(
            f"{arm}_required_graph_classes_captured",
            tuple(sorted(class_states)) == tuple(sorted(expected_graph_classes))
            and all(
                bool(class_states[name].get("fully_captured"))
                and int(class_states[name].get("entry_count", 0))
                == len(CAPTURE_SIZES)
                for name in expected_graph_classes
            ),
            graph_classes,
        )
        phase = initialization.get("phase") or {}
        check(
            f"{arm}_pd_graph_contract",
            phase.get("phase_mode") == "prefill_graph_remote_decode"
            and phase.get("park_resume_enabled") is True
            and phase.get("alias_graph_captured") is True
            and phase.get("remote_graph_captured") is True,
            phase,
        )
        segments = result.get("segments") or []
        check(
            f"{arm}_segment_order",
            [item.get("label") for item in segments] == EXPECTED_LABELS[arm],
            [item.get("label") for item in segments],
        )
        for segment in segments:
            label = str(segment.get("label"))
            measurements = segment.get("measurements") or []
            batch = int(measurements[0].get("batch_size", -1)) if measurements else -1
            prefill_steps = 1 if batch == 64 else 2
            schedule_steps = prefill_steps + 31
            tier = str(segment.get("tier"))
            expected_class = {"k2": "cold70", "k6": "cold89"}.get(tier)
            contract_ok = batch in BATCHES and len(measurements) == 3 and all(
                int(item.get("batch_size", -1)) == batch
                and int(item.get("prompt_tokens_per_request", -1)) == 128
                and int(item.get("output_tokens_per_request", -1)) == 32
                and int(item.get("prefill_step_count", -1)) == prefill_steps
                and int(item.get("decode_step_ms", {}).get("count", -1)) == 31
                and int(item.get("schedule_step_count", -1)) == schedule_steps
                and int(item.get("max_scheduler_running_requests", -1)) == batch
                for item in measurements
            )
            check(f"{label}_measurement_contract", contract_ok, measurements)
            graph_ok = len(measurements) == 3 and all(
                int(item.get("graph_wrapper_delta", {}).get("full_replay", 0))
                == schedule_steps
                and len(item.get("graph_step_deltas") or []) == schedule_steps
                and all(
                    int(step.get("full_replay", 0)) == 1
                    and int(step.get("full_capture", 0)) == 0
                    and int(step.get("none_passthrough", 0)) == 0
                    for step in item.get("graph_step_deltas") or []
                )
                for item in measurements
            )
            check(f"{label}_all_steps_graph_replay", graph_ok, measurements)
            pd_ok = len(measurements) == 3 and all(
                item.get("phase_after", {}).get("lifecycle_state")
                == "remote_parked"
                and item.get("phase_after", {}).get("remote_map_active") is True
                and int(item.get("phase_after", {}).get("phase_transition_count", 0))
                > int(item.get("phase_before", {}).get("phase_transition_count", 0))
                for item in measurements
            )
            check(f"{label}_fast_pd_switch", pd_ok, measurements)
            before = segment.get("audit_before") or {}
            after = segment.get("audit_after") or {}
            check(
                f"{label}_tier_and_graph_class_stable",
                expected_class is not None
                and before.get("plan", {}).get("active_plan_class") == expected_class
                and after.get("plan", {}).get("active_plan_class") == expected_class
                and before.get("graph_class", {}).get("active_class") == expected_class
                and after.get("graph_class", {}).get("active_class") == expected_class
                and before.get("kv", {}).get("current_target_blocks")
                == after.get("kv", {}).get("current_target_blocks"),
                {"before": before, "after": after},
            )
        initial_counter = initialization.get("graph_counter_after") or {}
        final_audit = result.get("final_audit") or {}
        final_counter = final_audit.get("graph_counter") or {}
        check(
            f"{arm}_no_graph_recapture",
            initial_counter.get("num_cudagraph_captured")
            == final_counter.get("num_cudagraph_captured"),
            {"initial": initial_counter, "final": final_counter},
        )
        final_classes = (final_audit.get("graph_class_diagnostics") or {}).get(
            "classes"
        ) or {}
        check(
            f"{arm}_required_graph_classes_retained",
            tuple(sorted(final_classes)) == tuple(sorted(expected_graph_classes))
            and all(
                bool(final_classes[name].get("fully_captured"))
                for name in expected_graph_classes
            ),
            final_classes,
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
        bad_markers = (
            "xid",
            "fault_pte",
            "fault_pde",
            "mmu fault",
            "gpu page fault",
            "illegal address",
        )
        bad = [marker for marker in bad_markers if marker in kernel_text]
        check(f"{arm}_kernel_clean", not bad, bad or "fault marker absent")

    if "roundtrip" in cases:
        path = [
            (int(item["old_target_blocks"]), int(item["new_target_blocks"]))
            for item in cases["roundtrip"].get("transitions") or []
            if int(item["old_target_blocks"]) != int(item["new_target_blocks"])
        ]
        expected_path = [(1364, 2730), (2730, 4095), (4095, 2730), (2730, 1364)]
        check("adjacent_2g_roundtrip", path == expected_path, path)
    else:
        path = []

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
        "pd_cutover_step_ms": lambda item: float(item["pd_cutover_step_ms"]),
        "decode_step_ms": lambda item: float(item["steady_decode_last16_ms"]["median"]),
        "ttft_ms": lambda item: float(item["request_timings"]["request_ttft_ms"]["median"]),
        "tpot_ms": lambda item: float(item["request_timings"]["request_tpot_ms"]["median"]),
        "e2e_ms": lambda item: float(item["request_timings"]["request_latency_ms"]["median"]),
    }
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    hashes: dict[str, list[str]] = {}
    for result in cases.values():
        for segment in result.get("segments") or []:
            label = str(segment["label"])
            measurements = segment.get("measurements") or []
            if not measurements:
                continue
            summaries[label] = {
                name: distribution([getter(item) for item in measurements])
                for name, getter in getters.items()
            }
            hashes[label] = [str(item["output_token_ids_sha256"]) for item in measurements]

    fixed_comparisons: list[dict[str, Any]] = []
    equivalence: list[dict[str, Any]] = []
    states = (
        ("k2_initial", "adaptive_k2_initial", "fixed_k2"),
        ("k6_after_grow", "adaptive_k6_after_grow", "fixed_k6"),
        ("k2_after_return", "adaptive_k2_after_return", "fixed_k2"),
    )
    for batch in BATCHES:
        k2_label = f"fixed_k2_short_b{batch}"
        k6_label = f"fixed_k6_short_b{batch}"
        if k2_label in summaries and k6_label in summaries:
            k2 = summaries[k2_label]
            k6 = summaries[k6_label]
            fixed_comparisons.append(
                {
                    "batch_size": batch,
                    "k2_over_k6_request_throughput": k2["request_per_second"]["median"]
                    / k6["request_per_second"]["median"],
                    "k2_over_k6_prefill_latency": k2["prefill_ms"]["median"]
                    / k6["prefill_ms"]["median"],
                    "k2_over_k6_decode_latency": k2["decode_step_ms"]["median"]
                    / k6["decode_step_ms"]["median"],
                    "k2_over_k6_e2e": k2["e2e_ms"]["median"]
                    / k6["e2e_ms"]["median"],
                }
            )
        for state, adaptive_prefix, fixed_prefix in states:
            adaptive_label = f"{adaptive_prefix}_short_b{batch}"
            fixed_label = f"{fixed_prefix}_short_b{batch}"
            if adaptive_label not in summaries or fixed_label not in summaries:
                continue
            output_equal = len(set(hashes[adaptive_label] + hashes[fixed_label])) == 1
            check(f"{state}_b{batch}_output_hash_equal", output_equal, {
                "adaptive": hashes[adaptive_label], "fixed": hashes[fixed_label]
            })
            for metric in getters:
                adaptive_value = float(summaries[adaptive_label][metric]["median"])
                fixed_value = float(summaries[fixed_label][metric]["median"])
                ratio = adaptive_value / fixed_value
                equivalence.append(
                    {
                        "state": state,
                        "batch_size": batch,
                        "metric": metric,
                        "fixed": fixed_value,
                        "adaptive": adaptive_value,
                        "adaptive_over_fixed": ratio,
                        "within_10pct": EQUIVALENCE_LOWER <= ratio <= EQUIVALENCE_UPPER,
                    }
                )

    status = "ok" if checks and all(item["ok"] for item in checks) else "error"
    all_equivalent = len(equivalence) == len(states) * len(BATCHES) * len(getters) and all(
        item["within_10pct"] for item in equivalence
    )
    transition_wall_ms = [
        float(item["wall_ms_excluded"])
        for item in cases.get("roundtrip", {}).get("transitions") or []
        if int(item["old_target_blocks"]) != int(item["new_target_blocks"])
    ]
    verification = {
        "schema": "eurosys_e13_short_graph_roundtrip_verification_v1",
        "status": status,
        "single_round": True,
        "adaptive_graph_class_set": list(GRAPH_CLASSES),
        "fixed_arms_capture_only_active_class": True,
        "equivalence_band": [EQUIVALENCE_LOWER, EQUIVALENCE_UPPER],
        "all_equivalence_metrics_within_10pct": all_equivalent,
        "transition_path": path,
        "transition_wall_ms_excluded": transition_wall_ms,
        "checks": checks,
        "summaries": summaries,
        "fixed_comparisons": fixed_comparisons,
        "equivalence": equivalence,
    }
    atomic_text(output, json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    markdown = [
        "# E13 FULL_SEPARATE Adaptive K2→K6→K2 steady-state round-trip",
        "",
        f"Verifier status: `{status}`. Adaptive pre-captures K2/K4/K6; each Fixed arm captures only its active class. Initialization, switch transactions and one target warmup per cell are excluded.",
        "",
        "| State | B | req/s | TTFT ms | TPOT ms | E2E ms | Prefill ms | P→D ms | Decode ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    rows = (
        ("Fixed-K2", "fixed_k2"),
        ("Fixed-K6", "fixed_k6"),
        ("Adaptive initial K2", "adaptive_k2_initial"),
        ("Adaptive grown K6", "adaptive_k6_after_grow"),
        ("Adaptive returned K2", "adaptive_k2_after_return"),
    )
    for state, prefix in rows:
        for batch in BATCHES:
            value = summaries.get(f"{prefix}_short_b{batch}")
            if value is None:
                continue
            markdown.append(
                f"| {state} | {batch} | {value['request_per_second']['median']:.4f} | "
                f"{value['ttft_ms']['median']:.2f} | {value['tpot_ms']['median']:.2f} | "
                f"{value['e2e_ms']['median']:.2f} | {value['prefill_ms']['median']:.2f} | "
                f"{value['pd_cutover_step_ms']['median']:.2f} | {value['decode_step_ms']['median']:.2f} |"
            )
    markdown.extend(
        [
            "",
            f"All {len(equivalence)} Adaptive/Fixed median metric ratios within ±10%: `{all_equivalent}`.",
            "Every measured serving step must be one FULL Graph replay; no eager fallback or Graph recapture is accepted.",
            "This is one cold-process round per arm with three in-process samples per cell, not a confidence-interval equivalence proof.",
            "",
        ]
    )
    atomic_text(root / "summary.md", "\n".join(markdown))
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
