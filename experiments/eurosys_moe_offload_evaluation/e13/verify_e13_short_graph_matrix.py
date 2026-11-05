#!/usr/bin/env python3
"""Verify the E13 Fixed-K2/K6 Short B64/B128 FULL_SEPARATE matrix."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any, Callable


ARMS = ("fixed_k2", "fixed_k6")
BATCHES = (64, 128)
CAPTURE_SIZES = (64, 128, 8192)


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
            and config.get("measurement_scope") == "short_graph_matrix"
            and config.get("full_separate_graph") is True
            and tuple(config.get("graph_capture_sizes") or ()) == CAPTURE_SIZES
            and tuple(config.get("short_graph_matrix_batches") or ()) == BATCHES
            and int(config.get("effective_max_num_seqs", -1)) == 128,
            config,
        )
        initialization = result.get("initialization") or {}
        compilation = initialization.get("realized_compilation") or {}
        phase = initialization.get("phase") or {}
        graph_calls = initialization.get("graph_wrapper_calls_after_load") or {}
        check(
            f"{arm}_full_separate_realized",
            compilation.get("mode") == "NONE"
            and compilation.get("cudagraph_mode") == "FULL_SEPARATE"
            and tuple(compilation.get("cudagraph_capture_sizes") or ()) == CAPTURE_SIZES
            and int(graph_calls.get("full_capture", 0)) >= len(CAPTURE_SIZES),
            {"compilation": compilation, "graph_calls": graph_calls},
        )
        check(
            f"{arm}_pd_graph_contract",
            phase.get("phase_mode") == "prefill_graph_remote_decode"
            and phase.get("park_resume_enabled") is True
            and phase.get("alias_graph_captured") is True
            and phase.get("remote_graph_captured") is True,
            phase,
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
            prefill_steps = 1 if batch == 64 else 2
            schedule_steps = prefill_steps + 31
            max_waiting = 0 if batch == 64 else 64
            measurement_ok = len(measurements) == 3 and all(
                int(item.get("batch_size", -1)) == batch
                and int(item.get("prompt_tokens_per_request", -1)) == 128
                and int(item.get("output_tokens_per_request", -1)) == 32
                and int(item.get("prefill_step_count", -1)) == prefill_steps
                and int(item.get("decode_step_ms", {}).get("count", -1)) == 31
                and int(item.get("schedule_step_count", -1)) == schedule_steps
                and int(item.get("max_scheduler_running_requests", -1)) == batch
                and int(item.get("max_scheduler_waiting_requests", -1)) == max_waiting
                for item in measurements
            )
            check(f"{arm}_b{batch}_measurement_contract", measurement_ok, measurements)
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
            check(f"{arm}_b{batch}_all_steps_graph_replay", graph_ok, measurements)
            pd_ok = len(measurements) == 3 and all(
                item.get("phase_after", {}).get("lifecycle_state") == "remote_parked"
                and item.get("phase_after", {}).get("remote_map_active") is True
                and int(item.get("phase_after", {}).get("phase_transition_count", 0))
                > int(item.get("phase_before", {}).get("phase_transition_count", 0))
                for item in measurements
            )
            check(f"{arm}_b{batch}_fast_pd_switch", pd_ok, measurements)
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
        initial_counter = initialization.get("graph_counter_after") or {}
        final_counter = (result.get("final_audit") or {}).get("graph_counter") or {}
        check(
            f"{arm}_no_graph_recapture",
            initial_counter.get("num_cudagraph_captured")
            == final_counter.get("num_cudagraph_captured"),
            {"initial": initial_counter, "final": final_counter},
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
    hashes: dict[str, dict[str, list[str]]] = {}
    for arm, result in cases.items():
        summaries[arm] = {}
        hashes[arm] = {}
        for segment in result.get("segments") or []:
            measurements = segment.get("measurements") or []
            if not measurements:
                continue
            batch = str(int(measurements[0]["batch_size"]))
            summaries[arm][batch] = {
                name: distribution([getter(item) for item in measurements])
                for name, getter in getters.items()
            }
            hashes[arm][batch] = [
                str(item["output_token_ids_sha256"]) for item in measurements
            ]

    comparisons: list[dict[str, Any]] = []
    for batch in BATCHES:
        key = str(batch)
        if not all(key in summaries.get(arm, {}) for arm in ARMS):
            continue
        k2 = summaries["fixed_k2"][key]
        k6 = summaries["fixed_k6"][key]
        output_equal = len(set(hashes["fixed_k2"][key] + hashes["fixed_k6"][key])) == 1
        check(f"b{batch}_output_hash_equal", output_equal, {arm: hashes[arm][key] for arm in ARMS})
        comparisons.append(
            {
                "batch_size": batch,
                "k2_over_k6_request_throughput": float(k2["request_per_second"]["median"])
                / float(k6["request_per_second"]["median"]),
                "k2_over_k6_prefill_latency": float(k2["prefill_ms"]["median"])
                / float(k6["prefill_ms"]["median"]),
                "k2_over_k6_decode_latency": float(k2["decode_step_ms"]["median"])
                / float(k6["decode_step_ms"]["median"]),
                "k2_over_k6_ttft": float(k2["ttft_ms"]["median"])
                / float(k6["ttft_ms"]["median"]),
                "k2_over_k6_tpot": float(k2["tpot_ms"]["median"])
                / float(k6["tpot_ms"]["median"]),
                "k2_over_k6_e2e": float(k2["e2e_ms"]["median"])
                / float(k6["e2e_ms"]["median"]),
            }
        )

    status = "ok" if checks and all(item["ok"] for item in checks) else "error"
    verification = {
        "schema": "eurosys_e13_short_graph_matrix_verification_v1",
        "status": status,
        "checks": checks,
        "summaries": summaries,
        "comparisons": comparisons,
    }
    output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown = [
        "# E13 Fixed-K2/K6 Short B64/B128 FULL_SEPARATE single-round matrix",
        "",
        f"Verifier status: `{status}`. Each arm is one cold process; each cell has one excluded warmup and three in-process samples.",
        "",
        "| B | Arm | req/s | TTFT ms | TPOT ms | E2E ms | Prefill ms | P→D ms | Decode step ms |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
                f"{value['pd_cutover_step_ms']['median']:.2f} | {value['decode_step_ms']['median']:.2f} |"
            )
    markdown.extend(
        [
            "",
            "Every measured serving step must be a FULL Graph replay; the verifier also requires the park/resume P→D transition and no Graph recapture.",
            "",
            "This is a single-round qualification; in-process samples are not independent cold-process repeats.",
            "",
        ]
    )
    atomic_text(root / "summary.md", "\n".join(markdown))
    print(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
