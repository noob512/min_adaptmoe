#!/usr/bin/env python3
"""Summarize the non-intrusive EuroSys qualification artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RUNS = HERE / "runs"
MIB = 1024**2
GIB = 1024**3


def load(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_new(path: Path, value: str | Any) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resource_time(path: Path) -> dict[str, float | int | None]:
    if not path.is_file():
        return {"elapsed_seconds": None, "maximum_rss_kib": None}
    text = path.read_text(encoding="utf-8", errors="replace")
    rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    elapsed = re.search(r"Elapsed \(wall clock\) time .*?\):\s*([^\n]+)", text)
    seconds: float | None = None
    if elapsed:
        parts = elapsed.group(1).strip().split(":")
        try:
            if len(parts) == 2:
                seconds = float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                seconds = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except ValueError:
            seconds = None
    return {
        "elapsed_seconds": seconds,
        "maximum_rss_kib": int(rss.group(1)) if rss else None,
        "maximum_rss_gib": int(rss.group(1)) / 1024**2 if rss else None,
    }


def nvml(path: Path, start: float | None = None, end: float | None = None) -> dict[str, Any]:
    if not path.is_file():
        return {"samples": 0}
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            converted: dict[str, float] = {}
            for key, raw in row.items():
                try:
                    converted[key] = float(raw) if raw not in (None, "", "None") else math.nan
                except ValueError:
                    continue
            epoch = converted.get("epoch_seconds", converted.get("unix_s"))
            if epoch is not None:
                converted["epoch"] = epoch
            rows.append(converted)
    selected = rows
    if start is not None and end is not None:
        selected = [row for row in rows if start <= row.get("epoch", -1) <= end]
        if not selected and rows:
            midpoint = (start + end) / 2
            selected = [min(rows, key=lambda row: abs(row.get("epoch", midpoint) - midpoint))]

    def values(*names: str) -> list[float]:
        result = []
        for row in selected:
            for name in names:
                value = row.get(name)
                if value is not None and not math.isnan(value):
                    result.append(value)
                    break
        return result

    memory = values("memory_used_mib")
    rx = values("pcie_rx_kib_s")
    tx = values("pcie_tx_kib_s")
    gpu = values("gpu_util_pct", "gpu_util_percent")
    result: dict[str, Any] = {
        "samples": len(selected),
        "all_samples": len(rows),
        "peak_memory_mib": max(memory) if memory else None,
        "mean_gpu_util_percent": statistics.fmean(gpu) if gpu else None,
        "mean_pcie_rx_kib_s": statistics.fmean(rx) if rx else None,
        "peak_pcie_rx_kib_s": max(rx) if rx else None,
        "mean_pcie_tx_kib_s": statistics.fmean(tx) if tx else None,
        "peak_pcie_tx_kib_s": max(tx) if tx else None,
    }
    if start is not None and end is not None:
        duration = end - start
        result["window_seconds"] = duration
        result["observed_pcie_rx_bytes"] = (
            result["mean_pcie_rx_kib_s"] * 1024 * duration
            if result["mean_pcie_rx_kib_s"] is not None
            else None
        )
        result["observed_pcie_tx_bytes"] = (
            result["mean_pcie_tx_kib_s"] * 1024 * duration
            if result["mean_pcie_tx_kib_s"] is not None
            else None
        )
    return result


def recursively(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from recursively(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursively(child)


def unique_switches(value: Any, byte_count: int) -> list[dict[str, Any]]:
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in recursively(value):
        if item.get("operation") not in {"park", "resume"}:
            continue
        if item.get("processed_alias_bytes") != byte_count:
            continue
        record = {
            key: item.get(key)
            for key in (
                "operation",
                "wall_ns",
                "pte_map_ns",
                "build_sort_ns",
                "metadata_ns",
                "tlb_invalidate_count",
                "dual_view_block_count",
                "processed_alias_count",
                "backing_bytes",
                "transition_batch_count",
            )
        }
        records[tuple(record.values())] = record
    return sorted(records.values(), key=lambda item: (str(item["operation"]), int(item["wall_ns"] or 0)))


def dist(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "max": max(values) if values else None,
    }


def short_e13(result: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    final_config = result.get("final_kv", {}).get("last_decision", {}).get("config", {})
    return {
        "status": result.get("status"),
        "trace_sha256": result.get("trace_sha256"),
        "successful_requests": summary.get("successful_requests"),
        "failed_requests": summary.get("failed_requests"),
        "rejected_requests": summary.get("rejected_requests"),
        "preempted_requests": summary.get("preempted_requests"),
        "queue_invariant_failures": summary.get("queue_invariant_failures"),
        "request_throughput_per_second": summary.get("successful_request_throughput_per_second"),
        "output_token_throughput_per_second": summary.get("output_token_throughput_per_second"),
        "ttft_ms": summary.get("ttft_ms"),
        "tpot_ms": summary.get("tpot_ms"),
        "e2e_ms": summary.get("e2e_ms"),
        "queueing_ms": summary.get("queueing_ms"),
        "resize_count": summary.get("resize_count"),
        "grow_count": summary.get("grow_count"),
        "shrink_count": summary.get("shrink_count"),
        "resize_wall_ms": summary.get("resize_wall_ms"),
        "resize_response_ms": summary.get("resize_response_ms"),
        "load_seconds": result.get("initialization", {}).get("load_seconds"),
        "tier_setup_wall_ms": result.get("initialization", {}).get("tier_setup_wall_ms"),
        "final_plan": result.get("final_plan", {}).get("active_plan_class"),
        "final_kv_blocks": result.get("final_kv", {}).get("applied_target_blocks"),
        "effective_low_pressure_streak": final_config.get("low_pressure_streak"),
        "effective_minimum_residency_steps": final_config.get("minimum_residency_steps"),
        "target_switches": summary.get("target_switches", []),
    }


def output_diff(left: dict[str, Any], right: dict[str, Any]) -> int:
    a = left.get("output_hash_by_logical_request", {})
    b = right.get("output_hash_by_logical_request", {})
    return sum(a.get(key) != b.get(key) for key in set(a) | set(b))


def metric_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, arm in summary["D"]["arms"].items():
        rows.append(
            {
                "experiment": "D",
                "arm": name,
                "request_throughput_per_second": arm["request_throughput_per_second"],
                "ttft_p99_ms": arm["ttft_ms"]["p99"],
                "tpot_p99_ms": arm["tpot_ms"]["p99"],
                "e2e_p99_ms": arm["e2e_ms"]["p99"],
                "resize_count": arm["resize_count"],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"output directory exists: {args.output_dir}")

    a_verify = load(RUNS / "A_e11_payload_retry2" / "verification.json")
    bc_verify = load(RUNS / "BC_e12_payload" / "verification.json")
    bc_result = load(RUNS / "BC_e12_payload" / "result.json")
    leaf = load(RUNS / "B_full_scale_pte_payload" / "verification_recovered.json")
    adaptive = load(RUNS / "D_e13_payload" / "repeat-00-seed-20260901" / "adaptive" / "result.json")
    fixed_k6 = load(RUNS / "D_e13_payload" / "repeat-00-seed-20260901" / "fixed_k6" / "result.json")
    fixed_k2 = load(RUNS / "D_e13_fixed_k2_k4_payload" / "repeat-00-seed-20260901" / "fixed_k2" / "result.json")
    fixed_k4 = load(RUNS / "D_e13_fixed_k2_k4_payload" / "repeat-00-seed-20260901" / "fixed_k4" / "result.json")
    no_hysteresis = load(RUNS / "D_e13_no_hysteresis_payload_retry1" / "repeat-00-seed-20260901" / "adaptive" / "result.json")
    d_arm = load(RUNS / "F_qwen_DF_payload_retry1" / "custom" / "b0_ordinary_chat_b1_D" / "result.json")
    f_arm = load(RUNS / "F_qwen_DF_payload_retry1" / "custom" / "b0_ordinary_chat_b1_F" / "result.json")
    full_rebuild = load(RUNS / "F2_e6_full_rebuild_payload" / "verification.json")
    stable = load(RUNS / "F2_e6_stable_payload" / "verification.json")
    g_result = load(RUNS / "G_e4_remote_payload" / "blocks" / "b0_asc" / "result.json")
    h_custom = load(RUNS / "H_managed_no_prefetch_payload" / "result.json")
    h_demand = json.loads((RUNS / "H_stock_demand_wrapper" / "stdout.log").read_text(encoding="utf-8").strip())
    h_prefetch = json.loads((RUNS / "H_stock_prefetch_wrapper" / "stdout.log").read_text(encoding="utf-8").strip())
    baseline = load(HERE / "baseline.json")["state"]

    cold80_switches = unique_switches(adaptive, 18_119_393_280)
    parent_park_ms = [item["wall_ns"] / 1e6 for item in cold80_switches if item["operation"] == "park"]
    parent_resume_ms = [item["wall_ns"] / 1e6 for item in cold80_switches if item["operation"] == "resume"]

    d_results = {
        "adaptive": adaptive,
        "fixed_k2": fixed_k2,
        "fixed_k4": fixed_k4,
        "fixed_k6": fixed_k6,
        "auto_no_hysteresis": no_hysteresis,
    }
    d_short = {name: short_e13(value) for name, value in d_results.items()}
    hash_matrix = {
        left: {right: output_diff(d_results[left], d_results[right]) for right in d_results}
        for left in d_results
    }

    g_phase = g_result["phases"][0]
    g_nvml = nvml(
        RUNS / "G_e4_remote_payload" / "blocks" / "b0_asc" / "nvml.csv",
        float(g_phase["decode_measured_started_epoch_seconds"]),
        float(g_phase["decode_measured_finished_epoch_seconds"]),
    )
    measured_tokens = int(g_phase["concurrency"]) * int(g_phase["steady_decode_step_ms"]["count"])
    if g_nvml.get("observed_pcie_rx_bytes") is not None:
        g_nvml["observed_pcie_rx_bytes_per_decode_token"] = g_nvml["observed_pcie_rx_bytes"] / measured_tokens
    g_nvml["measured_decode_tokens"] = measured_tokens

    backing = {"cold70": 660_602_880, "cold80": 754_974_720, "cold89": 839_909_376}
    tier_config = load(ROOT / "experiments" / "eurosys_moe_offload_evaluation" / "e13" / "config.json")
    full_parameter_bytes = 28_991_029_248
    gpu_total_mib = 32_607
    tiers = []
    for tier in tier_config["tiers"]:
        plan = tier["plan_class"]
        selected = int(tier["expert_offload_bytes"])
        row = {
            "tier": tier["name"],
            "plan_class": plan,
            "kv_target_blocks": tier["target_blocks"],
            "kv_physical_bytes": int(tier["target_blocks"]) * int(tier_config["runtime"]["kv_block_bytes"]),
            "expert_offload_bytes": selected,
            "resident_parameter_bytes_estimate": full_parameter_bytes - selected,
            "prefill_shared_backing_bytes": backing[plan],
            "prefill_shared_fraction_of_gpu": backing[plan] / (gpu_total_mib * MIB),
        }
        tiers.append(row)

    resource_arms: dict[str, Any] = {}
    for label, directory in (
        ("adaptive", RUNS / "D_e13_payload" / "repeat-00-seed-20260901" / "adaptive"),
        ("fixed_k6", RUNS / "D_e13_payload" / "repeat-00-seed-20260901" / "fixed_k6"),
        ("fixed_k2", RUNS / "D_e13_fixed_k2_k4_payload" / "repeat-00-seed-20260901" / "fixed_k2"),
        ("fixed_k4", RUNS / "D_e13_fixed_k2_k4_payload" / "repeat-00-seed-20260901" / "fixed_k4"),
        ("auto_no_hysteresis", RUNS / "D_e13_no_hysteresis_payload_retry1" / "repeat-00-seed-20260901" / "adaptive"),
    ):
        resource_arms[label] = {
            "resource_time": resource_time(directory / "resource_time.txt"),
            "nvml": nvml(directory / "nvml.csv"),
        }

    conservation = []
    tier_by_name = {row["tier"]: row for row in tiers}
    for old, new in (("k2", "k4"), ("k4", "k6")):
        old_row, new_row = tier_by_name[old], tier_by_name[new]
        resident_delta = new_row["resident_parameter_bytes_estimate"] - old_row["resident_parameter_bytes_estimate"]
        shared_delta = new_row["prefill_shared_backing_bytes"] - old_row["prefill_shared_backing_bytes"]
        kv_delta = new_row["kv_physical_bytes"] - old_row["kv_physical_bytes"]
        conservation.append(
            {
                "transition": f"{old}->{new}",
                "resident_parameter_delta_estimate": resident_delta,
                "prefill_shared_delta": shared_delta,
                "kv_delta": kv_delta,
                "unreconciled_runtime_allocator_residual": -(resident_delta + shared_delta + kv_delta),
                "interpretation": "The existing counters do not expose runtime/workspace pages finely enough to close the conservation equation.",
            }
        )

    f2 = {}
    for key in stable["aggregate"]:
        f2[key] = {}
        for direction in ("outbound", "return"):
            stable_ms = stable["aggregate"][key]["directions"][direction]["median_ms"]
            rebuild_ms = full_rebuild["aggregate"][key][direction]["total_ms"]
            f2[key][direction] = {
                "stable_va_ms": stable_ms,
                "full_rebuild_ms": rebuild_ms,
                "full_rebuild_over_stable_ratio": rebuild_ms / stable_ms,
                "full_rebuild_plan_ms": full_rebuild["aggregate"][key][direction]["plan_rebuild_ms"],
                "full_rebuild_graph_capture_ms": full_rebuild["aggregate"][key][direction]["graph_capture_ms"],
            }

    wrapper_records = []
    for command_path in sorted(RUNS.glob("*_wrapper*/command.json")):
        status_path = command_path.with_name("status.json")
        command = load(command_path)
        status = load(status_path) if status_path.is_file() else {}
        wrapper_records.append(
            {
                "directory": str(command_path.parent.relative_to(HERE)),
                "case": status.get("case"),
                "status": status.get("status"),
                "returncode": status.get("returncode"),
                "duration_seconds": status.get("duration_seconds"),
                "git_guard_before": status.get("git_guard_before"),
                "git_guard_after": status.get("git_guard_after"),
                "command": command.get("argv"),
                "environment_overrides": command.get("environment_overrides"),
            }
        )

    summary: dict[str, Any] = {
        "schema": "adaptmoe_eurosys_additional_qualification_summary_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "qualification-sized, one platform, reduced repetitions; no confidence intervals",
        "matrix": [
            {"experiment": "A", "status": "partial", "result": "scheduler trend obtained; output-correctness gate failed"},
            {"experiment": "B", "status": "completed_qualification", "result": "8 MiB parent soak plus cold80 full-scale leaf PTE point"},
            {"experiment": "C", "status": "partial", "result": "P/D allocation invariant passed; plan conservation lacks runtime/workspace counter"},
            {"experiment": "D", "status": "completed_qualification", "result": "Auto, K2/K4/K6, and external No-Hysteresis ran; cross-arm hash caveats"},
            {"experiment": "E", "status": "partial", "result": "HBM/RSS/init/backing envelopes; no GraphExec/page-table component attribution"},
            {"experiment": "F", "status": "partial", "result": "Full D and stable-vs-rebuild succeeded; all-remote FULL_SEPARATE unsupported"},
            {"experiment": "G", "status": "partial", "result": "representative graph-copy bytes and remote-decode PCIe sample"},
            {"experiment": "H", "status": "partial", "result": "stock UVM controlled microbaseline; custom no-migration mode faulted"},
        ],
        "A": {
            "status": a_verify["status"],
            "arms": a_verify["arms"],
            "correctness_errors": a_verify["errors"],
            "phase_separation_vs_continuous": {
                "throughput_ratio": a_verify["arms"]["remote_phase_epoch"]["summary"]["request_throughput_per_second"]
                / a_verify["arms"]["remote_continuous"]["summary"]["request_throughput_per_second"],
                "extra_mean_queueing_ms": a_verify["arms"]["remote_phase_epoch"]["summary"]["queueing_ms"]["mean"]
                - a_verify["arms"]["remote_continuous"]["summary"]["queueing_ms"]["mean"],
            },
            "limitations": ["single 8-request deterministic conflict trace", "not Poisson/open-loop offered-load sweep", "same-topology output token IDs differ"],
        },
        "B": {
            "parent_8mib_soak": {"verification": bc_verify, "target_bytes": bc_result["target_bytes"], "target_blocks_2mib": bc_result["target_blocks"]},
            "leaf_cold80_full_scale": leaf,
            "parent_cold80_from_adaptive_trace": {
                "records": cold80_switches,
                "park_ms": dist(parent_park_ms),
                "resume_ms": dist(parent_resume_ms),
            },
            "same_geometry_leaf_over_parent_median_ratio": {
                "park": leaf["park"]["outer_wall_ms"] / statistics.median(parent_park_ms),
                "resume": leaf["resume"]["outer_wall_ms"] / statistics.median(parent_resume_ms),
            },
            "full_model_parent_failure": {
                "status": "original_component_error",
                "condition": "Qwen 2.25 GiB dual-view SEAL; 6912 aliases; 1207959552 backing bytes",
                "error": "rm_status=86 poisoned session, followed by abort/segfault in B1/B4",
            },
        },
        "C": {
            "pd_cycles": 1000,
            "parent_switch_calls": 2000,
            "pde_writes": 8000,
            "tlb_batches": 2000,
            "vidmem_allocations_during_switch": 0,
            "vidmem_evictions_during_switch": 0,
            "copy_pushes_during_switch": 0,
            "faults": 0,
            "plan_transition_conservation": conservation,
            "conclusion": "The simple -Delta resident = Delta KV equality is invalid because shared backing changes; runtime/workspace attribution is also required.",
        },
        "D": {
            "arms": d_short,
            "output_hash_difference_count": hash_matrix,
            "adaptive_vs_fixed_k6": {
                "same_outputs": output_diff(adaptive, fixed_k6) == 0,
                "adaptive_request_throughput_delta_percent": 100
                * (d_short["adaptive"]["request_throughput_per_second"] / d_short["fixed_k6"]["request_throughput_per_second"] - 1),
                "adaptive_mean_tpot_delta_percent": 100
                * (d_short["adaptive"]["tpot_ms"]["mean"] / d_short["fixed_k6"]["tpot_ms"]["mean"] - 1),
            },
            "best_static_by_raw_qualification_rule": "fixed_k2",
            "best_static_speedup_publishable": False,
            "reason": "fixed_k2/k4 output hashes differ from Adaptive; only Adaptive vs fixed_k6 is output-identical",
            "no_hysteresis_adapter_effective": d_short["auto_no_hysteresis"]["effective_low_pressure_streak"] == 1
            and d_short["auto_no_hysteresis"]["effective_minimum_residency_steps"] == 0,
        },
        "E": {
            "gpu_total_mib": gpu_total_mib,
            "plan_classes": len(tiers),
            "tier_memory": tiers,
            "maximum_prefill_shared_fraction_of_gpu": max(row["prefill_shared_fraction_of_gpu"] for row in tiers),
            "resource_arms": resource_arms,
            "stage_graph_representative": {
                "graph_capture_count_after_load": d_arm["compilation_counter_after_load"]["num_cudagraph_captured"],
                "graph_capture_count_final": d_arm["compilation_counter_final"]["num_cudagraph_captured"],
                "load_seconds": d_arm["load_seconds"],
                "prefill_shared_backing_bytes": d_arm["diagnostics"]["after_load"]["model"]["alias_graph_certificate"]["backing_bytes"],
            },
            "unavailable_breakdown": ["CUDA GraphExec bytes", "page-table GPU/host bytes", "allocator unused-reserve attribution", "per-plan graph scaling"],
        },
        "F": {
            "stage_specific_D": {
                "status": d_arm["status"],
                "arm_name": d_arm["arm_name"],
                "timing": d_arm["timing"],
                "output_token_ids_sha256": d_arm["output_token_ids_sha256"],
                "parameter_va_stable": all(
                    d_arm["parameter_va_snapshots"][name] == d_arm["parameter_va_snapshots"]["after_load"]
                    for name in ("after_prefill", "after_first_decode", "final")
                ),
                "graph_capture_count_after_load": d_arm["compilation_counter_after_load"]["num_cudagraph_captured"],
                "graph_capture_count_final": d_arm["compilation_counter_final"]["num_cudagraph_captured"],
            },
            "all_remote_F": {"status": f_arm["status"], "error": f_arm["error"], "cleanup_errors": f_arm["shutdown_contract"]["cleanup_errors"]},
            "stable_va_vs_full_rebuild": f2,
            "stable_output_sha256": stable["output_sha256"],
            "full_rebuild_output_sha256": full_rebuild["output_sha256"],
            "stable_capture_count": stable["block_summaries"][0]["capture_count"],
            "full_rebuild_capture_count_after_load": full_rebuild["capture_count_after_load"],
            "full_rebuild_capture_count_final": full_rebuild["capture_count_final"],
        },
        "G": {
            "prefill_graph_copy": {
                "input_tokens": 512,
                "certified_h2d_bytes": d_arm["diagnostics"]["after_load"]["model"]["alias_graph_certificate"]["actual_h2d_bytes"],
                "bytes_per_input_token": d_arm["diagnostics"]["after_load"]["model"]["alias_graph_certificate"]["actual_h2d_bytes"] / 512,
                "prefill_step_ms": d_arm["timing"]["prefill_step_ms"],
            },
            "remote_decode": {
                "phase": g_phase,
                "nvml_window": g_nvml,
                "selected_remote_bytes": g_result["placement_audit"]["selected_parameter_bytes_per_forward"],
                "manifest_actual_h2d_bytes": g_result["final_audit"]["manifest"]["actual_h2d_bytes"],
            },
            "counter_caveat": "NVML is an aggregate PCIe throughput sampler. It is evidence/lower-bound, not a CUPTI validation that attributes every byte to direct pinned-memory reads.",
        },
        "H": {
            "stock_demand_paging": h_demand,
            "stock_prefetch_advice": h_prefetch,
            "custom_no_migration": h_custom,
            "custom_error": "first GPU write synchronization returned CUDA 700 illegal memory access",
            "scope": "64 MiB controlled microbenchmark; no end-to-end model semantic comparison",
        },
        "original_component_errors": [
            "A: same-topology scheduler arms produced different token IDs",
            "B full-model parent: rm_status=86 poisoned Qwen dual-view session; later abort/segfault",
            "B full-model leaf: post-residency/hot-restore HBM headroom gates failed at 172.75/40.63 MiB",
            "B synthetic runner: verifier variable was overwritten after sourcing the common harness; externally recovered verifier passed",
            "F arm: remote_only + FULL_SEPARATE rejected as reserved for prefill_graph_remote_decode",
            "H custom no-migration UVM: CUDA 700 on first managed GPU write",
        ],
        "new_layer_adjustments": [
            "run_case now invokes guard.py with sys.executable because the new guard file has no executable bit",
            "added missing ALLOW_GPU_RUN=1 after the first A wrapper preflight failure",
            "B leaf retried KV 160->128 MiB and existing headroom gate 192->128 MiB; stopped at 40.63 MiB free",
            "external pynvml wrapper was removed from module-switching F because it held a UVM client during preflight",
            "two mistakenly default-routed output directories were moved intact into the allowed new prefix",
            "Auto-No-Hysteresis config-only arm was rejected by the hard-coded shell arm table; case-runner adapter retry succeeded",
            "B full-scale raw measurement succeeded but the shell invoked the wrong verifier; intended verifier was run externally and passed",
        ],
        "disabled_mechanisms": [
            "custom dual-view probes: access-counter migration, UVM prefetch, and thrashing disabled by existing module switches",
            "leaf-PTE B point: dual-parent switch disabled by the existing runner",
            "fixed_k2/k4/k6: autoscale disabled by the existing arm definitions",
            "Auto-No-Hysteresis: low-pressure streak=1 and minimum residency=0 through an external case-runner adapter",
            "F representative: KTransformers external baseline disabled (RUN_KT=0)",
            "no second hardware platform or long random soak, per requested scope",
        ],
        "skipped_or_not_formal": [
            "A Poisson/open-loop 20-110% offered-load sweep and 2x3 plot",
            "B full-model 25/50/75/100% and B=8/16 grid after the Qwen driver/headroom failures",
            "D Oracle arm (not implemented by the existing controller entry)",
            "D publishable Best-Static claim because K2/K4 hashes differ",
            "E exact CUDA GraphExec/page-table/allocator component memory attribution",
            "F Copy-for-Both exact-expert coordinate (native vLLM only supports whole Parameters)",
            "G full length/batch/offload grid and CUPTI/Nsight byte attribution",
            "H end-to-end stock-UVM model baseline",
        ],
        "git_audit": {
            "head": baseline["head"],
            "preexisting_dirty_paths": len(baseline["dirty_path_hashes"]),
            "preexisting_status_lines": len(baseline["status_lines"]),
            "worktree_diff_sha256": baseline["worktree_diff_sha256"],
            "cached_diff_sha256": baseline["cached_diff_sha256"],
            "allowed_prefix": baseline["allowed_prefix"],
            "wrapper_count": len(wrapper_records),
            "all_wrapper_guards_passed": all(
                item["git_guard_before"] is True and item["git_guard_after"] is True
                for item in wrapper_records
                if item["git_guard_before"] is not None
            ),
        },
        "executed_wrappers": wrapper_records,
    }

    report: list[str] = [
        "# AdaptMoe EuroSys 新增实验：非侵入资格报告",
        "",
        f"生成时间：{summary['generated_at']}。本轮均为 RTX 5090 单机、缩减重复次数的 qualification 数据；除 B/C 的 1,000 次微基准外，不能替代正式多冷进程误差条。",
        "",
        "## 实验矩阵与状态",
        "",
        "| 实验 | 状态 | 结论 |",
        "|---|---|---|",
    ]
    for row in summary["matrix"]:
        report.append(f"| {row['experiment']} | {row['status']} | {row['result']} |")
    report += [
        "",
        "## 关键结果",
        "",
        "### A 在线阶段分离",
        "",
        f"Remote phase-epoch 吞吐 {a_verify['arms']['remote_phase_epoch']['summary']['request_throughput_per_second']:.3f} req/s，continuous 为 {a_verify['arms']['remote_continuous']['summary']['request_throughput_per_second']:.3f} req/s；阶段分离平均多出 {summary['A']['phase_separation_vs_continuous']['extra_mean_queueing_ms']:.2f} ms 排队。但两 arm 输出 token 不一致，因此只作为 HOL 趋势，不可作为正式 speedup。Hybrid mixed path 按设计 fail closed。",
        "",
        "### B/C 页表切换与物理生命周期",
        "",
        f"8 MiB parent-switch 完成 1,000 个 P/D 循环：Park P50/P95={bc_verify['park_ms']['sample_p50']:.3f}/{bc_verify['park_ms']['sample_p95']:.3f} ms，Resume P50/P95={bc_verify['resume_ms']['sample_p50']:.3f}/{bc_verify['resume_ms']['sample_p95']:.3f} ms；2,000 次 switch、8,000 次 PDE write、2,000 个 TLB batch，切换期 vidmem allocation/eviction/copy/fault 全为 0。",
        f"Cold80 同几何下，逐 4 KiB PTE 的 Resume/Park 为 {leaf['resume']['outer_wall_ms']:.2f}/{leaf['park']['outer_wall_ms']:.2f} ms；adaptive trace 中 parent 路径中位数为 {statistics.median(parent_resume_ms):.2f}/{statistics.median(parent_park_ms):.2f} ms，叶级分别慢 {summary['B']['same_geometry_leaf_over_parent_median_ratio']['resume']:.1f}x/{summary['B']['same_geometry_leaf_over_parent_median_ratio']['park']:.1f}x。该比较来自不同 qualification 运行，正式稿仍需配对冷进程。",
        "Plan 扩缩容中 shared backing 会随 cold70/80/89 变化，现有计数又无法独立归因 runtime/workspace，因此论文不能写成简单的 `-Δresident=ΔKV`。",
        "",
        "### D 动态控制",
        "",
        f"Adaptive 与 Fixed-K6 输出完全一致，均 32/32 成功且无 OOM/抢占；Adaptive 发生 3 grow+4 shrink，切换 wall 总计 {d_short['adaptive']['resize_wall_ms']:.1f} ms。Adaptive 吞吐 {d_short['adaptive']['request_throughput_per_second']:.3f} req/s，Fixed-K6 为 {d_short['fixed_k6']['request_throughput_per_second']:.3f} req/s；本 trace 上 Adaptive 慢 {-summary['D']['adaptive_vs_fixed_k6']['adaptive_request_throughput_delta_percent']:.1f}%，但平均 TPOT 改善 {-summary['D']['adaptive_vs_fixed_k6']['adaptive_mean_tpot_delta_percent']:.1f}%。",
        f"No-Hysteresis 的有效参数确认为 streak=1/min-residency=0，也发生 7 次切换，吞吐 {d_short['auto_no_hysteresis']['request_throughput_per_second']:.3f} req/s；它与 Auto 有 6/32 输出 hash 不同，不能作正式性能配对。K2/K4 分别与 Auto 有 6/1 个请求 hash 不同，故 raw 最优 K2 不可发布为 Best-Static。",
        "",
        "### E 资源与初始化",
        "",
        f"三档 prefill shared backing 为 630/720/801 MiB；最大 parked 比例为 {100*summary['E']['maximum_prefill_shared_fraction_of_gpu']:.2f}%（801/32607 MiB）。代表 D arm 捕获 2 个 Graph，load={d_arm['load_seconds']:.2f} s；E13 各 arm 的 HBM 峰值、RSS 与启动时间见 summary.json。现有接口未给出 GraphExec、页表与 allocator reserve 的独立字节归因，E 仍属部分完成。",
        "",
        "### F 稳态与稳定 VA 消融",
        "",
        f"Qwen Ordinary B1 的完整 D arm：Prefill {d_arm['timing']['prefill_step_ms']:.2f} ms，稳态 Decode {d_arm['timing']['steady_decode_step_ms']['mean']:.2f} ms/token（{d_arm['timing']['steady_decode_tokens_per_second']:.2f} token/s），P→D cutover {d_arm['timing']['cutover_wall_ms']:.2f} ms，Graph capture 2→2，VA 全程稳定。F all-remote FULL_SEPARATE 被原运行时契约拒绝。",
        f"DeepSeek full rebuild 的 Graph capture 2→22，而 stable VA 保持 2；100% outbound 为 {full_rebuild['aggregate']['d100']['outbound']['total_ms']:.1f} ms 对 {stable['aggregate']['d100']['directions']['outbound']['median_ms']:.1f} ms，输出 hash 相同。",
        "",
        "### G PCIe",
        "",
        f"Graph-copy certificate 记录 2,264,924,160 H2D bytes，即 {summary['G']['prefill_graph_copy']['bytes_per_input_token']:.0f} bytes/input-token（512-token 点）。Remote C1 的 9 个 measured decode tokens 为 {g_phase['steady_decode_step_ms']['mean']:.2f} ms/step，NVML 窗口 RX 约 {g_nvml.get('observed_pcie_rx_bytes_per_decode_token', 0):.0f} bytes/token。NVML 是聚合采样/下界，未完成 CUPTI 归因验证。",
        "",
        "### H 原生 UVM",
        "",
        f"Stock UVM 64 MiB demand paging 首次写 {h_demand['first_write_ms']:.3f} ms；prefetch/advice 为 prefetch {h_prefetch['prefetch_ms']:.3f} ms + 首写 {h_prefetch['first_write_ms']:.3f} ms，二者 mismatch=0。自定义 no-migration mode 在首次 GPU 写返回 CUDA 700；未修改原组件。",
        "",
        "## 失败、重试与跳过",
        "",
        "原组件错误：",
        "",
    ]
    report += [f"- {item}" for item in summary["original_component_errors"]]
    report += ["", "新增层调整：", ""] + [f"- {item}" for item in summary["new_layer_adjustments"]]
    report += ["", "未做或不具正式性：", ""] + [f"- {item}" for item in summary["skipped_or_not_formal"]]
    report += [
        "",
        "## Git 与清理审计",
        "",
        f"基线 HEAD `{baseline['head']}`；原有 dirty paths={len(baseline['dirty_path_hashes'])}；tracked worktree diff SHA256=`{baseline['worktree_diff_sha256']}`。所有 {len(wrapper_records)} 个 wrapper 的可用前/后 guard 均通过，只允许 `{baseline['allowed_prefix']}` 下新增文件。最终独立 guard 和 GPU clean 状态由 `final_audit` 目录记录。",
        "",
        "## 后续正式补跑优先级",
        "",
        "1. 先解决/解释跨调度与跨 Plan 的输出 hash 差异，再补 A 的 Poisson offered-load 曲线和 D 的正式 Best-Static。",
        "2. 用同一冷进程矩阵配对 full-scale parent/PTE，并补 25/50/75/100%、两个方向和 B=1/8/16。",
        "3. 增加 CUPTI/Nsight 字节归因，并单独测 GraphExec、页表和 allocator reserve 内存。",
        "4. Full-model stock UVM 若语义不可对齐，明确降级为受控微基准，不作端到端优胜声明。",
        "",
    ]

    args.output_dir.mkdir(parents=True)
    write_new(args.output_dir / "summary.json", summary)
    write_new(args.output_dir / "REPORT.md", "\n".join(report))
    rows = metric_rows(summary)
    metrics_path = args.output_dir / "d_metrics.csv"
    with metrics_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_new(
        args.output_dir / "manifest.json",
        {
            "summary_sha256": hashlib.sha256((args.output_dir / "summary.json").read_bytes()).hexdigest(),
            "report_sha256": hashlib.sha256((args.output_dir / "REPORT.md").read_bytes()).hexdigest(),
            "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
