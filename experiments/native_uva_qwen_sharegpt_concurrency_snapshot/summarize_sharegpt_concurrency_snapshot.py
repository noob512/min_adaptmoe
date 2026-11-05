#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path


CASE_PATTERN = re.compile(
    r"pair(?P<pair>\d+)_g(?P<gib>\d+)_(?P<label>baseline|offload)$"
)
PHASE_PATTERN = re.compile(r"sharegpt_c(?P<concurrency>\d+)$")
BLOCK_SIZE = 16
EXPECTED_OFFLOAD_GIB = {2, 6, 10, 14, 18}
EXPECTED_CONCURRENCIES = {1, 2, 3, 4, 8, 16, 24, 32}
EXPECTED_OUTPUT_TOKENS_PER_REQUEST = 64


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_hash(phase: dict) -> str:
    payload = [row.get("output_token_ids", []) for row in phase.get("requests", [])]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def pcie_rx_peak_mb_s(path: Path) -> int | None:
    values: list[int] = []
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[0].isdigit() and fields[3].isdigit():
            values.append(int(fields[3]))
    return max(values) if values else None


def scheduling_log_checks(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    running_values = [int(value) for value in re.findall(r"Running:\s*(\d+)", text)]
    waiting_values = [int(value) for value in re.findall(r"Waiting:\s*(\d+)", text)]
    kv_matches = re.findall(r"GPU KV cache size:\s*([\d,]+) tokens", text)
    lowered = text.lower()
    return {
        "scheduler_status_samples": max(len(running_values), len(waiting_values)),
        "max_logged_running_requests": max(running_values) if running_values else None,
        "max_logged_waiting_requests": max(waiting_values) if waiting_values else None,
        "preemption_keyword_seen": any(
            keyword in lowered for keyword in ("preempt", "preemption", "swapped")
        ),
        "kv_cache_tokens_reported": (
            int(kv_matches[-1].replace(",", "")) if kv_matches else None
        ),
    }


def phase_scheduling_log_checks(path: Path) -> dict[str, dict]:
    checks: dict[str, dict] = {}
    if not path.exists():
        return checks
    current_phase: str | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        begin_match = re.search(r"UVM_PHASE_BEGIN name=([^ ]+)", line)
        if begin_match:
            current_phase = begin_match.group(1)
            checks.setdefault(
                current_phase,
                {
                    "scheduler_status_samples": 0,
                    "max_logged_running_requests": None,
                    "max_logged_waiting_requests": None,
                    "max_logged_kv_cache_usage_pct": None,
                },
            )
            continue
        if current_phase is None:
            continue
        running_match = re.search(r"Running:\s*(\d+)", line)
        waiting_match = re.search(r"Waiting:\s*(\d+)", line)
        usage_match = re.search(r"GPU KV cache usage:\s*([\d.]+)%", line)
        if running_match or waiting_match or usage_match:
            phase_check = checks[current_phase]
            phase_check["scheduler_status_samples"] += 1
            if running_match:
                running = int(running_match.group(1))
                previous = phase_check["max_logged_running_requests"]
                phase_check["max_logged_running_requests"] = (
                    running if previous is None else max(previous, running)
                )
            if waiting_match:
                waiting = int(waiting_match.group(1))
                previous = phase_check["max_logged_waiting_requests"]
                phase_check["max_logged_waiting_requests"] = (
                    waiting if previous is None else max(previous, waiting)
                )
            if usage_match:
                usage = float(usage_match.group(1))
                previous = phase_check["max_logged_kv_cache_usage_pct"]
                phase_check["max_logged_kv_cache_usage_pct"] = (
                    usage if previous is None else max(previous, usage)
                )
        if re.search(r"UVM_PHASE_END name=", line):
            current_phase = None
    return checks


def rounded_kv_tokens(phase: dict) -> int:
    total = 0
    for request in phase.get("requests", []):
        sequence_tokens = int(request.get("prompt_tokens", 0)) + int(
            request.get("output_tokens", 0)
        )
        total += math.ceil(sequence_tokens / BLOCK_SIZE) * BLOCK_SIZE
    return total


def first_observed(rows: list[dict], metric: str, threshold_pct: float) -> int | None:
    for row in rows:
        if row[metric] >= threshold_pct:
            return int(row["concurrency"])
    return None


def first_confirmed(rows: list[dict], metric: str, threshold_pct: float) -> int | None:
    for row, next_row in zip(rows, rows[1:]):
        if row[metric] >= threshold_pct and next_row[metric] >= threshold_pct:
            return int(row["concurrency"])
    return None


def first_persistent(rows: list[dict], metric: str, threshold_pct: float) -> int | None:
    for index, row in enumerate(rows):
        tail = rows[index:]
        if len(tail) >= 2 and all(item[metric] >= threshold_pct for item in tail):
            return int(row["concurrency"])
    return None


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} RUN_ROOT")
    root = Path(sys.argv[1]).resolve()
    cases: list[dict] = []

    for case_root in sorted(root.iterdir()):
        match = CASE_PATTERN.fullmatch(case_root.name)
        if match is None:
            continue
        result_path = case_root / "native_uva" / "result.json"
        if not result_path.exists():
            continue
        result = read_json(result_path)
        config = result.get("config", {})
        label = match.group("label")
        requested_offload_gib = int(match.group("gib"))
        manifest = [
            row
            for row in read_jsonl(
                case_root / "native_uva" / "cpu_offload_manifest.jsonl"
            )
            if row.get("event") == "cpu_offload_parameter"
        ]
        expert_manifest = [
            row
            for row in manifest
            if "experts" in str(row.get("full_parameter_name", "")).lower()
        ]
        expert_layers: dict[int, int] = {}
        for row in expert_manifest:
            layer_match = re.search(
                r"model\.layers\.(\d+)\.mlp\.experts\.",
                str(row.get("full_parameter_name", "")),
            )
            if layer_match:
                layer = int(layer_match.group(1))
                expert_layers[layer] = expert_layers.get(layer, 0) + 1

        run_log_path = case_root / "native_uva" / "run.log"
        phase_log_checks = phase_scheduling_log_checks(run_log_path)
        phases: list[dict] = []
        for phase in result.get("phases", []):
            phase_match = PHASE_PATTERN.fullmatch(str(phase.get("phase", "")))
            if phase_match is None:
                continue
            concurrency = int(phase_match.group("concurrency"))
            output_per_request = int(phase["output_tokens_per_request"])
            mean_ttft_ms = float(phase["mean_ttft_ms"])
            mean_tpot_ms = float(phase["mean_tpot_ms"])
            requests = phase.get("requests", [])
            estimated_kv = rounded_kv_tokens(phase)
            per_phase_log = phase_log_checks.get(
                str(phase["phase"]),
                {
                    "scheduler_status_samples": 0,
                    "max_logged_running_requests": None,
                    "max_logged_waiting_requests": None,
                    "max_logged_kv_cache_usage_pct": None,
                },
            )
            phases.append(
                {
                    "concurrency": concurrency,
                    "request_count": int(phase["request_count"]),
                    "completed_requests": int(phase["completed_requests"]),
                    "prompt_tokens": int(phase["prompt_tokens"]),
                    "output_tokens": int(phase["output_tokens"]),
                    "output_tokens_per_request": output_per_request,
                    "elapsed_seconds": float(phase["elapsed_seconds"]),
                    "output_tok_s": float(phase["output_tokens_per_second"]),
                    "mean_ttft_ms": mean_ttft_ms,
                    "p99_ttft_ms": float(phase["p99_ttft_ms"]),
                    "mean_tpot_ms": mean_tpot_ms,
                    "p99_tpot_ms": float(phase["p99_tpot_ms"]),
                    "mean_e2e_latency_ms_derived": (
                        mean_ttft_ms + (output_per_request - 1) * mean_tpot_ms
                    ),
                    "estimated_peak_kv_tokens_rounded_16": estimated_kv,
                    "request_prompt_token_min": min(
                        int(row["prompt_tokens"]) for row in requests
                    ),
                    "request_prompt_token_max": max(
                        int(row["prompt_tokens"]) for row in requests
                    ),
                    "unique_request_tpot_values": len(
                        {round(float(row["tpot_ms"]), 9) for row in requests}
                    ),
                    **per_phase_log,
                    "output_sha256": output_hash(phase),
                }
            )

        fixture_path = case_root / "sharegpt_256_prompt320_360.json"
        fixture_meta_path = case_root / "fixture_metadata.json"
        returncode_path = case_root / "native_uva" / "returncode.txt"
        log_checks = scheduling_log_checks(run_log_path)
        kv_capacity = log_checks["kv_cache_tokens_reported"]
        cases.append(
            {
                "pair": int(match.group("pair")),
                "requested_offload_gib": requested_offload_gib,
                "label": label,
                "scan_order": (case_root / "scan_order.txt").read_text().strip(),
                "returncode": (
                    int(returncode_path.read_text().strip())
                    if returncode_path.exists()
                    else None
                ),
                "status": result.get("status"),
                "configured_cpu_offload_gib": float(
                    config.get("cpu_offload_gb", -1.0)
                ),
                "core_config": {
                    key: config.get(key)
                    for key in (
                        "model",
                        "kv_cache_memory_bytes",
                        "max_model_len",
                        "max_num_seqs",
                        "max_num_batched_tokens",
                        "enable_prefix_caching",
                    )
                },
                "load_seconds": float(result["load_seconds"]),
                "gpu_after_load_used_mib": int(
                    result.get("gpu_after_load", {}).get("used_mib", -1)
                ),
                "fixture_sha256": file_sha256(fixture_path),
                "fixture_metadata": read_json(fixture_meta_path),
                "offload_parameter_records": len(manifest),
                "offload_bytes": sum(int(row.get("bytes", 0)) for row in manifest),
                "expert_parameter_records": len(expert_manifest),
                "expert_offload_bytes": sum(
                    int(row.get("bytes", 0)) for row in expert_manifest
                ),
                "fully_offloaded_expert_layers": sorted(
                    layer for layer, count in expert_layers.items() if count == 4
                ),
                "partial_expert_layers": sorted(
                    layer for layer, count in expert_layers.items() if count != 4
                ),
                "all_offload_records_uva_pinned": bool(manifest)
                and all(
                    row.get("offload_backend") == "uva"
                    and bool(row.get("uva_offloading"))
                    and bool(row.get("cpu_data_pinned"))
                    for row in manifest
                ),
                "process_pcie_rx_peak_mb_s": pcie_rx_peak_mb_s(
                    case_root / "dmon_pcie.csv"
                ),
                **log_checks,
                "all_phase_kv_estimates_fit_reported_capacity": bool(phases)
                and kv_capacity is not None
                and all(
                    phase["estimated_peak_kv_tokens_rounded_16"] <= kv_capacity
                    for phase in phases
                ),
                "all_expected_phases_complete": (
                    {phase["concurrency"] for phase in phases}
                    == EXPECTED_CONCURRENCIES
                    and all(
                        phase["request_count"] == phase["concurrency"]
                        and phase["completed_requests"] == phase["concurrency"]
                        and phase["output_tokens_per_request"]
                        == EXPECTED_OUTPUT_TOKENS_PER_REQUEST
                        and phase["output_tokens"]
                        == phase["concurrency"]
                        * EXPECTED_OUTPUT_TOKENS_PER_REQUEST
                        for phase in phases
                    )
                ),
                "phases": sorted(phases, key=lambda row: row["concurrency"]),
            }
        )

    by_key = {(row["pair"], row["label"]): row for row in cases}
    snapshots: list[dict] = []
    for pair in sorted({row["pair"] for row in cases}):
        baseline = by_key.get((pair, "baseline"))
        offload = by_key.get((pair, "offload"))
        if baseline is None or offload is None:
            continue
        base_phases = {row["concurrency"]: row for row in baseline["phases"]}
        offload_phases = {row["concurrency"]: row for row in offload["phases"]}
        phase_sets_match = (
            set(base_phases) == set(offload_phases) == EXPECTED_CONCURRENCIES
        )
        rows: list[dict] = []
        for concurrency in sorted(set(base_phases) & set(offload_phases)):
            base = base_phases[concurrency]
            remote = offload_phases[concurrency]
            rows.append(
                {
                    "concurrency": concurrency,
                    "prompt_tokens": base["prompt_tokens"],
                    "estimated_peak_kv_tokens_rounded_16": base[
                        "estimated_peak_kv_tokens_rounded_16"
                    ],
                    "baseline_output_tok_s": base["output_tok_s"],
                    "offload_output_tok_s": remote["output_tok_s"],
                    "throughput_slowdown_pct": (
                        1.0 - remote["output_tok_s"] / base["output_tok_s"]
                    )
                    * 100.0,
                    "baseline_mean_ttft_ms": base["mean_ttft_ms"],
                    "offload_mean_ttft_ms": remote["mean_ttft_ms"],
                    "ttft_slowdown_pct": (
                        remote["mean_ttft_ms"] / base["mean_ttft_ms"] - 1.0
                    )
                    * 100.0,
                    "baseline_mean_tpot_ms": base["mean_tpot_ms"],
                    "offload_mean_tpot_ms": remote["mean_tpot_ms"],
                    "tpot_slowdown_pct": (
                        remote["mean_tpot_ms"] / base["mean_tpot_ms"] - 1.0
                    )
                    * 100.0,
                    "baseline_mean_e2e_latency_ms_derived": base[
                        "mean_e2e_latency_ms_derived"
                    ],
                    "offload_mean_e2e_latency_ms_derived": remote[
                        "mean_e2e_latency_ms_derived"
                    ],
                    "fixture_inputs_match": (
                        base["prompt_tokens"] == remote["prompt_tokens"]
                        and base["request_prompt_token_min"]
                        == remote["request_prompt_token_min"]
                        and base["request_prompt_token_max"]
                        == remote["request_prompt_token_max"]
                    ),
                    "outputs_match": base["output_sha256"]
                    == remote["output_sha256"],
                    "baseline_scheduler_status_samples": base[
                        "scheduler_status_samples"
                    ],
                    "baseline_max_logged_running_requests": base[
                        "max_logged_running_requests"
                    ],
                    "baseline_max_logged_waiting_requests": base[
                        "max_logged_waiting_requests"
                    ],
                    "baseline_max_logged_kv_cache_usage_pct": base[
                        "max_logged_kv_cache_usage_pct"
                    ],
                    "offload_scheduler_status_samples": remote[
                        "scheduler_status_samples"
                    ],
                    "offload_max_logged_running_requests": remote[
                        "max_logged_running_requests"
                    ],
                    "offload_max_logged_waiting_requests": remote[
                        "max_logged_waiting_requests"
                    ],
                    "offload_max_logged_kv_cache_usage_pct": remote[
                        "max_logged_kv_cache_usage_pct"
                    ],
                }
            )

        actual_bytes = offload["offload_bytes"]
        expert_bytes = offload["expert_offload_bytes"]
        snapshots.append(
            {
                "pair": pair,
                "requested_offload_gib": offload["requested_offload_gib"],
                "scan_order": offload["scan_order"],
                "actual_offload_bytes": actual_bytes,
                "actual_offload_gib": actual_bytes / (1024**3),
                "expert_offload_bytes": expert_bytes,
                "expert_offload_gib": expert_bytes / (1024**3),
                "expert_fraction_pct": (
                    expert_bytes / actual_bytes * 100.0 if actual_bytes else None
                ),
                "offload_parameter_records": offload["offload_parameter_records"],
                "expert_parameter_records": offload["expert_parameter_records"],
                "fully_offloaded_expert_layers": offload[
                    "fully_offloaded_expert_layers"
                ],
                "partial_expert_layers": offload["partial_expert_layers"],
                "all_offload_records_uva_pinned": offload[
                    "all_offload_records_uva_pinned"
                ],
                "baseline_gpu_after_load_used_mib": baseline[
                    "gpu_after_load_used_mib"
                ],
                "offload_gpu_after_load_used_mib": offload[
                    "gpu_after_load_used_mib"
                ],
                "offload_process_pcie_rx_peak_mb_s": offload[
                    "process_pcie_rx_peak_mb_s"
                ],
                "fixture_match": baseline["fixture_sha256"]
                == offload["fixture_sha256"],
                "phase_sets_match": phase_sets_match,
                "all_inputs_match": all(row["fixture_inputs_match"] for row in rows),
                "all_outputs_match": all(row["outputs_match"] for row in rows),
                "first_observed_throughput_drop_10pct": first_observed(
                    rows, "throughput_slowdown_pct", 10.0
                ),
                "first_confirmed_throughput_drop_10pct": first_confirmed(
                    rows, "throughput_slowdown_pct", 10.0
                ),
                "first_persistent_throughput_drop_10pct": first_persistent(
                    rows, "throughput_slowdown_pct", 10.0
                ),
                "first_observed_tpot_drop_10pct": first_observed(
                    rows, "tpot_slowdown_pct", 10.0
                ),
                "first_confirmed_tpot_drop_10pct": first_confirmed(
                    rows, "tpot_slowdown_pct", 10.0
                ),
                "gray_concurrencies_throughput_5_to_10pct": [
                    row["concurrency"]
                    for row in rows
                    if 5.0 <= row["throughput_slowdown_pct"] < 10.0
                ],
                "results": rows,
            }
        )

    fixture_hashes = [row["fixture_sha256"] for row in cases]
    observed_offload_gib = {
        row["requested_offload_gib"] for row in cases if row["label"] == "offload"
    }
    observed_pairs = {row["pair"] for row in cases}
    core_configs = {
        json.dumps(row["core_config"], sort_keys=True) for row in cases
    }
    complete_matrix = (
        len(cases) == 10
        and len(snapshots) == 5
        and observed_pairs == {1, 2, 3, 4, 5}
        and observed_offload_gib == EXPECTED_OFFLOAD_GIB
        and all(row["all_expected_phases_complete"] for row in cases)
    )
    summary = {
        "run_root": str(root),
        "source_selection": (
            read_json(root / "sharegpt_human_first_selection_metadata.json")
            if (root / "sharegpt_human_first_selection_metadata.json").exists()
            else None
        ),
        "method": {
            "kind": (
                "one cold baseline/offload process pair per offload level; "
                "single-burst offline LLM.generate screening snapshot"
            ),
            "workload": (
                "real ShareGPT human/user first-turn prompts filtered to "
                "300-320 Qwen tokens; controlled 64-token outputs"
            ),
            "submitted_concurrencies": sorted(
                {
                    row["concurrency"]
                    for snapshot in snapshots
                    for row in snapshot["results"]
                }
            ),
            "primary_metric": "phase output tokens per second including prefill and decode",
            "observed_boundary": "first tested concurrency with >=10% slowdown",
            "confirmed_boundary": (
                "first tested concurrency with >=10% slowdown whose next "
                "larger tested concurrency also has >=10% slowdown"
            ),
            "gray_zone": "5% <= slowdown < 10%",
            "no_confidence_intervals": True,
            "concurrency_caveat": (
                "phase count is the number of requests submitted in one burst, "
                "not an HTTP closed-loop sustained-concurrency client"
            ),
        },
        "cases": cases,
        "snapshots": sorted(snapshots, key=lambda row: row["requested_offload_gib"]),
        "integrity": {
            "complete_expected_matrix": complete_matrix,
            "case_count": len(cases),
            "pair_count": len(snapshots),
            "all_cases_ok": complete_matrix
            and all(row["status"] == "ok" and row["returncode"] == 0 for row in cases),
            "all_expected_phases_complete": complete_matrix,
            "all_fixtures_identical_across_cases": len(fixture_hashes) == 10
            and len(set(fixture_hashes)) == 1,
            "all_fixtures_match_within_pair": bool(snapshots)
            and all(row["fixture_match"] for row in snapshots),
            "all_inputs_match_within_pair": bool(snapshots)
            and all(row["all_inputs_match"] for row in snapshots),
            "all_outputs_match_within_pair": bool(snapshots)
            and all(row["all_outputs_match"] for row in snapshots),
            "all_offload_records_uva_pinned": bool(snapshots)
            and all(row["all_offload_records_uva_pinned"] for row in snapshots),
            "all_phase_sets_match_within_pair": complete_matrix
            and all(row["phase_sets_match"] for row in snapshots),
            "all_baseline_parameter_manifests_empty": complete_matrix
            and all(
                row["offload_parameter_records"] == 0
                for row in cases
                if row["label"] == "baseline"
            ),
            "all_offload_parameter_manifests_nonempty": complete_matrix
            and all(
                row["offload_parameter_records"] > 0
                for row in cases
                if row["label"] == "offload"
            ),
            "all_cpu_offload_configs_match_labels": complete_matrix
            and all(
                (
                    row["configured_cpu_offload_gib"] == 0.0
                    if row["label"] == "baseline"
                    else row["configured_cpu_offload_gib"]
                    == float(row["requested_offload_gib"])
                )
                for row in cases
            ),
            "all_core_configs_identical": complete_matrix
            and len(core_configs) == 1,
            "all_kv_cache_capacities_13104_tokens": complete_matrix
            and all(row["kv_cache_tokens_reported"] == 13104 for row in cases),
            "all_phase_kv_estimates_fit_reported_capacity": bool(cases)
            and all(row["all_phase_kv_estimates_fit_reported_capacity"] for row in cases),
            "all_cases_have_scheduler_status_samples": bool(cases)
            and all(row["scheduler_status_samples"] > 0 for row in cases),
            "no_positive_logged_waiting_requests": bool(cases)
            and all(row["scheduler_status_samples"] > 0 for row in cases)
            and all(row["max_logged_waiting_requests"] == 0 for row in cases),
            "no_preemption_keyword_seen": bool(cases)
            and not any(row["preemption_keyword_seen"] for row in cases),
        },
    }
    output_path = root / "snapshot_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
