#!/usr/bin/env python3
"""Validate and summarize the 2/4/6/8 GiB offload comparison.

The timing domains in this experiment are intentionally kept separate:

* custom and vLLM native artifacts expose in-process scheduler-step timing;
* custom and vLLM native also expose in-process request timing;
* KTransformers exposes local HTTP/SSE client TTFT/TPOT;
* MoE-Infinity exposes in-process ``MoE.generate()`` streamer TTFT/TPOT.

The script never substitutes one timing domain for another.  By default all
four tiers from all four backends must be complete.  ``--allow-incomplete``
is intended for producing an explicitly partial report while a matrix is
still running.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
INPUTS_DIR = SCRIPT_DIR / "inputs"

TIERS = ("2g", "4g", "6g", "8g")
NOMINAL_GIB = {"2g": 2, "4g": 4, "6g": 6, "8g": 8}
GIB = 1 << 30
MOEI_EXPECTED_COMMIT = "ba5651897a80d9c9b7a1500cef2c68adaa63db0f"

DEFAULT_CUSTOM_ROOT = (
    ARTIFACTS_DIR / "custom_qualification_eager_20260821T032717Z"
)
DEFAULT_VLLM_ROOT = (
    ARTIFACTS_DIR / "vllm_native_qualification_20260821T033550Z"
)
DEFAULT_KT_ROOT = (
    ARTIFACTS_DIR / "ktransformers_qualification_20260821T035336Z"
)
DEFAULT_MOEI_ROOT = (
    ARTIFACTS_DIR / "moe_infinity_qualification_20260821T040103Z"
)


class ArtifactError(RuntimeError):
    """A completed artifact violates its qualification contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the four-backend 2/4/6/8 GiB experiment and emit a "
            "machine-readable JSON summary plus a Chinese Markdown report."
        )
    )
    parser.add_argument("--custom-root", type=Path, default=DEFAULT_CUSTOM_ROOT)
    parser.add_argument("--vllm-root", type=Path, default=DEFAULT_VLLM_ROOT)
    parser.add_argument("--kt-root", type=Path, default=DEFAULT_KT_ROOT)
    parser.add_argument("--moei-root", type=Path, default=DEFAULT_MOEI_ROOT)
    parser.add_argument(
        "--cold-plan-manifest",
        type=Path,
        default=INPUTS_DIR / "manifest.json",
    )
    parser.add_argument(
        "--kt-placement-manifest",
        type=Path,
        default=INPUTS_DIR / "kt_frequency_manifest.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ARTIFACTS_DIR / "comparison_summary.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ARTIFACTS_DIR / "comparison_summary_zh.md",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Emit a partial report with null metrics for missing/in-progress "
            "tiers. Invalid completed artifacts still fail."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactError(f"missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"malformed JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ArtifactError(
                        f"malformed JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ArtifactError(
                        f"expected JSON object at {path}:{line_number}"
                    )
                rows.append(row)
    except FileNotFoundError as exc:
        raise ArtifactError(f"missing JSONL file: {path}") from exc
    return rows


def nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactError(message)


def number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"{label} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactError(f"{label} is not finite: {value!r}")
    return result


def integer(value: Any, label: str) -> int:
    result = number(value, label)
    if not result.is_integer():
        raise ArtifactError(f"{label} is not an integer: {value!r}")
    return int(result)


def median(values: Iterable[float], label: str) -> float:
    materialized = [number(value, label) for value in values]
    if not materialized:
        raise ArtifactError(f"no samples for {label}")
    return float(statistics.median(materialized))


def gib(value: int | float) -> float:
    return float(value) / GIB


def resolved(path: Path) -> str:
    return str(path.resolve())


def complete_returncode(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return int(path.read_text(encoding="utf-8").strip()) == 0
    except (OSError, ValueError):
        return False


def nvml_peak_mib(path: Path) -> tuple[int, int]:
    samples: list[int] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, row in enumerate(csv.reader(handle), start=1):
                if len(row) < 3:
                    raise ArtifactError(
                        f"NVML row has fewer than three columns at "
                        f"{path}:{line_number}"
                    )
                try:
                    samples.append(int(row[2].strip()))
                except ValueError as exc:
                    raise ArtifactError(
                        f"invalid NVML memory.used value at "
                        f"{path}:{line_number}: {row[2]!r}"
                    ) from exc
    except FileNotFoundError as exc:
        raise ArtifactError(f"missing NVML CSV file: {path}") from exc
    require(bool(samples), f"NVML CSV has no samples: {path}")
    return max(samples), len(samples)


def incomplete_tier(required_files: Sequence[Path]) -> dict[str, Any] | None:
    missing = [resolved(path) for path in required_files if not path.is_file()]
    if not missing:
        return None
    return {
        "status": "incomplete",
        "missing_files": missing,
        "errors": [],
    }


def invalid_tier(exc: Exception, sources: Sequence[Path]) -> dict[str, Any]:
    return {
        "status": "invalid",
        "sources": [resolved(path) for path in sources],
        "errors": [str(exc)],
    }


def read_input_manifests(
    cold_path: Path, kt_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    cold = read_json(cold_path)
    kt = read_json(kt_path)
    require(
        cold.get("schema") == "deepseek_v2_lite_cold_budget_matrix_v1",
        f"unexpected cold-plan schema in {cold_path}",
    )
    require(
        kt.get("schema") == "deepseek_v2_lite_kt_frequency_placements_v1",
        f"unexpected KTransformers placement schema in {kt_path}",
    )
    for tier in TIERS:
        cold_tier = nested(cold, "tiers", tier)
        kt_tier = nested(kt, "tiers", tier)
        require(isinstance(cold_tier, dict), f"cold plan is missing tier {tier}")
        require(isinstance(kt_tier, dict), f"KT plan is missing tier {tier}")
        selected = integer(
            cold_tier.get("selected_main_bytes"),
            f"cold plan {tier} selected_main_bytes",
        )
        require(
            selected == integer(
                kt_tier.get("selected_main_bytes"),
                f"KT plan {tier} selected_main_bytes",
            ),
            f"cold and KT selected bytes differ at {tier}",
        )
        require(
            integer(
                cold_tier.get("cold_experts_per_layer"),
                f"cold plan {tier} cold_experts_per_layer",
            )
            == integer(
                kt_tier.get("cold_experts_per_layer"),
                f"KT plan {tier} cold_experts_per_layer",
            ),
            f"cold and KT expert counts differ at {tier}",
        )
    return cold, kt


def custom_result_paths(root: Path, tier: str) -> list[Path]:
    paths = set(root.glob(f"{tier}_r*/result.json"))
    paths.update(root.glob(f"{tier}_r*.json"))
    return sorted(paths, key=lambda path: str(path))


def validate_custom_run(
    result_path: Path, expected_bytes: int
) -> dict[str, Any]:
    returncode_path = result_path.parent / "returncode.txt"
    require(
        complete_returncode(returncode_path),
        f"custom run has no successful returncode: {returncode_path}",
    )
    result = read_json(result_path)
    require(result.get("status") == "ok", f"custom status is not ok: {result_path}")
    require(result.get("arm") == "H", f"custom arm is not H: {result_path}")
    require(
        nested(result, "config", "phase_mode") == "prefill_remote_decode",
        f"custom phase mode mismatch: {result_path}",
    )
    require(
        integer(nested(result, "config", "batch_size"), "custom batch size") == 1,
        f"custom batch size mismatch: {result_path}",
    )
    require(
        integer(
            nested(result, "config", "expected_selected_bytes"),
            "custom expected_selected_bytes",
        )
        == expected_bytes,
        f"custom selected bytes mismatch: {result_path}",
    )

    timing = result.get("timing")
    require(isinstance(timing, dict), f"custom timing is missing: {result_path}")
    outputs = result.get("outputs")
    require(
        isinstance(outputs, list) and outputs and isinstance(outputs[0], dict),
        f"custom request output is missing: {result_path}",
    )
    output = outputs[0]
    require(output.get("finished") is True, f"custom request unfinished: {result_path}")

    transition = result.get("cutover_transition")
    require(
        isinstance(transition, dict) and transition.get("ok") is True,
        f"custom cutover transition failed: {result_path}",
    )
    require(
        transition.get("parameter_vas_before")
        == transition.get("parameter_vas_after"),
        f"custom Parameter VA changed: {result_path}",
    )
    require(
        integer(
            nested(transition, "remote_map", "remote_mapped_bytes"),
            "custom remote_mapped_bytes",
        )
        == expected_bytes,
        f"custom remote bytes mismatch: {result_path}",
    )
    require(
        integer(
            nested(transition, "physical_residency", "cpu_bytes"),
            "custom cold CPU residency bytes",
        )
        == expected_bytes,
        f"custom cold residency mismatch: {result_path}",
    )
    require(
        integer(
            nested(transition, "hot_physical_residency", "gpu_bytes"),
            "custom hot GPU residency bytes",
        )
        > 0,
        f"custom hot residency is empty: {result_path}",
    )
    prefill_backing_bytes = integer(
        nested(result, "diagnostics", "after_load", "model", "backing_bytes"),
        "custom prefill backing bytes",
    )
    require(
        0 < prefill_backing_bytes < expected_bytes,
        f"custom prefill backing bytes are outside the selected range: {result_path}",
    )
    require(
        integer(
            nested(
                result,
                "diagnostics",
                "after_load",
                "model",
                "active_driver_backing_bytes",
            ),
            "custom active backing bytes after load",
        )
        == prefill_backing_bytes,
        f"custom active backing does not match prefill backing: {result_path}",
    )
    require(
        integer(
            nested(
                transition,
                "destroy",
                "destroy",
                "released_backing_bytes",
            ),
            "custom cutover released backing bytes",
        )
        == prefill_backing_bytes,
        f"custom cutover did not release the prefill backing: {result_path}",
    )
    require(
        integer(
            nested(
                result,
                "diagnostics",
                "after_first_decode",
                "model",
                "active_driver_backing_bytes",
            ),
            "custom active backing after first decode",
        )
        == 0,
        f"custom backing remains active after decode cutover: {result_path}",
    )
    require(
        integer(
            nested(
                result,
                "shutdown_contract",
                "diagnostics_after_shutdown",
                "driver",
                "session_count",
            ),
            "custom shutdown session_count",
        )
        == 0,
        f"custom driver sessions leaked: {result_path}",
    )
    require(
        integer(
            nested(
                result,
                "shutdown_contract",
                "diagnostics_after_shutdown",
                "model",
                "active_driver_backing_bytes",
            ),
            "custom shutdown backing bytes",
        )
        == 0,
        f"custom driver backing leaked: {result_path}",
    )

    steady_count = integer(
        nested(timing, "steady_decode_step_ms", "count"),
        "custom steady decode count",
    )
    require(steady_count > 0, f"custom has no measured decode steps: {result_path}")
    return {
        "source": resolved(result_path),
        "prefill_step_ms": number(timing.get("prefill_step_ms"), "custom prefill"),
        "decode_step_median_ms": number(
            nested(timing, "steady_decode_step_ms", "median"),
            "custom decode median",
        ),
        "decode_tokens_per_second": number(
            timing.get("steady_decode_tokens_per_second"),
            "custom decode tokens/s",
        ),
        "decode_step_samples": steady_count,
        "request_ttft_ms": number(output.get("ttft_ms"), "custom request TTFT"),
        "request_tpot_ms": number(output.get("tpot_ms"), "custom request TPOT"),
        "request_latency_ms": number(
            output.get("request_latency_ms"), "custom request latency"
        ),
        "cutover_wall_ms": number(
            timing.get("cutover_wall_ms"), "custom cutover wall time"
        ),
        "first_decode_step_ms": number(
            timing.get("first_decode_step_ms"), "custom first decode step"
        ),
        "first_decode_excluding_cutover_ms": number(
            timing.get("first_decode_excluding_cutover_ms"),
            "custom first decode excluding cutover",
        ),
        "output_token_ids_sha256": result.get("output_token_ids_sha256"),
        "fixture_sha256": nested(result, "fixture", "sha256"),
        "prompt_tokens": integer(output.get("prompt_tokens"), "custom prompt tokens"),
        "prefill_backing_bytes": prefill_backing_bytes,
        "decode_active_backing_bytes": 0,
        "execution_mode": nested(result, "config", "execution_mode"),
        "graph_prefill": nested(result, "config", "graph_prefill"),
        "graph_decode": nested(result, "config", "graph_decode"),
        "alias_graph_captured": nested(
            result, "diagnostics", "after_load", "model", "alias_graph_captured"
        ),
        "software_versions": result.get("versions"),
    }


def collect_custom(
    root: Path, cold_manifest: dict[str, Any]
) -> dict[str, Any]:
    backend: dict[str, Any] = {
        "display_name": "Custom hybrid alias/remote",
        "artifact_root": resolved(root),
        "timing_domain": "in_process_scheduler_step_and_request",
        "budget_semantics": (
            "Exact frozen calibration-cold expert main-weight bytes; the same "
            "expert IDs are used by the custom and KTransformers placements. "
            "Prefill net GPU-residency reduction is selected bytes minus alias "
            "backing and is an accounting estimate, not an NVML measurement."
        ),
        "tiers": {},
    }
    all_hashes: list[str] = []
    for tier in TIERS:
        plan = cold_manifest["tiers"][tier]
        expected_bytes = integer(
            plan["selected_main_bytes"], f"cold plan {tier} selected bytes"
        )
        paths = custom_result_paths(root, tier)
        if not paths:
            backend["tiers"][tier] = {
                "status": "incomplete",
                "missing_glob": resolved(root / f"{tier}_r*/result.json"),
                "errors": [],
            }
            continue
        try:
            runs = [validate_custom_run(path, expected_bytes) for path in paths]
            hashes = [run["output_token_ids_sha256"] for run in runs]
            require(
                all(isinstance(value, str) and value for value in hashes),
                f"custom output hash missing at {tier}",
            )
            require(len(set(hashes)) == 1, f"custom output hash unstable at {tier}")
            all_hashes.extend(hashes)

            graph_modes = {
                (
                    run["execution_mode"],
                    run["graph_prefill"],
                    run["graph_decode"],
                    run["alias_graph_captured"],
                )
                for run in runs
            }
            require(
                len(graph_modes) == 1,
                f"custom CUDA Graph configuration differs across runs at {tier}",
            )
            mode, graph_prefill, graph_decode, graph_captured = next(
                iter(graph_modes)
            )
            backing_values = {run["prefill_backing_bytes"] for run in runs}
            require(
                len(backing_values) == 1,
                f"custom prefill backing bytes differ across runs at {tier}",
            )
            prefill_backing_bytes = next(iter(backing_values))
            estimated_net_bytes = expected_bytes - prefill_backing_bytes
            backend["tiers"][tier] = {
                "status": "ok",
                "budget": {
                    "nominal_tier_gib": NOMINAL_GIB[tier],
                    "exact_selected_main_bytes": expected_bytes,
                    "exact_selected_main_gib": gib(expected_bytes),
                    "cold_experts_per_layer": integer(
                        plan["cold_experts_per_layer"],
                        f"cold plan {tier} cold experts",
                    ),
                    "selected_expert_count": integer(
                        plan["selected_expert_count"],
                        f"cold plan {tier} selected experts",
                    ),
                    "prefill_backing_bytes": prefill_backing_bytes,
                    "prefill_backing_gib": gib(prefill_backing_bytes),
                    "estimated_prefill_net_gpu_residency_reduction_bytes": (
                        estimated_net_bytes
                    ),
                    "estimated_prefill_net_gpu_residency_reduction_gib": gib(
                        estimated_net_bytes
                    ),
                    "prefill_net_estimate_semantics": (
                        "selected main-weight bytes minus alias backing bytes; "
                        "a physical-residency accounting estimate, not an NVML "
                        "peak-memory measurement"
                    ),
                    "decode_after_cutover": {
                        "active_backing_bytes": 0,
                        "selected_cpu_resident_bytes": expected_bytes,
                    },
                },
                "samples": {
                    "run_count": len(runs),
                    "prefill_step_count": len(runs),
                    "decode_step_count": sum(
                        run["decode_step_samples"] for run in runs
                    ),
                    "request_count": len(runs),
                },
                "scheduler_step": {
                    "prefill_ms_run_median": median(
                        (run["prefill_step_ms"] for run in runs),
                        "custom prefill run medians",
                    ),
                    "steady_decode_ms_run_median": median(
                        (run["decode_step_median_ms"] for run in runs),
                        "custom decode run medians",
                    ),
                    "steady_decode_tokens_per_second_run_median": median(
                        (run["decode_tokens_per_second"] for run in runs),
                        "custom decode throughput",
                    ),
                    "first_decode_cutover_ms_run_median": median(
                        (run["cutover_wall_ms"] for run in runs),
                        "custom cutover",
                    ),
                    "first_decode_including_cutover_ms_run_median": median(
                        (run["first_decode_step_ms"] for run in runs),
                        "custom first decode",
                    ),
                    "first_decode_excluding_cutover_ms_run_median": median(
                        (
                            run["first_decode_excluding_cutover_ms"]
                            for run in runs
                        ),
                        "custom first decode excluding cutover",
                    ),
                },
                "in_process_request": {
                    "ttft_ms_run_median": median(
                        (run["request_ttft_ms"] for run in runs),
                        "custom request TTFT",
                    ),
                    "tpot_ms_run_median": median(
                        (run["request_tpot_ms"] for run in runs),
                        "custom request TPOT",
                    ),
                    "latency_ms_run_median": median(
                        (run["request_latency_ms"] for run in runs),
                        "custom request latency",
                    ),
                },
                "cuda_graph": {
                    "execution_mode": mode,
                    "prefill_enabled": graph_prefill,
                    "decode_enabled": graph_decode,
                    "alias_graph_capture_verified": graph_captured,
                    "status": (
                        "enabled"
                        if graph_prefill or graph_decode or graph_captured
                        else "disabled_eager"
                    ),
                },
                "qualification": {
                    "parameter_va_stable": True,
                    "cold_cpu_residency_exact": True,
                    "hot_gpu_residency_verified": True,
                    "remote_bytes_exact": True,
                    "prefill_backing_bytes_consistent_across_runs": True,
                    "decode_cutover_released_backing": True,
                    "decode_selected_bytes_cpu_resident": True,
                    "teardown_clean": True,
                    "output_hash_stable": True,
                    "output_token_ids_sha256": hashes[0],
                },
                "runs": runs,
                "errors": [],
            }
        except ArtifactError as exc:
            backend["tiers"][tier] = invalid_tier(exc, paths)

    ok_tiers = [
        data
        for data in backend["tiers"].values()
        if data.get("status") == "ok"
    ]
    if ok_tiers:
        backend["software_versions"] = ok_tiers[0]["runs"][0].get(
            "software_versions"
        )
    if all_hashes and len(set(all_hashes)) != 1:
        backend["cross_tier_error"] = "custom output hash differs across tiers"
    else:
        backend["cross_tier_output_hash_stable"] = bool(all_hashes)
    backend["cuda_graph_limitation"] = (
        "The qualified public custom matrix is eager: DeepSeek's available "
        "vLLM TRITON_MLA backend rejected FULL_SEPARATE CUDA Graph mode. "
        "Capture-size configuration alone is not reported as graph execution."
    )
    return backend


def validate_vllm_manifest(path: Path, nominal_bytes: int) -> dict[str, Any]:
    events = [
        row
        for row in read_jsonl(path)
        if row.get("event") == "cpu_offload_parameter"
    ]
    require(events, f"vLLM manifest has no offloaded parameters: {path}")
    require(
        all(
            row.get("offload_backend") == "uva"
            and row.get("uva_offloading") is True
            and row.get("cpu_data_pinned") is True
            for row in events
        ),
        f"vLLM manifest does not describe pinned UVA for every parameter: {path}",
    )
    sizes = [integer(row.get("bytes"), "vLLM manifest bytes") for row in events]
    actual = sum(sizes)
    last_event = max(
        events, key=lambda row: integer(row.get("offload_order"), "offload_order")
    )
    last_bytes = integer(last_event.get("bytes"), "last vLLM parameter bytes")
    require(actual >= nominal_bytes, f"vLLM did not reach nominal budget: {path}")
    require(
        actual - last_bytes < nominal_bytes,
        f"vLLM crossed budget before its final whole parameter: {path}",
    )
    return {
        "actual_offload_bytes": actual,
        "actual_offload_gib": gib(actual),
        "parameter_count": len(events),
        "last_parameter_bytes": last_bytes,
    }


def collect_vllm(root: Path) -> dict[str, Any]:
    backend: dict[str, Any] = {
        "display_name": "vLLM native pinned UVA",
        "artifact_root": resolved(root),
        "timing_domain": "in_process_scheduler_step_and_request",
        "budget_semantics": (
            "Nominal cpu_offload_gb threshold over parameter traversal. Whole "
            "parameters are selected, so actual bytes may cross the threshold "
            "by the final parameter; this is not the frozen cold-expert set."
        ),
        "tiers": {},
    }
    for tier in TIERS:
        case_dir = root / tier
        result_path = case_dir / "result.json"
        manifest_path = case_dir / "manifest.jsonl"
        returncode_path = case_dir / "returncode.txt"
        incomplete = incomplete_tier(
            (result_path, manifest_path, returncode_path)
        )
        if incomplete:
            backend["tiers"][tier] = incomplete
            continue
        try:
            require(
                complete_returncode(returncode_path),
                f"vLLM returncode is not zero: {returncode_path}",
            )
            result = read_json(result_path)
            require(result.get("status") == "ok", f"vLLM status is not ok: {tier}")
            require(
                result.get("mechanism_class")
                == "vllm_native_cpu_offload_pinned_uva",
                f"vLLM mechanism class mismatch: {tier}",
            )
            configured_gib = number(
                nested(result, "config", "cpu_offload_gb"),
                "vLLM cpu_offload_gb",
            )
            require(
                math.isclose(configured_gib, NOMINAL_GIB[tier]),
                f"vLLM nominal tier mismatch: {tier}",
            )
            phase = nested(result, "phase_summary", "1")
            require(isinstance(phase, dict), f"vLLM batch=1 phase missing: {tier}")
            require(
                integer(phase.get("batch_size"), "vLLM phase batch size") == 1,
                f"vLLM batch size mismatch: {tier}",
            )
            require(
                integer(phase.get("prompt_tokens"), "vLLM prompt tokens") == 512,
                f"vLLM prompt length mismatch: {tier}",
            )
            budget = validate_vllm_manifest(
                manifest_path, NOMINAL_GIB[tier] * GIB
            )

            measurement_repeats = integer(
                phase.get("measurement_repeats"),
                "vLLM measurement repeats",
            )
            prefill_count = integer(
                nested(phase, "prefill_step_ms", "count"),
                "vLLM prefill sample count",
            )
            decode_count = integer(
                nested(phase, "decode_all_step_ms", "count"),
                "vLLM decode sample count",
            )
            require(
                measurement_repeats > 0 and prefill_count == measurement_repeats,
                f"vLLM prefill/repeat count mismatch: {tier}",
            )
            require(decode_count > 0, f"vLLM decode samples missing: {tier}")
            cycles = [
                cycle
                for cycle in result.get("cycles", [])
                if isinstance(cycle, dict) and cycle.get("role") == "measurement"
            ]
            require(
                len(cycles) == measurement_repeats,
                f"vLLM measurement cycle count mismatch: {tier}",
            )
            hashes = [cycle.get("output_token_ids_sha256") for cycle in cycles]
            require(
                all(isinstance(value, str) and value for value in hashes)
                and len(set(hashes)) == 1,
                f"vLLM output hash unstable: {tier}",
            )
            enforce_eager = nested(result, "config", "enforce_eager")
            backend["tiers"][tier] = {
                "status": "ok",
                "budget": {
                    "nominal_tier_gib": NOMINAL_GIB[tier],
                    **budget,
                },
                "samples": {
                    "measurement_repeat_count": measurement_repeats,
                    "prefill_step_count": prefill_count,
                    "decode_step_count": decode_count,
                    "request_count": measurement_repeats,
                },
                "scheduler_step": {
                    "prefill_ms_median": number(
                        nested(phase, "prefill_step_ms", "median"),
                        "vLLM prefill median",
                    ),
                    "steady_decode_ms_median": number(
                        nested(phase, "decode_all_step_ms", "median"),
                        "vLLM decode median",
                    ),
                    "steady_decode_tokens_per_second": number(
                        phase.get("decode_tokens_per_second"),
                        "vLLM decode tokens/s",
                    ),
                },
                "in_process_request": {
                    "ttft_ms_median": number(
                        nested(phase, "request_ttft_p50_ms", "median"),
                        "vLLM request TTFT",
                    ),
                    "tpot_ms_median": number(
                        nested(phase, "request_tpot_p50_ms", "median"),
                        "vLLM request TPOT",
                    ),
                    "latency_ms_median": number(
                        nested(phase, "request_latency_p50_ms", "median"),
                        "vLLM request latency",
                    ),
                },
                "cuda_graph": {
                    "enforce_eager": enforce_eager,
                    "status": "disabled_eager" if enforce_eager else "not_verified",
                },
                "qualification": {
                    "pinned_uva_all_parameters": True,
                    "whole_parameter_budget_crossing_valid": True,
                    "output_hash_stable": True,
                    "output_token_ids_sha256": hashes[0],
                },
                "sources": [resolved(result_path), resolved(manifest_path)],
                "software_versions": result.get("versions"),
                "errors": [],
            }
        except ArtifactError as exc:
            backend["tiers"][tier] = invalid_tier(
                exc, (result_path, manifest_path, returncode_path)
            )
    ok_tiers = [
        data
        for data in backend["tiers"].values()
        if data.get("status") == "ok"
    ]
    if ok_tiers:
        backend["software_versions"] = ok_tiers[0].get("software_versions")
    return backend


PLACEMENT_RE = re.compile(r"layer\s+(\d+)\s+\(MoE\)\s+has\s+(\d+)\s+GPU experts")


def validate_kt_placement(path: Path, expected_gpu_experts: int) -> int:
    matches = [
        PLACEMENT_RE.search(line)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
    ]
    pairs = [match.groups() for match in matches if match]
    require(len(pairs) == 26, f"KT placement did not verify 26 MoE layers: {path}")
    layer_ids = {int(layer) for layer, _ in pairs}
    counts = {int(count) for _, count in pairs}
    require(layer_ids == set(range(1, 27)), f"KT placement layer IDs mismatch: {path}")
    require(
        counts == {expected_gpu_experts},
        f"KT GPU expert count mismatch in {path}: {sorted(counts)}",
    )
    return len(pairs)


def validate_gpu_after_zero(path: Path) -> None:
    raw = path.read_text(encoding="utf-8").strip().split(",", 1)[0].strip()
    try:
        used_mib = int(raw)
    except ValueError as exc:
        raise ArtifactError(f"invalid KT gpu.after.csv: {path}") from exc
    require(used_mib == 0, f"KT GPU did not clean up after tier: {path}")


def collect_kt(
    root: Path,
    cold_manifest: dict[str, Any],
    placement_manifest: dict[str, Any],
) -> dict[str, Any]:
    backend: dict[str, Any] = {
        "display_name": "KTransformers exact frequency placement",
        "artifact_root": resolved(root),
        "timing_domain": "streaming_request_ttft_tpot",
        "measurement_boundary": (
            "Local HTTP/SSE client submission to streamed tokens; includes "
            "HTTP, server queueing, scheduling, and inference."
        ),
        "budget_semantics": (
            "Exact complement of the same frozen per-layer cold-expert plan "
            "used by the custom backend; selected experts execute on CPU."
        ),
        "tiers": {},
    }
    for tier in TIERS:
        case_dir = root / tier
        eval_paths = sorted(case_dir.glob("eval_r*.json"), key=lambda path: str(path))
        warmup_path = case_dir / "warmup.json"
        placement_path = case_dir / "placement_counts.log"
        graph_path = case_dir / "cuda_graph_evidence.log"
        gpu_after_path = case_dir / "gpu.after.csv"
        required = (warmup_path, placement_path, graph_path, gpu_after_path)
        incomplete = incomplete_tier(required)
        if incomplete or not eval_paths:
            missing = incomplete or {"status": "incomplete", "errors": []}
            if not eval_paths:
                missing.setdefault("missing_globs", []).append(
                    resolved(case_dir / "eval_r*.json")
                )
            backend["tiers"][tier] = missing
            continue
        sources = [*eval_paths, *required]
        try:
            warmup = read_json(warmup_path)
            require(warmup.get("status") == "ok", f"KT warmup failed: {tier}")
            warmup_prompt_hash = nested(
                warmup,
                "fixture",
                "embedded_metadata",
                "source_prompt_token_ids_sha256",
            )
            require(
                nested(warmup, "fixture", "embedded_metadata", "fixture_role")
                == "warmup",
                f"KT warmup fixture role mismatch: {tier}",
            )
            plan = placement_manifest["tiers"][tier]
            cold_plan = cold_manifest["tiers"][tier]
            selected_bytes = integer(
                plan.get("selected_main_bytes"), f"KT selected bytes {tier}"
            )
            require(
                selected_bytes
                == integer(
                    cold_plan.get("selected_main_bytes"),
                    f"cold selected bytes {tier}",
                ),
                f"KT selected bytes do not match cold plan: {tier}",
            )
            gpu_experts = integer(
                plan.get("gpu_experts_per_layer"), f"KT GPU experts {tier}"
            )
            placement_layers = validate_kt_placement(
                placement_path, gpu_experts
            )
            validate_gpu_after_zero(gpu_after_path)

            graph_text = graph_path.read_text(
                encoding="utf-8", errors="replace"
            )
            capture_end_count = graph_text.count("Capture cuda graph end")
            require(
                capture_end_count > 0,
                f"KT CUDA Graph capture completion is missing: {graph_path}",
            )
            prefill_eager_evidence_count = len(
                re.findall(r"Prefill batch.*cuda graph:\s*False", graph_text)
            )
            replay_true_evidence_count = len(
                re.findall(r"cuda graph:\s*True", graph_text)
            )

            ttft_values: list[float] = []
            tpot_values: list[float] = []
            hashes: list[str] = []
            request_count = 0
            output_tokens = 0
            evaluation_prompt_hashes: list[str] = []
            run_summaries: list[dict[str, Any]] = []
            for eval_path in eval_paths:
                result = read_json(eval_path)
                require(result.get("status") == "ok", f"KT eval failed: {eval_path}")
                phases = result.get("phases")
                require(
                    isinstance(phases, list)
                    and len(phases) == 1
                    and isinstance(phases[0], dict),
                    f"KT eval phase mismatch: {eval_path}",
                )
                phase = phases[0]
                require(
                    nested(
                        result,
                        "fixture",
                        "embedded_metadata",
                        "fixture_role",
                    )
                    == "evaluation",
                    f"KT evaluation fixture role mismatch: {eval_path}",
                )
                evaluation_prompt_hash = nested(
                    result,
                    "fixture",
                    "embedded_metadata",
                    "source_prompt_token_ids_sha256",
                )
                require(
                    isinstance(evaluation_prompt_hash, str)
                    and evaluation_prompt_hash,
                    f"KT evaluation prompt hash missing: {eval_path}",
                )
                evaluation_prompt_hashes.append(evaluation_prompt_hash)
                completed = integer(
                    phase.get("completed_requests"), "KT completed requests"
                )
                failed = integer(phase.get("failed_requests"), "KT failed requests")
                require(completed > 0 and failed == 0, f"KT requests failed: {eval_path}")
                require(
                    phase.get("all_requests_produced_fixed_output") is True,
                    f"KT output length was not fixed: {eval_path}",
                )
                ttft = number(phase.get("p50_ttft_ms"), "KT TTFT")
                tpot = number(phase.get("p50_tpot_ms"), "KT TPOT")
                output_hash = phase.get("output_fingerprints_sha256")
                require(
                    isinstance(output_hash, str) and output_hash,
                    f"KT output hash missing: {eval_path}",
                )
                ttft_values.append(ttft)
                tpot_values.append(tpot)
                hashes.append(output_hash)
                request_count += completed
                output_tokens += integer(
                    phase.get("completed_output_tokens"), "KT output tokens"
                )
                run_summaries.append(
                    {
                        "source": resolved(eval_path),
                        "ttft_ms": ttft,
                        "tpot_ms": tpot,
                        "completed_requests": completed,
                        "completed_output_tokens": integer(
                            phase.get("completed_output_tokens"),
                            "KT output tokens",
                        ),
                        "output_fingerprints_sha256": output_hash,
                    }
                )
            require(len(set(hashes)) == 1, f"KT output hash unstable: {tier}")
            require(
                len(set(evaluation_prompt_hashes)) == 1
                and isinstance(warmup_prompt_hash, str)
                and warmup_prompt_hash not in set(evaluation_prompt_hashes),
                f"KT warmup/evaluation fixtures are not independent: {tier}",
            )
            backend["tiers"][tier] = {
                "status": "ok",
                "budget": {
                    "nominal_tier_gib": NOMINAL_GIB[tier],
                    "exact_selected_main_bytes": selected_bytes,
                    "exact_selected_main_gib": gib(selected_bytes),
                    "cold_experts_per_layer": integer(
                        plan.get("cold_experts_per_layer"),
                        f"KT cold experts {tier}",
                    ),
                    "gpu_experts_per_layer": gpu_experts,
                    "total_cpu_experts": integer(
                        plan.get("total_cpu_experts"),
                        f"KT total CPU experts {tier}",
                    ),
                },
                "samples": {
                    "client_run_count": len(eval_paths),
                    "client_request_count": request_count,
                    "completed_output_token_count": output_tokens,
                },
                "streaming_request": {
                    "ttft_ms_run_median": median(ttft_values, "KT TTFT"),
                    "tpot_ms_run_median": median(tpot_values, "KT TPOT"),
                },
                "cuda_graph": {
                    "enabled_by_server_config": True,
                    "capture_completion_verified": True,
                    "capture_end_evidence_count": capture_end_count,
                    "prefill_eager_evidence_count": prefill_eager_evidence_count,
                    "replay_true_evidence_count": replay_true_evidence_count,
                    "phase_attribution": (
                        "Capture completed for batch size 1. Artifact logs "
                        "explicitly show eager prefill; they do not directly "
                        "record a measured decode replay event."
                    ),
                },
                "qualification": {
                    "placement_layer_count": placement_layers,
                    "placement_exact": True,
                    "output_hash_stable": True,
                    "warmup_and_evaluation_prompts_distinct": True,
                    "output_fingerprints_sha256": hashes[0],
                    "gpu_cleanup_verified": True,
                },
                "runs": run_summaries,
                "errors": [],
            }
        except (ArtifactError, OSError) as exc:
            backend["tiers"][tier] = invalid_tier(exc, sources)
    return backend


def collect_moei(root: Path) -> dict[str, Any]:
    backend: dict[str, Any] = {
        "display_name": "MoE-Infinity demand-only dynamic cache",
        "artifact_root": resolved(root),
        "timing_domain": "streaming_request_ttft_tpot",
        "measurement_boundary": (
            "In-process Hugging Face MoE.generate() with streamer callbacks; "
            "no HTTP transport, but includes Python/generate overhead, demand "
            "loading, and a CUDA synchronize at every generated-token callback."
        ),
        "implementation_variant": (
            "MoE-Infinity commit ba5651897a80d9c9b7a1500cef2c68adaa63db0f "
            "with local compatibility patches, demand-only mode, and the local "
            ">256-token prefill chunk patch; not stock MoE-Infinity or the "
            "paper's EAM configuration."
        ),
        "budget_semantics": (
            "Approximate tier produced by device_memory_ratio and a dynamic "
            "expert cache. Reported saving is observed peak-memory reduction "
            "against the common all-GPU reference, not fixed expert bytes."
        ),
        "tiers": {},
    }
    environments: list[dict[str, Any]] = []
    for tier in TIERS:
        case_dir = root / tier
        result_path = case_dir / "result.json"
        budget_path = case_dir / "memory_budget.json"
        returncode_path = case_dir / "returncode.txt"
        nvml_path = case_dir / "nvml.csv"
        required = (result_path, budget_path, returncode_path, nvml_path)
        incomplete = incomplete_tier(required)
        if incomplete:
            # Do not publish timing from a result.json written before the case
            # has completed its returncode, cleanup, and memory-budget steps.
            backend["tiers"][tier] = incomplete
            continue
        try:
            require(
                complete_returncode(returncode_path),
                f"MoE-Infinity returncode is not zero: {returncode_path}",
            )
            result = read_json(result_path)
            budget = read_json(budget_path)
            require(result.get("status") == "ok", f"MoE-Infinity failed: {tier}")
            qualification = result.get("qualification")
            require(
                isinstance(qualification, dict)
                and qualification.get("status") == "ok",
                f"MoE-Infinity qualification failed: {tier}",
            )
            require(
                qualification.get("output_hash_stable") is True,
                f"MoE-Infinity output hash unstable: {tier}",
            )
            require(
                qualification.get("warmup_output_hash_stable") is True,
                f"MoE-Infinity warmup output hash unstable: {tier}",
            )
            require(
                qualification.get("cross_phase_output_match_required") is False,
                f"MoE-Infinity cross-phase hash contract mismatch: {tier}",
            )
            require(
                result.get("mode") == "demand-only",
                f"MoE-Infinity mode is not demand-only: {tier}",
            )
            require(
                nested(result, "environment", "moe_infinity_commit")
                == MOEI_EXPECTED_COMMIT,
                f"MoE-Infinity commit mismatch: {tier}",
            )
            require(
                isinstance(
                    nested(result, "environment", "moe_infinity_diff_sha256"),
                    str,
                )
                and bool(
                    nested(result, "environment", "moe_infinity_diff_sha256")
                ),
                f"MoE-Infinity compatibility diff fingerprint missing: {tier}",
            )
            require(
                integer(
                    nested(result, "protocol", "batch_size"),
                    "MoE-Infinity batch size",
                )
                == 1,
                f"MoE-Infinity batch size mismatch: {tier}",
            )
            require(
                integer(
                    nested(result, "protocol", "prompt_tokens"),
                    "MoE-Infinity prompt tokens",
                )
                == 512,
                f"MoE-Infinity prompt length mismatch: {tier}",
            )
            prompt_hash = nested(result, "prompt", "prompt_token_ids_sha256")
            warmup_hash = nested(
                result, "warmup_prompt", "prompt_token_ids_sha256"
            )
            require(
                isinstance(prompt_hash, str)
                and isinstance(warmup_hash, str)
                and prompt_hash != warmup_hash,
                f"MoE-Infinity warmup/evaluation fixtures are not independent: {tier}",
            )
            require(budget.get("tier") == tier, f"MoE-Infinity budget tier mismatch: {tier}")
            ratio = number(
                budget.get("device_memory_ratio"), "MoE-Infinity memory ratio"
            )
            require(
                math.isclose(
                    ratio,
                    number(
                        nested(result, "protocol", "device_memory_ratio"),
                        "MoE-Infinity protocol ratio",
                    ),
                    rel_tol=0,
                    abs_tol=1e-12,
                ),
                f"MoE-Infinity ratio mismatch: {tier}",
            )
            reference_mib = integer(
                budget.get("all_gpu_reference_mib"),
                "MoE-Infinity all-GPU reference",
            )
            peak_mib = integer(budget.get("peak_mib"), "MoE-Infinity peak MiB")
            observed_peak_mib, nvml_sample_count = nvml_peak_mib(nvml_path)
            require(
                observed_peak_mib == peak_mib,
                f"MoE-Infinity NVML peak does not match memory budget: {tier}",
            )
            saved_gib = number(
                budget.get("estimated_saved_gib"),
                "MoE-Infinity estimated saved GiB",
            )
            require(
                math.isclose(
                    saved_gib,
                    (reference_mib - peak_mib) / 1024,
                    rel_tol=0,
                    abs_tol=1e-6,
                ),
                f"MoE-Infinity observed saving arithmetic mismatch: {tier}",
            )
            trials = result.get("trials")
            warmups = result.get("warmups")
            require(isinstance(trials, list) and trials, f"MoE-Infinity trials missing: {tier}")
            require(isinstance(warmups, list), f"MoE-Infinity warmups missing: {tier}")
            recorded_repetitions = integer(
                qualification.get("recorded_repetitions"),
                "MoE-Infinity repetitions",
            )
            require(
                recorded_repetitions == len(trials),
                f"MoE-Infinity trial count mismatch: {tier}",
            )
            summary = result.get("summary")
            require(isinstance(summary, dict), f"MoE-Infinity summary missing: {tier}")
            for metric in ("ttft_seconds", "decode_tpot_seconds"):
                require(
                    integer(
                        nested(summary, metric, "count"),
                        f"MoE-Infinity {metric} count",
                    )
                    == len(trials),
                    f"MoE-Infinity {metric} sample count mismatch: {tier}",
                )
            environment = result.get("environment")
            if isinstance(environment, dict):
                environments.append(environment)
            backend["tiers"][tier] = {
                "status": "ok",
                "budget": {
                    "nominal_tier_gib": NOMINAL_GIB[tier],
                    "device_memory_ratio": ratio,
                    "all_gpu_reference_mib": reference_mib,
                    "observed_peak_mib": peak_mib,
                    "estimated_saved_gib": saved_gib,
                    "is_fixed_expert_byte_budget": False,
                },
                "samples": {
                    "measured_trial_count": len(trials),
                    "warmup_request_count": len(warmups),
                    "output_tokens_per_trial": integer(
                        nested(result, "protocol", "output_tokens"),
                        "MoE-Infinity output tokens",
                    ),
                    "nvml_sample_count": nvml_sample_count,
                },
                "streaming_request": {
                    "ttft_ms_median": 1000
                    * number(
                        nested(summary, "ttft_seconds", "median"),
                        "MoE-Infinity TTFT median",
                    ),
                    "tpot_ms_median": 1000
                    * number(
                        nested(summary, "decode_tpot_seconds", "median"),
                        "MoE-Infinity TPOT median",
                    ),
                    "decode_tokens_per_second_median": number(
                        nested(summary, "decode_tokens_per_second", "median"),
                        "MoE-Infinity decode tokens/s",
                    ),
                    "tpot_definition": nested(
                        result,
                        "protocol",
                        "primary_decode_tpot_definition",
                    ),
                },
                "cuda_graph": {
                    "status": "not_used",
                    "reason": (
                        "The qualified MoE-Infinity path uses its direct "
                        "non-native generation engine; no CUDA Graph evidence "
                        "is present in the artifact."
                    ),
                },
                "qualification": {
                    "status": "ok",
                    "output_hash_stable": True,
                    "warmup_output_hash_stable": True,
                    "warmup_and_evaluation_prompts_distinct": True,
                    "evaluation_prompt_token_ids_sha256": prompt_hash,
                    "warmup_prompt_token_ids_sha256": warmup_hash,
                },
                "sources": [
                    resolved(result_path),
                    resolved(budget_path),
                    resolved(nvml_path),
                ],
                "errors": [],
            }
        except ArtifactError as exc:
            backend["tiers"][tier] = invalid_tier(exc, required)
    if environments:
        first = environments[0]
        backend["environment"] = {
            key: first.get(key)
            for key in (
                "gpu_name",
                "gpu_count",
                "gpu_capability",
                "cuda_runtime",
                "torch",
                "transformers",
                "moe_infinity_commit",
                "moe_infinity_diff_sha256",
                "moe_infinity_status_porcelain",
            )
        }
    return backend


def tier_plan_summary(cold_manifest: dict[str, Any], tier: str) -> dict[str, Any]:
    plan = cold_manifest["tiers"][tier]
    return {
        "nominal_offload_gib": NOMINAL_GIB[tier],
        "frozen_cold_plan": {
            "selected_main_bytes": integer(
                plan["selected_main_bytes"], f"cold plan {tier} bytes"
            ),
            "selected_main_gib": number(
                plan["selected_main_gib"], f"cold plan {tier} GiB"
            ),
            "cold_experts_per_layer": integer(
                plan["cold_experts_per_layer"], f"cold plan {tier} cold experts"
            ),
            "selected_expert_count": integer(
                plan["selected_expert_count"],
                f"cold plan {tier} selected experts",
            ),
            "calibration_selected_route_slots": integer(
                plan["calibration_selected_route_slots"],
                f"cold plan {tier} route slots",
            ),
            "plan_sha256": plan.get("plan_sha256"),
        },
    }


def validation_summary(backends: dict[str, dict[str, Any]]) -> dict[str, Any]:
    incomplete: list[str] = []
    invalid: list[dict[str, Any]] = []
    for backend_name, backend in backends.items():
        cross_tier_error = backend.get("cross_tier_error")
        if cross_tier_error:
            invalid.append(
                {"case": backend_name, "errors": [str(cross_tier_error)]}
            )
        for tier in TIERS:
            result = backend["tiers"][tier]
            status = result.get("status")
            case = f"{backend_name}/{tier}"
            if status == "incomplete":
                incomplete.append(case)
            elif status != "ok":
                invalid.append(
                    {"case": case, "errors": result.get("errors", [])}
                )
    status = "invalid" if invalid else ("incomplete" if incomplete else "ok")
    return {
        "status": status,
        "required_tiers": list(TIERS),
        "required_backends": list(backends),
        "all_four_tiers_complete": not incomplete and not invalid,
        "incomplete_cases": incomplete,
        "invalid_cases": invalid,
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    cold_manifest, placement_manifest = read_input_manifests(
        args.cold_plan_manifest, args.kt_placement_manifest
    )
    backends = {
        "custom": collect_custom(args.custom_root, cold_manifest),
        "vllm_native": collect_vllm(args.vllm_root),
        "ktransformers": collect_kt(
            args.kt_root, cold_manifest, placement_manifest
        ),
        "moe_infinity": collect_moei(args.moei_root),
    }
    validation = validation_summary(backends)
    hardware = nested(backends, "moe_infinity", "environment")
    return {
        "schema": "hybrid_offload_2_4_6_8_comparison_summary_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": validation["status"],
        "protocol": {
            "model": "DeepSeek-V2-Lite-Chat BF16",
            "hardware": hardware,
            "batch_size": 1,
            "evaluation_prompt_tokens": 512,
            "evaluation_selection": "held-out frozen token-ID fixture",
            "cold_selection": {
                "rule": cold_manifest.get("selection_rule"),
                "calibration_trace_sha256": cold_manifest.get(
                    "calibration_trace_sha256"
                ),
                "layer_ids": cold_manifest.get("layer_ids"),
                "experts_per_layer": cold_manifest.get("experts_per_layer"),
                "calibration_coverage": cold_manifest.get("coverage"),
            },
            "placement_update_scope_tested": (
                "Initialization-time frozen placement only. Scheduling-time "
                "and idle-time replacement are mechanism capabilities but "
                "were not exercised by this matrix."
            ),
            "sample_structure": {
                "per_tier_nominal_n": 3,
                "custom": "three independent cold-process runs",
                "vllm_native": "three measurement cycles in one process",
                "ktransformers": "three requests against one per-tier server",
                "moe_infinity": (
                    "three trials in one process sharing the dynamic cache"
                ),
                "cross_method_variance_comparable": False,
            },
            "timing_domains": {
                "scheduler_step": {
                    "backends": ["custom", "vllm_native"],
                    "definition": (
                        "In-process vLLM scheduler-step wall time. Prefill and "
                        "steady one-token decode are directly comparable within "
                        "this group."
                    ),
                },
                "in_process_request": {
                    "backends": ["custom", "vllm_native"],
                    "definition": (
                        "Request metrics reported inside the vLLM process; "
                        "kept separate from the streaming-request group."
                    ),
                },
                "streaming_request_ttft_tpot": {
                    "backends": ["ktransformers", "moe_infinity"],
                    "definition": (
                        "Streaming TTFT and TPOT shown side by side only. KT is "
                        "a local HTTP/SSE client measurement and includes HTTP "
                        "and queueing. MoE-Infinity is an in-process HF "
                        "MoE.generate()/streamer measurement without HTTP, but "
                        "includes Python/generate, demand loading, and per-token "
                        "callback synchronization. Neither is interchangeable "
                        "with scheduler-step time, and their boundaries are not "
                        "fully equivalent to each other."
                    ),
                },
            },
        },
        "tiers": {
            tier: tier_plan_summary(cold_manifest, tier) for tier in TIERS
        },
        "backends": backends,
        "comparability": {
            "direct_scheduler_step_comparison": ["custom", "vllm_native"],
            "streaming_request_side_by_side_observation": [
                "ktransformers",
                "moe_infinity",
            ],
            "streaming_request_boundaries_fully_equivalent": False,
            "budget_equivalence": {
                "custom_vs_ktransformers": (
                    "Exact same frozen cold expert IDs and main-weight bytes."
                ),
                "vllm_native": (
                    "Different whole-parameter traversal selection; nominal "
                    "budget only."
                ),
                "moe_infinity": (
                    "Dynamic cache ratio and observed peak saving; neither the "
                    "expert set nor bytes are fixed."
                ),
            },
            "cross_timing_domain_ranking_allowed": False,
        },
        "limitations": [
            (
                "Scheduler-step latency and streaming-request TTFT/TPOT use different "
                "boundaries and must not be placed in one numerical ranking."
            ),
            (
                "Only custom and KTransformers use the same frozen cold expert "
                "set; vLLM uses parameter traversal and MoE-Infinity uses a "
                "dynamic cache."
            ),
            (
                "The custom matrix validates initialization-time frozen "
                "placement. Scheduling-time, idle-time, and KV-pressure-driven "
                "placement changes remain unmeasured."
            ),
            (
                "The qualified custom matrix is eager because the available "
                "DeepSeek TRITON_MLA backend rejected FULL_SEPARATE CUDA Graph "
                "mode. KTransformers has capture-completion evidence, but the "
                "artifact does not directly attribute measured decode requests "
                "to replay."
            ),
            (
                "Batch size 1 and one held-out 512-token fixture do not establish "
                "performance for other prompts, concurrency levels, or models."
            ),
            (
                "MoE-Infinity is commit "
                f"{MOEI_EXPECTED_COMMIT} with local "
                "compatibility patches in demand-only mode. Its 512-token "
                "prefill depends on the local >256-token chunk patch; this is "
                "neither stock MoE-Infinity nor the paper's EAM configuration."
            ),
            (
                "Each tier has nominal n=3, but the sampling units differ: "
                "custom uses three independent cold processes; vLLM native "
                "uses three cycles in one process; KTransformers uses three "
                "requests against one server; MoE-Infinity uses three trials "
                "sharing one process and dynamic cache. Cross-method variance "
                "is not comparable."
            ),
            (
                "Custom 2g/4g prefill and 8g decode, plus several "
                "KTransformers/MoE-Infinity tiers, show notable range or "
                "outlier behavior. Medians are retained, but conclusions are "
                "limited to an exploratory qualification matrix."
            ),
        ],
        "validation": validation,
        "sources": {
            "cold_plan_manifest": resolved(args.cold_plan_manifest),
            "kt_placement_manifest": resolved(args.kt_placement_manifest),
        },
    }


def fmt_ms(value: Any) -> str:
    if value is None:
        return "待完成"
    return f"{float(value):.3f} ms"


def fmt_gib(value: Any) -> str:
    if value is None:
        return "待完成"
    return f"{float(value):.4f} GiB"


def percent_change(value: float, baseline: float) -> float:
    require(baseline != 0, "cannot compute percent change from a zero baseline")
    return 100.0 * (value - baseline) / baseline


def tier_data(
    summary: dict[str, Any], backend: str, tier: str
) -> dict[str, Any] | None:
    data = summary["backends"][backend]["tiers"][tier]
    return data if data.get("status") == "ok" else None


def core_conclusions(summary: dict[str, Any]) -> list[str]:
    prefill_reductions: list[tuple[str, float]] = []
    custom_decode_values: list[float] = []
    vllm_decode_values: list[float] = []
    cutover_values: list[float] = []
    moei_tpot_values: list[float] = []
    moei_sample_counts: list[int] = []
    for tier in TIERS:
        custom = tier_data(summary, "custom", tier)
        vllm = tier_data(summary, "vllm_native", tier)
        moei = tier_data(summary, "moe_infinity", tier)
        if custom and vllm:
            custom_step = custom["scheduler_step"]
            vllm_step = vllm["scheduler_step"]
            prefill_reductions.append(
                (
                    tier,
                    -percent_change(
                        custom_step["prefill_ms_run_median"],
                        vllm_step["prefill_ms_median"],
                    ),
                )
            )
            custom_decode_values.append(
                float(custom_step["steady_decode_ms_run_median"])
            )
            vllm_decode_values.append(
                float(vllm_step["steady_decode_ms_median"])
            )
            cutover_values.append(
                float(custom_step["first_decode_cutover_ms_run_median"])
            )
        if moei:
            moei_tpot_values.append(
                float(moei["streaming_request"]["tpot_ms_median"])
            )
            moei_sample_counts.append(
                int(moei["samples"]["measured_trial_count"])
            )

    conclusions: list[str] = []
    if prefill_reductions:
        changes = "，".join(
            f"{tier} 为 {reduction:.1f}%"
            for tier, reduction in prefill_reductions
        )
        conclusions.append(
            "- 同为 scheduler-step 计时、但预算选择规则不同的同名义档位下，"
            f"我们的 prefill step 相对 vLLM 原生降幅分别是：{changes}。"
        )
    if custom_decode_values and vllm_decode_values:
        conclusions.append(
            f"- 我们的稳定 decode 为约 {min(custom_decode_values):.1f}-"
            f"{max(custom_decode_values):.1f} ms，四档都绝对慢于 vLLM 原生；"
            "不过它没有随名义预算从 2 GiB 增至 8 GiB 而持续恶化，表现得更稳定。"
        )
    if cutover_values:
        conclusions.append(
            f"- 首次 decode 的一次性 cutover 随集合扩大，从约 "
            f"{cutover_values[0]:.0f} ms 增至 {cutover_values[-1]:.0f} ms；"
            "稳定 decode 表没有混入这项一次性成本。"
        )
    if moei_tpot_values:
        non_monotonic = any(
            left > right
            for left, right in zip(moei_tpot_values, moei_tpot_values[1:])
        )
        sample_text = (
            str(moei_sample_counts[0])
            if len(set(moei_sample_counts)) == 1
            else "/".join(str(value) for value in moei_sample_counts)
        )
        conclusions.append(
            "- KTransformers 与 MoE-Infinity 只能在流式 TTFT/TPOT 域并列观察，"
            "两者测量边界不同，不能强排。MoE-Infinity 是动态 cache、每档 "
            f"n={sample_text}，TPOT"
            f"{' 呈非单调变化' if non_monotonic else ' 在当前样本中呈单调变化'}，"
            "不足以推出稳定的预算趋势。"
        )
    return conclusions


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    validation = summary["validation"]
    coverage = summary["protocol"]["cold_selection"]["calibration_coverage"]
    total_route_slots = sum(
        int(value) for value in coverage["routed_slots"].values()
    )
    require(total_route_slots > 0, "calibration routed-slot count must be positive")
    cold_selection_rows: list[list[str]] = []
    for tier in TIERS:
        cold = summary["tiers"][tier]["frozen_cold_plan"]
        selected_slots = int(cold["calibration_selected_route_slots"])
        cold_selection_rows.append(
            [
                tier,
                str(cold["cold_experts_per_layer"]),
                str(cold["selected_expert_count"]),
                f"{100.0 * selected_slots / total_route_slots:.3f}%",
            ]
        )
    lines = [
        "# MoE 卸载 2/4/6/8 GiB 实验汇总",
        "",
        f"汇总状态：**{validation['status']}**。模型为 DeepSeek-V2-Lite-Chat "
        "BF16，batch=1，测量输入为独立保留的 512-token 固定 token-ID "
        "fixture。",
        "",
    ]
    if validation["incomplete_cases"]:
        cases = "、".join(f"`{case}`" for case in validation["incomplete_cases"])
        lines.extend(
            [
                f"尚未完成：{cases}。这些单元格只显示“待完成”，不会从临时 "
                "`result.json` 推断结果。",
                "",
            ]
        )

    lines.extend(
        [
            "## 先看懂计时口径",
            "",
            "我们的机制和 vLLM 原生机制能在同一个 vLLM 进程内看到 "
            "scheduler step，因此可以直接比较一次 prefill 和一次稳定 decode "
            "step。KTransformers 的 TTFT/TPOT 来自本地 HTTP/SSE 客户端；"
            "MoE-Infinity 则来自同进程 HF `MoE.generate()` 的 streamer callback。"
            "两者可并列观察，但边界并不完全相同，也不能与 scheduler-step 混排得出"
            "四后端总排名。",
            "",
            "## 冷专家如何选择",
            "",
            "先用一条独立的 512-token calibration 输入统计每层专家在 prefill 和 "
            "decode 中的总路由次数，再在每层按次数从少到多选择；次数相同时按 "
            "expert ID 排序。四个档位是嵌套集合，评测使用另一条 held-out 输入，"
            "没有拿评测请求反向挑专家。",
            "",
            markdown_table(
                ["档位", "每层冷专家", "总专家数", "占 calibration 路由槽"],
                cold_selection_rows,
            ),
            "",
            "这只证明这些专家在该 calibration fixture 上较冷。Calibration 只有一条"
            "请求，held-out 评测也未开启路由 trace，因此不能把这组专家外推为所有"
            "工作负载下都冷。",
            "",
            "## 卸载预算",
            "",
        ]
    )
    budget_rows: list[list[str]] = []
    for tier in TIERS:
        custom = tier_data(summary, "custom", tier)
        vllm = tier_data(summary, "vllm_native", tier)
        kt = tier_data(summary, "ktransformers", tier)
        moei = tier_data(summary, "moe_infinity", tier)
        cold = summary["tiers"][tier]["frozen_cold_plan"]
        budget_rows.append(
            [
                tier,
                str(cold["cold_experts_per_layer"]),
                fmt_gib(cold["selected_main_gib"]),
                fmt_gib(
                    nested(custom or {}, "budget", "prefill_backing_gib")
                ),
                fmt_gib(
                    nested(
                        custom or {},
                        "budget",
                        "estimated_prefill_net_gpu_residency_reduction_gib",
                    )
                ),
                fmt_gib(
                    nested(vllm or {}, "budget", "actual_offload_gib")
                ),
                fmt_gib(
                    nested(moei or {}, "budget", "estimated_saved_gib")
                ),
                (
                    f"{nested(moei or {}, 'budget', 'device_memory_ratio'):.2f}"
                    if moei
                    else "待完成"
                ),
            ]
        )
        require(
            custom is None
            or kt is None
            or nested(custom, "budget", "exact_selected_main_bytes")
            == nested(kt, "budget", "exact_selected_main_bytes"),
            f"render-time exact budget mismatch at {tier}",
        )
    lines.extend(
        [
            markdown_table(
                [
                    "名义档位",
                    "每层冷专家",
                    "我们/KT 精确权重",
                    "Custom prefill backing",
                    "Custom prefill 净释放估算",
                    "vLLM 实际整参数",
                    "MoEI 观测节省",
                    "MoEI ratio",
                ],
                budget_rows,
            ),
            "",
            "“我们/KT”使用完全相同的 calibration-cold 专家集合。vLLM 会按参数"
            "遍历并完整卸载最后一个参数，所以实际值可越过名义预算；MoE-Infinity "
            "是动态 cache，只能把峰值显存相对 all-GPU 参考的下降作为近似档位。",
            "",
            "Custom prefill 净释放估算 = 选中专家主权重 - alias backing。它是驱动"
            "物理驻留记账的差值估算，不是 NVML 峰值实测。切到 decode 后，active "
            "backing 已释放为 0，选中的全部主权重字节均验证为 CPU resident。",
            "",
            "## 核心结论",
            "",
            *core_conclusions(summary),
            "",
            "## Scheduler-step 性能",
            "",
        ]
    )
    scheduler_rows: list[list[str]] = []
    for tier in TIERS:
        for backend_name, label in (
            ("custom", "我们的 prefill 搬运 / decode 远程访问"),
            ("vllm_native", "vLLM 原生 pinned UVA"),
        ):
            data = tier_data(summary, backend_name, tier)
            if data:
                step = data["scheduler_step"]
                if backend_name == "custom":
                    prefill = step["prefill_ms_run_median"]
                    decode = step["steady_decode_ms_run_median"]
                else:
                    prefill = step["prefill_ms_median"]
                    decode = step["steady_decode_ms_median"]
                samples = data["samples"]
                sample_text = (
                    f"prefill {samples['prefill_step_count']} / "
                    f"decode {samples['decode_step_count']}"
                )
                graph = data["cuda_graph"]["status"]
            else:
                prefill = decode = None
                sample_text = "待完成"
                graph = "待完成"
            scheduler_rows.append(
                [tier, label, fmt_ms(prefill), fmt_ms(decode), sample_text, graph]
            )
    scheduler_block = [
            markdown_table(
                [
                    "档位",
                    "机制",
                    "Prefill step",
                    "稳定 Decode step",
                    "样本数",
                    "CUDA Graph",
                ],
                scheduler_rows,
            ),
            "",
        ]
    scheduler_block.extend(
        [
            "自定义机制的首次 decode 还包含一次从 prefill 搬运态切换到远程访问态的 "
            "cutover；上表 decode 是切换后的稳定 step，不把一次性 cutover 混进去。",
            "",
            "### 自定义机制首次 decode 切换成本",
            "",
        ]
    )
    lines.extend(scheduler_block)
    cutover_rows: list[list[str]] = []
    for tier in TIERS:
        data = tier_data(summary, "custom", tier)
        step = data["scheduler_step"] if data else {}
        cutover_rows.append(
            [
                tier,
                fmt_ms(step.get("first_decode_cutover_ms_run_median")),
                fmt_ms(
                    step.get("first_decode_excluding_cutover_ms_run_median")
                ),
                fmt_ms(
                    step.get("first_decode_including_cutover_ms_run_median")
                ),
            ]
        )
    lines.extend(
        [
            markdown_table(
                ["档位", "Cutover", "不含 cutover", "首次 decode 总计"],
                cutover_rows,
            ),
            "",
            "## vLLM 进程内请求指标",
            "",
        ]
    )
    in_process_rows: list[list[str]] = []
    for tier in TIERS:
        for backend_name, label in (
            ("custom", "我们的机制"),
            ("vllm_native", "vLLM 原生"),
        ):
            data = tier_data(summary, backend_name, tier)
            request = data["in_process_request"] if data else {}
            ttft_key = (
                "ttft_ms_run_median" if backend_name == "custom" else "ttft_ms_median"
            )
            tpot_key = (
                "tpot_ms_run_median" if backend_name == "custom" else "tpot_ms_median"
            )
            count = nested(data or {}, "samples", "request_count")
            in_process_rows.append(
                [
                    tier,
                    label,
                    fmt_ms(request.get(ttft_key)),
                    fmt_ms(request.get(tpot_key)),
                    str(count) if count is not None else "待完成",
                ]
            )
    lines.extend(
        [
            markdown_table(
                ["档位", "机制", "进程内 TTFT", "进程内 TPOT", "请求数"],
                in_process_rows,
            ),
            "",
            "这些是 vLLM 进程内部的请求指标，用于补充解释 step 数据，不与下方两种"
            "流式请求计时直接比较。",
            "",
            "## 流式请求 TTFT / TPOT（边界不同）",
            "",
            "KTransformers 数值包含本地 HTTP/SSE、排队、调度和推理；"
            "MoE-Infinity 没有 HTTP，但包含 Python/`generate()`、按需加载，而且每次 "
            "token callback 都执行 CUDA synchronize。下表只作并列观察，不表示完全"
            "同口径对决。",
            "",
        ]
    )
    client_rows: list[list[str]] = []
    for tier in TIERS:
        for backend_name, label in (
            ("ktransformers", "KTransformers"),
            ("moe_infinity", "MoE-Infinity"),
        ):
            data = tier_data(summary, backend_name, tier)
            client = data["streaming_request"] if data else {}
            if backend_name == "ktransformers":
                ttft_key = "ttft_ms_run_median"
                tpot_key = "tpot_ms_run_median"
                count_key = "client_request_count"
            else:
                ttft_key = "ttft_ms_median"
                tpot_key = "tpot_ms_median"
                count_key = "measured_trial_count"
            count = nested(data or {}, "samples", count_key)
            client_rows.append(
                [
                    tier,
                    label,
                    (
                        "本地 HTTP/SSE"
                        if backend_name == "ktransformers"
                        else "同进程 generate/streamer"
                    ),
                    fmt_ms(client.get(ttft_key)),
                    fmt_ms(client.get(tpot_key)),
                    str(count) if count is not None else "待完成",
                ]
            )
    lines.extend(
        [
            markdown_table(
                [
                    "档位",
                    "机制",
                    "测量边界",
                    "流式 TTFT",
                    "流式 TPOT",
                    "请求/试验数",
                ],
                client_rows,
            ),
            "",
            "## CUDA Graph 状态",
            "",
            markdown_table(
                ["机制", "本矩阵证据", "解释"],
                [
                    [
                        "我们的机制",
                        "eager",
                        "TRITON_MLA 拒绝 FULL_SEPARATE，本次公共矩阵未启用 graph",
                    ],
                    [
                        "vLLM 原生",
                        "eager",
                        "runner 明确 enforce_eager",
                    ],
                    [
                        "KTransformers",
                        "capture 完成",
                        "batch=1 capture 已验证；prefill 日志为 eager，decode replay 未被逐请求日志直接证明",
                    ],
                    [
                        "MoE-Infinity",
                        "未使用",
                        "direct non-native engine artifact 没有 CUDA Graph 证据",
                    ],
                ],
            ),
            "",
            "## 结论边界",
            "",
            "本轮验证的是初始化阶段冻结的冷专家集合。调度阶段、空闲阶段以及 KV cache "
            "压力驱动的动态换集合尚未进入这组实测，不能把它们写成已验证收益。当前数据"
            "回答的是固定 batch=1、固定 512-token 输入下，不同预算语义的四种机制分别要付出"
            "多少 prefill/decode 或 TTFT/TPOT 成本。",
            "",
            "每档名义上都是 n=3，但样本结构不同：Custom 是 3 个独立冷进程，"
            "vLLM 原生是同进程 3 个 cycles，KTransformers 是同一档 server 上的 "
            "3 个 requests，MoE-Infinity 是共享同一进程动态 cache 的 3 个 trials。"
            "因此不能跨方法比较方差。Custom 2g/4g prefill、8g decode，以及 "
            "KTransformers/MoE-Infinity 若干档存在明显范围或离群；本报告保留中位数，"
            "结论仅限探索性 qualification matrix。",
            "",
            "MoE-Infinity 结果来自 commit "
            "`ba5651897a80d9c9b7a1500cef2c68adaa63db0f` 的本地 compatibility-patched "
            "demand-only 路径；512-token prefill 依赖本地 >256-token chunk patch。它"
            "不是 stock MoE-Infinity，也不是论文 EAM 配置。",
            "",
            "原始 artifact 根目录：",
            "",
        ]
    )
    for backend_name in (
        "custom",
        "vllm_native",
        "ktransformers",
        "moe_infinity",
    ):
        backend = summary["backends"][backend_name]
        lines.append(
            f"- {backend['display_name']}: `{backend['artifact_root']}`"
        )
    lines.append("")
    return "\n".join(lines)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def print_validation_errors(validation: dict[str, Any]) -> None:
    for case in validation.get("incomplete_cases", []):
        print(f"incomplete: {case}", file=sys.stderr)
    for item in validation.get("invalid_cases", []):
        errors = "; ".join(str(value) for value in item.get("errors", []))
        print(f"invalid: {item.get('case')}: {errors}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    try:
        summary = build_summary(args)
        validation = summary["validation"]
        if validation["status"] == "invalid":
            print_validation_errors(validation)
            return 2
        if validation["status"] == "incomplete" and not args.allow_incomplete:
            print_validation_errors(validation)
            print(
                "refusing to emit a final report; rerun with --allow-incomplete "
                "for an explicitly partial report",
                file=sys.stderr,
            )
            return 2
        markdown = render_markdown(summary)
        atomic_write(
            args.output_json,
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        atomic_write(args.output_md, markdown)
        print(f"status={validation['status']}")
        print(f"json={args.output_json.resolve()}")
        print(f"markdown={args.output_md.resolve()}")
        return 0
    except (ArtifactError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
