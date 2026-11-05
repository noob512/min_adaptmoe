#!/usr/bin/env python3
"""Summarize repeated vLLM-native-offload versus KT pilot runs.

The top-level run directory must contain the four fixed pilot cases
``vllm_a1``, ``kt_b1``, ``kt_b2``, and ``vllm_a2``.  vLLM cases provide
``result.json`` plus a CPU offload JSONL manifest; KT cases provide
``benchmark.json`` plus a converter manifest describing their hot/cold expert
split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


CONCURRENCIES = (1, 4, 8)
CASE_ORDER = ("vllm_a1", "kt_b1", "kt_b2", "vllm_a2")


class SummaryError(ValueError):
    """Raised when a pilot artifact is incomplete or internally inconsistent."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except OSError as exc:
        raise SummaryError(f"cannot read {path}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SummaryError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SummaryError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SummaryError(f"cannot read {path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SummaryError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise SummaryError(f"expected an object at {path}:{line_number}")
        records.append(value)
    return records


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise SummaryError(f"cannot hash {path}: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def output_token_ids_sha256(request_token_ids: list[list[int]]) -> str:
    """Use the same stable token-ID digest on both runtime result formats."""
    return sha256_bytes(
        json.dumps(request_token_ids, separators=(",", ":")).encode("utf-8")
    )


def strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SummaryError(f"{label} must be an integer, got {value!r}")
    return value


def positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryError(f"{label} must be numeric, got {value!r}")
    result = float(value)
    if not result > 0:
        raise SummaryError(f"{label} must be positive, got {result}")
    return result


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        normalized = path.resolve() if path.exists() else path.absolute()
        if normalized not in seen:
            result.append(path)
            seen.add(normalized)
    return result


def check_returncode(case_root: Path) -> dict[str, int]:
    """Validate conventional case/client return-code files when present."""
    candidates = (
        case_root / "returncode.txt",
        case_root / "client_returncode.txt",
        case_root / "benchmark_returncode.txt",
        case_root / "benchmark.returncode",
    )
    observed: dict[str, int] = {}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            value = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError) as exc:
            raise SummaryError(f"invalid return code file {path}: {exc}") from exc
        if value != 0:
            raise SummaryError(f"non-zero return code for {case_root.name}: {path.name}={value}")
        observed[path.name] = value
    return observed


def resolve_fixture_path(
    case_root: Path, run_root: Path, configured_path: Any
) -> Path | None:
    candidates: list[Path] = []
    if isinstance(configured_path, str) and configured_path:
        configured = Path(configured_path)
        if configured.is_absolute():
            candidates.append(configured)
        else:
            candidates.extend((case_root / configured, run_root / configured))
    candidates.extend(
        (
            run_root / "fixture.json",
            run_root / "fixtures" / "evaluation_fixture.json",
            case_root / "fixture.json",
            case_root / "evaluation_fixture.json",
        )
    )
    return next((path for path in unique_paths(candidates) if path.is_file()), None)


def fixture_metadata(
    *,
    case_root: Path,
    run_root: Path,
    configured_path: Any,
    declared_sha256: Any = None,
) -> dict[str, Any]:
    path = resolve_fixture_path(case_root, run_root, configured_path)
    recomputed = sha256_file(path) if path is not None else None
    declared = declared_sha256 if isinstance(declared_sha256, str) else None
    if declared is not None and recomputed is not None and declared != recomputed:
        raise SummaryError(
            f"fixture SHA mismatch for {case_root.name}: "
            f"declared={declared} recomputed={recomputed} path={path}"
        )
    digest = declared or recomputed
    if digest is None:
        raise SummaryError(
            f"cannot determine fixture SHA-256 for {case_root.name}; "
            "keep the configured fixture file or record its digest"
        )
    return {
        "path": str(path.resolve()) if path is not None else str(configured_path),
        "sha256": digest,
        "sha256_recomputed": recomputed,
    }


def concurrency_from_phase(phase: dict[str, Any], mechanism: str) -> int | None:
    raw = phase.get("concurrency")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    name = str(phase.get("phase", ""))
    match = re.fullmatch(r"(?:sharegpt_)?c(\d+)", name)
    if match is not None:
        return int(match.group(1))
    # vLLM's phased runner does not have a separate concurrency field.
    if mechanism == "vllm" and name in {f"c{value}" for value in CONCURRENCIES}:
        return strict_int(phase.get("request_count"), f"{mechanism}/{name}.request_count")
    return None


def select_phases(
    raw_phases: Any, mechanism: str, case_name: str
) -> dict[int, dict[str, Any]]:
    if not isinstance(raw_phases, list):
        raise SummaryError(f"{case_name}: phases must be a list")
    selected: dict[int, dict[str, Any]] = {}
    for index, phase in enumerate(raw_phases):
        if not isinstance(phase, dict):
            raise SummaryError(f"{case_name}: phases[{index}] is not an object")
        concurrency = concurrency_from_phase(phase, mechanism)
        if concurrency not in CONCURRENCIES:
            continue
        if concurrency in selected:
            raise SummaryError(f"{case_name}: duplicate c{concurrency} phase")
        selected[concurrency] = phase
    missing = [value for value in CONCURRENCIES if value not in selected]
    if missing:
        raise SummaryError(f"{case_name}: missing phases {['c' + str(v) for v in missing]}")
    return selected


def summarize_vllm_phase(
    phase: dict[str, Any], concurrency: int, case_name: str
) -> dict[str, Any]:
    label = f"{case_name}/c{concurrency}"
    request_count = strict_int(phase.get("request_count"), f"{label}.request_count")
    completed_requests = strict_int(
        phase.get("completed_requests"), f"{label}.completed_requests"
    )
    output_per_request = strict_int(
        phase.get("output_tokens_per_request"), f"{label}.output_tokens_per_request"
    )
    output_tokens = strict_int(phase.get("output_tokens"), f"{label}.output_tokens")
    prompt_tokens = strict_int(phase.get("prompt_tokens"), f"{label}.prompt_tokens")
    expected_output_tokens = concurrency * output_per_request
    if request_count != concurrency or completed_requests != concurrency:
        raise SummaryError(
            f"{label}: requests={request_count}, completed={completed_requests}, "
            f"expected={concurrency}"
        )
    if output_tokens != expected_output_tokens:
        raise SummaryError(
            f"{label}: output_tokens={output_tokens}, expected={expected_output_tokens}"
        )
    requests = phase.get("requests")
    if not isinstance(requests, list) or len(requests) != concurrency:
        raise SummaryError(f"{label}: expected {concurrency} request records")
    request_token_ids: list[list[int]] = []
    for index, record in enumerate(requests):
        if not isinstance(record, dict):
            raise SummaryError(f"{label}.requests[{index}] is not an object")
        record_output_tokens = strict_int(
            record.get("output_tokens"),
            f"{label}.requests[{index}].output_tokens",
        )
        if record_output_tokens != output_per_request:
            raise SummaryError(
                f"{label}.requests[{index}] output_tokens={record_output_tokens}, "
                f"expected={output_per_request}"
            )
        token_ids = record.get("output_token_ids")
        if not isinstance(token_ids, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in token_ids
        ):
            raise SummaryError(
                f"{label}.requests[{index}].output_token_ids is invalid"
            )
        if len(token_ids) != output_per_request:
            raise SummaryError(
                f"{label}.requests[{index}] has {len(token_ids)} token IDs, "
                f"expected={output_per_request}"
            )
        request_token_ids.append(token_ids)
    return {
        "concurrency": concurrency,
        "request_count": request_count,
        "completed_requests": completed_requests,
        "prompt_tokens": prompt_tokens,
        "output_tokens_per_request": output_per_request,
        "output_tokens": output_tokens,
        "output_token_ids_sha256": output_token_ids_sha256(request_token_ids),
        "output_tokens_per_second": positive_float(
            phase.get("output_tokens_per_second"), f"{label}.output_tokens_per_second"
        ),
        "elapsed_seconds": positive_float(
            phase.get("elapsed_seconds"), f"{label}.elapsed_seconds"
        ),
        "mean_ttft_ms": positive_float(
            phase.get("mean_ttft_ms"), f"{label}.mean_ttft_ms"
        ),
        "p99_ttft_ms": positive_float(
            phase.get("p99_ttft_ms"), f"{label}.p99_ttft_ms"
        ),
        "mean_tpot_ms": positive_float(
            phase.get("mean_tpot_ms"), f"{label}.mean_tpot_ms"
        ),
        "p99_tpot_ms": positive_float(
            phase.get("p99_tpot_ms"), f"{label}.p99_tpot_ms"
        ),
        "latency_clock_scope": "vllm_engine_request_timestamps",
    }


def summarize_kt_phase(
    phase: dict[str, Any], concurrency: int, output_per_request: int, case_name: str
) -> dict[str, Any]:
    label = f"{case_name}/c{concurrency}"
    request_count = strict_int(phase.get("request_count"), f"{label}.request_count")
    completed_requests = strict_int(
        phase.get("completed_requests"), f"{label}.completed_requests"
    )
    failed_requests = strict_int(phase.get("failed_requests"), f"{label}.failed_requests")
    expected_output_tokens = strict_int(
        phase.get("expected_output_tokens"), f"{label}.expected_output_tokens"
    )
    completed_output_tokens = strict_int(
        phase.get("completed_output_tokens"), f"{label}.completed_output_tokens"
    )
    expected = concurrency * output_per_request
    if (
        request_count != concurrency
        or completed_requests != concurrency
        or failed_requests != 0
    ):
        raise SummaryError(
            f"{label}: requests={request_count}, completed={completed_requests}, "
            f"failed={failed_requests}, expected={concurrency}/0"
        )
    if expected_output_tokens != expected or completed_output_tokens != expected:
        raise SummaryError(
            f"{label}: expected/completed output tokens="
            f"{expected_output_tokens}/{completed_output_tokens}, required={expected}"
        )
    if phase.get("all_requests_produced_fixed_output") is not True:
        raise SummaryError(f"{label}: all_requests_produced_fixed_output is not true")

    requests = phase.get("requests")
    if not isinstance(requests, list) or len(requests) != concurrency:
        raise SummaryError(f"{label}: expected {concurrency} request records")
    prompt_tokens = 0
    request_token_ids: list[list[int]] = []
    for index, record in enumerate(requests):
        if not isinstance(record, dict):
            raise SummaryError(f"{label}.requests[{index}] is not an object")
        if record.get("success") is not True:
            raise SummaryError(f"{label}.requests[{index}] did not succeed")
        completion_tokens = strict_int(
            record.get("completion_tokens"),
            f"{label}.requests[{index}].completion_tokens",
        )
        if completion_tokens != output_per_request:
            raise SummaryError(
                f"{label}.requests[{index}] completion_tokens={completion_tokens}, "
                f"expected={output_per_request}"
            )
        prompt_tokens += strict_int(
            record.get("server_prompt_tokens"),
            f"{label}.requests[{index}].server_prompt_tokens",
        )
        token_ids = record.get("output_token_ids")
        if not isinstance(token_ids, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in token_ids
        ):
            raise SummaryError(
                f"{label}.requests[{index}].output_token_ids is invalid"
            )
        if len(token_ids) != output_per_request:
            raise SummaryError(
                f"{label}.requests[{index}] has {len(token_ids)} token IDs, "
                f"expected={output_per_request}"
            )
        request_token_ids.append(token_ids)
    return {
        "concurrency": concurrency,
        "request_count": request_count,
        "completed_requests": completed_requests,
        "prompt_tokens": prompt_tokens,
        "output_tokens_per_request": output_per_request,
        "output_tokens": completed_output_tokens,
        "output_token_ids_sha256": output_token_ids_sha256(request_token_ids),
        "output_tokens_per_second": positive_float(
            phase.get("output_tokens_per_second"), f"{label}.output_tokens_per_second"
        ),
        "elapsed_seconds": positive_float(
            phase.get("elapsed_seconds"), f"{label}.elapsed_seconds"
        ),
        "mean_ttft_ms": positive_float(
            phase.get("mean_ttft_ms"), f"{label}.mean_ttft_ms"
        ),
        "p99_ttft_ms": positive_float(
            phase.get("p99_ttft_ms"), f"{label}.p99_ttft_ms"
        ),
        "mean_tpot_ms": positive_float(
            phase.get("mean_tpot_ms"), f"{label}.mean_tpot_ms"
        ),
        "p99_tpot_ms": positive_float(
            phase.get("p99_tpot_ms"), f"{label}.p99_tpot_ms"
        ),
        "latency_clock_scope": "localhost_sse_client_timestamps",
    }


def _parse_number(value: str, label: str) -> float:
    cleaned = value.strip()
    if cleaned in {"", "-", "N/A", "[N/A]"}:
        raise SummaryError(f"missing numeric value for {label}: {value!r}")
    try:
        return float(cleaned)
    except ValueError as exc:
        raise SummaryError(f"invalid numeric value for {label}: {value!r}") from exc


def summarize_nvml_csv(path: Path, scope: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
        raise SummaryError(f"cannot read NVML samples {path}: {exc}") from exc

    memory_used: list[float] = []
    gpu_util: list[float] = []
    memory_util: list[float] = []
    power_watts: list[float] = []
    pcie_generations: list[float] = []
    pcie_widths: list[float] = []
    for line_number, row in enumerate(rows, start=1):
        if not row or all(not item.strip() for item in row):
            continue
        # timestamp,index,memory.used,util.gpu,util.memory,power,...[,gen,width]
        if len(row) < 6:
            raise SummaryError(
                f"{path}:{line_number}: expected at least 6 CSV columns, got {len(row)}"
            )
        memory_used.append(_parse_number(row[2], f"{path}:{line_number}:memory.used"))
        gpu_util.append(_parse_number(row[3], f"{path}:{line_number}:utilization.gpu"))
        memory_util.append(
            _parse_number(row[4], f"{path}:{line_number}:utilization.memory")
        )
        power_watts.append(_parse_number(row[5], f"{path}:{line_number}:power.draw"))
        if len(row) >= 10:
            pcie_generations.append(
                _parse_number(row[8], f"{path}:{line_number}:pcie.gen")
            )
            pcie_widths.append(
                _parse_number(row[9], f"{path}:{line_number}:pcie.width")
            )
    if not memory_used:
        raise SummaryError(f"no valid NVML samples in {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "scope": scope,
        "sample_count": len(memory_used),
        "memory_used_mib_min": min(memory_used),
        "memory_used_mib_mean": statistics.fmean(memory_used),
        "memory_used_mib_peak": max(memory_used),
        "gpu_utilization_pct_mean": statistics.fmean(gpu_util),
        "gpu_utilization_pct_peak": max(gpu_util),
        "memory_utilization_pct_mean": statistics.fmean(memory_util),
        "memory_utilization_pct_peak": max(memory_util),
        "power_draw_w_mean": statistics.fmean(power_watts),
        "power_draw_w_peak": max(power_watts),
        "pcie_link_gen_min": min(pcie_generations) if pcie_generations else None,
        "pcie_link_gen_max": max(pcie_generations) if pcie_generations else None,
        "pcie_link_width_min": min(pcie_widths) if pcie_widths else None,
        "pcie_link_width_max": max(pcie_widths) if pcie_widths else None,
    }


def summarize_pcie_dmon(path: Path, scope: str) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise SummaryError(f"cannot read PCIe samples {path}: {exc}") from exc
    rx_mb_s: list[float] = []
    tx_mb_s: list[float] = []
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise SummaryError(
                f"{path}:{line_number}: expected date time gpu rxpci txpci, got {parts}"
            )
        try:
            int(parts[2])
            rx = float(parts[3])
            tx = float(parts[4])
        except ValueError as exc:
            raise SummaryError(f"invalid dmon sample at {path}:{line_number}") from exc
        rx_mb_s.append(rx)
        tx_mb_s.append(tx)
    if not rx_mb_s:
        raise SummaryError(f"no valid PCIe dmon samples in {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "scope": scope,
        "sample_interval_seconds": 1,
        "sample_count": len(rx_mb_s),
        "rx_mb_per_second_mean": statistics.fmean(rx_mb_s),
        "rx_mb_per_second_peak": max(rx_mb_s),
        "tx_mb_per_second_mean": statistics.fmean(tx_mb_s),
        "tx_mb_per_second_peak": max(tx_mb_s),
        "sampled_rx_mb_sum_assuming_1s_intervals": sum(rx_mb_s),
        "sampled_tx_mb_sum_assuming_1s_intervals": sum(tx_mb_s),
    }


def summarize_resources(case_root: Path, mechanism: str) -> dict[str, Any]:
    if mechanism == "vllm":
        nvml_path = case_root / "resource_nvml.csv"
        pcie_path = case_root / "resource_pcie_dmon.txt"
        scope = "process_load_warmup_and_measured_phases"
    else:
        nvml_path = case_root / "phase_nvml.csv"
        pcie_path = case_root / "phase_pcie_dmon.txt"
        scope = "measured_phases_only_after_server_ready"
    if not nvml_path.is_file() or not pcie_path.is_file():
        raise SummaryError(
            f"{case_root.name}: missing resource samples: {nvml_path}, {pcie_path}"
        )
    return {
        "nvml": summarize_nvml_csv(nvml_path, scope),
        "pcie_dmon": summarize_pcie_dmon(pcie_path, scope),
    }


def find_vllm_manifest(case_root: Path) -> Path:
    candidates = unique_paths(
        [case_root / "cpu_offload_manifest.jsonl"]
        + sorted(case_root.glob("*manifest*.jsonl"))
    )
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise SummaryError(f"{case_root.name}: no vLLM CPU-offload manifest found")
    if len(existing) > 1:
        raise SummaryError(f"{case_root.name}: ambiguous vLLM manifests: {existing}")
    return existing[0]


def summarize_vllm_manifest(path: Path) -> dict[str, Any]:
    records = read_jsonl(path)
    parameters = [record for record in records if record.get("event") == "cpu_offload_parameter"]
    if not parameters:
        raise SummaryError(f"no cpu_offload_parameter records in {path}")
    signature: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    backends: set[str] = set()
    for index, record in enumerate(parameters):
        name = record.get("full_parameter_name")
        if not isinstance(name, str) or not name:
            raise SummaryError(f"{path}: parameter record {index} has no name")
        if name in seen_names:
            raise SummaryError(f"{path}: duplicate offloaded parameter {name}")
        seen_names.add(name)
        backend = record.get("offload_backend")
        if not isinstance(backend, str) or not backend:
            raise SummaryError(f"{path}: {name} has no offload_backend")
        backends.add(backend)
        if backend == "copy" and record.get("copy_on_forward") is not True:
            raise SummaryError(f"{path}: {name} is not marked copy_on_forward")
        if backend == "uva":
            if record.get("uva_offloading") is not True:
                raise SummaryError(f"{path}: {name} is not marked uva_offloading")
            if not str(record.get("device_after", "")).startswith("cuda"):
                raise SummaryError(f"{path}: {name} has no CUDA UVA view")
        if backend not in {"copy", "uva"}:
            raise SummaryError(f"{path}: unsupported baseline backend {backend!r}")
        if record.get("cpu_data_pinned") is not True:
            raise SummaryError(f"{path}: {name} is not backed by pinned host memory")
        byte_count = strict_int(record.get("bytes"), f"{path}:{name}.bytes")
        if byte_count <= 0:
            raise SummaryError(f"{path}: {name} has non-positive bytes={byte_count}")
        signature.append(
            {
                "name": name,
                "bytes": byte_count,
                "shape": record.get("shape"),
                "dtype": record.get("dtype"),
                "order": strict_int(
                    record.get("offload_order"), f"{path}:{name}.offload_order"
                ),
            }
        )
    signature.sort(key=lambda item: item["order"])
    if len(backends) != 1:
        raise SummaryError(f"{path}: mixed vLLM offload backends: {sorted(backends)}")
    actual_bytes = sum(item["bytes"] for item in signature)
    expert_bytes = sum(
        item["bytes"] for item in signature if ".mlp.experts." in item["name"]
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "parameter_count": len(signature),
        "offload_backend": next(iter(backends)),
        "actual_offload_bytes": actual_bytes,
        "actual_offload_gib": actual_bytes / (1 << 30),
        "expert_parameter_bytes": expert_bytes,
        "expert_parameter_gib": expert_bytes / (1 << 30),
        "parameter_signature_sha256": canonical_sha256(signature),
    }


def extract_kt_plan(value: dict[str, Any], source: Path) -> dict[str, Any] | None:
    plans = value.get("plans")
    if isinstance(plans, list):
        if len(plans) != 1 or not isinstance(plans[0], dict):
            raise SummaryError(f"{source}: expected exactly one KT plan entry")
        entry = plans[0]
    elif all(key in value for key in ("hot_experts", "cold_experts")):
        entry = value
    else:
        return None

    hot = strict_int(entry.get("hot_experts"), f"{source}:hot_experts")
    cold = strict_int(entry.get("cold_experts"), f"{source}:cold_experts")
    total = strict_int(entry.get("total_experts", hot + cold), f"{source}:total_experts")
    if hot < 0 or cold < 0 or hot + cold != total or total <= 0:
        raise SummaryError(
            f"{source}: invalid KT split hot={hot} cold={cold} total={total}"
        )
    ratio_raw = entry.get("kt_gpu_experts_ratio_cli", entry.get("kt_gpu_experts_ratio"))
    try:
        ratio = float(ratio_raw)
    except (TypeError, ValueError) as exc:
        raise SummaryError(f"{source}: invalid KT GPU ratio {ratio_raw!r}") from exc
    recomputed_hot = int(total * ratio)
    if recomputed_hot != hot:
        raise SummaryError(
            f"{source}: KT ratio maps to {recomputed_hot} hot experts, expected {hot}"
        )
    return {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "hot_experts": hot,
        "cold_experts": cold,
        "total_experts": total,
        "kt_gpu_experts_ratio": ratio,
        "kt_gpu_experts_ratio_cli": str(ratio_raw),
        "target_gib": entry.get("target_gib"),
        "input_plan": entry.get("input_plan"),
        "input_plan_sha256": entry.get("input_plan_sha256"),
        "logical_count_sha256": entry.get("logical_count_sha256"),
    }


def add_kt_cold_bytes(
    manifest_summary: dict[str, Any], case_root: Path
) -> dict[str, Any]:
    candidates: list[Path] = []
    configured = manifest_summary.get("input_plan")
    if isinstance(configured, str) and configured:
        candidates.append(Path(configured))
    candidates.extend(sorted((case_root / "input_plan").glob("*.json")))
    plan_path = next((path for path in unique_paths(candidates) if path.is_file()), None)
    if plan_path is None:
        raise SummaryError(
            f"{case_root.name}: KT manifest has no readable input plan for byte accounting"
        )
    plan = read_json(plan_path)
    metadata = plan.get("metadata")
    if not isinstance(metadata, dict):
        raise SummaryError(f"{plan_path}: missing plan metadata")
    expert_bytes = strict_int(metadata.get("expert_bytes"), f"{plan_path}:expert_bytes")
    expert_scale_bytes = strict_int(
        metadata.get("expert_scale_bytes"), f"{plan_path}:expert_scale_bytes"
    )
    expert_bytes_including_scales = strict_int(
        metadata.get("expert_bytes_including_scales"),
        f"{plan_path}:expert_bytes_including_scales",
    )
    cold_main_weight_bytes = strict_int(
        metadata.get("actual_offload_bytes"), f"{plan_path}:actual_offload_bytes"
    )
    expected = manifest_summary["cold_experts"] * expert_bytes
    if cold_main_weight_bytes != expected:
        raise SummaryError(
            f"{plan_path}: cold bytes={cold_main_weight_bytes}, expected "
            f"cold_experts*expert_bytes={expected}"
        )
    if expert_bytes + expert_scale_bytes != expert_bytes_including_scales:
        raise SummaryError(
            f"{plan_path}: expert byte components do not add up: "
            f"{expert_bytes}+{expert_scale_bytes}!={expert_bytes_including_scales}"
        )
    cold_bytes_including_scales = strict_int(
        metadata.get("actual_offload_bytes_including_scales"),
        f"{plan_path}:actual_offload_bytes_including_scales",
    )
    expected_including_scales = (
        manifest_summary["cold_experts"] * expert_bytes_including_scales
    )
    if cold_bytes_including_scales != expected_including_scales:
        raise SummaryError(
            f"{plan_path}: cold bytes including scales={cold_bytes_including_scales}, "
            f"expected={expected_including_scales}"
        )
    input_sha = sha256_file(plan_path)
    declared_sha = manifest_summary.get("input_plan_sha256")
    if isinstance(declared_sha, str) and declared_sha != input_sha:
        raise SummaryError(
            f"{plan_path}: input plan SHA mismatch: declared={declared_sha} actual={input_sha}"
        )
    return manifest_summary | {
        "input_plan": str(plan_path.resolve()),
        "input_plan_sha256": input_sha,
        "expert_main_weight_bytes": expert_bytes,
        "expert_scale_bytes": expert_scale_bytes,
        "expert_bytes_including_scales": expert_bytes_including_scales,
        "cold_expert_main_weight_bytes": cold_main_weight_bytes,
        "cold_expert_main_weight_gib": cold_main_weight_bytes / (1 << 30),
        "cold_expert_bytes_including_scales": cold_bytes_including_scales,
        "cold_expert_gib_including_scales": cold_bytes_including_scales / (1 << 30),
        "byte_scope": "three FP8 expert matrices plus their weight_scale_inv tensors",
    }


def find_kt_manifest(case_root: Path, run_root: Path) -> dict[str, Any]:
    candidates = unique_paths(
        [
            case_root / "kt_stats" / "manifest.json",
            case_root / "manifest.json",
            case_root / "kt_manifest.json",
        ]
        + sorted(case_root.glob("*manifest*.json"))
        + [
            run_root / "kt_stats" / "manifest.json",
            run_root / "kt_manifest.json",
        ]
        + sorted(run_root.glob("*kt*manifest*.json"))
    )
    matches: list[dict[str, Any]] = []
    for path in candidates:
        if not path.is_file():
            continue
        value = read_json(path)
        extracted = extract_kt_plan(value, path)
        if extracted is not None:
            matches.append(extracted)
    if not matches:
        raise SummaryError(f"{case_root.name}: no KT hot/cold manifest found")
    identities = {
        (
            item["hot_experts"],
            item["cold_experts"],
            item["total_experts"],
            item["kt_gpu_experts_ratio"],
            item["input_plan_sha256"],
        )
        for item in matches
    }
    if len(identities) != 1:
        raise SummaryError(f"{case_root.name}: conflicting KT manifests: {matches}")
    return add_kt_cold_bytes(matches[0], case_root)


def summarize_vllm_case(case_root: Path, run_root: Path) -> dict[str, Any]:
    result_path = case_root / "result.json"
    result = read_json(result_path)
    if result.get("status") != "ok":
        raise SummaryError(f"{case_root.name}: status={result.get('status')!r}, expected 'ok'")
    returncodes = check_returncode(case_root)
    config = result.get("config")
    if not isinstance(config, dict):
        raise SummaryError(f"{case_root.name}: result config is missing")
    fixture = fixture_metadata(
        case_root=case_root,
        run_root=run_root,
        configured_path=config.get("fixture"),
        declared_sha256=result.get("fixture_sha256"),
    )
    selected = select_phases(result.get("phases"), "vllm", case_root.name)
    phases = {
        concurrency: summarize_vllm_phase(selected[concurrency], concurrency, case_root.name)
        for concurrency in CONCURRENCIES
    }
    manifest = summarize_vllm_manifest(find_vllm_manifest(case_root))
    mechanism = (
        "vllm_native_cpu_offload_pinned_uva"
        if manifest["offload_backend"] == "uva"
        else "vllm_copy_on_forward_ablation"
    )
    run_match = re.search(r"(\d+)$", case_root.name)
    return {
        "name": case_root.name,
        "run_id": run_match.group(1) if run_match else case_root.name,
        "mechanism": mechanism,
        "result_path": str(result_path.resolve()),
        "status": "ok",
        "returncodes": returncodes,
        "fixture": fixture,
        "manifest": manifest,
        "resources": summarize_resources(case_root, "vllm"),
        "phases": {f"c{key}": value for key, value in phases.items()},
    }


def summarize_kt_case(case_root: Path, run_root: Path) -> dict[str, Any]:
    result_path = case_root / "benchmark.json"
    result = read_json(result_path)
    if result.get("status") != "ok":
        raise SummaryError(f"{case_root.name}: status={result.get('status')!r}, expected 'ok'")
    returncodes = check_returncode(case_root)
    config = result.get("config")
    fixture_value = result.get("fixture")
    if not isinstance(config, dict) or not isinstance(fixture_value, dict):
        raise SummaryError(f"{case_root.name}: benchmark config/fixture is missing")
    output_per_request = strict_int(
        config.get("output_tokens_per_request"),
        f"{case_root.name}.config.output_tokens_per_request",
    )
    fixture = fixture_metadata(
        case_root=case_root,
        run_root=run_root,
        configured_path=fixture_value.get("path"),
        declared_sha256=fixture_value.get("sha256"),
    )
    selected = select_phases(result.get("phases"), "kt", case_root.name)
    phases = {
        concurrency: summarize_kt_phase(
            selected[concurrency], concurrency, output_per_request, case_root.name
        )
        for concurrency in CONCURRENCIES
    }
    run_match = re.search(r"(\d+)$", case_root.name)
    return {
        "name": case_root.name,
        "run_id": run_match.group(1) if run_match else case_root.name,
        "mechanism": "ktransformers_cpu_experts",
        "result_path": str(result_path.resolve()),
        "status": "ok",
        "returncodes": returncodes,
        "fixture": fixture,
        "manifest": find_kt_manifest(case_root, run_root),
        "resources": summarize_resources(case_root, "kt"),
        "phases": {f"c{key}": value for key, value in phases.items()},
    }


def validate_cross_case_integrity(
    vllm_cases: list[dict[str, Any]], kt_cases: list[dict[str, Any]]
) -> dict[str, Any]:
    fixture_hashes = {
        case["fixture"]["sha256"] for case in [*vllm_cases, *kt_cases]
    }
    if len(fixture_hashes) != 1:
        raise SummaryError(f"fixture SHA-256 differs across cases: {sorted(fixture_hashes)}")

    vllm_signatures = {
        (
            case["manifest"]["parameter_signature_sha256"],
            case["manifest"]["actual_offload_bytes"],
            case["manifest"]["offload_backend"],
        )
        for case in vllm_cases
    }
    if len(vllm_signatures) != 1:
        raise SummaryError(f"vLLM offload manifests differ across cases: {vllm_signatures}")
    kt_splits = {
        (
            case["manifest"]["hot_experts"],
            case["manifest"]["cold_experts"],
            case["manifest"]["total_experts"],
            case["manifest"]["kt_gpu_experts_ratio"],
            case["manifest"]["cold_expert_main_weight_bytes"],
        )
        for case in kt_cases
    }
    if len(kt_splits) != 1:
        raise SummaryError(f"KT expert splits differ across cases: {kt_splits}")

    hashes_by_concurrency: dict[str, dict[str, Any]] = {}
    for concurrency in CONCURRENCIES:
        phase_name = f"c{concurrency}"
        output_per_request = {
            case["phases"][phase_name]["output_tokens_per_request"]
            for case in [*vllm_cases, *kt_cases]
        }
        output_tokens = {
            case["phases"][phase_name]["output_tokens"]
            for case in [*vllm_cases, *kt_cases]
        }
        prompt_tokens = {
            case["phases"][phase_name]["prompt_tokens"]
            for case in [*vllm_cases, *kt_cases]
        }
        if len(output_per_request) != 1 or len(output_tokens) != 1:
            raise SummaryError(
                f"{phase_name}: output-token counts differ: "
                f"per_request={output_per_request}, total={output_tokens}"
            )
        if len(prompt_tokens) != 1:
            raise SummaryError(
                f"{phase_name}: prompt-token totals differ across runtimes: {prompt_tokens}"
            )
        vllm_output_hashes = {
            case["phases"][phase_name]["output_token_ids_sha256"]
            for case in vllm_cases
        }
        kt_output_hashes = {
            case["phases"][phase_name]["output_token_ids_sha256"]
            for case in kt_cases
        }
        if len(vllm_output_hashes) != 1 or len(kt_output_hashes) != 1:
            raise SummaryError(
                f"{phase_name}: output token hashes are not repeatable within a "
                f"runtime: vllm={sorted(vllm_output_hashes)} "
                f"kt={sorted(kt_output_hashes)}"
            )
        vllm_hash = next(iter(vllm_output_hashes))
        kt_hash = next(iter(kt_output_hashes))
        hashes_by_concurrency[phase_name] = {
            "vllm": vllm_hash,
            "ktransformers": kt_hash,
            "equal_across_mechanisms": vllm_hash == kt_hash,
        }
    return {
        "all_status_ok": True,
        "all_observed_returncodes_zero": True,
        "all_required_phases_present": True,
        "all_request_and_output_token_counts_valid": True,
        "prompt_token_totals_equal_across_runtimes": True,
        "output_token_hashes_repeatable_within_each_runtime": True,
        "output_token_hashes_equal_across_mechanisms": all(
            item["equal_across_mechanisms"]
            for item in hashes_by_concurrency.values()
        ),
        "output_token_ids_sha256_by_concurrency": hashes_by_concurrency,
        "fixture_sha256_equal": True,
        "fixture_sha256": next(iter(fixture_hashes)),
        "vllm_manifests_identical": True,
        "kt_manifests_identical": True,
    }


def build_comparisons(
    vllm_cases: list[dict[str, Any]], kt_cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    vllm_by_id = {case["run_id"]: case for case in vllm_cases}
    kt_by_id = {case["run_id"]: case for case in kt_cases}
    paired_ids = sorted(set(vllm_by_id) & set(kt_by_id))
    for concurrency in CONCURRENCIES:
        phase_name = f"c{concurrency}"
        vllm_runs = [
            {
                "case": case["name"],
                "run_id": case["run_id"],
                "output_tokens_per_second": case["phases"][phase_name][
                    "output_tokens_per_second"
                ],
            }
            for case in vllm_cases
        ]
        kt_runs = [
            {
                "case": case["name"],
                "run_id": case["run_id"],
                "output_tokens_per_second": case["phases"][phase_name][
                    "output_tokens_per_second"
                ],
            }
            for case in kt_cases
        ]
        vllm_values = [item["output_tokens_per_second"] for item in vllm_runs]
        kt_values = [item["output_tokens_per_second"] for item in kt_runs]
        vllm_mean = statistics.fmean(vllm_values)
        kt_mean = statistics.fmean(kt_values)
        paired = []
        for run_id in paired_ids:
            vllm_value = vllm_by_id[run_id]["phases"][phase_name][
                "output_tokens_per_second"
            ]
            kt_value = kt_by_id[run_id]["phases"][phase_name][
                "output_tokens_per_second"
            ]
            paired.append(
                {
                    "run_id": run_id,
                    "vllm_output_tokens_per_second": vllm_value,
                    "kt_output_tokens_per_second": kt_value,
                    "kt_over_vllm_throughput_ratio": kt_value / vllm_value,
                }
            )
        rows.append(
            {
                "concurrency": concurrency,
                "phase": phase_name,
                "vllm_runs": vllm_runs,
                "kt_runs": kt_runs,
                "vllm_run_count": len(vllm_runs),
                "kt_run_count": len(kt_runs),
                "vllm_mean_output_tokens_per_second": vllm_mean,
                "kt_mean_output_tokens_per_second": kt_mean,
                "kt_over_vllm_mean_throughput_ratio": kt_mean / vllm_mean,
                "kt_vs_vllm_mean_throughput_change_pct":
                    (kt_mean / vllm_mean - 1.0) * 100.0,
                "vllm_output_token_ids_sha256": vllm_cases[0]["phases"][
                    phase_name
                ]["output_token_ids_sha256"],
                "kt_output_token_ids_sha256": kt_cases[0]["phases"][phase_name][
                    "output_token_ids_sha256"
                ],
                "output_token_ids_equal_across_mechanisms": (
                    vllm_cases[0]["phases"][phase_name][
                        "output_token_ids_sha256"
                    ]
                    == kt_cases[0]["phases"][phase_name][
                        "output_token_ids_sha256"
                    ]
                ),
                "paired_runs": paired,
            }
        )
    return rows


def markdown_structure(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    columns = [
        "Concurrency",
        "vLLM runs (tok/s)",
        "KT runs (tok/s)",
        "vLLM mean",
        "KT mean",
        "KT/vLLM",
    ]
    rows = []
    for item in comparisons:
        rows.append(
            [
                f"c{item['concurrency']}",
                ", ".join(
                    f"{run['case']}={run['output_tokens_per_second']:.3f}"
                    for run in item["vllm_runs"]
                ),
                ", ".join(
                    f"{run['case']}={run['output_tokens_per_second']:.3f}"
                    for run in item["kt_runs"]
                ),
                f"{item['vllm_mean_output_tokens_per_second']:.3f}",
                f"{item['kt_mean_output_tokens_per_second']:.3f}",
                f"{item['kt_over_vllm_mean_throughput_ratio']:.4f}",
            ]
        )
    return {"columns": columns, "rows": rows}


def render_markdown(snapshot: dict[str, Any]) -> str:
    table = snapshot["markdown_table"]
    columns = table["columns"]
    lines = [
        "# KTransformers vs vLLM native offload pilot",
        "",
        f"- vLLM runs: {len(snapshot['cases']['vllm'])}",
        f"- KT runs: {len(snapshot['cases']['kt'])}",
        f"- Fixture SHA-256: `{snapshot['integrity']['fixture_sha256']}`",
        f"- vLLM actual offload: {snapshot['manifests']['vllm']['actual_offload_bytes']:,} bytes",
        (
            "- KT experts: "
            f"{snapshot['manifests']['kt']['cold_experts']:,} cold / "
            f"{snapshot['manifests']['kt']['hot_experts']:,} hot"
        ),
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in table["rows"])
    return "\n".join(lines) + "\n"


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize vLLM native-offload and KT c1/c4/c8 pilot repeats."
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        raise SystemExit(f"run root is not a directory: {run_root}")
    case_roots = {name: run_root / name for name in CASE_ORDER}
    required_results = {
        "vllm_a1": "result.json",
        "kt_b1": "benchmark.json",
        "kt_b2": "benchmark.json",
        "vllm_a2": "result.json",
    }
    missing = [
        str(case_roots[name] / result_name)
        for name, result_name in required_results.items()
        if not (case_roots[name] / result_name).is_file()
    ]
    if missing:
        raise SystemExit(
            "missing required fixed pilot artifacts: " + ", ".join(missing)
        )
    vllm_roots = [case_roots["vllm_a1"], case_roots["vllm_a2"]]
    kt_roots = [case_roots["kt_b1"], case_roots["kt_b2"]]

    try:
        vllm_cases = [summarize_vllm_case(path, run_root) for path in vllm_roots]
        kt_cases = [summarize_kt_case(path, run_root) for path in kt_roots]
        integrity = validate_cross_case_integrity(vllm_cases, kt_cases)
        comparisons = build_comparisons(vllm_cases, kt_cases)
    except SummaryError as exc:
        raise SystemExit(f"error: {exc}") from exc

    representative_vllm = vllm_cases[0]["manifest"]
    representative_kt = kt_cases[0]["manifest"]
    snapshot: dict[str, Any] = {
        "experiment": "ktransformers_vs_vllm_native_offload_pilot",
        "run_root": str(run_root),
        "case_order": list(CASE_ORDER),
        "required_concurrencies": list(CONCURRENCIES),
        "cases": {"vllm": vllm_cases, "kt": kt_cases},
        "manifests": {
            "vllm": representative_vllm,
            "kt": representative_kt,
            "vllm_actual_minus_kt_cold_main_weight_bytes": (
                representative_vllm["actual_offload_bytes"]
                - representative_kt["cold_expert_main_weight_bytes"]
            ),
            "vllm_actual_minus_kt_cold_bytes_including_scales": (
                representative_vllm["actual_offload_bytes"]
                - representative_kt["cold_expert_bytes_including_scales"]
            ),
        },
        "throughput_comparison": comparisons,
        "integrity": integrity,
        "measurement_notes": {
            "primary_metric": "complete phase output tokens per wall-clock second",
            "throughput_scope": "full runtime path for each serving stack",
            "latency_cross_runtime_comparison": False,
            "latency_reason": (
                "vLLM uses engine request timestamps while KTransformers uses "
                "localhost SSE client timestamps"
            ),
            "resource_samples_cross_runtime_comparison": False,
            "resource_reason": (
                "vLLM samples include load and warmup; KT samples cover only "
                "measured phases"
            ),
        },
    }
    snapshot["markdown_table"] = markdown_structure(comparisons)

    output = args.output or run_root / "snapshot.json"
    atomic_write(output, json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    if args.markdown_output is not None:
        atomic_write(args.markdown_output, render_markdown(snapshot))
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
