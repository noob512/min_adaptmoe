#!/usr/bin/env python3
"""Verify and summarize the complete E1 maximum-resident experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


SYSTEMS = ("ours", "ktransformers", "moe_infinity")
EXPECTED_REPEATS = 5
EXPECTED_MEASUREMENT_REQUESTS = 10
EXPECTED_MINIMUM_START_INTERVAL_SECONDS = 6.0
T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite verifier output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def phase_signature(client: dict[str, Any]) -> tuple[list[list[Any]] | None, list[str]]:
    failures: list[str] = []
    phases = client.get("phases")
    if not isinstance(phases, list) or len(phases) != 1:
        return None, ["client must contain exactly one measurement phase"]
    phase = phases[0]
    requests = phase.get("requests")
    config = client.get("config", {})
    checks = {
        "client status": client.get("status") == "ok",
        "concurrency": phase.get("concurrency") == 1,
        "configured request count": (
            config.get("measurement_requests") == EXPECTED_MEASUREMENT_REQUESTS
        ),
        "observed request count": (
            phase.get("request_count") == EXPECTED_MEASUREMENT_REQUESTS
            and isinstance(requests, list)
            and len(requests) == EXPECTED_MEASUREMENT_REQUESTS
        ),
        "successful request count": (
            phase.get("successful_requests") == EXPECTED_MEASUREMENT_REQUESTS
            and phase.get("failed_requests") == 0
        ),
        "configured rate limit": math.isclose(
            float(config.get("minimum_request_start_interval_seconds", -1)),
            EXPECTED_MINIMUM_START_INTERVAL_SECONDS,
            abs_tol=1e-9,
        ),
        "observed rate limit": phase.get("request_rate_limit_compliant") is True,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    if not isinstance(requests, list):
        return None, failures
    signature: list[list[Any]] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            failures.append(f"request {index} is not an object")
            continue
        token_hash = request.get("output_token_ids_sha256")
        if (
            request.get("success") is not True
            or request.get("http_status") != 200
            or request.get("completion_tokens") != 32
            or not isinstance(token_hash, str)
        ):
            failures.append(f"request {index} failed exact-output checks")
            continue
        signature.append([request.get("fixture_index"), token_hash])
    return signature if len(signature) == EXPECTED_MEASUREMENT_REQUESTS else None, failures


def parse_ready_hbm_mib(path: Path) -> int | None:
    value = load_json(path)
    rows = value.get("gpu_rows")
    if not isinstance(rows, list) or len(rows) != 1:
        return None
    fields = [field.strip() for field in str(rows[0]).split(",")]
    try:
        return int(fields[3])
    except (IndexError, ValueError):
        return None


def parse_nvml_peak_mib(path: Path) -> int | None:
    if not path.is_file():
        return None
    values: list[int] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            try:
                values.append(int(float(row[4].strip())))
            except (IndexError, ValueError):
                continue
    return max(values) if values else None


def candidate_attempts(repeat_root: Path) -> list[Path]:
    if not repeat_root.is_dir():
        return []
    return sorted(
        (path for path in repeat_root.glob("attempt_*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
        reverse=True,
    )


def option_value(command: list[str], option: str) -> str | None:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        return None
    return command[positions[0] + 1]


def maximum_resident_certificate(system: str, command_path: Path) -> dict[str, Any]:
    command = shlex.split(command_path.read_text(encoding="utf-8"))
    if system == "ours":
        checks = {
            "cpu_offload_gb_zero": option_value(command, "--cpu-offload-gb") == "0",
            "fixed_small_kv": option_value(command, "--kv-cache-memory-bytes") == "67108864",
            "single_sequence": option_value(command, "--max-num-seqs") == "1",
            "no_dynamic_expert_plan": not any("expert-plan" in value for value in command),
        }
        mode = "vLLM eager; CUDA Graph N/A at all-weight 32-GiB headroom"
    elif system == "ktransformers":
        checks = {
            "all_64_experts_per_layer": option_value(command, "--kt-num-gpu-experts") == "64",
            "uniform_placement": option_value(command, "--kt-expert-placement-strategy") == "uniform",
            "single_request": option_value(command, "--max-running-requests") == "1",
            "cuda_graph_enabled": "--disable-cuda-graph" not in command,
        }
        mode = "KTransformers maximum GPU experts with decode CUDA Graph"
    else:
        checks = {
            "maximum_batch_one": option_value(command, "--max-batch-size") == "1",
            "device_memory_ratio_frozen": option_value(command, "--device-memory-ratio") == "0.75",
            "kv_cache_ratio_frozen": option_value(command, "--kv-cache-ratio") == "0.01",
        }
        mode = "MoE-Infinity maximum-feasible public-demand cache; CUDA Graph N/A"
    return {"mode": mode, "checks": checks, "ok": all(checks.values())}


def summarize_blocks(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    mean = statistics.fmean(values)
    result: dict[str, Any] = {
        "n": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
    if len(values) > 1:
        stdev = statistics.stdev(values)
        critical = T_CRITICAL_95.get(len(values), 1.96)
        half = critical * stdev / math.sqrt(len(values))
        result.update(
            {
                "sample_stdev": stdev,
                "ci95_method": "two-sided Student-t over independent server-block means",
                "ci95_low": mean - half,
                "ci95_high": mean + half,
                "ci95_half_width": half,
            }
        )
    return result


def expected_order(repeat: int) -> list[str]:
    rotations = (
        ["ours", "ktransformers", "moe_infinity"],
        ["ktransformers", "moe_infinity", "ours"],
        ["moe_infinity", "ours", "ktransformers"],
    )
    return rotations[(repeat - 1) % 3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("e1_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol-id", required=True)
    args = parser.parse_args()

    failures: list[str] = []
    execution_path = args.e1_root / "execution.tsv"
    rows: list[dict[str, str]] = []
    if execution_path.is_file():
        with execution_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    else:
        failures.append("missing E1 execution.tsv")
    execution_attempts = Counter(row.get("system") for row in rows)

    first_orders: dict[tuple[int, str], int] = {}
    for row in rows:
        try:
            key = (int(row["repeat"]), row["system"])
            first_orders.setdefault(key, int(row["order"]))
        except (KeyError, TypeError, ValueError):
            failures.append("malformed execution.tsv row")
    balanced_order = True
    for repeat in range(1, EXPECTED_REPEATS + 1):
        for order, system in enumerate(expected_order(repeat), 1):
            if first_orders.get((repeat, system)) != order:
                balanced_order = False
    if not balanced_order:
        failures.append("balanced cyclic block order was not preserved")

    cases: list[dict[str, Any]] = []
    signatures: set[str] = set()
    verified_counts: Counter[str] = Counter()
    metrics: dict[str, dict[str, list[float]]] = {
        system: {
            "ttft_ms": [],
            "tpot_ms": [],
            "e2e_ms": [],
            "output_tokens_per_second": [],
            "ready_hbm_mib": [],
            "peak_hbm_mib": [],
        }
        for system in SYSTEMS
    }

    for system in SYSTEMS:
        for repeat in range(1, EXPECTED_REPEATS + 1):
            repeat_root = (
                args.e1_root / system / "deepseek_v2_lite_bf16"
                / "maximum_resident" / f"repeat_{repeat:02d}"
            )
            case_root: Path | None = None
            for attempt in candidate_attempts(repeat_root):
                verification_path = attempt / "verification.json"
                if not verification_path.is_file():
                    continue
                verification = load_json(verification_path)
                if (
                    verification.get("status") == "ok"
                    and verification.get("protocol_id") == args.protocol_id
                ):
                    case_root = attempt
                    break
            if case_root is None:
                failures.append(f"no verified E1 attempt: {system}/repeat_{repeat:02d}")
                continue
            verified_counts[system] += 1
            required = {
                "verification": case_root / "verification.json",
                "client": case_root / "short_dialogue" / "client.json",
                "ready": case_root / "gpu.ready.json",
                "nvml": case_root / "nvml.csv",
                "command": case_root / "command.txt",
            }
            missing = [str(path) for path in required.values() if not path.is_file()]
            if missing:
                failures.append(f"missing E1 artifacts: {missing}")
                continue
            verification = load_json(required["verification"])
            client = load_json(required["client"])
            signature, signature_failures = phase_signature(client)
            failures.extend(
                f"{system}/repeat_{repeat:02d}: {value}"
                for value in signature_failures
            )
            signature_sha = None
            if signature is not None:
                encoded = json.dumps(signature, separators=(",", ":")).encode()
                signature_sha = hashlib.sha256(encoded).hexdigest()
                signatures.add(signature_sha)
            phase = client.get("phases", [{}])[0]
            ready_hbm = parse_ready_hbm_mib(required["ready"])
            peak_hbm = parse_nvml_peak_mib(required["nvml"])
            certificate = maximum_resident_certificate(system, required["command"])
            if not certificate["ok"]:
                failures.append(f"{system}/repeat_{repeat:02d}: resident certificate failed")
            values = {
                "ttft_ms": phase.get("ttft_ms", {}).get("mean"),
                "tpot_ms": phase.get("tpot_ms", {}).get("mean"),
                "e2e_ms": phase.get("e2e_ms", {}).get("mean"),
                "output_tokens_per_second": phase.get("output_tokens_per_second"),
                "ready_hbm_mib": ready_hbm,
                "peak_hbm_mib": peak_hbm,
            }
            for name, value in values.items():
                if isinstance(value, (int, float)):
                    metrics[system][name].append(float(value))
                else:
                    failures.append(f"{system}/repeat_{repeat:02d}: missing {name}")
            cases.append(
                {
                    "system": system,
                    "repeat": repeat,
                    "attempt": case_root.name,
                    "case_root": str(case_root.resolve()),
                    "measurement_requests": EXPECTED_MEASUREMENT_REQUESTS,
                    "minimum_request_start_interval_seconds": (
                        phase.get("observed_minimum_request_start_interval_seconds")
                    ),
                    "output_signature_sha256": signature_sha,
                    "maximum_resident_certificate": certificate,
                    **values,
                }
            )

    for system in SYSTEMS:
        if verified_counts[system] != EXPECTED_REPEATS:
            failures.append(
                f"{system}: expected five verified repeats, found {verified_counts[system]}"
            )
    exact_match = len(signatures) == 1 and len(cases) == 15
    if not exact_match:
        failures.append(f"cross-framework/repeat output signatures differ: {sorted(signatures)}")

    summaries = {
        system: {
            name: summarize_blocks(values)
            for name, values in system_metrics.items()
        }
        for system, system_metrics in metrics.items()
    }
    result = {
        "schema_version": 2,
        "status": "ok" if not failures else "error",
        "experiment": "E1_maximum_resident_framework_calibration",
        "scope": "formal_performance_candidate",
        "protocol_id": args.protocol_id,
        "measurement_unit": "independent server process block",
        "request_policy": {
            "concurrency": 1,
            "measurement_requests_per_block": EXPECTED_MEASUREMENT_REQUESTS,
            "minimum_request_start_interval_seconds": (
                EXPECTED_MINIMUM_START_INTERVAL_SECONDS
            ),
            "maximum_request_starts_per_minute": 10,
        },
        "checks": {
            "expected_process_blocks": 15,
            "observed_verified_process_blocks": len(cases),
            "execution_attempts_per_system": dict(execution_attempts),
            "verified_repeats_per_system": dict(verified_counts),
            "balanced_cyclic_order": balanced_order,
            "cross_framework_exact_token_match_for_all_10_prompts": exact_match,
            "output_signature_sha256": (
                next(iter(signatures)) if len(signatures) == 1 else None
            ),
        },
        "system_summaries": summaries,
        "cases": cases,
        "failures": failures,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
