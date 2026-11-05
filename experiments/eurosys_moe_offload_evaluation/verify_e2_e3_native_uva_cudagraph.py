#!/usr/bin/env python3
"""Verify and summarize the native pinned-UVA + FULL_SEPARATE E2/E3 run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


WORKLOADS = (
    "short_dialogue",
    "ordinary_chat",
    "rag_summary",
    "long_generation",
)
E3_WORKLOADS = (
    "short_dialogue",
    "ordinary_chat",
    "medium_context",
    "rag_summary",
    "long_generation",
)
FATAL = re.compile(
    r"CUDA error|illegal memory access|\bXid\b|PDE.*fault|PTE.*fault|"
    r"GPU page fault|out of memory|Traceback \(most recent call last\)",
    re.IGNORECASE,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def single_dir(root: Path, pattern: str) -> Path:
    matches = sorted(path for path in root.glob(pattern) if path.is_dir())
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one directory for {pattern!r} under {root}, got {matches}"
        )
    return matches[0]


def manifest_is_native_uva(path: Path) -> bool:
    if not path.is_file():
        return False
    for line in read_text(path).splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            record.get("event") == "cpu_offload_parameter"
            and record.get("offload_backend") == "uva"
            and record.get("cpu_data_pinned") is True
            and record.get("uva_available") is True
            and record.get("uva_offloading") is True
            and record.get("copy_on_forward") is False
        ):
            return True
    return False


def graph_checks(log: str, expected_mode: str) -> dict[str, bool]:
    audit_lines = [
        line for line in log.splitlines() if "CUDA_GRAPH_RUNTIME_AUDIT" in line
    ]
    checks = {
        "requested_graph_mode_realized": (
            f"CUDAGraphMode.{expected_mode}" in log and "enforce_eager=False" in log
        ),
        "decode_graph_captured": "Capturing CUDA graphs (decode, FULL)" in log,
        "graph_capture_finished": "Graph capturing finished" in log,
        "decode_graph_replayed": any(
            "phase=enabled:decode_step" in line and "runtime_mode=FULL" in line
            for line in audit_lines
        ),
        "no_decode_eager_runtime_dispatch": bool(audit_lines) and not any(
            "phase=enabled:decode_step" in line and "runtime_mode=NONE" in line
            for line in audit_lines
        ),
    }
    if expected_mode == "FULL_SEPARATE":
        checks.update({
            "prefill_graph_captured": (
                "Capturing CUDA graphs (pure prefill, FULL)" in log
            ),
            "prefill_graph_replayed": any(
                "phase=enabled:prefill_step" in line and "runtime_mode=FULL" in line
                for line in audit_lines
            ),
            "no_prefill_eager_runtime_dispatch": not any(
                "phase=enabled:prefill_step" in line and "runtime_mode=NONE" in line
                for line in audit_lines
            ),
        })
    elif expected_mode == "FULL_DECODE_ONLY":
        checks["prefill_eager_by_protocol"] = any(
            "phase=enabled:prefill_step" in line and "runtime_mode=NONE" in line
            for line in audit_lines
        )
    return checks


def collect_e2_c1(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for workload in WORKLOADS:
        case = single_dir(
            root,
            f"e2/c1/cases/native_pinned_uva/{workload}/*",
        )
        verification = read_json(case / "verification.json")
        log = read_text(case / "server.log")
        graph = graph_checks(log, "FULL_DECODE_ONLY")
        client = verification.get("checks", {}).get("clients", {})
        phases = next(iter(client.values()), {}).get("phases", [])
        phase = phases[0] if len(phases) == 1 else {}
        checks = {
            "base_verifier_ok": verification.get("status") == "ok",
            "native_uva_manifest": manifest_is_native_uva(
                case / "offload_manifest.jsonl"
            ),
            "no_fatal_log": not FATAL.search(log),
            **graph,
        }
        records.append({
            "experiment": "E2",
            "concurrency": 1,
            "workload": workload,
            "case": str(case),
            "status": "ok" if all(checks.values()) else "error",
            "checks": checks,
            "ttft_ms": phase.get("mean_ttft_ms"),
            "tpot_ms": phase.get("mean_tpot_ms"),
            "e2e_ms": phase.get("mean_e2e_ms"),
            "output_hashes": phase.get("request_output_hashes"),
        })
    return records


def collect_e2_c16(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for workload in WORKLOADS:
        case = single_dir(
            root,
            f"e2/c16/cases/native_pinned_uva/{workload}/*",
        )
        verification = read_json(case / "verification.json")
        result = read_json(case / "result.json")
        log = read_text(case / "run.log")
        graph = graph_checks(log, "FULL_DECODE_ONLY")
        measurement = result.get("measurement") or {}
        checks = {
            "base_verifier_ok": verification.get("status") == "ok",
            "native_uva_manifest": manifest_is_native_uva(
                case / "offload_manifest.jsonl"
            ),
            "no_fatal_log": not FATAL.search(log),
            **graph,
        }
        requests = measurement.get("requests") or []
        records.append({
            "experiment": "E2",
            "concurrency": 16,
            "workload": workload,
            "case": str(case),
            "status": "ok" if all(checks.values()) else "error",
            "checks": checks,
            "ttft_ms": (measurement.get("ttft_ms") or {}).get("mean"),
            "tpot_ms": (measurement.get("tpot_ms") or {}).get("mean"),
            "e2e_ms": (measurement.get("e2e_ms") or {}).get("mean"),
            "successful_requests": measurement.get("successful_requests"),
            "output_hashes": [
                request.get("output_token_ids_sha256") for request in requests
            ],
        })
    return records


def collect_e3(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for workload in E3_WORKLOADS:
        for batch in (1, 16):
            case = root / "e3" / "cases" / f"{workload}_b{batch}_uva"
            result = read_json(case / "result.json")
            log = read_text(case / "run.log")
            config = result.get("config") or {}
            summary = (result.get("phase_summary") or {}).get(str(batch), {})
            graph = graph_checks(log, "FULL_SEPARATE")
            checks = {
                "runner_returncode_zero": (
                    read_text(case / "returncode.txt").strip() == "0"
                ),
                "result_status_ok": result.get("status") == "ok",
                "native_uva_manifest": manifest_is_native_uva(
                    case / "manifest.jsonl"
                ),
                "full_separate_configured": (
                    config.get("cudagraph_mode") == "FULL_SEPARATE"
                    and config.get("enforce_eager") is False
                    and (config.get("compilation_config") or {}).get(
                        "cudagraph_mode"
                    ) == "FULL_SEPARATE"
                ),
                "one_measurement_repeat": summary.get("measurement_repeats") == 1,
                "no_fatal_log": not FATAL.search(log),
                **graph,
            }
            cycles = [
                cycle for cycle in result.get("cycles", [])
                if cycle.get("role") == "measurement"
            ]
            records.append({
                "experiment": "E3",
                "batch": batch,
                "workload": workload,
                "case": str(case),
                "status": "ok" if all(checks.values()) else "error",
                "checks": checks,
                "prefill_ms": (summary.get("prefill_step_ms") or {}).get("median"),
                "decode_ms": (summary.get("decode_repeat_median_ms") or {}).get(
                    "median"
                ),
                "ttft_ms": (summary.get("request_ttft_p50_ms") or {}).get("median"),
                "tpot_ms": (summary.get("request_tpot_p50_ms") or {}).get("median"),
                "e2e_ms": (summary.get("request_latency_p50_ms") or {}).get(
                    "median"
                ),
                "output_hashes": [
                    cycle.get("output_token_ids_sha256") for cycle in cycles
                ],
            })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    output = args.output or root / "verification.json"
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        records.extend(collect_e2_c1(root))
        records.extend(collect_e2_c16(root))
        records.extend(collect_e3(root))
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    kernel_log = read_text(root / "kernel.since.log") \
        if (root / "kernel.since.log").is_file() else ""
    execution_rows = []
    execution_path = root / "execution.tsv"
    if execution_path.is_file():
        execution_rows = [
            line.split("\t", maxsplit=1)
            for line in read_text(execution_path).splitlines()[1:]
            if "\t" in line
        ]
    gpu_after = []
    gpu_after_path = root / "gpu.after.csv"
    if gpu_after_path.is_file():
        for line in read_text(gpu_after_path).splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 2:
                try:
                    gpu_after.append(int(fields[1]))
                except ValueError:
                    pass
    checks = {
        "e2_c1_4_of_4_ok": (
            len([r for r in records if r.get("experiment") == "E2" and r.get("concurrency") == 1 and r.get("status") == "ok"]) == 4
        ),
        "e2_c16_4_of_4_ok": (
            len([r for r in records if r.get("experiment") == "E2" and r.get("concurrency") == 16 and r.get("status") == "ok"]) == 4
        ),
        "e3_10_of_10_ok": (
            len([r for r in records if r.get("experiment") == "E3" and r.get("status") == "ok"]) == 10
        ),
        "kernel_log_collected": bool(kernel_log),
        "no_kernel_fault": not FATAL.search(kernel_log),
        "section_returncodes_zero": (
            bool(execution_rows)
            and all(returncode == "0" for _, returncode in execution_rows)
        ),
        "gpu_zero_after": bool(gpu_after) and all(value == 0 for value in gpu_after),
    }
    report = {
        "schema": "eurosys_e2_e3_native_uva_cudagraph_verification_v1",
        "run_root": str(root),
        "status": "ok" if not errors and all(checks.values()) else "error",
        "checks": checks,
        "errors": errors,
        "records": records,
    }
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
