#!/usr/bin/env python3
"""Summarize the C1 cross-mechanism runtime-hot-audit-off matrix."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


ARMS = (
    "native_all_gpu",
    "native_pinned_uva",
    "native_copy_on_forward",
    "ours",
    "ktransformers",
    "moe_infinity",
)
WORKLOADS = (
    "short_dialogue",
    "ordinary_chat",
    "rag_summary",
    "long_generation",
)
FATAL = re.compile(
    r"CUDA error|illegal memory access|out of memory|\bXid\b|PTE.*fault|"
    r"PDE.*fault|segmentation fault|Traceback",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def peak_hbm_mib(path: Path) -> int | None:
    if not path.is_file():
        return None
    peak: int | None = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for row in csv.reader(handle):
            if len(row) < 5:
                continue
            try:
                value = int(float(row[4].strip()))
            except ValueError:
                continue
            peak = value if peak is None else max(peak, value)
    return peak


def traceback_ends_in_keyboard_interrupt(lines: list[str], index: int) -> bool:
    saw_keyboard_interrupt = False
    for candidate in lines[index + 1 : index + 128]:
        if candidate.strip() == "KeyboardInterrupt":
            saw_keyboard_interrupt = True
        # SGLang's scheduler and detokenizer tracebacks can interleave, so a
        # second "Traceback" is not a reliable boundary.  The harness's
        # timestamped shutdown line is the stable delimiter.
        if "INFO:" in candidate and "Shutting down" in candidate:
            return saw_keyboard_interrupt
    return False


def first_fatal(
    path: Path, *, ignore_keyboard_interrupt_traceback: bool = False
) -> str | None:
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        if FATAL.search(line):
            if (
                ignore_keyboard_interrupt_traceback
                and line.startswith("Traceback")
                and traceback_ends_in_keyboard_interrupt(lines, index)
            ):
                continue
            return line[-2000:]
    return None


def discover_client(case_root: Path) -> Path | None:
    paths = sorted(case_root.glob("*_c1/client.json"))
    return paths[0] if len(paths) == 1 else None


def parse_attempt(row: dict[str, str]) -> dict[str, Any]:
    case_root = Path(row["case_root"])
    returncode = int(row["returncode"])
    verification_path = case_root / "verification.json"
    verification = load_json(verification_path) if verification_path.is_file() else {}
    client_path = discover_client(case_root)
    client = load_json(client_path) if client_path is not None else {}
    phases = client.get("phases")
    phase = phases[0] if isinstance(phases, list) and len(phases) == 1 else {}
    requests = phase.get("requests") if isinstance(phase, dict) else None
    request = requests[0] if isinstance(requests, list) and len(requests) == 1 else {}
    success = (
        returncode == 0
        and verification.get("status") == "ok"
        and client.get("status") == "ok"
        and isinstance(phase, dict)
        and phase.get("failed_requests") == 0
        and phase.get("request_count") == 1
    )
    transition_path = case_root / "phase_transition.json"
    transition_doc = load_json(transition_path) if transition_path.is_file() else {}
    transition = transition_doc.get("transition")
    if not isinstance(transition, dict):
        transition = {}
    server_log = case_root / "server.log"
    ttft_ms = (
        phase.get("ttft_ms", {}).get("mean") if isinstance(phase, dict) else None
    )
    tpot_ms = (
        phase.get("itl_ms", {}).get("mean") if isinstance(phase, dict) else None
    )
    return {
        "created_at": row["created_at"],
        "arm": row["arm"],
        "workload": row["workload"],
        "setting": row["setting"],
        "case_root": str(case_root),
        "returncode": returncode,
        "success": success,
        "verification_status": verification.get("status"),
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "slo_pass": (
            ttft_ms is not None
            and tpot_ms is not None
            and ttft_ms <= 2000.0
            and tpot_ms <= 100.0
        ),
        "e2e_ms": phase.get("e2e_ms", {}).get("mean")
        if isinstance(phase, dict)
        else None,
        "output_token_ids_sha256": request.get("output_token_ids_sha256")
        if isinstance(request, dict)
        else None,
        "prompt_token_ids_sha256": request.get("prompt_token_ids_sha256")
        if isinstance(request, dict)
        else None,
        "peak_hbm_mib": peak_hbm_mib(case_root / "nvml.csv"),
        "phase_transition_ms": (
            float(transition["wall_ns"]) / 1_000_000
            if isinstance(transition.get("wall_ns"), int)
            else None
        ),
        "hot_physical_audit_enabled": transition.get(
            "hot_physical_audit_enabled"
        ),
        "cold_physical_residency": transition.get("physical_residency"),
        # SGLang workers receive SIGINT during the harness's normal shutdown
        # and print KeyboardInterrupt tracebacks even after a successful
        # request.  Ignore those only for an otherwise successful KT case;
        # retain every traceback for failed attempts and every other arm.
        "fatal_log_excerpt": first_fatal(
            server_log,
            ignore_keyboard_interrupt_traceback=(
                row["arm"] == "ktransformers" and success
            ),
        ),
    }


def write_json(path: Path, value: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    execution = args.run_root / "execution.tsv"
    with execution.open(encoding="utf-8", newline="") as handle:
        attempts = [parse_attempt(row) for row in csv.DictReader(handle, delimiter="\t")]

    final: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for arm in ARMS:
        for workload in WORKLOADS:
            successes = [
                item
                for item in attempts
                if item["arm"] == arm
                and item["workload"] == workload
                and item["success"]
            ]
            if successes:
                if arm == "moe_infinity":
                    # The MoE-Infinity search may first find a feasible 0.85
                    # point and append a later 0.86--0.90 upper-bound probe.
                    # Select the successful point with the largest device
                    # cache ratio, i.e. the least offload for this workload.
                    final.append(
                        max(
                            successes,
                            key=lambda item: float(
                                re.search(
                                    r"device_ratio=([0-9.]+)", item["setting"]
                                ).group(1)
                            ),
                        )
                    )
                elif arm in {"ours", "ktransformers"}:
                    # A resumed precision-reservation search may append a new
                    # successful shape.  Keep the point with the fewest cold
                    # experts, which is the objective of this pilot.
                    final.append(
                        min(
                            successes,
                            key=lambda item: int(
                                re.search(
                                    r"cold_per_layer=([0-9]+)", item["setting"]
                                ).group(1)
                            ),
                        )
                    )
                else:
                    final.append(successes[0])
            else:
                missing.append({"arm": arm, "workload": workload})

    hashes: dict[str, dict[str, str | None]] = {}
    for workload in WORKLOADS:
        hashes[workload] = {
            item["arm"]: item["output_token_ids_sha256"]
            for item in final
            if item["workload"] == workload
        }
    exact_generation: dict[str, dict[str, bool | None]] = {}
    for workload, values in hashes.items():
        baseline = values.get("native_all_gpu")
        exact_generation[workload] = {
            arm: (value == baseline if baseline is not None and value is not None else None)
            for arm, value in values.items()
        }

    result = {
        "schema": "c1_mechanism_runtime_hot_audit_off_summary_v1",
        "status": "ok" if not missing else "incomplete",
        "expected_final_cells": len(ARMS) * len(WORKLOADS),
        "successful_final_cells": len(final),
        "slo_passed_final_cells": sum(bool(item["slo_pass"]) for item in final),
        "slo_thresholds_ms": {"ttft": 2000.0, "tpot": 100.0},
        "attempt_count": len(attempts),
        "runtime_hot_audit_policy": {
            "ours_hot_cuda_read_gate": False,
            "ours_hot_physical_residency_audit": False,
            "cold_remote_residency_audit_retained": True,
        },
        "final_results": final,
        "failed_attempts": [item for item in attempts if not item["success"]],
        "missing": missing,
        "output_hashes": hashes,
        "exact_generation_vs_native_all_gpu": exact_generation,
    }
    write_json(args.run_root / "summary.json", result, args.overwrite)

    metrics_path = args.run_root / "metrics.csv"
    if metrics_path.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {metrics_path}")
    columns = (
        "arm",
        "workload",
        "setting",
        "ttft_ms",
        "tpot_ms",
        "slo_pass",
        "e2e_ms",
        "peak_hbm_mib",
        "phase_transition_ms",
        "hot_physical_audit_enabled",
        "output_token_ids_sha256",
    )
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in final:
            writer.writerow({key: item.get(key) for key in columns})
    print(json.dumps({"status": result["status"], "final": len(final), "attempts": len(attempts)}))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
