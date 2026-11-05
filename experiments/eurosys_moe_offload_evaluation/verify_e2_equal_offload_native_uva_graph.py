#!/usr/bin/env python3
"""Verify and summarize the one-round E2 equal-requested-offload experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKLOADS = (
    "short_dialogue",
    "ordinary_chat",
    "rag_summary",
    "long_generation",
)
ARMS = ("ours", "native_pinned_uva")
FATAL = re.compile(
    r"Xid|MMU fault|PDE fault|PTE fault|illegal memory access|"
    r"CUDA error|Segmentation fault|Traceback \(most recent call last\)",
    re.IGNORECASE,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def one_case(root: Path, concurrency: int, arm: str, workload: str) -> Path:
    candidates = sorted(
        path.parent
        for path in (root / f"c{concurrency}" / "cases" / arm / workload).glob(
            "*/verification.json"
        )
        if load_json(path).get("status") == "ok"
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected one successful c{concurrency}/{arm}/{workload} case, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def manifest_stats(path: Path) -> dict[str, Any]:
    events = [json.loads(line) for line in path.read_text().splitlines() if line]
    config = next(event for event in events if event.get("event") == "cpu_offload_config")
    parameters = [
        event for event in events if event.get("event") == "cpu_offload_parameter"
    ]
    if not parameters:
        raise ValueError(f"no offloaded parameters in {path}")
    realized = int(parameters[-1]["cumulative_offload_bytes"])
    previous = int(parameters[-2]["cumulative_offload_bytes"]) if len(parameters) > 1 else 0
    return {
        "requested_bytes": int(config["max_offload_bytes"]),
        "realized_bytes": realized,
        "previous_prefix_bytes": previous,
        "parameter_count": len(parameters),
        "expert_slice_count": int(config.get("graph_static_alias_selected_expert_count", 0)),
        "parameter_names": [event["full_parameter_name"] for event in parameters],
    }


def c1_measurement(case: Path, workload: str) -> tuple[dict[str, float], list[list[int]]]:
    client = load_json(case / f"{workload}_c1" / "client.json")
    phase = client["phases"][0]
    metrics = {
        "ttft_ms": float(phase["ttft_ms"]["mean"]),
        "tpot_ms": float(phase["tpot_ms"]["mean"]),
        "e2e_ms": float(phase["e2e_ms"]["mean"]),
        "throughput_requests_per_second": float(phase["goodput_requests_per_second"]),
    }
    outputs = [request["output_token_ids"] for request in phase["requests"]]
    return metrics, outputs


def c16_measurement(case: Path) -> tuple[dict[str, float], list[list[int]]]:
    measurement = load_json(case / "result.json")["measurement"]
    metrics = {
        "ttft_ms": float(measurement["ttft_ms"]["mean"]),
        "tpot_ms": float(measurement["tpot_ms"]["mean"]),
        "e2e_ms": float(measurement["e2e_ms"]["mean"]),
        "throughput_requests_per_second": (
            float(measurement["successful_requests"])
            / float(measurement["elapsed_seconds"])
        ),
    }
    outputs = [request["output_token_ids"] for request in measurement["requests"]]
    return metrics, outputs


def output_hash(outputs: list[list[int]]) -> str:
    payload = json.dumps(outputs, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def graph_checks(
    case: Path,
    arm: str,
    concurrency: int,
    native_graph_mode: str,
) -> dict[str, bool]:
    log_name = "server.log" if concurrency == 1 else "run.log"
    log = (case / log_name).read_text(errors="replace")
    decode_audits = re.findall(
        r"CUDA_GRAPH_RUNTIME_AUDIT[^\n]*decode_step[^\n]*runtime_mode=([A-Z_]+)",
        log,
    )
    prefill_audits = re.findall(
        r"CUDA_GRAPH_RUNTIME_AUDIT[^\n]*prefill_step[^\n]*runtime_mode=([A-Z_]+)",
        log,
    )
    full_decode = bool(decode_audits) and all(
        mode == "FULL" for mode in decode_audits
    )
    if arm == "ours":
        prefill_ok = bool(prefill_audits) and all(
            mode == "FULL" for mode in prefill_audits
        )
        transition_ok = any(
            json.loads(line).get("event") == "cpu_offload_uvm_original_va_remote_transition"
            and json.loads(line).get("ok") is True
            and json.loads(line).get("remote_map", {}).get("reason")
                == "prebuilt_full_dual_view"
            for line in (case / "offload_manifest.jsonl").read_text().splitlines()
            if line
        )
    else:
        expected = "FULL" if native_graph_mode == "FULL_SEPARATE" else "NONE"
        prefill_ok = bool(prefill_audits) and all(
            mode == expected for mode in prefill_audits
        )
        transition_ok = True
    return {
        "decode_full_graph_replayed": full_decode,
        "prefill_protocol_observed": prefill_ok,
        "dual_view_transition_ok": transition_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--mode",
        choices=("native_requested_matches_ours", "realized_within_10_percent"),
        default="native_requested_matches_ours",
    )
    parser.add_argument(
        "--native-graph-mode",
        choices=("FULL_DECODE_ONLY", "FULL_SEPARATE"),
        default="FULL_DECODE_ONLY",
    )
    args = parser.parse_args()
    root = args.run_root.resolve()
    checks: dict[str, bool] = {}
    cells: list[dict[str, Any]] = []

    execution = {
        fields[0]: int(fields[1])
        for line in (root / "execution.tsv").read_text().splitlines()[1:]
        if len(fields := line.split("\t")) == 2 and fields[0] != "verification"
    }
    checks["launchers_succeeded"] = execution == {"c1": 0, "c16": 0}

    for concurrency in (1, 16):
        for workload in WORKLOADS:
            arm_data: dict[str, Any] = {}
            arm_outputs: dict[str, list[list[int]]] = {}
            for arm in ARMS:
                case = one_case(root, concurrency, arm, workload)
                verification = load_json(case / "verification.json")
                manifest = manifest_stats(case / "offload_manifest.jsonl")
                metrics, outputs = (
                    c1_measurement(case, workload)
                    if concurrency == 1
                    else c16_measurement(case)
                )
                graphs = graph_checks(
                    case, arm, concurrency, args.native_graph_mode
                )
                prefix = f"c{concurrency}_{workload}_{arm}"
                checks[f"{prefix}_case_ok"] = verification.get("status") == "ok"
                checks.update({f"{prefix}_{key}": value for key, value in graphs.items()})
                arm_data[arm] = {
                    "case_root": str(case),
                    "manifest": manifest,
                    "metrics": metrics,
                    "output_hash": output_hash(outputs),
                }
                arm_outputs[arm] = outputs

            ours = arm_data["ours"]
            native = arm_data["native_pinned_uva"]
            ours_bytes = ours["manifest"]["realized_bytes"]
            native_manifest = native["manifest"]
            requested_match = native_manifest["requested_bytes"] == ours_bytes
            native_covers_target = (
                native_manifest["previous_prefix_bytes"]
                < native_manifest["requested_bytes"]
                <= native_manifest["realized_bytes"]
            )
            outputs_match = arm_outputs["ours"] == arm_outputs["native_pinned_uva"]
            cell_prefix = f"c{concurrency}_{workload}"
            checks[f"{cell_prefix}_native_is_minimum_whole_parameter_prefix"] = native_covers_target
            native_bytes = native_manifest["realized_bytes"]
            realized_ratio = max(native_bytes, ours_bytes) / min(native_bytes, ours_bytes)
            if args.mode == "native_requested_matches_ours":
                checks[f"{cell_prefix}_native_requested_bytes_match_ours"] = requested_match
                match_definition = (
                    "same requested bytes; Native uses the minimum whole-Parameter "
                    "prefix whose realized bytes are not below Ours"
                )
            else:
                checks[f"{cell_prefix}_realized_bytes_within_10_percent"] = realized_ratio <= 1.10
                match_definition = (
                    "Ours uses one more cold expert per layer; realized byte counts "
                    "must differ by no more than 10 percent"
                )
            cells.append({
                "concurrency": concurrency,
                "workload": workload,
                "ours": ours,
                "native": native,
                "offload": {
                    "target_bytes": ours_bytes,
                    "native_realized_bytes": native_bytes,
                    "native_realized_over_target_ratio": native_bytes / ours_bytes,
                    "symmetric_realized_ratio": realized_ratio,
                    "native_realized_overshoot_bytes": native_bytes - ours_bytes,
                    "match_definition": match_definition,
                },
                "outputs_match": outputs_match,
                "native_over_ours": {
                    key: native["metrics"][key] / ours["metrics"][key]
                    for key in ("ttft_ms", "tpot_ms", "e2e_ms")
                },
            })

    kernel_log = (root / "kernel.since.log").read_text(errors="replace")
    checks["no_kernel_or_cuda_fault"] = not FATAL.search(kernel_log)
    gpu_rows = [line.split(",") for line in (root / "gpu.after.csv").read_text().splitlines()]
    checks["gpu_memory_zero_after"] = bool(gpu_rows) and all(
        int(row[1].strip()) == 0 for row in gpu_rows
    )
    status = "ok" if all(checks.values()) else "error"
    result = {
        "schema": "eurosys_e2_equal_offload_native_uva_graph_verification_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "protocol": {
            "repetitions": 1,
            "native_selection": "registration-order whole Parameters",
            "ours_selection": "cross-layer cold expert slices",
            "byte_matching": (
                "Native requested budget equals Ours realized bytes; because Native cannot "
                "split a Parameter, its realized prefix is the minimum prefix >= target"
                if args.mode == "native_requested_matches_ours"
                else "Ours adds one expert per layer and both realized byte counts are within 10 percent"
            ),
            "matching_mode": args.mode,
            "native_graph": (
                "FULL_SEPARATE Prefill/Decode with graph-safe Prefill chunks "
                "defined by the frozen per-cell profiles"
                if args.native_graph_mode == "FULL_SEPARATE"
                else "eager Prefill plus FULL_DECODE_ONLY"
            ),
            "ours_graph": "FULL_SEPARATE Prefill/Decode plus dual-page-table P-to-D",
        },
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "cells": cells,
    }
    atomic_json(root / "verification.json", result)
    atomic_json(root / "summary.json", {
        key: result[key] for key in ("schema", "created_at", "status", "protocol", "cells")
    })
    print(json.dumps({"status": status, "failed_checks": result["failed_checks"]}, indent=2))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
