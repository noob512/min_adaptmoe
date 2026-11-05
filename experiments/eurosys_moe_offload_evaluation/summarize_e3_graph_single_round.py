#!/usr/bin/env python3
"""Verify and summarize the direct-batch CUDA-Graph E3 single round."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


COMPARE_ARMS = {"G", "B", "C"}
COMPARE_CELLS = {
    ("short_dialogue", 1), ("short_dialogue", 8),
    ("ordinary_chat", 1), ("ordinary_chat", 8),
    ("medium_context", 1), ("medium_context", 8),
    ("long_generation", 1), ("long_generation", 8),
}
DUAL_GRAPH_ONLY_CELLS = {
    ("medium_context", 8),
    ("rag_summary", 1),
    ("rag_summary", 8),
}
COMPARE_CELLS -= DUAL_GRAPH_ONLY_CELLS
EXPECTED_ARM_SETS = {
    **{cell: COMPARE_ARMS for cell in COMPARE_CELLS},
    **{cell: {"D"} for cell in DUAL_GRAPH_ONLY_CELLS},
}
EXPECTED_CELLS = set(EXPECTED_ARM_SETS)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def token_hash(result: dict[str, Any]) -> str:
    payload = [item["output_token_ids"] for item in result["outputs"]]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def peak_hbm_mib(path: Path) -> float | None:
    if not path.is_file():
        return None
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            values.append(float(fields[1]))
        except ValueError:
            pass
    return max(values) if values else None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-current-tier-dual-view", action="store_true")
    parser.add_argument("roots", type=Path, nargs="+")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    metrics: list[dict[str, Any]] = []
    seen_labels: set[tuple[str, int, str]] = set()

    for root in args.roots:
        execution = root / "custom/execution.tsv"
        if not execution.is_file():
            errors.append(f"missing execution.tsv: {root}")
            continue
        with execution.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        plan_path = root / "provenance/expert_plan.json"
        if not plan_path.is_file():
            plan_path = root / "provenance/cold_decode_plan_2g.json"
        plan = load_json(plan_path)
        layer_counts = sorted({len(value) for value in plan["selected_by_layer"].values()})
        tier = str(plan.get("tier") or plan.get("metadata", {}).get("tier") or "unknown")
        for row in rows:
            label = (row["workload"], int(row["batch"]), row["arm"])
            if label in seen_labels:
                errors.append(f"duplicate run: {label}")
                continue
            seen_labels.add(label)
            case = root / "custom" / row["case_dir"]
            if row["status"] != "ok" or row["returncode"] != "0":
                errors.append(f"run failed: {label} status={row['status']} rc={row['returncode']}")
                continue
            try:
                result = load_json(case / "result.json")
            except Exception as exc:  # retain all independent case failures
                errors.append(f"result unreadable: {label}: {exc}")
                continue
            arm = row["arm"]
            config = result.get("config", {})
            steps = result.get("steps", {})
            measured = steps.get("measured_decode", [])
            if arm == "D":
                graph_ok = (
                    result.get("status") == "ok"
                    and config.get("execution_mode") == "full_graph_separate"
                    and config.get("graph_decode") is True
                    and config.get("graph_prefill") is True
                    and steps.get("prefill", {}).get("graph_wrapper_delta", {}).get("full_replay") == 1
                    and steps.get("first_decode", {}).get("graph_wrapper_delta", {}).get("full_replay") == 1
                    and measured
                    and all(item.get("graph_wrapper_delta", {}).get("full_replay") == 1 for item in measured)
                )
            else:
                graph_ok = (
                    result.get("status") == "ok"
                    and config.get("execution_mode") == "full_graph_decode"
                    and config.get("graph_decode") is True
                    and config.get("graph_prefill") is False
                    and steps.get("first_decode", {}).get("graph_wrapper_delta", {}).get("full_replay") == 1
                    and measured
                    and all(item.get("graph_wrapper_delta", {}).get("full_replay") == 1 for item in measured)
                )
            if not graph_ok:
                errors.append(f"Graph replay contract failed: {label}")
            prefill_schedule = steps.get("prefill", {}).get("schedule", {})
            first_decode_schedule = steps.get("first_decode", {}).get("schedule", {})
            batch = int(row["batch"])
            prompt = int(row["prompt"])
            pure_phase_ok = (
                len(prefill_schedule.get("new_request_ids", [])) == batch
                and not prefill_schedule.get("cached_request_ids", [])
                and int(prefill_schedule.get("total_num_scheduled_tokens", -1)) == prompt * batch
                and not first_decode_schedule.get("new_request_ids", [])
                and len(first_decode_schedule.get("cached_request_ids", [])) == batch
                and int(first_decode_schedule.get("total_num_scheduled_tokens", -1)) == batch
                and all(not item.get("schedule", {}).get("new_request_ids", []) for item in measured)
            )
            if not pure_phase_ok:
                errors.append(f"direct-batch pure-phase contract failed: {label}")
            expected_output = int(row["output"])
            if len(result.get("outputs", [])) != int(row["batch"]) or not all(
                len(item.get("output_token_ids", [])) == expected_output
                for item in result.get("outputs", [])
            ):
                errors.append(f"output cardinality/length failed: {label}")
            shutdown_sessions = (
                result.get("shutdown_contract", {})
                .get("diagnostics_after_shutdown", {})
                .get("driver", {})
                .get("session_count")
            )
            if shutdown_sessions != 0:
                errors.append(f"driver session leak: {label}: {shutdown_sessions}")
            log_text = (case / "run.log").read_text(encoding="utf-8", errors="replace")
            fault_markers = [
                marker for marker in
                ("CUDA illegal", "NVRM: Xid", "fault_pde", "fault_pte", "Traceback (most recent call last)")
                if marker.lower() in log_text.lower()
            ]
            if fault_markers:
                errors.append(f"fault marker in {label}: {fault_markers}")
            timing = result["timing"]
            transition = result.get("cutover_transition", {})
            transition_body = transition.get("transition", transition) if isinstance(transition, dict) else {}
            hot_audit = transition_body.get("hot_physical_audit_enabled")
            if arm in {"G", "D"} and hot_audit is not False:
                errors.append(f"hot-residency audit was not disabled: {label}: {hot_audit}")
            park = transition_body.get("park") if isinstance(transition_body, dict) else None
            remote_map = transition_body.get("remote_map") if isinstance(transition_body, dict) else None
            dual_view_ok = bool(
                isinstance(park, dict)
                and park.get("ok") is True
                and park.get("dual_view_full_coverage") is True
                and int(park.get("dual_view_block_count", 0)) > 0
                and transition_body.get("alias_detach_operation") == "park"
                and transition_body.get("remote_map_operation")
                == "prebuilt_full_dual_view"
                and isinstance(remote_map, dict)
                and remote_map.get("reason") == "prebuilt_full_dual_view"
                and int(remote_map.get("ioctl_calls", -1)) == 0
            )
            if (
                args.require_current_tier_dual_view
                and arm in {"G", "D"}
                and not dual_view_ok
            ):
                errors.append(f"current-tier dual-view contract failed: {label}")
            model_diag = result["diagnostics"]["after_load"]["model"]
            metrics.append({
                "source_root": str(root.resolve()),
                "workload": row["workload"],
                "prompt_tokens": int(row["prompt"]),
                "output_tokens": expected_output,
                "batch": int(row["batch"]),
                "arm": arm,
                "tier": tier,
                "cold_experts_per_layer": "/".join(map(str, layer_counts)),
                "selected_bytes": int(model_diag["selected_parameter_bytes_per_forward"]),
                "kv_bytes": int(row["kv_bytes"]),
                "prefill_ms": float(timing["prefill_step_ms"]),
                "cutover_ms": float(timing["cutover_wall_ms"]),
                "first_decode_excluding_cutover_ms": float(timing["first_decode_excluding_cutover_ms"]),
                "steady_decode_mean_ms": float(timing["steady_decode_step_ms"]["mean"]),
                "steady_decode_median_ms": float(timing["steady_decode_step_ms"]["median"]),
                "cycle_e2e_ms": float(timing["cycle_e2e_ms"]),
                "peak_hbm_mib": peak_hbm_mib(case / "nvml.csv"),
                "output_token_ids_sha256": token_hash(result),
                "graph_decode": True,
                "graph_prefill": arm == "D",
                "qualification_kind": (
                    "dual_graph_large_prefill" if arm == "D"
                    else "decode_graph_three_arm_compare"
                ),
                "direct_batch_pure_phase": pure_phase_ok,
                "hot_residency_audit": hot_audit if arm in {"G", "D"} else False,
                "dual_view_full_coverage": dual_view_ok if arm in {"G", "D"} else False,
                "dual_view_block_count": (
                    int(park.get("dual_view_block_count", 0))
                    if isinstance(park, dict)
                    else 0
                ),
                "remote_range_map_ioctl_calls": (
                    int(remote_map.get("ioctl_calls", -1))
                    if isinstance(remote_map, dict)
                    else -1
                ),
                "status": "ok" if graph_ok else "error",
            })

    by_cell: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for metric in metrics:
        by_cell[(metric["workload"], metric["batch"])][metric["arm"]] = metric
    actual_cells = set(by_cell)
    if actual_cells != EXPECTED_CELLS:
        errors.append(f"cell set mismatch: missing={sorted(EXPECTED_CELLS-actual_cells)} extra={sorted(actual_cells-EXPECTED_CELLS)}")

    winners: list[dict[str, Any]] = []
    for cell in sorted(EXPECTED_CELLS):
        arms = by_cell.get(cell, {})
        expected_arms = EXPECTED_ARM_SETS[cell]
        if set(arms) != expected_arms:
            errors.append(f"incomplete arm set for {cell}: {sorted(arms)}")
            continue
        if cell in DUAL_GRAPH_ONLY_CELLS:
            continue
        hashes = {item["output_token_ids_sha256"] for item in arms.values()}
        if len(hashes) != 1:
            errors.append(f"G/B/C output hash mismatch for {cell}")
        selected = {item["selected_bytes"] for item in arms.values()}
        if len(selected) != 1:
            errors.append(f"G/B/C selected-byte mismatch for {cell}")
        best = min(COMPARE_ARMS, key=lambda name: arms[name]["cycle_e2e_ms"])
        winners.append({
            "workload": cell[0], "batch": cell[1], "winner": best,
            "G_cycle_e2e_ms": arms["G"]["cycle_e2e_ms"],
            "B_cycle_e2e_ms": arms["B"]["cycle_e2e_ms"],
            "C_cycle_e2e_ms": arms["C"]["cycle_e2e_ms"],
            "selected_bytes": next(iter(selected)),
            "output_hash_equal": len(hashes) == 1,
        })

    metric_fields = list(metrics[0]) if metrics else ["status"]
    winner_fields = list(winners[0]) if winners else ["status"]
    write_csv(args.output_dir / "metrics.csv", metrics, metric_fields)
    write_csv(args.output_dir / "winners.csv", winners, winner_fields)
    verification = {
        "schema": "eurosys_e3_graph_single_round_verification_v1",
        "status": "ok" if not errors else "error",
        "qualification_only": True,
        "expected_cells": len(EXPECTED_CELLS),
        "successful_cells": len(by_cell),
        "successful_compare_cells": len(winners),
        "successful_dual_graph_only_cells": sum(
            cell in by_cell and set(by_cell[cell]) == {"D"}
            for cell in DUAL_GRAPH_ONLY_CELLS
        ),
        "expected_runs": sum(len(arms) for arms in EXPECTED_ARM_SETS.values()),
        "successful_runs": len(metrics),
        "compare_cells": len(COMPARE_CELLS),
        "dual_graph_only_cells": len(DUAL_GRAPH_ONLY_CELLS),
        "current_tier_dual_view_required": bool(
            args.require_current_tier_dual_view
        ),
        "winner_counts": {arm: sum(row["winner"] == arm for row in winners) for arm in sorted(COMPARE_ARMS)},
        "errors": errors,
        "source_roots": [str(root.resolve()) for root in args.roots],
    }
    (args.output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
