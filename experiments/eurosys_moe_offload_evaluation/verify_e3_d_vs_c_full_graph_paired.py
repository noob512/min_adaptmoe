#!/usr/bin/env python3
"""Verify and summarize paired E3 D versus C-full-Graph (arm F) runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED = {
    ("medium_context", 8): (2048, 128, 8606711808),
    ("rag_summary", 1): (8192, 128, 8606711808),
    ("rag_summary", 8): (8192, 128, 14269022208),
}
FAULT_MARKERS = (
    "cuda illegal",
    "nvrm: xid",
    "fault_pde",
    "fault_pte",
    "kernel panic",
    "traceback (most recent call last)",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_hash(result: dict[str, Any]) -> str:
    payload = [item["output_token_ids"] for item in result["outputs"]]
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def graph_replayed(result: dict[str, Any], *, require_prefill: bool) -> bool:
    steps = result.get("steps", {})
    measured = steps.get("measured_decode", [])
    prefill_ok = (
        steps.get("prefill", {}).get("graph_wrapper_delta", {}).get("full_replay") == 1
        if require_prefill
        else steps.get("prefill", {}).get("graph_wrapper_delta", {}).get("full_replay", 0) == 0
    )
    return bool(
        prefill_ok
        and steps.get("first_decode", {}).get("graph_wrapper_delta", {}).get("full_replay") == 1
        and measured
        and all(
            step.get("graph_wrapper_delta", {}).get("full_replay") == 1
            for step in measured
        )
    )


def pure_phase(result: dict[str, Any], batch: int, prompt: int) -> bool:
    steps = result.get("steps", {})
    prefill = steps.get("prefill", {}).get("schedule", {})
    first = steps.get("first_decode", {}).get("schedule", {})
    measured = steps.get("measured_decode", [])
    return bool(
        len(prefill.get("new_request_ids", [])) == batch
        and not prefill.get("cached_request_ids", [])
        and int(prefill.get("total_num_scheduled_tokens", -1)) == prompt * batch
        and not first.get("new_request_ids", [])
        and len(first.get("cached_request_ids", [])) == batch
        and int(first.get("total_num_scheduled_tokens", -1)) == batch
        and all(not step.get("schedule", {}).get("new_request_ids", []) for step in measured)
    )


def no_recapture(result: dict[str, Any], expected_captures: int) -> bool:
    before = result.get("compilation_counter_after_load", {})
    after = result.get("compilation_counter_final", {})
    return bool(
        int(before.get("num_cudagraph_captured", -1)) == expected_captures
        and int(after.get("num_cudagraph_captured", -2)) == expected_captures
        and int(before.get("num_gpu_runner_capture_triggers", -1))
        == int(after.get("num_gpu_runner_capture_triggers", -2))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--control-arm", choices=("C", "F"), default="F")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root / "summary").resolve()
    output.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    expected_arms = {"D", args.control_arm}
    runs: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    custom_module_hashes: set[str] = set()

    for execution in sorted((root / "groups").glob("*/custom/execution.tsv")):
        group = execution.parent.parent
        custom_module = group / "provenance/module.custom.json"
        stock_module = group / "provenance/module.after.json"
        if not stock_module.is_file():
            stock_module = group / "provenance/module.stock_after_custom_error.json"
        if custom_module.is_file():
            custom_module_hashes.add(str(load_json(custom_module).get("module_sha256")))
        else:
            errors.append(f"missing custom module snapshot: {group}")
        if not stock_module.is_file():
            errors.append(f"missing final module snapshot: {group}")
        else:
            stock = load_json(stock_module)
            if stock.get("gpu_clean") is not True or int(stock.get("memory_used_mib", -1)) != 0:
                errors.append(f"unclean final GPU state: {group}")
            if "/usr/lib/modules/" not in str(stock.get("module_path", "")):
                errors.append(f"stock UVM was not restored: {group}")
        with execution.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        for row in rows:
            cell = (row["workload"], int(row["batch"]))
            arm = row["arm"]
            if cell not in EXPECTED or arm not in expected_arms:
                errors.append(f"unexpected run: {cell}/{arm}")
                continue
            if arm in runs.setdefault(cell, {}):
                errors.append(f"duplicate run: {cell}/{arm}")
                continue
            case = group / "custom" / row["case_dir"]
            if row["status"] != "ok" or row["returncode"] != "0":
                errors.append(f"failed run: {cell}/{arm}: status={row['status']} rc={row['returncode']}")
                continue
            try:
                result = load_json(case / "result.json")
            except Exception as exc:
                errors.append(f"unreadable result: {cell}/{arm}: {exc}")
                continue

            prompt, output_tokens, selected = EXPECTED[cell]
            config = result.get("config", {})
            timing = result.get("timing", {})
            environment = result.get("environment", {})
            outputs = result.get("outputs", [])
            label = f"{cell}/{arm}"
            if result.get("status") != "ok":
                errors.append(f"result status is not ok: {label}")
            prefill_graph = arm in {"D", "F"}
            expected_execution_mode = (
                "full_graph_separate" if prefill_graph else "full_graph_decode"
            )
            if not (
                config.get("execution_mode") == expected_execution_mode
                and config.get("graph_prefill") is prefill_graph
                and config.get("graph_decode") is True
                and config.get("batch_size") == cell[1]
                and config.get("expected_selected_bytes") == selected
                and config.get("kv_cache_memory_bytes") == int(row["kv_bytes"])
                and environment.get("VLLM_ATTENTION_BACKEND") == "TRITON_ATTN"
            ):
                errors.append(f"configuration mismatch: {label}")
            expected_phase = "prefill_graph_remote_decode" if arm == "D" else "remote_only"
            if config.get("phase_mode") != expected_phase:
                errors.append(f"phase mode mismatch: {label}")
            if not graph_replayed(result, require_prefill=prefill_graph):
                errors.append(f"Prefill/Decode Graph replay contract failed: {label}")
            if not pure_phase(result, cell[1], prompt):
                errors.append(f"pure-phase schedule failed: {label}")
            if not no_recapture(result, 2 if prefill_graph else 1):
                errors.append(f"Graph recapture/capture-count mismatch: {label}")
            if len(outputs) != cell[1] or not all(
                len(item.get("output_token_ids", [])) == output_tokens for item in outputs
            ):
                errors.append(f"output cardinality/length mismatch: {label}")
            sessions = (
                result.get("shutdown_contract", {})
                .get("diagnostics_after_shutdown", {})
                .get("driver", {})
                .get("session_count")
            )
            if sessions != 0:
                errors.append(f"driver session leak: {label}: {sessions}")
            logs = "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in (case / "run.log", case / "uvm.log")
                if path.is_file()
            ).lower()
            found = [marker for marker in FAULT_MARKERS if marker in logs]
            if found:
                errors.append(f"fault marker in {label}: {found}")

            transitions = result.get("new_first_decode_transitions") or []
            cutover = result.get("cutover_transition")
            if arm == "D":
                body = cutover.get("transition", cutover) if isinstance(cutover, dict) else {}
                park = body.get("park", {}) if isinstance(body, dict) else {}
                remote = body.get("remote_map", {}) if isinstance(body, dict) else {}
                if not (
                    len(transitions) == 1
                    and body.get("ok") is True
                    and park.get("dual_view_full_coverage") is True
                    and int(park.get("dual_view_block_count", 0)) > 0
                    and body.get("remote_map_operation") == "prebuilt_full_dual_view"
                    and remote.get("reason") == "prebuilt_full_dual_view"
                    and int(remote.get("ioctl_calls", -1)) == 0
                ):
                    errors.append(f"D dual-view cutover contract failed: {label}")
            elif transitions or cutover is not None or float(timing.get("cutover_wall_ms", -1)) != 0.0:
                errors.append(f"F unexpectedly performed a phase cutover: {label}")

            runs[cell][arm] = {
                "result": result,
                "workload": cell[0],
                "batch": cell[1],
                "prompt_tokens": prompt,
                "output_tokens": output_tokens,
                "selected_bytes": selected,
                "kv_bytes": int(row["kv_bytes"]),
                "arm": arm,
                "prefill_ms": float(timing["prefill_step_ms"]),
                "cutover_ms": float(timing["cutover_wall_ms"]),
                "steady_decode_median_ms": float(timing["steady_decode_step_ms"]["median"]),
                "cycle_e2e_ms": float(timing["cycle_e2e_ms"]),
                "output_token_ids_sha256": output_hash(result),
            }

    if set(runs) != set(EXPECTED):
        errors.append(f"cell mismatch: expected={sorted(EXPECTED)} actual={sorted(runs)}")
    if len(custom_module_hashes) != 1:
        errors.append(f"custom module hash mismatch: {sorted(custom_module_hashes)}")

    comparisons: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = [
        {key: value for key, value in arm.items() if key != "result"}
        for arms in runs.values()
        for arm in arms.values()
    ]
    for cell in sorted(EXPECTED):
        arms = runs.get(cell, {})
        if set(arms) != expected_arms:
            errors.append(
                f"incomplete D/{args.control_arm} pair for {cell}: {sorted(arms)}"
            )
            continue
        d, control = arms["D"], arms[args.control_arm]
        for key in ("prompt_tokens", "output_tokens", "selected_bytes", "kv_bytes"):
            if d[key] != control[key]:
                errors.append(f"paired {key} mismatch for {cell}")
        hash_equal = (
            d["output_token_ids_sha256"] == control["output_token_ids_sha256"]
        )
        if not hash_equal:
            errors.append(f"D/{args.control_arm} output hash mismatch for {cell}")
        control_label = "C_full_graph" if args.control_arm == "F" else "C"
        comparisons.append({
            "workload": cell[0],
            "batch": cell[1],
            "control_arm": args.control_arm,
            "D_prefill_ms": d["prefill_ms"],
            f"{control_label}_prefill_ms": control["prefill_ms"],
            "prefill_speedup_D_over_C": control["prefill_ms"] / d["prefill_ms"],
            "D_cutover_ms": d["cutover_ms"],
            f"{control_label}_cutover_ms": control["cutover_ms"],
            "D_steady_decode_median_ms": d["steady_decode_median_ms"],
            f"{control_label}_steady_decode_median_ms": control["steady_decode_median_ms"],
            "steady_decode_delta_percent_D_vs_C":
                (d["steady_decode_median_ms"] / control["steady_decode_median_ms"] - 1.0) * 100.0,
            "D_cycle_e2e_ms": d["cycle_e2e_ms"],
            f"{control_label}_cycle_e2e_ms": control["cycle_e2e_ms"],
            "e2e_speedup_D_over_C": control["cycle_e2e_ms"] / d["cycle_e2e_ms"],
            "output_hash_equal": hash_equal,
        })

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fields = list(rows[0]) if rows else ["status"]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output / "metrics.csv", metrics)
    write_csv(output / "comparison.csv", comparisons)
    verification = {
        "schema": "eurosys_e3_d_vs_c_paired_verification_v2",
        "status": "ok" if not errors else "error",
        "qualification_only": True,
        "retries": 0,
        "expected_runs": 6,
        "successful_runs": len(metrics),
        "successful_pairs": len(comparisons),
        "custom_module_sha256": next(iter(custom_module_hashes), None),
        "control_arm": args.control_arm,
        "arm_alias": (
            {"F": "C-full-Graph (always remote)"}
            if args.control_arm == "F"
            else {"C": "eager remote Prefill + remote Decode Graph"}
        ),
        "errors": errors,
    }
    (output / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "comparison.json").write_text(
        json.dumps(comparisons, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
