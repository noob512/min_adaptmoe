#!/usr/bin/env python3
"""Verify the short BHHB no-offload CPU DMA-latency experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_ORDER = "BHHB"
EXPECTED_GAPS = [110, 500, 500, 110]


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL: {path}")
    values: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        require(isinstance(value, dict), f"{path}:{line_number}: not an object")
        values.append(value)
    return values


def load_env(path: Path) -> dict[str, str]:
    require(path.is_file(), f"missing environment: {path}")
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        require("=" in raw, f"{path}:{line_number}: malformed entry")
        key, value = raw.split("=", 1)
        require(key and key not in result, f"{path}:{line_number}: duplicate key")
        result[key] = value
    return result


def finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label}: not finite",
    )
    return float(value)


def summarize(values: list[float]) -> dict[str, Any]:
    require(bool(values), "cannot summarize empty values")
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "samples": values,
    }


def expected_calls() -> list[tuple[str, str, str, int, str]]:
    calls: list[tuple[str, str, str, int, str]] = []
    calls.extend(("auto_contiguous", "measure", "auto", 0, "sleep") for _ in range(2))
    for gap in EXPECTED_GAPS:
        calls.append(("auto_gap_sweep", "condition", "auto", 0, "sleep"))
        calls.append(("auto_gap_sweep", "measure", "auto", gap, "sleep"))
    for gap in EXPECTED_GAPS:
        calls.append(("auto_busy_gap_sweep", "condition", "auto", 0, "sleep"))
        calls.append(("auto_busy_gap_sweep", "measure", "auto", gap, "busy"))
    calls.extend(
        [
            ("locked_gap_sweep", "condition", "locked", 0, "sleep"),
            ("locked_gap_sweep", "measure", "locked", 0, "sleep"),
            ("auto_recovery", "condition", "auto_recovery", 0, "sleep"),
            ("auto_recovery", "measure", "auto_recovery", 0, "sleep"),
        ]
    )
    return calls


def call_contract(call: dict[str, Any]) -> tuple[str, str, str, int, str]:
    return (
        str(call.get("phase")),
        str(call.get("role")),
        str(call.get("clock_mode")),
        int(call.get("requested_gap_ms", -1)),
        str(call.get("idle_mode")),
    )


def verify_result(case_dir: Path, mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = load_json(case_dir / "result.json")
    require(result.get("schema") == "vllm_prefill_p3_clock_diagnosis_v1", "result schema")
    require(result.get("status") == "ok", "result status")
    require(result.get("mechanism_class") == "vllm_no_offload_control", "mechanism")
    expected_label = f"cpu_dma_latency_no_offload_{mode}"
    require(result.get("profile_label") == expected_label, "profile label")
    config = result.get("config")
    require(isinstance(config, dict), "result config")
    expected_config = {
        "concurrency": 4,
        "output_tokens": 1,
        "warmup_requests": 4,
        "continuous_batches": 2,
        "auto_gaps_ms": EXPECTED_GAPS,
        "busy_gaps_ms": EXPECTED_GAPS,
        "locked_gaps_ms": [0],
        "recovery_gaps_ms": [0],
        "cpu_offload_gb": 0.0,
        "disable_chunked_prefill": True,
    }
    for key, expected in expected_config.items():
        require(config.get(key) == expected, f"config {key}: {config.get(key)!r}")
    calls = result.get("calls")
    require(isinstance(calls, list), "calls absent")
    require([call_contract(call) for call in calls] == expected_calls(), "call schedule")
    require([call.get("call_index") for call in calls] == list(range(22)), "call indices")
    hashes: set[str] = set()
    for index, call in enumerate(calls):
        start = finite(call.get("start_epoch"), f"call {index} start")
        end = finite(call.get("end_epoch"), f"call {index} end")
        require(end >= start, f"call {index}: reversed window")
        require(finite(call.get("wall_ms"), f"call {index} wall") > 0, "nonpositive wall")
        idle = finite(call.get("actual_idle_ms"), f"call {index} idle")
        if call.get("role") == "measure":
            require(idle + 0.1 >= int(call["requested_gap_ms"]), f"call {index}: short gap")
        before = call.get("clock_before")
        after = call.get("clock_after")
        require(isinstance(before, dict) and isinstance(after, dict), "clock samples")
        for sample in (before, after):
            for key in ("pstate", "sm_mhz", "memory_mhz", "power_watts"):
                finite(sample.get(key), f"call {index} clock {key}")
        token_hash = call.get("output_token_ids_sha256")
        require(isinstance(token_hash, str) and len(token_hash) == 64, "token hash")
        hashes.add(token_hash)
    require(len(hashes) == 1, "outputs changed within process")
    require(result.get("output_token_ids_sha256") in hashes, "aggregate token hash")
    require(isinstance(result.get("clock_events"), list), "clock events absent")
    require(not any("error" in event for event in result["clock_events"]), "clock error")
    return result, calls


def verify_no_offload(case_dir: Path) -> None:
    environment = load_env(case_dir / "environment.txt")
    expected = {
        "VLLM_USE_UVM": "0",
        "VLLM_UVM_EXPERT_WEIGHTS_MANAGED": "0",
        "VLLM_UVM_EXPERT_ENVELOPE_ENABLE": "0",
        "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE": "0",
        "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE": "restore",
        "VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD": "0",
        "VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE": "",
    }
    for key, value in expected.items():
        require(environment.get(key) == value, f"no-offload environment {key}")
    manifest = load_jsonl(case_dir / "cpu_offload_manifest.jsonl")
    require(
        not any(record.get("event") == "cpu_offload_parameter" for record in manifest),
        "CPU-offloaded parameter present",
    )
    trace = load_jsonl(case_dir / "slot_trace.jsonl")
    forbidden = {"restore_layer", "register_layer_plans", "preallocate_second_bank"}
    require(not any(record.get("operation") in forbidden for record in trace), "P3 trace present")


def verify_guard(case_dir: Path, held: bool, result: dict[str, Any]) -> None:
    ready_path = case_dir / "control" / "guard.ready.json"
    events_path = case_dir / "control" / "guard.events.jsonl"
    if not held:
        require(not ready_path.exists() and not events_path.exists(), "baseline has guard evidence")
        return
    ready = load_json(ready_path)
    events = load_jsonl(events_path)
    require(ready.get("status") == "acquired", "guard not acquired")
    require(ready.get("latency_us") == 0, "guard latency")
    require(ready.get("character_device") is True, "guard device type")
    require(len(events) == 2, "guard event count")
    require([event.get("status") for event in events] == ["acquired", "released"], "guard lifecycle")
    require(events[0].get("pid") == events[1].get("pid") == ready.get("pid"), "guard PID")
    acquired = finite(ready.get("acquired_epoch"), "guard acquired")
    released = finite(events[1].get("released_epoch"), "guard released")
    require(acquired <= finite(result.get("started_epoch"), "result start"), "guard started late")
    require(released >= finite(result.get("shutdown_end_epoch"), "result shutdown"), "guard released early")
    require(events[1].get("reason") == "SIGTERM", "guard release reason")


def verify_gpu_clean(path: Path) -> None:
    require(path.is_file(), f"missing GPU snapshot: {path}")
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    require(rows and len(rows[0]) >= 4, f"malformed GPU snapshot: {path}")
    require(int(rows[0][3].strip()) == 0, f"GPU memory is not clean: {path}")
    require(len(rows) == 1, f"compute process remains: {path}")


def verify(run_root: Path) -> dict[str, Any]:
    root = run_root.resolve()
    config = load_json(root / "run_config.json")
    require(config.get("schema") == "p3_cpu_dma_latency_no_offload_v1", "suite schema")
    require(config.get("mode_order") == EXPECTED_ORDER, "suite order")
    require(config.get("custom_module_cycle") is False, "custom module cycle used")
    require(config.get("stock_srcversion") == config.get("loaded_srcversion"), "non-stock UVM")
    require(config.get("sleep_gaps_ms") == EXPECTED_GAPS, "suite sleep gaps")
    require(config.get("busy_gaps_ms") == EXPECTED_GAPS, "suite busy gaps")
    require(config.get("locked_gaps_ms") == [0], "suite locked gaps")
    require(config.get("recovery_gaps_ms") == [0], "suite recovery gaps")

    with (root / "execution_status.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(len(rows) == 4, "expected four processes")
    require("".join(str(row.get("symbol")) for row in rows) == EXPECTED_ORDER, "row order")
    require(all(row.get("returncode") == "0" for row in rows), "case failure")

    output_hashes: set[str] = set()
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    cases: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows, 1):
        symbol = str(row["symbol"])
        mode = "baseline" if symbol == "B" else "dma_latency_held"
        case_dir = root / str(row["directory"])
        require(int((case_dir / "returncode.txt").read_text().strip()) == 0, "return code")
        result, calls = verify_result(case_dir, mode)
        verify_no_offload(case_dir)
        verify_guard(case_dir, symbol == "H", result)
        verify_gpu_clean(case_dir / "gpu.before.csv")
        verify_gpu_clean(case_dir / "gpu.after.csv")
        output_hashes.add(str(result["output_token_ids_sha256"]))
        for call in calls:
            if call["role"] != "measure":
                continue
            grouped[(mode, str(call["phase"]), int(call["requested_gap_ms"]))].append(call)
        cases.append(
            {
                "ordinal": ordinal,
                "symbol": symbol,
                "mode": mode,
                "directory": str(case_dir),
                "call_count": len(calls),
                "output_token_ids_sha256": result["output_token_ids_sha256"],
            }
        )
    require(len(output_hashes) == 1, "outputs differ across processes")
    verify_gpu_clean(root / "gpu.initial.csv")
    verify_gpu_clean(root / "gpu.final.csv")

    phase_summary: dict[str, Any] = {}
    phase_keys = sorted({(phase, gap) for _, phase, gap in grouped})
    for phase, gap in phase_keys:
        key = f"{phase}:{gap}ms"
        item: dict[str, Any] = {"phase": phase, "requested_gap_ms": gap, "modes": {}}
        for mode in ("baseline", "dma_latency_held"):
            calls = grouped[(mode, phase, gap)]
            expected_count = 4 if phase in {
                "auto_contiguous",
                "auto_gap_sweep",
                "auto_busy_gap_sweep",
            } else 2
            require(
                len(calls) == expected_count,
                f"{key}/{mode}: expected {expected_count} samples, got {len(calls)}",
            )
            item["modes"][mode] = {
                "wall_ms": summarize([float(call["wall_ms"]) for call in calls]),
                "ttft_p50_ms": summarize([float(call["ttft_ms"]["p50"]) for call in calls]),
                "actual_idle_ms": summarize([float(call["actual_idle_ms"]) for call in calls]),
                "sm_mhz_before": summarize([float(call["clock_before"]["sm_mhz"]) for call in calls]),
            }
        baseline = item["modes"]["baseline"]["wall_ms"]["median"]
        held = item["modes"]["dma_latency_held"]["wall_ms"]["median"]
        item["held_minus_baseline_wall_ms"] = held - baseline
        phase_summary[key] = item

    sleep_busy: dict[str, Any] = {}
    for mode in ("baseline", "dma_latency_held"):
        for gap in (110, 500):
            sleep_values = grouped[(mode, "auto_gap_sweep", gap)]
            busy_values = grouped[(mode, "auto_busy_gap_sweep", gap)]
            sleep_median = statistics.median(float(call["wall_ms"]) for call in sleep_values)
            busy_median = statistics.median(float(call["wall_ms"]) for call in busy_values)
            sleep_busy[f"{mode}:{gap}ms"] = {
                "sleep_wall_ms_median": sleep_median,
                "busy_wall_ms_median": busy_median,
                "sleep_minus_busy_wall_ms": sleep_median - busy_median,
            }

    return {
        "schema": "p3_cpu_dma_latency_no_offload_verification_v1",
        "status": "pass",
        "run_root": str(root),
        "mode_order": EXPECTED_ORDER,
        "cases": cases,
        "output_token_ids_sha256": output_hashes.pop(),
        "phase_summary": phase_summary,
        "sleep_busy_comparison": sleep_busy,
        "interpretation_contract": {
            "held_sleep_recovers_busy_floor": "guest PM-QoS-controlled idle state is implicated",
            "held_sleep_unchanged_busy_fast": "scheduler/cache/host state is more likely",
            "held_sleep_partial_recovery": "idle state and another factor both contribute",
            "kvm_caveat": "a negative guest result does not exclude physical-host C-states",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = verify(args.run_root)
    except (OSError, ValueError, VerificationError) as exc:
        raise SystemExit(f"verification failed: {exc}") from exc
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
