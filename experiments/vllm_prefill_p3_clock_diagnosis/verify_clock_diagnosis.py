#!/usr/bin/env python3
"""Verify and summarize the controlled P3 idle/clock diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ARMS = ("p3", "pinned", "uva")
LAYER_BYTES = 603_979_776
REGIONS_PER_LAYER = 1152
TLB_PER_LAYER = 288


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"not a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL: {path}")
    result: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        require(isinstance(value, dict), f"{path}:{line_number}: not an object")
        result.append(value)
    return result


def load_env(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        require("=" in raw, f"{path}:{line_number}: malformed environment")
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
    require(bool(values), "cannot summarize an empty population")
    return {
        "count": len(values),
        "minimum": min(values),
        "median": statistics.median(values),
        "maximum": max(values),
        "samples": values,
    }


def verify_common_result(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = load_json(path)
    require(result.get("schema") == "vllm_prefill_p3_clock_diagnosis_v1", f"{path}: schema")
    require(result.get("status") == "ok", f"{path}: status")
    config = result.get("config")
    require(isinstance(config, dict), f"{path}: config absent")
    require(config.get("concurrency") == 4, f"{path}: concurrency")
    require(config.get("output_tokens") == 1, f"{path}: output tokens")
    require(config.get("warmup_requests") == 4, f"{path}: warmup requests")
    require(config.get("disable_chunked_prefill") is True, f"{path}: chunked prefill")
    calls = result.get("calls")
    require(isinstance(calls, list) and calls, f"{path}: calls absent")
    expected_indices = list(range(len(calls)))
    require([call.get("call_index") for call in calls] == expected_indices, f"{path}: call order")
    hashes: set[str] = set()
    for call in calls:
        require(isinstance(call, dict), f"{path}: malformed call")
        require(call.get("role") in {"condition", "measure"}, f"{path}: call role")
        require(
            call.get("clock_mode") in {"auto", "locked", "auto_recovery"},
            f"{path}: clock mode",
        )
        gap = call.get("requested_gap_ms")
        require(isinstance(gap, int) and gap >= 0, f"{path}: requested gap")
        idle = call.get("actual_idle_ms")
        require(isinstance(idle, (int, float)) and float(idle) >= 0, f"{path}: actual gap")
        if call.get("role") == "measure":
            require(float(idle) + 0.05 >= gap, f"{path}: requested gap was not honored")
        start = finite(call.get("start_epoch"), f"{path}: start")
        end = finite(call.get("end_epoch"), f"{path}: end")
        require(end >= start, f"{path}: reversed call window")
        require(finite(call.get("wall_ms"), f"{path}: wall") > 0, f"{path}: wall")
        before = call.get("clock_before")
        after = call.get("clock_after")
        require(isinstance(before, dict) and isinstance(after, dict), f"{path}: clock sample")
        for sample in (before, after):
            for key in ("pstate", "sm_mhz", "memory_mhz", "power_watts", "temperature_c"):
                finite(sample.get(key), f"{path}: clock {key}")
        if call.get("clock_mode") == "locked":
            requested_memory = int(config.get("locked_memory_mhz"))
            observed_memory = int(before.get("memory_mhz"))
            require(
                abs(observed_memory - requested_memory) <= 500,
                f"{path}: memory lock",
            )
        token_hash = call.get("output_token_ids_sha256")
        require(isinstance(token_hash, str) and len(token_hash) == 64, f"{path}: output hash")
        hashes.add(token_hash)
    require(len(hashes) == 1, f"{path}: repeated outputs differ")
    require(result.get("output_token_ids_sha256") in hashes, f"{path}: aggregate output hash")
    events = result.get("clock_events")
    require(isinstance(events, list) and events, f"{path}: clock events absent")
    require(not any("error" in event for event in events), f"{path}: clock command failed")
    return result, calls


def verify_pinned(case_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    require(env.get("VLLM_CPU_OFFLOAD_BACKEND") == "copy", "pinned backend")
    records = load_jsonl(case_dir / "cpu_offload_manifest.jsonl")
    parameters = [record for record in records if record.get("event") == "cpu_offload_parameter"]
    require(len(parameters) == 8, "pinned parameter count")
    require(all(record.get("cpu_data_pinned") is True for record in parameters), "unpinned source")
    require(all(record.get("copy_on_forward") is True for record in parameters), "not copy-on-forward")
    byte_count = sum(int(record.get("bytes", -1)) for record in parameters)
    require(byte_count == 4 * LAYER_BYTES, "pinned selected bytes")
    return {"parameters": len(parameters), "bytes": byte_count}


def verify_uva(case_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    require(env.get("VLLM_CPU_OFFLOAD_BACKEND") == "uva", "UVA backend")
    require(env.get("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE") == "remote_read", "UVA access mode")
    placement = load_json(case_dir / "runtime_remote_placement_manifest.json")
    require(placement.get("status") == "ok" and placement.get("complete") is True, "UVA placement")
    require(placement.get("remote_expert_count") == 512, "UVA expert count")
    require(placement.get("remote_region_count") == 4608, "UVA region count")
    require(placement.get("remote_region_bytes") == 4 * LAYER_BYTES, "UVA bytes")
    return {
        "experts": placement["remote_expert_count"],
        "regions": placement["remote_region_count"],
        "bytes": placement["remote_region_bytes"],
    }


def verify_restore(record: dict[str, Any], expected_layer: int) -> float:
    require(record.get("operation") == "restore_layer", "P3 restore operation")
    require(record.get("layer_id") == expected_layer and record.get("ok") is True, "P3 layer/order")
    low = record.get("restore")
    require(isinstance(low, dict) and low.get("ok") is True, "P3 ioctl failed")
    expected = {
        "plan_execute": True,
        "requested_bytes": LAYER_BYTES,
        "accepted_bytes": LAYER_BYTES,
        "requested_count": REGIONS_PER_LAYER,
        "success_count": REGIONS_PER_LAYER,
        "failure_count": 0,
        "copy_push_count": 1,
        "tlb_invalidate_count": TLB_PER_LAYER,
        "cuda_fence_accepted": True,
        "cuda_completion_signaled": True,
        "cuda_stream_wait_enqueued": True,
        "success_contract_ok": True,
        "poisoned": False,
        "kernel_poisoned": False,
    }
    for key, value in expected.items():
        require(low.get(key) == value, f"P3 {key} differs")
    wall_ns = low.get("ioctl_wall_ns")
    require(isinstance(wall_ns, int) and wall_ns > 0, "P3 ioctl timing absent")
    return wall_ns / 1_000_000.0


def records_in_window(
    records: list[dict[str, Any]], start_epoch: float, end_epoch: float
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if start_epoch <= finite(record.get("timestamp"), "P3 timestamp") <= end_epoch
    ]


def verify_p3(
    case_dir: Path, env: dict[str, str], result: dict[str, Any], calls: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    require(env.get("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE") == "slot_restore", "P3 access mode")
    trace = load_jsonl(case_dir / "slot_trace.jsonl")
    restores = [record for record in trace if record.get("operation") == "restore_layer"]
    require(len(restores) == 4 * (1 + len(calls)), "P3 transition count")
    warmup = result.get("warmup")
    require(isinstance(warmup, dict), "P3 warmup window")
    warm_records = records_in_window(
        restores,
        finite(warmup.get("start_epoch"), "warmup start"),
        finite(warmup.get("end_epoch"), "warmup end"),
    )
    require(len(warm_records) == 4, "P3 warmup transitions")
    for layer, record in enumerate(warm_records):
        verify_restore(record, layer)
    by_call: dict[int, dict[str, Any]] = {}
    seen: set[int] = {id(record) for record in warm_records}
    for call in calls:
        records = records_in_window(restores, float(call["start_epoch"]), float(call["end_epoch"]))
        require(len(records) == 4, f"P3 call {call['call_index']}: transition count")
        ioctl_ms = [verify_restore(record, layer) for layer, record in enumerate(records)]
        seen.update(id(record) for record in records)
        by_call[int(call["call_index"])] = {
            "ioctl_ms": ioctl_ms,
            "ioctl_sum_ms": sum(ioctl_ms),
            "ioctl_median_ms": statistics.median(ioctl_ms),
            "l0_ioctl_ms": ioctl_ms[0],
        }
    require(len(seen) == len(restores), "P3 transition outside declared windows")
    return {
        "transitions": len(restores),
        "measured_calls": len(calls),
        "bytes": len(restores) * LAYER_BYTES,
        "errors": 0,
    }, by_call


def call_contract(call: dict[str, Any]) -> tuple[Any, ...]:
    return (
        call.get("call_index"),
        call.get("phase"),
        call.get("role"),
        call.get("clock_mode"),
        call.get("requested_gap_ms"),
    )


def verify(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    config = load_json(run_root / "run_config.json")
    require(config.get("order") == "PCU", "diagnosis order must be PCU")
    with (run_root / "execution_status.tsv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(len(rows) == 3, "expected exactly three arm processes")
    require(all(row.get("returncode") == "0" for row in rows), "an arm failed")
    require([row.get("arm") for row in rows] == list(ARMS), "arm order differs")

    results: dict[str, dict[str, Any]] = {}
    calls_by_arm: dict[str, list[dict[str, Any]]] = {}
    evidence: dict[str, Any] = {}
    p3_timings: dict[int, dict[str, Any]] = {}
    output_hashes: set[str] = set()
    for row in rows:
        arm = str(row["arm"])
        case_dir = run_root / str(row["directory"])
        result, calls = verify_common_result(case_dir / "result.json")
        env = load_env(case_dir / "environment.txt")
        if arm == "p3":
            evidence[arm], p3_timings = verify_p3(case_dir, env, result, calls)
        elif arm == "pinned":
            evidence[arm] = verify_pinned(case_dir, env)
        else:
            evidence[arm] = verify_uva(case_dir, env)
        results[arm] = result
        calls_by_arm[arm] = calls
        output_hashes.add(str(result["output_token_ids_sha256"]))
    require(len(output_hashes) == 1, "outputs differ across mechanisms")

    reference = [call_contract(call) for call in calls_by_arm["p3"]]
    for arm in ARMS[1:]:
        require([call_contract(call) for call in calls_by_arm[arm]] == reference, f"{arm}: phase contract")

    grouped: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for arm in ARMS:
        for call in calls_by_arm[arm]:
            if call["role"] != "measure":
                continue
            key = (str(call["phase"]), int(call["requested_gap_ms"]))
            grouped[key][arm].append(call)

    phase_summary: dict[str, Any] = {}
    for (phase, gap), arms in sorted(grouped.items()):
        key = f"{phase}:{gap}ms"
        item: dict[str, Any] = {"phase": phase, "requested_gap_ms": gap, "arms": {}}
        for arm in ARMS:
            population = arms[arm]
            arm_summary: dict[str, Any] = {
                "wall_ms": summarize([float(call["wall_ms"]) for call in population]),
                "actual_idle_ms": summarize(
                    [float(call["actual_idle_ms"]) for call in population]
                ),
                "clock_before_sm_mhz": summarize(
                    [float(call["clock_before"]["sm_mhz"]) for call in population]
                ),
                "clock_before_memory_mhz": summarize(
                    [float(call["clock_before"]["memory_mhz"]) for call in population]
                ),
            }
            if arm == "p3":
                timings = [p3_timings[int(call["call_index"])] for call in population]
                arm_summary.update(
                    {
                        "ioctl_sum_ms": summarize(
                            [float(timing["ioctl_sum_ms"]) for timing in timings]
                        ),
                        "l0_ioctl_ms": summarize(
                            [float(timing["l0_ioctl_ms"]) for timing in timings]
                        ),
                        "ioctl_layer_samples_ms": [timing["ioctl_ms"] for timing in timings],
                    }
                )
            item["arms"][arm] = arm_summary
        item["p3_minus_pinned_wall_ms"] = (
            item["arms"]["p3"]["wall_ms"]["median"]
            - item["arms"]["pinned"]["wall_ms"]["median"]
        )
        phase_summary[key] = item

    return {
        "schema": "vllm_prefill_p3_clock_diagnosis_verification_v1",
        "status": "pass",
        "run_root": str(run_root),
        "output_token_ids_sha256": next(iter(output_hashes)),
        "evidence": evidence,
        "phase_summary": phase_summary,
        "interpretation_note": (
            "Calls within one process are repeated diagnostic observations, not "
            "independent statistical samples. Compare the idle dose response and "
            "locked-vs-auto behavior; do not treat call count as process-level n."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.run_root)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, VerificationError) as exc:
        print(f"verification failed: {exc}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
