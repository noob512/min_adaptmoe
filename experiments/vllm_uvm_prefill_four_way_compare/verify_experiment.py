#!/usr/bin/env python3
"""Fail-closed verifier and aggregator for the four-way prefill experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DUAL_BANK_BYTES = 1_207_959_552
LOGICAL_BYTES = 2_415_919_104
ALIAS_COUNT = 6_912
ALIAS_BYTES = 3_623_878_656
ARM_NAMES = {
    "N": "native_copy_eager",
    "A": "alias_eager",
    "G": "alias_full_graph",
    "R": "remote_eager",
    "C": "remote_decode_graph",
    "F": "native_copy_full_graph_control",
}
ARM_PROFILES = {
    "N": (
        "qwen_l0_l3_native_copy_eager",
        "vllm_native_copy_on_forward_exact_l0_l3",
    ),
    "A": (
        "qwen_l0_l3_uvm_original_va_alias_eager",
        "vllm_uvm_original_va_static_alias_eager_l0_l3",
    ),
    "G": (
        "qwen_l0_l3_uvm_original_va_alias_full_graph",
        "vllm_uvm_original_va_static_alias_full_graph_l0_l3",
    ),
    "R": (
        "qwen_l0_l3_uvm_original_va_remote_eager",
        "vllm_uvm_original_va_remote_read_eager_l0_l3",
    ),
    "C": (
        "qwen_l0_l3_uvm_original_va_remote_graph",
        "vllm_uvm_original_va_remote_read_graph_l0_l3",
    ),
    "F": (
        "qwen_l0_l3_native_copy_full_graph_control",
        "vllm_native_copy_on_forward_full_graph_control_l0_l3",
    ),
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} is not an object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{line_number} is not an object")
            records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    require(records, f"{path} is empty")
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_input_hashes(run_root: Path) -> dict[str, Any]:
    path = run_root / "input_sha256.txt"
    seen: dict[Path, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        require(len(parts) == 2, f"{path}:{line_number} malformed")
        expected, raw = parts
        target = Path(raw.strip()).resolve()
        require(target.is_file(), f"locked input missing: {target}")
        require(target not in seen, f"duplicate locked input: {target}")
        require(sha256_file(target) == expected, f"locked input drift: {target}")
        seen[target] = expected
    required = {
        Path(__file__).resolve(),
        (Path(__file__).resolve().parent / "run_experiment.sh").resolve(),
        (
            Path(__file__).resolve().parent / "benchmark_compare_steps.py"
        ).resolve(),
        (
            Path(__file__).resolve().parents[1]
            / "vllm_uvm_original_va_graph_integration"
            / "run_integration.sh"
        ).resolve(),
        (
            Path(__file__).resolve().parents[1]
            / "vllm_p3_static_alias_graph_integration"
            / "benchmark_full_graph_steps.py"
        ).resolve(),
    }
    require(required <= set(seen), "input hash manifest omits matrix code")
    return {"file_count": len(seen), "sha256": sha256_file(path)}


def read_status(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(rows, "execution_status.tsv is empty")
    return rows


def expected_graph_config(symbol: str) -> tuple[str, str, list[int]]:
    if symbol in {"G", "F"}:
        return "full_graph_all", "FULL", [1, 4, 314, 1217]
    if symbol == "C":
        return "full_graph_decode", "FULL_DECODE_ONLY", [1, 4]
    return "eager", "NONE", []


def expected_arm_config(symbol: str) -> dict[str, Any]:
    execution, mode, sizes = expected_graph_config(symbol)
    alias = symbol not in {"N", "F"}
    profile, mechanism = ARM_PROFILES[symbol]
    return {
        "arm": ARM_NAMES[symbol],
        "profile_label": profile,
        "mechanism_class": mechanism,
        "execution_mode": execution,
        "cudagraph_mode": mode,
        "cudagraph_capture_sizes": sizes,
        "cpu_offload_backend": "copy",
        "graph_static_alias": alias,
        "graph_static_alias_backend": "uvm_original_va" if alias else None,
        "managed_expert_weights": alias,
        "original_va_phase_mode": "remote_only" if symbol in {"R", "C"} else "off",
    }


def validate_run_config(run_config: dict[str, Any]) -> tuple[str, str]:
    require(
        run_config.get("schema") == "vllm_uvm_original_va_graph_run_v1",
        "run config schema differs",
    )
    require(run_config.get("extended_prefill_matrix") is True, "extended matrix flag absent")
    require(run_config.get("attention_backend") == "TRITON_ATTN", "run attention backend differs")
    require(run_config.get("cudagraph_mode") == "PER_ARM", "global Graph mode is not per-arm")
    require(run_config.get("cudagraph_capture_sizes") is None, "global Graph sizes are ambiguous")
    order = str(run_config.get("order", ""))
    required = str(run_config.get("required_arms", ""))
    require(order and required, "run order or required arms absent")
    require(set(order) <= set(ARM_NAMES), "run order contains an unknown arm")
    require(len(set(required)) == len(required), "required arm set has duplicates")
    require(set(required) <= set(ARM_NAMES), "required arm set differs")
    require(len(order) % len(required) == 0, "run order has an incomplete block")
    arm_configs = run_config.get("arm_configs")
    require(isinstance(arm_configs, dict), "per-arm run config absent")
    require(set(arm_configs) == set(order), "per-arm run config keys differ from order")
    for symbol in set(order):
        require(
            arm_configs.get(symbol) == expected_arm_config(symbol),
            f"run config for arm {symbol} differs",
        )
    for offset in range(0, len(order), len(required)):
        require(
            set(order[offset : offset + len(required)]) == set(required),
            f"run order block {offset // len(required) + 1} mixes arms",
        )
    return order, required


def validate_status_identity(
    rows: list[dict[str, str]], order: str, required: str
) -> None:
    require(len(rows) == len(order), "case count differs")
    seen_directories: set[str] = set()
    for index, row in enumerate(rows):
        symbol = order[index]
        block = index // len(required) + 1
        expected_batch_order = "4,1" if block % 2 == 0 else "1,4"
        expected_directory = f"{index + 1:02d}_{ARM_NAMES[symbol]}"
        require(row.get("ordinal") == str(index + 1), f"row {index + 1}: ordinal differs")
        require(row.get("block") == str(block), f"row {index + 1}: block differs")
        require(row.get("symbol") == symbol, f"row {index + 1}: order differs")
        require(row.get("arm") == ARM_NAMES[symbol], f"row {index + 1}: arm differs")
        require(
            row.get("batch_order") == expected_batch_order,
            f"row {index + 1}: batch order differs",
        )
        require(
            row.get("directory") == expected_directory,
            f"row {index + 1}: directory differs",
        )
        require(
            expected_directory not in seen_directories,
            f"row {index + 1}: directory reused",
        )
        seen_directories.add(expected_directory)


def validate_graph_steps(result: dict[str, Any], symbol: str, label: str) -> None:
    contract = result.get("graph_contract", {})
    expected_prefill = symbol in {"G", "F"}
    expected_decode = symbol in {"G", "C", "F"}
    require(
        contract.get("expected_prefill_graph") is expected_prefill,
        f"{label}: prefill Graph contract differs",
    )
    require(
        contract.get("expected_decode_graph") is expected_decode,
        f"{label}: decode Graph contract differs",
    )
    prefill_count = int(contract.get("prefill_step_count", 0))
    decode_count = int(contract.get("decode_step_count", 0))
    require(prefill_count > 0 and decode_count > 0, f"{label}: step evidence absent")
    require(
        int(contract.get("prefill_full_replay_count", -1))
        == (prefill_count if expected_prefill else 0),
        f"{label}: prefill replay count differs",
    )
    require(
        int(contract.get("decode_full_replay_count", -1))
        == (decode_count if expected_decode else 0),
        f"{label}: decode replay count differs",
    )


def validate_vas(result: dict[str, Any], symbol: str, label: str) -> None:
    snapshots = result.get("parameter_va_snapshots", {}).get("snapshots", [])
    require(isinstance(snapshots, list) and snapshots, f"{label}: VA snapshots absent")
    if symbol in {"N", "F"}:
        require(
            all(not item.get("vas") for item in snapshots),
            f"{label}: native arm unexpectedly has original-VA alias snapshots",
        )
        return
    fingerprints = [tuple(int(value) for value in item.get("vas", [])) for item in snapshots]
    require(all(len(value) == 8 for value in fingerprints), f"{label}: VA count differs")
    require(len(set(fingerprints)) == 1, f"{label}: Parameter VA changed")
    expected_state = "sealed" if symbol in {"A", "G"} else "remote"
    require(
        all(item.get("lifecycle_state") == expected_state for item in snapshots),
        f"{label}: lifecycle state changed across steps",
    )


def validate_shutdown(result: dict[str, Any], symbol: str, label: str) -> None:
    shutdown = result.get("shutdown_contract", {})
    post = shutdown.get("diagnostics_after_shutdown", {})
    driver = shutdown.get("driver_diagnostics_after_shutdown", {})
    if symbol not in {"N", "F"}:
        require(int(post.get("session_id", -1)) == 0, f"{label}: session survived")
        require(
            int(post.get("active_driver_backing_bytes", -1)) == 0,
            f"{label}: backing survived",
        )
        require(driver.get("session_count") == 0, f"{label}: driver cache survived")
    shutdown_results = shutdown.get("model_runner_shutdown_results", [])
    require(shutdown_results, f"{label}: model-runner shutdown result absent")
    final = shutdown_results[-1]
    require(final.get("ok") is True, f"{label}: shutdown failed")
    expected_graphs = 4 if symbol == "G" else 2 if symbol == "C" else 0
    require(
        int(final.get("destroyed_graph_count", 0)) == expected_graphs,
        f"{label}: destroyed Graph count differs",
    )
    if symbol in {"A", "G"}:
        destroy = final.get("destroy", {})
        require(destroy.get("ok") is True, f"{label}: cmd99 failed")
        require(int(destroy.get("released_alias_count", 0)) == ALIAS_COUNT, f"{label}: aliases")
        require(int(destroy.get("released_alias_bytes", 0)) == ALIAS_BYTES, f"{label}: alias bytes")
        require(
            int(destroy.get("released_backing_bytes", 0)) == DUAL_BANK_BYTES,
            f"{label}: backing release",
        )
        for field in ("remaining_alias_count", "remaining_alias_bytes", "remaining_backing_bytes"):
            require(int(destroy.get(field, -1)) == 0, f"{label}: {field}")


def validate_trace(records: list[dict[str, Any]], symbol: str, label: str) -> dict[str, Any]:
    forwards = [item for item in records if item.get("event") == "cpu_offload_copy_forward"]
    trace_enabled = bool(forwards)
    summary: dict[str, Any] = {"trace_enabled": trace_enabled, "forward_count": len(forwards)}
    if not trace_enabled:
        return summary
    selected = [item for item in forwards if int(item.get("copy_bytes", 0)) == 603_979_776]
    require(selected, f"{label}: selected-layer trace absent")
    if symbol in {"N", "F"}:
        require(all(item.get("graph_static_alias") is False for item in selected), f"{label}: native alias")
    elif symbol in {"A", "G"}:
        require(
            all(int(item.get("graph_static_alias_actual_h2d_bytes", 0)) == 603_979_776 for item in selected),
            f"{label}: alias capture/eager H2D differs",
        )
        require(
            all(item.get("graph_static_alias_lifecycle_state") == "sealed" for item in selected),
            f"{label}: alias trace state differs",
        )
    else:
        require(
            all(int(item.get("graph_static_alias_actual_h2d_bytes", -1)) == 0 for item in selected),
            f"{label}: remote path submitted H2D",
        )
        require(
            all(item.get("graph_static_alias_lifecycle_state") == "remote" for item in selected),
            f"{label}: remote trace state differs",
        )
    summary["selected_forward_count"] = len(selected)
    summary["selected_actual_h2d_bytes"] = sum(
        int(item.get("graph_static_alias_actual_h2d_bytes", 0)) for item in selected
    )
    return summary


def validate_case(
    run_root: Path,
    row: dict[str, str],
    run_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    symbol = row["symbol"]
    label = row["directory"]
    require(symbol in ARM_NAMES, f"{label}: unknown symbol")
    require(row["arm"] == ARM_NAMES[symbol], f"{label}: arm name differs")
    require(int(row["returncode"]) == 0, f"{label}: process failed")
    directory = run_root / label
    result = load_json(directory / "result.json")
    require(result.get("status") == "ok", f"{label}: result not ok")
    require(result.get("arm") == symbol, f"{label}: result arm differs")
    require(not result.get("wrapper_contract_errors"), f"{label}: wrapper errors")
    config = result.get("config", {})
    execution, mode, sizes = expected_graph_config(symbol)
    require(config.get("execution_mode") == execution, f"{label}: execution mode")
    require(config.get("realized_cudagraph_mode") == mode, f"{label}: Graph mode")
    require(config.get("realized_cudagraph_capture_sizes") == sizes, f"{label}: Graph sizes")
    expected_profile, expected_mechanism = ARM_PROFILES[symbol]
    require(config.get("profile_label") == expected_profile, f"{label}: profile differs")
    require(config.get("mechanism_class") == expected_mechanism, f"{label}: mechanism differs")
    require(result.get("profile_label") == expected_profile, f"{label}: top-level profile differs")
    require(result.get("mechanism_class") == expected_mechanism, f"{label}: top-level mechanism differs")
    expected_batch_order = [int(value) for value in row["batch_order"].split(",")]
    require(config.get("batch_sizes") == expected_batch_order, f"{label}: batch order differs")
    arm_config = run_config.get("arm_configs", {}).get(symbol)
    require(arm_config == expected_arm_config(symbol), f"{label}: per-arm run config differs")
    environment = result.get("reproducibility_environment", {})
    require(
        environment.get("VLLM_ATTENTION_BACKEND") == "TRITON_ATTN",
        f"{label}: attention backend differs",
    )
    expected_alias = symbol not in {"N", "F"}
    expected_environment = {
        "VLLM_PREFILL_COMPARE_ARM": symbol,
        "VLLM_CPU_OFFLOAD_BACKEND": "copy",
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS": "1" if expected_alias else "0",
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND": (
            "uvm_original_va" if expected_alias else "vmm"
        ),
        "VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER": "1" if symbol == "A" else "0",
        "VLLM_UVM_EXPERT_WEIGHTS_MANAGED": "1" if expected_alias else "0",
        "VLLM_UVM_ORIGINAL_VA_PHASE_MODE": "remote_only" if symbol in {"R", "C"} else "off",
    }
    for key, value in expected_environment.items():
        require(environment.get(key) == value, f"{label}: environment {key} differs")
    require(
        int(config.get("kv_cache_memory_bytes", -1))
        == int(run_config["kv_cache_memory_bytes"]),
        f"{label}: KV cache differs",
    )
    validate_graph_steps(result, symbol, label)
    validate_vas(result, symbol, label)
    validate_shutdown(result, symbol, label)
    trace = validate_trace(load_jsonl(directory / "cpu_offload_manifest.jsonl"), symbol, label)
    before = load_json(directory / "gpu.before.json")
    after = load_json(directory / "gpu.after.json")
    for name, snapshot in (("before", before), ("after", after)):
        require(snapshot.get("gpu_clean") is True, f"{label}: GPU {name} dirty")
        require(int(snapshot.get("memory_used_mib", -1)) == 0, f"{label}: GPU memory")
        require(not snapshot.get("compute_pids") and not snapshot.get("uvm_client_pids"), f"{label}: PID leak")
    return result, trace


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        symbol = str(result["arm"])
        for batch, values in result.get("phase_summary", {}).items():
            grouped[(symbol, str(batch))].append(values)
    output: dict[str, Any] = {}
    for (symbol, batch), values in sorted(grouped.items()):
        target = output.setdefault(symbol, {})
        metrics: dict[str, Any] = {}
        for output_name, path in (
            ("prefill_step_ms", ("prefill_step_ms", "median")),
            ("decode_step_ms", ("decode_repeat_median_ms", "median")),
            ("cycle_e2e_ms", ("cycle_e2e_ms", "median")),
            ("prefill_tokens_per_second", ("prefill_tokens_per_second",)),
            ("decode_tokens_per_second", ("decode_tokens_per_second",)),
        ):
            samples: list[float] = []
            for value in values:
                current: Any = value
                for key in path:
                    current = current[key]
                samples.append(float(current))
            metrics[output_name] = {
                "samples": samples,
                "process_count": len(samples),
                "mean": statistics.mean(samples),
                "median": statistics.median(samples),
                "min": min(samples),
                "max": max(samples),
                "sample_stdev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
            }
        target[batch] = metrics
    return output


def ratio(aggregate_data: dict[str, Any], candidate: str, baseline: str, batch: str, metric: str) -> dict[str, float]:
    candidate_ms = float(aggregate_data[candidate][batch][metric]["median"])
    baseline_ms = float(aggregate_data[baseline][batch][metric]["median"])
    return {
        f"{candidate}_ms": candidate_ms,
        f"{baseline}_ms": baseline_ms,
        f"{candidate}_over_{baseline}_ratio": candidate_ms / baseline_ms,
        f"{candidate}_vs_{baseline}_change_percent": (candidate_ms / baseline_ms - 1.0) * 100.0,
    }


def verify(run_root: Path) -> dict[str, Any]:
    run_config = load_json(run_root / "run_config.json")
    rows = read_status(run_root / "execution_status.tsv")
    order, required = validate_run_config(run_config)
    if run_config.get("mode") != "smoke":
        require(set(required) == set(ARM_NAMES), "run does not require all six controls")
    validate_status_identity(rows, order, required)
    results: list[dict[str, Any]] = []
    traces: dict[str, Any] = {}
    # FULL mode is a distinct vLLM/TRITON execution path.  F is the native
    # copy-on-forward FULL control for G, while N/A/R/C form the eager/remote
    # semantic reference group.  Require exact output identity within each
    # execution class instead of misattributing a generic FULL-mode numerical
    # difference to the alias transport.
    output_hashes: dict[tuple[str, int, str, int], str] = {}
    for row in rows:
        result, trace = validate_case(run_root, row, run_config)
        results.append(result)
        traces[row["directory"]] = trace
        output_class = "full_graph" if result["arm"] in {"G", "F"} else "eager_remote"
        for cycle in result.get("cycles", []):
            key = (
                output_class,
                int(cycle["batch_size"]),
                str(cycle["role"]),
                int(cycle["role_index"]),
            )
            digest = str(cycle["output_token_ids_sha256"])
            previous = output_hashes.setdefault(key, digest)
            require(previous == digest, f"output hash differs for {key}")

    before = load_json(run_root / "module.before.json")
    custom = load_json(run_root / "module.custom.json")
    after = load_json(run_root / "module.after.json")
    require(custom.get("manual_migration_fault_mode") == 2, "mode-2 not active")
    require(custom.get("slot_pte_remap_enable") == 1, "slot remap not active")
    for field in ("module_sha256", "loaded_srcversion"):
        require(before.get(field) == after.get(field), f"stock module {field} not restored")
    for name, snapshot in (("before", before), ("custom", custom), ("after", after)):
        require(snapshot.get("gpu_clean") is True, f"module {name} GPU state dirty")
        require(int(snapshot.get("memory_used_mib", -1)) == 0, f"module {name} GPU memory")
        require(
            not snapshot.get("compute_pids") and not snapshot.get("uvm_client_pids"),
            f"module {name} retained a GPU/UVM client",
        )

    aggregated = aggregate(results)
    comparisons: dict[str, Any] = {}
    if set(ARM_NAMES) <= set(aggregated):
        for batch in ("1", "4"):
            comparisons[batch] = {
                "alias_graph_vs_alias_eager_prefill": ratio(aggregated, "G", "A", batch, "prefill_step_ms"),
                "alias_eager_vs_native_prefill": ratio(aggregated, "A", "N", batch, "prefill_step_ms"),
                "alias_graph_vs_native_prefill": ratio(aggregated, "G", "N", batch, "prefill_step_ms"),
                "remote_vs_native_prefill": ratio(aggregated, "R", "N", batch, "prefill_step_ms"),
                "alias_eager_vs_remote_prefill": ratio(aggregated, "A", "R", batch, "prefill_step_ms"),
                "alias_graph_vs_remote_prefill": ratio(aggregated, "G", "R", batch, "prefill_step_ms"),
                "alias_graph_vs_alias_eager_decode": ratio(aggregated, "G", "A", batch, "decode_step_ms"),
                "remote_graph_vs_remote_eager_decode": ratio(aggregated, "C", "R", batch, "decode_step_ms"),
                "alias_graph_vs_native_graph_prefill": ratio(aggregated, "G", "F", batch, "prefill_step_ms"),
                "alias_graph_vs_native_graph_decode": ratio(aggregated, "G", "F", batch, "decode_step_ms"),
            }
    return {
        "schema": "vllm_uvm_prefill_four_way_verification_v1",
        "status": "ok",
        "run_root": str(run_root.resolve()),
        "case_count": len(results),
        "input_hashes": validate_input_hashes(run_root),
        "output_hashes": {
            f"{key[0]}_b{key[1]}_{key[2]}_{key[3]}": value
            for key, value in sorted(output_hashes.items())
        },
        "trace_summaries": traces,
        "aggregate_performance": aggregated,
        "comparisons": comparisons,
        "module_cycle": {"before": before, "custom": custom, "after": after},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = verify(args.run_root.resolve())
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
