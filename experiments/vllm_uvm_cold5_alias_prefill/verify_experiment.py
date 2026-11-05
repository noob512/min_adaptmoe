#!/usr/bin/env python3
"""Verify correctness and summarize cold-5 prefill performance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ARMS = {
    "T": "route_trace_control",
    "Z": "all_gpu_graph",
    "A": "alias_eager",
    "G": "alias_graph",
}
STATUS_COLUMNS = [
    "ordinal",
    "symbol",
    "arm",
    "batch_order",
    "directory",
    "returncode",
]
COMMON_CONFIG_KEYS = (
    "conditioning_repeats",
    "data_parallel_size",
    "enable_chunked_prefill",
    "enable_dbo",
    "enable_prefix_caching",
    "engine_core_multiprocessing",
    "fixture",
    "frozen_prompt_token_ids",
    "gpu_memory_utilization",
    "ignore_eos",
    "kv_cache_memory_bytes",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "measured_decode_steps",
    "measurement_repeats",
    "model",
    "pipeline_parallel_size",
    "seed",
    "temperature",
    "tensor_parallel_size",
    "total_output_tokens",
    "warmup_decode_steps",
)


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def load_run_contract(
    run_root: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], list[Path]]:
    run_config = load_json(run_root / "run_config.json")
    require(
        run_config.get("schema") == "vllm_uvm_cold5_alias_prefill_run_v1",
        "run config schema differs",
    )
    mode = str(run_config.get("mode"))
    order = str(run_config.get("order", ""))
    require(mode in {"qualify", "formal"}, f"unverifiable mode: {mode}")
    require(order and all(symbol in ARMS for symbol in order), "run order differs")
    arm_counts = Counter(order)
    if mode == "qualify":
        require(
            arm_counts == Counter({"T": 1, "Z": 1, "A": 1, "G": 1}),
            "qualify arm counts differ",
        )
    else:
        performance_counts = [arm_counts[symbol] for symbol in ("Z", "A", "G")]
        require(arm_counts["T"] == 1, "formal route-trace count differs")
        require(
            len(set(performance_counts)) == 1 and performance_counts[0] >= 3,
            f"formal performance arm counts differ: {arm_counts}",
        )
        recorded_count = run_config.get("formal_processes_per_arm")
        if recorded_count is not None:
            require(
                int(recorded_count) == performance_counts[0],
                "formal process count metadata differs",
            )

    status_path = run_root / "execution_status.tsv"
    with status_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require(reader.fieldnames == STATUS_COLUMNS, "execution status columns differ")
        rows = list(reader)
    require(len(rows) == len(order), "execution status row count differs")

    result_paths: list[Path] = []
    batch_orders: dict[str, Counter[str]] = defaultdict(Counter)
    for index, (symbol, row) in enumerate(zip(order, rows, strict=True)):
        ordinal = index + 1
        expected_arm = ARMS[symbol]
        expected_batch_order = "1,4" if index % 2 == 0 else "4,1"
        expected_directory = f"{ordinal:02d}_{expected_arm}"
        require(row["ordinal"] == str(ordinal), f"row {ordinal}: ordinal differs")
        require(row["symbol"] == symbol, f"row {ordinal}: symbol differs")
        require(row["arm"] == expected_arm, f"row {ordinal}: arm differs")
        require(
            row["batch_order"] == expected_batch_order,
            f"row {ordinal}: batch order differs",
        )
        require(
            row["directory"] == expected_directory,
            f"row {ordinal}: directory differs",
        )
        require(row["returncode"] == "0", f"row {ordinal}: return code differs")
        result_path = run_root / expected_directory / "result.json"
        require(result_path.is_file(), f"row {ordinal}: result is absent")
        result_paths.append(result_path)
        batch_orders[symbol][expected_batch_order] += 1

    if mode == "formal":
        for symbol in ("Z", "A", "G"):
            counts = batch_orders[symbol]
            require(
                set(counts) == {"1,4", "4,1"}
                and abs(counts["1,4"] - counts["4,1"]) <= 1,
                f"formal batch order is not balanced for {symbol}: {counts}",
            )

    initially_active = {
        line.strip()
        for line in (run_root / "services.initially_active.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    service_states: dict[str, str] = {}
    for line in (run_root / "services.after.tsv").read_text(
        encoding="utf-8"
    ).splitlines():
        service, state = line.split("\t", 1)
        service_states[service] = state
    require(
        all(service_states.get(service) == "active" for service in initially_active),
        "initially active service was not restored",
    )
    gpu_before = load_json(run_root / "gpu.before.json")
    gpu_stock = load_json(run_root / "gpu.stock.json")
    require(
        gpu_before.get("loaded_srcversion") == gpu_stock.get("loaded_srcversion"),
        "stock UVM was not restored",
    )
    for label, snapshot in (("before", gpu_before), ("stock", gpu_stock)):
        require(int(snapshot.get("memory_used_mib", -1)) == 0, f"GPU {label} memory")
        require(not snapshot.get("compute_pids"), f"GPU {label} compute clients")
        require(not snapshot.get("uvm_client_pids"), f"GPU {label} UVM clients")
    return run_config, rows, result_paths


def expected_output_hash_keys(
    *, conditioning_repeats: int, measurement_repeats: int
) -> set[tuple[int, str, int]]:
    return {
        (batch, role, role_index)
        for batch in (1, 4)
        for role, repeats in (
            ("conditioning", conditioning_repeats),
            ("measurement", measurement_repeats),
        )
        for role_index in range(repeats)
    }


def summarize(values: list[float]) -> dict[str, Any]:
    require(values, "cannot summarize an empty sample")
    ordered = sorted(values)
    return {
        "count": len(values),
        "samples_ms": values,
        "min_ms": ordered[0],
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "max_ms": ordered[-1],
    }


def output_hashes(result: dict[str, Any]) -> dict[tuple[int, str, int], str]:
    values: dict[tuple[int, str, int], str] = {}
    for cycle in result.get("cycles", []):
        key = (
            int(cycle["batch_size"]),
            str(cycle["role"]),
            int(cycle["role_index"]),
        )
        values[key] = str(cycle["output_token_ids_sha256"])
    return values


def validate_result_config(
    result: dict[str, Any],
    *,
    path: Path,
    symbol: str,
    row: dict[str, str],
    run_config: dict[str, Any],
    common_fingerprint: dict[str, Any] | None,
    versions_fingerprint: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = result.get("config")
    versions = result.get("versions")
    environment = result.get("reproducibility_environment")
    require(isinstance(config, dict), f"{path}: config absent")
    require(isinstance(versions, dict), f"{path}: versions absent")
    require(isinstance(environment, dict), f"{path}: environment absent")
    expected_batch_sizes = [int(value) for value in row["batch_order"].split(",")]
    require(config.get("batch_sizes") == expected_batch_sizes, f"{path}: batches")
    for key in (
        "conditioning_repeats",
        "measurement_repeats",
        "warmup_decode_steps",
        "measured_decode_steps",
    ):
        require(config.get(key) == run_config.get(key), f"{path}: {key} differs")
    require(
        Path(str(config.get("model"))).resolve()
        == Path(str(run_config.get("model"))).resolve(),
        f"{path}: model differs",
    )
    require(
        Path(str(config.get("fixture"))).resolve()
        == Path(str(run_config.get("fixture"))).resolve(),
        f"{path}: fixture differs",
    )

    graph = symbol in {"Z", "G"}
    alias = symbol in {"A", "G"}
    trace = symbol == "T"
    require(
        config.get("execution_mode") == ("full_graph_all" if graph else "eager"),
        f"{path}: execution mode differs",
    )
    require(config.get("enforce_eager") is (not graph), f"{path}: eager differs")
    require(
        config.get("realized_cudagraph_mode") == ("FULL" if graph else "NONE"),
        f"{path}: realized Graph mode differs",
    )
    require(
        float(config.get("cpu_offload_gb", -1.0)) == (1.25 if alias else 0.0),
        f"{path}: offload budget differs",
    )
    require(
        environment.get("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS")
        == ("1" if alias else "0"),
        f"{path}: alias environment differs",
    )
    require(
        environment.get("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER")
        == ("1" if symbol == "A" else "0"),
        f"{path}: eager alias environment differs",
    )
    require(
        environment.get("VLLM_MOE_TOKEN_EXPERT_TRACE_ENABLE")
        == ("1" if trace else "0"),
        f"{path}: trace environment differs",
    )
    expected_copy_trace = (
        str(int(run_config.get("performance_copy_trace", -1))) if alias else "0"
    )
    require(
        environment.get("VLLM_CPU_OFFLOAD_COPY_TRACE") == expected_copy_trace,
        f"{path}: copy trace environment differs",
    )
    for key in (
        "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE",
        "VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE",
        "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE",
        "VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE",
    ):
        require(environment.get(key) == "0", f"{path}: {key} differs")

    current_common = {key: config.get(key) for key in COMMON_CONFIG_KEYS}
    if common_fingerprint is not None:
        require(current_common == common_fingerprint, f"{path}: common config differs")
    if versions_fingerprint is not None:
        require(versions == versions_fingerprint, f"{path}: versions differ")
    return current_common, dict(versions)


def route_summary(run_root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    selected = {
        int(layer): {int(expert) for expert in experts}
        for layer, experts in plan["selected_by_layer"].items()
    }
    records: list[dict[str, Any]] = []
    trace_files = sorted(run_root.glob("*_route_trace_control/expert_trace.jsonl"))
    for path in trace_files:
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            value = json.loads(raw)
            if value.get("event") == "qwen3_moe_expert_trace" and int(
                value.get("num_tokens", 0)
            ) in {314, 1217}:
                records.append(value)
    require(records, "evaluation prefill route trace is absent")
    by_prompt: dict[int, dict[str, Any]] = {}
    for prompt_tokens in (314, 1217):
        prompt_records = [
            record for record in records if int(record["num_tokens"]) == prompt_tokens
        ]
        layers = Counter(int(record["layer_id"]) for record in prompt_records)
        require(set(layers) == set(range(48)), f"route trace layers differ for {prompt_tokens}")
        cold_hits = 0
        routed_slots = 0
        hit_pairs: set[tuple[int, int]] = set()
        for record in prompt_records:
            layer_id = int(record["layer_id"])
            counts = {
                int(expert): int(count)
                for expert, count in record.get("expert_token_counts", {}).items()
            }
            routed_slots += sum(counts.values())
            for expert_id in selected[layer_id]:
                count = counts.get(expert_id, 0)
                cold_hits += count
                if count:
                    hit_pairs.add((layer_id, expert_id))
        require(cold_hits > 0, f"no selected cold expert was routed for {prompt_tokens}")
        by_prompt[prompt_tokens] = {
            "record_count": len(prompt_records),
            "repeat_count": min(layers.values()),
            "routed_slots": routed_slots,
            "selected_cold_hits": cold_hits,
            "selected_cold_fraction": cold_hits / routed_slots,
            "hit_layer_expert_pair_count": len(hit_pairs),
        }
    return {
        "evidence_scope": "route_trace_control_eager_only",
        "trace_file_count": len(trace_files),
        "evaluation_record_count": len(records),
        "by_prompt_tokens": by_prompt,
    }


def verify(run_root: Path) -> dict[str, Any]:
    plan_path = HERE / "cold5_expert_plan.json"
    names_path = HERE / "exact_parameter_names.txt"
    plan = load_json(plan_path)
    require(plan.get("selected_expert_count") == 240, "frozen plan count differs")
    require(len(names_path.read_text(encoding="utf-8").splitlines()) == 96, "exact names differ")
    run_config, status_rows, result_paths = load_run_contract(run_root)
    require(
        int(run_config.get("selected_experts_per_layer", -1)) == 5
        and int(run_config.get("layer_count", -1)) == 48
        and int(run_config.get("parameter_count", -1)) == 96
        and int(run_config.get("selected_bytes_per_prefill", -1)) == 1_132_462_080
        and int(run_config.get("backing_bytes", -1)) == 47_185_920,
        "run geometry differs",
    )
    model_config_path = Path(str(run_config.get("model"))) / "config.json"
    fixture_path = Path(str(run_config.get("fixture")))
    require(model_config_path.is_file(), "model config is absent")
    require(fixture_path.is_file(), "fixture is absent")
    require(
        sha256_file(model_config_path) == run_config.get("model_config_sha256"),
        "model config hash differs",
    )
    require(
        sha256_file(fixture_path) == run_config.get("fixture_sha256"),
        "fixture hash differs",
    )
    require(
        set(run_root.glob("??_*/result.json")) == set(result_paths),
        "unexpected result file set",
    )

    results: list[tuple[Path, dict[str, Any]]] = []
    common_fingerprint: dict[str, Any] | None = None
    versions_fingerprint: dict[str, Any] | None = None
    for path, row in zip(result_paths, status_rows, strict=True):
        result = load_json(path)
        require(result.get("status") == "ok", f"{path}: status={result.get('status')}")
        require(not result.get("wrapper_contract_errors"), f"{path}: wrapper errors")
        arm = str(result.get("arm"))
        require(arm == row["symbol"], f"{path}: arm symbol differs")
        current_common, current_versions = validate_result_config(
            result,
            path=path,
            symbol=arm,
            row=row,
            run_config=run_config,
            common_fingerprint=common_fingerprint,
            versions_fingerprint=versions_fingerprint,
        )
        if common_fingerprint is None:
            common_fingerprint = current_common
        if versions_fingerprint is None:
            versions_fingerprint = current_versions
        results.append((path, result))

    seen_arms = {str(result["arm"]) for _, result in results}
    require(set(ARMS) == seen_arms, f"arm coverage differs: {seen_arms}")

    execution_reference_hashes: dict[
        str, dict[tuple[int, str, int], str]
    ] = {}
    execution_reference_arms = {
        "eager": {"T", "A"},
        "full_graph": {"Z", "G"},
    }
    hash_evidence: dict[str, int] = {}
    prefill_samples: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    decode_samples: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    expected_hash_keys = expected_output_hash_keys(
        conditioning_repeats=int(run_config["conditioning_repeats"]),
        measurement_repeats=int(run_config["measurement_repeats"]),
    )
    for path, result in results:
        arm = str(result["arm"])
        hashes = output_hashes(result)
        require(set(hashes) == expected_hash_keys, f"{path}: output hash keys differ")
        execution_class = next(
            name
            for name, members in execution_reference_arms.items()
            if arm in members
        )
        reference_hashes = execution_reference_hashes.get(execution_class)
        if reference_hashes is None:
            execution_reference_hashes[execution_class] = hashes
        else:
            require(
                hashes == reference_hashes,
                f"{path}: output hash differs within {execution_class}",
            )
        hash_evidence[arm] = hash_evidence.get(arm, 0) + len(hashes)
        if arm == "T":
            continue
        for cycle in result.get("cycles", []):
            if cycle.get("role") != "measurement":
                continue
            batch = int(cycle["batch_size"])
            prefill_samples[arm][batch].append(float(cycle["prefill_elapsed_ms"]))
            decode = cycle.get("decode_step_ms", {})
            values = [float(value) for value in decode.get("samples", [])]
            require(
                len(values) == int(run_config["measured_decode_steps"]),
                f"{path}: measured decode sample count differs",
            )
            decode_samples[arm][batch].extend(values)

    require(
        set(execution_reference_hashes) == set(execution_reference_arms),
        "correctness reference absent",
    )
    performance: dict[str, dict[str, Any]] = {}
    decode_performance: dict[str, dict[str, Any]] = {}
    arm_counts = Counter(str(result["arm"]) for _, result in results)
    measurement_repeats = int(run_config["measurement_repeats"])
    for arm in ("Z", "A", "G"):
        performance[arm] = {}
        decode_performance[arm] = {}
        for batch in (1, 4):
            expected_sample_count = arm_counts[arm] * measurement_repeats
            require(
                len(prefill_samples[arm][batch]) == expected_sample_count,
                f"{arm} batch {batch}: sample count differs",
            )
            performance[arm][str(batch)] = summarize(prefill_samples[arm][batch])
            expected_decode_count = (
                expected_sample_count * int(run_config["measured_decode_steps"])
            )
            require(
                len(decode_samples[arm][batch]) == expected_decode_count,
                f"{arm} batch {batch}: decode sample count differs",
            )
            decode_performance[arm][str(batch)] = summarize(
                decode_samples[arm][batch]
            )

    comparisons: dict[str, dict[str, float]] = {}
    for batch in (1, 4):
        control = performance["Z"][str(batch)]["median_ms"]
        eager = performance["A"][str(batch)]["median_ms"]
        graph = performance["G"][str(batch)]["median_ms"]
        eager_decode = decode_performance["A"][str(batch)]["median_ms"]
        graph_decode = decode_performance["G"][str(batch)]["median_ms"]
        comparisons[str(batch)] = {
            "alias_eager_vs_all_gpu_graph_percent":
                (eager / control - 1.0) * 100.0,
            "alias_graph_vs_all_gpu_graph_percent":
                (graph / control - 1.0) * 100.0,
            "alias_graph_vs_alias_eager_percent": (graph / eager - 1.0) * 100.0,
            "alias_graph_prefill_speedup": eager / graph,
            "alias_graph_prefill_latency_reduction_percent":
                (1.0 - graph / eager) * 100.0,
            "alias_graph_decode_speedup": eager_decode / graph_decode,
            "alias_graph_decode_latency_reduction_percent":
                (1.0 - graph_decode / eager_decode) * 100.0,
        }

    route_evidence = route_summary(run_root, plan)
    expected_route_repeats = arm_counts["T"] * (
        int(run_config["conditioning_repeats"])
        + int(run_config["measurement_repeats"])
    )
    require(
        int(route_evidence["trace_file_count"]) == arm_counts["T"],
        "route trace file count differs",
    )
    for prompt_tokens, evidence in route_evidence["by_prompt_tokens"].items():
        require(
            int(evidence["repeat_count"]) == expected_route_repeats,
            f"route repeat count differs for {prompt_tokens}",
        )
        require(
            int(evidence["record_count"]) == 48 * expected_route_repeats,
            f"route record count differs for {prompt_tokens}",
        )

    batch_order_counts: dict[str, dict[str, int]] = {}
    for symbol in ARMS:
        counts = Counter(
            row["batch_order"]
            for row in status_rows
            if row["symbol"] == symbol
        )
        batch_order_counts[symbol] = dict(sorted(counts.items()))

    return {
        "schema": "vllm_uvm_cold5_alias_prefill_verification_v1",
        "status": "ok",
        "run_root": str(run_root.resolve()),
        "case_count": len(results),
        "arms": sorted(seen_arms),
        "run_contract": {
            "mode": run_config["mode"],
            "order": run_config["order"],
            "all_returncodes_zero": True,
            "common_config_equal": True,
            "versions_equal": True,
            "arm_counts": dict(sorted(arm_counts.items())),
            "batch_order_counts": batch_order_counts,
            "measurement_repeats_per_process": measurement_repeats,
        },
        "plan_sha256": sha256_file(plan_path),
        "exact_names_sha256": sha256_file(names_path),
        "output_hash_evidence": hash_evidence,
        "output_hash_comparison_groups": {
            name: sorted(members)
            for name, members in execution_reference_arms.items()
        },
        "performance_comparison_note": (
            "Core Graph-support ablation is alias_graph versus alias_eager. "
            "Formal alias performance arms disable copy trace and per-forward "
            "manifest I/O. Primary legacy comparison is alias_graph versus "
            "all_gpu_graph. "
            "route_trace_control is not a performance baseline because tracing "
            "synchronizes routing data to the CPU."
        ),
        "route_evidence": route_evidence,
        "prefill_performance": performance,
        "decode_performance": decode_performance,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = verify(args.run_root.resolve())
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
