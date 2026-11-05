#!/usr/bin/env python3
"""Verify and summarize the footprint-matched vLLM prefill three-arm run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


LAYERS = tuple(range(4))
EXPERTS = 128
REGIONS_PER_EXPERT = 9
LAYER_BYTES = 603_979_776
TOTAL_BYTES = len(LAYERS) * LAYER_BYTES
REGIONS_PER_LAYER = EXPERTS * REGIONS_PER_EXPERT
TOTAL_REGIONS = len(LAYERS) * REGIONS_PER_LAYER
EXPECTED_NAMES = tuple(
    f"model.layers.{layer}.mlp.experts.{weight}"
    for layer in LAYERS
    for weight in ("w13_weight", "w2_weight")
)
ARM_NAMES = {"P": "p3", "C": "pinned", "U": "uva"}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def load_env(path: Path) -> dict[str, str]:
    require(path.is_file(), f"missing environment artifact: {path}")
    result: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw:
            continue
        require("=" in raw, f"{path}:{line_number}: malformed line")
        key, value = raw.split("=", 1)
        require(bool(key) and key not in result, f"{path}:{line_number}: bad key")
        result[key] = value
    return result


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL artifact: {path}")
    records: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        require(isinstance(value, dict), f"{path}:{line_number}: not an object")
        records.append(value)
    return records


def finite_number(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} is not finite",
    )
    return float(value)


def verify_plan(run_root: Path) -> None:
    plan = load_json(run_root / "inputs" / "exact_l0_l3_plan.json")
    cold = {
        (item.get("layer_id"), item.get("expert_id"))
        for item in plan.get("offload_plan", [])
        if isinstance(item, dict)
    }
    expected_cold = {(layer, expert) for layer in LAYERS for expert in range(EXPERTS)}
    universe = {(layer, expert) for layer in range(48) for expert in range(EXPERTS)}
    hot = {
        (item.get("layer_id"), item.get("expert_id"))
        for item in plan.get("hot_plan", [])
        if isinstance(item, dict)
    }
    require(cold == expected_cold, "plan is not exactly L0-L3/all experts")
    require(not cold & hot and cold | hot == universe, "cold/hot plan partition differs")
    metadata = plan.get("metadata")
    require(isinstance(metadata, dict), "plan metadata is absent")
    require(metadata.get("actual_offload_main_bytes") == TOTAL_BYTES, "plan byte total differs")
    require(metadata.get("cold_regions") == TOTAL_REGIONS, "plan region total differs")
    require(metadata.get("shared_hbm_bank_bytes") == LAYER_BYTES, "plan bank bytes differ")
    names = tuple(
        line.strip()
        for line in (run_root / "inputs" / "exact_parameter_names.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    require(names == EXPECTED_NAMES, "pinned exact-name list differs")


def verify_result(
    path: Path,
    *,
    expected_concurrency: int,
    expected_warmup_requests: int,
    expected_disable_chunked_prefill: bool,
) -> tuple[dict[str, Any], float, float, float, str, int]:
    result = load_json(path)
    require(result.get("status") == "ok", f"{path}: status is not ok")
    config = result.get("config")
    measurement = result.get("measurement")
    require(isinstance(config, dict) and isinstance(measurement, dict), f"{path}: sections absent")
    require(
        config.get("concurrency") == expected_concurrency,
        f"{path}: concurrency differs",
    )
    require(config.get("output_tokens") == 1, f"{path}: not prefill-only output=1")
    require(
        config.get("warmup_requests") == expected_warmup_requests,
        f"{path}: warmup request count differs",
    )
    require(
        bool(config.get("disable_chunked_prefill"))
        == expected_disable_chunked_prefill,
        f"{path}: chunked-prefill setting differs",
    )
    require(config.get("require_prompt_token_ids") is True, f"{path}: prompt IDs not required")
    require(
        measurement.get("completed_requests") == expected_concurrency,
        f"{path}: completion count differs",
    )
    require(
        measurement.get("output_tokens") == expected_concurrency,
        f"{path}: output count differs",
    )
    ttft = measurement.get("ttft_ms")
    require(isinstance(ttft, dict), f"{path}: TTFT summary absent")
    ttft_ms = finite_number(ttft.get("p50"), f"{path}: TTFT p50")
    ttft_mean_ms = finite_number(ttft.get("mean"), f"{path}: TTFT mean")
    require(ttft_ms > 0.0, f"{path}: TTFT is not positive")
    elapsed_ms = finite_number(
        measurement.get("elapsed_seconds"), f"{path}: elapsed seconds"
    ) * 1000.0
    require(elapsed_ms > 0.0, f"{path}: elapsed is not positive")
    output_hash = measurement.get("aggregate_output_token_ids_sha256")
    require(isinstance(output_hash, str) and len(output_hash) == 64, f"{path}: output hash absent")
    prompt_tokens = measurement.get("prompt_tokens")
    require(isinstance(prompt_tokens, int) and prompt_tokens > 0, f"{path}: prompt tokens absent")
    return result, ttft_ms, ttft_mean_ms, elapsed_ms, output_hash, prompt_tokens


def verify_pinned(case_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    require(env.get("VLLM_CPU_OFFLOAD_BACKEND") == "copy", "pinned backend differs")
    require(env.get("VLLM_CPU_OFFLOAD_STATIC_BANK") == "0", "pinned arm uses static bank")
    require(env.get("VLLM_CPU_OFFLOAD_COPY_SYNC") == "0", "pinned arm forced host sync")
    require(env.get("VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE") == "0", "pinned arm enabled Stage M")
    records = load_jsonl(case_dir / "cpu_offload_manifest.jsonl")
    parameters = [record for record in records if record.get("event") == "cpu_offload_parameter"]
    require(len(parameters) == len(EXPECTED_NAMES), "pinned parameter count differs")
    require(
        tuple(record.get("full_parameter_name") for record in parameters) == EXPECTED_NAMES,
        "pinned parameter order/selection differs",
    )
    require(all(record.get("cpu_data_pinned") is True for record in parameters), "source is not pinned")
    require(all(record.get("copy_on_forward") is True for record in parameters), "not copy-on-forward")
    copied_bytes = sum(int(record.get("bytes", -1)) for record in parameters)
    require(copied_bytes == TOTAL_BYTES, "pinned allocation bytes differ")
    return {"selected_parameters": len(parameters), "selected_bytes": copied_bytes}


def verify_uva(case_dir: Path, env: dict[str, str]) -> dict[str, Any]:
    require(env.get("VLLM_CPU_OFFLOAD_BACKEND") == "uva", "UVA backend differs")
    require(env.get("VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE") == "remote_read", "UVA is not remote-read")
    require(env.get("VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE") == "1", "UVA Stage M disabled")
    placement = load_json(case_dir / "runtime_remote_placement_manifest.json")
    require(placement.get("status") == "ok" and placement.get("complete") is True, "UVA placement failed")
    require(placement.get("remote_expert_count") == len(LAYERS) * EXPERTS, "UVA expert count differs")
    require(placement.get("remote_region_count") == TOTAL_REGIONS, "UVA region count differs")
    require(placement.get("remote_region_bytes") == TOTAL_BYTES, "UVA remote bytes differ")
    require(placement.get("offloaded_expert_count") == 0, "UVA contains restore-mode experts")
    require(placement.get("release_time_remote_mapping_verified") is True, "UVA remote mapping unverified")
    return {
        "remote_experts": placement["remote_expert_count"],
        "remote_regions": placement["remote_region_count"],
        "remote_bytes": placement["remote_region_bytes"],
    }


def verify_p3(
    case_dir: Path,
    env: dict[str, str],
    result: dict[str, Any],
    *,
    warmup_requests: int,
) -> dict[str, Any]:
    expected = {
        "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE": "slot_restore",
        "VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD": "1",
        "VLLM_UVM_SLOT_DIRECT_INIT_BANK": "1",
        "VLLM_UVM_SLOT_DIRECT_REUSE_BANK": "1",
        "VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES": "1",
        "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION": "1",
        "VLLM_UVM_SLOT_DIRECT_SHARED_PUSH": "1",
        "VLLM_UVM_SLOT_DIRECT_64K_PTE": "1",
        "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN": "1",
        "VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT": "1",
        "VLLM_UVM_SLOT_DIRECT_CUDA_FENCE": "1",
        "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES": str(2 * LAYER_BYTES),
        "VLLM_UVM_SLOT_DIRECT_MAX_SLOTS": str(2 * REGIONS_PER_LAYER),
    }
    for key, value in expected.items():
        require(env.get(key) == value, f"P3 environment mismatch: {key}")
    trace = load_jsonl(case_dir / "slot_trace.jsonl")
    registrations = [record for record in trace if record.get("operation") == "register_layer_plans"]
    restores = [record for record in trace if record.get("operation") == "restore_layer"]
    require(len(registrations) == 1 and registrations[0].get("ok") is True, "P3 registration failed")
    cycle_count = 1 + int(warmup_requests > 0)
    expected_layers = LAYERS * cycle_count
    require(
        len(restores) == len(expected_layers),
        "P3 prefill layer-cycle count differs",
    )
    all_ioctl_ms: list[float] = []
    for expected_layer, record in zip(expected_layers, restores):
        require(record.get("ok") is True and record.get("layer_id") == expected_layer, "P3 layer order/failure")
        low = record.get("restore")
        require(isinstance(low, dict) and low.get("ok") is True, "P3 ioctl failed")
        require(low.get("plan_execute") is True, "P3 bypassed registered plan")
        require(low.get("requested_bytes") == LAYER_BYTES, "P3 layer byte count differs")
        require(low.get("accepted_bytes") == LAYER_BYTES, "P3 accepted byte count differs")
        require(low.get("cuda_fence_accepted") is True, "P3 compute fence not accepted")
        require(low.get("cuda_completion_signaled") is True, "P3 completion fence not signaled")
        require(low.get("cuda_stream_wait_enqueued") is True, "P3 stream wait absent")
        require(low.get("success_contract_ok") is True, "P3 success contract failed")
        require(low.get("success_count") == REGIONS_PER_LAYER, "P3 success count differs")
        require(low.get("failure_count") == 0, "P3 contains failed entries")
        require(low.get("copy_push_count") == 1, "P3 did not use one shared copy push")
        require(
            low.get("tlb_invalidate_count") == LAYER_BYTES // (2 * 1024 * 1024),
            "P3 TLB invalidate count differs",
        )
        require(low.get("poisoned") is False, "P3 userspace session is poisoned")
        require(low.get("kernel_poisoned") is False, "P3 kernel session is poisoned")
        wall_ns = low.get("ioctl_wall_ns")
        require(isinstance(wall_ns, int) and wall_ns > 0, "P3 ioctl timing absent")
        all_ioctl_ms.append(wall_ns / 1_000_000.0)

    measurement_start = finite_number(
        result.get("measurement_start_epoch"), "P3 measurement start"
    )
    measurement_end = finite_number(
        result.get("measurement_end_epoch"), "P3 measurement end"
    )
    measured_restores = [
        record
        for record in restores
        if measurement_start
        <= finite_number(record.get("timestamp"), "P3 restore timestamp")
        <= measurement_end
    ]
    require(
        [record.get("layer_id") for record in measured_restores] == list(LAYERS),
        "P3 measurement window does not contain exactly L0-L3",
    )
    measured_ioctl_ms = [
        int(record["restore"]["ioctl_wall_ns"]) / 1_000_000.0
        for record in measured_restores
    ]
    return {
        "transitions": len(restores),
        "warmup_transitions": len(restores) - len(measured_restores),
        "measurement_transitions": len(measured_restores),
        "transition_bytes": len(measured_restores) * LAYER_BYTES,
        "ioctl_wall_ms_median": statistics.median(measured_ioctl_ms),
        "all_ioctl_wall_ms_median": statistics.median(all_ioctl_ms),
    }


def comparison(candidate: float, baseline: float) -> dict[str, float]:
    return {
        "delta_ms": candidate - baseline,
        "ratio": candidate / baseline,
        "percent_slower": (candidate / baseline - 1.0) * 100.0,
    }


def verify(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    verify_plan(run_root)
    config = load_json(run_root / "run_config.json")
    order = config.get("order")
    require(isinstance(order, str) and order, "run order absent")
    require(set(order) == set(ARM_NAMES), "run order must contain all three arms")
    concurrency = int(config.get("concurrency", 0))
    warmup_requests = int(config.get("warmup_requests", -1))
    disable_chunked_prefill = bool(config.get("disable_chunked_prefill", False))
    require(concurrency > 0, "run concurrency is invalid")
    require(warmup_requests >= 0, "run warmup request count is invalid")

    status_path = run_root / "execution_status.tsv"
    require(status_path.is_file(), "execution status is absent")
    with status_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(len(rows) == len(order), "execution row count differs")
    directories = [str(row.get("directory")) for row in rows]
    require(len(set(directories)) == len(directories), "execution directories are not unique")

    samples: dict[str, list[float]] = defaultdict(list)
    elapsed_samples: dict[str, list[float]] = defaultdict(list)
    output_hashes: set[str] = set()
    prompt_counts: set[int] = set()
    details: list[dict[str, Any]] = []
    for index, (symbol, row) in enumerate(zip(order, rows), 1):
        arm = ARM_NAMES[symbol]
        require(int(row.get("ordinal", -1)) == index, f"row {index}: ordinal differs")
        require(row.get("arm") == arm, f"row {index}: arm differs")
        require(int(row.get("returncode", -1)) == 0, f"row {index}: arm failed")
        case_dir = run_root / str(row.get("directory"))
        require(case_dir.is_dir(), f"row {index}: case directory is absent")
        returncode_path = case_dir / "returncode.txt"
        require(returncode_path.is_file(), f"row {index}: returncode file is absent")
        require(int(returncode_path.read_text().strip()) == 0, f"row {index}: returncode file failed")
        execution = load_env(case_dir / "execution.env")
        require(execution.get("ordinal") == str(index), f"row {index}: execution ordinal differs")
        require(execution.get("symbol") == symbol, f"row {index}: execution symbol differs")
        require(execution.get("arm") == arm, f"row {index}: execution arm differs")
        require(execution.get("returncode") == "0", f"row {index}: execution failed")
        (
            result,
            ttft_ms,
            ttft_mean_ms,
            elapsed_ms,
            output_hash,
            prompt_tokens,
        ) = verify_result(
            case_dir / "result.json",
            expected_concurrency=concurrency,
            expected_warmup_requests=warmup_requests,
            expected_disable_chunked_prefill=disable_chunked_prefill,
        )
        env = load_env(case_dir / "environment.txt")
        evidence: dict[str, Any]
        if arm == "p3":
            evidence = verify_p3(
                case_dir,
                env,
                result,
                warmup_requests=warmup_requests,
            )
        elif arm == "pinned":
            evidence = verify_pinned(case_dir, env)
        else:
            evidence = verify_uva(case_dir, env)
        require(result.get("mechanism_class") == config["mechanism_classes"][arm], f"{arm}: class differs")
        samples[arm].append(ttft_ms)
        elapsed_samples[arm].append(elapsed_ms)
        output_hashes.add(output_hash)
        prompt_counts.add(prompt_tokens)
        details.append(
            {
                "ordinal": index,
                "arm": arm,
                "directory": str(case_dir),
                "ttft_ms": ttft_ms,
                "ttft_mean_ms": ttft_mean_ms,
                "batch_elapsed_ms": elapsed_ms,
                "output_token_ids_sha256": output_hash,
                "evidence": evidence,
            }
        )

    require(len(output_hashes) == 1, "output tokens differ across arms")
    require(len(prompt_counts) == 1, "prompt token count differs across arms")
    require(all(samples[name] for name in ARM_NAMES.values()), "an arm has no samples")
    sample_counts = {len(samples[name]) for name in ARM_NAMES.values()}
    require(len(sample_counts) == 1, "arm sample counts differ")
    medians = {name: statistics.median(values) for name, values in samples.items()}
    elapsed_medians = {
        name: statistics.median(values) for name, values in elapsed_samples.items()
    }
    return {
        "schema": "vllm_prefill_p3_pinned_uva_compare_v1",
        "status": "pass",
        "run_root": str(run_root),
        "order": order,
        "offloaded_weight_contract": {
            "layers": list(LAYERS),
            "experts_per_layer": EXPERTS,
            "regions_per_layer": REGIONS_PER_LAYER,
            "bytes_per_layer": LAYER_BYTES,
            "total_bytes": TOTAL_BYTES,
            "total_gib": TOTAL_BYTES / 1024**3,
            "traffic_note": (
                "P3/pinned copy the full selection; UVA remote-read traffic "
                "is demand-dependent"
            ),
        },
        "measurement_contract": {
            "concurrency": concurrency,
            "output_tokens": 1,
            "warmup_requests": warmup_requests,
            "disable_chunked_prefill": disable_chunked_prefill,
            "metric": "ttft_ms",
            "prompt_tokens": next(iter(prompt_counts)),
        },
        "ttft_ms": {
            name: {"samples": values, "median": medians[name]}
            for name, values in samples.items()
        },
        "batch_elapsed_ms": {
            name: {"samples": values, "median": elapsed_medians[name]}
            for name, values in elapsed_samples.items()
        },
        "comparison": {
            "p3_vs_pinned": comparison(medians["p3"], medians["pinned"]),
            "p3_vs_uva": comparison(medians["p3"], medians["uva"]),
            "pinned_vs_uva": comparison(medians["pinned"], medians["uva"]),
        },
        "batch_elapsed_comparison": {
            "p3_vs_pinned": comparison(
                elapsed_medians["p3"], elapsed_medians["pinned"]
            ),
            "p3_vs_uva": comparison(
                elapsed_medians["p3"], elapsed_medians["uva"]
            ),
            "pinned_vs_uva": comparison(
                elapsed_medians["pinned"], elapsed_medians["uva"]
            ),
        },
        "output_token_ids_sha256": next(iter(output_hashes)),
        "cases": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        summary = verify(args.run_root)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, VerificationError) as exc:
        print(f"verification failed: {exc}")
        return 1
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
