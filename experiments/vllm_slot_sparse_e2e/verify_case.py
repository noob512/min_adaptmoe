#!/usr/bin/env python3
"""Verify one real 48-layer sparse-region registered-plan vLLM run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


LAYERS = tuple(range(48))
EXPERTS = 128
COLD_EXPERTS_PER_LAYER = 12
REGIONS_PER_EXPERT = 9
ENTRIES_PER_LAYER = COLD_EXPERTS_PER_LAYER * REGIONS_PER_EXPERT
EXPERT_BYTES = 4_718_592
LAYER_BYTES = COLD_EXPERTS_PER_LAYER * EXPERT_BYTES
BANK_SLOTS = ENTRIES_PER_LAYER
DUAL_BANK_SLOTS = 2 * BANK_SLOTS
DUAL_BANK_BYTES = 2 * LAYER_BYTES
EXPECTED_RESTORES = 3 * 4 * len(LAYERS)
EXPECTED_AGGREGATE_OUTPUT_SHA256 = (
    "61a02f296763ef269499e451d962b6571db165debf4f7124156eaa333589582e"
)
EXPECTED_REQUEST_OUTPUT_SHA256 = (
    "5e3205fac7ca66dc338eb0af28e428ada4c71f1432a04c1963191295d01a1d94"
)


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
        name, value = raw.split("=", 1)
        require(bool(name) and name not in result, f"{path}:{line_number}: bad key")
        result[name] = value
    return result


def load_stats(path: Path) -> dict[str, int]:
    require(path.is_file(), f"missing stats artifact: {path}")
    result: dict[str, int] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = raw.strip().split()
        if len(fields) != 2:
            continue
        try:
            result[fields[0]] = int(fields[1], 0)
        except ValueError:
            pass
    return result


def delta(before: dict[str, int], after: dict[str, int], name: str) -> int:
    require(name in before and name in after, f"stats lack {name}")
    value = after[name] - before[name]
    require(value >= 0, f"counter moved backwards: {name}")
    return value


def verify_shadow_stats(deltas: dict[str, int], enabled: int) -> None:
    calls = deltas["slot_direct_shadow_pte_prefill_calls"]
    blocks = deltas["slot_direct_shadow_pte_prefill_blocks"]
    if enabled:
        require(calls > 0, "shadow-PTE path was not used")
        require(blocks > 0, "no full block was shadow-prefilled")
    else:
        require(calls == 0, "shadow-PTE calls were observed while disabled")
        require(blocks == 0, "shadow-PTE blocks were observed while disabled")


def verify_plan(path: Path) -> dict[int, tuple[int, ...]]:
    plan = load_json(path)

    def coordinates(records: Any, label: str) -> set[tuple[int, int]]:
        require(isinstance(records, list), f"{label} is not a list")
        values: list[tuple[int, int]] = []
        for index, record in enumerate(records):
            require(isinstance(record, dict), f"{label}[{index}] is not an object")
            layer = record.get("layer_id")
            expert = record.get("expert_id")
            require(
                isinstance(layer, int) and not isinstance(layer, bool),
                f"{label}[{index}] has invalid layer",
            )
            require(
                isinstance(expert, int) and not isinstance(expert, bool),
                f"{label}[{index}] has invalid expert",
            )
            require(layer in LAYERS and 0 <= expert < EXPERTS, f"{label}[{index}] out of range")
            values.append((layer, expert))
        require(len(values) == len(set(values)), f"{label} contains duplicates")
        return set(values)

    cold = coordinates(plan.get("offload_plan"), "offload_plan")
    hot = coordinates(plan.get("hot_plan"), "hot_plan")
    universe = {(layer, expert) for layer in LAYERS for expert in range(EXPERTS)}
    require(not cold & hot, "cold/hot plan overlap")
    require(cold | hot == universe, "cold/hot plan does not cover 48x128")
    selected: dict[int, tuple[int, ...]] = {}
    for layer in LAYERS:
        values = tuple(sorted(expert for candidate, expert in cold if candidate == layer))
        require(len(values) == COLD_EXPERTS_PER_LAYER, f"layer {layer} is not sparse-12")
        selected[layer] = values

    metadata = plan.get("metadata")
    require(isinstance(metadata, dict), "plan metadata is absent")
    require(metadata.get("envelope_bytes") == 524_288, "envelope is not 512 KiB")
    require(metadata.get("regions_per_expert") == REGIONS_PER_EXPERT, "regions/expert mismatch")
    require(metadata.get("cold_regions") == len(LAYERS) * ENTRIES_PER_LAYER, "region total mismatch")
    require(metadata.get("actual_offload_main_bytes") == len(LAYERS) * LAYER_BYTES, "byte total mismatch")
    require(metadata.get("shared_hbm_bank_bytes") == LAYER_BYTES, "single-bank bytes mismatch")
    require(metadata.get("shared_hbm_bank_slots") == BANK_SLOTS, "single-bank slots mismatch")
    return selected


def verify_case(run_dir: Path, *, shadow_pte_prefill: int = 1) -> dict[str, Any]:
    require(shadow_pte_prefill in (0, 1), "shadow_pte_prefill must be 0 or 1")
    run_dir = run_dir.resolve()
    selected = verify_plan(run_dir / "inputs" / "exact_l0_l3_plan.json")
    result = load_json(run_dir / "result.json")
    require(result.get("status") == "ok", "benchmark status is not ok")
    execution = result.get("execution")
    measurement = result.get("measurement")
    require(isinstance(execution, dict) and isinstance(measurement, dict), "result sections absent")
    request_count = result.get("config", {}).get("sequential_requests")
    require(isinstance(request_count, int) and request_count > 0, "invalid request count")
    require(execution.get("generate_call_count") == request_count, "generate count mismatch")
    require(measurement.get("completed_requests") == request_count, "request completion mismatch")
    require(execution.get("single_llm_instance") is True, "LLM instance was not reused")
    require(execution.get("engine_shutdown_explicit") is True, "engine shutdown was not explicit")
    require(
        measurement.get("aggregate_output_token_ids_sha256")
        == EXPECTED_AGGREGATE_OUTPUT_SHA256,
        "aggregate output token hash mismatch",
    )
    requests = measurement.get("requests")
    require(isinstance(requests, list) and len(requests) == request_count, "request records mismatch")
    for index, request in enumerate(requests):
        require(isinstance(request, dict), f"request {index} is not an object")
        require(
            request.get("output_token_ids_sha256") == EXPECTED_REQUEST_OUTPUT_SHA256,
            f"request {index} output token hash mismatch",
        )

    environment = load_env(run_dir / "environment.txt")
    expected_environment = {
        "VLLM_UVM_EXPERT_ENVELOPE_ENABLE": "1",
        "VLLM_UVM_EXPERT_ENVELOPE_BYTES": "524288",
        "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE": "slot_restore",
        "VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD": "1",
        "VLLM_UVM_SLOT_DIRECT_REUSE_BANK": "1",
        "VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES": "1",
        "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION": "1",
        "VLLM_UVM_SLOT_DIRECT_SHARED_PUSH": "1",
        "VLLM_UVM_SLOT_DIRECT_64K_PTE": "1",
        "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN": "1",
        "VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT": "1",
        "VLLM_UVM_SLOT_DIRECT_CUDA_FENCE": "1",
        "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES": str(DUAL_BANK_BYTES),
        "VLLM_UVM_SLOT_DIRECT_MAX_SLOTS": str(DUAL_BANK_SLOTS),
    }
    for name, expected in expected_environment.items():
        require(environment.get(name) == expected, f"environment mismatch: {name}")

    shadow_parameter = "uvm_moe_slot_plan_shadow_pte_prefill"
    for state_name in ("module_state.before.env", "module_state.after.env"):
        module_state = load_env(run_dir / state_name)
        require(
            module_state.get(shadow_parameter) == str(shadow_pte_prefill),
            f"{state_name} shadow-PTE parameter mismatch",
        )

    registration_count = 0
    restore_layers: list[int] = []
    ioctl_ms: list[float] = []
    fence_count = 0
    release_ok = False
    previous_generation: int | None = None
    trace_path = run_dir / "slot_trace.jsonl"
    require(trace_path.is_file(), "slot trace is absent")
    for line_number, raw in enumerate(trace_path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        record = json.loads(raw)
        require(isinstance(record, dict), f"trace line {line_number} is not an object")
        operation = record.get("operation")
        if operation == "register_layer_plans":
            registration_count += 1
            require(record.get("ok") is True, "registered-plan setup failed")
            require(record.get("layer_count") == len(LAYERS), "registered layer count mismatch")
            require(record.get("lane_count") == ENTRIES_PER_LAYER, "lane count mismatch")
            require(record.get("bank_count") == 2 and record.get("dual_bank_event") is True, "not dual bank")
            expected_bytes = {str(layer): LAYER_BYTES for layer in LAYERS}
            require(record.get("requested_bytes_per_layer") == expected_bytes, "registered bytes mismatch")
        elif operation == "restore_layer":
            require(record.get("ok") is True, f"restore failed at trace line {line_number}")
            require(record.get("registered_plan_used") is True, "restore bypassed registered plan")
            require(record.get("dual_bank_event") is True, "restore bypassed dual bank")
            layer = record.get("layer_id")
            require(layer in LAYERS, f"unexpected restore layer: {layer!r}")
            expected_ids = list(selected[int(layer)])
            if "expert_ids" in record:
                require(record["expert_ids"] == expected_ids, f"layer {layer} expert set mismatch")
            restore_layers.append(int(layer))
            low = record.get("restore")
            require(isinstance(low, dict) and low.get("ok") is True, "registered ioctl failed")
            for field in ("cuda_fence_used", "cuda_fence_accepted", "cuda_completion_signaled", "cuda_stream_wait_enqueued"):
                require(low.get(field) is True, f"fence field is false: {field}")
            fence_count += 1
            require(low.get("plan_execute") is True, "compact plan execute was not used")
            require(low.get("requested_bytes") == LAYER_BYTES, "requested bytes mismatch")
            require(low.get("accepted_bytes") == LAYER_BYTES, "accepted bytes mismatch")
            require(low.get("success_count") == ENTRIES_PER_LAYER, "success count mismatch")
            require(low.get("remap_count") == ENTRIES_PER_LAYER, "remap count mismatch")
            require(low.get("failure_count") == 0 and low.get("conflict_count") == 0, "restore conflict/failure")
            require(low.get("poisoned") is False, "session was poisoned")
            generation = low.get("generation")
            new_generation = low.get("new_generation")
            require(isinstance(generation, int) and new_generation == generation + 1, "generation mismatch")
            if previous_generation is not None:
                require(generation == previous_generation, "generations are not contiguous")
            previous_generation = new_generation
            wall_ns = low.get("ioctl_wall_ns")
            require(isinstance(wall_ns, int) and wall_ns > 0, "ioctl timing absent")
            ioctl_ms.append(wall_ns / 1_000_000)
        elif operation == "release" and record.get("ok") is True:
            release_ok = True

    require(registration_count == 1, "expected one registered-plan setup")
    require(len(restore_layers) == EXPECTED_RESTORES, "restore count is not 3x4x48")
    for offset in range(0, len(restore_layers), len(LAYERS)):
        require(tuple(restore_layers[offset : offset + len(LAYERS)]) == LAYERS, "restore order mismatch")
    require(release_ok, "session release did not succeed")

    before = load_stats(run_dir / "latency_stats.before.txt")
    after = load_stats(run_dir / "latency_stats.after.txt")
    names = (
        "slot_plan_register_calls",
        "slot_plan_execute_calls",
        "slot_direct_successes",
        "slot_direct_errors",
        "slot_direct_pte_4k_blocks",
        "slot_direct_pte_64k_blocks",
        "slot_direct_partial_64k_blocks",
        "slot_direct_pte_2m_blocks",
        "slot_direct_dual_bank_copy_first_calls",
        "slot_direct_shadow_pte_prefill_calls",
        "slot_direct_shadow_pte_prefill_blocks",
        "slot_direct_copy_run_cache_hits",
        "slot_direct_metadata_recipe_cache_attempts",
        "slot_direct_target_recipe_cache_hits",
        "slot_direct_clear_recipe_cache_hits",
    )
    deltas = {name: delta(before, after, name) for name in names}
    restores = len(restore_layers)
    require(deltas["slot_plan_register_calls"] == len(LAYERS) + 1, "plan registration count mismatch")
    require(deltas["slot_plan_execute_calls"] == restores, "plan execute count mismatch")
    require(deltas["slot_direct_successes"] == restores and deltas["slot_direct_errors"] == 0, "kernel error count")
    require(deltas["slot_direct_pte_4k_blocks"] == 0, "4 KiB PTE fallback was used")
    require(deltas["slot_direct_pte_2m_blocks"] == 0, "2 MiB leaf mode was used")
    require(deltas["slot_direct_pte_64k_blocks"] > 0, "64 KiB PTE path was not used")
    require(deltas["slot_direct_partial_64k_blocks"] > 0, "partial-64K path was not used")
    verify_shadow_stats(deltas, shadow_pte_prefill)
    require(deltas["slot_direct_dual_bank_copy_first_calls"] == restores - 1, "dual-bank transitions mismatch")
    require(deltas["slot_direct_copy_run_cache_hits"] == restores, "copy-run cache missed")
    require(deltas["slot_direct_metadata_recipe_cache_attempts"] == restores, "metadata recipe attempts mismatch")
    require(deltas["slot_direct_target_recipe_cache_hits"] == restores, "target recipe cache missed")
    require(deltas["slot_direct_clear_recipe_cache_hits"] == max(restores - 2, 0), "clear recipe cache mismatch")

    latency = measurement.get("request_latency_ms", {})
    median = latency.get("median", latency.get("p50"))
    require(isinstance(median, (int, float)) and math.isfinite(float(median)), "request median absent")
    return {
        "schema": "vllm_sparse_region_e2e_verification_v1",
        "status": "pass",
        "run_dir": str(run_dir),
        "layers": len(LAYERS),
        "cold_experts_per_layer": COLD_EXPERTS_PER_LAYER,
        "regions_per_layer": ENTRIES_PER_LAYER,
        "bytes_per_layer": LAYER_BYTES,
        "single_bank_bytes": LAYER_BYTES,
        "dual_bank_bytes": DUAL_BANK_BYTES,
        "dual_bank_slots": DUAL_BANK_SLOTS,
        "shadow_pte_prefill": shadow_pte_prefill,
        "restore_cycles": restores // len(LAYERS),
        "restore_count": restores,
        "fence_count": fence_count,
        "ioctl_wall_ms": {
            "count": len(ioctl_ms),
            "median": statistics.median(ioctl_ms),
            "minimum": min(ioctl_ms),
            "maximum": max(ioctl_ms),
        },
        "request_latency_ms_median": float(median),
        "output_token_ids_sha256": measurement.get("aggregate_output_token_ids_sha256"),
        "latency_stats_delta": deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--shadow-pte-prefill", type=int, choices=(0, 1), default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        summary = verify_case(
            args.run_dir,
            shadow_pte_prefill=args.shadow_pte_prefill,
        )
    except (OSError, ValueError, json.JSONDecodeError, VerificationError) as exc:
        print(f"verification failed: {exc}")
        return 1
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
