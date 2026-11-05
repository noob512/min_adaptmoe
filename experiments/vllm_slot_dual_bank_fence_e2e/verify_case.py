#!/usr/bin/env python3
"""Verify one real vLLM dual-bank registered-plan fence case."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


CACHE_BYTES = 1_207_959_552
MAX_SLOTS = 2_304
LAYERS = (0, 1, 2, 3)
LAYER_BYTES = 603_979_776
BLOCK_BYTES = 2 * 1024 * 1024
BLOCKS_PER_LAYER = LAYER_BYTES // BLOCK_BYTES
ENTRIES_PER_LAYER = 1_152
SEQUENTIAL_REQUESTS = 3
REPO_ROOT = Path(__file__).resolve().parents[2]


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def load_environment(path: Path) -> dict[str, str]:
    require(path.is_file(), f"missing environment artifact: {path}")
    result: dict[str, str] = {}
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw:
            continue
        require("=" in raw, f"{path}:{line_number}: malformed environment line")
        name, value = raw.split("=", 1)
        require(bool(name), f"{path}:{line_number}: empty environment name")
        require(name not in result, f"{path}:{line_number}: duplicate {name}")
        result[name] = value
    return result


def load_stats(path: Path) -> dict[str, int]:
    require(path.is_file(), f"missing latency-stats artifact: {path}")
    result: dict[str, int] = {}
    section: str | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.endswith(":"):
            section = stripped[:-1]
            continue
        if raw[:1] and not raw[:1].isspace():
            section = None

        fields = stripped.split()
        if section == "slot_direct_profile" and raw[:1].isspace() and fields:
            stage = fields[0]
            parsed_profile = False
            for field in fields[1:]:
                if "=" not in field:
                    continue
                name, value = field.split("=", 1)
                try:
                    result[f"{section}.{stage}.{name}"] = int(value, 0)
                except ValueError:
                    continue
                parsed_profile = True
            if parsed_profile:
                continue

        if len(fields) != 2:
            continue
        try:
            result[fields[0]] = int(fields[1], 0)
        except ValueError:
            continue
    return result


def stats_delta(before: dict[str, int], after: dict[str, int], name: str) -> int:
    require(name in before, f"latency stats before lacks {name}")
    require(name in after, f"latency stats after lacks {name}")
    value = after[name] - before[name]
    require(value >= 0, f"latency counter {name} moved backwards")
    return value


def load_optional_run_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return load_json(path)


def verify_feature_contract(
    run_dir: Path,
    *,
    run_config: dict[str, Any],
    fence: int,
    metadata_recipe: int,
    fast_validation: int,
    shadow_pte_prefill: int,
) -> bool:
    feature_contract = (
        "fast_validation" in run_config or "shadow_pte_prefill" in run_config
    )
    if not feature_contract:
        require(
            fast_validation == 0 and shadow_pte_prefill == 0,
            "artifact predates fast/shadow telemetry and cannot prove it",
        )
        return False

    require(
        run_config.get("fence") == fence,
        "run_config fence differs from requested verification",
    )
    require(
        run_config.get("metadata_recipe") == metadata_recipe,
        "run_config metadata recipe differs from requested verification",
    )
    require(
        run_config.get("fast_validation") == fast_validation,
        "run_config fast-validation mode differs from requested verification",
    )
    require(
        run_config.get("shadow_pte_prefill") == shadow_pte_prefill,
        "run_config shadow-PTE mode differs from requested verification",
    )
    overlap_copy_prepare = run_config.get("overlap_copy_prepare")
    require(
        overlap_copy_prepare in (0, 1),
        "run_config has invalid overlap-copy-prepare mode",
    )
    if shadow_pte_prefill:
        require(
            overlap_copy_prepare == 0,
            "shadow-PTE case did not disable legacy overlap",
        )

    expected_parameters = {
        "uvm_moe_slot_pte_remap_enable": "1",
        "uvm_moe_slot_plan_overlap_copy_prepare": str(overlap_copy_prepare),
        "uvm_moe_slot_plan_register_pte_prewarm": "1",
        "uvm_moe_slot_plan_cache_copy_runs": "1",
        "uvm_moe_slot_transition_fast_index": "1",
        "uvm_moe_slot_transition_direct_cutover": "1",
        "uvm_moe_slot_plan_fast_validation": str(fast_validation),
        "uvm_moe_slot_plan_shadow_pte_prefill": str(shadow_pte_prefill),
        "uvm_moe_slot_plan_cache_metadata_recipe": str(metadata_recipe),
        "uvm_moe_slot_plan_dual_bank_event": "1",
        "uvm_moe_slot_plan_cuda_fence": str(fence),
    }
    states: list[dict[str, str]] = []
    for name in ("before", "after"):
        state = load_environment(run_dir / f"module_state.{name}.env")
        states.append(state)
        for parameter, expected in expected_parameters.items():
            require(
                state.get(parameter) == expected,
                f"module {name} mismatch: {parameter}",
            )
        try:
            budget = int(state["uvm_moe_slot_pool_vidmem_budget_bytes"], 0)
        except (KeyError, ValueError) as exc:
            raise VerificationError(
                f"module {name} has invalid slot-pool budget"
            ) from exc
        require(budget >= CACHE_BYTES, f"module {name} slot-pool budget is too small")
    for parameter in expected_parameters:
        require(
            states[0][parameter] == states[1][parameter],
            f"module parameter changed during run: {parameter}",
        )
    require(
        states[0].get("loaded_srcversion")
        == states[0].get("custom_srcversion")
        == states[1].get("loaded_srcversion")
        == states[1].get("custom_srcversion"),
        "loaded/custom module srcversion changed or mismatched",
    )
    return True


def verify_plan(path: Path) -> str:
    plan = load_json(path)
    cold = plan.get("offload_plan")
    hot = plan.get("hot_plan")
    require(isinstance(cold, list), "plan offload_plan is not a list")
    require(isinstance(hot, list), "plan hot_plan is not a list")

    def coordinates(records: list[Any], label: str) -> set[tuple[int, int]]:
        values: list[tuple[int, int]] = []
        for index, record in enumerate(records):
            require(isinstance(record, dict), f"{label}[{index}] is not an object")
            layer = record.get("layer_id")
            expert = record.get("expert_id")
            require(
                isinstance(layer, int) and not isinstance(layer, bool),
                f"{label}[{index}] has invalid layer_id",
            )
            require(
                isinstance(expert, int) and not isinstance(expert, bool),
                f"{label}[{index}] has invalid expert_id",
            )
            values.append((layer, expert))
        require(len(values) == len(set(values)), f"{label} contains duplicates")
        return set(values)

    cold_keys = coordinates(cold, "offload_plan")
    hot_keys = coordinates(hot, "hot_plan")
    expected_cold = {(layer, expert) for layer in LAYERS for expert in range(128)}
    expected_all = {
        (layer, expert) for layer in range(48) for expert in range(128)
    }
    require(cold_keys == expected_cold, "plan is not exact Qwen L0-L3 all-expert")
    require(not cold_keys & hot_keys, "plan cold/hot sets overlap")
    require(cold_keys | hot_keys == expected_all, "plan does not cover Qwen 48x128")
    metadata = plan.get("metadata")
    require(isinstance(metadata, dict), "plan metadata is absent")
    require(
        metadata.get("actual_offload_main_bytes") == 4 * LAYER_BYTES,
        "plan byte total is not four 576 MiB layers",
    )

    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_case(
    run_dir: Path,
    *,
    fence: int,
    metadata_recipe: int,
    fast_validation: int = 0,
    shadow_pte_prefill: int = 0,
) -> dict[str, Any]:
    require(fence in (0, 1), "fence must be 0 or 1")
    require(metadata_recipe in (0, 1), "metadata_recipe must be 0 or 1")
    require(fast_validation in (0, 1), "fast_validation must be 0 or 1")
    require(shadow_pte_prefill in (0, 1), "shadow_pte_prefill must be 0 or 1")
    run_dir = run_dir.resolve()
    result = load_json(run_dir / "result.json")
    environment = load_environment(run_dir / "environment.txt")
    run_config = load_optional_run_config(run_dir / "run_config.json")
    feature_contract = verify_feature_contract(
        run_dir,
        run_config=run_config,
        fence=fence,
        metadata_recipe=metadata_recipe,
        fast_validation=fast_validation,
        shadow_pte_prefill=shadow_pte_prefill,
    )
    plan_sha256 = verify_plan(run_dir / "inputs" / "exact_l0_l3_plan.json")

    require(result.get("status") == "ok", "benchmark status is not ok")
    config = result.get("config")
    execution = result.get("execution")
    measurement = result.get("measurement")
    require(isinstance(config, dict), "benchmark config is absent")
    require(isinstance(execution, dict), "benchmark execution is absent")
    require(isinstance(measurement, dict), "benchmark measurement is absent")
    require(
        config.get("sequential_requests") == SEQUENTIAL_REQUESTS,
        "benchmark did not request exactly three sequential calls",
    )
    require(
        execution.get("generate_call_count") == SEQUENTIAL_REQUESTS,
        "benchmark did not complete three generate calls",
    )
    require(
        measurement.get("completed_requests") == SEQUENTIAL_REQUESTS,
        "benchmark did not complete three requests",
    )
    require(execution.get("single_llm_instance") is True, "LLM instance was not reused")
    require(execution.get("engine_shutdown_explicit") is True, "engine shutdown was not explicit")

    expected_environment = {
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
        "VLLM_UVM_SLOT_DIRECT_CUDA_FENCE": str(fence),
        "VLLM_UVM_SLOT_DIRECT_CUDA_FENCE_KERNEL_LIB": str(
            (
                REPO_ROOT
                / "workloads/vllm/vllm/uvm_test/"
                "libuvm_slot_cuda_fence_kernels.so"
            ).resolve()
        ),
        "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES": str(CACHE_BYTES),
        "VLLM_UVM_SLOT_DIRECT_MAX_SLOTS": str(MAX_SLOTS),
    }
    for name, expected in expected_environment.items():
        require(environment.get(name) == expected, f"environment mismatch: {name}")

    trace_path = run_dir / "slot_trace.jsonl"
    require(trace_path.is_file(), f"missing slot trace: {trace_path}")
    registration_count = 0
    restore_count = 0
    restore_layers: list[int] = []
    ioctl_wall_ms: list[float] = []
    fence_used = 0
    fence_accepted = 0
    completion_signaled = 0
    release_ok = False
    previous_generation: int | None = None
    with trace_path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise VerificationError(
                    f"{trace_path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            require(isinstance(record, dict), f"trace line {line_number} is not an object")
            operation = record.get("operation")
            if operation == "register_layer_plans":
                registration_count += 1
                require(record.get("ok") is True, "registered-plan setup failed")
                require(record.get("layer_count") == 4, "registered plan is not L0-L3")
                require(record.get("bank_count") == 2, "registered plan is not dual-bank")
                require(record.get("dual_bank_event") is True, "dual-bank event is absent")
                requested = record.get("requested_bytes_per_layer")
                require(isinstance(requested, dict), "registered plan lacks byte map")
                require(
                    requested == {str(layer): LAYER_BYTES for layer in LAYERS},
                    "registered plan does not contain four 576 MiB layers",
                )
            elif operation == "restore_layer":
                restore_count += 1
                require(record.get("ok") is True, f"restore {restore_count} failed")
                require(record.get("registered_plan_used") is True, "restore bypassed plan")
                require(record.get("dual_bank_event") is True, "restore bypassed dual bank")
                layer = record.get("layer_id")
                require(layer in LAYERS, f"restore has unexpected layer {layer!r}")
                restore_layers.append(int(layer))
                low_level = record.get("restore")
                require(isinstance(low_level, dict), "restore lacks ioctl result")
                require(low_level.get("ok") is True, "registered-plan ioctl failed")
                used = low_level.get("cuda_fence_used") is True
                accepted = low_level.get("cuda_fence_accepted") is True
                signaled = low_level.get("cuda_completion_signaled") is True
                fence_used += int(used)
                fence_accepted += int(accepted)
                completion_signaled += int(signaled)
                require(used == bool(fence), "trace fence mode differs from requested case")
                require(
                    accepted == bool(fence),
                    "compute fence was not accepted exactly when enabled",
                )
                require(
                    signaled == bool(fence),
                    "completion fence was not signaled exactly when enabled",
                )
                if feature_contract:
                    require(
                        low_level.get("plan_execute") is True,
                        "restore did not use compact registered-plan execute",
                    )
                    require(
                        low_level.get("cuda_stream_wait_enqueued") == bool(fence),
                        "completion wait was not enqueued exactly when fenced",
                    )
                    require(
                        low_level.get("requested_bytes") == LAYER_BYTES
                        and low_level.get("accepted_bytes") == LAYER_BYTES,
                        "restore did not complete one full 576 MiB layer",
                    )
                    require(
                        low_level.get("success_count") == ENTRIES_PER_LAYER
                        and low_level.get("remap_count") == ENTRIES_PER_LAYER,
                        "restore did not publish all 1152 ranges",
                    )
                    require(
                        low_level.get("failure_count") == 0
                        and low_level.get("conflict_count") == 0,
                        "restore reported a failure or conflict",
                    )
                    require(
                        low_level.get("poisoned") is False,
                        "registered-plan session was poisoned",
                    )
                    require(
                        low_level.get("owner_cutover") is (restore_count > 1),
                        "owner cutover does not match bootstrap/transition state",
                    )
                    generation = low_level.get("generation")
                    new_generation = low_level.get("new_generation")
                    require(
                        isinstance(generation, int)
                        and isinstance(new_generation, int)
                        and new_generation == generation + 1,
                        "restore generation did not advance exactly once",
                    )
                    if previous_generation is not None:
                        require(
                            generation == previous_generation,
                            "restore generations are not contiguous",
                        )
                    previous_generation = new_generation
                wall_ns = low_level.get("ioctl_wall_ns")
                require(
                    isinstance(wall_ns, int) and wall_ns > 0,
                    "restore lacks a positive ioctl_wall_ns",
                )
                ioctl_wall_ms.append(wall_ns / 1_000_000.0)
            elif operation == "release" and record.get("ok") is True:
                release_ok = True

    require(registration_count == 1, "expected exactly one plan registration record")
    require(restore_count >= 4 * SEQUENTIAL_REQUESTS, "too few L0-L3 restore records")
    require(restore_count % 4 == 0, "restore trace ends with a partial layer cycle")
    for offset in range(0, restore_count, 4):
        require(
            tuple(restore_layers[offset : offset + 4]) == LAYERS,
            f"restore order is not L0-L3 at cycle {offset // 4}",
        )
    require(release_ok, "trace lacks successful session release")

    before = load_stats(run_dir / "latency_stats.before.txt")
    after = load_stats(run_dir / "latency_stats.after.txt")
    counter_names = [
        "slot_plan_register_calls",
        "slot_plan_execute_calls",
        "slot_direct_copy_run_cache_hits",
        "slot_direct_metadata_recipe_cache_attempts",
        "slot_direct_target_recipe_cache_hits",
        "slot_direct_clear_recipe_cache_hits",
    ]
    if feature_contract:
        counter_names.extend(
            (
                "slot_direct_ioctl_calls",
                "slot_direct_successes",
                "slot_direct_errors",
                "slot_direct_dual_bank_copy_first_calls",
                "slot_direct_transition_fast_index_calls",
                "slot_direct_transition_group_clear_calls",
                "slot_direct_transition_group_clear_fallbacks",
                "slot_direct_transition_direct_cutover_calls",
                "slot_direct_shadow_pte_prefill_calls",
                "slot_direct_shadow_pte_prefill_blocks",
                "slot_direct_profile.shadow_pte_prefill.calls",
                "slot_direct_profile.shadow_pte_prefill.ns",
                "slot_direct_profile.shadow_pte_prefill.bytes",
                "slot_direct_profile.shadow_pte_publish.calls",
                "slot_direct_profile.shadow_pte_publish.ns",
                "slot_direct_profile.shadow_pte_publish.bytes",
            )
        )
    deltas = {name: stats_delta(before, after, name) for name in counter_names}
    require(deltas["slot_plan_register_calls"] >= 5, "fewer than five plans registered")
    require(
        deltas["slot_plan_execute_calls"] >= restore_count,
        "kernel execute counter is below trace restore count",
    )
    require(deltas["slot_direct_copy_run_cache_hits"] > 0, "copy-run cache had no hit")
    if metadata_recipe:
        require(
            deltas["slot_direct_metadata_recipe_cache_attempts"] > 0,
            "metadata recipe cache had no attempt",
        )
        require(
            deltas["slot_direct_target_recipe_cache_hits"] > 0,
            "target recipe cache had no hit",
        )
        require(
            deltas["slot_direct_clear_recipe_cache_hits"] > 0,
            "clear recipe cache had no hit",
        )
    else:
        for name in (
            "slot_direct_metadata_recipe_cache_attempts",
            "slot_direct_target_recipe_cache_hits",
            "slot_direct_clear_recipe_cache_hits",
        ):
            require(deltas[name] == 0, f"{name} changed while recipe cache was off")

    transition_count = max(restore_count - 1, 0)
    if feature_contract:
        require(
            deltas["slot_plan_execute_calls"] == restore_count,
            "registered-plan execute delta differs from restore trace",
        )
        require(
            deltas["slot_direct_ioctl_calls"] == restore_count
            and deltas["slot_direct_successes"] == restore_count
            and deltas["slot_direct_errors"] == 0,
            "slot-direct success/error counters do not match restore trace",
        )
        for name in (
            "slot_direct_dual_bank_copy_first_calls",
            "slot_direct_transition_fast_index_calls",
            "slot_direct_transition_group_clear_calls",
            "slot_direct_transition_direct_cutover_calls",
        ):
            require(
                deltas[name] == transition_count,
                f"{name} does not cover every non-bootstrap transition",
            )
        require(
            deltas["slot_direct_transition_group_clear_fallbacks"] == 0,
            "group-clear path fell back",
        )
        require(
            deltas["slot_direct_copy_run_cache_hits"] == restore_count,
            "copy-run cache did not hit every execute",
        )
        if metadata_recipe:
            require(
                deltas["slot_direct_metadata_recipe_cache_attempts"]
                == restore_count,
                "metadata recipe was not attempted on every execute",
            )
            require(
                deltas["slot_direct_target_recipe_cache_hits"] == restore_count,
                "target recipe did not hit every execute",
            )
            require(
                deltas["slot_direct_clear_recipe_cache_hits"]
                == max(restore_count - 2, 0),
                "clear recipe hit count does not match two initially empty banks",
            )

        expected_shadow_calls = transition_count if shadow_pte_prefill else 0
        expected_shadow_blocks = expected_shadow_calls * BLOCKS_PER_LAYER
        require(
            deltas["slot_direct_shadow_pte_prefill_calls"]
            == expected_shadow_calls,
            "shadow-PTE call count does not match non-bootstrap transitions",
        )
        require(
            deltas["slot_direct_shadow_pte_prefill_blocks"]
            == expected_shadow_blocks,
            "shadow-PTE prefill did not cover all 288 blocks per transition",
        )
        expected_shadow_bytes = expected_shadow_calls * LAYER_BYTES
        for stage in ("shadow_pte_prefill", "shadow_pte_publish"):
            calls_key = f"slot_direct_profile.{stage}.calls"
            ns_key = f"slot_direct_profile.{stage}.ns"
            bytes_key = f"slot_direct_profile.{stage}.bytes"
            require(
                deltas[calls_key] == expected_shadow_calls,
                f"{stage} profile call count is incorrect",
            )
            require(
                deltas[bytes_key] == expected_shadow_bytes,
                f"{stage} profile byte count is incorrect",
            )
            if shadow_pte_prefill:
                require(deltas[ns_key] > 0, f"{stage} profile time is zero")
            else:
                require(deltas[ns_key] == 0, f"{stage} ran while disabled")

    request_latency = measurement.get("request_latency_ms")
    require(isinstance(request_latency, dict), "request latency summary is absent")
    latency_median = request_latency.get("median")
    if latency_median is None:
        latency_median = request_latency.get("p50")
    require(
        isinstance(latency_median, (int, float))
        and math.isfinite(float(latency_median)),
        "request latency median is invalid",
    )

    return {
        "schema": "vllm_slot_dual_bank_fence_e2e_verification_v1",
        "status": "pass",
        "run_dir": str(run_dir),
        "fence": fence,
        "metadata_recipe": metadata_recipe,
        "fast_validation": fast_validation,
        "shadow_pte_prefill": shadow_pte_prefill,
        "feature_contract": feature_contract,
        "plan_sha256": plan_sha256,
        "sequential_requests": SEQUENTIAL_REQUESTS,
        "restore_cycles": restore_count // 4,
        "restore_count": restore_count,
        "fence_used_count": fence_used,
        "fence_accepted_count": fence_accepted,
        "completion_signaled_count": completion_signaled,
        "non_bootstrap_transition_count": transition_count,
        "ioctl_wall_ms": {
            "count": len(ioctl_wall_ms),
            "median": statistics.median(ioctl_wall_ms),
            "minimum": min(ioctl_wall_ms),
            "maximum": max(ioctl_wall_ms),
        },
        "request_latency_ms_median": float(latency_median),
        "latency_stats_delta": deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--fence", type=int, choices=(0, 1), required=True)
    parser.add_argument(
        "--metadata-recipe", type=int, choices=(0, 1), required=True
    )
    parser.add_argument(
        "--fast-validation", type=int, choices=(0, 1), default=0
    )
    parser.add_argument(
        "--shadow-pte-prefill", type=int, choices=(0, 1), default=0
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        summary = verify_case(
            args.run_dir,
            fence=args.fence,
            metadata_recipe=args.metadata_recipe,
            fast_validation=args.fast_validation,
            shadow_pte_prefill=args.shadow_pte_prefill,
        )
    except VerificationError as exc:
        print(f"verification failed: {exc}")
        return 1
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
