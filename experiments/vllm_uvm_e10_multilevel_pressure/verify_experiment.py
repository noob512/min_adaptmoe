#!/usr/bin/env python3
"""Verify the single-process E10 2->4->6 GiB pressure qualification."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


BLOCK_BYTES = 1_572_864
MAX_BYTES = 8_589_410_304
TARGETS = (1364, 2730, 4095)
CLASSES = ("cold70", "cold80", "cold89")
SELECTED_BYTES = {
    "cold70": 15_854_469_120,
    "cold80": 18_119_393_280,
    "cold89": 20_157_825_024,
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"artifact is not an object: {path}")
    return value


def load_stats(path: Path) -> dict[str, int]:
    require(path.is_file(), f"missing artifact: {path}")
    result: dict[str, int] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.strip().split()
        if len(fields) != 2:
            continue
        try:
            result[fields[0]] = int(fields[1], 0)
        except ValueError:
            continue
    return result


def stats_delta(
    before: dict[str, int], after: dict[str, int], name: str
) -> int:
    require(name in before, f"latency stats before lacks {name}")
    require(name in after, f"latency stats after lacks {name}")
    value = after[name] - before[name]
    require(value >= 0, f"latency counter {name} moved backwards")
    return value


DUAL_COUNTER_NAMES = (
    "slot_alias_dual_prepare_calls",
    "slot_alias_dual_prepared_blocks",
    "slot_alias_dual_reused_blocks",
    "slot_alias_dual_skipped_blocks",
    "slot_alias_dual_switch_calls",
    "slot_alias_dual_cpu_blocks",
    "slot_alias_dual_gpu_blocks",
    "slot_alias_dual_pde_writes",
    "slot_alias_dual_tlb_batches",
    "slot_alias_dual_prepare_ns",
    "slot_alias_dual_switch_ns",
)


def available_stats_delta(
    before: dict[str, int], after: dict[str, int], names: tuple[str, ...]
) -> dict[str, int]:
    """Return counters present in both snapshots.

    Failure artifacts can come from a module built immediately before a new
    diagnostic counter was added. Preserve all comparable evidence instead of
    making the failure verifier itself fail on the newer optional counter.
    """
    return {
        name: after[name] - before[name]
        for name in names
        if name in before and name in after and after[name] >= before[name]
    }


def expected_gpu_bytes(target: int) -> int:
    return (int(target) + 1) * BLOCK_BYTES


def verify_physical(
    snapshot: dict[str, Any],
    target: int,
    label: str,
    *,
    maximum_bytes: int,
) -> None:
    gpu_bytes = expected_gpu_bytes(target)
    require(
        snapshot.get("schema") == "vllm_uvm_kv_physical_snapshot_v1",
        f"{label}: physical schema",
    )
    require(
        int(snapshot.get("reserved_va_bytes", -1)) == maximum_bytes
        and int(snapshot.get("covered_bytes", -1)) == maximum_bytes
        and int(snapshot.get("total_blocks", -1))
        == maximum_bytes // BLOCK_BYTES - 1,
        f"{label}: stable maximum allocation",
    )
    full = snapshot["full"]
    head = snapshot["head"]
    tail = snapshot["tail"]
    require(
        full.get("ok") is True
        and int(full.get("requested_bytes", -1)) == maximum_bytes
        and int(full.get("gpu_bytes", -1)) == gpu_bytes
        and int(full.get("cpu_bytes", -1)) == maximum_bytes - gpu_bytes
        and int(full.get("unknown_bytes", -1)) == 0,
        f"{label}: full residency",
    )
    require(
        head.get("ok") is True
        and int(head.get("requested_bytes", -1)) == gpu_bytes
        and int(head.get("gpu_bytes", -1)) == gpu_bytes
        and int(head.get("cpu_bytes", -1)) == 0,
        f"{label}: GPU prefix",
    )
    require(
        tail.get("ok") is True
        and int(tail.get("requested_bytes", -1)) == maximum_bytes - gpu_bytes
        and int(tail.get("cpu_bytes", -1)) == maximum_bytes - gpu_bytes
        and int(tail.get("gpu_bytes", -1)) == 0,
        f"{label}: CPU tail",
    )


def verify_queue(snapshot: dict[str, Any], target: int, label: str) -> None:
    require(snapshot.get("invariant_ok") is True, f"{label}: queue invariant")
    require(
        int(snapshot.get("applied_target_blocks", -1)) == target
        and int(snapshot.get("backup_target_blocks", -1)) == target - 16
        and int(snapshot.get("gpu_backup_blocks", -1)) == target - 16
        and int(snapshot.get("gpu_free_blocks", -1)) == 16
        and int(snapshot.get("active_blocks", -1)) == 0,
        f"{label}: queue partition",
    )


def finite_positive(value: Any, label: str) -> float:
    number = float(value)
    require(math.isfinite(number) and number > 0, label)
    return number


def parse_tier_specs(raw: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
    gib_values: list[int] = []
    targets: list[int] = []
    classes: list[str] = []
    for record in raw.split(","):
        fields = record.strip().split(":")
        require(len(fields) == 3, f"invalid tier specification: {record!r}")
        gib_values.append(int(fields[0]))
        targets.append(int(fields[1]))
        classes.append(fields[2].strip())
    require(gib_values == sorted(set(gib_values)), "tier GiB values")
    require(targets == sorted(set(targets)), "tier block targets")
    require(len(classes) == len(set(classes)), "tier plan classes")
    return tuple(gib_values), tuple(targets), tuple(classes)


def class_width(name: str) -> int:
    match = re.fullmatch(r"cold(\d+)", name)
    require(match is not None, f"unsupported plan class name: {name}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--expected-trigger-mode",
        choices=("pressure", "direct"),
        default="pressure",
    )
    parser.add_argument(
        "--tier-specs",
        default="2:1364:cold70,4:2730:cold80,6:4095:cold89",
    )
    parser.add_argument("--kv-max-bytes", type=int, default=MAX_BYTES)
    parser.add_argument("--plan-classes", type=Path)
    args = parser.parse_args()
    trigger_mode = str(args.expected_trigger_mode)
    tier_gibs, targets, classes = parse_tier_specs(args.tier_specs)
    maximum_bytes = int(args.kv_max_bytes)
    selected_bytes = {
        name: class_width(name) * 48 * 4_718_592 for name in classes
    }
    if args.plan_classes is not None:
        manifest = load_json(args.plan_classes)
        inputs = (manifest.get("metadata") or {}).get("inputs") or {}
        for name in classes:
            require(name in manifest.get("classes", {}), f"missing plan class {name}")
            if name in inputs:
                selected_bytes[name] = int(inputs[name]["selected_main_bytes"])

    root = args.run_root.resolve()
    case = root / "sequence"
    returncode = int((case / "returncode.txt").read_text().strip())
    result = load_json(case / "result.json")
    if returncode != 0:
        error = str(result.get("error") or "")
        model = (
            result.get("shutdown_contract", {})
            .get("diagnostics_after_shutdown", {})
            .get("model", {})
        )
        switches = model.get("plan_class_switch_history") or []
        failure_stage = "startup_plan_class_precompile_restore"
        if "UVM original-VA SEAL failed" in error:
            failure_stage = "startup_static_alias_seal_dual_prepare"
        if "initial KV tier is not 2 GiB" in error:
            failure_stage = "first_2g_request_unexpected_early_growth"
        kernel_log = (root / "kernel.since.log").read_text(
            encoding="utf-8", errors="replace"
        )
        module_before = load_json(root / "module.before.json")
        module_custom = load_json(root / "module.custom.json")
        module_after = load_json(root / "module.after.json")
        stats_before = load_stats(root / "latency_stats.before.txt")
        stats_after = load_stats(root / "latency_stats.after.txt")
        dual_stats = available_stats_delta(
            stats_before, stats_after, DUAL_COUNTER_NAMES
        )
        failure = {
            "schema": "eurosys_e10_kv_2_4_6_pressure_verification_v1",
            "status": "failed",
            "runner_returncode": returncode,
            "completed_sequences": 0,
            "statistical_repetitions": 0,
            "requests_measured": 0,
            "failure_stage": failure_stage,
            "error": result.get("error"),
            "initial_kv_physical": result.get("initial_kv_physical"),
            "startup_plan_class_restore_succeeded": any(
                item.get("old_class") == "cold89"
                and item.get("new_class") == "cold70"
                and item.get("ok") is True
                for item in switches
            ),
            "active_plan_class_at_failure": model.get("active_plan_class"),
            "plan_generation_at_failure": model.get("plan_generation"),
            "plan_class_switches": [
                {
                    key: item.get(key)
                    for key in (
                        "old_class",
                        "new_class",
                        "ok",
                        "reason",
                        "generation",
                        "wall_ns",
                        "error",
                    )
                }
                for item in switches
            ],
            "cpu_shadow": model.get("plan_class_immutable_cpu_shadow"),
            "source_prefill": model.get("plan_class_source_prefill"),
            "cold_residency": model.get("cold_residency_result"),
            "hot_residency": model.get("hot_residency_result"),
            "dual_view_parent_switch": {
                "enabled": int(
                    module_custom.get("static_alias_dual_view_enable", -1)
                )
                == 1,
                "latency_stats_delta": dual_stats,
            },
            "cleanup_errors": result.get("shutdown_contract", {}).get(
                "cleanup_errors", []
            ),
            "kernel_xid_count": kernel_log.count("NVRM: Xid"),
            "kernel_mmu_fault_count": kernel_log.count("MMU Fault"),
            "gpu_clean_after": load_json(case / "gpu.after.json").get(
                "gpu_clean"
            ),
            "stock_uvm_restored": (
                module_after.get("loaded_srcversion")
                == module_before.get("loaded_srcversion")
                and module_after.get("gpu_clean") is True
            ),
            "prefill_timing": None,
            "decode_timing": None,
        }
        args.output.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, sort_keys=True))
        return 1
    require(result.get("status") == "ok", "result status")
    module_custom = load_json(root / "module.custom.json")
    require(
        int(module_custom.get("manual_migration_fault_mode", -1)) == 2
        and int(module_custom.get("access_counter_migration_enable", -1)) == 0
        and int(module_custom.get("prefetch_enable", -1)) == 0
        and int(module_custom.get("thrashing_enable", -1)) == 0
        and int(module_custom.get("static_alias_dual_view_enable", -1)) == 1
        and int(
            module_custom.get("static_alias_dual_view_require_full", -1)
        )
        == 1,
        "automatic migration policy is not fail-fast/disabled",
    )
    config = result.get("config") or {}
    require(
        config.get("multilevel_pressure_qualification") is True
        and config.get("multilevel_trigger_mode") == trigger_mode
        and config.get("graph_prefill") is True
        and config.get("graph_decode") is True,
        "runner mode",
    )
    qualification = result.get("kv_multilevel_pressure") or {}
    require(
        qualification.get("schema")
        in {
            "vllm_uvm_kv_multilevel_pressure_2_4_6_v1",
            "vllm_uvm_kv_multilevel_pressure_trace_v2",
        }
        and qualification.get("trigger_mode") == trigger_mode
        and qualification.get("pressure_source")
        == (
            "real_scheduler_backup_queue_consumption"
            if trigger_mode == "pressure"
            else "direct_idle_request_boundary"
        )
        and qualification.get("prediction_enabled") is False,
        "qualification schema/source",
    )
    require(
        qualification.get("target_sequence_blocks") == list(targets)
        and qualification.get("plan_class_sequence") == list(classes)
        and int(qualification.get("initial_reserved_va_bytes", -1))
        == maximum_bytes,
        "frozen tier sequence",
    )
    initial = qualification["initial_physical"]
    verify_physical(
        initial,
        targets[0],
        "initial",
        maximum_bytes=maximum_bytes,
    )
    expected_va = initial["va_sha256"]

    tiers = qualification.get("tiers") or []
    require(len(tiers) == len(targets), "tier count")
    output_hashes: set[str] = set()
    timing: dict[str, Any] = {}
    graph_capture_counts: set[int] = set()
    for index, (tier, target, class_name) in enumerate(
        zip(tiers, targets, classes, strict=True)
    ):
        label = f"{target} blocks/{class_name}"
        require(
            int(tier.get("tier_gib", -1)) == tier_gibs[index]
            and int(tier.get("target_blocks", -1)) == target
            and tier.get("plan_class") == class_name,
            f"{label}: identity",
        )
        plan = tier.get("plan_class_after_request") or {}
        require(
            plan.get("active_plan_class") == class_name
            and int(plan.get("selected_parameter_bytes_per_forward", -1))
            == selected_bytes[class_name],
            f"{label}: expert placement",
        )
        verify_physical(
            tier["physical_after_request"],
            target,
            label,
            maximum_bytes=maximum_bytes,
        )
        require(
            tier["physical_after_request"].get("va_sha256") == expected_va,
            f"{label}: KV VA changed",
        )
        verify_queue(tier["queue_after_request"], target, label)
        require(len(tier.get("warmup_decode") or []) >= 1, f"{label}: warmup")
        measured = tier.get("measured_decode") or []
        require(len(measured) >= 1, f"{label}: measured Decode")
        require(
            all(
                int(item.get("graph_wrapper_delta", {}).get("full_replay", 0))
                == 1
                for item in measured
            ),
            f"{label}: Decode Graph replay",
        )
        tier_timing = tier.get("timing") or {}
        finite_positive(tier_timing.get("prefill_end_to_end_ms"), f"{label}: Prefill")
        finite_positive(
            tier_timing.get("steady_decode_step_ms", {}).get("median"),
            f"{label}: Decode",
        )
        timing[f"{tier_gibs[index]}g"] = tier_timing
        output_hashes.add(str(tier.get("output_token_ids_sha256")))
        graph_after = tier.get("graph_wrapper_calls_after")
        if graph_after is not None:
            graph_capture_counts.add(int(graph_after.get("full_capture", -1)))

        if index == 0:
            require(
                tier.get("trigger") == "initial_measurement_autoscale_disabled"
                and tier.get("trigger_mode") == trigger_mode
                and int(tier.get("cleared_pressure", {}).get("blocks", 0)) > 0,
                "2 GiB measured pressure audit",
            )
            continue

        require(
            tier.get("trigger")
            == (
                "real_scheduler_backup_queue_pressure"
                if trigger_mode == "pressure"
                else "direct_idle_request_boundary"
            )
            and tier.get("trigger_mode") == trigger_mode,
            f"{label}: trigger",
        )
        schedule_pressure = tier.get("schedule_pressure") or {}
        decision = (tier.get("autoscale_after_prefill") or {}).get(
            "last_decision"
        ) or {}
        require(
            int(schedule_pressure.get("pending_blocks", -1)) == 4
            and int(schedule_pressure.get("pending_events", -1)) == 1,
            f"{label}: request did not create the same backup pressure",
        )
        if trigger_mode == "pressure":
            require(
                decision.get("growth_trigger_policy") == "backup_threshold_only"
                and decision.get("backup_pressure_threshold_reached") is True
                and int(decision.get("predicted_blocks", -1)) == 0
                and int(decision.get("prediction_error_blocks", -1)) == 0,
                f"{label}: pressure-only decision",
            )
        else:
            direct_transition = tier.get("direct_transition") or {}
            require(
                direct_transition.get("ok") is True
                and tier_timing.get("tier_transition_placement")
                == "before_request_while_idle",
                f"{label}: direct idle transition",
            )
        rpcs = tier.get("rpc_records") or []
        methods = [item.get("method") for item in rpcs]
        require(
            methods.count("activate_uvm_expert_graph_class") == 1
            and methods.count("resize_uvm_kv_cache_target") == 1
            and methods.index("activate_uvm_expert_graph_class")
            < methods.index("resize_uvm_kv_cache_target"),
            f"{label}: expert-before-KV order",
        )
        activation = next(
            item
            for item in rpcs
            if item.get("method") == "activate_uvm_expert_graph_class"
        ).get("result") or {}
        resize_result = next(
            item
            for item in rpcs
            if item.get("method") == "resize_uvm_kv_cache_target"
        ).get("result") or {}
        early_resume = activation.get(
            "prefill_topology_resume_before_plan_switch"
        ) or {}
        plan_switch = activation.get("plan_switch") or {}
        delta_result = plan_switch.get("rebind") or {}
        dual_refresh = resize_result.get("dual_view_refresh") or {}
        refresh_ioctl = dual_refresh.get("refresh") or {}
        expected_delta_entries = (
            class_width(class_name) - class_width(classes[index - 1])
        ) * 450
        require(
            early_resume.get("ok") is True
            and early_resume.get("pre_state") == "remote_parked"
            and early_resume.get("post_state") == "sealed"
            and plan_switch.get("pre_state") == "sealed"
            and plan_switch.get("post_state") == "sealed"
            and plan_switch.get("dual_view_refresh_deferred") is True
            and int(plan_switch.get("quiesced_delta_add_count", -1))
            == expected_delta_entries
            and int(plan_switch.get("quiesced_delta_remove_count", -1)) == 0
            ,
            f"{label}: resumed sealed delta append",
        )
        if expected_delta_entries <= 4608:
            require(
                plan_switch.get("quiesced_delta_eligible") is True
                and plan_switch.get("quiesced_delta_used") is True
                and plan_switch.get("park") is None
                and plan_switch.get("resume") is None
                and delta_result.get("operation") == "map"
                and int(delta_result.get("requested_count", -1))
                == expected_delta_entries
                and int(delta_result.get("ioctl_calls", -1)) == 1,
                f"{label}: single-ioctl delta append",
            )
        else:
            require(
                plan_switch.get("quiesced_delta_used") is False
                and delta_result.get("ok") is True,
                f"{label}: large append fallback",
            )
        require(
            delta_result.get("dual_view_refresh_deferred") is True
            and resize_result.get("ok") is True
            and resize_result.get("made_progress") is True
            and dual_refresh.get("ok") is True
            and dual_refresh.get("state_before") == "sealed"
            and dual_refresh.get("state_after") == "sealed"
            and refresh_ioctl.get("ok") is True
            and refresh_ioctl.get("refresh_dual") is True
            and refresh_ioctl.get("sealed") is True,
            f"{label}: current-tier dual view was not rebuilt after KV",
        )
        decode_transitions = tier.get("first_decode_transitions") or []
        require(
            len(decode_transitions) == 1,
            f"{label}: first-Decode transition count",
        )
        decode_transition = decode_transitions[0]
        remote_map = decode_transition.get("remote_map") or {}
        remote_requested_bytes = int(remote_map.get("requested_bytes", -1))
        require(
            decode_transition.get("ok") is True
            and decode_transition.get("pre_state") == "sealed"
            and decode_transition.get("post_state") == "remote_parked"
            and decode_transition.get("remote_map_operation")
            == "prebuilt_full_dual_view"
            and decode_transition.get("post_map_physical_audit_enabled")
            is False
            and decode_transition.get("hot_physical_audit_requested") is False
            and decode_transition.get("hot_physical_audit_enabled") is False
            and decode_transition.get(
                "hot_physical_audit_diagnostics_skipped"
            )
            is True
            and decode_transition.get("physical_residency") is None
            and decode_transition.get("hot_physical_residency") is None,
            f"{label}: post-map residency audit was not removed",
        )
        require(
            remote_map.get("ok") is True
            and remote_map.get("reason") == "prebuilt_full_dual_view"
            and int(remote_map.get("ioctl_calls", -1)) == 0
            and int(remote_map.get("requested_count", -1))
            == int(remote_map.get("success_count", -2))
            and int(remote_map.get("failure_count", -1)) == 0
            and remote_requested_bytes > 0
            and int(remote_map.get("cpu_resident_bytes", -1))
            >= remote_requested_bytes
            and int(remote_map.get("remote_mapped_bytes", -1))
            >= remote_requested_bytes
            and int(remote_map.get("covered_bytes", -1))
            >= remote_requested_bytes
            and int(remote_map.get("unknown_bytes", -1)) == 0,
            f"{label}: mutation-time remote-map validation",
        )
        finite_positive(
            tier_timing.get("rebalance_rpc_wall_ms"),
            f"{label}: rebalance timing",
        )
        finite_positive(
            tier_timing.get("prefill_topology_resume_ms"),
            f"{label}: early Prefill resume timing",
        )
        require(
            float(
                tier_timing.get(
                    "rebalance_rpc_including_prefill_resume_ms", -1
                )
            )
            > float(tier_timing.get("rebalance_rpc_wall_ms", math.inf)),
            f"{label}: non-overlapping timing accounting",
        )
        before_capture = int(
            tier.get("graph_wrapper_calls_before", {}).get("full_capture", -1)
        )
        after_capture = int(
            tier.get("graph_wrapper_calls_after", {}).get("full_capture", -2)
        )
        require(before_capture == after_capture, f"{label}: Graph recapture")

    require(len(output_hashes) == 1 and "None" not in output_hashes, "output mismatch")
    require(len(graph_capture_counts) <= 1, "capture count changed across tiers")
    shutdown = result.get("shutdown_contract") or {}
    require(not (shutdown.get("cleanup_errors") or []), "cleanup errors")
    require(
        (case / "gpu.after.json").is_file()
        and load_json(case / "gpu.after.json").get("gpu_clean") is True,
        "case did not release GPU",
    )
    module_before = load_json(root / "module.before.json")
    module_after = load_json(root / "module.after.json")
    require(
        module_after.get("loaded_srcversion") == module_before.get("loaded_srcversion")
        and module_after.get("gpu_clean") is True
        and int(module_after.get("memory_used_mib", -1)) == 0,
        "stock UVM/GPU restoration",
    )
    kernel_log = (root / "kernel.since.log").read_text(
        encoding="utf-8", errors="replace"
    )
    fatal_patterns = (
        "Xid ",
        "NVRM: Xid",
        "MMU Fault",
        "BUG:",
        "kernel BUG",
        "general protection fault",
        "unable to handle kernel",
    )
    require(
        not any(pattern in kernel_log for pattern in fatal_patterns),
        "kernel fault signature",
    )

    stats_before = load_stats(root / "latency_stats.before.txt")
    stats_after = load_stats(root / "latency_stats.after.txt")
    dual_stats = {
        name: stats_delta(stats_before, stats_after, name)
        for name in DUAL_COUNTER_NAMES
    }
    require(
        dual_stats["slot_alias_dual_prepare_calls"] > 0
        and dual_stats["slot_alias_dual_prepared_blocks"] > 0
        and dual_stats["slot_alias_dual_reused_blocks"] > 0
        and dual_stats["slot_alias_dual_switch_calls"] >= 5
        and dual_stats["slot_alias_dual_cpu_blocks"] > 0
        and dual_stats["slot_alias_dual_gpu_blocks"] > 0
        and dual_stats["slot_alias_dual_pde_writes"]
        == dual_stats["slot_alias_dual_cpu_blocks"]
        + dual_stats["slot_alias_dual_gpu_blocks"]
        and 0 < dual_stats["slot_alias_dual_tlb_batches"]
        < dual_stats["slot_alias_dual_pde_writes"]
        and dual_stats["slot_alias_dual_prepare_ns"] > 0
        and dual_stats["slot_alias_dual_switch_ns"] > 0,
        "dual-view parent-entry fast path was not exercised consistently",
    )

    verification = {
        "schema": "eurosys_e10_kv_2_4_6_pressure_verification_v1",
        "status": "ok",
        "statistical_repetitions": 0,
        "completed_sequences": 1,
        "kv_reserved_va_bytes": maximum_bytes,
        "kv_target_blocks": list(targets),
        "kv_tiers_gib": list(tier_gibs),
        "expert_plan_classes": list(classes),
        "trigger_mode": trigger_mode,
        "expert_selected_bytes": selected_bytes,
        "runtime_post_map_residency_audit": False,
        "runtime_hot_residency_audit": False,
        "migration_heuristics": {
            "manual_fault_mode": 2,
            "access_counter_migration": 0,
            "prefetch": 0,
            "thrashing": 0,
        },
        "dual_view_parent_switch": {
            "enabled": True,
            "switch_batch_blocks": 64,
            "latency_stats_delta": dual_stats,
        },
        "output_token_ids_sha256": next(iter(output_hashes)),
        "timing": timing,
    }
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
