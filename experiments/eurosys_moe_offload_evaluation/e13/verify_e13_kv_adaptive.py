#!/usr/bin/env python3
"""Fail-closed verifier for an E13 Adaptive-versus-Fixed run root."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FATAL_KERNEL_PATTERNS = (
    re.compile(r"\bXid\b"),
    re.compile(r"MMU Fault", re.IGNORECASE),
    re.compile(r"NULL pointer", re.IGNORECASE),
    re.compile(r"general protection fault", re.IGNORECASE),
    re.compile(r"kernel BUG"),
    re.compile(r"BUG:"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("qualification", "formal", "capacity"),
        required=True,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: Any) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def verify(args: argparse.Namespace) -> dict[str, Any]:
    root = args.run_root.resolve()
    protocol = load_json(args.config)
    arms = {item["name"]: item for item in protocol["arms"]}
    tiers = {item["name"]: item for item in protocol["tiers"]}
    seeds = [int(value) for value in protocol["profiles"][args.profile]["seeds"]]
    ordered_targets = [
        int(item["target_blocks"]) for item in protocol["tiers"]
    ]
    allowed = set(ordered_targets)
    expected_scenarios = set(protocol["profiles"][args.profile]["scenario_names"])
    checks: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    results_by_repeat: dict[int, dict[str, dict[str, Any]]] = {}

    for repeat_index, seed in enumerate(seeds):
        repeat_dir = root / f"repeat-{repeat_index:02d}-seed-{seed}"
        add_check(checks, f"repeat_{repeat_index}_directory", repeat_dir.is_dir(), str(repeat_dir))
        repeat_results: dict[str, dict[str, Any]] = {}
        for arm_name, arm in arms.items():
            case_dir = repeat_dir / arm_name
            result_path = case_dir / "result.json"
            returncode_path = case_dir / "returncode.txt"
            case_checks: list[dict[str, Any]] = []
            if not result_path.is_file() or not returncode_path.is_file():
                add_check(case_checks, "artifacts_present", False, str(case_dir))
                cases.append(
                    {
                        "repeat_index": repeat_index,
                        "seed": seed,
                        "arm": arm_name,
                        "status": "missing",
                        "checks": case_checks,
                    }
                )
                continue
            result = load_json(result_path)
            repeat_results[arm_name] = result
            returncode = int(returncode_path.read_text(encoding="utf-8").strip())
            add_check(case_checks, "returncode_zero", returncode == 0, returncode)
            add_check(
                case_checks,
                "result_status_ok",
                result.get("status") == "ok",
                result.get("status"),
            )
            add_check(
                case_checks,
                "identity",
                result.get("arm") == arm_name
                and int(result.get("seed", -1)) == seed
                and int(result.get("repeat_index", -1)) == repeat_index
                and result.get("profile") == args.profile,
                {
                    "arm": result.get("arm"),
                    "seed": result.get("seed"),
                    "repeat_index": result.get("repeat_index"),
                    "profile": result.get("profile"),
                },
            )
            observed_scenarios = set((result.get("scenario_summaries") or {}).keys())
            add_check(
                case_checks,
                "scenario_coverage",
                observed_scenarios == expected_scenarios,
                sorted(observed_scenarios),
            )
            requests = result.get("requests") or []
            add_check(case_checks, "requests_nonempty", bool(requests), len(requests))
            request_ids = [item.get("logical_request") for item in requests]
            add_check(
                case_checks,
                "logical_request_ids_unique",
                len(request_ids) == len(set(request_ids)),
                len(request_ids),
            )
            successful = [item for item in requests if item.get("success")]
            add_check(
                case_checks,
                "successful_request_metrics_complete",
                all(
                    item.get("ttft_ms") is not None
                    and item.get("e2e_ms") is not None
                    and int(item.get("completion_tokens", -1))
                    == int(item.get("requested_output_tokens", -2))
                    for item in successful
                ),
                len(successful),
            )
            steps = result.get("time_series") or []
            observed_targets = {
                int(item.get("kv", {}).get("current_target_blocks", -1))
                for item in steps
            }
            add_check(
                case_checks,
                "targets_registered",
                bool(observed_targets) and observed_targets <= allowed,
                sorted(observed_targets),
            )
            add_check(
                case_checks,
                "queue_invariants",
                bool(steps)
                and all(item.get("kv", {}).get("invariant_ok") is True for item in steps),
                sum(item.get("kv", {}).get("invariant_ok") is False for item in steps),
            )
            initial = tiers[arm["initial_tier"]]
            runtime_snapshot = (
                result.get("configuration", {}).get("runtime", {})
            )
            synchronized_tiers = bool(
                runtime_snapshot.get("synchronized_tier_steps", False)
            )
            if synchronized_tiers:
                setup = result.get("initialization", {}).get(
                    "tier_setup_result", {}
                )
                setup_steps = setup.get("tier_steps") or []
                setup_edges = [
                    (
                        int(step.get("old_target_blocks", -1)),
                        int(step.get("new_target_blocks", -1)),
                    )
                    for step in setup_steps
                ]
                setup_chain_ok = bool(setup_steps) and all(
                    bool((step.get("result") or {}).get("ok"))
                    and old in ordered_targets
                    and new in ordered_targets
                    and (
                        old == new
                        or abs(
                            ordered_targets.index(old)
                            - ordered_targets.index(new)
                        )
                        == 1
                    )
                    and (
                        index == 0
                        or old
                        == int(
                            setup_steps[index - 1].get(
                                "new_target_blocks", -1
                            )
                        )
                    )
                    for index, (step, (old, new)) in enumerate(
                        zip(setup_steps, setup_edges)
                    )
                )
                setup_chain_ok = (
                    setup_chain_ok
                    and setup.get("ok") is True
                    and setup_edges[-1][1]
                    == int(initial["target_blocks"])
                )
                add_check(
                    case_checks,
                    "initial_tier_uses_synchronized_adjacent_steps",
                    setup_chain_ok,
                    setup_edges,
                )
            if arm["strategy"] == "fixed":
                expected_target = int(initial["target_blocks"])
                add_check(
                    case_checks,
                    "fixed_target_immutable",
                    observed_targets == {expected_target},
                    sorted(observed_targets),
                )
                changed = [
                    event
                    for event in (result.get("resize_events") or [])
                    if event.get("direction") in {"grow", "shrink"}
                ]
                add_check(case_checks, "fixed_runtime_resize_zero", not changed, len(changed))
            else:
                add_check(
                    case_checks,
                    "adaptive_controller_enabled",
                    all(item.get("kv", {}).get("enabled") is True for item in steps),
                    len(steps),
                )
                add_check(
                    case_checks,
                    "adaptive_all_tiers_covered",
                    observed_targets == allowed,
                    sorted(observed_targets),
                )
                online_resize = bool(
                    runtime_snapshot.get(
                        "scheduler_boundary_resize", False
                    )
                )
                if online_resize and synchronized_tiers:
                    nonadjacent_resizes = [
                        {
                            "scenario": event.get("scenario"),
                            "old": event.get("old_target_blocks"),
                            "new": event.get("new_target_blocks"),
                        }
                        for event in (result.get("resize_events") or [])
                        if event.get("direction") in {"grow", "shrink"}
                        and not (
                            int(event.get("old_target_blocks", -1))
                            in ordered_targets
                            and int(event.get("new_target_blocks", -1))
                            in ordered_targets
                            and abs(
                                ordered_targets.index(
                                    int(event.get("old_target_blocks", -1))
                                )
                                - ordered_targets.index(
                                    int(event.get("new_target_blocks", -1))
                                )
                            )
                            == 1
                        )
                    ]
                    add_check(
                        case_checks,
                        "adaptive_resize_uses_synchronized_adjacent_tiers",
                        not nonadjacent_resizes,
                        nonadjacent_resizes,
                    )
                cohort_controller_events = [
                    event
                    for epoch in (result.get("epochs") or [])
                    for event in (epoch.get("cohort_controller_events") or [])
                ]
                expected_late_cohorts = sum(
                    max(len(epoch.get("cohorts") or []) - 1, 0)
                    for scenario in protocol["scenarios"]
                    if scenario["name"] in expected_scenarios
                    for epoch in scenario.get("epochs") or []
                )
                late_controller_events = [
                    event
                    for event in cohort_controller_events
                    if event.get("late_arrival") is True
                ]
                add_check(
                    case_checks,
                    "adaptive_late_cohorts_controlled",
                    len(late_controller_events) == expected_late_cohorts
                    and all(
                        bool((event.get("controller_result") or {}).get("ok"))
                        and (
                            (event.get("controller_result") or {})
                            .get("pressure", {})
                            .get("decision_source")
                            == "unified_kv_autoscale_controller"
                        )
                        for event in late_controller_events
                    ),
                    {
                        "expected": expected_late_cohorts,
                        "observed": len(late_controller_events),
                    },
                )
                if "burst_recovery" in expected_scenarios:
                    burst_directions = [
                        event.get("direction")
                        for event in (result.get("resize_events") or [])
                        if event.get("scenario") == "burst_recovery"
                        and event.get("direction") in {"grow", "shrink"}
                    ]
                    first_burst_grow = next(
                        (
                            index
                            for index, direction in enumerate(burst_directions)
                            if direction == "grow"
                        ),
                        None,
                    )
                    recovered_after_grow = (
                        first_burst_grow is not None
                        and "shrink" in burst_directions[first_burst_grow + 1 :]
                    )
                    add_check(
                        case_checks,
                        "adaptive_burst_growth_and_recovery",
                        first_burst_grow is not None and recovered_after_grow,
                        burst_directions,
                    )
                    burst_late_events = [
                        event
                        for epoch in (result.get("epochs") or [])
                        if epoch.get("scenario") == "burst_recovery"
                        for event in (epoch.get("cohort_controller_events") or [])
                        if event.get("late_arrival") is True
                    ]
                    deferred_burst_requests = [
                        item
                        for item in requests
                        if item.get("scenario") == "burst_recovery"
                        and int(item.get("cohort_index", -1)) > 0
                        and item.get("admission_deferred_for_kv_grow") is True
                    ]
                    # Read the case snapshot rather than the verifier's current
                    # config so historical drain-at-idle artifacts remain
                    # reproducible after the protocol gained online switching.
                    if online_resize:
                        online_shrinks = [
                            event
                            for event in (result.get("resize_events") or [])
                            if event.get("direction") == "shrink"
                        ]
                        invalid_tail_proofs = [
                            {
                                "scenario": event.get("scenario"),
                                "target": event.get("new_target_blocks"),
                                "max_active": (event.get("kv_before") or {}).get(
                                    "max_active_block_id"
                                ),
                                "tail_free_start": (
                                    event.get("kv_before") or {}
                                ).get("tail_free_start_block_id"),
                            }
                            for event in online_shrinks
                            if not (
                                (event.get("kv_before") or {}).get(
                                    "active_block_ids_known"
                                )
                                is True
                                and (event.get("kv_before") or {}).get(
                                    "max_active_block_id"
                                )
                                is not None
                                and int(
                                    (event.get("kv_before") or {}).get(
                                        "max_active_block_id"
                                    )
                                )
                                <= int(event.get("new_target_blocks", -1))
                                and int(
                                    (event.get("kv_before") or {}).get(
                                        "tail_free_start_block_id", -1
                                    )
                                )
                                <= int(event.get("new_target_blocks", -1)) + 1
                            )
                        ]
                        add_check(
                            case_checks,
                            "adaptive_online_shrinks_use_contiguous_free_tail",
                            bool(online_shrinks) and not invalid_tail_proofs,
                            {
                                "shrink_count": len(online_shrinks),
                                "invalid": invalid_tail_proofs,
                            },
                        )
                        online_growth_events = [
                            event
                            for event in burst_late_events
                            if bool(
                                (
                                    (event.get("controller_result") or {}).get(
                                        "autoscale"
                                    )
                                    or {}
                                ).get("committed_with_running_requests")
                            )
                        ]
                        add_check(
                            case_checks,
                            "adaptive_burst_growth_committed_at_scheduler_boundary",
                            bool(online_growth_events)
                            and not deferred_burst_requests,
                            {
                                "online_growth_events": len(online_growth_events),
                                "deferred_requests": len(deferred_burst_requests),
                            },
                        )
                    else:
                        add_check(
                            case_checks,
                            "adaptive_burst_admission_waited_for_growth",
                            bool(deferred_burst_requests),
                            len(deferred_burst_requests),
                        )
            plan_class_by_target = {
                int(item["target_blocks"]): str(item["plan_class"])
                for item in protocol["tiers"]
            }
            tier_mismatches = [
                int(item.get("engine_step", -1))
                for item in steps
                if item.get("active_plan_class")
                != plan_class_by_target.get(
                    int(item.get("kv", {}).get("current_target_blocks", -1))
                )
            ]
            add_check(
                case_checks,
                "kv_plan_class_tier_consistent",
                not tier_mismatches,
                tier_mismatches[:10],
            )
            failed_resizes = [
                event
                for event in (result.get("resize_events") or [])
                if not bool((event.get("result") or {}).get("ok"))
            ]
            add_check(
                case_checks,
                "runtime_resize_attempts_succeeded",
                not failed_resizes,
                len(failed_resizes),
            )
            before_graph = (
                result.get("initialization", {})
                .get("graph_counter_after_load", {})
                .get("num_cudagraph_captured")
            )
            after_graph = (result.get("graph_counter_after_trace") or {}).get(
                "num_cudagraph_captured"
            )
            add_check(
                case_checks,
                "no_runtime_graph_recapture",
                before_graph is not None and before_graph == after_graph,
                {"before": before_graph, "after": after_graph},
            )
            gpu_after_path = case_dir / "gpu.after.json"
            gpu_after = load_json(gpu_after_path) if gpu_after_path.is_file() else {}
            add_check(
                case_checks,
                "gpu_clean_after",
                gpu_after.get("gpu_clean") is True,
                gpu_after,
            )
            nvml_path = case_dir / "nvml.csv"
            nvml_lines = (
                nvml_path.read_text(encoding="utf-8").splitlines()
                if nvml_path.is_file()
                else []
            )
            add_check(case_checks, "nvml_samples_present", len(nvml_lines) > 2, len(nvml_lines))
            kernel_path = case_dir / "kernel.since.log"
            kernel_text = (
                kernel_path.read_text(encoding="utf-8", errors="replace")
                if kernel_path.is_file()
                else ""
            )
            fatal_matches = [
                pattern.pattern
                for pattern in FATAL_KERNEL_PATTERNS
                if pattern.search(kernel_text)
            ]
            add_check(case_checks, "kernel_fault_free", not fatal_matches, fatal_matches)
            case_ok = all(item["ok"] for item in case_checks)
            cases.append(
                {
                    "repeat_index": repeat_index,
                    "seed": seed,
                    "arm": arm_name,
                    "status": "ok" if case_ok else "error",
                    "checks": case_checks,
                    "summary": result.get("summary"),
                }
            )
        results_by_repeat[repeat_index] = repeat_results

    for repeat_index, repeat_results in results_by_repeat.items():
        if set(repeat_results) != set(arms):
            add_check(
                checks,
                f"repeat_{repeat_index}_all_arms_present",
                False,
                sorted(repeat_results),
            )
            continue
        traces = {value.get("trace_sha256") for value in repeat_results.values()}
        add_check(
            checks,
            f"repeat_{repeat_index}_trace_hash_equal",
            len(traces) == 1 and None not in traces,
            sorted(str(value) for value in traces),
        )
        reference_name = next(iter(arms))
        reference = repeat_results[reference_name]
        reference_prompts = {
            item["logical_request"]: item["prompt_token_ids_sha256"]
            for item in reference.get("requests") or []
        }
        prompt_equal = True
        output_equal = True
        first_token_equal = True
        compared_outputs = 0
        matched_outputs = 0
        output_mismatches: list[str] = []
        for arm_name, result in repeat_results.items():
            if arm_name == reference_name:
                continue
            prompts = {
                item["logical_request"]: item["prompt_token_ids_sha256"]
                for item in result.get("requests") or []
            }
            prompt_equal = prompt_equal and prompts == reference_prompts
            common = set(reference.get("output_hash_by_logical_request") or {}) & set(
                result.get("output_hash_by_logical_request") or {}
            )
            for logical in common:
                compared_outputs += 1
                if (
                    reference["output_hash_by_logical_request"][logical]
                    != result["output_hash_by_logical_request"][logical]
                ):
                    output_equal = False
                    output_mismatches.append(f"{arm_name}:{logical}")
                    reference_record = next(
                        item
                        for item in reference.get("requests") or []
                        if item["logical_request"] == logical
                    )
                    result_record = next(
                        item
                        for item in result.get("requests") or []
                        if item["logical_request"] == logical
                    )
                    first_token_equal = first_token_equal and (
                        reference_record.get("output_token_ids", [None])[0]
                        == result_record.get("output_token_ids", [None])[0]
                    )
                else:
                    matched_outputs += 1
        add_check(
            checks,
            f"repeat_{repeat_index}_prompt_sequence_equal",
            prompt_equal,
            len(reference_prompts),
        )
        add_check(
            checks,
            f"repeat_{repeat_index}_common_first_tokens_equal",
            first_token_equal,
            output_mismatches[:20],
        )
        add_check(
            checks,
            f"repeat_{repeat_index}_full_output_hash_diagnostic",
            True,
            {
                "all_equal": output_equal,
                "matched": matched_outputs,
                "compared": compared_outputs,
                "match_fraction": (
                    matched_outputs / compared_outputs
                    if compared_outputs
                    else None
                ),
                "mismatches": output_mismatches[:20],
            },
        )

    adaptive_results = [
        values["adaptive"]
        for values in results_by_repeat.values()
        if "adaptive" in values and values["adaptive"].get("status") == "ok"
    ]
    adaptive_directions = [
        event.get("direction")
        for result in adaptive_results
        for event in result.get("resize_events") or []
    ]
    add_check(
        checks,
        "adaptive_exercised_growth",
        "grow" in adaptive_directions,
        adaptive_directions.count("grow"),
    )
    add_check(
        checks,
        "adaptive_exercised_shrink",
        "shrink" in adaptive_directions,
        adaptive_directions.count("shrink"),
    )

    cases_ok = len(cases) == len(seeds) * len(arms) and all(
        case["status"] == "ok" for case in cases
    )
    status = "ok" if cases_ok and all(item["ok"] for item in checks) else "error"
    return {
        "schema": "eurosys_e13_kv_adaptive_verification_v1",
        "status": status,
        "verified_at": utc_now(),
        "run_root": str(root),
        "profile": args.profile,
        "expected_case_count": len(seeds) * len(arms),
        "observed_case_count": len(cases),
        "checks": checks,
        "cases": cases,
    }


def main() -> int:
    args = parse_args()
    payload = verify(args)
    output = args.output or args.run_root / "verification.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "output": str(output)}))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
