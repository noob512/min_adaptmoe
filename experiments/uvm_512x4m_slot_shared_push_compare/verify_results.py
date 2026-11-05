#!/usr/bin/env python3
"""Strictly verify one or more saved slot shared-push benchmark runs."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any

import run_benchmark as runner


MODES = runner.MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--require-full", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
    print(f"PASS {message}")


def load(path: Path) -> tuple[Path, dict[str, Any]]:
    summary_path = path / "summary.json" if path.is_dir() else path
    if not summary_path.is_file():
        raise ValueError(f"missing summary: {summary_path}")
    import json

    document = json.loads(summary_path.read_text(encoding="utf-8"))
    return summary_path, document


def residency_equals(value: dict[str, Any], *, ranges: int, cpu: int, gpu: int, any_: int) -> bool:
    return (
        int(value["ranges"]) == ranges
        and int(value["cpu_bytes"]) == cpu
        and int(value["gpu_bytes"]) == gpu
        and int(value["any_bytes"]) == any_
        and int(value["unknown_bytes"]) == 0
    )


def verify_outcome(value: dict[str, Any], allocations: int, logical_bytes: int) -> None:
    require(
        int(value["calls"]) == 4
        and int(value["success"]) == allocations
        and int(value["failure"]) == 0
        and int(value["conflict"]) == 0
        and int(value["requested_bytes"]) == logical_bytes
        and int(value["accepted_bytes"]) == logical_bytes
        and int(value["bad_entries"]) == 0
        and int(value["completion_waits"]) == 4,
        "sample has four synchronous, complete batch results",
    )


def verify_sample(sample: dict[str, Any], config: dict[str, Any]) -> None:
    allocations = int(config["allocations"])
    logical_bytes = int(config["logical_bytes"])
    mode = sample["mode"]
    require(sample.get("schema_version") == 1 and sample.get("status") == "ok", "sample schema/status")
    require(mode in MODES, f"known mode {mode}")
    require(sample["config"] == {
        "device": config["device"],
        "allocations": allocations,
        "allocation_bytes": config["allocation_bytes"],
        "guard_bytes": config["guard_bytes"],
        "batch_size": config["batch_size"],
        "batches": 4,
        "logical_bytes": logical_bytes,
        "warmups": config["worker_warmups"],
        "slot_layer_id": config["slot_layer_id"],
        "slot_expert_id": config["slot_expert_id"],
    }, "sample geometry matches run")
    abi = sample["abi"]
    require(
        int(abi["generic_ioctl"]) == 90
        and int(abi["generic_max_ranges"]) == 128
        and int(abi["slot_ioctl"]) == 91
        and int(abi["slot_max_entries"]) == 128
        and int(abi["slot_params_size"]) == 6224
        and int(abi["slot_mode_prealloc"]) == 8
        and int(abi["slot_mode_per_block"]) == 10
        and int(abi["slot_mode_shared"]) == 11,
        "versioned ioctl 90/91 ABI and slot modes",
    )
    require(
        int(sample["layout"]["adjacent_test_ranges"]) == 0
        and int(sample["layout"]["min_gap_bytes"]) >= int(config["allocation_bytes"])
        and len(sample["layout"]["va_snapshot"]) == allocations,
        "all test VAs are present and guard-separated",
    )

    slot_mode = mode != "driver_4x128_sync"
    setup = sample["setup"]
    require(
        bool(setup["slot_vidmem_preallocated_before_timing"]) == slot_mode,
        "mode-8 slot preallocation is explicitly outside timing",
    )
    prealloc = setup["prealloc"]
    if slot_mode:
        verify_outcome(prealloc, allocations, logical_bytes)
        require(
            int(prealloc["remap"]) == 0
            and int(prealloc["tlb_invalidate"]) == 0
            and int(prealloc["copy_pushes"]) == 0,
            "mode-8 setup allocates backing without mapping",
        )
    else:
        require(all(int(value) == 0 for value in prealloc.values()), "generic mode has no slot setup")

    reset = sample["reset"]
    require(
        int(reset["accepted"]) == allocations
        and int(reset["last_prefetch_cpu_ranges"]) == allocations
        and bool(reset["va_stable"])
        and residency_equals(
            reset["source_residency"],
            ranges=allocations,
            cpu=logical_bytes,
            gpu=0,
            any_=logical_bytes,
        ),
        "all source ranges are CPU-resident immediately before timing",
    )

    measurement = sample["measurement"]
    verify_outcome(measurement["outcome"], allocations, logical_bytes)
    require(
        bool(measurement["cpu_reset_immediately_before_profile_and_timing"])
        and bool(measurement["host_monotonic_to_synchronous_ioctl_completion"])
        and int(measurement["wall_ns"]) > 0
        and math.isfinite(float(measurement["wall_ms"]))
        and float(measurement["wall_ms"]) > 0
        and math.isfinite(float(measurement["gib_s"]))
        and float(measurement["gib_s"]) > 0,
        "ioctl-completion e2e timing is finite",
    )
    if slot_mode:
        require(
            math.isfinite(float(measurement["copy_ms"]))
            and float(measurement["copy_ms"]) > 0
            and math.isfinite(float(measurement["slot_map_ms"]))
            and float(measurement["slot_map_ms"]) > 0
            and math.isfinite(float(measurement["driver_profile_total_ms"]))
            and float(measurement["driver_profile_total_ms"]) > 0
            and measurement["component_source"]
            == "slot_direct_profile_cumulative_nested_stages"
            and bool(measurement["components_are_nested_not_additive"]),
            "slot copy/map/total profile timings are explicit and non-additive",
        )
        require(
            measurement["noncomparable_generic_diagnostics"] is None,
            "slot mode has no generic-stage timing mixed into its components",
        )
    else:
        diagnostics = measurement["noncomparable_generic_diagnostics"]
        require(
            measurement["copy_ms"] is None
            and measurement["slot_map_ms"] is None
            and measurement["driver_profile_total_ms"] is None
            and measurement["component_source"] is None
            and not bool(measurement["components_are_nested_not_additive"])
            and isinstance(diagnostics, dict)
            and diagnostics["note"]
            == "nested CPU stages; not the slot copy/map timing contract"
            and all(
                math.isfinite(float(diagnostics[name]))
                and float(diagnostics[name]) >= 0
                for name in (
                    "migrate_copy_between_total_cpu_ms",
                    "migrate_block_map_cpu_ms",
                    "migrate_api_tracker_wait_cpu_ms",
                )
            ),
            "generic mode does not fabricate incomparable slot timing stages",
        )

    validation = sample["validation"]
    expected_cpu = logical_bytes if slot_mode else 0
    expected_gpu = 0 if slot_mode else logical_bytes
    require(
        residency_equals(
            validation["post_residency"],
            ranges=allocations,
            cpu=expected_cpu,
            gpu=expected_gpu,
            any_=logical_bytes,
        )
        and residency_equals(
            validation["post_checksum_residency"],
            ranges=allocations,
            cpu=expected_cpu,
            gpu=expected_gpu,
            any_=logical_bytes,
        ),
        "normal UVM residency distinguishes generic migration from slot mapping",
    )
    require(
        bool(validation["slot_mapping_is_not_uvm_gpu_residency"]) == slot_mode
        and int(validation["checksum"]) == int(validation["expected_checksum"])
        and bool(validation["full_gpu_checksum"])
        and bool(validation["va_stable"]),
        "full GPU checksum, slot semantics, and VA stability",
    )
    if slot_mode:
        require(
            int(validation["cpu_last_prefetch_after"]) == allocations,
            "slot copy preserves CPU source LastPrefetchLocation",
        )

    profile = sample["profile"]
    required = {name: int(value) for name, value in profile["required"].items()}
    delta = {name: int(value) for name, value in profile["delta"].items()}
    require(bool(profile["snapshot_after_cpu_reset"]), "profile boundary follows CPU reset")
    require(
        all(name in delta and delta[name] == value for name, value in required.items()),
        "raw driver profile satisfies every required exact counter",
    )
    if slot_mode:
        blocks = delta["slot_direct_blocks"]
        pushes = delta["slot_direct_copy_pushes"]
        methods = delta["slot_direct_memcopy_methods"]
        outcome = measurement["outcome"]
        require(
            blocks > 0
            and methods >= blocks
            and int(outcome["remap"]) == allocations
            and int(outcome["tlb_invalidate"]) == blocks
            and int(outcome["copy_pushes"]) == pushes
            and delta["slot_direct_vidmem_allocations"] == 0
            and delta["slot_direct_vidmem_evictions"] == 0
            and delta["slot_direct_fallbacks"] == 0
            and delta["slot_direct_errors"] == 0,
            "timed slot path maps all ranges and reuses backing without fallback/eviction",
        )
        require(
            all(
                delta[f"slot_direct_profile.{stage}.ns"] > 0
                for stage in ("total", "prepare", "copy", "map", "wait")
            ),
            "all slot-direct profile stages recorded positive time",
        )
        if mode == "slot_per_block_push":
            require(pushes == blocks, "baseline uses one CE push per copied block")
        else:
            require(
                4 <= pushes <= blocks,
                "shared path reports its real capacity-driven push flush count",
            )


def verify_run(path: Path, document: dict[str, Any], require_full: bool) -> dict[str, float]:
    require(document.get("schema_version") == 1 and document.get("status") == "ok", f"{path}: run schema/status")
    config = document["config"]
    samples = document["samples"]
    allocations = int(config["allocations"])
    allocation_bytes = int(config["allocation_bytes"])
    logical_bytes = int(config["logical_bytes"])
    blocks = int(config["blocks"])
    require(
        allocations == 4 * int(config["batch_size"])
        and int(config["batches"]) == 4
        and logical_bytes == allocations * allocation_bytes
        and config["schedule"] == "pairwise_abba_independent_processes",
        f"{path}: four-batch byte geometry and schedule",
    )
    if require_full:
        require(
            config["profile"] == "full"
            and allocations == 512
            and allocation_bytes == 4 * 1024 * 1024
            and int(config["batch_size"]) == 128
            and logical_bytes == 2 * 1024 * 1024 * 1024,
            f"{path}: exact 512x4MiB / 2GiB full geometry",
        )
    require(
        int(config["slot_vidmem_budget_bytes"]) == 0
        or int(config["slot_vidmem_budget_bytes"]) >= logical_bytes,
        f"{path}: slot budget can hold the complete working set",
    )
    require(len(samples) == blocks * 12, f"{path}: complete pairwise ABBA sample count")
    require(
        int(document["unique_process_count"]) == len(samples)
        and len({int(sample["pid"]) for sample in samples}) == len(samples),
        f"{path}: every measured sample ran in an independent process",
    )

    for block_index in range(blocks):
        selected = sorted(
            (sample for sample in samples if int(sample["schedule"]["block_index"]) == block_index),
            key=lambda sample: int(sample["schedule"]["global_index"]),
        )
        expected = runner.abba_schedule(block_index)
        observed = [
            (
                sample["schedule"]["pair"],
                int(sample["schedule"]["pair_slot"]),
                sample["mode"],
            )
            for sample in selected
        ]
        require(observed == expected, f"{path}: block {block_index} exact ABBA/BAAB order")
    for mode in MODES:
        require(
            sum(sample["mode"] == mode for sample in samples) == blocks * 4,
            f"{path}: balanced {mode} count",
        )
    for sample in samples:
        verify_sample(sample, config)

    aggregate = document["aggregate"]
    medians: dict[str, float] = {}
    push_medians: dict[str, float] = {}
    for mode in MODES:
        values = [float(sample["measurement"]["wall_ms"]) for sample in samples if sample["mode"] == mode]
        median = statistics.median(values)
        medians[mode] = median
        saved = aggregate["modes"][mode]
        require(
            int(saved["samples"]) == blocks * 4
            and math.isclose(float(saved["wall_ms_p50"]), median, rel_tol=1e-12),
            f"{path}: {mode} aggregate median",
        )
        if mode == "driver_4x128_sync":
            require(
                saved["copy_ms_p50"] is None
                and saved["slot_map_ms_p50"] is None
                and saved["copy_pushes_p50"] is None
                and saved["memcopy_methods_p50"] is None,
                f"{path}: generic aggregate keeps slot-only fields null",
            )
        else:
            copy_values = [
                float(sample["measurement"]["copy_ms"])
                for sample in samples
                if sample["mode"] == mode
            ]
            map_values = [
                float(sample["measurement"]["slot_map_ms"])
                for sample in samples
                if sample["mode"] == mode
            ]
            push_values = [
                int(sample["profile"]["delta"]["slot_direct_copy_pushes"])
                for sample in samples
                if sample["mode"] == mode
            ]
            push_medians[mode] = statistics.median(push_values)
            require(
                math.isclose(
                    float(saved["copy_ms_p50"]),
                    statistics.median(copy_values),
                    rel_tol=1e-12,
                )
                and math.isclose(
                    float(saved["slot_map_ms_p50"]),
                    statistics.median(map_values),
                    rel_tol=1e-12,
                )
                and math.isclose(
                    float(saved["copy_pushes_p50"]),
                    push_medians[mode],
                    rel_tol=1e-12,
                ),
                f"{path}: {mode} copy/map/push aggregates",
            )
    comparisons = aggregate["comparisons"]
    expected_shared_per_block = medians[MODES[1]] / medians[MODES[2]]
    expected_shared_driver = medians[MODES[0]] / medians[MODES[2]]
    require(
        math.isclose(
            float(comparisons["slot_shared_vs_slot_per_block_speedup"]),
            expected_shared_per_block,
            rel_tol=1e-12,
        )
        and math.isclose(
            float(comparisons["slot_shared_vs_driver_batch_speedup"]),
            expected_shared_driver,
            rel_tol=1e-12,
        )
        and math.isclose(
            float(comparisons["slot_shared_push_count_reduction"]),
            push_medians[MODES[1]] / push_medians[MODES[2]],
            rel_tol=1e-12,
        ),
        f"{path}: saved speedups/push reduction match raw independent samples",
    )
    return {
        "shared_vs_per_block": expected_shared_per_block,
        "shared_vs_driver": expected_shared_driver,
    }


def main() -> int:
    args = parse_args()
    ratios = []
    try:
        for requested in args.runs:
            path, document = load(requested)
            ratios.append(verify_run(path, document, args.require_full))
        print(
            "VERIFIED "
            f"runs={len(ratios)} "
            f"shared_vs_per_block_median={statistics.median(row['shared_vs_per_block'] for row in ratios):.4f} "
            f"shared_vs_driver_median={statistics.median(row['shared_vs_driver'] for row in ratios):.4f}"
        )
        return 0
    except Exception as error:  # noqa: BLE001 - verification boundary.
        print(f"FAIL {error}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
