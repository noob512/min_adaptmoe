#!/usr/bin/env python3
"""Strictly verify saved slot-PTE/CE benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping

import run_benchmark as runner


MIB = 1024 * 1024
GIB = 1024 * MIB
VA_BLOCK_BYTES = 2 * MIB
MODES = runner.MODES
GENERIC_COUNTERS = (
    "prefetch_gpu_batch_fast_ioctl_calls",
    "prefetch_gpu_batch_fast_core_calls",
    "prefetch_gpu_batch_fast_requested_ranges",
    "prefetch_gpu_batch_fast_completed_ranges",
    "prefetch_gpu_batch_fast_requested_bytes",
    "prefetch_gpu_batch_fast_completed_bytes",
    "prefetch_gpu_batch_fast_waits",
    "prefetch_gpu_batch_fast_fallbacks",
    "prefetch_gpu_batch_fast_errors",
)
SLOT_PROFILE_STAGES = (
    "total",
    "prepare",
    "unmap",
    "copy",
    "map",
    "wait",
    "prepare_complete",
    "copy_complete",
    "map_complete",
)
SLOT_COUNTERS = (
    "slot_direct_ioctl_calls",
    "slot_direct_per_block_calls",
    "slot_direct_shared_calls",
    "slot_direct_successes",
    "slot_direct_requested_ranges",
    "slot_direct_completed_ranges",
    "slot_direct_requested_bytes",
    "slot_direct_completed_bytes",
    "slot_direct_blocks",
    "slot_direct_copy_pushes",
    "slot_direct_memcopy_methods",
    "slot_direct_tracker_waits",
    "slot_direct_vidmem_allocations",
    "slot_direct_vidmem_evictions",
    "slot_direct_pte_map_ranges",
    "slot_direct_pte_map_bytes",
    "slot_direct_pte_4k_blocks",
    "slot_direct_pte_64k_blocks",
    "slot_direct_pte_2m_blocks",
    "slot_direct_pte_write_entries",
    "slot_direct_skipped_unmap_blocks",
    "slot_direct_stage_sync_calls",
    "slot_direct_fallbacks",
    "slot_direct_errors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument(
        "--pinned-result",
        type=Path,
        help="optional matching 2GiB/512-range pinned-copy result",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(requested: Path) -> tuple[Path, dict[str, Any]]:
    path = requested / "summary.json" if requested.is_dir() else requested
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load {path}: {error}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"{path}: top-level JSON is not an object")
    return path, document


def verify_pinned_result(path: Path) -> float:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(f"cannot load pinned result {path}: {error}") from error

    result_lines = [line for line in lines if line.startswith("RESULT ")]
    require(len(result_lines) == 1, f"{path}: exactly one filtered RESULT row")
    fields: dict[str, str] = {}
    for token in result_lines[0].split()[1:]:
        require("=" in token, f"{path}: malformed RESULT token {token!r}")
        name, value = token.split("=", 1)
        fields[name] = value

    require(
        fields.get("layout") == "multi_alloc"
        and fields.get("backend") == "pinned_copy"
        and fields.get("mode") == "tensor_batch"
        and int(fields["bytes"]) == 2 * GIB
        and int(fields["logical_ranges"]) == 512
        and int(fields["submitted_ranges"]) == 512
        and int(fields["functional_ok"]) == 1
        and int(fields["in_place_va"]) == 0,
        f"{path}: matching vLLM-style 2GiB/512-range pinned-copy contract",
    )
    require(
        any(
            line.startswith("benchmark_result ")
            and "functional_success=1" in line.split()
            and "filtered=1" in line.split()
            for line in lines
        ),
        f"{path}: filtered pinned benchmark completed successfully",
    )

    migration_ms = float(fields["migration_ms_p50"])
    migration_gib_s = float(fields["migration_gib_s"])
    require(
        math.isfinite(migration_ms)
        and migration_ms > 0
        and math.isfinite(migration_gib_s)
        and migration_gib_s > 0,
        f"{path}: positive pinned-copy timing and bandwidth",
    )
    computed_gib_s = 2.0 / (migration_ms / 1000.0)
    require(
        math.isclose(migration_gib_s, computed_gib_s, rel_tol=5e-5),
        f"{path}: pinned bandwidth is consistent with 2GiB/time",
    )
    return migration_gib_s


def residency_equals(
    value: Mapping[str, Any], *, ranges: int, cpu: int, gpu: int, any_: int
) -> bool:
    return (
        int(value["ranges"]) == ranges
        and int(value["cpu_bytes"]) == cpu
        and int(value["gpu_bytes"]) == gpu
        and int(value["any_bytes"]) == any_
        and int(value["unknown_bytes"]) == 0
    )


def verify_outcome(
    value: Mapping[str, Any], allocations: int, logical_bytes: int, *, transfer: bool
) -> None:
    require(
        int(value["calls"]) == 4
        and int(value["success"]) == allocations
        and int(value["failure"]) == 0
        and int(value["conflict"]) == 0
        and int(value["requested_bytes"]) == logical_bytes
        and int(value["accepted_bytes"]) == logical_bytes
        and int(value["bad_entries"]) == 0,
        "four synchronous ioctl batches completed every range",
    )
    if not transfer:
        require(
            int(value["remap"]) == 0
            and int(value["tlb_invalidate"]) == 0
            and int(value["copy_pushes"]) == 0,
            "mode-8 preallocation did not copy or map",
        )


def expected_grain(mode: str, blocks: int, delta: Mapping[str, int]) -> tuple[int, int, int]:
    if mode in ("slot_4k_baseline", "stage_sync_4k_ce"):
        return blocks, 0, 0
    if mode == "slot_64k":
        return 0, blocks, 0
    if mode == "slot_2m":
        return 0, 0, blocks
    observed = (
        int(delta["slot_direct_pte_4k_blocks"]),
        int(delta["slot_direct_pte_64k_blocks"]),
        int(delta["slot_direct_pte_2m_blocks"]),
    )
    require(
        all(value >= 0 for value in observed) and sum(observed) == blocks,
        "overwrite mode classified every VA block by existing PTE grain",
    )
    return observed


def verify_sample(sample: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    mode = str(sample["mode"])
    require(
        int(sample["schema_version"]) == 1
        and sample["status"] == "ok"
        and mode in MODES,
        "sample schema/status/mode",
    )
    allocations = int(config["allocations"])
    allocation_bytes = int(config["allocation_bytes"])
    logical_bytes = int(config["logical_bytes"])
    blocks = int(config["expected_va_blocks"])
    pages = logical_bytes // 4096
    require(
        sample["config"]
        == {
            "device": config["device"],
            "allocations": allocations,
            "allocation_bytes": allocation_bytes,
            "guard_bytes": config["guard_bytes"],
            "batch_size": config["batch_size"],
            "batches": 4,
            "logical_bytes": logical_bytes,
            "expected_va_blocks": blocks,
            "warmups": config["worker_warmups"],
            "slot_layer_id": config["slot_layer_id"],
            "slot_expert_id": config["slot_expert_id"],
        },
        "sample geometry matches its run",
    )

    abi = sample["abi"]
    expected_abi = {
        "generic_ioctl": 90,
        "generic_max_ranges": 128,
        "generic_params_size": 9280,
        "slot_ioctl": 91,
        "slot_max_entries": 128,
        "slot_params_size": 6224,
        "slot_mode_prealloc": 8,
        "slot_mode_slot_4k_baseline": 10,
        "slot_mode_slot_64k": 12,
        "slot_mode_slot_2m": 13,
        "slot_mode_direct_overwrite_existing_grain": 14,
        "slot_mode_stage_sync_4k_ce": 15,
    }
    require(
        {name: int(abi[name]) for name in expected_abi} == expected_abi,
        "versioned ioctl 90/91 and mode 8/10/12/13/14/15 ABI",
    )

    layout = sample["layout"]
    va_snapshot = [int(value, 16) for value in layout["va_snapshot"]]
    require(
        int(layout["adjacent_test_ranges"]) == 0
        and int(layout["min_gap_bytes"]) >= int(config["guard_bytes"])
        and bool(layout["all_regions_2m_aligned"])
        and len(va_snapshot) == allocations
        and len(set(va_snapshot)) == allocations
        and all(value % VA_BLOCK_BYTES == 0 for value in va_snapshot),
        "all 4MiB regions are unique, guard-separated, and 2MiB aligned",
    )

    setup = sample["setup"]
    require(
        bool(setup["slot_vidmem_preallocated_before_timing"]),
        "mode-8 slot preallocation is outside timing",
    )
    verify_outcome(setup["prealloc"], allocations, logical_bytes, transfer=False)
    prime = setup["uniform_old_pte_prime"]
    for reset_name in ("cpu_reset_before", "cpu_reset_after"):
        prime_reset = prime[reset_name]
        require(
            int(prime_reset["accepted"]) == allocations
            and int(prime_reset["last_prefetch_cpu_ranges"]) == allocations
            and bool(prime_reset["va_stable"])
            and residency_equals(
                prime_reset["source_residency"],
                ranges=allocations,
                cpu=logical_bytes,
                gpu=0,
                any_=logical_bytes,
            ),
            f"uniform old-PTE prime {reset_name} has complete CPU residency",
        )
    require(
        int(prime["checksum"]) == int(prime["expected_checksum"])
        and math.isfinite(float(prime["checksum_ms"]))
        and float(prime["checksum_ms"]) > 0,
        "uniform prime performed a full valid GPU read before the final CPU reset",
    )

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
        "all managed sources are CPU-resident immediately before timing",
    )

    measurement = sample["measurement"]
    outcome = measurement["outcome"]
    verify_outcome(outcome, allocations, logical_bytes, transfer=True)
    require(
        int(outcome["remap"]) == allocations
        and int(outcome["tlb_invalidate"]) == blocks
        and int(outcome["copy_pushes"]) == blocks,
        "timed ioctl mapped every range with one push per VA block",
    )
    require(
        bool(measurement["cpu_reset_immediately_before_profile_and_timing"])
        and bool(measurement["host_monotonic_to_synchronous_ioctl_completion"])
        and int(measurement["wall_ns"]) > 0
        and math.isfinite(float(measurement["wall_ms"]))
        and float(measurement["wall_ms"]) > 0
        and math.isfinite(float(measurement["gib_s"]))
        and float(measurement["gib_s"]) > 0
        and measurement["component_source"] == "slot_direct_profile"
        and bool(measurement["submit_stages_are_not_ce_completion"]),
        "e2e and submit-stage timings have explicit valid semantics",
    )
    for field in (
        "driver_profile_total_ms",
        "prepare_ms",
        "copy_ms",
        "map_ms",
        "wait_ms",
    ):
        require(
            math.isfinite(float(measurement[field]))
            and float(measurement[field]) > 0,
            f"positive {field}",
        )
    require(
        math.isfinite(float(measurement["unmap_ms"]))
        and (
            float(measurement["unmap_ms"]) == 0
            if mode == "direct_overwrite_existing_grain"
            else float(measurement["unmap_ms"]) > 0
        ),
        "separate-unmap timing is zero only for direct overwrite",
    )
    complete_fields = (
        "prepare_complete_ms",
        "copy_complete_ms",
        "map_complete_ms",
    )
    if mode == "stage_sync_4k_ce":
        require(
            bool(measurement["stage_sync_completion_boundaries"])
            and all(
                value is not None
                and math.isfinite(float(value))
                and float(value) > 0
                for value in (measurement[field] for field in complete_fields)
            ),
            "mode 15 exposes all three completion-boundary timings",
        )
    else:
        require(
            not bool(measurement["stage_sync_completion_boundaries"])
            and all(measurement[field] is None for field in complete_fields),
            "non-stage-sync modes do not fabricate completion timings",
        )

    validation = sample["validation"]
    require(
        residency_equals(
            validation["post_residency"],
            ranges=allocations,
            cpu=logical_bytes,
            gpu=0,
            any_=logical_bytes,
        )
        and residency_equals(
            validation["post_checksum_residency"],
            ranges=allocations,
            cpu=logical_bytes,
            gpu=0,
            any_=logical_bytes,
        ),
        "slot PTEs preserve CPU UVM residency before and after checksum",
    )
    require(
        bool(validation["slot_mapping_is_not_uvm_gpu_residency"])
        and int(validation["cpu_last_prefetch_after"]) == allocations
        and int(validation["checksum"]) == int(validation["expected_checksum"])
        and bool(validation["full_gpu_checksum"])
        and math.isfinite(float(validation["checksum_ms"]))
        and float(validation["checksum_ms"]) > 0
        and bool(validation["va_stable"]),
        "full original-VA GPU checksum, CPU location, and VA stability",
    )

    profile = sample["profile"]
    required = {name: int(value) for name, value in profile["required"].items()}
    delta = {name: int(value) for name, value in profile["delta"].items()}
    require(
        bool(profile["snapshot_after_cpu_reset"])
        and profile["first_boundary"] == profile["timing_boundary"],
        "unchanged consecutive profile baseline after CPU reset",
    )
    require(
        all(name in delta and delta[name] == value for name, value in required.items()),
        "raw driver profile satisfies every required exact field",
    )
    require(
        delta["migrate_profile.api_total.calls"] == 0
        and all(delta[name] == 0 for name in GENERIC_COUNTERS),
        "timed path used neither generic migrate nor ioctl-90 batch prefetch",
    )
    require(all(name in delta for name in SLOT_COUNTERS), "all slot telemetry fields exist")

    common = {
        "slot_direct_ioctl_calls": 4,
        "slot_direct_per_block_calls": 4,
        "slot_direct_shared_calls": 0,
        "slot_direct_successes": 4,
        "slot_direct_requested_ranges": allocations,
        "slot_direct_completed_ranges": allocations,
        "slot_direct_requested_bytes": logical_bytes,
        "slot_direct_completed_bytes": logical_bytes,
        "slot_direct_blocks": blocks,
        "slot_direct_copy_pushes": blocks,
        "slot_direct_memcopy_methods": blocks,
        "slot_direct_vidmem_allocations": 0,
        "slot_direct_vidmem_evictions": 0,
        "slot_direct_pte_map_ranges": allocations,
        "slot_direct_pte_map_bytes": logical_bytes,
        "slot_direct_fallbacks": 0,
        "slot_direct_errors": 0,
    }
    require(
        all(delta[name] == value for name, value in common.items()),
        "slot path completed exact bytes/blocks without allocation, eviction, or fallback",
    )
    stage_sync = mode == "stage_sync_4k_ce"
    require(
        delta["slot_direct_tracker_waits"] == (16 if stage_sync else 4)
        and delta["slot_direct_stage_sync_calls"] == (4 if stage_sync else 0)
        and delta["slot_direct_skipped_unmap_blocks"]
        == (blocks if mode == "direct_overwrite_existing_grain" else 0),
        "tracker-wait, stage-sync, and skipped-unmap counters match the mode",
    )
    pte_4k, pte_64k, pte_2m = expected_grain(mode, blocks, delta)
    require(
        (
            delta["slot_direct_pte_4k_blocks"],
            delta["slot_direct_pte_64k_blocks"],
            delta["slot_direct_pte_2m_blocks"],
        )
        == (pte_4k, pte_64k, pte_2m),
        "PTE grain counters match the selected mode",
    )
    expected_entries = pte_4k * 512 + pte_64k * 32 + pte_2m
    require(
        delta["slot_direct_pte_write_entries"] == expected_entries,
        "PTE write-entry count is exactly grain-weighted",
    )

    for stage in SLOT_PROFILE_STAGES:
        prefix = f"slot_direct_profile.{stage}"
        complete = stage.endswith("_complete")
        enabled = not complete or stage_sync
        expected_positive_ns = not (
            stage == "unmap" and mode == "direct_overwrite_existing_grain"
        )
        require(
            delta[f"{prefix}.calls"] == (4 if enabled else 0)
            and delta[f"{prefix}.pages"] == (pages if enabled else 0)
            and delta[f"{prefix}.bytes"] == (logical_bytes if enabled else 0)
            and (
                (
                    delta[f"{prefix}.ns"] > 0
                    if expected_positive_ns
                    else delta[f"{prefix}.ns"] == 0
                )
                if enabled
                else delta[f"{prefix}.ns"] == 0
            ),
            f"exact profile contract for {stage}",
        )


def compare_tree(observed: Any, expected: Any, path: str = "aggregate") -> None:
    if isinstance(expected, dict):
        require(isinstance(observed, dict), f"{path}: expected object")
        require(set(observed) == set(expected), f"{path}: key set mismatch")
        for key in expected:
            compare_tree(observed[key], expected[key], f"{path}.{key}")
        return
    if expected is None:
        require(observed is None, f"{path}: expected null")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(
            isinstance(observed, (int, float))
            and math.isclose(float(observed), float(expected), rel_tol=1e-12, abs_tol=0.0),
            f"{path}: {observed}/{expected}",
        )
        return
    require(observed == expected, f"{path}: {observed!r}/{expected!r}")


def verify_run(
    path: Path, document: dict[str, Any], require_full: bool
) -> dict[str, float]:
    require(
        document.get("schema_version") == 1 and document.get("status") == "ok",
        f"{path}: run schema/status",
    )
    config = document["config"]
    samples = document["samples"]
    profile = str(config["profile"])
    allocations = int(config["allocations"])
    allocation_bytes = int(config["allocation_bytes"])
    guard_bytes = int(config["guard_bytes"])
    batch_size = int(config["batch_size"])
    logical_bytes = int(config["logical_bytes"])
    blocks = int(config["blocks"])
    expected_va_blocks = logical_bytes // VA_BLOCK_BYTES

    expected_geometry = (
        (4, 4 * MIB, 4 * MIB, 1, 16 * MIB, 8)
        if profile == "smoke"
        else (512, 4 * MIB, 4 * MIB, 128, 2 * 1024 * MIB, 1024)
    )
    require(profile in ("smoke", "full"), f"{path}: known profile")
    require(
        (
            allocations,
            allocation_bytes,
            guard_bytes,
            batch_size,
            logical_bytes,
            expected_va_blocks,
        )
        == expected_geometry,
        f"{path}: exact {profile} geometry with complete 2MiB VA blocks",
    )
    if require_full:
        require(profile == "full", f"{path}: full profile required")
    require(
        int(config["batches"]) == 4
        and int(config["expected_va_blocks"]) == expected_va_blocks
        and blocks > 0
        and int(config["pairs_per_block"]) == 10
        and int(config["samples_per_mode"]) == blocks * 8
        and int(config["samples_per_block"]) == 40
        and config["schedule"] == "all_pairwise_abba_independent_processes"
        and tuple(config["modes"]) == MODES,
        f"{path}: four-batch/all-pairwise schedule contract",
    )
    require(
        int(config["slot_vidmem_budget_bytes"]) == 0
        or int(config["slot_vidmem_budget_bytes"]) >= logical_bytes,
        f"{path}: slot budget covers the complete working set",
    )
    require(
        len(samples) == blocks * 40
        and int(document["sample_count"]) == len(samples),
        f"{path}: complete all-pairwise ABBA sample count",
    )
    require(
        int(document["unique_process_count"]) == len(samples)
        and len({int(sample["pid"]) for sample in samples}) == len(samples),
        f"{path}: every sample ran in a fresh process",
    )

    global_indices = sorted(int(sample["schedule"]["global_index"]) for sample in samples)
    require(
        global_indices == list(range(len(samples))),
        f"{path}: contiguous unique global sample indices",
    )
    for block_index in range(blocks):
        selected = sorted(
            (
                sample
                for sample in samples
                if int(sample["schedule"]["block_index"]) == block_index
            ),
            key=lambda sample: int(sample["schedule"]["global_index"]),
        )
        expected = runner.pairwise_abba_schedule(block_index)
        observed = [
            (
                sample["schedule"]["pair"],
                int(sample["schedule"]["pair_slot"]),
                sample["mode"],
            )
            for sample in selected
        ]
        require(observed == expected, f"{path}: block {block_index} exact pairwise ABBA")
    for mode in MODES:
        require(
            sum(sample["mode"] == mode for sample in samples) == blocks * 8,
            f"{path}: balanced {mode} sample count",
        )
    for sample in samples:
        verify_sample(sample, config)

    require(
        len({int(sample["validation"]["expected_checksum"]) for sample in samples}) == 1,
        f"{path}: identical dataset checksum contract across fresh processes",
    )
    expected_aggregate = runner.aggregate(list(samples))
    compare_tree(document["aggregate"], expected_aggregate)
    return {
        name: float(value)
        for name, value in expected_aggregate["comparisons"].items()
        if name.endswith("_speedup")
    }


def main() -> int:
    args = parse_args()
    ratios: list[dict[str, float]] = []
    ce_gib_s_values: list[float] = []
    logical_bytes_values: list[int] = []
    try:
        for requested in args.runs:
            path, document = load(requested)
            ratios.append(verify_run(path, document, args.require_full))
            logical_bytes = int(document["config"]["logical_bytes"])
            logical_bytes_values.append(logical_bytes)
            logical_gib = float(logical_bytes) / GIB
            copy_complete_ms = float(
                document["aggregate"]["modes"]["stage_sync_4k_ce"][
                    "copy_complete_ms_p50"
                ]
            )
            ce_gib_s_values.append(logical_gib / (copy_complete_ms / 1000.0))
        keys = sorted(ratios[0])
        rendered = " ".join(
            f"{key}={statistics.median(row[key] for row in ratios):.4f}"
            for key in keys
        )
        ce_gib_s = statistics.median(ce_gib_s_values)
        extra = f" ce_copy_gib_s={ce_gib_s:.3f}"
        if args.pinned_result:
            require(
                all(value == 2 * GIB for value in logical_bytes_values),
                "--pinned-result requires only matching 2GiB runs",
            )
            pinned_gib_s = verify_pinned_result(args.pinned_result)
            extra += (
                f" pinned_copy_gib_s={pinned_gib_s:.3f}"
                f" ce_vs_pinned={ce_gib_s / pinned_gib_s:.4f}x"
            )
        print(f"VERIFIED runs={len(ratios)} {rendered}{extra}")
        return 0
    except Exception as error:  # noqa: BLE001 - verification boundary.
        print(f"FAIL {error}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
