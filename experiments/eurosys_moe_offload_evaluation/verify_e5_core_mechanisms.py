#!/usr/bin/env python3
"""Verify and summarize the four publication-facing E5 core ablations."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def performance_samples(
    verification: dict[str, Any], *, name: str
) -> dict[int, dict[int, dict[str, Any]]]:
    output: dict[int, dict[int, dict[str, Any]]] = {1: {}, 4: {}}
    audit_count = 0
    for sample in verification.get("samples", []):
        if sample.get("audit_trace") is True:
            audit_count += 1
            continue
        match = re.search(r"_s(\d+)_b(\d+)_", str(sample.get("case", "")))
        require(match is not None, f"{name}: cannot parse performance case")
        repeat, batch = map(int, match.groups())
        require(batch in output, f"{name}: unexpected batch {batch}")
        require(repeat not in output[batch], f"{name}: duplicate B{batch}/r{repeat}")
        output[batch][repeat] = sample
    require(audit_count == 2, f"{name}: expected two audit cases")
    require(set(output[1]) == set(output[4]), f"{name}: repeat sets differ")
    require(len(output[1]) == 3, f"{name}: expected three cold processes")
    return output


def prebuild_summary(on_root: Path, off_root: Path) -> dict[str, Any]:
    on = load(on_root / "verification.json")
    off = load(off_root / "verification.json")
    require(on.get("status") == off.get("status") == "ok", "prebuild verifier")
    on_config, off_config = on.get("run_config", {}), off.get("run_config", {})
    geometry_keys = (
        "model",
        "fixture",
        "batch_sizes",
        "logical_bytes_per_forward",
        "dual_bank_backing_bytes",
        "kv_cache_memory_bytes",
        "cudagraph_mode",
        "cudagraph_capture_sizes",
        "measured_decode_steps",
        "performance_repeats",
        "custom_module_sha256",
        "stock_module_srcversion",
    )
    geometry = {key: on_config.get(key) == off_config.get(key) for key in geometry_keys}
    require(all(geometry.values()), f"prebuild geometry mismatch: {geometry}")
    require(
        on_config.get("static_alias_dual_view_enable") == 1
        and on_config.get("static_alias_dual_view_require_full") == 1,
        "prebuild-on flags",
    )
    require(
        off_config.get("static_alias_dual_view_enable") == 0
        and off_config.get("static_alias_dual_view_require_full") == 0,
        "prebuild-off flags",
    )
    require(on.get("page_table_prebuild", {}).get("enabled") is True, "prebuild on")
    require(off.get("page_table_prebuild", {}).get("enabled") is False, "prebuild off")
    on_samples = performance_samples(on, name="prebuild-on")
    off_samples = performance_samples(off, name="prebuild-off")
    batches: dict[str, Any] = {}
    for batch in (1, 4):
        repeat_ids = sorted(on_samples[batch])
        require(repeat_ids == sorted(off_samples[batch]), "prebuild repeat IDs")
        for repeat in repeat_ids:
            require(
                on_samples[batch][repeat]["output_hash"]
                == off_samples[batch][repeat]["output_hash"],
                f"prebuild output mismatch B{batch}/r{repeat}",
            )
        metrics: dict[str, Any] = {}
        for metric in (
            "prefill_step_ms",
            "steady_decode_step_ms",
            "cutover_wall_ms",
            "resume_wall_ms",
            "repark_wall_ms",
        ):
            on_median = statistics.median(
                float(on_samples[batch][repeat][metric]) for repeat in repeat_ids
            )
            off_median = statistics.median(
                float(off_samples[batch][repeat][metric]) for repeat in repeat_ids
            )
            metrics[metric] = {
                "prebuild_on_median_ms": on_median,
                "prebuild_off_median_ms": off_median,
                "off_over_on_speedup": off_median / on_median,
            }
        batches[str(batch)] = metrics
    return {
        "status": "ok",
        "cold_processes_per_batch": 3,
        "controlled_geometry_equal": True,
        "geometry_checks": geometry,
        "all_outputs_equal": True,
        "batches": batches,
    }


def graph_summary(root: Path) -> dict[str, Any]:
    verification = load(root / "verification.json")
    config = load(root / "run_config.json")
    require(verification.get("status") == "ok", "Graph/eager verifier")
    contract = verification.get("run_contract", {})
    counts = contract.get("arm_counts", {})
    require(counts == {"A": 3, "G": 3, "T": 1, "Z": 3}, "Graph arm counts")
    require(config.get("performance_copy_trace") == 0, "Graph performance trace")
    require(contract.get("common_config_equal") is True, "Graph common config")
    require(contract.get("versions_equal") is True, "Graph version config")
    require(
        int(verification.get("route_evidence", {}).get("trace_file_count", 0)) == 1,
        "route evidence",
    )
    hashes_by_arm: dict[str, dict[int, set[str]]] = {
        "A": {1: set(), 4: set()},
        "G": {1: set(), 4: set()},
    }
    for path in sorted(root.glob("??_alias_*/result.json")):
        result = load(path)
        require(result.get("status") == "ok", f"{path}: result status")
        require(not result.get("wrapper_contract_errors"), f"{path}: wrapper errors")
        require(
            result.get("alias_contract", {}).get("copy_trace_enabled") is False
            and result.get("alias_contract", {}).get("forward_trace_count") == 0,
            f"{path}: per-forward trace not eliminated",
        )
        graph = result.get("arm") == "G"
        arm = str(result.get("arm"))
        require(arm in hashes_by_arm, f"{path}: unexpected arm")
        for cycle in result.get("cycles", []):
            if cycle.get("role") == "measurement":
                hashes_by_arm[arm][int(cycle["batch_size"])].add(
                    str(cycle["output_token_ids_sha256"])
                )
        replay = result.get("graph_contract", {})
        require(
            int(replay.get("prefill_full_replay_count", -1))
            == (int(replay.get("prefill_step_count", 0)) if graph else 0),
            f"{path}: prefill replay",
        )
        require(
            int(replay.get("decode_full_replay_count", -1))
            == (int(replay.get("decode_step_count", 0)) if graph else 0),
            f"{path}: decode replay",
        )
    output_hashes: dict[str, Any] = {}
    for batch in (1, 4):
        require(
            len(hashes_by_arm["A"][batch]) == len(hashes_by_arm["G"][batch]) == 1,
            f"Graph/eager B{batch}: output unstable within execution class",
        )
        eager_hash = next(iter(hashes_by_arm["A"][batch]))
        graph_hash = next(iter(hashes_by_arm["G"][batch]))
        output_hashes[str(batch)] = {
            "eager": eager_hash,
            "graph": graph_hash,
            "cross_execution_equal": eager_hash == graph_hash,
        }
    return {
        "status": "ok",
        "cold_processes_per_arm": 3,
        "performance_copy_trace": False,
        "per_forward_trace_count": 0,
        "output_hashes": output_hashes,
        "prefill_performance": verification["prefill_performance"],
        "decode_performance": verification["decode_performance"],
        "comparisons": verification["comparisons"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-switch", type=Path, required=True)
    parser.add_argument("--stable-va", type=Path, required=True)
    parser.add_argument("--prebuild-on", type=Path, required=True)
    parser.add_argument("--prebuild-off", type=Path, required=True)
    parser.add_argument("--graph-eager", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    direct = load(args.direct_switch / "comparison.json")
    stable = load(args.stable_va / "comparison.json")
    require(direct.get("status") == "ok", "direct-switch comparison")
    require(stable.get("status") == "ok", "stable-VA comparison")
    require(direct.get("process_pairs_per_batch") == 3, "direct-switch repeats")
    require(stable.get("process_pairs_per_generation") == 3, "stable-VA repeats")
    require(direct.get("controlled_geometry_equal") is True, "direct geometry")
    require(stable.get("all_outputs_equal") is True, "stable outputs")
    require(stable.get("all_fault_scans_clean") is True, "stable fault scan")
    require(stable.get("all_resources_released") is True, "stable cleanup")

    payload = {
        "schema": "e5_core_four_mechanism_verification_v1",
        "status": "ok",
        "mechanisms": {
            "direct_pd_page_table_switch": direct,
            "cuda_graph_support": graph_summary(args.graph_eager.resolve()),
            "stable_parameter_va": stable,
            "prebuilt_page_tables": prebuild_summary(
                args.prebuild_on.resolve(), args.prebuild_off.resolve()
            ),
        },
    }
    require(not args.output.exists(), f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
