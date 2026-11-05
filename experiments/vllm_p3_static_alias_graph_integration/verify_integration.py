#!/usr/bin/env python3
"""Verify Qwen L0-L3 static-alias CUDA Graph integration artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PHASE_EXPERIMENT = REPO_ROOT / "experiments" / "vllm_p3_pinned_uva_phase_compare"
sys.path.insert(0, str(PHASE_EXPERIMENT))
import benchmark_phase_steps as phase  # noqa: E402


LOGICAL_BYTES = 2_415_919_104
DUAL_BANK_BYTES = 1_207_959_552
PARAMETER_COUNT = 8
REFERENCE_FIXTURE_SHA256 = (
    "0f28e34b410774573c3f8bdcddaffddd0ac29d50daad09d5d28e0e09c14930a4"
)
REFERENCE_OUTPUT_SHA256 = {
    1: "83dd1cdf2e151863a154ba5ee4cd1dab1b0db5e02ac0e1cd6c97c5a0c7eb7cbf",
    4: "659c552b293f3060934ae3c63414bc4c7d75717617cb89d348f9117e5f9783e0",
}
ARM_NAMES = {
    "E": "eager_pinned",
    "A": "graph_static_alias",
    "D": "graph_direct_static_bank",
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
    if not isinstance(value, dict):
        raise VerificationError(f"{path} does not contain a JSON object")
    return value


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_status(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle, delimiter="\t"))
    except OSError as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    require(bool(records), "execution_status.tsv is empty")
    return records


def validate_graph_delta(
    delta: Any,
    *,
    graph_arm: bool,
    path: str,
) -> None:
    require(isinstance(delta, dict), f"{path}: graph delta is not an object")
    captures = sum(
        int(value) for key, value in delta.items() if str(key).endswith("_capture")
    )
    require(captures == 0, f"{path}: runtime unexpectedly captured a graph: {delta}")
    full_replays = int(delta.get("full_replay", 0))
    other_replays = sum(
        int(value)
        for key, value in delta.items()
        if str(key).endswith("_replay") and key != "full_replay"
    )
    if graph_arm:
        require(full_replays == 1, f"{path}: expected one FULL replay, got {delta}")
        require(other_replays == 0, f"{path}: non-FULL replay observed: {delta}")
    else:
        require(
            full_replays == 0 and other_replays == 0,
            f"{path}: eager arm replayed a graph: {delta}",
        )


def validate_result(
    result: dict[str, Any],
    *,
    symbol: str,
    config: dict[str, Any],
    directory: Path,
) -> tuple[dict[tuple[int, str, int], str], dict[int, dict[str, float]]]:
    label = directory.name
    require(
        result.get("schema") == "vllm_p3_static_alias_graph_steps_v1",
        f"{label}: wrong result schema",
    )
    require(result.get("status") == "ok", f"{label}: result is not ok")
    run_cfg = result.get("config")
    require(isinstance(run_cfg, dict), f"{label}: missing config")
    batch_order = run_cfg.get("batch_sizes")
    require(
        isinstance(batch_order, list)
        and len(batch_order) == 2
        and set(batch_order) == {1, 4},
        f"{label}: expected batch permutation [1,4], got {batch_order}",
    )
    graph_arm = symbol in ("A", "D")
    require(
        run_cfg.get("requested_compilation_mode") == "NONE"
        and run_cfg.get("realized_compilation_mode") == "NONE",
        f"{label}: Dynamo/Inductor compilation was unexpectedly enabled",
    )
    if graph_arm:
        require(
            run_cfg.get("execution_mode") == "full_graph_decode",
            f"{label}: graph arm execution mode differs",
        )
        require(
            run_cfg.get("realized_cudagraph_mode") == "FULL_DECODE_ONLY",
            f"{label}: graph mode was downgraded",
        )
        require(
            run_cfg.get("realized_cudagraph_capture_sizes") == [1, 4],
            f"{label}: graph capture sizes differ",
        )
        counter_delta = result.get("compilation_counter_load_delta", {})
        require(
            int(counter_delta.get("num_cudagraph_captured", 0)) >= 2,
            f"{label}: fewer than two graphs captured: {counter_delta}",
        )
        wrapper_load = result.get("graph_wrapper_calls_after_load", {})
        require(
            int(wrapper_load.get("full_capture", 0)) >= 2,
            f"{label}: wrapper saw fewer than two FULL captures: {wrapper_load}",
        )
    else:
        require(
            run_cfg.get("execution_mode") == "eager"
            and run_cfg.get("realized_cudagraph_mode") == "NONE",
            f"{label}: eager mode contract differs",
        )
        require(
            run_cfg.get("realized_cudagraph_capture_sizes") == [],
            f"{label}: eager arm retained CUDA Graph capture sizes",
        )

    alias_diagnostics = result.get("static_alias_diagnostics_after_load")
    require(
        isinstance(alias_diagnostics, dict),
        f"{label}: missing static-alias diagnostics",
    )
    if symbol == "A":
        require(alias_diagnostics.get("enabled") is True, f"{label}: alias disabled")
        require(
            alias_diagnostics.get("capture_phase") is False,
            f"{label}: capture phase leaked past model load",
        )
        require(
            int(alias_diagnostics.get("arena_count", 0)) == 4,
            f"{label}: expected four static-alias arenas: {alias_diagnostics}",
        )
        require(
            int(alias_diagnostics.get("backing_bytes", 0)) == DUAL_BANK_BYTES,
            f"{label}: diagnostic backing differs: {alias_diagnostics}",
        )
        arenas = alias_diagnostics.get("arenas")
        require(
            isinstance(arenas, list) and len(arenas) == 4,
            f"{label}: arena diagnostics differ",
        )
        require(
            {int(item["bank_id"]) for item in arenas} == {0, 1},
            f"{label}: diagnostic bank IDs differ",
        )
        require(
            len({int(item["upload_va"]) for item in arenas}) == 4
            and sum(len(item["consumer_vas"]) for item in arenas) == 8,
            f"{label}: diagnostic alias VA counts differ",
        )
    else:
        require(
            alias_diagnostics.get("enabled") is False,
            f"{label}: non-alias arm enabled static aliases",
        )

    expected_cycle_count = 2 * (
        int(config["conditioning_repeats"]) + int(config["measurement_repeats"])
    )
    cycles = result.get("cycles")
    require(
        isinstance(cycles, list) and len(cycles) == expected_cycle_count,
        f"{label}: expected {expected_cycle_count} cycles",
    )
    hashes: dict[tuple[int, str, int], str] = {}
    roles: dict[tuple[int, str], set[int]] = defaultdict(set)
    expected_runtime_replays = 0
    for cycle_index, cycle in enumerate(cycles):
        require(isinstance(cycle, dict), f"{label}: cycle {cycle_index} is invalid")
        batch = int(cycle["batch_size"])
        role = str(cycle["role"])
        role_index = int(cycle["role_index"])
        require(batch in (1, 4), f"{label}: unexpected batch {batch}")
        require(role in ("conditioning", "measurement"), f"{label}: bad role")
        roles[(batch, role)].add(role_index)
        phase.validate_prefill(cycle["prefill_schedule"], batch)
        prefill_delta = cycle.get("prefill_graph_wrapper_delta", {})
        require(
            not any(str(key).endswith("_replay") for key in prefill_delta),
            f"{label}: prefill unexpectedly replayed a graph: {prefill_delta}",
        )
        diagnostics_after_prefill = cycle.get(
            "static_alias_diagnostics_after_prefill"
        )
        diagnostics_after_decode = cycle.get(
            "static_alias_diagnostics_after_decode"
        )
        require(
            isinstance(diagnostics_after_prefill, dict)
            and isinstance(diagnostics_after_decode, dict),
            f"{label}: cycle lacks alias diagnostics",
        )
        if symbol == "A":
            require(
                int(diagnostics_after_prefill["forward_call_count"])
                == int(diagnostics_after_decode["forward_call_count"]),
                f"{label}: decode re-entered Python forward hook",
            )
        for category in ("warmup_decode_steps", "measured_decode_steps"):
            steps = cycle.get(category)
            require(isinstance(steps, list), f"{label}: missing {category}")
            expected = int(
                config[
                    "warmup_decode_steps"
                    if category.startswith("warmup")
                    else "measured_decode_steps"
                ]
            )
            require(len(steps) == expected, f"{label}: {category} count differs")
            for step_index, step in enumerate(steps):
                phase.validate_decode(step["schedule"], batch)
                validate_graph_delta(
                    step.get("graph_wrapper_delta", {}),
                    graph_arm=graph_arm,
                    path=f"{label}/{batch}/{role}/{category}/{step_index}",
                )
                expected_runtime_replays += int(graph_arm)

        outputs = cycle.get("outputs")
        require(isinstance(outputs, list) and len(outputs) == batch, f"{label}: outputs")
        token_ids = [item.get("output_token_ids") for item in outputs]
        digest = sha256_json(token_ids)
        require(
            digest == cycle.get("output_token_ids_sha256"),
            f"{label}: output hash does not match serialized tokens",
        )
        hashes[(batch, role, role_index)] = digest

    for batch in (1, 4):
        require(
            roles[(batch, "conditioning")]
            == set(range(int(config["conditioning_repeats"]))),
            f"{label}: conditioning repeat indexes differ for batch {batch}",
        )
        require(
            roles[(batch, "measurement")]
            == set(range(int(config["measurement_repeats"]))),
            f"{label}: measurement repeat indexes differ for batch {batch}",
        )

    final_wrapper = result.get("graph_wrapper_calls_final", {})
    if graph_arm:
        load_wrapper = result.get("graph_wrapper_calls_after_load", {})
        runtime_full_replays = int(final_wrapper.get("full_replay", 0)) - int(
            load_wrapper.get("full_replay", 0)
        )
        require(
            runtime_full_replays == expected_runtime_replays,
            f"{label}: replay aggregate {runtime_full_replays} != "
            f"step evidence {expected_runtime_replays}",
        )

    evidence = result.get("offload_manifest_evidence")
    require(isinstance(evidence, dict), f"{label}: no offload manifest evidence")
    require(
        int(evidence.get("parameter_record_count", -1)) == PARAMETER_COUNT,
        f"{label}: expected {PARAMETER_COUNT} offloaded parameters: {evidence}",
    )
    require(
        int(evidence.get("parameter_logical_bytes", -1)) == LOGICAL_BYTES,
        f"{label}: logical offload bytes differ: {evidence}",
    )
    if symbol == "A":
        require(
            int(evidence.get("graph_static_alias_parameter_count", 0))
            == PARAMETER_COUNT,
            f"{label}: missing graph-static-alias records",
        )
        require(
            evidence.get("bank_ids") == [0, 1],
            f"{label}: expected parity banks 0 and 1: {evidence}",
        )
        require(
            evidence.get("layer_indices") == [0, 1, 2, 3],
            f"{label}: layer mapping differs: {evidence}",
        )
        require(
            len(evidence.get("upload_vas", [])) == 4,
            f"{label}: expected four upload mappings: {evidence}",
        )
        require(
            len(evidence.get("consumer_vas", [])) == 8,
            f"{label}: expected eight consumer mappings: {evidence}",
        )
        require(
            len(evidence.get("pinned_source_vas", [])) == 8,
            f"{label}: expected eight stable pinned sources: {evidence}",
        )
        require(
            int(evidence.get("unique_physical_backing_bytes", -1))
            == DUAL_BANK_BYTES,
            f"{label}: backing is not exactly 1.125 GiB: {evidence}",
        )
        require(
            evidence.get("upload_access_values") == ["rw"]
            and evidence.get("consumer_access_values") == ["ro"],
            f"{label}: alias access protections differ: {evidence}",
        )
        require(
            int(evidence.get("max_graph_static_alias_total_backing_bytes", 0))
            == DUAL_BANK_BYTES,
            f"{label}: cumulative alias backing differs: {evidence}",
        )
        if bool(config.get("copy_trace")):
            capture_counts = evidence.get("forward_capture_active_counts", {})
            capture_phase_counts = evidence.get(
                "forward_capture_phase_counts", {}
            )
            require(
                int(capture_counts.get("true", 0)) > 0,
                f"{label}: trace has no capture-time forwards: {evidence}",
            )
            require(
                int(capture_phase_counts.get("true", 0)) > 0,
                f"{label}: trace has no lifecycle-marked capture forwards",
            )
            require(
                int(evidence.get("max_forward_call_index", 0)) > 0,
                f"{label}: trace has no forward call index",
            )
    else:
        require(
            int(evidence.get("graph_static_alias_record_count", 0)) == 0,
            f"{label}: non-alias arm emitted alias records",
        )

    summaries: dict[int, dict[str, float]] = {}
    phase_summary = result.get("phase_summary")
    require(isinstance(phase_summary, dict), f"{label}: no phase summary")
    for batch in (1, 4):
        summary = phase_summary.get(str(batch))
        require(isinstance(summary, dict), f"{label}: no summary for batch {batch}")
        values = {
            "prefill_step_ms": float(summary["prefill_step_ms"]["median"]),
            "decode_step_ms": float(summary["decode_repeat_median_ms"]["median"]),
            "cycle_e2e_ms": float(summary["cycle_e2e_ms"]["median"]),
        }
        require(
            all(math.isfinite(value) and value > 0 for value in values.values()),
            f"{label}: invalid timing summary {values}",
        )
        summaries[batch] = values
    return hashes, summaries


def verify(run_root: Path) -> dict[str, Any]:
    config = load_json(run_root / "run_config.json")
    require(
        config.get("schema") == "vllm_p3_static_alias_graph_run_v1",
        "wrong run schema",
    )
    require(config.get("batch_sizes") == [1, 4], "run batches differ")
    require(config.get("cudagraph_capture_sizes") == [1, 4], "capture sizes differ")
    require(config.get("module_cycle") is False, "module cycle must remain disabled")
    require(config.get("stock_uvm_required") is True, "stock UVM contract missing")
    require(int(config.get("logical_bytes_per_forward", 0)) == LOGICAL_BYTES, "bytes")
    require(
        int(config.get("dual_bank_backing_bytes", 0)) == DUAL_BANK_BYTES,
        "dual-bank bytes differ",
    )

    status_records = read_status(run_root / "execution_status.tsv")
    order = str(config["order"])
    require(len(status_records) == len(order), "execution row count differs from order")
    required_symbols = set(str(config["required_arms"]))
    results_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_hashes: dict[tuple[int, str, int], str] | None = None
    process_summaries: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    fixture_hash: str | None = None

    for index, (record, expected_symbol) in enumerate(zip(status_records, order), 1):
        symbol = record.get("symbol", "")
        require(symbol == expected_symbol, f"row {index}: symbol/order mismatch")
        require(symbol in ARM_NAMES, f"row {index}: unknown symbol {symbol!r}")
        require(record.get("arm") == ARM_NAMES[symbol], f"row {index}: arm name differs")
        require(record.get("returncode") == "0", f"row {index}: process failed")
        directory = run_root / str(record["directory"])
        result = load_json(directory / "result.json")
        hashes, summaries = validate_result(
            result, symbol=symbol, config=config, directory=directory
        )
        if all_hashes is None:
            all_hashes = hashes
        else:
            require(
                hashes == all_hashes,
                f"{directory.name}: token hashes differ across mechanisms/processes",
            )
        current_fixture_hash = str(result.get("fixture", {}).get("sha256", ""))
        if fixture_hash is None:
            fixture_hash = current_fixture_hash
        require(
            current_fixture_hash == fixture_hash and len(current_fixture_hash) == 64,
            f"{directory.name}: fixture hash differs",
        )
        results_by_symbol[symbol].append(result)
        for batch, summary in summaries.items():
            process_summaries[symbol][batch].append(summary)

    require(required_symbols <= set(results_by_symbol), "one or more required arms absent")
    if (
        fixture_hash == REFERENCE_FIXTURE_SHA256
        and int(config["warmup_decode_steps"]) == 5
        and int(config["measured_decode_steps"]) == 16
    ):
        assert all_hashes is not None
        for (batch, role, role_index), digest in all_hashes.items():
            require(
                digest == REFERENCE_OUTPUT_SHA256[batch],
                "published Qwen output hash changed for "
                f"batch={batch} {role}#{role_index}: {digest}",
            )
    minimum_processes = 3 if config.get("mode") == "performance" else 1
    for symbol in required_symbols:
        require(
            len(results_by_symbol[symbol]) >= minimum_processes,
            f"arm {symbol} has fewer than {minimum_processes} independent processes",
        )

    aggregate: dict[str, dict[str, Any]] = {}
    for symbol, batches in sorted(process_summaries.items()):
        aggregate[symbol] = {}
        for batch, samples in sorted(batches.items()):
            aggregate[symbol][str(batch)] = {
                metric: {
                    "process_count": len(samples),
                    "process_medians": [float(item[metric]) for item in samples],
                    "median": statistics.median(float(item[metric]) for item in samples),
                }
                for metric in ("prefill_step_ms", "decode_step_ms", "cycle_e2e_ms")
            }

    comparisons: dict[str, Any] = {}
    if "E" in aggregate and "A" in aggregate:
        for batch in (1, 4):
            eager = aggregate["E"][str(batch)]
            alias = aggregate["A"][str(batch)]
            comparisons[str(batch)] = {
                metric: {
                    "eager_ms": eager[metric]["median"],
                    "graph_alias_ms": alias[metric]["median"],
                    "graph_over_eager_ratio": (
                        alias[metric]["median"] / eager[metric]["median"]
                    ),
                    "graph_speedup_percent": (
                        100.0
                        * (eager[metric]["median"] - alias[metric]["median"])
                        / eager[metric]["median"]
                    ),
                }
                for metric in ("prefill_step_ms", "decode_step_ms", "cycle_e2e_ms")
            }

    return {
        "schema": "vllm_p3_static_alias_graph_verification_v1",
        "status": "ok",
        "run_root": str(run_root.resolve()),
        "mode": config["mode"],
        "process_count": len(status_records),
        "process_counts_by_symbol": {
            symbol: len(items) for symbol, items in sorted(results_by_symbol.items())
        },
        "fixture_sha256": fixture_hash,
        "token_hashes": {
            f"batch{batch}/{role}{role_index}": digest
            for (batch, role, role_index), digest in sorted(all_hashes.items())
        },
        "aggregate": aggregate,
        "comparisons": comparisons,
        "performance_is_acceptance_gate": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (args.run_root / "verification.json")
    try:
        result = verify(args.run_root)
    except Exception as exc:
        result = {
            "schema": "vllm_p3_static_alias_graph_verification_v1",
            "status": "failed",
            "run_root": str(args.run_root.resolve()),
            "error": repr(exc),
        }
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
