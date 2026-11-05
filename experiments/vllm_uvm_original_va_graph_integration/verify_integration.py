#!/usr/bin/env python3
"""Fail-closed verifier for the vLLM UVM original-VA experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DUAL_BANK_BYTES = 1_207_959_552
LOGICAL_BYTES = 2_415_919_104
ALIAS_COUNT = 6_912
ALIAS_BYTES = 3_623_878_656
REFERENCE_FIXTURE_SHA256 = (
    "0f28e34b410774573c3f8bdcddaffddd0ac29d50daad09d5d28e0e09c14930a4"
)
REFERENCE_OUTPUT_SHA256 = {
    1: "83dd1cdf2e151863a154ba5ee4cd1dab1b0db5e02ac0e1cd6c97c5a0c7eb7cbf",
    4: "659c552b293f3060934ae3c63414bc4c7d75717617cb89d348f9117e5f9783e0",
}
ARM_NAMES = {"E": "eager_pinned", "U": "uvm_original_va_graph"}
EXPECTED_UVM_ALLOCATOR_SIZES = sorted(
    [402_653_184] * 4
    + [201_326_592] * 4
    + [404_750_336] * 2
    + [203_423_744] * 2
)


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
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                require(
                    isinstance(value, dict),
                    f"{path}:{line_number} is not a JSON object",
                )
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    require(bool(records), f"{path} is empty")
    return records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_input_hashes(run_root: Path) -> dict[str, Any]:
    path = run_root / "input_sha256.txt"
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    except OSError as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    records: dict[Path, str] = {}
    for line_number, line in enumerate(lines, 1):
        parts = line.split(maxsplit=1)
        require(len(parts) == 2, f"{path}:{line_number} is malformed")
        digest, raw_file = parts
        current_path = Path(raw_file.strip()).resolve()
        require(current_path not in records, f"{path}: duplicate {current_path}")
        require(current_path.is_file(), f"{path}: missing {current_path}")
        require(
            len(digest) == 64 and sha256_file(current_path) == digest,
            f"{path}: hash drift for {current_path}",
        )
        records[current_path] = digest

    repo_root = Path(__file__).resolve().parents[2]
    required = {
        Path(__file__).resolve(),
        (repo_root / "experiments/vllm_uvm_original_va_graph_integration/run_integration.sh").resolve(),
        (repo_root / "experiments/vllm_uvm_original_va_graph_integration/benchmark_original_va_graph_steps.py").resolve(),
        (repo_root / "experiments/vllm_p3_static_alias_graph_integration/benchmark_full_graph_steps.py").resolve(),
        (repo_root / "experiments/vllm_p3_pinned_uva_phase_compare/benchmark_phase_steps.py").resolve(),
        (repo_root / "workloads/vllm/vllm/vllm/model_executor/models/utils.py").resolve(),
        (repo_root / "workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py").resolve(),
        (repo_root / "workloads/vllm/vllm/uvm_test/uvm_allocator.so").resolve(),
    }
    require(required <= set(records), "input hash manifest omits required code")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "file_count": len(records),
    }


def read_status(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            records = list(csv.DictReader(handle, delimiter="\t"))
    except OSError as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    require(bool(records), "execution_status.tsv is empty")
    return records


def output_digest(outputs: Any) -> str:
    require(isinstance(outputs, list), "outputs are not a list")
    payload = [record["output_token_ids"] for record in outputs]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def validate_exact_name_input(
    run_root: Path,
    config: dict[str, Any],
) -> tuple[Path, set[str]]:
    require(config.get("layers") == [0, 1, 2, 3], "selected layers differ")
    require(int(config.get("parameter_count", 0)) == 8, "parameter count differs")
    require(
        int(config.get("logical_bytes_per_forward", 0)) == LOGICAL_BYTES,
        "logical copy bytes differ",
    )
    require(
        int(config.get("dual_bank_backing_bytes", 0)) == DUAL_BANK_BYTES,
        "dual-bank bytes differ",
    )
    expected = {
        f"model.layers.{layer}.mlp.experts.{local_name}"
        for layer in range(4)
        for local_name in ("w13_weight", "w2_weight")
    }
    path = run_root / "inputs" / "exact_parameter_names.txt"
    try:
        names = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    except OSError as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    require(names == expected, "exact-name input is not the L0-L3 w13/w2 set")
    return path.resolve(), expected


def validate_result_configuration(
    result: dict[str, Any],
    *,
    run_config: dict[str, Any],
    row: dict[str, str],
    symbol: str,
    directory: Path,
) -> None:
    current = result.get("config")
    require(isinstance(current, dict), f"{directory.name}: result config absent")
    require(
        Path(str(current.get("model", ""))).resolve()
        == Path(str(run_config["model"])).resolve(),
        f"{directory.name}: model differs from run config",
    )
    require(
        Path(str(current.get("fixture", ""))).resolve()
        == Path(str(run_config["fixture"])).resolve(),
        f"{directory.name}: fixture differs from run config",
    )
    expected_batch_order = [int(value) for value in str(row["batch_order"]).split(",")]
    require(
        current.get("batch_sizes") == expected_batch_order,
        f"{directory.name}: batch order differs",
    )
    scalar_fields = (
        "conditioning_repeats",
        "measurement_repeats",
        "warmup_decode_steps",
        "measured_decode_steps",
        "kv_cache_memory_bytes",
    )
    for field in scalar_fields:
        require(
            int(current.get(field, -1)) == int(run_config[field]),
            f"{directory.name}: {field} differs from run config",
        )
    require(
        sorted(int(value) for value in current.get("cudagraph_capture_sizes", []))
        == sorted(int(value) for value in run_config["cudagraph_capture_sizes"]),
        f"{directory.name}: requested Graph capture sizes differ",
    )
    require(
        int(current.get("tensor_parallel_size", 0)) == 1
        and int(current.get("pipeline_parallel_size", 0)) == 1
        and int(current.get("data_parallel_size", 0)) == 1
        and current.get("enable_dbo") is False,
        f"{directory.name}: concurrency topology differs",
    )
    if symbol == "U":
        expected = {
            "profile_label": "qwen_l0_l3_uvm_original_va_full_graph",
            "mechanism_class": "vllm_uvm_original_va_static_alias_dual_bank_l0_l3",
            "execution_mode": "full_graph_decode",
            "realized_cudagraph_mode": "FULL_DECODE_ONLY",
            "enforce_eager": False,
        }
        require(
            sorted(current.get("realized_cudagraph_capture_sizes", [])) == [1, 4],
            f"{directory.name}: realized Graph capture sizes differ",
        )
    else:
        expected = {
            "profile_label": "qwen_l0_l3_eager_pinned",
            "mechanism_class": "vllm_pinned_copy_exact_l0_l3",
            "execution_mode": "eager",
            "realized_cudagraph_mode": "NONE",
            "enforce_eager": True,
        }
        require(
            current.get("realized_cudagraph_capture_sizes") == [],
            f"{directory.name}: eager arm realized a Graph capture size",
        )
    for field, expected_value in expected.items():
        require(
            current.get(field) == expected_value,
            f"{directory.name}: {field} differs",
        )
    require(
        result.get("profile_label") == expected["profile_label"]
        and result.get("mechanism_class") == expected["mechanism_class"],
        f"{directory.name}: top-level profile identity differs",
    )


def validate_parameter_manifest(
    result: dict[str, Any],
    *,
    directory: Path,
    exact_names_path: Path,
    expected_names: set[str],
    graph_arm: bool,
) -> None:
    path = directory / "cpu_offload_manifest.jsonl"
    records = load_jsonl(path)
    config_records = [record for record in records if record.get("event") == "cpu_offload_config"]
    parameter_records = [
        record for record in records if record.get("event") == "cpu_offload_parameter"
    ]
    require(len(config_records) == 1, f"{directory.name}: offload config count differs")
    require(len(parameter_records) == 8, f"{directory.name}: offload parameter count differs")
    require(
        {str(record.get("full_parameter_name")) for record in parameter_records}
        == expected_names,
        f"{directory.name}: offloaded parameter names differ",
    )
    require(
        sum(int(record.get("bytes", 0)) for record in parameter_records) == LOGICAL_BYTES,
        f"{directory.name}: offloaded parameter bytes differ",
    )
    require(
        all(
            record.get("copy_exact_names") is True
            and record.get("copy_on_forward") is True
            and record.get("offload_backend") == "copy"
            and Path(str(record.get("copy_exact_names_file", ""))).resolve()
            == exact_names_path
            for record in parameter_records
        ),
        f"{directory.name}: exact pinned-copy parameter contract differs",
    )
    config_record = config_records[0]
    require(
        config_record.get("copy_exact_names") is True
        and int(config_record.get("copy_exact_name_count", 0)) == 8
        and Path(str(config_record.get("copy_exact_names_file", ""))).resolve()
        == exact_names_path,
        f"{directory.name}: exact-name config record differs",
    )
    if graph_arm:
        require(
            config_record.get("graph_static_alias") is True
            and config_record.get("graph_static_alias_backend") == "uvm_original_va",
            f"{directory.name}: graph config record differs",
        )
        require(
            all(
                record.get("graph_static_alias") is True
                and record.get("graph_static_alias_backend") == "uvm_original_va"
                and record.get("uvm_offloading") is True
                and record.get("device_after") == "cuda:0"
                and record.get("uvm_original_parameter_va_retained") is True
                for record in parameter_records
            ),
            f"{directory.name}: original-VA parameter record differs",
        )
    else:
        require(
            config_record.get("graph_static_alias") is False,
            f"{directory.name}: eager config enabled static alias",
        )
        require(
            all(
                record.get("graph_static_alias") is False
                and record.get("uvm_offloading") is False
                and record.get("cpu_data_pinned") is True
                and record.get("device_after") == "cpu"
                for record in parameter_records
            ),
            f"{directory.name}: eager pinned parameter record differs",
        )
    evidence = result.get("offload_manifest_evidence", {})
    require(
        Path(str(evidence.get("path", ""))).resolve() == path.resolve()
        and int(evidence.get("parameter_record_count", 0)) == 8
        and int(evidence.get("parameter_logical_bytes", 0)) == LOGICAL_BYTES,
        f"{directory.name}: manifest evidence summary differs",
    )
    if graph_arm:
        final_manifest = result.get("shutdown_contract", {}).get(
            "manifest_after_destroy", {}
        )
        require(
            int(final_manifest.get("record_count", -1)) == len(records)
            and int(final_manifest.get("parameter_count", 0)) == 8
            and int(final_manifest.get("parameter_logical_bytes", 0))
            == LOGICAL_BYTES,
            f"{directory.name}: final manifest summary differs",
        )
    else:
        require(
            evidence.get("sha256") == sha256_file(path),
            f"{directory.name}: eager manifest hash differs",
        )


def validate_cycles(
    result: dict[str, Any], *, graph_arm: bool, label: str
) -> dict[tuple[int, str, int], str]:
    cycles = result.get("cycles")
    require(isinstance(cycles, list) and bool(cycles), f"{label}: no cycles")
    config = result.get("config", {})
    batch_sizes = [int(value) for value in config.get("batch_sizes", [])]
    conditioning_repeats = int(config.get("conditioning_repeats", -1))
    measurement_repeats = int(config.get("measurement_repeats", -1))
    warmup_steps = int(config.get("warmup_decode_steps", -1))
    measured_steps = int(config.get("measured_decode_steps", -1))
    expected_keys = {
        (batch, role, role_index)
        for batch in batch_sizes
        for role, count in (
            ("conditioning", conditioning_repeats),
            ("measurement", measurement_repeats),
        )
        for role_index in range(count)
    }
    require(
        len(cycles) == len(expected_keys),
        f"{label}: cycle count differs from requested repeats",
    )
    hashes: dict[tuple[int, str, int], str] = {}
    observed_keys: set[tuple[int, str, int]] = set()
    full_replays = 0
    decode_steps = 0
    for cycle_index, cycle in enumerate(cycles):
        require(isinstance(cycle, dict), f"{label}: invalid cycle {cycle_index}")
        batch = int(cycle["batch_size"])
        role = str(cycle["role"])
        role_index = int(cycle["role_index"])
        cycle_key = (batch, role, role_index)
        require(cycle_key not in observed_keys, f"{label}: duplicate cycle {cycle_key}")
        observed_keys.add(cycle_key)
        digest = output_digest(cycle.get("outputs"))
        require(
            digest == cycle.get("output_token_ids_sha256"),
            f"{label}: serialized output hash differs",
        )
        hashes[cycle_key] = digest
        prefill_delta = cycle.get("prefill_graph_wrapper_delta", {})
        require(
            not any(str(key).endswith("_replay") for key in prefill_delta),
            f"{label}: prefill unexpectedly replayed a Graph",
        )
        for category in ("warmup_decode_steps", "measured_decode_steps"):
            steps = cycle.get(category)
            require(isinstance(steps, list) and bool(steps), f"{label}: no {category}")
            expected_step_count = (
                warmup_steps if category == "warmup_decode_steps" else measured_steps
            )
            require(
                len(steps) == expected_step_count,
                f"{label}: {category} count differs",
            )
            require(
                [int(step.get("step", -1)) for step in steps]
                == list(range(1, expected_step_count + 1)),
                f"{label}: {category} indices differ",
            )
            for step in steps:
                decode_steps += 1
                delta = step.get("graph_wrapper_delta", {})
                current_full = int(delta.get("full_replay", 0))
                other_replays = sum(
                    int(value)
                    for key, value in delta.items()
                    if str(key).endswith("_replay") and key != "full_replay"
                )
                if graph_arm:
                    require(
                        current_full == 1 and other_replays == 0,
                        f"{label}: decode did not use exactly one FULL replay: {delta}",
                    )
                else:
                    require(
                        current_full == 0 and other_replays == 0,
                        f"{label}: eager arm replayed a Graph: {delta}",
                    )
                full_replays += current_full
    if graph_arm:
        require(full_replays == decode_steps, f"{label}: aggregate replay count differs")
    require(observed_keys == expected_keys, f"{label}: cycle roles/batches differ")
    for batch in {key[0] for key in hashes}:
        values = {digest for key, digest in hashes.items() if key[0] == batch}
        require(len(values) == 1, f"{label}: batch {batch} output is non-deterministic")
    return hashes


def validate_uvm_contract(result: dict[str, Any], label: str) -> None:
    contract = result.get("original_va_contract")
    require(isinstance(contract, dict), f"{label}: missing original-VA contract")
    require(contract.get("requested") is True, f"{label}: UVM backend not requested")
    active = contract.get("diagnostics_before_destroy")
    driver = contract.get("driver_diagnostics_before_destroy")
    manifest = contract.get("manifest_before_destroy")
    require(isinstance(active, dict), f"{label}: missing active diagnostics")
    require(active.get("enabled") is True, f"{label}: static alias disabled")
    require(active.get("backend") == "uvm_original_va", f"{label}: wrong backend")
    require(active.get("lifecycle_state") == "sealed", f"{label}: not sealed")
    require(active.get("sealed") is True, f"{label}: sealed flag false")
    require(int(active.get("session_id", 0)) > 0, f"{label}: session id absent")
    require(int(active.get("arena_count", 0)) == 4, f"{label}: arena count differs")
    require(
        int(active.get("backing_bytes", 0)) == DUAL_BANK_BYTES,
        f"{label}: backing is not 1.125 GiB",
    )
    arenas = active.get("arenas")
    require(isinstance(arenas, list) and len(arenas) == 4, f"{label}: arenas differ")
    consumer_vas = [int(va) for arena in arenas for va in arena["consumer_vas"]]
    require(len(set(consumer_vas)) == 8, f"{label}: original VAs not unique")
    require(
        all(int(arena.get("backing_layer_id", -1)) >= 0 for arena in arenas),
        f"{label}: arenas are not driver-backed",
    )
    require(
        isinstance(driver, dict)
        and driver.get("session_count") == 1
        and driver.get("sessions", [{}])[0].get("sealed") is True,
        f"{label}: userspace session cache differs",
    )
    seal = active.get("seal_result", {})
    require(
        seal.get("ok") is True
        and seal.get("sealed") is True
        and int(seal.get("alias_count", 0)) == ALIAS_COUNT
        and int(seal.get("alias_bytes", 0)) == ALIAS_BYTES
        and int(seal.get("backing_bytes", 0)) == DUAL_BANK_BYTES,
        f"{label}: exact SEAL topology differs",
    )
    require(isinstance(manifest, dict), f"{label}: manifest evidence absent")
    require(
        int(manifest.get("parameter_count", 0)) == 8
        and int(manifest.get("parameter_logical_bytes", 0)) == LOGICAL_BYTES,
        f"{label}: selected Qwen parameters differ",
    )
    require(
        int(manifest.get("original_va_retained_count", 0)) == 8
        and int(manifest.get("managed_parameter_count", 0)) == 8,
        f"{label}: original managed VA retention failed",
    )
    require(
        manifest.get("original_parameter_vas") == sorted(consumer_vas),
        f"{label}: manifest/live original VAs differ",
    )
    require(
        len(manifest.get("finalize_records", [])) == 1
        and manifest["finalize_records"][0].get("ok") is True,
        f"{label}: finalize evidence differs",
    )

    shutdown = result.get("shutdown_contract")
    require(isinstance(shutdown, dict), f"{label}: shutdown contract absent")
    require(
        shutdown.get("graphs_destroyed_before_cmd99") is True,
        f"{label}: Graph destruction ordering absent",
    )
    cleanup = shutdown.get("cleanup_result", {})
    require(cleanup.get("ok") is True, f"{label}: cmd99 failed: {cleanup}")
    destroy = cleanup.get("destroy", {})
    require(
        int(cleanup.get("destroyed_graph_count", 0)) == 2
        and destroy.get("ok") is True
        and destroy.get("cache_cleared") is True
        and destroy.get("session_cleared") is True
        and int(destroy.get("released_alias_count", 0)) == ALIAS_COUNT
        and int(destroy.get("released_alias_bytes", 0)) == ALIAS_BYTES
        and int(destroy.get("released_backing_bytes", 0)) == DUAL_BANK_BYTES
        and int(destroy.get("remaining_alias_count", -1)) == 0
        and int(destroy.get("remaining_alias_bytes", -1)) == 0
        and int(destroy.get("remaining_backing_bytes", -1)) == 0,
        f"{label}: exact DESTROY accounting differs",
    )
    post = shutdown.get("diagnostics_after_destroy", {})
    require(
        post.get("lifecycle_state") == "remote"
        and post.get("sealed") is False
        and int(post.get("session_id", -1)) == 0,
        f"{label}: post-cmd99 state differs",
    )
    require(
        shutdown.get("driver_diagnostics_after_destroy", {}).get("session_count") == 0,
        f"{label}: userspace session survived cmd99",
    )
    after_manifest = shutdown.get("manifest_after_destroy", {})
    require(
        len(after_manifest.get("destroy_records", [])) == 1
        and after_manifest["destroy_records"][0].get("ok") is True,
        f"{label}: destroy manifest evidence differs",
    )


def validate_eager_contract(result: dict[str, Any], label: str) -> None:
    contract = result.get("original_va_contract")
    require(isinstance(contract, dict), f"{label}: missing wrapper contract")
    require(contract.get("requested") is False, f"{label}: eager requested UVM alias")
    diagnostics = contract.get("diagnostics_before_destroy", {})
    require(diagnostics.get("enabled") is False, f"{label}: eager enabled alias")
    cleanup = result.get("shutdown_contract", {}).get("cleanup_result", {})
    require(cleanup.get("ok") is True, f"{label}: eager shutdown failed")


def validate_uvm_allocator_log(directory: Path, label: str) -> dict[str, Any]:
    path = directory / "uvm_allocations.log"
    try:
        contents = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VerificationError(f"{label}: cannot read allocator log: {exc}") from exc
    require(
        contents.count("vLLM UVM Allocator Session Started") == 1,
        f"{label}: allocator log does not contain exactly one session",
    )
    pattern = re.compile(
        r"TRACE_ALLOC alloc_id=(\d+) ptr=(0x[0-9a-f]+) "
        r"end=(0x[0-9a-f]+) size_bytes=(\d+)"
    )
    allocations = [
        {
            "allocation_id": int(match.group(1)),
            "start_va": int(match.group(2), 16),
            "end_va_inclusive": int(match.group(3), 16),
            "size_bytes": int(match.group(4)),
        }
        for match in pattern.finditer(contents)
    ]
    require(len(allocations) == 12, f"{label}: managed allocation count is not 12")
    require(
        [record["allocation_id"] for record in allocations] == list(range(1, 13)),
        f"{label}: managed allocation ids are not contiguous",
    )
    require(
        sorted(record["size_bytes"] for record in allocations)
        == EXPECTED_UVM_ALLOCATOR_SIZES,
        f"{label}: managed allocation sizes differ from 8 weights + 4 uploads",
    )
    require(
        all(
            record["end_va_inclusive"] - record["start_va"] + 1
            == record["size_bytes"]
            for record in allocations
        ),
        f"{label}: allocator VA envelope differs from allocation size",
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "allocation_count": len(allocations),
        "allocation_bytes": sum(record["size_bytes"] for record in allocations),
        "allocation_sizes": [record["size_bytes"] for record in allocations],
    }


def timing_summary(result: dict[str, Any]) -> dict[int, dict[str, float]]:
    summaries: dict[int, dict[str, float]] = {}
    raw = result.get("phase_summary")
    require(isinstance(raw, dict), "phase summary absent")
    for batch_text, item in raw.items():
        batch = int(batch_text)
        values = {
            "prefill_step_ms": float(item["prefill_step_ms"]["median"]),
            "decode_step_ms": float(item["decode_repeat_median_ms"]["median"]),
            "cycle_e2e_ms": float(item["cycle_e2e_ms"]["median"]),
        }
        require(all(math.isfinite(x) and x > 0 for x in values.values()), "bad timing")
        summaries[batch] = values
    return summaries


def validate_module_cycle(run_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    before = load_json(run_root / "module.before.json")
    custom = load_json(run_root / "module.custom.json")
    after = load_json(run_root / "module.after.json")
    stock_src = str(config["stock_module_srcversion"])
    custom_src = str(config["custom_module_srcversion"])
    require(before.get("loaded_srcversion") == stock_src, "initial UVM is not stock")
    require(custom.get("loaded_srcversion") == custom_src, "custom UVM was not loaded")
    require(after.get("loaded_srcversion") == stock_src, "stock UVM was not restored")
    require(before.get("gpu_clean") is True, "initial GPU was not clean")
    require(custom.get("gpu_clean") is True, "custom module load leaked GPU memory")
    require(after.get("gpu_clean") is True, "final GPU/client state is not clean")
    require(
        custom.get("module_sha256") == config.get("custom_module_sha256"),
        "custom module hash differs",
    )
    require(
        before.get("module_sha256") == config.get("stock_module_sha256")
        and after.get("module_sha256") == config.get("stock_module_sha256"),
        "stock module hash differs before/after",
    )
    require(
        int(config.get("slot_budget_bytes", 0)) >= DUAL_BANK_BYTES,
        "slot budget is below 1.125 GiB",
    )
    require(
        int(config.get("manual_migration_fault_mode", -1)) == 2
        and int(custom.get("manual_migration_fault_mode", -1)) == 2,
        "custom UVM did not disable automatic migration with fail-fast mode 2",
    )
    return {"before": before, "custom": custom, "after": after}


def verify(run_root: Path) -> dict[str, Any]:
    input_hashes = validate_input_hashes(run_root)
    config = load_json(run_root / "run_config.json")
    require(
        config.get("schema") == "vllm_uvm_original_va_graph_run_v1",
        "wrong run schema",
    )
    require(config.get("module_cycle") is True, "module cycle contract absent")
    module_cycle = validate_module_cycle(run_root, config)
    exact_names_path, expected_names = validate_exact_name_input(run_root, config)
    status = read_status(run_root / "execution_status.tsv")
    order = str(config["order"])
    require(len(status) == len(order), "execution/order lengths differ")

    hashes_reference: dict[tuple[int, str, int], str] | None = None
    fixture_hash: str | None = None
    process_counts: Counter[str] = Counter()
    timing: dict[str, dict[int, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    allocator_audits: list[dict[str, Any]] = []
    for index, (row, symbol) in enumerate(zip(status, order), 1):
        require(row.get("symbol") == symbol, f"row {index}: order differs")
        require(symbol in ARM_NAMES, f"row {index}: unknown arm")
        require(row.get("returncode") == "0", f"row {index}: process failed")
        directory = run_root / str(row["directory"])
        for filename in ("command.txt", "environment.txt", "gpu.before.json", "gpu.after.json"):
            require((directory / filename).is_file(), f"{directory.name}: missing {filename}")
        gpu_before = load_json(directory / "gpu.before.json")
        gpu_after = load_json(directory / "gpu.after.json")
        require(gpu_before.get("gpu_clean") is True, f"{directory.name}: dirty start")
        require(gpu_after.get("gpu_clean") is True, f"{directory.name}: dirty shutdown")
        result = load_json(directory / "result.json")
        require(
            result.get("schema") == "vllm_uvm_original_va_graph_steps_v1",
            f"{directory.name}: wrong result schema",
        )
        require(result.get("status") == "ok", f"{directory.name}: result not ok")
        require(not result.get("wrapper_contract_errors"), f"{directory.name}: contract errors")
        graph_arm = symbol == "U"
        validate_result_configuration(
            result,
            run_config=config,
            row=row,
            symbol=symbol,
            directory=directory,
        )
        validate_parameter_manifest(
            result,
            directory=directory,
            exact_names_path=exact_names_path,
            expected_names=expected_names,
            graph_arm=graph_arm,
        )
        hashes = validate_cycles(result, graph_arm=graph_arm, label=directory.name)
        if graph_arm:
            validate_uvm_contract(result, directory.name)
            allocator_audits.append(
                validate_uvm_allocator_log(directory, directory.name)
            )
        else:
            validate_eager_contract(result, directory.name)
        if hashes_reference is None:
            hashes_reference = hashes
        else:
            require(hashes == hashes_reference, f"{directory.name}: output hashes differ")
        current_fixture = str(result.get("fixture", {}).get("sha256", ""))
        if fixture_hash is None:
            fixture_hash = current_fixture
        require(current_fixture == fixture_hash and len(current_fixture) == 64, "fixture differs")
        process_counts[symbol] += 1
        for batch, values in timing_summary(result).items():
            timing[symbol][batch].append(values)

    required = set(str(config["required_arms"]))
    require(required <= set(process_counts), "required arm missing")
    minimum = 3 if config.get("mode") == "formal" else 1
    for symbol in required:
        require(process_counts[symbol] >= minimum, f"arm {symbol} has too few processes")
    if (
        fixture_hash == REFERENCE_FIXTURE_SHA256
        and int(config["warmup_decode_steps"]) == 5
        and int(config["measured_decode_steps"]) == 16
    ):
        assert hashes_reference is not None
        for (batch, _, _), digest in hashes_reference.items():
            require(digest == REFERENCE_OUTPUT_SHA256[batch], "reference output changed")

    aggregate: dict[str, Any] = {}
    for symbol, batches in timing.items():
        aggregate[symbol] = {}
        for batch, samples in batches.items():
            aggregate[symbol][str(batch)] = {
                metric: {
                    "process_count": len(samples),
                    "process_medians": [sample[metric] for sample in samples],
                    "median": statistics.median(sample[metric] for sample in samples),
                }
                for metric in ("prefill_step_ms", "decode_step_ms", "cycle_e2e_ms")
            }
    comparisons: dict[str, Any] = {}
    if "E" in aggregate and "U" in aggregate:
        for batch in sorted(set(aggregate["E"]) & set(aggregate["U"])):
            comparisons[batch] = {}
            for metric in ("prefill_step_ms", "decode_step_ms", "cycle_e2e_ms"):
                eager = float(aggregate["E"][batch][metric]["median"])
                uvm = float(aggregate["U"][batch][metric]["median"])
                comparisons[batch][metric] = {
                    "eager_ms": eager,
                    "uvm_original_va_graph_ms": uvm,
                    "uvm_over_eager_ratio": uvm / eager,
                    "uvm_speedup_percent": 100.0 * (eager - uvm) / eager,
                }
    return {
        "schema": "vllm_uvm_original_va_graph_verification_v1",
        "status": "ok",
        "run_root": str(run_root.resolve()),
        "mode": config["mode"],
        "order": order,
        "process_counts": dict(sorted(process_counts.items())),
        "fixture_sha256": fixture_hash,
        "output_hashes": {
            f"batch{batch}/{role}{role_index}": digest
            for (batch, role, role_index), digest in sorted((hashes_reference or {}).items())
        },
        "input_hashes": input_hashes,
        "module_cycle": module_cycle,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "uvm_allocator_audits": allocator_audits,
        "performance_is_acceptance_gate": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run_root / "verification.json"
    try:
        result = verify(args.run_root)
    except Exception as exc:
        result = {
            "schema": "vllm_uvm_original_va_graph_verification_v1",
            "status": "failed",
            "run_root": str(args.run_root.resolve()),
            "error": repr(exc),
        }
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
