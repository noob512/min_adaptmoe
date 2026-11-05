#!/usr/bin/env python3
"""Fail-closed verification for the layer-versus-cold-5 prefill experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
RUN_SCHEMA = "vllm_uvm_layer_vs_cold5_prefill_run_v1"
RESULT_SCHEMA = "vllm_uvm_layer_vs_cold5_prefill_steps_v1"
VERIFICATION_SCHEMA = "vllm_uvm_layer_vs_cold5_prefill_verification_v1"

DEFAULT_ARM_NAMES = {
    "Z": "all_gpu_graph",
    "C": "cold5_alias_graph",
    "B": "l0_l1_alias_graph",
    "L": "l0_l3_alias_graph",
}

# selected_bytes is both the selected logical geometry and the required actual
# H2D total for every model forward represented in the copy-forward manifest.
ARM_GEOMETRY: dict[str, dict[str, int]] = {
    "Z": {
        "selected_bytes": 0,
        "backing_bytes": 0,
        "parameter_count": 0,
        "semantic_expert_count": 0,
        "diagnostics_selected_expert_count": 0,
        "layers_per_forward": 0,
        "alias_count": 0,
        "alias_bytes": 0,
    },
    "C": {
        "selected_bytes": 1_132_462_080,
        "backing_bytes": 47_185_920,
        "parameter_count": 96,
        "semantic_expert_count": 240,
        "diagnostics_selected_expert_count": 240,
        "layers_per_forward": 48,
        "alias_count": 2_250,
        "alias_bytes": 1_179_648_000,
    },
    "B": {
        "selected_bytes": 1_207_959_552,
        "backing_bytes": 1_207_959_552,
        "parameter_count": 4,
        "semantic_expert_count": 256,
        "diagnostics_selected_expert_count": 0,
        "layers_per_forward": 2,
        "alias_count": 4_608,
        "alias_bytes": 2_415_919_104,
    },
    "L": {
        "selected_bytes": 2_415_919_104,
        "backing_bytes": 1_207_959_552,
        "parameter_count": 8,
        "semantic_expert_count": 512,
        "diagnostics_selected_expert_count": 0,
        "layers_per_forward": 4,
        "alias_count": 6_912,
        "alias_bytes": 3_623_878_656,
    },
}

STATUS_COLUMNS = [
    "ordinal",
    "symbol",
    "arm",
    "batch_order",
    "directory",
    "returncode",
]

COMMON_CONFIG_KEYS = (
    "conditioning_repeats",
    "cudagraph_capture_sizes",
    "data_parallel_size",
    "enable_chunked_prefill",
    "enable_dbo",
    "enable_prefix_caching",
    "engine_core_multiprocessing",
    "fixture",
    "frozen_prompt_token_ids",
    "gpu_memory_utilization",
    "ignore_eos",
    "kv_cache_memory_bytes",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "measured_decode_steps",
    "measurement_repeats",
    "model",
    "pipeline_parallel_size",
    "realized_compilation_mode",
    "realized_cudagraph_capture_sizes",
    "requested_compilation_mode",
    "seed",
    "temperature",
    "tensor_parallel_size",
    "total_output_tokens",
    "warmup_decode_steps",
)

DISABLED_UVM_ENVIRONMENT = (
    "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE",
    "VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE",
    "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE",
    "VLLM_UVM_INIT_WEIGHT_WARM_OFFLOAD_ENABLE",
)


class VerificationError(RuntimeError):
    """The artifact failed a required experiment contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise VerificationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def load_jsonl(path: Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, raw in enumerate(lines, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            require(
                isinstance(value, dict),
                f"{path}:{line_number} is not a JSON object",
            )
            records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    require(allow_empty or bool(records), f"{path} is empty")
    return records


def normalized_path(value: Any) -> str:
    return str(Path(str(value)).resolve())


def output_digest(outputs: Any) -> str:
    require(isinstance(outputs, list), "outputs are not a list")
    try:
        payload = [record["output_token_ids"] for record in outputs]
    except (KeyError, TypeError) as exc:
        raise VerificationError("serialized outputs lack token ids") from exc
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def first_dict(*values: Any) -> dict[str, Any]:
    for value in values:
        if isinstance(value, dict) and value:
            return value
    return {}


def optional_bool_false(value: dict[str, Any], key: str, label: str) -> None:
    if key in value:
        require(value[key] is False, f"{label}: {key} is not false")


def int_from_aliases(
    value: dict[str, Any], keys: Iterable[str], *, label: str
) -> int:
    observed: list[tuple[str, int]] = []
    for key in keys:
        if key in value:
            try:
                observed.append((key, int(value[key])))
            except (TypeError, ValueError) as exc:
                raise VerificationError(f"{label}: {key} is not an integer") from exc
    require(bool(observed), f"{label}: required geometry field is absent")
    distinct = {item for _, item in observed}
    require(len(distinct) == 1, f"{label}: aliased geometry fields disagree")
    return observed[0][1]


def normalized_arm_geometry(run_config: dict[str, Any]) -> dict[str, dict[str, int]]:
    configured = run_config.get("arm_geometry")
    require(isinstance(configured, dict), "run config arm_geometry is absent")
    require(set(configured) == set(ARM_GEOMETRY), "run config arm_geometry arms differ")
    normalized: dict[str, dict[str, int]] = {}
    for symbol, expected in ARM_GEOMETRY.items():
        raw = configured[symbol]
        require(isinstance(raw, dict), f"arm_geometry.{symbol} is not an object")
        selected = int_from_aliases(
            raw,
            (
                "selected_bytes",
                "selected_bytes_per_forward",
                "selected_bytes_per_prefill",
                "selected_parameter_bytes_per_prefill",
                "actual_h2d_bytes_per_forward",
                "expected_h2d_bytes_per_prefill",
            ),
            label=f"arm_geometry.{symbol}",
        )
        backing = int_from_aliases(
            raw,
            ("backing_bytes", "expected_backing_bytes"),
            label=f"arm_geometry.{symbol}",
        )
        parameters = int_from_aliases(
            raw,
            ("parameter_count", "selected_parameter_count"),
            label=f"arm_geometry.{symbol}",
        )
        require(selected == expected["selected_bytes"], f"{symbol}: selected bytes differ")
        require(backing == expected["backing_bytes"], f"{symbol}: backing bytes differ")
        require(parameters == expected["parameter_count"], f"{symbol}: Parameter count differs")
        semantic_experts = int_from_aliases(
            raw,
            ("selected_expert_count",),
            label=f"arm_geometry.{symbol}",
        )
        require(
            semantic_experts == expected["semantic_expert_count"],
            f"{symbol}: semantic expert count differs",
        )
        diagnostics_experts = int(
            raw.get("expected_diagnostics_selected_expert_count", semantic_experts)
        )
        require(
            diagnostics_experts == expected["diagnostics_selected_expert_count"],
            f"{symbol}: diagnostics expert count differs",
        )
        if "expected_alias_count" in raw:
            require(
                int(raw["expected_alias_count"]) == expected["alias_count"],
                f"{symbol}: expected alias count differs",
            )
        if "expected_alias_bytes" in raw:
            require(
                int(raw["expected_alias_bytes"]) == expected["alias_bytes"],
                f"{symbol}: expected alias bytes differ",
            )
        net_bytes = selected - backing
        if any(
            key in raw
            for key in (
                "net_bytes",
                "theoretical_net_bytes",
                "theoretical_net_residency_relief_bytes",
            )
        ):
            configured_net = int_from_aliases(
                raw,
                (
                    "net_bytes",
                    "theoretical_net_bytes",
                    "theoretical_net_residency_relief_bytes",
                ),
                label=f"arm_geometry.{symbol}",
            )
            require(configured_net == net_bytes, f"{symbol}: net bytes differ")
        normalized[symbol] = {
            "selected_bytes_per_forward": selected,
            "backing_bytes": backing,
            "parameter_count": parameters,
            "semantic_selected_expert_count": semantic_experts,
            "diagnostics_selected_expert_count": diagnostics_experts,
            "theoretical_net_bytes": net_bytes,
        }
    return normalized


def validate_input_hashes(
    run_root: Path, run_config: dict[str, Any]
) -> dict[str, str]:
    path = run_root / "input_sha256.txt"
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    except OSError as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    require(bool(lines), f"{path} is empty")
    records: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        parts = line.split(maxsplit=1)
        require(len(parts) == 2, f"{path}:{line_number} is malformed")
        digest, raw_file = parts
        require(len(digest) == 64, f"{path}:{line_number} has an invalid digest")
        current = Path(raw_file.strip())
        if not current.is_absolute():
            current = run_root / current
        current = current.resolve()
        key = str(current)
        require(key not in records, f"{path}: duplicate input {current}")
        require(current.is_file(), f"{path}: input is absent: {current}")
        require(sha256_file(current) == digest, f"{path}: hash drift for {current}")
        records[key] = digest

    verifier = str(Path(__file__).resolve())
    require(verifier in records, "input hashes do not cover this verifier")
    require(
        any(
            Path(raw).parent == HERE
            and Path(raw).name.startswith("benchmark")
            and Path(raw).suffix == ".py"
            for raw in records
        ),
        "input hashes do not cover the experiment wrapper",
    )
    require(
        any(Path(raw).name == "benchmark_full_graph_steps.py" for raw in records),
        "input hashes do not cover the shared FULL Graph runner",
    )

    configured_hashes = run_config.get("code_hashes")
    if configured_hashes is not None:
        require(isinstance(configured_hashes, dict), "run config code_hashes is invalid")
        manifest_digests = set(records.values())
        for label, digest in configured_hashes.items():
            require(
                isinstance(digest, str) and digest in manifest_digests,
                f"run config code hash is not in input manifest: {label}",
            )
    return records


def validate_geometry_inputs(
    run_config: dict[str, Any], input_hashes: dict[str, str]
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for symbol, expected in ARM_GEOMETRY.items():
        raw = run_config["arm_geometry"][symbol]
        require(
            raw.get("selected_layers") == list(range(expected["layers_per_forward"])),
            f"{symbol}: configured layer selection differs",
        )
        require(
            int(raw.get("selected_layer_count", -1)) == expected["layers_per_forward"],
            f"{symbol}: configured selected-layer count differs",
        )
        sources: dict[str, Any] = {}
        for key in ("exact_parameter_names", "expert_plan"):
            source = raw.get(key)
            if source is None:
                sources[key] = None
                continue
            require(isinstance(source, dict), f"{symbol}: {key} source is invalid")
            source_path = Path(str(source.get("path", ""))).resolve()
            digest = str(source.get("sha256", ""))
            require(source_path.is_file(), f"{symbol}: {key} file is absent")
            require(sha256_file(source_path) == digest, f"{symbol}: {key} hash differs")
            require(
                input_hashes.get(str(source_path)) == digest,
                f"{symbol}: {key} is not covered by input hashes",
            )
            sources[key] = {"path": str(source_path), "sha256": digest}
        if symbol == "Z":
            require(all(value is None for value in sources.values()), "Z: selection input exists")
        elif symbol == "C":
            require(all(value is not None for value in sources.values()), "C: selection input absent")
        else:
            require(sources["exact_parameter_names"] is not None, f"{symbol}: names absent")
            require(sources["expert_plan"] is None, f"{symbol}: expert plan should be absent")
        evidence[symbol] = sources
    return evidence


def validate_gpu_snapshot(
    path: Path,
    *,
    require_no_uvm_clients: bool,
) -> dict[str, Any]:
    snapshot = load_json(path)
    require(int(snapshot.get("memory_used_mib", -1)) == 0, f"{path}: GPU memory is not zero")
    require(not snapshot.get("compute_pids"), f"{path}: GPU compute clients remain")
    if require_no_uvm_clients:
        require(not snapshot.get("uvm_client_pids"), f"{path}: UVM clients remain")
    require(bool(snapshot.get("loaded_srcversion")), f"{path}: UVM srcversion absent")
    return snapshot


def validate_host_restoration(run_root: Path) -> dict[str, Any]:
    initially_active_path = run_root / "services.initially_active.txt"
    after_path = run_root / "services.after.tsv"
    try:
        initially_active = {
            line.strip()
            for line in initially_active_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        service_states: dict[str, str] = {}
        for line_number, line in enumerate(
            after_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            parts = line.split("\t", 1)
            require(len(parts) == 2, f"{after_path}:{line_number} is malformed")
            service, state = parts
            require(service not in service_states, f"{after_path}: duplicate {service}")
            service_states[service] = state
    except OSError as exc:
        raise VerificationError(f"cannot read service restoration evidence: {exc}") from exc
    require(
        all(service_states.get(service) == "active" for service in initially_active),
        "an initially active service was not restored",
    )

    gpu_before = validate_gpu_snapshot(
        run_root / "gpu.before.json", require_no_uvm_clients=True
    )
    gpu_custom = validate_gpu_snapshot(
        run_root / "gpu.custom.json", require_no_uvm_clients=True
    )
    gpu_stock = validate_gpu_snapshot(
        run_root / "gpu.stock.json", require_no_uvm_clients=True
    )
    final_path = run_root / "gpu.final.json"
    gpu_final = (
        validate_gpu_snapshot(final_path, require_no_uvm_clients=False)
        if final_path.is_file()
        else None
    )
    require(
        gpu_before["loaded_srcversion"] == gpu_stock["loaded_srcversion"],
        "stock UVM srcversion was not restored",
    )
    require(
        gpu_custom["loaded_srcversion"] != gpu_stock["loaded_srcversion"],
        "custom and stock UVM srcversions are indistinguishable",
    )
    return {
        "initially_active_services": sorted(initially_active),
        "services_after": dict(sorted(service_states.items())),
        "stock_srcversion": gpu_stock["loaded_srcversion"],
        "custom_srcversion": gpu_custom["loaded_srcversion"],
        "final_snapshot_checked": gpu_final is not None,
    }


def arm_names_from_config(run_config: dict[str, Any]) -> dict[str, str]:
    raw = run_config.get("arm_names", DEFAULT_ARM_NAMES)
    require(isinstance(raw, dict), "run config arm_names is invalid")
    require(set(raw) == set(DEFAULT_ARM_NAMES), "run config arm_names arms differ")
    values = {symbol: str(raw[symbol]) for symbol in DEFAULT_ARM_NAMES}
    require(all(values.values()), "run config contains an empty arm name")
    require(len(set(values.values())) == len(values), "run config arm names are not unique")
    require(
        all("/" not in value and ".." not in value for value in values.values()),
        "run config contains an unsafe arm name",
    )
    return values


def common_contract_from_config(run_config: dict[str, Any]) -> dict[str, Any]:
    value = run_config.get("common_contract")
    require(isinstance(value, dict), "run config common_contract is absent")
    require(value.get("execution_mode") == "full_graph_all", "common execution mode differs")
    require(value.get("batch_sizes") == [1, 4], "common batch sizes differ")
    require(
        value.get("cudagraph_capture_sizes") == [1, 4, 314, 1217],
        "common Graph capture sizes differ",
    )
    require(value.get("alias_backend") == "uvm_original_va", "common alias backend differs")
    require(str(value.get("alias_pte_mode", "")).lower() == "4k", "common PTE mode differs")
    require(int(value.get("alias_region_bytes", -1)) == 524_288, "common alias region differs")
    require(int(value.get("expected_arena_count", -1)) == 4, "common arena count differs")
    require(
        int(value.get("manual_migration_fault_mode", -1)) == 2,
        "common migration mode differs",
    )
    for key in (
        "batch_sizes",
        "cudagraph_capture_sizes",
        "conditioning_repeats",
        "measurement_repeats",
        "warmup_decode_steps",
        "measured_decode_steps",
        "kv_cache_memory_bytes",
    ):
        if key in run_config:
            require(run_config[key] == value.get(key), f"top-level/common {key} differs")
    return value


def load_run_contract(
    run_root: Path,
) -> tuple[dict[str, Any], list[dict[str, str]], list[Path], dict[str, str]]:
    run_config = load_json(run_root / "run_config.json")
    require(run_config.get("schema") == RUN_SCHEMA, "run config schema differs")
    mode = str(run_config.get("mode"))
    require(mode in {"qualify", "formal"}, f"unverifiable mode: {mode}")
    order = str(run_config.get("order", ""))
    require(order and all(symbol in ARM_GEOMETRY for symbol in order), "run order differs")
    expected_processes = 1 if mode == "qualify" else 3
    expected_measurements = 2 if mode == "qualify" else 3
    expected_counts = Counter({symbol: expected_processes for symbol in ARM_GEOMETRY})
    require(Counter(order) == expected_counts, f"{mode} arm process counts differ")
    common_contract = common_contract_from_config(run_config)
    require(
        int(common_contract.get("measurement_repeats", -1)) == expected_measurements,
        f"{mode} measurement repeats differ",
    )
    require(
        int(common_contract.get("conditioning_repeats", -1)) == 1,
        f"{mode} conditioning repeats differ",
    )
    arm_names = arm_names_from_config(run_config)
    require(
        run_config.get("arm_process_counts") == dict(sorted(expected_counts.items())),
        "run config arm_process_counts differs",
    )
    executions = run_config.get("executions")
    require(
        isinstance(executions, list) and len(executions) == len(order),
        "run config executions differ",
    )
    for index, (symbol, execution) in enumerate(zip(order, executions, strict=True)):
        require(isinstance(execution, dict), f"execution {index + 1} is invalid")
        require(
            execution
            == {
                "ordinal": index + 1,
                "symbol": symbol,
                "arm": arm_names[symbol],
                "batch_sizes": [1, 4] if index % 2 == 0 else [4, 1],
            },
            f"execution {index + 1} differs",
        )

    status_path = run_root / "execution_status.tsv"
    try:
        with status_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            require(reader.fieldnames == STATUS_COLUMNS, "execution status columns differ")
            rows = list(reader)
    except OSError as exc:
        raise VerificationError(f"cannot read {status_path}: {exc}") from exc
    require(len(rows) == len(order), "execution status row count differs")

    result_paths: list[Path] = []
    batch_orders: dict[str, Counter[str]] = defaultdict(Counter)
    for index, (symbol, row) in enumerate(zip(order, rows, strict=True)):
        ordinal = index + 1
        expected_arm = arm_names[symbol]
        expected_batch_order = "1,4" if index % 2 == 0 else "4,1"
        expected_directory = f"{ordinal:02d}_{expected_arm}"
        require(row["ordinal"] == str(ordinal), f"row {ordinal}: ordinal differs")
        require(row["symbol"] == symbol, f"row {ordinal}: symbol differs")
        require(row["arm"] == expected_arm, f"row {ordinal}: arm name differs")
        require(
            row["batch_order"] == expected_batch_order,
            f"row {ordinal}: batch order differs",
        )
        require(
            row["directory"] == expected_directory,
            f"row {ordinal}: directory differs",
        )
        require(row["returncode"] == "0", f"row {ordinal}: return code differs")
        case_dir = run_root / expected_directory
        returncode_path = case_dir / "returncode.txt"
        require(returncode_path.is_file(), f"row {ordinal}: returncode.txt absent")
        require(
            returncode_path.read_text(encoding="utf-8").strip() == "0",
            f"row {ordinal}: recorded process return code differs",
        )
        result_path = case_dir / "result.json"
        require(result_path.is_file(), f"row {ordinal}: result is absent")
        validate_gpu_snapshot(case_dir / "gpu.before.json", require_no_uvm_clients=True)
        validate_gpu_snapshot(case_dir / "gpu.after.json", require_no_uvm_clients=True)
        result_paths.append(result_path)
        batch_orders[symbol][expected_batch_order] += 1

    if mode == "formal":
        for symbol in ARM_GEOMETRY:
            counts = batch_orders[symbol]
            require(
                set(counts) == {"1,4", "4,1"}
                and abs(counts["1,4"] - counts["4,1"]) == 1,
                f"formal batch order is not balanced for {symbol}: {counts}",
            )

    return run_config, rows, result_paths, arm_names


def normalized_common_config(config: dict[str, Any]) -> dict[str, Any]:
    value = {key: config.get(key) for key in COMMON_CONFIG_KEYS}
    value["model"] = normalized_path(config.get("model"))
    value["fixture"] = normalized_path(config.get("fixture"))
    return value


def validate_result_config(
    result: dict[str, Any],
    *,
    path: Path,
    symbol: str,
    row: dict[str, str],
    run_config: dict[str, Any],
    common_fingerprint: dict[str, Any] | None,
    versions_fingerprint: dict[str, Any] | None,
    code_fingerprint: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    config = result.get("config")
    versions = result.get("versions")
    environment = result.get("reproducibility_environment")
    require(isinstance(config, dict), f"{path}: config absent")
    require(isinstance(versions, dict) and bool(versions), f"{path}: versions absent")
    require(isinstance(environment, dict), f"{path}: environment absent")
    expected_batches = [int(item) for item in row["batch_order"].split(",")]
    require(config.get("batch_sizes") == expected_batches, f"{path}: batches differ")
    common_contract = common_contract_from_config(run_config)
    for key in (
        "conditioning_repeats",
        "measurement_repeats",
        "warmup_decode_steps",
        "measured_decode_steps",
    ):
        require(config.get(key) == common_contract.get(key), f"{path}: {key} differs")
    require(
        normalized_path(config.get("model")) == normalized_path(run_config.get("model")),
        f"{path}: model differs",
    )
    require(
        normalized_path(config.get("fixture"))
        == normalized_path(run_config.get("fixture")),
        f"{path}: fixture differs",
    )
    for key in COMMON_CONFIG_KEYS:
        if key in common_contract and key not in {"model", "fixture"}:
            require(config.get(key) == common_contract.get(key), f"{path}: {key} differs")
        if key in run_config and key not in {"model", "fixture"}:
            require(config.get(key) == run_config.get(key), f"{path}: top-level {key} differs")

    require(config.get("execution_mode") == "full_graph_all", f"{path}: mode is not FULL")
    require(config.get("enforce_eager") is False, f"{path}: eager was enforced")
    require(config.get("requested_cudagraph_mode") == "FULL", f"{path}: requested Graph mode")
    require(config.get("realized_cudagraph_mode") == "FULL", f"{path}: realized Graph mode")

    alias = symbol != "Z"
    arm_config = run_config["arm_geometry"][symbol]
    require(
        float(config.get("cpu_offload_gb", -1.0))
        == float(arm_config.get("cpu_offload_gb", -2.0)),
        f"{path}: arm CPU-offload budget differs",
    )
    require(
        environment.get("VLLM_ATTENTION_BACKEND")
        == common_contract.get("attention_backend"),
        f"{path}: attention backend differs",
    )
    require(environment.get("VLLM_USE_UVM") == "0", f"{path}: legacy UVM path enabled")
    require(
        environment.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0",
        f"{path}: engine multiprocessing differs",
    )
    require(
        environment.get("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS") == ("1" if alias else "0"),
        f"{path}: static-alias environment differs",
    )
    require(
        environment.get("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_EAGER") == "0",
        f"{path}: eager-alias environment differs",
    )
    require(
        environment.get("VLLM_MOE_TOKEN_EXPERT_TRACE_ENABLE", "0") == "0",
        f"{path}: expert tracing was enabled",
    )
    if alias:
        require(
            environment.get("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_BACKEND")
            == "uvm_original_va",
            f"{path}: alias backend differs",
        )
        require(
            str(environment.get("VLLM_CPU_OFFLOAD_GRAPH_STATIC_ALIAS_PTE_MODE", "")).lower()
            == "4k",
            f"{path}: requested PTE mode is not 4K",
        )
    for key in DISABLED_UVM_ENVIRONMENT:
        require(environment.get(key) == "0", f"{path}: {key} differs")

    current_common = normalized_common_config(config)
    if common_fingerprint is not None:
        require(current_common == common_fingerprint, f"{path}: common config differs")
    if versions_fingerprint is not None:
        require(versions == versions_fingerprint, f"{path}: versions differ")
    current_code = result.get("code_hashes")
    if current_code is not None:
        require(isinstance(current_code, dict), f"{path}: code_hashes is invalid")
        if code_fingerprint is not None:
            require(current_code == code_fingerprint, f"{path}: code hashes differ")
    elif code_fingerprint is not None:
        raise VerificationError(f"{path}: code hashes disappeared")
    return current_common, dict(versions), current_code


def validate_cycles(
    result: dict[str, Any],
    *,
    path: Path,
    run_config: dict[str, Any],
) -> tuple[dict[tuple[int, str, int], str], dict[int, list[float]], dict[int, int]]:
    cycles = result.get("cycles")
    require(isinstance(cycles, list), f"{path}: cycles are absent")
    common_contract = common_contract_from_config(run_config)
    conditioning_repeats = int(common_contract["conditioning_repeats"])
    measurement_repeats = int(common_contract["measurement_repeats"])
    expected_keys = {
        (batch, role, role_index)
        for batch in (1, 4)
        for role, repeats in (
            ("conditioning", conditioning_repeats),
            ("measurement", measurement_repeats),
        )
        for role_index in range(repeats)
    }
    require(len(cycles) == len(expected_keys), f"{path}: cycle count differs")
    observed: set[tuple[int, str, int]] = set()
    hashes: dict[tuple[int, str, int], str] = {}
    measurements: dict[int, list[float]] = defaultdict(list)
    prompt_tokens: dict[int, int] = {}
    for cycle_index, cycle in enumerate(cycles):
        require(isinstance(cycle, dict), f"{path}: cycle {cycle_index} is invalid")
        try:
            key = (
                int(cycle["batch_size"]),
                str(cycle["role"]),
                int(cycle["role_index"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise VerificationError(f"{path}: cycle identity is invalid") from exc
        require(key in expected_keys, f"{path}: unexpected cycle {key}")
        require(key not in observed, f"{path}: duplicate cycle {key}")
        observed.add(key)
        digest = output_digest(cycle.get("outputs"))
        require(digest == cycle.get("output_token_ids_sha256"), f"{path}: output hash differs")
        hashes[key] = digest

        delta = cycle.get("prefill_graph_wrapper_delta")
        require(isinstance(delta, dict), f"{path}: prefill Graph delta absent")
        require(int(delta.get("full_replay", 0)) == 1, f"{path}: prefill lacked one FULL replay")
        require(
            sum(
                int(value)
                for name, value in delta.items()
                if str(name).endswith("_replay") and name != "full_replay"
            )
            == 0,
            f"{path}: prefill used a non-FULL replay",
        )
        elapsed = float(cycle.get("prefill_elapsed_ms", math.nan))
        require(math.isfinite(elapsed) and elapsed > 0.0, f"{path}: invalid prefill time")
        batch = key[0]
        current_prompt_tokens = int(cycle.get("prompt_tokens", -1))
        require(current_prompt_tokens > 0, f"{path}: prompt token count absent")
        prior_prompt_tokens = prompt_tokens.setdefault(batch, current_prompt_tokens)
        require(prior_prompt_tokens == current_prompt_tokens, f"{path}: prompt tokens changed")
        schedule = cycle.get("prefill_schedule", {})
        require(
            int(schedule.get("total_num_scheduled_tokens", -1)) == current_prompt_tokens,
            f"{path}: prefill schedule token count differs",
        )
        if key[1] == "measurement":
            measurements[batch].append(elapsed)

    require(observed == expected_keys, f"{path}: cycle roles or repeats differ")
    for batch in (1, 4):
        require(
            len(measurements[batch]) == measurement_repeats,
            f"{path}: batch {batch} measurement count differs",
        )
    graph_contract = result.get("graph_contract")
    require(isinstance(graph_contract, dict), f"{path}: graph contract absent")
    require(
        int(graph_contract.get("prefill_step_count", -1)) == len(cycles),
        f"{path}: Graph prefill step count differs",
    )
    require(
        int(graph_contract.get("prefill_full_replay_count", -1)) == len(cycles),
        f"{path}: aggregate FULL replay count differs",
    )
    return hashes, measurements, prompt_tokens


def manifest_path_for_result(path: Path, result: dict[str, Any]) -> Path:
    configured = result.get("config", {}).get("offload_manifest")
    require(configured, f"{path}: offload manifest path absent")
    manifest_path = Path(str(configured)).resolve()
    expected = (path.parent / "cpu_offload_manifest.jsonl").resolve()
    require(manifest_path == expected, f"{path}: manifest is outside its case directory")
    require(manifest_path.is_file(), f"{path}: manifest is absent")
    return manifest_path


def collect_va_snapshots(
    result: dict[str, Any], active: dict[str, Any]
) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []

    def append(label: str, value: Any) -> None:
        if not isinstance(value, dict):
            return
        vas = value.get("original_va_snapshot")
        if not isinstance(vas, list):
            return
        snapshots.append(
            {
                "label": label,
                "vas": [int(item) for item in vas],
                "state": value.get("lifecycle_state"),
            }
        )

    append("after_load", result.get("static_alias_diagnostics_after_load"))
    for index, cycle in enumerate(result.get("cycles", [])):
        if isinstance(cycle, dict):
            append(f"cycle_{index}_after_prefill", cycle.get("static_alias_diagnostics_after_prefill"))
            append(f"cycle_{index}_after_decode", cycle.get("static_alias_diagnostics_after_decode"))
    append("before_shutdown", active)
    direct = result.get("alias_contract", {}).get("parameter_va_snapshots", [])
    if isinstance(direct, list):
        for index, item in enumerate(direct):
            if isinstance(item, dict) and isinstance(item.get("vas"), list):
                snapshots.append(
                    {
                        "label": str(item.get("label", f"wrapper_{index}")),
                        "vas": [int(value) for value in item["vas"]],
                        "state": item.get("state"),
                    }
                )
    return snapshots


def active_contracts(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    mechanism = result.get("mechanism_contract", {})
    original_va = result.get("original_va_contract", {})
    alias_contract = result.get("alias_contract", {})
    active = first_dict(
        mechanism.get("diagnostics_before_shutdown"),
        original_va.get("diagnostics_before_destroy"),
        alias_contract.get("diagnostics_before_shutdown"),
        result.get("static_alias_diagnostics_final"),
        result.get("static_alias_diagnostics_after_load"),
    )
    driver = first_dict(
        mechanism.get("driver_diagnostics_before_shutdown"),
        original_va.get("driver_diagnostics_before_destroy"),
        alias_contract.get("driver_diagnostics_before_shutdown"),
    )
    return active, driver


def post_shutdown_contracts(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    shutdown = result.get("shutdown_contract", {})
    post = first_dict(
        shutdown.get("diagnostics_after_shutdown"),
        shutdown.get("diagnostics_after_destroy"),
        shutdown.get("post_diagnostics"),
    )
    driver = first_dict(
        shutdown.get("driver_diagnostics_after_shutdown"),
        shutdown.get("driver_diagnostics_after_destroy"),
        shutdown.get("post_driver_diagnostics"),
    )
    return post, driver


def validate_shutdown(
    result: dict[str, Any], *, path: Path, symbol: str
) -> dict[str, Any]:
    post, driver_post = post_shutdown_contracts(result)
    require(bool(post), f"{path}: post-shutdown diagnostics absent")
    require(bool(driver_post), f"{path}: post-shutdown driver diagnostics absent")
    require(int(post.get("session_id", 0)) == 0, f"{path}: session survived shutdown")
    require(
        int(post.get("active_driver_backing_bytes", 0)) == 0,
        f"{path}: active backing survived shutdown",
    )
    optional_bool_false(post, "cleanup_pending", str(path))
    optional_bool_false(post, "cleanup_unresolved", str(path))
    require(
        int(driver_post.get("session_count", -1)) == 0
        and not driver_post.get("sessions"),
        f"{path}: driver session survived shutdown",
    )
    if symbol == "Z":
        return {"session_count_after_shutdown": 0}

    expected = ARM_GEOMETRY[symbol]
    require(post.get("lifecycle_state") == "remote", f"{path}: post state is not remote")
    require(post.get("sealed") is False, f"{path}: post state remained sealed")
    shutdown = result.get("shutdown_contract", {})
    candidates: list[dict[str, Any]] = []
    values = shutdown.get("model_runner_shutdown_results")
    if isinstance(values, list):
        candidates.extend(item for item in values if isinstance(item, dict))
    cleanup = shutdown.get("cleanup_result")
    if isinstance(cleanup, dict):
        candidates.append(cleanup)
    require(bool(candidates), f"{path}: alias shutdown result absent")
    cleanup_result = next(
        (item for item in reversed(candidates) if isinstance(item.get("destroy"), dict)),
        {},
    )
    require(bool(cleanup_result), f"{path}: alias DESTROY result absent")
    require(cleanup_result.get("ok") is True, f"{path}: alias shutdown failed")
    require(
        int(cleanup_result.get("destroyed_graph_count", 0)) > 0,
        f"{path}: no Graph was destroyed before alias cleanup",
    )
    destroy = cleanup_result["destroy"]
    require(destroy.get("ok") is True, f"{path}: cmd99 DESTROY failed")
    require(destroy.get("cache_cleared") is True, f"{path}: alias cache was not cleared")
    require(destroy.get("session_cleared") is True, f"{path}: session was not cleared")
    require(
        int(destroy.get("released_alias_count", -1)) == expected["alias_count"],
        f"{path}: released alias count differs",
    )
    require(
        int(destroy.get("released_alias_bytes", -1)) == expected["alias_bytes"],
        f"{path}: released alias bytes differ",
    )
    require(
        int(destroy.get("released_backing_bytes", -1)) == expected["backing_bytes"],
        f"{path}: released backing bytes differ",
    )
    for key in ("remaining_alias_count", "remaining_alias_bytes", "remaining_backing_bytes"):
        require(int(destroy.get(key, -1)) == 0, f"{path}: {key} is nonzero")
    return {
        "session_count_after_shutdown": 0,
        "released_alias_count": expected["alias_count"],
        "released_alias_bytes": expected["alias_bytes"],
        "released_backing_bytes": expected["backing_bytes"],
    }


def validate_control_contract(
    result: dict[str, Any], *, path: Path, records: list[dict[str, Any]]
) -> dict[str, Any]:
    active, driver = active_contracts(result)
    require(bool(active), f"{path}: control diagnostics absent")
    require(active.get("enabled") is False, f"{path}: control enabled static alias")
    require(int(active.get("session_id", 0)) == 0, f"{path}: control has a session")
    require(
        int(active.get("active_driver_backing_bytes", 0)) == 0,
        f"{path}: control has active alias backing",
    )
    require(
        int(driver.get("session_count", 0)) == 0 and not driver.get("sessions"),
        f"{path}: control driver session exists",
    )
    require(
        not [item for item in records if item.get("event") == "cpu_offload_parameter"],
        f"{path}: control offloaded a Parameter",
    )
    return {"session_count_before_shutdown": 0, "parameter_count": 0}


def validate_alias_manifest(
    records: list[dict[str, Any]],
    *,
    path: Path,
    symbol: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    expected = ARM_GEOMETRY[symbol]
    parameter_records = [item for item in records if item.get("event") == "cpu_offload_parameter"]
    require(
        len(parameter_records) == expected["parameter_count"],
        f"{path}: manifest Parameter count differs",
    )
    require(
        sum(int(item.get("bytes", 0)) for item in parameter_records)
        == expected["selected_bytes"],
        f"{path}: manifest selected bytes differ",
    )
    require(
        all(
            item.get("graph_static_alias") is True
            and item.get("graph_static_alias_backend") == "uvm_original_va"
            and item.get("uvm_offloading") is True
            and item.get("uvm_original_parameter_va_retained") is True
            and int(item.get("uvm_original_parameter_va", 0))
            == int(item.get("graph_static_alias_consumer_va", -1))
            for item in parameter_records
        ),
        f"{path}: manifest original-VA Parameter contract differs",
    )
    expected_layers = set(range(expected["layers_per_forward"]))
    observed_layers = Counter(int(item.get("graph_static_alias_layer_index", -1)) for item in parameter_records)
    require(
        set(observed_layers) == expected_layers
        and set(observed_layers.values()) == {2},
        f"{path}: selected layers or Parameters differ",
    )
    if symbol == "C":
        require(
            all(len(item.get("graph_static_alias_expert_ids") or []) == 5 for item in parameter_records),
            f"{path}: cold-5 expert selection differs",
        )
    else:
        require(
            all(not item.get("graph_static_alias_expert_ids") for item in parameter_records),
            f"{path}: layer arm unexpectedly used an expert subset",
        )

    forward_records = [item for item in records if item.get("event") == "cpu_offload_copy_forward"]
    require(bool(forward_records), f"{path}: copy-forward trace absent")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in forward_records:
        index = int(item.get("forward_index", 0))
        require(index > 0, f"{path}: copy-forward group index absent")
        grouped[index].append(item)
    require(sorted(grouped) == list(range(1, len(grouped) + 1)), f"{path}: forward groups are not contiguous")
    group_summaries: list[dict[str, int]] = []
    for index, group in sorted(grouped.items()):
        require(
            len(group) == expected["layers_per_forward"],
            f"{path}: forward {index} selected layer count differs",
        )
        actual_h2d = sum(int(item.get("graph_static_alias_actual_h2d_bytes", -1)) for item in group)
        copy_bytes = sum(int(item.get("copy_bytes", -1)) for item in group)
        require(
            actual_h2d == expected["selected_bytes"]
            and copy_bytes == expected["selected_bytes"],
            f"{path}: forward {index} actual H2D bytes differ",
        )
        require(
            all(
                item.get("ok") is True
                and item.get("graph_static_alias_copy_submitted") is True
                and item.get("graph_static_alias_lifecycle_state") == "sealed"
                and int(item.get("graph_static_alias_backing_bytes", -1))
                == expected["backing_bytes"]
                for item in group
            ),
            f"{path}: forward {index} copy contract differs",
        )
        group_summaries.append(
            {"forward_index": index, "record_count": len(group), "actual_h2d_bytes": actual_h2d}
        )

    evidence = result.get("offload_manifest_evidence")
    require(isinstance(evidence, dict), f"{path}: manifest evidence absent")
    require(
        int(evidence.get("parameter_record_count", -1)) == expected["parameter_count"],
        f"{path}: manifest evidence Parameter count differs",
    )
    require(
        int(evidence.get("parameter_logical_bytes", -1)) == expected["selected_bytes"],
        f"{path}: manifest evidence logical bytes differ",
    )
    require(
        int(evidence.get("unique_physical_backing_bytes", -1)) == expected["backing_bytes"],
        f"{path}: manifest evidence backing bytes differ",
    )
    return {
        "parameter_count": len(parameter_records),
        "selected_layers": sorted(expected_layers),
        "model_forward_group_count": len(grouped),
        "actual_h2d_bytes_per_forward": expected["selected_bytes"],
        "forward_groups": group_summaries,
        "original_parameter_vas": sorted(
            int(item["uvm_original_parameter_va"]) for item in parameter_records
        ),
    }


def validate_alias_contract(
    result: dict[str, Any],
    *,
    path: Path,
    symbol: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = ARM_GEOMETRY[symbol]
    active, driver = active_contracts(result)
    require(bool(active), f"{path}: active alias diagnostics absent")
    require(active.get("enabled") is True, f"{path}: alias diagnostics disabled")
    require(active.get("backend") == "uvm_original_va", f"{path}: alias backend differs")
    require(active.get("lifecycle_state") == "sealed", f"{path}: alias is not sealed")
    require(active.get("sealed") is True, f"{path}: alias sealed flag is false")
    require(int(active.get("session_id", 0)) > 0, f"{path}: alias session absent")
    require(
        int(active.get("backing_bytes", -1)) == expected["backing_bytes"]
        and int(active.get("active_driver_backing_bytes", -1)) == expected["backing_bytes"],
        f"{path}: active backing bytes differ",
    )
    require(
        int(active.get("selected_parameter_count", -1)) == expected["parameter_count"],
        f"{path}: selected Parameter count differs",
    )
    require(
        int(active.get("selected_expert_count", -1))
        == expected["diagnostics_selected_expert_count"],
        f"{path}: diagnostics selected expert count differs",
    )
    require(
        int(active.get("selected_parameter_bytes_per_forward", -1))
        == expected["selected_bytes"],
        f"{path}: selected H2D bytes per forward differ",
    )
    arenas = active.get("arenas")
    require(isinstance(arenas, list) and bool(arenas), f"{path}: alias arenas absent")
    require(
        all(str(arena.get("pte_mode", "")).upper() == "4K" for arena in arenas),
        f"{path}: not every alias arena uses 4K PTEs",
    )
    require(
        int(driver.get("session_count", -1)) == 1
        and len(driver.get("sessions", [])) == 1
        and driver["sessions"][0].get("sealed") is True,
        f"{path}: live driver session differs",
    )
    seal = active.get("seal_result", {})
    require(
        seal.get("ok") is True
        and seal.get("sealed") is True
        and int(seal.get("alias_count", -1)) == expected["alias_count"]
        and int(seal.get("alias_bytes", -1)) == expected["alias_bytes"]
        and int(seal.get("backing_bytes", -1)) == expected["backing_bytes"],
        f"{path}: exact SEAL topology differs",
    )

    manifest = validate_alias_manifest(records, path=path, symbol=symbol, result=result)
    snapshots = collect_va_snapshots(result, active)
    fingerprints = [tuple(item["vas"]) for item in snapshots]
    require(bool(fingerprints), f"{path}: Parameter VA snapshots absent")
    require(
        all(len(item) == expected["parameter_count"] for item in fingerprints),
        f"{path}: Parameter VA snapshot length differs",
    )
    require(len(set(fingerprints)) == 1, f"{path}: a Parameter VA changed")
    require(
        all(item.get("state") in {None, "sealed"} for item in snapshots),
        f"{path}: alias lifecycle changed during execution",
    )
    require(
        sorted(fingerprints[0]) == manifest["original_parameter_vas"],
        f"{path}: manifest and live original VAs differ",
    )
    return {
        "session_id_before_shutdown": int(active["session_id"]),
        "lifecycle_state_before_shutdown": "sealed",
        "backing_bytes": expected["backing_bytes"],
        "parameter_count": expected["parameter_count"],
        "diagnostics_selected_expert_count": expected[
            "diagnostics_selected_expert_count"
        ],
        "pte_modes": sorted({str(arena["pte_mode"]).upper() for arena in arenas}),
        "parameter_va_snapshot_count": len(snapshots),
        "parameter_va_fingerprint": list(fingerprints[0]),
        "manifest": manifest,
    }


def summarize_process_medians(values: list[float]) -> dict[str, Any]:
    require(bool(values), "cannot summarize an empty process set")
    ordered = sorted(values)
    return {
        "process_count": len(values),
        "process_medians_ms": values,
        "min_process_median_ms": ordered[0],
        "median_of_process_medians_ms": statistics.median(values),
        "mean_process_median_ms": statistics.mean(values),
        "max_process_median_ms": ordered[-1],
    }


def build_performance(
    process_samples: dict[str, dict[int, list[dict[str, Any]]]],
    *,
    prompt_tokens: dict[int, int],
    arm_counts: Counter[str],
    measurement_repeats: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    performance: dict[str, dict[str, Any]] = {}
    medians: dict[str, dict[int, float]] = defaultdict(dict)
    for symbol in ARM_GEOMETRY:
        performance[symbol] = {}
        for batch in (1, 4):
            processes = process_samples[symbol][batch]
            require(
                len(processes) == arm_counts[symbol],
                f"{symbol} batch {batch}: process sample count differs",
            )
            for process in processes:
                require(
                    len(process["measurement_samples_ms"]) == measurement_repeats,
                    f"{symbol} batch {batch}: per-process measurement count differs",
                )
            process_medians = [float(item["process_median_ms"]) for item in processes]
            summary = summarize_process_medians(process_medians)
            median_ms = float(summary["median_of_process_medians_ms"])
            medians[symbol][batch] = median_ms
            performance[symbol][str(batch)] = {
                "batch_size": batch,
                "prompt_tokens": prompt_tokens[batch],
                "measurement_repeats_per_process": measurement_repeats,
                "processes": processes,
                **summary,
                "prefill_tokens_per_second": prompt_tokens[batch] / (median_ms / 1000.0),
            }

    comparisons: dict[str, Any] = {}
    for batch in (1, 4):
        control = medians["Z"][batch]
        absolute = {symbol: medians[symbol][batch] for symbol in ARM_GEOMETRY}
        relative: dict[str, dict[str, float]] = {}
        bandwidth: dict[str, float | None] = {"Z": None}
        per_net_gib: dict[str, float | None] = {"Z": None}
        for symbol, current in absolute.items():
            delta = current - control
            relative[symbol] = {
                "latency_delta_ms": delta,
                "latency_ratio_to_Z": current / control,
                "latency_change_percent": (current / control - 1.0) * 100.0,
                "throughput_ratio_to_Z": control / current,
                "throughput_change_percent": (control / current - 1.0) * 100.0,
            }
            if symbol == "Z":
                continue
            selected_gib = ARM_GEOMETRY[symbol]["selected_bytes"] / 1024**3
            bandwidth[symbol] = (
                selected_gib / (delta / 1000.0) if delta > 0.0 else None
            )
            net_bytes = (
                ARM_GEOMETRY[symbol]["selected_bytes"]
                - ARM_GEOMETRY[symbol]["backing_bytes"]
            )
            per_net_gib[symbol] = (
                delta / (net_bytes / 1024**3) if net_bytes > 0 else None
            )
        cold5_vs_layers: dict[str, dict[str, float]] = {}
        for layer_symbol in ("B", "L"):
            cold = absolute["C"]
            layer = absolute[layer_symbol]
            cold5_vs_layers[layer_symbol] = {
                "C_minus_layer_ms": cold - layer,
                "C_vs_layer_latency_percent": (cold / layer - 1.0) * 100.0,
                "layer_minus_C_ms": layer - cold,
                "layer_vs_C_latency_percent": (layer / cold - 1.0) * 100.0,
            }
        comparisons[str(batch)] = {
            "prompt_tokens": prompt_tokens[batch],
            "absolute_median_ms": absolute,
            "relative_to_Z": relative,
            "cold5_vs_layer_arms": cold5_vs_layers,
            "effective_h2d_bandwidth_gib_per_second": bandwidth,
            "added_latency_ms_per_theoretical_net_gib": per_net_gib,
        }
    return performance, comparisons


def verify(run_root: Path) -> dict[str, Any]:
    run_config, status_rows, result_paths, arm_names = load_run_contract(run_root)
    geometry = normalized_arm_geometry(run_config)
    input_hashes = validate_input_hashes(run_root, run_config)
    geometry_inputs = validate_geometry_inputs(run_config, input_hashes)
    restoration = validate_host_restoration(run_root)
    for result_path in result_paths:
        for snapshot_name in ("gpu.before.json", "gpu.after.json"):
            snapshot = load_json(result_path.parent / snapshot_name)
            require(
                snapshot.get("loaded_srcversion") == restoration["custom_srcversion"],
                f"{result_path.parent}: {snapshot_name} did not use custom UVM",
            )

    model_config_path = Path(str(run_config.get("model"))).resolve() / "config.json"
    fixture_path = Path(str(run_config.get("fixture"))).resolve()
    require(model_config_path.is_file(), "model config is absent")
    require(fixture_path.is_file(), "fixture is absent")
    require(
        sha256_file(model_config_path) == run_config.get("model_config_sha256"),
        "model config hash differs",
    )
    require(
        sha256_file(fixture_path) == run_config.get("fixture_sha256"),
        "fixture hash differs",
    )
    require(
        set(run_root.glob("??_*/result.json")) == set(result_paths),
        "unexpected result file set",
    )

    results: list[tuple[Path, dict[str, Any], dict[str, str]]] = []
    common_fingerprint: dict[str, Any] | None = None
    versions_fingerprint: dict[str, Any] | None = None
    code_fingerprint: dict[str, Any] | None = None
    process_samples: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    reference_hashes: dict[tuple[int, str, int], str] | None = None
    hash_evidence = Counter()
    prompt_token_reference: dict[int, int] | None = None
    contract_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path, row in zip(result_paths, status_rows, strict=True):
        result = load_json(path)
        require(result.get("schema") == RESULT_SCHEMA, f"{path}: result schema differs")
        require(result.get("status") == "ok", f"{path}: status={result.get('status')}")
        require(not result.get("wrapper_contract_errors"), f"{path}: wrapper contract errors")
        symbol = str(result.get("arm", ""))
        require(symbol == row["symbol"], f"{path}: arm symbol differs")
        current_common, current_versions, current_code = validate_result_config(
            result,
            path=path,
            symbol=symbol,
            row=row,
            run_config=run_config,
            common_fingerprint=common_fingerprint,
            versions_fingerprint=versions_fingerprint,
            code_fingerprint=code_fingerprint,
        )
        if common_fingerprint is None:
            common_fingerprint = current_common
        if versions_fingerprint is None:
            versions_fingerprint = current_versions
        if code_fingerprint is None and current_code is not None:
            code_fingerprint = current_code

        hashes, measurements, prompt_tokens = validate_cycles(
            result, path=path, run_config=run_config
        )
        if reference_hashes is None:
            reference_hashes = hashes
        else:
            require(hashes == reference_hashes, f"{path}: output hashes differ across arms/processes")
        hash_evidence[symbol] += len(hashes)
        if prompt_token_reference is None:
            prompt_token_reference = prompt_tokens
        else:
            require(prompt_tokens == prompt_token_reference, f"{path}: prompt tokens differ")

        manifest_path = manifest_path_for_result(path, result)
        records = load_jsonl(manifest_path, allow_empty=True)
        if symbol == "Z":
            active_evidence = validate_control_contract(result, path=path, records=records)
        else:
            active_evidence = validate_alias_contract(
                result, path=path, symbol=symbol, records=records
            )
        shutdown_evidence = validate_shutdown(result, path=path, symbol=symbol)
        contract_evidence[symbol].append(
            {
                "directory": row["directory"],
                "manifest_sha256": sha256_file(manifest_path),
                "active": active_evidence,
                "shutdown": shutdown_evidence,
            }
        )

        for batch in (1, 4):
            samples = measurements[batch]
            process_samples[symbol][batch].append(
                {
                    "directory": row["directory"],
                    "batch_order": row["batch_order"],
                    "measurement_samples_ms": samples,
                    "process_median_ms": statistics.median(samples),
                }
            )
        results.append((path, result, row))

    seen_arms = {str(result["arm"]) for _, result, _ in results}
    require(seen_arms == set(ARM_GEOMETRY), f"arm coverage differs: {seen_arms}")
    require(reference_hashes is not None, "output hash reference absent")
    require(prompt_token_reference is not None, "prompt token reference absent")
    arm_counts = Counter(str(result["arm"]) for _, result, _ in results)
    common_contract = common_contract_from_config(run_config)
    performance, comparisons = build_performance(
        process_samples,
        prompt_tokens=prompt_token_reference,
        arm_counts=arm_counts,
        measurement_repeats=int(common_contract["measurement_repeats"]),
    )

    batch_order_counts: dict[str, dict[str, int]] = {}
    for symbol in ARM_GEOMETRY:
        counts = Counter(row["batch_order"] for row in status_rows if row["symbol"] == symbol)
        batch_order_counts[symbol] = dict(sorted(counts.items()))

    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "ok",
        "run_root": str(run_root.resolve()),
        "case_count": len(results),
        "arms": list(ARM_GEOMETRY),
        "arm_names": arm_names,
        "arm_geometry": geometry,
        "geometry_inputs": geometry_inputs,
        "run_contract": {
            "mode": run_config["mode"],
            "order": run_config["order"],
            "all_returncodes_zero": True,
            "common_config_equal": True,
            "versions_equal": True,
            "code_hashes_verified": True,
            "input_hash_count": len(input_hashes),
            "arm_counts": dict(sorted(arm_counts.items())),
            "batch_order_counts": batch_order_counts,
            "conditioning_repeats_per_process": int(common_contract["conditioning_repeats"]),
            "measurement_repeats_per_process": int(common_contract["measurement_repeats"]),
        },
        "host_restoration": restoration,
        "output_hash_contract": {
            "comparison_scope": "all_Z_C_B_L_processes_and_roles",
            "hashes_per_process": len(reference_hashes),
            "evidence_count_by_arm": dict(sorted(hash_evidence.items())),
        },
        "alias_and_shutdown_contracts": dict(contract_evidence),
        "aggregation_method": (
            "For each process and batch, take the median of that process's "
            "measurement-role prefill samples; then take the median of those "
            "process medians across independent processes. Conditioning samples "
            "are excluded from performance aggregation."
        ),
        "prefill_performance": performance,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = verify(args.run_root.resolve())
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
