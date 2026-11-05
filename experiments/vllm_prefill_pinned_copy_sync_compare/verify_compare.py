#!/usr/bin/env python3
"""Verify the stock-UVM pinned-copy async/sync steady comparison.

The measured unit is one 4-request/1217-token batch.  Each process performs an
initial warmup, three conditioning batches, and five measured batches.  CUDA
timing is deliberately available only in the sync arm: the async arm records
host submission without introducing an event synchronization into each layer.
"""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
STEADY_ROOT = REPO_ROOT / "experiments" / "vllm_prefill_p3_steady_compare"
OLD_ROOT = REPO_ROOT / "experiments" / "vllm_prefill_p3_pinned_uva_compare"
sys.path.insert(0, str(STEADY_ROOT))
sys.path.insert(0, str(OLD_ROOT))
import verify_steady_compare as steady  # noqa: E402
import verify_compare as old  # noqa: E402


ORDER = "ASSA"
LAYERS = tuple(range(4))
LAYER_BYTES = 603_979_776
TOTAL_BYTES = 2_415_919_104
W13_BYTES = 402_653_184
W2_BYTES = 201_326_592
# vLLM executes one internal profiling/bootstrap forward after the selected
# parameters have been offloaded and before the benchmark's declared warmup.
# The manifest therefore contains one bootstrap plus the 9 declared batches
# (warmup + 3 conditioning + 5 measurement).
TOTAL_BATCHES = 10
DECLARED_BATCHES = 9
CONDITIONING_BATCHES = 3
MEASUREMENT_BATCHES = 5
EXPECTED_NAMES = tuple(
    f"model.layers.{layer}.mlp.experts.{weight}"
    for layer in LAYERS
    for weight in ("w13_weight", "w2_weight")
)
EXPECTED_PARAMETER_BYTES = {
    f"model.layers.{layer}.mlp.experts.w13_weight": W13_BYTES
    for layer in LAYERS
} | {
    f"model.layers.{layer}.mlp.experts.w2_weight": W2_BYTES
    for layer in LAYERS
}
MODES = {"A": "async", "S": "sync"}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    require(path.is_file(), f"missing JSONL artifact: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"{path}:{line_number}: expected object")
        records.append(value)
    return records


def finite(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label}: expected finite number",
    )
    return float(value)


def positive(value: Any, label: str) -> float:
    result = finite(value, label)
    require(result > 0.0, f"{label}: expected positive number")
    return result


def distribution(values: list[float]) -> dict[str, Any]:
    require(bool(values), "cannot summarize an empty sample")
    return {
        "count": len(values),
        "samples": values,
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_status(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing execution status: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def gpu_snapshot_clean(path: Path) -> None:
    require(path.is_file(), f"missing GPU snapshot: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(bool(lines), f"{path}: empty GPU snapshot")
    first = next(csv.reader([lines[0]]))
    require(len(first) >= 6, f"{path}: malformed GPU row")
    try:
        used_mib = int(first[3].strip())
    except ValueError as exc:
        raise VerificationError(f"{path}: invalid GPU memory value") from exc
    require(used_mib == 0, f"{path}: GPU used memory is {used_mib} MiB, expected 0")
    for line in lines[1:]:
        fields = next(csv.reader([line]))
        if len(fields) >= 2 and fields[1].strip().isdigit():
            raise VerificationError(f"{path}: compute process present: {line}")


def verify_plan(run_root: Path, config: dict[str, Any]) -> Path:
    names_path = run_root / "inputs" / "exact_parameter_names.txt"
    plan_path = run_root / "inputs" / "exact_l0_l3_plan.json"
    require(names_path.is_file() and plan_path.is_file(), "registered input files absent")
    names = tuple(
        line.strip()
        for line in names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    require(names == EXPECTED_NAMES, "exact parameter names differ from L0-L3 w13/w2")
    require(
        config.get("exact_names_sha256") == sha256(names_path),
        "exact-name SHA-256 differs",
    )
    plan = load_json(plan_path)
    metadata = plan.get("metadata")
    require(isinstance(metadata, dict), "placement plan metadata absent")
    require(
        tuple(metadata.get("copy_parameter_names", ())) == EXPECTED_NAMES,
        "placement plan copy names differ",
    )
    require(metadata.get("actual_offload_main_bytes") == TOTAL_BYTES, "plan bytes differ")
    require(metadata.get("cold_layers") == list(LAYERS), "plan layers differ")
    return names_path.resolve()


def expected_mode_environment(mode: str) -> dict[str, str]:
    return {
        "VLLM_USE_UVM": "0",
        "VLLM_CPU_OFFLOAD_BACKEND": "copy",
        "VLLM_CPU_OFFLOAD_STATIC_BANK": "0",
        "VLLM_CPU_OFFLOAD_COPY_TRACE": "1",
        "VLLM_CPU_OFFLOAD_COPY_SYNC": "1" if mode == "sync" else "0",
        "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE": "0",
        "VLLM_UVM_EXPERT_WEIGHTS_MANAGED": "0",
        "VLLM_UVM_EXPERT_ENVELOPE_ENABLE": "0",
    }


def verify_parameters(
    records: list[dict[str, Any]],
    *,
    names_path: Path,
    label: str,
) -> dict[str, Any]:
    parameters = [item for item in records if item.get("event") == "cpu_offload_parameter"]
    require(len(parameters) == len(EXPECTED_NAMES), f"{label}: parameter record count differs")
    require(
        tuple(item.get("full_parameter_name") for item in parameters) == EXPECTED_NAMES,
        f"{label}: parameter selection/order differs",
    )
    copied_bytes = 0
    for item in parameters:
        name = str(item.get("full_parameter_name"))
        require(
            item.get("bytes") == EXPECTED_PARAMETER_BYTES[name],
            f"{label}: {name} bytes differ",
        )
        copied_bytes += int(item["bytes"])
        expected = {
            "cpu_data_pinned": True,
            "copy_on_forward": True,
            "copy_exact_names": True,
            "copy_static_bank": False,
            "pin_memory": True,
            "uva_offloading": False,
            "device_after": "cpu",
        }
        for key, wanted in expected.items():
            require(item.get(key) == wanted, f"{label}: {name} field {key} differs")
        require(
            Path(str(item.get("copy_exact_names_file"))).resolve() == names_path,
            f"{label}: {name} exact-name file differs",
        )
    require(copied_bytes == TOTAL_BYTES, f"{label}: selected bytes differ")
    return {"parameter_count": len(parameters), "selected_bytes": copied_bytes}


def window_copy_records(
    copies: list[dict[str, Any]],
    batch_ranges: list[dict[str, Any]],
    *,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    windows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for ordinal, window in enumerate(batch_ranges, 1):
        start = finite(window.get("start_epoch"), f"{label}: batch start")
        end = finite(window.get("end_epoch"), f"{label}: batch end")
        contained = [
            item
            for item in copies
            if start <= finite(item.get("timestamp"), f"{label}: copy timestamp") <= end
        ]
        require(len(contained) == len(LAYERS), f"{label}: batch {ordinal} copy count differs")
        require(
            [item.get("module_prefix") for item in contained]
            == [f"model.layers.{layer}" for layer in LAYERS],
            f"{label}: batch {ordinal} layer order differs",
        )
        require(
            {item.get("forward_index") for item in contained} == {ordinal + 1},
            f"{label}: batch {ordinal} forward index differs",
        )
        for item in contained:
            identity = id(item)
            require(identity not in seen_ids, f"{label}: copy record belongs to two batches")
            seen_ids.add(identity)
        windows.append(
            {
                "role": window["role"],
                "index": window["index"],
                "forward_index": ordinal + 1,
                "records": contained,
            }
        )
    bootstrap = [item for item in copies if id(item) not in seen_ids]
    require(len(bootstrap) == len(LAYERS), f"{label}: bootstrap copy count differs")
    require(
        [item.get("module_prefix") for item in bootstrap]
        == [f"model.layers.{layer}" for layer in LAYERS],
        f"{label}: bootstrap layer order differs",
    )
    require(
        {item.get("forward_index") for item in bootstrap} == {1},
        f"{label}: bootstrap forward index differs",
    )
    first_declared_start = min(
        finite(window.get("start_epoch"), f"{label}: declared batch start")
        for window in batch_ranges
    )
    require(
        all(
            finite(item.get("timestamp"), f"{label}: bootstrap timestamp")
            < first_declared_start
            for item in bootstrap
        ),
        f"{label}: bootstrap copy overlaps a declared batch",
    )
    require(
        len(seen_ids) + len(bootstrap) == len(copies),
        f"{label}: copy record accounting differs",
    )
    return windows, bootstrap


def verify_copies(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    mode: str,
    names_path: Path,
    label: str,
) -> dict[str, Any]:
    copies = [item for item in records if item.get("event") == "cpu_offload_copy_forward"]
    require(len(copies) == TOTAL_BATCHES * len(LAYERS), f"{label}: copy record count differs")
    for item in copies:
        prefix = str(item.get("module_prefix"))
        require(
            prefix in {f"model.layers.{layer}" for layer in LAYERS},
            f"{label}: unexpected layer",
        )
        expected = {
            "copy_bytes": LAYER_BYTES,
            "copy_tensor_count": 2,
            "copy_exact_names": True,
            "copy_exact_name_count": 2,
            "copy_static_bank": False,
            "copy_sync": mode == "sync",
            "ok": True,
        }
        for key, wanted in expected.items():
            require(item.get(key) == wanted, f"{label}: {prefix} field {key} differs")
        positive(item.get("copy_submit_us"), f"{label}: copy_submit_us")
        require(
            Path(str(item.get("copy_exact_names_file"))).resolve() == names_path,
            f"{label}: copy exact-name file differs",
        )
        if mode == "sync":
            positive(item.get("copy_cuda_ms"), f"{label}: copy_cuda_ms")
        else:
            require(
                item.get("copy_cuda_ms") is None,
                f"{label}: async CUDA timing unexpectedly present",
            )

    windows, bootstrap = window_copy_records(
        copies, summary["batch_ranges"], label=label
    )
    require(
        len(windows) == DECLARED_BATCHES,
        f"{label}: declared copy-window count differs",
    )
    measured = [item for item in windows if item["role"] == "measurement"]
    require(len(measured) == MEASUREMENT_BATCHES, f"{label}: measured copy windows differ")
    measured_records = [record for window in measured for record in window["records"]]
    require(
        len(measured_records) == MEASUREMENT_BATCHES * len(LAYERS),
        f"{label}: measured copy count differs",
    )
    evidence: dict[str, Any] = {
        "copy_records": len(copies),
        "bootstrap_copy_records": len(bootstrap),
        "measurement_copy_records": len(measured_records),
        "copy_submit_us_all": distribution(
            [positive(item["copy_submit_us"], f"{label}: submit") for item in copies]
        ),
        "copy_submit_us_measurement": distribution(
            [
                positive(item["copy_submit_us"], f"{label}: measured submit")
                for item in measured_records
            ]
        ),
    }
    if mode == "sync":
        by_layer: dict[str, list[float]] = defaultdict(list)
        batch_sums: list[float] = []
        for window in measured:
            values = [
                positive(item["copy_cuda_ms"], f"{label}: measured CUDA copy")
                for item in window["records"]
            ]
            batch_sums.append(sum(values))
            for item, value in zip(window["records"], values):
                by_layer[str(item["module_prefix"])].append(value)
        evidence["copy_cuda_ms_by_layer"] = {
            layer: distribution(values) for layer, values in sorted(by_layer.items())
        }
        evidence["copy_cuda_ms_batch_sum"] = distribution(batch_sums)
    return evidence


def verify_case(
    case_dir: Path,
    *,
    ordinal: int,
    symbol: str,
    names_path: Path,
) -> dict[str, Any]:
    mode = MODES[symbol]
    label = f"case {ordinal}/{mode}"
    require(case_dir.is_dir(), f"{label}: directory absent")
    gpu_snapshot_clean(case_dir / "gpu.before.csv")
    gpu_snapshot_clean(case_dir / "gpu.after.csv")
    require(
        (case_dir / "returncode.txt").read_text().strip() == "0",
        f"{label}: nonzero return code",
    )
    env = old.load_env(case_dir / "environment.txt")
    for key, wanted in expected_mode_environment(mode).items():
        require(env.get(key) == wanted, f"{label}: environment {key} differs")
    require(
        Path(str(env.get("VLLM_CPU_OFFLOAD_EXACT_NAMES_FILE"))).resolve() == names_path,
        f"{label}: exact-name environment differs",
    )

    result, summary = steady.verify_result(
        case_dir / "result.json",
        concurrency=4,
        warmup_requests=4,
        conditioning_batches=CONDITIONING_BATCHES,
        measurement_batches=MEASUREMENT_BATCHES,
        disable_chunked_prefill=True,
    )
    require(
        result.get("mechanism_class") == f"vllm_prefill_pinned_copy_exact_l0_l3_{mode}",
        f"{label}: mechanism class differs",
    )
    config = result.get("config")
    require(isinstance(config, dict), f"{label}: result config absent")
    require(
        finite(config.get("cpu_offload_gb"), f"{label}: cpu_offload_gb") == 3.0,
        f"{label}: cpu offload budget differs",
    )
    records = load_jsonl(case_dir / "cpu_offload_manifest.jsonl")
    parameter_evidence = verify_parameters(records, names_path=names_path, label=label)
    copy_evidence = verify_copies(
        records,
        summary,
        mode=mode,
        names_path=names_path,
        label=label,
    )
    return {
        "ordinal": ordinal,
        "symbol": symbol,
        "mode": mode,
        "directory": str(case_dir),
        **summary,
        "parameters": parameter_evidence,
        "copy": copy_evidence,
    }


def comparison(sync_ms: float, async_ms: float) -> dict[str, float]:
    return {
        "sync_minus_async_ms": sync_ms - async_ms,
        "sync_over_async": sync_ms / async_ms,
        "async_host_submission_advantage_percent": (sync_ms / async_ms - 1.0) * 100.0,
    }


def verify(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    config = load_json(run_root / "run_config.json")
    expected_config = {
        "schema": "vllm_pinned_copy_sync_steady_v1",
        "order": ORDER,
        "modes": {"A": "async", "S": "sync"},
        "custom_module_cycle": False,
        "conditioning_batches": CONDITIONING_BATCHES,
        "measurement_batches": MEASUREMENT_BATCHES,
        "inter_batch_idle_ms": 0.0,
        "copy_trace": True,
        "layers": list(LAYERS),
        "bytes_per_layer": LAYER_BYTES,
        "total_bytes_per_batch": TOTAL_BYTES,
    }
    for key, wanted in expected_config.items():
        require(config.get(key) == wanted, f"run config {key} differs")
    stock = str(config.get("stock_srcversion", ""))
    loaded = str(config.get("loaded_srcversion", ""))
    require(bool(stock) and loaded == stock, "suite did not use stock UVM")
    require(
        not any(run_root.glob("*module_cycle*")),
        "custom module-cycle artifact unexpectedly present",
    )
    names_path = verify_plan(run_root, config)
    gpu_snapshot_clean(run_root / "gpu.initial.csv")
    gpu_snapshot_clean(run_root / "gpu.final.csv")

    rows = read_status(run_root / "execution_status.tsv")
    require(len(rows) == len(ORDER), "execution row count differs")
    cases: list[dict[str, Any]] = []
    for ordinal, (symbol, row) in enumerate(zip(ORDER, rows), 1):
        mode = MODES[symbol]
        require(row.get("symbol") == symbol, f"case {ordinal}: status symbol differs")
        require(row.get("mode") == mode, f"case {ordinal}: status mode differs")
        require(row.get("returncode") == "0", f"case {ordinal}: execution failed")
        case_dir = run_root / str(row.get("directory"))
        cases.append(
            verify_case(
                case_dir,
                ordinal=ordinal,
                symbol=symbol,
                names_path=names_path,
            )
        )

    eval_hashes = {item["evaluation_output_hash"] for item in cases}
    warmup_hashes = {item["warmup_output_hash"] for item in cases}
    require(len(eval_hashes) == 1, "evaluation output hashes differ")
    require(len(warmup_hashes) == 1, "warmup output hashes differ")

    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in cases:
        by_mode[item["mode"]].append(item)
    require(
        all(len(by_mode[mode]) == 2 for mode in ("async", "sync")),
        "mode process count differs",
    )

    process_medians = {
        mode: [float(item["elapsed_ms_process_median"]) for item in items]
        for mode, items in by_mode.items()
    }
    pooled_batches = {
        mode: [float(value) for item in items for value in item["elapsed_ms_batches"]]
        for mode, items in by_mode.items()
    }
    process_summary = {mode: distribution(values) for mode, values in process_medians.items()}
    pooled_summary = {mode: distribution(values) for mode, values in pooled_batches.items()}
    process_comparison = comparison(
        process_summary["sync"]["median"], process_summary["async"]["median"]
    )
    pooled_comparison = comparison(
        pooled_summary["sync"]["median"], pooled_summary["async"]["median"]
    )

    pair_specs = ((1, 2), (4, 3))
    pairs: list[dict[str, Any]] = []
    for async_ordinal, sync_ordinal in pair_specs:
        async_case = cases[async_ordinal - 1]
        sync_case = cases[sync_ordinal - 1]
        async_ms = float(async_case["elapsed_ms_process_median"])
        sync_ms = float(sync_case["elapsed_ms_process_median"])
        pairs.append(
            {
                "async_ordinal": async_ordinal,
                "sync_ordinal": sync_ordinal,
                **comparison(sync_ms, async_ms),
            }
        )

    sync_layer_samples: dict[str, list[float]] = defaultdict(list)
    sync_batch_sums: list[float] = []
    for item in by_mode["sync"]:
        copy = item["copy"]
        sync_batch_sums.extend(copy["copy_cuda_ms_batch_sum"]["samples"])
        for layer, values in copy["copy_cuda_ms_by_layer"].items():
            sync_layer_samples[layer].extend(values["samples"])

    submit_by_mode = {
        mode: distribution(
            [
                float(value)
                for item in items
                for value in item["copy"]["copy_submit_us_measurement"]["samples"]
            ]
        )
        for mode, items in by_mode.items()
    }
    return {
        "schema": "vllm_pinned_copy_sync_steady_verification_v1",
        "status": "pass",
        "run_root": str(run_root),
        "order": ORDER,
        "stock_uvm_srcversion": stock,
        "measurement_contract": {
            "concurrency": 4,
            "prompt_tokens": 1217,
            "output_tokens": 1,
            "warmup_batches": 1,
            "conditioning_batches": CONDITIONING_BATCHES,
            "measurement_batches": MEASUREMENT_BATCHES,
            "inter_batch_idle_ms": 0.0,
            "selected_bytes_per_batch": TOTAL_BYTES,
        },
        "elapsed_ms_process_median": process_summary,
        "elapsed_ms_pooled_batches": pooled_summary,
        "process_median_comparison": process_comparison,
        "pooled_batch_comparison": pooled_comparison,
        "paired_process_blocks": pairs,
        "copy_submit_us_measurement": submit_by_mode,
        "sync_copy_cuda_ms_by_layer": {
            layer: distribution(values) for layer, values in sorted(sync_layer_samples.items())
        },
        "sync_copy_cuda_ms_batch_sum": distribution(sync_batch_sums),
        "interpretation": (
            "sync-minus-async is the observed async host-submission advantage under "
            "this workload; it must not be described as eliminating the CE copy."
        ),
        "output_token_ids_sha256": next(iter(eval_hashes)),
        "warmup_output_token_ids_sha256": next(iter(warmup_hashes)),
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = verify(args.run_root)
    except (
        VerificationError,
        steady.VerificationError,
        old.VerificationError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
