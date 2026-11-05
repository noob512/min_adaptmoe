#!/usr/bin/env python3
"""Verify the AdaptMoE always-copy continuous-schedule probe."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        records.append(value)
    return records


def output_hashes(result: dict[str, Any]) -> dict[str, str]:
    return {
        str(record["logical_request"]): str(record["output_token_ids_sha256"])
        for record in result.get("requests", [])
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    result = load(args.run_root / "result.json")
    reference = load(args.reference)
    manifest = load_jsonl(args.run_root / "cpu_offload_manifest.jsonl")
    run_log = (args.run_root / "run.log").read_text(
        encoding="utf-8", errors="replace"
    )
    kernel_log = (args.run_root / "kernel.since.log").read_text(
        encoding="utf-8", errors="replace"
    )

    summary = result.get("summary", {})
    mixed_steps = [
        record
        for record in result.get("scheduler_steps", [])
        if record.get("mixed_new_and_cached") is True
    ]
    if result.get("status") != "ok":
        errors.append(f"runner status is {result.get('status')!r}")
    if summary.get("successful_requests") != 8:
        errors.append("not all eight requests completed")
    if summary.get("generated_tokens") != 128:
        errors.append("generated token count is not 8 x 16")
    if len(mixed_steps) != 1:
        errors.append(f"expected exactly one mixed scheduler step, got {len(mixed_steps)}")
    elif not (
        len(mixed_steps[0].get("scheduled_new_request_ids", [])) == 4
        and len(mixed_steps[0].get("scheduled_cached_request_ids", [])) == 4
        and mixed_steps[0].get("total_num_scheduled_tokens") == 516
    ):
        errors.append("mixed step is not the expected 4x128 Prefill + 4x1 Decode")

    if not re.search(
        r"CUDA_GRAPH_RUNTIME_AUDIT phase=enabled:mixed_step .*runtime_mode=NONE",
        run_log,
    ):
        errors.append("mixed step was not observed on the eager copy path")
    if not re.search(
        r"CUDA_GRAPH_RUNTIME_AUDIT phase=enabled:decode_step .*runtime_mode=FULL",
        run_log,
    ):
        errors.append("uniform Decode did not dispatch its copy-bearing Graph")

    configs = [
        record
        for record in manifest
        if record.get("event") == "cpu_offload_config"
    ]
    if len(configs) != 1 or not (
        configs[0].get("graph_static_alias") is True
        and configs[0].get("graph_static_alias_backend") == "uvm_original_va"
        and configs[0].get("uvm_original_va_phase_mode") == "off"
    ):
        errors.append("manifest does not certify fixed original-VA static aliases")
    finalized = [
        record
        for record in manifest
        if record.get("event") == "cpu_offload_uvm_static_alias_finalized"
    ]
    if len(finalized) != 1 or not (
        finalized[0].get("ok") is True
        and finalized[0].get("state") == "sealed"
        and finalized[0].get("sealed") is True
    ):
        errors.append("static aliases were not finalized in the sealed state")
    captured_copy_records = [
        record
        for record in manifest
        if record.get("event") == "cpu_offload_copy_forward"
        and record.get("graph_static_alias_capture_phase") is True
        and record.get("graph_static_alias_capture_active") is True
        and record.get("graph_static_alias_lifecycle_state") == "sealed"
        and int(record.get("graph_static_alias_actual_h2d_bytes", 0)) > 0
    ]
    if not captured_copy_records:
        errors.append("Decode Graph capture contains no certified H2D nodes")
    eager_copy_records = [
        record
        for record in manifest
        if record.get("event") == "cpu_offload_copy_forward"
        and record.get("graph_static_alias_capture_phase") is False
        and record.get("graph_static_alias_capture_active") is False
        and record.get("graph_static_alias_lifecycle_state") == "sealed"
        and int(record.get("graph_static_alias_actual_h2d_bytes", 0)) > 0
    ]
    if not eager_copy_records:
        errors.append("Prefill/mixed eager path contains no certified H2D copies")

    remote_events = [
        record
        for record in manifest
        if record.get("event")
        in (
            "cpu_offload_uvm_original_va_remote_transition",
            "cpu_offload_uvm_original_va_profile_all_remote_prepared",
            "cpu_offload_uvm_original_va_phase_poisoned",
        )
    ]
    if remote_events:
        errors.append("always-copy arm unexpectedly entered a remote/poisoned state")

    compared_hashes = output_hashes(result)
    reference_hashes = output_hashes(reference)
    matching_output_hashes = sum(
        compared_hashes.get(request) == digest
        for request, digest in reference_hashes.items()
    )
    if reference.get("status") != "ok" or len(reference_hashes) != 8:
        errors.append("continuous remote reference is not a complete successful run")
    if compared_hashes != reference_hashes:
        warnings.append(
            "output token hashes differ from the same-trace continuous reference "
            f"({matching_output_hashes}/8 requests match)"
        )

    fault_tokens = ("Xid ", "FAULT_PDE", "FAULT_PTE", "BUG:", "Oops:")
    if any(token in kernel_log for token in fault_tokens):
        errors.append("kernel log contains a GPU/kernel fault")
    gpu_after = load(args.run_root / "gpu.after.json")
    module_after = load(args.run_root / "module.after.json")
    if gpu_after.get("gpu_clean") is not True:
        errors.append("GPU did not clean after the runner")
    if module_after.get("gpu_clean") is not True:
        errors.append("stock-module restore snapshot is not clean")

    result_throughput = summary.get("request_throughput_per_second")
    reference_throughput = reference.get("summary", {}).get(
        "request_throughput_per_second"
    )
    throughput_ratio = None
    if isinstance(result_throughput, (int, float)) and isinstance(
        reference_throughput, (int, float)
    ) and reference_throughput > 0:
        throughput_ratio = result_throughput / reference_throughput

    def mean_metric(payload: dict[str, Any], name: str) -> float | None:
        value = payload.get("summary", {}).get(name, {}).get("mean")
        return float(value) if isinstance(value, (int, float)) else None

    def ratio(left: float | None, right: float | None) -> float | None:
        if left is None or right is None or right <= 0:
            return None
        return left / right

    ttft_mean = mean_metric(result, "ttft_ms")
    reference_ttft_mean = mean_metric(reference, "ttft_ms")
    tpot_mean = mean_metric(result, "tpot_ms")
    reference_tpot_mean = mean_metric(reference, "tpot_ms")
    e2e_mean = mean_metric(result, "e2e_ms")
    reference_e2e_mean = mean_metric(reference, "e2e_ms")
    selected_bytes_per_forward = (
        int(finalized[0].get("cold_residency_bytes", 0)) if finalized else 0
    )
    engine_steps = int(summary.get("engine_steps", 0))

    verification = {
        "schema": "adaptmoe_mixed_schedule_probe_verification_v1",
        "status": (
            "error"
            if errors
            else "ok_with_output_drift"
            if warnings
            else "ok"
        ),
        "errors": errors,
        "warnings": warnings,
        "claim_scope": "single-process compatibility qualification only",
        "mechanism": (
            "continuous scheduling with sealed aliases throughout; Prefill/mixed "
            "use eager H2D and Decode uses H2D-bearing FULL_DECODE_ONLY Graphs"
        ),
        "evidence": {
            "successful_requests": summary.get("successful_requests"),
            "generated_tokens": summary.get("generated_tokens"),
            "mixed_scheduler_steps": len(mixed_steps),
            "mixed_step_indices": [step.get("scheduler_step") for step in mixed_steps],
            "captured_h2d_record_count": len(captured_copy_records),
            "eager_h2d_record_count": len(eager_copy_records),
            "remote_or_poison_event_count": len(remote_events),
            "output_hashes_equal_reference": compared_hashes == reference_hashes,
            "matching_output_hash_requests": matching_output_hashes,
            "request_throughput_per_second": result_throughput,
            "reference_remote_request_throughput_per_second": reference_throughput,
            "throughput_over_remote_reference": throughput_ratio,
            "throughput_change_percent": (
                (throughput_ratio - 1.0) * 100.0
                if throughput_ratio is not None
                else None
            ),
            "mean_ttft_ms": ttft_mean,
            "reference_remote_mean_ttft_ms": reference_ttft_mean,
            "ttft_over_remote_reference": ratio(ttft_mean, reference_ttft_mean),
            "mean_tpot_ms": tpot_mean,
            "reference_remote_mean_tpot_ms": reference_tpot_mean,
            "tpot_over_remote_reference": ratio(tpot_mean, reference_tpot_mean),
            "mean_e2e_ms": e2e_mean,
            "reference_remote_mean_e2e_ms": reference_e2e_mean,
            "e2e_over_remote_reference": ratio(e2e_mean, reference_e2e_mean),
            "selected_h2d_bytes_per_forward": selected_bytes_per_forward,
            "logical_trace_h2d_bytes": selected_bytes_per_forward * engine_steps,
            "queueing_ms": summary.get("queueing_ms"),
            "ttft_ms": summary.get("ttft_ms"),
            "tpot_ms": summary.get("tpot_ms"),
        },
    }
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
