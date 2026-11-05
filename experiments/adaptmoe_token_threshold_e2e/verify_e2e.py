#!/usr/bin/env python3
"""Verify one real-model AdaptMoE token-threshold continuous-schedule run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        value
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for value in [json.loads(line)]
        if isinstance(value, dict)
    ]


def output_hashes(result: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["logical_request"]): str(item["output_token_ids_sha256"])
        for item in result.get("requests", [])
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=int)
    parser.add_argument("--resume-threshold", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    result = load(args.run_root / "result.json")
    plan = load(args.plan)
    reference = load(args.reference)
    manifest = load_jsonl(args.run_root / "cpu_offload_manifest.jsonl")
    kernel_log = (args.run_root / "kernel.since.log").read_text(
        encoding="utf-8", errors="replace"
    )
    errors: list[str] = []

    scheduler_steps = result.get("scheduler_steps", [])
    token_counts = [
        int(step.get("total_num_scheduled_tokens", 0))
        for step in scheduler_steps
    ]
    mixed = [step for step in scheduler_steps if step.get("mixed_new_and_cached")]
    if result.get("status") != "ok":
        errors.append(f"runner status is {result.get('status')!r}")
    if result.get("summary", {}).get("successful_requests") != 8:
        errors.append("not all eight requests completed")
    if len(mixed) != 1 or int(mixed[0].get("total_num_scheduled_tokens", 0)) != 516:
        errors.append("expected exactly one 516-token mixed scheduler step")
    if not any(0 < count <= args.threshold for count in token_counts):
        errors.append("scheduler emitted no remote-side token count")
    if not any(count >= args.resume_threshold for count in token_counts):
        errors.append("scheduler emitted no copy-side token count")

    configs = [item for item in manifest if item.get("event") == "cpu_offload_config"]
    if len(configs) != 1 or not (
        configs[0].get("uvm_original_va_phase_mode") == "token_threshold"
        and int(configs[0].get("uvm_original_va_token_threshold", -1))
        == args.threshold
        and int(
            configs[0].get("uvm_original_va_resume_token_threshold", -1)
        )
        == args.resume_threshold
        and configs[0].get("uvm_token_threshold_fast_transition") is True
        and configs[0].get("graph_static_alias_backend") == "uvm_original_va"
    ):
        errors.append(
            "manifest does not certify token-threshold fast mode with "
            f"park={args.threshold}, resume={args.resume_threshold}"
        )
    finalized = [
        item
        for item in manifest
        if item.get("event") == "cpu_offload_uvm_static_alias_finalized"
    ]
    selected_bytes = int(finalized[0].get("cold_residency_bytes", 0)) if finalized else 0
    if len(finalized) != 1 or not (
        finalized[0].get("ok") is True
        and finalized[0].get("state") == "sealed"
        and selected_bytes == int(plan.get("selected_main_bytes", -1))
    ):
        errors.append("fixed offload tier did not finalize with the planned bytes")

    parks = [
        item
        for item in manifest
        if item.get("event") == "cpu_offload_uvm_original_va_remote_transition"
        and item.get("reason") == "token_threshold_remote"
        and item.get("post_state") == "remote_parked"
        and item.get("ok") is True
    ]
    resumes = [
        item
        for item in manifest
        if item.get("event") == "cpu_offload_uvm_original_va_alias_prefill_activated"
        and item.get("pre_state") == "remote_parked"
        and item.get("post_state") == "sealed"
        and item.get("ok") is True
    ]
    poisoned = [
        item
        for item in manifest
        if item.get("event") == "cpu_offload_uvm_original_va_phase_poisoned"
    ]
    if len(parks) < 2:
        errors.append(f"expected at least two successful PARKs, got {len(parks)}")
    if not resumes:
        errors.append("mixed large-token step did not RESUME the copy topology")
    if poisoned:
        errors.append("phase topology was poisoned")
    if any(
        item.get("fast_runtime_transition") is not True
        or item.get("post_map_physical_audit_enabled") is not False
        or item.get("stream_barrier", {}).get("ok") is not True
        for item in parks
    ):
        errors.append("a PARK did not use the audited fast transition path")
    if any(
        item.get("fast_runtime_transition") is not True
        or item.get("stream_barrier", {}).get("ok") is not True
        for item in resumes
    ):
        errors.append("a RESUME did not use the audited fast transition path")
    runtime_read_gates = [
        item
        for item in manifest
        if item.get("event") == "cpu_offload_uvm_hot_cuda_read_gate_begin"
        and str(item.get("gate_stage", "")).startswith(
            ("runtime_remote_", "runtime_prefill_")
        )
    ]
    if runtime_read_gates:
        errors.append("runtime PARK/RESUME still executed CUDA read gates")

    copy_forwards = [
        item
        for item in manifest
        if item.get("event") == "cpu_offload_copy_forward"
        and item.get("graph_static_alias_lifecycle_state") == "sealed"
        and int(item.get("graph_static_alias_actual_h2d_bytes", 0)) > 0
    ]
    remote_forwards = [
        item
        for item in manifest
        if item.get("event") == "cpu_offload_copy_forward"
        and item.get("graph_static_alias_lifecycle_state") == "remote_parked"
        and int(item.get("graph_static_alias_actual_h2d_bytes", -1)) == 0
    ]
    if not copy_forwards:
        errors.append("no real H2D copy forward was recorded")
    if not remote_forwards:
        errors.append("no zero-H2D remote forward was recorded")

    if result.get("configuration", {}).get("compilation_config") != {
        "mode": "NONE",
        "cudagraph_mode": "NONE",
    }:
        errors.append("run was not fully eager")
    fault_tokens = ("Xid ", "FAULT_PDE", "FAULT_PTE", "BUG:", "Oops:")
    if any(token in kernel_log for token in fault_tokens):
        errors.append("kernel log contains a GPU/kernel fault")
    if load(args.run_root / "gpu.after.json").get("gpu_clean") is not True:
        errors.append("GPU was not clean after the model process")
    if load(args.run_root / "module.after.json").get("gpu_clean") is not True:
        errors.append("stock-module restore was not clean")

    hashes = output_hashes(result)
    reference_hashes = output_hashes(reference)
    verification = {
        "schema": "adaptmoe_token_threshold_e2e_verification_v2",
        "status": "error" if errors else "ok",
        "errors": errors,
        "fixed_tier": {
            "name": plan.get("tier"),
            "selected_expert_count": plan.get("selected_expert_count"),
            "selected_bytes": selected_bytes,
            "selected_gib": selected_bytes / (1 << 30),
        },
        "scheduler": {
            "token_counts": token_counts,
            "park_token_threshold": args.threshold,
            "resume_token_threshold": args.resume_threshold,
            "mixed_step_count": len(mixed),
            "mixed_step_tokens": [
                int(step["total_num_scheduled_tokens"]) for step in mixed
            ],
        },
        "topology": {
            "park_count": len(parks),
            "resume_count": len(resumes),
            "poison_count": len(poisoned),
            "copy_forward_record_count": len(copy_forwards),
            "remote_forward_record_count": len(remote_forwards),
            "runtime_read_gate_count": len(runtime_read_gates),
        },
        "outputs": {
            "successful_requests": result.get("summary", {}).get(
                "successful_requests"
            ),
            "hashes_equal_reference": hashes == reference_hashes,
            "matching_reference_hashes": sum(
                hashes.get(key) == value for key, value in reference_hashes.items()
            ),
        },
        "performance": result.get("summary", {}),
        "transition_performance_ms": {
            "park_wall": [float(item["wall_ns"]) / 1e6 for item in parks],
            "park_stream_dependency": [
                float(item.get("sync_ns", 0)) / 1e6 for item in parks
            ],
            "park_ioctl": [
                float(item.get("park", {}).get("wall_ns", 0)) / 1e6
                for item in parks
            ],
            "park_post_map_control": [
                float(item.get("physical_audit_ns", 0)) / 1e6
                for item in parks
            ],
            "resume_wall": [
                float(item["wall_ns"]) / 1e6 for item in resumes
            ],
            "resume_stream_dependency": [
                float(item.get("stream_dependency_ns", 0)) / 1e6
                for item in resumes
            ],
            "resume_ioctl": [
                float(item.get("resume", {}).get("wall_ns", 0)) / 1e6
                for item in resumes
            ],
        },
    }
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
