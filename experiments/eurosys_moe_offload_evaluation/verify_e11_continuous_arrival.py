#!/usr/bin/env python3
"""Verify the single-round E11 continuous-arrival qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    errors: list[str] = []
    arms: dict[str, dict[str, Any]] = {}
    for mode in (
        "remote_phase_epoch",
        "remote_continuous",
        "hybrid_fail_closed",
    ):
        arm_root = args.run_root / mode
        result_path = arm_root / "result.json"
        if not result_path.is_file():
            errors.append(f"{mode}: result.json is absent")
            continue
        result = load(result_path)
        arms[mode] = result
        kernel = (arm_root / "kernel.since.log").read_text(
            encoding="utf-8", errors="replace"
        ) if (arm_root / "kernel.since.log").is_file() else ""
        if any(token in kernel for token in ("Xid ", "FAULT_PDE", "FAULT_PTE", "BUG:", "Oops:")):
            errors.append(f"{mode}: kernel log contains a GPU/kernel fault")
        gpu_after = load(arm_root / "gpu.after.json")
        if gpu_after.get("gpu_clean") is not True:
            errors.append(f"{mode}: GPU did not clean after the arm")

    for mode in ("remote_phase_epoch", "remote_continuous"):
        result = arms.get(mode, {})
        if result.get("status") != "ok":
            errors.append(f"{mode}: status is not ok")
            continue
        summary = result.get("summary", {})
        if summary.get("successful_requests") != 8:
            errors.append(f"{mode}: not all eight requests completed")
        mixed = int(summary.get("mixed_scheduler_steps", -1))
        if mode == "remote_phase_epoch" and mixed != 0:
            errors.append(
                "remote_phase_epoch: mixed scheduler work was observed"
            )
        if mode == "remote_continuous" and mixed < 1:
            errors.append("remote_continuous: no mixed scheduler work was observed")

    fail_closed = arms.get("hybrid_fail_closed", {})
    fail_error = str(fail_closed.get("error", ""))
    if fail_closed.get("status") != "error":
        errors.append("hybrid_fail_closed: expected an explicit error")
    accepted_fail_closed_errors = (
        "mixed prefill/decode batches are unsupported",
        "requires an unmixed request",
    )
    if not any(text in fail_error for text in accepted_fail_closed_errors):
        errors.append("hybrid_fail_closed: expected mixed-step rejection is absent")

    if all(
        mode in arms
        for mode in ("remote_phase_epoch", "remote_continuous")
    ):
        left = {
            record["logical_request"]: record["output_token_ids_sha256"]
            for record in arms["remote_phase_epoch"].get("requests", [])
        }
        right = {
            record["logical_request"]: record["output_token_ids_sha256"]
            for record in arms["remote_continuous"].get("requests", [])
        }
        if left != right:
            errors.append(
                "same-topology scheduler arms produced different token IDs"
            )

    module_after = load(args.run_root / "module.after.json")
    if module_after.get("gpu_clean") is not True:
        errors.append("final stock-module snapshot is not clean")
    verification = {
        "schema": "eurosys_e11_continuous_arrival_verification_v1",
        "status": "ok" if not errors else "error",
        "errors": errors,
        "correctness_comparison": {
            "topology": "remote_only_eager",
            "varying_factor": "phase_epoch_scheduler_only",
        },
        "arms": {
            mode: {
                "status": result.get("status"),
                "summary": result.get("summary"),
                "error": result.get("error"),
            }
            for mode, result in arms.items()
        },
    }
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
