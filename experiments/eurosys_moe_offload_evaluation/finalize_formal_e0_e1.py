#!/usr/bin/env python3
"""Finalize the immutable E0/E1 v2 artifact bundle and top-level summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
EXTERNAL_INPUTS = (
    SCRIPT_DIR / "protocol_template.json",
    SCRIPT_DIR / "configs/e0_validation_c1_v2.json",
    SCRIPT_DIR / "configs/e1_short_c1.json",
    SCRIPT_DIR / "streaming_client.py",
    SCRIPT_DIR / "formal_openai_block.py",
    SCRIPT_DIR / "verify_formal_e0.py",
    SCRIPT_DIR / "verify_formal_e1.py",
    SCRIPT_DIR / "verify_ktransformers_e0.py",
    SCRIPT_DIR / "diagnose_formal_e1_outputs.py",
    SCRIPT_DIR / "finalize_formal_e0_e1.py",
    SCRIPT_DIR / "run_formal_e1.sh",
    SCRIPT_DIR / "run_ktransformers_e0.sh",
    SCRIPT_DIR / "launch_moe_infinity_server.py",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite finalizer output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def file_record(path: Path, base: Path | None = None) -> dict[str, Any]:
    name = str(path.resolve()) if base is None else str(path.relative_to(base))
    return {
        "path": name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def metric(summary: dict[str, Any], name: str) -> dict[str, Any]:
    value = summary[name]
    return {
        "mean": value["mean"],
        "ci95_low": value.get("ci95_low"),
        "ci95_high": value.get("ci95_high"),
        "n_server_blocks": value["n"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    protocol = load_json(root / "protocol.json")
    e0_path = root / "cases/E0/verification.json"
    e1_path = root / "cases/E1/verification.json"
    diagnostic_path = root / "cases/E1/output_divergence_diagnostic.json"
    e0 = load_json(e0_path)
    e1 = load_json(e1_path)
    diagnostic = load_json(diagnostic_path)

    protocol_id = protocol.get("protocol_id")
    if protocol_id != "eurosys-pa-v2-20260824":
        raise SystemExit(f"unexpected protocol: {protocol_id}")
    if e0.get("status") != "ok":
        raise SystemExit("E0 qualification did not pass")
    if not diagnostic.get("execution_complete"):
        raise SystemExit("E1 execution did not complete")
    if not diagnostic.get("request_policy", {}).get("all_blocks_compliant"):
        raise SystemExit("E1 request-rate policy did not pass")
    if e1.get("checks", {}).get("observed_verified_process_blocks") != 15:
        raise SystemExit("E1 does not contain 15 verified process blocks")

    now = datetime.now(timezone.utc).isoformat()
    manifest_path = root / "provenance/e0_e1_completion_manifest.json"
    summary_path = root / "summary.json"
    run_status_path = root / "run_status.json"
    excluded = {manifest_path, summary_path, run_status_path}
    artifact_files = [
        path for path in sorted(root.rglob("*"))
        if path.is_file() and path not in excluded
    ]
    missing_inputs = [str(path) for path in EXTERNAL_INPUTS if not path.is_file()]
    if missing_inputs:
        raise SystemExit(f"missing registered inputs: {missing_inputs}")
    artifact_records = [file_record(path, root) for path in artifact_files]
    external_records = [file_record(path) for path in EXTERNAL_INPUTS]
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "registered_at": now,
        "registration": (
            "post-run v2 provenance attestation; no experimental treatment, "
            "workload, ordering, request-rate, or measurement value was changed"
        ),
        "artifact_root": str(root),
        "artifact_file_count": len(artifact_records),
        "artifact_total_bytes": sum(record["bytes"] for record in artifact_records),
        "artifact_files": artifact_records,
        "external_input_files": external_records,
        "immutable_result_files": {
            "e0_verification": file_record(e0_path, root),
            "e1_verification": file_record(e1_path, root),
            "e1_output_diagnostic": file_record(diagnostic_path, root),
        },
        "reporting_amendment": {
            "reason": (
                "The frozen protocol required v2 provenance and a strict numerical "
                "gate. The final verifiers separate a passed compatibility check "
                "from the unresolved broad-generation numerical gate."
            ),
            "changes_measurements": False,
            "changes_treatments": False,
            "changes_request_rate": False,
        },
    }
    atomic_json(manifest_path, manifest)

    systems: dict[str, Any] = {}
    for system, values in e1["system_summaries"].items():
        systems[system] = {
            "ttft_ms": metric(values, "ttft_ms"),
            "tpot_ms": metric(values, "tpot_ms"),
            "e2e_ms": metric(values, "e2e_ms"),
            "output_tokens_per_second": metric(
                values, "output_tokens_per_second"
            ),
            "peak_hbm_mib": metric(values, "peak_hbm_mib"),
            "ready_hbm_mib": metric(values, "ready_hbm_mib"),
        }
    exact_gate = diagnostic["cross_system"]["exact_generation_gate_passed"]
    summary = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "status": "complete" if exact_gate else "complete_with_correctness_hold",
        "completed_at": now,
        "request_policy": diagnostic["request_policy"],
        "E0": {
            "qualification_status": "passed",
            "shared_validation_output_token_ids_sha256": (
                e0["fresh_e0_validation"]["shared_output_token_ids_sha256"]
            ),
            "capability_matrix": e0["capability_matrix"],
            "publication_correctness_gate": e0["publication_correctness_gate"],
            "verification": str(e0_path),
        },
        "E1": {
            "execution_status": "completed",
            "verified_independent_server_blocks": 15,
            "blocks_per_system": 5,
            "balanced_cyclic_order": e1["checks"]["balanced_cyclic_order"],
            "system_summaries": systems,
            "descriptive_anchor_metrics_available": True,
            "cross_system_speedup_eligible": exact_gate,
            "output_stability": diagnostic["system_stability"],
            "unanimous_evaluation_prompts": diagnostic["cross_system"][
                "unanimous_prompts_across_all_15_blocks"
            ],
            "total_evaluation_prompts": diagnostic["cross_system"]["total_prompts"],
            "verification": str(e1_path),
            "output_diagnostic": str(diagnostic_path),
        },
        "conclusion": (
            "E0 base compatibility passed and E1 execution completed. E1 metrics "
            "remain descriptive only because the cross-framework/repeat generation "
            "gate failed; no cross-system speedup is publication-eligible until the "
            "teacher-forced numerical gate passes."
        ),
        "provenance_manifest": str(manifest_path),
    }
    atomic_json(summary_path, summary)

    old_status = load_json(run_status_path)
    run_status = {
        **old_status,
        "status": summary["status"],
        "verification_status": (
            "qualified" if exact_gate else "compatibility_passed_numerical_gate_pending"
        ),
        "completed_at": now,
        "E0": "passed",
        "E1_execution": "completed",
        "E1_publication_gate": "passed" if exact_gate else "held",
        "request_rate_policy": "passed",
        "summary": str(summary_path),
        "provenance_manifest": str(manifest_path),
    }
    atomic_json(run_status_path, run_status, replace=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
