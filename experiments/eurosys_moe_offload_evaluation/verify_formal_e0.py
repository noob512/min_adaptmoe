#!/usr/bin/env python3
"""Register and verify E0 compatibility/correctness under the v2 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MECHANISMS = {
    "dual_graph_park_resume": (
        REPO_ROOT / "experiments/eurosys_moe_offload_evaluation/artifacts/"
        "eurosys-pa-20260823T132800Z-e0-qualification/cases/ours/qwen3_fp8/"
        "dual_topology/verification.json"
    ),
    "stable_va_large_kv": (
        REPO_ROOT / "experiments/vllm_uvm_large_kv_multilevel/artifacts/"
        "20260824_large_kv_multilevel_qwen_base70_full_v1/verification.json"
    ),
    "physical_residency_cleanup": (
        REPO_ROOT / "experiments/eurosys_moe_offload_evaluation/artifacts/"
        "eurosys-pa-20260823T132800Z-e0-qualification/cases/ours/qwen3_fp8/"
        "physical_kv_resize/verification.json"
    ),
    "fixed_graph_classes": (
        REPO_ROOT / "experiments/hybrid_offload_2_4_6_8_compare/artifacts/"
        "graph_classes_20260823T142842Z/result.json"
    ),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite verifier output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def command_has(command_path: Path, option: str, expected: str | None = None) -> bool:
    command = shlex.split(command_path.read_text(encoding="utf-8"))
    if option not in command:
        return False
    if expected is None:
        return True
    index = command.index(option)
    return index + 1 < len(command) and command[index + 1] == expected


def latest_verified_case(e1_root: Path, system: str, repeat: int) -> Path | None:
    repeat_root = (
        e1_root / system / "deepseek_v2_lite_bf16" / "maximum_resident"
        / f"repeat_{repeat:02d}"
    )
    attempts = sorted(
        (path for path in repeat_root.glob("attempt_*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
        reverse=True,
    )
    for attempt in attempts:
        verification = attempt / "verification.json"
        if verification.is_file() and load_json(verification).get("status") == "ok":
            return attempt
    return None


def verified_openai_validation(
    case_root: Path,
    system: str,
    protocol_id: str,
    failures: list[str],
) -> dict[str, Any]:
    """Read the single frozen E0 validation request from an OpenAI case."""
    verification_path = case_root / "verification.json"
    client_path = case_root / "short_dialogue/client.json"
    record: dict[str, Any] = {
        "case_root": str(case_root.resolve()),
        "verification_path": str(verification_path.resolve()),
        "client_path": str(client_path.resolve()),
    }
    if not verification_path.is_file() or not client_path.is_file():
        failures.append(f"missing fresh E0 artifacts: {system}")
        record["ok"] = False
        return record

    verification = load_json(verification_path)
    client = load_json(client_path)
    phases = client.get("phases", [])
    phase = phases[0] if isinstance(phases, list) and len(phases) == 1 else {}
    requests = phase.get("requests", [])
    request = requests[0] if isinstance(requests, list) and len(requests) == 1 else {}
    config = client.get("config", {})
    checks = {
        "verification_ok": verification.get("status") == "ok",
        "protocol_matches": verification.get("protocol_id") == protocol_id,
        "one_frozen_request": (
            phase.get("request_count") == 1
            and phase.get("successful_requests") == 1
            and phase.get("failed_requests") == 0
            and len(requests) == 1
            and request.get("fixture_index") == 0
        ),
        "greedy_32_tokens": (
            request.get("success") is True
            and request.get("http_status") == 200
            and request.get("completion_tokens") == 32
            and isinstance(request.get("output_token_ids_sha256"), str)
        ),
        "concurrency_one": phase.get("concurrency") == 1,
        "rate_limiter_configured": (
            config.get("measurement_requests") == 1
            and float(config.get("minimum_request_start_interval_seconds", -1))
            == 6.0
        ),
        "gpu_cleanup": verification.get("checks", {}).get("gpu_cleanup") is True,
    }
    if system == "moe_infinity":
        counters = verification.get("checks", {}).get(
            "moe_infinity_admin_stats", {}
        )
        checks.update(
            {
                "admin_counters_available": counters.get("available") is True,
                "admin_counters_clean": counters.get("clean") is True,
                "request_counter_exact": (
                    counters.get("observed_completed_requests")
                    == counters.get("expected_completed_requests")
                ),
                "token_counter_exact": (
                    counters.get("observed_generated_tokens")
                    == counters.get("expected_generated_tokens")
                ),
                "no_pending_or_cancelled_requests": (
                    counters.get("pending_requests") == 0
                    and counters.get("cancelled_requests") == 0
                ),
            }
        )
    ok = all(checks.values())
    if not ok:
        failures.append(f"fresh E0 OpenAI validation failed: {system}")
    record.update(
        {
            "verification_sha256": sha256_file(verification_path),
            "client_sha256": sha256_file(client_path),
            "output_token_ids_sha256": request.get("output_token_ids_sha256"),
            "checks": checks,
            "ok": ok,
        }
    )
    return record


def verified_ktransformers_validation(
    case_root: Path,
    protocol_id: str,
    failures: list[str],
) -> dict[str, Any]:
    """Read the frozen KTransformers eager/Graph E0 pair."""
    verification_path = case_root / "verification.json"
    record: dict[str, Any] = {
        "case_root": str(case_root.resolve()),
        "verification_path": str(verification_path.resolve()),
    }
    if not verification_path.is_file():
        failures.append("missing fresh KTransformers E0 verification")
        record["ok"] = False
        return record
    verification = load_json(verification_path)
    hashes: dict[str, str | None] = {}
    client_checks: dict[str, bool] = {}
    for mode in ("eager", "graph"):
        client_path = case_root / mode / "client.json"
        if not client_path.is_file():
            client_checks[f"{mode}_client_present"] = False
            hashes[mode] = None
            continue
        client = load_json(client_path)
        phases = client.get("phases", [])
        phase = phases[0] if isinstance(phases, list) and len(phases) == 1 else {}
        requests = phase.get("requests", [])
        first = requests[0] if isinstance(requests, list) and requests else {}
        hashes[mode] = first.get("output_token_ids_sha256")
        client_checks[f"{mode}_ten_requests"] = (
            phase.get("request_count") == 10
            and phase.get("successful_requests") == 10
            and phase.get("failed_requests") == 0
            and len(requests) == 10
        )
        client_checks[f"{mode}_concurrency_one"] = phase.get("concurrency") == 1
        client_checks[f"{mode}_rate_limit"] = (
            client.get("config", {}).get("minimum_request_start_interval_seconds")
            == 6.0
            and phase.get("request_rate_limit_compliant") is True
        )
        client_checks[f"{mode}_fixture_zero_present"] = (
            first.get("fixture_index") == 0
            and first.get("completion_tokens") == 32
            and isinstance(hashes[mode], str)
        )
    checks = {
        "verification_ok": verification.get("status") == "ok",
        "protocol_matches": verification.get("protocol_id") == protocol_id,
        "eager_graph_output_equal": (
            verification.get("checks", {}).get("eager_graph_output_hash_equal")
            is True
            and hashes.get("eager") == hashes.get("graph")
            and hashes.get("eager") is not None
        ),
        **client_checks,
    }
    ok = all(checks.values())
    if not ok:
        failures.append("fresh v2 KTransformers eager/Graph gate failed")
    record.update(
        {
            "verification_sha256": sha256_file(verification_path),
            "output_token_ids_sha256": hashes.get("graph"),
            "checks": checks,
            "ok": ok,
        }
    )
    return record


def verify_mechanisms(failures: list[str]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, path in DEFAULT_MECHANISMS.items():
        if not path.is_file():
            failures.append(f"missing re-attested mechanism artifact: {path}")
            continue
        value = load_json(path)
        checks: dict[str, bool]
        if name == "dual_graph_park_resume":
            checks = {
                "status": value.get("status") == "ok",
                "full_separate_graph": value.get("run_config", {}).get("cudagraph_mode") == "FULL_SEPARATE",
                "park_resume": value.get("run_config", {}).get("park_resume") is True,
                "pte_audit": all(sample.get("audit_trace") is True for sample in value.get("samples", [])),
                "outputs_present": all(bool(sample.get("output_hash")) for sample in value.get("samples", [])),
            }
        elif name == "stable_va_large_kv":
            checks = {
                "status": value.get("status") == "ok",
                "outputs_match": value.get("outputs_match") is True,
                "parameter_va_stable": value.get("parameter_va_stable_within_each_process") is True,
                "source_va_stable": value.get("source_va_stable_within_each_process") is True,
                "kv_va_stable": value.get("kv_va_stable_within_each_process") is True,
                "prefill_decode_graph": value.get("prefill_decode_full_graph_replayed") is True,
                "park_resume": value.get("prefill_resume_decode_park_validated") is True,
                "no_recapture": value.get("cuda_graph_recaptured") is False,
                "no_faults": value.get("faults_detected") is False,
                "cleanup": value.get("resources_released") is True,
            }
        elif name == "physical_residency_cleanup":
            checks = {
                "status": value.get("status") == "ok",
                "residency": value.get("kv_va_stable_within_each_process") is True,
                "outputs_match": value.get("outputs_match_direct_baselines") is True,
                "no_faults": value.get("faults_detected") is False,
                "cleanup": value.get("resources_released") is True,
                "no_recapture": value.get("cuda_graph_recaptured_during_resize") is False,
            }
        else:
            hashes = {
                record.get("output_sha256") for record in value.get("records", [])
            }
            stable = value.get("stable_vas", {})
            checks = {
                "status": value.get("status") == "ok",
                "all_classes": value.get("sequence") == ["c2", "c4", "c6", "c8", "c2"],
                "no_recapture": value.get("capture_count_after_load") == value.get("capture_count_final"),
                "stable_parameter_va": bool(stable.get("parameter_vas")),
                "stable_source_va": bool(stable.get("source_vas")),
                "stable_upload_va": bool(stable.get("upload_vas")),
                "outputs_match": len(hashes) == 1 and None not in hashes,
                "cleanup": value.get("cleanup_errors") == [],
            }
        if not all(checks.values()):
            failures.append(f"re-attested mechanism checks failed: {name}")
        records[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "provenance_disposition": (
                "pre-v2 qualification artifact re-attested by hash under v2; "
                "not counted as a new performance observation"
            ),
            "checks": checks,
            "ok": all(checks.values()),
        }
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--e1-root", required=True, type=Path)
    parser.add_argument("--ours-e0-root", required=True, type=Path)
    parser.add_argument("--moe-infinity-e0-root", required=True, type=Path)
    parser.add_argument("--kt-e0-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    protocol_path = args.run_root / "protocol.json"
    if not protocol_path.is_file():
        raise SystemExit(f"missing protocol: {protocol_path}")
    protocol = load_json(protocol_path)
    protocol_id = protocol.get("protocol_id")
    provenance_paths = [
        args.run_root / "provenance/git_commits.json",
        args.run_root / "provenance/model_manifest.json",
        args.run_root / "provenance/environment.txt",
        args.run_root / "provenance/fixture_manifest.json",
    ]
    provenance_ok = all(path.is_file() for path in provenance_paths)
    if not provenance_ok:
        failures.append("v2 provenance bundle is incomplete")

    model_manifest = (
        load_json(args.run_root / "provenance/model_manifest.json")
        if (args.run_root / "provenance/model_manifest.json").is_file()
        else {}
    )
    model_config_path = Path(
        model_manifest.get("deepseek_v2_lite_bf16", {})
        .get("files", {})
        .get("config.json", {})
        .get("path", "")
    )
    checkpoint_declares_bf16 = (
        model_config_path.is_file()
        and load_json(model_config_path).get("torch_dtype") == "bfloat16"
    )

    fixture_manifest_path = (
        SCRIPT_DIR / "inputs/generated" / str(protocol_id)
        / "deepseek_v2_lite_bf16/fixture_manifest.json"
    )
    if not fixture_manifest_path.is_file():
        failures.append("missing v2 DeepSeek fixture manifest")
        fixture_ok = False
    else:
        fixture = load_json(fixture_manifest_path)
        fixture_ok = (
            fixture.get("status") == "ok"
            and fixture.get("protocol_id") == protocol_id
            and fixture.get("all_fixture_source_sets_disjoint") is True
        )
        if not fixture_ok:
            failures.append("v2 frozen fixture manifest failed")

    ours_e0 = verified_openai_validation(
        args.ours_e0_root, "ours", str(protocol_id), failures
    )
    moe_infinity_e0 = verified_openai_validation(
        args.moe_infinity_e0_root, "moe_infinity", str(protocol_id), failures
    )
    kt_e0 = verified_ktransformers_validation(
        args.kt_e0_root, str(protocol_id), failures
    )
    validation_hashes = {
        ours_e0.get("output_token_ids_sha256"),
        moe_infinity_e0.get("output_token_ids_sha256"),
        kt_e0.get("output_token_ids_sha256"),
    }
    fresh_validation_exact = len(validation_hashes) == 1 and None not in validation_hashes
    if not fresh_validation_exact:
        failures.append("fresh v2 three-system frozen validation output differs")

    e1_verification_path = args.e1_root / "verification.json"
    e1 = load_json(e1_verification_path)
    e1_execution_complete = (
        e1.get("protocol_id") == protocol_id
        and e1.get("checks", {}).get("observed_verified_process_blocks") == 15
        and e1.get("checks", {}).get("balanced_cyclic_order") is True
    )
    if not e1_execution_complete:
        failures.append("fresh E1 execution is incomplete")
    broad_generation_exact = e1.get("checks", {}).get(
        "cross_framework_exact_token_match_for_all_10_prompts"
    ) is True

    fresh_cases: dict[str, list[str]] = {system: [] for system in ("ours", "ktransformers", "moe_infinity")}
    moe_counters_ok = True
    model_precision_ok = checkpoint_declares_bf16
    cleanup_ok = True
    for system in fresh_cases:
        for repeat in range(1, 6):
            case = latest_verified_case(args.e1_root, system, repeat)
            if case is None:
                failures.append(f"missing fresh E1 case for E0: {system}/{repeat}")
                continue
            fresh_cases[system].append(str(case.resolve()))
            command = case / "command.txt"
            verification = load_json(case / "verification.json")
            cleanup_ok &= verification.get("checks", {}).get("gpu_cleanup") is True
            if system == "ours":
                model_precision_ok &= command_has(command, "--dtype", "bfloat16")
            elif system == "ktransformers":
                model_precision_ok &= command_has(command, "--kt-method", "BF16")
            else:
                stats = verification.get("checks", {}).get("moe_infinity_admin_stats", {})
                moe_counters_ok &= stats.get("available") is True and stats.get("clean") is True
    if not model_precision_ok:
        failures.append("DeepSeek BF16 command certificate failed")
    if not moe_counters_ok:
        failures.append("MoE-Infinity counter/cleanup certificate failed")
    if not cleanup_ok:
        failures.append("one or more fresh server blocks did not clean GPU state")

    mechanism_records = verify_mechanisms(failures)
    capability_matrix = {
        "ours": {
            "model_precision": "DeepSeek-V2-Lite-Chat BF16: verified",
            "frozen_input_and_exact_output": "one shared E0 validation prompt: verified; E1 broad-generation gate: pending",
            "eager_path": "verified in fresh E1",
            "cuda_graph_path": "DeepSeek maximum-resident: N/A (HBM headroom); Qwen FULL_SEPARATE re-attested",
            "placement": "maximum-resident, no dynamic expert switch",
            "va_pte_residency_fault_cleanup": "verified by hashed Qwen UVM mechanism qualifications",
        },
        "ktransformers": {
            "model_precision": "DeepSeek-V2-Lite-Chat BF16: verified",
            "frozen_input_and_exact_output": "one shared E0 validation prompt plus eager/Graph 10-prompt identity: verified; E1 cross-framework gate: pending",
            "eager_path": "verified in fresh E0",
            "cuda_graph_path": "verified in fresh E0 and E1",
            "placement": "fresh E0 frequency 54/layer; fresh E1 uniform 64/layer",
            "va_pte_residency_fault_cleanup": "N/A (not the UVM implementation under test)",
        },
        "moe_infinity": {
            "model_precision": "DeepSeek-V2-Lite-Chat checkpoint remote-code BF16: verified",
            "frozen_input_and_exact_output": "one shared E0 validation prompt: verified; E1 repeat/cross-framework gate: pending",
            "eager_path": "public-demand path verified in fresh E1",
            "cuda_graph_path": "N/A (public server path exposes no CUDA Graph mode)",
            "placement": "public-demand cache with frozen device/kv ratios",
            "counters": "admin request/token/KV cleanup counters verified; speculative counters N/A",
            "va_pte_residency_fault_cleanup": "N/A (not the UVM implementation under test)",
        },
    }
    result = {
        "schema_version": 2,
        "status": "ok" if not failures else "error",
        "experiment": "E0_compatibility_correctness",
        "scope": "qualification_only_not_formal_performance",
        "protocol_id": protocol_id,
        "checks": {
            "v2_provenance_complete": provenance_ok,
            "v2_fixtures_frozen_and_disjoint": fixture_ok,
            "fresh_deepseek_three_system_validation_exact_output": fresh_validation_exact,
            "fresh_ktransformers_eager_graph": kt_e0.get("ok") is True,
            "e1_execution_complete": e1_execution_complete,
            "broad_e1_generation_exact_output": broad_generation_exact,
            "deepseek_bf16_commands": model_precision_ok,
            "deepseek_checkpoint_declares_bf16": checkpoint_declares_bf16,
            "moe_infinity_admin_counters_clean": moe_counters_ok,
            "fresh_gpu_cleanup": cleanup_ok,
        },
        "fresh_e0_validation": {
            "ours": ours_e0,
            "ktransformers": kt_e0,
            "moe_infinity": moe_infinity_e0,
            "shared_output_token_ids_sha256": (
                next(iter(validation_hashes)) if fresh_validation_exact else None
            ),
        },
        "fresh_e1_cases": fresh_cases,
        "ktransformers_e0": {
            "path": str(args.kt_e0_root.resolve()),
            "verification_sha256": kt_e0.get("verification_sha256"),
        },
        "re_attested_ours_mechanisms": mechanism_records,
        "capability_matrix": capability_matrix,
        "unsupported_policy": "Unsupported capabilities are reported as N/A, never inferred.",
        "publication_correctness_gate": {
            "status": "passed" if broad_generation_exact else "pending",
            "reason": (
                None if broad_generation_exact else
                "E1 generation hashes diverge across frameworks/repeats; run the "
                "pre-registered teacher-forced logits/top-1/perplexity gate before "
                "publishing cross-system speedups."
            ),
            "e1_verification": str(e1_verification_path.resolve()),
        },
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
