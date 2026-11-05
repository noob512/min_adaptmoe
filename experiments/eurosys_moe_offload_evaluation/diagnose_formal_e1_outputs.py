#!/usr/bin/env python3
"""Diagnose E1 generation-output stability without rerunning the experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


SYSTEMS = ("ours", "ktransformers", "moe_infinity")
REPEATS = 5
PROMPTS = 10


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def latest_verified_case(root: Path, system: str, repeat: int, protocol: str) -> Path:
    repeat_root = (
        root / system / "deepseek_v2_lite_bf16" / "maximum_resident"
        / f"repeat_{repeat:02d}"
    )
    attempts = sorted(
        (path for path in repeat_root.glob("attempt_*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
        reverse=True,
    )
    for attempt in attempts:
        verification = attempt / "verification.json"
        if not verification.is_file():
            continue
        value = load_json(verification)
        if value.get("status") == "ok" and value.get("protocol_id") == protocol:
            return attempt
    raise FileNotFoundError(f"no verified attempt for {system}/repeat_{repeat:02d}")


def signature(hashes: list[str]) -> str:
    encoded = json.dumps(hashes, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("e1_root", type=Path)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    by_system: dict[str, list[list[str]]] = {system: [] for system in SYSTEMS}
    prompt_inputs: list[list[tuple[str | None, str | None]]] = [
        [] for _ in range(PROMPTS)
    ]
    cases: list[dict[str, Any]] = []
    execution_failures: list[str] = []
    for system in SYSTEMS:
        for repeat in range(1, REPEATS + 1):
            case = latest_verified_case(args.e1_root, system, repeat, args.protocol_id)
            client_path = case / "short_dialogue/client.json"
            client = load_json(client_path)
            phases = client.get("phases", [])
            phase = phases[0] if len(phases) == 1 else {}
            requests = phase.get("requests", [])
            hashes = [request.get("output_token_ids_sha256") for request in requests]
            valid_hashes = (
                len(hashes) == PROMPTS
                and all(isinstance(value, str) for value in hashes)
            )
            rate_ok = (
                phase.get("concurrency") == 1
                and phase.get("request_count") == PROMPTS
                and phase.get("successful_requests") == PROMPTS
                and phase.get("failed_requests") == 0
                and phase.get("request_rate_limit_compliant") is True
                and client.get("config", {}).get(
                    "minimum_request_start_interval_seconds"
                ) == 6.0
            )
            if not valid_hashes:
                execution_failures.append(f"invalid output hashes: {system}/{repeat}")
            if not rate_ok:
                execution_failures.append(f"request policy violation: {system}/{repeat}")
            if valid_hashes:
                by_system[system].append(hashes)
                for index, request in enumerate(requests):
                    prompt_inputs[index].append(
                        (
                            request.get("prompt_token_ids_sha256"),
                            request.get("serialized_body_sha256"),
                        )
                    )
            cases.append(
                {
                    "system": system,
                    "repeat": repeat,
                    "case_root": str(case.resolve()),
                    "client_path": str(client_path.resolve()),
                    "signature_sha256": signature(hashes) if valid_hashes else None,
                    "request_policy_compliant": rate_ok,
                    "observed_minimum_request_start_interval_seconds": phase.get(
                        "observed_minimum_request_start_interval_seconds"
                    ),
                }
            )

    system_stability: dict[str, Any] = {}
    modal_hashes: dict[str, list[str | None]] = {}
    for system, repeats in by_system.items():
        per_prompt_unique = [
            len({values[index] for values in repeats})
            for index in range(PROMPTS)
        ] if len(repeats) == REPEATS else []
        signatures = [signature(values) for values in repeats]
        modal: list[str | None] = []
        modal_support: list[int] = []
        for index in range(PROMPTS):
            counts = Counter(values[index] for values in repeats)
            if not counts:
                modal.append(None)
                modal_support.append(0)
                continue
            value, support = counts.most_common(1)[0]
            modal.append(value)
            modal_support.append(support)
        modal_hashes[system] = modal
        system_stability[system] = {
            "verified_process_blocks": len(repeats),
            "unique_10_prompt_signatures": len(set(signatures)),
            "stable_across_all_five_blocks": len(set(signatures)) == 1,
            "stable_prompt_count": sum(value == 1 for value in per_prompt_unique),
            "per_prompt_unique_hash_counts": per_prompt_unique,
            "modal_hash_support_out_of_five": modal_support,
            "signature_sha256": signatures[0] if len(set(signatures)) == 1 else None,
        }

    prompt_comparison: list[dict[str, Any]] = []
    unanimous_all_blocks = 0
    frozen_inputs_identical = True
    for index in range(PROMPTS):
        all_values = [
            values[index]
            for system in SYSTEMS
            for values in by_system[system]
        ]
        unanimous = len(all_values) == len(SYSTEMS) * REPEATS and len(set(all_values)) == 1
        input_identical = (
            len(prompt_inputs[index]) == len(SYSTEMS) * REPEATS
            and len(set(prompt_inputs[index])) == 1
            and all(value is not None for value in prompt_inputs[index][0])
        )
        frozen_inputs_identical &= input_identical
        unanimous_all_blocks += int(unanimous)
        prompt_comparison.append(
            {
                "fixture_index": index,
                "frozen_prompt_and_request_body_identical": input_identical,
                "unanimous_across_all_15_blocks": unanimous,
                "unique_hashes_across_all_15_blocks": len(set(all_values)),
                "modal_hash_equal_ours_vs_ktransformers": (
                    modal_hashes["ours"][index]
                    == modal_hashes["ktransformers"][index]
                ),
                "modal_hash_equal_ours_vs_moe_infinity": (
                    modal_hashes["ours"][index]
                    == modal_hashes["moe_infinity"][index]
                ),
                "modal_hash_equal_ktransformers_vs_moe_infinity": (
                    modal_hashes["ktransformers"][index]
                    == modal_hashes["moe_infinity"][index]
                ),
            }
        )

    e1_verification_path = args.e1_root / "verification.json"
    e1_verification = load_json(e1_verification_path)
    execution_complete = (
        not execution_failures
        and e1_verification.get("checks", {}).get("observed_verified_process_blocks")
        == len(SYSTEMS) * REPEATS
        and e1_verification.get("checks", {}).get("balanced_cyclic_order") is True
    )
    exact_gate = unanimous_all_blocks == PROMPTS
    result = {
        "schema_version": 1,
        "experiment": "E1_output_divergence_diagnostic",
        "protocol_id": args.protocol_id,
        "status": "diagnosed" if execution_complete else "error",
        "execution_complete": execution_complete,
        "request_policy": {
            "concurrency": 1,
            "measurement_requests_per_server_block": 10,
            "minimum_request_start_interval_seconds": 6.0,
            "maximum_request_starts_per_minute": 10,
            "all_blocks_compliant": all(
                case["request_policy_compliant"] for case in cases
            ),
        },
        "system_stability": system_stability,
        "cross_system": {
            "frozen_prompt_and_request_body_identical_all_15_blocks": (
                frozen_inputs_identical
            ),
            "unanimous_prompts_across_all_15_blocks": unanimous_all_blocks,
            "total_prompts": PROMPTS,
            "exact_generation_gate_passed": exact_gate,
            "prompt_comparison": prompt_comparison,
        },
        "publication_disposition": {
            "descriptive_anchor_metrics_available": execution_complete,
            "cross_system_speedup_eligible": execution_complete and exact_gate,
            "required_next_gate": (
                None if exact_gate else
                "teacher-forced logits relative-L1, top-1 agreement, and perplexity"
            ),
            "reason": (
                None if exact_gate else
                "Generation hashes differ across frameworks and MoE-Infinity also "
                "varies across independent process blocks."
            ),
        },
        "e1_verification": {
            "path": str(e1_verification_path.resolve()),
            "status": e1_verification.get("status"),
            "performance_summaries_retained": True,
        },
        "cases": cases,
        "execution_failures": execution_failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if execution_complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
