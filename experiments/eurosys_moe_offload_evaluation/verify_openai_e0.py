#!/usr/bin/env python3
"""Verify one unified OpenAI/SSE E0 qualification case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def token_hash(token_ids: list[int]) -> str:
    payload = json.dumps(
        token_ids, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite verifier output: {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def measured_request(client: dict[str, Any]) -> dict[str, Any] | None:
    phases = client.get("phases")
    if not isinstance(phases, list) or len(phases) != 1:
        return None
    requests = phases[0].get("requests")
    if not isinstance(requests, list) or len(requests) != 1:
        return None
    request = requests[0]
    return request if isinstance(request, dict) else None


def check_request(
    record: Any,
    *,
    expected_tokens: int,
    label: str,
    failures: list[str],
) -> str | None:
    if not isinstance(record, dict):
        failures.append(f"{label}: record is not an object")
        return None
    token_ids = record.get("output_token_ids")
    if not isinstance(token_ids, list) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in token_ids
    ):
        failures.append(f"{label}: exact output token IDs are absent")
        return None
    observed_hash = token_hash(token_ids)
    stored_hash = record.get("output_token_ids_sha256")
    checks = {
        "success": record.get("success") is True,
        "http_status": record.get("http_status") == 200,
        "completion_tokens": record.get("completion_tokens") == expected_tokens,
        "token_id_count": len(token_ids) == expected_tokens,
        "token_hash": stored_hash == observed_hash,
        "saw_done": record.get("saw_done") is True,
        "connection_reused": record.get("connection_reused") is True,
        "connection_not_created": (
            record.get("connection_created_during_measurement") is False
        ),
        "no_fallback_events": record.get("fallback_token_events") == 0,
        "no_grouped_events": record.get("grouped_token_events") == 0,
    }
    for name, passed in checks.items():
        if not passed:
            failures.append(f"{label}: failed {name}")
    intervals = record.get("itl_ms")
    if (
        not isinstance(intervals, list)
        or len(intervals) != expected_tokens - 1
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or value <= 0
            for value in intervals
        )
    ):
        failures.append(f"{label}: per-token arrival intervals are invalid")
    return observed_hash


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_root", type=Path)
    parser.add_argument(
        "--system",
        required=True,
        choices=("ours", "ktransformers", "moe_infinity"),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--same-framework-reference", type=Path)
    parser.add_argument("--cross-framework-client", type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    checks: dict[str, Any] = {}
    client_path = args.case_root / "client.json"
    log_path = args.case_root / "server.log"
    stats_path = args.case_root / "admin_stats.json"
    for path in (client_path, log_path):
        if not path.is_file():
            failures.append(f"missing artifact: {path}")
    if failures:
        result = {
            "schema_version": 1,
            "status": "error",
            "system": args.system,
            "failures": failures,
        }
        atomic_json(args.output, result)
        return 1

    client = load_json(client_path)
    if client.get("status") != "ok":
        failures.append(f"client status={client.get('status')!r}")
    config = client.get("config", {})
    if config.get("system") != args.system:
        failures.append("client system label differs")
    if (
        config.get("concurrencies") != [1]
        or config.get("output_tokens") != 32
        or config.get("warmup_requests") != 3
        or config.get("warmup_output_tokens") != 8
    ):
        failures.append("client smoke configuration differs")
    fixture = client.get("fixture", {}).get("metadata", {})
    if (
        fixture.get("fixture_role") != "validation"
        or fixture.get("workload") != "short_dialogue"
        or fixture.get("prompt_tokens") != 128
        or fixture.get("output_tokens") != 32
        or fixture.get("selection_frozen_before_measurement") is not True
    ):
        failures.append("frozen validation fixture metadata differs")

    warmups = client.get("warmups")
    warmup_hashes: list[str | None] = []
    if not isinstance(warmups, list) or len(warmups) != 3:
        failures.append("expected exactly three warmups")
    else:
        for index, warmup in enumerate(warmups):
            warmup_hashes.append(
                check_request(
                    warmup,
                    expected_tokens=8,
                    label=f"warmup[{index}]",
                    failures=failures,
                )
            )

    request = measured_request(client)
    measured_hash = check_request(
        request,
        expected_tokens=32,
        label="measured",
        failures=failures,
    )
    phase = client.get("phases", [{}])[0]
    if (
        phase.get("concurrency") != 1
        or phase.get("request_count") != 1
        or phase.get("successful_requests") != 1
        or phase.get("failed_requests") != 0
    ):
        failures.append("measured C1 phase counts differ")

    internal_hash: str | None = None
    if args.same_framework_reference is not None:
        reference = load_json(args.same_framework_reference)
        internal_hash = reference.get("output_token_ids_sha256")
        if reference.get("status") != "ok" or internal_hash != measured_hash:
            failures.append("same-framework greedy output differs")
    checks["same_framework_output_hash"] = internal_hash
    checks["same_framework_output_equal"] = (
        internal_hash == measured_hash if internal_hash is not None else None
    )

    cross_hash: str | None = None
    if args.cross_framework_client is not None:
        cross_client = load_json(args.cross_framework_client)
        cross_request = measured_request(cross_client)
        if cross_request is None:
            failures.append("cross-framework reference client is malformed")
        else:
            cross_hash = cross_request.get("output_token_ids_sha256")
    cross_equal = (
        cross_hash == measured_hash
        if cross_hash is not None and measured_hash is not None
        else None
    )
    checks["cross_framework_output_hash"] = cross_hash
    checks["cross_framework_token_equal"] = cross_equal
    checks["cross_framework_numerical_gate_required"] = cross_equal is False

    if stats_path.is_file():
        stats = load_json(stats_path)
        stats_ok = (
            stats.get("pending_requests") == 0
            and stats.get("completed_requests") == 4
            and stats.get("cancelled_requests") == 0
            and stats.get("num_steps") == 56
            and stats.get("total_generated_tokens") == 56
            and stats.get("kv_cache_num_blocks")
            == stats.get("kv_cache_free_blocks")
        )
        checks["admin_stats_clean"] = stats_ok
        if not stats_ok:
            failures.append("final server stats are not clean")

    log = log_path.read_text(encoding="utf-8", errors="replace")
    fatal_patterns = (
        r"Traceback \(most recent call last\)",
        r"CUDA out of memory",
        r"OutOfMemoryError",
        r"illegal memory access",
        r"CUDA error:",
    )
    fatal_hits = [pattern for pattern in fatal_patterns if re.search(pattern, log)]
    checks["fatal_log_patterns"] = fatal_hits
    if fatal_hits:
        failures.append(f"fatal server log patterns: {fatal_hits}")

    result = {
        "schema_version": 1,
        "experiment": "unified_openai_sse_e0_qualification",
        "status": "ok" if not failures else "error",
        "scope": "qualification_only_not_formal_performance",
        "system": args.system,
        "case_root": str(args.case_root.resolve()),
        "client_result": str(client_path.resolve()),
        "measured_output_token_ids_sha256": measured_hash,
        "warmup_output_token_ids_sha256": warmup_hashes,
        "mean_ttft_ms": phase.get("ttft_ms", {}).get("mean"),
        "mean_tpot_ms": phase.get("tpot_ms", {}).get("mean"),
        "mean_e2e_ms": phase.get("e2e_ms", {}).get("mean"),
        "checks": checks,
        "failures": failures,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
