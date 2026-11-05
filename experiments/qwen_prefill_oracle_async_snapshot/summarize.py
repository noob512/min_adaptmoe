#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CASES = ("baseline", "remote", "sync", "async")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                values.append(value)
    return values


def output_hash(phase: dict[str, Any]) -> str:
    token_ids = [request["output_token_ids"] for request in phase["requests"]]
    payload = json.dumps(token_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-requests", type=int, default=4)
    parser.add_argument("--expected-prompt-tokens", type=int, default=1217)
    args = parser.parse_args()
    output = args.output or args.run_root / "snapshot.json"

    results: dict[str, Any] = {}
    for case in CASES:
        case_root = args.run_root / case
        returncode = int((case_root / "returncode.txt").read_text().strip())
        result = read_json(case_root / "result.json")
        phases = result.get("phases") or []
        if returncode != 0 or result.get("status") != "ok" or len(phases) != 1:
            raise SystemExit(
                f"invalid case {case}: rc={returncode} status={result.get('status')} "
                f"phases={len(phases)}"
            )
        phase = phases[0]
        expected_output_tokens = args.expected_requests
        expected_values = {
            "request_count": args.expected_requests,
            "completed_requests": args.expected_requests,
            "prompt_tokens": args.expected_prompt_tokens,
            "output_tokens": expected_output_tokens,
            "output_tokens_per_request": 1,
        }
        mismatched = {
            key: {"actual": phase.get(key), "expected": expected}
            for key, expected in expected_values.items()
            if int(phase.get(key, -1)) != expected
        }
        if phase.get("phase") != "prefill_eval" or mismatched:
            raise SystemExit(
                f"invalid prefill-only phase for {case}: "
                f"phase={phase.get('phase')} mismatched={mismatched}"
            )
        traces = read_jsonl(case_root / "oracle_trace.jsonl")
        if case in ("sync", "async"):
            matching = [
                item
                for item in traces
                if item.get("event") == "qwen3_moe_prefill_oracle_prefetch"
            ]
            if len(matching) != 1 or matching[0].get("mode") != case:
                raise SystemExit(f"invalid oracle trace for {case}: {matching!r}")
            oracle_trace: dict[str, Any] | None = matching[0]
            if (
                int(oracle_trace.get("num_tokens", -1))
                != args.expected_prompt_tokens
                or int(oracle_trace.get("layer_count", -1)) != 48
            ):
                raise SystemExit(
                    f"invalid oracle trace shape for {case}: "
                    f"tokens={oracle_trace.get('num_tokens')} "
                    f"layers={oracle_trace.get('layer_count')}"
                )
        else:
            oracle_trace = None
        results[case] = {
            "load_seconds": result["load_seconds"],
            "elapsed_seconds": phase["elapsed_seconds"],
            "prompt_tokens": phase["prompt_tokens"],
            "output_tokens": phase["output_tokens"],
            "output_tokens_per_second": phase["output_tokens_per_second"],
            "mean_ttft_ms": phase["mean_ttft_ms"],
            "p99_ttft_ms": phase["p99_ttft_ms"],
            "completed_requests": phase["completed_requests"],
            "output_token_hash": output_hash(phase),
            "oracle_trace": oracle_trace,
        }

    elapsed = {case: float(results[case]["elapsed_seconds"]) for case in CASES}
    ttft = {case: float(results[case]["mean_ttft_ms"]) for case in CASES}
    hashes = {results[case]["output_token_hash"] for case in CASES}
    snapshot = {
        "experiment": "qwen_prefill_oracle_async_snapshot",
        "run_root": str(args.run_root),
        "plan": read_json(args.run_root / "plans" / "plan_summary.json"),
        "fixture": read_json(args.run_root / "fixtures" / "fixture_metadata.json"),
        "driver": read_json(args.run_root / "driver_metadata.json"),
        "results": results,
        "comparisons": {
            "async_vs_remote_elapsed_speedup": elapsed["remote"] / elapsed["async"],
            "async_vs_sync_elapsed_speedup": elapsed["sync"] / elapsed["async"],
            "async_vs_remote_elapsed_reduction_pct": (
                1.0 - elapsed["async"] / elapsed["remote"]
            )
            * 100.0,
            "async_vs_sync_elapsed_reduction_pct": (
                1.0 - elapsed["async"] / elapsed["sync"]
            )
            * 100.0,
            "async_vs_remote_ttft_reduction_pct": (
                1.0 - ttft["async"] / ttft["remote"]
            )
            * 100.0,
            "async_vs_sync_ttft_reduction_pct": (
                1.0 - ttft["async"] / ttft["sync"]
            )
            * 100.0,
            "async_vs_baseline_elapsed_overhead_pct": (
                elapsed["async"] / elapsed["baseline"] - 1.0
            )
            * 100.0,
        },
        "integrity": {
            "all_returncodes_zero": True,
            "all_status_ok": True,
            "all_output_token_hashes_equal": len(hashes) == 1,
            "output_token_hashes": sorted(hashes),
        },
        "limitations": {
            "oracle_plan_uses_same_requests": True,
            "prefill_only_max_tokens": 1,
            "online_prediction_tested": False,
            "experts_remain_gpu_resident_after_prefetch": True,
            "single_snapshot_no_confidence_interval": True,
        },
    }
    if len(hashes) != 1:
        raise SystemExit(f"output token hashes differ: {hashes}")
    output.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
