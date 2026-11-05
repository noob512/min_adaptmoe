#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


T_CRITICAL_95 = {
    1: 12.706204736,
    2: 4.302652730,
    3: 3.182446305,
    4: 2.776445105,
    5: 2.570581836,
    6: 2.446911851,
    7: 2.364624252,
    8: 2.306004135,
    9: 2.262157163,
    10: 2.228138852,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def output_hash(phase: dict) -> str:
    payload = [
        request.get("output_token_ids", []) for request in phase.get("requests", [])
    ]
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def summarize(values: list[float]) -> dict:
    result = {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "ci95_low": None,
        "ci95_high": None,
    }
    if len(values) > 1:
        critical = T_CRITICAL_95.get(len(values) - 1, 1.96)
        half_width = critical * statistics.stdev(values) / math.sqrt(len(values))
        result["ci95_low"] = result["mean"] - half_width
        result["ci95_high"] = result["mean"] + half_width
    return result


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    cases: list[dict] = []

    for case_root in sorted(root.glob("r*_*")):
        result_path = case_root / "native_uva" / "result.json"
        if not result_path.exists():
            continue
        result = read_json(result_path)
        repeat_part, label = case_root.name.split("_", 1)
        repeat = int(repeat_part[1:])
        fixture_path = case_root / "sharegpt_256_prompt320_360.json"
        manifest = [
            record
            for record in read_jsonl(
                case_root / "native_uva" / "cpu_offload_manifest.jsonl"
            )
            if record.get("event") == "cpu_offload_parameter"
        ]
        expert_manifest = [
            record
            for record in manifest
            if "expert" in str(record.get("full_parameter_name", "")).lower()
        ]
        phases = []
        for phase in result.get("phases", []):
            name = str(phase.get("phase", ""))
            if not name.startswith("decode_b"):
                continue
            batch = int(name.removeprefix("decode_b"))
            mean_tpot_ms = float(phase["mean_tpot_ms"])
            phases.append(
                {
                    "batch_tokens": batch,
                    "phase": name,
                    "request_count": int(phase["request_count"]),
                    "prompt_tokens": int(phase["prompt_tokens"]),
                    "output_tokens_per_request": int(
                        phase["output_tokens_per_request"]
                    ),
                    "elapsed_seconds": float(phase["elapsed_seconds"]),
                    "output_tok_s": float(phase["output_tokens_per_second"]),
                    "mean_tpot_ms": mean_tpot_ms,
                    "p99_tpot_ms": float(phase["p99_tpot_ms"]),
                    "steady_decode_tok_s": batch * 1000.0 / mean_tpot_ms,
                    "output_sha256": output_hash(phase),
                }
            )
        cases.append(
            {
                "repeat": repeat,
                "label": label,
                "scan_order": (case_root / "scan_order.txt").read_text().strip(),
                "status": result.get("status"),
                "load_seconds": float(result["load_seconds"]),
                "gpu_after_load_used_mib": int(
                    result.get("gpu_after_load", {}).get("used_mib", -1)
                ),
                "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                "offload_records": len(manifest),
                "offload_bytes": sum(int(row.get("bytes", 0)) for row in manifest),
                "expert_offload_bytes": sum(
                    int(row.get("bytes", 0)) for row in expert_manifest
                ),
                "all_uva_pinned": bool(manifest) and all(
                    row.get("offload_backend") == "uva"
                    and bool(row.get("uva_offloading"))
                    and bool(row.get("cpu_data_pinned"))
                    for row in manifest
                ),
                "phases": sorted(phases, key=lambda row: row["batch_tokens"]),
            }
        )

    by_case = {(row["repeat"], row["label"]): row for row in cases}
    paired_by_batch: dict[int, list[dict]] = {}
    for repeat in sorted({row["repeat"] for row in cases}):
        baseline = by_case.get((repeat, "baseline"))
        offload = by_case.get((repeat, "offload20"))
        if baseline is None or offload is None:
            continue
        baseline_phases = {
            row["batch_tokens"]: row for row in baseline.get("phases", [])
        }
        offload_phases = {
            row["batch_tokens"]: row for row in offload.get("phases", [])
        }
        for batch in sorted(set(baseline_phases) & set(offload_phases)):
            base = baseline_phases[batch]
            remote = offload_phases[batch]
            paired_by_batch.setdefault(batch, []).append(
                {
                    "repeat": repeat,
                    "scan_order": baseline["scan_order"],
                    "baseline_mean_tpot_ms": base["mean_tpot_ms"],
                    "offload_mean_tpot_ms": remote["mean_tpot_ms"],
                    "tpot_slowdown_pct":
                        (remote["mean_tpot_ms"] / base["mean_tpot_ms"] - 1.0)
                        * 100.0,
                    "baseline_steady_decode_tok_s": base["steady_decode_tok_s"],
                    "offload_steady_decode_tok_s": remote["steady_decode_tok_s"],
                    "decode_throughput_change_pct":
                        (
                            remote["steady_decode_tok_s"]
                            / base["steady_decode_tok_s"]
                            - 1.0
                        )
                        * 100.0,
                    "end_to_end_throughput_change_pct":
                        (remote["output_tok_s"] / base["output_tok_s"] - 1.0)
                        * 100.0,
                    "outputs_match": base["output_sha256"]
                    == remote["output_sha256"],
                }
            )

    batches = []
    for batch, pairs in sorted(paired_by_batch.items()):
        slowdowns = [row["tpot_slowdown_pct"] for row in pairs]
        slowdown_stats = summarize(slowdowns)
        clear_drop_5pct = (
            slowdown_stats["ci95_low"] is not None
            and slowdown_stats["ci95_low"] > 5.0
        )
        batches.append(
            {
                "batch_tokens": batch,
                "paired": pairs,
                "tpot_slowdown_pct": slowdown_stats,
                "decode_throughput_change_pct": summarize(
                    [row["decode_throughput_change_pct"] for row in pairs]
                ),
                "end_to_end_throughput_change_pct": summarize(
                    [row["end_to_end_throughput_change_pct"] for row in pairs]
                ),
                "clear_drop_5pct": clear_drop_5pct,
                "strong_drop_10pct": slowdown_stats["ci95_low"] is not None
                and slowdown_stats["ci95_low"] > 10.0,
                "all_outputs_match": all(row["outputs_match"] for row in pairs),
            }
        )

    clear = [row["batch_tokens"] for row in batches if row["clear_drop_5pct"]]
    strong = [row["batch_tokens"] for row in batches if row["strong_drop_10pct"]]
    summary = {
        "run_root": str(root),
        "method": {
            "primary_metric": "paired steady-decode mean TPOT slowdown",
            "clear_drop_rule": (
                "paired-mean TPOT slowdown 95% CI lower bound > 5%"
            ),
            "strong_drop_rule": (
                "paired-mean TPOT slowdown 95% CI lower bound > 10%"
            ),
            "decode_forward_tokens": (
                "batch size: EngineCore multiprocessing is disabled so all requests "
                "are queued before execution; equal output lengths plus ignore_eos "
                "keep every request active for each steady decode forward"
            ),
        },
        "cases": cases,
        "batches": batches,
        "conclusion": {
            "first_clear_drop_batch_tokens": min(clear) if clear else None,
            "first_strong_drop_batch_tokens": min(strong) if strong else None,
            "all_outputs_match": bool(batches)
            and all(row["all_outputs_match"] for row in batches),
        },
    }
    output_path = root / "comparison_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
