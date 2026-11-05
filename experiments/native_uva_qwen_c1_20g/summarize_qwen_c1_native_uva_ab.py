#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


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


def sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def summarize_values(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sample_std": sample_std(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    cases: list[dict] = []
    for case_root in sorted(root.glob("r*_*")):
        result_path = case_root / "native_uva" / "result.json"
        if not result_path.exists():
            continue
        result = read_json(result_path)
        measure = next(
            phase for phase in result.get("phases", []) if phase.get("phase") == "measure"
        )
        manifest = [
            record
            for record in read_jsonl(
                case_root / "native_uva" / "cpu_offload_manifest.jsonl"
            )
            if record.get("event") == "cpu_offload_parameter"
        ]
        expert_records = [
            record
            for record in manifest
            if "expert" in str(record.get("full_parameter_name", "")).lower()
        ]
        label = case_root.name.split("_", 1)[1]
        repeat = int(case_root.name.split("_", 1)[0][1:])
        cases.append(
            {
                "repeat": repeat,
                "label": label,
                "status": result.get("status"),
                "fixture_sha256": hashlib.sha256(
                    (case_root / "sharegpt_256_prompt320_360.json").read_bytes()
                ).hexdigest(),
                "load_seconds": float(result["load_seconds"]),
                "gpu_after_load_used_mib": int(
                    result.get("gpu_after_load", {}).get("used_mib", -1)
                ),
                "measure_elapsed_seconds": float(measure["elapsed_seconds"]),
                "measure_output_tokens": int(measure["output_tokens"]),
                "measure_output_tok_s": float(measure["output_tokens_per_second"]),
                "measure_mean_tpot_ms": float(measure["mean_tpot_ms"]),
                "measure_p99_tpot_ms": float(measure["p99_tpot_ms"]),
                "output_sha256": output_hash(measure),
                "offload_records": len(manifest),
                "offload_bytes": sum(int(record.get("bytes", 0)) for record in manifest),
                "expert_offload_records": len(expert_records),
                "expert_offload_bytes": sum(
                    int(record.get("bytes", 0)) for record in expert_records
                ),
                "all_uva": all(
                    record.get("offload_backend") == "uva"
                    and bool(record.get("uva_offloading"))
                    for record in manifest
                ),
                "all_pinned": all(
                    bool(record.get("cpu_data_pinned")) for record in manifest
                ),
            }
        )

    grouped: dict[str, list[dict]] = {
        label: [case for case in cases if case["label"] == label]
        for label in sorted({case["label"] for case in cases})
    }
    aggregate = {}
    for label, selected in grouped.items():
        aggregate[label] = {
            "measure_output_tok_s": summarize_values(
                [case["measure_output_tok_s"] for case in selected]
            ),
            "measure_mean_tpot_ms": summarize_values(
                [case["measure_mean_tpot_ms"] for case in selected]
            ),
            "measure_elapsed_seconds": summarize_values(
                [case["measure_elapsed_seconds"] for case in selected]
            ),
            "load_seconds": summarize_values(
                [case["load_seconds"] for case in selected]
            ),
        }

    paired = []
    by_key = {(case["repeat"], case["label"]): case for case in cases}
    for repeat in sorted({case["repeat"] for case in cases}):
        baseline = by_key.get((repeat, "baseline"))
        offload = by_key.get((repeat, "offload20"))
        if baseline is None or offload is None:
            continue
        paired.append(
            {
                "repeat": repeat,
                "throughput_change_pct": (
                    offload["measure_output_tok_s"]
                    / baseline["measure_output_tok_s"]
                    - 1.0
                )
                * 100.0,
                "mean_tpot_change_pct": (
                    offload["measure_mean_tpot_ms"]
                    / baseline["measure_mean_tpot_ms"]
                    - 1.0
                )
                * 100.0,
                "elapsed_change_pct": (
                    offload["measure_elapsed_seconds"]
                    / baseline["measure_elapsed_seconds"]
                    - 1.0
                )
                * 100.0,
                "outputs_match": baseline["output_sha256"]
                == offload["output_sha256"],
            }
        )

    throughput_changes = [row["throughput_change_pct"] for row in paired]
    equivalence_margin_pct = 5.0
    conclusion = {
        "predeclared_equivalence_margin_pct": equivalence_margin_pct,
        "median_throughput_change_pct": (
            statistics.median(throughput_changes) if throughput_changes else None
        ),
        "all_pair_changes_within_margin": bool(throughput_changes)
        and all(abs(value) <= equivalence_margin_pct for value in throughput_changes),
        "all_outputs_match": bool(paired)
        and all(row["outputs_match"] for row in paired),
    }
    summary = {
        "run_root": str(root),
        "cases": cases,
        "aggregate": aggregate,
        "paired_comparisons": paired,
        "conclusion": conclusion,
    }
    output_path = root / "comparison_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
