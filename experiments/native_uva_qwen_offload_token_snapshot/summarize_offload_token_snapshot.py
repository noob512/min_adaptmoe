#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path


CASE_PATTERN = re.compile(r"pair(?P<pair>\d+)_g(?P<gib>\d+)_(?P<label>baseline|offload)$")


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
    payload = [row.get("output_token_ids", []) for row in phase.get("requests", [])]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def pcie_rx_peak_mb_s(path: Path) -> int | None:
    values: list[int] = []
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[0].isdigit() and fields[3].isdigit():
            values.append(int(fields[3]))
    return max(values) if values else None


def scheduling_log_checks(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    waiting_values = [
        int(value) for value in re.findall(r"Waiting:\s*(\d+)", text)
    ]
    lowered = text.lower()
    return {
        "max_logged_waiting_requests": max(waiting_values, default=0),
        "preemption_keyword_seen": any(
            keyword in lowered for keyword in ("preempt", "preemption", "swapped")
        ),
    }


def first_persistent_threshold(rows: list[dict], threshold_pct: float) -> int | None:
    for index, row in enumerate(rows):
        tail = rows[index:]
        if len(tail) >= 2 and all(
            item["tpot_slowdown_pct"] >= threshold_pct for item in tail
        ):
            return int(row["batch_tokens"])
    return None


def first_observed_threshold(rows: list[dict], threshold_pct: float) -> int | None:
    for row in rows:
        if row["tpot_slowdown_pct"] >= threshold_pct:
            return int(row["batch_tokens"])
    return None


def first_confirmed_threshold(rows: list[dict], threshold_pct: float) -> int | None:
    for row, next_row in zip(rows, rows[1:]):
        if (
            row["tpot_slowdown_pct"] >= threshold_pct
            and next_row["tpot_slowdown_pct"] >= threshold_pct
        ):
            return int(row["batch_tokens"])
    return None


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    cases: list[dict] = []
    for case_root in sorted(root.iterdir()):
        match = CASE_PATTERN.fullmatch(case_root.name)
        if match is None:
            continue
        result_path = case_root / "native_uva" / "result.json"
        if not result_path.exists():
            continue
        result = read_json(result_path)
        manifest = [
            row
            for row in read_jsonl(
                case_root / "native_uva" / "cpu_offload_manifest.jsonl"
            )
            if row.get("event") == "cpu_offload_parameter"
        ]
        expert_manifest = [
            row
            for row in manifest
            if "experts" in str(row.get("full_parameter_name", "")).lower()
        ]
        expert_layers: dict[int, int] = {}
        for row in expert_manifest:
            layer_match = re.search(
                r"model\.layers\.(\d+)\.mlp\.experts\.",
                str(row.get("full_parameter_name", "")),
            )
            if layer_match:
                layer = int(layer_match.group(1))
                expert_layers[layer] = expert_layers.get(layer, 0) + 1
        phases = []
        for phase in result.get("phases", []):
            name = str(phase.get("phase", ""))
            if not name.startswith("decode_b"):
                continue
            batch = int(name.removeprefix("decode_b"))
            tpot_ms = float(phase["mean_tpot_ms"])
            phases.append(
                {
                    "batch_tokens": batch,
                    "request_count": int(phase["request_count"]),
                    "prompt_tokens": int(phase["prompt_tokens"]),
                    "output_tokens_per_request": int(
                        phase["output_tokens_per_request"]
                    ),
                    "mean_tpot_ms": tpot_ms,
                    "phase_output_tok_s": float(phase["output_tokens_per_second"]),
                    "steady_decode_tok_s": batch * 1000.0 / tpot_ms,
                    "output_sha256": output_hash(phase),
                }
            )
        fixture_path = case_root / "sharegpt_256_prompt320_360.json"
        cases.append(
            {
                "pair": int(match.group("pair")),
                "requested_offload_gib": int(match.group("gib")),
                "label": match.group("label"),
                "scan_order": (case_root / "scan_order.txt").read_text().strip(),
                "status": result.get("status"),
                "load_seconds": float(result["load_seconds"]),
                "gpu_after_load_used_mib": int(
                    result.get("gpu_after_load", {}).get("used_mib", -1)
                ),
                "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                "offload_parameter_records": len(manifest),
                "offload_bytes": sum(int(row.get("bytes", 0)) for row in manifest),
                "expert_parameter_records": len(expert_manifest),
                "expert_offload_bytes": sum(
                    int(row.get("bytes", 0)) for row in expert_manifest
                ),
                "fully_offloaded_expert_layers": sorted(
                    layer for layer, count in expert_layers.items() if count == 4
                ),
                "partial_expert_layers": sorted(
                    layer for layer, count in expert_layers.items() if count != 4
                ),
                "all_offload_records_uva_pinned": bool(manifest)
                and all(
                    row.get("offload_backend") == "uva"
                    and bool(row.get("uva_offloading"))
                    and bool(row.get("cpu_data_pinned"))
                    for row in manifest
                ),
                "pcie_rx_peak_mb_s": pcie_rx_peak_mb_s(
                    case_root / "dmon_pcie.csv"
                ),
                **scheduling_log_checks(case_root / "native_uva" / "run.log"),
                "phases": sorted(phases, key=lambda row: row["batch_tokens"]),
            }
        )

    by_key = {(row["pair"], row["label"]): row for row in cases}
    snapshots = []
    for pair in sorted({row["pair"] for row in cases}):
        baseline = by_key.get((pair, "baseline"))
        offload = by_key.get((pair, "offload"))
        if baseline is None or offload is None:
            continue
        base_phases = {row["batch_tokens"]: row for row in baseline["phases"]}
        offload_phases = {row["batch_tokens"]: row for row in offload["phases"]}
        rows = []
        for batch in sorted(set(base_phases) & set(offload_phases)):
            base = base_phases[batch]
            remote = offload_phases[batch]
            rows.append(
                {
                    "batch_tokens": batch,
                    "baseline_tpot_ms": base["mean_tpot_ms"],
                    "offload_tpot_ms": remote["mean_tpot_ms"],
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
                    "outputs_match": base["output_sha256"]
                    == remote["output_sha256"],
                }
            )
        actual_bytes = offload["offload_bytes"]
        expert_bytes = offload["expert_offload_bytes"]
        snapshots.append(
            {
                "pair": pair,
                "requested_offload_gib": offload["requested_offload_gib"],
                "scan_order": offload["scan_order"],
                "actual_offload_bytes": actual_bytes,
                "actual_offload_gib": actual_bytes / (1024**3),
                "expert_offload_bytes": expert_bytes,
                "expert_offload_gib": expert_bytes / (1024**3),
                "expert_fraction_pct":
                    expert_bytes / actual_bytes * 100.0 if actual_bytes else None,
                "offload_parameter_records": offload["offload_parameter_records"],
                "expert_parameter_records": offload["expert_parameter_records"],
                "fully_offloaded_expert_layers": offload[
                    "fully_offloaded_expert_layers"
                ],
                "partial_expert_layers": offload["partial_expert_layers"],
                "all_offload_records_uva_pinned": offload[
                    "all_offload_records_uva_pinned"
                ],
                "baseline_gpu_after_load_used_mib": baseline[
                    "gpu_after_load_used_mib"
                ],
                "offload_gpu_after_load_used_mib": offload[
                    "gpu_after_load_used_mib"
                ],
                "offload_pcie_rx_peak_mb_s": offload["pcie_rx_peak_mb_s"],
                "fixture_match": baseline["fixture_sha256"]
                == offload["fixture_sha256"],
                "all_outputs_match": all(row["outputs_match"] for row in rows),
                "first_observed_drop_10pct": first_observed_threshold(rows, 10.0),
                "first_confirmed_drop_10pct": first_confirmed_threshold(
                    rows, 10.0
                ),
                "first_persistent_drop_5pct": first_persistent_threshold(rows, 5.0),
                "first_persistent_drop_10pct": first_persistent_threshold(rows, 10.0),
                "gray_batches_5_to_10pct": [
                    row["batch_tokens"]
                    for row in rows
                    if 5.0 <= row["tpot_slowdown_pct"] < 10.0
                ],
                "results": rows,
            }
        )

    summary = {
        "run_root": str(root),
        "method": {
            "kind": "single cold-process pair per offload level; screening only",
            "batch_tokens": sorted(
                {
                    row["batch_tokens"]
                    for snapshot in snapshots
                    for row in snapshot["results"]
                }
            ),
            "screening_boundary": (
                "first tested batch where TPOT slowdown remains >=10% at all "
                "larger tested batches, with at least one larger batch available"
            ),
            "observed_boundary": "first tested batch with TPOT slowdown >=10%",
            "confirmed_boundary": (
                "first tested batch with TPOT slowdown >=10% whose next larger "
                "tested batch also has slowdown >=10%"
            ),
            "gray_zone": "5% <= TPOT slowdown < 10%",
            "no_confidence_intervals": True,
        },
        "cases": cases,
        "snapshots": sorted(
            snapshots, key=lambda row: row["requested_offload_gib"]
        ),
        "integrity": {
            "all_cases_ok": bool(cases)
            and all(row["status"] == "ok" for row in cases),
            "all_fixtures_match_within_pair": bool(snapshots)
            and all(row["fixture_match"] for row in snapshots),
            "all_outputs_match_within_pair": bool(snapshots)
            and all(row["all_outputs_match"] for row in snapshots),
            "all_offload_records_uva_pinned": bool(snapshots)
            and all(row["all_offload_records_uva_pinned"] for row in snapshots),
            "no_logged_waiting_requests": bool(cases)
            and all(row["max_logged_waiting_requests"] == 0 for row in cases),
            "no_preemption_keyword_seen": bool(cases)
            and not any(row["preemption_keyword_seen"] for row in cases),
        },
    }
    output_path = root / "snapshot_summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
