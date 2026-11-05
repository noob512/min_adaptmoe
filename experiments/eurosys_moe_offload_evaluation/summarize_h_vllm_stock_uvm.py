#!/usr/bin/env python3
"""Summarize the single-round Device-resident versus Stock-UVM experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def metric(result: dict, *keys: str) -> float | None:
    value: object = result
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()

    records: dict[tuple[str, str, int], dict[str, dict]] = {}
    with (root / "execution.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            case_root = Path(row["case_root"])
            result_path = case_root / "result.json"
            result = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.is_file()
                else {}
            )
            mode = row["mode"]
            key = (row["model"], row["workload"], int(row["concurrency"]))
            records.setdefault(key, {})[mode] = {
                "status": row["status"],
                "returncode": int(row["returncode"]),
                "case_root": str(case_root),
                "result": result,
            }

    rows: list[dict] = []
    comparable_matches = 0
    for (model, workload, concurrency), arms in sorted(records.items()):
        device = arms["device_resident"]
        uvm = arms["stock_uvm_demand"]
        d_result = device["result"]
        u_result = uvm["result"]
        d_ttft = metric(d_result, "measurement", "ttft_ms", "p50")
        d_tpot = metric(d_result, "measurement", "tpot_ms", "p50")
        d_e2e = metric(d_result, "measurement", "e2e_ms", "p50")
        u_ttft = metric(u_result, "measurement", "ttft_ms", "p50")
        u_tpot = metric(u_result, "measurement", "tpot_ms", "p50")
        u_e2e = metric(u_result, "measurement", "e2e_ms", "p50")
        d_hashes = [
            request["output_token_ids_sha256"]
            for request in (d_result.get("measurement") or {}).get("requests", [])
        ]
        u_hashes = [
            request["output_token_ids_sha256"]
            for request in (u_result.get("measurement") or {}).get("requests", [])
        ]
        comparable = device["status"] == "ok" and uvm["status"] == "ok"
        outputs_match = comparable and d_hashes == u_hashes
        comparable_matches += int(outputs_match)
        rows.append(
            {
                "model": model,
                "workload": workload,
                "concurrency": concurrency,
                "device_status": device["status"],
                "uvm_status": uvm["status"],
                "device_ttft_ms": d_ttft,
                "uvm_ttft_ms": u_ttft,
                "ttft_ratio": ratio(u_ttft, d_ttft),
                "device_tpot_ms": d_tpot,
                "uvm_tpot_ms": u_tpot,
                "tpot_ratio": ratio(u_tpot, d_tpot),
                "device_e2e_ms": d_e2e,
                "uvm_e2e_ms": u_e2e,
                "e2e_ratio": ratio(u_e2e, d_e2e),
                "comparable": comparable,
                "outputs_match": outputs_match if comparable else None,
            }
        )

    logs = sorted(root.glob("cases/stock_uvm_demand/*/*/c*/uvm_allocations.log"))
    forbidden = (
        "ManagedPrefetchGpu",
        "ManagedAdvisePrefetchGpu",
        "cudaMemPrefetchAsync",
        "cudaMemAdvise",
    )
    required_flags = (
        "policy_enabled=0",
        "policy_warmup_prefetch_enabled=0",
        "policy_warmup_advise_gpu=0",
        "gap_watch_enabled=0",
        "device_direct_enable=0",
        "kv_device_direct_enable=0",
        "fixed_weight_device_direct_enable=0",
    )
    log_audit = []
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        log_audit.append(
            {
                "path": str(path),
                "required_flags_present": all(flag in text for flag in required_flags),
                "forbidden_explicit_operations": [item for item in forbidden if item in text],
                "trace_alloc_present": "TRACE_ALLOC" in text,
            }
        )

    comparable = [row for row in rows if row["comparable"]]
    summary = {
        "schema": "h_vllm_stock_uvm_summary_v1",
        "run_root": str(root),
        "cells": len(rows) * 2,
        "logical_workload_cells": len(rows),
        "device_ok": sum(row["device_status"] == "ok" for row in rows),
        "device_oom": sum(row["device_status"] == "OOM" for row in rows),
        "uvm_ok": sum(row["uvm_status"] == "ok" for row in rows),
        "comparable_cells": len(comparable),
        "comparable_output_matches": comparable_matches,
        "ratio_ranges": {
            name: {
                "min": min(row[name] for row in comparable),
                "max": max(row[name] for row in comparable),
            }
            for name in ("ttft_ratio", "tpot_ratio", "e2e_ratio")
        },
        "uvm_allocator_audit": {
            "log_count": len(logs),
            "all_required_disable_flags_present": all(
                item["required_flags_present"] for item in log_audit
            ),
            "explicit_prefetch_or_advice_matches": sum(
                len(item["forbidden_explicit_operations"]) for item in log_audit
            ),
            "all_have_managed_allocation_trace": all(
                item["trace_alloc_present"] for item in log_audit
            ),
            "logs": log_audit,
        },
        "rows": rows,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fieldnames = list(rows[0])
    with (root / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    valid = (
        len(rows) == 18
        and summary["device_ok"] + summary["device_oom"] == 18
        and summary["uvm_ok"] == 18
        and comparable_matches == len(comparable)
        and len(logs) == 18
        and summary["uvm_allocator_audit"]["all_required_disable_flags_present"]
        and summary["uvm_allocator_audit"]["explicit_prefetch_or_advice_matches"] == 0
        and summary["uvm_allocator_audit"]["all_have_managed_allocation_trace"]
    )
    print(json.dumps({key: summary[key] for key in (
        "cells", "device_ok", "device_oom", "uvm_ok", "comparable_cells",
        "comparable_output_matches", "ratio_ranges", "uvm_allocator_audit"
    ) if key != "uvm_allocator_audit"} | {
        "uvm_allocator_audit": {
            key: summary["uvm_allocator_audit"][key]
            for key in (
                "log_count", "all_required_disable_flags_present",
                "explicit_prefetch_or_advice_matches",
                "all_have_managed_allocation_trace",
            )
        },
        "valid": valid,
    }, ensure_ascii=False, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
