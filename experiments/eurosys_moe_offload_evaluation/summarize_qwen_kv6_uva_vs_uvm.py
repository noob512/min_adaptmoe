#!/usr/bin/env python3
"""Summarize the Qwen 6-GiB-KV Native-UVA versus Stock-UVM run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


GIB = 1024**3
EXPECTED_KV_BYTES = 6 * GIB
WORKLOAD_LABELS = {
    "short_dialogue": "Short",
    "ordinary_chat": "Ordinary",
    "medium_context": "Medium",
    "rag_summary": "RAG",
    "long_generation": "Long",
}
MODES = ("native_uva_6g", "stock_uvm_demand")


def nested_number(value: dict[str, Any], *keys: str) -> float | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    return float(current)


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def actual_offload_bytes(path: Path) -> int | None:
    if not path.is_file():
        return None
    maximum: int | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = record.get("cumulative_offload_bytes")
        if isinstance(value, int):
            maximum = value if maximum is None else max(maximum, value)
    return maximum


def output_hashes(result: dict[str, Any]) -> list[str]:
    measurement = result.get("measurement") or {}
    return [
        str(request.get("output_token_ids_sha256"))
        for request in measurement.get("requests", [])
    ]


def load_result(case_root: Path) -> dict[str, Any]:
    path = case_root / "result.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def fmt_ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:,.2f}×"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()

    records: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    with (root / "execution.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            workload = row["workload"]
            concurrency = int(row["concurrency"])
            mode = row["mode"]
            case_root = Path(row["case_root"])
            result = load_result(case_root)
            records.setdefault((workload, concurrency), {})[mode] = {
                "status": row["status"],
                "returncode": int(row["returncode"]),
                "kv_cache_bytes": int(row["kv_cache_bytes"]),
                "case_root": str(case_root),
                "result": result,
                "ttft_ms": nested_number(result, "measurement", "ttft_ms", "p50"),
                "tpot_ms": nested_number(result, "measurement", "tpot_ms", "p50"),
                # Full-matrix E2E is when the final request in the admitted batch
                # completes. Direct-batch requests normally finish on one step,
                # but use max explicitly rather than request-level P50.
                "e2e_ms": nested_number(result, "measurement", "e2e_ms", "max"),
                "actual_offload_bytes": actual_offload_bytes(
                    case_root / "offload_manifest.jsonl"
                ) if mode == "native_uva_6g" else None,
            }

    rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for (workload, concurrency), arms in sorted(
        records.items(), key=lambda item: (
            list(WORKLOAD_LABELS).index(item[0][0]), item[0][1]
        )
    ):
        for mode in MODES:
            if mode not in arms:
                continue
            arm = arms[mode]
            rows.append({
                "workload": workload,
                "concurrency": concurrency,
                "mode": mode,
                "status": arm["status"],
                "returncode": arm["returncode"],
                "kv_cache_bytes": arm["kv_cache_bytes"],
                "actual_offload_bytes": arm["actual_offload_bytes"],
                "actual_offload_gib": (
                    arm["actual_offload_bytes"] / GIB
                    if arm["actual_offload_bytes"] is not None else None
                ),
                "ttft_ms": arm["ttft_ms"],
                "tpot_ms": arm["tpot_ms"],
                "e2e_ms": arm["e2e_ms"],
                "case_root": arm["case_root"],
            })
        if all(mode in arms for mode in MODES):
            uva, uvm = (arms[mode] for mode in MODES)
            comparable = uva["status"] == "ok" and uvm["status"] == "ok"
            pairs.append({
                "workload": workload,
                "concurrency": concurrency,
                "comparable": comparable,
                "outputs_match": (
                    output_hashes(uva["result"]) == output_hashes(uvm["result"])
                    if comparable else None
                ),
                "uvm_over_uva_ttft": ratio(uvm["ttft_ms"], uva["ttft_ms"]),
                "uvm_over_uva_tpot": ratio(uvm["tpot_ms"], uva["tpot_ms"]),
                "uvm_over_uva_e2e": ratio(uvm["e2e_ms"], uva["e2e_ms"]),
            })

    uvm_logs = sorted(root.glob("cases/stock_uvm_demand/qwen/*/c*/uvm_allocations.log"))
    forbidden = ("ManagedPrefetchGpu", "ManagedAdvisePrefetchGpu",
                 "cudaMemPrefetchAsync", "cudaMemAdvise")
    required_flags = (
        "policy_enabled=0", "policy_warmup_prefetch_enabled=0",
        "policy_warmup_advise_gpu=0", "gap_watch_enabled=0",
        "device_direct_enable=0", "kv_device_direct_enable=0",
        "fixed_weight_device_direct_enable=0",
    )
    audit_rows = []
    for path in uvm_logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        audit_rows.append({
            "path": str(path),
            "required_flags_present": all(flag in text for flag in required_flags),
            "forbidden_explicit_operations": [op for op in forbidden if op in text],
            "trace_alloc_present": "TRACE_ALLOC" in text,
        })

    complete_protocol = len(rows) == 20 and len(pairs) == 10
    comparable_pairs = [pair for pair in pairs if pair["comparable"]]
    summary = {
        "schema": "qwen_kv6_uva_vs_stock_uvm_summary_v1",
        "run_root": str(root),
        "kv_cache_bytes": EXPECTED_KV_BYTES,
        "kv_cache_gib": 6,
        "cells": len(rows),
        "logical_pairs": len(pairs),
        "uva_ok": sum(row["mode"] == MODES[0] and row["status"] == "ok" for row in rows),
        "uvm_ok": sum(row["mode"] == MODES[1] and row["status"] == "ok" for row in rows),
        "comparable_pairs": len(comparable_pairs),
        "matching_output_pairs": sum(pair["outputs_match"] is True for pair in pairs),
        "complete_protocol": complete_protocol,
        "uvm_allocator_audit": {
            "log_count": len(uvm_logs),
            "all_required_disable_flags_present": bool(audit_rows) and all(
                row["required_flags_present"] for row in audit_rows
            ),
            "explicit_prefetch_or_advice_matches": sum(
                len(row["forbidden_explicit_operations"]) for row in audit_rows
            ),
            "all_have_managed_allocation_trace": bool(audit_rows) and all(
                row["trace_alloc_present"] for row in audit_rows
            ),
            "logs": audit_rows,
        },
        "rows": rows,
        "pairs": pairs,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if rows:
        with (root / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# Qwen3-30B-A3B FP8：6 GiB KV cache 下 Native UVA 与 Stock UVM",
        "",
        "单轮冷进程；direct batch；TRITON_ATTN；3 个 warmup 请求；"
        "TTFT/TPOT 为请求级 P50，E2E 为整批最后一个请求完成时间。",
        "",
        "| 场景 | C | 机制 | KV cache (GiB) | 实际显式卸载 (GiB) | TTFT (ms) | TPOT (ms) | E2E (ms) | 状态 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        mechanism = "Native UVA" if row["mode"] == MODES[0] else "Stock UVM demand"
        explicit = fmt(row["actual_offload_gib"], 3) if row["mode"] == MODES[0] else "0"
        lines.append(
            f"| {WORKLOAD_LABELS[row['workload']]} | {row['concurrency']} | {mechanism} "
            f"| 6 | {explicit} | {fmt(row['ttft_ms'])} | {fmt(row['tpot_ms'])} "
            f"| {fmt(row['e2e_ms'])} | {row['status']} |"
        )
    lines.extend([
        "",
        "比值均为 Stock UVM / Native UVA（大于 1 表示 UVM 延迟更高）。",
        "",
        "| 场景 | C | TTFT 比值 | TPOT 比值 | E2E 比值 | 输出一致 |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for pair in pairs:
        matches = "是" if pair["outputs_match"] is True else (
            "否" if pair["outputs_match"] is False else "—"
        )
        lines.append(
            f"| {WORKLOAD_LABELS[pair['workload']]} | {pair['concurrency']} "
            f"| {fmt_ratio(pair['uvm_over_uva_ttft'])} "
            f"| {fmt_ratio(pair['uvm_over_uva_tpot'])} "
            f"| {fmt_ratio(pair['uvm_over_uva_e2e'])} | {matches} |"
        )
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    audit = summary["uvm_allocator_audit"]
    valid = (
        complete_protocol
        and all(row["kv_cache_bytes"] == EXPECTED_KV_BYTES for row in rows)
        and all(
            row["status"] in ("ok", "OOM")
            for row in rows if row["mode"] == MODES[0]
        )
        and summary["uvm_ok"] == 10
        and summary["matching_output_pairs"] == summary["comparable_pairs"]
        and audit["log_count"] == 10
        and audit["all_required_disable_flags_present"]
        and audit["explicit_prefetch_or_advice_matches"] == 0
        and audit["all_have_managed_allocation_trace"]
    )
    print(json.dumps({
        "cells": summary["cells"], "logical_pairs": summary["logical_pairs"],
        "uva_ok": summary["uva_ok"], "uvm_ok": summary["uvm_ok"],
        "comparable_pairs": summary["comparable_pairs"],
        "matching_output_pairs": summary["matching_output_pairs"],
        "complete_protocol": complete_protocol, "valid": valid,
    }, ensure_ascii=False, indent=2))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
