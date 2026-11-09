#!/usr/bin/env python3
"""Summarize fresh execution records without replacing the stored figure data."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def inspect_stock_outputs(root: Path) -> tuple[list[dict], list[dict]]:
    """Compare completed stock-UVM cells with the matching completed baseline."""
    audits, failures = [], []
    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    for protocol in sorted(root.rglob("protocol.json")):
        try:
            schema = json.loads(protocol.read_text()).get("schema")
        except (ValueError, OSError) as exc:
            failures.append({"file": str(protocol.relative_to(root)), "error": str(exc)})
            continue
        baseline = {"h_vllm_stock_uvm_v1": "device_resident",
                    "qwen_kv6_uva_vs_stock_uvm_v1": "native_uva_6g"}.get(schema)
        if baseline is None:
            continue
        table = protocol.parent / "execution.tsv"
        pairs, cells = [], {}
        try:
            with table.open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
        except OSError as exc:
            failures.append({"file": str(table.relative_to(root)), "error": str(exc)})
            continue
        for row in rows:
            if row["status"] != "ok":
                continue
            case = Path(row["case_root"])
            try:
                result = json.loads((case / "result.json").read_text())
                measurement = result["measurement"]
                requests = measurement["requests"]
                batch = int(row["concurrency"])
                require(result["status"] == "ok", "result status differs from completed execution row")
                require(measurement["successful_requests"] == len(requests) == batch,
                        "completed request count differs from concurrency")
                signatures = {}
                for request in requests:
                    ids = request["output_token_ids"]
                    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
                    require(request["success"] is True, "request did not complete")
                    require(request["completion_tokens"] == len(ids), "output length differs from receipt")
                    require(len(ids) == int(result["configuration"]["output_tokens"]),
                            "output length differs from the configured request")
                    require(request["output_token_ids_sha256"] == digest, "stored output hash differs from token IDs")
                    key = request["fixture_index"]
                    require(key not in signatures, "duplicate fixture index")
                    signatures[key] = (request["prompt_token_ids_sha256"], digest)
                cell = (row.get("model", "qwen"), row["workload"], batch)
                require((row["mode"], cell) not in cells, "duplicate completed cell")
                cells[row["mode"], cell] = signatures
            except (KeyError, ValueError, OSError) as exc:
                failures.append({"file": str(table.relative_to(root)), "case": str(case),
                                 "error": f"Invalid completed output: {type(exc).__name__}: {exc}"})
        for (mode, cell), signatures in sorted(cells.items()):
            if mode != "stock_uvm_demand" or (baseline, cell) not in cells:
                continue
            equal = signatures == cells[baseline, cell]
            entry = {"model": cell[0], "workload": cell[1], "concurrency": cell[2],
                     "baseline": baseline, "prompt_and_output_hashes_equal": equal}
            pairs.append(entry)
            if not equal:
                failures.append({"file": str(table.relative_to(root)), **entry})
        audits.append({"file": str(table.relative_to(root)), "valid_completed_cells": len(cells),
                       "completed_pairs": pairs})
    return audits, failures


def inspect_run(root: Path) -> dict:
    """Keep cell failures, verifier status and performance claims distinct."""
    checks, tables, failures, claims = [], [], [], []
    for path in sorted(root.rglob("verification.json")):
        relative = str(path.relative_to(root))
        try:
            value = json.loads(path.read_text())
        except (ValueError, OSError) as exc:
            failures.append({"file": relative, "error": str(exc)})
            continue
        status = value.get("status")
        checks.append({"file": relative, "status": status})
        if status not in ("ok", "passed", "pass"):
            failures.append({"file": relative, "status": status})
        performance = {key: item for key, item in value.items()
                       if key in ("claim_gate", "claims", "hypotheses")
                       or key.startswith("all_equivalence_")}
        if performance:
            claims.append({"file": relative, **performance})
    for path in sorted(root.rglob("execution*.tsv")):
        with path.open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        statuses = Counter()
        bad, skipped = [], []
        for ordinal, row in enumerate(rows, 1):
            if row.get("status"):
                statuses[row["status"]] += 1
            if any(str(value).startswith("skipped") for value in row.values()):
                skipped.append({"row": ordinal, "values": row})
                continue
            problems = {k: row[k] for k in ("returncode", "verification", "launcher_rc")
                        if row.get(k) not in (None, "", "0")}
            if row.get("status") not in (None, "", "ok", "pass", "passed", "0"):
                problems["status"] = row["status"]
            if problems:
                bad.append({"row": ordinal, "values": row, "problems": problems})
        tables.append({"file": str(path.relative_to(root)), "rows": len(rows),
                       "status_counts": dict(statuses), "unsuccessful_rows": bad,
                       "skipped_rows": skipped})
        failures.extend({"file": str(path.relative_to(root)), **entry} for entry in bad)
    stock_outputs, output_failures = inspect_stock_outputs(root)
    failures.extend(output_failures)
    return {"verification": checks, "execution_tables": tables,
            "stock_output_audits": stock_outputs,
            "unsuccessful_records": failures, "performance_claims": claims,
            "note": "An unsuccessful capacity cell can be an expected observation; inspect its protocol. "
                    "A zero launcher return code alone does not establish correctness or performance agreement."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    report = {"schema": "adaptmoe_figure_run_report_v1", "runs": []}
    for root in args.runs:
        execution = root / "execution.json"
        if execution.exists():
            receipt = json.loads(execution.read_text())
            for job in receipt["jobs"]:
                report["runs"].append({"root": str(root.resolve()), "job": job,
                    "observations": inspect_run(root / job["name"])})
        else:
            report["runs"].append({"root": str(root.resolve()),
                                  "observations": inspect_run(root)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
