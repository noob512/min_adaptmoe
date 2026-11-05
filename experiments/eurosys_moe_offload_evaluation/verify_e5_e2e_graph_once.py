#!/usr/bin/env python3
"""Summarize the one-cold-process E5 H/G end-to-end ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.run_root.resolve()
    output = (args.output or root / "verification.json").resolve()
    rows: list[dict[str, str]] = []
    with (root / "execution_status.tsv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    require(len(rows) == 4, f"expected four H/G cells, got {len(rows)}")

    cells: dict[str, dict[str, Any]] = {}
    for row in rows:
        require(row["returncode"] == "0", f"failed cell: {row}")
        require(row["symbol"] in {"H", "G"}, f"unexpected arm: {row}")
        result = load(root / row["directory"] / "result.json")
        require(result.get("status") == "ok", f"bad result: {row['directory']}")
        timing = result.get("timing")
        require(isinstance(timing, dict), f"missing timing: {row['directory']}")
        steady = timing.get("steady_decode_step_ms")
        require(isinstance(steady, dict), f"missing steady decode: {row['directory']}")
        key = f"{row['symbol']}_B{row['batch']}"
        cells[key] = {
            "arm": result.get("arm_name"),
            "batch": int(row["batch"]),
            "prefill_ms": float(timing["prefill_step_ms"]),
            "cutover_ms": float(timing["cutover_wall_ms"]),
            "first_decode_ms": float(timing["first_decode_step_ms"]),
            "steady_decode_ms": float(steady["median"]),
            "decode_tokens_per_second": float(
                timing["steady_decode_tokens_per_second"]
            ),
            "cycle_e2e_ms": float(timing["cycle_e2e_ms"]),
            "output_sha256": result.get("output_token_ids_sha256"),
        }

    comparisons: dict[str, Any] = {}
    for batch in (1, 4):
        eager = cells[f"H_B{batch}"]
        graph = cells[f"G_B{batch}"]
        comparisons[f"B{batch}"] = {
            "decode_graph_speedup": (
                eager["steady_decode_ms"] / graph["steady_decode_ms"]
            ),
            "cycle_e2e_speedup": eager["cycle_e2e_ms"] / graph["cycle_e2e_ms"],
            "prefill_ratio_eager_over_decode_graph": (
                eager["prefill_ms"] / graph["prefill_ms"]
            ),
        }

    payload = {
        "schema": "e5_e2e_graph_once_verification_v1",
        "status": "ok",
        "protocol": {
            "cold_processes_per_cell": 1,
            "semantic_cross_arm_gate": "not_applied_by_request",
            "model": "Qwen3-30B-A3B-FP8",
            "logical_parameter_bytes": 2_415_919_104,
            "batches": [1, 4],
        },
        "cells": cells,
        "comparisons": comparisons,
    }
    require(not output.exists(), f"refusing to overwrite {output}")
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
