#!/usr/bin/env python3
"""Validate and summarize matched KTransformers eager/Graph run roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


COUNTER_RE = re.compile(
    r'^sglang:cuda_graph_passes_total\{(?P<labels>[^}]*)\}\s+(?P<value>[0-9.eE+-]+)$',
    re.MULTILINE,
)
MODE_RE = re.compile(r'mode="([^"]+)"')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def final_counters(path: Path) -> dict[str, float]:
    text = path.read_text()
    result: dict[str, float] = {}
    for match in COUNTER_RE.finditer(text):
        mode_match = MODE_RE.search(match.group("labels"))
        if mode_match:
            mode = mode_match.group(1)
            result[mode] = result.get(mode, 0.0) + float(match.group("value"))
    return result


def validate_root(root: Path, expected_mode: str) -> dict[str, Any]:
    result_path = root / "result.json"
    result = read_json(result_path)
    placement = read_json(root / "run_summary.json")
    if result.get("status") != "ok":
        raise ValueError(f"{root}: benchmark status is not ok")
    if placement.get("runtime_mask_verified") is not True:
        raise ValueError(f"{root}: runtime mask was not verified")
    if int(placement.get("cold_experts", -1)) != 512:
        raise ValueError(f"{root}: cold expert count differs")
    if int(placement.get("cold_main_weight_bytes", -1)) != 2_415_919_104:
        raise ValueError(f"{root}: cold main-weight bytes differ")

    summary = result.get("phase_summary", {})
    if sorted(summary) != ["1", "4"]:
        raise ValueError(f"{root}: batch summaries differ")
    hashes = {
        batch: tuple(summary[batch]["output_token_ids_sha256"])
        for batch in ("1", "4")
    }
    if any(len(values) != 1 for values in hashes.values()):
        raise ValueError(f"{root}: output hash drift")

    log = (root / "server.log").read_text(errors="replace")
    counters = final_counters(root / "metrics_final.txt")
    graph_true_logs = log.count("cuda graph: True")
    graph_false_logs = log.count("cuda graph: False")
    if expected_mode == "graph":
        if counters.get("decode_cuda_graph", 0.0) < 168 or graph_true_logs <= 0:
            raise ValueError(f"{root}: decode CUDA Graph evidence is incomplete")
    elif expected_mode == "eager":
        if counters.get("decode_none", 0.0) < 168 or graph_true_logs != 0:
            raise ValueError(f"{root}: eager decode evidence is incomplete")
    else:
        raise ValueError(expected_mode)

    return {
        "mode": expected_mode,
        "root": str(root.resolve()),
        "raw_result_sha256": sha256(result_path),
        "commits": {
            "ktransformers": (root / "ktransformers.commit").read_text().strip(),
            "sglang": (root / "sglang.commit").read_text().strip(),
        },
        "placement": placement,
        "cuda_graph": {
            "final_pass_counters": counters,
            "server_log_true_lines": graph_true_logs,
            "server_log_false_lines": graph_false_logs,
        },
        "phase_summary": summary,
        "output_hashes": hashes,
    }


def percent(candidate: float, baseline: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", required=True, type=Path)
    parser.add_argument("--eager-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    graph = validate_root(args.graph_root, "graph")
    eager = validate_root(args.eager_root, "eager")
    if graph["commits"] != eager["commits"]:
        raise ValueError("Graph/eager commits differ")
    if graph["output_hashes"] != eager["output_hashes"]:
        raise ValueError("Graph/eager output hashes differ")

    comparisons: dict[str, Any] = {}
    for batch in ("1", "4"):
        graph_summary = graph["phase_summary"][batch]
        eager_summary = eager["phase_summary"][batch]
        graph_ttft = float(graph_summary["client_ttft_batch_ms"]["median"])
        eager_ttft = float(eager_summary["client_ttft_batch_ms"]["median"])
        graph_decode = float(graph_summary["decode_step_ms"]["median"])
        eager_decode = float(eager_summary["decode_step_ms"]["median"])
        comparisons[batch] = {
            "graph_client_ttft_ms": graph_ttft,
            "eager_client_ttft_ms": eager_ttft,
            "graph_ttft_change_percent": percent(graph_ttft, eager_ttft),
            "graph_decode_step_ms": graph_decode,
            "eager_decode_step_ms": eager_decode,
            "graph_decode_change_percent": percent(graph_decode, eager_decode),
            "graph_decode_speedup": eager_decode / graph_decode,
            "graph_decode_tokens_per_second": float(
                graph_summary["decode_tokens_per_second"]["median"]
            ),
            "eager_decode_tokens_per_second": float(
                eager_summary["decode_tokens_per_second"]["median"]
            ),
        }

    payload = {
        "schema": "ktransformers_l0_l3_phase_comparison_v1",
        "status": "ok",
        "protocol": {
            "model": "Qwen3-30B-A3B-FP8 d206ba7",
            "cold_set": "L0-L3, 512 experts",
            "cold_main_weight_bytes": 2_415_919_104,
            "prompt_tokens": {"1": 314, "4": 1217},
            "output_tokens_per_request": 22,
            "decode_warmup_steps": 5,
            "decode_measured_steps": 16,
            "timing": "localhost batched HTTP/SSE client boundaries",
        },
        "graph": graph,
        "eager": eager,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ok", "comparisons": comparisons}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
