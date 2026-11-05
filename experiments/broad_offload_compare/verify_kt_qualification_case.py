#!/usr/bin/env python3
"""Verify that a KT qualification server actually used the requested profile."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any


def option_value(tokens: list[str], option: str) -> str | None:
    matches = [index for index, token in enumerate(tokens) if token == option]
    if len(matches) != 1:
        return None
    index = matches[0]
    if index + 1 >= len(tokens):
        return None
    return tokens[index + 1]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("exact", "vendor-best", "vendor-static"),
        required=True,
    )
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument(
        "--cuda-graph",
        choices=("enabled", "disabled"),
        default="enabled",
        help="Expected CUDA Graph state for this server process.",
    )
    parser.add_argument(
        "--placement", choices=("frequency", "uniform"), default="frequency"
    )
    parser.add_argument(
        "--overlap", choices=("enabled", "disabled"), default="enabled"
    )
    parser.add_argument(
        "--mixed-chunk", choices=("enabled", "disabled"), default="enabled"
    )
    parser.add_argument("--command-file", required=True, type=Path)
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--stats-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    command_text = args.command_file.read_text(encoding="utf-8").strip()
    command = shlex.split(command_text)
    log = args.server_log.read_text(encoding="utf-8", errors="replace")
    stats = json.loads(args.stats_manifest.read_text(encoding="utf-8"))
    failures: list[str] = []
    checks: dict[str, Any] = {}

    expected_options = {
        "--kt-method": "FP8",
        "--kt-cpuinfer": str(args.threads),
        "--kt-threadpool-count": "1",
        "--kt-numa-nodes": "0",
        "--kt-expert-placement-strategy": args.placement,
        "--kt-gpu-experts-ratio": stats["placement"]["gpu_experts_ratio_cli"],
        "--kt-max-deferred-experts-per-token": "0" if args.mode == "exact" else "2",
        "--kt-gpu-prefill-token-threshold": "0" if args.mode == "exact" else "1024",
    }
    for option, expected in expected_options.items():
        actual = option_value(command, option)
        checks[f"command:{option}"] = {"expected": expected, "actual": actual}
        if actual != expected:
            failures.append(f"{option}={actual!r}, expected {expected!r}")

    required_flags = {"--disable-radix-cache"}
    forbidden_flags: set[str] = set()
    if args.mixed_chunk == "enabled":
        required_flags.add("--enable-mixed-chunk")
    else:
        forbidden_flags.add("--enable-mixed-chunk")
    if args.overlap == "disabled":
        required_flags.add("--disable-overlap-schedule")
    else:
        forbidden_flags.add("--disable-overlap-schedule")
    if args.cuda_graph == "disabled":
        required_flags.add("--disable-cuda-graph")
    else:
        forbidden_flags.add("--disable-cuda-graph")
    if args.mode == "vendor-best":
        required_flags.add("--kt-enable-dynamic-expert-update")
    else:
        forbidden_flags.add("--kt-enable-dynamic-expert-update")
    for flag in sorted(required_flags):
        present = flag in command
        checks[f"command:{flag}"] = present
        if not present:
            failures.append(f"required flag missing: {flag}")
    for flag in sorted(forbidden_flags):
        present = flag in command
        checks[f"command:not:{flag}"] = not present
        if present:
            failures.append(f"forbidden flag present: {flag}")

    stats_path = option_value(command, "--init-expert-location")
    checks["frequency_stats_path"] = stats_path
    if args.placement == "frequency":
        if stats_path is None:
            failures.append("--init-expert-location is missing")
        elif Path(stats_path).resolve() != Path(stats["normalized_stats"]["path"]).resolve():
            failures.append("--init-expert-location does not match normalized stats manifest")
    elif stats_path is not None:
        failures.append("uniform placement unexpectedly received --init-expert-location")

    expected_server_repr = {
        "disable_cuda_graph": "True" if args.cuda_graph == "disabled" else "False",
        "disable_overlap_schedule": (
            "False" if args.overlap == "enabled" else "True"
        ),
        "enable_mixed_chunk": (
            "True" if args.mixed_chunk == "enabled" else "False"
        ),
        "kt_enable_dynamic_expert_update": (
            "True" if args.mode == "vendor-best" else "False"
        ),
        "kt_max_deferred_experts_per_token": (
            "0" if args.mode == "exact" else "2"
        ),
        "kt_gpu_prefill_token_threshold": (
            "0" if args.mode == "exact" else "1024"
        ),
    }
    for field, expected in expected_server_repr.items():
        pattern = rf"\b{re.escape(field)}={re.escape(expected)}\b"
        found = re.search(pattern, log) is not None
        checks[f"server_args:{field}"] = found
        if not found:
            failures.append(f"server log does not confirm {field}={expected}")

    frequency_loaded = (
        "Loading activation frequency from" in log
        and "Using frequency-based strategy with activation frequency data" in log
        and "WITHOUT activation frequency data" not in log
    )
    checks["server_frequency_stats_loaded_without_fallback"] = frequency_loaded
    if args.placement == "frequency" and not frequency_loaded:
        failures.append("server did not confirm real frequency-stat loading")
    if args.placement == "uniform" and frequency_loaded:
        failures.append("uniform placement unexpectedly loaded frequency stats")

    pairs = re.findall(
        r"KT GPU experts: layer (\d+) \(MoE\) has (\d+) GPU experts", log
    )
    observed = {int(layer): int(count) for layer, count in pairs}
    expected_per_layer = stats["placement"]["hot_experts_per_layer"]
    checks["runtime_mask_layers"] = len(observed)
    checks["runtime_hot_experts"] = sum(observed.values())
    if len(observed) != len(expected_per_layer):
        failures.append(
            f"runtime logged {len(observed)} MoE layers, expected {len(expected_per_layer)}"
        )
    elif args.placement == "frequency":
        actual_per_layer = [observed[index] for index in range(len(expected_per_layer))]
        if actual_per_layer != expected_per_layer:
            failures.append("runtime per-layer GPU expert counts differ from frozen stats")
    else:
        actual_per_layer = [observed[index] for index in range(len(expected_per_layer))]
        if (
            sum(actual_per_layer) != int(stats["placement"]["hot_experts"])
            or max(actual_per_layer) - min(actual_per_layer) > 1
        ):
            failures.append("runtime uniform placement does not preserve the frozen budget")

    layerwise_hits = len(re.findall(r"KT layerwise prefill: layer \d+ compute", log))
    dynamic_hits = len(re.findall(r"expert update =", log))
    checks["layerwise_prefill_log_hits"] = layerwise_hits
    checks["dynamic_expert_update_log_hits"] = dynamic_hits
    if args.mode != "exact":
        if layerwise_hits == 0:
            failures.append(f"{args.mode} benchmark never triggered layerwise prefill")
    if args.mode == "vendor-best":
        if dynamic_hits == 0:
            failures.append("vendor-best benchmark never logged a dynamic expert update")

    result = {
        "schema_version": 1,
        "mode": args.mode,
        "threads": args.threads,
        "cuda_graph": args.cuda_graph,
        "placement": args.placement,
        "overlap": args.overlap,
        "mixed_chunk": args.mixed_chunk,
        "status": "ok" if not failures else "error",
        "checks": checks,
        "failures": failures,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
