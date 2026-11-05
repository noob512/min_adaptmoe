#!/usr/bin/env python3
"""Verify the formal DeepSeek BF16 KTransformers eager/Graph E0 smoke."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def option_value(command: list[str], option: str) -> str | None:
    positions = [index for index, token in enumerate(command) if token == option]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        return None
    return command[positions[0] + 1]


def graph_passes(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    pattern = re.compile(
        r'^sglang:cuda_graph_passes_total\{([^}]*)\}\s+'
        r'([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)$',
        re.MULTILINE,
    )
    for labels, raw_value in pattern.findall(text):
        match = re.search(r'(?:^|,)mode="([^"]+)"(?:,|$)', labels)
        if match is not None:
            mode = match.group(1)
            result[mode] = result.get(mode, 0.0) + float(raw_value)
    return result


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--placement-manifest", required=True, type=Path)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    checks: dict[str, Any] = {}
    placement = load_json(args.placement_manifest)
    tier = placement["tiers"]["4g"]
    expected_frequency = Path(tier["frequency_path"]).resolve()
    expected_gpu_experts = str(tier["gpu_experts_per_layer"])
    output_signatures: dict[str, list[list[Any]]] = {}
    mode_results: dict[str, Any] = {}

    for mode in ("eager", "graph"):
        root = args.run_root / mode
        required = {
            "command": root / "server_command.txt",
            "log": root / "server.log",
            "client": root / "client.json",
            "metrics": root / "metrics.after.prom",
            "server_rc": root / "server_returncode.txt",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            failures.append(f"{mode}: missing artifacts: {missing}")
            continue

        command = shlex.split(required["command"].read_text(encoding="utf-8"))
        log = required["log"].read_text(encoding="utf-8", errors="replace")
        client = load_json(required["client"])
        metrics_text = required["metrics"].read_text(
            encoding="utf-8", errors="replace"
        )
        counters = graph_passes(metrics_text)

        expected_options = {
            "--kt-method": "BF16",
            "--kt-cpuinfer": "14",
            "--kt-threadpool-count": "1",
            "--kt-numa-nodes": "0",
            "--kt-num-gpu-experts": expected_gpu_experts,
            "--kt-expert-placement-strategy": "frequency",
            "--kt-max-deferred-experts-per-token": "0",
            "--kt-gpu-prefill-token-threshold": "0",
            "--cuda-graph-max-bs": "1",
        }
        for option, expected in expected_options.items():
            actual = option_value(command, option)
            checks[f"{mode}:command:{option}"] = actual
            if actual != expected:
                failures.append(
                    f"{mode}: {option}={actual!r}, expected {expected!r}"
                )
        frequency = option_value(command, "--init-expert-location")
        checks[f"{mode}:frequency_path"] = frequency
        if frequency is None or Path(frequency).resolve() != expected_frequency:
            failures.append(f"{mode}: frozen frequency placement path differs")
        for flag in ("--enable-mixed-chunk", "--disable-radix-cache", "--enable-metrics"):
            present = flag in command
            checks[f"{mode}:command:{flag}"] = present
            if not present:
                failures.append(f"{mode}: required flag is absent: {flag}")
        graph_disabled = "--disable-cuda-graph" in command
        checks[f"{mode}:graph_disabled"] = graph_disabled
        if graph_disabled != (mode == "eager"):
            failures.append(f"{mode}: CUDA Graph command state differs")

        frequency_loaded = (
            "Loading activation frequency from" in log
            and "Using frequency-based strategy with activation frequency data" in log
            and "WITHOUT activation frequency data" not in log
        )
        checks[f"{mode}:frequency_loaded"] = frequency_loaded
        if not frequency_loaded:
            failures.append(f"{mode}: server did not load frozen frequency data")
        pairs = re.findall(
            r"KT GPU experts: layer (\d+) \(MoE\) has (\d+) GPU experts", log
        )
        observed = {int(layer): int(count) for layer, count in pairs}
        checks[f"{mode}:placement_layers"] = len(observed)
        checks[f"{mode}:placement_counts"] = sorted(set(observed.values()))
        if len(observed) != 26 or set(observed.values()) != {int(expected_gpu_experts)}:
            failures.append(f"{mode}: runtime KT placement is not 26 x {expected_gpu_experts}")

        capture_begin = "Capture cuda graph begin" in log
        capture_end = "Capture cuda graph end" in log
        checks[f"{mode}:capture_begin"] = capture_begin
        checks[f"{mode}:capture_end"] = capture_end
        if mode == "graph" and not (capture_begin and capture_end):
            failures.append("graph: CUDA Graph capture certificate is incomplete")
        if mode == "eager" and (capture_begin or capture_end):
            failures.append("eager: CUDA Graph was captured despite eager mode")
        fatal_patterns = (
            r"Traceback \(most recent call last\)",
            r"CUDA out of memory",
            r"OutOfMemoryError",
            r"illegal memory access",
            r"CUDA error:",
        )
        fatal_hits = [pattern for pattern in fatal_patterns if re.search(pattern, log)]
        checks[f"{mode}:fatal_log_patterns"] = fatal_hits
        if fatal_hits:
            failures.append(f"{mode}: fatal server log patterns: {fatal_hits}")

        if client.get("status") != "ok":
            failures.append(f"{mode}: client status={client.get('status')!r}")
            continue
        if client.get("config", {}).get("system") != "ktransformers":
            failures.append(f"{mode}: client system label differs")
        fixture_meta = client.get("fixture", {}).get("metadata", {})
        if (
            fixture_meta.get("fixture_role") != "validation"
            or fixture_meta.get("workload") != "short_dialogue"
            or fixture_meta.get("prompt_tokens") != 128
            or fixture_meta.get("output_tokens") != 32
            or fixture_meta.get("protocol_id") != args.protocol_id
        ):
            failures.append(f"{mode}: client fixture metadata differs")
        warmups = client.get("warmups", [])
        if len(warmups) != 3 or any(
            not item.get("success") or item.get("completion_tokens") != 8
            for item in warmups
        ):
            failures.append(f"{mode}: three exact warmups did not complete")
        phases = client.get("phases", [])
        if len(phases) != 1:
            failures.append(f"{mode}: expected one measured phase")
            continue
        phase = phases[0]
        if (
            phase.get("concurrency") != 1
            or phase.get("request_count") != 10
            or phase.get("successful_requests") != 10
            or phase.get("failed_requests") != 0
            or phase.get("request_rate_limit_compliant") is not True
        ):
            failures.append(f"{mode}: measured C1 phase did not pass")
            continue
        requests = phase.get("requests", [])
        signature: list[list[Any]] = []
        for index, request in enumerate(requests):
            token_ids = request.get("output_token_ids")
            if (
                not request.get("success")
                or request.get("prompt_tokens") != 128
                or request.get("completion_tokens") != 32
                or not request.get("saw_done")
                or request.get("connection_reused") is not True
                or request.get("connection_created_during_measurement") is not False
                or request.get("fallback_token_events") != 0
                or not isinstance(token_ids, list)
                or len(token_ids) != 32
                or not isinstance(request.get("output_token_ids_sha256"), str)
            ):
                failures.append(
                    f"{mode}: measured request {index} protocol/token gate failed"
                )
            else:
                signature.append(
                    [request.get("fixture_index"), request["output_token_ids_sha256"]]
                )
        output_signatures[mode] = signature

        expected_counter = "decode_none" if mode == "eager" else "decode_cuda_graph"
        checks[f"{mode}:cuda_graph_passes"] = counters
        if counters.get(expected_counter, 0.0) <= 0.0:
            failures.append(
                f"{mode}: metric {expected_counter} has no observed forward passes"
            )
        mode_results[mode] = {
            "client_result": str(required["client"].resolve()),
            "metrics": str(required["metrics"].resolve()),
            "cuda_graph_passes": counters,
            "output_signature": output_signatures.get(mode),
            "measurement_requests": phase.get("request_count"),
            "mean_ttft_ms": phase.get("ttft_ms", {}).get("mean"),
            "mean_tpot_ms": phase.get("tpot_ms", {}).get("mean"),
            "e2e_ms": phase.get("e2e_ms", {}).get("mean"),
        }

    hashes_equal = (
        len(output_signatures.get("eager", [])) == 10
        and output_signatures.get("eager") == output_signatures.get("graph")
    )
    checks["eager_graph_output_hash_equal"] = hashes_equal
    if not hashes_equal:
        failures.append("eager/Graph output token hashes differ")

    result = {
        "schema_version": 1,
        "experiment": "ktransformers_deepseek_bf16_e0_qualification",
        "status": "ok" if not failures else "error",
        "scope": "qualification_only_not_formal_performance",
        "protocol_id": args.protocol_id,
        "placement_manifest": str(args.placement_manifest.resolve()),
        "placement_tier": "4g",
        "modes": mode_results,
        "checks": checks,
        "failures": failures,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
