#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from statistics import mean
from typing import Any


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return ordered[index]


def load_fixture(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload.get("requests", [])
    if not isinstance(requests, list):
        raise ValueError(f"fixture requests must be a list: {path}")
    return requests


def parse_csv(value: str, cast=str) -> list[Any]:
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MoE-Infinity on the current ShareGPT fixture.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--offload-dir", required=True, type=Path)
    parser.add_argument("--device-memory-ratio", type=float, default=0.75)
    parser.add_argument("--phase-names", default="final")
    parser.add_argument("--phase-request-counts", default="256")
    parser.add_argument("--phase-output-tokens", default="64")
    parser.add_argument("--use-native-engine", action="store_true")
    parser.add_argument("--prefetch", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    args.offload_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "source": "moe_infinity_phased_benchmark",
        "model": args.model,
        "fixture": str(args.fixture),
        "offload_dir": str(args.offload_dir),
        "device_memory_ratio": args.device_memory_ratio,
        "use_native_engine": args.use_native_engine,
        "prefetch": args.prefetch,
        "phases": [],
        "status": "initializing",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def progress(record: dict[str, Any]) -> None:
        with args.progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": time.time(), **record}, sort_keys=True) + "\n")

    try:
        import torch
        from transformers import AutoTokenizer
        from moe_infinity import MoE

        requests = load_fixture(args.fixture)
        phase_names = parse_csv(args.phase_names, str)
        phase_counts = parse_csv(args.phase_request_counts, int)
        phase_outputs = parse_csv(args.phase_output_tokens, int)
        if not len(phase_names) == len(phase_counts) == len(phase_outputs):
            raise ValueError("phase names/counts/output tokens must align")
        if any(count > len(requests) for count in phase_counts):
            raise ValueError("phase request count exceeds fixture size")

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_start = time.perf_counter()
        config = {
            "offload_path": str(args.offload_dir),
            "device_memory_ratio": args.device_memory_ratio,
            "use_native_engine": args.use_native_engine,
            "prefetch": args.prefetch,
        }
        model = MoE(args.model, config)
        load_seconds = time.perf_counter() - load_start
        result["load_seconds"] = load_seconds
        result["status"] = "running"
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        for phase_name, count, output_tokens in zip(phase_names, phase_counts, phase_outputs):
            selected = requests[:count]
            progress({"action": "phase_begin", "phase": phase_name, "count": count, "output_tokens_per_request": output_tokens})
            phase_start = time.perf_counter()
            records: list[dict[str, Any]] = []
            for index, request in enumerate(selected):
                prompt = str(request["prompt"])
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda:0")
                request_start = time.perf_counter()
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=output_tokens,
                        min_new_tokens=output_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                elapsed = time.perf_counter() - request_start
                generated = int(output_ids.shape[-1] - input_ids.shape[-1])
                records.append({
                    "request": int(request.get("request", index)),
                    "prompt_tokens": int(input_ids.shape[-1]),
                    "output_tokens": generated,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": generated / elapsed if elapsed else None,
                })
                if (index + 1) % 8 == 0 or index + 1 == len(selected):
                    progress({"action": "request_progress", "phase": phase_name, "completed": index + 1, "count": count})
            phase_elapsed = time.perf_counter() - phase_start
            output_count = sum(int(record["output_tokens"]) for record in records)
            per_request_tps = [float(record["tokens_per_second"]) for record in records if record.get("tokens_per_second")]
            phase_result = {
                "phase": phase_name,
                "request_count": count,
                "output_tokens_per_request": output_tokens,
                "elapsed_seconds": phase_elapsed,
                "output_tokens": output_count,
                "output_tokens_per_second": output_count / phase_elapsed if phase_elapsed else None,
                "mean_request_tokens_per_second": mean(per_request_tps) if per_request_tps else None,
                "p50_request_tokens_per_second": percentile(per_request_tps, 0.50),
                "p99_request_tokens_per_second": percentile(per_request_tps, 0.99),
                "requests": records,
            }
            result["phases"].append(phase_result)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            progress({"action": "phase_end", **{key: value for key, value in phase_result.items() if key != "requests"}})

        result["status"] = "ok"
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        result["status"] = "error"
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        progress({"action": "error", "error_type": type(exc).__name__, "error": str(exc)})
        print(result["traceback"])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
