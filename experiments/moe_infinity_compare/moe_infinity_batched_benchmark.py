#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any


def load_fixture(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload.get("requests", [])
    if not isinstance(requests, list):
        raise ValueError(f"fixture requests must be a list: {path}")
    return requests


def gpu_used_mib() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(completed.stdout.splitlines()[0].strip())
    except (IndexError, OSError, subprocess.SubprocessError, ValueError):
        return None


class GPUMemoryMonitor:
    def __init__(self, interval_seconds: float = 0.1) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, float | int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        while not self._stop.is_set():
            used_mib = gpu_used_mib()
            if used_mib is not None:
                self.samples.append({"ts": time.time(), "used_mib": used_mib})
            self._stop.wait(self.interval_seconds)

    @property
    def peak_used_mib(self) -> int | None:
        if not self.samples:
            return None
        return max(int(sample["used_mib"]) for sample in self.samples)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run MoE-Infinity with real batched HuggingFace generation."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--offload-dir", required=True, type=Path)
    parser.add_argument("--device-memory-ratio", type=float, required=True)
    parser.add_argument("--request-count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--prefetch", action="store_true")
    args = parser.parse_args()

    if args.request_count <= 0:
        raise ValueError("--request-count must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.output_tokens <= 0:
        raise ValueError("--output-tokens must be positive")
    if not 0.0 < args.device_memory_ratio <= 1.0:
        raise ValueError("--device-memory-ratio must be in (0, 1]")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.progress.parent.mkdir(parents=True, exist_ok=True)
    args.offload_dir.mkdir(parents=True, exist_ok=True)
    args.progress.write_text("", encoding="utf-8")

    result: dict[str, Any] = {
        "source": "moe_infinity_batched_benchmark",
        "model": args.model,
        "fixture": str(args.fixture),
        "offload_dir": str(args.offload_dir),
        "device_memory_ratio": args.device_memory_ratio,
        "request_count": args.request_count,
        "batch_size": args.batch_size,
        "output_tokens_per_request": args.output_tokens,
        "prefetch": args.prefetch,
        "status": "initializing",
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def progress(record: dict[str, Any]) -> None:
        with args.progress.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"ts": time.time(), **record}, sort_keys=True) + "\n")

    monitor = GPUMemoryMonitor()
    monitor.start()
    try:
        import torch
        from transformers import AutoTokenizer
        from moe_infinity import MoE

        requests = load_fixture(args.fixture)
        if args.request_count > len(requests):
            raise ValueError("request count exceeds fixture size")
        selected = requests[: args.request_count]

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, trust_remote_code=True, use_fast=False
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        load_started = time.perf_counter()
        model = MoE(
            args.model,
            {
                "offload_path": str(args.offload_dir),
                "device_memory_ratio": args.device_memory_ratio,
                "use_native_engine": False,
                "prefetch": args.prefetch,
            },
        )
        load_seconds = time.perf_counter() - load_started
        result["load_seconds"] = load_seconds
        result["gpu_after_load_mib"] = gpu_used_mib()
        result["status"] = "running"
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        total_output_tokens = 0
        batch_records: list[dict[str, Any]] = []
        generation_started = time.perf_counter()
        for batch_index, start in enumerate(
            range(0, args.request_count, args.batch_size)
        ):
            batch_requests = selected[start : start + args.batch_size]
            prompts = [str(request["prompt"]) for request in batch_requests]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to("cuda:0")
            attention_mask = encoded["attention_mask"].to("cuda:0")
            prompt_lengths = [int(value) for value in attention_mask.sum(dim=1).tolist()]

            progress(
                {
                    "action": "batch_begin",
                    "batch_index": batch_index,
                    "batch_size": len(batch_requests),
                    "min_prompt_tokens": min(prompt_lengths),
                    "max_prompt_tokens": max(prompt_lengths),
                }
            )
            torch.cuda.synchronize()
            batch_started = time.perf_counter()
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.output_tokens,
                    min_new_tokens=args.output_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            batch_elapsed = time.perf_counter() - batch_started
            generated_per_request = int(output_ids.shape[-1] - input_ids.shape[-1])
            batch_output_tokens = generated_per_request * len(batch_requests)
            if generated_per_request != args.output_tokens:
                raise RuntimeError(
                    f"expected {args.output_tokens} generated tokens per request, "
                    f"got {generated_per_request}"
                )
            total_output_tokens += batch_output_tokens
            batch_record = {
                "batch_index": batch_index,
                "batch_size": len(batch_requests),
                "elapsed_seconds": batch_elapsed,
                "output_tokens": batch_output_tokens,
                "output_tokens_per_second": batch_output_tokens / batch_elapsed,
                "gpu_after_batch_mib": gpu_used_mib(),
            }
            batch_records.append(batch_record)
            progress({"action": "batch_end", **batch_record})

        wall_seconds = time.perf_counter() - generation_started
        generation_seconds = sum(
            float(batch["elapsed_seconds"]) for batch in batch_records
        )
        result.update(
            {
                "status": "ok",
                "generation_seconds": generation_seconds,
                "wall_seconds": wall_seconds,
                "output_tokens": total_output_tokens,
                "output_tokens_per_second": total_output_tokens
                / generation_seconds,
                "gpu_after_generation_mib": gpu_used_mib(),
                "batches": batch_records,
            }
        )
    except BaseException as exc:
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return_code = 1
    else:
        return_code = 0
    finally:
        monitor.stop()
        result["peak_gpu_used_mib"] = monitor.peak_used_mib
        result["gpu_memory_samples"] = monitor.samples
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
