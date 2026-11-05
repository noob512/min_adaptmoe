#!/usr/bin/env python3
"""Replay a frozen ShareGPT JSONL trace against an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import aiohttp


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q / 100.0
    lower, upper = int(position), int(position + 1)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def merge_token_ids(current: list[int], ids: list[int], usage_tokens: int | None) -> list[int]:
    # Continuous usage disambiguates delta IDs from cumulative IDs. In
    # particular, a repeated second token must not be mistaken for an
    # unchanged one-token cumulative prefix.
    if usage_tokens is not None and len(ids) == usage_tokens - len(current):
        return current + ids
    if len(ids) >= len(current) and ids[:len(current)] == current:
        return list(ids)
    return current + ids


def load_trace(path: Path, limit: int | None) -> list[dict[str, Any]]:
    # Prompts may contain Unicode line/paragraph separators (U+2028/U+2029).
    # ``str.splitlines()`` treats those as record separators even though the
    # frozen JSONL generator only uses LF between records, corrupting a valid
    # JSON object.  Split on the physical LF delimiter only.
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
    rows = rows[:limit] if limit is not None else rows
    if not rows:
        raise ValueError("trace is empty")
    required = {
        "request_id",
        "prompt_token_ids",
        "prompt_token_ids_sha256",
        "prompt_tokens",
        "max_tokens",
        "arrival_ms",
    }
    for row in rows:
        if not required.issubset(row):
            raise ValueError(f"trace record is missing fields: {sorted(required - row.keys())}")
        token_ids = row["prompt_token_ids"]
        token_hash = sha256_bytes(
            ",".join(str(token) for token in token_ids).encode("ascii")
        )
        if token_hash != row["prompt_token_ids_sha256"]:
            raise ValueError(f"prompt token hash mismatch for {row['request_id']}")
        if row["prompt_tokens"] != len(token_ids):
            raise ValueError(f"prompt length mismatch for {row['request_id']}")
    if any(left["arrival_ms"] > right["arrival_ms"] for left, right in zip(rows, rows[1:])):
        raise ValueError("arrival_ms is not monotonic")
    return rows


async def iter_sse(response: aiohttp.ClientResponse):
    buffer = ""
    data_lines: list[str] = []
    async for chunk in response.content.iter_any():
        buffer += chunk.decode("utf-8")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines = []
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


async def send_one(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    row: dict[str, Any],
    scheduled_at: float,
    timeout_seconds: float,
    seed: int,
    warmup: bool = False,
) -> dict[str, Any]:
    if scheduled_at > time.perf_counter():
        await asyncio.sleep(scheduled_at - time.perf_counter())
    sent = time.perf_counter()
    payload = {
        "model": model,
        "prompt": row["prompt_token_ids"],
        "max_tokens": int(row["max_tokens"] if not warmup else 8),
        "min_tokens": int(row["max_tokens"] if not warmup else 8),
        "ignore_eos": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "return_token_ids": True,
    }
    result: dict[str, Any] = {
        "request_id": row["request_id"],
        "trace_arrival_ms": row["arrival_ms"],
        "scheduled_at_perf": scheduled_at,
        "sent_at_perf": sent,
        "warmup": warmup,
        "pressure_phase": row.get("pressure_phase"),
        "source_request_id": row.get("source_request_id", row["request_id"]),
        "prompt_tokens": row["prompt_tokens"],
        "expected_output_tokens": int(row["max_tokens"] if not warmup else 8),
        "output_token_ids": [],
        "success": False,
    }
    token_times: list[float] = []
    status: int | None = None
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with session.post(
            endpoint,
            data=json.dumps(payload, separators=(",", ":")),
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        ) as response:
            status = response.status
            if status != 200:
                raise RuntimeError(f"HTTP {status}: {(await response.text())[:1000]}")
            usage_tokens: int | None = None
            async for raw in iter_sse(response):
                if raw == "[DONE]":
                    break
                event = json.loads(raw)
                if event.get("error") is not None:
                    raise RuntimeError(str(event["error"]))
                usage = event.get("usage")
                if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                    usage_tokens = int(usage["completion_tokens"])
                choices = event.get("choices") or []
                choice = choices[0] if choices else {}
                ids = choice.get("token_ids") if isinstance(choice, dict) else None
                if isinstance(ids, list) and all(isinstance(value, int) for value in ids):
                    before = len(result["output_token_ids"])
                    result["output_token_ids"] = merge_token_ids(
                        result["output_token_ids"], ids, usage_tokens,
                    )
                    delta = len(result["output_token_ids"]) - before
                    token_times.extend([time.perf_counter()] * delta)
                elif isinstance(choice, dict) and choice.get("text"):
                    token_times.append(time.perf_counter())
            result["usage_completion_tokens"] = usage_tokens
        finished = time.perf_counter()
        result["finished_at_perf"] = finished
        result["first_token_at_perf"] = token_times[0] if token_times else None
        result["token_times_perf"] = token_times
        result["http_status"] = status
        result["completion_tokens"] = int(
            usage_tokens if usage_tokens is not None else len(result["output_token_ids"])
        )
        result["success"] = result["completion_tokens"] == result["expected_output_tokens"]
        result["ttft_ms"] = ((token_times[0] if token_times else finished) - sent) * 1000.0
        result["tpot_ms"] = (
            statistics.fmean((right - left) * 1000.0 for left, right in zip(token_times, token_times[1:]))
            if len(token_times) > 1
            else None
        )
        result["e2e_ms"] = (finished - sent) * 1000.0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["finished_at_perf"] = time.perf_counter()
        result["http_status"] = status
    return result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    all_rows = load_trace(args.trace, None)
    rows = all_rows[: args.limit] if args.limit is not None else all_rows
    if args.arrival_scale <= 0:
        raise ValueError("arrival scale must be positive")
    endpoint = args.base_url.rstrip("/") + "/v1/completions"
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)
    connector = aiohttp.TCPConnector(limit=max(args.max_connections, 1), force_close=False)
    started = time.perf_counter()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Warmup is outside the formal trace window and uses cloned prompts.
        if args.warmup > 0:
            warmup_source = (
                sorted(
                    all_rows,
                    key=lambda row: int(row["prompt_tokens"]),
                    reverse=True,
                )
                if args.warmup_selection == "longest_prompt"
                else rows
            )
            warmup_rows = warmup_source[: args.warmup]
            warmup_tasks = [
                asyncio.create_task(
                    send_one(session, endpoint, args.model, row, time.perf_counter(), args.timeout_seconds, args.seed, True)
                )
                for row in warmup_rows
            ]
            warmup_results = await asyncio.gather(*warmup_tasks)
            if any(not row.get("success") for row in warmup_results):
                raise RuntimeError("Warmup failed; refusing to start formal measurement")
            await asyncio.sleep(args.warmup_settle_seconds)
        trace_start = time.perf_counter()
        trace_start_epoch = time.time()
        progress_path = args.output.with_suffix(".progress.jsonl")
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        async def measured_request(row: dict[str, Any]) -> dict[str, Any]:
            result = await send_one(
                session, endpoint, args.model, row,
                trace_start + float(row["arrival_ms"]) * args.arrival_scale / 1000.0,
                args.timeout_seconds, args.seed,
            )
            # One event-loop thread writes completed records sequentially.
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            return result
        tasks = [
            asyncio.create_task(
                measured_request(row)
            )
            for row in rows
        ]
        results = await asyncio.gather(*tasks)
    finished = time.perf_counter()
    successful = [row for row in results if row.get("success")]
    ttft = [float(row["ttft_ms"]) for row in successful]
    tpot = [float(row["tpot_ms"]) for row in successful if row.get("tpot_ms") is not None]
    e2e = [float(row["e2e_ms"]) for row in successful]
    generated = sum(int(row.get("completion_tokens", 0)) for row in successful)
    trace_window = max(0.001, finished - trace_start)
    return {
        "schema": "eurosys_sharegpt_trace_replay_v1",
        "status": "ok" if len(successful) == len(rows) else "incomplete",
        "trace": str(args.trace.resolve()),
        "trace_sha256": sha256_bytes(args.trace.read_bytes()),
        "model": args.model,
        "base_url": args.base_url,
        "request_count": len(rows),
        "successful_requests": len(successful),
        "failed_requests": len(rows) - len(successful),
        "warmup_requests": args.warmup,
        "warmup_selection": args.warmup_selection,
        "arrival_scale": args.arrival_scale,
        "trace_start_perf": trace_start,
        "trace_start_epoch": trace_start_epoch,
        "max_connections": args.max_connections,
        "measurement_window_seconds": trace_window,
        "offered_qps": len(rows)
        / max(0.001, float(rows[-1]["arrival_ms"]) * args.arrival_scale / 1000.0),
        "qps": len(successful) / trace_window,
        "output_tok_s": generated / trace_window,
        "ttft_ms": distribution(ttft),
        "tpot_ms": distribution(tpot),
        "e2e_ms": distribution(e2e),
        "requests": results,
        "started_at_epoch": time.time() - (finished - started),
        "finished_at_epoch": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument(
        "--warmup-selection",
        choices=("trace_prefix", "longest_prompt"),
        default="trace_prefix",
        help="Select cloned warmup prompts; warmup remains outside the trace window",
    )
    parser.add_argument("--warmup-settle-seconds", type=float, default=2.0)
    parser.add_argument("--max-connections", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument(
        "--arrival-scale",
        type=float,
        default=1.0,
        help="Multiply frozen trace arrival offsets without changing request data",
    )
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "request_count", "successful_requests", "qps", "output_tok_s", "ttft_ms", "tpot_ms", "e2e_ms")}, indent=2))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
