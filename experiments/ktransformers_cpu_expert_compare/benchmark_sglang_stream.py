#!/usr/bin/env python3
"""Run deterministic, burst-style benchmarks against SGLang's /generate API.

The client deliberately reads the first N prompts from an existing fixture without
sampling or shuffling.  Token timing comes from the cumulative
``meta_info.completion_tokens`` field in SGLang SSE events, not from decoded text
chunks (one token does not necessarily produce text immediately).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Iterable


SAMPLING_PARAMS = {
    "temperature": 0.0,
    "ignore_eos": True,
    "sampling_seed": 0,
    "stream_interval": 1,
}


def parse_positive_ints(raw: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid comma-separated integer {part!r}"
            ) from exc
        if value <= 0:
            raise argparse.ArgumentTypeError("concurrencies must be positive")
        if value in seen:
            raise argparse.ArgumentTypeError(f"duplicate concurrency: {value}")
        values.append(value)
        seen.add(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one concurrency is required")
    return values


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(payload)


def percentile(values: Iterable[float], pct: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def mean(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else None


def format_metric(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else "null"


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_fixture(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_bytes = path.read_bytes()
    try:
        payload = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON fixture {path}: {exc}") from exc

    requests = payload.get("requests") if isinstance(payload, dict) else payload
    if not isinstance(requests, list) or not requests:
        raise ValueError("fixture must be a non-empty list or contain a requests list")

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(requests):
        if not isinstance(item, dict):
            raise ValueError(f"fixture request {index} is not an object")
        prompt_token_ids = item.get("prompt_token_ids")
        if prompt_token_ids is not None and (
            not isinstance(prompt_token_ids, list)
            or not prompt_token_ids
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int)
                for token_id in prompt_token_ids
            )
        ):
            raise ValueError(
                f"fixture request {index} has invalid prompt_token_ids"
            )
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or (not prompt and prompt_token_ids is None):
            raise ValueError(
                f"fixture request {index} has neither text nor prompt_token_ids"
            )
        declared_prompt_tokens = item.get("prompt_tokens")
        if prompt_token_ids is not None and len(prompt_token_ids) != declared_prompt_tokens:
            raise ValueError(
                f"fixture request {index} prompt_token_ids length does not "
                "match prompt_tokens"
            )
        normalized.append(
            {
                "fixture_index": index,
                "fixture_request_id": item.get("request", index),
                "prompt": prompt,
                "prompt_token_ids": prompt_token_ids,
                "prompt_token_ids_sha256": (
                    sha256_json(prompt_token_ids)
                    if prompt_token_ids is not None
                    else None
                ),
                "declared_prompt_tokens": declared_prompt_tokens,
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            }
        )

    metadata = {
        "path": str(path.resolve()),
        "sha256": sha256_bytes(raw_bytes),
        "request_count": len(normalized),
        "prompt_list_sha256": sha256_json(
            [request["prompt"] for request in normalized]
        ),
        "prompt_token_ids_list_sha256": (
            sha256_json([request["prompt_token_ids"] for request in normalized])
            if all(request["prompt_token_ids"] is not None for request in normalized)
            else None
        ),
        "all_requests_have_prompt_token_ids": all(
            request["prompt_token_ids"] is not None for request in normalized
        ),
        "embedded_metadata": payload.get("metadata")
        if isinstance(payload, dict)
        else None,
    }
    return normalized, metadata


class BurstGate:
    """A one-shot async gate that releases all request tasks together."""

    def __init__(self, parties: int) -> None:
        self.parties = parties
        self.ready = 0
        self._condition = asyncio.Condition()
        self._released = asyncio.Event()

    async def arrive_and_wait(self) -> None:
        async with self._condition:
            self.ready += 1
            self._condition.notify_all()
        await self._released.wait()

    async def wait_until_ready(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self.ready == self.parties)

    def release(self) -> None:
        self._released.set()


async def iter_sse_data(content: Any) -> AsyncIterator[str]:
    """Parse SSE records without relying on network chunk boundaries."""

    data_lines: list[str] = []
    async for raw_line in content:
        try:
            line = raw_line.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"SSE response is not UTF-8: {exc}") from exc
        if line == "":
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue
        # event/id/retry are legal SSE fields and do not carry response data.
        if line.startswith(("event:", "id:", "retry:")):
            continue
        raise RuntimeError(f"unexpected SSE line: {line[:200]!r}")
    if data_lines:
        yield "\n".join(data_lines)


def merge_output_ids(
    reconstructed: list[int], raw_ids: Any, previous_count: int, count: int
) -> tuple[list[int], str | None]:
    """Handle both cumulative and disjoint SGLang output_ids streams."""

    if raw_ids is None:
        return reconstructed, None
    if not isinstance(raw_ids, list) or any(
        not isinstance(token_id, int) for token_id in raw_ids
    ):
        return reconstructed, "output_ids is not a list of integers"

    if len(raw_ids) == count:
        return list(raw_ids), None
    delta = count - previous_count
    if delta >= 0 and len(raw_ids) == delta and len(reconstructed) == previous_count:
        return reconstructed + raw_ids, None
    if len(raw_ids) >= len(reconstructed) and raw_ids[: len(reconstructed)] == reconstructed:
        return list(raw_ids), None
    return reconstructed, (
        "cannot reconstruct output_ids: "
        f"previous_count={previous_count} count={count} "
        f"reconstructed={len(reconstructed)} event_ids={len(raw_ids)}"
    )


@dataclass
class RequestContext:
    fixture_index: int
    fixture_request_id: Any
    prompt: str
    prompt_token_ids: list[int] | None
    prompt_token_ids_sha256: str | None
    prompt_sha256: str
    declared_prompt_tokens: Any


async def run_one_request(
    *,
    session: Any,
    endpoint: str,
    request: RequestContext,
    output_tokens: int,
    gate: BurstGate,
    phase_started: float,
) -> dict[str, Any]:
    await gate.arrive_and_wait()
    submitted_at = time.perf_counter()
    input_payload: dict[str, Any]
    if request.prompt_token_ids is not None:
        input_payload = {"input_ids": request.prompt_token_ids}
    else:
        input_payload = {"text": request.prompt}
    payload = {
        **input_payload,
        "sampling_params": {
            **SAMPLING_PARAMS,
            "max_new_tokens": output_tokens,
        },
        "stream": True,
    }

    first_token_at: float | None = None
    last_token_at: float | None = None
    finished_at: float | None = None
    completion_tokens = 0
    prompt_tokens: int | None = None
    text = ""
    output_ids: list[int] = []
    output_ids_warnings: list[str] = []
    finish_reason: Any = None
    saw_done = False
    sse_events = 0
    token_events = 0
    grouped_token_events = 0
    http_status: int | None = None
    error: str | None = None

    try:
        async with session.post(endpoint, json=payload) as response:
            http_status = response.status
            if response.status != 200:
                body = (await response.text())[:4096]
                raise RuntimeError(f"HTTP {response.status}: {body}")

            async for raw_data in iter_sse_data(response.content):
                received_at = time.perf_counter()
                if raw_data == "[DONE]":
                    saw_done = True
                    break
                sse_events += 1
                try:
                    event = json.loads(raw_data)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"invalid SSE JSON event {raw_data[:200]!r}: {exc}"
                    ) from exc
                if not isinstance(event, dict):
                    raise RuntimeError("SSE JSON event is not an object")
                if event.get("error") is not None:
                    raise RuntimeError(f"server error event: {event['error']!r}")

                event_text = event.get("text")
                if event_text is not None:
                    if not isinstance(event_text, str):
                        raise RuntimeError("event text is not a string")
                    text = event_text

                meta = event.get("meta_info")
                if not isinstance(meta, dict):
                    raise RuntimeError("SSE event has no meta_info object")
                raw_count = meta.get("completion_tokens")
                if isinstance(raw_count, bool) or not isinstance(raw_count, int):
                    raise RuntimeError(
                        f"invalid meta_info.completion_tokens: {raw_count!r}"
                    )
                if raw_count < completion_tokens:
                    raise RuntimeError(
                        "completion_tokens decreased from "
                        f"{completion_tokens} to {raw_count}"
                    )
                if raw_count > output_tokens:
                    raise RuntimeError(
                        f"server produced {raw_count} tokens, expected at most {output_tokens}"
                    )

                previous_count = completion_tokens
                completion_tokens = raw_count
                if completion_tokens > previous_count:
                    delta = completion_tokens - previous_count
                    token_events += 1
                    grouped_token_events += int(delta > 1)
                    if first_token_at is None:
                        first_token_at = received_at
                    last_token_at = received_at

                output_ids, ids_warning = merge_output_ids(
                    output_ids,
                    event.get("output_ids"),
                    previous_count,
                    completion_tokens,
                )
                if ids_warning and ids_warning not in output_ids_warnings:
                    output_ids_warnings.append(ids_warning)

                raw_prompt_tokens = meta.get("prompt_tokens")
                if isinstance(raw_prompt_tokens, int) and not isinstance(
                    raw_prompt_tokens, bool
                ):
                    prompt_tokens = raw_prompt_tokens
                if meta.get("finish_reason") is not None:
                    finish_reason = meta["finish_reason"]

        if not saw_done:
            raise RuntimeError("stream ended without data: [DONE]")
        if completion_tokens != output_tokens:
            raise RuntimeError(
                f"completed {completion_tokens} output tokens; expected {output_tokens}"
            )
        if request.prompt_token_ids is not None and prompt_tokens != len(
            request.prompt_token_ids
        ):
            raise RuntimeError(
                "server prompt token count does not match submitted input_ids: "
                f"server={prompt_tokens!r} submitted={len(request.prompt_token_ids)}"
            )
        if first_token_at is None or last_token_at is None:
            raise RuntimeError("stream completed without a token-bearing SSE event")
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # Return partial measurements for diagnosis.
        error = f"{type(exc).__name__}: {exc}"
    finally:
        finished_at = time.perf_counter()

    ids_complete = len(output_ids) == completion_tokens and completion_tokens > 0
    output_token_ids = output_ids if ids_complete else None
    output_ids_sha256 = sha256_json(output_ids) if ids_complete else None
    text_sha256 = sha256_bytes(text.encode("utf-8"))
    fingerprint_kind = "output_token_ids" if ids_complete else "text_utf8"
    fingerprint_sha256 = output_ids_sha256 or text_sha256
    ttft_ms = (
        (first_token_at - submitted_at) * 1000.0
        if first_token_at is not None
        else None
    )
    tpot_ms = (
        (last_token_at - first_token_at) * 1000.0 / (completion_tokens - 1)
        if first_token_at is not None
        and last_token_at is not None
        and completion_tokens > 1
        else None
    )

    return {
        "fixture_index": request.fixture_index,
        "fixture_request_id": request.fixture_request_id,
        "prompt_sha256": request.prompt_sha256,
        "submitted_input_kind": (
            "input_ids" if request.prompt_token_ids is not None else "text"
        ),
        "submitted_prompt_token_ids_sha256": request.prompt_token_ids_sha256,
        "submitted_prompt_token_ids_count": (
            len(request.prompt_token_ids)
            if request.prompt_token_ids is not None
            else None
        ),
        "server_prompt_token_count_matches_submission": (
            prompt_tokens == len(request.prompt_token_ids)
            if request.prompt_token_ids is not None and prompt_tokens is not None
            else None
        ),
        "declared_prompt_tokens": request.declared_prompt_tokens,
        "server_prompt_tokens": prompt_tokens,
        "success": error is None,
        "error": error,
        "http_status": http_status,
        "submitted_after_phase_start_ms": (submitted_at - phase_started) * 1000.0,
        "latency_ms": (finished_at - submitted_at) * 1000.0,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "saw_done": saw_done,
        "sse_events": sse_events,
        "token_events": token_events,
        "grouped_token_events": grouped_token_events,
        "output_token_ids": output_token_ids,
        "output_token_ids_sha256": output_ids_sha256,
        "output_ids_warnings": output_ids_warnings,
        "output_text_bytes": len(text.encode("utf-8")),
        "output_text_sha256": text_sha256,
        "output_fingerprint_kind": fingerprint_kind,
        "output_fingerprint_sha256": fingerprint_sha256,
    }


async def flush_server_cache(session: Any, base_url: str) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/flush_cache"
    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    # SGLang may emit the final SSE event just before its scheduler removes the
    # request.  A flush sent in that small window returns HTTP 400 (pending
    # requests) even though the measured request is complete.  Retry only this
    # synchronization operation; no inference request is retried.
    for attempt in range(1, 26):
        try:
            async with session.post(url, json={}) as response:
                body = (await response.text())[:4096]
                attempts.append(
                    {"attempt": attempt, "http_status": response.status, "response": body}
                )
                if response.status == 200:
                    return {
                        "ok": True,
                        "url": url,
                        "attempt_count": attempt,
                        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                        "response": body,
                        "attempts": attempts,
                    }
        except Exception as exc:
            attempts.append(
                {"attempt": attempt, "error": f"{type(exc).__name__}: {exc}"}
            )
        if attempt < 25:
            await asyncio.sleep(0.2)
    return {
        "ok": False,
        "url": url,
        "attempt_count": len(attempts),
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "error": "cache remained busy after bounded retries",
        "attempts": attempts,
    }


async def run_phase(
    *,
    session: Any,
    endpoint: str,
    fixture_requests: list[dict[str, Any]],
    concurrency: int,
    output_tokens: int,
    barrier_timeout: float,
) -> dict[str, Any]:
    selected = fixture_requests[:concurrency]
    gate = BurstGate(concurrency)
    # Tasks cannot pass the gate until all N have reached it.
    phase_started_placeholder = time.perf_counter()
    tasks = [
        asyncio.create_task(
            run_one_request(
                session=session,
                endpoint=endpoint,
                request=RequestContext(**item),
                output_tokens=output_tokens,
                gate=gate,
                phase_started=phase_started_placeholder,
            )
        )
        for item in selected
    ]

    try:
        await asyncio.wait_for(gate.wait_until_ready(), timeout=barrier_timeout)
    except Exception:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    phase_started_epoch = time.time()
    phase_started = time.perf_counter()
    # run_one_request received the placeholder only for its diagnostic skew. Its
    # own latency and TTFT use the actual post-gate submission timestamp.
    gate.release()
    records = await asyncio.gather(*tasks)
    phase_finished = time.perf_counter()
    elapsed_seconds = phase_finished - phase_started

    # Normalize the diagnostic offset to the real gate release. The placeholder
    # was captured immediately before task creation, so subtract that setup gap.
    setup_gap_ms = (phase_started - phase_started_placeholder) * 1000.0
    for record in records:
        record["submitted_after_phase_start_ms"] -= setup_gap_ms

    successful = [record for record in records if record["success"]]
    completed_tokens = sum(record["completion_tokens"] for record in successful)
    ttfts = [record["ttft_ms"] for record in successful if record["ttft_ms"] is not None]
    tpots = [record["tpot_ms"] for record in successful if record["tpot_ms"] is not None]
    submission_offsets = [record["submitted_after_phase_start_ms"] for record in records]
    output_fingerprints = [
        {
            "fixture_index": record["fixture_index"],
            "kind": record["output_fingerprint_kind"],
            "sha256": record["output_fingerprint_sha256"],
        }
        for record in records
    ]

    return {
        "phase": f"c{concurrency}",
        "concurrency": concurrency,
        "fixture_indices": list(range(concurrency)),
        "prompt_list_sha256": sha256_json(
            [item["prompt"] for item in selected]
        ),
        "prompt_token_ids_list_sha256": (
            sha256_json([item["prompt_token_ids"] for item in selected])
            if all(item["prompt_token_ids"] is not None for item in selected)
            else None
        ),
        "phase_started_epoch_s": phase_started_epoch,
        "elapsed_seconds": elapsed_seconds,
        "request_count": concurrency,
        "completed_requests": len(successful),
        "failed_requests": concurrency - len(successful),
        "expected_output_tokens": concurrency * output_tokens,
        "completed_output_tokens": completed_tokens,
        "all_requests_produced_fixed_output": len(successful) == concurrency
        and completed_tokens == concurrency * output_tokens,
        "output_tokens_per_second": completed_tokens / elapsed_seconds
        if elapsed_seconds > 0
        else None,
        "mean_ttft_ms": mean(ttfts),
        "p50_ttft_ms": percentile(ttfts, 50.0),
        "p99_ttft_ms": percentile(ttfts, 99.0),
        "mean_tpot_ms": mean(tpots),
        "p50_tpot_ms": percentile(tpots, 50.0),
        "p99_tpot_ms": percentile(tpots, 99.0),
        "barrier_ready_count": gate.ready,
        "request_submit_skew_ms": max(submission_offsets) - min(submission_offsets)
        if submission_offsets
        else None,
        "output_fingerprints_sha256": sha256_json(output_fingerprints),
        "requests": records,
    }


async def run_benchmark(
    args: argparse.Namespace,
    fixture_requests: list[dict[str, Any]],
    result: dict[str, Any],
) -> None:
    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError(
            "aiohttp is required; run this client in the SGLang/KTransformers environment"
        ) from exc

    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for concurrency in args.concurrencies:
            cache_flush = None
            if args.flush_cache_before_phase:
                cache_flush = await flush_server_cache(session, args.base_url)
                if not cache_flush["ok"]:
                    raise RuntimeError(
                        f"cache flush before c{concurrency} failed: {cache_flush['error']}"
                    )
            if args.phase_settle_seconds:
                await asyncio.sleep(args.phase_settle_seconds)

            phase = await run_phase(
                session=session,
                endpoint=args.endpoint,
                fixture_requests=fixture_requests,
                concurrency=concurrency,
                output_tokens=args.output_tokens,
                barrier_timeout=args.barrier_timeout,
            )
            phase["cache_flush_before_phase"] = cache_flush
            result["phases"].append(phase)
            atomic_write_json(args.output, result)
            print(
                f"phase=c{concurrency} completed={phase['completed_requests']}/"
                f"{concurrency} output_tok_s="
                f"{format_metric(phase['output_tokens_per_second'])} "
                f"mean_ttft_ms={format_metric(phase['mean_ttft_ms'])} "
                f"mean_tpot_ms={format_metric(phase['mean_tpot_ms'])}",
                flush=True,
            )
            if phase["failed_requests"]:
                raise RuntimeError(f"c{concurrency} has failed requests")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SGLang /generate streaming with fixed fixture prefixes and "
            "simultaneous burst submission."
        )
    )
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:30000", help="SGLang server root"
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Native generate endpoint (default: BASE_URL/generate)",
    )
    parser.add_argument(
        "--concurrencies",
        type=parse_positive_ints,
        default=parse_positive_ints("1,2,3,4,8,16,24,32"),
    )
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument(
        "--request-timeout", type=float, default=3600.0, help="Per-request seconds"
    )
    parser.add_argument(
        "--barrier-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for all client tasks to reach the burst gate",
    )
    parser.add_argument(
        "--flush-cache-before-phase",
        action="store_true",
        help="POST /flush_cache before every measured phase",
    )
    parser.add_argument("--phase-settle-seconds", type=float, default=0.0)
    parser.add_argument(
        "--require-prompt-token-ids",
        action="store_true",
        help="Reject text-only fixtures and submit native /generate input_ids.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    args.endpoint = args.endpoint or args.base_url + "/generate"

    if args.output_tokens < 2:
        parser.error("--output-tokens must be at least 2 to measure TPOT")
    if args.request_timeout <= 0 or args.barrier_timeout <= 0:
        parser.error("timeouts must be positive")
    if args.phase_settle_seconds < 0:
        parser.error("--phase-settle-seconds must be non-negative")

    try:
        fixture_requests, fixture_metadata = load_fixture(args.fixture)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if max(args.concurrencies) > len(fixture_requests):
        parser.error(
            f"largest concurrency {max(args.concurrencies)} exceeds fixture size "
            f"{len(fixture_requests)}"
        )
    if args.require_prompt_token_ids and any(
        request["prompt_token_ids"] is None for request in fixture_requests
    ):
        parser.error("--require-prompt-token-ids received a text-only fixture")

    result: dict[str, Any] = {
        "status": "running",
        "created_at": utc_now(),
        "config": {
            "base_url": args.base_url,
            "endpoint": args.endpoint,
            "concurrencies": args.concurrencies,
            "output_tokens_per_request": args.output_tokens,
            "sampling_params": {
                **SAMPLING_PARAMS,
                "max_new_tokens": args.output_tokens,
            },
            "stream": True,
            "request_timeout_seconds": args.request_timeout,
            "barrier_timeout_seconds": args.barrier_timeout,
            "flush_cache_before_phase": args.flush_cache_before_phase,
            "phase_settle_seconds": args.phase_settle_seconds,
            "selection": "strict_fixture_prefix_without_shuffle",
            "require_prompt_token_ids": args.require_prompt_token_ids,
        },
        "fixture": fixture_metadata,
        "phases": [],
    }
    atomic_write_json(args.output, result)

    try:
        asyncio.run(run_benchmark(args, fixture_requests, result))
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        result["finished_at"] = utc_now()
        result["error"] = "KeyboardInterrupt"
        atomic_write_json(args.output, result)
        return 130
    except Exception as exc:
        result["status"] = "error"
        result["finished_at"] = utc_now()
        result["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(args.output, result)
        print(result["error"], file=sys.stderr)
        return 1

    result["status"] = "ok"
    result["finished_at"] = utc_now()
    atomic_write_json(args.output, result)
    print(f"result={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
