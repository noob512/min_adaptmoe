#!/usr/bin/env python3
"""Common pre-tokenized localhost SSE client for the EuroSys evaluation.

The same client is used for vLLM, SGLang/KTransformers, and MoE-Infinity.
Payload serialization and TCP connection setup happen before measurement.  A
request's clock starts in aiohttp's ``on_request_chunk_sent`` callback, i.e.
when its already-serialized body is handed to the transport.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator


SLO = {
    "short_dialogue": (2000.0, 100.0),
    "ordinary_chat": (5000.0, 150.0),
    "medium_context": (15000.0, 200.0),
    "rag_summary": (30000.0, 200.0),
    "long_generation": (5000.0, 150.0),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
        "p99": percentile(values, 99.0),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def positive_ints(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(item <= 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("values must be unique positive integers")
    return result


@dataclass(frozen=True)
class FixtureRequest:
    fixture_index: int
    request_id: Any
    prompt_token_ids: list[int]
    prompt_token_ids_sha256: str
    source_indices: list[int]


def load_fixture(path: Path) -> tuple[dict[str, Any], list[FixtureRequest]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("metadata"), dict):
        raise ValueError("fixture must contain an object-valued metadata field")
    records = value.get("requests")
    if not isinstance(records, list) or not records:
        raise ValueError("fixture requests must be a non-empty list")
    metadata = value["metadata"]
    normalized: list[FixtureRequest] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"fixture request {index} is not an object")
        token_ids = record.get("prompt_token_ids")
        if not isinstance(token_ids, list) or not token_ids or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in token_ids
        ):
            raise ValueError(f"fixture request {index} has invalid prompt_token_ids")
        if record.get("prompt_tokens") != len(token_ids):
            raise ValueError(f"fixture request {index} prompt length metadata drift")
        token_hash = sha256_bytes(
            ",".join(str(token) for token in token_ids).encode("ascii")
        )
        if record.get("prompt_token_ids_sha256") != token_hash:
            raise ValueError(f"fixture request {index} token hash drift")
        sources = record.get("source_indices", [])
        if not isinstance(sources, list) or any(
            isinstance(source, bool) or not isinstance(source, int) or source < 0
            for source in sources
        ):
            raise ValueError(f"fixture request {index} has invalid source indices")
        normalized.append(
            FixtureRequest(
                fixture_index=index,
                request_id=record.get("request", index),
                prompt_token_ids=list(token_ids),
                prompt_token_ids_sha256=token_hash,
                source_indices=list(sources),
            )
        )
    declared_count = metadata.get("requests")
    if declared_count is not None and declared_count != len(normalized):
        raise ValueError("fixture request count metadata drift")
    return metadata, normalized


async def iter_sse_data(content: Any) -> AsyncIterator[str]:
    """Parse SSE records independently of HTTP chunk boundaries."""
    buffer = ""
    data_lines: list[str] = []
    async for chunk in content.iter_any():
        try:
            buffer += chunk.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError("SSE response is not UTF-8") from exc
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line == "":
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
                continue
            if line.startswith(("event:", "id:", "retry:")):
                continue
            raise RuntimeError(f"unexpected SSE line: {line[:200]!r}")
    if buffer.rstrip("\r"):
        raise RuntimeError("SSE stream ended with an incomplete line")
    if data_lines:
        yield "\n".join(data_lines)


@dataclass
class TokenAccumulator:
    system: str
    completion_tokens: int = 0
    prompt_tokens: int | None = None
    token_arrival_s: list[float] = field(default_factory=list)
    output_token_ids: list[int] = field(default_factory=list)
    output_text_parts: list[str] = field(default_factory=list)
    finish_reason: Any = None
    sse_events: int = 0
    grouped_token_events: int = 0
    fallback_token_events: int = 0

    def observe(self, event: dict[str, Any], received_at: float) -> None:
        self.sse_events += 1
        if event.get("error") is not None or event.get("object") == "error":
            raise RuntimeError(f"server error event: {event}")
        usage = event.get("usage")
        usage_count: int | None = None
        if isinstance(usage, dict):
            raw_prompt = usage.get("prompt_tokens")
            if isinstance(raw_prompt, int) and not isinstance(raw_prompt, bool):
                self.prompt_tokens = raw_prompt
            raw_completion = usage.get("completion_tokens")
            if isinstance(raw_completion, int) and not isinstance(raw_completion, bool):
                usage_count = raw_completion
        choices = event.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        token_ids: list[int] = []
        text = ""
        if isinstance(choice, dict):
            raw_ids = choice.get("token_ids")
            if isinstance(raw_ids, list) and all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in raw_ids
            ):
                token_ids = list(raw_ids)
            raw_text = choice.get("text")
            if isinstance(raw_text, str):
                text = raw_text
                self.output_text_parts.append(raw_text)
            if choice.get("finish_reason") is not None:
                self.finish_reason = choice["finish_reason"]

        before = self.completion_tokens
        if usage_count is not None:
            if usage_count < before:
                raise RuntimeError("stream completion token count decreased")
            self.completion_tokens = usage_count
        elif token_ids:
            self.completion_tokens += len(token_ids)
        elif self.system == "moe_infinity" and isinstance(choice, dict):
            # Its public server emits exactly one choice-bearing SSE record per
            # token but currently omits continuous usage and token IDs.
            self.completion_tokens += 1
            self.fallback_token_events += 1
        elif text:
            # Diagnostic fallback for an OpenAI server without token counts.
            self.completion_tokens += 1
            self.fallback_token_events += 1

        delta = self.completion_tokens - before
        if delta > 0:
            self.token_arrival_s.extend([received_at] * delta)
            self.grouped_token_events += int(delta > 1)
        if token_ids:
            if len(token_ids) == delta:
                self.output_token_ids.extend(token_ids)
            elif token_ids[: len(self.output_token_ids)] == self.output_token_ids:
                self.output_token_ids = token_ids


class BurstGate:
    def __init__(self, size: int) -> None:
        self.size = size
        self.ready = 0
        self.condition = asyncio.Condition()
        self.release_event = asyncio.Event()

    async def arrive(self) -> None:
        async with self.condition:
            self.ready += 1
            self.condition.notify_all()
        await self.release_event.wait()

    async def wait_ready(self) -> None:
        async with self.condition:
            await self.condition.wait_for(lambda: self.ready == self.size)

    def release(self) -> None:
        self.release_event.set()


def make_payload(
    request: FixtureRequest, *, model: str, output_tokens: int, seed: int
) -> bytes:
    payload = {
        "model": model,
        "prompt": request.prompt_token_ids,
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "best_of": 1,
        "seed": seed,
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": True,
        },
        "return_token_ids": True,
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


async def run_request(
    *,
    session: Any,
    endpoint: str,
    system: str,
    request: FixtureRequest,
    body: bytes,
    output_tokens: int,
    gate: BurstGate,
    phase_started: float,
) -> dict[str, Any]:
    trace_context: dict[str, Any] = {
        "phase": "measured",
        "body_sent_perf": None,
        "body_sent_epoch": None,
        "connection_created": False,
        "connection_reused": False,
    }
    await gate.arrive()
    accumulator = TokenAccumulator(system=system)
    http_status: int | None = None
    saw_done = False
    response_finished: float | None = None
    error: str | None = None
    try:
        async with session.post(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            trace_request_ctx=trace_context,
        ) as response:
            http_status = response.status
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:4096]}")
            async for raw in iter_sse_data(response.content):
                received_at = time.perf_counter()
                if raw == "[DONE]":
                    saw_done = True
                    break
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid SSE JSON: {raw[:200]!r}") from exc
                if not isinstance(event, dict):
                    raise RuntimeError("SSE JSON event is not an object")
                accumulator.observe(event, received_at)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        response_finished = time.perf_counter()

    body_sent = trace_context["body_sent_perf"]
    token_times = accumulator.token_arrival_s
    if body_sent is None and error is None:
        error = "ProtocolError: request body send timestamp is absent"
    if trace_context["connection_created"] and error is None:
        error = "ProtocolError: measured request opened a new TCP connection"
    if not saw_done and error is None:
        error = "ProtocolError: SSE stream ended without [DONE]"
    if accumulator.completion_tokens != output_tokens and error is None:
        error = (
            "ProtocolError: generated "
            f"{accumulator.completion_tokens}/{output_tokens} required tokens"
        )
    if len(token_times) != accumulator.completion_tokens and error is None:
        error = "ProtocolError: token arrival timestamps are incomplete"
    if accumulator.prompt_tokens not in (None, len(request.prompt_token_ids)) and error is None:
        error = "ProtocolError: server prompt token count differs from fixture"

    ttft_ms = None
    e2e_ms = None
    itl_ms: list[float] = []
    if body_sent is not None and token_times:
        ttft_ms = (token_times[0] - body_sent) * 1000.0
        e2e_ms = (token_times[-1] - body_sent) * 1000.0
        itl_ms = [
            (right - left) * 1000.0
            for left, right in zip(token_times, token_times[1:])
        ]
    tpot_ms = statistics.fmean(itl_ms) if itl_ms else None
    ids_complete = len(accumulator.output_token_ids) == accumulator.completion_tokens
    output_text = "".join(accumulator.output_text_parts)
    return {
        "fixture_index": request.fixture_index,
        "fixture_request_id": request.request_id,
        "source_indices": request.source_indices,
        "prompt_tokens": len(request.prompt_token_ids),
        "prompt_token_ids_sha256": request.prompt_token_ids_sha256,
        "serialized_body_sha256": sha256_bytes(body),
        "success": error is None,
        "error": error,
        "http_status": http_status,
        "body_sent_epoch_s": trace_context["body_sent_epoch"],
        "submitted_after_phase_start_ms": (
            (body_sent - phase_started) * 1000.0 if body_sent is not None else None
        ),
        "connection_created_during_measurement": trace_context["connection_created"],
        "connection_reused": trace_context["connection_reused"],
        "ttft_ms": ttft_ms,
        "itl_ms": itl_ms,
        "tpot_ms": tpot_ms,
        "e2e_ms": e2e_ms,
        "response_tail_ms": (
            (response_finished - token_times[-1]) * 1000.0
            if token_times and response_finished is not None
            else None
        ),
        "completion_tokens": accumulator.completion_tokens,
        "server_prompt_tokens": accumulator.prompt_tokens,
        "finish_reason": accumulator.finish_reason,
        "saw_done": saw_done,
        "sse_events": accumulator.sse_events,
        "grouped_token_events": accumulator.grouped_token_events,
        "fallback_token_events": accumulator.fallback_token_events,
        "output_token_ids": accumulator.output_token_ids if ids_complete else None,
        "output_token_ids_sha256": (
            sha256_json(accumulator.output_token_ids) if ids_complete else None
        ),
        "output_text_bytes": len(output_text.encode("utf-8")),
        "output_text_sha256": sha256_bytes(output_text.encode("utf-8")),
    }


async def warm_sessions(sessions: list[Any], health_url: str) -> None:
    """Establish or refresh keep-alive connections outside timed requests."""

    async def warm(session: Any) -> None:
        async with session.get(
            health_url, trace_request_ctx={"phase": "connection_warmup"}
        ) as response:
            await response.read()
            if response.status != 200:
                raise RuntimeError(f"health endpoint returned HTTP {response.status}")

    await asyncio.gather(*(warm(session) for session in sessions))


async def create_sessions(count: int, health_url: str, timeout_s: float) -> list[Any]:
    import aiohttp

    async def on_connection_create_start(_session: Any, ctx: Any, _params: Any) -> None:
        if ctx.trace_request_ctx.get("phase") == "measured":
            ctx.trace_request_ctx["connection_created"] = True

    async def on_connection_reuse(_session: Any, ctx: Any, _params: Any) -> None:
        if ctx.trace_request_ctx.get("phase") == "measured":
            ctx.trace_request_ctx["connection_reused"] = True

    async def on_chunk_sent(_session: Any, ctx: Any, params: Any) -> None:
        state = ctx.trace_request_ctx
        if state.get("phase") == "measured" and params.chunk and state["body_sent_perf"] is None:
            state["body_sent_perf"] = time.perf_counter()
            state["body_sent_epoch"] = time.time()

    trace = aiohttp.TraceConfig()
    trace.on_connection_create_start.append(on_connection_create_start)
    trace.on_connection_reuseconn.append(on_connection_reuse)
    trace.on_request_chunk_sent.append(on_chunk_sent)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    sessions = [
        aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=1, limit_per_host=1),
            trace_configs=[trace],
        )
        for _ in range(count)
    ]
    try:
        await warm_sessions(sessions, health_url)
        return sessions
    except BaseException:
        await asyncio.gather(*(session.close() for session in sessions))
        raise


def summarize_phase(
    *, concurrency: int, records: list[dict[str, Any]], phase_elapsed: float,
    workload: str,
) -> dict[str, Any]:
    successful = [record for record in records if record["success"]]
    ttft = [float(record["ttft_ms"]) for record in successful]
    tpot = [float(record["tpot_ms"]) for record in successful if record["tpot_ms"] is not None]
    e2e = [float(record["e2e_ms"]) for record in successful]
    itl = [float(value) for record in successful for value in record["itl_ms"]]
    output_tokens = sum(int(record["completion_tokens"]) for record in successful)
    ttft_slo, tpot_slo = SLO[workload]
    slo_good = [
        record for record in successful
        if record["ttft_ms"] <= ttft_slo
        and record["tpot_ms"] is not None
        and record["tpot_ms"] <= tpot_slo
    ]
    submit_offsets = [
        float(record["submitted_after_phase_start_ms"])
        for record in records
        if record["submitted_after_phase_start_ms"] is not None
    ]
    return {
        "concurrency": concurrency,
        "elapsed_seconds": phase_elapsed,
        "requests": records,
        "request_count": len(records),
        "successful_requests": len(successful),
        "failed_requests": len(records) - len(successful),
        "success_rate": len(successful) / len(records),
        "completed_output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / phase_elapsed,
        "goodput_requests_per_second": len(slo_good) / phase_elapsed,
        "slo_satisfied_requests": len(slo_good),
        "slo": {"ttft_ms": ttft_slo, "tpot_ms": tpot_slo},
        "ttft_ms": distribution(ttft),
        "tpot_ms": distribution(tpot),
        "itl_ms": distribution(itl),
        "e2e_ms": distribution(e2e),
        "request_submit_skew_ms": (
            max(submit_offsets) - min(submit_offsets) if submit_offsets else None
        ),
    }


async def run_phase(
    *, sessions: list[Any], endpoint: str, system: str, requests: list[FixtureRequest],
    model: str, output_tokens: int, seed: int, barrier_timeout: float,
    workload: str,
) -> dict[str, Any]:
    concurrency = len(requests)
    bodies = [
        make_payload(request, model=model, output_tokens=output_tokens, seed=seed)
        for request in requests
    ]
    gate = BurstGate(concurrency)
    placeholder = time.perf_counter()
    tasks = [
        asyncio.create_task(
            run_request(
                session=sessions[index], endpoint=endpoint, system=system,
                request=request, body=bodies[index], output_tokens=output_tokens,
                gate=gate, phase_started=placeholder,
            )
        )
        for index, request in enumerate(requests)
    ]
    try:
        await asyncio.wait_for(gate.wait_ready(), timeout=barrier_timeout)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    phase_started = time.perf_counter()
    gate.release()
    records = await asyncio.gather(*tasks)
    phase_elapsed = time.perf_counter() - phase_started
    setup_gap_ms = (phase_started - placeholder) * 1000.0
    for record in records:
        if record["submitted_after_phase_start_ms"] is not None:
            record["submitted_after_phase_start_ms"] -= setup_gap_ms
    return summarize_phase(
        concurrency=concurrency,
        records=records,
        phase_elapsed=phase_elapsed,
        workload=workload,
    )


class RequestStartLimiter:
    """Pace inference request starts without contaminating service timings."""

    def __init__(self, minimum_interval_seconds: float) -> None:
        self.minimum_interval_seconds = minimum_interval_seconds
        self._last_start: float | None = None

    async def wait(self) -> float:
        waited = 0.0
        if self._last_start is not None and self.minimum_interval_seconds > 0:
            remaining = (
                self._last_start + self.minimum_interval_seconds
                - time.perf_counter()
            )
            if remaining > 0:
                started = time.perf_counter()
                await asyncio.sleep(remaining)
                waited = time.perf_counter() - started
        self._last_start = time.perf_counter()
        return waited


def combine_serial_waves(
    *,
    concurrency: int,
    waves: list[dict[str, Any]],
    workload: str,
    wall_elapsed_seconds: float,
    rate_limit_wait_seconds: float,
    minimum_start_interval_seconds: float,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    service_elapsed_seconds = 0.0
    sequence = 0
    for wave_index, wave in enumerate(waves):
        service_elapsed_seconds += float(wave["elapsed_seconds"])
        for record in wave["requests"]:
            record["measurement_sequence"] = sequence
            record["measurement_wave"] = wave_index
            records.append(record)
            sequence += 1
    result = summarize_phase(
        concurrency=concurrency,
        records=records,
        phase_elapsed=service_elapsed_seconds,
        workload=workload,
    )
    starts = sorted(
        float(record["body_sent_epoch_s"])
        for record in records
        if isinstance(record.get("body_sent_epoch_s"), (int, float))
    )
    intervals = [right - left for left, right in zip(starts, starts[1:])]
    observed_minimum = min(intervals) if intervals else None
    tolerance_seconds = 0.05
    result.update(
        {
            "measurement_waves": len(waves),
            "service_elapsed_seconds": service_elapsed_seconds,
            "wall_elapsed_seconds": wall_elapsed_seconds,
            "rate_limit_wait_seconds": rate_limit_wait_seconds,
            "minimum_request_start_interval_seconds": (
                minimum_start_interval_seconds
            ),
            "observed_minimum_request_start_interval_seconds": observed_minimum,
            "request_start_intervals_seconds": intervals,
            "request_rate_limit_compliant": (
                observed_minimum is None
                or observed_minimum
                >= minimum_start_interval_seconds - tolerance_seconds
            ),
        }
    )
    return result


async def run(args: argparse.Namespace, result: dict[str, Any]) -> None:
    eval_meta, evaluation = load_fixture(args.fixture)
    warm_meta: dict[str, Any] | None = None
    warmups: list[FixtureRequest] = []
    if args.warmup_requests:
        if args.warmup_fixture is None:
            raise ValueError("--warmup-fixture is required when warmups are enabled")
        warm_meta, warmups = load_fixture(args.warmup_fixture)
        if len(warmups) < args.warmup_requests:
            raise ValueError("warmup fixture is too small")
        eval_sources = {value for request in evaluation for value in request.source_indices}
        warm_sources = {value for request in warmups for value in request.source_indices}
        if eval_sources & warm_sources:
            raise ValueError("warmup and evaluation fixtures overlap")

    max_concurrency = max(args.concurrencies)
    if len(evaluation) < args.measurement_requests:
        raise ValueError("evaluation fixture is smaller than measurement request count")
    if any(args.measurement_requests % value for value in args.concurrencies):
        raise ValueError(
            "measurement request count must be divisible by every concurrency"
        )
    if args.minimum_request_start_interval > 0 and max_concurrency != 1:
        raise ValueError(
            "a positive request-start interval is only valid for concurrency=1"
        )
    sessions = await create_sessions(max_concurrency, args.health_url, args.request_timeout)
    limiter = RequestStartLimiter(args.minimum_request_start_interval)
    try:
        # Long generations can outlive a server's keep-alive timeout.  Refresh
        # each connection between phases, outside the timed POST boundary.  The
        # first phase reuses the connection established by create_sessions().
        needs_rewarm = False
        for index in range(args.warmup_requests):
            await limiter.wait()
            if needs_rewarm:
                await warm_sessions(sessions[:1], args.health_url)
            phase = await run_phase(
                sessions=sessions[:1], endpoint=args.endpoint, system=args.system,
                requests=[warmups[index]], model=args.model,
                output_tokens=args.warmup_output_tokens, seed=args.seed,
                barrier_timeout=args.barrier_timeout,
                workload=str(warm_meta["workload"]),
            )
            result["warmups"].append(phase["requests"][0])
            atomic_json(args.output, result)
            if phase["failed_requests"]:
                raise RuntimeError(f"warmup request {index} failed")
            needs_rewarm = True
        for concurrency in args.concurrencies:
            waves: list[dict[str, Any]] = []
            rate_limit_wait_seconds = 0.0
            wall_started = time.perf_counter()
            for offset in range(0, args.measurement_requests, concurrency):
                rate_limit_wait_seconds += await limiter.wait()
                if needs_rewarm:
                    await warm_sessions(sessions[:concurrency], args.health_url)
                waves.append(
                    await run_phase(
                        sessions=sessions[:concurrency], endpoint=args.endpoint,
                        system=args.system,
                        requests=evaluation[offset : offset + concurrency],
                        model=args.model, output_tokens=args.output_tokens,
                        seed=args.seed, barrier_timeout=args.barrier_timeout,
                        workload=str(eval_meta["workload"]),
                    )
                )
                needs_rewarm = True
            phase = combine_serial_waves(
                concurrency=concurrency,
                waves=waves,
                workload=str(eval_meta["workload"]),
                wall_elapsed_seconds=time.perf_counter() - wall_started,
                rate_limit_wait_seconds=rate_limit_wait_seconds,
                minimum_start_interval_seconds=(
                    args.minimum_request_start_interval
                ),
            )
            result["phases"].append(phase)
            atomic_json(args.output, result)
    finally:
        await asyncio.gather(*(session.close() for session in sessions))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=("ours", "ktransformers", "moe_infinity"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--warmup-fixture", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint")
    parser.add_argument("--health-url")
    parser.add_argument("--concurrencies", type=positive_ints, default=positive_ints("1,4,8,16"))
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--warmup-requests", type=int, default=3)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--measurement-requests", type=int, default=1)
    parser.add_argument(
        "--minimum-request-start-interval", type=float, default=0.0
    )
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--barrier-timeout", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260823)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    args.endpoint = args.endpoint or args.base_url + "/v1/completions"
    args.health_url = args.health_url or args.base_url + "/health"
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    if args.warmup_requests < 0 or args.warmup_output_tokens < 2:
        parser.error("warmup counts must be non-negative and output at least two tokens")
    if args.measurement_requests <= 0:
        parser.error("measurement request count must be positive")
    if args.minimum_request_start_interval < 0:
        parser.error("minimum request-start interval must be non-negative")
    if args.request_timeout <= 0 or args.barrier_timeout <= 0:
        parser.error("timeouts must be positive")
    try:
        fixture_meta, fixture_requests = load_fixture(args.fixture)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    declared_output = fixture_meta.get("output_tokens")
    args.output_tokens = args.output_tokens or declared_output
    if not isinstance(args.output_tokens, int) or args.output_tokens < 2:
        parser.error("output token count must be at least two")
    if declared_output != args.output_tokens:
        parser.error("--output-tokens differs from frozen fixture metadata")
    workload = fixture_meta.get("workload")
    if workload not in SLO:
        parser.error(f"fixture has unknown workload: {workload!r}")
    if max(args.concurrencies) > len(fixture_requests):
        parser.error("requested concurrency exceeds fixture size")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "created_at": utc_now(),
        "timing_boundary": {
            "start": "aiohttp on_request_chunk_sent for pre-serialized request body",
            "ttft_end": "first parsed token-bearing SSE event",
            "e2e_end": "last generated-token SSE arrival",
            "tokenization_included": False,
            "json_serialization_included": False,
            "connection_setup_included": False,
            "response_tail_excluded_from_e2e": True,
        },
        "config": {
            "system": args.system,
            "model": args.model,
            "base_url": args.base_url,
            "endpoint": args.endpoint,
            "health_url": args.health_url,
            "concurrencies": args.concurrencies,
            "output_tokens": args.output_tokens,
            "warmup_requests": args.warmup_requests,
            "warmup_output_tokens": args.warmup_output_tokens,
            "measurement_requests": args.measurement_requests,
            "minimum_request_start_interval_seconds": (
                args.minimum_request_start_interval
            ),
            "seed": args.seed,
        },
        "fixture": {
            "path": str(args.fixture.resolve()),
            "sha256": sha256_file(args.fixture),
            "metadata": fixture_meta,
        },
        "warmup_fixture": (
            {"path": str(args.warmup_fixture.resolve()), "sha256": sha256_file(args.warmup_fixture)}
            if args.warmup_fixture is not None else None
        ),
        "warmups": [],
        "phases": [],
    }
    atomic_json(args.output, result)
    try:
        asyncio.run(run(args, result))
    except KeyboardInterrupt:
        result.update(status="interrupted", finished_at=utc_now(), error="KeyboardInterrupt")
        atomic_json(args.output, result)
        return 130
    except Exception as exc:
        result.update(status="error", finished_at=utc_now(), error=f"{type(exc).__name__}: {exc}")
        atomic_json(args.output, result)
        print(result["error"], flush=True)
        return 1
    result.update(status="ok", finished_at=utc_now())
    atomic_json(args.output, result)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
