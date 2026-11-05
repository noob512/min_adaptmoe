#!/usr/bin/env python3
"""Rate-limited E2 PARK/RESUME diagnostic with a 16-sequence final wave."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from streaming_client import iter_sse_data, load_fixture


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def payload(
    *, model: str, prompt_token_ids: list[int], output_tokens: int, n: int
) -> bytes:
    return json.dumps(
        {
            "model": model,
            "prompt": prompt_token_ids,
            "max_tokens": output_tokens,
            "min_tokens": output_tokens,
            "ignore_eos": True,
            # vLLM rejects n>1 for greedy sampling before scheduler admission.
            # A tiny non-zero temperature keeps the diagnostic reproducible
            # while allowing two sequences per HTTP request.
            "temperature": 0.01 if n > 1 else 0.0,
            "top_p": 1.0,
            "n": n,
            "best_of": n,
            "seed": 20260823,
            "stream": True,
            "stream_options": {
                "include_usage": True,
                "continuous_usage_stats": True,
            },
            "return_token_ids": True,
        },
        separators=(",", ":"),
    ).encode("utf-8")


async def issue_request(
    *, session: Any, endpoint: str, body: bytes, n: int, start_gate: asyncio.Event
) -> dict[str, Any]:
    await start_gate.wait()
    started_epoch = time.time()
    http_status: int | None = None
    saw_done = False
    event_count = 0
    completion_tokens: int | None = None
    finished_choices: set[int] = set()
    error: str | None = None
    try:
        async with session.post(
            endpoint, data=body, headers={"Content-Type": "application/json"}
        ) as response:
            http_status = response.status
            if response.status != 200:
                raise RuntimeError(
                    f"HTTP {response.status}: {(await response.text())[:2048]}"
                )
            async for raw in iter_sse_data(response.content):
                if raw == "[DONE]":
                    saw_done = True
                    break
                event = json.loads(raw)
                event_count += 1
                if event.get("error") is not None or event.get("object") == "error":
                    raise RuntimeError(f"server error event: {event}")
                usage = event.get("usage")
                if isinstance(usage, dict) and isinstance(
                    usage.get("completion_tokens"), int
                ):
                    completion_tokens = int(usage["completion_tokens"])
                choices = event.get("choices")
                if isinstance(choices, list):
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        if choice.get("finish_reason") is not None:
                            index = choice.get("index", 0)
                            if isinstance(index, int):
                                finished_choices.add(index)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "request_start_epoch_s": started_epoch,
        "http_status": http_status,
        "saw_done": saw_done,
        "event_count": event_count,
        "completion_tokens": completion_tokens,
        "requested_sequences": n,
        "finished_choice_indices": sorted(finished_choices),
        "error": error,
        "success": (
            error is None
            and http_status == 200
            and saw_done
            and len(finished_choices) == n
        ),
    }


async def warm_connections(sessions: list[Any], health_url: str) -> None:
    async def warm(session: Any) -> None:
        async with session.get(health_url) as response:
            await response.read()
            if response.status != 200:
                raise RuntimeError(f"health returned HTTP {response.status}")

    await asyncio.gather(*(warm(session) for session in sessions))


async def run(args: argparse.Namespace) -> dict[str, Any]:
    import aiohttp

    _, warmup_requests = load_fixture(args.warmup_fixture)
    _, evaluation_requests = load_fixture(args.fixture)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    sessions = [
        aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(limit=1, limit_per_host=1),
        )
        for _ in range(args.wave_requests)
    ]
    records: list[dict[str, Any]] = []
    last_start: float | None = None
    try:
        await warm_connections(sessions, args.health_url)
        for index in range(args.warmup_requests):
            if last_start is not None:
                await asyncio.sleep(
                    max(0.0, last_start + args.minimum_start_interval - time.time())
                )
            gate = asyncio.Event()
            gate.set()
            record = await issue_request(
                session=sessions[0],
                endpoint=args.endpoint,
                body=payload(
                    model=args.model,
                    prompt_token_ids=warmup_requests[index].prompt_token_ids,
                    output_tokens=args.warmup_output_tokens,
                    n=1,
                ),
                n=1,
                start_gate=gate,
            )
            records.append({"phase": "warmup", "index": index, **record})
            last_start = float(record["request_start_epoch_s"])
            if not record["success"]:
                return summarize(args, records)

        if last_start is not None:
            await asyncio.sleep(
                max(0.0, last_start + args.minimum_start_interval - time.time())
            )
        await warm_connections(sessions, args.health_url)
        gate = asyncio.Event()
        tasks = [
            asyncio.create_task(
                issue_request(
                    session=sessions[index],
                    endpoint=args.endpoint,
                    body=payload(
                        model=args.model,
                        prompt_token_ids=evaluation_requests[index].prompt_token_ids,
                        output_tokens=args.output_tokens,
                        n=args.sequences_per_request,
                    ),
                    n=args.sequences_per_request,
                    start_gate=gate,
                )
            )
            for index in range(args.wave_requests)
        ]
        await asyncio.sleep(0)
        gate.set()
        wave = await asyncio.gather(*tasks)
        records.extend(
            {"phase": "wave", "index": index, **record}
            for index, record in enumerate(wave)
        )
        return summarize(args, records)
    finally:
        await asyncio.gather(*(session.close() for session in sessions))


def summarize(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    starts = sorted(float(record["request_start_epoch_s"]) for record in records)
    max_starts_per_60s = max(
        (
            sum(start <= candidate < start + 60.0 for candidate in starts)
            for start in starts
        ),
        default=0,
    )
    expected_requests = args.warmup_requests + args.wave_requests
    rate_limit_enforced = args.max_inference_starts_per_60s > 0
    rate_limit_compliant = (
        not rate_limit_enforced
        or max_starts_per_60s <= args.max_inference_starts_per_60s
    )
    status = (
        "ok"
        if len(records) == expected_requests
        and all(record["success"] for record in records)
        and rate_limit_compliant
        else "error"
    )
    return {
        "schema_version": 1,
        "status": status,
        "rate_limit": {
            "policy": (
                "local vLLM inference POST start limit is disabled"
                if not rate_limit_enforced
                else "local vLLM inference POST starts <= configured maximum "
                "in every 60-second window"
            ),
            "enforced": rate_limit_enforced,
            "maximum_starts_per_60s": args.max_inference_starts_per_60s,
            "max_observed_starts_per_60s": max_starts_per_60s,
            "request_start_epoch_s": starts,
            "compliant": rate_limit_compliant,
        },
        "configuration": {
            "warmup_requests": args.warmup_requests,
            "wave_requests": args.wave_requests,
            "sequences_per_wave_request": args.sequences_per_request,
            "wave_sequences": args.wave_requests * args.sequences_per_request,
            "minimum_warmup_start_interval_seconds": args.minimum_start_interval,
            "max_inference_starts_per_60s": (
                args.max_inference_starts_per_60s
            ),
        },
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--warmup-fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:30201")
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--wave-requests", type=int, default=8)
    parser.add_argument("--sequences-per-request", type=int, default=2)
    parser.add_argument("--warmup-output-tokens", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--minimum-start-interval", type=float, default=6.1)
    parser.add_argument(
        "--max-inference-starts-per-60s",
        type=int,
        default=10,
        help="zero disables the local vLLM POST start-rate gate",
    )
    parser.add_argument("--timeout", type=float, default=7200.0)
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    args.endpoint = args.base_url + "/v1/completions"
    args.health_url = args.base_url + "/health"
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    if args.max_inference_starts_per_60s < 0:
        parser.error("max inference starts per 60s must be non-negative")
    if (
        args.max_inference_starts_per_60s > 0
        and args.warmup_requests + args.wave_requests
        > args.max_inference_starts_per_60s
    ):
        parser.error(
            "diagnostic request count exceeds the configured local vLLM "
            "POST start-rate gate"
        )
    if args.minimum_start_interval <= 6.0:
        parser.error("minimum start interval must be greater than 6 seconds")
    return args


def main() -> int:
    args = parse_args()
    result = asyncio.run(run(args))
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
