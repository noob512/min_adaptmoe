#!/usr/bin/env python3
"""Measure matched prefill/decode phases through SGLang's batched SSE API."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

import aiohttp


METRIC_RE = re.compile(
    r'^sglang:cuda_graph_passes_total\{(?P<labels>[^}]*)\}\s+(?P<value>[0-9.eE+-]+)$'
)
LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--conditioning-repeats", type=int, default=1)
    parser.add_argument("--measurement-repeats", type=int, default=3)
    parser.add_argument("--warmup-decode-steps", type=int, default=5)
    parser.add_argument("--measured-decode-steps", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--formal-e3",
        action="store_true",
        help="allow frozen E3 B1/B4/B8 subsets instead of only 1,4",
    )
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def summarize(values: list[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"invalid samples: {values}")
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sample_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "samples": values,
    }


def load_fixture(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    requests = payload["requests"]
    if not isinstance(requests, list) or len(requests) < 4:
        raise ValueError("fixture needs at least four requests")
    normalized = []
    for index, request in enumerate(requests):
        ids = request.get("prompt_token_ids")
        if not isinstance(ids, list) or not ids or any(
            isinstance(value, bool) or not isinstance(value, int) for value in ids
        ):
            raise ValueError(f"request {index} has invalid prompt_token_ids")
        if int(request.get("prompt_tokens", -1)) != len(ids):
            raise ValueError(f"request {index} prompt length differs")
        normalized.append({"index": index, "prompt_token_ids": ids})
    return payload.get("metadata", {}), normalized, sha256_bytes(raw)


def parse_graph_metrics(text: str) -> dict[str, float]:
    result = {"true": 0.0, "false": 0.0}
    for line in text.splitlines():
        match = METRIC_RE.match(line)
        if not match:
            continue
        labels = dict(LABEL_RE.findall(match.group("labels")))
        mode = labels.get("mode")
        if mode in {"true", "decode_cuda_graph"}:
            result["true"] += float(match.group("value"))
        elif mode in {"false", "decode_eager", "decode_none"}:
            result["false"] += float(match.group("value"))
    return result


async def read_sse(response: aiohttp.ClientResponse):
    data_lines: list[str] = []
    async for raw in response.content:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


async def scrape_metrics(session: aiohttp.ClientSession, base_url: str) -> dict[str, float]:
    async with session.get(f"{base_url}/metrics") as response:
        response.raise_for_status()
        return parse_graph_metrics(await response.text())


def merge_ids(current: list[int], raw: Any, old_count: int, new_count: int) -> list[int]:
    if not isinstance(raw, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw
    ):
        raise ValueError("output_ids is not an integer list")
    if len(raw) == new_count:
        return list(raw)
    if len(raw) == new_count - old_count:
        return current + list(raw)
    if len(raw) >= len(current) and raw[: len(current)] == current:
        return list(raw)
    raise ValueError(
        f"cannot merge output IDs old={old_count} new={new_count} raw={len(raw)}"
    )


async def run_cycle(
    session: aiohttp.ClientSession,
    base_url: str,
    requests: list[dict[str, Any]],
    batch_size: int,
    output_tokens: int,
    warmup_decode_steps: int,
    measured_decode_steps: int,
    role: str,
    role_index: int,
) -> dict[str, Any]:
    selected = requests[:batch_size]
    payload = {
        "input_ids": [item["prompt_token_ids"] for item in selected],
        "sampling_params": [
            {
                "temperature": 0.0,
                "max_new_tokens": output_tokens,
                "min_new_tokens": output_tokens,
                "ignore_eos": True,
                "sampling_seed": 0,
                "stream_interval": 1,
            }
            for _ in selected
        ],
        "stream": True,
        "log_metrics": True,
    }
    counts = [0] * batch_size
    ids: list[list[int]] = [[] for _ in selected]
    token_times: list[list[float]] = [[] for _ in selected]
    final_meta: list[dict[str, Any] | None] = [None] * batch_size

    metrics_before = await scrape_metrics(session, base_url)
    started = time.perf_counter()
    async with session.post(f"{base_url}/generate", json=payload) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:4096]}")
        async for data in read_sse(response):
            received = time.perf_counter()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            index = int(event["index"])
            meta = event["meta_info"]
            count = int(meta["completion_tokens"])
            old_count = counts[index]
            if count < old_count or count - old_count > 1:
                raise RuntimeError(
                    f"request {index} token count jump {old_count} -> {count}"
                )
            if count > old_count:
                token_times[index].append(received)
            ids[index] = merge_ids(ids[index], event.get("output_ids"), old_count, count)
            counts[index] = count
            if final_meta[index] is None:
                final_meta[index] = {}
            final_meta[index].update(meta)
    finished = time.perf_counter()
    metrics_after = await scrape_metrics(session, base_url)

    if counts != [output_tokens] * batch_size:
        raise RuntimeError(f"output token counts differ: {counts}")
    if any(len(times) != output_tokens for times in token_times):
        raise RuntimeError("one SSE token event per output token was not observed")
    if any(len(output_ids) != output_tokens for output_ids in ids):
        raise RuntimeError("output ID reconstruction failed")
    metas = [meta for meta in final_meta if meta is not None]
    if len(metas) != batch_size:
        raise RuntimeError("final metadata missing")

    # One batched request guarantees one prefill batch. The slowest SSE arrival for
    # each token ordinal is the client-visible completion boundary of that step.
    step_boundaries = [
        max(token_times[request_index][token_index] for request_index in range(batch_size))
        for token_index in range(output_tokens)
    ]
    decode_intervals_ms = [
        (step_boundaries[index] - step_boundaries[index - 1]) * 1000.0
        for index in range(1, output_tokens)
    ]
    warmup_ms = decode_intervals_ms[:warmup_decode_steps]
    measured_ms = decode_intervals_ms[warmup_decode_steps:]
    if len(measured_ms) != measured_decode_steps:
        raise RuntimeError("measured decode interval count differs")

    prefill_launch = [meta.get("prefill_launch_latency") for meta in metas]
    prefill_launch_ms = (
        [float(value) * 1000.0 for value in prefill_launch]
        if all(value is not None for value in prefill_launch)
        else []
    )
    prefill_forward_ms = [
        (float(meta["prefill_finished_time"]) - float(meta["forward_entry_time"]))
        * 1000.0
        for meta in metas
        if meta.get("prefill_finished_time") is not None
        and meta.get("forward_entry_time") is not None
    ]
    graph_delta = {
        mode: metrics_after[mode] - metrics_before[mode] for mode in ("true", "false")
    }
    output_hash = sha256_bytes(
        json.dumps(ids, separators=(",", ":")).encode("utf-8")
    )
    return {
        "role": role,
        "role_index": role_index,
        "batch_size": batch_size,
        "prompt_tokens": sum(len(item["prompt_token_ids"]) for item in selected),
        "output_tokens_per_request": output_tokens,
        "server_prefill_launch_ms": (
            summarize(prefill_launch_ms) if prefill_launch_ms else None
        ),
        "server_prefill_forward_ms": (
            summarize(prefill_forward_ms) if prefill_forward_ms else None
        ),
        "client_ttft_batch_ms": (step_boundaries[0] - started) * 1000.0,
        "warmup_decode_step_ms": summarize(warmup_ms),
        "measured_decode_step_ms": summarize(measured_ms),
        "decode_tokens_per_second": batch_size * 1000.0 / statistics.median(measured_ms),
        "cycle_client_wall_ms": (finished - started) * 1000.0,
        "cuda_graph_pass_delta": graph_delta,
        "output_token_ids_sha256": output_hash,
        "output_token_ids": ids,
        "final_meta_info": metas,
    }


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    metadata, requests, fixture_sha256 = load_fixture(args.fixture)
    batches = [int(value) for value in args.batch_sizes.split(",")]
    if not args.formal_e3 and batches != [1, 4]:
        raise ValueError("this matched experiment requires batch order 1,4")
    if not batches or any(batch not in (1, 4, 8) for batch in batches):
        raise ValueError("formal E3 batches must be a non-empty subset of 1,4,8")
    if max(batches) > len(requests):
        raise ValueError("fixture has fewer requests than the largest batch")
    if args.conditioning_repeats < 1 or args.measurement_repeats < 1:
        raise ValueError("positive repeat counts required")
    output_tokens = 1 + args.warmup_decode_steps + args.measured_decode_steps
    timeout = aiohttp.ClientTimeout(total=args.timeout_seconds)
    result: dict[str, Any] = {
        "schema": "ktransformers_l0_l3_phase_stream_v1",
        "status": "running",
        "fixture": {"path": str(args.fixture.resolve()), "sha256": fixture_sha256, "metadata": metadata},
        "config": {
            "batch_sizes": batches,
            "conditioning_repeats": args.conditioning_repeats,
            "measurement_repeats": args.measurement_repeats,
            "warmup_decode_steps": args.warmup_decode_steps,
            "measured_decode_steps": args.measured_decode_steps,
            "output_tokens_per_request": output_tokens,
            "sampling": "greedy_seed0_ignore_eos",
            "input": "frozen_prompt_token_ids",
            "transport": "localhost_batched_http_sse",
        },
        "cycles": [],
    }
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for batch_size in batches:
            roles = ["conditioning"] * args.conditioning_repeats + [
                "measurement"
            ] * args.measurement_repeats
            role_indices = {"conditioning": 0, "measurement": 0}
            for role in roles:
                role_index = role_indices[role]
                role_indices[role] += 1
                cycle = await run_cycle(
                    session,
                    args.base_url,
                    requests,
                    batch_size,
                    output_tokens,
                    args.warmup_decode_steps,
                    args.measured_decode_steps,
                    role,
                    role_index,
                )
                result["cycles"].append(cycle)

    result["phase_summary"] = {}
    for batch_size in batches:
        cycles = [
            cycle
            for cycle in result["cycles"]
            if cycle["role"] == "measurement" and cycle["batch_size"] == batch_size
        ]
        result["phase_summary"][str(batch_size)] = {
            "batch_size": batch_size,
            "prompt_tokens": cycles[0]["prompt_tokens"],
            "measurement_repeats": len(cycles),
            "server_prefill_launch_ms": (
                summarize(
                    [cycle["server_prefill_launch_ms"]["median"] for cycle in cycles]
                )
                if all(cycle["server_prefill_launch_ms"] is not None for cycle in cycles)
                else None
            ),
            "server_prefill_forward_ms": (
                summarize(
                    [
                        cycle["server_prefill_forward_ms"]["median"]
                        for cycle in cycles
                    ]
                )
                if all(
                    cycle["server_prefill_forward_ms"] is not None
                    for cycle in cycles
                )
                else None
            ),
            "client_ttft_batch_ms": summarize(
                [cycle["client_ttft_batch_ms"] for cycle in cycles]
            ),
            "decode_step_ms": summarize(
                [cycle["measured_decode_step_ms"]["median"] for cycle in cycles]
            ),
            "decode_tokens_per_second": summarize(
                [cycle["decode_tokens_per_second"] for cycle in cycles]
            ),
            "cycle_client_wall_ms": summarize(
                [cycle["cycle_client_wall_ms"] for cycle in cycles]
            ),
            "cuda_graph_pass_delta": {
                mode: sum(cycle["cuda_graph_pass_delta"][mode] for cycle in cycles)
                for mode in ("true", "false")
            },
            "output_token_ids_sha256": sorted(
                {cycle["output_token_ids_sha256"] for cycle in cycles}
            ),
        }
    result["status"] = "ok"
    return result


def main() -> int:
    args = parse_args()
    result = asyncio.run(async_main(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "phase_summary": result["phase_summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
