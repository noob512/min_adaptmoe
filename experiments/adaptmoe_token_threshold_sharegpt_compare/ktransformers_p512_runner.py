#!/usr/bin/env python3
"""Replay the frozen P512/N128 trace through KTransformers/SGLang directly."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_hash(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def prompt_hash(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    def percentile(q: float) -> float:
        position = (len(ordered) - 1) * q
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {"mean": statistics.fmean(values), "median": statistics.median(values),
            "p95": percentile(.95), "min": min(values), "max": max(values)}


def prepare_placement(plan: Path, target: Path) -> dict:
    import torch
    from kt_kernel import generate_gpu_experts_masks

    cold = json.loads(plan.read_text())
    selected = cold["selected_by_layer"]
    assert set(map(int, selected)) == set(range(1, 27))
    width = len(selected['1'])
    assert width in (8,11)
    counts = torch.zeros((1, 27, 64), dtype=torch.int64)
    for layer in range(1, 27):
        ids = selected[str(layer)]
        assert len(ids) == len(set(ids)) == width and all(0 <= e < 64 for e in ids)
        counts[0, layer] = 1
        counts[0, layer, ids] = 0
    masks = generate_gpu_experts_masks(counts.sum(dim=0).float(), (64-width) * 26)
    for layer in range(1, 27):
        assert set(torch.where(~masks[layer])[0].tolist()) == set(selected[str(layer)])
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"logical_count": counts}, target)
    expected_bytes = 26 * width * 3 * 2048 * 1408 * 2
    assert cold["selected_main_bytes"] == expected_bytes
    manifest = {
        "cold_plan": str(plan.resolve()), "cold_plan_sha256": digest(plan),
        "frequency_plan": str(target.resolve()), "frequency_plan_sha256": digest(target),
        "gpu_experts_per_moe_layer": 64-width, "cpu_experts_per_moe_layer": width,
        "moe_layers": 26, "selected_expert_count": width*26,
        "selected_main_bytes": expected_bytes,
        "selected_by_layer": selected, "exact_mask_verified_with_kt_kernel": True,
        "dense_layer_zero_does_not_consume_moe_topk": True,
    }
    write_json(target.with_suffix(".json"), manifest)
    return manifest


async def replay(engine, fixture: dict, rate: float, directory: Path) -> dict:
    records: list[dict] = []
    started = time.perf_counter()
    started_epoch = time.time()
    sampling = {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                "max_new_tokens": 16, "min_new_tokens": 16, "ignore_eos": True}

    async def one(index: int, request: dict) -> dict:
        planned = index / rate
        await asyncio.sleep(max(0.0, started + planned - time.perf_counter()))
        dispatched = time.perf_counter() - started
        stream = await engine.async_generate(
            input_ids=request["prompt_token_ids"], sampling_params=sampling,
            stream=True, rid=f"p{len(request['prompt_token_ids'])}-{index:03d}")
        ids: list[int] = []
        times: list[float] = []
        completion = 0
        grouped = 0
        final_meta = {}
        async for chunk in stream:
            observed = time.perf_counter() - started
            meta = chunk.get("meta_info") or {}
            current = int(meta.get("completion_tokens", completion))
            assert current >= completion
            new = current - completion
            chunk_ids = list(map(int, chunk.get("output_ids", [])))
            if len(chunk_ids) == current:
                ids = chunk_ids
            elif new and len(chunk_ids) >= new:
                ids.extend(chunk_ids[-new:])
            elif new:
                raise RuntimeError(f"request {index}: missing streamed output token IDs")
            times.extend([observed] * new)
            grouped += int(new > 1)
            completion = current
            final_meta = meta
        finished = time.perf_counter() - started
        if len(ids) != 16 or len(times) != 16 or completion != 16:
            raise RuntimeError(f"request {index}: output length {len(ids)}/{len(times)}/{completion}")
        itl = [(b - a) * 1000 for a, b in zip(times, times[1:])]
        record = {
            "fixture_index": index, "fixture_request_id": request.get("request"),
            "planned_arrival_offset_seconds": planned,
            "dispatched_offset_seconds": dispatched,
            "admission_delay_ms": (dispatched - planned) * 1000,
            "prompt_token_ids_sha256": prompt_hash(request["prompt_token_ids"]),
            "prompt_tokens": len(request['prompt_token_ids']), "completion_tokens": completion,
            "output_token_ids": ids, "output_token_ids_sha256": token_hash(ids),
            "token_observed_offsets_seconds": times, "stream_finished_offset_seconds": finished,
            "ttft_ms": (times[0] - planned) * 1000,
            "tpot_ms": statistics.fmean(itl), "itl_ms": itl,
            "e2e_ms": (finished - planned) * 1000,
            "grouped_stream_events": grouped, "server_meta": final_meta, "success": True,
        }
        records.append(record)
        with (directory / "requests.jsonl").open("a") as output:
            output.write(json.dumps(record, default=str) + "\n")
        if len(records) % 16 == 0:
            print(f"TRACE_PROGRESS completed={len(records)}/128 elapsed={finished:.3f}s", flush=True)
        return record

    tasks = [asyncio.create_task(one(i, request)) for i, request in enumerate(fixture["requests"])]
    try:
        ordered = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    # Match the baseline's completion-origin metric, excluding shutdown.
    elapsed = max(record["stream_finished_offset_seconds"] for record in ordered)
    return {
        "request_count": 128, "successful_requests": len(ordered), "failed_requests": 0,
        "actual_qps": len(ordered) / elapsed, "elapsed_seconds": elapsed,
        "started_epoch": started_epoch, "finished_epoch": started_epoch + elapsed,
        "planned_arrival_span_seconds": 127 / rate,
        "drain_seconds": elapsed - 127 / rate,
        "grouped_stream_events": sum(r["grouped_stream_events"] for r in ordered),
        "ttft_ms": distribution([r["ttft_ms"] for r in ordered]),
        "tpot_ms": distribution([r["tpot_ms"] for r in ordered]),
        "e2e_ms": distribution([r["e2e_ms"] for r in ordered]),
        "requests": ordered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--frequency-plan", type=Path, required=True)
    parser.add_argument("--cold-plan", type=Path)
    parser.add_argument("--prepare-placement", action="store_true")
    parser.add_argument("--qps", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=576)
    parser.add_argument("--kv-budget-bytes", type=int, default=536870912)
    parser.add_argument("--kv-effective-tokens", type=int, default=17248)
    args = parser.parse_args()
    if args.prepare_placement:
        print(json.dumps(prepare_placement(args.cold_plan, args.frequency_plan), indent=2))
        return
    if not args.qps or args.qps <= 0 or not args.output or args.output.exists():
        raise SystemExit("positive --qps and new --output are required")
    fixture = json.loads(args.fixture.read_text())
    assert len(fixture["requests"]) == 128
    assert digest(args.fixture) == {512:"febf1babf6981abb9f1d65cfd600e33e7a8833ad799b87e09e66cc913b8edef7",
                                   1024:"a4cc14060c94efb8c86f0524f2c49fbf85c0ff284eb12636e1e5c133d85bb9f2"}[args.prompt_tokens]
    for request in fixture["requests"]:
        assert len(request["prompt_token_ids"]) == args.prompt_tokens
        assert prompt_hash(request["prompt_token_ids"]) == request["prompt_token_ids_sha256"]
    placement = json.loads(args.frequency_plan.with_suffix(".json").read_text())
    assert digest(args.frequency_plan) == placement["frequency_plan_sha256"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"schema": "kt_p512_n128_fixed_arrival_v1", "status": "initializing",
              "started_at": utc_now(), "configuration": {
                  "model": str(args.model), "fixture": str(args.fixture),
                  "fixture_sha256": digest(args.fixture), "arrival_qps": args.qps,
                  "requests": 128, "prompt_tokens": args.prompt_tokens, "output_tokens": 16,
                  "latency_origin": "planned_arrival", "additional_conditioning_requests": 0,
                  "max_num_seqs": args.max_num_seqs, "max_num_batched_tokens": 8192,
                  "max_model_len": args.max_model_len, "kv_budget_reference_bytes": args.kv_budget_bytes,
                  "kv_effective_tokens": args.kv_effective_tokens, "kv_buffer_bytes_with_dummy_slot": (args.kv_effective_tokens+1) * 576 * 2 * 27,
                  "cpu_threads": 14, "attention_backend": "triton",
                  "cuda_graph": False, "mixed_chunk": True,
                  "transport": "direct_engine_async_streaming", "seed": 20260831,
                  "placement": placement}}
    write_json(args.output, result)
    engine = None
    try:
        import sglang as sgl
        engine = sgl.Engine(
            model_path=str(args.model), dtype="bfloat16", kv_cache_dtype="auto", tp_size=1,
            context_length=args.max_model_len, max_running_requests=args.max_num_seqs, max_total_tokens=args.kv_effective_tokens,
            mem_fraction_static=0.97, page_size=1, chunked_prefill_size=8192,
            max_prefill_tokens=8192, schedule_policy="fcfs", stream_interval=1,
            random_seed=20260831, trust_remote_code=True,
            kt_weight_path=str(args.model), kt_method="BF16", kt_cpuinfer=14,
            kt_threadpool_count=1, kt_numa_nodes=[0], kt_num_gpu_experts=placement['gpu_experts_per_moe_layer'],
            kt_expert_placement_strategy="frequency", init_expert_location=str(args.frequency_plan),
            kt_max_deferred_experts_per_token=0, kt_gpu_prefill_token_threshold=0,
            kt_enable_dynamic_expert_update=False, attention_backend="triton",
            enable_mixed_chunk=True, disable_cuda_graph=True, enable_torch_compile=False,
            disable_shared_experts_fusion=True, disable_radix_cache=True,
            enable_metrics=True, decode_log_interval=1, log_level="info",
        )
        # get_server_info() requires the tokenizer reply loop, normally started
        # by the first generate call. Use the startup ready payload here.
        result["server_info_before"] = {**dataclasses.asdict(engine.server_args),
                                        **engine.scheduler_info}
        actual = result["server_info_before"]
        assert actual["max_total_num_tokens"] == args.kv_effective_tokens, actual
        for key, expected in {"disable_cuda_graph": True, "enable_mixed_chunk": True,
                              "max_running_requests": args.max_num_seqs, "attention_backend": "triton"}.items():
            assert actual[key] == expected, (key, actual[key])
        result["status"] = "measuring"
        write_json(args.output, result)
        print(f"TRACE_BEGIN qps={args.qps} requests=128", flush=True)
        result["measurement"] = engine.loop.run_until_complete(
            replay(engine, fixture, args.qps, args.output.parent))
        result["server_info_after"] = engine.get_server_info()
        result["status"] = "ok"
        print("TRACE_END " + json.dumps({key: value for key, value in result["measurement"].items()
                                         if key != "requests"}), flush=True)
    except BaseException as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["finished_at"] = utc_now()
        write_json(args.output, result)
        if engine is not None:
            engine.shutdown()


if __name__ == "__main__":
    main()
