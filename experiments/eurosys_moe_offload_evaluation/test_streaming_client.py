from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("streaming_client.py")
SPEC = importlib.util.spec_from_file_location("streaming_client", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_accumulator_prefers_continuous_usage() -> None:
    value = MODULE.TokenAccumulator(system="ktransformers")
    value.observe(
        {"choices": [{"text": "a"}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}},
        1.0,
    )
    value.observe(
        {"choices": [{"text": "bc"}], "usage": {"prompt_tokens": 3, "completion_tokens": 3}},
        2.0,
    )
    assert value.prompt_tokens == 3
    assert value.completion_tokens == 3
    assert value.token_arrival_s == [1.0, 2.0, 2.0]
    assert value.grouped_token_events == 1


def test_accumulator_uses_vllm_token_ids() -> None:
    value = MODULE.TokenAccumulator(system="ours")
    value.observe({"choices": [{"text": "x", "token_ids": [7, 8]}]}, 4.0)
    assert value.completion_tokens == 2
    assert value.output_token_ids == [7, 8]
    assert value.token_arrival_s == [4.0, 4.0]


def test_moe_infinity_one_choice_event_is_one_token() -> None:
    value = MODULE.TokenAccumulator(system="moe_infinity")
    value.observe({"choices": [{"text": ""}]}, 1.0)
    value.observe({"choices": [{"text": "z", "finish_reason": "length"}]}, 2.0)
    assert value.completion_tokens == 2
    assert value.fallback_token_events == 2


def test_loads_formal_fixture(tmp_path: Path) -> None:
    token_ids = [1, 2, 3]
    token_hash = MODULE.sha256_bytes(b"1,2,3")
    fixture = {
        "metadata": {"requests": 1, "workload": "short_dialogue", "output_tokens": 32},
        "requests": [{
            "request": 0,
            "prompt_tokens": 3,
            "prompt_token_ids": token_ids,
            "prompt_token_ids_sha256": token_hash,
            "source_indices": [9],
        }],
    }
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")
    metadata, requests = MODULE.load_fixture(path)
    assert metadata["workload"] == "short_dialogue"
    assert requests[0].prompt_token_ids == token_ids


def test_percentile_linear_interpolation() -> None:
    assert MODULE.percentile([1.0, 3.0], 50.0) == 2.0


def test_localhost_sse_boundary_uses_warmed_connection() -> None:
    async def scenario() -> None:
        from aiohttp import web

        async def health(_request: web.Request) -> web.Response:
            return web.Response(text="ok")

        async def completion(request: web.Request) -> web.StreamResponse:
            payload = await request.json()
            assert payload["prompt"] == [1, 2, 3]
            response = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await response.prepare(request)
            for count, token in ((1, 7), (2, 8)):
                event = {
                    "choices": [{"text": "x", "token_ids": [token]}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": count},
                }
                await response.write(f"data: {json.dumps(event)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_get("/health", health)
        app.router.add_post("/v1/completions", completion)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        sockets = site._server.sockets
        assert sockets
        port = sockets[0].getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        sessions = await MODULE.create_sessions(1, base + "/health", 10.0)
        try:
            request = MODULE.FixtureRequest(0, 0, [1, 2, 3], "fixture-hash", [4])
            phase = await MODULE.run_phase(
                sessions=sessions,
                endpoint=base + "/v1/completions",
                system="ours",
                requests=[request],
                model="test-model",
                output_tokens=2,
                seed=1,
                barrier_timeout=2.0,
                workload="short_dialogue",
            )
            record = phase["requests"][0]
            assert record["success"] is True
            assert record["connection_reused"] is True
            assert record["connection_created_during_measurement"] is False
            assert record["output_token_ids"] == [7, 8]
            assert record["ttft_ms"] is not None
            assert record["e2e_ms"] >= record["ttft_ms"]
        finally:
            await asyncio.gather(*(session.close() for session in sessions))
            await runner.cleanup()

    asyncio.run(scenario())


def test_rewarm_happens_outside_timed_request_after_server_closes_socket() -> None:
    async def scenario() -> None:
        from aiohttp import web

        async def health(_request: web.Request) -> web.Response:
            return web.Response(text="ok")

        async def completion(request: web.Request) -> web.StreamResponse:
            await request.json()
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream", "Connection": "close"},
            )
            response.force_close()
            await response.prepare(request)
            for count, token in ((1, 7), (2, 8)):
                event = {
                    "choices": [{"text": "x", "token_ids": [token]}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": count},
                }
                await response.write(f"data: {json.dumps(event)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        app = web.Application()
        app.router.add_get("/health", health)
        app.router.add_post("/v1/completions", completion)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        sockets = site._server.sockets
        assert sockets
        base = f"http://127.0.0.1:{sockets[0].getsockname()[1]}"
        sessions = await MODULE.create_sessions(1, base + "/health", 10.0)
        request = MODULE.FixtureRequest(0, 0, [1, 2, 3], "fixture-hash", [4])
        try:
            first = await MODULE.run_phase(
                sessions=sessions,
                endpoint=base + "/v1/completions",
                system="ktransformers",
                requests=[request],
                model="test-model",
                output_tokens=2,
                seed=1,
                barrier_timeout=2.0,
                workload="short_dialogue",
            )
            assert first["requests"][0]["success"] is True
            await MODULE.warm_sessions(sessions, base + "/health")
            second = await MODULE.run_phase(
                sessions=sessions,
                endpoint=base + "/v1/completions",
                system="ktransformers",
                requests=[request],
                model="test-model",
                output_tokens=2,
                seed=1,
                barrier_timeout=2.0,
                workload="short_dialogue",
            )
            record = second["requests"][0]
            assert record["success"] is True
            assert record["connection_reused"] is True
            assert record["connection_created_during_measurement"] is False
        finally:
            await asyncio.gather(*(session.close() for session in sessions))
            await runner.cleanup()

    asyncio.run(scenario())
