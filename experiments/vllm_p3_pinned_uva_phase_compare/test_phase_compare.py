from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import benchmark_phase_steps as bench  # noqa: E402
import verify_phase_compare as verify  # noqa: E402


def test_request_metrics_are_enabled_without_periodic_loggers() -> None:
    output_processor = SimpleNamespace(log_stats=False)
    engine = SimpleNamespace(
        log_stats=False,
        logger_manager=None,
        output_processor=output_processor,
    )

    bench.enable_request_metrics_without_default_loggers(
        SimpleNamespace(llm_engine=engine)
    )

    assert engine.log_stats is True
    assert output_processor.log_stats is True
    assert engine.logger_manager is None


def test_numeric_request_id_sort_is_stable_after_many_cycles() -> None:
    outputs = []
    for request_id in (11, 8, 10, 9):
        outputs.append(
            SimpleNamespace(
                request_id=str(request_id),
                finished=True,
                prompt_token_ids=[request_id],
                outputs=[SimpleNamespace(token_ids=[request_id + 100])],
            )
        )
    records = bench.output_records(outputs)
    assert [record["request_id"] for record in records] == ["8", "9", "10", "11"]
    with pytest.raises(RuntimeError, match="RequestStateStats"):
        bench.request_timing_summaries(records)


def test_request_state_stats_use_one_monotonic_clock_domain() -> None:
    output = SimpleNamespace(
        request_id="0",
        finished=True,
        prompt_token_ids=[1, 2],
        outputs=[SimpleNamespace(token_ids=[10, 11, 12, 13])],
        metrics=SimpleNamespace(
            first_token_latency=0.010,
            first_token_ts=25.000,
            last_token_ts=25.030,
            # These deliberately use unrelated values.  The implementation
            # must not subtract them from monotonic *_ts fields.
            arrival_time=1_900_000_000.0,
            finished_time=1_900_000_001.0,
        ),
    )
    records = bench.output_records([output])
    assert records[0]["ttft_ms"] == pytest.approx(10.0)
    assert records[0]["tpot_ms"] == pytest.approx(10.0)
    assert records[0]["request_latency_ms"] == pytest.approx(40.0)
    summaries = bench.request_timing_summaries(records)
    assert summaries["request_ttft_ms"]["median"] == pytest.approx(10.0)


def test_batch_parser_and_strict_schedule_validation() -> None:
    assert bench.parse_positive_ints("1,4") == [1, 4]
    with pytest.raises(ValueError):
        bench.parse_positive_ints("1,1")
    prefill = {
        "new_request_ids": ["0", "1"],
        "new_request_prompt_tokens": {"0": 3, "1": 4},
        "num_scheduled_tokens": {"0": 3, "1": 4},
        "total_num_scheduled_tokens": 7,
        "cached_request_ids": [],
        "resumed_request_ids": [],
        "scheduled_spec_decode_request_ids": [],
        "scheduled_encoder_request_ids": [],
        "preempted_request_ids": [],
    }
    bench.validate_prefill(prefill, 2)
    decode = {
        "new_request_ids": [],
        "num_scheduled_tokens": {"0": 1, "1": 1},
        "total_num_scheduled_tokens": 2,
        "cached_request_ids": ["0", "1"],
        "resumed_request_ids": [],
        "scheduled_spec_decode_request_ids": [],
        "scheduled_encoder_request_ids": [],
        "preempted_request_ids": [],
    }
    bench.validate_decode(decode, 2)
    decode["resumed_request_ids"] = ["0"]
    with pytest.raises(RuntimeError):
        bench.validate_decode(decode, 2)
    decode["resumed_request_ids"] = []
    decode["cached_request_ids"] = ["0", "different"]
    with pytest.raises(RuntimeError):
        bench.validate_decode(decode, 2)


def _prefill_schedule() -> dict[str, object]:
    return {
        "new_request_ids": ["0"],
        "new_request_prompt_tokens": {"0": 2},
        "num_scheduled_tokens": {"0": 2},
        "total_num_scheduled_tokens": 2,
        "cached_request_ids": [],
        "resumed_request_ids": [],
        "scheduled_spec_decode_request_ids": [],
        "scheduled_encoder_request_ids": [],
        "preempted_request_ids": [],
    }


def _decode_schedule() -> dict[str, object]:
    return {
        "new_request_ids": [],
        "num_scheduled_tokens": {"0": 1},
        "total_num_scheduled_tokens": 1,
        "cached_request_ids": ["0"],
        "resumed_request_ids": [],
        "scheduled_spec_decode_request_ids": [],
        "scheduled_encoder_request_ids": [],
        "preempted_request_ids": [],
    }


def _cycle(role: str, role_index: int, prefill_ms: float, decode_ms: float) -> dict[str, object]:
    output_ids = [[10, 11, 12, 13]]
    return {
        "role": role,
        "role_index": role_index,
        "batch_size": 1,
        "prompt_tokens": 2,
        "prefill_elapsed_ms": prefill_ms,
        "prefill_schedule": _prefill_schedule(),
        "warmup_decode_steps": [
            {"step": 1, "elapsed_ms": decode_ms + 1.0, "schedule": _decode_schedule()}
        ],
        "measured_decode_steps": [
            {"step": 1, "elapsed_ms": decode_ms - 0.5, "schedule": _decode_schedule()},
            {"step": 2, "elapsed_ms": decode_ms + 0.5, "schedule": _decode_schedule()},
        ],
        "decode_step_ms": {"median": decode_ms},
        "cycle_elapsed_ms": prefill_ms + 3.0 * decode_ms + 1.0,
        "request_ttft_ms": {"median": prefill_ms + 0.25},
        "request_tpot_ms": {"median": decode_ms + 0.25},
        "request_latency_ms": {"median": prefill_ms + 3.0 * decode_ms},
        "outputs": [
            {
                "request_id": "0",
                "finished": True,
                "prompt_tokens": 2,
                "output_token_ids": output_ids[0],
            }
        ],
        "output_token_ids_sha256": verify.sha256_json(output_ids),
    }


def _result_payload(tmp_path: Path) -> dict[str, object]:
    fixture = tmp_path / "fixture.json"
    fixture.write_text("{}", encoding="utf-8")
    cycles = [
        _cycle("conditioning", 0, 10.0, 4.0),
        _cycle("measurement", 0, 9.0, 3.0),
        _cycle("measurement", 1, 11.0, 5.0),
        _cycle("measurement", 2, 10.0, 4.0),
    ]
    return {
        "schema": "vllm_p3_pinned_uva_phase_steps_v1",
        "status": "ok",
        "mechanism_class": verify.MECHANISMS["p3"],
        "config": {
            "batch_sizes": [1],
            "conditioning_repeats": 1,
            "measurement_repeats": 3,
            "warmup_decode_steps": 1,
            "measured_decode_steps": 2,
            "max_model_len": 32,
            "max_num_seqs": 1,
            "max_num_batched_tokens": 32,
            "enforce_eager": True,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "engine_core_multiprocessing": False,
            "frozen_prompt_token_ids": True,
            "disable_log_stats": True,
            "request_metrics_without_default_loggers": True,
            "seed": 0,
            "cpu_offload_gb": 0.0,
            "model": str(tmp_path / "model"),
            "fixture": str(fixture),
        },
        "fixture": {"sha256": verify.sha256_file(fixture)},
        "cycles": cycles,
        "phase_summary": {
            "1": {
                "prefill_step_ms": {"median": 10.0},
                "decode_repeat_median_ms": {"median": 4.0},
                "cycle_e2e_ms": {"median": 23.0},
                "request_ttft_p50_ms": {"median": 10.25},
                "request_tpot_p50_ms": {"median": 4.25},
                "request_latency_p50_ms": {"median": 22.0},
            }
        },
    }


def test_result_verifier_separates_prefill_decode_and_e2e(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(_result_payload(tmp_path)), encoding="utf-8")
    _result, summaries, hashes = verify.verify_result(
        path,
        mechanism=verify.MECHANISMS["p3"],
        batch_order=[1],
        prompt_lengths={1: [2]},
        conditioning_repeats=1,
        measurement_repeats=3,
        warmup_decode_steps=1,
        measured_decode_steps=2,
        max_model_len=32,
        max_num_seqs=1,
        max_num_batched_tokens=32,
        expected_model=tmp_path / "model",
        expected_fixture=tmp_path / "fixture.json",
        expected_fixture_sha256=verify.sha256_file(tmp_path / "fixture.json"),
        expected_cpu_offload_gb=0.0,
    )
    assert summaries[1] == {
        "prefill_step_ms": 10.0,
        "decode_step_ms": 4.0,
        "cycle_e2e_ms": 23.0,
        "request_ttft_p50_ms": 10.25,
        "request_tpot_p50_ms": 4.25,
        "request_latency_p50_ms": 22.0,
    }
    assert len(hashes[1]) == 64


def test_result_verifier_rejects_resumed_decode(tmp_path: Path) -> None:
    payload = _result_payload(tmp_path)
    payload["cycles"][1]["measured_decode_steps"][0]["schedule"][
        "resumed_request_ids"
    ] = ["0"]
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(verify.VerificationError):
        verify.verify_result(
            path,
            mechanism=verify.MECHANISMS["p3"],
            batch_order=[1],
            prompt_lengths={1: [2]},
            conditioning_repeats=1,
            measurement_repeats=3,
            warmup_decode_steps=1,
            measured_decode_steps=2,
            max_model_len=32,
            max_num_seqs=1,
            max_num_batched_tokens=32,
            expected_model=tmp_path / "model",
            expected_fixture=tmp_path / "fixture.json",
            expected_fixture_sha256=verify.sha256_file(tmp_path / "fixture.json"),
            expected_cpu_offload_gb=0.0,
        )
