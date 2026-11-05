from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_module("e13_runner_test", "e13_kv_adaptive_runner.py")
aggregate = load_module("e13_aggregate_test", "aggregate_e13_kv_adaptive.py")
teacher_analysis = load_module(
    "e13_teacher_analysis_test", "analyze_e13_teacher_forced.py"
)


def test_percentile_linear_interpolation() -> None:
    assert runner.percentile([1.0, 2.0, 3.0], 50.0) == 2.0
    assert runner.percentile([0.0, 10.0], 95.0) == 9.5
    assert aggregate.percentile([0.0, 10.0], 99.0) == 9.9


def test_variation_is_across_cold_process_values() -> None:
    value = aggregate.variation([1.0, 2.0, 3.0])
    assert value == {
        "count": 3,
        "mean": 2.0,
        "sample_stddev": 1.0,
        "min": 1.0,
        "max": 3.0,
    }


def test_best_fixed_prefers_correctness_before_speed() -> None:
    rows = [
        {
            "arm": "fast_but_failed",
            "failed_requests_mean": 1.0,
            "rejected_requests_mean": 0.0,
            "successful_requests_mean": 9.0,
            "request_throughput_mean": 100.0,
            "e2e_p99_pooled_ms": 1.0,
        },
        {
            "arm": "complete",
            "failed_requests_mean": 0.0,
            "rejected_requests_mean": 0.0,
            "successful_requests_mean": 10.0,
            "request_throughput_mean": 10.0,
            "e2e_p99_pooled_ms": 10.0,
        },
    ]
    assert aggregate.select_best(rows)["arm"] == "complete"


def test_frozen_protocol_has_bidirectional_and_fair_fixed_design() -> None:
    protocol = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    targets = [int(item["target_blocks"]) for item in protocol["tiers"]]
    assert targets == [1364, 2730, 4095]
    assert protocol["runtime"]["backup_reserve_blocks"] == 16
    assert protocol["runtime"]["safety_blocks"] == 16
    assert protocol["runtime"]["scheduler_boundary_resize"] is True
    assert protocol["runtime"]["synchronized_tier_steps"] is True
    assert len(protocol["profiles"]["formal"]["seeds"]) == 3
    assert len(protocol["profiles"]["capacity"]["seeds"]) == 3
    assert protocol["profiles"]["capacity"]["scenario_names"] == [
        "capacity_envelope"
    ]
    assert sum(item["strategy"] == "fixed" for item in protocol["arms"]) == 3
    assert {item["name"] for item in protocol["scenarios"]} == {
        "low_pressure",
        "high_pressure",
        "dynamic_mixed",
        "burst_recovery",
        "capacity_envelope",
    }
    for fixture in protocol["fixtures"].values():
        assert (REPO_ROOT / fixture).is_file()


def test_distribution_empty_is_explicit() -> None:
    value = runner.distribution([])
    assert value["count"] == 0
    assert value["p95"] is None


def test_resize_summary_separates_attempts_from_real_switches() -> None:
    events = [
        {
            "direction": "hold",
            "wall_ms": 2.0,
            "requested_bytes": 0,
            "result": {"ok": False, "reason": "prepare_failed"},
        },
        {
            "direction": "grow",
            "wall_ms": 5.0,
            "requested_bytes": 1024,
            "result": {"ok": True, "reason": "applied"},
        },
        {
            "direction": "shrink",
            "wall_ms": 7.0,
            "requested_bytes": 512,
            "result": {"ok": True, "reason": "applied"},
        },
    ]

    value = runner.summarize_requests([], [], events)

    assert value["resize_count"] == 2
    assert value["resize_attempt_count"] == 3
    assert value["failed_resize_attempt_count"] == 1
    assert value["grow_count"] == 1
    assert value["shrink_count"] == 1
    assert value["resize_wall_ms"] == 12.0
    assert value["resize_response_ms"]["count"] == 2
    assert value["resize_response_ms"]["p50"] == 6.0
    assert value["resize_attempt_wall_ms"] == 14.0
    assert value["requested_kv_h2d_bytes"] == 1024
    assert value["requested_kv_d2h_bytes"] == 512


def test_teacher_forced_comparison_reports_top1_and_perplexity() -> None:
    import torch

    reference = {
        "target_token_ids": [0, 1],
        "logits": torch.tensor([[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]]),
    }
    candidate = {
        "target_token_ids": [0, 1],
        "logits": torch.tensor([[2.9, 1.1, 0.0], [0.0, 0.9, 1.0]]),
    }
    request, tokens = teacher_analysis.compare_request(
        repeat_index=0,
        seed=1,
        arm="fixed_k2",
        logical="scenario/epoch/c0/0",
        reference=reference,
        candidate=candidate,
    )
    assert request["token_positions"] == 2
    assert request["top1_equal_positions"] == 1
    assert request["top1_agreement_fraction"] == 0.5
    assert request["perplexity_relative_delta"] > 0
    assert tokens[0]["top1_equal"] is True
    assert tokens[1]["top1_equal"] is False
