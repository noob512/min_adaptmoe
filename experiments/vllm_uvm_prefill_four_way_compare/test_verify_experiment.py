from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).with_name("verify_experiment.py")
SPEC = importlib.util.spec_from_file_location("prefill_four_way_verify", PATH)
assert SPEC and SPEC.loader
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)

RUNNER_PATH = Path(__file__).with_name("benchmark_compare_steps.py")
RUNNER_SPEC = importlib.util.spec_from_file_location(
    "prefill_four_way_runner", RUNNER_PATH
)
assert RUNNER_SPEC and RUNNER_SPEC.loader
runner = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(runner)


@pytest.mark.parametrize(
    ("symbol", "execution", "mode", "sizes"),
    [
        ("N", "eager", "NONE", []),
        ("A", "eager", "NONE", []),
        ("G", "full_graph_all", "FULL", [1, 4, 314, 1217]),
        ("R", "eager", "NONE", []),
        ("C", "full_graph_decode", "FULL_DECODE_ONLY", [1, 4]),
        ("F", "full_graph_all", "FULL", [1, 4, 314, 1217]),
    ],
)
def test_expected_graph_config(symbol, execution, mode, sizes):
    assert verify.expected_graph_config(symbol) == (execution, mode, sizes)


def test_aggregate_uses_process_phase_medians():
    results = []
    for symbol, samples in (("A", [10.0, 12.0, 11.0]), ("G", [8.0, 9.0, 7.0])):
        for sample in samples:
            results.append(
                {
                    "arm": symbol,
                    "phase_summary": {
                        "1": {
                            "prefill_step_ms": {"median": sample},
                            "decode_repeat_median_ms": {"median": sample + 1},
                            "cycle_e2e_ms": {"median": sample + 2},
                            "prefill_tokens_per_second": 100.0,
                            "decode_tokens_per_second": 20.0,
                        }
                    },
                }
            )
    aggregated = verify.aggregate(results)
    assert aggregated["A"]["1"]["prefill_step_ms"]["median"] == 11.0
    assert aggregated["G"]["1"]["prefill_step_ms"]["median"] == 8.0
    comparison = verify.ratio(aggregated, "G", "A", "1", "prefill_step_ms")
    assert comparison["G_vs_A_change_percent"] == pytest.approx(-27.272727)


def test_graph_contract_requires_prefill_replay_only_for_g():
    good = {
        "cycles": [
            {
                "prefill_graph_wrapper_delta": {"full_replay": 1},
                "warmup_decode_steps": [
                    {"graph_wrapper_delta": {"full_replay": 1}}
                ],
                "measured_decode_steps": [
                    {"graph_wrapper_delta": {"full_replay": 1}}
                ],
            }
        ]
    }
    contract = runner.graph_contract(good, "G")
    assert contract["prefill_full_replay_count"] == 1
    assert contract["decode_full_replay_count"] == 2
    with pytest.raises(runner.ContractError, match="prefill Graph replay"):
        runner.graph_contract(good, "C")


def complete_run_config(order: str = "NAGRCF"):
    return {
        "schema": "vllm_uvm_original_va_graph_run_v1",
        "mode": "qualification",
        "order": order,
        "required_arms": "NAGRCF",
        "extended_prefill_matrix": True,
        "attention_backend": "TRITON_ATTN",
        "cudagraph_mode": "PER_ARM",
        "cudagraph_capture_sizes": None,
        "arm_configs": {
            symbol: verify.expected_arm_config(symbol) for symbol in set(order)
        },
    }


def test_run_config_is_explicitly_per_arm():
    order, required = verify.validate_run_config(complete_run_config())
    assert order == "NAGRCF"
    assert required == "NAGRCF"
    assert verify.expected_arm_config("G")["cudagraph_capture_sizes"] == [
        1,
        4,
        314,
        1217,
    ]


def test_run_config_rejects_global_decode_only_description():
    config = complete_run_config()
    config["cudagraph_mode"] = "FULL_DECODE_ONLY"
    config["cudagraph_capture_sizes"] = [1, 4]
    with pytest.raises(verify.VerificationError, match="global Graph mode"):
        verify.validate_run_config(config)


def test_run_config_rejects_cross_arm_contract_mix():
    config = complete_run_config()
    config["arm_configs"]["G"] = copy.deepcopy(config["arm_configs"]["C"])
    with pytest.raises(verify.VerificationError, match="arm G differs"):
        verify.validate_run_config(config)


def test_status_identity_rejects_cross_arm_directory_mix():
    rows = []
    for index, symbol in enumerate("NAGRCF", 1):
        rows.append(
            {
                "ordinal": str(index),
                "block": "1",
                "symbol": symbol,
                "arm": verify.ARM_NAMES[symbol],
                "batch_order": "1,4",
                "directory": f"{index:02d}_{verify.ARM_NAMES[symbol]}",
                "returncode": "0",
            }
        )
    verify.validate_status_identity(rows, "NAGRCF", "NAGRCF")
    rows[2]["directory"] = rows[1]["directory"]
    with pytest.raises(verify.VerificationError, match="directory differs"):
        verify.validate_status_identity(rows, "NAGRCF", "NAGRCF")
