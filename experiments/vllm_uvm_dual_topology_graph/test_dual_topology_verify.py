from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).with_name("verify_experiment.py")
SPEC = importlib.util.spec_from_file_location("dual_topology_verify", PATH)
assert SPEC is not None and SPEC.loader is not None
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


def sample(batch: int, value: float, *, audit: bool = False, digest: str = "same"):
    return {
        "case": f"b{batch}_{value}",
        "batch_size": batch,
        "audit_trace": audit,
        "output_hash": digest,
        "prefill_step_ms": value,
        "steady_decode_step_ms": value + 1,
        "first_decode_step_ms": value + 2,
        "cutover_wall_ms": value + 3,
        "resume_wall_ms": value + 4,
        "resume_ioctl_ms": value + 5,
        "repark_wall_ms": value + 6,
        "repark_ioctl_ms": value + 7,
        "cycle_e2e_ms": value + 8,
        "cuda_used_after_load_bytes": value + 9,
        "cuda_max_allocated_bytes": value + 10,
    }


def test_normalized_graph_entry_contract():
    assert verify.normalized_entries(
        [[4, True, False], [1, True, False]]
    ) == verify.REMOTE_ENTRIES
    assert verify.normalized_entries(
        [[1217, False, False], [314, False, False]]
    ) == verify.ALIAS_ENTRIES


def test_batched_transition_certificate():
    transition = {
        "processed_alias_count": 4_608,
        "transition_batch_count": 1,
        "batch_count_reported": True,
        "batch_count_valid": True,
        "ioctl_calls": 1,
        "tlb_invalidate_count": 1_152,
    }
    verify.validate_batched_transition(transition, "case", required=True)


def test_prebuilt_dual_view_certificate():
    verify.validate_prebuild_certificate(
        {
            "dual_view_block_count": 1_152,
            "dual_view_full_coverage": True,
            "tlb_invalidate_count": 18,
        },
        "case",
        enabled=True,
    )


def test_non_prebuilt_dual_view_certificate():
    verify.validate_prebuild_certificate(
        {
            "dual_view_block_count": 0,
            "dual_view_full_coverage": False,
            "tlb_invalidate_count": 1_152,
        },
        "case",
        enabled=False,
    )


def test_batched_transition_certificate_rejects_missing_batches():
    transition = {
        "processed_alias_count": 4_608,
        "transition_batch_count": 0,
        "batch_count_reported": False,
        "batch_count_valid": True,
        "ioctl_calls": 1,
        "tlb_invalidate_count": 1_152,
    }
    with pytest.raises(verify.ContractError, match="batch count absent"):
        verify.validate_batched_transition(transition, "case", required=True)


def test_non_prebuilt_grouped_transition_certificate():
    transition = {
        "processed_alias_count": 4_608,
        "transition_batch_count": 1,
        "transition_mode": "grouped",
        "batch_count_reported": True,
        "batch_count_valid": True,
        "ioctl_calls": 1,
        "tlb_invalidate_count": 1_152,
    }
    verify.validate_batched_transition(
        transition,
        "case",
        required=True,
        expected_grouped_transactions=1,
    )


def test_sequential_transition_certificate():
    transition = {
        "processed_alias_count": 4_608,
        "transition_batch_count": 4_608,
        "transition_mode": "sequential",
        "sequential_requested": True,
        "batch_count_reported": True,
        "batch_count_valid": True,
        "ioctl_calls": 1,
        "tlb_invalidate_count": 4_608,
    }
    verify.validate_batched_transition(
        transition,
        "case",
        required=True,
        transition_mode="sequential",
    )


def test_aggregate_uses_trace_free_samples_and_stable_hash():
    values = [
        sample(1, 99.0, audit=True),
        sample(1, 10.0),
        sample(1, 12.0),
        sample(4, 199.0, audit=True),
        sample(4, 20.0),
        sample(4, 24.0),
    ]
    result = verify.aggregate(values)
    assert result["1"]["sample_count"] == 2
    assert result["1"]["prefill_step_ms"]["median"] == 11.0
    assert result["1"]["prefill_step_ms"]["p95"] == pytest.approx(11.9)
    assert result["1"]["prefill_step_ms"]["p99"] == pytest.approx(11.98)
    assert len(result["1"]["prefill_step_ms"]["median_bootstrap_95_ci"]) == 2
    assert result["4"]["steady_decode_step_ms"]["median"] == 23.0


def test_aggregate_rejects_output_hash_drift():
    values = [sample(1, 10.0), sample(1, 11.0, digest="different"), sample(4, 20.0)]
    with pytest.raises(verify.ContractError, match="output hash drift"):
        verify.aggregate(values)


def test_aggregate_accepts_batch_one_smoke():
    result = verify.aggregate([sample(1, 10.0, audit=True)])
    assert set(result) == {"1"}
