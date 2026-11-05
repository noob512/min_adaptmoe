"""Synthetic artifact tests for the P3-prefill/remote-decode verifier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import pytest


MODULE_PATH = Path(__file__).with_name("verify_compare.py")
SPEC = importlib.util.spec_from_file_location("prefill_p3_decode_remote_verify", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def cold_experts(layer: int) -> tuple[int, ...]:
    base = (0, 7, 15, 23, 31, 47, 63, 79, 95, 111, 119, 127)
    return tuple(sorted((value + layer) % VERIFY.EXPERTS for value in base))


def make_plan(root: Path) -> None:
    cold = {(layer, expert) for layer in VERIFY.LAYERS for expert in cold_experts(layer)}
    universe = {(layer, expert) for layer in VERIFY.LAYERS for expert in range(VERIFY.EXPERTS)}
    write_json(
        root / "inputs" / "exact_l0_l3_plan.json",
        {
            "metadata": {
                "envelope_bytes": 524_288,
                "regions_per_expert": VERIFY.REGIONS_PER_EXPERT,
                "cold_experts": VERIFY.TOTAL_EXPERTS,
                "cold_regions": VERIFY.TOTAL_REGIONS,
                "actual_offload_main_bytes": VERIFY.TOTAL_BYTES,
                "shared_hbm_bank_bytes": VERIFY.LAYER_BYTES,
                "shared_hbm_bank_slots": VERIFY.REGIONS_PER_LAYER,
            },
            "offload_plan": [
                {"layer_id": layer, "expert_id": expert} for layer, expert in sorted(cold)
            ],
            "hot_plan": [
                {"layer_id": layer, "expert_id": expert}
                for layer, expert in sorted(universe - cold)
            ],
            "warm_plan": [],
        },
    )


def p3_env(hybrid: bool) -> dict[str, str]:
    return {
        "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE": "slot_restore",
        "VLLM_UVM_SLOT_FORMAL_RESTORE_ALL_COLD": "1",
        "VLLM_UVM_SLOT_DIRECT_INIT_BANK": "1",
        "VLLM_UVM_SLOT_DIRECT_REUSE_BANK": "1",
        "VLLM_UVM_SLOT_DIRECT_PRECOMPUTED_LANES": "1",
        "VLLM_UVM_SLOT_DIRECT_LAYER_TRANSITION": "1",
        "VLLM_UVM_SLOT_DIRECT_SHARED_PUSH": "1",
        "VLLM_UVM_SLOT_DIRECT_64K_PTE": "1",
        "VLLM_UVM_SLOT_DIRECT_REGISTERED_PLAN": "1",
        "VLLM_UVM_SLOT_DIRECT_DUAL_BANK_EVENT": "1",
        "VLLM_UVM_SLOT_DIRECT_CUDA_FENCE": "1",
        "VLLM_UVM_SLOT_DIRECT_CACHE_BYTES": str(2 * VERIFY.LAYER_BYTES),
        "VLLM_UVM_SLOT_DIRECT_MAX_SLOTS": str(2 * VERIFY.REGIONS_PER_LAYER),
        "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ": "1" if hybrid else "0",
        "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_BATCH": "1" if hybrid else "0",
    }


def write_env(path: Path, values: dict[str, str]) -> None:
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")


def result(mechanism: str, output_hash: str = "a" * 64) -> dict[str, Any]:
    summary = {"count": 1, "p50": 10.0, "mean": 10.0, "min": 10.0, "max": 10.0}
    return {
        "status": "ok",
        "mechanism_class": mechanism,
        "config": {
            "concurrency": 1,
            "output_tokens": VERIFY.OUTPUT_TOKENS,
            "warmup_requests": 0,
            "max_num_seqs": 1,
            "disable_chunked_prefill": True,
            "enforce_eager": True,
            "require_prompt_token_ids": True,
        },
        "measurement": {
            "completed_requests": 1,
            "prompt_tokens": 512,
            "output_tokens": VERIFY.OUTPUT_TOKENS,
            "ttft_ms": summary,
            "tpot_ms": summary,
            "request_latency_ms": summary,
            "aggregate_output_token_ids_sha256": output_hash,
        },
    }


def registration() -> dict[str, Any]:
    return {
        "operation": "register_layer_plans",
        "ok": True,
        "layer_count": len(VERIFY.LAYERS),
        "lane_count": VERIFY.REGIONS_PER_LAYER,
        "bank_count": 2,
    }


def restore(index: int) -> dict[str, Any]:
    layer = index % len(VERIFY.LAYERS)
    return {
        "operation": "restore_layer",
        "ok": True,
        "layer_id": layer,
        "expert_ids": list(cold_experts(layer)),
        "restore": {
            "ok": True,
            "plan_execute": True,
            "requested_bytes": VERIFY.LAYER_BYTES,
            "accepted_bytes": VERIFY.LAYER_BYTES,
            "success_count": VERIFY.REGIONS_PER_LAYER,
            "remap_count": VERIFY.REGIONS_PER_LAYER,
            "failure_count": 0,
            "conflict_count": 0,
            "poisoned": False,
            "cuda_fence_accepted": True,
            "cuda_completion_signaled": True,
            "cuda_stream_wait_enqueued": True,
            "ioctl_wall_ns": 1_000_000,
        },
    }


def write_stats(case_dir: Path, executes: int) -> None:
    (case_dir / "latency_stats.before.txt").write_text(
        "slot_plan_execute_calls 10\nslot_direct_errors 4\n", encoding="utf-8"
    )
    (case_dir / "latency_stats.after.txt").write_text(
        f"slot_plan_execute_calls {10 + executes}\nslot_direct_errors 4\n",
        encoding="utf-8",
    )


def make_tree(root: Path) -> Path:
    make_plan(root)
    mechanisms = {
        "hybrid": "vllm_sparse512k_p3_prefill_custom_remote_decode",
        "p3": "vllm_sparse512k_pure_p3_all_token_forwards",
        "remote": "custom_sparse512k_remote_read_prefill_decode",
    }
    write_json(
        root / "run_config.json",
        {
            "schema": "vllm_prefill_p3_decode_remote_run_v1",
            "order": "HPU",
            "layers": len(VERIFY.LAYERS),
            "cold_experts_per_layer": VERIFY.COLD_EXPERTS_PER_LAYER,
            "regions_per_layer": VERIFY.REGIONS_PER_LAYER,
            "bytes_per_layer": VERIFY.LAYER_BYTES,
            "total_regions": VERIFY.TOTAL_REGIONS,
            "total_bytes": VERIFY.TOTAL_BYTES,
            "concurrency": 1,
            "output_tokens": VERIFY.OUTPUT_TOKENS,
            "max_num_seqs": 1,
            "chunked_prefill": False,
            "enforce_eager": True,
            "hybrid_remote_batch_max_ranges": VERIFY.REMOTE_BATCH_MAX_RANGES,
            "hybrid_remote_batch_ioctl_calls": VERIFY.REMOTE_BATCH_IOCTL_CALLS,
            "hybrid_remote_mapped_bytes": VERIFY.TOTAL_BYTES,
            "mechanism_classes": mechanisms,
        },
    )
    (root / "execution_status.tsv").write_text(
        "ordinal\tarm\tdirectory\treturncode\n"
        "1\thybrid\t01_hybrid\t0\n"
        "2\tp3\t02_p3\t0\n"
        "3\tremote\t03_remote\t0\n",
        encoding="utf-8",
    )

    hybrid = root / "01_hybrid"
    hybrid.mkdir()
    write_json(hybrid / "result.json", result(mechanisms["hybrid"]))
    write_env(hybrid / "environment.txt", p3_env(True))
    write_jsonl(
        hybrid / "slot_trace.jsonl",
        [registration()]
        + [restore(index) for index in range(VERIFY.PREFILL_TRANSITIONS)]
        + [{"operation": "release", "scope": "session", "ok": True}],
    )
    write_stats(hybrid, VERIFY.PREFILL_TRANSITIONS)
    write_json(
        hybrid / "transition_result.json",
        {
            "schema": "vllm_prefill_p3_decode_remote_transition_v1",
            "ok": True,
            "reason": "ok",
            "module_count": len(VERIFY.LAYERS),
            "expert_count": VERIFY.TOTAL_EXPERTS,
            "region_count": VERIFY.TOTAL_REGIONS,
            "close_contract_verified": True,
            "remote_batch": True,
            "remote_batch_ioctl_calls": VERIFY.REMOTE_BATCH_IOCTL_CALLS,
            "remote_batch_total_ioctl_calls": VERIFY.REMOTE_BATCH_IOCTL_CALLS + 1,
            "remote_batch_fd_probe_ioctl_calls": 1,
            "remote_batch_fd_candidate_attempts": 2,
            "remote_mapped_bytes": VERIFY.TOTAL_BYTES,
            "sync_ns": 10,
            "close_ns": 20,
            "remote_map_ns": 30,
            "wall_ns": 60,
        },
    )

    pure = root / "02_p3"
    pure.mkdir()
    write_json(pure / "result.json", result(mechanisms["p3"]))
    write_env(pure / "environment.txt", p3_env(False))
    write_jsonl(
        pure / "slot_trace.jsonl",
        [registration()] + [restore(index) for index in range(VERIFY.PURE_P3_TRANSITIONS)],
    )
    write_stats(pure, VERIFY.PURE_P3_TRANSITIONS)

    remote = root / "03_remote"
    remote.mkdir()
    write_json(remote / "result.json", result(mechanisms["remote"]))
    write_env(
        remote / "environment.txt",
        {
            "VLLM_CPU_OFFLOAD_BACKEND": "uva",
            "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE": "remote_read",
            "VLLM_UVM_INIT_WEIGHT_OFFLOAD_ENABLE": "1",
            "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_READ": "0",
            "VLLM_UVM_PREFILL_P3_DECODE_REMOTE_BATCH": "0",
        },
    )
    write_jsonl(remote / "slot_trace.jsonl", [])
    write_stats(remote, 0)
    write_json(
        remote / "runtime_remote_placement_manifest.json",
        {
            "status": "ok",
            "complete": True,
            "remote_expert_count": VERIFY.TOTAL_EXPERTS,
            "remote_region_count": VERIFY.TOTAL_REGIONS,
            "remote_region_bytes": VERIFY.TOTAL_BYTES,
            "offloaded_expert_count": 0,
            "release_time_remote_mapping_verified": True,
        },
    )
    return root


def test_verify_accepts_exact_phase_contract(tmp_path: Path) -> None:
    summary = VERIFY.verify(make_tree(tmp_path))
    assert summary["status"] == "pass"
    assert summary["measurement_contract"]["chunked_prefill"] is False
    assert summary["cases"][0]["evidence"]["transitions"] == 48
    assert summary["cases"][1]["evidence"]["transitions"] == 384
    assert summary["cases"][0]["evidence"]["remote_regions"] == 5184
    assert summary["cases"][0]["evidence"]["remote_batch_ioctl_calls"] == 81
    assert summary["cases"][0]["evidence"]["remote_mapped_bytes"] == 2_717_908_992
    assert summary["hybrid_remote_cutover_contract"] == {
        "batch": True,
        "max_ranges_per_ioctl": 64,
        "ioctl_calls": 81,
        "module_count": 48,
        "expert_count": 576,
        "region_count": 5_184,
        "remote_mapped_bytes": 2_717_908_992,
    }
    assert summary["output_token_ids_sha256"] == "a" * 64


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda root: write_jsonl(
                root / "01_hybrid" / "slot_trace.jsonl",
                [registration()]
                + [restore(index) for index in range(VERIFY.PREFILL_TRANSITIONS - 1)]
                + [{"operation": "release", "scope": "session", "ok": True}],
            ),
            "expected 48 P3 transitions",
        ),
        (
            lambda root: write_json(
                root / "01_hybrid" / "transition_result.json",
                {
                    **json.loads((root / "01_hybrid" / "transition_result.json").read_text()),
                    "module_count": len(VERIFY.LAYERS) - 1,
                },
            ),
            "hybrid transition mismatch: module_count",
        ),
        (
            lambda root: write_json(
                root / "01_hybrid" / "transition_result.json",
                {
                    **json.loads((root / "01_hybrid" / "transition_result.json").read_text()),
                    "expert_count": VERIFY.TOTAL_EXPERTS - 1,
                },
            ),
            "hybrid transition mismatch: expert_count",
        ),
        (
            lambda root: write_json(
                root / "01_hybrid" / "transition_result.json",
                {
                    **json.loads((root / "01_hybrid" / "transition_result.json").read_text()),
                    "region_count": VERIFY.TOTAL_REGIONS - 1,
                },
            ),
            "hybrid transition mismatch: region_count",
        ),
        (
            lambda root: write_json(
                root / "01_hybrid" / "transition_result.json",
                {
                    **json.loads((root / "01_hybrid" / "transition_result.json").read_text()),
                    "remote_batch_ioctl_calls": VERIFY.REMOTE_BATCH_IOCTL_CALLS - 1,
                },
            ),
            "hybrid transition mismatch: remote_batch_ioctl_calls",
        ),
        (
            lambda root: write_json(
                root / "01_hybrid" / "transition_result.json",
                {
                    **json.loads((root / "01_hybrid" / "transition_result.json").read_text()),
                    "remote_mapped_bytes": VERIFY.TOTAL_BYTES - 1,
                },
            ),
            "hybrid transition mismatch: remote_mapped_bytes",
        ),
        (
            lambda root: write_jsonl(
                root / "02_p3" / "slot_trace.jsonl",
                [registration()]
                + [restore(index) for index in range(VERIFY.PURE_P3_TRANSITIONS - 1)],
            ),
            "expected 384 P3 transitions",
        ),
        (
            lambda root: write_json(
                root / "03_remote" / "runtime_remote_placement_manifest.json",
                {
                    **json.loads(
                        (root / "03_remote" / "runtime_remote_placement_manifest.json").read_text()
                    ),
                    "release_time_remote_mapping_verified": False,
                },
            ),
            "remote placement mismatch",
        ),
        (
            lambda root: write_json(
                root / "03_remote" / "result.json",
                result("custom_sparse512k_remote_read_prefill_decode", "b" * 64),
            ),
            "output token hashes differ",
        ),
    ],
)
def test_verify_rejects_contract_violation(
    tmp_path: Path,
    mutate: Callable[[Path], None],
    match: str,
) -> None:
    root = make_tree(tmp_path)
    mutate(root)
    with pytest.raises(VERIFY.VerificationError, match=match):
        VERIFY.verify(root)
