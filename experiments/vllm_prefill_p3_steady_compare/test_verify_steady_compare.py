from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import verify_steady_compare as verify


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def batch(role: str, index: int, start: float, elapsed_ms: float) -> dict[str, object]:
    return {
        "role": role,
        "index": index,
        "start_epoch": start,
        "end_epoch": start + elapsed_ms / 1000.0,
        "elapsed_seconds": elapsed_ms / 1000.0,
        "requested_idle_ms": 0.0,
        "actual_idle_ms": None if role == "warmup" else 0.1,
        "completed_requests": 4,
        "prompt_tokens": 1226 if role == "warmup" else 1217,
        "output_tokens": 4,
        "aggregate_output_token_ids_sha256": (
            "b" * 64 if role == "warmup" else "a" * 64
        ),
        "ttft_ms": {"p50": elapsed_ms - 2.0},
    }


def result(arm: str, base_ms: float) -> dict[str, object]:
    cursor = 1000.0
    warmup = batch("warmup", 0, cursor, base_ms)
    cursor = float(warmup["end_epoch"]) + 0.0001
    calls: list[dict[str, object]] = []
    for role, count in (("conditioning", 3), ("measurement", 5)):
        for index in range(count):
            item = batch(role, index, cursor, base_ms + index)
            calls.append(item)
            cursor = float(item["end_epoch"]) + 0.0001
    return {
        "schema": "vllm_prefill_three_arm_steady_batches_v1",
        "status": "ok",
        "mechanism_class": verify.MECHANISMS[arm],
        "config": {
            "concurrency": 4,
            "warmup_requests": 4,
            "conditioning_batches": 3,
            "measurement_batches": 5,
            "disable_chunked_prefill": True,
            "output_tokens": 1,
            "warmup_output_tokens": 1,
            "require_prompt_token_ids": True,
            "inter_batch_idle_ms": 0.0,
            "max_model_len": 512,
            "max_num_batched_tokens": 2048,
            "max_num_seqs": 4,
        },
        "warmup": warmup,
        "calls": calls,
    }


def make_run(tmp_path: Path, order: str = "PCUCUP") -> Path:
    write_json(
        tmp_path / "run_config.json",
        {
            "order": order,
            "concurrency": 4,
            "warmup_requests": 4,
            "disable_chunked_prefill": True,
            "steady_conditioning_batches": 3,
            "steady_measurement_batches": 5,
        },
    )
    with (tmp_path / "execution_status.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("ordinal", "arm", "directory", "returncode"),
            delimiter="\t",
        )
        writer.writeheader()
        counts = {"p3": 0, "pinned": 0, "uva": 0}
        for ordinal, symbol in enumerate(order, 1):
            arm = verify.ARM_NAMES[symbol]
            counts[arm] += 1
            directory = f"{ordinal:02d}_{arm}"
            case = tmp_path / directory
            case.mkdir()
            (case / "environment.txt").write_text("", encoding="utf-8")
            base = {"p3": 130.0, "pinned": 110.0, "uva": 150.0}[arm]
            write_json(case / "result.json", result(arm, base + counts[arm]))
            writer.writerow(
                {
                    "ordinal": ordinal,
                    "arm": arm,
                    "directory": directory,
                    "returncode": 0,
                }
            )
    return tmp_path


def bypass_mechanism_artifacts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify.old, "verify_plan", lambda root: None)
    monkeypatch.setattr(verify.old, "verify_pinned", lambda case, env: {"ok": True})
    monkeypatch.setattr(verify.old, "verify_uva", lambda case, env: {"ok": True})
    monkeypatch.setattr(
        verify,
        "verify_p3",
        lambda case, env, summary, total_batches: {"ok": True},
    )


def test_process_medians_are_the_independent_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bypass_mechanism_artifacts(monkeypatch)
    output = verify.verify(make_run(tmp_path))
    assert output["status"] == "pass"
    assert output["process_blocks"] == ["PCU", "CUP"]
    assert output["elapsed_ms"]["p3"]["count"] == 2
    assert output["elapsed_ms"]["pinned"]["count"] == 2
    assert output["elapsed_ms"]["uva"]["count"] == 2
    assert output["measurement_contract"]["legacy_engine_geometry_match"] is True
    assert output["comparisons"]["p3_vs_pinned"]["delta_ms"] == pytest.approx(20.0)
    assert len(output["block_paired"]["p3_vs_uva"]["pairs"]) == 2


def test_rejects_non_permutation_process_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bypass_mechanism_artifacts(monkeypatch)
    with pytest.raises(verify.VerificationError, match="permutation"):
        verify.verify(make_run(tmp_path, order="PPC"))


def test_rejects_intentional_hot_window_idle(tmp_path: Path) -> None:
    payload = result("pinned", 110.0)
    payload["calls"][3]["requested_idle_ms"] = 1.0
    path = tmp_path / "result.json"
    write_json(path, payload)
    with pytest.raises(verify.VerificationError, match="intentional idle"):
        verify.verify_result(
            path,
            concurrency=4,
            warmup_requests=4,
            conditioning_batches=3,
            measurement_batches=5,
            disable_chunked_prefill=True,
        )
