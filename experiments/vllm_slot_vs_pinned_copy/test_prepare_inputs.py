"""CPU-only regression tests for exact and sparse Qwen plan generation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("prepare_inputs.py")
SPEC = importlib.util.spec_from_file_location("vllm_slot_prepare_inputs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare_inputs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_inputs)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generate(tmp_path: Path, *args: str) -> tuple[dict, dict, Path]:
    output = tmp_path / "inputs"
    assert prepare_inputs.main(["--output-dir", str(output), *args]) == 0
    plan_path = output / "exact_l0_l3_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    summary = json.loads((output / "input_summary.json").read_text(encoding="utf-8"))
    return plan, summary, output


def _coordinates(records: list[dict[str, int]]) -> set[tuple[int, int]]:
    return {(record["layer_id"], record["expert_id"]) for record in records}


def test_default_outputs_remain_byte_for_byte_compatible(tmp_path: Path) -> None:
    plan, summary, output = _generate(tmp_path)

    assert _sha256(output / "exact_l0_l3_plan.json") == (
        "a0e5695409eb1da7b99993a0590aa2e33f62d8637e1bf255e7a1406d4b67fddf"
    )
    assert _sha256(output / "exact_parameter_names.txt") == (
        "473d3f28f1f70788fa6dc08e333108950d8d89a423973daa7dfb74751ab46eb9"
    )
    assert len(plan["offload_plan"]) == 4 * 128
    assert summary["one_layer_bank_bytes"] == 576 * 1024 * 1024
    assert summary["one_layer_bank_slots"] == 1152


def test_sparse_48_by_12_plan_is_complete_equal_and_rotating(tmp_path: Path) -> None:
    plan, summary, output = _generate(
        tmp_path,
        "--cold-layers",
        "0-47",
        "--cold-experts-per-layer",
        "12",
        "--cold-expert-rotation-step",
        "1",
    )
    cold = _coordinates(plan["offload_plan"])
    hot = _coordinates(plan["hot_plan"])
    universe = {(layer, expert) for layer in range(48) for expert in range(128)}

    assert len(cold) == 48 * 12
    assert len(hot) == 48 * (128 - 12)
    assert not cold & hot
    assert cold | hot == universe
    for layer in range(48):
        expected = {(expert + layer) % 128 for expert in range(12)}
        assert {expert for candidate, expert in cold if candidate == layer} == expected

    assert summary["region_count"] == 48 * 12 * 9
    assert summary["logical_bytes"] == 48 * 12 * 4_718_592
    assert summary["one_layer_bank_bytes"] == 12 * 4_718_592
    assert summary["one_layer_bank_slots"] == 12 * 9
    assert summary["dual_bank_bytes"] == 2 * 12 * 4_718_592
    assert summary["dual_bank_slots"] == 2 * 12 * 9
    assert summary["exact_parameter_copy_supported"] is False
    assert (output / "exact_parameter_names.txt").read_bytes() == b""


def test_explicit_ids_wrap_and_layer_spelling_is_canonical(tmp_path: Path) -> None:
    common = (
        "--cold-expert-ids",
        "0,7,127",
        "--cold-expert-rotation-step",
        "2",
    )
    plan_a, summary_a, output_a = _generate(
        tmp_path / "a", "--cold-layers", "0,2,5", *common
    )
    plan_b, summary_b, output_b = _generate(
        tmp_path / "b",
        "--cold-layers",
        "5,0,2",
        "--cold-expert-ids",
        "127,7,0",
        "--cold-expert-rotation-step",
        "2",
    )

    assert (output_a / "exact_l0_l3_plan.json").read_bytes() == (
        output_b / "exact_l0_l3_plan.json"
    ).read_bytes()
    assert summary_a["plan_sha256"] == summary_b["plan_sha256"]
    selected = plan_a["metadata"]["selected_by_layer"]
    assert selected == {
        "0": [0, 7, 127],
        "2": [1, 2, 9],
        "5": [3, 4, 11],
    }


@pytest.mark.parametrize(
    "args",
    [
        ("--cold-layers", "48"),
        ("--cold-layers", "3-1"),
        ("--cold-layers", "1,1"),
        ("--cold-expert-ids", "128"),
        ("--cold-expert-ids", "7,7"),
        ("--cold-expert-ids", "9-3"),
        ("--cold-experts-per-layer", "0"),
        ("--cold-experts-per-layer", "129"),
        ("--cold-expert-ids", "1", "--cold-experts-per-layer", "2"),
    ],
)
def test_invalid_selection_fails_closed(tmp_path: Path, args: tuple[str, ...]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        prepare_inputs.main(["--output-dir", str(tmp_path / "bad"), *args])
    assert exc_info.value.code == 2
