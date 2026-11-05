#!/usr/bin/env python3
"""Freeze five calibration-cold experts per Qwen3 MoE layer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_FROZEN = HERE / "cold5_expert_plan.json"
EXPECTED_SOURCE_SHA256 = (
    "912732708a46291b73e70c93291758747f285b364437ce77f58d53501d072f81"
)
EXPECTED_CANONICAL_SHA256 = (
    "86e8cd516045ccb5b5d2f1c0cd140dfd8bc23311c3ca981ed69cb72206f1e5c4"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_plan(source: Path) -> dict[str, Any]:
    import torch

    source_sha256 = sha256_file(source)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"calibration input hash differs: {source_sha256}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    logical_count = payload["logical_count"]
    if tuple(logical_count.shape) != (11, 48, 128):
        raise RuntimeError(f"logical_count shape differs: {logical_count.shape}")

    counts = logical_count[:4].to(torch.int64).sum(0)
    selected: dict[str, list[int]] = {}
    selected_slots = 0
    zero_pairs = 0
    for layer_id, layer_counts in enumerate(counts):
        cold = sorted(
            (int(count), expert_id)
            for expert_id, count in enumerate(layer_counts)
        )[:5]
        expert_ids = sorted(expert_id for _, expert_id in cold)
        selected[str(layer_id)] = expert_ids
        selected_slots += sum(int(counts[layer_id, expert]) for expert in expert_ids)
        zero_pairs += sum(int(counts[layer_id, expert]) == 0 for expert in expert_ids)

    canonical = json.dumps(selected, sort_keys=True, separators=(",", ":"))
    canonical_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
    if canonical_sha256 != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(f"selected expert fingerprint differs: {canonical_sha256}")
    total_slots = int(counts.sum())
    return {
        "schema": "vllm_uvm_cold5_expert_plan_v1",
        "source": str(source.relative_to(REPO_ROOT)),
        "source_sha256": source_sha256,
        "logical_count_shape": list(logical_count.shape),
        "calibration_passes": [0, 1, 2, 3],
        "calibration_prefill_tokens": total_slots // (48 * 8),
        "selection_rule": (
            "five minimum summed calibration-prefill route counts; "
            "ties by expert_id; output ids sorted"
        ),
        "selected_by_layer": selected,
        "selected_by_layer_canonical_sha256": canonical_sha256,
        "selected_expert_count": sum(map(len, selected.values())),
        "calibration_selected_route_slots": selected_slots,
        "calibration_total_route_slots": total_slots,
        "calibration_selected_fraction": selected_slots / total_slots,
        "calibration_zero_count_pairs": zero_pairs,
    }


def validate_frozen_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema") != "vllm_uvm_cold5_expert_plan_v1":
        raise RuntimeError(f"unexpected frozen-plan schema: {path}")
    if plan.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"frozen-plan source hash differs: {path}")

    selected = plan.get("selected_by_layer")
    if not isinstance(selected, dict) or set(selected) != {
        str(layer_id) for layer_id in range(48)
    }:
        raise RuntimeError(f"frozen plan must contain exactly layers 0..47: {path}")
    for layer_id, expert_ids in selected.items():
        if (
            not isinstance(expert_ids, list)
            or len(expert_ids) != 5
            or expert_ids != sorted(set(expert_ids))
            or any(
                not isinstance(expert_id, int) or not 0 <= expert_id < 128
                for expert_id in expert_ids
            )
        ):
            raise RuntimeError(
                f"layer {layer_id} must contain five unique sorted experts in [0, 127]"
            )

    canonical = json.dumps(selected, sort_keys=True, separators=(",", ":"))
    canonical_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
    if canonical_sha256 != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(
            f"frozen selected-expert fingerprint differs: {canonical_sha256}"
        )
    if plan.get("selected_by_layer_canonical_sha256") != canonical_sha256:
        raise RuntimeError(f"embedded selected-expert fingerprint differs: {path}")
    if plan.get("selected_expert_count") != 240:
        raise RuntimeError(f"frozen selected-expert count differs: {path}")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help="explicit calibration tensor; omitted to validate the frozen plan",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--verify",
        type=Path,
        default=DEFAULT_FROZEN,
    )
    args = parser.parse_args()
    frozen = validate_frozen_plan(args.verify.resolve())
    if args.source:
        plan = build_plan(args.source.resolve())
        if plan != frozen:
            raise RuntimeError(f"frozen plan differs: {args.verify}")
    else:
        plan = frozen
    if args.output:
        args.output.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(plan, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
