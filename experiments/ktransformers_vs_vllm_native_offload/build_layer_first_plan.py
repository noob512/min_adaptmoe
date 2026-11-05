#!/usr/bin/env python3
"""Build a deterministic layer-first whole-expert CPU-cold plan.

The plan schema matches ``cold_plan_to_kt_stats.py``: ``offload_plan`` is the
authoritative CPU-cold set and ``hot_plan`` is its complete GPU-hot complement.
Selection is the lexicographic prefix of ``(layer_id, expert_id)`` over the
fixed Qwen3-30B-A3B topology (48 layers x 128 experts).

This builder deliberately works at *whole expert* granularity.  It therefore
does not claim to reproduce a vLLM ``cpu_offload_gb`` boundary that offloads
only one tensor (for example, ``w13`` but not ``w2``) of a layer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from pathlib import Path
from typing import Any


NUM_LAYERS = 48
NUM_EXPERTS_PER_LAYER = 128
EXPERT_BYTES = 4_718_592
EXPERT_SCALE_BYTES = 1_152
EXPERT_BYTES_INCLUDING_SCALES = EXPERT_BYTES + EXPERT_SCALE_BYTES
GIB_BYTES = 1 << 30
TOTAL_EXPERTS = NUM_LAYERS * NUM_EXPERTS_PER_LAYER
TOTAL_EXPERT_BYTES = TOTAL_EXPERTS * EXPERT_BYTES


def _nonnegative_decimal(text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"invalid GiB value: {text!r}") from exc
    if not value.is_finite() or value < 0:
        raise argparse.ArgumentTypeError(
            f"target GiB must be finite and non-negative, got {text!r}"
        )
    return value


def _bounded_cold_experts(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"cold expert count must be an integer, got {text!r}"
        ) from exc
    if not 0 <= value <= TOTAL_EXPERTS:
        raise argparse.ArgumentTypeError(
            f"cold expert count must be in [0, {TOTAL_EXPERTS}], got {value}"
        )
    return value


def _json_number(value: Decimal) -> int | float:
    integral = value.to_integral_value()
    return int(integral) if value == integral else float(value)


def _coordinate_sha256(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            f"{record['layer_id']}:{record['expert_id']}\n".encode("ascii")
        )
    return digest.hexdigest()


def _atomic_json_write(value: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic 48x128 layer-first whole-expert cold plan "
            "for the KTransformers versus vLLM native-offload comparison."
        )
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--target-gib",
        type=_nonnegative_decimal,
        help=(
            "CPU-cold main-weight budget in GiB. The selected whole-expert "
            "count is floor(target_bytes / 4718592)."
        ),
    )
    target.add_argument(
        "--cold-experts",
        type=_bounded_cold_experts,
        help="Exact number of whole experts to mark CPU-cold.",
    )
    parser.add_argument(
        "--output",
        "--output-plan",
        dest="output",
        required=True,
        type=Path,
        help="Output plan JSON path.",
    )
    parser.add_argument(
        "--label",
        help="Optional human-readable comparison label stored in metadata.",
    )
    return parser.parse_args()


def build_plan(
    *,
    target_gib: Decimal | None,
    requested_cold_experts: int | None,
    label: str | None = None,
) -> dict[str, Any]:
    if (target_gib is None) == (requested_cold_experts is None):
        raise ValueError("exactly one target mode is required")

    if target_gib is not None:
        target_bytes_decimal = target_gib * GIB_BYTES
        target_bytes = int(
            target_bytes_decimal.to_integral_value(rounding=ROUND_FLOOR)
        )
        if target_bytes > TOTAL_EXPERT_BYTES:
            raise ValueError(
                f"target {target_gib} GiB is {target_bytes} bytes, exceeding "
                f"all {TOTAL_EXPERTS} experts ({TOTAL_EXPERT_BYTES} bytes, "
                f"{TOTAL_EXPERT_BYTES / GIB_BYTES:.6f} GiB)"
            )
        cold_experts = target_bytes // EXPERT_BYTES
        target_mode = "target_gib_floor"
    else:
        assert requested_cold_experts is not None
        cold_experts = requested_cold_experts
        target_bytes = cold_experts * EXPERT_BYTES
        target_mode = "exact_cold_experts"

    hot_experts = TOTAL_EXPERTS - cold_experts
    actual_offload_bytes = cold_experts * EXPERT_BYTES
    actual_offload_bytes_including_scales = (
        cold_experts * EXPERT_BYTES_INCLUDING_SCALES
    )
    actual_offload_gib = actual_offload_bytes / GIB_BYTES

    offload_plan: list[dict[str, Any]] = []
    hot_plan: list[dict[str, Any]] = []
    cold_per_layer: list[int] = []

    for layer_id in range(NUM_LAYERS):
        layer_cold = min(
            NUM_EXPERTS_PER_LAYER,
            max(0, cold_experts - layer_id * NUM_EXPERTS_PER_LAYER),
        )
        cold_per_layer.append(layer_cold)
        for expert_id in range(NUM_EXPERTS_PER_LAYER):
            global_index = layer_id * NUM_EXPERTS_PER_LAYER + expert_id
            base_record = {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "selection_index": global_index,
            }
            if global_index < cold_experts:
                offload_plan.append(
                    {**base_record, "reason": "layer_first_cpu_cold"}
                )
            else:
                hot_plan.append(
                    {**base_record, "reason": "layer_first_hot_complement"}
                )

    if len(offload_plan) != cold_experts:
        raise AssertionError(
            f"internal cold count mismatch: {len(offload_plan)} != {cold_experts}"
        )
    if len(hot_plan) != hot_experts:
        raise AssertionError(
            f"internal hot count mismatch: {len(hot_plan)} != {hot_experts}"
        )

    full_cold_layer_count, partial_count = divmod(
        cold_experts, NUM_EXPERTS_PER_LAYER
    )
    partial_cold_layer = (
        {
            "layer_id": full_cold_layer_count,
            "cold_experts": partial_count,
            "cold_expert_id_begin_inclusive": 0,
            "cold_expert_id_end_exclusive": partial_count,
        }
        if partial_count
        else None
    )
    hot_per_layer = [NUM_EXPERTS_PER_LAYER - count for count in cold_per_layer]
    cold_count_by_layer = {
        str(layer_id): count for layer_id, count in enumerate(cold_per_layer)
    }
    hot_count_by_layer = {
        str(layer_id): count for layer_id, count in enumerate(hot_per_layer)
    }

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "generator": "build_layer_first_plan.py",
        "label": label,
        "policy": "layer_first_whole_expert_cpu_cold_v1",
        "selection_policy": "lexicographic_prefix_by_layer_then_expert",
        "selection_uses_routing_data": False,
        "target_mode": target_mode,
        "target_gib": _json_number(target_gib) if target_gib is not None else None,
        "target_gib_text": str(target_gib) if target_gib is not None else None,
        "requested_cold_experts": requested_cold_experts,
        "target_bytes": target_bytes,
        "target_expert_records_floor": cold_experts,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS_PER_LAYER,
        "num_experts_per_layer": NUM_EXPERTS_PER_LAYER,
        "total_expert_records": TOTAL_EXPERTS,
        "expert_bytes": EXPERT_BYTES,
        "expert_scale_bytes": EXPERT_SCALE_BYTES,
        "expert_bytes_including_scales": EXPERT_BYTES_INCLUDING_SCALES,
        "expert_payload": "w13_weight+w2_weight",
        "expert_granularity": "whole_expert",
        "actual_offload_bytes": actual_offload_bytes,
        "actual_offload_gib": actual_offload_gib,
        "actual_offload_bytes_including_scales": (
            actual_offload_bytes_including_scales
        ),
        "actual_offload_gib_including_scales": (
            actual_offload_bytes_including_scales / GIB_BYTES
        ),
        "unused_target_bytes": target_bytes - actual_offload_bytes,
        "offload_plan_records": cold_experts,
        "hot_plan_records": hot_experts,
        "warm_plan_records": 0,
        "prefetch_plan_records": hot_experts,
        "full_cold_layer_count": full_cold_layer_count,
        "full_cold_layers": list(range(full_cold_layer_count)),
        "partial_cold_layer": partial_cold_layer,
        "cold_experts_per_layer": cold_count_by_layer,
        "hot_experts_per_layer": hot_count_by_layer,
        "min_cold_experts_per_layer": min(cold_per_layer),
        "max_cold_experts_per_layer": max(cold_per_layer),
        "min_hot_experts_per_layer": min(hot_per_layer),
        "max_hot_experts_per_layer": max(hot_per_layer),
        "offload_coordinate_sha256": _coordinate_sha256(offload_plan),
        "hot_coordinate_sha256": _coordinate_sha256(hot_plan),
        "hot_is_exact_complement": True,
        "record_order": "ascending_layer_id_then_expert_id",
        "comparison_limitation": (
            "whole-expert placement cannot reproduce a native cpu_offload_gb "
            "boundary that places only part of an expert on CPU"
        ),
    }

    return {
        "metadata": metadata,
        "offload_plan": offload_plan,
        "hot_plan": hot_plan,
        "warm_plan": [],
        "prefetch_plan": hot_plan,
        "offload_plan_records": cold_experts,
        "hot_plan_records": hot_experts,
        "warm_plan_records": 0,
        "prefetch_plan_records": hot_experts,
    }


def main() -> int:
    args = parse_args()
    try:
        plan = build_plan(
            target_gib=args.target_gib,
            requested_cold_experts=args.cold_experts,
            label=args.label,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    _atomic_json_write(plan, args.output)
    summary = {
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "target_mode": plan["metadata"]["target_mode"],
        "label": plan["metadata"]["label"],
        "target_gib": plan["metadata"]["target_gib"],
        "cold_experts": plan["offload_plan_records"],
        "hot_experts": plan["hot_plan_records"],
        "actual_offload_bytes": plan["metadata"]["actual_offload_bytes"],
        "actual_offload_gib": plan["metadata"]["actual_offload_gib"],
        "actual_offload_bytes_including_scales": plan["metadata"][
            "actual_offload_bytes_including_scales"
        ],
        "actual_offload_gib_including_scales": plan["metadata"][
            "actual_offload_gib_including_scales"
        ],
        "full_cold_layer_count": plan["metadata"]["full_cold_layer_count"],
        "partial_cold_layer": plan["metadata"]["partial_cold_layer"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
