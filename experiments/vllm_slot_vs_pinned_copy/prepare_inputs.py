#!/usr/bin/env python3
"""Create exact or sparse Qwen3 MoE placement manifests.

With no sparse-selection arguments this intentionally emits the historical
L0-L3/all-128-expert files byte-for-byte.  Sparse plans retain an equal cold
expert count in every selected layer so they can share the fixed bank lanes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path


LAYERS = 48
EXPERTS = 128
COLD_LAYERS = range(4)
W13_BYTES = 384 * 1024 * 1024
W2_BYTES = 192 * 1024 * 1024
LAYER_BYTES = W13_BYTES + W2_BYTES
EXPERT_BYTES = LAYER_BYTES // EXPERTS
ENVELOPE_BYTES = 512 * 1024
REGIONS_PER_EXPERT = EXPERT_BYTES // ENVELOPE_BYTES
_INDEX_TOKEN = re.compile(r"^(\d+)(?:-(\d+))?$")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_index_spec(value: str, *, name: str, upper_bound: int) -> tuple[int, ...]:
    """Parse an inclusive ``0,2,5-8`` index specification."""
    text = str(value).strip()
    if not text:
        raise argparse.ArgumentTypeError(f"{name} must not be empty")

    parsed: list[int] = []
    seen: set[int] = set()
    for raw_token in text.split(","):
        token = raw_token.strip()
        match = _INDEX_TOKEN.fullmatch(token)
        if match is None:
            raise argparse.ArgumentTypeError(
                f"{name} contains invalid token {token!r}"
            )
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if first > last:
            raise argparse.ArgumentTypeError(
                f"{name} range {token!r} is descending"
            )
        if last >= upper_bound:
            raise argparse.ArgumentTypeError(
                f"{name} index {last} is outside 0-{upper_bound - 1}"
            )
        for index in range(first, last + 1):
            if index in seen:
                raise argparse.ArgumentTypeError(
                    f"{name} repeats index {index}"
                )
            seen.add(index)
            parsed.append(index)

    # Index specifications describe sets. Canonical ordering makes equivalent
    # spellings produce identical registered-plan bytes and cache keys.
    return tuple(sorted(parsed))


def positive_expert_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "cold experts per layer must be an integer"
        ) from exc
    if not 1 <= count <= EXPERTS:
        raise argparse.ArgumentTypeError(
            f"cold experts per layer must be between 1 and {EXPERTS}"
        )
    return count


def build_cold_experts_by_layer(
    cold_layers: Sequence[int],
    base_expert_ids: Sequence[int],
    *,
    rotation_step: int,
) -> dict[int, tuple[int, ...]]:
    """Return one equal-sized, optionally rotated expert set per layer."""
    base = tuple(int(expert_id) for expert_id in base_expert_ids)
    if not base or len(base) != len(set(base)):
        raise ValueError("base cold expert IDs must be non-empty and unique")
    if any(expert_id < 0 or expert_id >= EXPERTS for expert_id in base):
        raise ValueError(f"cold expert IDs must be inside 0-{EXPERTS - 1}")

    return {
        int(layer): tuple(
            sorted(
                (expert_id + ordinal * int(rotation_step)) % EXPERTS
                for expert_id in base
            )
        )
        for ordinal, layer in enumerate(cold_layers)
    }


def build_outputs(
    cold_experts_by_layer: dict[int, tuple[int, ...]],
    *,
    base_expert_ids: Sequence[int],
    rotation_step: int,
) -> tuple[dict[str, object], list[str], dict[str, object]]:
    cold_layers = tuple(sorted(cold_experts_by_layer))
    cold_keys = {
        (layer, expert)
        for layer in cold_layers
        for expert in cold_experts_by_layer[layer]
    }
    all_keys = {
        (layer, expert)
        for layer in range(LAYERS)
        for expert in range(EXPERTS)
    }
    hot_keys = all_keys - cold_keys
    equal_counts = {len(experts) for experts in cold_experts_by_layer.values()}
    if not cold_layers or len(equal_counts) != 1:
        raise ValueError("every cold layer must contain the same non-zero expert count")
    experts_per_layer = equal_counts.pop()
    if not experts_per_layer:
        raise ValueError("every cold layer must contain at least one cold expert")

    full_tensor_copy_equivalent = experts_per_layer == EXPERTS
    exact_names = (
        [
            f"model.layers.{layer}.mlp.experts.{weight}"
            for layer in cold_layers
            for weight in ("w13_weight", "w2_weight")
        ]
        if full_tensor_copy_equivalent
        else []
    )
    total_bytes = len(cold_keys) * EXPERT_BYTES
    total_regions = len(cold_keys) * REGIONS_PER_EXPERT
    bank_bytes = experts_per_layer * EXPERT_BYTES
    bank_slots = experts_per_layer * REGIONS_PER_EXPERT
    legacy = (
        cold_layers == tuple(COLD_LAYERS)
        and experts_per_layer == EXPERTS
        and cold_keys
        == {
            (layer, expert)
            for layer in COLD_LAYERS
            for expert in range(EXPERTS)
        }
    )

    if legacy:
        metadata: dict[str, object] = {
            "schema_version": 1,
            "policy": "formal_l0_l3_all_experts_exact_main_weights_v1",
            "actual_offload_main_bytes": total_bytes,
            "actual_offload_main_gib": total_bytes / 1024**3,
            "expert_main_bytes": EXPERT_BYTES,
            "envelope_bytes": ENVELOPE_BYTES,
            "regions_per_expert": REGIONS_PER_EXPERT,
            "cold_layers": list(COLD_LAYERS),
            "cold_experts": len(cold_keys),
            "cold_regions": total_regions,
            "copy_parameter_names": exact_names,
            "shared_hbm_bank_bytes": LAYER_BYTES,
            "shared_hbm_bank_slots": LAYER_BYTES // ENVELOPE_BYTES,
        }
    else:
        selected_by_layer = {
            str(layer): list(cold_experts_by_layer[layer]) for layer in cold_layers
        }
        metadata = {
            "schema_version": 1,
            "policy": "formal_sparse_equal_cold_experts_rotating_v1",
            "actual_offload_main_bytes": total_bytes,
            "actual_offload_main_gib": total_bytes / 1024**3,
            "expert_main_bytes": EXPERT_BYTES,
            "envelope_bytes": ENVELOPE_BYTES,
            "regions_per_expert": REGIONS_PER_EXPERT,
            "cold_layers": list(cold_layers),
            "cold_experts": len(cold_keys),
            "cold_experts_per_layer": experts_per_layer,
            "cold_expert_base_ids": list(base_expert_ids),
            "cold_expert_rotation_step": int(rotation_step),
            "selected_by_layer": selected_by_layer,
            "cold_regions": total_regions,
            # Packed Qwen expert slices are not independently named parameters.
            # An exact-name pinned-copy arm would copy the whole w13/w2 tensors,
            # so leave this empty instead of silently claiming equivalence.
            "copy_parameter_names": exact_names,
            "exact_parameter_copy_supported": full_tensor_copy_equivalent,
            "shared_hbm_bank_bytes": bank_bytes,
            "shared_hbm_bank_slots": bank_slots,
        }

    plan: dict[str, object] = {
        "metadata": metadata,
        "offload_plan": [
            {"layer_id": layer, "expert_id": expert}
            for layer, expert in sorted(cold_keys)
        ],
        "hot_plan": [
            {"layer_id": layer, "expert_id": expert}
            for layer, expert in sorted(hot_keys)
        ],
        "warm_plan": [],
    }
    summary: dict[str, object] = {
        "parameter_count": len(exact_names),
        "cold_expert_count": len(cold_keys),
        "region_count": total_regions,
        "logical_bytes": total_bytes,
        "logical_gib": total_bytes / 1024**3,
        "one_layer_bank_bytes": bank_bytes,
        "one_layer_bank_slots": bank_slots,
    }
    if not legacy:
        summary.update(
            {
                "policy": metadata["policy"],
                "cold_layers": list(cold_layers),
                "cold_experts_per_layer": experts_per_layer,
                "cold_expert_rotation_step": int(rotation_step),
                "selected_by_layer": metadata["selected_by_layer"],
                "exact_parameter_copy_supported": full_tensor_copy_equivalent,
                "dual_bank_bytes": 2 * bank_bytes,
                "dual_bank_slots": 2 * bank_slots,
            }
        )
    return plan, exact_names, summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cold-layers",
        metavar="SPEC",
        help="inclusive layer IDs, for example 0-47 or 0,2,5-8",
    )
    expert_selector = parser.add_mutually_exclusive_group()
    expert_selector.add_argument(
        "--cold-expert-ids",
        metavar="SPEC",
        help="base cold expert IDs, for example 0-10 or 0,7,31,64,127",
    )
    expert_selector.add_argument(
        "--cold-experts-per-layer",
        type=positive_expert_count,
        metavar="COUNT",
        help="select base expert IDs 0 through COUNT-1 in every cold layer",
    )
    parser.add_argument(
        "--cold-expert-rotation-step",
        type=int,
        default=0,
        metavar="STEP",
        help="rotate the base expert IDs by STEP for each successive cold layer",
    )
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cold_layers = (
            parse_index_spec(
                args.cold_layers,
                name="cold layers",
                upper_bound=LAYERS,
            )
            if args.cold_layers is not None
            else tuple(COLD_LAYERS)
        )
        # Rotation is a property of model-layer order. parse_index_spec()
        # canonicalizes explicit sets before assigning rotation ordinals.
        if args.cold_expert_ids is not None:
            base_expert_ids = parse_index_spec(
                args.cold_expert_ids,
                name="cold expert IDs",
                upper_bound=EXPERTS,
            )
        elif args.cold_experts_per_layer is not None:
            base_expert_ids = tuple(range(args.cold_experts_per_layer))
        else:
            base_expert_ids = tuple(range(EXPERTS))
        cold_experts_by_layer = build_cold_experts_by_layer(
            cold_layers,
            base_expert_ids,
            rotation_step=args.cold_expert_rotation_step,
        )
        plan, exact_names, summary = build_outputs(
            cold_experts_by_layer,
            base_expert_ids=base_expert_ids,
            rotation_step=args.cold_expert_rotation_step,
        )
    except (argparse.ArgumentTypeError, ValueError) as exc:
        parser.error(str(exc))

    assert EXPERT_BYTES == 4_718_592
    assert REGIONS_PER_EXPERT == 9
    plan_path = args.output_dir / "exact_l0_l3_plan.json"
    names_path = args.output_dir / "exact_parameter_names.txt"
    plan_path.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    names_path.write_text(
        "\n".join(exact_names) + ("\n" if exact_names else ""),
        encoding="utf-8",
    )
    summary = {
        **summary,
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256(plan_path),
        "exact_names": str(names_path.resolve()),
        "exact_names_sha256": sha256(names_path),
    }
    summary_path = args.output_dir / "input_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
