#!/usr/bin/env python3
"""Freeze nested DeepSeek-V2-Lite cold-expert plans from a calibration trace."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


LAYER_IDS = tuple(range(1, 27))
EXPERT_COUNT = 64
EXPERT_BYTES = 3 * 2048 * 1408 * 2
TIERS = {"2g": 5, "4g": 10, "6g": 14, "8g": 19}
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")
TIER_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_tier_widths(value: str) -> dict[str, int]:
    tiers: dict[str, int] = {}
    for item in value.split(","):
        try:
            tier, raw_width = item.split("=", 1)
            width = int(raw_width)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid tier width {item!r}; expected LABEL=COUNT"
            ) from exc
        if not TIER_RE.fullmatch(tier):
            raise argparse.ArgumentTypeError(f"invalid tier label: {tier!r}")
        if tier in tiers:
            raise argparse.ArgumentTypeError(f"duplicate tier label: {tier!r}")
        if not 1 <= width <= EXPERT_COUNT:
            raise argparse.ArgumentTypeError(
                f"tier {tier!r} width must be in [1, {EXPERT_COUNT}]"
            )
        tiers[tier] = width
    if not tiers:
        raise argparse.ArgumentTypeError("at least one tier width is required")
    return tiers


def load_counts(
    path: Path, *, serving_start_step: int, prefill_step_count: int = 1
) -> tuple[dict[int, Counter[int]], dict[str, Any]]:
    if serving_start_step < 0:
        raise ValueError("serving_start_step must be non-negative")
    if prefill_step_count <= 0:
        raise ValueError("prefill_step_count must be positive")
    counts = {layer_id: Counter() for layer_id in LAYER_IDS}
    records = Counter()
    token_rows = Counter()
    topk_slots = Counter()
    step_layers: dict[int, set[int]] = {}
    step_num_tokens: dict[int, int] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        match = LAYER_RE.search(str(value.get("layer_name", "")))
        if match is None:
            raise ValueError(f"trace line {line_number} has an unknown layer name")
        layer_id = int(match.group(1))
        if layer_id not in counts:
            raise ValueError(f"trace line {line_number} has layer {layer_id}")
        step = int(value.get("step", -1))
        if step < serving_start_step:
            continue
        layers = step_layers.setdefault(step, set())
        if layer_id in layers:
            raise ValueError(
                f"trace step {step} repeats layer {layer_id} at line {line_number}"
            )
        layers.add(layer_id)
        raw = value.get("expert_token_counts")
        if not isinstance(raw, dict):
            raise ValueError(f"trace line {line_number} has no expert counts")
        row_total = 0
        for raw_expert, raw_count in raw.items():
            expert_id = int(raw_expert)
            count = int(raw_count)
            if not 0 <= expert_id < EXPERT_COUNT or count <= 0:
                raise ValueError(f"invalid expert count at trace line {line_number}")
            counts[layer_id][expert_id] += count
            row_total += count
        num_tokens = int(value.get("num_tokens", 0))
        top_k = int(value.get("top_k", 0))
        if num_tokens <= 0 or top_k <= 0 or row_total != num_tokens * top_k:
            raise ValueError(f"routing-slot total differs at trace line {line_number}")
        previous_tokens = step_num_tokens.setdefault(step, num_tokens)
        if previous_tokens != num_tokens:
            raise ValueError(f"trace step {step} has inconsistent token counts")
        # Batched decode has num_tokens equal to the number of live requests,
        # so num_tokens > 1 cannot distinguish it from prefill.  The formal
        # calibration disables chunked prefill and submits a strict batch;
        # consequently its first serving scheduler step is prefill and all
        # later scheduler steps are decode.
        phase = (
            "prefill"
            if step < serving_start_step + prefill_step_count
            else "decode"
        )
        records[phase] += 1
        token_rows[phase] += num_tokens
        topk_slots[phase] += row_total

    serving_steps = sorted(step_layers)
    if not serving_steps or serving_steps[0] != serving_start_step:
        raise ValueError(
            f"calibration trace does not start at serving step {serving_start_step}"
        )
    expected_steps = list(range(serving_start_step, serving_steps[-1] + 1))
    if serving_steps != expected_steps:
        missing_steps = sorted(set(expected_steps) - set(serving_steps))
        raise ValueError(f"calibration trace has missing serving steps: {missing_steps}")
    expected_layers = set(LAYER_IDS)
    incomplete = {
        step: sorted(expected_layers - layers)
        for step, layers in step_layers.items()
        if layers != expected_layers
    }
    if incomplete:
        raise ValueError(f"calibration trace has incomplete step layers: {incomplete}")
    missing = [layer_id for layer_id, values in counts.items() if not values]
    if missing:
        raise ValueError(f"calibration trace is missing layers: {missing}")
    if (
        records["prefill"] != prefill_step_count * len(LAYER_IDS)
        or records["decode"] == 0
    ):
        raise ValueError(f"unexpected calibration phase coverage: {dict(records)}")
    prefill_steps = serving_steps[:prefill_step_count]
    if any(step_num_tokens[step] <= 1 for step in prefill_steps):
        raise ValueError("formal prefill step did not contain multiple tokens")
    return counts, {
        "serving_start_step": serving_start_step,
        "prefill_step_count": prefill_step_count,
        "phase_classification": (
            "strict batch with chunked prefill disabled: first serving scheduler "
            "step(s) are prefill; remaining steps are batched decode"
        ),
        "scheduler_steps": {
            "prefill": prefill_step_count,
            "decode": len(serving_steps) - prefill_step_count,
            "total": len(serving_steps),
        },
        "prefill_num_tokens": [step_num_tokens[step] for step in prefill_steps],
        "decode_num_tokens_min": min(
            step_num_tokens[step] for step in serving_steps[prefill_step_count:]
        ),
        "decode_num_tokens_max": max(
            step_num_tokens[step] for step in serving_steps[prefill_step_count:]
        ),
        "record_counts": dict(records),
        "token_rows": dict(token_rows),
        "routed_slots": dict(topk_slots),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--serving-start-step",
        type=int,
        default=3,
        help="first real-request MoE step; earlier steps are vLLM load warmups",
    )
    parser.add_argument(
        "--prefill-step-count",
        type=int,
        default=1,
        help=(
            "number of initial serving scheduler steps that belong to prefill; "
            "formal strict-batch calibration disables chunked prefill and uses one"
        ),
    )
    parser.add_argument(
        "--tier-widths",
        type=parse_tier_widths,
        default=None,
        metavar="LABEL=COUNT,...",
        help=(
            "override the default cold experts per layer, for example "
            "2g=8,4g=13,6g=17,8g=22"
        ),
    )
    args = parser.parse_args()
    tiers = args.tier_widths or TIERS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts, coverage = load_counts(
        args.trace,
        serving_start_step=args.serving_start_step,
        prefill_step_count=args.prefill_step_count,
    )

    ranked = {
        layer_id: sorted(
            range(EXPERT_COUNT), key=lambda expert_id: (counts[layer_id][expert_id], expert_id)
        )
        for layer_id in LAYER_IDS
    }
    exact_names = [
        f"model.layers.{layer_id}.mlp.experts.{local_name}"
        for layer_id in LAYER_IDS
        for local_name in ("w13_weight", "w2_weight")
    ]
    names_path = args.output_dir / "exact_parameter_names.txt"
    names_path.write_text("\n".join(exact_names) + "\n", encoding="utf-8")

    manifest: dict[str, Any] = {
        "schema": "deepseek_v2_lite_cold_budget_matrix_v1",
        "selection_rule": (
            "minimum calibration prefill+decode routed slots per layer; "
            "ties by expert_id; tier plans are nested"
        ),
        "calibration_trace": str(args.trace.resolve()),
        "calibration_trace_sha256": sha256_file(args.trace),
        "coverage": coverage,
        "layer_ids": list(LAYER_IDS),
        "experts_per_layer": EXPERT_COUNT,
        "expert_main_bytes": EXPERT_BYTES,
        "tier_width_source": "command_line" if args.tier_widths else "default",
        "full_parameter_count": len(exact_names),
        "full_expert_main_bytes": len(LAYER_IDS) * EXPERT_COUNT * EXPERT_BYTES,
        "tiers": {},
    }
    for tier, width in tiers.items():
        selected = {
            str(layer_id): sorted(ranked[layer_id][:width]) for layer_id in LAYER_IDS
        }
        selected_count = len(LAYER_IDS) * width
        selected_bytes = selected_count * EXPERT_BYTES
        selected_slots = sum(
            counts[layer_id][expert_id]
            for layer_id in LAYER_IDS
            for expert_id in selected[str(layer_id)]
        )
        plan = {
            "schema": "vllm_uvm_cold_expert_plan_v1",
            "model": "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "tier": tier,
            "selected_by_layer": selected,
            "selected_expert_count": selected_count,
            "selected_main_bytes": selected_bytes,
            "selected_main_gib": selected_bytes / 2**30,
            "calibration_selected_route_slots": selected_slots,
            "calibration_total_route_slots": sum(
                sum(layer_counts.values()) for layer_counts in counts.values()
            ),
            "calibration_trace_sha256": manifest["calibration_trace_sha256"],
        }
        plan_path = args.output_dir / f"cold_plan_{tier}.json"
        atomic_json(plan_path, plan)
        manifest["tiers"][tier] = {
            "cold_experts_per_layer": width,
            "selected_expert_count": selected_count,
            "selected_main_bytes": selected_bytes,
            "selected_main_gib": selected_bytes / 2**30,
            "plan": plan_path.name,
            "plan_sha256": sha256_file(plan_path),
            "calibration_selected_route_slots": selected_slots,
        }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
