#!/usr/bin/env python3
"""Validate and freeze real SGLang recorder counts for KT frequency placement.

The SGLang ``stat`` recorder writes a timestamped ``.pt`` file whose
``logical_count`` tensor has shape ``[recorded_forward_passes, layers, experts]``.
KTransformers sums the first dimension before choosing its global top-k GPU
experts.  This tool performs that sum once, checks that calibration/evaluation
fixtures are disjoint, derives the mask with the installed KT implementation,
and emits a compact, auditable artifact plus a JSON manifest.

Unlike ``cold_plan_to_kt_stats.py``, the output counts remain real activation
counts.  They are never replaced by an authoritative zero/one mask.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Any, torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(value, temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def fixture_metadata(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("requests"), list):
        raise ValueError(f"{path}: fixture must contain a requests list")
    embedded = payload.get("metadata")
    if not isinstance(embedded, dict):
        raise ValueError(f"{path}: fixture must contain metadata")
    indices = embedded.get("source_indices")
    if not isinstance(indices, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in indices
    ):
        raise ValueError(f"{path}: metadata.source_indices must be an integer list")
    prompts = [request.get("prompt") for request in payload["requests"]]
    if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError(f"{path}: every request must have a non-empty prompt")
    prompt_digest = hashlib.sha256(
        json.dumps(
            prompts, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "request_count": len(prompts),
        "source_indices": indices,
        "prompt_list_sha256": prompt_digest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze real SGLang recorder counts for KT frequency placement."
    )
    parser.add_argument("--input", required=True, type=Path, help="Raw recorder .pt")
    parser.add_argument("--output", required=True, type=Path, help="Normalized .pt")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--calibration-fixture", required=True, type=Path)
    parser.add_argument("--evaluation-fixture", required=True, type=Path)
    parser.add_argument("--gpu-experts-ratio", required=True, type=float)
    parser.add_argument("--expected-layers", type=int, default=48)
    parser.add_argument("--expected-experts", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.gpu_experts_ratio <= 1.0:
        raise SystemExit("--gpu-experts-ratio must be in [0, 1]")
    if args.expected_layers <= 0 or args.expected_experts <= 0:
        raise SystemExit("expected topology dimensions must be positive")

    import torch
    from kt_kernel import generate_gpu_experts_masks

    calibration = fixture_metadata(args.calibration_fixture)
    evaluation = fixture_metadata(args.evaluation_fixture)
    overlap = sorted(set(calibration["source_indices"]) & set(evaluation["source_indices"]))
    if overlap:
        raise SystemExit(
            "calibration and evaluation fixtures overlap at source indices "
            f"{overlap[:16]}"
        )

    raw = torch.load(args.input, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or "logical_count" not in raw:
        raise SystemExit("recorder artifact must be a dict containing logical_count")
    logical_count = raw["logical_count"]
    if not isinstance(logical_count, torch.Tensor):
        raise SystemExit("logical_count is not a torch.Tensor")
    if logical_count.dim() != 3:
        raise SystemExit(
            "logical_count must have shape [passes,layers,experts], got "
            f"{tuple(logical_count.shape)}"
        )
    expected_tail = (args.expected_layers, args.expected_experts)
    if tuple(logical_count.shape[1:]) != expected_tail:
        raise SystemExit(
            f"logical_count tail is {tuple(logical_count.shape[1:])}, "
            f"expected {expected_tail}"
        )
    if logical_count.shape[0] == 0:
        raise SystemExit("recorder artifact contains zero forward passes")
    if logical_count.is_floating_point():
        if not torch.isfinite(logical_count).all():
            raise SystemExit("logical_count contains non-finite values")
        if not torch.equal(logical_count, logical_count.round()):
            raise SystemExit("logical_count contains non-integral values")
    if bool((logical_count < 0).any()):
        raise SystemExit("logical_count contains negative values")

    aggregate = logical_count.to(torch.int64).sum(dim=0, keepdim=True)
    per_layer_slots = aggregate.sum(dim=(0, 2))
    empty_layers = torch.nonzero(per_layer_slots == 0, as_tuple=False).flatten().tolist()
    if empty_layers:
        raise SystemExit(f"recorder has no routed slots for layers {empty_layers}")
    total_slots = int(aggregate.sum().item())
    if total_slots <= 0:
        raise SystemExit("recorder has zero routed slots")

    total_experts = args.expected_layers * args.expected_experts
    hot_experts = int(total_experts * args.gpu_experts_ratio)
    mask = generate_gpu_experts_masks(aggregate.sum(dim=0).float(), hot_experts)
    if tuple(mask.shape) != expected_tail:
        raise SystemExit(f"KT mask has unexpected shape {tuple(mask.shape)}")
    mask = mask.to(device="cpu", dtype=torch.bool).contiguous()
    if int(mask.sum().item()) != hot_experts:
        raise SystemExit(
            f"KT selected {int(mask.sum().item())} hot experts, expected {hot_experts}"
        )

    counts_2d = aggregate[0]
    hot_slots = int(counts_2d[mask].sum().item())
    per_layer_hot_experts = mask.sum(dim=1).tolist()
    per_layer_hot_slots = (counts_2d * mask).sum(dim=1)
    per_layer_hit_rate = []
    for layer_hot, layer_total in zip(per_layer_hot_slots, per_layer_slots):
        per_layer_hit_rate.append(float(layer_hot.item() / layer_total.item()))

    mask_bytes = mask.to(torch.uint8).numpy().tobytes(order="C")
    mask_sha256 = hashlib.sha256(mask_bytes).hexdigest()
    normalized = {
        "logical_count": aggregate.contiguous(),
        "qualification_expected_gpu_mask": mask,
    }
    atomic_torch_save(args.output, normalized, torch)

    manifest = {
        "schema_version": 1,
        "selection_kind": "real_sglang_stat_recorder_counts_global_kt_topk",
        "binary_authoritative_mask_used_as_counts": False,
        "raw_recorder": {
            "path": str(args.input.resolve()),
            "sha256": sha256_file(args.input),
            "logical_count_shape": list(logical_count.shape),
            "logical_count_dtype": str(logical_count.dtype),
            "rank": raw.get("rank"),
        },
        "normalized_stats": {
            "path": str(args.output.resolve()),
            "sha256": sha256_file(args.output),
            "logical_count_shape": list(aggregate.shape),
            "logical_count_dtype": str(aggregate.dtype),
        },
        "calibration_fixture": calibration,
        "evaluation_fixture": evaluation,
        "fixtures_disjoint_by_source_index": True,
        "topology": {
            "layers": args.expected_layers,
            "experts_per_layer": args.expected_experts,
            "total_experts": total_experts,
        },
        "placement": {
            "gpu_experts_ratio_cli": format(args.gpu_experts_ratio, ".17g"),
            "hot_experts": hot_experts,
            "cold_experts": total_experts - hot_experts,
            "hot_experts_per_layer": per_layer_hot_experts,
            "cold_experts_per_layer": [
                args.expected_experts - value for value in per_layer_hot_experts
            ],
            "gpu_mask_uint8_sha256": mask_sha256,
        },
        "routing": {
            "recorded_forward_passes": int(logical_count.shape[0]),
            "total_routed_slots": total_slots,
            "hot_routed_slots": hot_slots,
            "cold_routed_slots": total_slots - hot_slots,
            "frequency_mask_hot_slot_fraction": hot_slots / total_slots,
            "frequency_mask_cold_slot_fraction": 1.0 - hot_slots / total_slots,
            "per_layer_hot_slot_fraction": per_layer_hit_rate,
            "per_layer_total_slots": [int(value) for value in per_layer_slots.tolist()],
        },
    }
    if not math.isclose(
        manifest["routing"]["frequency_mask_hot_slot_fraction"]
        + manifest["routing"]["frequency_mask_cold_slot_fraction"],
        1.0,
    ):
        raise AssertionError("internal routed-slot fraction error")
    atomic_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
