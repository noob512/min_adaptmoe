#!/usr/bin/env python3
"""Convert custom hot/cold expert plans into KT frequency-stat tensors.

KTransformers' ``frequency`` placement selects a global top-k over a tensor
with shape ``[buffer, layers, experts]``.  The custom plans deliberately use a
balanced per-layer policy, so their raw pilot frequencies cannot be passed to
KT directly: doing that would generally select a different global top-k.  This
converter encodes the authoritative hot set as ones and the cold set as zeros.
With exactly ``k`` ones, KT's top-k is guaranteed to reproduce the plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torch


DEFAULT_NUM_LAYERS = 48
DEFAULT_NUM_EXPERTS = 128


class PlanValidationError(ValueError):
    """Raised when a plan cannot unambiguously define a KT hot-expert mask."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanValidationError(f"{field} must be an integer, got {value!r}")
    return value


def _read_expert_set(
    plan_path: Path,
    plan: dict[str, Any],
    key: str,
    num_layers: int,
    num_experts: int,
) -> set[tuple[int, int]]:
    records = plan.get(key)
    if not isinstance(records, list):
        raise PlanValidationError(
            f"{plan_path}: {key!r} must be a list (it is required for "
            "hot/cold complement validation)"
        )

    result: set[tuple[int, int]] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise PlanValidationError(
                f"{plan_path}: {key}[{index}] must be an object"
            )
        layer_id = _strict_int(
            record.get("layer_id"), f"{plan_path}:{key}[{index}].layer_id"
        )
        expert_id = _strict_int(
            record.get("expert_id"), f"{plan_path}:{key}[{index}].expert_id"
        )
        if not 0 <= layer_id < num_layers:
            raise PlanValidationError(
                f"{plan_path}: {key}[{index}] layer_id={layer_id} is outside "
                f"[0, {num_layers})"
            )
        if not 0 <= expert_id < num_experts:
            raise PlanValidationError(
                f"{plan_path}: {key}[{index}] expert_id={expert_id} is outside "
                f"[0, {num_experts})"
            )
        coordinate = (layer_id, expert_id)
        if coordinate in result:
            raise PlanValidationError(
                f"{plan_path}: duplicate {key} coordinate {coordinate}"
            )
        result.add(coordinate)
    return result


def _sample_coordinates(values: set[tuple[int, int]], limit: int = 8) -> str:
    ordered = sorted(values)
    suffix = " ..." if len(ordered) > limit else ""
    return f"{ordered[:limit]}{suffix}"


def _check_declared_counts(
    plan_path: Path,
    plan: dict[str, Any],
    num_layers: int,
    num_experts: int,
    cold_count: int,
    hot_count: int,
) -> None:
    metadata = plan.get("metadata", {})
    if metadata is not None and not isinstance(metadata, dict):
        raise PlanValidationError(f"{plan_path}: metadata must be an object")
    metadata = metadata or {}

    checks = (
        (metadata, "num_layers", num_layers, "metadata."),
        (metadata, "num_experts", num_experts, "metadata."),
        (metadata, "offload_plan_records", cold_count, "metadata."),
        (metadata, "hot_plan_records", hot_count, "metadata."),
        (plan, "offload_plan_records", cold_count, ""),
        (plan, "hot_plan_records", hot_count, ""),
    )
    for container, key, expected, prefix in checks:
        if key not in container:
            continue
        actual = _strict_int(container[key], f"{plan_path}:{prefix}{key}")
        if actual != expected:
            raise PlanValidationError(
                f"{plan_path}: {prefix}{key}={actual}, expected {expected}"
            )


def _kt_topk_mask(activation_freq: torch.Tensor, num_gpu_experts: int) -> torch.Tensor:
    """Mirror kt_kernel.experts_base.generate_gpu_experts_masks exactly."""
    if activation_freq.dim() != 2:
        raise PlanValidationError(
            f"internal error: activation_freq must be 2-D, got {activation_freq.shape}"
        )
    num_layers, num_experts = activation_freq.shape
    total_experts = num_layers * num_experts
    num_gpu_experts = max(0, min(num_gpu_experts, total_experts))
    if num_gpu_experts == 0:
        return torch.zeros(
            num_layers, num_experts, dtype=torch.bool, device="cpu"
        )

    flat_freq = activation_freq.view(-1).to(device="cpu")
    _, top_indices = torch.topk(
        flat_freq, k=num_gpu_experts, largest=True, sorted=False
    )
    mask = torch.zeros(total_experts, dtype=torch.bool, device="cpu")
    mask[top_indices] = True
    return mask.view(num_layers, num_experts)


def _infer_target_gib(plan_path: Path, metadata: dict[str, Any]) -> int | float | None:
    target = metadata.get("target_gib")
    if isinstance(target, (int, float)) and not isinstance(target, bool):
        return target
    match = re.fullmatch(r"plan_g(\d+(?:\.\d+)?)", plan_path.stem)
    if match is None:
        return None
    text = match.group(1)
    return int(text) if text.isdigit() else float(text)


def _prepare_plan(
    plan_path: Path, num_layers: int, num_experts: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    try:
        plan_bytes = plan_path.read_bytes()
    except OSError as exc:
        raise PlanValidationError(f"cannot read {plan_path}: {exc}") from exc
    try:
        plan = json.loads(plan_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PlanValidationError(f"invalid JSON in {plan_path}: {exc}") from exc
    if not isinstance(plan, dict):
        raise PlanValidationError(f"{plan_path}: top-level JSON value must be an object")

    cold = _read_expert_set(
        plan_path, plan, "offload_plan", num_layers, num_experts
    )
    hot = _read_expert_set(plan_path, plan, "hot_plan", num_layers, num_experts)
    overlap = cold & hot
    if overlap:
        raise PlanValidationError(
            f"{plan_path}: offload_plan and hot_plan overlap at "
            f"{_sample_coordinates(overlap)}"
        )

    expected = {
        (layer_id, expert_id)
        for layer_id in range(num_layers)
        for expert_id in range(num_experts)
    }
    missing = expected - (cold | hot)
    if missing:
        raise PlanValidationError(
            f"{plan_path}: hot/cold plans are not a complete complement; "
            f"missing {len(missing)} coordinates: {_sample_coordinates(missing)}"
        )
    unexpected = (cold | hot) - expected
    if unexpected:
        # Range validation should make this unreachable, but retain the invariant.
        raise PlanValidationError(
            f"{plan_path}: plan contains unexpected coordinates: "
            f"{_sample_coordinates(unexpected)}"
        )

    total_experts = num_layers * num_experts
    cold_count = len(cold)
    hot_count = len(hot)
    if cold_count + hot_count != total_experts:
        raise PlanValidationError(
            f"{plan_path}: cold ({cold_count}) + hot ({hot_count}) != "
            f"{total_experts}"
        )
    _check_declared_counts(
        plan_path,
        plan,
        num_layers,
        num_experts,
        cold_count,
        hot_count,
    )

    hot_mask = torch.zeros(num_layers, num_experts, dtype=torch.bool, device="cpu")
    for layer_id, expert_id in hot:
        hot_mask[layer_id, expert_id] = True

    # [buffer=1, layers, experts].  Ones authoritatively mean GPU-hot.
    logical_count = hot_mask.to(dtype=torch.int64).unsqueeze(0).contiguous()
    activation_freq = logical_count.sum(dim=0).float()
    reproduced_mask = _kt_topk_mask(activation_freq, hot_count)
    if not torch.equal(reproduced_mask, hot_mask):
        different = torch.nonzero(reproduced_mask != hot_mask, as_tuple=False)
        sample = different[:8].tolist()
        raise PlanValidationError(
            f"{plan_path}: KT top-k failed to reproduce hot_plan; "
            f"{different.shape[0]} differing coordinates, first={sample}"
        )

    ratio = hot_count / total_experts
    ratio_cli = format(ratio, ".17g")
    ratio_float = float(ratio_cli)
    ratio_recomputed_count = int(total_experts * ratio_float)
    if ratio_recomputed_count != hot_count:
        raise PlanValidationError(
            f"{plan_path}: --kt-gpu-experts-ratio {ratio_cli} maps to "
            f"{ratio_recomputed_count}, not {hot_count}, under KT's int() rule"
        )

    hot_per_layer = hot_mask.sum(dim=1).tolist()
    cold_per_layer = (~hot_mask).sum(dim=1).tolist()
    quotient, remainder = divmod(hot_count, num_layers)
    ceil_per_layer = math.ceil(hot_count / num_layers)
    metadata = plan.get("metadata") or {}
    entry: dict[str, Any] = {
        "input_plan": str(plan_path.resolve()),
        "input_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "target_gib": _infer_target_gib(plan_path, metadata),
        "cold_experts": cold_count,
        "hot_experts": hot_count,
        "total_experts": total_experts,
        "cold_experts_per_layer": cold_per_layer,
        "hot_experts_per_layer": hot_per_layer,
        "min_cold_experts_per_layer": min(cold_per_layer),
        "max_cold_experts_per_layer": max(cold_per_layer),
        "min_hot_experts_per_layer": min(hot_per_layer),
        "max_hot_experts_per_layer": max(hot_per_layer),
        "kt_gpu_experts_ratio": ratio,
        "kt_gpu_experts_ratio_cli": ratio_cli,
        "kt_ratio_int_recomputed_hot_experts": ratio_recomputed_count,
        "kt_num_gpu_experts": {
            "semantics": "integer per MoE layer; total is value * num_moe_layers",
            "exact": quotient if remainder == 0 else None,
            "quotient": quotient,
            "remainder_layers": remainder,
            "floor": quotient,
            "floor_total": quotient * num_layers,
            "ceil": ceil_per_layer,
            "ceil_total": ceil_per_layer * num_layers,
        },
        "logical_count_shape": list(logical_count.shape),
        "logical_count_dtype": str(logical_count.dtype),
        "logical_count_sum": int(logical_count.sum().item()),
        "kt_topk_mask_verified_equal_to_hot_plan": True,
    }
    return logical_count, entry


def _atomic_torch_save(value: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(value, temporary)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json_save(value: Any, output: Path) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert one or more balanced custom cold-plan JSON files to "
            "KTransformers logical_count .pt files."
        )
    )
    parser.add_argument("plans", nargs="+", type=Path, help="plan_g*.json files")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="manifest path (default: OUTPUT_DIR/manifest.json)",
    )
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_layers != DEFAULT_NUM_LAYERS or args.num_experts != DEFAULT_NUM_EXPERTS:
        raise SystemExit(
            "this comparison requires exactly 48 layers x 128 experts; got "
            f"{args.num_layers} x {args.num_experts}"
        )

    prepared: list[tuple[Path, torch.Tensor, dict[str, Any]]] = []
    output_names: set[str] = set()
    try:
        for plan_path in args.plans:
            output_name = f"{plan_path.stem}_logical_count.pt"
            if output_name in output_names:
                raise PlanValidationError(
                    f"multiple inputs map to the same output filename {output_name!r}"
                )
            output_names.add(output_name)
            logical_count, entry = _prepare_plan(
                plan_path, args.num_layers, args.num_experts
            )
            prepared.append((args.output_dir / output_name, logical_count, entry))
    except PlanValidationError as exc:
        raise SystemExit(f"error: {exc}") from exc

    entries: list[dict[str, Any]] = []
    for output_path, logical_count, entry in prepared:
        _atomic_torch_save({"logical_count": logical_count}, output_path)
        loaded = torch.load(output_path, map_location="cpu", weights_only=True)
        if not isinstance(loaded, dict) or not torch.equal(
            loaded.get("logical_count"), logical_count
        ):
            raise SystemExit(f"error: round-trip verification failed for {output_path}")
        entry["logical_count_path"] = str(output_path.resolve())
        entry["logical_count_sha256"] = _sha256(output_path)
        entry["launch_args"] = [
            "--kt-gpu-experts-ratio",
            entry["kt_gpu_experts_ratio_cli"],
            "--kt-expert-placement-strategy",
            "frequency",
            "--init-expert-location",
            str(output_path.resolve()),
        ]
        entries.append(entry)

    manifest = {
        "schema_version": 1,
        "num_layers": args.num_layers,
        "num_experts_per_layer": args.num_experts,
        "num_moe_layers": args.num_layers,
        "total_experts": args.num_layers * args.num_experts,
        "logical_count_semantics": (
            "shape [1, layers, experts]; 1 means the custom plan's GPU-hot "
            "expert and 0 means CPU-cold"
        ),
        "placement_requirements": {
            "kt_expert_placement_strategy": "frequency",
            "kt_enable_dynamic_expert_update": False,
            "reason": (
                "dynamic updates would mutate the initially identical hot-expert set"
            ),
        },
        "plans": entries,
    }
    manifest_path = args.manifest or args.output_dir / "manifest.json"
    _atomic_json_save(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
