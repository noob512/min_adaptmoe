#!/usr/bin/env python3
"""Freeze a custom-UVM cold-expert plan from a qualified KT GPU mask.

The output uses the complement of ``qualification_expected_gpu_mask``.  It
therefore matches KTransformers by (layer, expert) coordinate rather than by a
nominal GiB flag.  Loading is CPU-only; this script never initializes CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


QWEN_EXPERT_MAIN_BYTES = 4_718_592
QWEN_EXPERT_SCALE_BYTES = 1_152
EXPECTED_KT_COMMIT = "eb9b70c4115cff151ace2cae5b0fc9db3690e31e"
EXPECTED_SGLANG_COMMIT = "bc7f0058fafe6c05dc6ad52d6b1c854af2a673b2"
EXPECTED_MODEL_SNAPSHOT = "d206ba732169f29bb77fbf80fc2c4b81d4d30782"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coordinate_sha256(coordinates: list[tuple[int, int]]) -> str:
    payload = "\n".join(f"{layer_id}:{expert_id}" for layer_id, expert_id in coordinates)
    if payload:
        payload += "\n"
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a qualified KT expected GPU mask to a custom cold plan."
    )
    parser.add_argument("--stats", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--qualification-snapshot",
        required=True,
        type=Path,
        help="Passed KT qualification snapshot containing exact runtime verification.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--empty-output",
        type=Path,
        help="Also write a topology-identical empty cold plan for custom-empty ABBA.",
    )
    parser.add_argument("--expected-layers", type=int, default=48)
    parser.add_argument("--expected-experts", type=int, default=128)
    parser.add_argument("--expert-main-bytes", type=int, default=QWEN_EXPERT_MAIN_BYTES)
    parser.add_argument("--expert-scale-bytes", type=int, default=QWEN_EXPERT_SCALE_BYTES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_layers <= 0 or args.expected_experts <= 0:
        raise SystemExit("expected topology must be positive")
    if args.expert_main_bytes <= 0 or args.expert_scale_bytes < 0:
        raise SystemExit("expert byte sizes are invalid")
    if not all(
        path.is_file()
        for path in (args.stats, args.manifest, args.qualification_snapshot)
    ):
        raise SystemExit("stats, manifest and qualification snapshot must exist")
    if args.empty_output is not None and args.empty_output.resolve() == args.output.resolve():
        raise SystemExit("--output and --empty-output must be different paths")

    # Import after argument/file validation. map_location=cpu and the explicit
    # CUDA guard make accidental GPU initialization a hard error.
    import torch

    if torch.cuda.is_initialized():
        raise SystemExit("CUDA was unexpectedly initialized before stats loading")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    qualification = json.loads(
        args.qualification_snapshot.read_text(encoding="utf-8")
    )
    recorded_stats_sha = (
        manifest.get("normalized_stats", {}).get("sha256")
        if isinstance(manifest, dict)
        else None
    )
    actual_stats_sha = sha256_file(args.stats)
    if recorded_stats_sha != actual_stats_sha:
        raise SystemExit(
            f"stats SHA mismatch: manifest={recorded_stats_sha!r} actual={actual_stats_sha}"
        )
    if manifest.get("fixtures_disjoint_by_source_index") is not True:
        raise SystemExit("qualification manifest does not prove disjoint fixtures")
    if (
        not isinstance(qualification, dict)
        or qualification.get("experiment")
        != "ktransformers_documentation_qualification"
        or qualification.get("status") != "ok"
        or qualification.get("failures") != []
        or qualification.get("exact_output_hashes_equal_across_threads") is not True
    ):
        raise SystemExit("KT qualification snapshot did not pass its exact gate")
    embedded_manifest = qualification.get("frequency_stats_manifest")
    if not isinstance(embedded_manifest, dict):
        raise SystemExit("KT qualification snapshot lacks its frequency manifest")
    if embedded_manifest.get("normalized_stats", {}).get("sha256") != actual_stats_sha:
        raise SystemExit("KT qualification snapshot refers to different frequency stats")
    if (
        embedded_manifest.get("placement", {}).get("gpu_mask_uint8_sha256")
        != manifest.get("placement", {}).get("gpu_mask_uint8_sha256")
    ):
        raise SystemExit("KT qualification snapshot/manifest GPU-mask SHA mismatch")

    exact_cases: list[dict[str, Any]] = []
    for case in qualification.get("cases", []):
        if not isinstance(case, dict) or case.get("mode") != "exact":
            continue
        metadata = case.get("metadata")
        runtime = case.get("runtime_verification")
        if not isinstance(metadata, dict) or not isinstance(runtime, dict):
            continue
        checks = runtime.get("checks")
        if not isinstance(checks, dict):
            continue
        if (
            runtime.get("status") != "ok"
            or runtime.get("failures") != []
            or metadata.get("ktransformers_commit") != EXPECTED_KT_COMMIT
            or metadata.get("sglang_commit") != EXPECTED_SGLANG_COMMIT
            or Path(str(metadata.get("model", ""))).name != EXPECTED_MODEL_SNAPSHOT
            or metadata.get("frequency_stats", {}).get("sha256")
            != actual_stats_sha
            or checks.get("runtime_mask_layers") != args.expected_layers
            or checks.get("runtime_hot_experts")
            != int(manifest.get("placement", {}).get("hot_experts", -1))
            or checks.get("server_frequency_stats_loaded_without_fallback") is not True
            or checks.get("dynamic_expert_update_log_hits") != 0
            or checks.get("command:not:--kt-enable-dynamic-expert-update") is not True
            or checks.get("command:--kt-max-deferred-experts-per-token", {}).get(
                "actual"
            )
            != "0"
            or checks.get("command:--kt-gpu-prefill-token-threshold", {}).get(
                "actual"
            )
            != "0"
        ):
            raise SystemExit(
                f"KT exact runtime provenance failed for case {case.get('name')!r}"
            )
        exact_cases.append(case)
    if not exact_cases:
        raise SystemExit("KT qualification snapshot has no verified exact case")

    artifact = torch.load(args.stats, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict):
        raise SystemExit("stats artifact is not a dictionary")
    counts = artifact.get("logical_count")
    gpu_mask = artifact.get("qualification_expected_gpu_mask")
    if not isinstance(counts, torch.Tensor) or not isinstance(gpu_mask, torch.Tensor):
        raise SystemExit("stats artifact lacks logical_count or expected GPU mask")
    expected_shape = (args.expected_layers, args.expected_experts)
    if tuple(gpu_mask.shape) != expected_shape:
        raise SystemExit(f"GPU mask shape {tuple(gpu_mask.shape)} != {expected_shape}")
    if tuple(counts.shape) == (1, *expected_shape):
        counts_2d = counts[0]
    elif tuple(counts.shape) == expected_shape:
        counts_2d = counts
    else:
        raise SystemExit(f"logical_count shape {tuple(counts.shape)} is invalid")
    if bool((counts_2d < 0).any()):
        raise SystemExit("logical_count contains negative values")

    mask = gpu_mask.to(device="cpu", dtype=torch.bool).contiguous()
    mask_digest = hashlib.sha256(mask.to(torch.uint8).numpy().tobytes(order="C")).hexdigest()
    recorded_mask_digest = manifest.get("placement", {}).get("gpu_mask_uint8_sha256")
    if recorded_mask_digest != mask_digest:
        raise SystemExit(
            f"GPU mask SHA mismatch: manifest={recorded_mask_digest!r} actual={mask_digest}"
        )

    cold_plan: list[dict[str, Any]] = []
    hot_plan: list[dict[str, Any]] = []
    cold_coordinates: list[tuple[int, int]] = []
    hot_coordinates: list[tuple[int, int]] = []
    for layer_id in range(args.expected_layers):
        for expert_id in range(args.expected_experts):
            is_hot = bool(mask[layer_id, expert_id].item())
            record = {
                "layer_id": layer_id,
                "expert_id": expert_id,
                "calibration_routed_slots": int(counts_2d[layer_id, expert_id].item()),
                "reason": "kt_frequency_mask_hot" if is_hot else "kt_frequency_mask_cold",
            }
            coordinate = (layer_id, expert_id)
            if is_hot:
                hot_plan.append(record)
                hot_coordinates.append(coordinate)
            else:
                cold_plan.append(record)
                cold_coordinates.append(coordinate)

    expected_cold = int(manifest.get("placement", {}).get("cold_experts", -1))
    expected_hot = int(manifest.get("placement", {}).get("hot_experts", -1))
    if len(cold_plan) != expected_cold or len(hot_plan) != expected_hot:
        raise SystemExit(
            "manifest/mask expert count mismatch: "
            f"cold={len(cold_plan)}/{expected_cold} hot={len(hot_plan)}/{expected_hot}"
        )
    if len(cold_plan) + len(hot_plan) != args.expected_layers * args.expected_experts:
        raise AssertionError("internal topology coverage error")

    cold_main_bytes = len(cold_plan) * args.expert_main_bytes
    cold_scale_bytes = len(cold_plan) * args.expert_scale_bytes
    metadata = {
        "schema_version": 1,
        "policy": "kt_qualified_frequency_coordinate_exact_complement_v1",
        "semantic_tier": "exact",
        "source_stats": str(args.stats.resolve()),
        "source_stats_sha256": actual_stats_sha,
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256_file(args.manifest),
        "source_qualification_snapshot": str(
            args.qualification_snapshot.resolve()
        ),
        "source_qualification_snapshot_sha256": sha256_file(
            args.qualification_snapshot
        ),
        "qualified_exact_cases": [str(case.get("name")) for case in exact_cases],
        "qualified_ktransformers_commit": EXPECTED_KT_COMMIT,
        "qualified_sglang_commit": EXPECTED_SGLANG_COMMIT,
        "qualified_model_snapshot": EXPECTED_MODEL_SNAPSHOT,
        "calibration_fixture": manifest.get("calibration_fixture"),
        "evaluation_fixture": manifest.get("evaluation_fixture"),
        "fixtures_disjoint_by_source_index": True,
        "num_layers": args.expected_layers,
        "num_experts": args.expected_experts,
        "offload_plan_records": len(cold_plan),
        "hot_plan_records": len(hot_plan),
        "expert_main_bytes": args.expert_main_bytes,
        "expert_scale_bytes": args.expert_scale_bytes,
        "actual_offload_main_bytes": cold_main_bytes,
        "actual_offload_main_gib": cold_main_bytes / (1 << 30),
        "kt_cpu_expert_bytes_including_scales": cold_main_bytes + cold_scale_bytes,
        "kt_cpu_expert_gib_including_scales": (cold_main_bytes + cold_scale_bytes)
        / (1 << 30),
        "scale_byte_delta": cold_scale_bytes,
        "gpu_mask_uint8_sha256": mask_digest,
        "cold_coordinate_sha256": coordinate_sha256(cold_coordinates),
        "hot_coordinate_sha256": coordinate_sha256(hot_coordinates),
        "same_expert_coordinates_as_kt_expected_mask": True,
        "same_expert_coordinates_as_kt_runtime_verified": True,
        "custom_scales_remain_gpu_resident": True,
    }
    plan = {
        "metadata": metadata,
        "offload_plan": cold_plan,
        "hot_plan": hot_plan,
        "warm_plan": [],
        "prefetch_plan": hot_plan,
        "offload_plan_records": len(cold_plan),
        "hot_plan_records": len(hot_plan),
        "warm_plan_records": 0,
        "prefetch_plan_records": len(hot_plan),
    }
    atomic_json(args.output, plan)
    summary = {
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "metadata": metadata,
    }
    if args.empty_output is not None:
        all_hot = sorted(
            (
                {
                    **record,
                    "reason": "custom_empty_hot_topology_complement",
                }
                for record in [*cold_plan, *hot_plan]
            ),
            key=lambda item: (int(item["layer_id"]), int(item["expert_id"])),
        )
        empty_metadata = {
            **metadata,
            "policy": "custom_empty_same_kt_qualified_topology_v1",
            "semantic_tier": "reference",
            "offload_plan_records": 0,
            "hot_plan_records": len(all_hot),
            "actual_offload_main_bytes": 0,
            "actual_offload_main_gib": 0.0,
            "kt_cpu_expert_bytes_including_scales": 0,
            "kt_cpu_expert_gib_including_scales": 0.0,
            "scale_byte_delta": 0,
            "cold_coordinate_sha256": coordinate_sha256([]),
            "hot_coordinate_sha256": coordinate_sha256(
                [
                    (int(item["layer_id"]), int(item["expert_id"]))
                    for item in all_hot
                ]
            ),
            "same_expert_coordinates_as_kt_expected_mask": False,
            "same_expert_coordinates_as_kt_runtime_verified": False,
            "reference_for_treatment_plan_sha256": sha256_file(args.output),
        }
        empty_plan = {
            "metadata": empty_metadata,
            "offload_plan": [],
            "hot_plan": all_hot,
            "warm_plan": [],
            "prefetch_plan": all_hot,
            "offload_plan_records": 0,
            "hot_plan_records": len(all_hot),
            "warm_plan_records": 0,
            "prefetch_plan_records": len(all_hot),
        }
        atomic_json(args.empty_output, empty_plan)
        summary["empty_output"] = str(args.empty_output.resolve())
        summary["empty_output_sha256"] = sha256_file(args.empty_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
