#!/usr/bin/env python3
"""Prepare CPU-only DeepSeek-V2-Lite assets for a quick custom-UVM screen.

The fixed last-16 policy is intentionally *not* routing calibrated.  The plan
is exact only in the narrower sense that it is a complete, non-overlapping
partition of all 26 x 64 routed-expert coordinates.  Nothing in this script
imports CUDA libraries or queries a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MODEL = Path(
    "/home/ubuntu/.cache/huggingface/hub/"
    "models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/"
    "85864749cd611b4353ce1decdb286193298f64c7"
)
DEFAULT_EVALUATION_SOURCE = (
    SCRIPT_DIR / "fixtures/moei_deepseek_v2_lite_evaluation_prompt512_v1.json"
)
DEFAULT_CALIBRATION_SOURCE = (
    SCRIPT_DIR / "fixtures/moei_deepseek_v2_lite_calibration_prompt512_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/eurosys_moe_offload_evaluation/inputs/generated/"
    "deepseek_v2_lite_bf16"
)

EXPECTED_SOURCE_FILE_SHA256 = {
    "evaluation": "5315aa367325a0adea8ac30fc7d16659bd0bdcbcfe608f8f5c3d59ea7d415bd5",
    "calibration": "14ce7cb4960109fc3744ecd10fbcbb7a8c8fca41a82ee89b7de5a118fd4eaffe",
}
EXPECTED_SOURCE_TOKEN_SHA256 = {
    "evaluation": "70004006f8762a0f7a68744ac7925815ad430b89cb66055bf1a6d27887800237",
    "calibration": "ffc4ffab5ed1b727350f682f89b64eab322019f0ce8d81013ae05c700a55a69a",
}
EXPECTED_MODEL_SNAPSHOT = "85864749cd611b4353ce1decdb286193298f64c7"
EXPECTED_CONFIG_SHA256 = "f346286b0f1c8b044252fd54cb4fa78b9fab6472a6e8bebb9edfe03d414ea03d"
EXPECTED_INDEX_SHA256 = "d2cdb2f325f6682cf3ad1ad2526a9f979d857390b579380c0331d975136e0acf"

LAYER_IDS = tuple(range(1, 27))
EXPERT_IDS = tuple(range(64))
COLD_EXPERT_IDS = tuple(range(48, 64))
HOT_EXPERT_IDS = tuple(range(48))
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 1408
BF16_BYTES = 2
EXPERT_MAIN_BYTES = 3 * HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE * BF16_BYTES
EXPERT_SCALE_BYTES = 0
UVM_ENVELOPE_BYTES = 512 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def token_sha256(token_ids: list[int]) -> str:
    payload = ",".join(str(value) for value in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def coordinate_sha256(coordinates: list[tuple[int, int]]) -> str:
    payload = "".join(f"{layer_id}:{expert_id}\n" for layer_id, expert_id in coordinates)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_or_absolute(path: Path) -> str:
    # Preserve Hugging Face snapshot symlink paths in provenance.  The file
    # digest separately proves the resolved blob content.
    resolved = path.absolute()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def load_source_fixture(path: Path, role: str) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_SOURCE_FILE_SHA256[role]:
        raise SystemExit(f"{role} source fixture file SHA-256 mismatch: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("fixture_role") != role:
        raise SystemExit(f"{role} source fixture has the wrong role")
    token_ids = payload.get("prompt_token_ids")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != 512
        or any(isinstance(value, bool) or not isinstance(value, int) for value in token_ids)
    ):
        raise SystemExit(f"{role} source fixture does not contain 512 integer token IDs")
    actual_token_sha = token_sha256(token_ids)
    if (
        payload.get("prompt_length") != 512
        or payload.get("prompt_token_ids_sha256") != actual_token_sha
        or actual_token_sha != EXPECTED_SOURCE_TOKEN_SHA256[role]
    ):
        raise SystemExit(f"{role} source token-ID hash mismatch")
    if payload.get("model_snapshot_name") != EXPECTED_MODEL_SNAPSHOT:
        raise SystemExit(f"{role} source fixture refers to the wrong model snapshot")
    return payload


def benchmark_fixture(
    source: dict[str, Any], source_path: Path, role: str, source_index: int
) -> dict[str, Any]:
    token_ids = [int(value) for value in source["prompt_token_ids"]]
    benchmark_role = "warmup" if role == "calibration" else "evaluation"
    return {
        "metadata": {
            "schema_version": 1,
            "fixture_role": benchmark_role,
            "source_fixture_role": role,
            "source_fixture": relative_or_absolute(source_path),
            "source_fixture_sha256": sha256_file(source_path),
            "source_prompt_token_ids_sha256": source["prompt_token_ids_sha256"],
            "source_seed_text_sha256": source["seed_text_sha256"],
            "source_indices": [source_index],
            "source_index_namespace": (
                "synthetic_fixture_role_ids; evaluation=0, calibration=1"
            ),
            "one_request_fixture": True,
            "frozen_prompt_token_ids": True,
            "prompt_field_semantics": (
                "empty sentinel required by benchmark schema; prompt_token_ids are consumed"
            ),
            "model_snapshot": source["model_snapshot"],
            "model_snapshot_name": source["model_snapshot_name"],
            "prompt_tokens": 512,
            "selection_usage": (
                "held_out_measurement" if role == "evaluation" else "warmup_only"
            ),
        },
        "requests": [
            {
                "request": 0,
                "prompt": "",
                "prompt_tokens": 512,
                "prompt_token_ids": token_ids,
            }
        ],
    }


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise SystemExit(f"invalid safetensors header: {path}")
        length = struct.unpack("<Q", raw_length)[0]
        raw_header = handle.read(length)
        if len(raw_header) != length:
            raise SystemExit(f"truncated safetensors header: {path}")
    value = json.loads(raw_header)
    if not isinstance(value, dict):
        raise SystemExit(f"safetensors header is not an object: {path}")
    return value


def validate_model_and_expert_bytes(model: Path) -> dict[str, Any]:
    config_path = model / "config.json"
    index_path = model / "model.safetensors.index.json"
    if model.name != EXPECTED_MODEL_SNAPSHOT:
        raise SystemExit(f"unexpected model snapshot directory: {model}")
    if sha256_file(config_path) != EXPECTED_CONFIG_SHA256:
        raise SystemExit("model config SHA-256 mismatch")
    if sha256_file(index_path) != EXPECTED_INDEX_SHA256:
        raise SystemExit("model safetensors index SHA-256 mismatch")
    config = read_json(config_path)
    expected_config = {
        "num_hidden_layers": 27,
        "n_routed_experts": 64,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "torch_dtype": "bfloat16",
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"model config mismatch for {key}: {config.get(key)!r} != {expected!r}"
            )
    index = read_json(index_path)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise SystemExit("model safetensors index lacks weight_map")
    header_cache: dict[str, dict[str, Any]] = {}
    projection_shapes = {
        "gate_proj": [MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE],
        "up_proj": [MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE],
        "down_proj": [HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE],
    }
    observed_bytes = 0
    checked_tensors = 0
    for layer_id in LAYER_IDS:
        for expert_id in EXPERT_IDS:
            expert_bytes = 0
            for projection, expected_shape in projection_shapes.items():
                key = (
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}."
                    f"{projection}.weight"
                )
                shard_name = weight_map.get(key)
                if not isinstance(shard_name, str):
                    raise SystemExit(f"checkpoint index lacks {key}")
                if shard_name not in header_cache:
                    header_cache[shard_name] = safetensors_header(model / shard_name)
                tensor = header_cache[shard_name].get(key)
                if not isinstance(tensor, dict):
                    raise SystemExit(f"safetensors shard lacks {key}")
                if tensor.get("dtype") != "BF16" or tensor.get("shape") != expected_shape:
                    raise SystemExit(
                        f"unexpected checkpoint tensor metadata for {key}: {tensor}"
                    )
                tensor_bytes = BF16_BYTES
                for dimension in expected_shape:
                    tensor_bytes *= dimension
                expert_bytes += tensor_bytes
                checked_tensors += 1
            if expert_bytes != EXPERT_MAIN_BYTES:
                raise AssertionError("per-expert byte derivation is inconsistent")
            observed_bytes += expert_bytes
    expected_total = len(LAYER_IDS) * len(EXPERT_IDS) * EXPERT_MAIN_BYTES
    if observed_bytes != expected_total:
        raise AssertionError("topology byte derivation is inconsistent")
    return {
        "model_snapshot": str(model.resolve()),
        "model_snapshot_name": model.name,
        "config": relative_or_absolute(config_path),
        "config_sha256": sha256_file(config_path),
        "safetensors_index": relative_or_absolute(index_path),
        "safetensors_index_sha256": sha256_file(index_path),
        "checkpoint_expert_tensors_checked": checked_tensors,
        "checkpoint_shards_read_header_only": sorted(header_cache),
        "checkpoint_dtype": "BF16",
        "checkpoint_projection_shapes": projection_shapes,
        "expert_main_bytes_derivation": "3 * 2048 * 1408 * 2",
        "expert_main_bytes": EXPERT_MAIN_BYTES,
        "topology_main_bytes": expected_total,
        "topology_main_gib": expected_total / (1 << 30),
    }


def plan_record(layer_id: int, expert_id: int, cold: bool) -> dict[str, Any]:
    return {
        "expert_id": expert_id,
        "layer_id": layer_id,
        "reason": (
            "deterministic_last16_uncalibrated_screening_cold"
            if cold
            else "deterministic_last16_uncalibrated_screening_hot_complement"
        ),
    }


def treatment_plan(
    model_provenance: dict[str, Any],
    evaluation_fixture_path: Path,
    warmup_fixture_path: Path,
) -> dict[str, Any]:
    cold_coordinates = [
        (layer_id, expert_id)
        for layer_id in LAYER_IDS
        for expert_id in COLD_EXPERT_IDS
    ]
    hot_coordinates = [
        (layer_id, expert_id)
        for layer_id in LAYER_IDS
        for expert_id in HOT_EXPERT_IDS
    ]
    cold_bytes = len(cold_coordinates) * EXPERT_MAIN_BYTES
    if EXPERT_MAIN_BYTES % UVM_ENVELOPE_BYTES != 0:
        raise AssertionError("expert bytes are not envelope aligned")
    metadata = {
        "schema_version": 1,
        "policy": "deterministic_last16_per_layer_uncalibrated_screening_v1",
        "semantic_tier": "screening",
        "exactness_scope": "complete_nonoverlapping_coordinate_partition_only",
        "selection_calibrated_from_routing": False,
        "same_instantaneous_cache_as_moe_infinity": False,
        "publication_eligible": False,
        "publication_blocker": (
            "cold IDs are a deterministic quick-screen policy, not selected on an "
            "independent routing trace and not matched to MoE-Infinity runtime cache"
        ),
        "model_provenance": model_provenance,
        "num_layers": len(LAYER_IDS),
        "layer_ids": list(LAYER_IDS),
        "num_experts": len(EXPERT_IDS),
        "cold_experts_per_layer": len(COLD_EXPERT_IDS),
        "cold_expert_ids_per_layer": list(COLD_EXPERT_IDS),
        "hot_expert_ids_per_layer": list(HOT_EXPERT_IDS),
        "offload_plan_records": len(cold_coordinates),
        "hot_plan_records": len(hot_coordinates),
        "expert_main_bytes": EXPERT_MAIN_BYTES,
        "expert_scale_bytes": EXPERT_SCALE_BYTES,
        "actual_offload_main_bytes": cold_bytes,
        "actual_offload_main_gib": cold_bytes / (1 << 30),
        "uvm_envelope_bytes": UVM_ENVELOPE_BYTES,
        "uvm_envelopes_per_expert": EXPERT_MAIN_BYTES // UVM_ENVELOPE_BYTES,
        "planned_remote_region_count": (
            len(cold_coordinates) * EXPERT_MAIN_BYTES // UVM_ENVELOPE_BYTES
        ),
        "cold_coordinate_sha256": coordinate_sha256(cold_coordinates),
        "hot_coordinate_sha256": coordinate_sha256(hot_coordinates),
        "fixtures_disjoint_by_source_index": True,
        "evaluation_fixture": {
            "path": relative_or_absolute(evaluation_fixture_path),
            "sha256": sha256_file(evaluation_fixture_path),
        },
        "warmup_fixture": {
            "path": relative_or_absolute(warmup_fixture_path),
            "sha256": sha256_file(warmup_fixture_path),
        },
    }
    return {
        "metadata": metadata,
        "offload_plan": [
            plan_record(layer_id, expert_id, True)
            for layer_id, expert_id in cold_coordinates
        ],
        "hot_plan": [
            plan_record(layer_id, expert_id, False)
            for layer_id, expert_id in hot_coordinates
        ],
        "warm_plan": [],
        "offload_plan_records": len(cold_coordinates),
        "hot_plan_records": len(hot_coordinates),
        "warm_plan_records": 0,
    }


def empty_plan(treatment: dict[str, Any], treatment_path: Path) -> dict[str, Any]:
    all_coordinates = [
        (layer_id, expert_id) for layer_id in LAYER_IDS for expert_id in EXPERT_IDS
    ]
    treatment_meta = treatment["metadata"]
    metadata = {
        **treatment_meta,
        "policy": "custom_empty_same_deepseek_topology_screening_reference_v1",
        "semantic_tier": "screening_reference",
        "exactness_scope": "complete_empty_cold_coordinate_partition",
        "offload_plan_records": 0,
        "hot_plan_records": len(all_coordinates),
        "actual_offload_main_bytes": 0,
        "actual_offload_main_gib": 0.0,
        "planned_remote_region_count": 0,
        "cold_coordinate_sha256": coordinate_sha256([]),
        "hot_coordinate_sha256": coordinate_sha256(all_coordinates),
        "reference_for_treatment_plan": relative_or_absolute(treatment_path),
        "reference_for_treatment_plan_sha256": sha256_file(treatment_path),
    }
    return {
        "metadata": metadata,
        "offload_plan": [],
        "hot_plan": [
            {
                "expert_id": expert_id,
                "layer_id": layer_id,
                "reason": "custom_empty_hot_complete_topology",
            }
            for layer_id, expert_id in all_coordinates
        ],
        "warm_plan": [],
        "offload_plan_records": 0,
        "hot_plan_records": len(all_coordinates),
        "warm_plan_records": 0,
    }


def validate_derived_fixture(path: Path, expected_source_index: int) -> dict[str, Any]:
    value = read_json(path)
    metadata = value.get("metadata") if isinstance(value, dict) else None
    requests = value.get("requests") if isinstance(value, dict) else None
    if not isinstance(metadata, dict) or metadata.get("source_indices") != [
        expected_source_index
    ]:
        raise AssertionError(f"invalid metadata in {path}")
    if not isinstance(requests, list) or len(requests) != 1:
        raise AssertionError(f"{path} is not a one-request fixture")
    request = requests[0]
    if (
        request.get("prompt") != ""
        or request.get("prompt_tokens") != 512
        or len(request.get("prompt_token_ids", [])) != 512
    ):
        raise AssertionError(f"invalid request schema in {path}")
    return value


def validate_plan(plan: dict[str, Any], expect_empty: bool) -> None:
    cold = plan.get("offload_plan")
    hot = plan.get("hot_plan")
    if not isinstance(cold, list) or not isinstance(hot, list):
        raise AssertionError("plan lacks cold/hot lists")
    cold_keys = {(int(item["layer_id"]), int(item["expert_id"])) for item in cold}
    hot_keys = {(int(item["layer_id"]), int(item["expert_id"])) for item in hot}
    expected = {(layer_id, expert_id) for layer_id in LAYER_IDS for expert_id in EXPERT_IDS}
    if len(cold_keys) != len(cold) or len(hot_keys) != len(hot):
        raise AssertionError("plan contains duplicate coordinates")
    if cold_keys & hot_keys or cold_keys | hot_keys != expected:
        raise AssertionError("plan is not a complete disjoint topology partition")
    if expect_empty and cold:
        raise AssertionError("empty plan contains cold experts")
    if not expect_empty and cold_keys != {
        (layer_id, expert_id)
        for layer_id in LAYER_IDS
        for expert_id in COLD_EXPERT_IDS
    }:
        raise AssertionError("treatment plan is not the fixed last-16 policy")
    expected_bytes = 0 if expect_empty else len(cold) * EXPERT_MAIN_BYTES
    if plan["metadata"].get("actual_offload_main_bytes") != expected_bytes:
        raise AssertionError("plan byte provenance is inconsistent")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--evaluation-source", type=Path, default=DEFAULT_EVALUATION_SOURCE
    )
    parser.add_argument(
        "--calibration-source", type=Path, default=DEFAULT_CALIBRATION_SOURCE
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluation_source = load_source_fixture(args.evaluation_source, "evaluation")
    calibration_source = load_source_fixture(args.calibration_source, "calibration")
    if evaluation_source["prompt_token_ids"] == calibration_source["prompt_token_ids"]:
        raise SystemExit("evaluation and calibration token-ID streams are identical")
    model_provenance = validate_model_and_expert_bytes(args.model)

    output_dir = args.output_dir.resolve()
    evaluation_path = output_dir / "evaluation_fixture.json"
    warmup_path = output_dir / "warmup_fixture.json"
    treatment_path = output_dir / "remote_last16_plan.json"
    empty_path = output_dir / "empty_plan.json"
    manifest_path = output_dir / "asset_manifest.json"

    atomic_json(
        evaluation_path,
        benchmark_fixture(evaluation_source, args.evaluation_source, "evaluation", 0),
    )
    atomic_json(
        warmup_path,
        benchmark_fixture(calibration_source, args.calibration_source, "calibration", 1),
    )
    derived_evaluation = validate_derived_fixture(evaluation_path, 0)
    derived_warmup = validate_derived_fixture(warmup_path, 1)
    if set(derived_evaluation["metadata"]["source_indices"]) & set(
        derived_warmup["metadata"]["source_indices"]
    ):
        raise AssertionError("derived fixtures have overlapping source indices")

    treatment = treatment_plan(model_provenance, evaluation_path, warmup_path)
    validate_plan(treatment, expect_empty=False)
    atomic_json(treatment_path, treatment)
    empty = empty_plan(treatment, treatment_path)
    validate_plan(empty, expect_empty=True)
    atomic_json(empty_path, empty)

    files = {
        "evaluation_fixture": evaluation_path,
        "warmup_fixture": warmup_path,
        "remote_plan": treatment_path,
        "empty_plan": empty_path,
    }
    manifest = {
        "schema_version": 1,
        "status": "ok",
        "purpose": "custom DeepSeek-V2-Lite c1 512+32 quick screening inputs",
        "cpu_only_generation": True,
        "gpu_started_or_queried": False,
        "screening_only": True,
        "screening_reason": (
            "deterministic last-16 expert selection is not routing calibrated"
        ),
        "fixtures_distinct_by_source_file": True,
        "fixtures_distinct_by_token_hash": True,
        "fixtures_disjoint_by_source_index": True,
        "model_provenance": model_provenance,
        "bytes": {
            "expert_main_bytes": EXPERT_MAIN_BYTES,
            "expert_scale_bytes": EXPERT_SCALE_BYTES,
            "cold_experts": len(LAYER_IDS) * len(COLD_EXPERT_IDS),
            "remote_main_bytes": len(LAYER_IDS)
            * len(COLD_EXPERT_IDS)
            * EXPERT_MAIN_BYTES,
            "remote_main_gib": (
                len(LAYER_IDS) * len(COLD_EXPERT_IDS) * EXPERT_MAIN_BYTES
            )
            / (1 << 30),
            "uvm_envelope_bytes": UVM_ENVELOPE_BYTES,
            "planned_remote_region_count": len(LAYER_IDS)
            * len(COLD_EXPERT_IDS)
            * EXPERT_MAIN_BYTES
            // UVM_ENVELOPE_BYTES,
        },
        "files": {
            key: {
                "path": relative_or_absolute(path),
                "sha256": sha256_file(path),
            }
            for key, path in files.items()
        },
        "recommended_run_environment": {
            "ALLOW_GPU_RUN": "1",
            "PROFILE": "custom_remote",
            "MODEL_FAMILY": "deepseek",
            "MODEL": str(args.model.resolve()),
            "EVAL_FIXTURE": str(evaluation_path),
            "WARMUP_FIXTURE": str(warmup_path),
            "PLAN": str(treatment_path),
            "EMPTY_PLAN": str(empty_path),
            "RUN_ROOT": str(
                REPO_ROOT
                / "experiments/eurosys_moe_offload_evaluation/artifacts/"
                "pilot_deepseek_custom_remote_c1"
            ),
            "CONCURRENCY": "1",
            "OUTPUT_TOKENS": "32",
            "WARMUP_REQUESTS": "1",
            "WARMUP_OUTPUT_TOKENS": "8",
            "KV_CACHE_BYTES": "1288490189",
            "MAX_MODEL_LEN": "544",
            "MAX_NUM_BATCHED_TOKENS": "2048",
            "MAX_NUM_SEQS": "1",
            "GPU_MEMORY_UTILIZATION": "0.97",
            "TIMEOUT_SEC": "3600",
            "ROUTING_TRACE": "0",
            "FORMAL_RUN": "1",
            "REQUIRE_PROMPT_TOKEN_IDS": "1",
            "CPUSET": "0-13",
            "CPU_THREADS": "8",
        },
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
