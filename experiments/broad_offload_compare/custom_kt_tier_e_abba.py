#!/usr/bin/env python3
"""CPU-only control, integrity gates, and summary for custom-vs-KT Tier E.

GPU work is intentionally absent from this module.  The companion shell
orchestrator owns calls to the existing custom and KTransformers ``run-case``
interfaces.  This controller freezes inputs/schedule, updates status with
atomic replacement, validates completed artifacts, and performs a paired
complete-block cluster bootstrap.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "custom-kt-tier-e-abba-v1"
QUALITY_SCHEMA = "custom-kt-cross-framework-quality-v1"
TUNING_SCHEMA = "kt-dense-fp8-tuning-v1"
EXPECTED_KT_COMMIT = "eb9b70c4115cff151ace2cae5b0fc9db3690e31e"
EXPECTED_SGLANG_COMMIT = "bc7f0058fafe6c05dc6ad52d6b1c854af2a673b2"
EXPECTED_MODEL_SNAPSHOT = "d206ba732169f29bb77fbf80fc2c4b81d4d30782"
EXPECTED_MASK_SHA = "6ce8e2a0c5a68c0516366a42a5ee03c28cb6b5790311bf985c5737d496b37418"
EXPECTED_HOT_EXPERTS = 4736
EXPECTED_COLD_EXPERTS = 1408
EXPECTED_COLD_MAIN_BYTES = 6_643_777_536
EXPECTED_SHAPES = ((5120, 2048), (2048, 4096))
EXPECTED_DEVICE_SLUG = "NVIDIA_GeForce_RTX_5090"
CONCURRENCIES = (1, 4, 8)
BLOCKS = 3
ODD_SEQUENCE = ("A", "B", "B", "A")
EVEN_SEQUENCE = ("B", "A", "A", "B")
CUSTOM_KV_CACHE_BYTES = 1_288_490_189
KV_TOKENS = 13_104
CPUSET = "0-13"
CPU_THREADS = 12
OUTPUT_TOKENS = 64
WARMUP_REQUESTS = 8
WARMUP_OUTPUT_TOKENS = 8
MAX_MODEL_LEN = 512
MAX_NUM_BATCHED_TOKENS = 4096
MAX_NUM_SEQS = 32
BOOTSTRAP_DRAWS = 20_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def nonnegative_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("mean requires at least one value")
    return statistics.fmean(materialized)


def percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def coefficient_of_variation(values: list[float]) -> float | None:
    if len(values) < 2 or statistics.fmean(values) == 0:
        return None
    return statistics.stdev(values) / statistics.fmean(values)


def git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def option_value(tokens: list[str], option: str) -> str | None:
    indices = [index for index, token in enumerate(tokens) if token == option]
    if len(indices) != 1 or indices[0] + 1 >= len(tokens):
        return None
    return tokens[indices[0] + 1]


def expected_bundle_filenames() -> set[str]:
    return {
        (
            f"N={n},K={k},device_name={EXPECTED_DEVICE_SLUG},dtype=fp8_w8a8,"
            "block_shape=[128, 128].json"
        )
        for n, k in EXPECTED_SHAPES
    }


def validate_dense_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Validate only a completed installable bundle; never inspect fragments."""

    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "dense bundle has no manifest.json; wait for tuning/bundle completion"
        )
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema") != TUNING_SCHEMA
        or manifest.get("kind") != "installable-bundle"
    ):
        raise ValueError("dense tuning bundle manifest is incomplete or invalid")
    if manifest.get("expected_kt_commit") != EXPECTED_KT_COMMIT:
        raise ValueError("dense bundle KTransformers commit differs")
    if manifest.get("expected_sglang_commit") != EXPECTED_SGLANG_COMMIT:
        raise ValueError("dense bundle SGLang commit differs")
    if manifest.get("activation_output_dtype") != "bfloat16":
        raise ValueError("dense bundle output dtype is not bfloat16")
    if manifest.get("device_name") != "NVIDIA GeForce RTX 5090":
        raise ValueError("dense bundle device is not the frozen RTX 5090")
    records = manifest.get("files")
    if (
        not isinstance(records, list)
        or len(records) != 2
        or any(not isinstance(item, dict) for item in records)
        or {item.get("filename") for item in records}
        != expected_bundle_filenames()
    ):
        raise ValueError("dense bundle must contain exactly two expected configs")
    files: list[dict[str, Any]] = []
    for record in records:
        filename = record["filename"]
        source = bundle_dir / "configs" / filename
        if not source.is_file() or sha256_file(source) != record.get("sha256"):
            raise ValueError(f"dense bundle config hash mismatch: {filename}")
        config = load_json(source)
        if not isinstance(config, dict) or not config:
            raise ValueError(f"dense bundle config is empty: {filename}")
        if sorted(map(int, config)) != record.get("M_keys"):
            raise ValueError(f"dense bundle M keys differ: {filename}")
        for m_value, kernel in config.items():
            if not isinstance(kernel, dict) or kernel.get("BLOCK_SIZE_K") != 128:
                raise ValueError(
                    f"dense bundle config M={m_value} is invalid or would clamp"
                )
        files.append(
            {"filename": filename, "sha256": record["sha256"], "source": source}
        )
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "files": sorted(files, key=lambda item: item["filename"]),
    }


def validate_token_fixture(path: Path, label: str) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("requests"), list):
        raise ValueError(f"{label} fixture is not an object with requests")
    metadata = payload.get("metadata")
    requests = payload["requests"]
    if not isinstance(metadata, dict) or metadata.get("prompt_token_ids_frozen") is not True:
        raise ValueError(f"{label} fixture is not marked frozen")
    if metadata.get("tokenizer_add_special_tokens") is not False:
        raise ValueError(f"{label} fixture tokenizer special-token policy differs")
    if len(requests) < max(CONCURRENCIES, default=0):
        raise ValueError(f"{label} fixture has fewer than eight requests")
    all_ids: list[list[int]] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict) or not isinstance(request.get("prompt"), str):
            raise ValueError(f"{label} request {index} lacks a prompt")
        ids = request.get("prompt_token_ids")
        if (
            not isinstance(ids, list)
            or not ids
            or any(isinstance(value, bool) or not isinstance(value, int) for value in ids)
        ):
            raise ValueError(f"{label} request {index} has invalid token IDs")
        if request.get("prompt_tokens") != len(ids):
            raise ValueError(f"{label} request {index} token count differs")
        if request.get("prompt_token_ids_sha256") != sha256_json(ids):
            raise ValueError(f"{label} request {index} token-ID SHA differs")
        all_ids.append(ids)
    ids_sha = sha256_json(all_ids)
    if metadata.get("prompt_token_ids_list_sha256") != ids_sha:
        raise ValueError(f"{label} fixture aggregate token-ID SHA differs")
    source_indices = metadata.get("source_indices")
    if not isinstance(source_indices, list) or len(source_indices) != len(requests):
        raise ValueError(f"{label} fixture source indices are incomplete")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "prompt_token_ids_list_sha256": ids_sha,
        "source_indices": source_indices,
        "request_count": len(requests),
        "prefix_prompt_token_ids_sha256": {
            str(concurrency): sha256_json(all_ids[:concurrency])
            for concurrency in CONCURRENCIES
        },
    }


def plan_coordinates(records: Any, label: str) -> set[tuple[int, int]]:
    if not isinstance(records, list):
        raise ValueError(f"plan {label} is not a list")
    coordinates: list[tuple[int, int]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"plan {label}[{index}] is not an object")
        layer = record.get("layer_id")
        expert = record.get("expert_id")
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or not 0 <= layer < 48
            or isinstance(expert, bool)
            or not isinstance(expert, int)
            or not 0 <= expert < 128
        ):
            raise ValueError(f"plan {label}[{index}] has an invalid coordinate")
        coordinates.append((layer, expert))
    if len(coordinates) != len(set(coordinates)):
        raise ValueError(f"plan {label} contains duplicate coordinates")
    return set(coordinates)


def validate_plan(path: Path, stats_manifest: dict[str, Any]) -> dict[str, Any]:
    plan = load_json(path)
    if not isinstance(plan, dict):
        raise ValueError("custom plan is not an object")
    cold = plan_coordinates(plan.get("offload_plan"), "offload_plan")
    hot = plan_coordinates(
        plan.get("hot_plan", plan.get("prefetch_plan")), "hot_plan"
    )
    expected = {(layer, expert) for layer in range(48) for expert in range(128)}
    if cold & hot or cold | hot != expected:
        raise ValueError("custom plan is not a disjoint complete Qwen topology")
    if len(cold) != EXPECTED_COLD_EXPERTS or len(hot) != EXPECTED_HOT_EXPERTS:
        raise ValueError("custom plan hot/cold expert counts differ")
    metadata = plan.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("custom plan metadata is missing")
    placement = stats_manifest.get("placement", {})
    required = {
        "actual_offload_main_bytes": EXPECTED_COLD_MAIN_BYTES,
        "offload_plan_records": EXPECTED_COLD_EXPERTS,
        "hot_plan_records": EXPECTED_HOT_EXPERTS,
        "gpu_mask_uint8_sha256": EXPECTED_MASK_SHA,
        "same_expert_coordinates_as_kt_expected_mask": True,
        "same_expert_coordinates_as_kt_runtime_verified": True,
        "semantic_tier": "exact",
    }
    for key, expected_value in required.items():
        if metadata.get(key) != expected_value:
            raise ValueError(f"custom plan metadata {key} differs")
    if placement.get("gpu_mask_uint8_sha256") != EXPECTED_MASK_SHA:
        raise ValueError("stats manifest mask SHA differs")
    if metadata.get("source_stats_sha256") != stats_manifest.get(
        "normalized_stats", {}
    ).get("sha256"):
        raise ValueError("custom plan does not pin the normalized KT stats")
    return {
        "path": path.resolve(),
        "sha256": sha256_file(path),
        "cold": cold,
        "hot": hot,
        "cold_coordinate_sha256": metadata.get("cold_coordinate_sha256"),
    }


def validate_stats(stats: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("frequency stats manifest is not an object")
    normalized = manifest.get("normalized_stats", {})
    placement = manifest.get("placement", {})
    topology = manifest.get("topology", {})
    if sha256_file(stats) != normalized.get("sha256"):
        raise ValueError("frequency stats SHA differs from its manifest")
    if normalized.get("logical_count_shape") != [1, 48, 128]:
        raise ValueError("frequency stats shape differs")
    if topology != {"layers": 48, "experts_per_layer": 128, "total_experts": 6144}:
        raise ValueError("frequency stats topology differs")
    if (
        placement.get("gpu_mask_uint8_sha256") != EXPECTED_MASK_SHA
        or placement.get("hot_experts") != EXPECTED_HOT_EXPERTS
        or placement.get("cold_experts") != EXPECTED_COLD_EXPERTS
        or sum(placement.get("hot_experts_per_layer", [])) != EXPECTED_HOT_EXPERTS
        or len(placement.get("hot_experts_per_layer", [])) != 48
        or manifest.get("fixtures_disjoint_by_source_index") is not True
        or manifest.get("binary_authoritative_mask_used_as_counts") is not False
    ):
        raise ValueError("frequency placement/integrity metadata differs")
    return {
        "manifest": manifest,
        "stats_sha256": sha256_file(stats),
        "manifest_sha256": sha256_file(manifest_path),
    }


def validate_qualification_snapshot(
    path: Path, stats_manifest: dict[str, Any], stats_sha256: str
) -> dict[str, Any]:
    snapshot = load_json(path)
    failures: list[str] = []
    if snapshot.get("status") != "ok" or snapshot.get("failures") != []:
        failures.append("qualification snapshot is not clean")
    if snapshot.get("exact_output_hashes_equal_across_threads") is not True:
        failures.append("exact qualification hashes were unstable across threads")
    case = next(
        (
            item
            for item in snapshot.get("cases", [])
            if isinstance(item, dict) and item.get("name") == "exact_t12"
        ),
        None,
    )
    if not isinstance(case, dict):
        failures.append("qualification lacks exact_t12")
    else:
        runtime = case.get("runtime_verification", {})
        checks = runtime.get("checks", {})
        metadata = case.get("metadata", {})
        if runtime.get("status") != "ok" or runtime.get("failures") != []:
            failures.append("qualification exact_t12 runtime gate failed")
        if checks.get("runtime_hot_experts") != EXPECTED_HOT_EXPERTS:
            failures.append("qualification exact_t12 hot count differs")
        if checks.get("runtime_mask_layers") != 48:
            failures.append("qualification exact_t12 mask layer count differs")
        if checks.get("server_frequency_stats_loaded_without_fallback") is not True:
            failures.append("qualification exact_t12 did not load frequency stats")
        if metadata.get("frequency_stats", {}).get("sha256") != stats_sha256:
            failures.append("qualification exact_t12 used different stats")
        if metadata.get("ktransformers_commit") != EXPECTED_KT_COMMIT:
            failures.append("qualification exact_t12 KT commit differs")
        if metadata.get("sglang_commit") != EXPECTED_SGLANG_COMMIT:
            failures.append("qualification exact_t12 SGLang commit differs")
    frozen = snapshot.get("frequency_stats_manifest", {})
    if frozen.get("placement") != stats_manifest.get("placement"):
        failures.append("qualification snapshot placement differs from manifest")
    if failures:
        raise ValueError("; ".join(failures))
    return {"path": path.resolve(), "sha256": sha256_file(path)}


def build_schedule(run_root: Path) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    global_index = 0
    for concurrency in CONCURRENCIES:
        for block in range(1, BLOCKS + 1):
            sequence = ODD_SEQUENCE if block % 2 else EVEN_SEQUENCE
            for slot, arm in enumerate(sequence, start=1):
                global_index += 1
                slot_id = f"c{concurrency}-b{block}-s{slot}-{arm}"
                slot_root = (
                    run_root / f"c{concurrency}" / f"block{block}" / f"slot{slot}_{arm}"
                )
                framework = "custom" if arm == "A" else "ktransformers"
                artifact_root = (
                    slot_root
                    if arm == "A"
                    else slot_root / "kt_runner" / "cases" / "formal"
                )
                schedule.append(
                    {
                        "index": global_index,
                        "slot_id": slot_id,
                        "concurrency": concurrency,
                        "block": block,
                        "slot": slot,
                        "arm": arm,
                        "framework": framework,
                        "profile": (
                            "custom_remote" if arm == "A" else "kt_exact_t12_tuned"
                        ),
                        "slot_root": str(slot_root),
                        "artifact_root": str(artifact_root),
                    }
                )
    return schedule


def schedule_tsv(schedule: list[dict[str, Any]]) -> str:
    columns = (
        "index",
        "slot_id",
        "concurrency",
        "block",
        "slot",
        "arm",
        "framework",
        "profile",
        "slot_root",
        "artifact_root",
    )
    lines = ["\t".join(columns)]
    for item in schedule:
        lines.append("\t".join(str(item[column]) for column in columns))
    return "\n".join(lines) + "\n"


def source_fingerprint(repo_root: Path, paths: list[Path]) -> dict[str, Any]:
    records = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    custom_paths = [
        "workloads/vllm/vllm/vllm/device_allocator/uvm.py",
        "workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py",
        "workloads/vllm/vllm/vllm/device_allocator/uvm_region.py",
        "workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py",
        "workloads/vllm/vllm/vllm/model_executor/models/qwen3_moe.py",
        "workloads/vllm/vllm/vllm/model_executor/models/utils.py",
        "workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py",
        "workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py",
    ]
    patch = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--binary", "HEAD", "--", *custom_paths],
        check=True,
        capture_output=True,
    ).stdout
    return {
        "repo_commit": git_head(repo_root),
        "custom_implementation_patch_sha256": sha256_bytes(patch),
        "files": records,
    }


def preregister(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"refusing to reuse RUN_ROOT: {run_root}")
    paths = {
        "model": args.model.resolve(),
        "evaluation": args.evaluation_fixture.resolve(),
        "warmup": args.warmup_fixture.resolve(),
        "plan": args.plan.resolve(),
        "stats": args.frequency_stats.resolve(),
        "stats_manifest": args.stats_manifest.resolve(),
        "qualification": args.qualification_snapshot.resolve(),
        "bundle": args.dense_bundle_dir.resolve(),
        "kt_root": args.kt_root.resolve(),
    }
    for label in ("evaluation", "warmup", "plan", "stats", "stats_manifest", "qualification"):
        if not paths[label].is_file():
            raise FileNotFoundError(f"missing {label}: {paths[label]}")
    if not paths["model"].is_dir() or not (paths["model"] / "config.json").is_file():
        raise FileNotFoundError(f"invalid model snapshot: {paths['model']}")
    if paths["model"].name != EXPECTED_MODEL_SNAPSHOT:
        raise ValueError("model snapshot differs from the frozen Qwen snapshot")
    if git_head(paths["kt_root"]) != EXPECTED_KT_COMMIT:
        raise ValueError("KTransformers checkout commit differs")
    if git_head(paths["kt_root"] / "third_party/sglang") != EXPECTED_SGLANG_COMMIT:
        raise ValueError("SGLang checkout commit differs")

    evaluation = validate_token_fixture(paths["evaluation"], "evaluation")
    warmup = validate_token_fixture(paths["warmup"], "warmup")
    if set(evaluation["source_indices"]) & set(warmup["source_indices"]):
        raise ValueError("evaluation and warmup fixture source indices overlap")
    stats = validate_stats(paths["stats"], paths["stats_manifest"])
    plan = validate_plan(paths["plan"], stats["manifest"])
    qualification = validate_qualification_snapshot(
        paths["qualification"], stats["manifest"], stats["stats_sha256"]
    )
    bundle = validate_dense_bundle(paths["bundle"])

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    implementation = source_fingerprint(
        repo_root,
        [
            Path(__file__),
            script_dir / "run_custom_kt_tier_e_abba.sh",
            script_dir / "run_custom_comparison.sh",
            script_dir / "run_kt_qualification.sh",
            script_dir / "kt_dense_fp8_tuning.py",
            script_dir / "kt_dense_fp8_ablation.py",
            script_dir / "benchmark_custom_offload.py",
            repo_root
            / "experiments/ktransformers_cpu_expert_compare/benchmark_sglang_stream.py",
        ],
    )

    temporary = run_root.with_name(f".{run_root.name}.preregister-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary preregistration root exists: {temporary}")
    try:
        (temporary / "inputs/dense_bundle/configs").mkdir(parents=True)
        (temporary / "logs").mkdir()
        (temporary / "states").mkdir()
        shutil.copy2(paths["evaluation"], temporary / "inputs/evaluation_fixture.json")
        shutil.copy2(paths["warmup"], temporary / "inputs/warmup_fixture.json")
        shutil.copy2(paths["plan"], temporary / "inputs/custom_plan.json")
        shutil.copy2(paths["stats"], temporary / "inputs/frequency_stats.pt")
        shutil.copy2(
            paths["stats_manifest"], temporary / "inputs/frequency_stats_manifest.json"
        )
        shutil.copy2(
            paths["qualification"], temporary / "inputs/kt_qualification_snapshot.json"
        )
        shutil.copy2(
            bundle["manifest_path"], temporary / "inputs/dense_bundle/manifest.json"
        )
        for record in bundle["files"]:
            shutil.copy2(
                record["source"],
                temporary / "inputs/dense_bundle/configs" / record["filename"],
            )

        schedule = build_schedule(run_root)
        schedule_document = {
            "schema": SCHEMA,
            "kind": "frozen-schedule",
            "created_at": utc_now(),
            "slots": schedule,
        }
        atomic_json(temporary / "schedule.json", schedule_document)
        atomic_text(temporary / "schedule.tsv", schedule_tsv(schedule))
        status = {
            "schema": SCHEMA,
            "kind": "execution-status",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "schedule_sha256": sha256_file(temporary / "schedule.json"),
            "slots": [
                {
                    "slot_id": item["slot_id"],
                    "index": item["index"],
                    "status": "pending",
                    "returncode": None,
                    "started_at": None,
                    "ended_at": None,
                    "detail": "preregistered; not started",
                    "history": [],
                }
                for item in schedule
            ],
        }
        atomic_json(temporary / "execution_status.json", status)

        snapshot_paths = {
            "evaluation_fixture": run_root / "inputs/evaluation_fixture.json",
            "warmup_fixture": run_root / "inputs/warmup_fixture.json",
            "custom_plan": run_root / "inputs/custom_plan.json",
            "frequency_stats": run_root / "inputs/frequency_stats.pt",
            "frequency_stats_manifest": run_root
            / "inputs/frequency_stats_manifest.json",
            "kt_qualification_snapshot": run_root
            / "inputs/kt_qualification_snapshot.json",
            "dense_bundle": run_root / "inputs/dense_bundle",
        }
        manifest = {
            "schema": SCHEMA,
            "kind": "preregistered-run",
            "created_at": utc_now(),
            "run_root": str(run_root),
            "schedule": {
                "sha256": sha256_file(temporary / "schedule.json"),
                "concurrencies": list(CONCURRENCIES),
                "blocks_per_concurrency": BLOCKS,
                "slots_per_block": 4,
                "cold_processes_per_arm_per_concurrency": 6,
                "odd_blocks": list(ODD_SEQUENCE),
                "even_blocks": list(EVEN_SEQUENCE),
            },
            "arms": {
                "A": {
                    "framework": "custom-vLLM",
                    "profile": "custom_remote",
                    "semantics": "fixed 1,408-expert managed-UVM remote read",
                },
                "B": {
                    "framework": "KTransformers/SGLang",
                    "profile": "exact_t12_tuned",
                    "semantics": "fixed frequency mask; all expert contributions; tuned dense FP8 configs",
                },
            },
            "protocol": {
                "tier": "E-coordinate-exact",
                "output_tokens": OUTPUT_TOKENS,
                "warmup_requests": WARMUP_REQUESTS,
                "warmup_concurrency": WARMUP_REQUESTS,
                "warmup_output_tokens": WARMUP_OUTPUT_TOKENS,
                "kv_tokens": KV_TOKENS,
                "custom_kv_cache_bytes": CUSTOM_KV_CACHE_BYTES,
                "cpuset": CPUSET,
                "cpu_expert_threads": CPU_THREADS,
                "max_model_len": MAX_MODEL_LEN,
                "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
                "max_num_seqs": MAX_NUM_SEQS,
                "greedy_temperature": 0.0,
                "ignore_eos": True,
                "seed": 0,
                "routing_trace": False,
            },
            "inputs": {
                "model": {
                    "path": str(paths["model"]),
                    "snapshot": paths["model"].name,
                    "config_sha256": sha256_file(paths["model"] / "config.json"),
                },
                "evaluation_fixture": {
                    **{key: value for key, value in evaluation.items() if key != "source_indices"},
                    "snapshot": str(snapshot_paths["evaluation_fixture"]),
                },
                "warmup_fixture": {
                    **{key: value for key, value in warmup.items() if key != "source_indices"},
                    "snapshot": str(snapshot_paths["warmup_fixture"]),
                },
                "fixtures_disjoint_by_source_index": True,
                "custom_plan": {
                    "source": str(paths["plan"]),
                    "snapshot": str(snapshot_paths["custom_plan"]),
                    "sha256": plan["sha256"],
                    "cold_experts": EXPECTED_COLD_EXPERTS,
                    "hot_experts": EXPECTED_HOT_EXPERTS,
                    "cold_main_bytes": EXPECTED_COLD_MAIN_BYTES,
                    "gpu_mask_uint8_sha256": EXPECTED_MASK_SHA,
                },
                "frequency_stats": {
                    "source": str(paths["stats"]),
                    "snapshot": str(snapshot_paths["frequency_stats"]),
                    "sha256": stats["stats_sha256"],
                },
                "frequency_stats_manifest": {
                    "source": str(paths["stats_manifest"]),
                    "snapshot": str(snapshot_paths["frequency_stats_manifest"]),
                    "sha256": stats["manifest_sha256"],
                    "gpu_mask_uint8_sha256": EXPECTED_MASK_SHA,
                },
                "kt_qualification_snapshot": {
                    "source": str(paths["qualification"]),
                    "snapshot": str(snapshot_paths["kt_qualification_snapshot"]),
                    "sha256": qualification["sha256"],
                    "status": "ok",
                    "exact_t12_runtime_mask_qualified": True,
                },
                "dense_bundle": {
                    "source": str(paths["bundle"]),
                    "snapshot": str(snapshot_paths["dense_bundle"]),
                    "manifest_sha256": bundle["manifest_sha256"],
                    "files": [
                        {
                            "filename": item["filename"],
                            "sha256": item["sha256"],
                        }
                        for item in bundle["files"]
                    ],
                },
            },
            "software": {
                **implementation,
                "ktransformers_commit": EXPECTED_KT_COMMIT,
                "sglang_commit": EXPECTED_SGLANG_COMMIT,
            },
            "claim_scope": {
                "allowed": "complete-system-stack throughput comparison",
                "pure_mechanism_attribution": False,
                "mechanism_attribution_blockers": [
                    "custom benchmark_custom_offload.py hard-codes enforce_eager=True while KT exact keeps CUDA graph enabled",
                    "custom uses VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1 while KT follows its own server initialization/warmup path",
                    "custom uses offline vLLM LLM.generate while KT uses SGLang HTTP/SSE",
                    "GPU fused-MoE remote reads and KT CPU-expert compute change kernels, compute device, and activation handling together",
                ],
                "future_custom_graph_interface": (
                    "A future graph-qualified custom arm must be a separately frozen profile "
                    "with its own output/VRAM/stability qualification and same-framework "
                    "eager-vs-graph ABBA; this runner will not relabel eager results."
                ),
            },
            "quality_gate": {
                "required_schema": QUALITY_SCHEMA,
                "path": str(run_root / "cross_framework_quality_gate.json"),
                "required_for_main_table": True,
                "cross_framework_output_hash_equality_required": False,
            },
        }
        atomic_json(temporary / "run_manifest.json", manifest)
        os.replace(temporary, run_root)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def read_run_documents(run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    manifest = load_json(run_root / "run_manifest.json")
    schedule_doc = load_json(run_root / "schedule.json")
    status = load_json(run_root / "execution_status.json")
    if manifest.get("schema") != SCHEMA or manifest.get("kind") != "preregistered-run":
        raise ValueError("invalid run manifest")
    if schedule_doc.get("schema") != SCHEMA or schedule_doc.get("kind") != "frozen-schedule":
        raise ValueError("invalid schedule")
    if sha256_file(run_root / "schedule.json") != manifest.get("schedule", {}).get("sha256"):
        raise ValueError("schedule SHA differs from preregistration")
    if status.get("schedule_sha256") != manifest.get("schedule", {}).get("sha256"):
        raise ValueError("execution status names a different schedule")
    expected = build_schedule(run_root.resolve())
    if schedule_doc.get("slots") != expected:
        raise ValueError("schedule contents differ from the fixed ABBA/BAAB design")
    return manifest, expected, status


def update_status(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    lock_path = run_root / ".execution_status.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _, schedule, status = read_run_documents(run_root)
        scheduled_ids = [item["slot_id"] for item in schedule]
        if [item.get("slot_id") for item in status.get("slots", [])] != scheduled_ids:
            raise ValueError("execution status slot order differs from schedule")
        try:
            index = scheduled_ids.index(args.slot_id)
        except ValueError as exc:
            raise ValueError(f"unknown slot ID: {args.slot_id}") from exc
        record = status["slots"][index]
        old = record.get("status")
        allowed = {
            "pending": {"running", "skipped"},
            "running": {"success", "failed"},
        }
        if args.status not in allowed.get(old, set()):
            raise ValueError(f"invalid status transition {old!r} -> {args.status!r}")
        if args.status == "success" and args.returncode != 0:
            raise ValueError("success requires returncode=0")
        if args.status in {"failed", "skipped"} and args.returncode == 0:
            raise ValueError(f"{args.status} requires a nonzero return code")
        timestamp = utc_now()
        record.setdefault("history", []).append(
            {
                "from": old,
                "to": args.status,
                "at": timestamp,
                "returncode": args.returncode,
                "detail": args.detail,
            }
        )
        record["status"] = args.status
        record["returncode"] = args.returncode
        record["detail"] = args.detail
        if args.status == "running":
            record["started_at"] = timestamp
        else:
            record["ended_at"] = timestamp
        status["updated_at"] = timestamp
        atomic_json(run_root / "execution_status.json", status)
        return record


def required_json(path: Path, failures: list[str], label: str) -> dict[str, Any]:
    if not path.is_file():
        failures.append(f"missing {label}: {path.name}")
        return {}
    try:
        value = load_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        failures.append(f"invalid {label}: {exc}")
        return {}
    if not isinstance(value, dict):
        failures.append(f"{label} is not an object")
        return {}
    return value


def expected_prompt_prefix_sha(manifest: dict[str, Any], concurrency: int) -> str:
    return manifest["inputs"]["evaluation_fixture"][
        "prefix_prompt_token_ids_sha256"
    ][str(concurrency)]


def output_hash_and_request_metrics(
    requests: Any,
    *,
    concurrency: int,
    custom: bool,
    failures: list[str],
) -> tuple[str | None, dict[str, float | None]]:
    if not isinstance(requests, list) or len(requests) != concurrency:
        failures.append("measured request records are incomplete")
        return None, {}
    outputs: list[list[int]] = []
    ttft: list[float] = []
    tpot: list[float] = []
    latency: list[float] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            failures.append(f"request {index} is not an object")
            continue
        ids = request.get("output_token_ids")
        if (
            not isinstance(ids, list)
            or len(ids) != OUTPUT_TOKENS
            or any(isinstance(value, bool) or not isinstance(value, int) for value in ids)
        ):
            failures.append(f"request {index} output token IDs differ")
            continue
        if request.get("output_token_ids_sha256") != sha256_json(ids):
            failures.append(f"request {index} output token-ID SHA differs")
        if custom:
            if (
                request.get("prompt_token_ids_exact") is not True
                or request.get("output_tokens") != OUTPUT_TOKENS
            ):
                failures.append(f"custom request {index} frozen/fixed-output gate failed")
            ttft_value = request.get("ttft_ms")
            tpot_value = request.get("tpot_ms")
            latency_value = request.get("request_latency_ms")
        else:
            if (
                request.get("success") is not True
                or request.get("submitted_input_kind") != "input_ids"
                or request.get("server_prompt_token_count_matches_submission") is not True
                or request.get("completion_tokens") != OUTPUT_TOKENS
            ):
                failures.append(f"KT request {index} frozen/fixed-output gate failed")
            ttft_value = request.get("ttft_ms")
            tpot_value = request.get("tpot_ms")
            latency_value = request.get("latency_ms")
        for name, value, destination in (
            ("ttft", ttft_value, ttft),
            ("tpot", tpot_value, tpot),
            ("latency", latency_value, latency),
        ):
            if not nonnegative_finite(value):
                failures.append(f"request {index} has invalid {name}")
            else:
                destination.append(float(value))
        outputs.append(ids)
    if len(outputs) != concurrency:
        return None, {}
    metrics: dict[str, float | None] = {}
    for label, values in (("ttft_ms", ttft), ("tpot_ms", tpot), ("latency_ms", latency)):
        metrics[f"mean_{label}"] = statistics.fmean(values) if values else None
        metrics[f"p50_{label}"] = percentile(values, 0.50)
        metrics[f"p95_{label}"] = percentile(values, 0.95)
        metrics[f"p99_{label}"] = percentile(values, 0.99)
    return sha256_json(outputs), metrics


def frozen_prompt_ids(path: Path, concurrency: int) -> list[list[int]]:
    payload = load_json(path)
    return [
        [int(value) for value in request["prompt_token_ids"]]
        for request in payload["requests"][:concurrency]
    ]


def validate_custom_slot(
    slot: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    root = Path(slot["artifact_root"])
    failures: list[str] = []
    result = required_json(root / "result.json", failures, "custom result")
    gate = required_json(root / "integrity_gate.json", failures, "custom integrity gate")
    placement = required_json(
        root / "runtime_remote_placement_manifest.json", failures, "custom placement"
    )
    expected_plan = load_json(Path(manifest["inputs"]["custom_plan"]["snapshot"]))
    expected_remote = sorted(
        [int(item["layer_id"]), int(item["expert_id"])]
        for item in expected_plan.get("offload_plan", [])
    )
    expected_hot = sorted(
        [int(item["layer_id"]), int(item["expert_id"])]
        for item in expected_plan.get("hot_plan", [])
    )

    if (root / "returncode.txt").is_file():
        try:
            if int((root / "returncode.txt").read_text().strip()) != 0:
                failures.append("custom runner return code is nonzero")
        except ValueError:
            failures.append("custom runner return code is invalid")
    else:
        failures.append("custom runner return code is missing")

    config = result.get("config", {})
    expected_config = {
        "concurrency": slot["concurrency"],
        "output_tokens": OUTPUT_TOKENS,
        "warmup_requests": WARMUP_REQUESTS,
        "warmup_output_tokens": WARMUP_OUTPUT_TOKENS,
        "kv_cache_memory_bytes": CUSTOM_KV_CACHE_BYTES,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "max_num_seqs": MAX_NUM_SEQS,
        "require_prompt_token_ids": True,
        "enforce_eager": True,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": 0,
        "profile_label": "custom_remote",
    }
    if result.get("status") != "ok" or result.get("profile_label") != "custom_remote":
        failures.append("custom result status/profile differs")
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            failures.append(f"custom config {key} differs")
    if Path(config.get("model", "")).name != EXPECTED_MODEL_SNAPSHOT:
        failures.append("custom model snapshot differs")
    fixtures = result.get("fixtures", {})
    if fixtures.get("evaluation_sha256") != manifest["inputs"]["evaluation_fixture"]["sha256"]:
        failures.append("custom evaluation fixture SHA differs")
    if fixtures.get("warmup_sha256") != manifest["inputs"]["warmup_fixture"]["sha256"]:
        failures.append("custom warmup fixture SHA differs")
    if fixtures.get("disjoint_source_indices") is not True:
        failures.append("custom fixtures are not recorded disjoint")

    measurement = result.get("measurement", {})
    if (
        measurement.get("completed_requests") != slot["concurrency"]
        or measurement.get("output_tokens") != slot["concurrency"] * OUTPUT_TOKENS
    ):
        failures.append("custom measured request/output counts differ")
    output_hash, request_metrics = output_hash_and_request_metrics(
        measurement.get("requests"),
        concurrency=slot["concurrency"],
        custom=True,
        failures=failures,
    )
    if output_hash != measurement.get("aggregate_output_token_ids_sha256"):
        failures.append("custom aggregate output token-ID SHA differs")
    expected_prompt_ids = frozen_prompt_ids(
        Path(manifest["inputs"]["evaluation_fixture"]["snapshot"]),
        slot["concurrency"],
    )
    measured_requests = measurement.get("requests", [])
    if isinstance(measured_requests, list) and len(measured_requests) == slot["concurrency"]:
        for index, (request, prompt_ids) in enumerate(
            zip(measured_requests, expected_prompt_ids)
        ):
            if (
                request.get("fixture_request") != index
                or request.get("expected_prompt_token_ids_sha256")
                != sha256_json(prompt_ids)
            ):
                failures.append(f"custom request {index} is not the frozen fixture prefix")
    throughput = measurement.get("output_tokens_per_second")
    if not positive_finite(throughput):
        failures.append("custom throughput is invalid")

    gate_requirements = {
        "status": "pass",
        "profile": "custom_remote",
        "formal_run": True,
        "prompt_token_ids_frozen": True,
        "metric_counts_complete": True,
        "aggregate_metrics_finite_positive": True,
        "placement_qualified": True,
        "release_fail_fast": True,
        "runtime_remote_expert_coordinates_match_plan": True,
        "runtime_remote_expert_count": EXPECTED_COLD_EXPERTS,
        "runtime_hot_expert_count": EXPECTED_HOT_EXPERTS,
        "runtime_remote_region_bytes": EXPECTED_COLD_MAIN_BYTES,
        "release_time_remote_mapping_verified": True,
        "snapshot_time_remote_cpu_residency_verified": True,
    }
    for key, expected in gate_requirements.items():
        if gate.get(key) != expected:
            failures.append(f"custom integrity gate {key} differs")
    if gate.get("aggregate_output_token_ids_sha256") != output_hash:
        failures.append("custom integrity output SHA differs")

    placement_requirements = {
        "status": "ok",
        "complete": True,
        "failures": [],
        "layer_count": 48,
        "expert_count": 6144,
        "remote_expert_count": EXPECTED_COLD_EXPERTS,
        "hot_expert_count": EXPECTED_HOT_EXPERTS,
        "remote_region_bytes": EXPECTED_COLD_MAIN_BYTES,
        "offloaded_expert_count": 0,
        "release_time_remote_mapping_verified": True,
    }
    for key, expected in placement_requirements.items():
        if placement.get(key) != expected:
            failures.append(f"custom placement {key} differs")
    if sorted(placement.get("remote_experts", [])) != expected_remote:
        failures.append("custom runtime remote coordinates differ from the frozen plan")
    if sorted(placement.get("hot_experts", [])) != expected_hot:
        failures.append("custom runtime hot coordinates differ from the frozen plan")
    if placement.get("plan_sha256") != manifest["inputs"]["custom_plan"]["sha256"]:
        failures.append("custom runtime plan SHA differs")
    physical = placement.get("snapshot_time_remote_physical_residency", {})
    if (
        physical.get("ok") is not True
        or physical.get("queried_bytes") != EXPECTED_COLD_MAIN_BYTES
        or physical.get("cpu_bytes") != EXPECTED_COLD_MAIN_BYTES
        or physical.get("gpu_bytes") != 0
        or physical.get("unknown_bytes") != 0
    ):
        failures.append("custom remote bytes are not wholly CPU resident")

    environment_path = root / "environment.txt"
    command_path = root / "command.txt"
    log_path = root / "run.log"
    if not environment_path.is_file() or not command_path.is_file() or not log_path.is_file():
        failures.append("custom command/environment/run log evidence is incomplete")
    else:
        environment = set(environment_path.read_text(encoding="utf-8").splitlines())
        for value in (
            "OMP_NUM_THREADS=12",
            "MKL_NUM_THREADS=12",
            "VLLM_UVM_CONTROLLED_REMAP_SKIP_POST_INIT_WARMUP=1",
            "VLLM_UVM_EXPERT_OFFLOAD_ACCESS_MODE=remote_read",
            "VLLM_UVM_DYNAMIC_EXPERT_SWAP_ENABLE=0",
            "VLLM_UVM_INCREMENTAL_EXPERT_OFFLOAD_ENABLE=0",
            "VLLM_UVM_WEIGHT_PREFETCH_ENABLE=0",
            "VLLM_UVM_MOE_ROUTING_TRACE_ENABLE=0",
        ):
            if value not in environment:
                failures.append(f"custom environment lacks {value}")
        tokens = shlex.split(command_path.read_text(encoding="utf-8"))
        taskset_indices = [index for index, token in enumerate(tokens) if token == "taskset"]
        if not any(tokens[index : index + 3] == ["taskset", "-c", CPUSET] for index in taskset_indices):
            failures.append("custom command CPU affinity differs")
        log = log_path.read_text(encoding="utf-8", errors="replace")
        if not re.search(r"GPU KV cache size:\s*13,104 tokens", log):
            failures.append("custom runtime did not confirm a 13,104-token KV cache")

    metrics = {
        "output_tokens_per_second": float(throughput) if positive_finite(throughput) else None,
        **request_metrics,
        "ready_memory_used_mib": result.get("post_warmup_ready_nvml", {}).get("used_mib"),
        "peak_allocated_bytes": result.get("measurement_torch_peak", {}).get("max_allocated_bytes"),
        "process_max_rss_kib": result.get("process_memory", {}).get("ru_maxrss_kib"),
    }
    return {
        "schema": SCHEMA,
        "kind": "cross-framework-slot-integrity",
        "slot_id": slot["slot_id"],
        "arm": "A",
        "framework": "custom",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "output_token_ids_sha256": output_hash,
        "metrics": metrics,
        "placement": {
            "remote_experts": placement.get("remote_expert_count"),
            "hot_experts": placement.get("hot_expert_count"),
            "remote_bytes": placement.get("remote_region_bytes"),
            "cpu_residency_verified": not any(
                "CPU resident" in failure for failure in failures
            ),
        },
        "stack": {
            "enforce_eager": config.get("enforce_eager"),
            "controlled_remap_skip_post_init_warmup": True,
            "client": "offline_vllm_generate",
        },
    }


def validate_kt_warmup(
    warmup: dict[str, Any], manifest: dict[str, Any], failures: list[str]
) -> None:
    config = warmup.get("config", {})
    phases = warmup.get("phases")
    phase = phases[0] if isinstance(phases, list) and len(phases) == 1 else {}
    fixture = warmup.get("fixture", {})
    sampling = config.get("sampling_params", {})
    if (
        warmup.get("status") != "ok"
        or config.get("concurrencies") != [WARMUP_REQUESTS]
        or config.get("output_tokens_per_request") != WARMUP_OUTPUT_TOKENS
        or config.get("require_prompt_token_ids") is not True
        or sampling.get("temperature") != 0.0
        or sampling.get("ignore_eos") is not True
        or sampling.get("sampling_seed") != 0
        or fixture.get("sha256") != manifest["inputs"]["warmup_fixture"]["sha256"]
        or fixture.get("all_requests_have_prompt_token_ids") is not True
        or phase.get("concurrency") != WARMUP_REQUESTS
        or phase.get("completed_requests") != WARMUP_REQUESTS
        or phase.get("failed_requests") != 0
        or phase.get("completed_output_tokens")
        != WARMUP_REQUESTS * WARMUP_OUTPUT_TOKENS
        or phase.get("all_requests_produced_fixed_output") is not True
        or phase.get("prompt_token_ids_list_sha256")
        != manifest["inputs"]["warmup_fixture"][
            "prefix_prompt_token_ids_sha256"
        ][str(WARMUP_REQUESTS)]
    ):
        failures.append("KT held-out frozen application warmup gate failed")


def validate_kt_slot(slot: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    root = Path(slot["artifact_root"])
    slot_root = Path(slot["slot_root"])
    failures: list[str] = []
    benchmark = required_json(root / "benchmark.json", failures, "KT benchmark")
    warmup = required_json(root / "warmup.json", failures, "KT warmup")
    runtime = required_json(
        root / "runtime_verification.json", failures, "KT runtime verification"
    )
    metadata = required_json(root / "case_metadata.json", failures, "KT metadata")
    warning = required_json(
        root / "dense_fp8_warning_gate.json", failures, "KT dense warning gate"
    )
    before = required_json(
        slot_root / "kt_config_state_before.json", failures, "KT config state before"
    )
    after = required_json(
        slot_root / "kt_config_state_after.json", failures, "KT config state after"
    )

    config = benchmark.get("config", {})
    sampling = config.get("sampling_params", {})
    fixture = benchmark.get("fixture", {})
    phases = benchmark.get("phases")
    phase = phases[0] if isinstance(phases, list) and len(phases) == 1 else {}
    concurrency = slot["concurrency"]
    if (
        benchmark.get("status") != "ok"
        or config.get("concurrencies") != [concurrency]
        or config.get("output_tokens_per_request") != OUTPUT_TOKENS
        or config.get("require_prompt_token_ids") is not True
        or sampling.get("temperature") != 0.0
        or sampling.get("ignore_eos") is not True
        or sampling.get("sampling_seed") != 0
        or fixture.get("sha256") != manifest["inputs"]["evaluation_fixture"]["sha256"]
        or fixture.get("all_requests_have_prompt_token_ids") is not True
    ):
        failures.append("KT benchmark config/frozen-fixture gate failed")
    if (
        phase.get("concurrency") != concurrency
        or phase.get("completed_requests") != concurrency
        or phase.get("failed_requests") != 0
        or phase.get("completed_output_tokens") != concurrency * OUTPUT_TOKENS
        or phase.get("all_requests_produced_fixed_output") is not True
        or phase.get("prompt_token_ids_list_sha256")
        != expected_prompt_prefix_sha(manifest, concurrency)
    ):
        failures.append("KT measured phase request/prompt/output gate failed")
    output_hash, request_metrics = output_hash_and_request_metrics(
        phase.get("requests"),
        concurrency=concurrency,
        custom=False,
        failures=failures,
    )
    throughput = phase.get("output_tokens_per_second")
    if not positive_finite(throughput):
        failures.append("KT throughput is invalid")
    validate_kt_warmup(warmup, manifest, failures)

    checks = runtime.get("checks", {})
    if (
        runtime.get("status") != "ok"
        or runtime.get("mode") != "exact"
        or runtime.get("threads") != CPU_THREADS
        or runtime.get("failures") != []
        or checks.get("runtime_hot_experts") != EXPECTED_HOT_EXPERTS
        or checks.get("runtime_mask_layers") != 48
        or checks.get("server_frequency_stats_loaded_without_fallback") is not True
        or checks.get("dynamic_expert_update_log_hits") != 0
        or checks.get("layerwise_prefill_log_hits") != 0
    ):
        failures.append("KT runtime exact-mask/profile gate failed")
    if (
        metadata.get("mode") != "exact"
        or metadata.get("cpu_threads") != CPU_THREADS
        or metadata.get("ktransformers_commit") != EXPECTED_KT_COMMIT
        or metadata.get("sglang_commit") != EXPECTED_SGLANG_COMMIT
        or Path(metadata.get("model", "")).name != EXPECTED_MODEL_SNAPSHOT
        or metadata.get("frequency_stats", {}).get("sha256")
        != manifest["inputs"]["frequency_stats"]["sha256"]
        or metadata.get("evaluation_fixture", {}).get("sha256")
        != manifest["inputs"]["evaluation_fixture"]["sha256"]
        or metadata.get("gpu_experts_ratio_cli") != "0.77083333333333337"
        or metadata.get("kv_tokens") != KV_TOKENS
    ):
        failures.append("KT case metadata differs from preregistration")

    bundle_sha = manifest["inputs"]["dense_bundle"]["manifest_sha256"]
    for label, state in (("before", before), ("after", after)):
        if (
            state.get("state") != "tuned"
            or state.get("bundle_manifest_sha256") != bundle_sha
        ):
            failures.append(f"KT dense config state {label} was not pinned tuned")
    parsed = warning.get("parsed", {})
    tuned_shapes = {
        tuple(item.get("shape", []))
        for item in parsed.get("dense_tuned_loads", [])
        if isinstance(item, dict)
    }
    if (
        warning.get("status") != "ok"
        or warning.get("expect") != "tuned"
        or warning.get("failures") != []
        or parsed.get("sha256")
        != (sha256_file(root / "server.log") if (root / "server.log").is_file() else None)
        or parsed.get("fp8_gemm_runner_backend") != "triton"
        or parsed.get("dense_fallbacks") != []
        or not set(EXPECTED_SHAPES).issubset(tuned_shapes)
        or parsed.get("dense_config_clamp_warning_count") != 0
    ):
        failures.append("KT tuned dense warning gate failed")

    command_path = root / "server_command.txt"
    if not command_path.is_file():
        failures.append("KT server command is missing")
    else:
        tokens = shlex.split(command_path.read_text(encoding="utf-8"))
        expected_values = {
            "--context-length": str(MAX_MODEL_LEN),
            "--max-running-requests": str(MAX_NUM_SEQS),
            "--max-total-tokens": str(KV_TOKENS),
            "--chunked-prefill-size": str(MAX_NUM_BATCHED_TOKENS),
            "--max-prefill-tokens": str(MAX_NUM_BATCHED_TOKENS),
            "--kt-method": "FP8",
            "--kt-cpuinfer": str(CPU_THREADS),
            "--kt-threadpool-count": "1",
            "--kt-numa-nodes": "0",
            "--kt-expert-placement-strategy": "frequency",
            "--kt-max-deferred-experts-per-token": "0",
            "--kt-gpu-prefill-token-threshold": "0",
            "--attention-backend": "flashinfer",
            "--fp8-gemm-backend": "triton",
        }
        for flag, expected in expected_values.items():
            if option_value(tokens, flag) != expected:
                failures.append(f"KT server command {flag} differs")
        if tokens[:3] != ["taskset", "-c", CPUSET]:
            failures.append("KT server command CPU affinity differs")
        for flag in ("--kt-enable-dynamic-expert-update", "--disable-cuda-graph", "--disable-overlap-schedule"):
            if flag in tokens:
                failures.append(f"KT exact command unexpectedly contains {flag}")

    metrics = {
        "output_tokens_per_second": float(throughput) if positive_finite(throughput) else None,
        **request_metrics,
    }
    ready_path = root / "ready_gpu.csv"
    if ready_path.is_file():
        try:
            metrics["ready_memory_used_mib"] = float(
                ready_path.read_text(encoding="utf-8").splitlines()[0].split(",")[1].strip()
            )
        except (IndexError, ValueError):
            metrics["ready_memory_used_mib"] = None
    else:
        metrics["ready_memory_used_mib"] = None
    return {
        "schema": SCHEMA,
        "kind": "cross-framework-slot-integrity",
        "slot_id": slot["slot_id"],
        "arm": "B",
        "framework": "ktransformers",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "output_token_ids_sha256": output_hash,
        "metrics": metrics,
        "placement": {
            "runtime_mask_layers": checks.get("runtime_mask_layers"),
            "hot_experts": checks.get("runtime_hot_experts"),
            "cold_experts": EXPECTED_COLD_EXPERTS,
            "frequency_mask_sha256": EXPECTED_MASK_SHA,
            "frequency_stats_loaded_without_fallback": checks.get(
                "server_frequency_stats_loaded_without_fallback"
            ),
        },
        "dense_warning": {
            "fallback_count": len(parsed.get("dense_fallbacks", [])),
            "tuned_shapes": [list(value) for value in sorted(tuned_shapes)],
            "clamp_warning_count": parsed.get("dense_config_clamp_warning_count"),
            "bundle_manifest_sha256": bundle_sha,
        },
        "stack": {
            "cuda_graph_enabled": True,
            "overlap_schedule_enabled": True,
            "client": "sglang_http_sse",
        },
    }


def validate_slot(slot: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    try:
        return (
            validate_custom_slot(slot, manifest)
            if slot["arm"] == "A"
            else validate_kt_slot(slot, manifest)
        )
    except BaseException as exc:
        return {
            "schema": SCHEMA,
            "kind": "cross-framework-slot-integrity",
            "slot_id": slot.get("slot_id"),
            "arm": slot.get("arm"),
            "framework": slot.get("framework"),
            "status": "fail",
            "failures": [f"fail-closed validator exception: {type(exc).__name__}: {exc}"],
            "output_token_ids_sha256": None,
            "metrics": {},
        }


def verify_slot(args: argparse.Namespace) -> dict[str, Any]:
    manifest, schedule, _ = read_run_documents(args.run_root.resolve())
    slot = next((item for item in schedule if item["slot_id"] == args.slot_id), None)
    if slot is None:
        raise ValueError(f"unknown slot ID: {args.slot_id}")
    result = validate_slot(slot, manifest)
    output = args.output or Path(slot["slot_root"]) / "cross_framework_integrity_gate.json"
    atomic_json(output, result)
    return result


def cluster_bootstrap(
    blocks: list[dict[str, list[float]]], *, seed: int, draws: int = BOOTSTRAP_DRAWS
) -> dict[str, Any] | None:
    """Bootstrap whole four-slot ABBA blocks, never individual processes."""

    if not blocks or any(len(block.get(arm, [])) != 2 for block in blocks for arm in ("A", "B")):
        return None
    observed_a = [value for block in blocks for value in block["A"]]
    observed_b = [value for block in blocks for value in block["B"]]
    observed_ratio = mean(observed_a) / mean(observed_b)
    generator = random.Random(seed)
    ratios: list[float] = []
    for _ in range(draws):
        selected = [blocks[generator.randrange(len(blocks))] for _ in blocks]
        values_a = [value for block in selected for value in block["A"]]
        values_b = [value for block in selected for value in block["B"]]
        ratios.append(mean(values_a) / mean(values_b))
    return {
        "statistic": "mean(custom output tok/s) / mean(KT output tok/s)",
        "cluster_unit": "complete four-slot ABBA/BAAB block",
        "complete_clusters": len(blocks),
        "draws": draws,
        "seed": seed,
        "ratio": observed_ratio,
        "ci95_percentile": [percentile(ratios, 0.025), percentile(ratios, 0.975)],
    }


def validate_quality_gate(
    path: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    failures: list[str] = []
    if not path.is_file():
        return {
            "status": "missing",
            "pass": False,
            "path": str(path),
            "sha256": None,
            "failures": [
                "cross-framework quality gate is missing; output-hash equality is not required, but an independent preregistered quality evaluation is required"
            ],
        }
    try:
        gate = load_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "invalid",
            "pass": False,
            "path": str(path),
            "sha256": sha256_file(path),
            "failures": [f"quality gate is unreadable: {exc}"],
        }
    expected = {
        "schema": QUALITY_SCHEMA,
        "status": "pass",
        "same_model_snapshot": True,
        "same_prompt_token_ids": True,
        "quality_thresholds_preregistered": True,
        "quality_within_threshold": True,
        "model_snapshot": EXPECTED_MODEL_SNAPSHOT,
        "evaluation_fixture_sha256": manifest["inputs"]["evaluation_fixture"]["sha256"],
        "prompt_token_ids_list_sha256": manifest["inputs"]["evaluation_fixture"][
            "prompt_token_ids_list_sha256"
        ],
        "covered_concurrencies": list(CONCURRENCIES),
    }
    if not isinstance(gate, dict):
        failures.append("quality gate is not an object")
        gate = {}
    for key, expected_value in expected.items():
        if gate.get(key) != expected_value:
            failures.append(f"quality gate {key} differs")
    methods = gate.get("methods")
    if not isinstance(methods, list) or not methods or not set(methods) & {
        "teacher_forced_logits",
        "teacher_forced_perplexity",
        "fixed_task_accuracy",
    }:
        failures.append("quality gate lacks an accepted independent method")
    evidence_sha = gate.get("evidence_sha256")
    if not isinstance(evidence_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", evidence_sha):
        failures.append("quality gate lacks a valid evidence SHA")
    return {
        "status": "pass" if not failures else "fail",
        "pass": not failures,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "failures": failures,
        "methods": gate.get("methods"),
    }


def build_summary(run_root: Path, quality_gate_path: Path | None = None) -> dict[str, Any]:
    manifest, schedule, status = read_run_documents(run_root)
    status_by_id = {item["slot_id"]: item for item in status.get("slots", [])}
    validations = {item["slot_id"]: validate_slot(item, manifest) for item in schedule}
    global_failures: list[str] = []
    if len(status_by_id) != len(schedule):
        global_failures.append("execution status does not cover the frozen schedule")
    for item in schedule:
        state = status_by_id.get(item["slot_id"], {})
        if state.get("status") != "success" or state.get("returncode") != 0:
            global_failures.append(
                f"{item['slot_id']}: scheduled slot status is {state.get('status')!r}"
            )
        validation = validations[item["slot_id"]]
        if validation.get("status") != "pass":
            global_failures.extend(
                f"{item['slot_id']}: {failure}"
                for failure in validation.get("failures", ["integrity validation failed"])
            )

    points: dict[str, Any] = {}
    all_repetition_gates = True
    for concurrency in CONCURRENCIES:
        selected = [item for item in schedule if item["concurrency"] == concurrency]
        complete_blocks: list[dict[str, list[float]]] = []
        block_records: list[dict[str, Any]] = []
        for block in range(1, BLOCKS + 1):
            block_slots = [item for item in selected if item["block"] == block]
            block_ok = all(
                status_by_id.get(item["slot_id"], {}).get("status") == "success"
                and status_by_id.get(item["slot_id"], {}).get("returncode") == 0
                and validations[item["slot_id"]].get("status") == "pass"
                and positive_finite(
                    validations[item["slot_id"]]
                    .get("metrics", {})
                    .get("output_tokens_per_second")
                )
                for item in block_slots
            )
            arm_values = {
                arm: [
                    float(
                        validations[item["slot_id"]]["metrics"][
                            "output_tokens_per_second"
                        ]
                    )
                    for item in block_slots
                    if item["arm"] == arm
                    and positive_finite(
                        validations[item["slot_id"]]
                        .get("metrics", {})
                        .get("output_tokens_per_second")
                    )
                ]
                for arm in ("A", "B")
            }
            if block_ok and all(len(arm_values[arm]) == 2 for arm in ("A", "B")):
                complete_blocks.append(arm_values)
            block_records.append(
                {
                    "block": block,
                    "sequence": [item["arm"] for item in block_slots],
                    "complete": block_ok,
                    "throughput_by_arm": arm_values,
                }
            )

        hashes = {
            arm: [
                validations[item["slot_id"]].get("output_token_ids_sha256")
                for item in selected
                if item["arm"] == arm
                and validations[item["slot_id"]].get("status") == "pass"
            ]
            for arm in ("A", "B")
        }
        hash_stable = {
            arm: len(values) == 6
            and all(value is not None for value in values)
            and len(set(values)) == 1
            for arm, values in hashes.items()
        }
        throughput = {
            arm: [value for block in complete_blocks for value in block[arm]]
            for arm in ("A", "B")
        }
        cv = {
            arm: coefficient_of_variation(values) if values else None
            for arm, values in throughput.items()
        }
        minimum_complete = len(complete_blocks) == BLOCKS
        high_cv_extension_required = any(
            value is not None and value > 0.05 and len(throughput[arm]) < 10
            for arm, value in cv.items()
        )
        repetition_gate = (
            minimum_complete
            and all(hash_stable.values())
            and not high_cv_extension_required
        )
        all_repetition_gates &= repetition_gate
        points[f"c{concurrency}"] = {
            "concurrency": concurrency,
            "scheduled_blocks": BLOCKS,
            "complete_blocks": len(complete_blocks),
            "scheduled_cold_processes_per_arm": 6,
            "included_cold_processes_per_arm": {
                arm: len(throughput[arm]) for arm in ("A", "B")
            },
            "blocks": block_records,
            "framework_internal_output_hash_stable": hash_stable,
            "framework_output_hashes": {
                arm: sorted(set(value for value in hashes[arm] if value is not None))
                for arm in ("A", "B")
            },
            "cross_framework_output_hash_equal": (
                len(set(hashes["A"])) == 1
                and len(set(hashes["B"])) == 1
                and hashes["A"][0] == hashes["B"][0]
            )
            if hashes["A"] and hashes["B"]
            else None,
            "throughput": {
                "custom_mean_output_tokens_per_second": (
                    statistics.fmean(throughput["A"]) if throughput["A"] else None
                ),
                "kt_mean_output_tokens_per_second": (
                    statistics.fmean(throughput["B"]) if throughput["B"] else None
                ),
                "custom_cv": cv["A"],
                "kt_cv": cv["B"],
                "custom_over_kt_complete_block_cluster_bootstrap": cluster_bootstrap(
                    complete_blocks, seed=20260813 + concurrency
                ),
            },
            "high_cv_requires_at_least_ten_processes_per_arm": high_cv_extension_required,
            "performance_repetition_gate_pass": repetition_gate,
        }

    quality_path = quality_gate_path or Path(
        manifest["quality_gate"]["path"]
    )
    quality = validate_quality_gate(quality_path, manifest)
    schedule_complete = not global_failures
    system_table_eligible = schedule_complete and all_repetition_gates and quality["pass"]
    publication_blockers = []
    if global_failures:
        publication_blockers.append("one or more scheduled/status/integrity gates failed")
    if not all_repetition_gates:
        publication_blockers.append(
            "one or more concurrency points lacks three complete blocks, stable within-framework hashes, or the required high-CV extension"
        )
    if not quality["pass"]:
        publication_blockers.extend(quality["failures"])
    mechanism_blockers = manifest["claim_scope"]["mechanism_attribution_blockers"]
    result = {
        "schema": SCHEMA,
        "kind": "formal-summary",
        "created_at": utc_now(),
        "run_root": str(run_root),
        "schedule": {
            "total_slots": len(schedule),
            "successful_slots": sum(
                status_by_id.get(item["slot_id"], {}).get("status") == "success"
                for item in schedule
            ),
            "failed_slots": [
                {
                    "slot_id": item["slot_id"],
                    "status": status_by_id.get(item["slot_id"], {}).get("status"),
                    "returncode": status_by_id.get(item["slot_id"], {}).get("returncode"),
                    "detail": status_by_id.get(item["slot_id"], {}).get("detail"),
                }
                for item in schedule
                if status_by_id.get(item["slot_id"], {}).get("status") != "success"
            ],
            "all_integrity_gates_pass": schedule_complete,
        },
        "points": points,
        "quality_gate": quality,
        "claim_scope": {
            "system_throughput_table_eligible": system_table_eligible,
            "main_table_eligible": system_table_eligible,
            "pure_mechanism_attribution_eligible": False,
            "allowed_if_eligible": "complete-system-stack throughput only",
            "mechanism_attribution_blockers": mechanism_blockers,
        },
        "publication_eligible": system_table_eligible,
        "publication_blockers": publication_blockers,
        "failures": global_failures,
        "slot_integrity": [validations[item["slot_id"]] for item in schedule],
    }
    return result


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Custom remote vs KTransformers exact/t12/tuned — Tier E",
        "",
        f"- Publication/main-table eligible (system throughput only): `{str(summary['publication_eligible']).lower()}`",
        f"- Successful scheduled slots: `{summary['schedule']['successful_slots']}/{summary['schedule']['total_slots']}`",
        f"- Cross-framework quality gate: `{summary['quality_gate']['status']}`",
        "- Pure mechanism attribution eligible: `false`",
        "",
        "## Complete-block results",
        "",
        "| concurrency | complete blocks | custom tok/s | KT tok/s | custom/KT | cluster-bootstrap 95% CI | A hash stable | B hash stable | point gate |",
        "|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for name in ("c1", "c4", "c8"):
        point = summary["points"][name]
        throughput = point["throughput"]
        bootstrap = throughput["custom_over_kt_complete_block_cluster_bootstrap"]
        ratio = bootstrap["ratio"] if bootstrap else None
        interval = bootstrap["ci95_percentile"] if bootstrap else None
        format_value = lambda value: "—" if value is None else f"{value:.3f}"
        interval_text = (
            "—"
            if interval is None
            else f"[{interval[0]:.3f}, {interval[1]:.3f}]"
        )
        stable = point["framework_internal_output_hash_stable"]
        lines.append(
            "| {c} | {blocks}/3 | {a} | {b} | {ratio} | {ci} | {ha} | {hb} | {gate} |".format(
                c=point["concurrency"],
                blocks=point["complete_blocks"],
                a=format_value(throughput["custom_mean_output_tokens_per_second"]),
                b=format_value(throughput["kt_mean_output_tokens_per_second"]),
                ratio=format_value(ratio),
                ci=interval_text,
                ha=str(stable["A"]).lower(),
                hb=str(stable["B"]).lower(),
                gate=str(point["performance_repetition_gate_pass"]).lower(),
            )
        )
    lines.extend(
        [
            "",
            "The CI resamples complete four-slot ABBA/BAAB blocks. Failed or partial blocks remain in the denominator and are not converted into independent process samples.",
            "",
            "## Publication blockers",
            "",
        ]
    )
    blockers = summary["publication_blockers"]
    lines.extend(f"- {value}" for value in blockers)
    if not blockers:
        lines.append("- None for the complete-system-stack throughput table.")
    lines.extend(
        [
            "",
            "## Mechanism-attribution blockers",
            "",
            *(
                f"- {value}"
                for value in summary["claim_scope"]["mechanism_attribution_blockers"]
            ),
            "",
            "Even after every publication gate passes, this result supports only a complete-system-stack comparison. It does not isolate transfer cost or prove a pure offload-mechanism speedup.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    summary = build_summary(
        run_root,
        args.quality_gate.resolve() if args.quality_gate is not None else None,
    )
    output_json = args.output_json or run_root / "CUSTOM_KT_TIER_E_SUMMARY.json"
    output_md = args.output_md or run_root / "CUSTOM_KT_TIER_E_SUMMARY.md"
    atomic_json(output_json, summary)
    atomic_text(output_md, render_summary_markdown(summary))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("preregister", help="freeze inputs and all 36 slots")
    register.add_argument("--run-root", type=Path, required=True)
    register.add_argument("--model", type=Path, required=True)
    register.add_argument("--evaluation-fixture", type=Path, required=True)
    register.add_argument("--warmup-fixture", type=Path, required=True)
    register.add_argument("--plan", type=Path, required=True)
    register.add_argument("--frequency-stats", type=Path, required=True)
    register.add_argument("--stats-manifest", type=Path, required=True)
    register.add_argument("--qualification-snapshot", type=Path, required=True)
    register.add_argument("--dense-bundle-dir", type=Path, required=True)
    register.add_argument("--kt-root", type=Path, required=True)

    mark = sub.add_parser("mark", help="atomically transition one scheduled slot")
    mark.add_argument("--run-root", type=Path, required=True)
    mark.add_argument("--slot-id", required=True)
    mark.add_argument("--status", choices=("running", "success", "failed", "skipped"), required=True)
    mark.add_argument("--returncode", type=int, required=True)
    mark.add_argument("--detail", required=True)

    verify = sub.add_parser("verify-slot", help="validate one completed slot")
    verify.add_argument("--run-root", type=Path, required=True)
    verify.add_argument("--slot-id", required=True)
    verify.add_argument("--output", type=Path)

    summary = sub.add_parser("summarize", help="validate and cluster-bootstrap a run")
    summary.add_argument("--run-root", type=Path, required=True)
    summary.add_argument("--quality-gate", type=Path)
    summary.add_argument("--output-json", type=Path)
    summary.add_argument("--output-md", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "preregister":
            value = preregister(args)
        elif args.command == "mark":
            value = update_status(args)
        elif args.command == "verify-slot":
            value = verify_slot(args)
        elif args.command == "summarize":
            value = summarize(args)
        else:  # pragma: no cover - argparse owns this branch
            raise AssertionError(args.command)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    if args.command == "verify-slot" and value.get("status") != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
