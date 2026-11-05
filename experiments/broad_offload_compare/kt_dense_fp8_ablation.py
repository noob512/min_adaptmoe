#!/usr/bin/env python3
"""CPU-only control, integrity gates, and summary for KT dense-config D-T-T-D.

This file never imports torch and never starts a server.  GPU execution is
owned by ``run_kt_dense_fp8_ablation.sh`` through the already-qualified KT
runner.  Configuration transitions are restricted to the two manifest-pinned
RTX 5090 JSON files and are journaled with hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "kt-dense-fp8-ablation-v1"
TUNING_SCHEMA = "kt-dense-fp8-tuning-v1"
EXPECTED_KT_COMMIT = "eb9b70c4115cff151ace2cae5b0fc9db3690e31e"
EXPECTED_SGLANG_COMMIT = "bc7f0058fafe6c05dc6ad52d6b1c854af2a673b2"
EXPECTED_MODEL_SNAPSHOT = "d206ba732169f29bb77fbf80fc2c4b81d4d30782"
EXPECTED_MASK_SHA = "6ce8e2a0c5a68c0516366a42a5ee03c28cb6b5790311bf985c5737d496b37418"
EXPECTED_HOT_EXPERTS = 4736
EXPECTED_COLD_EXPERTS = 1408
EXPECTED_SHAPES = ((5120, 2048), (2048, 4096))
DEVICE_SLUG = "NVIDIA_GeForce_RTX_5090"
CONFIG_SUFFIX = "third_party/sglang/python/sglang/srt/layers/quantization/configs"
SCHEDULE = ((1, "D", "default"), (2, "T", "tuned"), (3, "T", "tuned"), (4, "D", "default"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
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


def git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def expected_filenames() -> set[str]:
    return {
        (
            f"N={n},K={k},device_name={DEVICE_SLUG},dtype=fp8_w8a8,"
            "block_shape=[128, 128].json"
        )
        for n, k in EXPECTED_SHAPES
    }


def validate_bundle(bundle_dir: Path) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema") != TUNING_SCHEMA or manifest.get("kind") != "installable-bundle":
        raise ValueError("invalid dense tuning bundle manifest")
    if manifest.get("expected_kt_commit") != EXPECTED_KT_COMMIT:
        raise ValueError("bundle KT commit differs")
    if manifest.get("expected_sglang_commit") != EXPECTED_SGLANG_COMMIT:
        raise ValueError("bundle SGLang commit differs")
    if manifest.get("activation_output_dtype") != "bfloat16":
        raise ValueError("bundle was not tuned for BF16 output")
    if manifest.get("device_name") != "NVIDIA GeForce RTX 5090":
        raise ValueError("bundle device differs")
    records = manifest.get("files")
    if (
        not isinstance(records, list)
        or len(records) != len(expected_filenames())
        or any(not isinstance(item, dict) for item in records)
        or {item.get("filename") for item in records} != expected_filenames()
    ):
        raise ValueError("bundle must contain exactly the two expected dense configs")
    files = []
    for item in records:
        filename = item["filename"]
        source = bundle_dir / "configs" / filename
        if not source.is_file() or sha256_file(source) != item.get("sha256"):
            raise ValueError(f"bundle config hash mismatch: {filename}")
        config = load_json(source)
        if not isinstance(config, dict) or not config:
            raise ValueError(f"empty runtime config: {filename}")
        if sorted(map(int, config)) != item.get("M_keys"):
            raise ValueError(f"M keys differ from manifest: {filename}")
        for m, kernel in config.items():
            if not isinstance(kernel, dict):
                raise ValueError(f"M={m} has a non-object kernel config")
            if kernel.get("BLOCK_SIZE_K") != 128:
                raise ValueError(f"M={m} would be clamped at runtime")
        files.append(
            {"filename": filename, "sha256": item["sha256"], "source": str(source)}
        )
    return {
        "path": str(bundle_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "files": sorted(files, key=lambda item: item["filename"]),
    }


def config_target_dir(kt_root: Path) -> Path:
    target = (kt_root.resolve() / CONFIG_SUFFIX).resolve()
    if not target.as_posix().endswith(CONFIG_SUFFIX):
        raise ValueError("resolved config target escaped exact SGLang scope")
    return target


def inspect_config_state(bundle_dir: Path, kt_root: Path) -> dict[str, Any]:
    bundle = validate_bundle(bundle_dir)
    target_dir = config_target_dir(kt_root)
    records = []
    for source_record in bundle["files"]:
        target = target_dir / source_record["filename"]
        exists = target.is_file()
        actual_sha = sha256_file(target) if exists else None
        records.append(
            {
                "filename": target.name,
                "target": str(target),
                "exists": exists,
                "actual_sha256": actual_sha,
                "bundle_sha256": source_record["sha256"],
                "matches_bundle": exists and actual_sha == source_record["sha256"],
            }
        )
    if all(not item["exists"] for item in records):
        state = "default"
    elif all(item["matches_bundle"] for item in records):
        state = "tuned"
    else:
        state = "invalid"
    return {
        "schema": SCHEMA,
        "kind": "config-state",
        "created_at": utc_now(),
        "state": state,
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "target_dir": str(target_dir),
        "files": records,
    }


def transition_config(
    *,
    bundle_dir: Path,
    kt_root: Path,
    target_state: str,
    journal: Path,
    verify_checkout: bool = True,
) -> dict[str, Any]:
    if target_state not in {"default", "tuned"}:
        raise ValueError("target state must be default or tuned")
    if journal.exists():
        raise FileExistsError(f"refusing to overwrite transition journal {journal}")
    kt_root = kt_root.resolve()
    if verify_checkout:
        if git_head(kt_root) != EXPECTED_KT_COMMIT:
            raise ValueError("KTransformers checkout drift")
        if git_head(kt_root / "third_party/sglang") != EXPECTED_SGLANG_COMMIT:
            raise ValueError("SGLang checkout drift")
    bundle = validate_bundle(bundle_dir)
    before = inspect_config_state(bundle_dir, kt_root)
    if before["state"] == "invalid":
        raise ValueError("config directory is partial or differs from the bundle")
    operations: list[dict[str, Any]] = []
    backups: dict[Path, bytes] = {}
    created: list[Path] = []
    try:
        if before["state"] != target_state:
            for record in bundle["files"]:
                source = Path(record["source"])
                target = config_target_dir(kt_root) / record["filename"]
                if target_state == "default":
                    if sha256_file(target) != record["sha256"]:
                        raise ValueError(f"refusing to remove non-bundle config {target}")
                    backups[target] = target.read_bytes()
                    target.unlink()
                    operations.append({"operation": "remove", "target": str(target)})
                else:
                    if target.exists():
                        raise FileExistsError(f"refusing to replace {target}")
                    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
                    temporary.write_bytes(source.read_bytes())
                    os.replace(temporary, target)
                    created.append(target)
                    operations.append({"operation": "install", "target": str(target)})
        after = inspect_config_state(bundle_dir, kt_root)
        if after["state"] != target_state:
            raise RuntimeError("configuration transition did not reach its target")
    except Exception:
        for target in created:
            if target.exists() and target.name in expected_filenames():
                target.unlink()
        for target, content in backups.items():
            if not target.exists():
                temporary = target.with_name(f".{target.name}.rollback-{os.getpid()}")
                temporary.write_bytes(content)
                os.replace(temporary, target)
        raise
    receipt = {
        "schema": SCHEMA,
        "kind": "config-transition-receipt",
        "created_at": utc_now(),
        "bundle": {
            "path": bundle["path"],
            "manifest_sha256": bundle["manifest_sha256"],
            "files": bundle["files"],
        },
        "requested_state": target_state,
        "before": before,
        "operations": operations,
        "after": after,
        "recoverable_from_bundle": True,
    }
    atomic_json(journal, receipt)
    return receipt


def snapshot_copy(source: Path, target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {"path": str(target.resolve()), "sha256": sha256_file(target)}


def fixture_metadata(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    requests = payload.get("requests") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict) or metadata.get("prompt_token_ids_frozen") is not True:
        raise ValueError(f"fixture is not frozen: {path}")
    if not isinstance(requests, list) or len(requests) < 8:
        raise ValueError(f"fixture has fewer than eight requests: {path}")
    ids = [item.get("prompt_token_ids") if isinstance(item, dict) else None for item in requests]
    if any(
        not isinstance(values, list)
        or not values
        or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
        for values in ids
    ):
        raise ValueError(f"fixture contains invalid prompt IDs: {path}")
    if any(item.get("prompt_tokens") != len(values) for item, values in zip(requests, ids)):
        raise ValueError(f"fixture prompt lengths differ from token IDs: {path}")
    actual_ids_sha = sha256_json(ids)
    if metadata.get("prompt_token_ids_list_sha256") != actual_ids_sha:
        raise ValueError(f"fixture token-ID SHA is inconsistent: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "prompt_token_ids_list_sha256": actual_ids_sha,
        "c8_prompt_token_ids_list_sha256": sha256_json(ids[:8]),
        "c8_prompt_lengths": [len(values) for values in ids[:8]],
    }


def write_status_files(run_root: Path, status: dict[str, Any]) -> None:
    atomic_json(run_root / "execution_status.json", status)
    lines = ["slot\tarm\tconfig_state\tcase_root\tstatus\treturncode\tstarted_at\tended_at\tdetail"]
    for item in status["slots"]:
        values = [
            item["slot"],
            item["arm"],
            item["config_state"],
            item["case_root"],
            item["status"],
            "" if item["returncode"] is None else item["returncode"],
            item.get("started_at") or "",
            item.get("ended_at") or "",
            item.get("detail") or "",
        ]
        lines.append("\t".join(str(value).replace("\t", " ").replace("\n", " ") for value in values))
    atomic_text(run_root / "execution_status.tsv", "\n".join(lines) + "\n")


def preregister(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"run root already exists: {run_root}")
    kt_root = args.kt_root.resolve()
    if git_head(kt_root) != EXPECTED_KT_COMMIT:
        raise ValueError("KTransformers checkout drift")
    if git_head(kt_root / "third_party/sglang") != EXPECTED_SGLANG_COMMIT:
        raise ValueError("SGLang checkout drift")
    source_bundle = validate_bundle(args.bundle_dir)
    initial_state = inspect_config_state(args.bundle_dir, kt_root)
    if initial_state["state"] == "invalid":
        raise ValueError("initial dense config state is invalid")
    eval_meta = fixture_metadata(args.evaluation_fixture)
    warm_meta = fixture_metadata(args.warmup_fixture)
    stats_manifest = load_json(args.stats_manifest)
    if stats_manifest.get("placement", {}).get("gpu_mask_uint8_sha256") != EXPECTED_MASK_SHA:
        raise ValueError("frequency mask SHA differs")
    if stats_manifest.get("placement", {}).get("hot_experts") != EXPECTED_HOT_EXPERTS:
        raise ValueError("frequency mask hot-expert count differs")
    if stats_manifest.get("placement", {}).get("cold_experts") != EXPECTED_COLD_EXPERTS:
        raise ValueError("frequency mask cold-expert count differs")
    if sha256_file(args.frequency_stats) != stats_manifest.get("normalized_stats", {}).get("sha256"):
        raise ValueError("frequency stats SHA differs from manifest")

    run_root.mkdir(parents=True)
    for directory in ("logs", "states", "transitions", "status", "inputs", "bundle_snapshot", "metadata"):
        (run_root / directory).mkdir()
    snapshot_bundle = run_root / "bundle_snapshot"
    snapshot_copy(Path(source_bundle["manifest_path"]), snapshot_bundle / "manifest.json")
    for item in source_bundle["files"]:
        snapshot_copy(Path(item["source"]), snapshot_bundle / "configs" / item["filename"])
    validated_snapshot = validate_bundle(snapshot_bundle)
    if validated_snapshot["manifest_sha256"] != source_bundle["manifest_sha256"]:
        raise RuntimeError("bundle snapshot manifest changed")
    eval_snapshot = snapshot_copy(args.evaluation_fixture, run_root / "inputs/evaluation_fixture_token_ids.json")
    warm_snapshot = snapshot_copy(args.warmup_fixture, run_root / "inputs/calibration_fixture_token_ids.json")
    snapshot_copy(args.stats_manifest, run_root / "inputs/frequency_stats_manifest.json")
    snapshot_copy(args.frequency_stats, run_root / "inputs/frequency_stats.pt")

    schedule = []
    for slot, arm, state in SCHEDULE:
        case_root = run_root / "cases/block1" / f"slot{slot}_{arm}"
        schedule.append(
            {
                "slot": slot,
                "arm": arm,
                "config_state": state,
                "case_label": f"block1/slot{slot}_{arm}",
                "case_root": str(case_root),
            }
        )
    manifest = {
        "schema": SCHEMA,
        "kind": "preregistered-run",
        "created_at": utc_now(),
        "run_root": str(run_root),
        "schedule": schedule,
        "initial_config_state": initial_state,
        "bundle": {
            "source_path": source_bundle["path"],
            "snapshot_path": str(snapshot_bundle),
            "manifest_sha256": validated_snapshot["manifest_sha256"],
            "files": validated_snapshot["files"],
        },
        "inputs": {
            "evaluation_fixture": {**eval_meta, **eval_snapshot},
            "warmup_fixture": {**warm_meta, **warm_snapshot},
            "frequency_stats": {
                "runtime_path": str(args.frequency_stats.resolve()),
                "snapshot_path": str((run_root / "inputs/frequency_stats.pt").resolve()),
                "sha256": sha256_file(args.frequency_stats),
            },
            "frequency_stats_manifest": {
                "runtime_path": str(args.stats_manifest.resolve()),
                "snapshot_path": str((run_root / "inputs/frequency_stats_manifest.json").resolve()),
                "sha256": sha256_file(args.stats_manifest),
                "gpu_mask_uint8_sha256": EXPECTED_MASK_SHA,
                "hot_experts": EXPECTED_HOT_EXPERTS,
                "cold_experts": EXPECTED_COLD_EXPERTS,
            },
        },
        "fixed_protocol": {
            "profile": "exact",
            "cpu_threads": 12,
            "concurrency": 8,
            "output_tokens": 64,
            "warmup_concurrency": 8,
            "warmup_output_tokens": 8,
            "kv_tokens": 13104,
            "gpu_experts_ratio": "0.77083333333333337",
            "fp8_gemm_backend": "triton",
            "mixed_chunk": True,
            "cuda_graph": True,
            "overlap_schedule": True,
            "radix_cache": False,
            "shared_experts_fusion": False,
            "dynamic_expert_update": False,
            "deferred_experts_per_token": 0,
            "gpu_prefill_token_threshold": 0,
        },
        "commits": {
            "ktransformers": git_head(kt_root),
            "sglang": git_head(kt_root / "third_party/sglang"),
        },
    }
    atomic_json(run_root / "run_manifest.json", manifest)
    atomic_json(run_root / "initial_config_state.json", initial_state)
    atomic_json(run_root / "abba_order.json", {"schema": SCHEMA, "schedule": schedule})
    order_lines = ["slot\tarm\tconfig_state\tcase_label\tcase_root"]
    for item in schedule:
        order_lines.append(
            f"{item['slot']}\t{item['arm']}\t{item['config_state']}\t{item['case_label']}\t{item['case_root']}"
        )
    atomic_text(run_root / "abba_order.tsv", "\n".join(order_lines) + "\n")
    status = {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "slots": [
            {
                **item,
                "status": "preregistered",
                "returncode": None,
                "started_at": None,
                "ended_at": None,
                "detail": None,
            }
            for item in schedule
        ],
    }
    write_status_files(run_root, status)
    return manifest


def update_status(args: argparse.Namespace) -> dict[str, Any]:
    path = args.run_root / "execution_status.json"
    status = load_json(path)
    matches = [item for item in status["slots"] if item["slot"] == args.slot]
    if len(matches) != 1:
        raise ValueError("slot is absent or duplicated")
    item = matches[0]
    if args.status == "running":
        if item["status"] != "preregistered":
            raise ValueError("only a preregistered slot can start")
        item["started_at"] = args.timestamp or utc_now()
    elif args.status in {"ok", "failed", "skipped"}:
        if item["status"] not in {"running", "preregistered"}:
            raise ValueError("slot is already terminal")
        item["ended_at"] = args.timestamp or utc_now()
    item["status"] = args.status
    item["returncode"] = args.returncode
    item["detail"] = args.detail
    write_status_files(args.run_root, status)
    return item


def command_value(tokens: list[str], flag: str) -> str | None:
    if tokens.count(flag) != 1:
        return None
    index = tokens.index(flag)
    return tokens[index + 1] if index + 1 < len(tokens) else None


def positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def validate_case_config_state(
    state: dict[str, Any], expected_state: str, expected_bundle_sha: str
) -> list[str]:
    failures: list[str] = []
    if state.get("schema") != SCHEMA or state.get("kind") != "config-state":
        failures.append("state receipt schema/kind differs")
    if state.get("state") != expected_state:
        failures.append(f"state is not {expected_state}")
    if state.get("bundle_manifest_sha256") != expected_bundle_sha:
        failures.append("state bundle SHA differs")
    files = state.get("files")
    if not isinstance(files, list) or len(files) != 2:
        failures.append("state does not cover exactly two dense config files")
        return failures
    if {item.get("filename") for item in files if isinstance(item, dict)} != expected_filenames():
        failures.append("state dense config filenames differ")
    for item in files:
        if not isinstance(item, dict):
            failures.append("state contains a non-object file record")
            continue
        if expected_state == "tuned":
            if not item.get("exists") or not item.get("matches_bundle"):
                failures.append(f"tuned config mismatch: {item.get('filename')}")
        elif item.get("exists") or item.get("matches_bundle"):
            failures.append(f"default arm unexpectedly has config: {item.get('filename')}")
    return failures


def read_nvml_metrics(case_dir: Path) -> dict[str, Any]:
    ready = None
    ready_path = case_dir / "ready_gpu.csv"
    if ready_path.is_file():
        fields = [item.strip() for item in ready_path.read_text().strip().split(",")]
        if len(fields) >= 2:
            ready = int(fields[1])
    samples = []
    phase_path = case_dir / "phase_nvml.csv"
    if phase_path.is_file():
        for line in phase_path.read_text().splitlines():
            fields = [item.strip() for item in line.split(",")]
            if len(fields) >= 3:
                try:
                    samples.append({"memory_used_mib": int(fields[2]), "gpu_util_percent": int(fields[3])})
                except (ValueError, IndexError):
                    pass
    return {
        "ready_memory_used_mib": ready,
        "phase_peak_memory_used_mib": max((item["memory_used_mib"] for item in samples), default=None),
        "phase_peak_gpu_util_percent": max((item["gpu_util_percent"] for item in samples), default=None),
        "phase_sample_count": len(samples),
    }


def verify_slot(args: argparse.Namespace) -> dict[str, Any]:
    case_dir = args.case_dir.resolve()
    manifest = load_json(args.run_manifest)
    failures: list[str] = []
    expected_state = "default" if args.arm == "D" else "tuned"
    scheduled = [
        item
        for item in manifest.get("schedule", [])
        if Path(item.get("case_root", "")).resolve() == case_dir
    ]
    if len(scheduled) != 1:
        failures.append("case directory is absent or duplicated in the preregistered schedule")
    elif (
        scheduled[0].get("arm") != args.arm
        or scheduled[0].get("config_state") != expected_state
    ):
        failures.append("requested arm differs from the preregistered schedule")
    required = [
        "benchmark.json",
        "warmup.json",
        "runtime_verification.json",
        "case_metadata.json",
        "server_command.txt",
        "server.log",
        "dense_fp8_warning_gate.json",
        "config_state_before.json",
        "config_state_after.json",
        "config_transition_receipt.json",
        "runner.returncode.txt",
    ]
    for name in required:
        if not (case_dir / name).is_file():
            failures.append(f"missing {name}")
    if failures:
        result = {
            "schema": SCHEMA,
            "kind": "slot-integrity-gate",
            "created_at": utc_now(),
            "arm": args.arm,
            "case_dir": str(case_dir),
            "status": "failed",
            "failures": failures,
        }
        atomic_json(args.output, result)
        return result

    benchmark = load_json(case_dir / "benchmark.json")
    warmup = load_json(case_dir / "warmup.json")
    runtime = load_json(case_dir / "runtime_verification.json")
    metadata = load_json(case_dir / "case_metadata.json")
    warning = load_json(case_dir / "dense_fp8_warning_gate.json")
    before = load_json(case_dir / "config_state_before.json")
    after = load_json(case_dir / "config_state_after.json")
    receipt = load_json(case_dir / "config_transition_receipt.json")
    try:
        runner_rc = int((case_dir / "runner.returncode.txt").read_text().strip())
    except ValueError:
        runner_rc = -999
    if runner_rc != 0:
        failures.append(f"qualification runner returned {runner_rc}")
    for label, state in (("before", before), ("after", after)):
        failures.extend(
            f"config {label}: {failure}"
            for failure in validate_case_config_state(
                state, expected_state, manifest["bundle"]["manifest_sha256"]
            )
        )
    if receipt.get("schema") != SCHEMA or receipt.get("kind") != "config-transition-receipt":
        failures.append("transition receipt schema/kind differs")
    if receipt.get("requested_state") != expected_state:
        failures.append("transition receipt requested the wrong state")
    if receipt.get("after", {}).get("state") != expected_state:
        failures.append("transition receipt does not establish the requested state")
    if receipt.get("bundle", {}).get("manifest_sha256") != manifest["bundle"]["manifest_sha256"]:
        failures.append("transition receipt bundle SHA differs")
    if warning.get("status") != "ok":
        failures.append("dense warning gate failed")
    if warning.get("expect") != expected_state:
        failures.append("dense warning gate checked the wrong arm")
    if warning.get("parsed", {}).get("sha256") != sha256_file(case_dir / "server.log"):
        failures.append("dense warning gate does not cover this server log")
    warning_parsed = warning.get("parsed", {})
    fallback_shapes = {
        tuple(item.get("shape", []))
        for item in warning_parsed.get("dense_fallbacks", [])
        if isinstance(item, dict)
    }
    tuned_shapes = {
        tuple(item.get("shape", []))
        for item in warning_parsed.get("dense_tuned_loads", [])
        if isinstance(item, dict)
    }
    if warning_parsed.get("fp8_gemm_runner_backend") != "triton":
        failures.append("warning audit did not confirm the Triton dense FP8 backend")
    if expected_state == "default" and (
        not set(EXPECTED_SHAPES).issubset(fallback_shapes)
        or set(EXPECTED_SHAPES) & tuned_shapes
    ):
        failures.append("default warning evidence differs from both dense fallback shapes")
    if expected_state == "tuned" and (
        fallback_shapes or not set(EXPECTED_SHAPES).issubset(tuned_shapes)
    ):
        failures.append("tuned warning evidence differs from both dense config loads")
    if expected_state == "tuned" and warning_parsed.get("dense_config_clamp_warning_count") != 0:
        failures.append("tuned warning evidence contains a BLOCK_SIZE_K clamp")
    if runtime.get("status") != "ok" or runtime.get("failures"):
        failures.append("runtime placement/config verification failed")
    checks = runtime.get("checks", {})
    if checks.get("runtime_hot_experts") != EXPECTED_HOT_EXPERTS:
        failures.append("runtime hot expert count differs")
    if checks.get("runtime_mask_layers") != 48:
        failures.append("runtime mask does not cover 48 layers")
    if metadata.get("mode") != "exact" or metadata.get("cpu_threads") != 12:
        failures.append("case is not exact/t12")
    if metadata.get("ktransformers_commit") != EXPECTED_KT_COMMIT:
        failures.append("case KTransformers commit differs")
    if metadata.get("sglang_commit") != EXPECTED_SGLANG_COMMIT:
        failures.append("case SGLang commit differs")
    if Path(metadata.get("model", "")).name != EXPECTED_MODEL_SNAPSHOT:
        failures.append("case model snapshot differs")
    if metadata.get("gpu_experts_ratio_cli") != "0.77083333333333337":
        failures.append("case GPU-expert ratio differs")
    if metadata.get("frequency_stats", {}).get("sha256") != manifest["inputs"]["frequency_stats"]["sha256"]:
        failures.append("case frequency stats SHA differs")
    if metadata.get("evaluation_fixture", {}).get("sha256") != manifest["inputs"]["evaluation_fixture"]["sha256"]:
        failures.append("case evaluation fixture SHA differs")
    if metadata.get("kv_tokens") != 13104:
        failures.append("case KV token capacity differs")

    config = benchmark.get("config", {})
    if benchmark.get("status") != "ok":
        failures.append("benchmark status is not ok")
    if config.get("concurrencies") != [8] or config.get("output_tokens_per_request") != 64:
        failures.append("benchmark is not frozen c8/output64")
    if config.get("require_prompt_token_ids") is not True:
        failures.append("benchmark did not require frozen input IDs")
    sampling = config.get("sampling_params", {})
    if sampling.get("temperature") != 0.0 or sampling.get("ignore_eos") is not True or sampling.get("sampling_seed") != 0:
        failures.append("sampling protocol differs")
    fixture = benchmark.get("fixture", {})
    if fixture.get("sha256") != manifest["inputs"]["evaluation_fixture"]["sha256"]:
        failures.append("benchmark fixture SHA differs")
    if fixture.get("all_requests_have_prompt_token_ids") is not True:
        failures.append("benchmark fixture lacks frozen input IDs")
    phases = benchmark.get("phases")
    phase = phases[0] if isinstance(phases, list) and len(phases) == 1 else {}
    if phase.get("concurrency") != 8 or phase.get("completed_requests") != 8 or phase.get("failed_requests") != 0:
        failures.append("measured phase request counts differ")
    if phase.get("expected_output_tokens") != 512 or phase.get("completed_output_tokens") != 512:
        failures.append("measured phase output count differs")
    if phase.get("all_requests_produced_fixed_output") is not True:
        failures.append("measured phase did not produce fixed output")
    if phase.get("prompt_token_ids_list_sha256") != manifest["inputs"]["evaluation_fixture"]["c8_prompt_token_ids_list_sha256"]:
        failures.append("measured c8 prompt-ID SHA differs")
    for key in (
        "output_tokens_per_second",
        "elapsed_seconds",
        "mean_ttft_ms",
        "p50_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "p50_tpot_ms",
        "p99_tpot_ms",
    ):
        if not positive_finite(phase.get(key)):
            failures.append(f"measured phase has invalid {key}")
    requests = phase.get("requests") if isinstance(phase, dict) else None
    if not isinstance(requests, list) or len(requests) != 8:
        failures.append("measured request records are incomplete")
        requests = []
    outputs = []
    request_hashes = []
    for index, request in enumerate(requests):
        output_ids = request.get("output_token_ids")
        if (
            request.get("success") is not True
            or request.get("submitted_input_kind") != "input_ids"
            or request.get("server_prompt_token_count_matches_submission") is not True
            or request.get("completion_tokens") != 64
            or not isinstance(output_ids, list)
            or len(output_ids) != 64
        ):
            failures.append(f"request {index} failed frozen/fixed-output gates")
            continue
        if sha256_json(output_ids) != request.get("output_token_ids_sha256"):
            failures.append(f"request {index} output SHA is inconsistent")
        outputs.append(output_ids)
        request_hashes.append(request.get("output_token_ids_sha256"))

    warm_config = warmup.get("config", {})
    warm_fixture = warmup.get("fixture", {})
    warm_phases = warmup.get("phases")
    warm_phase = warm_phases[0] if isinstance(warm_phases, list) and len(warm_phases) == 1 else {}
    warm_sampling = warm_config.get("sampling_params", {})
    if (
        warmup.get("status") != "ok"
        or warm_config.get("concurrencies") != [8]
        or warm_config.get("output_tokens_per_request") != 8
        or warm_config.get("require_prompt_token_ids") is not True
        or warm_sampling.get("temperature") != 0.0
        or warm_sampling.get("ignore_eos") is not True
        or warm_sampling.get("sampling_seed") != 0
        or warm_fixture.get("sha256") != manifest["inputs"]["warmup_fixture"]["sha256"]
        or warm_fixture.get("all_requests_have_prompt_token_ids") is not True
        or warm_phase.get("concurrency") != 8
        or warm_phase.get("completed_requests") != 8
        or warm_phase.get("failed_requests") != 0
        or warm_phase.get("expected_output_tokens") != 64
        or warm_phase.get("completed_output_tokens") != 64
        or warm_phase.get("all_requests_produced_fixed_output") is not True
        or warm_phase.get("prompt_token_ids_list_sha256")
        != manifest["inputs"]["warmup_fixture"]["c8_prompt_token_ids_list_sha256"]
    ):
        failures.append("held-out frozen application warmup gate failed")

    import shlex

    tokens = shlex.split((case_dir / "server_command.txt").read_text())
    expected_values = {
        "--model-path": metadata.get("model"),
        "--context-length": "512",
        "--max-running-requests": "32",
        "--chunked-prefill-size": "4096",
        "--max-prefill-tokens": "4096",
        "--schedule-policy": "fcfs",
        "--stream-interval": "1",
        "--random-seed": "0",
        "--kt-weight-path": metadata.get("model"),
        "--kt-cpuinfer": "12",
        "--kt-method": "FP8",
        "--kt-threadpool-count": "1",
        "--kt-numa-nodes": "0",
        "--kt-gpu-experts-ratio": "0.77083333333333337",
        "--kt-expert-placement-strategy": "frequency",
        "--init-expert-location": metadata.get("frequency_stats", {}).get("path"),
        "--kt-max-deferred-experts-per-token": "0",
        "--kt-gpu-prefill-token-threshold": "0",
        "--max-total-tokens": "13104",
        "--attention-backend": "flashinfer",
        "--fp8-gemm-backend": "triton",
    }
    for flag, expected in expected_values.items():
        if command_value(tokens, flag) != expected:
            failures.append(f"server command {flag} differs")
    if tokens[:3] != ["taskset", "-c", "0-13"]:
        failures.append("server command CPU affinity differs from 0-13")
    for flag in (
        "--trust-remote-code",
        "--enable-mixed-chunk",
        "--disable-radix-cache",
        "--disable-shared-experts-fusion",
    ):
        if tokens.count(flag) != 1:
            failures.append(f"server command must contain {flag} exactly once")
    for flag in ("--kt-enable-dynamic-expert-update", "--disable-cuda-graph", "--disable-overlap-schedule"):
        if tokens.count(flag):
            failures.append(f"server command unexpectedly contains {flag}")

    metrics = {
        "output_tokens_per_second": phase.get("output_tokens_per_second"),
        "elapsed_seconds": phase.get("elapsed_seconds"),
        "mean_ttft_ms": phase.get("mean_ttft_ms"),
        "p50_ttft_ms": phase.get("p50_ttft_ms"),
        "p99_ttft_ms": phase.get("p99_ttft_ms"),
        "mean_tpot_ms": phase.get("mean_tpot_ms"),
        "p50_tpot_ms": phase.get("p50_tpot_ms"),
        "p99_tpot_ms": phase.get("p99_tpot_ms"),
        **read_nvml_metrics(case_dir),
    }
    if metrics["ready_memory_used_mib"] is None:
        failures.append("ready GPU memory sample is missing")
    if metrics["phase_sample_count"] < 1:
        failures.append("measured phase has no NVML samples")
    result = {
        "schema": SCHEMA,
        "kind": "slot-integrity-gate",
        "created_at": utc_now(),
        "arm": args.arm,
        "config_state": expected_state,
        "case_dir": str(case_dir),
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "bundle_manifest_sha256": manifest["bundle"]["manifest_sha256"],
        "frequency_mask_sha256": EXPECTED_MASK_SHA,
        "output": {
            "request_token_id_sha256": request_hashes,
            "c8_output_token_ids_sha256": sha256_json(outputs) if len(outputs) == 8 else None,
        },
        "metrics": metrics,
        "warning": {
            "dense_fallback_count": len(warning.get("parsed", {}).get("dense_fallbacks", [])),
            "dense_tuned_load_count": len(warning.get("parsed", {}).get("dense_tuned_loads", [])),
            "faq_ignored_fused_moe_warning_count": warning.get("parsed", {}).get("faq_ignored_fused_moe_warning_count"),
            "dense_config_clamp_warning_count": warning.get("parsed", {}).get("dense_config_clamp_warning_count"),
        },
        "placement": {
            "runtime_hot_experts": checks.get("runtime_hot_experts"),
            "runtime_cold_experts": EXPECTED_COLD_EXPERTS,
            "runtime_mask_layers": checks.get("runtime_mask_layers"),
        },
    }
    atomic_json(args.output, result)
    return result


def cv(values: list[float]) -> float | None:
    if len(values) < 2 or statistics.mean(values) == 0:
        return None
    return statistics.stdev(values) / statistics.mean(values)


def summarize(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    manifest = load_json(run_root / "run_manifest.json")
    status = load_json(run_root / "execution_status.json")
    failures: list[str] = []
    expected_order = [(slot, arm, state) for slot, arm, state in SCHEDULE]
    actual_order = [(item["slot"], item["arm"], item["config_state"]) for item in status["slots"]]
    if actual_order != expected_order:
        failures.append("execution order differs from preregistration")
    if any(item["status"] != "ok" or item["returncode"] != 0 for item in status["slots"]):
        failures.append("one or more preregistered slots did not pass")
    slots = []
    for item in status["slots"]:
        gate_path = Path(item["case_root"]) / "integrity_gate.json"
        if not gate_path.is_file():
            failures.append(f"slot {item['slot']} lacks integrity gate")
            continue
        gate = load_json(gate_path)
        if gate.get("status") != "ok":
            failures.append(f"slot {item['slot']} integrity gate failed")
        if (
            gate.get("schema") != SCHEMA
            or gate.get("arm") != item["arm"]
            or gate.get("config_state") != item["config_state"]
            or Path(gate.get("case_dir", "")).resolve() != Path(item["case_root"]).resolve()
        ):
            failures.append(f"slot {item['slot']} integrity gate identity differs")
        slots.append({**item, "gate": gate})
    bundle_shas = {item["gate"].get("bundle_manifest_sha256") for item in slots}
    if bundle_shas != {manifest["bundle"]["manifest_sha256"]}:
        failures.append("slots did not use one frozen tuning bundle")
    output_hash_values = [
        item["gate"].get("output", {}).get("c8_output_token_ids_sha256")
        for item in slots
    ]
    output_hashes = {value for value in output_hash_values if value is not None}
    output_exact = (
        len(slots) == 4
        and len(output_hash_values) == 4
        and all(value is not None for value in output_hash_values)
        and len(output_hashes) == 1
    )
    if not output_exact:
        failures.append("D/T output token-ID hashes are not exactly identical")

    arms: dict[str, Any] = {}
    for arm in ("D", "T"):
        arm_slots = [item for item in slots if item["arm"] == arm]
        if len(arm_slots) != 2:
            failures.append(f"arm {arm} does not contain exactly two slots")
        metrics: dict[str, Any] = {}
        for key in (
            "output_tokens_per_second",
            "mean_ttft_ms",
            "mean_tpot_ms",
            "ready_memory_used_mib",
            "phase_peak_memory_used_mib",
        ):
            values = [float(item["gate"]["metrics"][key]) for item in arm_slots if item["gate"]["metrics"].get(key) is not None]
            metrics[key] = {
                "values": values,
                "mean": statistics.mean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "cv": cv(values),
            }
        arms[arm] = {"sample_count": len(arm_slots), "metrics": metrics}
    d_tps = arms["D"]["metrics"]["output_tokens_per_second"]["mean"]
    t_tps = arms["T"]["metrics"]["output_tokens_per_second"]["mean"]
    d_ttft = arms["D"]["metrics"]["mean_ttft_ms"]["mean"]
    t_ttft = arms["T"]["metrics"]["mean_ttft_ms"]["mean"]
    d_tpot = arms["D"]["metrics"]["mean_tpot_ms"]["mean"]
    t_tpot = arms["T"]["metrics"]["mean_tpot_ms"]["mean"]
    comparisons = {
        "tuned_over_default_output_tok_s": t_tps / d_tps if d_tps and t_tps else None,
        "default_over_tuned_mean_ttft": d_ttft / t_ttft if d_ttft and t_ttft else None,
        "default_over_tuned_mean_tpot": d_tpot / t_tpot if d_tpot and t_tpot else None,
    }
    result = {
        "schema": SCHEMA,
        "kind": "dttd-summary",
        "created_at": utc_now(),
        "run_root": str(run_root),
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "schedule": status["slots"],
        "bundle_manifest_sha256": manifest["bundle"]["manifest_sha256"],
        "output_hash_exact_across_d_t": output_exact,
        "c8_output_token_ids_sha256": next(iter(output_hashes)) if len(output_hashes) == 1 else None,
        "arms": arms,
        "comparisons": comparisons,
        "interpretation": (
            "screening-only dense-kernel config ablation; not a custom-vs-KT speedup sample"
        ),
    }
    atomic_json(run_root / "DENSE_FP8_DTTD_SUMMARY.json", result)
    lines = [
        "# KT dense FP8 tuned-vs-default D-T-T-D",
        "",
        f"- Status: `{result['status']}`",
        f"- Bundle SHA-256: `{result['bundle_manifest_sha256']}`",
        f"- D/T output token-ID exact: `{str(output_exact).lower()}`",
        f"- Common c8 output hash: `{result['c8_output_token_ids_sha256']}`",
        "",
        "| Arm | n | output tok/s | mean TTFT ms | mean TPOT ms |",
        "|---|---:|---:|---:|---:|",
    ]

    def markdown_number(value: float | None) -> str:
        return "nan" if value is None else f"{value:.6f}"

    for arm in ("D", "T"):
        metric = arms[arm]["metrics"]
        lines.append(
            f"| {arm} | {arms[arm]['sample_count']} | "
            f"{markdown_number(metric['output_tokens_per_second']['mean'])} | "
            f"{markdown_number(metric['mean_ttft_ms']['mean'])} | "
            f"{markdown_number(metric['mean_tpot_ms']['mean'])} |"
        )
    lines.extend(
        [
            "",
            f"- Tuned/default output tok/s: `{comparisons['tuned_over_default_output_tok_s']}`",
            f"- Default/tuned mean TTFT: `{comparisons['default_over_tuned_mean_ttft']}`",
            f"- Default/tuned mean TPOT: `{comparisons['default_over_tuned_mean_tpot']}`",
            "",
            "This is a dense-config screening ablation, not a cross-system speedup result.",
        ]
    )
    if failures:
        lines.extend(["", "## Failures", ""] + [f"- {item}" for item in failures])
    atomic_text(run_root / "DENSE_FP8_DTTD_SUMMARY.md", "\n".join(lines) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    state = sub.add_parser("config-state")
    state.add_argument("--bundle-dir", required=True, type=Path)
    state.add_argument("--kt-root", required=True, type=Path)
    state.add_argument("--output", type=Path)
    transition = sub.add_parser("transition")
    transition.add_argument("--bundle-dir", required=True, type=Path)
    transition.add_argument("--kt-root", required=True, type=Path)
    transition.add_argument("--target", choices=["default", "tuned"], required=True)
    transition.add_argument("--journal", required=True, type=Path)
    prereg = sub.add_parser("preregister")
    prereg.add_argument("--run-root", required=True, type=Path)
    prereg.add_argument("--bundle-dir", required=True, type=Path)
    prereg.add_argument("--kt-root", required=True, type=Path)
    prereg.add_argument("--evaluation-fixture", required=True, type=Path)
    prereg.add_argument("--warmup-fixture", required=True, type=Path)
    prereg.add_argument("--frequency-stats", required=True, type=Path)
    prereg.add_argument("--stats-manifest", required=True, type=Path)
    update = sub.add_parser("update-status")
    update.add_argument("--run-root", required=True, type=Path)
    update.add_argument("--slot", required=True, type=int)
    update.add_argument("--status", choices=["running", "ok", "failed", "skipped"], required=True)
    update.add_argument("--returncode", type=int)
    update.add_argument("--timestamp")
    update.add_argument("--detail")
    verify = sub.add_parser("verify-slot")
    verify.add_argument("--case-dir", required=True, type=Path)
    verify.add_argument("--run-manifest", required=True, type=Path)
    verify.add_argument("--arm", choices=["D", "T"], required=True)
    verify.add_argument("--output", required=True, type=Path)
    summary = sub.add_parser("summarize")
    summary.add_argument("--run-root", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "config-state":
            result = inspect_config_state(args.bundle_dir, args.kt_root)
            if args.output:
                atomic_json(args.output, result)
        elif args.command == "transition":
            result = transition_config(
                bundle_dir=args.bundle_dir,
                kt_root=args.kt_root,
                target_state=args.target,
                journal=args.journal,
            )
        elif args.command == "preregister":
            result = preregister(args)
        elif args.command == "update-status":
            result = update_status(args)
        elif args.command == "verify-slot":
            result = verify_slot(args)
        elif args.command == "summarize":
            result = summarize(args.run_root)
        else:
            raise AssertionError(args.command)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.get("status") not in {"failed", "invalid"} else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
