#!/usr/bin/env python3
"""Preregister, gate, and summarize cold-process common-trace ABBA runs.

This module is deliberately CPU-only.  The companion shell runner owns GPU
processes and calls this module before and after each immutable schedule slot.
The independent statistical unit is an ABBA block; the repeated iterations
inside one microbenchmark process are technical repeats only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "common-trace-abba-v1"
EXPECTED_KT_COMMIT = "eb9b70c4115cff151ace2cae5b0fc9db3690e31e"
EXPECTED_SGLANG_COMMIT = "bc7f0058fafe6c05dc6ad52d6b1c854af2a673b2"
EXPECTED_REMOTE_BYTES = 603_979_776
MIN_BLOCKS = 3
ARMS = {"A": "kt", "B": "custom"}
REPO_ROOT = Path(__file__).resolve().parents[2]
MICROBENCH = Path(__file__).with_name("common_trace_moe_microbench.py")
RUNNER = Path(__file__).with_name("run_common_trace_abba.sh")
CUSTOM_IMPLEMENTATION_PATHS = (
    "workloads/vllm/vllm/vllm/device_allocator/uvm.py",
    "workloads/vllm/vllm/vllm/device_allocator/uvm_pool_residency.py",
    "workloads/vllm/vllm/vllm/device_allocator/uvm_region.py",
    "workloads/vllm/vllm/vllm/model_executor/layers/fused_moe/layer.py",
    "workloads/vllm/vllm/vllm/model_executor/models/qwen3_moe.py",
    "workloads/vllm/vllm/vllm/model_executor/models/utils.py",
    "workloads/vllm/vllm/vllm/v1/worker/gpu_model_runner.py",
    "workloads/vllm/vllm/vllm/v1/worker/gpu_worker.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


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


def git_output(root: Path, *args: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    return completed.stdout


def copy_snapshot(source: Path, target: Path) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)
    return {
        "source_path": str(source),
        "snapshot_path": str(target.resolve()),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
    }


def expected_sequence(block: int) -> list[str]:
    if block <= 0:
        raise ValueError("block must be positive")
    return ["A", "B", "B", "A"] if block % 2 else ["B", "A", "A", "B"]


def output_content_sha(path: Path) -> str:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        bits = archive["output_bf16_bits"].copy()
    return sha256_bytes(bits.tobytes(order="C"))


def validate_reference(
    reference: Path, fixture_sha: str, model_index_sha: str, layer: int
) -> dict[str, Any]:
    import numpy as np

    with np.load(reference, allow_pickle=False) as archive:
        required = {
            "output_bf16_bits",
            "output_float32",
            "fixture_sha256",
            "model_index_sha256",
            "layer",
        }
        if not required.issubset(archive.files):
            raise ValueError("reference output lacks required provenance fields")
        if str(archive["fixture_sha256"].item()) != fixture_sha:
            raise ValueError("reference fixture SHA differs")
        if str(archive["model_index_sha256"].item()) != model_index_sha:
            raise ValueError("reference model-index SHA differs")
        if int(archive["layer"].item()) != layer:
            raise ValueError("reference layer differs")
        shape = list(archive["output_bf16_bits"].shape)
    return {
        "sha256": sha256_file(reference),
        "output_content_sha256": output_content_sha(reference),
        "shape": shape,
    }


def module_provenance() -> dict[str, Any]:
    loaded_path = Path("/sys/module/nvidia_uvm/srcversion")
    module_path = (
        REPO_ROOT
        / "kernel-module/nvidia-module/kernel-open/nvidia-uvm.ko"
    )
    loaded = loaded_path.read_text(encoding="utf-8").strip() if loaded_path.is_file() else None
    worktree = None
    if module_path.is_file():
        completed = subprocess.run(
            ["modinfo", "-F", "srcversion", str(module_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            worktree = completed.stdout.strip()
    return {
        "loaded_srcversion": loaded,
        "worktree_srcversion": worktree,
        "match": bool(loaded and worktree and loaded == worktree),
    }


def preregister(args: argparse.Namespace) -> dict[str, Any]:
    if args.blocks < MIN_BLOCKS:
        raise ValueError(f"formal ABBA requires at least {MIN_BLOCKS} blocks")
    if args.threads <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("invalid threads/warmup/iterations")
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"refusing to reuse run root: {run_root}")
    model = args.model.resolve()
    model_config = model / "config.json"
    model_index = model / "model.safetensors.index.json"
    for path in (model_config, model_index, MICROBENCH, RUNNER):
        if not path.is_file():
            raise FileNotFoundError(path)

    fixture_manifest = load_json(args.fixture_manifest)
    fixture_meta = fixture_manifest.get("fixture")
    shape = fixture_manifest.get("shape")
    weights = fixture_manifest.get("weights")
    if not all(isinstance(item, dict) for item in (fixture_meta, shape, weights)):
        raise ValueError("fixture manifest is incomplete")
    fixture_sha = sha256_file(args.fixture)
    if fixture_sha != fixture_meta.get("sha256"):
        raise ValueError("fixture SHA differs from its manifest")
    if sha256_file(model_config) != weights.get("config_sha256"):
        raise ValueError("model config differs from fixture provenance")
    model_index_sha = sha256_file(model_index)
    if model_index_sha != weights.get("index_sha256"):
        raise ValueError("model index differs from fixture provenance")
    layer = shape.get("layer")
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise ValueError("fixture layer is invalid")

    tolerance = load_json(args.tolerance)
    thresholds = tolerance.get("thresholds")
    held_out = tolerance.get("held_out")
    if not isinstance(thresholds, dict) or not isinstance(held_out, dict):
        raise ValueError("tolerance preregistration is incomplete")
    if tolerance.get("registered_before_held_out_run") is not True:
        raise ValueError("tolerance was not preregistered before held-out execution")
    if held_out.get("fixture_layer") != layer:
        raise ValueError("tolerance held-out layer differs from fixture")
    for key in ("atol", "rtol", "max_relative_l1"):
        value = thresholds.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"invalid tolerance threshold: {key}")
    reference = validate_reference(
        args.reference_output, fixture_sha, model_index_sha, layer
    )

    kt_root = REPO_ROOT / "KTransformers"
    kt_commit = str(git_output(kt_root, "rev-parse", "HEAD")).strip()
    sglang_commit = str(
        git_output(kt_root / "third_party/sglang", "rev-parse", "HEAD")
    ).strip()
    if kt_commit != EXPECTED_KT_COMMIT or sglang_commit != EXPECTED_SGLANG_COMMIT:
        raise ValueError("KTransformers/SGLang commit drift")
    custom_diff = git_output(
        REPO_ROOT,
        "diff",
        "--binary",
        "HEAD",
        "--",
        *CUSTOM_IMPLEMENTATION_PATHS,
        binary=True,
    )
    assert isinstance(custom_diff, bytes)
    uvm = module_provenance()
    if not uvm["match"]:
        raise ValueError("loaded/worktree UVM srcversion mismatch")

    inputs = run_root / "inputs"
    inputs.mkdir(parents=True)
    snapshots = {
        "fixture": copy_snapshot(args.fixture, inputs / "fixture.npz"),
        "fixture_manifest": copy_snapshot(
            args.fixture_manifest, inputs / "fixture_manifest.json"
        ),
        "reference_output": copy_snapshot(
            args.reference_output, inputs / "reference_output.npz"
        ),
        "tolerance": copy_snapshot(args.tolerance, inputs / "tolerance.json"),
    }
    schedule: list[dict[str, Any]] = []
    for block in range(1, args.blocks + 1):
        for slot, arm in enumerate(expected_sequence(block), start=1):
            mechanism = ARMS[arm]
            case_root = run_root / f"block{block}" / f"slot{slot}_{arm}_{mechanism}" / "result"
            schedule.append(
                {
                    "block": block,
                    "slot": slot,
                    "arm": arm,
                    "mechanism": mechanism,
                    "case_root": str(case_root),
                }
            )
    manifest = {
        "schema": SCHEMA,
        "kind": "preregistered-common-trace-abba",
        "created_at": utc_now(),
        "run_root": str(run_root),
        "model": {
            "path": str(model),
            "config_sha256": sha256_file(model_config),
            "index_sha256": model_index_sha,
        },
        "inputs": snapshots,
        "reference": reference,
        "thresholds": {key: float(thresholds[key]) for key in ("atol", "rtol", "max_relative_l1")},
        "fixture": {
            "sha256": fixture_sha,
            "layer": layer,
            "tokens": shape.get("tokens"),
            "seed": fixture_meta.get("seed"),
        },
        "runtime": {
            "threads": args.threads,
            "numa_node": args.numa_node,
            "cpuset": args.cpuset,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "provenance": {
            "repo_commit": str(git_output(REPO_ROOT, "rev-parse", "HEAD")).strip(),
            "custom_relevant_diff_sha256": sha256_bytes(custom_diff),
            "ktransformers_commit": kt_commit,
            "sglang_commit": sglang_commit,
            "microbench_sha256": sha256_file(MICROBENCH),
            "orchestrator_sha256": sha256_file(Path(__file__).resolve()),
            "runner_sha256": sha256_file(RUNNER),
            "uvm": uvm,
        },
        "statistics": {
            "independent_unit": "one complete four-slot cold-process ABBA block",
            "technical_repeats_per_process": args.iterations,
            "minimum_blocks": MIN_BLOCKS,
            "bootstrap_unit": "complete ABBA block",
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 0,
        },
        "schedule": schedule,
        "interpretation": {
            "eligible_scope": "path-level common-trace comparison only",
            "system_main_table_eligible": False,
            "causal_warning": (
                "The arms also differ in CPU/GPU kernel, compute device, and "
                "activation quantization; a ratio is not a pure transfer effect."
            ),
        },
    }
    atomic_json(run_root / "run_manifest.json", manifest)
    status = {
        "schema": SCHEMA,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "slots": [
            {
                **item,
                "status": "pending",
                "returncode": None,
                "started_at": None,
                "ended_at": None,
                "detail": None,
            }
            for item in schedule
        ],
    }
    atomic_json(run_root / "execution_status.json", status)
    order_path = run_root / "abba_order.tsv"
    with order_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            fieldnames=["block", "slot", "arm", "mechanism", "case_root"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(schedule)
    return manifest


def update_status(args: argparse.Namespace) -> dict[str, Any]:
    path = args.run_root.resolve() / "execution_status.json"
    status = load_json(path)
    matches = [
        item
        for item in status.get("slots", [])
        if item.get("block") == args.block and item.get("slot") == args.slot
    ]
    if len(matches) != 1:
        raise ValueError("status coordinate is not unique")
    item = matches[0]
    now = utc_now()
    if args.status == "running":
        if item.get("status") != "pending":
            raise ValueError("only a pending slot may start")
        item["started_at"] = now
    else:
        if item.get("status") != "running":
            raise ValueError("only a running slot may finish")
        item["ended_at"] = now
    item["status"] = args.status
    item["returncode"] = args.returncode
    item["detail"] = args.detail
    status["updated_at"] = now
    atomic_json(path, status)
    return item


def validate_manifest(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    invalidation = run_root / "PRELAUNCH_INVALID.md"
    if invalidation.exists():
        raise ValueError(f"run root was invalidated before launch: {invalidation}")
    manifest = load_json(run_root / "run_manifest.json")
    failures: list[str] = []
    if manifest.get("schema") != SCHEMA:
        failures.append("manifest schema differs")
    if Path(manifest.get("run_root", "")).resolve() != run_root:
        failures.append("manifest run-root identity differs")
    provenance = manifest.get("provenance", {})
    current_hashes = {
        "microbench_sha256": sha256_file(MICROBENCH),
        "orchestrator_sha256": sha256_file(Path(__file__).resolve()),
        "runner_sha256": sha256_file(RUNNER),
    }
    for key, value in current_hashes.items():
        if provenance.get(key) != value:
            failures.append(f"{key} drift")
    for name, record in manifest.get("inputs", {}).items():
        if not isinstance(record, dict):
            failures.append(f"input record is invalid: {name}")
            continue
        path = Path(record.get("snapshot_path", ""))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            failures.append(f"input snapshot drift: {name}")
    model = manifest.get("model", {})
    model_root = Path(model.get("path", ""))
    if (
        not (model_root / "config.json").is_file()
        or sha256_file(model_root / "config.json") != model.get("config_sha256")
    ):
        failures.append("model config drift")
    if (
        not (model_root / "model.safetensors.index.json").is_file()
        or sha256_file(model_root / "model.safetensors.index.json") != model.get("index_sha256")
    ):
        failures.append("model index drift")
    current_uvm = module_provenance()
    if current_uvm != provenance.get("uvm") or not current_uvm.get("match"):
        failures.append("UVM module provenance drift")
    kt_root = REPO_ROOT / "KTransformers"
    if str(git_output(kt_root, "rev-parse", "HEAD")).strip() != provenance.get("ktransformers_commit"):
        failures.append("KTransformers commit drift")
    if str(git_output(kt_root / "third_party/sglang", "rev-parse", "HEAD")).strip() != provenance.get("sglang_commit"):
        failures.append("SGLang commit drift")
    schedule = manifest.get("schedule", [])
    for block in range(1, max((int(item.get("block", 0)) for item in schedule), default=0) + 1):
        rows = [item for item in schedule if item.get("block") == block]
        if [item.get("arm") for item in rows] != expected_sequence(block):
            failures.append(f"schedule drift at block {block}")
        for item in rows:
            case_root = Path(item.get("case_root", "")).resolve()
            if run_root not in case_root.parents:
                failures.append(f"case root escapes run root at block {block}")
    result = {
        "schema": SCHEMA,
        "kind": "manifest-validation",
        "created_at": utc_now(),
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "current_hashes": current_hashes,
        "uvm": current_uvm,
    }
    return result


def finite_positive(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    )


def verify_slot(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    manifest = load_json(run_root / "run_manifest.json")
    matches = [
        item
        for item in manifest.get("schedule", [])
        if item.get("block") == args.block and item.get("slot") == args.slot
    ]
    if len(matches) != 1:
        raise ValueError("manifest coordinate is not unique")
    schedule = matches[0]
    case_root = Path(schedule["case_root"])
    result_path = case_root / "result.json"
    failures: list[str] = []
    try:
        result = load_json(result_path)
    except Exception as exc:
        result = {}
        failures.append(f"result unavailable: {exc}")
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    mechanism = result.get("mechanism") if isinstance(result.get("mechanism"), dict) else {}
    timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
    validation = result.get("numerical_validation") if isinstance(result.get("numerical_validation"), dict) else {}
    if result.get("status") != "ok":
        failures.append("microbenchmark status is not ok")
    if metadata.get("command") != f"run-{schedule['mechanism']}":
        failures.append("command/mechanism schedule mismatch")
    if metadata.get("fixture_sha256") != manifest["fixture"]["sha256"]:
        failures.append("fixture SHA mismatch")
    if metadata.get("model_index_sha256") != manifest["model"]["index_sha256"]:
        failures.append("model-index SHA mismatch")
    if metadata.get("script_sha256") != manifest["provenance"]["microbench_sha256"]:
        failures.append("microbenchmark source drift")
    if validation.get("passed") is not True:
        failures.append("numerical validation did not pass")
    if not finite_positive(timing.get("inclusive_boundary_ms", {}).get("mean")):
        failures.append("inclusive mean is invalid")

    metrics: dict[str, Any] = {
        "inclusive_boundary_ms": timing.get("inclusive_boundary_ms", {}).get("mean"),
        "weight_load_seconds": mechanism.get("weight_load_seconds"),
    }
    placement_gate = None
    if schedule["mechanism"] == "kt":
        if mechanism.get("name") != "kt_kernel_fp8_cpu_expert":
            failures.append("KT mechanism identity differs")
        if mechanism.get("all_experts_on_cpu") is not True:
            failures.append("KT did not report all experts on CPU")
        if mechanism.get("threads") != manifest["runtime"]["threads"]:
            failures.append("KT thread count differs")
        compute = timing.get("compute_only_ms", {}).get("mean")
    else:
        if mechanism.get("name") != "custom_uvm_gpu_remote_read":
            failures.append("custom mechanism identity differs")
        if mechanism.get("all_experts_remote") is not True:
            failures.append("custom did not report all experts remote")
        compute = timing.get("compute_only_synchronized_wall_ms", {}).get("mean")
        placement_path = case_root / "placement_manifest.json"
        try:
            placement = load_json(placement_path)
        except Exception as exc:
            placement = {}
            failures.append(f"placement manifest unavailable: {exc}")
        expected = placement.get("expected_main_weight_bytes")
        post = placement.get("post_measurement_residency")
        gate = placement.get("measurement_window_residency_gate")
        if expected != EXPECTED_REMOTE_BYTES:
            failures.append("custom remote byte topology differs")
        if not isinstance(post, dict) or (
            post.get("ok") is not True
            or post.get("queried_bytes") != EXPECTED_REMOTE_BYTES
            or post.get("cpu_bytes") != EXPECTED_REMOTE_BYTES
            or post.get("gpu_bytes") != 0
            or post.get("unknown_bytes") != 0
        ):
            failures.append("post-measurement CPU residency gate failed")
        if not isinstance(gate, dict) or gate.get("status") != "pass":
            failures.append("measurement-window residency declaration missing")
        if mechanism.get("placement_manifest_sha256") != (
            sha256_file(placement_path) if placement_path.is_file() else None
        ):
            failures.append("placement manifest SHA mismatch")
        placement_gate = {
            "expected_bytes": expected,
            "post_measurement_residency": post,
            "status": gate.get("status") if isinstance(gate, dict) else None,
        }
    if not finite_positive(compute):
        failures.append("synchronized compute mean is invalid")
    metrics["compute_synchronized_wall_ms"] = compute
    if schedule["mechanism"] == "custom":
        metrics["compute_cuda_event_ms"] = timing.get("compute_only_ms", {}).get("mean")

    output_path = case_root / "output.npz"
    output_hash = None
    try:
        output_hash = output_content_sha(output_path)
    except Exception as exc:
        failures.append(f"output content unavailable: {exc}")
    gate_result = {
        "schema": SCHEMA,
        "kind": "common-trace-slot-gate",
        "created_at": utc_now(),
        "block": args.block,
        "slot": args.slot,
        "arm": schedule["arm"],
        "mechanism": schedule["mechanism"],
        "case_root": str(case_root),
        "status": "pass" if not failures else "failed",
        "failures": failures,
        "metrics": metrics,
        "output_content_sha256": output_hash,
        "numerical_validation": validation,
        "placement_gate": placement_gate,
    }
    atomic_json(case_root.parent / "slot_gate.json", gate_result)
    return gate_result


def cv(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    return statistics.stdev(values) / mean if mean else None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def cluster_bootstrap_ratio(
    blocks: list[dict[str, list[float]]], *, samples: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    if not blocks or any(not block.get("kt") or not block.get("custom") for block in blocks):
        raise ValueError("each bootstrap block needs both arms")
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        selected = [rng.choice(blocks) for _ in blocks]
        kt = [value for block in selected for value in block["kt"]]
        custom = [value for block in selected for value in block["custom"]]
        estimates.append(statistics.fmean(kt) / statistics.fmean(custom))
    return {
        "samples": samples,
        "seed": seed,
        "lower_95": percentile(estimates, 0.025),
        "upper_95": percentile(estimates, 0.975),
    }


def metric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "sample_stddev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        "cv": cv(values),
        "values": values,
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    validation = validate_manifest(args)
    if validation["status"] != "ok":
        raise ValueError(f"run manifest validation failed: {validation['failures']}")
    manifest = load_json(run_root / "run_manifest.json")
    execution = load_json(run_root / "execution_status.json")
    failures: list[str] = []
    expected = manifest.get("schedule", [])
    actual = execution.get("slots", [])
    identity = lambda item: (item.get("block"), item.get("slot"), item.get("arm"), item.get("mechanism"), item.get("case_root"))
    if [identity(item) for item in actual] != [identity(item) for item in expected]:
        failures.append("execution schedule identity/order differs from preregistration")
    if any(item.get("status") != "success" or item.get("returncode") != 0 for item in actual):
        failures.append("one or more preregistered slots did not complete successfully")

    slots: list[dict[str, Any]] = []
    for item in expected:
        gate_path = Path(item["case_root"]).parent / "slot_gate.json"
        if not gate_path.is_file():
            failures.append(f"missing slot gate: block={item['block']} slot={item['slot']}")
            continue
        gate = load_json(gate_path)
        if gate.get("status") != "pass":
            failures.append(f"failed slot gate: block={item['block']} slot={item['slot']}")
        if identity(gate)[:4] != identity(item)[:4]:
            failures.append(f"slot gate identity mismatch: block={item['block']} slot={item['slot']}")
        slots.append({**item, "gate": gate})

    block_count = int(max((item["block"] for item in expected), default=0))
    sequences = {
        block: [item["arm"] for item in expected if item["block"] == block]
        for block in range(1, block_count + 1)
    }
    for block, sequence in sequences.items():
        if sequence != expected_sequence(block):
            failures.append(f"block {block} sequence differs from preregistration rule")

    arms: dict[str, Any] = {}
    for mechanism in ("kt", "custom"):
        arm_slots = [item for item in slots if item["mechanism"] == mechanism]
        compute_values = [
            float(item["gate"]["metrics"]["compute_synchronized_wall_ms"])
            for item in arm_slots
            if finite_positive(item["gate"].get("metrics", {}).get("compute_synchronized_wall_ms"))
        ]
        inclusive_values = [
            float(item["gate"]["metrics"]["inclusive_boundary_ms"])
            for item in arm_slots
            if finite_positive(item["gate"].get("metrics", {}).get("inclusive_boundary_ms"))
        ]
        hashes = [item["gate"].get("output_content_sha256") for item in arm_slots]
        stable = bool(hashes) and all(value is not None for value in hashes) and len(set(hashes)) == 1
        if not stable:
            failures.append(f"{mechanism} output is not stable across cold processes")
        arms[mechanism] = {
            "cold_processes": len(arm_slots),
            "compute_synchronized_wall_ms": metric_summary(compute_values),
            "inclusive_boundary_ms": metric_summary(inclusive_values),
            "output_content_stable": stable,
            "output_content_sha256": hashes[0] if stable else None,
        }

    comparisons: dict[str, Any] = {}
    for metric in ("compute_synchronized_wall_ms", "inclusive_boundary_ms"):
        blocks: list[dict[str, list[float]]] = []
        block_ratios: list[dict[str, Any]] = []
        for block in range(1, block_count + 1):
            by_arm = {
                mechanism: [
                    float(item["gate"]["metrics"][metric])
                    for item in slots
                    if item["block"] == block and item["mechanism"] == mechanism
                ]
                for mechanism in ("kt", "custom")
            }
            if len(by_arm["kt"]) != 2 or len(by_arm["custom"]) != 2:
                failures.append(f"block {block} lacks two successful processes per arm")
                continue
            blocks.append(by_arm)
            block_ratios.append(
                {
                    "block": block,
                    "kt_over_custom": statistics.fmean(by_arm["kt"]) / statistics.fmean(by_arm["custom"]),
                }
            )
        kt_values = arms["kt"][metric]["values"]
        custom_values = arms["custom"][metric]["values"]
        comparisons[metric] = {
            "ratio_definition": "mean KT latency / mean custom latency",
            "kt_over_custom": (
                statistics.fmean(kt_values) / statistics.fmean(custom_values)
                if kt_values and custom_values
                else None
            ),
            "block_ratios": block_ratios,
            "paired_cluster_bootstrap_95": (
                cluster_bootstrap_ratio(blocks) if len(blocks) == block_count and blocks else None
            ),
        }

    minimum_complete = block_count >= MIN_BLOCKS and len(slots) == len(expected)
    path_eligible = minimum_complete and not failures
    if not minimum_complete:
        failures.append(f"fewer than {MIN_BLOCKS} complete ABBA blocks")
        path_eligible = False
    high_cv = [
        {"mechanism": mechanism, "metric": metric, "cv": arms[mechanism][metric]["cv"]}
        for mechanism in ("kt", "custom")
        for metric in ("compute_synchronized_wall_ms", "inclusive_boundary_ms")
        if arms[mechanism][metric]["cv"] is not None
        and arms[mechanism][metric]["cv"] > 0.05
    ]
    if high_cv:
        failures.append(
            "one or more arm-level cold-process CV values exceed 5%; "
            "extend to at least 10 cold processes per arm"
        )
        path_eligible = False
    result = {
        "schema": SCHEMA,
        "kind": "common-trace-abba-summary",
        "created_at": utc_now(),
        "run_root": str(run_root),
        "status": "ok" if path_eligible else "failed",
        "failures": failures,
        "block_count": block_count,
        "minimum_blocks_complete": minimum_complete,
        "arms": arms,
        "comparisons": comparisons,
        "high_cv_over_5_percent": high_cv,
        "additional_repeats_required": bool(high_cv),
        "eligible_for_path_level_table": path_eligible,
        "eligible_for_system_main_table": False,
        "interpretation": (
            "Cold-process path-level comparison on one frozen layer/shape. "
            "The ratio also changes CPU/GPU kernel, compute device, and activation "
            "quantization, so it is not a pure transfer-mechanism effect."
        ),
    }
    atomic_json(run_root / "COMMON_TRACE_ABBA_SUMMARY.json", result)
    lines = [
        "# Common routing trace cold-process ABBA",
        "",
        f"- Status: `{result['status']}`",
        f"- Complete ABBA blocks: `{block_count}`",
        f"- Path-level table eligible: `{str(path_eligible).lower()}`",
        "- System main-table eligible: `false`",
        "",
        "| Metric | KT mean ms | custom mean ms | KT/custom | block-bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in ("compute_synchronized_wall_ms", "inclusive_boundary_ms"):
        comp = comparisons[metric]
        ci = comp["paired_cluster_bootstrap_95"]
        ci_text = "—" if ci is None else f"[{ci['lower_95']:.4f}, {ci['upper_95']:.4f}]"
        lines.append(
            f"| {metric} | {arms['kt'][metric]['mean']:.6f} | "
            f"{arms['custom'][metric]['mean']:.6f} | {comp['kt_over_custom']:.6f} | {ci_text} |"
        )
    lines.extend(
        [
            "",
            "The bootstrap resamples complete ABBA blocks. In-process iterations are technical repeats only.",
            "This remains a path-level comparison and cannot identify a pure transfer effect.",
        ]
    )
    if failures:
        lines.extend(["", "## Failures", ""] + [f"- {item}" for item in failures])
    atomic_text(run_root / "COMMON_TRACE_ABBA_RESULTS.md", "\n".join(lines) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prereg = sub.add_parser("preregister")
    prereg.add_argument("--run-root", required=True, type=Path)
    prereg.add_argument("--model", required=True, type=Path)
    prereg.add_argument("--fixture", required=True, type=Path)
    prereg.add_argument("--fixture-manifest", required=True, type=Path)
    prereg.add_argument("--reference-output", required=True, type=Path)
    prereg.add_argument("--tolerance", required=True, type=Path)
    prereg.add_argument("--blocks", type=int, default=3)
    prereg.add_argument("--threads", type=int, default=12)
    prereg.add_argument("--numa-node", type=int, default=0)
    prereg.add_argument("--cpuset", default="0-13")
    prereg.add_argument("--warmup", type=int, default=3)
    prereg.add_argument("--iterations", type=int, default=20)
    update = sub.add_parser("update-status")
    update.add_argument("--run-root", required=True, type=Path)
    update.add_argument("--block", required=True, type=int)
    update.add_argument("--slot", required=True, type=int)
    update.add_argument("--status", choices=["running", "success", "failed", "skipped"], required=True)
    update.add_argument("--returncode", type=int)
    update.add_argument("--detail")
    validate = sub.add_parser("validate-manifest")
    validate.add_argument("--run-root", required=True, type=Path)
    verify = sub.add_parser("verify-slot")
    verify.add_argument("--run-root", required=True, type=Path)
    verify.add_argument("--block", required=True, type=int)
    verify.add_argument("--slot", required=True, type=int)
    summary = sub.add_parser("summarize")
    summary.add_argument("--run-root", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "preregister":
            result = preregister(args)
        elif args.command == "update-status":
            result = update_status(args)
        elif args.command == "validate-manifest":
            result = validate_manifest(args)
        elif args.command == "verify-slot":
            result = verify_slot(args)
        elif args.command == "summarize":
            result = summarize(args)
        else:
            raise AssertionError(args.command)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.get("status") not in {"failed", "invalid"} else 1
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=os.sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
