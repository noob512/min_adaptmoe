#!/usr/bin/env python3
"""One-layer common-routing-trace microbenchmark for KT and custom UVM.

The module deliberately imports neither torch nor either runtime before a GPU
subcommand has parsed ``--allow-gpu-run``.  ``prepare`` is CPU-only and writes
the exact BF16 hidden states, top-k expert IDs, and top-k weights consumed by
both execution paths.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
EXPECTED_MODEL = {
    "hidden_size": 2048,
    "moe_intermediate_size": 768,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
}
EXPECTED_BLOCK_SHAPE = [128, 128]
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
WEIGHT_SUFFIXES = ("weight", "weight_scale_inv")
REPO_ROOT = Path(__file__).resolve().parents[2]
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def run_capture(argv: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def run_capture_bytes(argv: list[str], cwd: Path | None = None) -> bytes | None:
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def implementation_provenance(command: str) -> dict[str, Any]:
    main_commit = run_capture(["git", "rev-parse", "HEAD"], REPO_ROOT)
    main_diff = run_capture_bytes(
        ["git", "diff", "--binary", "HEAD", "--", *CUSTOM_IMPLEMENTATION_PATHS],
        REPO_ROOT,
    )
    ktransformers = REPO_ROOT / "KTransformers"
    kt_commit = run_capture(["git", "rev-parse", "HEAD"], ktransformers)
    kt_diff = run_capture_bytes(
        ["git", "diff", "--binary", "HEAD"], ktransformers
    )
    kt_status = run_capture(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        ktransformers,
    )
    sglang_commit = run_capture(
        ["git", "rev-parse", "HEAD"], ktransformers / "third_party" / "sglang"
    )
    loaded_srcversion = None
    loaded_path = Path("/sys/module/nvidia_uvm/srcversion")
    if loaded_path.is_file():
        loaded_srcversion = loaded_path.read_text(encoding="utf-8").strip()
    module_path = (
        REPO_ROOT
        / "kernel-module"
        / "nvidia-module"
        / "kernel-open"
        / "nvidia-uvm.ko"
    )
    worktree_srcversion = (
        run_capture(["modinfo", "-F", "srcversion", str(module_path)])
        if module_path.is_file()
        else None
    )
    return {
        "active_path": command,
        "main_repo_commit": main_commit,
        "custom_relevant_diff_sha256": (
            hashlib.sha256(main_diff).hexdigest()
            if main_diff is not None
            else None
        ),
        "ktransformers_commit": kt_commit,
        "ktransformers_tracked_diff_sha256": (
            hashlib.sha256(kt_diff).hexdigest()
            if kt_diff is not None
            else None
        ),
        "ktransformers_status": kt_status,
        "sglang_commit": sglang_commit,
        "uvm_loaded_srcversion": loaded_srcversion,
        "uvm_worktree_srcversion": worktree_srcversion,
        "uvm_srcversion_match": bool(
            loaded_srcversion
            and worktree_srcversion
            and loaded_srcversion == worktree_srcversion
        ),
    }


def tensor_key(layer: int, expert: int, projection: str, suffix: str) -> str:
    return (
        f"model.layers.{layer}.mlp.experts.{expert}."
        f"{projection}.{suffix}"
    )


def required_tensor_keys(layer: int, num_experts: int) -> list[str]:
    return [
        tensor_key(layer, expert, projection, suffix)
        for expert in range(num_experts)
        for projection in PROJECTIONS
        for suffix in WEIGHT_SUFFIXES
    ]


def inspect_model(model: Path, layer: int) -> dict[str, Any]:
    model = model.resolve()
    config_path = model / "config.json"
    index_path = model / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError(
            f"model must contain config.json and model.safetensors.index.json: {model}"
        )
    config = load_json(config_path)
    for name, expected in EXPECTED_MODEL.items():
        actual = config.get(name)
        if actual != expected:
            raise ValueError(
                f"unsupported model config: {name}={actual!r}, expected {expected!r}"
            )
    if not 0 <= layer < EXPECTED_MODEL["num_hidden_layers"]:
        raise ValueError(f"layer must be in [0, 47], got {layer}")
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise ValueError("model lacks quantization_config")
    if quantization.get("quant_method") != "fp8":
        raise ValueError("model is not an FP8 checkpoint")
    if quantization.get("weight_block_size") != EXPECTED_BLOCK_SHAPE:
        raise ValueError(
            "model does not use the required 128x128 block-wise FP8 scales"
        )

    index = load_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("safetensors index lacks weight_map")
    keys = required_tensor_keys(layer, EXPECTED_MODEL["num_experts"])
    missing = [key for key in keys if key not in weight_map]
    if missing:
        raise ValueError(f"checkpoint is missing expert tensors: {missing[:8]}")

    shard_names = sorted({str(weight_map[key]) for key in keys})
    shard_records: list[dict[str, Any]] = []
    for shard_name in shard_names:
        shard_path = model / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"missing checkpoint shard: {shard_path}")
        resolved = shard_path.resolve()
        stat = resolved.stat()
        shard_records.append(
            {
                "file": shard_name,
                "resolved_blob": resolved.name,
                "size_bytes": int(stat.st_size),
            }
        )
    return {
        "model": model,
        "config": config,
        "config_path": config_path,
        "index_path": index_path,
        "weight_map": weight_map,
        "required_keys": keys,
        "shards": shard_records,
    }


def fp32_to_bf16_bits(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float32).copy()
    bits = array.view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding) >> np.uint32(16)).astype(np.uint16)


def prepare_fixture(args: argparse.Namespace) -> int:
    import numpy as np

    model_info = inspect_model(args.model, args.layer)
    if args.tokens <= 0:
        raise ValueError("--tokens must be positive")
    if args.tokens * EXPECTED_MODEL["num_experts_per_tok"] < EXPECTED_MODEL["num_experts"]:
        raise ValueError(
            "balanced trace must cover all 128 experts; use at least 16 tokens"
        )
    if not math.isfinite(args.hidden_scale) or args.hidden_scale <= 0:
        raise ValueError("--hidden-scale must be finite and positive")
    fixture = args.fixture.resolve()
    manifest = args.manifest.resolve()
    if fixture == manifest:
        raise ValueError("fixture and manifest paths must differ")
    for output in (fixture, manifest):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    hidden = (
        rng.standard_normal(
            (args.tokens, EXPECTED_MODEL["hidden_size"]), dtype=np.float32
        )
        * np.float32(args.hidden_scale)
    )
    hidden_bits = fp32_to_bf16_bits(hidden)
    token_offsets = (
        np.arange(args.tokens, dtype=np.int64)[:, None]
        * EXPECTED_MODEL["num_experts_per_tok"]
    )
    topk_offsets = np.arange(
        EXPECTED_MODEL["num_experts_per_tok"], dtype=np.int64
    )[None, :]
    topk_ids = (token_offsets + topk_offsets) % EXPECTED_MODEL["num_experts"]
    logits = rng.standard_normal(topk_ids.shape, dtype=np.float32)
    logits -= logits.max(axis=1, keepdims=True)
    topk_weights = np.exp(logits).astype(np.float32)
    topk_weights /= topk_weights.sum(axis=1, keepdims=True)

    temporary = fixture.with_name(f".{fixture.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            hidden_bf16_bits=hidden_bits,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(fixture)

    active, counts = np.unique(topk_ids, return_counts=True)
    model = model_info["model"]
    assert isinstance(model, Path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "qwen3_fp8_common_routing_trace",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fixture": {
            "path": str(fixture),
            "sha256": sha256_file(fixture),
            "hidden_storage": "raw_bf16_bits_uint16",
            "topk_ids_dtype": "int64",
            "topk_weights_dtype": "float32",
            "seed": args.seed,
            "hidden_scale": args.hidden_scale,
            "routing": "balanced_round_robin",
        },
        "shape": {
            **EXPECTED_MODEL,
            "tokens": args.tokens,
            "layer": args.layer,
            "active_experts": int(active.size),
            "expert_assignment_min": int(counts.min()),
            "expert_assignment_max": int(counts.max()),
        },
        "weights": {
            "model_path": str(model),
            "config_sha256": sha256_file(model_info["config_path"]),
            "index_sha256": sha256_file(model_info["index_path"]),
            "required_tensor_count": len(model_info["required_keys"]),
            "required_tensor_keys_sha256": hashlib.sha256(
                "\n".join(model_info["required_keys"]).encode("utf-8")
            ).hexdigest(),
            "shards": model_info["shards"],
            "expert_main_weight_bytes": 4_718_592,
            "expert_scale_bytes": 1_152,
            "fp8_block_shape": EXPECTED_BLOCK_SHAPE,
        },
    }
    write_json(manifest, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def load_fixture(
    fixture_path: Path, manifest_path: Path, model_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import numpy as np

    fixture_path = fixture_path.resolve()
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported fixture manifest schema")
    fixture_meta = manifest.get("fixture")
    shape = manifest.get("shape")
    weights_meta = manifest.get("weights")
    if not all(isinstance(value, dict) for value in (fixture_meta, shape, weights_meta)):
        raise ValueError("fixture manifest is incomplete")
    expected_fixture_sha = fixture_meta.get("sha256")
    actual_fixture_sha = sha256_file(fixture_path)
    if actual_fixture_sha != expected_fixture_sha:
        raise ValueError("fixture SHA-256 does not match its manifest")

    layer = shape.get("layer")
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise ValueError("manifest has invalid layer")
    model_info = inspect_model(model_path, layer)
    if sha256_file(model_info["config_path"]) != weights_meta.get("config_sha256"):
        raise ValueError("model config differs from fixture provenance")
    if sha256_file(model_info["index_path"]) != weights_meta.get("index_sha256"):
        raise ValueError("model safetensors index differs from fixture provenance")

    with np.load(fixture_path, allow_pickle=False) as archive:
        names = set(archive.files)
        required = {"hidden_bf16_bits", "topk_ids", "topk_weights"}
        if names != required:
            raise ValueError(f"fixture arrays mismatch: got {sorted(names)}")
        arrays = {name: archive[name].copy() for name in required}
    hidden_bits = arrays["hidden_bf16_bits"]
    topk_ids = arrays["topk_ids"]
    topk_weights = arrays["topk_weights"]
    tokens = shape.get("tokens")
    expected_hidden = (tokens, EXPECTED_MODEL["hidden_size"])
    expected_topk = (tokens, EXPECTED_MODEL["num_experts_per_tok"])
    if hidden_bits.dtype != np.uint16 or hidden_bits.shape != expected_hidden:
        raise ValueError("hidden_bf16_bits has the wrong dtype or shape")
    if topk_ids.dtype != np.int64 or topk_ids.shape != expected_topk:
        raise ValueError("topk_ids has the wrong dtype or shape")
    if topk_weights.dtype != np.float32 or topk_weights.shape != expected_topk:
        raise ValueError("topk_weights has the wrong dtype or shape")
    if np.any(topk_ids < 0) or np.any(topk_ids >= EXPECTED_MODEL["num_experts"]):
        raise ValueError("topk_ids is outside the expert topology")
    if any(np.unique(row).size != row.size for row in topk_ids):
        raise ValueError("a token routes to a duplicate expert")
    if not np.isfinite(topk_weights).all() or np.any(topk_weights < 0):
        raise ValueError("topk_weights is not finite and nonnegative")
    if not np.allclose(topk_weights.sum(axis=1), 1.0, rtol=0, atol=2e-6):
        raise ValueError("topk_weights rows are not normalized")
    return arrays, manifest, model_info


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, Any]:
    if not values or not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValueError("timing samples must be finite and nonnegative")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "samples": values,
    }


def validate_run_args(args: argparse.Namespace) -> None:
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("--warmup must be nonnegative and --iterations positive")
    if args.device < 0:
        raise ValueError("--device must be nonnegative")
    if args.output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse an existing run directory: {args.output_dir}"
        )
    if args.reference_output is not None:
        tolerances = (args.atol, args.rtol, args.max_relative_l1)
        if any(value is None for value in tolerances):
            raise ValueError(
                "--reference-output requires explicit --atol, --rtol, and "
                "--max-relative-l1"
            )
        if not all(math.isfinite(value) and value >= 0 for value in tolerances):
            raise ValueError("all numerical tolerances must be finite and nonnegative")
    elif any(
        value is not None for value in (args.atol, args.rtol, args.max_relative_l1)
    ):
        raise ValueError("tolerances are only valid together with --reference-output")


def require_gpu_authorization(args: argparse.Namespace) -> None:
    if not args.allow_gpu_run:
        raise PermissionError(
            "GPU execution is fail-closed; repeat with --allow-gpu-run after "
            "checking that the GPU is idle"
        )


def torch_fixture(arrays: dict[str, Any], torch: Any) -> tuple[Any, Any, Any]:
    hidden_bits = torch.from_numpy(arrays["hidden_bf16_bits"].copy())
    hidden = hidden_bits.view(torch.bfloat16).contiguous()
    topk_ids = torch.from_numpy(arrays["topk_ids"].copy()).contiguous()
    topk_weights = torch.from_numpy(arrays["topk_weights"].copy()).contiguous()
    return hidden, topk_ids, topk_weights


def pinned_copy(tensor: Any, torch: Any) -> Any:
    result = torch.empty_like(tensor, device="cpu", pin_memory=True)
    result.copy_(tensor)
    return result


def gpu_metadata(torch: Any, device: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": device,
        "name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "capability": [int(properties.major), int(properties.minor)],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def common_metadata(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    model_info: dict[str, Any],
    torch: Any,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": args.command,
        "argv": sys.argv,
        "fixture_sha256": manifest["fixture"]["sha256"],
        "fixture_manifest_sha256": sha256_file(args.manifest.resolve()),
        "model_config_sha256": sha256_file(model_info["config_path"]),
        "model_index_sha256": sha256_file(model_info["index_path"]),
        "layer": manifest["shape"]["layer"],
        "shape": manifest["shape"],
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "implementation": implementation_provenance(args.command),
        "host": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        "gpu": gpu_metadata(torch, args.device),
    }


def save_output(
    output_path: Path,
    output_cpu: Any,
    metadata: dict[str, Any],
    torch: Any,
) -> None:
    import numpy as np

    output = output_cpu.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    bits = output.view(torch.uint16).numpy().copy()
    float32 = output.float().numpy().copy()
    with output_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            output_bf16_bits=bits,
            output_float32=float32,
            fixture_sha256=np.asarray(metadata["fixture_sha256"]),
            model_index_sha256=np.asarray(metadata["model_index_sha256"]),
            layer=np.asarray(metadata["layer"], dtype=np.int64),
        )


def numerical_validation(
    current_output: Path,
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    if args.establish_reference:
        return {
            "status": "candidate_reference_not_validated",
            "passed": None,
            "note": "A second implementation must compare against this output.",
        }
    assert args.reference_output is not None
    reference_path = args.reference_output.resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f"reference output does not exist: {reference_path}")
    with np.load(current_output, allow_pickle=False) as current_archive:
        current = current_archive["output_float32"].copy()
    with np.load(reference_path, allow_pickle=False) as reference_archive:
        required = {
            "output_float32",
            "fixture_sha256",
            "model_index_sha256",
            "layer",
        }
        if not required.issubset(reference_archive.files):
            raise ValueError("reference output lacks provenance fields")
        reference = reference_archive["output_float32"].copy()
        reference_fixture = str(reference_archive["fixture_sha256"].item())
        reference_index = str(reference_archive["model_index_sha256"].item())
        reference_layer = int(reference_archive["layer"].item())
    if reference_fixture != metadata["fixture_sha256"]:
        raise ValueError("reference uses a different routing fixture")
    if reference_index != metadata["model_index_sha256"]:
        raise ValueError("reference uses a different safetensors index")
    if reference_layer != metadata["layer"]:
        raise ValueError("reference uses a different MoE layer")
    if current.shape != reference.shape:
        raise ValueError("reference output shape differs")
    finite = bool(np.isfinite(current).all() and np.isfinite(reference).all())
    difference = np.abs(current - reference)
    limit = float(args.atol) + float(args.rtol) * np.abs(reference)
    violations = difference > limit
    denominator = max(float(np.abs(reference).sum(dtype=np.float64)), 1e-30)
    relative_l1 = float(difference.sum(dtype=np.float64) / denominator)
    passed = bool(
        finite
        and not bool(violations.any())
        and relative_l1 <= float(args.max_relative_l1)
    )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "reference_output": str(reference_path),
        "atol": float(args.atol),
        "rtol": float(args.rtol),
        "max_relative_l1_allowed": float(args.max_relative_l1),
        "relative_l1": relative_l1,
        "max_abs_error": float(difference.max(initial=0.0)),
        "violating_elements": int(violations.sum()),
        "elements": int(current.size),
        "finite": finite,
    }


def finalize_run(
    args: argparse.Namespace,
    result: dict[str, Any],
    output_cpu: Any,
    torch: Any,
) -> int:
    output_path = args.output_dir / "output.npz"
    save_output(output_path, output_cpu, result["metadata"], torch)
    validation = numerical_validation(output_path, args, result["metadata"])
    result["output"] = {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
    }
    result["numerical_validation"] = validation
    result["status"] = "ok" if validation.get("passed") is not False else "failed"
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


def run_kt(args: argparse.Namespace) -> int:
    validate_run_args(args)
    if args.threads <= 0 or args.numa_node < 0:
        raise ValueError("--threads must be positive and --numa-node nonnegative")
    arrays, manifest, model_info = load_fixture(
        args.fixture, args.manifest, args.model
    )
    require_gpu_authorization(args)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(args.device)
    from kt_kernel import KTMoEWrapper, __cpu_variant__

    args.output_dir.mkdir(parents=True, exist_ok=False)
    hidden, topk_ids, topk_weights = torch_fixture(arrays, torch)
    tokens = int(hidden.shape[0])
    layer = int(manifest["shape"]["layer"])
    gpu_mask = torch.zeros(EXPECTED_MODEL["num_experts"], dtype=torch.bool)

    load_start = time.perf_counter()
    wrapper = KTMoEWrapper(
        layer_idx=layer,
        num_experts=EXPECTED_MODEL["num_experts"],
        num_experts_per_tok=EXPECTED_MODEL["num_experts_per_tok"],
        hidden_size=EXPECTED_MODEL["hidden_size"],
        moe_intermediate_size=EXPECTED_MODEL["moe_intermediate_size"],
        gpu_experts_mask=gpu_mask,
        cpuinfer_threads=args.threads,
        threadpool_count=1,
        weight_path=str(args.model.resolve()),
        chunked_prefill_size=max(tokens, 1),
        method="FP8",
        numa_nodes=[args.numa_node],
        max_deferred_experts_per_token=0,
    )
    physical_to_logical = torch.arange(
        EXPECTED_MODEL["num_experts"], dtype=torch.int64
    ).contiguous()
    wrapper.load_weights(physical_to_logical)
    weight_load_s = time.perf_counter() - load_start
    if wrapper.moe is None:
        raise RuntimeError("KT FP8 MoE failed to initialize")

    cpu_hidden = pinned_copy(hidden, torch)
    cpu_ids = pinned_copy(topk_ids, torch)
    cpu_weights = pinned_copy(topk_weights, torch)
    cpu_output = torch.empty_like(cpu_hidden, pin_memory=True)
    qlen = torch.tensor([tokens], dtype=torch.int32, pin_memory=True)

    def cpu_compute() -> None:
        wrapper.cpu_infer.submit(
            wrapper.moe.forward_task(
                qlen.data_ptr(),
                EXPECTED_MODEL["num_experts_per_tok"],
                cpu_ids.data_ptr(),
                cpu_weights.data_ptr(),
                cpu_hidden.data_ptr(),
                cpu_output.data_ptr(),
                False,
            )
        )
        wrapper.cpu_infer.sync()

    for _ in range(args.warmup):
        cpu_compute()
    compute_ms: list[float] = []
    for _ in range(args.iterations):
        start = time.perf_counter_ns()
        cpu_compute()
        compute_ms.append((time.perf_counter_ns() - start) / 1e6)
    pure_output = cpu_output.clone()

    # Common standalone boundary: CPU fixture -> GPU caller state -> KT CPU
    # expert -> GPU result -> CPU captured result.
    source_hidden = pinned_copy(hidden, torch)
    source_ids = pinned_copy(topk_ids, torch)
    source_weights = pinned_copy(topk_weights, torch)
    gpu_hidden = torch.empty_like(hidden, device=f"cuda:{args.device}")
    gpu_ids = torch.empty_like(topk_ids, device=f"cuda:{args.device}")
    gpu_weights = torch.empty_like(topk_weights, device=f"cuda:{args.device}")
    gpu_output = torch.empty_like(hidden, device=f"cuda:{args.device}")
    captured_output = torch.empty_like(hidden, device="cpu", pin_memory=True)
    stream = torch.cuda.current_stream(args.device)

    def inclusive_once() -> None:
        gpu_hidden.copy_(source_hidden, non_blocking=True)
        gpu_ids.copy_(source_ids, non_blocking=True)
        gpu_weights.copy_(source_weights, non_blocking=True)
        stream.synchronize()
        cpu_hidden.copy_(gpu_hidden, non_blocking=True)
        cpu_ids.copy_(gpu_ids, non_blocking=True)
        cpu_weights.copy_(gpu_weights, non_blocking=True)
        stream.synchronize()
        cpu_compute()
        gpu_output.copy_(cpu_output, non_blocking=True)
        captured_output.copy_(gpu_output, non_blocking=True)
        stream.synchronize()

    for _ in range(args.warmup):
        inclusive_once()
    inclusive_ms: list[float] = []
    for _ in range(args.iterations):
        start = time.perf_counter_ns()
        inclusive_once()
        inclusive_ms.append((time.perf_counter_ns() - start) / 1e6)
    if not torch.equal(pure_output, captured_output):
        raise RuntimeError("KT pure and boundary-inclusive outputs differ")

    input_bytes = (
        hidden.numel() * hidden.element_size()
        + topk_ids.numel() * topk_ids.element_size()
        + topk_weights.numel() * topk_weights.element_size()
    )
    output_bytes = hidden.numel() * hidden.element_size()
    metadata = common_metadata(args, manifest, model_info, torch)
    result = {
        "metadata": metadata,
        "mechanism": {
            "name": "kt_kernel_fp8_cpu_expert",
            "backend": "KTMoEWrapper(method=FP8)",
            "cpu_variant": __cpu_variant__,
            "threads": args.threads,
            "threadpool_count": 1,
            "numa_node": args.numa_node,
            "all_experts_on_cpu": True,
            "deferred_experts": 0,
            "weight_load_seconds": weight_load_s,
        },
        "timing": {
            "compute_only_ms": summarize(compute_ms),
            "inclusive_boundary_ms": summarize(inclusive_ms),
            "compute_clock": "time.perf_counter_ns around CPUInfer submit+sync",
            "inclusive_boundary": (
                "CPU fixture H2D; GPU-to-pinned-CPU hidden/topk; KT CPU compute; "
                "pinned-CPU output H2D; result D2H"
            ),
            "inclusive_transfer_bytes": int(2 * (input_bytes + output_bytes)),
        },
    }
    return finalize_run(args, result, pure_output, torch)


def checkpoint_records(model_info: dict[str, Any], layer: int) -> dict[str, list[Any]]:
    records: dict[str, list[Any]] = defaultdict(list)
    weight_map = model_info["weight_map"]
    for expert in range(EXPECTED_MODEL["num_experts"]):
        for projection in PROJECTIONS:
            for suffix in WEIGHT_SUFFIXES:
                key = tensor_key(layer, expert, projection, suffix)
                records[str(weight_map[key])].append(
                    (expert, projection, suffix, key)
                )
    return records


def load_custom_weights(
    model_info: dict[str, Any],
    layer: int,
    device: int,
    torch: Any,
) -> tuple[Any, Any, Any, Any, float]:
    from safetensors import safe_open
    from vllm.device_allocator.uvm import uvm_local_memory_pool

    experts = EXPECTED_MODEL["num_experts"]
    hidden = EXPECTED_MODEL["hidden_size"]
    intermediate = EXPECTED_MODEL["moe_intermediate_size"]
    cuda_device = f"cuda:{device}"
    with uvm_local_memory_pool(enable_prefetch=False):
        w13 = torch.empty(
            (experts, 2 * intermediate, hidden),
            dtype=torch.float8_e4m3fn,
            device=cuda_device,
        )
        w2 = torch.empty(
            (experts, hidden, intermediate),
            dtype=torch.float8_e4m3fn,
            device=cuda_device,
        )
    w13_scale = torch.empty(
        (experts, 2 * (intermediate // 128), hidden // 128),
        dtype=torch.float32,
        device=cuda_device,
    )
    w2_scale = torch.empty(
        (experts, hidden // 128, intermediate // 128),
        dtype=torch.float32,
        device=cuda_device,
    )

    records = checkpoint_records(model_info, layer)
    model = model_info["model"]
    start = time.perf_counter()
    with torch.no_grad():
        for shard_name, shard_records in sorted(records.items()):
            with safe_open(model / shard_name, framework="pt", device="cpu") as shard:
                for expert, projection, suffix, key in shard_records:
                    value = shard.get_tensor(key).contiguous()
                    if suffix == "weight":
                        if value.dtype != torch.float8_e4m3fn:
                            raise TypeError(f"{key}: expected float8_e4m3fn")
                        if projection == "gate_proj":
                            w13[expert, :intermediate].copy_(value)
                        elif projection == "up_proj":
                            w13[expert, intermediate:].copy_(value)
                        else:
                            w2[expert].copy_(value)
                    else:
                        if value.dtype != torch.float32:
                            raise TypeError(f"{key}: expected float32 scale")
                        if projection == "gate_proj":
                            w13_scale[expert, : intermediate // 128].copy_(value)
                        elif projection == "up_proj":
                            w13_scale[expert, intermediate // 128 :].copy_(value)
                        else:
                            w2_scale[expert].copy_(value)
    torch.cuda.synchronize(device)
    return w13, w2, w13_scale, w2_scale, time.perf_counter() - start


def release_custom_weights(
    w13: Any,
    w2: Any,
    layer: int,
    device: int,
) -> tuple[dict[str, Any], list[str]]:
    from vllm.device_allocator.uvm_pool_residency import (
        query_uvm_regions_physical_residency,
        register_uvm_region,
        release_uvm_regions,
    )

    page = 64 * 1024
    region_ids: list[str] = []
    region_records: list[dict[str, Any]] = []
    for name, weight in (("w13", w13), ("w2", w2)):
        for expert in range(EXPECTED_MODEL["num_experts"]):
            view = weight[expert]
            base = int(view.data_ptr())
            length = int(view.numel() * view.element_size())
            if base % page or length % page:
                raise RuntimeError(
                    f"{name} expert {expert} is not 64-KiB aligned: "
                    f"base=0x{base:x} length={length}"
                )
            region_id = f"common-trace:l{layer}:{name}:e{expert}"
            register_uvm_region(region_id, base, length, state="unknown")
            region_ids.append(region_id)
            region_records.append(
                {
                    "region_id": region_id,
                    "tensor": name,
                    "expert": expert,
                    "base": base,
                    "length": length,
                }
            )
    release = release_uvm_regions(
        region_ids,
        device=device,
        remote_read=True,
        include_audit=True,
    )
    if release.get("ok") is not True or release.get("failed_region_ids"):
        raise RuntimeError(
            f"remote-read release failed: {release.get('failed_region_ids')}"
        )
    audits = release.get("audits")
    if not isinstance(audits, dict) or len(audits) != len(region_ids):
        raise RuntimeError("remote-read release returned incomplete audits")
    for region in region_records:
        audit = audits.get(region["region_id"])
        if not isinstance(audit, dict):
            raise RuntimeError(f"missing audit for {region['region_id']}")
        if (
            audit.get("ok") is not True
            or audit.get("remote_read") is not True
            or audit.get("app_state_after") != "cpu"
            or int(audit.get("registered_length", -1)) != region["length"]
            or int(audit.get("failure_blocks", -1)) != 0
            or int(audit.get("failed_bytes", -1)) != 0
            or int(audit.get("remote_map_failure_blocks", -1)) != 0
            or int(audit.get("cpu_resident_bytes", -1)) != region["length"]
            or int(audit.get("released_bytes", -1)) != region["length"]
            or int(audit.get("remote_mapped_bytes", -1)) != region["length"]
        ):
            raise RuntimeError(
                f"remote-read audit gate failed for {region['region_id']}: {audit}"
            )
    residency = query_uvm_regions_physical_residency(
        region_ids,
        device=device,
        wait_for_trackers=True,
    )
    expected_bytes = sum(int(record["length"]) for record in region_records)
    topology_bytes = EXPECTED_MODEL["num_experts"] * 4_718_592
    if expected_bytes != topology_bytes:
        raise RuntimeError(
            "managed main-weight byte count differs from the frozen topology: "
            f"got={expected_bytes} expected={topology_bytes}"
        )
    if (
        residency.get("ok") is not True
        or int(residency.get("region_count", -1)) != len(region_ids)
        or int(residency.get("expected_bytes", -1)) != expected_bytes
        or int(residency.get("queried_bytes", -1)) != expected_bytes
        or int(residency.get("cpu_bytes", -1)) != expected_bytes
        or int(residency.get("gpu_bytes", -1)) != 0
        or int(residency.get("unknown_bytes", -1)) != 0
    ):
        raise RuntimeError(
            f"post-release physical-residency gate failed: {residency}"
        )
    placement = {
        "schema_version": SCHEMA_VERSION,
        "access_mode": "remote_read",
        "region_count": len(region_ids),
        "expected_main_weight_bytes": expected_bytes,
        "regions": region_records,
        "release": release,
        "post_release_residency": residency,
    }
    return placement, region_ids


def run_custom(args: argparse.Namespace) -> int:
    validate_run_args(args)
    arrays, manifest, model_info = load_fixture(
        args.fixture, args.manifest, args.model
    )
    require_gpu_authorization(args)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.set_device(args.device)
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.device_allocator.uvm import release_uvm_local_memory_pool
    from vllm.device_allocator.uvm_pool_residency import (
        query_uvm_regions_physical_residency,
        unregister_uvm_regions,
    )
    from vllm.model_executor.layers.fused_moe.config import (
        fp8_w8a8_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    args.output_dir.mkdir(parents=True, exist_ok=False)
    hidden, topk_ids, topk_weights = torch_fixture(arrays, torch)
    layer = int(manifest["shape"]["layer"])
    w13, w2, w13_scale, w2_scale, weight_load_s = load_custom_weights(
        model_info, layer, args.device, torch
    )
    placement, region_ids = release_custom_weights(w13, w2, layer, args.device)
    write_json(args.output_dir / "placement_manifest.json", placement)

    cuda_device = f"cuda:{args.device}"
    gpu_hidden = hidden.to(cuda_device)
    gpu_ids = topk_ids.to(cuda_device)
    gpu_weights = topk_weights.to(cuda_device)
    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=EXPECTED_BLOCK_SHAPE,
    )
    vllm_config = VllmConfig(parallel_config=ParallelConfig())

    def gpu_compute(x: Any, ids: Any, weights: Any) -> Any:
        return fused_experts(
            x,
            w13,
            w2,
            weights,
            ids,
            inplace=False,
            activation="silu",
            quant_config=quant_config,
            allow_deep_gemm=False,
            allow_cutlass_block_scaled_grouped_gemm=False,
        )

    with set_current_vllm_config(vllm_config):
        for _ in range(args.warmup):
            _ = gpu_compute(gpu_hidden, gpu_ids, gpu_weights)
        torch.cuda.synchronize(args.device)
        compute_device_ms: list[float] = []
        compute_wall_ms: list[float] = []
        pure_output = None
        for _ in range(args.iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter_ns()
            start_event.record()
            pure_output = gpu_compute(gpu_hidden, gpu_ids, gpu_weights)
            end_event.record()
            end_event.synchronize()
            compute_device_ms.append(float(start_event.elapsed_time(end_event)))
            compute_wall_ms.append((time.perf_counter_ns() - wall_start) / 1e6)
        assert pure_output is not None
        pure_output_cpu = pure_output.to("cpu")

        # Common standalone boundary: CPU fixture H2D, remote-read GPU MoE,
        # and result D2H.
        source_hidden = pinned_copy(hidden, torch)
        source_ids = pinned_copy(topk_ids, torch)
        source_weights = pinned_copy(topk_weights, torch)
        boundary_hidden = torch.empty_like(hidden, device=cuda_device)
        boundary_ids = torch.empty_like(topk_ids, device=cuda_device)
        boundary_weights = torch.empty_like(topk_weights, device=cuda_device)
        captured_output = torch.empty_like(hidden, device="cpu", pin_memory=True)

        def inclusive_once() -> None:
            boundary_hidden.copy_(source_hidden, non_blocking=True)
            boundary_ids.copy_(source_ids, non_blocking=True)
            boundary_weights.copy_(source_weights, non_blocking=True)
            output = gpu_compute(boundary_hidden, boundary_ids, boundary_weights)
            captured_output.copy_(output, non_blocking=True)
            torch.cuda.current_stream(args.device).synchronize()

        for _ in range(args.warmup):
            inclusive_once()
        inclusive_ms: list[float] = []
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            inclusive_once()
            inclusive_ms.append((time.perf_counter_ns() - start) / 1e6)
    if not torch.equal(pure_output_cpu, captured_output):
        raise RuntimeError("custom pure and boundary-inclusive outputs differ")

    # This snapshot is intentionally outside the measured window and before
    # unregistering the regions.  The release-time snapshot alone cannot show
    # that the main weights remained CPU resident throughout the iterations.
    expected_remote_bytes = int(placement["expected_main_weight_bytes"])
    post_measurement_residency = query_uvm_regions_physical_residency(
        region_ids,
        device=args.device,
        wait_for_trackers=True,
    )
    if (
        post_measurement_residency.get("ok") is not True
        or int(post_measurement_residency.get("region_count", -1))
        != len(region_ids)
        or int(post_measurement_residency.get("expected_bytes", -1))
        != expected_remote_bytes
        or int(post_measurement_residency.get("queried_bytes", -1))
        != expected_remote_bytes
        or int(post_measurement_residency.get("cpu_bytes", -1))
        != expected_remote_bytes
        or int(post_measurement_residency.get("gpu_bytes", -1)) != 0
        or int(post_measurement_residency.get("unknown_bytes", -1)) != 0
    ):
        raise RuntimeError(
            "post-measurement physical-residency gate failed: "
            f"{post_measurement_residency}"
        )
    placement["post_measurement_residency"] = post_measurement_residency
    placement["measurement_window_residency_gate"] = {
        "status": "pass",
        "note": (
            "release-time and post-measurement snapshots both show every "
            "managed main-weight byte CPU resident; the post snapshot is not "
            "included in timing"
        ),
    }
    write_json(args.output_dir / "placement_manifest.json", placement)

    input_bytes = (
        hidden.numel() * hidden.element_size()
        + topk_ids.numel() * topk_ids.element_size()
        + topk_weights.numel() * topk_weights.element_size()
    )
    output_bytes = hidden.numel() * hidden.element_size()
    metadata = common_metadata(args, manifest, model_info, torch)
    result = {
        "metadata": metadata,
        "mechanism": {
            "name": "custom_uvm_gpu_remote_read",
            "backend": "vllm.fused_experts eager Triton path",
            "weights": "cudaMallocManaged main weights; CUDA-resident scales",
            "access_mode": "remote_read",
            "all_experts_remote": True,
            "weight_load_seconds": weight_load_s,
            "placement_manifest": str(
                args.output_dir / "placement_manifest.json"
            ),
            "placement_manifest_sha256": sha256_file(
                args.output_dir / "placement_manifest.json"
            ),
        },
        "timing": {
            "compute_only_ms": summarize(compute_device_ms),
            "compute_only_synchronized_wall_ms": summarize(compute_wall_ms),
            "inclusive_boundary_ms": summarize(inclusive_ms),
            "compute_clock": "CUDA events around fused_experts",
            "inclusive_boundary": (
                "CPU fixture H2D; remote-read GPU fused_experts; result D2H"
            ),
            "inclusive_transfer_bytes": int(input_bytes + output_bytes),
        },
    }
    unregister = unregister_uvm_regions(region_ids)
    if unregister.get("ok") is not True or unregister.get("failed_region_ids"):
        raise RuntimeError(
            "failed to unregister standalone microbenchmark regions: "
            f"{unregister.get('failed_region_ids')}"
        )
    torch.cuda.synchronize(args.device)
    del gpu_compute, inclusive_once, quant_config
    del pure_output, gpu_hidden, gpu_ids, gpu_weights
    del boundary_hidden, boundary_ids, boundary_weights
    del w13, w2, w13_scale, w2_scale
    del start_event, end_event
    gc.collect()
    torch.cuda.empty_cache()
    pool_released = release_uvm_local_memory_pool()
    result["mechanism"]["standalone_cleanup"] = {
        "regions_unregistered": len(region_ids),
        "local_managed_pool_released": pool_released,
    }
    return finalize_run(args, result, pure_output_cpu, torch)


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--allow-gpu-run",
        action="store_true",
        help="mandatory acknowledgement before importing torch/runtime code",
    )
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument(
        "--establish-reference",
        action="store_true",
        help="write an explicitly unvalidated candidate reference",
    )
    reference.add_argument("--reference-output", type=Path)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--max-relative-l1", type=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-layer common-trace comparison of KT FP8 CPU experts and "
            "custom UVM GPU remote-read"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="CPU-only: freeze hidden/top-k fixture and weight provenance"
    )
    prepare.add_argument("--model", required=True, type=Path)
    prepare.add_argument("--fixture", required=True, type=Path)
    prepare.add_argument("--manifest", required=True, type=Path)
    prepare.add_argument("--layer", type=int, default=0)
    prepare.add_argument("--tokens", type=int, default=128)
    prepare.add_argument("--seed", type=int, default=20260813)
    prepare.add_argument("--hidden-scale", type=float, default=0.5)
    prepare.set_defaults(func=prepare_fixture)

    kt = subparsers.add_parser(
        "run-kt", help="run KT-Kernel FP8 CPU experts on the frozen trace"
    )
    add_common_run_args(kt)
    kt.add_argument("--threads", type=int, default=12)
    kt.add_argument("--numa-node", type=int, default=0)
    kt.set_defaults(func=run_kt)

    custom = subparsers.add_parser(
        "run-custom", help="run custom UVM GPU remote-read on the frozen trace"
    )
    add_common_run_args(custom)
    custom.set_defaults(func=run_custom)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, FileExistsError, PermissionError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
