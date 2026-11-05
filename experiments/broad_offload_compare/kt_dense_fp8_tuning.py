#!/usr/bin/env python3
"""Prepare and audit workload-specific SGLang dense W8A8 FP8 tuning.

All commands except ``tune-one`` are CPU-only.  The tuning command deliberately
wraps SGLang's pinned tuner instead of changing the pinned checkout.  In
particular, it uses the runtime BF16 output dtype and excludes BLOCK_SIZE_K=64:
the pinned runtime clamps such entries to 128, so timing them as 64 would not
measure the configuration that the server eventually executes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "kt-dense-fp8-tuning-v1"
EXPECTED_KT_COMMIT = "eb9b70c4115cff151ace2cae5b0fc9db3690e31e"
EXPECTED_SGLANG_COMMIT = "bc7f0058fafe6c05dc6ad52d6b1c854af2a673b2"
EXPECTED_DEVICE = "NVIDIA GeForce RTX 5090"
EXPECTED_SHAPES = ((5120, 2048), (2048, 4096))
BLOCK_N = 128
BLOCK_K = 128
CONFIG_KEYS = {
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "num_warps",
    "num_stages",
}

DENSE_FALLBACK_RE = re.compile(
    r"Using default W8A8 Block FP8 kernel config\. Performance might be "
    r"sub-optimal! Config file not found at (?P<path>[^\r\n]+?\.json)"
)
DENSE_TUNED_RE = re.compile(
    r"Using configuration from (?P<path>[^\r\n]+?\.json) for W8A8 Block FP8 kernel\."
)
SHAPE_RE = re.compile(r"N=(?P<n>\d+),K=(?P<k>\d+),")
PREFILL_RE = re.compile(
    r"Prefill batch(?: \[\d+\])?, #new-seq: (?P<new_seq>\d+), "
    r"#new-token: (?P<new_token>\d+), #cached-token: (?P<cached>\d+), "
    r".*?#running-req: (?P<running>\d+),"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def sha256_json(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def parse_positive_csv(raw: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must be unique positive integers")
    return values


def shape_from_path(raw: str) -> tuple[int, int] | None:
    match = SHAPE_RE.search(raw)
    return (int(match.group("n")), int(match.group("k"))) if match else None


def parse_server_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    dense_fallbacks = [
        {"path": match.group("path"), "shape": shape_from_path(match.group("path"))}
        for match in DENSE_FALLBACK_RE.finditer(text)
    ]
    dense_tuned = [
        {"path": match.group("path"), "shape": shape_from_path(match.group("path"))}
        for match in DENSE_TUNED_RE.finditer(text)
    ]
    prefills = []
    for match in PREFILL_RE.finditer(text):
        prefills.append(
            {
                "new_sequences": int(match.group("new_seq")),
                "new_tokens": int(match.group("new_token")),
                "cached_tokens": int(match.group("cached")),
                "running_requests": int(match.group("running")),
            }
        )

    def last_bool(name: str) -> bool | None:
        matches = re.findall(rf"\b{re.escape(name)}=(True|False)", text)
        return matches[-1] == "True" if matches else None

    backend_matches = re.findall(r"\bfp8_gemm_runner_backend='([^']+)'", text)
    graph_bs_matches = re.findall(r"\bcuda_graph_bs=\[([^]]*)\]", text)
    graph_bs = None
    if graph_bs_matches:
        graph_bs = [int(value) for value in re.findall(r"\d+", graph_bs_matches[-1])]
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "dense_fallbacks": dense_fallbacks,
        "dense_tuned_loads": dense_tuned,
        "faq_ignored_fused_moe_warning_count": text.count(
            "Using default MoE kernel config. Performance might be sub-optimal!"
        )
        + text.count(
            "Using MoE kernel config with down_moe=False. Performance might be sub-optimal!"
        ),
        "dense_config_clamp_warning_count": text.count("Clamped BLOCK_SIZE_K up to"),
        "fp8_gemm_runner_backend": backend_matches[-1] if backend_matches else None,
        "enable_mixed_chunk": last_bool("enable_mixed_chunk"),
        "disable_cuda_graph": last_bool("disable_cuda_graph"),
        "disable_cuda_graph_padding": last_bool("disable_cuda_graph_padding"),
        "cuda_graph_bs": graph_bs,
        "prefill_events": prefills,
    }


def audit_log(log: Path, expect: str) -> dict[str, Any]:
    parsed = parse_server_log(log)
    fallback_shapes = {
        tuple(item["shape"]) for item in parsed["dense_fallbacks"] if item["shape"]
    }
    tuned_shapes = {
        tuple(item["shape"])
        for item in parsed["dense_tuned_loads"]
        if item["shape"]
    }
    failures: list[str] = []
    if parsed["fp8_gemm_runner_backend"] != "triton":
        failures.append("server did not normalize fp8_gemm_runner_backend='triton'")
    if expect == "default":
        missing = set(EXPECTED_SHAPES) - fallback_shapes
        if missing:
            failures.append(f"default arm lacks fallback evidence for {sorted(missing)}")
        if set(EXPECTED_SHAPES) & tuned_shapes:
            failures.append("default arm loaded a tuned config")
    elif expect == "tuned":
        if fallback_shapes:
            failures.append(f"dense fallback remained for {sorted(fallback_shapes)}")
        missing = set(EXPECTED_SHAPES) - tuned_shapes
        if missing:
            failures.append(f"tuned arm lacks config-load evidence for {sorted(missing)}")
        if parsed["dense_config_clamp_warning_count"]:
            failures.append("runtime clamped BLOCK_SIZE_K; tuned timing is not faithful")
    result = {
        "schema": SCHEMA,
        "kind": "dense-warning-audit",
        "created_at": utc_now(),
        "expect": expect,
        "required_shapes": [list(shape) for shape in EXPECTED_SHAPES],
        "parsed": parsed,
        "faq_scope": {
            "ignored": "SGLang fused-MoE config warnings only",
            "not_ignored": "W8A8 Block FP8 dense GEMM warnings",
            "source": "KTransformers/doc/en/FAQ.md section 1",
        },
        "failures": failures,
        "status": "ok" if not failures else "failed",
    }
    return result


def fixture_info(path: Path) -> tuple[dict[str, Any], list[list[int]]]:
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("requests"), list):
        raise ValueError("fixture must be an object with a requests list")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("prompt_token_ids_frozen") is not True:
        raise ValueError("fixture is not marked prompt_token_ids_frozen=true")
    all_ids: list[list[int]] = []
    for index, request in enumerate(payload["requests"]):
        ids = request.get("prompt_token_ids") if isinstance(request, dict) else None
        if not isinstance(ids, list) or not ids or any(
            isinstance(value, bool) or not isinstance(value, int) for value in ids
        ):
            raise ValueError(f"fixture request {index} has invalid token IDs")
        if request.get("prompt_tokens") != len(ids):
            raise ValueError(f"fixture request {index} token count differs from token IDs")
        all_ids.append(ids)
    declared = metadata.get("prompt_token_ids_list_sha256")
    actual = sha256_json(all_ids)
    if declared != actual:
        raise ValueError("fixture prompt-token-ID list SHA-256 is inconsistent")
    return (
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "prompt_token_ids_list_sha256": actual,
            "request_count": len(all_ids),
            "prompt_lengths": [len(ids) for ids in all_ids],
        },
        all_ids,
    )


def validate_benchmark(
    path: Path, all_ids: list[list[int]], requested: set[int]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = load_json(path)
    if payload.get("status") != "ok":
        raise ValueError(f"{path}: benchmark status is not ok")
    config = payload.get("config", {})
    if config.get("require_prompt_token_ids") is not True:
        raise ValueError(f"{path}: benchmark did not require frozen prompt token IDs")
    phases = payload.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError(f"{path}: benchmark has no phases")
    validated = []
    for phase in phases:
        concurrency = phase.get("concurrency")
        if concurrency not in requested:
            continue
        expected_ids = all_ids[:concurrency]
        if phase.get("prompt_token_ids_list_sha256") != sha256_json(expected_ids):
            raise ValueError(f"{path}: c{concurrency} token-ID prefix SHA differs")
        records = phase.get("requests")
        if not isinstance(records, list) or len(records) != concurrency:
            raise ValueError(f"{path}: c{concurrency} request records are incomplete")
        for index, record in enumerate(records):
            if record.get("success") is not True:
                raise ValueError(f"{path}: c{concurrency} has a failed request")
            if record.get("submitted_input_kind") != "input_ids":
                raise ValueError(f"{path}: c{concurrency} did not submit input_ids")
            if record.get("server_prompt_token_count_matches_submission") is not True:
                raise ValueError(f"{path}: c{concurrency} server token count differs")
            if record.get("submitted_prompt_token_ids_sha256") != sha256_json(
                expected_ids[index]
            ):
                raise ValueError(f"{path}: c{concurrency} request {index} ID SHA differs")
        validated.append(
            {
                "concurrency": concurrency,
                "prompt_tokens": sum(map(len, expected_ids)),
                "output_tokens_per_request": config.get("output_tokens_per_request"),
            }
        )
    return (
        {"path": str(path.resolve()), "sha256": sha256_file(path)},
        validated,
    )


def match_prefill_segments(
    events: list[dict[str, Any]], phases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    cursor = 0
    matches: list[dict[str, Any]] = []
    for phase in phases:
        target = phase["prompt_tokens"]
        found: tuple[int, int] | None = None
        for start in range(cursor, len(events)):
            total = 0
            for end in range(start, len(events)):
                total += events[end]["new_tokens"]
                if total == target:
                    found = (start, end + 1)
                    break
                if total > target:
                    break
            if found:
                break
        if found is None:
            raise ValueError(
                f"could not match ordered c{phase['concurrency']} prefill total {target}"
            )
        start, end = found
        selected = events[start:end]
        if any(item["cached_tokens"] != 0 for item in selected):
            raise ValueError("matched measured prefill used cached prompt tokens")
        matches.append({**phase, "event_indices": [start, end], "events": selected})
        cursor = end
    return matches


def decode_dense_m(concurrency: int, server: dict[str, Any]) -> int:
    if server["disable_cuda_graph"] is True:
        return concurrency
    buckets = server.get("cuda_graph_bs")
    if not buckets:
        raise ValueError("server log does not expose CUDA graph buckets")
    if server["disable_cuda_graph_padding"] is True:
        return concurrency
    eligible = [value for value in buckets if value >= concurrency]
    if not eligible:
        raise ValueError(f"no CUDA graph bucket covers concurrency {concurrency}")
    return min(eligible)


def create_plan(
    fixture: Path, concurrencies: list[int], cases: list[str]
) -> dict[str, Any]:
    fixture_meta, all_ids = fixture_info(fixture)
    if max(concurrencies) > len(all_ids):
        raise ValueError("concurrency exceeds frozen fixture request count")
    requested = set(concurrencies)
    evidence: list[dict[str, Any]] = []
    observed: dict[int, list[dict[str, Any]]] = {value: [] for value in concurrencies}
    for raw_case in cases:
        if "::" not in raw_case:
            raise ValueError("--case must be SERVER_LOG::BENCHMARK_JSON")
        log_raw, benchmark_raw = raw_case.split("::", 1)
        log_path, benchmark_path = Path(log_raw), Path(benchmark_raw)
        server = parse_server_log(log_path)
        benchmark_meta, phases = validate_benchmark(
            benchmark_path, all_ids, requested
        )
        if not phases:
            raise ValueError(f"{benchmark_path}: no requested concurrency")
        if server["fp8_gemm_runner_backend"] != "triton":
            raise ValueError(f"{log_path}: FP8 GEMM backend is not Triton")
        if server["enable_mixed_chunk"] is not True:
            raise ValueError(f"{log_path}: expected enable_mixed_chunk=True")
        matched = match_prefill_segments(server["prefill_events"], phases)
        case_record = {
            "server_log": {
                "path": server["path"],
                "sha256": server["sha256"],
            },
            "benchmark": benchmark_meta,
            "phases": [],
        }
        for phase in matched:
            prefill_ms = [
                event["new_tokens"] + event["running_requests"]
                for event in phase["events"]
            ]
            decode_m = decode_dense_m(phase["concurrency"], server)
            record = {
                **phase,
                "prefill_dense_m": prefill_ms,
                "decode_dense_m": decode_m,
                "mixed_chunk_rule": "new_tokens + running_requests",
            }
            observed[phase["concurrency"]].append(record)
            case_record["phases"].append(record)
        evidence.append(case_record)

    missing = [value for value, records in observed.items() if not records]
    recommended_ms = sorted(
        {
            m
            for records in observed.values()
            for record in records
            for m in [*record["prefill_dense_m"], record["decode_dense_m"]]
        }
    )
    predicted = {
        str(value): {
            "whole_burst_prefill_m": sum(map(len, all_ids[:value])),
            "decode_m_without_padding": value,
        }
        for value in concurrencies
    }
    return {
        "schema": SCHEMA,
        "kind": "workload-m-plan",
        "created_at": utc_now(),
        "fixture": fixture_meta,
        "concurrencies": concurrencies,
        "runtime": {
            "model": "Qwen3-30B-A3B-FP8",
            "activation_output_dtype": "bfloat16",
            "block_shape": [BLOCK_N, BLOCK_K],
            "dense_shapes": [list(shape) for shape in EXPECTED_SHAPES],
        },
        "predicted_without_scheduler_trace": predicted,
        "evidence": evidence,
        "observed_by_concurrency": {str(key): value for key, value in observed.items()},
        "recommended_ms": recommended_ms,
        "missing_formal_concurrencies": missing,
        "qualification_ready": not missing and bool(cases),
        "notes": [
            "recommended_ms is the union across all supplied cold-process traces",
            "mixed prefill dense M includes one token for every running decode request",
            "CUDA-graph capture-only buckets are startup work and are not added unless used by measured decode padding",
        ],
    }


def validate_config(config: Any) -> dict[str, int]:
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError(f"kernel config keys must be exactly {sorted(CONFIG_KEYS)}")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in config.values()):
        raise ValueError("all kernel config values must be positive integers")
    if config["BLOCK_SIZE_K"] != BLOCK_K:
        raise ValueError("BLOCK_SIZE_K must be 128 so tuning equals runtime execution")
    return config


def import_tuner(sglang_root: Path) -> tuple[Any, Any, Any, Path]:
    python_root = (sglang_root / "python").resolve()
    tuner_path = (
        sglang_root / "benchmark/kernels/quantization/tuning_block_wise_kernel.py"
    ).resolve()
    if not python_root.is_dir() or not tuner_path.is_file():
        raise ValueError("invalid pinned SGLang root")
    sys.path.insert(0, str(python_root))
    spec = importlib.util.spec_from_file_location("pinned_sglang_dense_tuner", tuner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned SGLang tuner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import sglang  # type: ignore
    import torch  # type: ignore
    import triton  # type: ignore

    imported = Path(sglang.__file__).resolve()
    if python_root not in imported.parents:
        raise RuntimeError(f"sglang import escaped pinned source: {imported}")
    return module, torch, triton, tuner_path


def tune_one(args: argparse.Namespace) -> dict[str, Any]:
    sglang_root = args.sglang_root.resolve()
    if git_head(sglang_root) != EXPECTED_SGLANG_COMMIT:
        raise ValueError("SGLang commit differs from the frozen comparison commit")
    module, torch, triton, tuner_path = import_tuner(sglang_root)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("tune-one requires exactly one CUDA-visible idle GPU")
    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    if device_name != EXPECTED_DEVICE or capability != [12, 0]:
        raise RuntimeError(
            f"expected {EXPECTED_DEVICE} capability [12, 0], got {device_name} {capability}"
        )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    search_space = [
        config
        for config in module.get_configs_compute_bound()
        if config["BLOCK_SIZE_K"] >= BLOCK_K
        and config["BLOCK_SIZE_K"] % BLOCK_K == 0
        and args.k % config["BLOCK_SIZE_K"] == 0
    ]
    if not search_space:
        raise RuntimeError("corrected search space is empty")
    started = time.perf_counter()
    best = module.tune(
        args.m,
        args.n,
        args.k,
        [BLOCK_N, BLOCK_K],
        torch.bfloat16,
        search_space,
        "fp8",
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    best = validate_config(best)
    fragment = {
        "schema": SCHEMA,
        "kind": "tuning-fragment",
        "created_at": utc_now(),
        "shape": {
            "M": args.m,
            "N": args.n,
            "K": args.k,
            "block_shape": [BLOCK_N, BLOCK_K],
            "input_dtype": "float8_e4m3fn",
            "output_dtype": "bfloat16",
        },
        "best_config": best,
        "search": {
            "candidate_count": len(search_space),
            "candidate_sha256": sha256_json(search_space),
            "constraint": "BLOCK_SIZE_K >= 128 and divisible by 128; K divisible by BLOCK_SIZE_K",
            "seed": args.seed,
            "elapsed_seconds": elapsed,
        },
        "environment": {
            "device_name": device_name,
            "compute_capability": capability,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": triton.__version__,
            "sglang_commit": git_head(sglang_root),
            "sglang_root": str(sglang_root),
            "tuner_path": str(tuner_path),
            "tuner_sha256": sha256_file(tuner_path),
            "fp8_kernel_sha256": sha256_file(
                sglang_root / "python/sglang/srt/layers/quantization/fp8_kernel.py"
            ),
        },
        "limitations": [
            "upstream tuner ranks latency candidates but does not validate model-level output",
            "formal server output and tuned-vs-default ablation remain required gates",
        ],
    }
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    atomic_json(args.output, fragment)
    return fragment


def plan_jobs(plan_path: Path) -> list[tuple[int, int, int]]:
    plan = load_json(plan_path)
    if plan.get("schema") != SCHEMA or plan.get("kind") != "workload-m-plan":
        raise ValueError("invalid workload M plan")
    if plan.get("qualification_ready") is not True:
        raise ValueError("plan lacks a formal frozen trace for every concurrency")
    ms = plan.get("recommended_ms")
    if not isinstance(ms, list) or not ms:
        raise ValueError("plan has no recommended M values")
    return [(n, k, m) for n, k in EXPECTED_SHAPES for m in ms]


def merge_fragments(plan_path: Path, fragments_dir: Path, bundle_dir: Path) -> dict[str, Any]:
    jobs = plan_jobs(plan_path)
    expected = {(n, k, m) for n, k, m in jobs}
    found: dict[tuple[int, int, int], tuple[Path, dict[str, Any]]] = {}
    for path in sorted(fragments_dir.rglob("*.fragment.json")):
        fragment = load_json(path)
        if fragment.get("schema") != SCHEMA or fragment.get("kind") != "tuning-fragment":
            continue
        shape = fragment.get("shape", {})
        key = (shape.get("N"), shape.get("K"), shape.get("M"))
        if key not in expected:
            continue
        validate_config(fragment.get("best_config"))
        environment = fragment.get("environment", {})
        if environment.get("device_name") != EXPECTED_DEVICE:
            raise ValueError(f"{path}: wrong tuning device")
        if environment.get("sglang_commit") != EXPECTED_SGLANG_COMMIT:
            raise ValueError(f"{path}: wrong SGLang commit")
        if key in found and sha256_json(found[key][1]) != sha256_json(fragment):
            raise ValueError(f"conflicting fragments for {key}")
        found[key] = (path, fragment)
    missing = expected - set(found)
    if missing:
        raise ValueError(f"missing tuning fragments: {sorted(missing)}")

    configs_dir = bundle_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    installed_files = []
    device_slug = EXPECTED_DEVICE.replace(" ", "_")
    required_ms = sorted({m for _, _, m in jobs})
    for n, k in EXPECTED_SHAPES:
        runtime = {
            str(m): found[(n, k, m)][1]["best_config"]
            for m in required_ms
        }
        filename = (
            f"N={n},K={k},device_name={device_slug},dtype=fp8_w8a8,"
            f"block_shape=[{BLOCK_N}, {BLOCK_K}].json"
        )
        target = configs_dir / filename
        atomic_json(target, runtime)
        installed_files.append(
            {
                "filename": filename,
                "sha256": sha256_file(target),
                "M_keys": sorted(map(int, runtime)),
            }
        )
    manifest = {
        "schema": SCHEMA,
        "kind": "installable-bundle",
        "created_at": utc_now(),
        "plan": {"path": str(plan_path.resolve()), "sha256": sha256_file(plan_path)},
        "expected_kt_commit": EXPECTED_KT_COMMIT,
        "expected_sglang_commit": EXPECTED_SGLANG_COMMIT,
        "device_name": EXPECTED_DEVICE,
        "activation_output_dtype": "bfloat16",
        "files": installed_files,
        "fragments": [
            {
                "path": str(found[key][0].resolve()),
                "sha256": sha256_file(found[key][0]),
                "shape": list(key),
            }
            for key in sorted(found)
        ],
    }
    atomic_json(bundle_dir / "manifest.json", manifest)
    return manifest


def install_bundle(
    bundle_dir: Path, kt_root: Path, receipt_path: Path
) -> dict[str, Any]:
    kt_root = kt_root.resolve()
    sglang_root = kt_root / "third_party/sglang"
    if git_head(kt_root) != EXPECTED_KT_COMMIT or git_head(sglang_root) != EXPECTED_SGLANG_COMMIT:
        raise ValueError("checkout commits differ from the frozen comparison environment")
    manifest_path = bundle_dir / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema") != SCHEMA or manifest.get("kind") != "installable-bundle":
        raise ValueError("invalid tuning bundle manifest")
    target_dir = sglang_root / "python/sglang/srt/layers/quantization/configs"
    operations = []
    for item in manifest.get("files", []):
        filename = item["filename"]
        if Path(filename).name != filename:
            raise ValueError("bundle filename is not a basename")
        source = bundle_dir / "configs" / filename
        if sha256_file(source) != item["sha256"]:
            raise ValueError(f"bundle hash mismatch for {filename}")
        target = target_dir / filename
        if target.exists() and sha256_file(target) != item["sha256"]:
            raise FileExistsError(f"refusing to replace different config {target}")
        operations.append((source, target, not target.exists()))

    created: list[Path] = []
    try:
        for source, target, should_create in operations:
            if should_create:
                temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
                temporary.write_bytes(source.read_bytes())
                os.replace(temporary, target)
                created.append(target)
    except Exception:
        for target in created:
            if target.exists():
                target.unlink()
        raise
    receipt = {
        "schema": SCHEMA,
        "kind": "install-receipt",
        "created_at": utc_now(),
        "bundle_manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "target_dir": str(target_dir.resolve()),
        "allowed_filenames": [target.name for _, target, _ in operations],
        "files": [
            {
                "target": str(target.resolve()),
                "sha256": sha256_file(target),
                "created_by_install": should_create,
            }
            for _, target, should_create in operations
        ],
    }
    atomic_json(receipt_path, receipt)
    return receipt


def uninstall_receipt(receipt_path: Path) -> dict[str, Any]:
    receipt = load_json(receipt_path)
    if receipt.get("schema") != SCHEMA or receipt.get("kind") != "install-receipt":
        raise ValueError("invalid install receipt")
    target_dir = Path(receipt.get("target_dir", "")).resolve()
    expected_suffix = "third_party/sglang/python/sglang/srt/layers/quantization/configs"
    if not target_dir.as_posix().endswith(expected_suffix):
        raise ValueError("receipt target is not an SGLang quantization config directory")
    expected_names = {
        (
            f"N={n},K={k},device_name={EXPECTED_DEVICE.replace(' ', '_')},"
            f"dtype=fp8_w8a8,block_shape=[{BLOCK_N}, {BLOCK_K}].json"
        )
        for n, k in EXPECTED_SHAPES
    }
    allowed_names = set(receipt.get("allowed_filenames", []))
    if allowed_names != expected_names:
        raise ValueError("receipt does not name exactly the two expected dense configs")
    removed = []
    for item in receipt.get("files", []):
        if not item.get("created_by_install"):
            continue
        target = Path(item["target"]).resolve()
        if target.parent != target_dir or target.name not in expected_names:
            raise ValueError("receipt contains a target outside the exact config scope")
        if not target.exists():
            raise FileNotFoundError(f"installed config disappeared: {target}")
        if sha256_file(target) != item["sha256"]:
            raise ValueError(f"installed config changed; refusing to remove {target}")
    for item in receipt.get("files", []):
        if item.get("created_by_install"):
            target = Path(item["target"])
            target.unlink()
            removed.append(str(target))
    return {"status": "ok", "removed": removed, "recoverable_from_bundle": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit-log", help="classify/fail dense fallback warnings")
    audit.add_argument("--log", required=True, type=Path)
    audit.add_argument("--expect", choices=["default", "tuned", "report"], default="report")
    audit.add_argument("--output", type=Path)

    plan = sub.add_parser("plan", help="derive actual dense M values from frozen runs")
    plan.add_argument("--fixture", required=True, type=Path)
    plan.add_argument("--concurrencies", type=parse_positive_csv, default=[1, 4, 8])
    plan.add_argument(
        "--case",
        action="append",
        default=[],
        help="repeat SERVER_LOG::BENCHMARK_JSON; benchmark must use frozen input_ids",
    )
    plan.add_argument("--output", required=True, type=Path)

    jobs = sub.add_parser("jobs", help="print plan jobs as N<TAB>K<TAB>M")
    jobs.add_argument("--plan", required=True, type=Path)

    tune = sub.add_parser("tune-one", help="GPU: tune one (M,N,K) fragment")
    tune.add_argument("--sglang-root", required=True, type=Path)
    tune.add_argument("--M", dest="m", required=True, type=int)
    tune.add_argument("--N", dest="n", required=True, type=int)
    tune.add_argument("--K", dest="k", required=True, type=int)
    tune.add_argument("--seed", type=int, default=0)
    tune.add_argument("--output", required=True, type=Path)
    tune.add_argument("--force", action="store_true")

    merge = sub.add_parser("merge", help="validate fragments and build runtime JSON files")
    merge.add_argument("--plan", required=True, type=Path)
    merge.add_argument("--fragments-dir", required=True, type=Path)
    merge.add_argument("--bundle-dir", required=True, type=Path)

    install = sub.add_parser("install", help="atomically install a reviewed bundle")
    install.add_argument("--bundle-dir", required=True, type=Path)
    install.add_argument("--kt-root", required=True, type=Path)
    install.add_argument("--receipt", required=True, type=Path)

    uninstall = sub.add_parser("uninstall", help="remove only files created by a receipt")
    uninstall.add_argument("--receipt", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "audit-log":
            result = audit_log(args.log, args.expect)
            if args.output:
                atomic_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if result["status"] == "ok" else 1
        if args.command == "plan":
            result = create_plan(args.fixture, args.concurrencies, args.case)
            atomic_json(args.output, result)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if result["qualification_ready"] else 2
        if args.command == "jobs":
            for n, k, m in plan_jobs(args.plan):
                print(f"{n}\t{k}\t{m}")
            return 0
        if args.command == "tune-one":
            if (args.n, args.k) not in EXPECTED_SHAPES or args.m <= 0:
                raise ValueError("shape is outside the frozen two-shape plan")
            result = tune_one(args)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "merge":
            result = merge_fragments(args.plan, args.fragments_dir, args.bundle_dir)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "install":
            result = install_bundle(args.bundle_dir, args.kt_root, args.receipt)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "uninstall":
            print(json.dumps(uninstall_receipt(args.receipt), indent=2, sort_keys=True))
            return 0
        raise AssertionError(args.command)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
